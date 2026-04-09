"""
CAP THEOREM AND CONSISTENCY MODELS
=====================================

Problem Statement:
Distributed databases must choose trade-offs between Consistency, Availability,
and Partition Tolerance. Understanding CAP theorem and the consistency spectrum
is essential for designing distributed systems.

CAP Theorem (Brewer's Theorem):
  In a distributed system with network partitions, you can guarantee at most
  two of the three properties simultaneously:

  C — Consistency      : Every read returns the most recent write or an error
  A — Availability     : Every request receives a (non-error) response
  P — Partition Tolerance: System works despite network partitions

  Since network partitions are unavoidable, real systems choose CP or AP:
    CP: Sacrifice availability (refuse requests) to maintain consistency
        Examples: HBase, Zookeeper, Etcd, MongoDB (default)
    AP: Sacrifice consistency (return stale data) to stay available
        Examples: Cassandra, CouchDB, DynamoDB, DNS

PACELC Extension:
  Also trade-off between Latency and Consistency when there is NO partition:
  Low latency → accept eventual consistency
  Strong consistency → accept higher latency

Consistency Models (strongest to weakest):
  Linearizability  : Real-time globally ordered writes (Redis single-node)
  Sequential       : All nodes see same order but not necessarily real-time
  Causal           : Causally related operations ordered; concurrent may vary
  Read-Your-Writes : You always see your own writes
  Monotonic Read   : You never see older data after seeing newer
  Eventual         : All nodes will eventually converge (no ordering guarantee)
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time
import random
import threading


class ConsistencyModel(Enum):
    LINEARIZABLE    = "linearizable"
    SEQUENTIAL      = "sequential"
    CAUSAL          = "causal"
    READ_YOUR_WRITES= "read_your_writes"
    MONOTONIC_READ  = "monotonic_read"
    EVENTUAL        = "eventual"


class CAPChoice(Enum):
    CP = "cp"   # Consistent + Partition Tolerant
    AP = "ap"   # Available + Partition Tolerant


@dataclass
class Version:
    """Vector clock entry for tracking causality."""
    node_id  : str
    counter  : int = 0

    def __lt__(self, other: "Version") -> bool:
        return self.counter < other.counter


@dataclass
class DataItem:
    key      : str
    value    : object
    version  : int
    timestamp: float = field(default_factory=time.time)
    node_id  : str = ""


# ─────────────────────────────────────────────
# DISTRIBUTED NODE (simulated)
# ─────────────────────────────────────────────

class DistributedNode:
    def __init__(self, node_id: str, cap_choice: CAPChoice,
                 consistency: ConsistencyModel = ConsistencyModel.EVENTUAL):
        self.node_id     = node_id
        self.cap_choice  = cap_choice
        self.consistency = consistency
        self._data       : Dict[str, DataItem] = {}
        self._peers      : List["DistributedNode"] = []
        self._partitioned: bool = False   # simulated network partition
        self._version    = 0
        self.reads       = 0
        self.writes      = 0
        self.rejected    = 0   # reads/writes rejected during partition (CP)

    def connect_peer(self, peer: "DistributedNode"):
        self._peers.append(peer)

    def _can_reach_quorum(self) -> bool:
        if not self._partitioned:
            return True
        reachable = sum(1 for p in self._peers if not p._partitioned)
        return reachable >= len(self._peers) // 2   # majority

    def write(self, key: str, value: object) -> Tuple[bool, str]:
        self.writes += 1
        if self.cap_choice == CAPChoice.CP and self._partitioned:
            if not self._can_reach_quorum():
                self.rejected += 1
                return False, f"CP: rejected write during partition (no quorum)"

        self._version += 1
        item = DataItem(key, value, self._version, node_id=self.node_id)
        self._data[key] = item

        # Propagate to peers (async for AP, sync for CP)
        for peer in self._peers:
            if not peer._partitioned:
                peer._replicate(item)

        return True, f"written v{self._version} on {self.node_id}"

    def _replicate(self, item: DataItem):
        """Accept replicated write if it's newer."""
        existing = self._data.get(item.key)
        if not existing or item.version > existing.version:
            self._data[item.key] = item

    def read(self, key: str) -> Tuple[Optional[object], str]:
        self.reads += 1
        if self.cap_choice == CAPChoice.CP and self._partitioned:
            if not self._can_reach_quorum():
                self.rejected += 1
                return None, f"CP: rejected read during partition (no quorum)"

        item = self._data.get(key)
        if item:
            return item.value, f"v{item.version} from {self.node_id}"
        return None, f"not found on {self.node_id}"

    def simulate_partition(self, is_partitioned: bool):
        self._partitioned = is_partitioned
        status = "PARTITIONED" if is_partitioned else "RECONNECTED"
        print(f"  Network: {self.node_id} → {status}")

    def report(self):
        print(f"    {self.node_id} [{self.cap_choice.value}]: "
              f"reads={self.reads}  writes={self.writes}  "
              f"rejected={self.rejected}  data_keys={len(self._data)}")


# ─────────────────────────────────────────────
# EVENTUAL CONSISTENCY DEMO
# ─────────────────────────────────────────────

class EventualConsistencyDemo:
    """Shows how divergent writes eventually converge."""

    def __init__(self):
        self._nodes: Dict[str, Dict] = {}   # node_id → {key: (value, version)}

    def write_to(self, node_id: str, key: str, value: object, version: int):
        self._nodes.setdefault(node_id, {})[key] = (value, version)

    def converge(self) -> Dict:
        """Last-write-wins (LWW) — highest version wins."""
        result = {}
        for data in self._nodes.values():
            for k, (v, ver) in data.items():
                existing = result.get(k)
                if not existing or ver > existing[1]:
                    result[k] = (v, ver)
        return {k: v for k, (v, _) in result.items()}

    def show_divergence(self, key: str):
        print(f"  Key='{key}' across nodes:")
        for node_id, data in self._nodes.items():
            item = data.get(key)
            print(f"    {node_id}: {item[0] if item else 'missing'} "
                  f"(v{item[1] if item else 0})")


# ─────────────────────────────────────────────
# CONSISTENCY LEVEL SELECTOR (Cassandra-like)
# ─────────────────────────────────────────────

class ConsistencyLevelSelector:
    """
    Cassandra-style: choose consistency level per-operation.
    R + W > N → strong consistency (no stale reads).
    R = 1, W = 1 → eventual consistency (fast but stale possible).
    """

    @staticmethod
    def quorum(n: int) -> int:
        return n // 2 + 1

    @staticmethod
    def analyze(n: int, write_cl: int, read_cl: int) -> Dict:
        strong = write_cl + read_cl > n
        return {
            "N": n,
            "W": write_cl,
            "R": read_cl,
            "W+R": write_cl + read_cl,
            "strong_consistency": strong,
            "write_latency": "fast" if write_cl == 1 else ("moderate" if write_cl <= n//2 else "slow"),
            "read_latency": "fast" if read_cl == 1 else "moderate",
            "availability": "high" if write_cl == 1 and read_cl == 1 else "moderate",
        }


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_cap():
    print("=" * 65)
    print("CAP THEOREM AND CONSISTENCY MODELS")
    print("=" * 65)

    # ── CP vs AP during partition ─────────────
    print("\n[1] CP vs AP DURING NETWORK PARTITION")
    print("─" * 55)

    # CP nodes
    cp1 = DistributedNode("cp-node-1", CAPChoice.CP)
    cp2 = DistributedNode("cp-node-2", CAPChoice.CP)
    cp3 = DistributedNode("cp-node-3", CAPChoice.CP)
    cp1.connect_peer(cp2)
    cp1.connect_peer(cp3)
    cp2.connect_peer(cp1)

    # AP nodes
    ap1 = DistributedNode("ap-node-1", CAPChoice.AP)
    ap2 = DistributedNode("ap-node-2", CAPChoice.AP)
    ap3 = DistributedNode("ap-node-3", CAPChoice.AP)
    ap1.connect_peer(ap2)
    ap1.connect_peer(ap3)

    # Normal write
    ok, msg = cp1.write("balance", 1000)
    ok2, msg2 = ap1.write("balance", 1000)
    print(f"  Normal write — CP: {msg}")
    print(f"  Normal write — AP: {msg2}")

    # Simulate partition
    print()
    cp1.simulate_partition(True)
    ap1.simulate_partition(True)

    # CP: rejects writes (sacrifices availability)
    ok_cp, msg_cp = cp1.write("balance", 1200)
    print(f"  During partition — CP write: {ok_cp} — {msg_cp}")

    # AP: accepts writes (sacrifices consistency)
    ok_ap, msg_ap = ap1.write("balance", 1200)
    print(f"  During partition — AP write: {ok_ap} — {msg_ap}")

    # Reads during partition
    val_cp, _ = cp1.read("balance")
    val_ap, _ = ap1.read("balance")
    print(f"  During partition — CP read: {val_cp} (refused or stale)")
    print(f"  During partition — AP read: {val_ap} (returns potentially stale data)")

    print()
    cp1.simulate_partition(False)
    ap1.simulate_partition(False)

    print("\n  CP cluster stats:")
    for n in [cp1, cp2, cp3]:
        n.report()
    print("  AP cluster stats:")
    for n in [ap1, ap2, ap3]:
        n.report()

    # ── Eventual Consistency ──────────────────
    print("\n\n[2] EVENTUAL CONSISTENCY — DIVERGENCE THEN CONVERGENCE")
    print("─" * 55)
    demo = EventualConsistencyDemo()
    # Concurrent writes to different nodes (partition scenario)
    demo.write_to("node-A", "counter", 5,  version=1)
    demo.write_to("node-B", "counter", 10, version=2)   # newer
    demo.write_to("node-C", "counter", 3,  version=1)

    demo.show_divergence("counter")
    converged = demo.converge()
    print(f"  After gossip/sync → converged value: {converged}")
    print(f"  (Last-Write-Wins: version 2 wins → value=10)")

    # ── Cassandra Consistency Levels ──────────
    print("\n\n[3] CASSANDRA CONSISTENCY LEVELS (N=3 nodes)")
    print("─" * 55)
    selector = ConsistencyLevelSelector()
    N = 3
    configs = [
        (1, 1,                "ANY / ONE — eventual, fastest"),
        (2, 1,                "W=QUORUM, R=ONE — write durable, fast reads"),
        (2, 2,                "W=QUORUM, R=QUORUM — strong consistency"),
        (3, 1,                "W=ALL, R=ONE — very durable, slow writes"),
        (1, 3,                "W=ONE, R=ALL — fast writes, slow reads"),
    ]
    print(f"  {'W':<4} {'R':<4} {'W+R':<6} {'Strong?':<10} {'Write':<12} {'Read':<12} Notes")
    print(f"  {'─'*75}")
    for w, r, notes in configs:
        a = selector.analyze(N, w, r)
        print(f"  {w:<4} {r:<4} {a['W+R']:<6} "
              f"{'✅' if a['strong_consistency'] else '❌':<10} "
              f"{a['write_latency']:<12} {a['read_latency']:<12} {notes}")

    # ── Consistency Model Hierarchy ───────────
    print("\n\n[4] CONSISTENCY MODEL SPECTRUM")
    print("─" * 55)
    models = [
        ("Linearizable",    "Highest",  "Single-node Redis, etcd, Zookeeper"),
        ("Sequential",      "Strong",   "Most SQL databases"),
        ("Causal",          "Medium",   "MongoDB causal consistency, COPS"),
        ("Read-Your-Writes","Medium",   "Read primary after write (pg routing)"),
        ("Monotonic Read",  "Weak",     "Consistent replica reads (no going back)"),
        ("Eventual",        "Weakest",  "Cassandra ONE, DNS, CDN edge caches"),
    ]
    print(f"  {'Model':<22} {'Strength':<12} {'Examples'}")
    print(f"  {'─'*70}")
    for model, strength, examples in models:
        print(f"  {model:<22} {strength:<12} {examples}")

    # ── PACELC ────────────────────────────────
    print("\n\n[5] PACELC — EXTENDED CAP (Latency vs Consistency)")
    print("─" * 55)
    pacelc = [
        ("DynamoDB",   "PA / EL", "Available during partition; low latency (eventual)"),
        ("Cassandra",  "PA / EL", "Available; tunable consistency (default eventual)"),
        ("HBase",      "PC / EC", "Consistent during partition; strong consistency"),
        ("MongoDB",    "PC / EC", "Consistent with majority writes"),
        ("MySQL",      "PC / EC", "ACID; strong consistency always"),
        ("Riak",       "PA / EL", "Available; eventual consistency (CRDT)"),
    ]
    print(f"  {'System':<14} {'PACELC':<12} {'Notes'}")
    print(f"  {'─'*65}")
    for system, pacelc_val, notes in pacelc:
        print(f"  {system:<14} {pacelc_val:<12} {notes}")
    print("\n  PA/EL = Partition → Available, no partition → Low Latency (eventual)")
    print("  PC/EC = Partition → Consistent, no partition → Strong Consistency")


if __name__ == "__main__":
    demonstrate_cap()
