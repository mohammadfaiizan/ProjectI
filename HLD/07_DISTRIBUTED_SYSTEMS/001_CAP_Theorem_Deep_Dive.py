"""
CAP THEOREM DEEP DIVE
======================

Problem Statement:
In a distributed system, what guarantees can you actually provide?
CAP Theorem (Brewer, 2000) states: a distributed system can provide at most 2 of:
  C — Consistency:   every read receives the most recent write or an error.
  A — Availability:  every request receives a (non-error) response.
  P — Partition Tolerance: the system continues despite network partition.

The Critical Insight:
  Network partitions WILL happen in any real distributed system (hardware faults,
  network congestion, datacenter splits). Therefore P is NOT optional.
  The real choice is: CP or AP?

CP systems (Consistency + Partition tolerance):
  During a partition: return an error rather than serve stale data.
  Examples: HBase, Zookeeper, etcd, MongoDB (with write concern majority).
  Use when: money transfers, inventory counts, distributed locks.

AP systems (Availability + Partition tolerance):
  During a partition: return the best available (possibly stale) data.
  Examples: Cassandra, CouchDB, DynamoDB, DNS.
  Use when: shopping carts, social media feeds, user profiles, analytics.

CA systems (Consistency + Availability):
  Only possible with a single node (no partition possible).
  Examples: a single PostgreSQL instance, SQLite.
  In practice: not achievable at scale.

Common Misconceptions:
  "You pick 2 of 3" — No, P is mandatory. You choose C or A during partition.
  "Eventual consistency = AP" — Not exactly. AP just means you return data.
  "CP = always consistent" — Only during partitions. Normal ops can be highly available.
  CAP doesn't say anything about performance, durability, or latency.

Extended Reality (PACELC):
  Even without partition, there's a Latency vs Consistency trade-off.
  See 002_PACELC_Theorem.py for the full picture.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import time
import threading
import random


# ─────────────────────────────────────────────
# SIMULATED NETWORK PARTITION
# ─────────────────────────────────────────────

class NetworkPartition:
    """Controls which nodes can communicate with each other."""

    def __init__(self, nodes: List[str]):
        self._partitioned: set = set()

    def partition(self, node_id: str):
        self._partitioned.add(node_id)

    def heal(self, node_id: str):
        self._partitioned.discard(node_id)

    def can_communicate(self, src: str, dst: str) -> bool:
        return src not in self._partitioned and dst not in self._partitioned


# ─────────────────────────────────────────────
# CP SYSTEM: Refuses reads on partition
# ─────────────────────────────────────────────

class CPNode:
    """
    CP node: during partition, refuses to serve stale reads.
    Prefers consistency over availability.
    Like ZooKeeper — if it can't confirm quorum, returns error.
    """

    def __init__(self, node_id: str, network: NetworkPartition, peers: List[str]):
        self.node_id    = node_id
        self.network    = network
        self.peers      = peers
        self._data      : Dict[str, Any] = {}
        self._version   : Dict[str, int] = {}
        self.served     = 0
        self.refused    = 0

    def write(self, key: str, value: Any) -> bool:
        """Write requires quorum acknowledgement."""
        reachable = [p for p in self.peers
                     if self.network.can_communicate(self.node_id, p)]
        quorum = len(self.peers) // 2 + 1
        if len(reachable) + 1 < quorum:   # +1 for self
            return False   # reject write — can't reach quorum
        self._data[key]    = value
        self._version[key] = self._version.get(key, 0) + 1
        return True

    def read(self, key: str) -> Tuple[Optional[Any], str]:
        """Read requires confirming with majority. Error if partitioned."""
        reachable = [p for p in self.peers
                     if self.network.can_communicate(self.node_id, p)]
        quorum = len(self.peers) // 2 + 1
        if len(reachable) + 1 < quorum:
            self.refused += 1
            return None, "ERROR: not enough peers reachable for consistent read"
        self.served += 1
        return self._data.get(key), "OK"


# ─────────────────────────────────────────────
# AP SYSTEM: Serves stale data on partition
# ─────────────────────────────────────────────

class APNode:
    """
    AP node: during partition, serves possibly stale data rather than error.
    Prefers availability over consistency.
    Like Cassandra with ONE consistency level.
    """

    def __init__(self, node_id: str):
        self.node_id  = node_id
        self._data    : Dict[str, Any]  = {}
        self._version : Dict[str, int]  = {}
        self.served   = 0
        self.stale_served = 0

    def write(self, key: str, value: Any, version: int):
        """Accept write. In real AP system, resolve conflicts via last-write-wins or CRDT."""
        if version >= self._version.get(key, 0):
            self._data[key]    = value
            self._version[key] = version

    def read(self, key: str, global_version: int) -> Tuple[Optional[Any], bool]:
        """Always responds. Returns (value, is_stale)."""
        self.served += 1
        value   = self._data.get(key)
        current = self._version.get(key, 0)
        stale   = value is not None and current < global_version
        if stale:
            self.stale_served += 1
        return value, stale


# ─────────────────────────────────────────────
# CAP SCENARIO SIMULATOR
# ─────────────────────────────────────────────

class CAPSimulator:
    def __init__(self, n_nodes: int = 3):
        self.n_nodes = n_nodes
        self.network = NetworkPartition([f"N{i}" for i in range(n_nodes)])

    def simulate_cp(self) -> Dict:
        """CP: during partition, reads fail rather than return stale data."""
        nodes    = [f"N{i}" for i in range(self.n_nodes)]
        cp_nodes = {nid: CPNode(nid, self.network, [n for n in nodes if n != nid])
                    for nid in nodes}

        # Write before partition
        cp_nodes["N0"].write("balance", 1000)

        # Partition N2 from cluster
        self.network.partition("N2")

        # N0 can still serve (has quorum with N1)
        val0, status0 = cp_nodes["N0"].read("balance")
        # N2 cannot serve (isolated — no quorum)
        val2, status2 = cp_nodes["N2"].read("balance")

        self.network.heal("N2")

        return {
            "N0_response": (val0, status0),
            "N2_response": (val2, status2),
            "N0_served"  : cp_nodes["N0"].served,
            "N2_refused" : cp_nodes["N2"].refused,
        }

    def simulate_ap(self) -> Dict:
        """AP: during partition, each node serves its local (possibly stale) data."""
        n1 = APNode("N1")
        n2 = APNode("N2")

        # Both start with version 1
        n1.write("cart", ["item-A"], version=1)
        n2.write("cart", ["item-A"], version=1)

        # N1 gets an update (version 2) — but is partitioned from N2
        n1.write("cart", ["item-A", "item-B"], version=2)

        # Both nodes serve reads
        v1, stale1 = n1.read("cart", global_version=2)
        v2, stale2 = n2.read("cart", global_version=2)

        return {
            "N1": {"value": v1, "stale": stale1},
            "N2": {"value": v2, "stale": stale2},
            "N1_stale_served": n1.stale_served,
            "N2_stale_served": n2.stale_served,
        }


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_cap():
    print("=" * 65)
    print("CAP THEOREM DEEP DIVE")
    print("=" * 65)

    sim = CAPSimulator(n_nodes=3)

    # ── CP Behaviour ──────────────────────────────
    print("\n[1] CP SYSTEM — CONSISTENCY DURING PARTITION")
    print("─" * 55)
    cp_result = sim.simulate_cp()
    print(f"  N0 read (has quorum):  {cp_result['N0_response']}")
    print(f"  N2 read (partitioned): {cp_result['N2_response']}")
    print(f"  N0 served: {cp_result['N0_served']}  N2 refused: {cp_result['N2_refused']}")
    print(f"  → N2 returns ERROR rather than potentially stale data")

    # ── AP Behaviour ──────────────────────────────
    print("\n\n[2] AP SYSTEM — AVAILABILITY DURING PARTITION")
    print("─" * 55)
    ap_result = sim.simulate_ap()
    print(f"  N1 read: value={ap_result['N1']['value']} stale={ap_result['N1']['stale']}")
    print(f"  N2 read: value={ap_result['N2']['value']} stale={ap_result['N2']['stale']}")
    print(f"  N2 served stale data: {ap_result['N2_stale_served']} time(s)")
    print(f"  → N2 returns available (stale) data rather than error")

    # ── Decision Guide ────────────────────────────
    print("\n\n[3] CP vs AP DECISION GUIDE")
    print("─" * 55)
    scenarios = [
        ("Bank account balance",        "CP",  "Stale balance = incorrect charge"),
        ("Shopping cart",               "AP",  "Stale cart = minor UX issue"),
        ("Inventory reservation",       "CP",  "Oversell = business problem"),
        ("Social media follower count", "AP",  "Slightly stale count = acceptable"),
        ("Distributed lock",            "CP",  "Two lock holders = disaster"),
        ("User profile read",           "AP",  "Slightly outdated bio = fine"),
        ("Payment idempotency key",     "CP",  "Duplicate charge = critical bug"),
        ("DNS record",                  "AP",  "Short-lived stale routing = OK"),
        ("Seat reservation (flight)",   "CP",  "Double booking = unacceptable"),
        ("Product catalog price",       "AP",  "10s stale price = acceptable"),
    ]
    print(f"  {'Scenario':<34} {'Choice':<8} {'Reason'}")
    print(f"  {'─'*72}")
    for scenario, choice, reason in scenarios:
        print(f"  {scenario:<34} {choice:<8} {reason}")

    # ── Common Examples ───────────────────────────
    print("\n\n[4] REAL-WORLD SYSTEM CLASSIFICATIONS")
    print("─" * 55)
    systems = [
        ("ZooKeeper",            "CP",  "Coordination: locks, leader election"),
        ("etcd",                 "CP",  "Kubernetes config store"),
        ("HBase",                "CP",  "Strong consistency required"),
        ("Cassandra",            "AP",  "Tunable — default AP"),
        ("DynamoDB",             "AP",  "Eventually consistent reads"),
        ("CouchDB",              "AP",  "Multi-master, eventual consistency"),
        ("PostgreSQL (single)",  "CA",  "No network partition possible"),
        ("MongoDB (w:majority)", "CP",  "Majority write concern = CP"),
        ("MongoDB (w:1)",        "AP",  "Single ack = potentially inconsistent"),
        ("Redis (single)",       "CA",  "No partition possible on single node"),
    ]
    print(f"  {'System':<26} {'Type':<6} {'Notes'}")
    print(f"  {'─'*65}")
    for system, cap_type, notes in systems:
        print(f"  {system:<26} {cap_type:<6} {notes}")

    # ── Key Nuances ───────────────────────────────
    print("\n\n[5] CAP NUANCES AND COMMON MISTAKES")
    print("─" * 55)
    nuances = [
        "Partition Tolerance is NOT optional — networks fail. Always choose C or A.",
        "CAP is a binary model. Reality is a spectrum of consistency levels.",
        "CP system is available during NORMAL operation — only rejects during partition.",
        "AP system can implement conflict resolution (LWW, CRDTs) to reduce staleness.",
        "'Consistent' in CAP = linearizability, not serializability or read-your-writes.",
        "Many systems are tunable: Cassandra can be CP with ALL consistency level.",
        "PACELC extends CAP: even without partition, latency vs consistency matters.",
    ]
    for nuance in nuances:
        print(f"  • {nuance}")


if __name__ == "__main__":
    demonstrate_cap()
