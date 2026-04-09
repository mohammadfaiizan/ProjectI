"""
QUORUM-BASED READS AND WRITES
================================

Problem Statement:
In a replicated system with N nodes, how do you ensure consistency without
requiring all N nodes to agree (which would be slow and unavailable if any fails)?
Quorums: require a majority or configurable subset. Overlap guarantees correctness.

Core Quorum Theorem:
  Write quorum W + Read quorum R > N  →  reads always see the latest write.
  Reason: any W nodes + any R nodes must overlap (by pigeonhole). The overlap
  contains the latest write. The read coordinator picks the highest version.

Common Configurations (N=5):
  W=3, R=3: strong consistency (W+R=6>5). Tolerates 2 failures for both reads/writes.
  W=5, R=1: write-all. Reads are fast and always consistent. No write if any node down.
  W=1, R=5: read-all. Writes are fast. Reads are slow but always consistent.
  W=3, R=1: W+R=4 < 5. NOT consistent. Fast reads but stale possible.
  W=1, R=1: W+R=2 < 5. Fastest. No consistency guarantee.

Dynamo-Style Sloppy Quorum:
  Normal quorum is "preferred list" of N nodes.
  On failure, route to "hint" nodes (not in preferred list).
  Hinted handoff: hint node stores data temporarily, forwards when original recovers.
  Achieves availability at cost of consistency.

Read Repair:
  On quorum read: coordinator sees multiple responses.
  If some are stale → coordinator sends latest value back to stale nodes.
  Heals divergence lazily on read path.

Write Coordinator:
  Any node can be coordinator (Dynamo/Cassandra client-facing node).
  Coordinator fans out to W nodes, waits for W acks, then returns.
  Coordinator handles response tracking.

Consistency Levels in Cassandra (N=3 by default):
  ONE:    W=1 or R=1. Fastest. Eventual consistency.
  TWO:    W=2 or R=2. Partial.
  QUORUM: W=ceil(N/2+1) or R=ceil(N/2+1). Strong consistency.
  ALL:    W=N or R=N. Strongest. No availability if any replica down.
  LOCAL_QUORUM: quorum within local datacenter. Good for multi-region.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import time
import threading
import random


# ─────────────────────────────────────────────
# VERSION-STAMPED VALUE
# ─────────────────────────────────────────────

@dataclass
class VersionedValue:
    value    : Any
    version  : int    # monotonically increasing
    node_id  : str
    timestamp: float = field(default_factory=time.time)


# ─────────────────────────────────────────────
# REPLICA NODE
# ─────────────────────────────────────────────

class QuorumNode:
    def __init__(self, node_id: str, available: bool = True,
                 latency_ms: float = 1.0):
        self.node_id    = node_id
        self.available  = available
        self.latency_ms = latency_ms
        self._store     : Dict[str, VersionedValue] = {}
        self._lock      = threading.Lock()
        self.reads      = 0
        self.writes     = 0

    def write(self, key: str, value: VersionedValue) -> bool:
        if not self.available:
            return False
        time.sleep(self.latency_ms / 1000)
        with self._lock:
            existing = self._store.get(key)
            if not existing or value.version >= existing.version:
                self._store[key] = value
                self.writes += 1
        return True

    def read(self, key: str) -> Optional[VersionedValue]:
        if not self.available:
            return None
        time.sleep(self.latency_ms / 1000)
        with self._lock:
            self.reads += 1
            return self._store.get(key)

    def repair(self, key: str, value: VersionedValue):
        """Read repair: update stale value silently."""
        with self._lock:
            existing = self._store.get(key)
            if not existing or value.version > existing.version:
                self._store[key] = value


# ─────────────────────────────────────────────
# QUORUM COORDINATOR
# ─────────────────────────────────────────────

class QuorumCoordinator:
    """
    Dynamo/Cassandra-style quorum coordinator.
    Fans out writes to all N nodes, waits for W acks.
    Fans out reads to R nodes, picks highest version, repairs stale nodes.
    """

    def __init__(self, nodes: List[QuorumNode], w: int, r: int):
        self.nodes       = nodes
        self.N           = len(nodes)
        self.W           = w
        self.R           = r
        self._version    = 0
        self._lock       = threading.Lock()
        self.write_ok    = 0
        self.write_fail  = 0
        self.read_ok     = 0
        self.read_fail   = 0
        self.read_repairs= 0

    def write(self, key: str, value: Any) -> bool:
        with self._lock:
            self._version += 1
            version = self._version

        vv   = VersionedValue(value=value, version=version, node_id="coordinator")
        acks = 0
        for node in self.nodes:
            if node.write(key, vv):
                acks += 1
            if acks >= self.W:
                break   # stop once quorum reached (fire rest async in real systems)

        # Write to remaining nodes async (best-effort)
        for node in self.nodes[acks:]:
            node.write(key, vv)

        if acks >= self.W:
            self.write_ok += 1
            return True
        self.write_fail += 1
        return False

    def read(self, key: str) -> Tuple[Optional[Any], bool]:
        """Returns (value, is_consistent)."""
        responses : List[VersionedValue] = []
        responded_nodes: List[QuorumNode] = []

        for node in self.nodes:
            vv = node.read(key)
            if vv is not None:
                responses.append(vv)
                responded_nodes.append(node)
            if len(responses) >= self.R:
                break

        if len(responses) < self.R:
            self.read_fail += 1
            return None, False

        # Find highest version
        best = max(responses, key=lambda v: v.version)

        # Read repair: fix stale nodes
        for i, (node, vv) in enumerate(zip(responded_nodes, responses)):
            if vv.version < best.version:
                node.repair(key, best)
                self.read_repairs += 1

        self.read_ok += 1
        return best.value, True

    def is_strongly_consistent(self) -> bool:
        return self.W + self.R > self.N

    def stats(self) -> Dict:
        return {
            "N": self.N, "W": self.W, "R": self.R,
            "strong": self.is_strongly_consistent(),
            "writes_ok": self.write_ok, "writes_fail": self.write_fail,
            "reads_ok" : self.read_ok,  "reads_fail" : self.read_fail,
            "read_repairs": self.read_repairs,
        }


# ─────────────────────────────────────────────
# SLOPPY QUORUM + HINTED HANDOFF
# ─────────────────────────────────────────────

class SloppyQuorumStore:
    """
    Sloppy quorum: if preferred nodes are unavailable, use substitute "hint" nodes.
    Hinted handoff: hint node stores data temporarily, forwards on recovery.
    """

    def __init__(self, preferred: List[QuorumNode], hints: List[QuorumNode], w: int):
        self.preferred = preferred
        self.hints     = hints
        self.W         = w
        self._hint_queue: List[Tuple[str, VersionedValue, str]] = []   # (key, val, target)
        self._version   = 0

    def write(self, key: str, value: Any) -> Tuple[bool, int, int]:
        """Returns (success, primary_acks, hint_acks)."""
        self._version += 1
        vv = VersionedValue(value=value, version=self._version, node_id="coordinator")

        primary_acks = 0
        hint_acks    = 0
        hint_iter    = iter(self.hints)

        for pnode in self.preferred:
            if pnode.write(key, vv):
                primary_acks += 1
            else:
                # Use a hint node instead
                hnode = next(hint_iter, None)
                if hnode and hnode.write(key, vv):
                    hint_acks += 1
                    self._hint_queue.append((key, vv, pnode.node_id))

        total = primary_acks + hint_acks
        return total >= self.W, primary_acks, hint_acks

    def handoff_hints(self):
        """When preferred nodes recover, deliver hinted writes."""
        pref_map = {n.node_id: n for n in self.preferred}
        delivered = 0
        remaining = []
        for key, vv, target_id in self._hint_queue:
            target = pref_map.get(target_id)
            if target and target.available:
                target.write(key, vv)
                delivered += 1
            else:
                remaining.append((key, vv, target_id))
        self._hint_queue = remaining
        return delivered


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_quorum():
    print("=" * 65)
    print("QUORUM-BASED READS AND WRITES")
    print("=" * 65)

    random.seed(17)

    # ── Strong vs Weak Quorum ─────────────────────
    print("\n[1] STRONG QUORUM (W=3, R=3, N=5) vs WEAK (W=1, R=1, N=5)")
    print("─" * 55)

    n5_nodes_strong = [QuorumNode(f"N{i}") for i in range(5)]
    n5_nodes_weak   = [QuorumNode(f"N{i}") for i in range(5)]
    strong_coord    = QuorumCoordinator(n5_nodes_strong, w=3, r=3)
    weak_coord      = QuorumCoordinator(n5_nodes_weak,   w=1, r=1)

    strong_coord.write("balance", 1000)
    weak_coord.write("balance",   1000)

    # Both read back correctly under normal conditions
    val_strong, ok_s = strong_coord.read("balance")
    val_weak,   ok_w = weak_coord.read("balance")
    print(f"  Strong (W=3,R=3): value={val_strong} strong={strong_coord.is_strongly_consistent()}")
    print(f"  Weak   (W=1,R=1): value={val_weak}   strong={weak_coord.is_strongly_consistent()}")

    # ── Stale Read on Weak Quorum ─────────────────
    print("\n\n[2] STALE READ — WEAK QUORUM MISSES LATEST WRITE")
    print("─" * 55)

    nodes3 = [QuorumNode(f"N{i}") for i in range(3)]
    coord  = QuorumCoordinator(nodes3, w=3, r=1)   # W+R = 4 > 3 → strong? W=3,R=1: 4>3 yes

    coord.write("counter", 1)
    # Manually make N0 stale (didn't receive update)
    nodes3[0]._store.pop("counter", None)

    # Read only from N0 (stale!)
    stale_read = nodes3[0].read("counter")
    print(f"  N0 stale read: {stale_read}")
    print(f"  Strong quorum read would repair this via read quorum R≥2")

    # ── Read Repair in Action ─────────────────────
    print("\n\n[3] READ REPAIR — HEALING STALE REPLICAS")
    print("─" * 55)

    repair_nodes = [QuorumNode(f"N{i}") for i in range(5)]
    rc           = QuorumCoordinator(repair_nodes, w=3, r=3)

    rc.write("profile", {"name": "Alice"})
    # Corrupt N0 (stale)
    repair_nodes[0]._store.clear()

    val, ok = rc.read("profile")
    print(f"  Read: value={val} repairs={rc.read_repairs}")
    print(f"  N0 after repair: {repair_nodes[0]._store.get('profile')}")

    # ── Sloppy Quorum + Hinted Handoff ───────────
    print("\n\n[4] SLOPPY QUORUM + HINTED HANDOFF")
    print("─" * 55)

    pref   = [QuorumNode(f"P{i}", available=(i != 1)) for i in range(3)]   # P1 down
    hints  = [QuorumNode(f"H{i}") for i in range(2)]
    sloppy = SloppyQuorumStore(pref, hints, w=2)

    ok, p_acks, h_acks = sloppy.write("config", "production")
    print(f"  P1 is down. Write: success={ok} primary_acks={p_acks} hint_acks={h_acks}")
    print(f"  Hint queue: {len(sloppy._hint_queue)} entry")

    # P1 recovers
    pref[1].available = True
    delivered = sloppy.handoff_hints()
    print(f"  P1 recovered. Hints delivered: {delivered}")
    print(f"  P1 now has config: {pref[1]._store.get('config')}")

    # ── Quorum Configuration Guide ────────────────
    print("\n\n[5] QUORUM CONFIGURATION GUIDE (N=5)")
    print("─" * 55)
    configs = [
        (1, 1, "Lowest latency, no consistency guarantee"),
        (3, 1, "Strong writes; fast cheap reads (W+R=4<5: NOT consistent)"),
        (1, 3, "Strong reads; fast writes (W+R=4<5: NOT consistent)"),
        (3, 3, "Strong both ways. Tolerates 2 failures each"),
        (5, 1, "Write-all consistency; no writes if any node down"),
        (1, 5, "Read-all consistency; no reads if any node down"),
        (2, 4, "Heavy read workload with consistency"),
    ]
    print(f"  {'W':<4} {'R':<4} {'W+R':<6} {'>N?':<6} {'Notes'}")
    print(f"  {'─'*60}")
    for w, r, notes in configs:
        strong = "YES" if w + r > 5 else "NO "
        print(f"  {w:<4} {r:<4} {w+r:<6} {strong:<6} {notes}")

    print("\n\n[6] QUORUM DESIGN PRINCIPLES")
    print("─" * 55)
    principles = [
        "W + R > N guarantees at least one overlapping node between write and read sets",
        "Coordinator picks highest version among R responses",
        "Read repair: heal stale replicas lazily on the read path",
        "Anti-entropy: background process for full repair (Merkle tree comparison)",
        "Sloppy quorum: use substitute nodes when preferred are unavailable (AP tradeoff)",
        "Hinted handoff: hint nodes deliver writes to recovered preferred nodes",
        "Tunable: adjust W/R per request based on consistency requirements",
    ]
    for p in principles:
        print(f"  • {p}")


if __name__ == "__main__":
    demonstrate_quorum()
