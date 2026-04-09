"""
SYSTEM DESIGN: DISTRIBUTED KEY-VALUE STORE (like DynamoDB / Cassandra)
=======================================================================

Problem Statement:
Design a distributed key-value store that scales horizontally,
provides tunable consistency, and tolerates node failures.

Functional Requirements:
  - get(key) → value
  - put(key, value)
  - delete(key)
  - Scan with key prefix

Non-Functional Requirements:
  - Handle 1M QPS reads, 100K QPS writes
  - Data partitioned across N nodes; each node handles fraction
  - Continue operating when minority of nodes fail
  - Tunable consistency: eventual or strong

CAP Theorem:
  A distributed system can guarantee at most 2 of 3:
  Consistency (C):  Every read returns most recent write.
  Availability (A): Every request receives a response (may be stale).
  Partition (P):    System continues during network partitions.
  CP systems: HBase, Zookeeper. AP systems: Cassandra, DynamoDB.

Consistency Models:
  Strong:       Read your own writes. Linearizable. Requires quorum.
  Eventual:     All replicas converge eventually. High availability.
  Read-your-writes: You always see your own writes.
  Monotonic read: Never see older data after seeing newer.
  Causal:       Respect causal ordering of operations.

Quorum (R + W > N):
  N = replication factor (typically 3)
  W = write quorum (writes must ACK from W nodes)
  R = read quorum (read from R nodes, return latest)
  Strong consistency: R + W > N → e.g., N=3, W=2, R=2
  High availability:  W=1, R=1 → eventual consistency

Vector Clocks:
  Track causality: {node_id: counter}.
  On write: increment own counter.
  On receive: merge (take max per node).
  Detect conflict: neither vector dominates the other.

LSM Tree Storage (simplified):
  Write: append to MemTable (sorted in memory).
  Flush: MemTable → SSTable (immutable sorted file) when full.
  Read:  Check MemTable → SSTables (newest first) → merge.

Consistent Hashing:
  Maps keys to nodes using a hash ring.
  Virtual nodes (vnodes) for even distribution.
  See 007_Design_Distributed_Cache.py for full implementation.

Gossip Protocol:
  Nodes spread cluster state (membership, tokens) by gossiping.
  Each node periodically exchanges state with K random peers.
  Eventually all nodes know all node states (O(log N) rounds).
"""

from __future__ import annotations

import time
import hashlib
import random
import threading
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set
from enum import Enum
from collections import defaultdict


# ─────────────────────────────────────────────
# VECTOR CLOCK
# ─────────────────────────────────────────────

class VectorClock:
    """Tracks causal ordering of writes across nodes."""

    def __init__(self, clock: Optional[Dict[str, int]] = None):
        self._clock = dict(clock or {})

    def increment(self, node_id: str) -> "VectorClock":
        new_clock = dict(self._clock)
        new_clock[node_id] = new_clock.get(node_id, 0) + 1
        return VectorClock(new_clock)

    def merge(self, other: "VectorClock") -> "VectorClock":
        """Merge (take max per node)."""
        merged = dict(self._clock)
        for nid, cnt in other._clock.items():
            merged[nid] = max(merged.get(nid, 0), cnt)
        return VectorClock(merged)

    def dominates(self, other: "VectorClock") -> bool:
        """True if self is causally after other."""
        for nid, cnt in other._clock.items():
            if self._clock.get(nid, 0) < cnt:
                return False
        return self._clock != other._clock or True

    def concurrent_with(self, other: "VectorClock") -> bool:
        """Neither dominates the other → conflict."""
        return (not self.dominates(other) and
                not other.dominates(self) and
                self._clock != other._clock)

    def to_dict(self) -> Dict:
        return dict(self._clock)

    def __repr__(self):
        return f"VC{self._clock}"

    def __eq__(self, other):
        return isinstance(other, VectorClock) and self._clock == other._clock


# ─────────────────────────────────────────────
# VALUE WITH METADATA
# ─────────────────────────────────────────────

@dataclass
class VersionedValue:
    value:     Any
    clock:     VectorClock
    timestamp: float
    node_id:   str

    def is_newer_than(self, other: "VersionedValue") -> bool:
        return self.clock.dominates(other.clock)


# ─────────────────────────────────────────────
# STORAGE ENGINE (LSM-like)
# ─────────────────────────────────────────────

class MemTable:
    """In-memory sorted write buffer."""

    def __init__(self, max_size: int = 1000):
        self._data:   Dict[str, VersionedValue] = {}
        self._max     = max_size

    def put(self, key: str, val: VersionedValue):
        self._data[key] = val

    def get(self, key: str) -> Optional[VersionedValue]:
        return self._data.get(key)

    def delete(self, key: str):
        # Tombstone marker
        self._data[key] = VersionedValue(None, VectorClock(), time.time(), "__tombstone__")

    def is_full(self) -> bool:
        return len(self._data) >= self._max

    def flush(self) -> "SSTable":
        table = SSTable(dict(sorted(self._data.items())))
        self._data.clear()
        return table

    def size(self) -> int:
        return len(self._data)


class SSTable:
    """Immutable sorted string table (on-disk simulation)."""

    def __init__(self, data: Dict[str, VersionedValue]):
        self._data = data
        self._bloom: Set[str] = set(data.keys())   # simplified bloom filter

    def might_contain(self, key: str) -> bool:
        return key in self._bloom

    def get(self, key: str) -> Optional[VersionedValue]:
        return self._data.get(key)

    def scan(self, prefix: str) -> List[Tuple[str, VersionedValue]]:
        return [(k, v) for k, v in sorted(self._data.items())
                if k.startswith(prefix)]

    def size(self) -> int:
        return len(self._data)


class StorageEngine:
    """LSM-tree storage engine per partition."""

    def __init__(self, node_id: str):
        self._node_id = node_id
        self._memtable = MemTable()
        self._sstables: List[SSTable] = []

    def put(self, key: str, value: Any, clock: Optional[VectorClock] = None
            ) -> VersionedValue:
        clock = (clock or VectorClock()).increment(self._node_id)
        vv    = VersionedValue(value, clock, time.time(), self._node_id)
        self._memtable.put(key, vv)
        if self._memtable.is_full():
            self._sstables.insert(0, self._memtable.flush())
        return vv

    def delete(self, key: str):
        self._memtable.delete(key)

    def get(self, key: str) -> Optional[VersionedValue]:
        # Check MemTable first
        vv = self._memtable.get(key)
        if vv:
            return None if vv.node_id == "__tombstone__" else vv

        # Check SSTables (newest first due to reverse insertion)
        for sst in self._sstables:
            if sst.might_contain(key):
                vv = sst.get(key)
                if vv:
                    return None if vv.node_id == "__tombstone__" else vv
        return None

    def scan(self, prefix: str) -> List[Tuple[str, Any]]:
        result: Dict[str, VersionedValue] = {}
        # Merge from all sources (newest wins)
        for sst in reversed(self._sstables):
            for k, vv in sst.scan(prefix):
                if k not in result:
                    result[k] = vv
        for k, vv in self._memtable._data.items():
            if k.startswith(prefix):
                result[k] = vv
        return [(k, vv.value) for k, vv in sorted(result.items())
                if vv.node_id != "__tombstone__" and vv.value is not None]


# ─────────────────────────────────────────────
# KV NODE
# ─────────────────────────────────────────────

class NodeStatus(Enum):
    UP   = "up"
    DOWN = "down"


class KVNode:
    def __init__(self, node_id: str):
        self.node_id   = node_id
        self.status    = NodeStatus.UP
        self._storage  = StorageEngine(node_id)
        self._replicas: List["KVNode"] = []

    def put(self, key: str, value: Any,
            clock: Optional[VectorClock] = None) -> VersionedValue:
        if self.status == NodeStatus.DOWN:
            raise ConnectionError(f"Node {self.node_id} is down")
        return self._storage.put(key, value, clock)

    def get(self, key: str) -> Optional[VersionedValue]:
        if self.status == NodeStatus.DOWN:
            raise ConnectionError(f"Node {self.node_id} is down")
        return self._storage.get(key)

    def delete(self, key: str):
        if self.status == NodeStatus.DOWN:
            raise ConnectionError(f"Node {self.node_id} is down")
        self._storage.delete(key)

    def scan(self, prefix: str) -> List[Tuple[str, Any]]:
        return self._storage.scan(prefix)


# ─────────────────────────────────────────────
# PARTITION RING (consistent hashing simplified)
# ─────────────────────────────────────────────

class PartitionRing:
    def __init__(self, nodes: List[KVNode], replication_factor: int = 3):
        self._nodes = sorted(nodes, key=lambda n: n.node_id)
        self._rf    = min(replication_factor, len(nodes))

    def _hash(self, key: str) -> int:
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def preference_list(self, key: str) -> List[KVNode]:
        """Returns ordered list of N nodes responsible for this key."""
        h   = self._hash(key)
        idx = h % len(self._nodes)
        result = []
        for i in range(self._rf):
            result.append(self._nodes[(idx + i) % len(self._nodes)])
        return result


# ─────────────────────────────────────────────
# DISTRIBUTED KV STORE
# ─────────────────────────────────────────────

class ConsistencyLevel(Enum):
    ONE    = 1   # fastest; eventual
    QUORUM = 2   # majority; balanced
    ALL    = 3   # slowest; strongest


class DistributedKVStore:
    """
    Distributed key-value store with tunable consistency.
    N=3 replication factor. Quorum = N//2 + 1 = 2.
    """

    def __init__(self, n_nodes: int = 3, rf: int = 3):
        self._nodes  = [KVNode(f"node-{i}") for i in range(n_nodes)]
        self._ring   = PartitionRing(self._nodes, rf)
        self._rf     = rf

    def put(self, key: str, value: Any,
            consistency: ConsistencyLevel = ConsistencyLevel.QUORUM) -> bool:
        targets = self._ring.preference_list(key)
        success_count = 0
        last_vv: Optional[VersionedValue] = None

        for node in targets:
            try:
                clock = last_vv.clock if last_vv else None
                vv    = node.put(key, value, clock)
                last_vv = vv
                success_count += 1
            except ConnectionError:
                pass

        return success_count >= consistency.value

    def get(self, key: str,
            consistency: ConsistencyLevel = ConsistencyLevel.QUORUM
            ) -> Optional[Any]:
        targets = self._ring.preference_list(key)
        versions: List[VersionedValue] = []

        for node in targets:
            try:
                vv = node.get(key)
                if vv:
                    versions.append(vv)
                if len(versions) >= consistency.value:
                    break
            except ConnectionError:
                pass

        if not versions:
            return None

        # Read repair: return latest version
        latest = max(versions, key=lambda v: v.timestamp)
        return latest.value

    def delete(self, key: str) -> bool:
        targets = self._ring.preference_list(key)
        for node in targets:
            try:
                node.delete(key)
            except ConnectionError:
                pass
        return True

    def scan(self, prefix: str) -> List[Tuple[str, Any]]:
        """Scan from one node (not quorum; best-effort for demo)."""
        for node in self._nodes:
            if node.status == NodeStatus.UP:
                return node.scan(prefix)
        return []

    def simulate_failure(self, node_idx: int):
        self._nodes[node_idx].status = NodeStatus.DOWN

    def restore_node(self, node_idx: int):
        self._nodes[node_idx].status = NodeStatus.UP


# ─────────────────────────────────────────────
# GOSSIP PROTOCOL SIMULATION
# ─────────────────────────────────────────────

@dataclass
class GossipState:
    node_id:    str
    status:     NodeStatus
    generation: int
    heartbeat:  int
    updated_at: float


class GossipProtocol:
    """
    Each node maintains a view of cluster state.
    Periodically exchanges state with random peers.
    """

    def __init__(self, nodes: List[str]):
        self._states: Dict[str, Dict[str, GossipState]] = {
            nid: {nid: GossipState(nid, NodeStatus.UP, 1, 0, time.time())}
            for nid in nodes
        }

    def heartbeat(self, node_id: str):
        state = self._states[node_id].get(node_id)
        if state:
            state.heartbeat += 1
            state.updated_at = time.time()

    def gossip(self, node_a: str, node_b: str):
        """Exchange state between two nodes."""
        state_a = self._states[node_a]
        state_b = self._states[node_b]
        # Merge: take newer heartbeat per node
        for nid in set(state_a) | set(state_b):
            a = state_a.get(nid)
            b = state_b.get(nid)
            if a and b:
                newer = a if a.heartbeat >= b.heartbeat else b
                state_a[nid] = newer
                state_b[nid] = newer
            elif a:
                state_b[nid] = a
            elif b:
                state_a[nid] = b

    def mark_dead(self, node_id: str, from_node: str):
        if node_id in self._states[from_node]:
            self._states[from_node][node_id].status = NodeStatus.DOWN

    def cluster_view(self, from_node: str) -> Dict[str, str]:
        return {nid: s.status.value
                for nid, s in self._states[from_node].items()}


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_kv_store():
    print("=" * 65)
    print("SYSTEM DESIGN: DISTRIBUTED KEY-VALUE STORE")
    print("=" * 65)

    kv = DistributedKVStore(n_nodes=3, rf=3)

    # ── Basic Operations ──────────────────────
    print("\n[1] BASIC OPERATIONS")
    print("─" * 55)

    kv.put("user:alice", {"name": "Alice", "email": "alice@example.com"})
    kv.put("user:bob",   {"name": "Bob",   "email": "bob@example.com"})
    kv.put("session:xyz", "token_abc123", ConsistencyLevel.ONE)

    for key in ["user:alice", "user:bob", "session:xyz", "nonexistent"]:
        val = kv.get(key)
        print(f"  GET {key} → {val}")

    # ── Scan ──────────────────────────────────
    print("\n[2] PREFIX SCAN")
    print("─" * 55)

    for prefix in ["user:", "session:"]:
        results = kv.scan(prefix)
        print(f"  SCAN '{prefix}' → {len(results)} results")
        for k, v in results:
            print(f"    {k}: {v}")

    # ── Node Failure (partition tolerance) ────
    print("\n[3] NODE FAILURE (N=3, RF=3, quorum=2)")
    print("─" * 55)

    print("  Simulating node-0 failure...")
    kv.simulate_failure(0)

    # With quorum=2, still works (2 out of 3 nodes up)
    kv.put("product:123", {"name": "Widget", "price": 9.99})
    val = kv.get("product:123")
    print(f"  PUT/GET with 1 failed node: {val}")

    # Kill second node
    kv.simulate_failure(1)
    kv_put_ok = kv.put("key_x", "val", ConsistencyLevel.QUORUM)
    print(f"\n  With 2 failed nodes (only 1 up):")
    print(f"  QUORUM write succeeded: {kv_put_ok} (cannot reach quorum=2)")
    # ONE consistency still works
    kv_one_ok = kv.put("key_x", "val", ConsistencyLevel.ONE)
    print(f"  ONE consistency write: {kv_one_ok}")

    kv.restore_node(0)
    kv.restore_node(1)

    # ── Vector Clocks ─────────────────────────
    print("\n[4] VECTOR CLOCKS")
    print("─" * 55)

    vc_a = VectorClock().increment("node-0")
    vc_b = vc_a.increment("node-1")
    vc_c = vc_a.increment("node-2")   # concurrent with vc_b

    print(f"  vc_a (after write on node-0): {vc_a}")
    print(f"  vc_b (after write on node-1): {vc_b}")
    print(f"  vc_c (concurrent write on node-2): {vc_c}")
    print(f"  vc_b dominates vc_a:  {vc_b.dominates(vc_a)}")
    print(f"  vc_b concurrent vc_c: {vc_b.concurrent_with(vc_c)}")
    print(f"  → Concurrent means CONFLICT: need last-write-wins or app resolution")

    # ── LSM Tree Storage ──────────────────────
    print("\n[5] LSM TREE STORAGE")
    print("─" * 55)

    engine = StorageEngine("demo-node")
    for i in range(20):
        engine.put(f"key_{i:03d}", f"value_{i}")

    print(f"  After 20 writes:")
    print(f"    MemTable size: {engine._memtable.size()}")
    print(f"    SSTables: {len(engine._sstables)}")

    # Overwrite some keys
    engine.put("key_005", "updated_value_5")
    val = engine.get("key_005")
    print(f"  GET key_005 (after update): {val.value if val else None}")

    engine.delete("key_010")
    val = engine.get("key_010")
    print(f"  GET key_010 (after delete): {val}")

    scan = engine.scan("key_01")
    print(f"  SCAN 'key_01': {[(k, v) for k, v in scan]}")

    # ── Gossip Protocol ───────────────────────
    print("\n[6] GOSSIP PROTOCOL")
    print("─" * 55)

    nodes = ["node-0", "node-1", "node-2", "node-3"]
    gossip = GossipProtocol(nodes)

    # Heartbeats
    for _ in range(3):
        for nid in nodes:
            gossip.heartbeat(nid)

    # node-0 and node-1 gossip
    gossip.gossip("node-0", "node-1")
    gossip.gossip("node-1", "node-2")
    gossip.gossip("node-2", "node-3")   # node-3 learns about node-0 after 3 rounds

    # Simulate node-2 going down
    gossip.mark_dead("node-2", "node-0")

    print("  Cluster view from node-0:")
    for nid, status in gossip.cluster_view("node-0").items():
        print(f"    {nid}: {status}")

    # ── Consistency Comparison ─────────────────
    print("\n[7] CONSISTENCY LEVEL COMPARISON")
    print("─" * 55)

    print(f"  {'Level':<12} {'W':<5} {'R':<5} {'Consistency':<15} {'Availability'}")
    print("  " + "─" * 55)
    configs = [
        ("ONE",    1, 1, "Eventual",       "High (any node up)"),
        ("QUORUM", 2, 2, "Strong",          "Medium (majority up)"),
        ("ALL",    3, 3, "Linearizable",    "Low (all nodes up)"),
    ]
    for level, w, r, consistency, avail in configs:
        print(f"  {level:<12} W={w}  R={r}  {consistency:<15} {avail}")

    # ── Architecture ──────────────────────────
    print("\n[8] DISTRIBUTED KV STORE ARCHITECTURE")
    print("─" * 55)

    arch = [
        ("Partitioning",  "Consistent hashing; vnodes for load balance"),
        ("Replication",   "RF=3; write to N coordinator + RF-1 replicas"),
        ("Consistency",   "Tunable: ONE/QUORUM/ALL via R+W>N rule"),
        ("Write path",    "MemTable → WAL → async SSTable flush"),
        ("Compaction",    "Merge SSTables; remove tombstones; leveled or STCS"),
        ("Failure detect","Gossip protocol; phi accrual failure detector"),
        ("Read repair",   "On quorum read: return latest; async update stale replicas"),
        ("Hinted handoff","Write to temp node if target down; replay when restored"),
        ("Anti-entropy",  "Merkle tree comparison between replicas; sync diffs"),
        ("Examples",      "DynamoDB, Cassandra, Riak, etcd, TiKV"),
    ]
    for component, detail in arch:
        print(f"  {component:<18} {detail}")


if __name__ == "__main__":
    demonstrate_kv_store()
