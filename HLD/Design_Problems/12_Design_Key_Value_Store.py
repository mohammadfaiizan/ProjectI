"""
Distributed Key-Value Store - Core Implementation
Demonstrates: consistent hash ring with virtual nodes, quorum reads/writes,
vector clocks, LSM tree (MemTable + SSTables + compaction), Bloom filter,
gossip protocol simulation. Standard library only.
"""

import hashlib
import math
import random
import time
from bisect import bisect_left, insort
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Bloom Filter
# ---------------------------------------------------------------------------

class BloomFilter:
    """
    Space-efficient probabilistic set membership check.
    False positive rate ~1% with 10 bits/key.
    Never false negatives: if filter says NOT present, key definitely absent.
    """

    def __init__(self, capacity: int = 10000, error_rate: float = 0.01):
        self.capacity = capacity
        self.error_rate = error_rate
        self.bit_size = self._optimal_bit_size(capacity, error_rate)
        self.hash_count = self._optimal_hash_count(self.bit_size, capacity)
        self.bit_array = bytearray(math.ceil(self.bit_size / 8))
        self.count = 0

    @staticmethod
    def _optimal_bit_size(n: int, p: float) -> int:
        return int(-n * math.log(p) / (math.log(2) ** 2))

    @staticmethod
    def _optimal_hash_count(m: int, n: int) -> int:
        return max(1, int(m / n * math.log(2)))

    def _hashes(self, key: str) -> List[int]:
        results = []
        for i in range(self.hash_count):
            h = int(hashlib.md5(f"{key}:{i}".encode()).hexdigest(), 16)
            results.append(h % self.bit_size)
        return results

    def add(self, key: str):
        for bit_pos in self._hashes(key):
            byte_idx, bit_idx = divmod(bit_pos, 8)
            self.bit_array[byte_idx] |= (1 << bit_idx)
        self.count += 1

    def contains(self, key: str) -> bool:
        """Returns False -> definitely not present. True -> probably present."""
        for bit_pos in self._hashes(key):
            byte_idx, bit_idx = divmod(bit_pos, 8)
            if not (self.bit_array[byte_idx] & (1 << bit_idx)):
                return False
        return True

    @property
    def false_positive_rate(self) -> float:
        k, m, n = self.hash_count, self.bit_size, self.count
        return (1 - math.exp(-k * n / m)) ** k if n else 0.0


# ---------------------------------------------------------------------------
# Vector Clock
# ---------------------------------------------------------------------------

class VectorClock:
    """Tracks causality for conflict detection in distributed writes."""

    def __init__(self, clocks: Optional[Dict[str, int]] = None):
        self.clocks: Dict[str, int] = clocks or {}

    def increment(self, node_id: str) -> "VectorClock":
        new_clocks = dict(self.clocks)
        new_clocks[node_id] = new_clocks.get(node_id, 0) + 1
        return VectorClock(new_clocks)

    def merge(self, other: "VectorClock") -> "VectorClock":
        """Take element-wise maximum."""
        all_nodes = set(self.clocks) | set(other.clocks)
        return VectorClock({
            n: max(self.clocks.get(n, 0), other.clocks.get(n, 0))
            for n in all_nodes
        })

    def dominates(self, other: "VectorClock") -> bool:
        """True if self is strictly newer than other (happens-after)."""
        all_nodes = set(self.clocks) | set(other.clocks)
        all_leq = all(self.clocks.get(n, 0) >= other.clocks.get(n, 0) for n in all_nodes)
        any_gt  = any(self.clocks.get(n, 0) >  other.clocks.get(n, 0) for n in all_nodes)
        return all_leq and any_gt

    def concurrent_with(self, other: "VectorClock") -> bool:
        return not self.dominates(other) and not other.dominates(self)

    def __repr__(self):
        return f"VC{dict(sorted(self.clocks.items()))}"


# ---------------------------------------------------------------------------
# LSM Tree Storage Engine
# ---------------------------------------------------------------------------

@dataclass
class SSTableEntry:
    key: str
    value: Optional[Any]  # None = tombstone (deleted)
    clock: VectorClock
    timestamp: float


class SSTable:
    """Immutable sorted table flushed from MemTable."""

    def __init__(self, entries: List[SSTableEntry]):
        self.entries: Dict[str, SSTableEntry] = {e.key: e for e in entries}
        self.bloom = BloomFilter(capacity=max(len(entries), 10))
        for e in entries:
            self.bloom.add(e.key)
        self.created_at = time.time()

    def get(self, key: str) -> Optional[SSTableEntry]:
        if not self.bloom.contains(key):
            return None  # Bloom filter eliminates unnecessary disk read
        return self.entries.get(key)

    def __len__(self):
        return len(self.entries)


class LSMTree:
    """
    Write-optimized storage engine.
    Write path: WAL -> MemTable -> (flush) -> SSTable L0 -> (compact) -> L1 -> L2
    Read path:  MemTable -> L0 SSTables -> L1 -> L2 (bloom filters skip absent keys)
    """

    MEMTABLE_SIZE_LIMIT = 5  # flush after N entries (small for demo; production = 64MB)
    COMPACTION_THRESHOLD = 3  # compact when L0 has >= N SSTables

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.memtable: Dict[str, SSTableEntry] = {}
        self.wal: List[Tuple[str, SSTableEntry]] = []  # append-only log
        self.sstables_l0: List[SSTable] = []  # L0: recently flushed, may overlap
        self.sstables_l1: List[SSTable] = []  # L1: compacted, no overlap
        self.write_count = 0
        self.compaction_count = 0

    def put(self, key: str, value: Any, clock: Optional[VectorClock] = None):
        clock = (clock or VectorClock()).increment(self.node_id)
        entry = SSTableEntry(key=key, value=value, clock=clock, timestamp=time.time())
        # 1. Append to WAL (durability guarantee)
        self.wal.append(("PUT", entry))
        # 2. Update MemTable
        self.memtable[key] = entry
        self.write_count += 1
        # 3. Check if MemTable should be flushed
        if len(self.memtable) >= self.MEMTABLE_SIZE_LIMIT:
            self._flush_memtable()
        return clock

    def delete(self, key: str):
        """Write a tombstone entry — actual deletion happens during compaction."""
        self.put(key, None)

    def get(self, key: str) -> Optional[SSTableEntry]:
        # Search order: MemTable -> L0 (newest first) -> L1
        if key in self.memtable:
            entry = self.memtable[key]
            return None if entry.value is None else entry

        for sst in reversed(self.sstables_l0):  # newest L0 first
            entry = sst.get(key)
            if entry is not None:
                return None if entry.value is None else entry

        for sst in reversed(self.sstables_l1):
            entry = sst.get(key)
            if entry is not None:
                return None if entry.value is None else entry

        return None

    def _flush_memtable(self):
        """Flush MemTable to a new L0 SSTable."""
        entries = list(self.memtable.values())
        sst = SSTable(entries)
        self.sstables_l0.append(sst)
        self.memtable = {}
        print(f"  [LSM:{self.node_id}] Flushed MemTable -> SSTable L0 "
              f"({len(entries)} entries, total L0={len(self.sstables_l0)})")
        if len(self.sstables_l0) >= self.COMPACTION_THRESHOLD:
            self._compact_l0_to_l1()

    def _compact_l0_to_l1(self):
        """
        Merge all L0 SSTables into a single L1 SSTable.
        Key strategy: for each key, keep the entry with the latest timestamp.
        Drops tombstones when no older version exists.
        """
        merged: Dict[str, SSTableEntry] = {}
        # Process oldest first so newer entries overwrite
        for sst in self.sstables_l0:
            for key, entry in sst.entries.items():
                existing = merged.get(key)
                if existing is None or entry.timestamp > existing.timestamp:
                    merged[key] = entry

        # Remove tombstones (value=None) during compaction
        live_entries = [e for e in merged.values() if e.value is not None]

        if live_entries:
            self.sstables_l1.append(SSTable(live_entries))
        self.compaction_count += 1
        print(f"  [LSM:{self.node_id}] Compacted {len(self.sstables_l0)} L0 SSTables "
              f"-> 1 L1 SSTable ({len(live_entries)} live entries)")
        self.sstables_l0 = []

    def stats(self) -> Dict:
        return {
            "memtable_size": len(self.memtable),
            "l0_sstables": len(self.sstables_l0),
            "l1_sstables": len(self.sstables_l1),
            "writes": self.write_count,
            "compactions": self.compaction_count,
        }


# ---------------------------------------------------------------------------
# Consistent Hash Ring
# ---------------------------------------------------------------------------

class ConsistentHashRing:
    """
    Hash ring [0, 2^128) with virtual nodes for even distribution.
    Uses MD5 for speed (SHA-256 in production).
    """

    def __init__(self, virtual_nodes: int = 150):
        self.virtual_nodes = virtual_nodes
        self._ring: List[int] = []          # sorted list of hash positions
        self._ring_map: Dict[int, str] = {} # hash position -> node_id
        self._nodes: Set[str] = set()

    def add_node(self, node_id: str):
        self._nodes.add(node_id)
        for i in range(self.virtual_nodes):
            pos = self._hash(f"{node_id}#vn{i}")
            self._ring_map[pos] = node_id
            insort(self._ring, pos)

    def remove_node(self, node_id: str):
        self._nodes.discard(node_id)
        for i in range(self.virtual_nodes):
            pos = self._hash(f"{node_id}#vn{i}")
            if pos in self._ring_map:
                del self._ring_map[pos]
                idx = bisect_left(self._ring, pos)
                if idx < len(self._ring) and self._ring[idx] == pos:
                    self._ring.pop(idx)

    def get_nodes(self, key: str, n: int = 1) -> List[str]:
        """Return n distinct physical nodes responsible for key (clockwise walk)."""
        if not self._ring:
            return []
        pos = self._hash(key)
        idx = bisect_left(self._ring, pos) % len(self._ring)
        seen: Set[str] = set()
        result: List[str] = []
        attempts = 0
        while len(result) < n and attempts < len(self._ring):
            node_id = self._ring_map[self._ring[idx % len(self._ring)]]
            if node_id not in seen:
                seen.add(node_id)
                result.append(node_id)
            idx += 1
            attempts += 1
        return result

    def _hash(self, key: str) -> int:
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def distribution(self, keys: List[str]) -> Dict[str, int]:
        """Show how many keys land on each node (primary only)."""
        dist: Dict[str, int] = defaultdict(int)
        for k in keys:
            nodes = self.get_nodes(k, n=1)
            if nodes:
                dist[nodes[0]] += 1
        return dict(dist)


# ---------------------------------------------------------------------------
# Gossip Protocol (Failure Detection)
# ---------------------------------------------------------------------------

@dataclass
class NodeState:
    node_id: str
    heartbeat: int = 0
    status: str = "UP"  # UP, SUSPECT, DOWN
    last_updated: float = field(default_factory=time.time)


class GossipProtocol:
    """
    Simulates gossip-based membership and failure detection.
    Each 'round' represents one gossip cycle (1 second in production).
    """

    SUSPECT_THRESHOLD = 3   # rounds without heartbeat -> SUSPECT
    DOWN_THRESHOLD = 8      # rounds without heartbeat -> DOWN
    FANOUT = 3              # gossip to 3 random peers per round

    def __init__(self):
        self.nodes: Dict[str, NodeState] = {}
        self.round = 0

    def add_node(self, node_id: str):
        self.nodes[node_id] = NodeState(node_id=node_id)

    def heartbeat(self, node_id: str):
        """A live node increments its own heartbeat counter."""
        if node_id in self.nodes:
            self.nodes[node_id].heartbeat += 1
            self.nodes[node_id].last_updated = time.time()
            self.nodes[node_id].status = "UP"

    def gossip_round(self, initiating_node: str):
        """
        Simulates one gossip exchange: node sends its state to FANOUT peers.
        Merge rule: take max heartbeat per node.
        """
        if initiating_node not in self.nodes:
            return
        peers = random.sample(
            [n for n in self.nodes if n != initiating_node],
            min(self.FANOUT, len(self.nodes) - 1)
        )
        sender_state = self.nodes[initiating_node]
        for peer_id in peers:
            peer_state = self.nodes.get(peer_id)
            if peer_state:
                peer_state.heartbeat = max(peer_state.heartbeat, sender_state.heartbeat)

    def detect_failures(self, node_id: str, rounds_since_update: Dict[str, int]):
        """Mark nodes SUSPECT or DOWN based on heartbeat staleness."""
        for nid, rounds in rounds_since_update.items():
            state = self.nodes.get(nid)
            if not state:
                continue
            if rounds >= self.DOWN_THRESHOLD:
                state.status = "DOWN"
            elif rounds >= self.SUSPECT_THRESHOLD:
                state.status = "SUSPECT"

    def live_nodes(self) -> List[str]:
        return [nid for nid, s in self.nodes.items() if s.status == "UP"]

    def summary(self) -> str:
        lines = [f"  Round {self.round}: Cluster State"]
        for nid, s in sorted(self.nodes.items()):
            lines.append(f"    {nid}: status={s.status}, heartbeat={s.heartbeat}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Distributed Key-Value Store
# ---------------------------------------------------------------------------

class KeyValueStore:
    """
    Distributed KV store with consistent hashing, quorum reads/writes,
    and LSM tree storage per node.
    """

    def __init__(self, node_ids: List[str], replication_factor: int = 3,
                 write_quorum: int = 2, read_quorum: int = 2,
                 virtual_nodes: int = 50):
        self.N = replication_factor
        self.W = write_quorum
        self.R = read_quorum

        self.ring = ConsistentHashRing(virtual_nodes=virtual_nodes)
        self.storage: Dict[str, LSMTree] = {}
        self.gossip = GossipProtocol()

        for nid in node_ids:
            self.ring.add_node(nid)
            self.storage[nid] = LSMTree(nid)
            self.gossip.add_node(nid)

        print(f"[KVStore] Initialized with {len(node_ids)} nodes, "
              f"N={self.N}, W={self.W}, R={self.R}")

    def put(self, key: str, value: Any) -> Tuple[bool, str]:
        """
        Write to W replicas. Returns (success, message).
        Coordinator picks N replicas; waits for W ACKs.
        """
        replicas = self.ring.get_nodes(key, n=self.N)
        live_replicas = [r for r in replicas if r in self.gossip.live_nodes()]

        if len(live_replicas) < self.W:
            return False, f"Only {len(live_replicas)} live replicas, need W={self.W}"

        clock = VectorClock()
        written = 0
        for node_id in live_replicas:
            clock = self.storage[node_id].put(key, value, clock)
            written += 1
            if written >= self.W:
                break  # quorum satisfied

        # Write to remaining replicas asynchronously (background)
        for node_id in live_replicas[written:]:
            self.storage[node_id].put(key, value, clock)

        return True, f"Written to W={written} replicas (of N={len(live_replicas)} live)"

    def get(self, key: str) -> Tuple[Optional[Any], str]:
        """
        Read from R replicas, return latest version.
        Detects conflicts via vector clock comparison.
        """
        replicas = self.ring.get_nodes(key, n=self.N)
        live_replicas = [r for r in replicas if r in self.gossip.live_nodes()]

        if len(live_replicas) < self.R:
            return None, f"Only {len(live_replicas)} live replicas, need R={self.R}"

        results: List[SSTableEntry] = []
        for node_id in live_replicas[:self.R]:
            entry = self.storage[node_id].get(key)
            if entry is not None:
                results.append(entry)

        if not results:
            return None, "Key not found"

        # Check for conflicts
        for i, a in enumerate(results):
            for b in results[i+1:]:
                if a.clock.concurrent_with(b.clock):
                    return (
                        [r.value for r in results],
                        f"CONFLICT: {len(results)} concurrent versions detected"
                    )

        # Return the entry with the latest vector clock
        winner = max(results, key=lambda e: e.timestamp)
        return winner.value, f"Read from R={self.R} replicas"

    def delete(self, key: str) -> Tuple[bool, str]:
        replicas = self.ring.get_nodes(key, n=self.N)
        live_replicas = [r for r in replicas if r in self.gossip.live_nodes()]
        if len(live_replicas) < self.W:
            return False, "Insufficient live replicas"
        for node_id in live_replicas:
            self.storage[node_id].delete(key)
        return True, f"Deleted from {len(live_replicas)} replicas"

    def add_node(self, node_id: str):
        """Simulate node joining the cluster."""
        self.ring.add_node(node_id)
        self.storage[node_id] = LSMTree(node_id)
        self.gossip.add_node(node_id)
        print(f"[KVStore] Node {node_id} joined the cluster")

    def fail_node(self, node_id: str):
        """Simulate node failure."""
        self.gossip.nodes[node_id].status = "DOWN"
        print(f"[KVStore] Node {node_id} marked DOWN")

    def recover_node(self, node_id: str):
        self.gossip.nodes[node_id].status = "UP"
        self.gossip.nodes[node_id].heartbeat += 100
        print(f"[KVStore] Node {node_id} recovered")

    def distribution_report(self, sample_keys: int = 1000) -> Dict[str, int]:
        """Show key distribution across nodes."""
        keys = [f"key:{i}" for i in range(sample_keys)]
        return self.ring.distribution(keys)


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def demo_bloom_filter():
    print("=== Bloom Filter Demo ===")
    bf = BloomFilter(capacity=1000, error_rate=0.01)
    for i in range(500):
        bf.add(f"user:{i}")

    # True positives
    tp = sum(1 for i in range(500) if bf.contains(f"user:{i}"))
    # False positives (keys never added)
    fp = sum(1 for i in range(500, 1000) if bf.contains(f"user:{i}"))

    print(f"  True positives (should be 500): {tp}")
    print(f"  False positives (should be ~5 at 1%): {fp}")
    print(f"  Actual FP rate: {fp/500:.2%}")
    print(f"  Bit array size: {bf.bit_size:,} bits "
          f"({bf.bit_size/8/1024:.1f} KB) for 500 keys")


def demo_vector_clocks():
    print("\n=== Vector Clock Demo ===")
    # Node A writes
    vc_a = VectorClock().increment("nodeA")
    print(f"  nodeA writes: {vc_a}")

    # Node B writes concurrently (doesn't know about A's write)
    vc_b = VectorClock().increment("nodeB")
    print(f"  nodeB writes concurrently: {vc_b}")

    # Conflict detection
    print(f"  Concurrent? {vc_a.concurrent_with(vc_b)} (expected: True)")

    # Node A reads both and merges
    vc_merged = vc_a.merge(vc_b).increment("nodeA")
    print(f"  After merge + nodeA increment: {vc_merged}")
    print(f"  Merged dominates A? {vc_merged.dominates(vc_a)} (expected: True)")
    print(f"  Merged dominates B? {vc_merged.dominates(vc_b)} (expected: True)")


def demo_lsm_tree():
    print("\n=== LSM Tree Demo ===")
    lsm = LSMTree("node1")

    # Write enough to trigger flush and compaction
    for i in range(12):
        lsm.put(f"key:{i}", f"value:{i}")

    # Read test
    entry = lsm.get("key:3")
    print(f"  Read key:3 = {entry.value if entry else 'NOT FOUND'}")

    # Delete test
    lsm.delete("key:3")
    entry = lsm.get("key:3")
    print(f"  After delete, key:3 = {entry}")

    # Overwrite
    lsm.put("key:5", "updated_value")
    entry = lsm.get("key:5")
    print(f"  After overwrite, key:5 = {entry.value if entry else None}")

    print(f"  LSM Stats: {lsm.stats()}")


def demo_distributed_kvstore():
    print("\n=== Distributed KV Store Demo ===")
    nodes = [f"node_{i}" for i in range(5)]
    store = KeyValueStore(nodes, replication_factor=3, write_quorum=2, read_quorum=2,
                         virtual_nodes=30)

    # Basic operations
    ok, msg = store.put("user:alice", {"name": "Alice", "age": 30})
    print(f"  PUT user:alice: {ok} — {msg}")

    ok, msg = store.put("user:bob", {"name": "Bob", "age": 25})
    print(f"  PUT user:bob: {ok} — {msg}")

    val, msg = store.get("user:alice")
    print(f"  GET user:alice: {val} — {msg}")

    # Node failure test
    print("\n  --- Simulating node failure ---")
    store.fail_node("node_0")
    store.fail_node("node_1")
    val, msg = store.get("user:alice")
    print(f"  GET with 2 nodes DOWN: {val is not None} — {msg}")

    # Try write with too many nodes down
    store.fail_node("node_2")
    ok, msg = store.put("user:charlie", "data")
    print(f"  PUT with 3 nodes DOWN (W=2 unavailable): success={ok} — {msg}")

    # Recover nodes
    store.recover_node("node_0")
    store.recover_node("node_1")
    store.recover_node("node_2")

    # Distribution report
    dist = store.distribution_report(sample_keys=1000)
    print(f"\n  Key distribution (1000 keys across {len(nodes)} nodes):")
    for nid, count in sorted(dist.items()):
        bar = "#" * (count // 5)
        print(f"    {nid}: {count:4d} keys {bar}")


def demo_gossip():
    print("\n=== Gossip Protocol Demo ===")
    gossip = GossipProtocol()
    for i in range(6):
        gossip.add_node(f"node_{i}")

    # Simulate 5 rounds where node_5 stops heartbeating
    rounds_stale = defaultdict(int)
    for round_num in range(10):
        gossip.round = round_num
        for nid in [f"node_{i}" for i in range(5)]:  # node_5 is silent
            gossip.heartbeat(nid)
            gossip.gossip_round(nid)
        rounds_stale["node_5"] = round_num + 1
        if round_num in (2, 7, 9):
            gossip.detect_failures("node_0", dict(rounds_stale))
            print(gossip.summary())


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_bloom_filter()
    demo_vector_clocks()
    demo_lsm_tree()
    demo_distributed_kvstore()
    demo_gossip()
