"""
DISTRIBUTED CACHE DESIGN
==========================

Problem Statement:
A single Redis node can serve ~100K ops/sec and store data up to its RAM limit.
When your app needs 1M+ ops/sec or datasets > single node RAM, you need a
distributed cache cluster: consistent hashing, replication, and coordination.

Redis Cluster:
  Data sharded across N primary nodes using consistent hashing (hash slot 0-16383).
  Each primary has 1+ replicas for high availability.
  Client (redis-py, Jedis) handles routing: computes slot, connects to correct node.
  Automatic failover: replica promoted if primary is unreachable.
  No cross-slot transactions (MULTI must target same slot → use hash tags {user}.*)

Memcached Cluster:
  Client-side sharding only (no server-side coordination).
  Clients compute which server to use. Simpler, no replication.
  Stateless servers — just raw key-value, no data structures.
  Cannot failover: losing a node = cache miss for that shard.

Consistent Hashing:
  Maps keys and servers to a ring (0..2^32).
  Key → nearest server clockwise.
  Adding/removing servers: only ~1/N of keys remapped (vs modulo: all keys).
  Virtual nodes: each server mapped to V positions → better balance.

Cache Node Failure Modes:
  1. Node crash: all keys on that node become misses → stampede on DB
  2. Network partition: client can't reach node → choose retry or serve stale
  3. OOM eviction: under memory pressure, entries evicted → unexpected misses
  4. Hot key: one key receives 10K QPS → single node bottleneck

Anti-patterns:
  Large keys (>10KB): serialization overhead, network strain
  Key explosion: dynamic key patterns produce unbounded key namespaces
  No TTL: cache fills up, stale data never expires
  Storing sessions in only one node without replication
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import hashlib
import random
import time
from collections import defaultdict


# ─────────────────────────────────────────────
# CONSISTENT HASH RING
# ─────────────────────────────────────────────

class ConsistentHashRing:
    """
    Maps cache nodes to a ring via virtual nodes.
    Ensures minimal key remapping when nodes are added/removed.
    """

    def __init__(self, virtual_nodes: int = 150):
        self.virtual_nodes = virtual_nodes
        self._ring        : Dict[int, str] = {}    # position → node_id
        self._sorted_keys : List[int] = []
        self._nodes       : List[str] = []

    def _hash(self, key: str) -> int:
        return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2 ** 32)

    def add_node(self, node_id: str):
        self._nodes.append(node_id)
        for i in range(self.virtual_nodes):
            pos = self._hash(f"{node_id}#{i}")
            self._ring[pos] = node_id
        self._sorted_keys = sorted(self._ring.keys())

    def remove_node(self, node_id: str):
        if node_id in self._nodes:
            self._nodes.remove(node_id)
        for i in range(self.virtual_nodes):
            pos = self._hash(f"{node_id}#{i}")
            self._ring.pop(pos, None)
        self._sorted_keys = sorted(self._ring.keys())

    def get_node(self, key: str) -> Optional[str]:
        if not self._ring:
            return None
        h = self._hash(key)
        # Binary search for first position >= h
        for pos in self._sorted_keys:
            if h <= pos:
                return self._ring[pos]
        return self._ring[self._sorted_keys[0]]   # wrap around

    def node_key_distribution(self, keys: List[str]) -> Dict[str, int]:
        dist : Dict[str, int] = defaultdict(int)
        for key in keys:
            node = self.get_node(key)
            if node:
                dist[node] += 1
        return dict(dist)


# ─────────────────────────────────────────────
# CACHE NODE
# ─────────────────────────────────────────────

@dataclass
class CacheNodeStats:
    hits       : int = 0
    misses     : int = 0
    evictions  : int = 0
    total_ops  : int = 0

    @property
    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class CacheNode:
    """Simulates a single Redis node in the cluster."""

    def __init__(self, node_id: str, capacity: int = 10_000,
                 is_replica: bool = False):
        self.node_id   = node_id
        self.capacity  = capacity
        self.is_replica= is_replica
        self._store    : Dict[str, Tuple[Any, float]] = {}   # key → (value, expires_at)
        self.stats     = CacheNodeStats()
        self._alive    = True
        self._primary  : Optional[str] = None   # replica's primary node_id

    def get(self, key: str) -> Optional[Any]:
        if not self._alive:
            raise ConnectionError(f"Node {self.node_id} is down")
        self.stats.total_ops += 1
        entry = self._store.get(key)
        if entry is None or (entry[1] and time.time() > entry[1]):
            if key in self._store:
                del self._store[key]
            self.stats.misses += 1
            return None
        self.stats.hits += 1
        return entry[0]

    def set(self, key: str, value: Any, ttl_s: Optional[float] = 300.0):
        if not self._alive:
            raise ConnectionError(f"Node {self.node_id} is down")
        if len(self._store) >= self.capacity and key not in self._store:
            # LRU-approximate eviction: remove random entry
            evict_key = next(iter(self._store))
            del self._store[evict_key]
            self.stats.evictions += 1
        expires_at = time.time() + ttl_s if ttl_s else None
        self._store[key] = (value, expires_at)
        self.stats.total_ops += 1

    def delete(self, key: str):
        self._store.pop(key, None)
        self.stats.total_ops += 1

    def simulate_crash(self):
        self._alive = False

    def recover(self):
        self._alive = True

    @property
    def is_alive(self) -> bool:
        return self._alive

    def key_count(self) -> int:
        return len(self._store)

    def replicate_from(self, primary: "CacheNode"):
        """Sync state from primary (simplified async replication)."""
        self._store = dict(primary._store)


# ─────────────────────────────────────────────
# DISTRIBUTED CACHE CLUSTER
# ─────────────────────────────────────────────

class DistributedCacheCluster:
    """
    Redis Cluster-like distributed cache.
    Consistent hashing for key routing.
    Primary + replica per shard for HA.
    """

    def __init__(self, n_primaries: int = 3, replicas_per_primary: int = 1,
                 virtual_nodes: int = 150):
        self._ring      = ConsistentHashRing(virtual_nodes)
        self._primaries : Dict[str, CacheNode] = {}
        self._replicas  : Dict[str, CacheNode] = {}   # primary_id → replica
        self.total_gets = 0
        self.total_sets = 0
        self.errors     = 0

        for i in range(n_primaries):
            pid  = f"primary-{i}"
            rid  = f"replica-{i}"
            primary = CacheNode(pid, capacity=50_000)
            replica = CacheNode(rid, capacity=50_000, is_replica=True)
            replica._primary = pid
            self._primaries[pid]  = primary
            self._replicas[pid]   = replica
            self._ring.add_node(pid)

    def _route(self, key: str) -> Optional[CacheNode]:
        node_id = self._ring.get_node(key)
        if not node_id:
            return None
        primary = self._primaries.get(node_id)
        if primary and primary.is_alive:
            return primary
        # Failover to replica
        replica = self._replicas.get(node_id)
        if replica and replica.is_alive:
            return replica
        return None

    def get(self, key: str) -> Optional[Any]:
        self.total_gets += 1
        node = self._route(key)
        if not node:
            self.errors += 1
            return None
        try:
            return node.get(key)
        except ConnectionError:
            self.errors += 1
            return None

    def set(self, key: str, value: Any, ttl_s: float = 300.0):
        self.total_sets += 1
        node = self._route(key)
        if not node:
            self.errors += 1
            return
        try:
            node.set(key, value, ttl_s)
            # Async replicate to replica
            pid = node.node_id
            if pid in self._replicas:
                self._replicas[pid].set(key, value, ttl_s)
        except ConnectionError:
            self.errors += 1

    def delete(self, key: str):
        node = self._route(key)
        if node:
            node.delete(key)

    def node_distribution(self, sample_keys: List[str]) -> Dict[str, int]:
        return self._ring.node_key_distribution(sample_keys)

    def simulate_node_failure(self, primary_id: str):
        if primary_id in self._primaries:
            self._primaries[primary_id].simulate_crash()

    def simulate_node_recovery(self, primary_id: str):
        if primary_id in self._primaries:
            self._primaries[primary_id].recover()
            # Re-sync from replica
            replica = self._replicas.get(primary_id)
            primary = self._primaries.get(primary_id)
            if replica and primary:
                primary._store = dict(replica._store)

    def add_node(self, node_id: str):
        """Scale out: add new primary + replica."""
        pid = node_id
        rid = f"replica-{node_id}"
        self._primaries[pid] = CacheNode(pid, capacity=50_000)
        self._replicas[pid]  = CacheNode(rid, capacity=50_000, is_replica=True)
        self._ring.add_node(pid)

    def cluster_stats(self) -> Dict:
        total_keys  = sum(n.key_count() for n in self._primaries.values())
        alive_nodes = sum(1 for n in self._primaries.values() if n.is_alive)
        total_hits  = sum(n.stats.hits for n in self._primaries.values())
        total_misses= sum(n.stats.misses for n in self._primaries.values())
        return {
            "primaries"   : len(self._primaries),
            "alive"       : alive_nodes,
            "total_keys"  : total_keys,
            "total_hits"  : total_hits,
            "total_misses": total_misses,
            "hit_ratio"   : total_hits / max(1, total_hits + total_misses),
            "errors"      : self.errors,
        }


# ─────────────────────────────────────────────
# HOT KEY DETECTOR
# ─────────────────────────────────────────────

class HotKeyDetector:
    """
    Detects keys receiving disproportionately high traffic.
    Hot keys cause single-node bottlenecks in distributed caches.
    Mitigations: local replica per app instance, key sharding (key#suffix).
    """

    def __init__(self, window_size: int = 1000, threshold_ratio: float = 0.05):
        self._counts      : Dict[str, int] = defaultdict(int)
        self._total       = 0
        self.window_size  = window_size
        self.threshold    = threshold_ratio   # key is "hot" if > 5% of traffic

    def record(self, key: str):
        self._counts[key] += 1
        self._total       += 1

    def hot_keys(self) -> List[Tuple[str, float]]:
        """Returns [(key, traffic_ratio)] for keys exceeding threshold."""
        hot = []
        for key, count in self._counts.items():
            ratio = count / max(1, self._total)
            if ratio >= self.threshold:
                hot.append((key, ratio))
        return sorted(hot, key=lambda x: x[1], reverse=True)

    def suggest_sharding(self, key: str, n_shards: int = 4) -> List[str]:
        """Returns sharded key variants to distribute hot key across nodes."""
        return [f"{key}#{i}" for i in range(n_shards)]


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_distributed_cache():
    print("=" * 65)
    print("DISTRIBUTED CACHE DESIGN")
    print("=" * 65)

    random.seed(42)

    # ── Consistent Hash Ring ───────────────────
    print("\n[1] CONSISTENT HASHING — KEY DISTRIBUTION")
    print("─" * 55)
    ring = ConsistentHashRing(virtual_nodes=150)
    nodes = ["cache-node-0", "cache-node-1", "cache-node-2"]
    for n in nodes:
        ring.add_node(n)

    sample_keys = [f"user:{i}" for i in range(1000)]
    dist = ring.node_key_distribution(sample_keys)
    print(f"  {len(sample_keys)} keys distributed across {len(nodes)} nodes:")
    for node, count in sorted(dist.items()):
        bar = "█" * (count // 10)
        pct = count / len(sample_keys)
        print(f"    {node}: {count:4d} keys ({pct:.1%})  {bar}")

    # Add a node — minimal remapping
    ring.add_node("cache-node-3")
    dist_after = ring.node_key_distribution(sample_keys)
    remapped = sum(1 for k in sample_keys
                   if ring.get_node(k) != (lambda d, r: list(d.keys())[0] if d else None)(dist, ring))
    print(f"\n  After adding cache-node-3:")
    for node, count in sorted(dist_after.items()):
        print(f"    {node}: {count:4d} keys ({count/len(sample_keys):.1%})")

    # ── Cluster Operations ─────────────────────
    print("\n\n[2] DISTRIBUTED CACHE CLUSTER (3 primaries + replicas)")
    print("─" * 55)
    cluster = DistributedCacheCluster(n_primaries=3, replicas_per_primary=1)

    # Populate
    print("  Inserting 500 items...")
    for i in range(500):
        cluster.set(f"product:{i}", {"id": i, "price": i * 9.99})

    # Read
    for i in range(500):
        cluster.get(f"product:{i}")

    stats = cluster.cluster_stats()
    print(f"  Cluster: {stats['primaries']} primaries  {stats['alive']} alive")
    print(f"  Total keys: {stats['total_keys']}  Hit ratio: {stats['hit_ratio']:.1%}")

    # Show per-node distribution
    print(f"\n  Per-node key distribution:")
    for pid, node in cluster._primaries.items():
        print(f"    {pid}: {node.key_count()} keys  "
              f"hits={node.stats.hits} misses={node.stats.misses}")

    # ── Node Failure + Failover ────────────────
    print("\n\n[3] NODE FAILURE AND REPLICA FAILOVER")
    print("─" * 55)
    print("  Simulating primary-0 crash...")
    cluster.simulate_node_failure("primary-0")

    # Try to read — should hit replica
    hits_after_failure = 0
    for i in range(500):
        val = cluster.get(f"product:{i}")
        if val is not None:
            hits_after_failure += 1

    print(f"  Reads after primary-0 crash: {hits_after_failure}/500 served")
    print(f"  (keys on primary-0 served by replica-0 — HA maintained)")
    print(f"  Errors during failure: {cluster.errors}")

    # Recovery
    cluster.simulate_node_recovery("primary-0")
    print(f"\n  primary-0 recovered — synced from replica")

    # ── Hot Key Detection ──────────────────────
    print("\n\n[4] HOT KEY DETECTION")
    print("─" * 55)
    detector = HotKeyDetector(threshold_ratio=0.05)

    # Simulate 80/20 access: a few keys get huge traffic
    keys = [f"product:{i}" for i in range(100)]
    hot  = ["product:1", "product:2", "product:3"]   # very hot
    for _ in range(2000):
        if random.random() < 0.6:
            detector.record(random.choice(hot))
        else:
            detector.record(random.choice(keys))

    hot_keys = detector.hot_keys()
    print(f"  Traffic distribution (2000 requests, 100 keys):")
    for key, ratio in hot_keys:
        print(f"    {key}: {ratio:.1%} of traffic  ← HOT KEY")

    # Suggest sharding
    print(f"\n  Hot key mitigation — shard across multiple nodes:")
    for key, _ in hot_keys[:1]:
        shards = detector.suggest_sharding(key, n_shards=4)
        print(f"    {key} → {shards}")
        print(f"    (app randomly picks a shard → distributes load across 4 nodes)")

    # ── Architecture Patterns ──────────────────
    print("\n\n[5] REDIS CLUSTER ARCHITECTURE")
    print("─" * 55)
    print("  Hash Slots: 16,384 slots (0-16383)")
    print("  Sharding:   hash_slot = CRC16(key) % 16384")
    print()
    config_rows = [
        ("3 primaries",     "0-5460", "5461-10922", "10923-16383"),
        ("Node assignment", "primary-0","primary-1","primary-2"),
    ]
    for label, *vals in config_rows:
        print(f"  {label:<18} {vals[0]:<18} {vals[1]:<18} {vals[2]}")
    print()
    print("  Hash Tags: {user}.cart → always on same node")
    print("  Multi-key operations: MGET, pipelines require same slot")
    print()
    print("  Scale-out: add primary (rehashes ~25% of slots if 3→4 nodes)")
    print("  HA: sentinel or cluster mode promotes replica in <30s")

    # ── Comparison ────────────────────────────
    print("\n\n[6] REDIS CLUSTER vs MEMCACHED")
    print("─" * 55)
    comparison = [
        ("Sharding",       "Server-side (cluster)",    "Client-side only"),
        ("Replication",    "Built-in per shard",       "None (client responsibility)"),
        ("Data structures","Rich (lists, sets, hashes)","String only"),
        ("Persistence",    "RDB + AOF",                "None (pure cache)"),
        ("Failover",       "Auto (cluster)",           "Manual re-routing"),
        ("Multi-threaded", "Single-threaded per shard","Multi-threaded"),
        ("Memory overhead","~3x vs raw data",          "~1.5x"),
        ("Use case",       "General purpose, sessions","Pure high-throughput cache"),
    ]
    print(f"  {'Aspect':<20} {'Redis Cluster':<28} {'Memcached'}")
    print(f"  {'─'*70}")
    for aspect, redis, memcached in comparison:
        print(f"  {aspect:<20} {redis:<28} {memcached}")


if __name__ == "__main__":
    demonstrate_distributed_cache()
