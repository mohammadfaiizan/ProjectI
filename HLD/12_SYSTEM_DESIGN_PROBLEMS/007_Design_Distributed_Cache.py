"""
SYSTEM DESIGN: DISTRIBUTED CACHE (Redis/Memcached)
====================================================

Problem Statement:
Design a distributed in-memory cache that supports get/set/delete
operations at millions of QPS, with horizontal scalability and
configurable eviction policies.

Functional Requirements:
  - get(key) → value or null
  - set(key, value, ttl)
  - delete(key)
  - Support TTL (time-to-live) expiration
  - Support multiple eviction policies (LRU, LFU, TTL)

Non-Functional Requirements:
  - 1M QPS read, 100K QPS write
  - Sub-millisecond latency (< 1ms p99)
  - Horizontal scalability (add nodes without downtime)
  - HA: replication with automatic failover

Architecture Decisions:

  1. Sharding Strategy:
     Consistent hashing: add/remove nodes → minimal key migration.
     Virtual nodes (vnodes): better load balance.
     Hash slot (Redis Cluster): 16384 slots; each node owns a range.
     CRC16(key) % 16384 → slot → node.

  2. Eviction Policies:
     LRU: Least Recently Used. Good for temporal locality.
     LFU: Least Frequently Used. Better for Zipf access patterns.
     FIFO: First In First Out.
     TTL-only: expire strictly on TTL; no extra eviction.
     Random: simple, lower overhead.

  3. Replication:
     Leader-replica: sync writes to leader; async replicate to replicas.
     Read from replica: OK for eventual consistency.
     Redis Sentinel: monitors leaders; promotes replica on failure.
     Redis Cluster: built-in HA; 1 primary + 1+ replica per slot range.

  4. Persistence (Redis options):
     RDB: periodic snapshot. Fast restart, some data loss.
     AOF: append-only file of every write. Slower but durable.
     No persistence: pure cache. Lose all on restart (fine for cache).

  5. Cache Invalidation:
     TTL: simplest. Cache-aside pattern.
     Write-through: write DB and cache together.
     Write-behind: write cache; async write to DB.
     Cache-aside: app manages: read miss → load from DB → store in cache.

Common Failure Modes:
  Cache stampede: TTL expires; many requests hit DB simultaneously.
                  Fix: probabilistic early expiration (PER) or mutex lock.
  Hot key:        Single key gets 1M+ QPS. Single node becomes bottleneck.
                  Fix: local read-through cache (in-process); key sharding.
  Large key:      value > 10MB blocks serialization.
                  Fix: compress values; split into sub-keys; store in S3.
"""

from __future__ import annotations

import time
import hashlib
import random
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import OrderedDict
from enum import Enum


# ─────────────────────────────────────────────
# CACHE ENTRY
# ─────────────────────────────────────────────

@dataclass
class CacheEntry:
    key:        str
    value:      Any
    created_at: float
    expires_at: Optional[float]   # None = no expiry
    access_count: int = 0
    last_access:  float = field(default_factory=time.time)

    def is_expired(self) -> bool:
        return self.expires_at is not None and time.time() > self.expires_at

    def touch(self):
        self.access_count += 1
        self.last_access   = time.time()


# ─────────────────────────────────────────────
# EVICTION POLICY
# ─────────────────────────────────────────────

class EvictionPolicy(Enum):
    LRU    = "lru"
    LFU    = "lfu"
    FIFO   = "fifo"
    RANDOM = "random"
    TTL    = "ttl"


# ─────────────────────────────────────────────
# LRU CACHE (doubly-linked list + hash map)
# ─────────────────────────────────────────────

class LRUCache:
    """
    O(1) get/set/delete with LRU eviction.
    Uses Python's OrderedDict as a doubly-linked list + hash map.
    """

    def __init__(self, capacity: int):
        self._capacity = capacity
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock  = threading.RLock()
        self.hits   = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self.misses += 1
                return None
            if entry.is_expired():
                del self._store[key]
                self.misses += 1
                return None
            # Move to end (most recently used)
            self._store.move_to_end(key)
            entry.touch()
            self.hits += 1
            return entry.value

    def set(self, key: str, value: Any, ttl_s: Optional[float] = None):
        with self._lock:
            expires = time.time() + ttl_s if ttl_s else None
            if key in self._store:
                self._store.move_to_end(key)
                self._store[key].value      = value
                self._store[key].expires_at = expires
            else:
                if len(self._store) >= self._capacity:
                    evicted_key, _ = self._store.popitem(last=False)
                    self.evictions += 1
                self._store[key] = CacheEntry(key, value, time.time(), expires)

    def delete(self, key: str) -> bool:
        with self._lock:
            return self._store.pop(key, None) is not None

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def size(self) -> int:
        return len(self._store)


# ─────────────────────────────────────────────
# LFU CACHE (frequency-based eviction)
# ─────────────────────────────────────────────

class LFUCache:
    """
    Evicts least frequently used item.
    O(1) for get/set using frequency bucket approach.
    """

    def __init__(self, capacity: int):
        self._capacity = capacity
        self._min_freq  = 0
        self._key_to_val: Dict[str, CacheEntry] = {}
        self._key_to_freq: Dict[str, int] = {}
        self._freq_to_keys: Dict[int, OrderedDict] = {}
        self._lock = threading.RLock()
        self.hits = 0; self.misses = 0; self.evictions = 0

    def _update_freq(self, key: str):
        freq = self._key_to_freq[key]
        self._key_to_freq[key] = freq + 1
        self._freq_to_keys[freq].pop(key, None)
        if not self._freq_to_keys[freq] and freq == self._min_freq:
            self._min_freq += 1
        self._freq_to_keys.setdefault(freq + 1, OrderedDict())[key] = None

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._key_to_val:
                self.misses += 1
                return None
            entry = self._key_to_val[key]
            if entry.is_expired():
                self._evict_key(key)
                self.misses += 1
                return None
            self._update_freq(key)
            entry.touch()
            self.hits += 1
            return entry.value

    def set(self, key: str, value: Any, ttl_s: Optional[float] = None):
        with self._lock:
            if self._capacity <= 0:
                return
            expires = time.time() + ttl_s if ttl_s else None
            if key in self._key_to_val:
                self._key_to_val[key].value      = value
                self._key_to_val[key].expires_at = expires
                self._update_freq(key)
                return
            if len(self._key_to_val) >= self._capacity:
                self._evict_lfu()
            # Insert with freq=1
            self._key_to_val[key] = CacheEntry(key, value, time.time(), expires)
            self._key_to_freq[key] = 1
            self._freq_to_keys.setdefault(1, OrderedDict())[key] = None
            self._min_freq = 1

    def _evict_lfu(self):
        keys = self._freq_to_keys.get(self._min_freq, OrderedDict())
        if keys:
            evict_key = next(iter(keys))
            self._evict_key(evict_key)
            self.evictions += 1

    def _evict_key(self, key: str):
        freq = self._key_to_freq.pop(key, 0)
        self._freq_to_keys.get(freq, {}).pop(key, None)
        self._key_to_val.pop(key, None)

    def delete(self, key: str) -> bool:
        with self._lock:
            if key not in self._key_to_val:
                return False
            self._evict_key(key)
            return True

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


# ─────────────────────────────────────────────
# CONSISTENT HASHING
# ─────────────────────────────────────────────

class ConsistentHashRing:
    """
    Hash ring with virtual nodes for load balancing.
    Maps cache keys to nodes.
    """

    def __init__(self, nodes: List[str], vnodes: int = 150):
        self._vnodes  = vnodes
        self._ring:   Dict[int, str] = {}
        self._sorted: List[int]      = []
        for node in nodes:
            self._add_node(node)

    def _hash(self, key: str) -> int:
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def _add_node(self, node: str):
        for i in range(self._vnodes):
            vkey   = f"{node}:vnode:{i}"
            hashed = self._hash(vkey)
            self._ring[hashed] = node
        self._sorted = sorted(self._ring.keys())

    def add_node(self, node: str):
        self._add_node(node)

    def remove_node(self, node: str):
        to_remove = [h for h, n in self._ring.items() if n == node]
        for h in to_remove:
            del self._ring[h]
        self._sorted = sorted(self._ring.keys())

    def get_node(self, key: str) -> Optional[str]:
        if not self._ring:
            return None
        hashed = self._hash(key)
        # Binary search for first node ≥ hash
        lo, hi = 0, len(self._sorted)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._sorted[mid] < hashed:
                lo = mid + 1
            else:
                hi = mid
        idx = lo % len(self._sorted)
        return self._ring[self._sorted[idx]]

    def distribution(self, keys: List[str]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for k in keys:
            node = self.get_node(k)
            counts[node] = counts.get(node, 0) + 1
        return counts


# ─────────────────────────────────────────────
# DISTRIBUTED CACHE NODE
# ─────────────────────────────────────────────

class CacheNode:
    def __init__(self, node_id: str, capacity: int = 10_000,
                 policy: EvictionPolicy = EvictionPolicy.LRU):
        self.node_id  = node_id
        self._policy  = policy
        if policy == EvictionPolicy.LFU:
            self._cache = LFUCache(capacity)
        else:
            self._cache = LRUCache(capacity)

    def get(self, key: str) -> Optional[Any]:
        return self._cache.get(key)

    def set(self, key: str, value: Any, ttl_s: Optional[float] = None):
        self._cache.set(key, value, ttl_s)

    def delete(self, key: str) -> bool:
        return self._cache.delete(key)

    @property
    def hit_rate(self) -> float:
        return self._cache.hit_rate

    @property
    def size(self) -> int:
        return self._cache.size()


# ─────────────────────────────────────────────
# DISTRIBUTED CACHE CLUSTER
# ─────────────────────────────────────────────

class DistributedCache:
    """
    Sharded cache cluster using consistent hashing.
    Each shard is a CacheNode.
    """

    def __init__(self, node_ids: List[str], capacity_per_node: int = 10_000,
                 policy: EvictionPolicy = EvictionPolicy.LRU):
        self._nodes: Dict[str, CacheNode] = {
            nid: CacheNode(nid, capacity_per_node, policy)
            for nid in node_ids
        }
        self._ring = ConsistentHashRing(node_ids)

    def _node_for(self, key: str) -> Optional[CacheNode]:
        node_id = self._ring.get_node(key)
        return self._nodes.get(node_id) if node_id else None

    def get(self, key: str) -> Optional[Any]:
        node = self._node_for(key)
        return node.get(key) if node else None

    def set(self, key: str, value: Any, ttl_s: Optional[float] = None):
        node = self._node_for(key)
        if node:
            node.set(key, value, ttl_s)

    def delete(self, key: str) -> bool:
        node = self._node_for(key)
        return node.delete(key) if node else False

    def add_node(self, node_id: str, capacity: int = 10_000):
        self._nodes[node_id] = CacheNode(node_id, capacity)
        self._ring.add_node(node_id)

    def remove_node(self, node_id: str):
        self._ring.remove_node(node_id)
        self._nodes.pop(node_id, None)

    def cluster_stats(self) -> Dict:
        total_hits   = sum(n._cache.hits      for n in self._nodes.values())
        total_misses = sum(n._cache.misses     for n in self._nodes.values())
        total_evictions = sum(n._cache.evictions for n in self._nodes.values())
        return {
            "nodes":       len(self._nodes),
            "total_hits":  total_hits,
            "total_misses":total_misses,
            "evictions":   total_evictions,
            "hit_rate":    total_hits / max(total_hits + total_misses, 1),
            "per_node":    {nid: {"size": n.size, "hit_rate": f"{n.hit_rate:.2%}"}
                            for nid, n in self._nodes.items()},
        }


# ─────────────────────────────────────────────
# CACHE STAMPEDE PREVENTION
# ─────────────────────────────────────────────

class ProbabilisticEarlyRevalidation:
    """
    PER: Refresh cache entry early with probability that increases
    as expiry approaches. Prevents thundering herd on expiry.
    """

    def __init__(self, beta: float = 1.0):
        self._beta = beta   # higher = more aggressive prefetch

    def should_recompute(self, entry: CacheEntry,
                         computation_time_s: float) -> bool:
        """Returns True if this request should recompute early."""
        if entry.expires_at is None:
            return False
        delta = entry.expires_at - time.time()
        if delta <= 0:
            return True   # already expired
        # Probability of early recompute increases as delta shrinks
        prob = math.exp(-delta / (computation_time_s * self._beta))
        return random.random() < prob

import math


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_cache():
    print("=" * 65)
    print("SYSTEM DESIGN: DISTRIBUTED CACHE")
    print("=" * 65)

    random.seed(42)

    # ── LRU Cache ─────────────────────────────
    print("\n[1] LRU CACHE")
    print("─" * 55)

    lru = LRUCache(capacity=5)
    ops = [
        ("set", "user:1",   {"name": "Alice"},  60),
        ("set", "user:2",   {"name": "Bob"},    60),
        ("set", "user:3",   {"name": "Carol"},  60),
        ("get", "user:1",   None,               None),
        ("set", "user:4",   {"name": "Dave"},   60),
        ("set", "user:5",   {"name": "Eve"},    60),
        ("set", "user:6",   {"name": "Frank"},  60),   # evicts LRU (user:2)
        ("get", "user:2",   None,               None), # should miss
        ("get", "user:1",   None,               None), # should hit (recently accessed)
    ]
    for op, key, val, ttl in ops:
        if op == "set":
            lru.set(key, val, ttl)
            print(f"  SET {key}")
        else:
            result = lru.get(key)
            print(f"  GET {key} → {'HIT' if result else 'MISS'}")

    print(f"\n  Hit rate: {lru.hit_rate:.1%}  Size: {lru.size()}")
    print(f"  Evictions: {lru.evictions}")

    # ── LFU Cache ─────────────────────────────
    print("\n[2] LFU CACHE")
    print("─" * 55)

    lfu = LFUCache(capacity=3)
    # A is accessed most, B second, C least
    lfu.set("A", "apple")
    lfu.set("B", "banana")
    lfu.set("C", "cherry")
    for _ in range(5): lfu.get("A")
    for _ in range(3): lfu.get("B")
    lfu.get("C")

    # D should evict C (least frequently used)
    lfu.set("D", "durian")

    print(f"  A: {lfu.get('A')} (5 accesses → stays)")
    print(f"  B: {lfu.get('B')} (3 accesses → stays)")
    print(f"  C: {lfu.get('C')} (1 access  → evicted)")
    print(f"  D: {lfu.get('D')} (new item  → inserted)")

    # ── Consistent Hashing ────────────────────
    print("\n[3] CONSISTENT HASHING")
    print("─" * 55)

    nodes3 = ["node-1", "node-2", "node-3"]
    ring   = ConsistentHashRing(nodes3, vnodes=150)

    test_keys = [f"key_{i}" for i in range(1000)]
    dist_3    = ring.distribution(test_keys)
    print(f"  3 nodes, 150 vnodes each:")
    for node, count in sorted(dist_3.items()):
        pct = count / len(test_keys) * 100
        bar = "█" * int(pct / 2)
        print(f"    {node}: {count:>4} keys ({pct:.1f}%) {bar}")

    # Add a node
    ring.add_node("node-4")
    dist_4   = ring.distribution(test_keys)
    migrated = sum(1 for k in test_keys
                   if ring.get_node(k) != ConsistentHashRing(nodes3).get_node(k))
    print(f"\n  After adding node-4 (total 4 nodes):")
    for node, count in sorted(dist_4.items()):
        pct = count / len(test_keys) * 100
        print(f"    {node}: {count:>4} keys ({pct:.1f}%)")

    # ── Distributed Cache ─────────────────────
    print("\n[4] DISTRIBUTED CACHE CLUSTER")
    print("─" * 55)

    cluster = DistributedCache(
        node_ids=["cache-1", "cache-2", "cache-3"],
        capacity_per_node=1000,
        policy=EvictionPolicy.LRU,
    )

    # Simulate mixed workload (Zipf-like: 20% of keys = 80% of reads)
    hot_keys  = [f"product:{i}" for i in range(20)]
    cold_keys = [f"product:{i}" for i in range(20, 100)]

    # Warm cache
    for k in hot_keys + cold_keys:
        cluster.set(k, f"data_for_{k}", ttl_s=300)

    # Simulate reads
    for _ in range(10000):
        if random.random() < 0.8:
            key = random.choice(hot_keys)
        else:
            key = random.choice(cold_keys)
        cluster.get(key)

    stats = cluster.cluster_stats()
    print(f"  Nodes:      {stats['nodes']}")
    print(f"  Hit rate:   {stats['hit_rate']:.2%}")
    print(f"  Evictions:  {stats['evictions']}")
    print("  Per node:")
    for nid, ns in stats["per_node"].items():
        print(f"    {nid}: size={ns['size']}  hit_rate={ns['hit_rate']}")

    # ── TTL Expiry ────────────────────────────
    print("\n[5] TTL EXPIRATION")
    print("─" * 55)

    short_cache = LRUCache(capacity=100)
    short_cache.set("session:abc", "user_data", ttl_s=0.05)   # 50ms TTL
    print(f"  Before TTL: {short_cache.get('session:abc')}")
    time.sleep(0.06)
    print(f"  After  TTL: {short_cache.get('session:abc')} (expired)")

    # ── Cache Patterns ────────────────────────
    print("\n[6] CACHE DESIGN PATTERNS")
    print("─" * 55)

    patterns = [
        ("Cache-Aside",     "Read: miss→load DB→store cache. Write: invalidate cache"),
        ("Write-Through",   "Write DB AND cache together; consistent but slower writes"),
        ("Write-Behind",    "Write cache; async flush to DB; fast writes, risk of loss"),
        ("Read-Through",    "Cache is the only read path; cache loads from DB on miss"),
        ("Refresh-Ahead",   "Proactively refresh popular keys before expiry"),
    ]
    for name, desc in patterns:
        print(f"  {name:<18} {desc}")

    # ── Failure Modes ─────────────────────────
    print("\n[7] COMMON FAILURE MODES AND FIXES")
    print("─" * 55)

    failures = [
        ("Cache Stampede",  "Many requests miss at same time → DB overload"),
        ("  Fix:",          "PER early revalidation; mutex lock; stale-while-revalidate"),
        ("Hot Key",         "1 key gets 1M+ QPS → single shard bottleneck"),
        ("  Fix:",          "Local in-process cache; shard key (key_1, key_2...key_N)"),
        ("Large Value",     "10MB+ value blocks I/O, causes OOM"),
        ("  Fix:",          "Compress (zstd); split into chunks; store blob in S3"),
        ("Cache Avalanche", "Many TTLs expire together → DB spike"),
        ("  Fix:",          "Add TTL jitter: ttl = base + random(0, base*0.1)"),
    ]
    for label, desc in failures:
        print(f"  {label:<20} {desc}")


if __name__ == "__main__":
    demonstrate_cache()
