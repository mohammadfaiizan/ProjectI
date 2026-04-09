"""
MULTI-LEVEL CACHE ARCHITECTURE
=================================

Problem Statement:
No single cache layer is optimal for all access patterns. A request for a
product page might benefit from: L1 (in-process, 0ms, per-instance),
L2 (Redis cluster, 0.5ms, shared), and L3 (CDN, 5ms, geo-distributed).
Multi-level caching stacks layers to maximize hit ratio and minimize latency.

Cache Hierarchy (smallest/fastest to largest/slowest):
  L0: CPU cache (hardware) — μs, managed by CPU
  L1: In-process (dict/OrderedDict) — ns-μs, per app instance, not shared
  L2: Distributed (Redis/Memcached) — 0.5-2ms, shared, HA
  L3: CDN edge (Cloudflare/Fastly) — 5-50ms, geographic, HTTP only
  L4: Origin DB / API — 10-500ms, canonical source of truth

Cache Population Strategy:
  On L1 miss → check L2 → on L2 miss → check L3 → on L3 miss → origin
  Populate upward: L3 hit → also set L2 + L1 for next request

Cache Eviction Across Levels:
  L1 eviction: LRU with small capacity (instance memory limit)
  L2 eviction: LRU + TTL (Redis maxmemory-policy: volatile-lru)
  L3 eviction: TTL-based (Cache-Control headers)

Invalidation Propagation:
  Write → invalidate L1 (this instance) + L2 (all instances via key delete)
  L1 invalidation: only affects current instance; other instances still have stale L1
  Fix: short L1 TTL (5-30s) to limit cross-instance stale window
  Fix: pub/sub invalidation broadcast to all instances' L1 caches

Write Strategy per Level:
  L1: write-aside (not written unless read)
  L2: write-invalidate or write-through
  L3: write-around (set via HTTP response headers, not direct write)

Local vs Shared Cache Trade-offs:
  L1 (in-process): fastest, but each instance has separate cache
    → 10 instances = 10 copies of same data
    → scales linearly but may have stale data per-instance
  L2 (Redis): shared source of truth for all instances
    → 1 copy, always consistent
    → network hop overhead
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from collections import OrderedDict
import time
import random
import threading


# ─────────────────────────────────────────────
# INDIVIDUAL CACHE LEVELS
# ─────────────────────────────────────────────

@dataclass
class CacheLevelConfig:
    name         : str
    capacity     : int
    ttl_s        : float
    latency_ms   : float   # simulated latency for this level
    level         : int    # 1 = fastest


class L1InProcessCache:
    """Per-instance in-process LRU cache (thread-safe)."""

    def __init__(self, config: CacheLevelConfig):
        self.config = config
        self._store : OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self._lock  = threading.Lock()
        self.hits   = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self.misses += 1
                return None
            value, expires_at = entry
            if time.time() > expires_at:
                del self._store[key]
                self.misses += 1
                return None
            self._store.move_to_end(key)
            self.hits += 1
            return value

    def set(self, key: str, value: Any):
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            elif len(self._store) >= self.config.capacity:
                self._store.popitem(last=False)
            self._store[key] = (value, time.time() + self.config.ttl_s)

    def delete(self, key: str):
        with self._lock:
            self._store.pop(key, None)

    def size(self) -> int:
        return len(self._store)

    @property
    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class L2DistributedCache:
    """Simulated Redis distributed cache (shared across instances)."""

    def __init__(self, config: CacheLevelConfig):
        self.config   = config
        self._store   : Dict[str, Tuple[Any, float]] = {}
        self._lock    = threading.Lock()
        self.hits     = 0
        self.misses   = 0

    def get(self, key: str) -> Optional[Any]:
        time.sleep(self.config.latency_ms / 1000)   # network round-trip
        with self._lock:
            entry = self._store.get(key)
            if entry is None or time.time() > entry[1]:
                if entry:
                    del self._store[key]
                self.misses += 1
                return None
            self.hits += 1
            return entry[0]

    def set(self, key: str, value: Any):
        time.sleep(self.config.latency_ms / 1000)
        with self._lock:
            if len(self._store) >= self.config.capacity and key not in self._store:
                # Evict random (approximate LRU)
                evict = next(iter(self._store))
                del self._store[evict]
            self._store[key] = (value, time.time() + self.config.ttl_s)

    def delete(self, key: str):
        with self._lock:
            self._store.pop(key, None)

    def size(self) -> int:
        return len(self._store)

    @property
    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class OriginDataSource:
    """The authoritative data source (database or API)."""

    def __init__(self, latency_ms: float = 20.0):
        self.latency_ms = latency_ms
        self.reads      = 0
        self._data : Dict[str, Any] = {
            f"product:{i}": {"id": i, "name": f"Product {i}", "price": i * 9.99}
            for i in range(1, 10001)
        }

    def get(self, key: str) -> Optional[Any]:
        time.sleep(random.uniform(self.latency_ms * 0.5, self.latency_ms * 1.5) / 1000)
        self.reads += 1
        return self._data.get(key)


# ─────────────────────────────────────────────
# MULTI-LEVEL CACHE
# ─────────────────────────────────────────────

class MultiLevelCache:
    """
    L1 (in-process) → L2 (Redis) → Origin
    On miss at level N: go to N+1, populate all levels above on return.
    """

    def __init__(self, l1: L1InProcessCache, l2: L2DistributedCache, origin: OriginDataSource):
        self.l1     = l1
        self.l2     = l2
        self.origin = origin
        self.l1_hits = 0
        self.l2_hits = 0
        self.origin_hits = 0
        self.total_latency_ms : List[float] = []

    def get(self, key: str) -> Optional[Any]:
        start = time.perf_counter()

        # L1 check
        val = self.l1.get(key)
        if val is not None:
            self.l1_hits += 1
            self.total_latency_ms.append((time.perf_counter() - start) * 1000)
            return val

        # L2 check
        val = self.l2.get(key)
        if val is not None:
            self.l2_hits += 1
            self.l1.set(key, val)   # warm L1
            self.total_latency_ms.append((time.perf_counter() - start) * 1000)
            return val

        # Origin
        val = self.origin.get(key)
        if val is not None:
            self.origin_hits += 1
            self.l2.set(key, val)   # warm L2
            self.l1.set(key, val)   # warm L1
        self.total_latency_ms.append((time.perf_counter() - start) * 1000)
        return val

    def invalidate(self, key: str):
        """Invalidate across all levels."""
        self.l1.delete(key)
        self.l2.delete(key)
        # Origin is the source of truth — unchanged

    @property
    def total_requests(self) -> int:
        return self.l1_hits + self.l2_hits + self.origin_hits

    @property
    def avg_latency_ms(self) -> float:
        return sum(self.total_latency_ms) / len(self.total_latency_ms) if self.total_latency_ms else 0.0

    def report(self):
        total = max(1, self.total_requests)
        print(f"  L1 hits: {self.l1_hits:5d} ({self.l1_hits/total:.1%})  avg lat: ~0.01ms")
        print(f"  L2 hits: {self.l2_hits:5d} ({self.l2_hits/total:.1%})  avg lat: ~1ms")
        print(f"  Origin:  {self.origin_hits:5d} ({self.origin_hits/total:.1%})  avg lat: ~20ms")
        print(f"  Overall avg latency: {self.avg_latency_ms:.2f}ms")


# ─────────────────────────────────────────────
# CROSS-INSTANCE L1 INVALIDATION
# ─────────────────────────────────────────────

class InvalidationBus:
    """
    Pub/Sub bus for broadcasting L1 cache invalidation to all instances.
    In production: Redis Pub/Sub or internal event bus.
    """

    def __init__(self):
        self._subscribers : List[L1InProcessCache] = []
        self.broadcasts   = 0

    def subscribe(self, cache: L1InProcessCache):
        self._subscribers.append(cache)

    def broadcast_invalidation(self, key: str):
        """Delete key from all subscribed L1 caches."""
        for cache in self._subscribers:
            cache.delete(key)
        self.broadcasts += 1


# ─────────────────────────────────────────────
# CACHE EFFECTIVENESS SIMULATION
# ─────────────────────────────────────────────

class CacheAccessSimulator:
    """Simulates realistic access patterns for multi-level cache analysis."""

    def __init__(self, n_keys: int, hot_ratio: float = 0.2, hot_traffic: float = 0.8):
        self.n_keys     = n_keys
        self.hot_keys   = [f"product:{i}" for i in range(1, int(n_keys * hot_ratio) + 1)]
        self.cold_keys  = [f"product:{i}" for i in range(int(n_keys * hot_ratio) + 1, n_keys + 1)]
        self.hot_traffic= hot_traffic

    def next_key(self) -> str:
        if random.random() < self.hot_traffic:
            return random.choice(self.hot_keys)
        return random.choice(self.cold_keys)


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_multi_level_cache():
    print("=" * 65)
    print("MULTI-LEVEL CACHE ARCHITECTURE")
    print("=" * 65)

    random.seed(42)

    # ── Setup ──────────────────────────────────
    l1_cfg = CacheLevelConfig("L1-InProcess", capacity=50,   ttl_s=30.0,  latency_ms=0.01, level=1)
    l2_cfg = CacheLevelConfig("L2-Redis",     capacity=5000, ttl_s=300.0, latency_ms=1.0,  level=2)

    l1     = L1InProcessCache(l1_cfg)
    l2     = L2DistributedCache(l2_cfg)
    origin = OriginDataSource(latency_ms=20.0)
    mlc    = MultiLevelCache(l1, l2, origin)

    sim = CacheAccessSimulator(n_keys=200, hot_ratio=0.1, hot_traffic=0.85)

    # ── Cold Start ─────────────────────────────
    print("\n[1] COLD START — FIRST 20 REQUESTS")
    print("─" * 55)
    for i in range(20):
        mlc.get(sim.next_key())

    print(f"  After 20 requests:")
    mlc.report()
    print(f"  L1 cache size: {l1.size()}")
    print(f"  L2 cache size: {l2.size()}")

    # ── Steady State ───────────────────────────
    print("\n\n[2] STEADY STATE — 500 REQUESTS")
    print("─" * 55)
    # Reset stats
    mlc.l1_hits = mlc.l2_hits = mlc.origin_hits = 0
    mlc.total_latency_ms = []
    l1.hits = l1.misses = l2.hits = l2.misses = 0

    for _ in range(500):
        mlc.get(sim.next_key())

    print(f"  After 500 requests (hot_keys=10% of 200, 85% traffic):")
    mlc.report()
    print(f"  L1 size: {l1.size()}/{l1_cfg.capacity}  L2 size: {l2.size()}/{l2_cfg.capacity}")

    # ── Latency Comparison ─────────────────────
    print("\n\n[3] LATENCY PER CACHE LEVEL")
    print("─" * 55)
    levels = [
        ("L1 (in-process)", 0.01,   "per-instance dict, no network"),
        ("L2 (Redis local)","1.0",   "network round-trip, shared"),
        ("L2 (Redis x-AZ)", "2-5",   "cross-AZ Redis replica"),
        ("Origin (DB SSD)", "10-50", "query + disk read"),
        ("Origin (DB HDD)", "50-200","query + seek + read"),
    ]
    for level, lat, note in levels:
        bar = "█" * min(40, int(float(str(lat).split("-")[0]) * 2))
        print(f"  {level:<20} {str(lat)+'ms':<12} {note}  {bar}")

    # ── Cross-Instance Invalidation ────────────
    print("\n\n[4] CROSS-INSTANCE L1 INVALIDATION (pub/sub)")
    print("─" * 55)

    # Simulate 3 app instances with their own L1 caches
    bus      = InvalidationBus()
    instance_caches = [L1InProcessCache(l1_cfg) for _ in range(3)]
    for c in instance_caches:
        bus.subscribe(c)

    # All instances cache the same key
    for ic in instance_caches:
        ic.set("product:42", {"name": "Hot Product"})

    hits_before = sum(1 for ic in instance_caches if ic.get("product:42") is not None)
    print(f"  3 instances, all have product:42 cached: {hits_before}/3 hit")

    # Product updated → broadcast invalidation
    bus.broadcast_invalidation("product:42")
    hits_after = sum(1 for ic in instance_caches if ic.get("product:42") is not None)
    print(f"  After broadcast invalidation: {hits_after}/3 hit")
    print(f"  Broadcasts sent: {bus.broadcasts}")
    print(f"\n  Without pub/sub: L1 stays stale for up to {l1_cfg.ttl_s}s per instance")
    print(f"  With pub/sub: all instances' L1 caches cleared within <1ms")

    # ── Cache Sizing Guide ─────────────────────
    print("\n\n[5] MULTI-LEVEL CACHE SIZING GUIDE")
    print("─" * 55)
    sizing = [
        ("L1 (per instance)", "Small (50-500 keys)", "Only hottest keys",
         "Keep hit ratio, not capacity — bigger L1 = more memory per instance"),
        ("L2 (Redis)",        "Medium (10K-10M keys)", "All hot + warm keys",
         "Add nodes to cluster as dataset grows"),
        ("L3 (CDN)",          "Unlimited (CDN manages)", "HTTP-cacheable responses",
         "CDN capacity = budget — pay per storage and transfer"),
    ]
    for level, size, what, note in sizing:
        print(f"\n  {level}:")
        print(f"    Size: {size}  What: {what}")
        print(f"    Note: {note}")

    # ── Architecture Diagram ───────────────────
    print("\n\n[6] MULTI-LEVEL CACHE REQUEST FLOW")
    print("─" * 55)
    flow = """
  Browser ──────────────────────────────────────────────────────┐
           └→ CDN Edge (L3, 5-50ms)                             │
                └[miss]→ App Server                             │
                            └→ L1 Cache (in-process, 0.01ms)   │
                                └[miss]→ L2 Redis (1ms)         │
                                          └[miss]→ Database     │
                                                    (10-500ms)  │
  Write path:                                                    │
    DB write → invalidate L2 → broadcast invalidate L1 ─────────┘
               CDN invalidated via surrogate key purge
    """
    for line in flow.strip().split("\n"):
        print(line)

    # ── Trade-offs ────────────────────────────
    print("\n\n[7] MULTI-LEVEL CACHE TRADE-OFFS")
    print("─" * 55)
    tradeoffs = [
        ("L1 hit ratio vs memory", "Larger L1 → better hit ratio but more heap per instance"),
        ("L1 staleness",           "Short TTL (5-30s) = near-consistent but more L2 load"),
        ("L2 vs origin load",      "Well-tuned L2 absorbs 80-95% of origin load"),
        ("Invalidation complexity", "Each level needs invalidation logic — bugs = stale data"),
        ("Debugging",              "Multi-level makes cache miss debugging harder — add metrics"),
        ("Cold start",             "All instances start with empty L1 → L2 absorbs spike"),
    ]
    for concern, note in tradeoffs:
        print(f"  {concern:<30} {note}")


if __name__ == "__main__":
    demonstrate_multi_level_cache()
