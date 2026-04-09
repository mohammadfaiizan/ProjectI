"""
CACHING FUNDAMENTALS AND BENEFITS
=====================================

Problem Statement:
Every database read takes 1-10ms. At 100K QPS that's 100-1000 database
connections running simultaneously — most DBs saturate well before that.
A cache intercepts repeat reads and returns results from memory in <1ms,
reducing DB load by 90%+ for typical read-heavy workloads.

What is a Cache?
  A fast, temporary store that holds copies of frequently accessed data.
  Sits between the application and the origin (DB, API, file system).

Cache Hit / Miss:
  Hit  : requested key exists in cache → return cached value (fast)
  Miss : key not found → fetch from origin → optionally store in cache

Hit Ratio:
  hit_ratio = hits / (hits + misses)
  Target: > 90% for production caches (80% is acceptable minimum)
  Factors: cache size, data hotness (Pareto: 20% keys = 80% traffic)

Latency Comparison:
  L1 CPU cache      : ~0.5 ns
  L2 CPU cache      : ~7 ns
  RAM (in-process)  : ~100 ns
  Redis (network)   : ~0.5 ms   (500 µs)
  SSD database read : ~1-5 ms
  HDD database read : ~10-50 ms
  Remote DB (cloud) : ~5-30 ms

Benefits:
  1. Latency reduction : sub-ms vs ms
  2. Throughput increase: 10-100x QPS
  3. Cost reduction    : fewer DB reads = cheaper DB tier
  4. Resilience        : cache absorbs traffic during DB overload

Caching Layers (hierarchy):
  Browser Cache → CDN Edge → API Gateway Cache → App In-Process Cache
  → Distributed Cache (Redis) → Database Query Cache → DB Disk Cache
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from collections import OrderedDict
import time
import random
import statistics


@dataclass
class CacheStats:
    hits      : int = 0
    misses    : int = 0
    sets      : int = 0
    evictions : int = 0
    total_get_ns: float = 0.0   # nanoseconds for cache gets
    total_origin_ns: float = 0.0

    @property
    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    @property
    def avg_cache_latency_us(self) -> float:
        total = self.hits + self.misses
        return (self.total_get_ns / total / 1000) if total else 0.0

    def report(self, name: str):
        total = self.hits + self.misses
        print(f"  [{name}] requests={total}  hits={self.hits}  misses={self.misses}  "
              f"hit_ratio={self.hit_ratio:.1%}  evictions={self.evictions}")


# ─────────────────────────────────────────────
# IN-PROCESS CACHE (L1 — fastest)
# ─────────────────────────────────────────────

class InProcessCache:
    """
    Local in-process dictionary cache (L1).
    Fastest possible — no network — but not shared across instances.
    """

    def __init__(self, max_size: int = 1000, default_ttl_s: float = 300.0):
        self._store     : OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self.max_size   = max_size
        self.default_ttl= default_ttl_s
        self.stats      = CacheStats()

    def get(self, key: str) -> Optional[Any]:
        start = time.perf_counter_ns()
        entry = self._store.get(key)
        if entry is None:
            self.stats.misses += 1
            self.stats.total_get_ns += time.perf_counter_ns() - start
            return None
        value, expires_at = entry
        if time.time() > expires_at:
            del self._store[key]
            self.stats.misses += 1
            self.stats.total_get_ns += time.perf_counter_ns() - start
            return None
        # Move to end (LRU)
        self._store.move_to_end(key)
        self.stats.hits += 1
        self.stats.total_get_ns += time.perf_counter_ns() - start
        return value

    def set(self, key: str, value: Any, ttl_s: float = None) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        elif len(self._store) >= self.max_size:
            self._store.popitem(last=False)   # evict LRU (first)
            self.stats.evictions += 1
        ttl = ttl_s or self.default_ttl
        self._store[key] = (value, time.time() + ttl)
        self.stats.sets += 1

    def delete(self, key: str):
        self._store.pop(key, None)

    def size(self) -> int:
        return len(self._store)


# ─────────────────────────────────────────────
# SIMULATED REDIS CACHE (L2 — distributed)
# ─────────────────────────────────────────────

class SimulatedRedisCache:
    """
    Simulates a distributed cache (Redis) with network latency.
    Shared across multiple app instances, but adds ~0.5ms network overhead.
    """
    NETWORK_LATENCY_MS = 0.5

    def __init__(self, max_size: int = 100_000):
        self._store   : Dict[str, Tuple[Any, float]] = {}
        self.max_size = max_size
        self.stats    = CacheStats()

    def get(self, key: str) -> Optional[Any]:
        time.sleep(self.NETWORK_LATENCY_MS / 1000)   # simulate network
        entry = self._store.get(key)
        if entry is None or time.time() > entry[1]:
            if key in self._store:
                del self._store[key]
            self.stats.misses += 1
            return None
        self.stats.hits += 1
        return entry[0]

    def set(self, key: str, value: Any, ttl_s: float = 300.0):
        if len(self._store) >= self.max_size and key not in self._store:
            # Simple eviction: remove random entry
            evict_key = next(iter(self._store))
            del self._store[evict_key]
            self.stats.evictions += 1
        self._store[key] = (value, time.time() + ttl_s)
        self.stats.sets += 1

    def delete(self, key: str):
        self._store.pop(key, None)

    def exists(self, key: str) -> bool:
        return key in self._store and time.time() <= self._store[key][1]


# ─────────────────────────────────────────────
# SIMULATED DATABASE (origin)
# ─────────────────────────────────────────────

class SimulatedDatabase:
    """Origin data store — slow but always correct."""
    READ_LATENCY_MS = 10.0   # 10ms average

    def __init__(self):
        self._data   = {f"user:{i}": {"id": i, "name": f"User{i}", "email": f"u{i}@ex.com"}
                        for i in range(1, 1001)}
        self.reads   = 0
        self.total_latency_ms = 0.0

    def get(self, key: str) -> Optional[Any]:
        latency = random.uniform(5, 20)   # 5-20ms
        time.sleep(latency / 1000)
        self.reads += 1
        self.total_latency_ms += latency
        return self._data.get(key)

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.reads if self.reads else 0.0


# ─────────────────────────────────────────────
# MULTI-LAYER CACHE
# ─────────────────────────────────────────────

class MultiLayerCache:
    """
    L1 (in-process) → L2 (Redis) → Origin (DB)
    Check each layer in order; on miss, populate all layers above origin.
    """

    def __init__(self, l1: InProcessCache, l2: SimulatedRedisCache, db: SimulatedDatabase):
        self.l1 = l1
        self.l2 = l2
        self.db = db
        self.l1_hits = 0
        self.l2_hits = 0
        self.db_hits = 0

    def get(self, key: str) -> Optional[Any]:
        # L1 check
        val = self.l1.get(key)
        if val is not None:
            self.l1_hits += 1
            return val

        # L2 check
        val = self.l2.get(key)
        if val is not None:
            self.l2_hits += 1
            self.l1.set(key, val, ttl_s=60.0)   # warm L1
            return val

        # DB (origin)
        val = self.db.get(key)
        self.db_hits += 1
        if val is not None:
            self.l2.set(key, val, ttl_s=300.0)
            self.l1.set(key, val, ttl_s=60.0)
        return val


# ─────────────────────────────────────────────
# CACHE EFFECTIVENESS ANALYZER
# ─────────────────────────────────────────────

class CacheEffectivenessAnalyzer:
    """Measures cache performance under different access patterns."""

    def __init__(self, db: SimulatedDatabase):
        self.db = db

    def simulate_uniform_access(self, n_requests: int, n_keys: int, cache_size: int) -> CacheStats:
        """Uniform random access — all keys equally likely."""
        cache = InProcessCache(max_size=cache_size, default_ttl_s=3600)
        for _ in range(n_requests):
            key = f"user:{random.randint(1, n_keys)}"
            if cache.get(key) is None:
                val = {"id": key, "data": "..."}
                cache.set(key, val)
        return cache.stats

    def simulate_pareto_access(self, n_requests: int, n_keys: int, cache_size: int) -> CacheStats:
        """80/20 Pareto access — 20% of keys get 80% of requests."""
        cache  = InProcessCache(max_size=cache_size, default_ttl_s=3600)
        hot_keys  = [f"user:{i}" for i in range(1, int(n_keys * 0.2) + 1)]
        cold_keys = [f"user:{i}" for i in range(int(n_keys * 0.2) + 1, n_keys + 1)]
        for _ in range(n_requests):
            if random.random() < 0.8:
                key = random.choice(hot_keys)
            else:
                key = random.choice(cold_keys)
            if cache.get(key) is None:
                cache.set(key, {"id": key})
        return cache.stats

    def cache_size_vs_hit_ratio(self, n_requests: int, n_keys: int) -> List[Tuple[int, float]]:
        """How does hit ratio scale with cache size?"""
        results = []
        for pct in [5, 10, 20, 30, 50, 80, 100]:
            size  = max(1, int(n_keys * pct / 100))
            stats = self.simulate_pareto_access(n_requests, n_keys, size)
            results.append((pct, stats.hit_ratio))
        return results


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_caching_fundamentals():
    print("=" * 65)
    print("CACHING FUNDAMENTALS AND BENEFITS")
    print("=" * 65)

    random.seed(42)

    # ── Latency Comparison ─────────────────────
    print("\n[1] LATENCY HIERARCHY")
    print("─" * 55)
    latencies = [
        ("L1 CPU cache",        0.0005, "ns"),
        ("L2 CPU cache",        0.007,  "µs"),
        ("RAM (in-process)",    0.1,    "µs"),
        ("In-process dict",     0.5,    "µs"),
        ("Redis (local)",       500,    "µs"),
        ("Redis (cross-AZ)",    1500,   "µs"),
        ("PostgreSQL SSD",      5000,   "µs"),
        ("PostgreSQL HDD",      30000,  "µs"),
    ]
    for name, lat, unit in latencies:
        bar = "█" * min(50, int(lat / 600))
        print(f"  {name:<25} {lat:>8.1f} {unit}  {bar}")

    # ── In-Process Cache ─────────────────────
    print("\n\n[2] IN-PROCESS CACHE (L1)")
    print("─" * 55)
    l1    = InProcessCache(max_size=100, default_ttl_s=60)
    db    = SimulatedDatabase()

    keys  = [f"user:{i}" for i in range(1, 21)]
    print("  Warming cache with 20 users...")
    for key in keys:
        val = db.get(key)
        l1.set(key, val)

    # Access pattern: hot keys repeated
    print("  Accessing 200 requests (same 20 keys repeatedly)...")
    for _ in range(200):
        key = random.choice(keys)
        result = l1.get(key)
        if result is None:
            val = db.get(key)
            l1.set(key, val)

    l1.stats.report("L1 In-Process")
    print(f"    DB reads:       {db.reads}  (avg latency: {db.avg_latency_ms:.1f}ms)")
    print(f"    Cache size:     {l1.size()}/{l1.max_size}")

    # ── Hit Ratio vs Cache Size ───────────────
    print("\n\n[3] HIT RATIO vs CACHE SIZE (Pareto 80/20 access)")
    print("─" * 55)
    analyzer = CacheEffectivenessAnalyzer(db)
    results  = analyzer.cache_size_vs_hit_ratio(n_requests=1000, n_keys=200)
    print(f"  {'Cache Size':<12} {'Hit Ratio':<12} {'Visual'}")
    for pct, ratio in results:
        bar = "█" * int(ratio * 40)
        print(f"  {pct:>4}% of keys  {ratio:.1%}         {bar}")

    print("\n  Insight: 20% cache size → 70%+ hit ratio (Pareto distribution)")
    print("  You don't need to cache all data — just the hot 20%")

    # ── Uniform vs Pareto ─────────────────────
    print("\n\n[4] UNIFORM vs PARETO ACCESS PATTERN (cache_size=20%, keys=100)")
    print("─" * 55)
    uniform = analyzer.simulate_uniform_access(500, 100, 20)
    pareto  = analyzer.simulate_pareto_access(500, 100, 20)
    uniform.report("Uniform (random)")
    pareto.report("Pareto (80/20)")
    print(f"\n  Pareto hit ratio is {pareto.hit_ratio / max(uniform.hit_ratio, 0.01):.1f}x better")
    print("  Real production traffic is highly skewed → cache is very effective")

    # ── Caching Benefits Math ─────────────────
    print("\n\n[5] CACHING ECONOMICS")
    print("─" * 55)
    qps         = 10_000
    hit_ratio   = 0.90
    db_cost_ms  = 10.0
    cache_cost_ms = 0.5
    cache_qps   = qps * hit_ratio
    db_qps      = qps * (1 - hit_ratio)
    print(f"  Total QPS: {qps:,}  hit_ratio: {hit_ratio:.0%}")
    print(f"  Cache serves: {cache_qps:,.0f} QPS at {cache_cost_ms}ms  (cost: {cache_qps * cache_cost_ms:,.0f} ms·req/s)")
    print(f"  DB serves:    {db_qps:,.0f} QPS at {db_cost_ms}ms   (cost: {db_qps * db_cost_ms:,.0f} ms·req/s)")
    print(f"\n  Without cache: all {qps:,} QPS hit DB → {qps * db_cost_ms:,.0f} ms·req/s DB load")
    print(f"  With cache:    only {db_qps:,.0f} QPS hit DB → {db_qps * db_cost_ms:,.0f} ms·req/s DB load")
    print(f"  DB load reduction: {(1 - db_qps/qps):.0%}")

    # ── Caching Layers ────────────────────────
    print("\n\n[6] CACHING LAYER HIERARCHY")
    print("─" * 55)
    layers = [
        ("Browser Cache",       "Static assets (CSS/JS/images)",     "Hours-days",   "Client RAM"),
        ("CDN Edge Cache",      "HTML, API responses, media",         "Seconds-hours","Edge PoP RAM"),
        ("API Gateway Cache",   "Identical request responses",        "Seconds",      "Gateway RAM"),
        ("App In-Process (L1)", "Hot data per instance",              "Seconds-mins", "App heap"),
        ("Redis / Memcached",   "Shared data across instances",       "Minutes-hours","Redis RAM"),
        ("DB Query Cache",      "Result of repeated queries",         "Seconds",      "DB RAM"),
        ("DB Buffer Pool",      "Hot pages from disk",                "Continuous",   "DB RAM"),
    ]
    print(f"  {'Layer':<26} {'What to Cache':<38} {'TTL':<14} {'Storage'}")
    print(f"  {'─'*90}")
    for layer, what, ttl, storage in layers:
        print(f"  {layer:<26} {what:<38} {ttl:<14} {storage}")


if __name__ == "__main__":
    demonstrate_caching_fundamentals()
