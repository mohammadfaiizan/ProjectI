"""
DATABASE CACHING STRATEGIES
==============================

Problem Statement:
Databases can typically handle 1K-10K queries/sec. Your app needs 100K+ QPS.
Caching sits between the app and DB, serving hot data from memory at
microsecond latency. The art is in cache invalidation: when to cache,
when to evict, and how to prevent stale data.

Caching Patterns:
  Cache-Aside    : App checks cache first; on miss, reads DB then populates cache
  Read-Through   : Cache layer handles DB read on miss (transparent to app)
  Write-Through  : Write to cache AND DB simultaneously (consistent but slower)
  Write-Behind   : Write to cache first, async to DB (fast writes, risk of loss)
  Refresh-Ahead  : Proactively refresh before expiry (reduces miss spikes)

Eviction Policies:
  LRU (Least Recently Used)   : evict oldest unused (most common)
  LFU (Least Frequently Used) : evict least-accessed items
  TTL (Time-To-Live)          : expire after fixed time
  FIFO (First In, First Out)  : evict oldest inserted

Cache Invalidation Strategies:
  TTL-based    : simple, may serve stale data until expiry
  Event-based  : invalidate on write events (pub/sub)
  Write-through: cache always current
  Versioning   : cache key includes version (cache_key:v3)

Cache Stampede (Thundering Herd):
  All cache misses hit DB simultaneously after TTL expiry.
  Fix: mutex lock on miss, staggered TTL jitter, background refresh.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
import time
import threading
import random
import hashlib


class CachePattern(Enum):
    CACHE_ASIDE    = "cache_aside"
    READ_THROUGH   = "read_through"
    WRITE_THROUGH  = "write_through"
    WRITE_BEHIND   = "write_behind"
    REFRESH_AHEAD  = "refresh_ahead"


class EvictionPolicy(Enum):
    LRU   = "lru"
    LFU   = "lfu"
    TTL   = "ttl"
    FIFO  = "fifo"


@dataclass
class CacheEntry:
    key        : str
    value      : Any
    created_at : float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)
    hit_count  : int = 0
    ttl_s      : Optional[float] = None

    @property
    def is_expired(self) -> bool:
        if self.ttl_s is None:
            return False
        return time.time() - self.created_at > self.ttl_s

    @property
    def age_s(self) -> float:
        return time.time() - self.created_at


# ─────────────────────────────────────────────
# LRU CACHE
# ─────────────────────────────────────────────

class LRUCache:
    """
    Least Recently Used cache using ordered dict pattern.
    O(1) get/put. Evicts least recently used entry when full.
    """

    def __init__(self, capacity: int, default_ttl_s: float = None):
        self.capacity    = capacity
        self.default_ttl = default_ttl_s
        self._store      : Dict[str, CacheEntry] = {}
        self._order      : List[str] = []   # most recent at end
        self.hits        = 0
        self.misses      = 0
        self.evictions   = 0
        self._lock       = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._store.get(key)
            if entry is None or entry.is_expired:
                if entry and entry.is_expired:
                    self._evict(key)
                self.misses += 1
                return None
            # Move to end (most recently used)
            if key in self._order:
                self._order.remove(key)
            self._order.append(key)
            entry.last_access = time.time()
            entry.hit_count  += 1
            self.hits         += 1
            return entry.value

    def set(self, key: str, value: Any, ttl_s: float = None):
        with self._lock:
            if key in self._store:
                self._order.remove(key)
            elif len(self._store) >= self.capacity:
                # Evict LRU (first in list)
                lru_key = self._order.pop(0)
                self._evict(lru_key)
            self._store[key] = CacheEntry(key, value, ttl_s=ttl_s or self.default_ttl)
            self._order.append(key)

    def delete(self, key: str):
        with self._lock:
            if key in self._store:
                self._evict(key)

    def _evict(self, key: str):
        self._store.pop(key, None)
        if key in self._order:
            self._order.remove(key)
        self.evictions += 1

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    @property
    def size(self) -> int:
        return len(self._store)

    def report(self):
        print(f"    LRU Cache: size={self.size}/{self.capacity}  "
              f"hits={self.hits}  misses={self.misses}  "
              f"hit_rate={self.hit_rate:.1%}  evictions={self.evictions}")


# ─────────────────────────────────────────────
# CACHE PATTERNS
# ─────────────────────────────────────────────

class CacheAsideService:
    """
    Application manages cache explicitly.
    Read: check cache → hit? return. miss? read DB → set cache → return.
    Write: write DB → invalidate or update cache.
    """

    def __init__(self, cache: LRUCache, db_latency_ms: float = 10.0):
        self.cache       = cache
        self.db_latency  = db_latency_ms
        self._db         : Dict[str, Any] = {}
        self.db_reads    = 0
        self.db_writes   = 0

    def get(self, key: str) -> Optional[Any]:
        # Check cache first
        val = self.cache.get(key)
        if val is not None:
            return val

        # Cache miss — go to DB
        time.sleep(self.db_latency / 1000)
        self.db_reads += 1
        val = self._db.get(key)
        if val is not None:
            self.cache.set(key, val, ttl_s=300)   # cache for 5 min
        return val

    def set(self, key: str, value: Any):
        # Write to DB
        time.sleep(self.db_latency / 1000)
        self.db_writes += 1
        self._db[key] = value
        # Invalidate cache (write-invalidate)
        self.cache.delete(key)

    def set_initial(self, key: str, value: Any):
        self._db[key] = value


class WriteThroughCache:
    """
    Write to cache AND DB simultaneously.
    Cache is always consistent with DB. Higher write latency.
    """

    def __init__(self, cache: LRUCache, db_latency_ms: float = 10.0):
        self.cache      = cache
        self.db_latency = db_latency_ms
        self._db        : Dict[str, Any] = {}
        self.db_writes  = 0

    def get(self, key: str) -> Optional[Any]:
        val = self.cache.get(key)
        if val:
            return val
        return self._db.get(key)

    def set(self, key: str, value: Any):
        # Write to BOTH simultaneously
        self.cache.set(key, value, ttl_s=3600)
        time.sleep(self.db_latency / 1000)
        self._db[key] = value
        self.db_writes += 1


class WriteBehindCache:
    """
    Write to cache only; asynchronously flush to DB.
    Fast writes, risk of data loss if cache crashes before flush.
    """

    def __init__(self, cache: LRUCache, flush_interval_s: float = 1.0):
        self.cache          = cache
        self.flush_interval = flush_interval_s
        self._db            : Dict[str, Any] = {}
        self._dirty         : Dict[str, Any] = {}   # pending DB writes
        self._lock          = threading.Lock()
        self.db_writes      = 0
        self.cache_writes   = 0

    def set(self, key: str, value: Any):
        self.cache.set(key, value, ttl_s=300)
        self.cache_writes += 1
        with self._lock:
            self._dirty[key] = value

    def flush(self):
        with self._lock:
            for key, val in self._dirty.items():
                self._db[key] = val
                self.db_writes += 1
            self._dirty.clear()
        print(f"  WriteBehind: flushed {self.db_writes} records to DB")


# ─────────────────────────────────────────────
# CACHE STAMPEDE PROTECTION
# ─────────────────────────────────────────────

class StampedeProtectedCache:
    """
    Prevents thundering herd: when a cached key expires,
    only one request fetches from DB; others wait.
    """

    def __init__(self, ttl_s: float = 60.0, jitter_s: float = 10.0):
        self._cache   : Dict[str, CacheEntry] = {}
        self._locks   : Dict[str, threading.Lock] = {}
        self._meta_lock = threading.Lock()
        self.ttl_s    = ttl_s
        self.jitter_s = jitter_s   # randomize TTL to prevent mass expiry
        self.stampede_blocks = 0

    def _get_lock(self, key: str) -> threading.Lock:
        with self._meta_lock:
            if key not in self._locks:
                self._locks[key] = threading.Lock()
            return self._locks[key]

    def get_or_compute(self, key: str, loader: Callable) -> Any:
        entry = self._cache.get(key)
        if entry and not entry.is_expired:
            return entry.value

        # Cache miss or expired — use mutex to prevent stampede
        lock = self._get_lock(key)
        if lock.acquire(blocking=False):
            try:
                # Double-check after acquiring lock
                entry = self._cache.get(key)
                if entry and not entry.is_expired:
                    return entry.value
                # Load from DB
                value = loader(key)
                jitter = random.uniform(0, self.jitter_s)
                ttl    = self.ttl_s + jitter
                self._cache[key] = CacheEntry(key, value, ttl_s=ttl)
                return value
            finally:
                lock.release()
        else:
            # Another thread is loading — wait for it
            self.stampede_blocks += 1
            lock.acquire()
            lock.release()
            entry = self._cache.get(key)
            return entry.value if entry else None


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_db_caching():
    print("=" * 65)
    print("DATABASE CACHING STRATEGIES")
    print("=" * 65)

    random.seed(42)

    # ── Cache-Aside ────────────────────────────
    print("\n[1] CACHE-ASIDE PATTERN")
    print("─" * 55)
    cache = LRUCache(capacity=100, default_ttl_s=300)
    service = CacheAsideService(cache, db_latency_ms=10.0)

    # Pre-populate DB
    for i in range(20):
        service.set_initial(f"user:{i}", {"name": f"User{i}", "email": f"u{i}@ex.com"})

    # Access pattern: 80/20 rule — 20% of keys get 80% of traffic
    hot_keys   = [f"user:{i}" for i in range(4)]   # popular users
    all_keys   = [f"user:{i}" for i in range(20)]

    print("  First access (cold cache — all misses → DB reads):")
    for key in hot_keys:
        val = service.get(key)

    print("  Subsequent access (hot cache — hits from cache):")
    for _ in range(20):
        key = random.choice(hot_keys + all_keys[:6])
        service.get(key)

    print(f"  Cache stats:")
    cache.report()
    print(f"    DB reads: {service.db_reads}  DB writes: {service.db_writes}")

    # ── LRU Eviction ──────────────────────────
    print("\n\n[2] LRU EVICTION (capacity=5)")
    print("─" * 55)
    small_cache = LRUCache(capacity=5)
    for i in range(8):
        small_cache.set(f"k{i}", f"value_{i}")
        print(f"  SET k{i} — size={small_cache.size}  "
              f"LRU order={[k[-1] for k in small_cache._order]}")
    print(f"  (keys k0, k1, k2 evicted as capacity=5 was exceeded)")

    # ── Write-Through ─────────────────────────
    print("\n\n[3] WRITE-THROUGH PATTERN")
    print("─" * 55)
    wt_cache = LRUCache(capacity=50)
    wt = WriteThroughCache(wt_cache, db_latency_ms=5.0)

    wt.set("product:p1", {"name": "Laptop", "price": 999.99})
    wt.set("product:p2", {"name": "Mouse",  "price": 29.99})
    print(f"  Wrote 2 products to cache+DB simultaneously")
    p1 = wt.get("product:p1")
    print(f"  Read product:p1 from cache: {p1}")
    print(f"  DB writes: {wt.db_writes}  (synchronous — always consistent)")

    # ── Write-Behind ──────────────────────────
    print("\n\n[4] WRITE-BEHIND PATTERN (async DB flush)")
    print("─" * 55)
    wb_cache = LRUCache(capacity=100)
    wb = WriteBehindCache(wb_cache)
    for i in range(5):
        wb.set(f"event:{i}", {"type": "click", "user": f"u{i}"})
        print(f"  SET event:{i} → cache immediately (DB async)")
    print(f"  Cache writes: {wb.cache_writes}  DB writes so far: {wb.db_writes}")
    wb.flush()
    print(f"  After flush: DB writes: {wb.db_writes}")

    # ── Stampede Protection ───────────────────
    print("\n\n[5] CACHE STAMPEDE PROTECTION")
    print("─" * 55)
    protected = StampedeProtectedCache(ttl_s=0.1, jitter_s=0.05)   # short TTL for demo
    db_calls = {"count": 0}

    def slow_db_loader(key: str) -> Any:
        db_calls["count"] += 1
        time.sleep(0.01)   # simulate 10ms DB read
        return {"data": f"db_value_{key}", "loaded_at": time.time()}

    # First access — loads from DB
    val = protected.get_or_compute("hot-key", slow_db_loader)
    print(f"  First access: loaded from DB ({val['data']})")

    # Subsequent accesses — from cache
    for _ in range(5):
        protected.get_or_compute("hot-key", slow_db_loader)
    print(f"  5 more accesses: DB calls = {db_calls['count']} (only 1 — cache hit)")

    # Concurrent access after expiry — only 1 should reload
    time.sleep(0.15)   # let cache expire
    threads = [threading.Thread(target=lambda: protected.get_or_compute("hot-key", slow_db_loader))
               for _ in range(5)]
    for t in threads: t.start()
    for t in threads: t.join()
    print(f"  After expiry, 5 concurrent misses: DB calls = {db_calls['count']} "
          f"(only 2 total — stampede prevented), "
          f"blocked={protected.stampede_blocks}")

    # ── Strategy Comparison ───────────────────
    print("\n\n[6] CACHING STRATEGY COMPARISON")
    print("─" * 55)
    strategies = [
        ("Cache-Aside",   "App manages cache",      "Flexible, most common",    "Code complexity"),
        ("Read-Through",  "Cache fetches on miss",  "Transparent to app",       "Cold start on miss"),
        ("Write-Through", "Dual write: cache+DB",   "Always consistent",        "Higher write latency"),
        ("Write-Behind",  "Write cache; async DB",  "Very fast writes",         "Data loss risk"),
        ("Refresh-Ahead", "Proactive refresh",      "No miss latency spikes",   "May cache unused data"),
    ]
    print(f"  {'Pattern':<18} {'How':<25} {'Pros':<28} {'Cons'}")
    print(f"  {'─'*80}")
    for name, how, pros, cons in strategies:
        print(f"  {name:<18} {how:<25} {pros:<28} {cons}")


if __name__ == "__main__":
    demonstrate_db_caching()
