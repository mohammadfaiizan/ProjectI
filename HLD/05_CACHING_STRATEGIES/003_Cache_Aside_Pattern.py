"""
CACHE-ASIDE PATTERN (LAZY LOADING)
=====================================

Problem Statement:
The most common caching pattern: the application controls the cache explicitly.
On read: check cache, if miss fetch from DB and populate cache.
On write: write to DB, then invalidate (or update) cache entry.

Cache-Aside (Lazy Loading):
  READ:
    1. Check cache for key
    2. HIT  → return cached value
    3. MISS → query DB → store result in cache → return value

  WRITE (invalidation strategy):
    1. Write to DB
    2. DELETE cache entry (invalidate)
    3. Next read will reload from DB (lazy)

  WRITE (update strategy):
    1. Write to DB
    2. SET cache entry to new value
    Problem: race condition if two writers compete

Why Invalidate (not update) on Write?
  Concurrent writes: Writer A and B both update DB.
  If both try to set cache: A's update may overwrite B's in wrong order.
  Invalidation is safer: next read will fetch fresh from DB.

Advantages:
  ✓ Cache only contains what's actually requested (no unnecessary data)
  ✓ DB failures degrade gracefully (cache still serves stale hits)
  ✓ Works with any DB type
  ✓ App controls what to cache and for how long

Disadvantages:
  ✗ First request always misses (cold cache)
  ✗ Race condition: read miss → DB read → write (another update invalidates) → stale set
  ✗ Higher code complexity vs transparent caching

Cache Warming (mitigate cold start):
  Pre-populate cache on startup for known-hot keys.
  Or: run a background job to fill cache from DB during low-traffic hours.

Stale Data Window:
  Between DB write and cache invalidation: old value still in cache.
  Solutions: short TTL, synchronous invalidation, pub/sub invalidation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
import time
import threading
import random
from collections import defaultdict


# ─────────────────────────────────────────────
# SUPPORTING CLASSES
# ─────────────────────────────────────────────

@dataclass
class CacheEntry:
    value     : Any
    created_at: float = field(default_factory=time.time)
    ttl_s     : float = 300.0

    @property
    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl_s


class SimpleCache:
    def __init__(self):
        self._store : Dict[str, CacheEntry] = {}
        self.hits   = 0
        self.misses = 0
        self.sets   = 0
        self.deletes= 0

    def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if entry is None or entry.is_expired:
            if entry:
                del self._store[key]
            self.misses += 1
            return None
        self.hits += 1
        return entry.value

    def set(self, key: str, value: Any, ttl_s: float = 300.0):
        self._store[key] = CacheEntry(value, ttl_s=ttl_s)
        self.sets += 1

    def delete(self, key: str):
        if key in self._store:
            del self._store[key]
            self.deletes += 1

    def flush(self):
        self._store.clear()

    @property
    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def size(self) -> int:
        return len(self._store)


class SlowDatabase:
    """Simulates a slow primary database."""

    def __init__(self):
        self._data    : Dict[str, Any] = {}
        self.reads    = 0
        self.writes   = 0
        self._latency_ms = 10.0   # avg 10ms
        self._lock    = threading.Lock()
        self.version  : Dict[str, int] = defaultdict(int)

    def get(self, key: str) -> Optional[Any]:
        time.sleep(random.uniform(5, 15) / 1000)
        self.reads += 1
        return self._data.get(key)

    def set(self, key: str, value: Any):
        time.sleep(random.uniform(5, 15) / 1000)
        with self._lock:
            self._data[key]    = value
            self.version[key] += 1
        self.writes += 1

    def bulk_load(self, data: Dict[str, Any]):
        self._data.update(data)

    def get_version(self, key: str) -> int:
        return self.version.get(key, 0)


# ─────────────────────────────────────────────
# CACHE-ASIDE SERVICE
# ─────────────────────────────────────────────

class CacheAsideService:
    """
    Application-managed cache with cache-aside (lazy loading) pattern.
    Exposes get() and update() with invalidation.
    """

    def __init__(self, cache: SimpleCache, db: SlowDatabase, default_ttl_s: float = 300.0):
        self.cache       = cache
        self.db          = db
        self.default_ttl = default_ttl_s
        self._lock       : Dict[str, threading.Lock] = {}
        self._meta_lock  = threading.Lock()
        self.cache_populates = 0

    def _key_lock(self, key: str) -> threading.Lock:
        with self._meta_lock:
            if key not in self._lock:
                self._lock[key] = threading.Lock()
            return self._lock[key]

    def get(self, key: str) -> Optional[Any]:
        # 1. Check cache
        val = self.cache.get(key)
        if val is not None:
            return val

        # 2. Cache miss — fetch from DB (per-key lock to avoid stampede)
        lock = self._key_lock(key)
        with lock:
            # Double-check after acquiring lock
            val = self.cache.get(key)
            if val is not None:
                return val
            val = self.db.get(key)
            if val is not None:
                self.cache.set(key, val, ttl_s=self.default_ttl)
                self.cache_populates += 1
            return val

    def update(self, key: str, value: Any):
        """Write to DB, then invalidate cache (invalidation strategy)."""
        self.db.set(key, value)
        self.cache.delete(key)   # Invalidate — next read fetches fresh

    def update_and_repopulate(self, key: str, value: Any):
        """Write to DB, then update cache (update strategy — race condition risk)."""
        self.db.set(key, value)
        self.cache.set(key, value, ttl_s=self.default_ttl)   # May have race condition

    def warm_cache(self, keys: List[str]):
        """Pre-populate cache for known-hot keys (cache warming)."""
        for key in keys:
            val = self.db.get(key)
            if val:
                self.cache.set(key, val, ttl_s=self.default_ttl)


# ─────────────────────────────────────────────
# RACE CONDITION DEMONSTRATOR
# ─────────────────────────────────────────────

class RaceConditionDemo:
    """
    Demonstrates the read-then-write race condition in cache-aside.
    Thread A: cache miss → DB read (slow) → set cache
    Thread B: writes new value to DB → invalidates cache
    Thread A: sets stale value into cache (overwrites B's invalidation)
    """

    def __init__(self):
        self._db    : Dict[str, str] = {"user:1": "Alice_v1"}
        self._cache : Dict[str, str] = {}
        self._log   : List[str] = []
        self._lock  = threading.Lock()

    def log(self, msg: str):
        with self._lock:
            self._log.append(f"[{time.time():.4f}] {msg}")

    def thread_a_read(self):
        """Reader: notices cache miss, slow DB read, then sets cache."""
        self.log("Thread A: cache MISS for user:1")
        time.sleep(0.02)   # simulate slow DB read
        val = self._db["user:1"]   # reads OLD value Alice_v1
        self.log(f"Thread A: DB returned '{val}'")
        time.sleep(0.01)   # simulate more delay
        self._cache["user:1"] = val   # sets STALE value after B already invalidated
        self.log(f"Thread A: set cache to '{val}' (STALE!)")

    def thread_b_write(self):
        """Writer: updates DB then invalidates cache."""
        time.sleep(0.01)   # starts slightly after A
        self.log("Thread B: updating user:1 to 'Alice_v2' in DB")
        self._db["user:1"] = "Alice_v2"
        if "user:1" in self._cache:
            del self._cache["user:1"]
        self.log("Thread B: invalidated cache for user:1")

    def simulate(self) -> Tuple:
        t_a = threading.Thread(target=self.thread_a_read)
        t_b = threading.Thread(target=self.thread_b_write)
        t_a.start()
        t_b.start()
        t_a.join()
        t_b.join()
        final_cache = self._cache.get("user:1", "NOT_CACHED")
        final_db    = self._db.get("user:1")
        return final_cache, final_db, self._log

    # fix re-import issue
    from typing import Tuple


# ─────────────────────────────────────────────
# CACHE WARMING STRATEGIES
# ─────────────────────────────────────────────

class CacheWarmer:
    """Strategies to pre-populate cache and avoid cold-start latency spikes."""

    def __init__(self, cache: SimpleCache, db: SlowDatabase):
        self.cache = cache
        self.db    = db

    def warm_from_access_log(self, access_log: List[str], top_n: int = 100):
        """Count access frequencies from log, warm top N keys."""
        freq : Dict[str, int] = defaultdict(int)
        for key in access_log:
            freq[key] += 1
        top_keys = sorted(freq, key=freq.get, reverse=True)[:top_n]
        loaded = 0
        for key in top_keys:
            val = self.db.get(key)
            if val:
                self.cache.set(key, val, ttl_s=600)
                loaded += 1
        return loaded, top_keys[:5]

    def warm_from_query(self, keys: List[str]) -> int:
        loaded = 0
        for key in keys:
            val = self.db.get(key)
            if val:
                self.cache.set(key, val, ttl_s=600)
                loaded += 1
        return loaded


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_cache_aside():
    print("=" * 65)
    print("CACHE-ASIDE PATTERN (LAZY LOADING)")
    print("=" * 65)

    random.seed(42)

    # ── Basic Cache-Aside ─────────────────────
    print("\n[1] CACHE-ASIDE BASIC FLOW")
    print("─" * 55)

    cache   = SimpleCache()
    db      = SlowDatabase()
    service = CacheAsideService(cache, db, default_ttl_s=60.0)

    # Pre-load DB
    users = {f"user:{i}": {"id": i, "name": f"User{i}", "role": "member"}
             for i in range(1, 51)}
    db.bulk_load(users)

    # Cold cache — all misses
    hot_keys = [f"user:{i}" for i in range(1, 6)]
    print("  COLD cache reads (first access → DB miss):")
    start = time.perf_counter()
    for key in hot_keys:
        service.get(key)
    cold_ms = (time.perf_counter() - start) * 1000
    print(f"  5 reads: {cold_ms:.1f}ms  (all DB reads)  cache_size={cache.size()}")

    # Warm cache — all hits
    print("\n  WARM cache reads (subsequent access → cache hit):")
    start = time.perf_counter()
    for key in hot_keys:
        service.get(key)
    warm_ms = (time.perf_counter() - start) * 1000
    print(f"  5 reads: {warm_ms:.2f}ms  (all cache hits)  speedup={cold_ms/max(warm_ms,0.01):.1f}x")

    # Mixed traffic
    print("\n  Mixed traffic (200 requests, 50 keys, 80/20 access pattern):")
    all_keys = [f"user:{i}" for i in range(1, 51)]
    for _ in range(200):
        if random.random() < 0.8:
            key = random.choice(hot_keys)
        else:
            key = random.choice(all_keys)
        service.get(key)

    print(f"  Cache: hits={cache.hits}  misses={cache.misses}  "
          f"hit_ratio={cache.hit_ratio:.1%}  DB reads={db.reads}")

    # ── Write with Invalidation ───────────────
    print("\n\n[2] WRITE-INVALIDATE (correct approach)")
    print("─" * 55)
    key = "user:1"
    service.get(key)
    print(f"  Cache before update: {cache.get(key) is not None} (key in cache)")
    service.update(key, {"id": 1, "name": "Alice_Updated", "role": "admin"})
    after = cache.get(key)
    print(f"  Cache after update:  {after is not None} (key invalidated — next read fetches fresh)")
    fresh = service.get(key)
    print(f"  Fresh read:          {fresh}")

    # ── Race Condition Illustration ────────────
    print("\n\n[3] RACE CONDITION IN CACHE-ASIDE")
    print("─" * 55)
    demo = RaceConditionDemo()
    final_cache, final_db, log = demo.simulate()
    for line in log:
        print(f"  {line}")
    print(f"\n  DB value (correct): '{final_db}'")
    print(f"  Cache value (actual): '{final_cache}'")
    if final_cache != final_db:
        print(f"  ⚠ STALE DATA in cache — race condition occurred!")
        print(f"  Fix: use short TTL so stale entry expires quickly")
        print(f"  Fix: use cache versioning (cache key includes version)")

    # ── Cache Warming ─────────────────────────
    print("\n\n[4] CACHE WARMING (avoid cold-start spike)")
    print("─" * 55)
    warm_cache = SimpleCache()
    warm_db    = SlowDatabase()
    warm_db.bulk_load({f"item:{i}": {"id": i, "price": i * 9.99} for i in range(1, 201)})
    warmer = CacheWarmer(warm_cache, warm_db)

    # Simulate access log from previous day
    access_log = []
    hot_items = [f"item:{i}" for i in range(1, 11)]
    for _ in range(1000):
        if random.random() < 0.7:
            access_log.append(random.choice(hot_items))
        else:
            access_log.append(f"item:{random.randint(1, 200)}")

    loaded, top5 = warmer.warm_from_access_log(access_log, top_n=20)
    print(f"  Warmed {loaded} keys from yesterday's access log")
    print(f"  Top 5 keys pre-loaded: {top5}")
    print(f"  Cache size after warming: {warm_cache.size()}")
    print(f"  First request hit_ratio (warm): {warm_cache.hit_ratio:.1%}")

    # ── Pattern Summary ───────────────────────
    print("\n\n[5] CACHE-ASIDE SUMMARY")
    print("─" * 55)
    flow = [
        ("READ HIT",  "1→ check cache  2→ return value  (DB not consulted)"),
        ("READ MISS", "1→ check cache  2→ miss  3→ query DB  4→ set cache  5→ return"),
        ("WRITE",     "1→ write DB  2→ invalidate cache key  (next read reloads)"),
        ("WARM",      "On startup: pre-load hot keys from DB into cache"),
    ]
    for op, steps in flow:
        print(f"  {op:<12} {steps}")

    print("\n  Tradeoffs:")
    print("  ✓ Only caches requested data (memory efficient)")
    print("  ✓ Works with any DB, any cache")
    print("  ✗ Cold start: first request misses")
    print("  ✗ Race condition: write invalidation vs stale read")
    print("  ✗ Application must manage cache explicitly")


if __name__ == "__main__":
    demonstrate_cache_aside()
