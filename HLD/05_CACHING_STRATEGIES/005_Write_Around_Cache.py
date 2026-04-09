"""
WRITE-AROUND CACHE
====================

Problem Statement:
Not all data should be cached immediately on write. Large files, batch
imports, one-time reports, and rarely-re-read data waste cache memory
if cached on write. Write-around avoids polluting the cache with data
that has a low probability of being read back soon.

Write-Around:
  WRITE: bypass cache → write directly to DB/storage
  READ : check cache → miss → load from DB → optionally cache

When to Use Write-Around:
  1. Write-once, read-never (or rarely) data: raw logs, audit trails
  2. Large objects: 50MB PDFs, video files (cache would evict hot small items)
  3. Batch imports: loading 1M records at night should not fill the cache
     and evict the hot data that the day-shift users need
  4. Pre-computed reports: results stored in DB, only specific users re-read
  5. User uploads: photo uploaded once, re-read by CDN not app cache

Cache Pollution Problem:
  Without write-around: a bulk import of 10K products fills the LRU cache,
  evicting the 100 hot product pages that get 90% of traffic.
  With write-around: bulk import goes to DB; cache retains hot products.

Combining Write-Around + Cache-Aside for Reads:
  This is the most common read pattern in production:
  - Write-around: all writes go to DB
  - Cache-aside: reads check cache, miss → DB → populate cache
  The cache self-organizes around what's actually being read.

Selective Write-Through (Hybrid):
  For some known-hot data (featured products, top users),
  bypass the write-around rule and write-through to pre-populate cache.
  Use access tier hints from the application to decide.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
import time
import random
from collections import defaultdict, OrderedDict


# ─────────────────────────────────────────────
# LRU CACHE (capacity-bounded)
# ─────────────────────────────────────────────

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity  = capacity
        self._store    : OrderedDict[str, Any] = OrderedDict()
        self.hits      = 0
        self.misses    = 0
        self.evictions = 0

    def get(self, key: str) -> Optional[Any]:
        if key not in self._store:
            self.misses += 1
            return None
        self._store.move_to_end(key)
        self.hits += 1
        return self._store[key]

    def set(self, key: str, value: Any):
        if key in self._store:
            self._store.move_to_end(key)
        elif len(self._store) >= self.capacity:
            self._store.popitem(last=False)
            self.evictions += 1
        self._store[key] = value

    def delete(self, key: str):
        self._store.pop(key, None)

    @property
    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def current_keys(self) -> List[str]:
        return list(self._store.keys())

    def size(self) -> int:
        return len(self._store)


class Database:
    def __init__(self, read_latency_ms: float = 10.0):
        self._data          : Dict[str, Any] = {}
        self.reads          = 0
        self.writes         = 0
        self.read_latency   = read_latency_ms

    def get(self, key: str) -> Optional[Any]:
        time.sleep(self.read_latency / 1000)
        self.reads += 1
        return self._data.get(key)

    def set(self, key: str, value: Any):
        self._data[key] = value
        self.writes += 1

    def bulk_set(self, items: Dict[str, Any]):
        self._data.update(items)
        self.writes += len(items)

    def contains(self, key: str) -> bool:
        return key in self._data


# ─────────────────────────────────────────────
# WRITE-AROUND CACHE LAYER
# ─────────────────────────────────────────────

class WriteAroundCacheLayer:
    """
    Write-around pattern with optional selective write-through for hot keys.
    All writes bypass cache.
    Reads: cache-aside (check cache → miss → DB → populate cache).
    """

    def __init__(self, cache: LRUCache, db: Database,
                 cache_on_read: bool = True,
                 hot_key_prefixes: Set[str] = None):
        self.cache            = cache
        self.db               = db
        self.cache_on_read    = cache_on_read
        self.hot_prefixes     = hot_key_prefixes or set()
        self.write_bypasses   = 0     # writes that bypassed cache
        self.write_throughs   = 0     # selective writes cached (hot keys)
        self.read_populates   = 0     # misses that populated cache

    def _is_hot(self, key: str) -> bool:
        return any(key.startswith(p) for p in self.hot_prefixes)

    def write(self, key: str, value: Any):
        """Write-around: always goes to DB. Cache updated only for hot keys."""
        self.db.set(key, value)

        if self._is_hot(key):
            # Selective write-through for known hot data
            self.cache.set(key, value)
            self.write_throughs += 1
        else:
            # Invalidate cache if stale entry exists
            self.cache.delete(key)
            self.write_bypasses += 1

    def bulk_write(self, items: Dict[str, Any]):
        """Bulk import — always write-around (protect cache from bulk pollution)."""
        self.db.bulk_set(items)
        self.write_bypasses += len(items)
        # Invalidate any cached entries that were overwritten
        for key in items:
            self.cache.delete(key)

    def read(self, key: str) -> Optional[Any]:
        """Cache-aside read: check cache → miss → DB → optionally cache."""
        val = self.cache.get(key)
        if val is not None:
            return val

        val = self.db.get(key)
        if val is not None and self.cache_on_read:
            self.cache.set(key, val)
            self.read_populates += 1
        return val


# ─────────────────────────────────────────────
# CACHE POLLUTION SIMULATOR
# ─────────────────────────────────────────────

class CachePollutionSimulator:
    """
    Demonstrates how a bulk write pollutes cache vs write-around behavior.
    """

    def __init__(self, cache_capacity: int = 10):
        self.cache_capacity = cache_capacity

    def simulate_without_write_around(self,
                                       hot_keys: List[str],
                                       bulk_keys: List[str]) -> Dict:
        """Without write-around: bulk write populates cache, evicts hot entries."""
        cache = LRUCache(self.cache_capacity)
        db    = Database(read_latency_ms=5.0)

        # Load hot data (simulates normal operation)
        for key in hot_keys:
            db.set(key, {"data": f"hot_{key}"})
            cache.set(key, {"data": f"hot_{key}"})   # pre-cached

        hot_in_cache_before = sum(1 for k in hot_keys if cache.get(k) is not None)
        cache.hits = cache.misses = 0   # reset stats

        # Bulk write: all go through cache (naive write-through)
        for key in bulk_keys:
            val = {"data": f"bulk_{key}"}
            db.set(key, val)
            cache.set(key, val)   # write-through: fills cache with bulk data

        hot_in_cache_after = sum(1 for k in hot_keys if k in cache._store)
        return {
            "hot_in_cache_before" : hot_in_cache_before,
            "hot_in_cache_after"  : hot_in_cache_after,
            "cache_size"          : cache.size(),
            "evictions"           : cache.evictions,
        }

    def simulate_with_write_around(self,
                                    hot_keys: List[str],
                                    bulk_keys: List[str]) -> Dict:
        """With write-around: bulk write bypasses cache, hot entries preserved."""
        cache = LRUCache(self.cache_capacity)
        db    = Database(read_latency_ms=5.0)
        layer = WriteAroundCacheLayer(cache, db)

        # Load hot data
        for key in hot_keys:
            db.set(key, {"data": f"hot_{key}"})
            cache.set(key, {"data": f"hot_{key}"})

        hot_in_cache_before = sum(1 for k in hot_keys if k in cache._store)

        # Bulk write: write-around (bypasses cache)
        bulk_dict = {key: {"data": f"bulk_{key}"} for key in bulk_keys}
        layer.bulk_write(bulk_dict)

        hot_in_cache_after = sum(1 for k in hot_keys if k in cache._store)
        return {
            "hot_in_cache_before" : hot_in_cache_before,
            "hot_in_cache_after"  : hot_in_cache_after,
            "cache_size"          : cache.size(),
            "evictions"           : cache.evictions,
            "write_bypasses"      : layer.write_bypasses,
        }


# ─────────────────────────────────────────────
# READ RATIO ANALYZER
# ─────────────────────────────────────────────

class ReadRatioAnalyzer:
    """
    Analyzes whether caching makes sense for a given read:write ratio.
    Write-around is justified when reads << writes or read ratio < threshold.
    """

    @staticmethod
    def cache_value_score(reads_per_write: float, ttl_s: float,
                           write_cost_ms: float, read_cost_ms: float) -> float:
        """
        Estimate the value of caching based on how many reads the cached entry
        will serve during its TTL relative to write cost.
        Score > 1.0 means caching is beneficial.
        """
        reads_during_ttl = reads_per_write   # expected reads before expiry
        cache_read_ms    = 0.5               # cache hit latency
        saved_per_read   = read_cost_ms - cache_read_ms
        total_saved      = reads_during_ttl * saved_per_read
        score            = total_saved / write_cost_ms
        return score

    @staticmethod
    def recommendation(reads_per_write: float) -> str:
        if reads_per_write < 0.1:
            return "Write-Around: data is rarely re-read, don't pollute cache"
        elif reads_per_write < 1.0:
            return "Write-Around with short TTL, or Cache-Aside only"
        elif reads_per_write < 10:
            return "Cache-Aside (lazy load on first read)"
        else:
            return "Write-Through: very read-heavy, pre-populate cache on write"


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_write_around():
    print("=" * 65)
    print("WRITE-AROUND CACHE")
    print("=" * 65)

    random.seed(42)

    # ── Cache Pollution Demo ───────────────────
    print("\n[1] CACHE POLLUTION WITHOUT WRITE-AROUND")
    print("─" * 55)
    sim = CachePollutionSimulator(cache_capacity=10)

    hot_keys  = [f"product:{i}" for i in range(1, 8)]    # 7 hot products
    bulk_keys = [f"import:{i}"  for i in range(1, 20)]   # 19 bulk imported items

    without = sim.simulate_without_write_around(hot_keys, bulk_keys)
    with_wa = sim.simulate_with_write_around(hot_keys, bulk_keys)

    print(f"  Scenario: cache_size=10, {len(hot_keys)} hot products pre-cached, "
          f"{len(bulk_keys)} bulk import")
    print()
    print(f"  {'Metric':<32} {'Without Write-Around':<22} {'With Write-Around'}")
    print(f"  {'─'*70}")
    metrics = [
        ("Hot keys in cache before bulk", without["hot_in_cache_before"], with_wa["hot_in_cache_before"]),
        ("Hot keys in cache after bulk",  without["hot_in_cache_after"],  with_wa["hot_in_cache_after"]),
        ("Cache evictions",               without["evictions"],           with_wa["evictions"]),
        ("Cache size",                    without["cache_size"],           with_wa["cache_size"]),
    ]
    for name, wout, ww in metrics:
        print(f"  {name:<32} {wout:<22} {ww}")

    print(f"\n  Without write-around: bulk import evicted {without['hot_in_cache_before'] - without['hot_in_cache_after']} hot products!")
    print(f"  With write-around: all {with_wa['hot_in_cache_after']} hot products preserved in cache")

    # ── Basic Write-Around Flow ────────────────
    print("\n\n[2] WRITE-AROUND BASIC FLOW")
    print("─" * 55)
    cache = LRUCache(capacity=20)
    db    = Database(read_latency_ms=8.0)
    layer = WriteAroundCacheLayer(cache, db)

    # Write log entries (write-around)
    print("  Writing 10 log entries (write-around → bypasses cache):")
    for i in range(10):
        layer.write(f"log:{i}", {"message": f"Event {i}", "level": "INFO"})
    print(f"  Cache size: {cache.size()}  Write bypasses: {layer.write_bypasses}")

    # Read back specific entries
    print("\n  Reading 3 log entries (first read → DB, second → cache):")
    for i in range(3):
        val = layer.read(f"log:{i}")
    print(f"  Cache size: {cache.size()}  Read-populated: {layer.read_populates}")

    for i in range(3):
        val = layer.read(f"log:{i}")   # should hit cache now
    print(f"  Cache hits={cache.hits}  misses={cache.misses}  hit_ratio={cache.hit_ratio:.1%}")

    # ── Selective Write-Through (Hot Keys) ─────
    print("\n\n[3] SELECTIVE WRITE-THROUGH FOR HOT KEYS")
    print("─" * 55)
    cache2 = LRUCache(capacity=20)
    db2    = Database(read_latency_ms=8.0)
    layer2 = WriteAroundCacheLayer(
        cache2, db2,
        hot_key_prefixes={"featured:", "homepage:"}   # pre-cache hot sections
    )

    # Hot data: write-through automatically
    featured_items = {f"featured:{i}": {"id": i, "name": f"Featured Product {i}"}
                      for i in range(5)}
    for key, val in featured_items.items():
        layer2.write(key, val)

    # Cold data: write-around
    cold_items = {f"archived:{i}": {"id": i, "name": f"Old Product {i}"}
                  for i in range(50)}
    for key, val in cold_items.items():
        layer2.write(key, val)

    print(f"  Hot keys (featured:*) written: {layer2.write_throughs} → cached immediately")
    print(f"  Cold keys (archived:*) written: {layer2.write_bypasses} → bypassed cache")
    print(f"  Cache size: {cache2.size()} (only hot keys)")

    # Reads of hot keys: zero latency
    for key in featured_items:
        cache2.get(key)
    print(f"\n  Reads of featured items: all {cache2.hits} cache hits (pre-warmed)")

    # ── Read:Write Ratio Analysis ─────────────
    print("\n\n[4] WHEN TO USE WRITE-AROUND (read:write analysis)")
    print("─" * 55)
    analyzer = ReadRatioAnalyzer()
    scenarios = [
        ("Raw logs",           0.01, "write-once, almost never re-read"),
        ("Audit trail",        0.05, "compliance writes, rarely queried"),
        ("Bulk product import",0.1,  "nightly sync, few reads day after"),
        ("User blog post",     2.0,  "read by author + few visitors"),
        ("News article",       50.0, "one write, thousands of reads"),
        ("Home page content",  1000.0,"one write, millions of reads"),
    ]
    print(f"  {'Data Type':<28} {'Reads/Write':<14} {'Recommendation'}")
    print(f"  {'─'*75}")
    for data_type, rpw, note in scenarios:
        rec = analyzer.recommendation(rpw)
        score = analyzer.cache_value_score(rpw, ttl_s=300, write_cost_ms=1.0, read_cost_ms=10.0)
        print(f"  {data_type:<28} {rpw:<14.2f} {rec}")

    # ── Summary ───────────────────────────────
    print("\n\n[5] WRITE-AROUND DECISION GUIDE")
    print("─" * 55)
    print("  USE write-around when:")
    print("    • Data is written once, rarely or never re-read (logs, archives)")
    print("    • Large objects that would evict many small hot items")
    print("    • Batch/bulk imports that should not pollute production cache")
    print("    • Data with high write volume, low read volume")
    print()
    print("  AVOID write-around when:")
    print("    • Data is immediately re-read after write (sessions, profiles)")
    print("    • You want cache to always reflect DB (use write-through)")
    print("    • Hot data is written frequently (use write-through selectively)")
    print()
    print("  COMBINE with:")
    print("    • Cache-aside on reads: lazy-load on first read after write-around")
    print("    • Selective write-through for known-hot keys (featured, homepage)")
    print("    • Short TTL on read-populated entries to limit staleness window")


if __name__ == "__main__":
    demonstrate_write_around()
