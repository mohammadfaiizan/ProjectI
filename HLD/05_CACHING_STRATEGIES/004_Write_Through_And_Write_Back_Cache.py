"""
WRITE-THROUGH AND WRITE-BACK CACHE
======================================

Problem Statement:
In cache-aside, the application manages cache reads/writes explicitly.
Write-through and write-back are alternative write strategies that
optimize for different consistency and performance tradeoffs.

Write-Through:
  Every write goes to cache AND database synchronously.
  READ : always hits cache (pre-populated on every write)
  WRITE: write cache + write DB in same request (higher write latency)
  Consistency: always consistent (cache = DB)
  Use: financial data, inventory counts, any write-then-read workload

Write-Back (Write-Behind):
  Writes go to cache immediately; DB updated asynchronously in batches.
  READ : hits cache (fast)
  WRITE: write cache only → return fast → background worker flushes to DB
  Consistency: window of potential data loss (crash before flush)
  Use: high-write IoT sensors, gaming leaderboards, analytics counters

Write-Around:
  Writes go directly to DB, bypassing cache entirely.
  Cache is only populated on reads (cache-aside behavior for reads).
  Use: infrequently read data (logs, backups) — avoids polluting cache
       with data that won't be re-read.

Comparison:
                   Write Latency    Read Latency    Consistency    Data Loss Risk
  Write-Through  : DB + cache         cache hit     Strong         None
  Write-Back     : cache only         cache hit     Eventual       Crash window
  Write-Around   : DB only            cache miss    Strong         None
  Cache-Aside    : DB only (manual)   cache/DB      Strong         None

Combined Strategies:
  Most systems combine patterns:
  - Write-through for critical data (orders, payments)
  - Write-behind for high-volume counters (views, likes)
  - Write-around for large blobs (images) that aren't re-read immediately
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import time
import threading
import random
from collections import defaultdict, deque


# ─────────────────────────────────────────────
# SHARED COMPONENTS
# ─────────────────────────────────────────────

class InMemoryCache:
    def __init__(self):
        self._store : Dict[str, Any] = {}
        self.hits   = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        val = self._store.get(key)
        if val is None:
            self.misses += 1
        else:
            self.hits += 1
        return val

    def set(self, key: str, value: Any):
        self._store[key] = value

    def delete(self, key: str):
        self._store.pop(key, None)

    @property
    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def size(self) -> int:
        return len(self._store)


class PrimaryDB:
    def __init__(self, latency_ms: float = 10.0):
        self._data     : Dict[str, Any] = {}
        self.reads     = 0
        self.writes    = 0
        self.latency   = latency_ms

    def get(self, key: str) -> Optional[Any]:
        time.sleep(self.latency / 1000)
        self.reads += 1
        return self._data.get(key)

    def set(self, key: str, value: Any):
        time.sleep(self.latency / 1000)
        self._data[key] = value
        self.writes += 1

    def bulk_set(self, items: Dict[str, Any]):
        time.sleep(self.latency / 1000)   # batch write: single latency hit
        self._data.update(items)
        self.writes += len(items)

    def get_all(self) -> Dict[str, Any]:
        return dict(self._data)


# ─────────────────────────────────────────────
# WRITE-THROUGH CACHE
# ─────────────────────────────────────────────

class WriteThroughCache:
    """
    Every write updates cache AND DB synchronously.
    Strong consistency: cache always reflects DB.
    Trade-off: higher write latency (cache + DB in same call).
    """

    def __init__(self, cache: InMemoryCache, db: PrimaryDB):
        self.cache = cache
        self.db    = db
        self.write_latencies_ms : List[float] = []

    def get(self, key: str) -> Optional[Any]:
        # Cache is always warm for written keys — no DB fallback needed
        val = self.cache.get(key)
        if val is not None:
            return val
        # Cold read (key never written through this path)
        val = self.db.get(key)
        if val:
            self.cache.set(key, val)
        return val

    def set(self, key: str, value: Any):
        start = time.perf_counter()
        # Write to BOTH: cache first (fast), then DB (slow)
        self.cache.set(key, value)
        self.db.set(key, value)   # Synchronous — caller waits
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.write_latencies_ms.append(elapsed_ms)

    @property
    def avg_write_latency_ms(self) -> float:
        return sum(self.write_latencies_ms) / len(self.write_latencies_ms) if self.write_latencies_ms else 0.0


# ─────────────────────────────────────────────
# WRITE-BACK (WRITE-BEHIND) CACHE
# ─────────────────────────────────────────────

@dataclass
class DirtyEntry:
    key       : str
    value     : Any
    written_at: float = field(default_factory=time.time)


class WriteBackCache:
    """
    Write to cache immediately; flush to DB asynchronously in batches.
    Lowest write latency; risk of data loss on crash before flush.
    """

    def __init__(self, cache: InMemoryCache, db: PrimaryDB,
                 flush_interval_s: float = 1.0, batch_size: int = 100):
        self.cache          = cache
        self.db             = db
        self.flush_interval = flush_interval_s
        self.batch_size     = batch_size
        self._dirty         : Dict[str, DirtyEntry] = {}
        self._lock          = threading.Lock()
        self._flush_count   = 0
        self._total_flushed = 0
        self.write_latencies_ms : List[float] = []

    def get(self, key: str) -> Optional[Any]:
        val = self.cache.get(key)
        if val is not None:
            return val
        # Check dirty buffer (in case not in cache but pending flush)
        entry = self._dirty.get(key)
        if entry:
            self.cache.misses -= 1   # correct miss count (was in dirty)
            self.cache.hits   += 1
            return entry.value
        # DB fallback
        return self.db.get(key)

    def set(self, key: str, value: Any):
        start = time.perf_counter()
        # Write to cache immediately
        self.cache.set(key, value)
        # Mark dirty (pending DB flush)
        with self._lock:
            self._dirty[key] = DirtyEntry(key, value)
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.write_latencies_ms.append(elapsed_ms)

    def flush(self) -> int:
        """Flush all dirty entries to DB in one batch."""
        with self._lock:
            if not self._dirty:
                return 0
            batch = dict(self._dirty)
            self._dirty.clear()

        self.db.bulk_set(batch)   # single network round-trip for batch
        self._flush_count   += 1
        self._total_flushed += len(batch)
        return len(batch)

    def flush_periodic(self, duration_s: float):
        """Run periodic flushes for `duration_s` seconds (simulates background worker)."""
        start = time.time()
        while time.time() - start < duration_s:
            time.sleep(self.flush_interval)
            self.flush()

    @property
    def pending_writes(self) -> int:
        return len(self._dirty)

    @property
    def avg_write_latency_ms(self) -> float:
        return sum(self.write_latencies_ms) / len(self.write_latencies_ms) if self.write_latencies_ms else 0.0


# ─────────────────────────────────────────────
# WRITE-AROUND CACHE
# ─────────────────────────────────────────────

class WriteAroundCache:
    """
    Writes bypass cache and go directly to DB.
    Cache is populated only on reads (lazy loading).
    Best for write-once, rarely-read data (logs, raw uploads).
    Avoids polluting cache with data unlikely to be re-read.
    """

    def __init__(self, cache: InMemoryCache, db: PrimaryDB, ttl_s: float = 300.0):
        self.cache = cache
        self.db    = db
        self.ttl   = ttl_s
        self.write_latencies_ms : List[float] = []

    def get(self, key: str) -> Optional[Any]:
        val = self.cache.get(key)
        if val is not None:
            return val
        # Load from DB and cache for future reads
        val = self.db.get(key)
        if val:
            self.cache.set(key, val)
        return val

    def set(self, key: str, value: Any):
        start = time.perf_counter()
        # Write directly to DB — cache NOT updated
        self.db.set(key, value)
        # If key was cached, invalidate it (consistency)
        self.cache.delete(key)
        self.write_latencies_ms.append((time.perf_counter() - start) * 1000)

    @property
    def avg_write_latency_ms(self) -> float:
        return sum(self.write_latencies_ms) / len(self.write_latencies_ms) if self.write_latencies_ms else 0.0


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_write_strategies():
    print("=" * 65)
    print("WRITE-THROUGH AND WRITE-BACK CACHE")
    print("=" * 65)

    random.seed(42)
    N_WRITES = 20

    # ── Write-Through ─────────────────────────
    print("\n[1] WRITE-THROUGH CACHE")
    print("─" * 55)
    wt_cache = InMemoryCache()
    wt_db    = PrimaryDB(latency_ms=8.0)
    wt       = WriteThroughCache(wt_cache, wt_db)

    print(f"  Writing {N_WRITES} products (cache + DB synchronously)...")
    start = time.perf_counter()
    for i in range(N_WRITES):
        wt.set(f"product:{i}", {"id": i, "name": f"Product{i}", "price": i * 10.99})
    total_write_ms = (time.perf_counter() - start) * 1000

    print(f"  Total write time: {total_write_ms:.1f}ms  avg/write: {wt.avg_write_latency_ms:.1f}ms")
    print(f"  DB writes: {wt_db.writes}  Cache size: {wt_cache.size()}")

    # Read — always hits cache
    start = time.perf_counter()
    for i in range(N_WRITES):
        wt.get(f"product:{i}")
    read_ms = (time.perf_counter() - start) * 1000

    print(f"\n  Read {N_WRITES} products: {read_ms:.2f}ms  (all cache hits, DB reads: {wt_db.reads})")
    print(f"  Hit ratio: {wt_cache.hit_ratio:.1%}")
    print(f"  Consistency: STRONG (cache = DB always)")

    # ── Write-Back ────────────────────────────
    print("\n\n[2] WRITE-BACK (WRITE-BEHIND) CACHE")
    print("─" * 55)
    wb_cache = InMemoryCache()
    wb_db    = PrimaryDB(latency_ms=8.0)
    wb       = WriteBackCache(wb_cache, wb_db, flush_interval_s=0.5, batch_size=50)

    print(f"  Writing {N_WRITES} events (cache only — async DB flush)...")
    start = time.perf_counter()
    for i in range(N_WRITES):
        wb.set(f"event:{i}", {"id": i, "type": "click", "ts": time.time()})
    total_write_ms = (time.perf_counter() - start) * 1000

    print(f"  Total write time: {total_write_ms:.2f}ms  avg/write: {wb.avg_write_latency_ms:.3f}ms")
    print(f"  Cache size: {wb_cache.size()}  Pending DB writes: {wb.pending_writes}")
    print(f"  DB writes so far: {wb_db.writes}  (async — not yet flushed)")

    # Flush
    flushed = wb.flush()
    print(f"\n  After flush: flushed={flushed} entries  DB writes: {wb_db.writes}")
    print(f"  Speedup vs write-through: write-back cache writes ~{wt.avg_write_latency_ms / max(wb.avg_write_latency_ms, 0.001):.0f}x faster")
    print(f"  ⚠ Risk: if crash before flush, {N_WRITES} writes LOST")

    # ── Write-Around ──────────────────────────
    print("\n\n[3] WRITE-AROUND CACHE")
    print("─" * 55)
    wa_cache = InMemoryCache()
    wa_db    = PrimaryDB(latency_ms=8.0)
    wa       = WriteAroundCache(wa_cache, wa_db)

    print(f"  Writing {N_WRITES} log entries (bypass cache → DB only)...")
    for i in range(N_WRITES):
        wa.set(f"log:{i}", {"message": f"Log entry {i}", "level": "INFO"})
    print(f"  Cache size after writes: {wa_cache.size()}  (0 — cache bypassed)")
    print(f"  DB writes: {wa_db.writes}")

    # Only specific reads populate cache
    print(f"\n  Reading 5 log entries (first read → DB, subsequent → cache):")
    for i in range(5):
        wa.get(f"log:{i}")
    print(f"  Cache size: {wa_cache.size()}  (only read entries cached)")
    for i in range(5):
        wa.get(f"log:{i}")
    print(f"  Second read of same 5: cache hits={wa_cache.hits}  misses={wa_cache.misses}")

    # ── Latency Comparison ─────────────────────
    print("\n\n[4] WRITE STRATEGY LATENCY COMPARISON")
    print("─" * 55)
    print(f"  DB write latency: ~8ms per write")
    print()
    strategies = [
        ("Write-Through",  f"{wt.avg_write_latency_ms:.1f}ms",  "cache + DB sync",   "Strong",   "None"),
        ("Write-Back",     f"{wb.avg_write_latency_ms:.3f}ms",  "cache only (async)","Eventual", "Crash window"),
        ("Write-Around",   f"{wa.avg_write_latency_ms:.1f}ms",  "DB only",           "Strong",   "None"),
    ]
    print(f"  {'Strategy':<16} {'Latency':<12} {'Path':<22} {'Consistency':<12} {'Data Loss'}")
    print(f"  {'─'*80}")
    for name, lat, path, cons, risk in strategies:
        print(f"  {name:<16} {lat:<12} {path:<22} {cons:<12} {risk}")

    # ── Decision Guide ─────────────────────────
    print("\n\n[5] WRITE STRATEGY SELECTION GUIDE")
    print("─" * 55)
    guide = [
        ("Write-Through",
         "Financial transactions, inventory, user profiles",
         "Write-then-read workload, strong consistency critical"),
        ("Write-Back",
         "Analytics counters, social likes, IoT sensors, gaming scores",
         "High write throughput, some data loss acceptable"),
        ("Write-Around",
         "Logs, raw media uploads, cold data",
         "Write-once, rarely re-read — avoid cache pollution"),
        ("Cache-Aside",
         "General reads, mixed workloads",
         "Want control over what gets cached, read-heavy"),
    ]
    for strategy, use_cases, when in guide:
        print(f"\n  {strategy}:")
        print(f"    Use for: {use_cases}")
        print(f"    When:    {when}")

    # ── Combined Example ───────────────────────
    print("\n\n[6] COMBINED STRATEGY (production example)")
    print("─" * 55)
    combined = [
        ("Orders / Payments",   "Write-Through",  "ACID + instant cache consistency"),
        ("Product views count", "Write-Back",     "High write volume, eventual ok"),
        ("User uploads (S3)",   "Write-Around",   "Large blob, don't cache immediately"),
        ("User sessions",       "Write-Through",  "Must read after write instantly"),
        ("Search suggestions",  "Write-Around",   "DB is source, full-text rebuild async"),
    ]
    print(f"  {'Data Type':<24} {'Strategy':<16} {'Reason'}")
    print(f"  {'─'*65}")
    for data, strategy, reason in combined:
        print(f"  {data:<24} {strategy:<16} {reason}")


if __name__ == "__main__":
    demonstrate_write_strategies()
