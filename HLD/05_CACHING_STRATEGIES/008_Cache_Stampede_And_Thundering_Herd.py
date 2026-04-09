"""
CACHE STAMPEDE AND THUNDERING HERD
=====================================

Problem Statement:
When a popular cache entry expires, hundreds of concurrent requests all miss
simultaneously and rush to query the database. Each request, seeing a cache
miss, independently fires a DB query. The DB is suddenly overwhelmed by N
identical queries all at once — the "thundering herd" or "cache stampede."

Scenario:
  - Cache key "homepage" expires at 12:00:00.000
  - 500 requests arrive at 12:00:00.001
  - All 500 see cache miss → all 500 fire DB query
  - DB receives 500x normal load → timeout cascade

Solutions:

  1. Mutex/Lock on Cache Miss (Request Coalescing):
     First thread to see miss acquires a per-key lock.
     Subsequent threads wait. First thread queries DB, populates cache.
     All waiting threads read from cache once populated.
     Problem: threads block while waiting — adds latency.

  2. Probabilistic Early Recompute (PER):
     Before TTL expires, probabilistically decide to recompute.
     Probability increases as entry approaches expiry.
     Background refresh: no stampede, entry never actually expires.
     Formula: P = exp(-(ttl_remaining) / (beta * compute_time))

  3. Stale-While-Revalidate:
     Serve stale entry immediately; one background thread refreshes.
     Eliminates stampede — only one DB query despite N concurrent reads.

  4. TTL Jitter:
     Add random offset to TTL: ttl + random(0, jitter).
     Prevents synchronized mass expiry of many keys.
     Spreads load over time rather than one spike.

  5. Hot Key Duplication:
     For extremely hot keys, maintain N copies (hot-key#0, hot-key#1...).
     Each copy has a slightly offset TTL — one expires at a time.

  6. Background Refresh Thread:
     Dedicated thread refreshes entries proactively before expiry.
     Entries never actually expire in production.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
import time
import threading
import random
import math
from collections import defaultdict


# ─────────────────────────────────────────────
# SIMULATED DATABASE
# ─────────────────────────────────────────────

class SlowDatabase:
    def __init__(self, latency_ms: float = 50.0):
        self._data    = {"homepage": {"content": "Home page v1", "version": 1}}
        self.queries  = 0
        self.latency  = latency_ms
        self._lock    = threading.Lock()

    def query(self, key: str) -> Optional[Any]:
        time.sleep(self.latency / 1000)
        with self._lock:
            self.queries += 1
        return self._data.get(key)


# ─────────────────────────────────────────────
# NO PROTECTION (naive cache)
# ─────────────────────────────────────────────

class NaiveCache:
    """No stampede protection — all threads query DB independently on miss."""

    def __init__(self, db: SlowDatabase):
        self.db   = db
        self._store : Dict[str, Any] = {}
        self._expires: Dict[str, float] = {}
        self.hits  = 0
        self.misses= 0

    def get(self, key: str, ttl_s: float = 0.2) -> Optional[Any]:
        entry = self._store.get(key)
        if entry is not None and time.time() < self._expires.get(key, 0):
            self.hits += 1
            return entry

        # Cache miss — every thread independently queries DB (stampede!)
        self.misses += 1
        val = self.db.query(key)
        self._store[key]   = val
        self._expires[key] = time.time() + ttl_s
        return val


# ─────────────────────────────────────────────
# MUTEX / REQUEST COALESCING
# ─────────────────────────────────────────────

class MutexProtectedCache:
    """
    Per-key lock ensures only one thread queries DB on miss.
    Other threads wait for the first thread to populate cache.
    """

    def __init__(self, db: SlowDatabase):
        self.db         = db
        self._store     : Dict[str, Any]   = {}
        self._expires   : Dict[str, float] = {}
        self._locks     : Dict[str, threading.Lock] = {}
        self._meta_lock = threading.Lock()
        self.hits       = 0
        self.misses     = 0
        self.coalesced  = 0   # requests served from cache after waiting for lock

    def _get_lock(self, key: str) -> threading.Lock:
        with self._meta_lock:
            if key not in self._locks:
                self._locks[key] = threading.Lock()
            return self._locks[key]

    def get(self, key: str, ttl_s: float = 0.2) -> Optional[Any]:
        # Fast path — check without lock
        entry = self._store.get(key)
        if entry is not None and time.time() < self._expires.get(key, 0):
            self.hits += 1
            return entry

        # Slow path — acquire per-key lock
        lock = self._get_lock(key)
        with lock:
            # Double-check after acquiring lock (another thread may have populated)
            entry = self._store.get(key)
            if entry is not None and time.time() < self._expires.get(key, 0):
                self.hits    += 1
                self.coalesced += 1
                return entry

            # This thread is responsible for querying DB
            self.misses += 1
            val = self.db.query(key)
            self._store[key]   = val
            self._expires[key] = time.time() + ttl_s
            return val


# ─────────────────────────────────────────────
# PROBABILISTIC EARLY RECOMPUTE (PER)
# ─────────────────────────────────────────────

class ProbabilisticEarlyRecomputeCache:
    """
    Before expiry, probabilistically decide to recompute (background refresh).
    Uses the XFetch algorithm: P = exp(-ttl_remaining / (beta * compute_time))
    As expiry approaches, probability approaches 1.0 → entry recomputed early.
    """

    def __init__(self, db: SlowDatabase, beta: float = 1.0):
        self.db       = db
        self.beta     = beta   # higher beta → more aggressive early refresh
        self._store   : Dict[str, Dict] = {}
        self.hits     = 0
        self.misses   = 0
        self.early_refreshes = 0

    def _should_early_recompute(self, entry: Dict) -> bool:
        ttl_remaining = entry["expires_at"] - time.time()
        if ttl_remaining <= 0:
            return True
        compute_time = entry.get("compute_time_s", 0.05)
        # XFetch: P = exp(-(ttl_remaining) / (beta * compute_time))
        prob = math.exp(-ttl_remaining / max(0.001, self.beta * compute_time))
        return random.random() < prob

    def get(self, key: str, ttl_s: float = 1.0) -> Optional[Any]:
        entry = self._store.get(key)
        now   = time.time()

        if entry is None or now > entry["expires_at"]:
            # Cache miss
            self.misses += 1
            start = time.perf_counter()
            val   = self.db.query(key)
            compute_time = time.perf_counter() - start
            self._store[key] = {
                "value": val, "expires_at": now + ttl_s,
                "compute_time_s": compute_time
            }
            return val

        if self._should_early_recompute(entry):
            # Proactively refresh before expiry (in background)
            self.early_refreshes += 1
            self._background_refresh(key, ttl_s)

        self.hits += 1
        return entry["value"]

    def _background_refresh(self, key: str, ttl_s: float):
        def refresh():
            val = self.db.query(key)
            self._store[key] = {
                "value": val, "expires_at": time.time() + ttl_s,
                "compute_time_s": 0.05
            }
        t = threading.Thread(target=refresh, daemon=True)
        t.start()


# ─────────────────────────────────────────────
# TTL JITTER
# ─────────────────────────────────────────────

class JitteredTTLCache:
    """
    Adds random jitter to TTL to prevent synchronized mass expiry.
    Instead of 1000 keys all expiring at T+60s, they expire over T+55s to T+65s.
    """

    def __init__(self, base_ttl_s: float = 60.0, jitter_s: float = 10.0):
        self.base_ttl  = base_ttl_s
        self.jitter_s  = jitter_s
        self._store    : Dict[str, Any]   = {}
        self._expires  : Dict[str, float] = {}

    def _jittered_ttl(self) -> float:
        return self.base_ttl + random.uniform(-self.jitter_s / 2, self.jitter_s / 2)

    def set(self, key: str, value: Any):
        ttl = self._jittered_ttl()
        self._store[key]   = value
        self._expires[key] = time.time() + ttl

    def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if entry is None or time.time() > self._expires.get(key, 0):
            return None
        return entry

    def expiry_distribution(self, n_keys: int) -> Dict[str, int]:
        """Show how jitter spreads expiry times."""
        buckets : Dict[str, int] = defaultdict(int)
        base    = time.time()
        for _ in range(n_keys):
            ttl = self._jittered_ttl()
            bucket = int((ttl - self.base_ttl + self.jitter_s / 2) / (self.jitter_s / 10))
            buckets[f"T+{self.base_ttl + bucket * (self.jitter_s / 10):.0f}s"] += 1
        return dict(sorted(buckets.items()))


# ─────────────────────────────────────────────
# STAMPEDE SIMULATOR
# ─────────────────────────────────────────────

class StampedeSimulator:
    """Simulates N concurrent requests hitting an expired cache entry."""

    def __init__(self, n_threads: int = 20):
        self.n_threads = n_threads

    def simulate(self, cache_cls, **cache_kwargs) -> Dict:
        db    = SlowDatabase(latency_ms=30.0)
        cache = cache_cls(db=db, **cache_kwargs)

        # Pre-warm cache (short TTL for demo)
        cache.get("homepage", ttl_s=0.05)
        time.sleep(0.06)   # let it expire

        # N concurrent requests after expiry
        db.queries = 0
        results = []
        start   = time.perf_counter()

        threads = [
            threading.Thread(target=lambda: results.append(cache.get("homepage", ttl_s=0.5)))
            for _ in range(self.n_threads)
        ]
        for t in threads: t.start()
        for t in threads: t.join()

        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "db_queries"  : db.queries,
            "cache_hits"  : getattr(cache, "hits", 0),
            "coalesced"   : getattr(cache, "coalesced", 0),
            "elapsed_ms"  : elapsed_ms,
            "all_returned": all(r is not None for r in results),
        }


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_cache_stampede():
    print("=" * 65)
    print("CACHE STAMPEDE AND THUNDERING HERD")
    print("=" * 65)

    random.seed(42)
    N = 20   # concurrent threads

    # ── Naive vs Protected ─────────────────────
    print(f"\n[1] CACHE STAMPEDE: {N} CONCURRENT REQUESTS ON EXPIRED ENTRY")
    print("─" * 55)
    sim = StampedeSimulator(n_threads=N)

    print("  Running naive cache (no protection)...")
    naive = sim.simulate(NaiveCache)
    print(f"  Naive:   DB queries={naive['db_queries']}  elapsed={naive['elapsed_ms']:.0f}ms")
    print(f"           → {naive['db_queries']} identical DB queries fired simultaneously!")

    print("\n  Running mutex-protected cache...")
    protected = sim.simulate(MutexProtectedCache)
    print(f"  Mutex:   DB queries={protected['db_queries']}  coalesced={protected['coalesced']}  "
          f"elapsed={protected['elapsed_ms']:.0f}ms")
    print(f"           → only {protected['db_queries']} DB query, {protected['coalesced']} requests waited + got cache hit")

    reduction = (naive['db_queries'] - protected['db_queries']) / naive['db_queries']
    print(f"\n  DB query reduction: {reduction:.0%}  (from {naive['db_queries']} to {protected['db_queries']})")

    # ── TTL Jitter ────────────────────────────
    print("\n\n[2] TTL JITTER — SPREADING EXPIRY OVER TIME")
    print("─" * 55)
    jitter_cache = JitteredTTLCache(base_ttl_s=60.0, jitter_s=20.0)

    # No jitter: all expire at T+60 → 1000 simultaneous misses
    print("  WITHOUT jitter: 1000 keys all expire at exactly T+60s")
    print("  → 1000 simultaneous cache misses → stampede")

    # With jitter: distributed expiry
    dist = jitter_cache.expiry_distribution(n_keys=1000)
    print(f"\n  WITH jitter (base=60s, jitter=±10s): 1000 keys expiry distribution:")
    for bucket, count in list(dist.items())[:10]:
        bar = "█" * (count // 5)
        print(f"    {bucket}: {count:4d} keys  {bar}")
    print(f"  → Load spread over ~20s window instead of all at once")

    # ── Probabilistic Early Recompute ─────────
    print("\n\n[3] PROBABILISTIC EARLY RECOMPUTE (PER/XFetch)")
    print("─" * 55)
    per_db    = SlowDatabase(latency_ms=20.0)
    per_cache = ProbabilisticEarlyRecomputeCache(per_db, beta=1.0)

    # Access pattern over time
    print("  Simulating 10 reads over time (TTL=0.3s):")
    for i in range(10):
        val = per_cache.get("data", ttl_s=0.3)
        print(f"  t={i*0.05:.2f}s: {'HIT' if per_cache.hits > 0 else 'MISS'} "
              f"early_refreshes={per_cache.early_refreshes}")
        time.sleep(0.05)

    print(f"\n  Total: hits={per_cache.hits}  misses={per_cache.misses}  "
          f"early_refreshes={per_cache.early_refreshes}  DB queries={per_db.queries}")
    print("  → Early refreshes prevent expiry without user-visible miss")

    # ── Hot Key Duplication ────────────────────
    print("\n\n[4] HOT KEY DUPLICATION (distributing hot key load)")
    print("─" * 55)
    N_SHARDS = 4
    print(f"  Hot key 'featured_product' → sharded to {N_SHARDS} copies:")
    for i in range(N_SHARDS):
        shard_key = f"featured_product#{i}"
        ttl = 60 + i * 15   # staggered TTLs: 60s, 75s, 90s, 105s
        print(f"    {shard_key}: TTL={ttl}s")
    print(f"\n  Each request picks random shard → load distributed across {N_SHARDS} nodes")
    print(f"  Staggered TTLs → only 1 shard expires at a time → no simultaneous stampede")

    # ── Background Refresh ─────────────────────
    print("\n\n[5] BACKGROUND REFRESH (proactive cache warming)")
    print("─" * 55)
    print("  Strategy: dedicated thread refreshes entries BEFORE they expire")
    print()
    steps = [
        ("Monitor",   "Track TTL remaining for hot keys"),
        ("Threshold", "When TTL < 20% remaining, queue for refresh"),
        ("Refresh",   "Background thread fetches fresh value from DB"),
        ("Update",    "Atomically replace cache entry"),
        ("Result",    "Entry never actually expires in production"),
    ]
    for step, desc in steps:
        print(f"  {step:<12} {desc}")

    print()
    print("  Pseudocode:")
    print("    while True:")
    print("      for key in watched_keys:")
    print("        if cache.ttl_remaining(key) < 0.2 * base_ttl:")
    print("          new_val = db.query(key)")
    print("          cache.set(key, new_val, ttl=base_ttl + jitter())")
    print("      sleep(10)")

    # ── Strategy Comparison ────────────────────
    print("\n\n[6] STAMPEDE PROTECTION STRATEGIES COMPARISON")
    print("─" * 55)
    strategies = [
        ("No protection",      "Simple",   "All threads hit DB",    "Never for hot keys"),
        ("Mutex (coalescing)", "Low",      "1 DB query, others wait","Most common approach"),
        ("Stale-while-reval",  "Low",      "1 background refresh",  "CDN, public content"),
        ("PER (XFetch)",       "Medium",   "Proactive before expiry","Variable load spikes"),
        ("TTL Jitter",         "Low",      "Spreads expiry times",  "Mass-TTL scenarios"),
        ("Hot key sharding",   "Medium",   "N copies, 1 expires/time","Extreme hot keys"),
        ("Background refresh", "High",     "Entry never expires",   "Critical hot paths"),
    ]
    print(f"  {'Strategy':<24} {'Complexity':<10} {'DB Impact':<26} {'Use When'}")
    print(f"  {'─'*80}")
    for strategy, complexity, db_impact, use_when in strategies:
        print(f"  {strategy:<24} {complexity:<10} {db_impact:<26} {use_when}")


if __name__ == "__main__":
    demonstrate_cache_stampede()
