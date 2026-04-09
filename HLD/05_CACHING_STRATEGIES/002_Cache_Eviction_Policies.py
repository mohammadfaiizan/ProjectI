"""
CACHE EVICTION POLICIES
==========================

Problem Statement:
Cache memory is finite. When the cache is full, which entry to remove to
maximize future hit ratio? The eviction policy determines cache effectiveness.

Eviction Policies:

  LRU (Least Recently Used):
    Evict the entry that hasn't been accessed in the longest time.
    O(1) with doubly linked list + hashmap.
    Best for: general workloads with temporal locality.
    Weakness: "scan" of large dataset pollutes cache (cache pollution).

  LFU (Least Frequently Used):
    Evict the entry accessed the fewest times.
    O(1) with frequency buckets + hashmap.
    Best for: stable "popularity" distributions (music streaming).
    Weakness: newly inserted hot items have low count → premature eviction.

  FIFO (First In First Out):
    Evict the oldest inserted entry, regardless of access.
    O(1). Simple, predictable.
    Best for: streaming data where recency matters linearly.
    Weakness: evicts hot entries that happen to be old.

  TTL (Time-To-Live):
    Entries expire after a fixed time window.
    Correctness-focused: ensures data freshness.
    Not a capacity eviction — entries auto-expire.
    Combined with LRU for both freshness and capacity management.

  Random:
    Evict a random entry.
    O(1). Surprisingly competitive vs LRU in many workloads.
    Used in: CPU hardware caches.

  ARC (Adaptive Replacement Cache):
    Self-tuning split between recency (LRU) and frequency (LFU).
    Maintains 4 lists: T1 (recent), T2 (frequent), B1 (ghost recent), B2 (ghost frequent).
    Adapts the T1/T2 split based on miss patterns.
    Used in ZFS, some storage systems.

  SLRU (Segmented LRU):
    Splits cache into probation and protected segments.
    New entries go to probation; re-accessed entries promoted to protected.
    Redis uses this in volatile segments.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, OrderedDict
import random
import time


# ─────────────────────────────────────────────
# LRU CACHE — O(1) with OrderedDict
# ─────────────────────────────────────────────

class LRUCache:
    """Least Recently Used — O(1) get/set using OrderedDict."""

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
        self._store.move_to_end(key)   # mark as recently used
        self.hits += 1
        return self._store[key]

    def put(self, key: str, value: Any):
        if key in self._store:
            self._store.move_to_end(key)
        else:
            if len(self._store) >= self.capacity:
                self._store.popitem(last=False)   # remove LRU (oldest)
                self.evictions += 1
        self._store[key] = value

    @property
    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def size(self) -> int:
        return len(self._store)


# ─────────────────────────────────────────────
# LFU CACHE — O(1) with frequency buckets
# ─────────────────────────────────────────────

class LFUCache:
    """
    Least Frequently Used — O(1) using:
    - key → (value, freq)
    - freq → OrderedDict[key] (ordered by insertion for LRU tiebreak)
    - min_freq tracker
    """

    def __init__(self, capacity: int):
        self.capacity  = capacity
        self._vals     : Dict[str, Any] = {}
        self._freq     : Dict[str, int] = {}
        self._buckets  : Dict[int, OrderedDict] = defaultdict(OrderedDict)
        self._min_freq = 0
        self.hits      = 0
        self.misses    = 0
        self.evictions = 0

    def get(self, key: str) -> Optional[Any]:
        if key not in self._vals:
            self.misses += 1
            return None
        self._increment(key)
        self.hits += 1
        return self._vals[key]

    def put(self, key: str, value: Any):
        if self.capacity <= 0:
            return
        if key in self._vals:
            self._vals[key] = value
            self._increment(key)
            return
        if len(self._vals) >= self.capacity:
            # Evict LFU (and LRU among ties)
            lfu_bucket = self._buckets[self._min_freq]
            evict_key, _ = lfu_bucket.popitem(last=False)
            del self._vals[evict_key]
            del self._freq[evict_key]
            self.evictions += 1
        self._vals[key]      = value
        self._freq[key]      = 1
        self._buckets[1][key] = True
        self._min_freq       = 1

    def _increment(self, key: str):
        f = self._freq[key]
        self._freq[key] = f + 1
        del self._buckets[f][key]
        if not self._buckets[f] and f == self._min_freq:
            self._min_freq += 1
        self._buckets[f + 1][key] = True

    @property
    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def size(self) -> int:
        return len(self._vals)


# ─────────────────────────────────────────────
# FIFO CACHE
# ─────────────────────────────────────────────

class FIFOCache:
    """First In First Out — evicts oldest inserted regardless of access."""

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
        # FIFO: do NOT move to end on access
        self.hits += 1
        return self._store[key]

    def put(self, key: str, value: Any):
        if key in self._store:
            self._store[key] = value   # update value but keep position
            return
        if len(self._store) >= self.capacity:
            self._store.popitem(last=False)   # remove oldest inserted
            self.evictions += 1
        self._store[key] = value

    @property
    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


# ─────────────────────────────────────────────
# RANDOM EVICTION CACHE
# ─────────────────────────────────────────────

class RandomCache:
    """Evicts a random entry — simple, no overhead, surprisingly competitive."""

    def __init__(self, capacity: int):
        self.capacity  = capacity
        self._store    : Dict[str, Any] = {}
        self.hits      = 0
        self.misses    = 0
        self.evictions = 0

    def get(self, key: str) -> Optional[Any]:
        val = self._store.get(key)
        if val is None:
            self.misses += 1
            return None
        self.hits += 1
        return val

    def put(self, key: str, value: Any):
        if key not in self._store and len(self._store) >= self.capacity:
            evict_key = random.choice(list(self._store.keys()))
            del self._store[evict_key]
            self.evictions += 1
        self._store[key] = value

    @property
    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


# ─────────────────────────────────────────────
# SEGMENTED LRU (SLRU)
# ─────────────────────────────────────────────

class SLRUCache:
    """
    Segmented LRU: probation + protected segments.
    New entries → probation (smaller).
    Re-accessed entries → protected (larger).
    Protects hot entries from one-time scan pollution.
    """

    def __init__(self, capacity: int, protected_ratio: float = 0.8):
        protected_cap = max(1, int(capacity * protected_ratio))
        probation_cap = max(1, capacity - protected_cap)
        self._protected = LRUCache(protected_cap)
        self._probation = LRUCache(probation_cap)
        self.hits       = 0
        self.misses     = 0
        self.evictions  = 0

    def get(self, key: str) -> Optional[Any]:
        # Check protected first
        val = self._protected.get(key)
        if val is not None:
            self.hits += 1
            return val
        # Check probation
        val = self._probation.get(key)
        if val is not None:
            # Promote to protected
            self._protected.put(key, val)
            self.hits += 1
            return val
        self.misses += 1
        return None

    def put(self, key: str, value: Any):
        # New entries go to probation
        self._probation.put(key, value)

    @property
    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


# ─────────────────────────────────────────────
# POLICY COMPARATOR
# ─────────────────────────────────────────────

class PolicyComparator:
    """Runs access trace against multiple eviction policies and compares."""

    def __init__(self, capacity: int):
        self.capacity = capacity

    def _run(self, cache, trace: List[str]) -> float:
        for key in trace:
            val = cache.get(key)
            if val is None:
                cache.put(key, f"val_{key}")
        return cache.hit_ratio

    def compare(self, trace: List[str]) -> Dict[str, float]:
        caches = {
            "LRU"   : LRUCache(self.capacity),
            "LFU"   : LFUCache(self.capacity),
            "FIFO"  : FIFOCache(self.capacity),
            "Random": RandomCache(self.capacity),
            "SLRU"  : SLRUCache(self.capacity),
        }
        results = {}
        for name, cache in caches.items():
            results[name] = self._run(cache, trace)
        return results

    @staticmethod
    def generate_trace_pareto(n_requests: int, n_keys: int) -> List[str]:
        """80% of requests target top 20% of keys."""
        hot  = [f"k{i}" for i in range(int(n_keys * 0.2))]
        cold = [f"k{i}" for i in range(int(n_keys * 0.2), n_keys)]
        trace = []
        for _ in range(n_requests):
            if random.random() < 0.8:
                trace.append(random.choice(hot))
            else:
                trace.append(random.choice(cold))
        return trace

    @staticmethod
    def generate_trace_scan(n_requests: int, n_keys: int) -> List[str]:
        """Sequential scan — worst case for LRU (cache pollution)."""
        trace = []
        for i in range(n_requests):
            trace.append(f"k{i % n_keys}")
        return trace

    @staticmethod
    def generate_trace_recency(n_requests: int, n_keys: int) -> List[str]:
        """Recent keys are more likely to be requested again."""
        trace = []
        window = min(n_keys, 50)   # last 50 accessed keys are hot
        recent = []
        for _ in range(n_requests):
            if recent and random.random() < 0.7:
                trace.append(random.choice(recent[-window:]))
            else:
                key = f"k{random.randint(0, n_keys - 1)}"
                recent.append(key)
                trace.append(key)
        return trace


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_eviction_policies():
    print("=" * 65)
    print("CACHE EVICTION POLICIES")
    print("=" * 65)

    random.seed(42)

    # ── LRU Mechanics ─────────────────────────
    print("\n[1] LRU MECHANICS (capacity=3)")
    print("─" * 55)
    lru = LRUCache(capacity=3)
    ops = [("PUT", "A"), ("PUT", "B"), ("PUT", "C"), ("GET", "A"),
           ("PUT", "D"), ("GET", "B"), ("PUT", "E")]
    for op, key in ops:
        if op == "PUT":
            lru.put(key, f"val_{key}")
            contents = list(lru._store.keys())
            print(f"  PUT {key}  → cache={contents}")
        else:
            val = lru.get(key)
            contents = list(lru._store.keys())
            result = "HIT" if val else "MISS"
            print(f"  GET {key}  → {result}  cache={contents}")

    # ── LFU Mechanics ─────────────────────────
    print("\n\n[2] LFU MECHANICS (capacity=3)")
    print("─" * 55)
    lfu = LFUCache(capacity=3)
    ops_lfu = [("PUT","X"), ("PUT","Y"), ("PUT","Z"),
               ("GET","X"), ("GET","X"), ("GET","Y"),
               ("PUT","W")]   # evicts Z (freq=1) not Y(freq=2) or X(freq=3)
    for op, key in ops_lfu:
        if op == "PUT":
            lfu.put(key, f"val_{key}")
            freqs = {k: lfu._freq[k] for k in lfu._vals}
            print(f"  PUT {key}  → cache_freqs={freqs}")
        else:
            lfu.get(key)
            freqs = {k: lfu._freq[k] for k in lfu._vals}
            print(f"  GET {key}  → cache_freqs={freqs}")

    # ── Policy Comparison: Pareto ──────────────
    print("\n\n[3] POLICY COMPARISON — PARETO ACCESS (80/20)")
    print("─" * 55)
    comparator = PolicyComparator(capacity=20)   # cache 20 of 100 keys
    trace_pareto = PolicyComparator.generate_trace_pareto(2000, 100)
    results_pareto = comparator.compare(trace_pareto)
    print(f"  Trace: 2000 requests, 100 keys, cache_size=20")
    for policy, ratio in sorted(results_pareto.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(ratio * 50)
        print(f"  {policy:<8} {ratio:.1%}  {bar}")

    # ── Policy Comparison: Scan (LRU weakness) ─
    print("\n\n[4] POLICY COMPARISON — SEQUENTIAL SCAN (LRU weakness)")
    print("─" * 55)
    trace_scan = PolicyComparator.generate_trace_scan(2000, 100)
    results_scan = comparator.compare(trace_scan)
    print(f"  Trace: sequential scan, 2000 requests, 100 keys, cache_size=20")
    for policy, ratio in sorted(results_scan.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(ratio * 50)
        print(f"  {policy:<8} {ratio:.1%}  {bar}")
    print("  → LRU hurt by scan: new entries evict hot ones (cache pollution)")
    print("  → SLRU probation segment protects hot entries from scans")

    # ── Policy Comparison: Recency ─────────────
    print("\n\n[5] POLICY COMPARISON — RECENCY BIAS (LRU strength)")
    print("─" * 55)
    trace_recency = PolicyComparator.generate_trace_recency(2000, 100)
    results_recency = comparator.compare(trace_recency)
    print(f"  Trace: recent keys more likely re-accessed, 100 keys, cache_size=20")
    for policy, ratio in sorted(results_recency.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(ratio * 50)
        print(f"  {policy:<8} {ratio:.1%}  {bar}")
    print("  → LRU wins when recent = likely reused (most web workloads)")

    # ── When to Use Which ──────────────────────
    print("\n\n[6] EVICTION POLICY SELECTION GUIDE")
    print("─" * 55)
    guide = [
        ("LRU",    "General web cache, session store",     "O(1) with OrderedDict",  "Scan pollution"),
        ("LFU",    "Music/video streaming, ad targeting",  "O(1) with freq buckets", "Cold start for new items"),
        ("FIFO",   "Log buffer, simple streaming",         "O(1) ring buffer",       "Evicts hot old entries"),
        ("SLRU",   "Database buffer pool (InnoDB)",        "O(1) two-segment",       "More complex to tune"),
        ("Random", "CPU L2 cache, simplicity needed",      "O(1) trivial",           "No access pattern awareness"),
        ("TTL",    "API response cache, session expiry",   "O(1) + background sweep","Not capacity-based"),
        ("ARC",    "ZFS, storage systems",                 "O(1) self-tuning",       "Complex implementation"),
    ]
    print(f"  {'Policy':<8} {'Use Case':<35} {'Complexity':<25} {'Weakness'}")
    print(f"  {'─'*90}")
    for policy, use_case, complexity, weakness in guide:
        print(f"  {policy:<8} {use_case:<35} {complexity:<25} {weakness}")


if __name__ == "__main__":
    demonstrate_eviction_policies()
