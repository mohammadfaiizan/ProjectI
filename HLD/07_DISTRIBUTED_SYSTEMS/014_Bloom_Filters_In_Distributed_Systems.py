"""
BLOOM FILTERS IN DISTRIBUTED SYSTEMS
=======================================

Problem Statement:
Given a large set S, can you quickly answer "is element X in S?"
Naive: store all elements. O(N) space, O(1) amortized.
Problem: for billions of elements, storing all IDs is impractical.
Bloom filter: probabilistic data structure. Space-efficient. O(1) time.

How It Works:
  Fixed-size bit array of M bits, all initialized to 0.
  K independent hash functions h1...hk.

  Add(x):    Set bits h1(x) % M, h2(x) % M, ..., hk(x) % M to 1.
  Query(x):  Check all K bit positions.
             If any bit is 0 → x is DEFINITELY NOT in the set.
             If all bits are 1 → x is PROBABLY in the set (may be false positive).

Properties:
  False positives: possible (other elements set those bits).
  False negatives: IMPOSSIBLE (a member's bits are never cleared).
  Cannot delete elements (use Counting Bloom Filter for deletion).
  Space: O(M) bits regardless of element count.

Optimal Parameters:
  Given N elements and desired false positive rate p:
    M = -N * ln(p) / (ln(2))²  bits
    K = (M / N) * ln(2)        hash functions
  Example: 1M elements, p=1%: M≈9.6MB (vs 8MB for raw 64-bit IDs), K≈7.

Use Cases in Distributed Systems:
  1. Cassandra: checks Bloom filter before SSTable I/O.
     Is the key in this SSTable? Avoid disk read if NO (no false negatives).
     ~1% false positive rate → 99% of unnecessary disk reads eliminated.

  2. Google Bigtable: row bloom filter to skip unnecessary tablet lookups.

  3. Akamai CDN: "Is this URL likely to be requested again?"
     Cache only URLs seen ≥ 2 times. Use Bloom filter to check first occurrence.

  4. Distributed deduplication: "Have we seen this URL/event ID before?"
     Sharded Bloom filters, one per shard. Fast membership check.

  5. Network routers: IP blacklist check at line rate. O(1) with tiny memory.

  6. Chrome malware URLs: downloaded Bloom filter checked locally before server query.

Counting Bloom Filter:
  Replace each bit with a counter. Allows deletion by decrementing.
  Uses 4x more space. Used when deletions needed.

Scalable Bloom Filter:
  Starts small, adds new filters as capacity fills.
  Each new filter has lower false positive rate.
  Capacity scales without knowing N upfront.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import math
import hashlib
import random
import time


# ─────────────────────────────────────────────
# BLOOM FILTER
# ─────────────────────────────────────────────

class BloomFilter:
    """
    Classic Bloom filter using multiple hash functions (simulated via salted MD5).
    """

    def __init__(self, expected_elements: int, false_positive_rate: float = 0.01):
        self.n = expected_elements
        self.p = false_positive_rate

        # Optimal size and hash count
        self.m = max(1, int(-self.n * math.log(self.p) / (math.log(2) ** 2)))
        self.k = max(1, int((self.m / self.n) * math.log(2)))

        self._bits          = bytearray((self.m + 7) // 8)
        self._count         = 0
        self.false_positives_detected = 0

    def _hash(self, item: str, seed: int) -> int:
        h = hashlib.md5(f"{seed}:{item}".encode()).hexdigest()
        return int(h, 16) % self.m

    def _bit_positions(self, item: str) -> List[int]:
        return [self._hash(item, i) for i in range(self.k)]

    def add(self, item: str):
        for pos in self._bit_positions(item):
            byte_idx = pos // 8
            bit_idx  = pos % 8
            self._bits[byte_idx] |= (1 << bit_idx)
        self._count += 1

    def might_contain(self, item: str) -> bool:
        """True = probably in set. False = definitely NOT in set."""
        for pos in self._bit_positions(item):
            byte_idx = pos // 8
            bit_idx  = pos % 8
            if not (self._bits[byte_idx] & (1 << bit_idx)):
                return False
        return True

    def size_bytes(self) -> int:
        return len(self._bits)

    def fill_ratio(self) -> float:
        """Fraction of bits set to 1. Higher → more false positives."""
        set_bits = sum(bin(b).count('1') for b in self._bits)
        return set_bits / self.m

    @property
    def estimated_fp_rate(self) -> float:
        return (1 - math.exp(-self.k * self._count / self.m)) ** self.k


# ─────────────────────────────────────────────
# COUNTING BLOOM FILTER (supports deletion)
# ─────────────────────────────────────────────

class CountingBloomFilter:
    def __init__(self, expected_elements: int, false_positive_rate: float = 0.01):
        m_bits = max(1, int(-expected_elements * math.log(false_positive_rate) / (math.log(2) ** 2)))
        self.m = m_bits
        self.k = max(1, int((m_bits / expected_elements) * math.log(2)))
        self._counters = [0] * self.m

    def _positions(self, item: str) -> List[int]:
        return [int(hashlib.md5(f"{i}:{item}".encode()).hexdigest(), 16) % self.m
                for i in range(self.k)]

    def add(self, item: str):
        for pos in self._positions(item):
            self._counters[pos] += 1

    def remove(self, item: str):
        for pos in self._positions(item):
            if self._counters[pos] > 0:
                self._counters[pos] -= 1

    def might_contain(self, item: str) -> bool:
        return all(self._counters[pos] > 0 for pos in self._positions(item))


# ─────────────────────────────────────────────
# SCALABLE BLOOM FILTER
# ─────────────────────────────────────────────

class ScalableBloomFilter:
    """
    Adds new Bloom filters as capacity fills.
    False positive rate tightens with each new filter.
    """

    def __init__(self, initial_capacity: int = 100, fp_rate: float = 0.01,
                 growth_factor: int = 2):
        self._filters     : List[BloomFilter] = []
        self.initial_cap  = initial_capacity
        self.fp_rate      = fp_rate
        self.growth_factor = growth_factor
        self._current_cap  = initial_capacity
        self._add_filter()

    def _add_filter(self):
        rate = self.fp_rate * (0.5 ** len(self._filters))
        self._filters.append(BloomFilter(self._current_cap, max(rate, 0.0001)))
        self._current_cap *= self.growth_factor

    def add(self, item: str):
        current = self._filters[-1]
        if current._count >= current.n:
            self._add_filter()
        self._filters[-1].add(item)

    def might_contain(self, item: str) -> bool:
        return any(f.might_contain(item) for f in self._filters)

    @property
    def filter_count(self) -> int:
        return len(self._filters)


# ─────────────────────────────────────────────
# CASSANDRA-STYLE SSTABLE FILTER
# ─────────────────────────────────────────────

class SSTableWithBloom:
    """
    Simulates Cassandra's use of Bloom filter per SSTable.
    Bloom filter tells: "does this SSTable contain key X?"
    Eliminates unnecessary disk reads for missing keys.
    """

    def __init__(self, sstable_id: str, keys: List[str], fp_rate: float = 0.01):
        self.sstable_id  = sstable_id
        self._data       = {k: f"value_of_{k}" for k in keys}
        self.bloom       = BloomFilter(len(keys) + 10, fp_rate)
        for key in keys:
            self.bloom.add(key)
        self.disk_reads  = 0
        self.bloom_hits  = 0
        self.bloom_misses = 0

    def get(self, key: str) -> Optional[str]:
        if not self.bloom.might_contain(key):
            self.bloom_misses += 1
            return None   # definitely not here — skip disk read
        # Bloom says "maybe" → must do disk read
        self.bloom_hits += 1
        self.disk_reads += 1
        return self._data.get(key)


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_bloom_filters():
    print("=" * 65)
    print("BLOOM FILTERS IN DISTRIBUTED SYSTEMS")
    print("=" * 65)

    random.seed(42)

    # ── Basic Bloom Filter ────────────────────────
    print("\n[1] BASIC BLOOM FILTER — PROPERTIES")
    print("─" * 55)

    bf = BloomFilter(expected_elements=1000, false_positive_rate=0.01)
    print(f"  Target: N=1000 elements, p=1% false positive rate")
    print(f"  Optimal bit array size: {bf.m:,} bits ({bf.size_bytes():,} bytes = {bf.size_bytes()/1024:.1f} KB)")
    print(f"  Optimal hash functions: {bf.k}")
    print(f"  Raw storage for 1000 IDs (64-bit): {1000*8:,} bytes")

    # Add 1000 elements
    added = {f"user:{i}" for i in range(1000)}
    for item in added:
        bf.add(item)

    print(f"\n  After adding 1000 elements:")
    print(f"  Fill ratio: {bf.fill_ratio():.2%}")
    print(f"  Estimated FP rate: {bf.estimated_fp_rate:.2%} (target: 1%)")

    # Check for false positives
    fp_count = 0
    test_count = 10000
    for i in range(1000, 1000 + test_count):
        if bf.might_contain(f"user:{i}"):
            fp_count += 1

    print(f"\n  False positive test: {fp_count}/{test_count} false positives "
          f"({fp_count/test_count:.2%})")

    # Zero false negatives
    fn_count = sum(1 for item in added if not bf.might_contain(item))
    print(f"  False negatives: {fn_count} (always 0)")

    # ── Counting Bloom Filter ─────────────────────
    print("\n\n[2] COUNTING BLOOM FILTER — SUPPORTS DELETION")
    print("─" * 55)

    cbf = CountingBloomFilter(100, fp_rate=0.05)
    cbf.add("session:abc")
    cbf.add("session:def")
    print(f"  Added session:abc and session:def")
    print(f"  Contains session:abc: {cbf.might_contain('session:abc')}")
    cbf.remove("session:abc")
    print(f"  After removing session:abc: {cbf.might_contain('session:abc')}")
    print(f"  session:def still present: {cbf.might_contain('session:def')}")

    # ── Scalable Bloom Filter ─────────────────────
    print("\n\n[3] SCALABLE BLOOM FILTER — GROWS DYNAMICALLY")
    print("─" * 55)

    sbf = ScalableBloomFilter(initial_capacity=50, fp_rate=0.01)
    for i in range(200):
        sbf.add(f"item:{i}")

    print(f"  Added 200 items (initial capacity was 50)")
    print(f"  Filters created: {sbf.filter_count}")
    print(f"  Contains item:0: {sbf.might_contain('item:0')}")
    print(f"  Contains item:199: {sbf.might_contain('item:199')}")
    print(f"  Contains item:999 (not added): {sbf.might_contain('item:999')}")

    # ── Cassandra SSTable Optimization ───────────
    print("\n\n[4] CASSANDRA SSTABLE BLOOM FILTER — DISK I/O SAVINGS")
    print("─" * 55)

    keys_in_sstable = [f"key:{i}" for i in range(500)]
    sstable = SSTableWithBloom("SST-001", keys_in_sstable, fp_rate=0.01)

    queries_present = [f"key:{i}" for i in range(100)]          # exist
    queries_absent  = [f"key:{i}" for i in range(1000, 1100)]   # don't exist

    for key in queries_present + queries_absent:
        sstable.get(key)

    print(f"  Queried: 100 present + 100 absent keys")
    print(f"  Disk reads: {sstable.disk_reads} (bloom passed)")
    print(f"  Bloom-filtered (no disk): {sstable.bloom_misses}")
    print(f"  → Bloom saved ~{sstable.bloom_misses/(sstable.disk_reads+sstable.bloom_misses)*100:.0f}% "
          f"of disk reads for absent keys")

    # ── Space Comparison ──────────────────────────
    print("\n\n[5] SPACE EFFICIENCY COMPARISON")
    print("─" * 55)
    configs = [
        (1_000,      0.01),
        (1_000_000,  0.01),
        (1_000_000,  0.001),
        (100_000_000, 0.01),
    ]
    print(f"  {'N elements':<16} {'FP rate':<10} {'Bloom size':<16} {'Raw 64-bit IDs':<18} {'Ratio'}")
    print(f"  {'─'*70}")
    for n, p in configs:
        m_bits   = int(-n * math.log(p) / (math.log(2) ** 2))
        bloom_kb = m_bits // 8 // 1024
        raw_kb   = n * 8 // 1024
        ratio    = raw_kb / max(bloom_kb, 1)
        print(f"  {n:<16,} {p:<10.3%} {bloom_kb:<16,}KB {raw_kb:<18,}KB {ratio:.1f}x larger raw")

    # ── Use Cases ─────────────────────────────────
    print("\n\n[6] BLOOM FILTER USE CASES IN DISTRIBUTED SYSTEMS")
    print("─" * 55)
    uses = [
        ("Cassandra/HBase",     "Skip SSTables that don't contain a key"),
        ("Deduplication",       "Have we seen this event/URL/message ID?"),
        ("CDN (Akamai)",        "Cache only URLs seen ≥2 times (once-hit-wonder)"),
        ("Browsers",            "Chrome malware URL check (local filter, low latency)"),
        ("Network routers",     "IP blacklist at line rate (O(1) per packet)"),
        ("Distributed join",    "Filter records before expensive shuffle/join"),
        ("Weak password check", "Is this password in the known-bad list?"),
    ]
    for system, use in uses:
        print(f"  {system:<24} {use}")


if __name__ == "__main__":
    demonstrate_bloom_filters()
