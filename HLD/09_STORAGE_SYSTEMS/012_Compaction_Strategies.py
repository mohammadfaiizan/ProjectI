"""
COMPACTION STRATEGIES
======================

Problem Statement:
LSM-tree based databases (RocksDB, Cassandra, LevelDB) accumulate multiple
SSTables with stale/duplicate keys. Compaction merges them to:
  1. Remove stale versions (reduce space amplification).
  2. Merge sorted files for faster reads (reduce read amplification).
  3. Apply tombstones (delete markers) to remove deleted data.

Trade-offs:
  More aggressive compaction → less space, fewer files, faster reads.
  Less compaction → fewer write I/O, but slower reads and more space.
  Write amplification: each byte written multiple times during compaction.

Compaction Strategies:

  1. Size-Tiered Compaction (Cassandra default for write-heavy):
     Group files by similar size. When N same-size files exist → merge.
     Good for: write-heavy workloads. Bad for: read latency.
     Space amplification: 2x during compaction.

  2. Leveled Compaction (LevelDB/RocksDB default):
     Files organized in levels. L1 = 10MB, L2 = 100MB, L3 = 1GB (10x each).
     Within each level, key ranges don't overlap.
     L0 files overlap; when too many → compact into L1.
     Better read performance. Higher write I/O than size-tiered.

  3. TWCS (Time-Window Compaction Strategy, Cassandra):
     Group SSTables by time window. Only compact within same window.
     Excellent for time-series data (old data never rewritten).
     Old windows: one SSTable per window after compaction.

  4. FIFO Compaction (RocksDB):
     Simple: just drop oldest files when total size exceeds limit.
     Only suitable for cache-like data (losing data is acceptable).

Write Amplification:
  Size-Tiered: 10-20x
  Leveled:     20-30x (RocksDB: 10-30x depending on levels)
  TWCS:        ~1.5x for pure time-series

Tombstones:
  Delete = write tombstone record (not actual deletion).
  Tombstone removed during compaction (after gc_grace_seconds in Cassandra).
  Risk: tombstone flood can slow reads (scanning stale markers).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum
import time
import bisect
import random


# ─────────────────────────────────────────────
# SSTable
# ─────────────────────────────────────────────

TOMBSTONE = "__DELETED__"


@dataclass
class SSTableMeta:
    sst_id     : int
    level      : int
    size_mb    : float
    min_key    : Any
    max_key    : Any
    created_at : float = field(default_factory=time.time)
    data       : Dict[Any, Any] = field(default_factory=dict, repr=False)

    def overlaps_with(self, other: "SSTableMeta") -> bool:
        return not (self.max_key < other.min_key or self.min_key > other.max_key)

    def key_count(self) -> int:
        return len(self.data)

    def tombstone_count(self) -> int:
        return sum(1 for v in self.data.values() if v == TOMBSTONE)


def make_sst(sst_id: int, level: int, data: Dict,
             size_mb: float = None) -> SSTableMeta:
    if not data:
        return SSTableMeta(sst_id=sst_id, level=level, size_mb=0,
                            min_key=None, max_key=None, data={})
    keys = sorted(data.keys())
    sz   = size_mb or len(data) * 0.001  # ~1KB per key simulated
    return SSTableMeta(sst_id=sst_id, level=level, size_mb=sz,
                        min_key=keys[0], max_key=keys[-1], data=dict(data))


# ─────────────────────────────────────────────
# SIZE-TIERED COMPACTION
# ─────────────────────────────────────────────

class SizeTieredCompaction:
    """
    Groups SSTables of similar size into tiers.
    When a tier reaches `min_threshold` files, merge all in that tier.
    """

    def __init__(self, min_threshold: int = 4, bucket_low: float = 0.5,
                 bucket_high: float = 1.5):
        self.min_threshold = min_threshold
        self.bucket_low    = bucket_low
        self.bucket_high   = bucket_high
        self.compactions   = 0
        self._next_id      = 1000

    def _get_bucket(self, size_mb: float) -> float:
        """Returns normalized bucket center for grouping."""
        return round(size_mb / self.bucket_high, 1)

    def maybe_compact(self, sstables: List[SSTableMeta]) -> Tuple[List[SSTableMeta], List[SSTableMeta]]:
        """Returns (new_sstables, removed_sstables). Compacts one tier if eligible."""
        # Group by size tier
        tiers: Dict[float, List[SSTableMeta]] = {}
        for sst in sstables:
            bucket = self._get_bucket(sst.size_mb)
            if bucket not in tiers:
                tiers[bucket] = []
            tiers[bucket].append(sst)

        # Find compactable tier
        for bucket, tier in sorted(tiers.items()):
            if len(tier) >= self.min_threshold:
                # Merge this tier
                merged_data: Dict = {}
                for sst in sorted(tier, key=lambda s: s.created_at):
                    merged_data.update(sst.data)

                # Remove tombstones
                merged_data = {k: v for k, v in merged_data.items()
                               if v != TOMBSTONE}

                new_size = sum(s.size_mb for s in tier) * 0.8  # compression
                self._next_id += 1
                new_sst  = make_sst(self._next_id, 0, merged_data, new_size)
                remaining = [s for s in sstables if s not in tier]
                remaining.append(new_sst)
                self.compactions += 1
                return remaining, tier

        return sstables, []


# ─────────────────────────────────────────────
# LEVELED COMPACTION
# ─────────────────────────────────────────────

class LeveledCompaction:
    """
    Leveled compaction: organize files in non-overlapping levels.
    L0: raw flushes (overlapping OK). L1+: non-overlapping key ranges.
    """

    def __init__(self, level_size_mb: List[float] = None):
        # L0, L1, L2, L3 size targets in MB
        self.level_targets  = level_size_mb or [0, 10, 100, 1000]
        self._levels        : List[List[SSTableMeta]] = [[] for _ in self.level_targets]
        self._next_id       = 2000
        self.compactions    = 0
        self.bytes_written  = 0.0

    def add_l0(self, sst: SSTableMeta):
        self._levels[0].append(sst)
        if len(self._levels[0]) >= 4:
            self._compact_l0_to_l1()

    def _compact_l0_to_l1(self):
        """Merge all L0 files into L1 (sorted, non-overlapping)."""
        merged_data: Dict = {}
        for sst in self._levels[0]:
            merged_data.update(sst.data)

        # Split into non-overlapping L1 files
        sorted_items = sorted(merged_data.items())
        file_size    = max(1, len(sorted_items) // 3)  # 3 files per L1 compaction
        new_files    = []
        for i in range(0, len(sorted_items), file_size):
            chunk   = dict(sorted_items[i:i + file_size])
            self._next_id += 1
            new_sst = make_sst(self._next_id, 1, chunk)
            new_files.append(new_sst)
            self.bytes_written += new_sst.size_mb

        # Merge with existing L1
        existing_data: Dict = {}
        for sst in self._levels[1]:
            existing_data.update(sst.data)
        existing_data.update(merged_data)
        existing_data = {k: v for k, v in existing_data.items() if v != TOMBSTONE}

        sorted_items = sorted(existing_data.items())
        new_l1       = []
        for i in range(0, len(sorted_items), max(1, len(sorted_items) // max(1, len(self._levels[1]) + len(new_files)))):
            chunk = dict(sorted_items[i:i + max(1, file_size)])
            self._next_id += 1
            new_l1.append(make_sst(self._next_id, 1, chunk))

        self._levels[0] = []
        self._levels[1] = new_l1
        self.compactions += 1

    def query(self, key: Any) -> Optional[Any]:
        # L0: check all (overlapping, newest first)
        for sst in reversed(self._levels[0]):
            if sst.data.get(key) is not None:
                v = sst.data[key]
                return None if v == TOMBSTONE else v
        # L1+: sorted, no overlap per level (binary search)
        for level in self._levels[1:]:
            for sst in level:
                if sst.min_key is not None and sst.min_key <= key <= sst.max_key:
                    v = sst.data.get(key)
                    if v is not None:
                        return None if v == TOMBSTONE else v
        return None

    def stats(self) -> Dict:
        return {
            "levels": {
                f"L{i}": {"files": len(lvl),
                           "keys": sum(sst.key_count() for sst in lvl)}
                for i, lvl in enumerate(self._levels)
            },
            "compactions"  : self.compactions,
            "bytes_written": self.bytes_written,
        }


# ─────────────────────────────────────────────
# TWCS (Time-Window Compaction)
# ─────────────────────────────────────────────

class TWCSCompaction:
    """
    Time-Window Compaction: group SSTables by time window.
    Only compacts SSTables within the same window.
    Old windows = single SSTable each (ideal for time-series).
    """

    def __init__(self, window_size_s: float = 3600):
        self.window_size = window_size_s
        self._windows   : Dict[int, List[SSTableMeta]] = {}
        self.compactions = 0
        self._next_id    = 3000

    def _window_key(self, ts: float) -> int:
        return int(ts // self.window_size)

    def add(self, sst: SSTableMeta):
        wk = self._window_key(sst.created_at)
        if wk not in self._windows:
            self._windows[wk] = []
        self._windows[wk].append(sst)

    def compact_old_windows(self, current_time: float, min_windows_old: int = 2) -> int:
        """Compact windows older than N windows. Returns files compacted."""
        current_wk  = self._window_key(current_time)
        compacted   = 0
        for wk in sorted(self._windows.keys()):
            if wk >= current_wk - min_windows_old:
                continue
            files = self._windows[wk]
            if len(files) <= 1:
                continue
            # Merge window into one SSTable
            merged: Dict = {}
            for sst in sorted(files, key=lambda s: s.created_at):
                merged.update(sst.data)
            merged = {k: v for k, v in merged.items() if v != TOMBSTONE}
            self._next_id += 1
            self._windows[wk] = [make_sst(self._next_id, 0, merged)]
            self.compactions += 1
            compacted += len(files) - 1
        return compacted

    def stats(self) -> Dict:
        return {
            "windows"   : len(self._windows),
            "files"     : sum(len(f) for f in self._windows.values()),
            "compactions": self.compactions,
        }


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_compaction():
    print("=" * 65)
    print("COMPACTION STRATEGIES")
    print("=" * 65)

    random.seed(42)

    # ── Size-Tiered Compaction ────────────────────
    print("\n[1] SIZE-TIERED COMPACTION")
    print("─" * 55)

    stcs = SizeTieredCompaction(min_threshold=4)
    sstables = []
    for i in range(8):
        data = {f"key{j}": f"val{j}_v{i}" for j in range(10 + i)}
        sst  = make_sst(i, 0, data, size_mb=2.0 + random.uniform(-0.3, 0.3))
        sstables.append(sst)
        print(f"  Added SST {i}: {sst.key_count()} keys, {sst.size_mb:.2f}MB")

    print(f"\n  Before compaction: {len(sstables)} files")
    sstables, removed = stcs.maybe_compact(sstables)
    print(f"  After compaction: {len(sstables)} files ({len(removed)} merged)")
    print(f"  Compactions done: {stcs.compactions}")
    if sstables:
        newest = sstables[-1]
        print(f"  New merged file: {newest.key_count()} keys, {newest.size_mb:.2f}MB")

    # ── Leveled Compaction ────────────────────────
    print("\n\n[2] LEVELED COMPACTION")
    print("─" * 55)

    lc = LeveledCompaction()
    # Simulate flushing 12 L0 SSTables
    for i in range(12):
        data = {f"key{j:03d}": f"val_{j}_flush{i}" for j in range(i*5, i*5+5)}
        sst  = make_sst(i, 0, data, size_mb=0.5)
        lc.add_l0(sst)
        print(f"  Flush L0 SST {i}: keys={list(data.keys())[:2]}...")

    stats = lc.stats()
    print(f"\n  Final structure:")
    for level, info in stats["levels"].items():
        print(f"    {level}: {info['files']} files, {info['keys']} keys")
    print(f"  Compactions: {stats['compactions']}")
    print(f"  Bytes written: {stats['bytes_written']:.2f}MB")

    # Query test
    v = lc.query("key010")
    print(f"  Query 'key010': {v}")

    # ── TWCS ──────────────────────────────────────
    print("\n\n[3] TIME-WINDOW COMPACTION (time-series)")
    print("─" * 55)

    twcs     = TWCSCompaction(window_size_s=3600)
    base_ts  = 1_700_000_000.0   # fixed base time

    # Create SSTables across 5 hours, multiple flushes per hour
    for hour in range(5):
        for flush in range(3):
            ts = base_ts + hour * 3600 + flush * 600
            data = {f"metric_{hour}_{flush}_{i}": i * 1.5 for i in range(10)}
            sst  = make_sst(hour * 10 + flush, 0, data)
            sst.created_at = ts
            twcs.add(sst)

    stats_before = twcs.stats()
    current_time = base_ts + 5 * 3600   # now = 5 hours later
    compacted    = twcs.compact_old_windows(current_time, min_windows_old=2)

    stats_after  = twcs.stats()
    print(f"  Before compaction: {stats_before['files']} files in {stats_before['windows']} windows")
    print(f"  Compact old windows (>2 ago): {compacted} files merged")
    print(f"  After compaction: {stats_after['files']} files in {stats_after['windows']} windows")
    print(f"  Old windows compacted to 1 file each (ideal for time-series)")

    # ── Tombstone Analysis ────────────────────────
    print("\n\n[4] TOMBSTONES — DELETE MARKERS")
    print("─" * 55)

    data_with_deletes = {
        "user:1": "Alice",
        "user:2": "Bob",
        "user:3": TOMBSTONE,   # deleted
        "user:4": TOMBSTONE,   # deleted
        "user:5": "Carol",
    }
    sst_with_ts = make_sst(999, 0, data_with_deletes)
    print(f"  SSTable: {sst_with_ts.key_count()} keys, "
          f"{sst_with_ts.tombstone_count()} tombstones")
    print(f"  Before compaction GC: user:3 = {sst_with_ts.data.get('user:3')}")

    # After compaction: tombstones removed
    compacted_data = {k: v for k, v in data_with_deletes.items() if v != TOMBSTONE}
    sst_clean = make_sst(1000, 1, compacted_data)
    print(f"  After compaction GC:  {sst_clean.key_count()} keys, "
          f"{sst_clean.tombstone_count()} tombstones")

    # ── Strategy Comparison ───────────────────────
    print("\n\n[5] COMPACTION STRATEGY COMPARISON")
    print("─" * 55)

    rows = [
        ("Size-Tiered",  "Write-heavy",    "Low",    "High", "Cassandra writes"),
        ("Leveled",      "Read-heavy",     "High",   "Low",  "RocksDB default"),
        ("TWCS",         "Time-series",    "Minimal","Low",  "IoT/metrics"),
        ("FIFO",         "Cache/ephemeral","None",   "None", "Hot data eviction"),
    ]
    print(f"  {'Strategy':<16} {'Best for':<16} {'Write amp':<12} {'Space amp':<12} {'Example'}")
    print(f"  {'─'*72}")
    for strategy, best_for, write_amp, space_amp, example in rows:
        print(f"  {strategy:<16} {best_for:<16} {write_amp:<12} {space_amp:<12} {example}")


if __name__ == "__main__":
    demonstrate_compaction()
