"""
LSM TREE vs B-TREE COMPARISON
================================

Problem Statement:
Database indexes must handle both reads and writes efficiently.
Two dominant tree structures make different trade-offs:
  B-Tree: optimized for reads (good for OLTP, balanced reads/writes).
  LSM Tree: optimized for writes (good for write-heavy workloads).

B-Tree (Balanced Tree):
  Used by: PostgreSQL, MySQL InnoDB, SQLite, Oracle.
  Structure: balanced n-ary tree. Leaf nodes hold data (B+ Tree).
  Read: O(log N), typically 3-4 disk seeks for billion-row table.
  Write: random I/O — find page, update in-place, fsync.
  Write amplification: ~10x (one logical write = many page rewrites).
  Read amplification: low (follow tree path).
  Space amplification: moderate (page fill factor ~70%).

LSM Tree (Log-Structured Merge Tree):
  Used by: LevelDB, RocksDB, Cassandra, HBase, InfluxDB.
  Structure: in-memory buffer (MemTable) + on-disk sorted runs (SSTables).
  Write: append to MemTable → WAL → flush to L0 SSTable → compaction.
  Read: check MemTable → bloom filters → SSTables (newest to oldest).
  Write amplification: lower than B-Tree for sequential writes.
  Read amplification: higher (may read multiple SSTables).
  Space amplification: higher (stale versions until compaction).

MemTable (Memory Table):
  In-memory sorted data structure (usually red-black tree or skip list).
  Absorbs writes in memory for fast response.
  When full: flush to disk as immutable SSTable.

SSTable (Sorted String Table):
  Immutable sorted key-value file on disk.
  Each flush produces a new L0 SSTable.
  Compaction merges/sorts SSTables into larger, sorted L1, L2... files.

Compaction:
  Merges SSTables to remove stale versions and maintain sorted order.
  Size-tiered: merge similar-sized files (RocksDB).
  Level-tiered: maintain sorted levels (LevelDB default).
  Write amplification: each byte written multiple times during compaction.

Read Path in LSM:
  1. Check MemTable (in-memory hash/tree lookup).
  2. Check each SSTable level, newest first.
  3. Bloom filter on each SSTable — skip if key probably not present.
  4. Binary search in SSTable block index.
  5. Read compressed block → decompress → find key.

Write Amplification Comparison:
  B-Tree:   ~10-20x (page splits, COW B-Tree).
  LSM:      ~10-30x (depends on compaction strategy).
  But LSM writes are sequential → SSD/HDD friendly.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple
import bisect
import time
import hashlib
import random


# ─────────────────────────────────────────────
# B-TREE NODE
# ─────────────────────────────────────────────

class BTreeNode:
    def __init__(self, order: int = 4, is_leaf: bool = True):
        self.order    = order
        self.is_leaf  = is_leaf
        self.keys     : List[Any] = []
        self.values   : List[Any] = []    # only in leaf nodes
        self.children : List["BTreeNode"] = []
        self.reads    = 0
        self.writes   = 0


class BTree:
    """
    Simplified B+ Tree (all data in leaves, internal nodes as router).
    Demonstrates: balanced reads, random write I/O pattern.
    """

    def __init__(self, order: int = 4):
        self.order      = order
        self._root      = BTreeNode(order=order, is_leaf=True)
        self.disk_reads = 0
        self.disk_writes= 0

    def insert(self, key: Any, value: Any):
        result = self._insert_recursive(self._root, key, value)
        if result:
            # Root was split
            new_root              = BTreeNode(order=self.order, is_leaf=False)
            new_root.keys         = [result[0]]
            new_root.children     = [self._root, result[1]]
            self._root            = new_root
        self.disk_writes += 1

    def _insert_recursive(self, node: BTreeNode, key: Any,
                           value: Any) -> Optional[Tuple]:
        if node.is_leaf:
            pos = bisect.bisect_left(node.keys, key)
            if pos < len(node.keys) and node.keys[pos] == key:
                node.values[pos] = value   # update in place
            else:
                node.keys.insert(pos, key)
                node.values.insert(pos, value)

            if len(node.keys) >= self.order:
                return self._split_leaf(node)
            return None
        else:
            pos = bisect.bisect_right(node.keys, key)
            result = self._insert_recursive(node.children[pos], key, value)
            if result:
                mid_key, new_child = result
                node.keys.insert(pos, mid_key)
                node.children.insert(pos + 1, new_child)
                if len(node.keys) >= self.order:
                    return self._split_internal(node)
            return None

    def _split_leaf(self, node: BTreeNode) -> Tuple:
        mid = len(node.keys) // 2
        new = BTreeNode(order=self.order, is_leaf=True)
        new.keys   = node.keys[mid:]
        new.values = node.values[mid:]
        node.keys  = node.keys[:mid]
        node.values= node.values[:mid]
        self.disk_writes += 2
        return new.keys[0], new

    def _split_internal(self, node: BTreeNode) -> Tuple:
        mid     = len(node.keys) // 2
        mid_key = node.keys[mid]
        new     = BTreeNode(order=self.order, is_leaf=False)
        new.keys     = node.keys[mid+1:]
        new.children = node.children[mid+1:]
        node.keys    = node.keys[:mid]
        node.children= node.children[:mid+1]
        self.disk_writes += 2
        return mid_key, new

    def search(self, key: Any) -> Optional[Any]:
        node = self._root
        while not node.is_leaf:
            pos = bisect.bisect_right(node.keys, key)
            self.disk_reads += 1
            node = node.children[pos]
        self.disk_reads += 1
        pos = bisect.bisect_left(node.keys, key)
        if pos < len(node.keys) and node.keys[pos] == key:
            return node.values[pos]
        return None

    def range_scan(self, start: Any, end: Any) -> List[Tuple]:
        """Efficient range scan along leaf level."""
        # Find leftmost leaf
        node = self._root
        while not node.is_leaf:
            pos = bisect.bisect_left(node.keys, start)
            self.disk_reads += 1
            node = node.children[min(pos, len(node.children)-1)]

        result = []
        while node:
            for k, v in zip(node.keys, node.values):
                if start <= k <= end:
                    result.append((k, v))
                elif k > end:
                    return result
            self.disk_reads += 1
            # In real B+ tree, leaf nodes have next pointer (simplified here)
            break   # simplified: single leaf scan
        return result

    def height(self) -> int:
        h, node = 0, self._root
        while not node.is_leaf:
            h += 1
            node = node.children[0]
        return h


# ─────────────────────────────────────────────
# LSM TREE COMPONENTS
# ─────────────────────────────────────────────

class MemTable:
    """In-memory sorted structure (simulated with sorted list)."""

    def __init__(self, max_size: int = 100):
        self._data     : Dict[Any, Any] = {}
        self.max_size  = max_size
        self._writes   = 0

    def put(self, key: Any, value: Any):
        self._data[key] = value
        self._writes += 1

    def get(self, key: Any) -> Optional[Any]:
        return self._data.get(key)

    def is_full(self) -> bool:
        return len(self._data) >= self.max_size

    def flush(self) -> "SSTable":
        """Flush to immutable SSTable (sorted)."""
        sorted_data = sorted(self._data.items())
        sst = SSTable(data=sorted_data)
        self._data.clear()
        return sst


@dataclass
class SSTable:
    """Immutable sorted file on disk (simulated in memory)."""
    data     : List[Tuple]           # sorted (key, value) pairs
    level    : int = 0
    created_at: float = field(default_factory=time.time)
    _bloom   : set = field(default_factory=set, repr=False)

    def __post_init__(self):
        self._bloom = {k for k, _ in self.data}

    def might_contain(self, key: Any) -> bool:
        """Bloom filter check (false positives possible)."""
        return key in self._bloom

    def get(self, key: Any) -> Optional[Any]:
        # Binary search
        lo, hi = 0, len(self.data) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            k, v = self.data[mid]
            if k == key:
                return v
            elif k < key:
                lo = mid + 1
            else:
                hi = mid - 1
        return None

    def range_scan(self, start: Any, end: Any) -> List[Tuple]:
        lo = bisect.bisect_left([k for k, _ in self.data], start)
        result = []
        for i in range(lo, len(self.data)):
            k, v = self.data[i]
            if k > end:
                break
            result.append((k, v))
        return result

    @property
    def size(self) -> int:
        return len(self.data)


class LSMTree:
    """
    LSM Tree with MemTable + multi-level SSTables + compaction.
    """

    MAX_L0_FILES = 4   # trigger compaction when L0 has this many files

    def __init__(self, memtable_size: int = 20):
        self._memtable  = MemTable(max_size=memtable_size)
        self._levels    : List[List[SSTable]] = [[], [], []]  # L0, L1, L2
        self.disk_reads = 0
        self.disk_writes= 0
        self.compactions= 0
        self._write_bytes= 0
        self._logical_bytes = 0

    def put(self, key: Any, value: Any):
        self._memtable.put(key, value)
        self._logical_bytes += 1
        if self._memtable.is_full():
            self._flush()

    def _flush(self):
        """Flush MemTable to L0 SSTable."""
        sst = self._memtable.flush()
        sst.level = 0
        self._levels[0].append(sst)
        self._write_bytes += sst.size
        self.disk_writes += 1
        if len(self._levels[0]) >= self.MAX_L0_FILES:
            self._compact(0)

    def _compact(self, level: int):
        """Merge all SSTables at level into next level."""
        if level + 1 >= len(self._levels):
            return
        # Merge all files at this level
        all_data: Dict[Any, Any] = {}
        for sst in self._levels[level]:
            for k, v in sst.data:
                all_data[k] = v   # newer overwrites older (same level treated as sorted)
        if self._levels[level + 1]:
            for sst in self._levels[level + 1]:
                for k, v in sst.data:
                    if k not in all_data:
                        all_data[k] = v

        sorted_data = sorted(all_data.items())
        new_sst = SSTable(data=sorted_data, level=level + 1)
        self._levels[level]     = []
        self._levels[level + 1] = [new_sst]
        self._write_bytes += new_sst.size
        self.disk_writes += 1
        self.compactions += 1

    def get(self, key: Any) -> Optional[Any]:
        # 1. Check MemTable
        v = self._memtable.get(key)
        if v is not None:
            return v

        # 2. Check each level (newest first)
        for level_ssts in self._levels:
            for sst in reversed(level_ssts):
                if sst.might_contain(key):
                    self.disk_reads += 1
                    v = sst.get(key)
                    if v is not None:
                        return v
        return None

    def range_scan(self, start: Any, end: Any) -> List[Tuple]:
        merged: Dict[Any, Any] = {}
        for k, v in self._memtable._data.items():
            if start <= k <= end:
                merged[k] = v
        for level_ssts in self._levels:
            for sst in level_ssts:
                for k, v in sst.range_scan(start, end):
                    if k not in merged:
                        merged[k] = v
        return sorted(merged.items())

    def write_amplification(self) -> float:
        return self._write_bytes / max(self._logical_bytes, 1)

    def stats(self) -> Dict:
        total_sstable_keys = sum(sst.size for level in self._levels for sst in level)
        return {
            "memtable_keys"  : len(self._memtable._data),
            "sstable_keys"   : total_sstable_keys,
            "l0_files"       : len(self._levels[0]),
            "l1_files"       : len(self._levels[1]),
            "l2_files"       : len(self._levels[2]),
            "compactions"    : self.compactions,
            "disk_reads"     : self.disk_reads,
            "disk_writes"    : self.disk_writes,
            "write_amp"      : self.write_amplification(),
        }


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_lsm_vs_btree():
    print("=" * 65)
    print("LSM TREE vs B-TREE COMPARISON")
    print("=" * 65)

    random.seed(42)
    keys   = list(range(200))
    random.shuffle(keys)

    # ── B-Tree ────────────────────────────────────
    print("\n[1] B-TREE — BALANCED READS, RANDOM WRITES")
    print("─" * 55)

    btree = BTree(order=5)
    t0 = time.time()
    for k in keys:
        btree.insert(k, f"val_{k}")
    write_ms = (time.time() - t0) * 1000

    t0 = time.time()
    for k in keys[:50]:
        btree.search(k)
    read_ms = (time.time() - t0) * 1000

    results = btree.range_scan(50, 60)
    print(f"  Inserted {len(keys)} keys")
    print(f"  Tree height: {btree.height()} (log₅(200) ≈ 3)")
    print(f"  Write time: {write_ms:.1f}ms  Read time (50 ops): {read_ms:.2f}ms")
    print(f"  Disk writes: {btree.disk_writes}  Disk reads: {btree.disk_reads}")
    print(f"  Range scan [50,60]: {len(results)} results")
    print(f"  Point lookup: key=42 → {btree.search(42)}")

    # ── LSM Tree ──────────────────────────────────
    print("\n\n[2] LSM TREE — SEQUENTIAL WRITES, MULTI-LEVEL READS")
    print("─" * 55)

    lsm = LSMTree(memtable_size=20)
    t0  = time.time()
    for k in keys:
        lsm.put(k, f"val_{k}")
    lsm._flush()   # flush remaining MemTable
    write_ms_lsm = (time.time() - t0) * 1000

    t0  = time.time()
    for k in keys[:50]:
        lsm.get(k)
    read_ms_lsm = (time.time() - t0) * 1000

    s = lsm.stats()
    print(f"  Inserted {len(keys)} keys")
    print(f"  Write time: {write_ms_lsm:.1f}ms  Read time (50 ops): {read_ms_lsm:.2f}ms")
    print(f"  Compactions: {s['compactions']}")
    print(f"  Levels: L0={s['l0_files']} files L1={s['l1_files']} L2={s['l2_files']}")
    print(f"  Write amplification: {s['write_amp']:.1f}x")
    print(f"  Disk reads: {s['disk_reads']}  Disk writes: {s['disk_writes']}")
    scan = lsm.range_scan(50, 60)
    print(f"  Range scan [50,60]: {len(scan)} results")
    print(f"  Point lookup: key=42 → {lsm.get(42)}")

    # ── Write-Heavy Workload ───────────────────────
    print("\n\n[3] WRITE-HEAVY WORKLOAD COMPARISON (1000 writes)")
    print("─" * 55)

    btree2 = BTree(order=8)
    lsm2   = LSMTree(memtable_size=50)
    N = 1000

    t0 = time.time()
    for i in range(N):
        btree2.insert(i, i * 2)
    bt_ms = (time.time() - t0) * 1000

    t0 = time.time()
    for i in range(N):
        lsm2.put(i, i * 2)
    lsm2._flush()
    lsm_ms = (time.time() - t0) * 1000

    print(f"  {N} writes:")
    print(f"    B-Tree:   {bt_ms:.2f}ms  disk_writes={btree2.disk_writes}")
    print(f"    LSM Tree: {lsm_ms:.2f}ms  disk_writes={lsm2.stats()['disk_writes']}  "
          f"compactions={lsm2.stats()['compactions']}")

    # ── Comparison Table ──────────────────────────
    print("\n\n[4] LSM TREE vs B-TREE COMPARISON")
    print("─" * 55)

    rows = [
        ("Write pattern",    "Random in-place update",  "Sequential append"),
        ("Write amplification","~10-20x",               "~10-30x (but sequential)"),
        ("Read amplification","Low (1-4 seeks)",        "Higher (MemTable+SSTables)"),
        ("Space amplification","Moderate (~1.3x)",      "Higher (~1.5-2x during compaction)"),
        ("Range scans",      "Excellent (B+ tree leaf)", "Good (but multi-file merge)"),
        ("Write latency",    "Spikes on splits",        "Consistent (MemTable absorbs)"),
        ("Read latency",     "Predictable O(logN)",     "Variable (bloom+SSTable scan)"),
        ("Compaction pauses","No compaction",           "Occasional I/O spikes"),
        ("Bloom filters",    "Not needed",              "Essential for read perf"),
        ("Used by",          "PostgreSQL, MySQL, SQLite","RocksDB, Cassandra, LevelDB"),
    ]
    print(f"  {'Property':<24} {'B-Tree':<28} {'LSM Tree'}")
    print(f"  {'─'*80}")
    for prop, bt, lsm in rows:
        print(f"  {prop:<24} {bt:<28} {lsm}")

    # ── When to Use ───────────────────────────────
    print("\n\n[5] WHEN TO CHOOSE")
    print("─" * 55)
    scenarios = [
        ("Read-heavy OLTP",       "B-Tree", "Low read amplification; predictable latency"),
        ("Write-heavy (IoT/logs)", "LSM",   "Sequential writes; absorbs write bursts"),
        ("Time-series data",       "LSM",   "Append-only; compaction handles old data"),
        ("Mixed workload",         "B-Tree","PostgreSQL default; well-understood"),
        ("SSD storage",            "LSM",   "Sequential writes = fewer SSD write cycles"),
        ("HDD storage",            "B-Tree","Random seek latency hidden by tree height"),
    ]
    for workload, choice, reason in scenarios:
        print(f"  {workload:<28} → {choice:<8} {reason}")


if __name__ == "__main__":
    demonstrate_lsm_vs_btree()
