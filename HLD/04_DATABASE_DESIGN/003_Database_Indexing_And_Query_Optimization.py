"""
DATABASE INDEXING AND QUERY OPTIMIZATION
==========================================

Problem Statement:
A table scan on 100M rows takes minutes. An index lookup takes milliseconds.
Indexing is the single most impactful performance optimization in database
design. But indexes have costs (write overhead, storage) so they must be
chosen carefully.

Index Types:
  B-Tree     : Balanced tree — default. Range queries, equality, ORDER BY
  Hash       : Exact equality only. O(1) lookup. Not for ranges.
  GiST/GIN   : Generalized Search Tree — arrays, JSON, full-text
  BRIN       : Block Range Index — huge tables with natural ordering (time)
  Composite  : Multiple columns — order matters (left-prefix rule)
  Partial    : Index on subset of rows (WHERE condition)
  Covering   : Index contains all queried columns — avoids table lookup

Index Selection Rules:
  1. Index columns used in WHERE, JOIN ON, ORDER BY, GROUP BY
  2. Composite index column order matters — most selective first
  3. Don't index low-cardinality columns (boolean, status with 3 values)
  4. Covering index: include all SELECT columns to avoid heap fetch
  5. Partial index: index only active rows (WHERE deleted_at IS NULL)

Query Optimizer:
  EXPLAIN ANALYZE shows the execution plan
  Seq Scan: full table scan — usually bad
  Index Scan: uses index — usually good
  Bitmap Scan: uses index then fetches rows in page order — good for batches
  Nested Loop Join: good for small tables
  Hash Join: good for large unsorted tables
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import time
import random
import bisect
import hashlib


class IndexType(Enum):
    BTREE    = "btree"
    HASH     = "hash"
    COMPOSITE= "composite"
    PARTIAL  = "partial"
    COVERING = "covering"
    BRIN     = "brin"


class ScanType(Enum):
    SEQ_SCAN    = "Seq Scan"
    INDEX_SCAN  = "Index Scan"
    BITMAP_SCAN = "Bitmap Index Scan"
    INDEX_ONLY  = "Index Only Scan"


@dataclass
class QueryPlan:
    scan_type    : ScanType
    rows_scanned : int
    rows_returned: int
    latency_ms   : float
    index_used   : Optional[str]
    cost_estimate: float

    def explain(self, query: str):
        print(f"\n  EXPLAIN ANALYZE: {query[:60]}")
        print(f"    Plan     : {self.scan_type.value}")
        print(f"    Index    : {self.index_used or 'none (full table scan)'}")
        print(f"    Scanned  : {self.rows_scanned:,} rows")
        print(f"    Returned : {self.rows_returned:,} rows")
        print(f"    Latency  : {self.latency_ms:.2f} ms")
        print(f"    Selectivity: {self.rows_returned/max(1,self.rows_scanned):.2%}")


# ─────────────────────────────────────────────
# B-TREE INDEX SIMULATION
# ─────────────────────────────────────────────

class BTreeIndex:
    """
    Simplified B-Tree index: sorted list of (key, row_id) pairs.
    Supports equality, range, and ordered queries in O(log N).
    """

    def __init__(self, name: str, column: str):
        self.name    = name
        self.column  = column
        self._keys   : List[Any]  = []   # sorted
        self._rowids : List[int]  = []   # parallel to _keys
        self.lookups = 0

    def build(self, rows: List[Dict], col: str):
        pairs = sorted((row[col], i) for i, row in enumerate(rows) if col in row)
        self._keys   = [p[0] for p in pairs]
        self._rowids = [p[1] for p in pairs]

    def lookup_eq(self, value: Any) -> List[int]:
        """Equality lookup — O(log N)."""
        self.lookups += 1
        lo = bisect.bisect_left(self._keys, value)
        hi = bisect.bisect_right(self._keys, value)
        return self._rowids[lo:hi]

    def lookup_range(self, lo: Any, hi: Any) -> List[int]:
        """Range lookup — O(log N + result_size)."""
        self.lookups += 1
        lo_idx = bisect.bisect_left(self._keys, lo)
        hi_idx = bisect.bisect_right(self._keys, hi)
        return self._rowids[lo_idx:hi_idx]

    def size_estimate_mb(self, row_count: int) -> float:
        return row_count * 8 / 1e6   # ~8 bytes per entry


# ─────────────────────────────────────────────
# HASH INDEX SIMULATION
# ─────────────────────────────────────────────

class HashIndex:
    """
    Hash index: O(1) exact lookups. No range queries.
    """

    def __init__(self, name: str, column: str):
        self.name    = name
        self.column  = column
        self._buckets: Dict[int, List[int]] = {}

    def build(self, rows: List[Dict], col: str):
        for i, row in enumerate(rows):
            if col not in row:
                continue
            h = hash(row[col]) % 1024
            self._buckets.setdefault(h, []).append(i)

    def lookup(self, value: Any) -> List[int]:
        h    = hash(value) % 1024
        candidates = self._buckets.get(h, [])
        return candidates   # caller verifies equality (collision handling)


# ─────────────────────────────────────────────
# COMPOSITE INDEX
# ─────────────────────────────────────────────

class CompositeIndex:
    """
    Multi-column index: (col_a, col_b).
    Left-prefix rule: usable for (col_a), (col_a, col_b) but NOT (col_b) alone.
    """

    def __init__(self, name: str, columns: List[str]):
        self.name    = name
        self.columns = columns
        self._index  : Dict[Tuple, List[int]] = {}

    def build(self, rows: List[Dict]):
        for i, row in enumerate(rows):
            key = tuple(row.get(c) for c in self.columns)
            self._index.setdefault(key, []).append(i)

    def lookup(self, **kwargs) -> List[int]:
        key = tuple(kwargs.get(c) for c in self.columns)
        return self._index.get(key, [])


# ─────────────────────────────────────────────
# QUERY EXECUTOR
# ─────────────────────────────────────────────

class QueryExecutor:
    """Simulates query execution with and without indexes."""

    def __init__(self, rows: List[Dict]):
        self.rows    = rows
        self._indexes: Dict[str, Any] = {}
        self.total_rows = len(rows)

    def add_index(self, name: str, index):
        self._indexes[name] = index

    def seq_scan(self, where_col: str, where_val: Any,
                 select_cols: List[str] = None) -> Tuple[List[Dict], QueryPlan]:
        start = time.perf_counter()
        result = [r for r in self.rows if r.get(where_col) == where_val]
        latency = (time.perf_counter() - start) * 1000 + self.total_rows * 0.001
        if select_cols:
            result = [{c: r.get(c) for c in select_cols} for r in result]
        plan = QueryPlan(ScanType.SEQ_SCAN, self.total_rows, len(result),
                          round(latency, 2), None,
                          cost_estimate=self.total_rows * 0.01)
        return result, plan

    def index_scan(self, index_name: str, index, value: Any,
                   select_cols: List[str] = None) -> Tuple[List[Dict], QueryPlan]:
        start = time.perf_counter()
        if isinstance(index, BTreeIndex):
            rowids = index.lookup_eq(value)
        elif isinstance(index, HashIndex):
            rowids = index.lookup(value)
        else:
            rowids = []
        result = [self.rows[i] for i in rowids if i < len(self.rows)]
        latency = (time.perf_counter() - start) * 1000 + len(rowids) * 0.01 + 2.0
        if select_cols:
            result = [{c: r.get(c) for c in select_cols} for r in result]
        plan = QueryPlan(ScanType.INDEX_SCAN, len(rowids), len(result),
                          round(latency, 2), index_name,
                          cost_estimate=8.5 + len(rowids) * 0.01)
        return result, plan

    def range_scan(self, index: BTreeIndex, lo: Any, hi: Any) -> Tuple[List[Dict], QueryPlan]:
        start  = time.perf_counter()
        rowids = index.lookup_range(lo, hi)
        result = [self.rows[i] for i in rowids if i < len(self.rows)]
        latency = (time.perf_counter() - start) * 1000 + len(rowids) * 0.01 + 3.0
        plan   = QueryPlan(ScanType.INDEX_SCAN, len(rowids), len(result),
                            round(latency, 2), index.name,
                            cost_estimate=10 + len(rowids) * 0.01)
        return result, plan


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_indexing():
    print("=" * 65)
    print("DATABASE INDEXING AND QUERY OPTIMIZATION")
    print("=" * 65)

    # Generate test data (100K user rows)
    random.seed(42)
    N = 100_000
    statuses = ["active", "inactive", "pending"]
    countries = ["US", "UK", "DE", "FR", "IN", "JP", "BR", "AU"]
    rows = [
        {
            "user_id"   : i,
            "email"     : f"user{i}@example.com",
            "name"      : f"User {i}",
            "country"   : random.choice(countries),
            "status"    : random.choice(statuses),
            "age"       : random.randint(18, 80),
            "created_at": 1700000000 + random.randint(0, 31536000),
        }
        for i in range(N)
    ]
    print(f"\n  Dataset: {N:,} user rows")

    # Build indexes
    idx_email    = BTreeIndex("idx_users_email",   "email")
    idx_email.build(rows, "email")
    idx_age      = BTreeIndex("idx_users_age",     "age")
    idx_age.build(rows, "age")
    idx_hash_status = HashIndex("idx_hash_status", "status")
    idx_hash_status.build(rows, "status")
    idx_comp     = CompositeIndex("idx_country_status", ["country", "status"])
    idx_comp.build(rows)

    executor = QueryExecutor(rows)
    executor.add_index("idx_users_email",   idx_email)
    executor.add_index("idx_users_age",     idx_age)
    executor.add_index("idx_hash_status",   idx_hash_status)

    # ── Seq Scan vs Index Scan ────────────────
    print("\n[1] SEQ SCAN vs INDEX SCAN (equality)")
    print("─" * 55)
    target_email = rows[50000]["email"]

    _, plan_seq = executor.seq_scan("email", target_email, ["user_id", "name", "email"])
    plan_seq.explain(f"SELECT user_id, name FROM users WHERE email = '{target_email}'")

    _, plan_idx = executor.index_scan("idx_users_email", idx_email, target_email,
                                       ["user_id", "name", "email"])
    plan_idx.explain(f"SELECT user_id, name FROM users WHERE email = '{target_email}'")
    print(f"\n  Speedup: {plan_seq.latency_ms / plan_idx.latency_ms:.0f}x faster with index")

    # ── Range Query ───────────────────────────
    print("\n\n[2] RANGE QUERY (B-Tree index)")
    print("─" * 55)
    _, plan_range = executor.range_scan(idx_age, 25, 35)
    plan_range.explain("SELECT * FROM users WHERE age BETWEEN 25 AND 35")

    # ── Composite Index ───────────────────────
    print("\n\n[3] COMPOSITE INDEX — LEFT PREFIX RULE")
    print("─" * 55)
    print("  Index: (country, status)")
    print("  ✅ Usable: WHERE country='US'")
    print("  ✅ Usable: WHERE country='US' AND status='active'")
    print("  ❌ NOT usable: WHERE status='active' (skips country — left column)")

    us_active = idx_comp.lookup(country="US", status="active")
    print(f"\n  WHERE country='US' AND status='active' → {len(us_active)} rows via composite index")

    # ── Partial Index ─────────────────────────
    print("\n\n[4] PARTIAL INDEX (index only active users)")
    print("─" * 55)
    active_rows = [(i, r) for i, r in enumerate(rows) if r["status"] == "active"]
    partial_keys  = sorted((r["age"], i) for i, r in active_rows)
    active_count  = len(active_rows)
    total_count   = len(rows)
    print(f"  Full index on age: {total_count:,} entries")
    print(f"  Partial index on age WHERE status='active': {active_count:,} entries")
    print(f"  Space saving: {(1 - active_count/total_count):.0%} smaller")
    print("  CREATE INDEX idx_active_age ON users(age) WHERE status = 'active';")

    # ── Index Cost Analysis ───────────────────
    print("\n\n[5] INDEX COST ANALYSIS")
    print("─" * 55)
    print("  Indexes speed up reads but slow writes:")
    print(f"  {'Operation':<20} {'No Index':<15} {'With 5 Indexes'}")
    print(f"  {'─'*50}")
    ops = [
        ("SELECT by email", "Full scan ~100ms", "Index scan ~0.1ms"),
        ("INSERT row",      "+0ms overhead",     "+5ms (update 5 indexes)"),
        ("UPDATE row",      "+0ms overhead",     "+5ms (update affected indexes)"),
        ("DELETE row",      "+0ms overhead",     "+5ms (update 5 indexes)"),
        ("Storage",         "Table only",        "Table + index files (20-40% extra)"),
    ]
    for op, no_idx, with_idx in ops:
        print(f"  {op:<20} {no_idx:<15} {with_idx}")

    # ── Index Design Rules ────────────────────
    print("\n\n[6] INDEX DESIGN RULES")
    print("─" * 55)
    rules = [
        ("Do index",   "Primary key (always)",              "Unique, clustered"),
        ("Do index",   "Foreign keys (JOIN columns)",        "Avoid full scan on join"),
        ("Do index",   "WHERE filter columns",              "Equality and ranges"),
        ("Do index",   "ORDER BY columns",                  "Avoid sort operation"),
        ("Do index",   "Covering index (include SELECT cols)","Avoid heap fetch"),
        ("Don't index","Boolean / low-cardinality columns", "100K active=50K → no help"),
        ("Don't index","Rarely queried columns",            "Write overhead not worth it"),
        ("Don't index","Very small tables (<1K rows)",       "Seq scan is faster"),
    ]
    for action, target, reason in rules:
        icon = "✅" if action == "Do index" else "❌"
        print(f"  {icon} {target:<40} {reason}")

    print(f"\n  Index size estimate for {N:,} rows:")
    for idx in [idx_email, idx_age]:
        print(f"    {idx.name}: ~{idx.size_estimate_mb(N):.1f} MB")


if __name__ == "__main__":
    demonstrate_indexing()
