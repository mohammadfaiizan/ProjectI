"""
PERFORMANCE PROFILING
======================

Problem Statement:
Metrics tell you something is slow; profiling tells you exactly WHERE.
Profiling identifies the hot paths in CPU, memory, and I/O so engineers
can focus optimization effort where it matters most.

Profiling Types:
  CPU Profiling:
    Sampling:    Interrupt at N Hz, record call stack. Low overhead (~1%).
                 Tools: py-spy, Linux perf, async-profiler (Java).
    Instrumented: Hooks at every function entry/exit. High overhead (10-30×).
                 Tools: cProfile (Python), Valgrind callgrind.

  Memory Profiling:
    Heap snapshot:  Capture all live objects at a point in time.
    Allocation trace: Record every malloc/free. Very high overhead.
    Tools: memory_profiler (Python), Heapster (Go), MAT (Java).

  Flame Graphs:
    Visualization of CPU profiling data.
    X axis: time spent. Y axis: call stack depth.
    Wide bars = hot paths to optimize.
    Created by Brendan Gregg (Netflix).

  Continuous Profiling:
    Low-overhead sampling profiler running in production.
    Aggregates profiles over time; stores compressed.
    Tools: Pyroscope (Grafana), Parca, Google Cloud Profiler.
    Overhead: <1% CPU, <100MB RAM.

  Database Query Profiling:
    EXPLAIN ANALYZE in PostgreSQL.
    Identify: seq scans on large tables, nested loops, bad estimates.
    Fix: add index, rewrite query, update statistics.

  I/O Profiling:
    strace (Linux): trace syscalls.
    iotop: per-process disk I/O.
    tcpdump/Wireshark: network traffic.

Profiling Workflow:
  1. Reproduce the slow scenario with production-like load.
  2. Collect CPU profile for 30-60 seconds.
  3. Identify top 3 functions by self time.
  4. Optimize hot path. Measure improvement.
  5. Repeat until SLO is met.

Amdahl's Law Reminder:
  Optimize only what matters. If a function is 1% of total time,
  even eliminating it entirely saves only 1%.
"""

from __future__ import annotations

import time
import cProfile
import pstats
import io
import functools
import tracemalloc
import random
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────
# PROFILING DECORATORS
# ─────────────────────────────────────────────

def timeit(func: Callable) -> Callable:
    """Decorator: measure wall-clock time of a function call."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start  = time.perf_counter()
        result = func(*args, **kwargs)
        end    = time.perf_counter()
        wrapper._last_ms = (end - start) * 1000
        return result
    wrapper._last_ms = 0.0
    return wrapper


def cpu_profile(func: Callable) -> Callable:
    """Decorator: cProfile the function; return (result, stats_string)."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        pr  = cProfile.Profile()
        pr.enable()
        result = func(*args, **kwargs)
        pr.disable()
        buf = io.StringIO()
        ps  = pstats.Stats(pr, stream=buf).sort_stats("cumulative")
        ps.print_stats(10)  # top 10 functions
        wrapper._profile_output = buf.getvalue()
        return result
    wrapper._profile_output = ""
    return wrapper


# ─────────────────────────────────────────────
# CALL TREE NODE (flame graph data)
# ─────────────────────────────────────────────

@dataclass
class CallNode:
    name:      str
    self_time_us:  float     = 0.0    # time spent in this function (not children)
    total_time_us: float     = 0.0    # self + children
    calls:         int       = 0
    children:      List["CallNode"] = field(default_factory=list)

    @property
    def avg_time_us(self) -> float:
        return self.total_time_us / max(self.calls, 1)


class FakeProfiler:
    """
    Simulates a sampling profiler output.
    Returns a call tree with simulated timings.
    """

    def profile(self, func_name: str) -> CallNode:
        """Build a fake but realistic call tree for demonstration."""
        random.seed(42)

        # Simulate a typical web request call tree
        root = CallNode("handle_request", self_time_us=50, total_time_us=45000, calls=1000)
        root.children = [
            CallNode("authenticate", 200, 800, 1000, children=[
                CallNode("jwt_verify",    150, 400, 1000),
                CallNode("db_user_lookup", 50, 400, 1000, children=[
                    CallNode("execute_query", 350, 395, 1000),
                ]),
            ]),
            CallNode("validate_input", 300, 500, 1000),
            CallNode("business_logic", 100, 32000, 1000, children=[
                CallNode("fetch_products", 200, 20000, 1000, children=[
                    CallNode("db_query_products", 100, 18000, 1000, children=[
                        CallNode("execute_query_slow",15000, 17800, 1000),  # HOT PATH
                    ]),
                    CallNode("cache_miss",            100,  1800, 900),
                ]),
                CallNode("apply_discounts",   8000, 10000, 1000),          # HOT PATH
                CallNode("serialize_response",1000,  2000, 1000),
            ]),
            CallNode("log_request", 500, 700, 1000),
        ]
        return root

    def flame_graph_rows(self, node: CallNode, depth: int = 0
                         ) -> List[Tuple[int, str, float]]:
        """Returns (depth, name, pct_of_root) for ASCII flame graph."""
        rows = []
        root_total = node.total_time_us if depth == 0 else None

        def traverse(n: CallNode, d: int, root_t: float):
            pct = n.total_time_us / root_t * 100 if root_t > 0 else 0
            rows.append((d, n.name, pct, n.total_time_us, n.self_time_us))
            for child in sorted(n.children, key=lambda c: -c.total_time_us):
                traverse(child, d + 1, root_t)

        traverse(node, 0, node.total_time_us)
        return rows


# ─────────────────────────────────────────────
# MEMORY PROFILER SIMULATION
# ─────────────────────────────────────────────

@dataclass
class AllocationRecord:
    filename:  str
    lineno:    int
    size_kb:   float
    count:     int


class MemoryProfiler:
    """
    Wraps tracemalloc to record top allocations.
    """

    def __init__(self):
        self._snapshot_before = None

    def start(self):
        tracemalloc.start()
        self._snapshot_before = tracemalloc.take_snapshot()

    def stop_and_report(self, top_n: int = 10) -> List[AllocationRecord]:
        snapshot = tracemalloc.take_snapshot()
        tracemalloc.stop()

        if self._snapshot_before:
            stats = snapshot.compare_to(self._snapshot_before, "lineno")
        else:
            stats = snapshot.statistics("lineno")

        records = []
        for stat in stats[:top_n]:
            frame = stat.traceback[0] if stat.traceback else None
            records.append(AllocationRecord(
                filename = frame.filename.split("/")[-1] if frame else "unknown",
                lineno   = frame.lineno if frame else 0,
                size_kb  = stat.size / 1024,
                count    = stat.count,
            ))
        return records


# ─────────────────────────────────────────────
# HOT PATH DEMONSTRATIONS
# ─────────────────────────────────────────────

def slow_sum(n: int) -> int:
    """Intentionally slow: redundant computation."""
    result = 0
    for i in range(n):
        result += int(math.sqrt(i) ** 2)  # sqrt then square → just use i
    return result


def fast_sum(n: int) -> int:
    """Optimized version."""
    return n * (n - 1) // 2


def build_dict_slow(items: List[str]) -> Dict[str, int]:
    """String concatenation in loop: O(n²) due to immutable strings."""
    result = {}
    key = ""
    for i, item in enumerate(items):
        key = key + item   # creates new string every time
        result[item] = i
    return result


def build_dict_fast(items: List[str]) -> Dict[str, int]:
    """Direct dict comprehension."""
    return {item: i for i, item in enumerate(items)}


def n_plus_one_slow(records: List[Dict]) -> List[str]:
    """
    N+1 query pattern: for each record, fetch related data separately.
    Simulated: O(n) lookups where one batch query would suffice.
    """
    db = {i: f"user_{i}" for i in range(10000)}  # simulated DB
    results = []
    for rec in records:
        user = db.get(rec["user_id"], "unknown")  # N individual lookups
        results.append(f"{rec['name']} ({user})")
    return results


def n_plus_one_fast(records: List[Dict]) -> List[str]:
    """Batch lookup: 1 query instead of N."""
    db       = {i: f"user_{i}" for i in range(10000)}
    user_ids = {rec["user_id"] for rec in records}
    users    = {uid: db[uid] for uid in user_ids if uid in db}  # 1 batch
    return [f"{rec['name']} ({users.get(rec['user_id'], 'unknown')})"
            for rec in records]


# ─────────────────────────────────────────────
# QUERY EXPLAIN ANALYZER (simulated)
# ─────────────────────────────────────────────

@dataclass
class QueryPlan:
    sql:          str
    plan_type:    str    # SeqScan, IndexScan, HashJoin, etc.
    cost:         float  # planner estimate
    actual_ms:    float
    rows:         int
    issue:        Optional[str] = None
    fix:          Optional[str] = None


class QueryAnalyzer:
    """Simulates EXPLAIN ANALYZE output for common slow query patterns."""

    def analyze(self, sql: str) -> QueryPlan:
        sql_lower = sql.lower()

        # No WHERE clause → seq scan
        if "where" not in sql_lower:
            return QueryPlan(sql, "SeqScan", 50000, 2500.0, 1_000_000,
                             "Full table scan — no WHERE clause",
                             "Add WHERE clause or LIMIT")

        # LIKE '%pattern' → can't use index
        if re.search(r"like\s+['\"]%", sql_lower) if (
            __import__("re").search(r"like\s+['\"]%", sql_lower)) else False:
            return QueryPlan(sql, "SeqScan", 40000, 1200.0, 500_000,
                             "Leading wildcard LIKE prevents index use",
                             "Use full-text search (tsvector) or prefix search")

        # Unindexed column pattern
        if "unindexed_col" in sql_lower:
            return QueryPlan(sql, "SeqScan", 30000, 800.0, 300_000,
                             "Filter on unindexed column",
                             "CREATE INDEX ON table(unindexed_col)")

        # SELECT * with join
        if "select *" in sql_lower and "join" in sql_lower:
            return QueryPlan(sql, "HashJoin", 500, 45.0, 1000,
                             "SELECT * fetches unused columns",
                             "Select only needed columns")

        # Good query
        return QueryPlan(sql, "IndexScan", 10, 2.0, 1,
                         None, None)

    def format(self, plan: QueryPlan) -> str:
        lines = [
            f"  SQL: {plan.sql[:60]}...",
            f"  Plan: {plan.plan_type}  cost={plan.cost:.0f}  "
            f"actual={plan.actual_ms:.0f}ms  rows={plan.rows:,}",
        ]
        if plan.issue:
            lines.append(f"  Issue: {plan.issue}")
            lines.append(f"  Fix:   {plan.fix}")
        return "\n".join(lines)


# Add re import for query analyzer
import re


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_profiling():
    print("=" * 65)
    print("PERFORMANCE PROFILING")
    print("=" * 65)

    # ── CPU Hotspot: Slow vs Fast ─────────────
    print("\n[1] CPU HOT PATH OPTIMIZATION")
    print("─" * 55)

    N = 50000

    t0 = time.perf_counter()
    slow_sum(N)
    slow_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    fast_sum(N)
    fast_ms = (time.perf_counter() - t0) * 1000

    speedup = slow_ms / max(fast_ms, 0.001)
    print(f"  slow_sum({N}):  {slow_ms:.2f}ms  [sqrt + square each element]")
    print(f"  fast_sum({N}):  {fast_ms:.4f}ms  [n*(n-1)//2 formula]")
    print(f"  Speedup: {speedup:.0f}×")

    # ── String concatenation ──────────────────
    print("\n[2] STRING CONCATENATION ANTI-PATTERN")
    print("─" * 55)

    items = [f"item_{i}" for i in range(2000)]

    t0 = time.perf_counter()
    build_dict_slow(items)
    slow2_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    build_dict_fast(items)
    fast2_ms = (time.perf_counter() - t0) * 1000

    speedup2 = slow2_ms / max(fast2_ms, 0.001)
    print(f"  slow (str concat in loop): {slow2_ms:.2f}ms")
    print(f"  fast (dict comprehension): {fast2_ms:.4f}ms")
    print(f"  Speedup: {speedup2:.0f}×")
    print("  Lesson: string concatenation in loop = O(n²)")

    # ── N+1 Query Pattern ─────────────────────
    print("\n[3] N+1 QUERY PATTERN")
    print("─" * 55)

    records = [{"name": f"item_{i}", "user_id": i % 100} for i in range(1000)]

    t0 = time.perf_counter()
    n_plus_one_slow(records)
    nplus1_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    n_plus_one_fast(records)
    batch_ms = (time.perf_counter() - t0) * 1000

    speedup3 = nplus1_ms / max(batch_ms, 0.001)
    print(f"  N+1 pattern (1000 records): {nplus1_ms:.2f}ms  [1000 individual lookups]")
    print(f"  Batch fetch:                {batch_ms:.2f}ms   [1 batch lookup]")
    print(f"  Speedup: {speedup3:.0f}×")
    print("  In prod: 1000 DB round trips vs 1 → orders of magnitude worse")

    # ── Flame Graph Simulation ────────────────
    print("\n[4] FLAME GRAPH (ASCII representation)")
    print("─" * 55)

    profiler = FakeProfiler()
    root     = profiler.profile("handle_request")
    rows     = profiler.flame_graph_rows(root)

    print(f"  {'Function':<35} {'% Time':>8}  {'Total (ms)':>12}  {'Self (ms)':>10}")
    print("  " + "─" * 70)
    for depth, name, pct, total_us, self_us in rows[:12]:
        indent = "  " * depth
        total_ms = total_us / 1000
        self_ms  = self_us  / 1000
        flag     = " ← HOT" if pct > 30 else ""
        print(f"  {indent}{name:<{35-depth*2}} {pct:>7.1f}%  "
              f"{total_ms:>10.1f}ms  {self_ms:>8.1f}ms{flag}")

    # ── Memory Profiling ──────────────────────
    print("\n[5] MEMORY PROFILING (tracemalloc)")
    print("─" * 55)

    mp = MemoryProfiler()
    mp.start()

    # Allocate some memory
    big_list    = [random.random() for _ in range(100_000)]
    big_dict    = {i: str(i) * 10 for i in range(10_000)}
    nested      = [[i * j for j in range(100)] for i in range(100)]

    records_mem = mp.stop_and_report(top_n=5)
    print(f"  {'File':<30} {'Line':>6} {'KB':>8}  {'Count':>6}")
    print("  " + "─" * 55)
    for r in records_mem[:5]:
        print(f"  {r.filename:<30} {r.lineno:>6} {r.size_kb:>7.1f}  {r.count:>6}")

    # ── Query Analysis ────────────────────────
    print("\n[6] SLOW QUERY ANALYSIS (EXPLAIN ANALYZE)")
    print("─" * 55)

    analyzer = QueryAnalyzer()
    queries  = [
        "SELECT * FROM orders",
        "SELECT id, amount FROM orders WHERE unindexed_col = 'x'",
        "SELECT * FROM users JOIN orders ON users.id = orders.user_id",
        "SELECT id FROM orders WHERE status = 'pending' AND created_at > NOW() - '1d'",
    ]
    for sql in queries:
        plan = analyzer.analyze(sql)
        print(f"\n  {analyzer.format(plan)}")

    # ── cProfile example ─────────────────────
    print("\n[7] cProfile SUMMARY (fast_sum profiled)")
    print("─" * 55)

    pr  = cProfile.Profile()
    pr.enable()
    for _ in range(1000):
        fast_sum(1000)
    pr.disable()

    buf = io.StringIO()
    ps  = pstats.Stats(pr, stream=buf).sort_stats("tottime")
    ps.print_stats(5)
    output = buf.getvalue()
    # Print just the relevant lines
    for line in output.split("\n")[4:12]:
        if line.strip():
            print(f"  {line}")

    # ── Optimization Principles ───────────────
    print("\n[8] PROFILING PRINCIPLES")
    print("─" * 55)

    principles = [
        "Measure first, optimize second — never guess",
        "Amdahl: optimizing 1% of total time → max 1% improvement",
        "Top 3 hot paths by self-time = where to focus",
        "Flame graph width = time; look for wide unexpectedly wide bars",
        "Memory: look for retained objects growing over time (leak)",
        "N+1 queries: one-row-at-a-time DB access → batch or JOIN",
        "String concat in loop: use ''.join(list) or list comprehension",
        "Cache hot reads: if same DB row read 1000×/sec, cache it",
        "Profile with production traffic, not microbenchmarks",
        "Continuous profiling in prod: Pyroscope/Parca < 1% overhead",
    ]
    for i, p in enumerate(principles, 1):
        print(f"  {i:>2}. {p}")


if __name__ == "__main__":
    demonstrate_profiling()
