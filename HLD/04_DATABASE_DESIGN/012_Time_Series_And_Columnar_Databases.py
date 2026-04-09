"""
TIME-SERIES AND COLUMNAR DATABASES
=====================================

Problem Statement:
Metric data (CPU, QPS, temperature), log data, and financial tick data are
fundamentally time-ordered streams. Standard row-store DBs are inefficient
for this use case. Time-series databases and columnar stores provide 10-100x
better compression and query performance for this data.

Time-Series Database (TSDB):
  Key characteristics:
  - Ordered by timestamp (append-only, rarely updated)
  - High write volume (millions of points/sec)
  - Range queries (last 5 min, last 24 hours)
  - Downsampling (1s → 1min → 1hr → 1day aggregates)
  - Automatic TTL/retention (keep last 90 days)

Examples: InfluxDB, TimescaleDB (PostgreSQL ext), Prometheus, VictoriaMetrics

Columnar Store (OLAP):
  Row store (OLTP):  [(row1_col1, row1_col2...), (row2_col1, row2_col2...)]
  Column store (OLAP): [(col1_row1, col1_row2...), (col2_row1, col2_row2...)]

  Column storage advantage:
  - Only read columns you need (SELECT avg(cpu) skips all other columns)
  - Same type data → much better compression (RLE, delta encoding)
  - Vectorized SIMD operations → faster aggregations

Examples: Redshift, BigQuery, Snowflake, ClickHouse, Parquet, Apache Arrow
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import time
import random
import struct
from collections import defaultdict


class RetentionPolicy(Enum):
    RAW_1H   = "raw_1h"      # 1-second resolution for 1 hour
    MINUTE_7D= "minute_7d"   # 1-minute resolution for 7 days
    HOUR_30D = "hour_30d"    # 1-hour resolution for 30 days
    DAY_2Y   = "day_2y"      # 1-day resolution for 2 years


@dataclass
class DataPoint:
    metric     : str
    tags       : Dict[str, str]
    value      : float
    timestamp  : float = field(default_factory=time.time)

    @property
    def tag_key(self) -> str:
        return ",".join(f"{k}={v}" for k, v in sorted(self.tags.items()))


@dataclass
class AggregatedPoint:
    metric    : str
    tags      : Dict[str, str]
    min_val   : float
    max_val   : float
    avg_val   : float
    sum_val   : float
    count     : int
    bucket_ts : float   # start of time bucket


# ─────────────────────────────────────────────
# TIME-SERIES STORE
# ─────────────────────────────────────────────

class TimeSeriesStore:
    """
    Simplified TSDB simulation.
    Supports: ingestion, range queries, downsampling, retention.
    """

    def __init__(self, retention_hours: int = 24):
        self.retention_s = retention_hours * 3600
        # Storage: metric+tags → sorted list of (timestamp, value)
        self._data       : Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        self.write_count = 0
        self.read_count  = 0

    def _series_key(self, metric: str, tags: Dict[str, str]) -> str:
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{metric}{{{tag_str}}}"

    def write(self, point: DataPoint):
        self.write_count += 1
        key = self._series_key(point.metric, point.tags)
        self._data[key].append((point.timestamp, point.value))

    def query_range(self, metric: str, tags: Dict[str, str],
                     start_ts: float, end_ts: float) -> List[Tuple[float, float]]:
        self.read_count += 1
        key    = self._series_key(metric, tags)
        series = self._data.get(key, [])
        return [(ts, val) for ts, val in series if start_ts <= ts <= end_ts]

    def downsample(self, metric: str, tags: Dict[str, str],
                    bucket_s: float, start_ts: float, end_ts: float) -> List[AggregatedPoint]:
        """Group data into time buckets and compute min/max/avg."""
        self.read_count += 1
        points = self.query_range(metric, tags, start_ts, end_ts)
        if not points:
            return []

        buckets: Dict[float, List[float]] = defaultdict(list)
        for ts, val in points:
            bucket = (ts // bucket_s) * bucket_s
            buckets[bucket].append(val)

        result = []
        for bucket_ts in sorted(buckets):
            vals = buckets[bucket_ts]
            result.append(AggregatedPoint(
                metric=metric, tags=tags,
                min_val=min(vals), max_val=max(vals),
                avg_val=sum(vals)/len(vals), sum_val=sum(vals),
                count=len(vals), bucket_ts=bucket_ts
            ))
        return result

    def apply_retention(self):
        """Evict data older than retention period."""
        cutoff = time.time() - self.retention_s
        evicted = 0
        for key in list(self._data.keys()):
            before = len(self._data[key])
            self._data[key] = [(ts, v) for ts, v in self._data[key] if ts >= cutoff]
            evicted += before - len(self._data[key])
        return evicted

    def series_count(self) -> int:
        return len(self._data)

    def total_points(self) -> int:
        return sum(len(v) for v in self._data.values())


# ─────────────────────────────────────────────
# COLUMNAR STORE
# ─────────────────────────────────────────────

class ColumnStore:
    """
    Columnar storage: data organized by column, not by row.
    Efficient for OLAP aggregations that touch only a few columns.
    """

    def __init__(self):
        self._columns    : Dict[str, List[Any]] = {}
        self._row_count  = 0
        self.read_bytes  = 0
        self.write_bytes = 0

    def define_schema(self, columns: List[Tuple[str, str]]):
        for col_name, _ in columns:
            self._columns[col_name] = []

    def insert_row(self, row: Dict[str, Any]):
        for col_name in self._columns:
            self._columns[col_name].append(row.get(col_name))
            self.write_bytes += 8   # approximate per value
        self._row_count += 1

    def scan_column(self, column: str) -> List[Any]:
        """Scan entire column — reads only that column's data."""
        col_bytes = len(self._columns.get(column, [])) * 8
        self.read_bytes += col_bytes
        return self._columns.get(column, [])

    def aggregate(self, column: str, func: str = "avg",
                   where_col: str = None, where_val: Any = None) -> float:
        """Aggregate a column with optional filter — reads 1-2 columns only."""
        values = self.scan_column(column)
        if where_col and where_val is not None:
            filter_col = self.scan_column(where_col)
            values = [v for v, f in zip(values, filter_col) if f == where_val]

        if not values:
            return 0.0
        if func == "avg":
            return sum(v for v in values if v is not None) / len(values)
        if func == "sum":
            return sum(v for v in values if v is not None)
        if func == "max":
            return max(v for v in values if v is not None)
        if func == "min":
            return min(v for v in values if v is not None)
        return 0.0

    def compression_ratio_estimate(self, column: str) -> float:
        """
        Columnar data compresses much better:
        - Repeated values (enums, status) → RLE compression
        - Sorted integers → delta encoding
        - Floats → gorilla encoding (used in Prometheus/InfluxDB)
        """
        values = self._columns.get(column, [])
        if not values:
            return 1.0
        unique = len(set(str(v) for v in values if v is not None))
        total  = len(values)
        # More repeated values = better compression
        ratio  = max(1.0, total / max(1, unique) * 2)
        return min(ratio, 20.0)   # cap at 20x

    def report(self):
        print(f"\n  ColumnStore: {self._row_count:,} rows  "
              f"{len(self._columns)} columns")
        print(f"    Read bytes:  {self.read_bytes:,}  "
              f"Write bytes: {self.write_bytes:,}")
        total_size = sum(len(v) for v in self._columns.values()) * 8
        print(f"    Estimated storage: {total_size / 1024:.1f} KB")


# ─────────────────────────────────────────────
# ROW STORE (for comparison)
# ─────────────────────────────────────────────

class RowStore:
    """Traditional row-based storage for OLTP comparison."""

    def __init__(self):
        self._rows      : List[Dict] = []
        self.read_bytes = 0

    def insert(self, row: Dict):
        self._rows.append(row)

    def aggregate_column(self, column: str) -> float:
        """Must read ALL columns of ALL rows to compute aggregate."""
        row_size = 8 * 10   # assume 10 columns per row
        self.read_bytes += len(self._rows) * row_size
        vals = [r.get(column) for r in self._rows if column in r]
        return sum(v for v in vals if v is not None) / max(1, len(vals))


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_tsdb_columnar():
    print("=" * 65)
    print("TIME-SERIES AND COLUMNAR DATABASES")
    print("=" * 65)

    random.seed(42)

    # ── Time-Series Ingestion ─────────────────
    print("\n[1] TIME-SERIES INGESTION")
    print("─" * 55)
    tsdb = TimeSeriesStore(retention_hours=1)

    now = time.time()
    hosts = ["web-1", "web-2", "web-3"]
    print(f"  Writing 1800 data points (1 per second × 3 hosts × 10 minutes)...")
    for host in hosts:
        for sec in range(600):
            # Simulate CPU metric with realistic pattern
            cpu  = 30 + 20 * abs(sec % 60 / 60 - 0.5) + random.uniform(-5, 5)
            tsdb.write(DataPoint(
                metric="cpu_usage",
                tags={"host": host, "dc": "us-east"},
                value=round(cpu, 2),
                timestamp=now - (600 - sec)
            ))

    print(f"  Ingested: {tsdb.write_count} points  "
          f"Series: {tsdb.series_count()}")

    # Range query
    start_ts = now - 300   # last 5 minutes
    end_ts   = now
    points = tsdb.query_range("cpu_usage", {"host": "web-1", "dc": "us-east"},
                               start_ts, end_ts)
    print(f"\n  Range query [last 5 min] for web-1: {len(points)} points")
    if points:
        vals = [v for _, v in points]
        print(f"  avg={sum(vals)/len(vals):.1f}%  min={min(vals):.1f}%  max={max(vals):.1f}%")

    # Downsampling
    buckets = tsdb.downsample("cpu_usage", {"host": "web-1", "dc": "us-east"},
                               bucket_s=60.0, start_ts=start_ts, end_ts=end_ts)
    print(f"\n  Downsampled to 1-minute buckets: {len(buckets)} buckets")
    for b in buckets[:3]:
        import datetime
        ts_str = datetime.datetime.fromtimestamp(b.bucket_ts).strftime("%H:%M")
        print(f"    {ts_str}: avg={b.avg_val:.1f}  min={b.min_val:.1f}  "
              f"max={b.max_val:.1f}  count={b.count}")

    # ── Row vs Columnar ───────────────────────
    print("\n\n[2] ROW STORE vs COLUMN STORE — OLAP QUERY")
    print("─" * 55)
    N = 100_000
    row_store = RowStore()
    col_store = ColumnStore()
    col_store.define_schema([
        ("order_id", "BIGINT"), ("user_id", "BIGINT"),
        ("product_id", "BIGINT"), ("category", "VARCHAR"),
        ("quantity", "INT"), ("unit_price", "DECIMAL"),
        ("total_usd", "DECIMAL"), ("status", "VARCHAR"),
        ("country", "VARCHAR"), ("created_at", "TIMESTAMP"),
    ])

    categories = ["electronics", "clothing", "books", "home", "sports"]
    countries  = ["US", "UK", "DE", "FR", "IN"]
    print(f"  Loading {N:,} order rows...")
    for i in range(N):
        row = {
            "order_id": i, "user_id": random.randint(1, 10000),
            "product_id": random.randint(1, 5000),
            "category": random.choice(categories),
            "quantity": random.randint(1, 5),
            "unit_price": round(random.uniform(5, 200), 2),
            "total_usd": round(random.uniform(5, 1000), 2),
            "status": random.choice(["completed", "shipped", "pending"]),
            "country": random.choice(countries),
            "created_at": now - random.randint(0, 86400 * 30),
        }
        row_store.insert(row)
        col_store.insert_row(row)

    # Run same aggregate query
    row_avg = row_store.aggregate_column("total_usd")
    col_avg = col_store.aggregate("total_usd")

    print(f"\n  Query: SELECT AVG(total_usd) FROM orders WHERE country='US'")
    print(f"    Row store: reads {row_store.read_bytes:,} bytes (entire table)")
    col_store.read_bytes = 0   # reset
    col_avg_us = col_store.aggregate("total_usd", where_col="country", where_val="US")
    print(f"    Col store: reads {col_store.read_bytes:,} bytes (2 columns only)")
    row_cols = 10   # total columns in row
    print(f"    Column store reads {row_cols}x less data")

    # ── Compression ───────────────────────────
    print(f"\n\n[3] COLUMNAR COMPRESSION RATIOS")
    print(f"─" * 55)
    for col_name in ["category", "country", "status", "total_usd", "user_id"]:
        ratio = col_store.compression_ratio_estimate(col_name)
        print(f"  {col_name:<15}: ~{ratio:.0f}x compression")
    print("  (Enum-like cols compress dramatically — RLE; floats use delta/gorilla)")

    col_store.report()

    # ── TSDB Retention ────────────────────────
    print(f"\n\n[4] RETENTION POLICY AND DOWNSAMPLING")
    print(f"─" * 55)
    retention = [
        ("1 second",  "1 hour",   "High-res, short window (live dashboard)"),
        ("1 minute",  "7 days",   "Operational monitoring"),
        ("1 hour",    "90 days",  "Trend analysis"),
        ("1 day",     "2 years",  "Long-term capacity planning"),
    ]
    print(f"  {'Resolution':<14} {'Retention':<12} Purpose")
    for res, ret, purpose in retention:
        print(f"  {res:<14} {ret:<12} {purpose}")

    # ── Use Cases ─────────────────────────────
    print(f"\n\n[5] WHEN TO USE TSDB vs COLUMNAR vs ROW")
    print(f"─" * 55)
    guide = [
        ("Metrics/monitoring",    "TSDB",     "InfluxDB, Prometheus, TimescaleDB"),
        ("IoT sensor streams",    "TSDB",     "InfluxDB, Apache Druid"),
        ("Financial tick data",   "TSDB",     "KDB+, TimescaleDB"),
        ("Analytics warehouse",   "Columnar", "Redshift, BigQuery, ClickHouse"),
        ("Log aggregation",       "Columnar", "Elasticsearch, ClickHouse"),
        ("OLTP transactions",     "Row store","PostgreSQL, MySQL"),
        ("Mixed OLTP+OLAP",       "Hybrid",   "Snowflake, Aurora, TiDB"),
    ]
    print(f"  {'Use Case':<28} {'Type':<12} {'Examples'}")
    print(f"  {'─'*65}")
    for use_case, db_type, examples in guide:
        print(f"  {use_case:<28} {db_type:<12} {examples}")


if __name__ == "__main__":
    demonstrate_tsdb_columnar()
