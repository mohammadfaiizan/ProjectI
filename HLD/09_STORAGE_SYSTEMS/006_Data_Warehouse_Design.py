"""
DATA WAREHOUSE DESIGN
======================

Problem Statement:
Analytical queries (GROUP BY, aggregations, time-series) over billions of rows
are too slow on OLTP databases optimized for row-level transactions.
Data warehouses are built for OLAP: columnar storage, bulk loads, complex aggregations.

OLTP vs OLAP:
  OLTP: Many small read/write transactions. Row-oriented. Normalized (3NF).
        Low latency per query. Examples: PostgreSQL, MySQL.
  OLAP: Few large analytical queries. Columnar. Denormalized (star/snowflake).
        High throughput aggregations. Examples: Redshift, BigQuery, Snowflake.

Storage: Columnar vs Row:
  Row storage: all columns of a row together. Good for row fetch (OLTP).
  Columnar: each column stored separately. Good for: aggregations, projections.
            Read only columns you need. Better compression (similar values together).
            Vectorized execution: SIMD operations on column arrays.

Data Modeling:
  Star Schema:      Fact table (events) at center.
                    Dimension tables (users, products, dates) as star arms.
                    Denormalized: fast joins (pre-joined dimensions).
  Snowflake Schema: Normalized dimensions (sub-dimensions).
                    More joins, less redundancy.
  Wide Table:       Single mega-table with all dimensions pre-joined.
                    BigQuery/Redshift recommendation for analytics.

ETL vs ELT:
  ETL (Extract-Transform-Load): Transform before loading (traditional DW).
  ELT (Extract-Load-Transform): Load raw, transform inside warehouse.
  Modern: ELT preferred (warehouse is cheap for compute, dbt for transforms).

Materialized Views / Pre-aggregation:
  Pre-compute heavy aggregations. Serve queries from materialized results.
  Trade-off: storage + maintenance overhead vs query speed.

Distribution (Redshift / Snowflake):
  Hash distribution: rows with same key go to same node → co-locate joins.
  Round-robin: even distribution, no co-location.
  Replicated: small dimension tables copied to all nodes.

Sort Keys / Clustering:
  Sort data on disk by common filter columns (date, region).
  Range queries read fewer blocks → less I/O.
  BigQuery: cluster by (date, user_id) avoids full table scans.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple
from enum import Enum
from collections import defaultdict
import time
import math


# ─────────────────────────────────────────────
# COLUMNAR STORAGE SIMULATION
# ─────────────────────────────────────────────

class ColumnStore:
    """
    Columnar storage engine: each column stored as a separate array.
    Supports projection (select specific columns) and predicate pushdown.
    """

    def __init__(self):
        self._columns: Dict[str, List[Any]] = {}
        self._n_rows  = 0

    def insert_rows(self, rows: List[Dict]):
        if not rows:
            return
        if not self._columns:
            for col in rows[0]:
                self._columns[col] = []

        for row in rows:
            for col in self._columns:
                self._columns[col].append(row.get(col))
            self._n_rows += 1

    def scan(self, select_cols: List[str] = None,
             where: Dict[str, Any] = None,
             limit: int = None) -> Tuple[List[Dict], int]:
        """Returns (rows, blocks_read). Columnar scan with projection."""
        if not self._columns:
            return [], 0

        cols   = select_cols or list(self._columns.keys())
        # Columnar predicate: build row mask
        mask   = [True] * self._n_rows
        blocks_read = 0
        if where:
            for col, val in where.items():
                if col not in self._columns:
                    continue
                col_data = self._columns[col]
                blocks_read += 1   # read only this column
                for i in range(self._n_rows):
                    if mask[i] and col_data[i] != val:
                        mask[i] = False

        # Projection: only read requested columns
        blocks_read += len(cols)

        result = []
        for i in range(self._n_rows):
            if not mask[i]:
                continue
            row = {c: self._columns[c][i] for c in cols if c in self._columns}
            result.append(row)
            if limit and len(result) >= limit:
                break

        return result, blocks_read

    def aggregate(self, group_by: List[str], metrics: Dict[str, str],
                  where: Dict = None) -> List[Dict]:
        """
        Group-by aggregation. metrics = {"col": "sum"|"count"|"avg"|"max"|"min"}.
        Columns: GROUP BY + aggregate.
        """
        rows, _ = self.scan(where=where)
        groups: Dict[Tuple, Dict[str, List]] = defaultdict(lambda: defaultdict(list))
        for row in rows:
            key = tuple(row.get(c) for c in group_by)
            for m_col in metrics:
                groups[key][m_col].append(row.get(m_col, 0) or 0)

        result = []
        for key, col_data in groups.items():
            agg_row = dict(zip(group_by, key))
            for m_col, func in metrics.items():
                vals = col_data[m_col]
                if func == "sum"  : agg_row[f"{func}_{m_col}"] = sum(vals)
                elif func == "count": agg_row[f"{func}_{m_col}"] = len(vals)
                elif func == "avg" : agg_row[f"{func}_{m_col}"] = sum(vals) / len(vals) if vals else 0
                elif func == "max" : agg_row[f"{func}_{m_col}"] = max(vals) if vals else None
                elif func == "min" : agg_row[f"{func}_{m_col}"] = min(vals) if vals else None
            result.append(agg_row)

        return sorted(result, key=lambda r: r.get(group_by[0], ""))

    @property
    def n_rows(self) -> int:
        return self._n_rows

    def compression_stats(self) -> Dict:
        """Simulate compression: repeated values in a column compress well."""
        stats = {}
        for col, data in self._columns.items():
            unique_ratio = len(set(str(v) for v in data)) / max(len(data), 1)
            simulated_ratio = 0.1 + unique_ratio * 0.8  # low cardinality compresses better
            stats[col] = {"unique_ratio": unique_ratio, "compression": simulated_ratio}
        return stats


# ─────────────────────────────────────────────
# STAR SCHEMA
# ─────────────────────────────────────────────

@dataclass
class DimUser:
    user_id   : str
    name      : str
    country   : str
    plan      : str


@dataclass
class DimProduct:
    product_id : str
    name       : str
    category   : str
    price      : float


@dataclass
class DimDate:
    date_key  : str   # "2024-01-15"
    year      : int
    month     : int
    day       : int
    day_of_week: str
    quarter   : int


@dataclass
class FactOrder:
    order_id   : str
    user_id    : str
    product_id : str
    date_key   : str
    quantity   : int
    amount     : float
    discount   : float


class StarSchemaWarehouse:
    """
    Star schema: fact table + dimension tables.
    Denormalized dimensions for fast analytics.
    """

    def __init__(self):
        self._dim_users    : Dict[str, DimUser]    = {}
        self._dim_products : Dict[str, DimProduct] = {}
        self._dim_dates    : Dict[str, DimDate]    = {}
        self._fact_orders  = ColumnStore()

    def load_dim_user(self, user: DimUser):
        self._dim_users[user.user_id] = user

    def load_dim_product(self, product: DimProduct):
        self._dim_products[product.product_id] = product

    def load_dim_date(self, date: DimDate):
        self._dim_dates[date.date_key] = date

    def load_fact(self, order: FactOrder):
        # Denormalize: join dimension attributes into fact row
        user    = self._dim_users.get(order.user_id)
        product = self._dim_products.get(order.product_id)
        date    = self._dim_dates.get(order.date_key)
        row = {
            "order_id"    : order.order_id,
            "user_id"     : order.user_id,
            "user_country": user.country if user else None,
            "user_plan"   : user.plan    if user else None,
            "product_id"  : order.product_id,
            "category"    : product.category if product else None,
            "date_key"    : order.date_key,
            "year"        : date.year    if date else None,
            "month"       : date.month   if date else None,
            "quantity"    : order.quantity,
            "amount"      : order.amount,
            "discount"    : order.discount,
        }
        self._fact_orders.insert_rows([row])

    def query_revenue_by_category(self, year: int) -> List[Dict]:
        return self._fact_orders.aggregate(
            group_by=["category"],
            metrics={"amount": "sum", "order_id": "count"},
            where={"year": year},
        )

    def query_revenue_by_country(self, year: int, month: int) -> List[Dict]:
        rows, _ = self._fact_orders.scan(where={"year": year, "month": month})
        agg: Dict[str, float] = defaultdict(float)
        cnt: Dict[str, int]   = defaultdict(int)
        for row in rows:
            c = row.get("user_country", "unknown")
            agg[c] += row.get("amount", 0)
            cnt[c] += 1
        return [{"country": c, "revenue": agg[c], "orders": cnt[c]}
                for c in sorted(agg)]

    def total_rows(self) -> int:
        return self._fact_orders.n_rows


# ─────────────────────────────────────────────
# MATERIALIZED VIEW
# ─────────────────────────────────────────────

class MaterializedView:
    """Pre-computed aggregation refreshed on schedule."""

    def __init__(self, name: str, refresh_interval_s: float = 3600):
        self.name             = name
        self._data            : List[Dict] = []
        self._last_refreshed  : Optional[float] = None
        self.refresh_interval = refresh_interval_s
        self.refreshes        = 0

    def refresh(self, query_fn) -> int:
        self._data          = query_fn()
        self._last_refreshed = time.time()
        self.refreshes      += 1
        return len(self._data)

    def query(self) -> List[Dict]:
        return self._data

    def is_stale(self) -> bool:
        if self._last_refreshed is None:
            return True
        return (time.time() - self._last_refreshed) > self.refresh_interval

    def staleness_s(self) -> float:
        if self._last_refreshed is None:
            return float("inf")
        return time.time() - self._last_refreshed


# ─────────────────────────────────────────────
# QUERY COST ESTIMATOR
# ─────────────────────────────────────────────

def estimate_query_cost(
    n_rows: int,
    n_columns_total: int,
    n_columns_selected: int,
    has_index: bool = False,
    is_columnar: bool = True,
) -> Dict[str, float]:
    """Rough I/O cost model for row vs columnar storage."""
    row_storage_blocks = math.ceil(n_rows / 100)   # 100 rows per block (row store)
    col_storage_blocks = math.ceil(n_rows / 1000)  # 1000 values per block (columnar)

    if is_columnar:
        # Only read selected columns
        blocks_read = col_storage_blocks * n_columns_selected
    else:
        # Row store: read all blocks even for partial column set
        blocks_read = row_storage_blocks if not has_index else row_storage_blocks // 10

    return {
        "rows_scanned": n_rows,
        "blocks_read" : blocks_read,
        "column_ratio": n_columns_selected / n_columns_total,
        "relative_cost": blocks_read,
    }


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_data_warehouse():
    print("=" * 65)
    print("DATA WAREHOUSE DESIGN")
    print("=" * 65)

    wh = StarSchemaWarehouse()

    # ── Load Dimensions ───────────────────────────
    print("\n[1] STAR SCHEMA — LOAD DIMENSIONS")
    print("─" * 55)

    users = [
        DimUser("u1", "Alice", "US", "premium"),
        DimUser("u2", "Bob",   "UK", "free"),
        DimUser("u3", "Carol", "US", "premium"),
        DimUser("u4", "Dave",  "DE", "free"),
    ]
    products = [
        DimProduct("p1", "Laptop",    "Electronics", 999.0),
        DimProduct("p2", "T-Shirt",   "Apparel",     29.0),
        DimProduct("p3", "Headphones","Electronics", 149.0),
        DimProduct("p4", "Notebook",  "Stationery",  5.0),
    ]
    dates = [
        DimDate("2024-01-15", 2024, 1, 15, "Mon", 1),
        DimDate("2024-01-16", 2024, 1, 16, "Tue", 1),
        DimDate("2024-02-10", 2024, 2, 10, "Sat", 1),
        DimDate("2024-03-05", 2024, 3, 5,  "Tue", 1),
    ]
    for u in users:    wh.load_dim_user(u)
    for p in products: wh.load_dim_product(p)
    for d in dates:    wh.load_dim_date(d)
    print(f"  Loaded: {len(users)} users, {len(products)} products, {len(dates)} dates")

    # ── Load Facts ────────────────────────────────
    print("\n\n[2] LOAD FACT TABLE (denormalized)")
    print("─" * 55)

    import random as _r; _r.seed(42)
    orders = [
        FactOrder(f"o{i}", _r.choice(["u1","u2","u3","u4"]),
                  _r.choice(["p1","p2","p3","p4"]),
                  _r.choice(["2024-01-15","2024-01-16","2024-02-10","2024-03-05"]),
                  _r.randint(1,5),
                  round(_r.uniform(5, 1000), 2),
                  round(_r.uniform(0, 0.2), 2))
        for i in range(20)
    ]
    for o in orders:
        wh.load_fact(o)
    print(f"  Fact rows: {wh.total_rows()} (denormalized with dim attributes)")

    # ── Columnar Aggregation ──────────────────────
    print("\n\n[3] ANALYTICAL QUERIES")
    print("─" * 55)

    rev_by_cat = wh.query_revenue_by_category(2024)
    print("  Revenue by category (2024):")
    for row in sorted(rev_by_cat, key=lambda r: -r.get("sum_amount", 0)):
        print(f"    {row.get('category','?'):<14} ${row.get('sum_amount',0):>8.2f}  "
              f"orders={row.get('count_order_id',0)}")

    rev_by_country = wh.query_revenue_by_country(2024, 1)
    print("\n  Revenue by country (Jan 2024):")
    for row in sorted(rev_by_country, key=lambda r: -r["revenue"]):
        print(f"    {row['country']:<6} ${row['revenue']:>8.2f}  orders={row['orders']}")

    # ── Columnar Compression ──────────────────────
    print("\n\n[4] COLUMNAR COMPRESSION ADVANTAGE")
    print("─" * 55)

    cstats = wh._fact_orders.compression_stats()
    print(f"  {'Column':<16} {'Unique ratio':>14} {'Compression ratio':>18}")
    print(f"  {'─'*50}")
    for col, s in list(cstats.items())[:7]:
        print(f"  {col:<16} {s['unique_ratio']:>14.2f} {s['compression']:>18.2f}")
    print(f"  Low-cardinality cols (country, category) compress ~10x")

    # ── Materialized View ─────────────────────────
    print("\n\n[5] MATERIALIZED VIEW — PRE-AGGREGATED RESULTS")
    print("─" * 55)

    mv = MaterializedView("mv_revenue_by_category_2024", refresh_interval_s=3600)
    n  = mv.refresh(lambda: wh.query_revenue_by_category(2024))
    print(f"  Refreshed materialized view: {n} rows, stale={mv.is_stale()}")
    print(f"  Query hits MV directly (no re-aggregation):")
    for row in mv.query()[:3]:
        print(f"    {row}")

    # ── I/O Cost: Row vs Columnar ─────────────────
    print("\n\n[6] QUERY I/O COST: ROW vs COLUMNAR STORAGE")
    print("─" * 55)

    scenarios = [
        ("Select 2 of 50 cols", 50, 2, False),
        ("Select 10 of 50 cols", 50, 10, False),
        ("Select 50 of 50 cols", 50, 50, False),
    ]
    print(f"  {'Scenario':<28} {'Columnar blocks':>16} {'Row-store blocks':>17}")
    print(f"  {'─'*63}")
    for desc, total_cols, sel_cols, has_idx in scenarios:
        col_cost = estimate_query_cost(1_000_000, total_cols, sel_cols, has_idx, True)
        row_cost = estimate_query_cost(1_000_000, total_cols, sel_cols, has_idx, False)
        print(f"  {desc:<28} {col_cost['blocks_read']:>16,} {row_cost['blocks_read']:>17,}")

    # ── Design Summary ────────────────────────────
    print("\n\n[7] DATA WAREHOUSE DESIGN DECISIONS")
    print("─" * 55)
    rows_summary = [
        ("Columnar storage",  "Read only needed columns; 10-50x I/O reduction"),
        ("Star schema",       "Single fact + dim tables; minimal joins"),
        ("Denormalization",   "Pre-join dims into fact → faster GROUP BY"),
        ("Partitioning",      "Partition by date → skip non-matching partitions"),
        ("Sort keys",         "Sort by common filter col → contiguous scan"),
        ("Materialized views","Pre-aggregate frequent queries → sub-second"),
        ("ELT pattern",       "Load raw first, transform in warehouse (dbt)"),
        ("Redshift DIST keys","Co-locate join keys on same node → no shuffle"),
    ]
    for decision, reason in rows_summary:
        print(f"  {decision:<24} {reason}")


if __name__ == "__main__":
    demonstrate_data_warehouse()
