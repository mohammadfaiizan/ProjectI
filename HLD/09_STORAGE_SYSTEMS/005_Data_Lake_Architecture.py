"""
DATA LAKE ARCHITECTURE
========================

Problem Statement:
Enterprises accumulate massive volumes of raw data from many sources
(logs, IoT, events, databases). A data lake stores ALL raw data cheaply
and allows schema-on-read analysis at scale.

Data Lake vs Data Warehouse:
  Data Lake:       Raw + unstructured + schema-on-read. Cheap (object storage).
                   Flexible but harder to query. Risk: data swamp.
  Data Warehouse:  Structured + schema-on-write. Expensive.
                   Fast queries but requires ETL transformation first.
  Modern Lakehouse: Combines both — raw data in lake + open table format
                   (Delta Lake, Apache Iceberg, Apache Hudi) for ACID queries.

Layers:
  Raw / Bronze:  Unprocessed data exactly as received.
                 Never delete. Append-only. Source of truth.
  Processed / Silver: Cleaned, deduplicated, typed. Joins applied.
  Curated / Gold: Business aggregates. Ready for dashboards, ML.

Open Table Formats (lakehouse):
  Delta Lake:    Transaction log (JSON) over Parquet. ACID, time travel.
  Apache Iceberg: Manifest files + snapshot model. Hidden partitioning.
  Apache Hudi:   Record-level upserts. Incremental processing.
  Key features: ACID transactions, schema evolution, time travel,
                partition pruning, compaction.

Ingestion Patterns:
  Batch:   Nightly ETL from operational DB. Simple but stale.
  Micro-batch: Spark Streaming / Flink every few minutes.
  Streaming: Kafka → Flink → sink to lake in real-time (seconds latency).

Partitioning Strategy:
  By date: year=2024/month=01/day=15 → prune most time-range queries.
  By region, tenant, event_type as additional sub-partitions.
  Avoid: too many small files (small file problem → compaction needed).

Data Catalog:
  AWS Glue, Apache Atlas, DataHub.
  Stores schema, lineage, tags, ownership.
  Enables discovery without reading files.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple
from enum import Enum
import time
import uuid
import json
import hashlib


# ─────────────────────────────────────────────
# LAKE ZONES
# ─────────────────────────────────────────────

class LakeZone(Enum):
    RAW       = "raw"        # Bronze
    PROCESSED = "processed"  # Silver
    CURATED   = "curated"    # Gold


# ─────────────────────────────────────────────
# SCHEMA
# ─────────────────────────────────────────────

@dataclass
class ColumnDef:
    name     : str
    data_type: str    # "string", "int", "timestamp", "double", "boolean"
    nullable : bool = True
    comment  : str  = ""


@dataclass
class Schema:
    name     : str
    version  : int
    columns  : List[ColumnDef]
    partition_keys: List[str] = field(default_factory=list)

    def column_names(self) -> List[str]:
        return [c.name for c in self.columns]

    def evolve(self, new_column: ColumnDef) -> "Schema":
        """Schema evolution: add nullable column (backward compatible)."""
        if not new_column.nullable:
            raise ValueError("Schema evolution: new columns must be nullable")
        return Schema(
            name=self.name,
            version=self.version + 1,
            columns=self.columns + [new_column],
            partition_keys=self.partition_keys,
        )


# ─────────────────────────────────────────────
# PARQUET FILE SIMULATION
# ─────────────────────────────────────────────

@dataclass
class ParquetFile:
    """Simulates a Parquet file: columnar, compressed, schema-embedded."""
    file_path    : str
    schema       : Schema
    rows         : List[Dict]
    partition    : Dict[str, str]      # partition key → value
    size_bytes   : int = 0
    row_count    : int = 0
    created_at   : float = field(default_factory=time.time)

    def __post_init__(self):
        self.row_count  = len(self.rows)
        self.size_bytes = sum(len(json.dumps(r).encode()) for r in self.rows)

    def min_max(self, column: str) -> Tuple[Any, Any]:
        vals = [r.get(column) for r in self.rows if column in r]
        return (min(vals), max(vals)) if vals else (None, None)


# ─────────────────────────────────────────────
# DELTA LOG (simplified transaction log)
# ─────────────────────────────────────────────

@dataclass
class DeltaAction:
    action    : str        # "add", "remove", "metadata"
    file_path : str
    partition : Dict[str, str]
    stats     : Dict       # min/max/count per column
    timestamp : float = field(default_factory=time.time)


@dataclass
class DeltaCommit:
    version   : int
    actions   : List[DeltaAction]
    operation : str       # "WRITE", "DELETE", "MERGE", "OPTIMIZE"
    timestamp : float = field(default_factory=time.time)


class DeltaTable:
    """
    Simplified Delta Lake table: transaction log + Parquet files.
    Supports: ACID writes, time travel, schema evolution.
    """

    def __init__(self, table_name: str, schema: Schema, location: str):
        self.table_name  = table_name
        self._schema     = schema
        self.location    = location
        self._log        : List[DeltaCommit] = []       # ordered commits
        self._files      : Dict[str, ParquetFile] = {}  # active files
        self._version    = 0

    @property
    def current_version(self) -> int:
        return self._version

    def write(self, rows: List[Dict], partition: Dict[str, str],
              mode: str = "append") -> int:
        """Write rows. Returns new version number."""
        if mode == "overwrite":
            # Remove all existing files for this partition
            to_remove = [f for f, pf in self._files.items()
                         if pf.partition == partition]
            for f in to_remove:
                del self._files[f]

        file_id = str(uuid.uuid4())[:8]
        file_path = f"{self.location}/{'/'.join(f'{k}={v}' for k, v in partition.items())}/{file_id}.parquet"

        pfile = ParquetFile(file_path=file_path, schema=self._schema,
                            rows=rows, partition=partition)
        self._files[file_path] = pfile

        # Stats for partition pruning
        stats = {col.name: {"min": pfile.min_max(col.name)[0],
                             "max": pfile.min_max(col.name)[1],
                             "count": pfile.row_count}
                 for col in self._schema.columns[:3]}

        action = DeltaAction(action="add", file_path=file_path,
                             partition=partition, stats=stats)
        commit = DeltaCommit(version=self._version + 1,
                             actions=[action], operation="WRITE")
        self._log.append(commit)
        self._version += 1
        return self._version

    def read(self, filters: Dict = None, time_travel_version: int = None) -> List[Dict]:
        """Read with optional partition pruning and time travel."""
        if time_travel_version is not None:
            return self._read_at_version(time_travel_version, filters)

        result = []
        for fpath, pfile in self._files.items():
            # Partition pruning
            if filters:
                skip = False
                for k, v in filters.items():
                    if k in pfile.partition and str(pfile.partition[k]) != str(v):
                        skip = True
                        break
                if skip:
                    continue
            # Row filtering
            for row in pfile.rows:
                if filters:
                    match = all(str(row.get(k)) == str(v)
                                for k, v in filters.items()
                                if k not in pfile.partition)
                    if not match:
                        continue
                result.append(row)
        return result

    def _read_at_version(self, target_version: int, filters: Dict = None) -> List[Dict]:
        """Time travel: reconstruct state at a past version."""
        active_files: Dict[str, ParquetFile] = {}
        for commit in self._log:
            if commit.version > target_version:
                break
            for action in commit.actions:
                if action.action == "add":
                    # We need the actual file; grab from current state
                    if action.file_path in self._files:
                        active_files[action.file_path] = self._files[action.file_path]
                elif action.action == "remove":
                    active_files.pop(action.file_path, None)

        result = []
        for pfile in active_files.values():
            if filters:
                skip = any(k in pfile.partition and str(pfile.partition[k]) != str(v)
                           for k, v in filters.items())
                if skip:
                    continue
            result.extend(pfile.rows)
        return result

    def evolve_schema(self, new_column: ColumnDef):
        self._schema = self._schema.evolve(new_column)
        commit = DeltaCommit(
            version=self._version + 1,
            actions=[DeltaAction("metadata", "", {}, {"new_column": new_column.name})],
            operation="SCHEMA_CHANGE",
        )
        self._log.append(commit)
        self._version += 1

    def optimize(self) -> Dict:
        """Compact small files into larger ones (simulated)."""
        small_files = [fp for fp, pf in self._files.items()
                       if pf.size_bytes < 10_000]
        return {"compacted": len(small_files), "version": self._version}

    def stats(self) -> Dict:
        total_rows  = sum(pf.row_count for pf in self._files.values())
        total_bytes = sum(pf.size_bytes for pf in self._files.values())
        return {
            "version"    : self._version,
            "files"      : len(self._files),
            "total_rows" : total_rows,
            "total_bytes": total_bytes,
            "commits"    : len(self._log),
        }


# ─────────────────────────────────────────────
# DATA CATALOG
# ─────────────────────────────────────────────

@dataclass
class CatalogEntry:
    table_name  : str
    zone        : LakeZone
    location    : str
    schema      : Schema
    owner       : str
    tags        : List[str]
    created_at  : float = field(default_factory=time.time)
    description : str   = ""


class DataCatalog:
    """Glue/Atlas-like catalog for table discovery and lineage."""

    def __init__(self):
        self._tables  : Dict[str, CatalogEntry] = {}
        self._lineage : Dict[str, List[str]] = {}   # table → source tables

    def register(self, entry: CatalogEntry, sources: List[str] = None):
        self._tables[entry.table_name] = entry
        if sources:
            self._lineage[entry.table_name] = sources

    def search(self, tag: str = None, zone: LakeZone = None) -> List[CatalogEntry]:
        results = list(self._tables.values())
        if tag:
            results = [e for e in results if tag in e.tags]
        if zone:
            results = [e for e in results if e.zone == zone]
        return results

    def lineage(self, table_name: str, depth: int = 2) -> Dict:
        """Return upstream lineage graph."""
        graph = {}
        queue = [(table_name, 0)]
        while queue:
            name, d = queue.pop(0)
            if d >= depth or name in graph:
                continue
            sources = self._lineage.get(name, [])
            graph[name] = sources
            for src in sources:
                queue.append((src, d + 1))
        return graph


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_data_lake():
    print("=" * 65)
    print("DATA LAKE ARCHITECTURE")
    print("=" * 65)

    # ── Schema Definition ─────────────────────────
    events_schema = Schema(
        name="user_events",
        version=1,
        columns=[
            ColumnDef("user_id",    "string"),
            ColumnDef("event_type", "string"),
            ColumnDef("timestamp",  "timestamp"),
            ColumnDef("properties", "string"),
        ],
        partition_keys=["date", "event_type"],
    )

    # ── Bronze Layer: Raw Ingestion ────────────────
    print("\n[1] BRONZE LAYER — RAW INGESTION (append-only)")
    print("─" * 55)

    bronze = DeltaTable("raw.user_events", events_schema, "s3://lake/raw/user_events")
    raw_batches = [
        ({"date": "2024-01-15", "event_type": "click"}, [
            {"user_id": "u1", "event_type": "click", "timestamp": "2024-01-15T10:00", "properties": '{"page":"/home"}'},
            {"user_id": "u2", "event_type": "click", "timestamp": "2024-01-15T10:01", "properties": '{"page":"/cart"}'},
        ]),
        ({"date": "2024-01-15", "event_type": "purchase"}, [
            {"user_id": "u1", "event_type": "purchase", "timestamp": "2024-01-15T10:05", "properties": '{"amount":99.99}'},
        ]),
        ({"date": "2024-01-16", "event_type": "click"}, [
            {"user_id": "u3", "event_type": "click", "timestamp": "2024-01-16T09:00", "properties": '{"page":"/checkout"}'},
        ]),
    ]
    for partition, rows in raw_batches:
        v = bronze.write(rows, partition)
        print(f"  Wrote {len(rows)} rows → partition={partition} v={v}")

    s = bronze.stats()
    print(f"  Table stats: {s['files']} files, {s['total_rows']} rows, {s['total_bytes']}B")

    # ── Partition Pruning ─────────────────────────
    print("\n\n[2] PARTITION PRUNING — FILTER PUSHDOWN")
    print("─" * 55)

    all_rows  = bronze.read()
    date_rows = bronze.read(filters={"date": "2024-01-15"})
    type_rows = bronze.read(filters={"date": "2024-01-15", "event_type": "click"})
    print(f"  All rows: {len(all_rows)}")
    print(f"  date=2024-01-15: {len(date_rows)} rows (pruned 1 partition)")
    print(f"  date=2024-01-15 + event_type=click: {len(type_rows)} rows")

    # ── Time Travel ───────────────────────────────
    print("\n\n[3] TIME TRAVEL — READ PAST VERSION")
    print("─" * 55)

    rows_v1 = bronze.read(time_travel_version=1)
    rows_v3 = bronze.read(time_travel_version=3)
    print(f"  Data at v1: {len(rows_v1)} rows")
    print(f"  Data at v3: {len(rows_v3)} rows")
    print(f"  Current v{bronze.current_version}: {len(bronze.read())} rows")

    # ── Schema Evolution ──────────────────────────
    print("\n\n[4] SCHEMA EVOLUTION — ADD NULLABLE COLUMN")
    print("─" * 55)

    bronze.evolve_schema(ColumnDef("session_id", "string", nullable=True))
    print(f"  Schema evolved to v{bronze._schema.version}")
    print(f"  New columns: {bronze._schema.column_names()}")

    try:
        bronze.evolve_schema(ColumnDef("user_age", "int", nullable=False))
    except ValueError as e:
        print(f"  Non-nullable column rejected: {e}")

    # ── Silver Layer: Cleaned + Joined ────────────
    print("\n\n[5] SILVER LAYER — CLEANED + TYPED")
    print("─" * 55)

    silver_schema = Schema(
        name="clean_events",
        version=1,
        columns=[
            ColumnDef("user_id",    "string"),
            ColumnDef("event_type", "string"),
            ColumnDef("timestamp",  "timestamp"),
            ColumnDef("page",       "string"),
            ColumnDef("amount",     "double", nullable=True),
        ],
        partition_keys=["date"],
    )
    silver = DeltaTable("silver.events", silver_schema, "s3://lake/silver/events")

    # Transform raw → clean
    import json as _json
    for row in bronze.read():
        props = {}
        try:
            props = _json.loads(row.get("properties", "{}"))
        except Exception:
            pass
        clean = {
            "user_id"   : row["user_id"],
            "event_type": row["event_type"],
            "timestamp" : row["timestamp"],
            "page"      : props.get("page", ""),
            "amount"    : props.get("amount"),
        }
        date = row["timestamp"][:10]
        silver.write([clean], {"date": date})

    ss = silver.stats()
    print(f"  Silver: {ss['files']} files, {ss['total_rows']} rows")

    # ── Gold Layer: Aggregates ────────────────────
    print("\n\n[6] GOLD LAYER — DAILY USER METRICS")
    print("─" * 55)

    from collections import defaultdict
    daily_clicks: Dict = defaultdict(int)
    daily_revenue: Dict = defaultdict(float)
    for row in silver.read():
        date = row["timestamp"][:10]
        if row["event_type"] == "click":
            daily_clicks[date] += 1
        elif row["event_type"] == "purchase" and row.get("amount"):
            daily_revenue[date] += row["amount"]

    gold_schema = Schema("daily_metrics", 1, [
        ColumnDef("date", "string"), ColumnDef("clicks", "int"),
        ColumnDef("revenue", "double"),
    ])
    gold = DeltaTable("gold.daily_metrics", gold_schema, "s3://lake/gold/daily_metrics")
    for date in sorted(set(list(daily_clicks) + list(daily_revenue))):
        gold.write([{"date": date, "clicks": daily_clicks[date],
                     "revenue": daily_revenue[date]}], {"date": date})
    print(f"  Gold metrics:")
    for row in gold.read():
        print(f"    {row['date']}: clicks={row['clicks']} revenue=${row['revenue']:.2f}")

    # ── Data Catalog ──────────────────────────────
    print("\n\n[7] DATA CATALOG — DISCOVERY + LINEAGE")
    print("─" * 55)

    catalog = DataCatalog()
    catalog.register(CatalogEntry("raw.user_events", LakeZone.RAW,
                                  "s3://lake/raw/user_events", events_schema,
                                  "data-eng", ["events", "clickstream"],
                                  description="Raw clickstream from web app"))
    catalog.register(CatalogEntry("silver.events", LakeZone.PROCESSED,
                                  "s3://lake/silver/events", silver_schema,
                                  "data-eng", ["events", "cleaned"]),
                     sources=["raw.user_events"])
    catalog.register(CatalogEntry("gold.daily_metrics", LakeZone.CURATED,
                                  "s3://lake/gold/daily_metrics", gold_schema,
                                  "analytics", ["kpi", "dashboard"]),
                     sources=["silver.events"])

    clickstream_tables = catalog.search(tag="events")
    print(f"  Tables tagged 'events': {[e.table_name for e in clickstream_tables]}")

    lineage = catalog.lineage("gold.daily_metrics")
    print(f"  Lineage for gold.daily_metrics: {lineage}")

    # ── Architecture Summary ──────────────────────
    print("\n\n[8] DATA LAKE DESIGN SUMMARY")
    print("─" * 55)
    rows_summary = [
        ("Raw/Bronze zone",       "Append-only, never delete — source of truth"),
        ("Parquet format",        "Columnar: 10x compression, vectorized reads"),
        ("Partition by date",     "Pruning: skip 99% of files for time-range queries"),
        ("Delta / Iceberg",       "ACID transactions + time travel on object storage"),
        ("Schema evolution",      "Add nullable columns without rewriting existing data"),
        ("Compaction (optimize)", "Merge small files → fewer, larger Parquet files"),
        ("Data catalog",          "Discover tables without reading files; track lineage"),
        ("Bronze→Silver→Gold",    "ETL pipeline: raw → cleaned → business metrics"),
    ]
    for decision, reason in rows_summary:
        print(f"  {decision:<26} {reason}")


if __name__ == "__main__":
    demonstrate_data_lake()
