"""
CHOOSING THE RIGHT DATABASE
==============================

Problem Statement:
With 20+ database categories available, engineers need a structured framework
to match workload characteristics to the right storage engine. The wrong
choice costs orders of magnitude in performance, cost, or reliability.

Decision Dimensions:
  Data Model     : relational, document, key-value, graph, time-series, vector
  Access Pattern : point lookup, range scan, full-text, graph traversal
  Consistency    : strong (CP) vs eventual (AP) — CAP theorem
  Scale          : single node, read replicas, sharding, geo-distributed
  Latency        : microseconds (in-memory) → milliseconds (disk) → seconds (warehouse)
  Durability     : ephemeral cache vs WAL-backed vs RAID-replicated
  Query Complexity: SQL joins, aggregations, ad-hoc vs pre-defined access patterns

Database Categories:
  Relational      : PostgreSQL, MySQL, Aurora — OLTP, ACID, SQL joins
  Document        : MongoDB, Firestore — flexible schema, nested documents
  Key-Value       : Redis, DynamoDB, Memcached — O(1) point lookup
  Wide-Column     : Cassandra, HBase — time-series, write-heavy, massive scale
  Graph           : Neo4j, Amazon Neptune — relationship traversal
  Time-Series     : InfluxDB, TimescaleDB, Prometheus — metrics, IoT
  Search          : Elasticsearch, OpenSearch, Typesense — full-text, relevance
  Vector          : Pinecone, pgvector, Weaviate — ML embeddings, semantic search
  Data Warehouse  : BigQuery, Redshift, Snowflake, ClickHouse — OLAP analytics
  NewSQL          : CockroachDB, Spanner, TiDB — distributed SQL, strong consistency
  In-Memory       : Redis, Memcached — sub-millisecond, cache, sessions

The Polyglot Persistence Pattern:
  Modern systems use multiple databases — each for what it does best.
  Example (e-commerce):
    PostgreSQL  → orders, users (ACID, financial correctness)
    Redis       → sessions, cart, rate limits (sub-ms, ephemeral ok)
    Elasticsearch → product search (BM25 relevance)
    Cassandra   → clickstream events (write-heavy, time-ordered)
    Neo4j       → recommendation engine (graph traversal)
    S3 + Parquet→ analytics warehouse (cheap, columnar)
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import time


class DataModel(Enum):
    RELATIONAL   = "relational"
    DOCUMENT     = "document"
    KEY_VALUE    = "key_value"
    WIDE_COLUMN  = "wide_column"
    GRAPH        = "graph"
    TIME_SERIES  = "time_series"
    SEARCH       = "search"
    VECTOR       = "vector"
    WAREHOUSE    = "data_warehouse"
    IN_MEMORY    = "in_memory"


class ConsistencyLevel(Enum):
    STRONG    = "strong"      # CP — linearizable
    EVENTUAL  = "eventual"    # AP — BASE
    TUNABLE   = "tunable"     # configurable per-query (Cassandra)


class ScaleModel(Enum):
    SINGLE_NODE    = "single_node"
    READ_REPLICAS  = "read_replicas"
    SHARDED        = "sharded"
    GEO_DISTRIBUTED= "geo_distributed"


@dataclass
class DatabaseProfile:
    name             : str
    category         : DataModel
    examples         : List[str]
    consistency      : ConsistencyLevel
    scale_model      : ScaleModel
    latency_p99_ms   : float        # typical p99 for point reads
    write_throughput : str          # qualitative: low/med/high/very_high
    query_flexibility: str          # sql/limited/none
    strengths        : List[str]
    weaknesses       : List[str]
    ideal_for        : List[str]
    avoid_when       : List[str]
    managed_services : List[str]    # cloud-managed options


@dataclass
class WorkloadRequirements:
    name              : str
    data_model_needed : DataModel
    consistency_needed: ConsistencyLevel
    read_pattern      : str   # "point", "range", "fulltext", "graph", "analytics"
    write_volume      : str   # "low", "medium", "high", "extreme"
    latency_sla_ms    : float
    joins_needed      : bool
    schema_flexible   : bool
    geo_distribution  : bool
    max_dataset_gb    : float


# ─────────────────────────────────────────────
# DATABASE CATALOG
# ─────────────────────────────────────────────

def build_database_catalog() -> Dict[str, DatabaseProfile]:
    return {
        "postgresql": DatabaseProfile(
            name="PostgreSQL", category=DataModel.RELATIONAL,
            examples=["PostgreSQL", "Aurora PostgreSQL", "Supabase"],
            consistency=ConsistencyLevel.STRONG,
            scale_model=ScaleModel.READ_REPLICAS,
            latency_p99_ms=5.0,
            write_throughput="medium",
            query_flexibility="sql",
            strengths=["ACID transactions", "Complex JOINs", "Rich SQL", "JSONB for semi-structured",
                       "Mature ecosystem", "Full-text search (tsvector)", "Row-level security"],
            weaknesses=["Vertical scaling limit", "Schema migrations can lock",
                        "Not designed for massive write throughput (10K+ TPS)"],
            ideal_for=["E-commerce orders", "Financial transactions", "User management",
                       "Content management", "Any OLTP needing correctness"],
            avoid_when=["Storing billions of time-series points", "Graph traversal 5+ hops",
                        "Full-text search at Google scale", "Sub-millisecond latency required"],
            managed_services=["Amazon RDS", "Cloud SQL", "Azure Database", "Supabase", "Neon"]
        ),

        "mongodb": DatabaseProfile(
            name="MongoDB", category=DataModel.DOCUMENT,
            examples=["MongoDB Atlas", "DocumentDB", "Firestore"],
            consistency=ConsistencyLevel.EVENTUAL,
            scale_model=ScaleModel.SHARDED,
            latency_p99_ms=10.0,
            write_throughput="high",
            query_flexibility="limited",
            strengths=["Flexible schema (no migrations for new fields)", "Nested documents",
                       "Horizontal sharding built-in", "Good for hierarchical data"],
            weaknesses=["No joins across collections", "No multi-collection ACID (w/o transactions)",
                        "Poor for complex reporting", "Denormalization leads to large documents"],
            ideal_for=["Product catalogs (variable attributes)", "CMS content",
                       "User profiles with varied fields", "Event logs"],
            avoid_when=["Financial ledgers needing ACID", "Complex cross-entity reports",
                        "Highly relational data with many joins"],
            managed_services=["MongoDB Atlas", "AWS DocumentDB", "Google Firestore"]
        ),

        "redis": DatabaseProfile(
            name="Redis", category=DataModel.IN_MEMORY,
            examples=["Redis", "Memcached", "Elasticache", "Upstash"],
            consistency=ConsistencyLevel.EVENTUAL,
            scale_model=ScaleModel.READ_REPLICAS,
            latency_p99_ms=0.5,
            write_throughput="very_high",
            query_flexibility="none",
            strengths=["Sub-millisecond latency", "Rich data structures (sorted sets, streams)",
                       "Pub/Sub", "TTL-based expiry", "Atomic operations (INCR, SETNX)"],
            weaknesses=["Data must fit in RAM", "Persistence is async (potential data loss)",
                        "No complex queries", "Expensive at scale"],
            ideal_for=["Session store", "Rate limiting (token bucket)", "Leaderboards",
                       "Cache (LRU eviction)", "Distributed locks", "Real-time counters",
                       "Job queues (BullMQ)", "Pub/Sub messaging"],
            avoid_when=["Primary database for business data", "Large datasets > RAM size",
                        "Complex querying", "Strict durability required"],
            managed_services=["Amazon ElastiCache", "Upstash Redis", "Redis Cloud", "Momento"]
        ),

        "cassandra": DatabaseProfile(
            name="Cassandra", category=DataModel.WIDE_COLUMN,
            examples=["Apache Cassandra", "DataStax Astra", "ScyllaDB"],
            consistency=ConsistencyLevel.TUNABLE,
            scale_model=ScaleModel.GEO_DISTRIBUTED,
            latency_p99_ms=5.0,
            write_throughput="very_high",
            query_flexibility="limited",
            strengths=["Massive write throughput (millions/sec)", "Linear horizontal scaling",
                       "Multi-datacenter replication", "No single point of failure",
                       "Tunable consistency (ONE/QUORUM/ALL)"],
            weaknesses=["No JOINs", "Limited secondary indexes", "Data model must match queries",
                        "No ACID across partitions", "Hard to change queries after design"],
            ideal_for=["IoT sensor data", "Clickstream events", "Message histories",
                       "Recommendation features", "Audit logs at massive scale"],
            avoid_when=["Ad-hoc queries not known at design time", "Relational/JOIN-heavy workloads",
                        "Strong consistency required globally"],
            managed_services=["DataStax Astra", "Amazon Keyspaces", "ScyllaDB Cloud"]
        ),

        "elasticsearch": DatabaseProfile(
            name="Elasticsearch", category=DataModel.SEARCH,
            examples=["Elasticsearch", "OpenSearch", "Typesense", "Meilisearch"],
            consistency=ConsistencyLevel.EVENTUAL,
            scale_model=ScaleModel.SHARDED,
            latency_p99_ms=50.0,
            write_throughput="high",
            query_flexibility="limited",
            strengths=["Full-text search with BM25 relevance", "Faceted search / aggregations",
                       "Geo-search", "Log analytics (ELK stack)", "Autocomplete"],
            weaknesses=["Not a primary database (eventual consistency)", "High memory usage",
                        "Expensive to operate", "No ACID transactions", "Schema mapping can be tricky"],
            ideal_for=["Product search", "Log aggregation (ELK)", "Application search",
                       "Autocomplete", "Anomaly detection on logs"],
            avoid_when=["Primary transactional store", "Simple key-value lookups",
                        "Strong consistency needed", "Budget-constrained small apps"],
            managed_services=["Elastic Cloud", "Amazon OpenSearch", "Bonsai"]
        ),

        "neo4j": DatabaseProfile(
            name="Neo4j", category=DataModel.GRAPH,
            examples=["Neo4j", "Amazon Neptune", "TigerGraph"],
            consistency=ConsistencyLevel.STRONG,
            scale_model=ScaleModel.READ_REPLICAS,
            latency_p99_ms=20.0,
            write_throughput="medium",
            query_flexibility="limited",
            strengths=["O(k) per-hop traversal (index-free adjacency)", "Cypher query language",
                       "Pattern matching across relationships", "Graph algorithms (PageRank, communities)"],
            weaknesses=["Not designed for massive write throughput", "Memory-intensive for large graphs",
                        "Limited sharding (single graph topology)", "Niche skill set"],
            ideal_for=["Social networks (friend-of-friend)", "Fraud ring detection",
                       "Recommendation engines", "Knowledge graphs", "Network topology"],
            avoid_when=["Simple key-value or document lookups", "Massive write throughput needed",
                        "Large-scale distributed graph (billions of nodes)"],
            managed_services=["Neo4j Aura", "Amazon Neptune"]
        ),

        "influxdb": DatabaseProfile(
            name="InfluxDB", category=DataModel.TIME_SERIES,
            examples=["InfluxDB", "TimescaleDB", "Prometheus", "VictoriaMetrics"],
            consistency=ConsistencyLevel.EVENTUAL,
            scale_model=ScaleModel.SHARDED,
            latency_p99_ms=10.0,
            write_throughput="very_high",
            query_flexibility="limited",
            strengths=["Optimized for time-ordered ingestion", "Automatic downsampling",
                       "TTL-based retention policies", "Compression (10-100x)", "Fast range queries"],
            weaknesses=["Poor for relational queries", "Limited update/delete support",
                        "Not for business entity storage"],
            ideal_for=["Infrastructure metrics (CPU, memory, disk)", "IoT sensor streams",
                       "Financial tick data", "Application performance monitoring",
                       "Prometheus-compatible alerting"],
            avoid_when=["General-purpose OLTP", "Flexible ad-hoc queries", "Non-time-ordered data"],
            managed_services=["InfluxDB Cloud", "TimescaleDB Cloud", "Amazon Timestream"]
        ),

        "bigquery": DatabaseProfile(
            name="BigQuery/Redshift", category=DataModel.WAREHOUSE,
            examples=["BigQuery", "Redshift", "Snowflake", "ClickHouse"],
            consistency=ConsistencyLevel.STRONG,
            scale_model=ScaleModel.GEO_DISTRIBUTED,
            latency_p99_ms=1000.0,
            write_throughput="medium",
            query_flexibility="sql",
            strengths=["Petabyte-scale analytics", "Columnar storage (10x compression)",
                       "Complex SQL aggregations", "Serverless (BigQuery)", "Pay-per-query"],
            weaknesses=["High latency (seconds to minutes)", "Not for OLTP",
                        "Expensive for frequent small queries", "Limited UPDATE/DELETE"],
            ideal_for=["Business intelligence dashboards", "Ad-hoc analytics",
                       "Data lake queries", "Historical trend analysis", "ML feature engineering"],
            avoid_when=["Low-latency transactional workloads", "Frequent point lookups",
                        "Real-time OLTP", "Budget-sensitive small datasets"],
            managed_services=["Google BigQuery", "Amazon Redshift", "Snowflake", "ClickHouse Cloud"]
        ),

        "pinecone": DatabaseProfile(
            name="Vector DB (Pinecone)", category=DataModel.VECTOR,
            examples=["Pinecone", "Weaviate", "Milvus", "pgvector", "Qdrant"],
            consistency=ConsistencyLevel.EVENTUAL,
            scale_model=ScaleModel.SHARDED,
            latency_p99_ms=20.0,
            write_throughput="medium",
            query_flexibility="none",
            strengths=["Approximate nearest neighbor (ANN) search", "Semantic similarity",
                       "High-dimensional vector indexing (HNSW, IVF)", "ML-native"],
            weaknesses=["Only useful with ML embeddings", "No relational queries",
                        "Relatively new — fewer operational patterns", "Can be expensive"],
            ideal_for=["RAG (retrieval-augmented generation)", "Semantic search",
                       "Image/audio similarity search", "Recommendation via embeddings",
                       "Anomaly detection"],
            avoid_when=["Exact keyword search (use Elasticsearch)", "Tabular data without ML",
                        "Simple CRUD application"],
            managed_services=["Pinecone", "Weaviate Cloud", "Zilliz (Milvus)", "pgvector on Supabase"]
        ),

        "cockroachdb": DatabaseProfile(
            name="CockroachDB/Spanner", category=DataModel.RELATIONAL,
            examples=["CockroachDB", "Google Spanner", "TiDB", "YugabyteDB"],
            consistency=ConsistencyLevel.STRONG,
            scale_model=ScaleModel.GEO_DISTRIBUTED,
            latency_p99_ms=15.0,
            write_throughput="high",
            query_flexibility="sql",
            strengths=["Distributed ACID transactions", "Horizontal sharding + SQL",
                       "Multi-region active-active", "Automatic failover",
                       "Postgres-compatible wire protocol"],
            weaknesses=["Higher latency than single-node Postgres (consensus overhead)",
                        "More complex to operate", "Higher cost than vanilla Postgres",
                        "Some Postgres features missing"],
            ideal_for=["Global financial applications", "Multi-region OLTP",
                       "When you need PostgreSQL semantics at planetary scale"],
            avoid_when=["Single-region app (Postgres is simpler/faster)", "Analytics (use warehouse)",
                        "Cost-sensitive projects"],
            managed_services=["CockroachDB Serverless", "Google Spanner", "TiDB Cloud"]
        ),
    }


# ─────────────────────────────────────────────
# DECISION ENGINE
# ─────────────────────────────────────────────

class DatabaseSelector:
    """
    Given workload requirements, score and rank database options.
    """

    def __init__(self, catalog: Dict[str, DatabaseProfile]):
        self.catalog = catalog

    def score(self, workload: WorkloadRequirements, profile: DatabaseProfile) -> Tuple[float, List[str]]:
        score  = 100.0
        notes  : List[str] = []

        # Data model match
        if profile.category == workload.data_model_needed:
            score += 30
            notes.append("+30 data model match")
        elif workload.data_model_needed == DataModel.RELATIONAL and profile.query_flexibility == "sql":
            score += 20
            notes.append("+20 SQL support")

        # Consistency
        if workload.consistency_needed == ConsistencyLevel.STRONG:
            if profile.consistency == ConsistencyLevel.STRONG:
                score += 20
                notes.append("+20 strong consistency match")
            elif profile.consistency == ConsistencyLevel.EVENTUAL:
                score -= 40
                notes.append("-40 eventual only (need strong)")

        # Latency SLA
        if profile.latency_p99_ms <= workload.latency_sla_ms:
            score += 15
            notes.append(f"+15 latency ok ({profile.latency_p99_ms}ms <= {workload.latency_sla_ms}ms)")
        elif profile.latency_p99_ms > workload.latency_sla_ms * 5:
            score -= 30
            notes.append(f"-30 too slow ({profile.latency_p99_ms}ms >> {workload.latency_sla_ms}ms)")

        # Write volume
        wv_map = {"low": 1, "medium": 2, "high": 3, "very_high": 4, "extreme": 4}
        needed = wv_map.get(workload.write_volume, 2)
        avail  = wv_map.get(profile.write_throughput, 2)
        if avail >= needed:
            score += 10
        else:
            score -= 20
            notes.append(f"-20 write throughput insufficient")

        # JOIN requirement
        if workload.joins_needed and profile.query_flexibility != "sql":
            score -= 25
            notes.append("-25 joins needed but no SQL")

        # Schema flexibility
        if workload.schema_flexible and profile.category == DataModel.DOCUMENT:
            score += 10
            notes.append("+10 document DB matches flexible schema need")

        # Geo distribution
        if workload.geo_distribution and profile.scale_model != ScaleModel.GEO_DISTRIBUTED:
            score -= 15
            notes.append("-15 geo distribution needed")

        return max(0.0, score), notes

    def recommend(self, workload: WorkloadRequirements, top_n: int = 3) -> List[Tuple[str, float, List[str]]]:
        scores = []
        for db_id, profile in self.catalog.items():
            s, notes = self.score(workload, profile)
            scores.append((db_id, s, notes))
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_database_selection():
    print("=" * 65)
    print("CHOOSING THE RIGHT DATABASE")
    print("=" * 65)

    catalog  = build_database_catalog()
    selector = DatabaseSelector(catalog)

    # ── Catalog Overview ──────────────────────
    print("\n[1] DATABASE CATALOG OVERVIEW")
    print("─" * 55)
    print(f"  {'Name':<22} {'Category':<14} {'Consistency':<12} {'P99 Latency':<14} {'Write'}")
    print(f"  {'─'*75}")
    for db_id, p in catalog.items():
        print(f"  {p.name:<22} {p.category.value:<14} {p.consistency.value:<12} "
              f"{p.latency_p99_ms:<14.1f} {p.write_throughput}")

    # ── Workload Scenarios ─────────────────────
    print("\n\n[2] WORKLOAD → DATABASE MATCHING")
    print("─" * 55)

    workloads = [
        WorkloadRequirements(
            name="E-Commerce Orders",
            data_model_needed=DataModel.RELATIONAL,
            consistency_needed=ConsistencyLevel.STRONG,
            read_pattern="point",
            write_volume="medium",
            latency_sla_ms=20.0,
            joins_needed=True,
            schema_flexible=False,
            geo_distribution=False,
            max_dataset_gb=500.0
        ),
        WorkloadRequirements(
            name="Product Search",
            data_model_needed=DataModel.SEARCH,
            consistency_needed=ConsistencyLevel.EVENTUAL,
            read_pattern="fulltext",
            write_volume="low",
            latency_sla_ms=100.0,
            joins_needed=False,
            schema_flexible=True,
            geo_distribution=False,
            max_dataset_gb=50.0
        ),
        WorkloadRequirements(
            name="Session Cache",
            data_model_needed=DataModel.IN_MEMORY,
            consistency_needed=ConsistencyLevel.EVENTUAL,
            read_pattern="point",
            write_volume="high",
            latency_sla_ms=2.0,
            joins_needed=False,
            schema_flexible=True,
            geo_distribution=False,
            max_dataset_gb=10.0
        ),
        WorkloadRequirements(
            name="IoT Sensor Metrics",
            data_model_needed=DataModel.TIME_SERIES,
            consistency_needed=ConsistencyLevel.EVENTUAL,
            read_pattern="range",
            write_volume="extreme",
            latency_sla_ms=50.0,
            joins_needed=False,
            schema_flexible=False,
            geo_distribution=False,
            max_dataset_gb=10_000.0
        ),
        WorkloadRequirements(
            name="Social Graph Recommendations",
            data_model_needed=DataModel.GRAPH,
            consistency_needed=ConsistencyLevel.STRONG,
            read_pattern="graph",
            write_volume="medium",
            latency_sla_ms=100.0,
            joins_needed=False,
            schema_flexible=False,
            geo_distribution=False,
            max_dataset_gb=200.0
        ),
        WorkloadRequirements(
            name="Analytics Dashboard",
            data_model_needed=DataModel.WAREHOUSE,
            consistency_needed=ConsistencyLevel.STRONG,
            read_pattern="analytics",
            write_volume="low",
            latency_sla_ms=5000.0,
            joins_needed=True,
            schema_flexible=False,
            geo_distribution=True,
            max_dataset_gb=100_000.0
        ),
        WorkloadRequirements(
            name="AI Semantic Search (RAG)",
            data_model_needed=DataModel.VECTOR,
            consistency_needed=ConsistencyLevel.EVENTUAL,
            read_pattern="vector",
            write_volume="medium",
            latency_sla_ms=50.0,
            joins_needed=False,
            schema_flexible=True,
            geo_distribution=False,
            max_dataset_gb=100.0
        ),
    ]

    for workload in workloads:
        print(f"\n  Workload: {workload.name}")
        print(f"    Needs: {workload.data_model_needed.value}, "
              f"consistency={workload.consistency_needed.value}, "
              f"latency<{workload.latency_sla_ms}ms, "
              f"joins={workload.joins_needed}")
        recs = selector.recommend(workload, top_n=3)
        for i, (db_id, score, notes) in enumerate(recs, 1):
            p = catalog[db_id]
            print(f"    #{i} {p.name:<22} score={score:.0f}  ({', '.join(p.examples[:2])})")

    # ── Polyglot Persistence Example ───────────
    print("\n\n[3] POLYGLOT PERSISTENCE — E-COMMERCE PLATFORM")
    print("─" * 55)
    components = [
        ("Users & Orders",        "PostgreSQL",    "ACID transactions, financial correctness"),
        ("Product Catalog",       "MongoDB",       "Flexible attributes per category"),
        ("Product Search",        "Elasticsearch", "Full-text, faceted, relevance ranking"),
        ("Sessions & Cart",       "Redis",         "Sub-ms, TTL-based expiry"),
        ("Clickstream Events",    "Cassandra",     "Write-heavy, time-ordered, append-only"),
        ("Inventory Metrics",     "InfluxDB",      "Time-series, downsampling, retention"),
        ("Recommendation Graph",  "Neo4j",         "friend-of-friend, collaborative filtering"),
        ("Analytics Dashboard",   "BigQuery",      "Petabyte SQL, columnar, pay-per-query"),
        ("Semantic Search (AI)",  "Pinecone",      "Vector embeddings for product similarity"),
        ("CDN / Static Assets",   "S3 + CloudFront","Cheap storage, global edge caching"),
    ]
    print(f"  {'Component':<26} {'Database':<18} {'Reason'}")
    print(f"  {'─'*75}")
    for comp, db, reason in components:
        print(f"  {comp:<26} {db:<18} {reason}")

    # ── Quick Reference Decision Tree ──────────
    print("\n\n[4] QUICK DECISION REFERENCE")
    print("─" * 55)
    decisions = [
        ("Need ACID + SQL JOINs?",              "→ PostgreSQL / Aurora"),
        ("Need sub-millisecond latency?",        "→ Redis / Memcached"),
        ("Need full-text search relevance?",     "→ Elasticsearch / Typesense"),
        ("Highly connected data traversal?",     "→ Neo4j / Amazon Neptune"),
        ("Metrics/monitoring time-series?",      "→ InfluxDB / TimescaleDB / Prometheus"),
        ("Massive write throughput (millions/s)?","→ Cassandra / ScyllaDB"),
        ("Petabyte analytics SQL?",              "→ BigQuery / Redshift / Snowflake"),
        ("Flexible document schema?",            "→ MongoDB / Firestore"),
        ("Semantic AI / vector search?",         "→ Pinecone / pgvector / Weaviate"),
        ("Distributed SQL globally?",            "→ CockroachDB / Google Spanner"),
        ("Everything + single solution?",        "→ There is no one-size-fits-all — use polyglot"),
    ]
    for question, answer in decisions:
        print(f"  {question:<48} {answer}")

    # ── Anti-Patterns ─────────────────────────
    print("\n\n[5] COMMON DATABASE ANTI-PATTERNS")
    print("─" * 55)
    antipatterns = [
        ("Using MongoDB for everything",
         "Developers avoid schema design. Then complex queries become unwritable."),
        ("Redis as primary database",
         "Data > RAM → eviction → data loss. Redis is a cache, not a source of truth."),
        ("PostgreSQL for 1B+ time-series rows",
         "Table bloat, slow vacuum, poor compression. Use InfluxDB/TimescaleDB."),
        ("Elasticsearch as sole data store",
         "Eventual consistency + data loss risk on split. Always have a primary DB."),
        ("Cassandra without query-driven design",
         "Designed tables for flexibility → hot partition, allow filtering, ALLOW FILTERING chaos."),
        ("BigQuery for OLTP",
         "Seconds latency, expensive per-query. Use Postgres for transactions."),
        ("Single DB for all microservices",
         "Schema coupling. One team's migration breaks another. Each service owns its DB."),
    ]
    for pattern, consequence in antipatterns:
        print(f"\n  ❌ {pattern}")
        print(f"     {consequence}")


if __name__ == "__main__":
    demonstrate_database_selection()
