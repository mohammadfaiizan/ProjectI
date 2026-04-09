"""
RELATIONAL VS NOSQL DATABASES
================================

Problem Statement:
Choosing the wrong database type leads to poor query performance, complex
data migrations, or inability to scale. Engineers must understand the
fundamental trade-offs between relational (SQL) and NoSQL databases.

Relational Databases (RDBMS):
  - Tables with fixed schema (rows and columns)
  - ACID transactions (Atomicity, Consistency, Isolation, Durability)
  - SQL: powerful joins, aggregations, filtering
  - Vertical scaling primarily; horizontal scaling via read replicas
  - Strong consistency; supports complex business logic
  - Examples: PostgreSQL, MySQL, Oracle, SQL Server

NoSQL Databases:
  Document  : JSON-like documents (MongoDB, CouchDB)
  Key-Value : Simple K→V lookups (Redis, DynamoDB)
  Column-Family: Wide columns, time-series (Cassandra, HBase)
  Graph     : Nodes + edges (Neo4j, Amazon Neptune)
  Search    : Full-text search (Elasticsearch, OpenSearch)

Decision Factors:
  → Schema flexibility needed? → NoSQL (document)
  → Complex joins? → SQL
  → Massive write scale (>100K writes/s)? → NoSQL
  → ACID transactions across tables? → SQL
  → Graph relationships? → Graph DB
  → Low-latency cache? → Key-Value
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import time
import json


class DBType(Enum):
    RELATIONAL   = "relational"
    DOCUMENT     = "document"
    KEY_VALUE    = "key_value"
    COLUMN_FAMILY= "column_family"
    GRAPH        = "graph"
    SEARCH       = "search"


class ConsistencyLevel(Enum):
    STRONG       = "strong"
    EVENTUAL     = "eventual"
    CAUSAL       = "causal"
    READ_COMMITTED = "read_committed"


@dataclass
class TableSchema:
    table_name : str
    columns    : Dict[str, str]   # name → type
    primary_key: List[str]
    indexes    : List[str] = field(default_factory=list)
    foreign_keys: List[Tuple[str, str]] = field(default_factory=list)  # col → ref_table

    def ddl(self) -> str:
        lines = [f"CREATE TABLE {self.table_name} ("]
        for col, dtype in self.columns.items():
            pk_note = " PRIMARY KEY" if col in self.primary_key and len(self.primary_key) == 1 else ""
            lines.append(f"    {col} {dtype}{pk_note},")
        if len(self.primary_key) > 1:
            lines.append(f"    PRIMARY KEY ({', '.join(self.primary_key)}),")
        for col, ref in self.foreign_keys:
            lines.append(f"    FOREIGN KEY ({col}) REFERENCES {ref},")
        # Remove trailing comma from last line
        lines[-1] = lines[-1].rstrip(",")
        lines.append(");")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# RELATIONAL DB SIMULATION
# ─────────────────────────────────────────────

class RelationalDB:
    """Simple SQL-like database simulation."""

    def __init__(self, name: str):
        self.name   = name
        self._tables: Dict[str, List[Dict]] = {}
        self._schemas: Dict[str, TableSchema] = {}
        self.query_count = 0

    def create_table(self, schema: TableSchema):
        self._tables[schema.table_name]  = []
        self._schemas[schema.table_name] = schema

    def insert(self, table: str, row: Dict):
        schema = self._schemas.get(table)
        if schema:
            for col in schema.columns:
                if col not in row and col not in schema.primary_key:
                    row[col] = None
        self._tables[table].append(row)

    def select(self, table: str, where: Dict = None,
               columns: List[str] = None) -> List[Dict]:
        self.query_count += 1
        rows = self._tables.get(table, [])
        if where:
            rows = [r for r in rows if all(r.get(k) == v for k, v in where.items())]
        if columns:
            rows = [{c: r.get(c) for c in columns} for r in rows]
        return rows

    def join(self, table_a: str, table_b: str,
             join_key_a: str, join_key_b: str) -> List[Dict]:
        self.query_count += 1
        result = []
        for row_a in self._tables.get(table_a, []):
            for row_b in self._tables.get(table_b, []):
                if row_a.get(join_key_a) == row_b.get(join_key_b):
                    merged = {**{f"a.{k}": v for k, v in row_a.items()},
                               **{f"b.{k}": v for k, v in row_b.items()}}
                    result.append(merged)
        return result

    def aggregate(self, table: str, group_by: str,
                   agg_col: str, func: str = "COUNT") -> Dict[str, Any]:
        self.query_count += 1
        groups: Dict[str, List] = {}
        for row in self._tables.get(table, []):
            key = row.get(group_by)
            groups.setdefault(key, []).append(row.get(agg_col, 0))
        if func == "COUNT":
            return {k: len(v) for k, v in groups.items()}
        if func == "SUM":
            return {k: sum(v) for k, v in groups.items()}
        if func == "AVG":
            return {k: sum(v)/len(v) for k, v in groups.items()}
        return {}

    def show_tables(self):
        for tname, schema in self._schemas.items():
            print(f"\n  Table: {tname}  ({len(self._tables[tname])} rows)")
            print(f"  {schema.ddl()}")


# ─────────────────────────────────────────────
# DOCUMENT DB SIMULATION
# ─────────────────────────────────────────────

class DocumentDB:
    """MongoDB-like document store — flexible schema."""

    def __init__(self, name: str):
        self.name        = name
        self._collections: Dict[str, List[Dict]] = {}
        self.query_count = 0

    def insert(self, collection: str, doc: Dict):
        self._collections.setdefault(collection, []).append(doc)

    def find(self, collection: str, query: Dict = None,
             projection: List[str] = None) -> List[Dict]:
        self.query_count += 1
        docs = self._collections.get(collection, [])
        if query:
            def matches(doc: Dict, q: Dict) -> bool:
                for k, v in q.items():
                    if isinstance(v, dict):
                        op = list(v.keys())[0]
                        val = v[op]
                        doc_val = doc.get(k)
                        if op == "$gt"  and not (doc_val is not None and doc_val > val): return False
                        if op == "$lt"  and not (doc_val is not None and doc_val < val): return False
                        if op == "$gte" and not (doc_val is not None and doc_val >= val):return False
                        if op == "$in"  and doc_val not in val: return False
                    else:
                        if doc.get(k) != v:
                            return False
                return True
            docs = [d for d in docs if matches(d, query)]
        if projection:
            docs = [{k: d.get(k) for k in projection} for d in docs]
        return docs

    def update(self, collection: str, query: Dict, update: Dict):
        for doc in self._collections.get(collection, []):
            if all(doc.get(k) == v for k, v in query.items()):
                if "$set" in update:
                    doc.update(update["$set"])
                if "$push" in update:
                    for k, v in update["$push"].items():
                        doc.setdefault(k, []).append(v)

    def show_schema_flexibility(self):
        print("  Document DB: each document can have different fields")
        print("  (perfect for heterogeneous product catalog)")


# ─────────────────────────────────────────────
# KEY-VALUE STORE SIMULATION
# ─────────────────────────────────────────────

class KeyValueStore:
    """Redis/DynamoDB-like key-value store."""

    def __init__(self, name: str):
        self.name   = name
        self._store : Dict[str, Any] = {}
        self._ttls  : Dict[str, float] = {}
        self.hits   = 0
        self.misses = 0

    def set(self, key: str, value: Any, ttl_s: int = None):
        self._store[key] = value
        if ttl_s:
            self._ttls[key] = time.time() + ttl_s

    def get(self, key: str) -> Optional[Any]:
        if key in self._ttls and time.time() > self._ttls[key]:
            del self._store[key]
            del self._ttls[key]
            self.misses += 1
            return None
        if key in self._store:
            self.hits += 1
            return self._store[key]
        self.misses += 1
        return None

    def delete(self, key: str):
        self._store.pop(key, None)
        self._ttls.pop(key, None)

    def increment(self, key: str, by: int = 1) -> int:
        self._store[key] = self._store.get(key, 0) + by
        return self._store[key]


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_sql_vs_nosql():
    print("=" * 65)
    print("RELATIONAL VS NOSQL DATABASES")
    print("=" * 65)

    # ── Relational DB ──────────────────────────
    print("\n[1] RELATIONAL DATABASE — E-COMMERCE SCHEMA")
    print("─" * 55)
    rdb = RelationalDB("ecommerce_db")

    users_schema = TableSchema(
        "users",
        {"user_id": "UUID", "name": "VARCHAR(100)", "email": "VARCHAR(255)", "created_at": "TIMESTAMP"},
        ["user_id"]
    )
    orders_schema = TableSchema(
        "orders",
        {"order_id": "UUID", "user_id": "UUID", "total_usd": "DECIMAL(10,2)", "status": "VARCHAR(20)", "created_at": "TIMESTAMP"},
        ["order_id"],
        foreign_keys=[("user_id", "users(user_id)")]
    )
    rdb.create_table(users_schema)
    rdb.create_table(orders_schema)
    rdb.show_tables()

    # Insert data
    for i in range(1, 5):
        rdb.insert("users",  {"user_id": f"u{i}", "name": f"User{i}", "email": f"u{i}@ex.com", "created_at": "2024-01-01"})
    for i in range(1, 7):
        rdb.insert("orders", {"order_id": f"o{i}", "user_id": f"u{(i%4)+1}", "total_usd": i*10.0, "status": "completed", "created_at": "2024-02-01"})

    # Queries
    print(f"\n  SELECT * FROM orders WHERE user_id = 'u1':")
    results = rdb.select("orders", where={"user_id": "u1"})
    for r in results:
        print(f"    {r}")

    print(f"\n  JOIN users ↔ orders:")
    joined = rdb.join("orders", "users", "user_id", "user_id")
    for r in joined[:3]:
        print(f"    order={r.get('a.order_id')}  user={r.get('b.name')}  total={r.get('a.total_usd')}")

    print(f"\n  COUNT orders GROUP BY status:")
    agg = rdb.aggregate("orders", "status", "order_id", "COUNT")
    print(f"    {agg}")

    # ── Document DB ───────────────────────────
    print("\n\n[2] DOCUMENT DB — PRODUCT CATALOG (FLEXIBLE SCHEMA)")
    print("─" * 55)
    doc_db = DocumentDB("catalog_db")
    doc_db.show_schema_flexibility()

    products = [
        {"_id": "p1", "name": "Laptop", "category": "electronics",
         "price": 999.99, "specs": {"cpu": "i7", "ram": "16GB"}, "tags": ["computing"]},
        {"_id": "p2", "name": "T-Shirt", "category": "clothing",
         "price": 29.99, "size": ["S","M","L"], "color": "blue"},
        {"_id": "p3", "name": "Book", "category": "books",
         "price": 14.99, "author": "Knuth", "isbn": "978-0-00"},
    ]
    for p in products:
        doc_db.insert("products", p)

    # Flexible queries
    cheap = doc_db.find("products", {"price": {"$lt": 30}}, ["name", "price"])
    print(f"\n  Products under $30: {cheap}")

    electronics = doc_db.find("products", {"category": "electronics"})
    print(f"  Electronics: {[p['name'] for p in electronics]}")

    # Nested update — add a review (no schema migration needed)
    doc_db.update("products", {"_id": "p1"}, {"$push": {"reviews": {"user": "alice", "rating": 5}}})
    updated = doc_db.find("products", {"_id": "p1"}, ["name", "reviews"])
    print(f"  After adding review: {updated}")

    # ── Key-Value Store ───────────────────────
    print("\n\n[3] KEY-VALUE STORE — SESSION CACHE")
    print("─" * 55)
    kv = KeyValueStore("session_cache")
    kv.set("session:abc123", {"user_id": "u1", "cart": [1, 2]}, ttl_s=3600)
    kv.set("rate:1.2.3.4",   42, ttl_s=60)
    kv.set("counter:visits", 0)

    print(f"  GET session:abc123 → {kv.get('session:abc123')}")
    kv.increment("counter:visits")
    kv.increment("counter:visits")
    print(f"  INCR counter:visits → {kv.get('counter:visits')}")
    print(f"  GET missing_key → {kv.get('missing_key')}")
    print(f"  Cache stats: hits={kv.hits}  misses={kv.misses}")

    # ── Decision Guide ────────────────────────
    print("\n\n[4] DATABASE SELECTION GUIDE")
    print("─" * 55)
    guide = [
        ("E-commerce orders",    "PostgreSQL",  "ACID, complex joins, financial"),
        ("User sessions",        "Redis",       "Low latency K-V, TTL support"),
        ("Product catalog",      "MongoDB",     "Flexible schema, nested docs"),
        ("Social graph",         "Neo4j",       "Graph traversal, relationships"),
        ("Time-series metrics",  "InfluxDB",    "Columnar, high write throughput"),
        ("Full-text search",     "Elasticsearch","Inverted index, fuzzy search"),
        ("Real-time leaderboard","Redis Sorted Set","O(log N) rank operations"),
        ("Event store",          "Cassandra",   "Append-only, wide column"),
        ("Config/metadata",      "DynamoDB",    "Serverless K-V, global tables"),
        ("Analytics warehouse",  "Redshift/BigQuery","columnar, OLAP, huge joins"),
    ]
    print(f"  {'Use Case':<28} {'Database':<20} {'Why'}")
    print(f"  {'─'*75}")
    for use_case, db, why in guide:
        print(f"  {use_case:<28} {db:<20} {why}")

    # ── Comparison Table ──────────────────────
    print("\n\n[5] SQL vs NOSQL TRADE-OFFS")
    print("─" * 55)
    rows = [
        ("Schema",           "Fixed (DDL required)",   "Flexible/dynamic"),
        ("Query language",   "SQL — rich, standardized","Varies by DB type"),
        ("Transactions",     "Full ACID",              "Limited (some support)"),
        ("Joins",            "Native, efficient",      "Manual/denormalized"),
        ("Horizontal scale", "Hard (sharding complex)","Designed for it"),
        ("Consistency",      "Strong by default",      "Eventual by default"),
        ("Maturity",         "50+ years",              "10-15 years"),
        ("Write throughput", "Moderate",               "Very high (Cassandra)"),
        ("Read patterns",    "Any (planner optimizes)","Must know access pattern"),
    ]
    print(f"  {'Aspect':<22} {'SQL':<28} {'NoSQL'}")
    print(f"  {'─'*70}")
    for aspect, sql, nosql in rows:
        print(f"  {aspect:<22} {sql:<28} {nosql}")


if __name__ == "__main__":
    demonstrate_sql_vs_nosql()
