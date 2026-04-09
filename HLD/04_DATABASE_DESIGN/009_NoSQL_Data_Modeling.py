"""
NOSQL DATA MODELING
=====================

Problem Statement:
NoSQL databases don't enforce a schema, but that doesn't mean you don't
need to design one. Poor NoSQL data modeling leads to hot partitions,
expensive cross-partition queries, and inability to scale.

Core Principle: Model your data around your access patterns, not
around your entities (opposite of relational design).

MongoDB Document Modeling:
  Embed: related data inside same document → fast reads, no joins
  Reference: store ID and lookup separately → normalized, slower reads
  Rule: embed when data is always accessed together; reference when
        data is large, frequently updated, or accessed independently

DynamoDB Single-Table Design:
  Put everything in one table; use PK/SK to model different entity types.
  Access patterns first — design table around queries, not entities.
  GSI (Global Secondary Index) for alternate access patterns.

Cassandra Data Modeling:
  Design tables per query. One query = one table (materialized query table).
  Partition key determines data distribution. Clustering key determines order.
  Avoid: cross-partition queries, unbounded collections, wide rows.

Key Rules:
  1. Know your access patterns before modeling
  2. Denormalize aggressively — reads are cheap; joins don't exist
  3. Avoid hot partitions — spread write load evenly
  4. Pre-aggregate for read performance
  5. Use TTL for time-limited data (caches, sessions, logs)
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import time
import hashlib
import uuid


class NoSQLType(Enum):
    DOCUMENT     = "document"    # MongoDB
    KEY_VALUE    = "key_value"   # DynamoDB / Redis
    COLUMN_FAMILY= "column_family"  # Cassandra


# ─────────────────────────────────────────────
# MONGODB DOCUMENT MODELING
# ─────────────────────────────────────────────

class MongoDocumentModeling:
    """
    Shows embed vs reference trade-offs in document databases.
    """

    @staticmethod
    def embedded_approach() -> Dict:
        """User with embedded address and last 3 orders — always read together."""
        return {
            "_id": "user_001",
            "email": "alice@example.com",
            "name": "Alice Smith",
            "address": {                   # embed — always read with user
                "street": "123 Main St",
                "city": "New York",
                "zip": "10001"
            },
            "recent_orders": [             # embed last N orders
                {"order_id": "o1", "total": 59.99, "status": "shipped"},
                {"order_id": "o2", "total": 19.99, "status": "delivered"},
            ],
            "preferences": {               # embed — always read with user
                "newsletter": True,
                "theme": "dark"
            }
        }

    @staticmethod
    def referenced_approach() -> Tuple[Dict, Dict]:
        """User with reference to orders — orders can grow unboundedly."""
        user = {
            "_id": "user_001",
            "email": "alice@example.com",
            "name": "Alice Smith",
            # order_ids NOT embedded — orders can be thousands
            "order_count": 142,   # pre-aggregated counter
        }
        order = {
            "_id": "o1",
            "user_id": "user_001",   # reference
            "items": [{"product_id": "p1", "qty": 2, "price": 29.99}],
            "total": 59.99,
            "status": "shipped",
            "created_at": "2024-01-15"
        }
        return user, order

    @staticmethod
    def anti_patterns() -> List[Tuple[str, str, str]]:
        return [
            ("Unbounded arrays",
             "user.orders = [o1, o2, ... o10000]",
             "Store orders in separate collection, reference by user_id"),
            ("Deeply nested",
             "order.items[0].product.category.parent.name",
             "Flatten: store category_name directly, or separate collection"),
            ("Duplicate huge blobs",
             "Embed full product object in each order item",
             "Embed product snapshot (name, price) + product_id reference"),
            ("Missing indexes",
             "No index on user_id in orders collection",
             "Create index on all query fields: db.orders.createIndex({user_id: 1})"),
        ]


# ─────────────────────────────────────────────
# DYNAMODB SINGLE-TABLE DESIGN
# ─────────────────────────────────────────────

class DynamoDBSingleTable:
    """
    Single-table design: one DynamoDB table holds all entity types.
    PK (partition key) and SK (sort key) encode entity type and relationships.

    Pattern: PK = entity type + ID, SK = type#data or METADATA
    """

    def __init__(self):
        self._table    : Dict[Tuple[str, str], Dict] = {}
        self._gsi1     : Dict[Tuple[str, str], Dict] = {}   # GSI1PK → items
        self.reads     = 0
        self.writes    = 0

    def put(self, pk: str, sk: str, item: Dict,
            gsi1_pk: str = None, gsi1_sk: str = None):
        self._table[(pk, sk)] = {**item, "PK": pk, "SK": sk}
        if gsi1_pk:
            self._gsi1[(gsi1_pk, gsi1_sk or "")] = {**item, "GSI1PK": gsi1_pk}
        self.writes += 1

    def get_item(self, pk: str, sk: str) -> Optional[Dict]:
        self.reads += 1
        return self._table.get((pk, sk))

    def query(self, pk: str, sk_prefix: str = "") -> List[Dict]:
        """Query all items with given PK, optionally filter by SK prefix."""
        self.reads += 1
        result = []
        for (p, s), item in self._table.items():
            if p == pk and s.startswith(sk_prefix):
                result.append(item)
        return sorted(result, key=lambda x: x.get("SK", ""))

    def query_gsi1(self, gsi1_pk: str) -> List[Dict]:
        """Query via GSI — alternate access pattern."""
        self.reads += 1
        return [item for (gpk, _), item in self._gsi1.items() if gpk == gsi1_pk]

    def load_ecommerce_data(self):
        """
        Models: Users, Orders, OrderItems in one table.
        Access patterns:
          - Get user by ID
          - Get all orders for user
          - Get order detail with items
          - Get all orders by status (GSI)
        """
        # User entity
        self.put("USER#alice", "METADATA",
                  {"name": "Alice", "email": "alice@ex.com", "status": "active"})

        # Orders for user (SK sorts chronologically)
        self.put("USER#alice", "ORDER#2024-01-15#o001",
                  {"order_id": "o001", "total": 59.99, "status": "shipped"},
                  gsi1_pk="STATUS#shipped", gsi1_sk="2024-01-15#o001")

        self.put("USER#alice", "ORDER#2024-01-20#o002",
                  {"order_id": "o002", "total": 19.99, "status": "delivered"},
                  gsi1_pk="STATUS#delivered", gsi1_sk="2024-01-20#o002")

        # Order items (same PK as order, different SK)
        self.put("ORDER#o001", "METADATA",
                  {"user_id": "alice", "total": 59.99, "status": "shipped"})
        self.put("ORDER#o001", "ITEM#1",
                  {"product": "Laptop Case", "qty": 1, "price": 39.99})
        self.put("ORDER#o001", "ITEM#2",
                  {"product": "USB Hub", "qty": 2, "price": 19.99})

    def show_access_patterns(self):
        print("  Access patterns:")
        # 1. Get user profile
        user = self.get_item("USER#alice", "METADATA")
        print(f"  1. GetUser(alice): {user.get('name')}, {user.get('email')}")

        # 2. Get all user's orders
        orders = self.query("USER#alice", "ORDER#")
        print(f"  2. GetUserOrders(alice): {len(orders)} orders")
        for o in orders:
            print(f"    SK={o['SK']}  status={o.get('status')}  total={o.get('total')}")

        # 3. Get order with all items (one query)
        items = self.query("ORDER#o001")
        print(f"  3. GetOrderWithItems(o001): {len(items)} records (metadata + items)")
        for item in items:
            print(f"    SK={item['SK']}")

        # 4. Query by status via GSI
        shipped = self.query_gsi1("STATUS#shipped")
        print(f"  4. GetOrdersByStatus(shipped) via GSI1: {len(shipped)} orders")


# ─────────────────────────────────────────────
# CASSANDRA DATA MODELING
# ─────────────────────────────────────────────

class CassandraTable:
    """
    Cassandra: design one table per query pattern.
    Partition key → data distribution.
    Clustering key → sort order within partition.
    """

    def __init__(self, table_name: str, partition_key: str,
                 clustering_keys: List[str] = None):
        self.table_name      = table_name
        self.partition_key   = partition_key
        self.clustering_keys = clustering_keys or []
        self._partitions     : Dict[str, List[Dict]] = {}

    def insert(self, row: Dict):
        pk = str(row[self.partition_key])
        self._partitions.setdefault(pk, []).append(row)

    def select(self, partition_val: str, limit: int = 100,
               reverse: bool = False) -> List[Dict]:
        rows = self._partitions.get(str(partition_val), [])
        if self.clustering_keys:
            ck = self.clustering_keys[0]
            rows = sorted(rows, key=lambda r: r.get(ck, ""), reverse=reverse)
        return rows[:limit]

    def partition_sizes(self) -> Dict[str, int]:
        return {pk: len(rows) for pk, rows in self._partitions.items()}

    def show_schema(self):
        ck_str = ", ".join(self.clustering_keys)
        print(f"  CREATE TABLE {self.table_name} (")
        print(f"    PRIMARY KEY ({self.partition_key}, {ck_str})")
        print(f"  ) WITH CLUSTERING ORDER BY ({self.clustering_keys[0] if self.clustering_keys else ''} DESC);")


# ─────────────────────────────────────────────
# HOT PARTITION DETECTOR
# ─────────────────────────────────────────────

class HotPartitionDetector:
    """Detects hot partitions in time-series data."""

    @staticmethod
    def bad_partition_key_example(user_id: str) -> str:
        """Bad: partition by date → all writes today go to one partition."""
        import datetime
        return datetime.date.today().isoformat()   # HOT! All writes to today's partition

    @staticmethod
    def good_partition_key_example(user_id: str) -> str:
        """Good: partition by user_id → distributed writes."""
        return f"user_{user_id}"

    @staticmethod
    def bucket_partition_key(user_id: str, n_buckets: int = 10) -> str:
        """Better: add shard suffix to distribute popular user_ids."""
        bucket = int(hashlib.md5(user_id.encode()).hexdigest(), 16) % n_buckets
        return f"user_{user_id}_bucket_{bucket}"


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_nosql_modeling():
    print("=" * 65)
    print("NOSQL DATA MODELING")
    print("=" * 65)

    # ── MongoDB Embed vs Reference ────────────
    print("\n[1] MONGODB — EMBED vs REFERENCE")
    print("─" * 55)
    mongo = MongoDocumentModeling()
    print("  Embedded document (user + address + preferences):")
    import json
    doc = mongo.embedded_approach()
    print(f"  {json.dumps(doc, indent=4)[:400]}...")

    print("\n  Referenced approach (user + separate orders):")
    user, order = mongo.referenced_approach()
    print(f"  User: {json.dumps(user)}")
    print(f"  Order: {json.dumps(order)}")

    print("\n  Embed vs Reference rules:")
    rules = [
        ("EMBED when",   "Data always queried with parent document"),
        ("EMBED when",   "Data has 1:1 or 1:few relationship"),
        ("EMBED when",   "Data rarely updated independently"),
        ("REFERENCE when","Data grows unboundedly (user.orders → millions)"),
        ("REFERENCE when","Same data shared by many documents (product info)"),
        ("REFERENCE when","Data updated frequently (avoid rewriting huge docs)"),
    ]
    for action, condition in rules:
        print(f"  {'✅' if 'EMBED' in action else '📎'} {action:<20} {condition}")

    print("\n  Anti-patterns:")
    for bad, example, fix in mongo.anti_patterns():
        print(f"  ❌ {bad}")
        print(f"     Bad: {example}")
        print(f"     Fix: {fix}")

    # ── DynamoDB Single-Table ─────────────────
    print("\n\n[2] DYNAMODB SINGLE-TABLE DESIGN")
    print("─" * 55)
    ddb = DynamoDBSingleTable()
    ddb.load_ecommerce_data()

    print("  Table structure (PK + SK encode entity type):")
    print(f"  {'PK':<20} {'SK':<35} {'Attributes'}")
    print(f"  {'─'*75}")
    for (pk, sk), item in sorted(ddb._table.items()):
        attrs = {k: v for k, v in item.items() if k not in ("PK", "SK")}
        print(f"  {pk:<20} {sk:<35} {str(attrs)[:40]}")

    print()
    ddb.show_access_patterns()
    print(f"\n  Reads: {ddb.reads}  Writes: {ddb.writes}")

    # ── Cassandra ────────────────────────────
    print("\n\n[3] CASSANDRA — ONE TABLE PER QUERY")
    print("─" * 55)
    print("  Access patterns for a messaging app:")

    # Table 1: messages by user (inbox)
    messages_by_user = CassandraTable("messages_by_user", "user_id", ["created_at"])
    messages_by_user.show_schema()

    for i in range(5):
        messages_by_user.insert({
            "user_id": "alice", "message_id": f"m{i}",
            "content": f"Message {i}", "sender": "bob",
            "created_at": f"2024-01-{20-i:02d}T10:00:00"
        })

    msgs = messages_by_user.select("alice", limit=3)
    print(f"\n  GetMessagesForUser(alice, limit=3):")
    for m in msgs:
        print(f"    {m['created_at']}: {m['content']}")

    # Table 2: messages by conversation
    messages_by_conv = CassandraTable("messages_by_conversation", "conversation_id",
                                       ["created_at", "message_id"])
    messages_by_conv.show_schema()

    print("\n  Two tables for different access patterns:")
    print("  messages_by_user: partition=user_id → GetInbox(user)")
    print("  messages_by_conv: partition=conversation_id → GetThread(conv)")
    print("  (Same data duplicated — Cassandra trade-off)")

    # ── Hot Partition ─────────────────────────
    print("\n\n[4] HOT PARTITION PROBLEM AND SOLUTION")
    print("─" * 55)
    hpd = HotPartitionDetector()
    users = ["alice", "bob", "carol", "dave"]
    print("  Bad: partition by date → ALL writes go to today")
    for uid in users:
        pk = hpd.bad_partition_key_example(uid)
        print(f"    user={uid} → partition={pk}  ← same for everyone!")

    print("\n  Good: partition by user_id → distributed")
    for uid in users:
        pk = hpd.good_partition_key_example(uid)
        print(f"    user={uid} → partition={pk}")

    print("\n  Better: bucketed partition key for very popular users")
    for uid in ["viral_user", "viral_user", "alice", "bob"]:
        pk = hpd.bucket_partition_key(uid, n_buckets=10)
        print(f"    user={uid} → partition={pk}")

    # ── NoSQL Decision Guide ──────────────────
    print("\n\n[5] NOSQL TYPE SELECTION GUIDE")
    print("─" * 55)
    guide = [
        ("User profiles, blog posts", "MongoDB", "Flexible schema, nested docs"),
        ("Session store, cache",       "Redis",   "Low latency, TTL, data structures"),
        ("E-commerce catalog",         "MongoDB", "Variable attributes per product"),
        ("Time-series metrics",        "InfluxDB/Cassandra","High write throughput"),
        ("Social graph",               "Neo4j",   "Traversal queries"),
        ("Full-text search",           "Elasticsearch","Inverted index, relevance"),
        ("High-scale K-V",             "DynamoDB","Serverless, single-digit ms"),
        ("IoT sensor data",            "Cassandra","Append-only, wide column"),
    ]
    print(f"  {'Use Case':<30} {'DB':<18} {'Why'}")
    print(f"  {'─'*70}")
    for use_case, db, why in guide:
        print(f"  {use_case:<30} {db:<18} {why}")


if __name__ == "__main__":
    demonstrate_nosql_modeling()
