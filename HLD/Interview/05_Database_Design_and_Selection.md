# Database Design and Selection — Interview Q&A

> 20 questions | Easy: Q1–Q7 | Medium: Q8–Q15 | Hard: Q16–Q20

---

## EASY (Q1–Q7)

---

### Q1. How do you choose between SQL and NoSQL databases?

**Answer:**

The choice between SQL (relational) and NoSQL depends on data structure, access patterns, scale requirements, and consistency needs — not on popularity or default assumptions.

**Choose SQL when:**

| Factor | Details |
|---|---|
| Complex relationships | Multi-table joins; foreign key constraints needed |
| ACID transactions | Money transfers, inventory, bookings — no partial updates |
| Flexible queries | Ad-hoc queries with complex WHERE, GROUP BY, aggregations |
| Data integrity | Schema enforcement; constraints prevent bad data |
| Reporting/analytics | Complex SQL queries for business intelligence |
| Team familiarity | Most engineers know SQL; lower learning curve |

**Choose NoSQL when:**

| Factor | Details |
|---|---|
| Massive write throughput | Cassandra: > 1M writes/sec; SQL primary limits at ~50K writes/sec |
| Dynamic schema | Fields vary per record; schema evolves rapidly |
| Specific access patterns | Know exactly how data will be queried (design for queries) |
| Horizontal scalability | Need to shard beyond what SQL can handle cleanly |
| Semi-structured data | JSON documents with varying structure |
| Low-latency at scale | Key-value lookups at O(1) from distributed nodes |

**Decision flowchart:**

```
Is the data highly relational? (complex joins, FK constraints)
  YES → SQL (PostgreSQL, MySQL)
  
Is there a need for ACID transactions? (money, inventory, orders)
  YES → SQL
  
Is the access pattern simple and known? (always query by user_id, sensor_id)
  YES → Consider NoSQL
  
Will the write volume exceed ~50K writes/sec on a single table?
  YES → Consider NoSQL (Cassandra, DynamoDB)
  
Is the data semi-structured or schema changes frequently?
  YES → Consider MongoDB or DynamoDB
  
All else equal → Default to PostgreSQL; migrate to NoSQL when specific pain is felt
```

**The pragmatic rule:** Start with PostgreSQL for most new projects. It supports JSON fields, can handle millions of rows without sharding, and is far more operationally familiar. Migrate to NoSQL when you have a specific, validated scaling problem.

---

### Q2. What is CAP theorem, and how does it apply to real databases?

**Answer:**

CAP theorem states a distributed system can guarantee at most two of three properties during a network partition: Consistency, Availability, and Partition tolerance. Since partitions are unavoidable, the real trade-off is CP vs AP.

**Definitions:**
- **Consistency (C):** Every read returns the most recent write or an error.
- **Availability (A):** Every request receives a response (not necessarily the latest data).
- **Partition tolerance (P):** System continues operating despite network partitions.

**Real database classification:**

| Database | CAP Type | Partition Behavior | Notes |
|---|---|---|---|
| PostgreSQL | CP (when replicated) | Rejects writes to maintain consistency | Single-node: CA (no partition) |
| MySQL (with async replica) | AP | Accepts writes on primary; replicas may be stale | Eventual consistency across replicas |
| Apache Cassandra | AP | Accepts writes on all nodes; reconciles later | Tunable consistency levels |
| Apache HBase | CP | Refuses operations if region server lost without replacement | ZooKeeper coordinates |
| Amazon DynamoDB (default) | AP | Eventual consistent reads by default | Strong consistency optional |
| DynamoDB (strong consistency) | CP | Waits for quorum before responding | Higher latency |
| ZooKeeper / etcd | CP | Refuses reads/writes without quorum | Leader election, config |
| MongoDB (default) | CP | Primary only accepts writes; replica lag is normal | Can configure for AP |
| CouchDB | AP | Accepts writes on all nodes; multi-version conflict resolution | MVCC |
| Redis Cluster | AP | Accepts writes; async replication can lose data | Failover is not instant |

**Example scenarios:**

**Choosing CP — Financial system:**
```
Bank account balance:
  User deposits $1,000 from mobile app
  Simultaneous query from web app: "What is my balance?"
  
  CP behavior (preferred): Web app blocks or gets error until
  primary DB confirms write
  
  AP behavior (wrong): Web app sees old balance immediately
  → User thinks deposit failed → double deposits → major issue
  
  Use CP: PostgreSQL with synchronous replication, or strong-consistency DynamoDB
```

**Choosing AP — Social network:**
```
User posts a tweet:
  Write to primary node
  User's followers query their feed
  
  AP behavior (acceptable): Followers may see tweet after 1-2 seconds lag
  CP behavior (unnecessary): All follower queries wait for replication
  → Latency impact with no real user benefit
  
  Use AP: Cassandra with eventual consistency; acceptable 1-2s feed freshness
```

---

### Q3. When would you use Cassandra vs DynamoDB vs MongoDB vs PostgreSQL?

**Answer:**

Each database is optimized for different workloads. Understanding their strengths and limitations is essential for database selection.

**PostgreSQL:**
```
Architecture: Relational, ACID, single-primary (or Citus for sharding)
Best for:
  - Complex queries with joins, aggregations, window functions
  - ACID transactions (payment systems, orders, inventory)
  - Mixed read/write with complex filtering
  - Schema changes are common during development
  - Medium scale: up to ~10TB, ~50K writes/sec without sharding

Limitations:
  - Vertical scaling primary path; horizontal write scaling requires Citus or sharding
  - Schema migrations can be painful at scale
  - Not designed for document/JSON-first workflows (though JSONB is supported)

Examples: Financial systems, e-commerce orders, HR systems, SaaS apps
```

**MongoDB:**
```
Architecture: Document store (JSON/BSON), CP/PA configurable
Best for:
  - Semi-structured or polymorphic data (products with varying attributes)
  - Rapid schema iteration during development
  - Hierarchical data (nested documents avoid joins)
  - Content management systems, catalogs

Limitations:
  - Multi-document transactions added late (4.0+) and still limited
  - No true ACID across multiple collections
  - Can become a "schema-less mess" without discipline

Examples: Product catalogs, CMS, user-generated content, real-time analytics
```

**Cassandra:**
```
Architecture: Wide-column, AP, leaderless, linear horizontal scaling
Best for:
  - Extremely high write throughput (millions of writes/sec)
  - Time-series data (metrics, IoT, logs, events)
  - Globally distributed data with multi-region writes
  - Known access patterns (query-first data modeling)

Limitations:
  - No joins; all data must be denormalized for access pattern
  - No ACID (transactions limited to single partition)
  - Strong upfront data modeling knowledge required
  - Operational complexity (tuning, compaction, repair)

Examples: Netflix (metadata), Facebook Messenger, time-series metrics, IoT telemetry
```

**DynamoDB:**
```
Architecture: Key-value + document, managed, AP default
Best for:
  - Serverless applications (pay-per-request pricing)
  - Unpredictable traffic spikes (auto-scaling built-in)
  - Simple access patterns (get by primary key, query by partition key)
  - AWS-native applications (Lambda, ECS integration)

Limitations:
  - Very limited query flexibility (no ad-hoc queries without full scan)
  - 400 KB item size limit
  - Joins not supported; GSI/LSI are limited workarounds
  - Expensive for high, sustained read/write volume

Examples: Shopping carts, user sessions, leaderboards, serverless backends
```

**Decision table:**
| Use Case | Recommended DB |
|---|---|
| Financial transactions | PostgreSQL |
| Product catalog (varied attributes) | MongoDB or PostgreSQL (JSONB) |
| IoT sensor data (10M writes/sec) | Cassandra or DynamoDB |
| Social graph | Neo4j or PostgreSQL (recursive CTEs) |
| Real-time leaderboard | Redis (ZSet) + PostgreSQL |
| Serverless + AWS | DynamoDB |
| Complex reporting | PostgreSQL or Redshift |

---

### Q4. What is database sharding and what are the main sharding strategies?

**Answer:**

Sharding (horizontal partitioning) distributes data across multiple database nodes, where each node (shard) holds a subset of the data. This enables scaling writes and storage beyond a single server.

**Why sharding:**
```
Single PostgreSQL server limits:
  Max writes: ~50K QPS
  Max storage: ~10-100 TB (practical limit with performance)
  Memory for indexes: Single server RAM

With sharding (10 shards):
  Max writes: ~500K QPS
  Max storage: ~1 PB
  Index memory: 10× single server
```

**Sharding strategies:**

**Range-based sharding:**
```
Partition data by range of a key (e.g., user_id, date):
  Shard 0: user_id 1–1,000,000
  Shard 1: user_id 1,000,001–2,000,000
  Shard 2: user_id 2,000,001–3,000,000

Pros:
  + Simple to implement and understand
  + Range queries efficient (order_date BETWEEN Jan 1 AND Feb 1 → one shard)
  + Easy to add shards at the end of range

Cons:
  - HOT SPOT: If most active users are in range 1M-2M → Shard 1 overloaded
  - New users (highest IDs) → all go to last shard → unbalanced
```

**Hash-based sharding:**
```
Partition by hash of the key:
  shard_id = hash(user_id) % num_shards

  user_id=123 → hash=567 → 567%4 = 3 → Shard 3
  user_id=124 → hash=891 → 891%4 = 2 → Shard 2

Pros:
  + Uniform distribution (no hot spots for random keys)
  + Good for workloads with no natural range queries

Cons:
  - Range queries require scatter-gather across ALL shards
    (SELECT * WHERE created_at > X → hits all N shards)
  - Adding shards requires rehashing ALL keys (use consistent hashing)
```

**Directory-based sharding:**
```
Lookup service maps key → shard:
  [Lookup Table]
  user_id=123  → shard_id=2
  user_id=456  → shard_id=0
  user_id=789  → shard_id=3

Pros:
  + Maximum flexibility: can rebalance keys without rehashing
  + Can handle hot spots by moving specific keys to dedicated shards

Cons:
  - Lookup service is a SPOF (must be highly available and fast)
  - Extra network hop for every query
  - Additional operational complexity
```

**Geographic sharding:**
```
Partition by user's region:
  Shard EU: all EU users
  Shard US: all US users
  Shard APAC: all APAC users

Pros:
  + Data residency compliance (GDPR: EU data stays in EU)
  + Low latency (users served from local shard)

Cons:
  - Cross-shard queries needed for global analytics
  - Uneven distribution if one region grows faster
```

---

### Q5. What is the difference between synchronous and asynchronous replication?

**Answer:**

Replication creates copies of data on multiple nodes for durability and read scaling. The replication mode determines when the primary acknowledges a write to the client.

**Synchronous replication:**
```
Client ──write──> Primary ──replicates──> Replica 1
                                       └> Replica 2
                  WAITS for ACK from replicas
                  ←── ACK to client (after all replicas confirmed)

Sequence:
  1. Client sends write
  2. Primary writes to its WAL (Write Ahead Log)
  3. Primary sends replication data to all sync replicas
  4. Each replica writes and sends ACK
  5. Primary sends ACK to client
  6. Client receives confirmation

Properties:
  RPO = 0 (zero data loss: replicas always in sync with primary)
  Latency = Primary write time + round trip to replica
  Write availability: If any sync replica is unreachable → writes stall
```

**Asynchronous replication:**
```
Client ──write──> Primary ──ACK immediately──> Client
                  │
                  └──replicates (async)──> Replica 1
                                        └> Replica 2
                  Does NOT wait for replica ACK

Sequence:
  1. Client sends write
  2. Primary writes to its WAL
  3. Primary ACKs to client immediately
  4. Primary replicates to replicas in background (replication lag: 10ms-seconds)

Properties:
  RPO = replication lag (data on replica may be seconds behind)
  Latency = Primary write time only (lower than sync)
  Write availability: Primary can write even if replicas are down
```

**Comparison:**

| Property | Synchronous | Asynchronous |
|---|---|---|
| RPO (data loss) | Zero | Seconds (replication lag) |
| Write latency | Higher (wait for replicas) | Lower (immediate ACK) |
| Write availability | Lower (replica failure stalls writes) | Higher |
| Read consistency | Strong (replicas always current) | Eventual (slight lag) |
| Use case | Financial, critical data | Analytics replicas, geo-replicas |

**PostgreSQL example:**
```sql
-- Synchronous standby list
synchronous_standby_names = 'replica1'
synchronous_commit = on   -- Wait for replica WAL flush

-- Per-transaction override:
BEGIN;
SET LOCAL synchronous_commit = off;  -- Async for this transaction
INSERT INTO analytics_events ...;
COMMIT;
```

---

### Q6. What is a primary-replica database setup, and when is multi-primary used?

**Answer:**

**Primary-Replica (Leader-Follower):**
```
         Writes only
Client ──────────────> [Primary/Master]
                              │
                    ┌─────────┴──────────┐
                    ▼                    ▼
             [Replica 1]          [Replica 2]
         (Read scaling)        (Failover standby)
              ↑
         Reads only

Rules:
  ALL writes → Primary only
  Reads → Replicas (for scaling) or Primary (for strong consistency)
  Replication → Primary to Replicas (unidirectional)
```

**Primary-Replica use cases:**
- Scale reads with read replicas (10 replicas = 10× read capacity).
- Analytics queries run on replica to avoid impacting production.
- Disaster recovery: promote replica to primary on failure.

**Multi-Primary (Active-Active):**
```
Client A → [Primary 1]    [Primary 2] ← Client B
                  ↕ bidirectional replication ↕
Both primaries accept writes
Both replicate to each other

Use cases:
  1. Multi-region active-active: Primary in us-east + Primary in eu-west
     Users routed to nearest primary for low write latency
     
  2. High write availability: If one primary fails, other still accepts writes
```

**Multi-primary trade-offs:**

| Property | Primary-Replica | Multi-Primary |
|---|---|---|
| Write availability | Single SPOF (primary) | Active-active, no write SPOF |
| Complexity | Simple, well-understood | Complex conflict resolution |
| Conflict potential | None (one writer) | High (concurrent writes to same row) |
| Consistency | Strong (all writes to one node) | Complex (eventual, or sync with high latency) |
| Use case | Most OLTP applications | Multi-region, active-active, high write availability |

**Conflict resolution in multi-primary:**
```
Both primaries accept write to same row concurrently:
  Primary 1: UPDATE price=100 WHERE product_id=123
  Primary 2: UPDATE price=200 WHERE product_id=123

Resolution strategies:
  Last-write-wins (LWW): Higher timestamp wins → 200 (if P2 was later)
  Application-level merge: Business logic decides (e.g., lowest price wins)
  Conflict detection + manual resolution: Error raised, human resolves
  CRDT (Conflict-free Replicated Data Types): Only for certain data types (counters, sets)
```

**PostgreSQL: BDR (Bi-Directional Replication)** and **CockroachDB** provide multi-primary with conflict resolution. Most teams avoid multi-primary unless multi-region is required.

---

### Q7. What are the common database indexing strategies?

**Answer:**

Indexes are data structures that allow the database to find rows quickly without scanning the entire table. Choosing the right index type is critical for query performance.

**B-tree Index (default):**
```
Most common; supports =, <, >, BETWEEN, LIKE 'prefix%'

Table: orders(id, user_id, created_at, status, total)
B-tree on user_id:

                [user_id=500]
               /             \
      [user_id=250]    [user_id=750]
       /      \              /    \
 [1-125]  [125-250]  [500-625] [750-1000]

Query: WHERE user_id = 123
  → B-tree traversal: O(log N) → find leaf node → row pointer → row data
  → Without index: O(N) full table scan
```

**Composite Index:**
```
Index on multiple columns:
  CREATE INDEX idx_user_status ON orders(user_id, status, created_at);

Leftmost prefix rule:
  Useful for: WHERE user_id = X
              WHERE user_id = X AND status = Y
              WHERE user_id = X AND status = Y AND created_at > Z
  
  NOT useful for: WHERE status = Y (user_id not first)
                  WHERE created_at > Z (skips user_id and status)

Column order matters:
  High cardinality first: user_id (millions of values) before status (5 values)
  Most frequently filtered first
```

**Covering Index:**
```
Index includes ALL columns needed by a query → no need to fetch actual row

CREATE INDEX idx_covering ON orders(user_id, status) INCLUDE (total, created_at);

Query: SELECT total, created_at FROM orders WHERE user_id=123 AND status='shipped';
  → ALL needed columns are in the index → index-only scan → no table row fetch
  → Extremely fast: avoids expensive random I/O to table pages
```

**Partial Index:**
```
Index only a subset of rows matching a condition

CREATE INDEX idx_active_users ON users(email) WHERE active = true;

Use: If 95% of users are inactive, index only active users
  → Index is 20× smaller → fits in memory → faster queries
  
Query: WHERE email = 'alice@example.com' AND active = true
  → Uses partial index → fast

Query: WHERE email = 'alice@example.com' AND active = false
  → Cannot use partial index → falls back to sequential scan
  (but inactive user queries are rare, so this is acceptable)
```

**Full-text Index:**
```
CREATE INDEX idx_text ON articles USING gin(to_tsvector('english', content));

Query: SELECT * FROM articles WHERE to_tsvector('english', content) @@ 'database';
  → Full-text search with stemming, stop words, ranking
  → For complex search: use Elasticsearch instead
```

**Summary table:**
| Index Type | Best For | Limitations |
|---|---|---|
| B-tree | Equality, range queries | Not for full-text or JSON |
| Hash | Equality only (=) | Not for range queries |
| Composite | Multi-column filtering | Leftmost prefix rule |
| Covering | Eliminate row fetches | Larger index storage |
| Partial | Sparse high-value subsets | Only usable when condition matches |
| GIN | Arrays, JSONB, full-text | Slower writes |
| BRIN | Large sequential data (logs, time-series) | Only useful for correlated data |

---

## MEDIUM (Q8–Q15)

---

### Q8. What is the N+1 query problem and how do you solve it?

**Answer:**

The N+1 problem occurs when an application executes 1 query to fetch a list of records, then N additional queries to fetch related data for each record individually. It is one of the most common and costly database performance anti-patterns.

**Example:**
```python
# N+1 PROBLEM:
users = db.query("SELECT * FROM users LIMIT 100")   # 1 query → 100 users

for user in users:
    orders = db.query("SELECT * FROM orders WHERE user_id = ?", user.id)
    # 100 queries (one per user!)

# Total: 1 + 100 = 101 queries for 100 users
# At 10ms per query: 1,010ms total vs. 10ms with JOIN

# N+1 in ORMs — the classic trap:
users = User.objects.all()[:100]     # 1 query
for user in users:
    print(user.orders.all())          # 1 query per user = 100 queries
```

**Solution 1: JOIN query:**
```sql
-- 1 query with JOIN instead of N+1
SELECT u.id, u.name, o.id as order_id, o.total
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE u.id IN (SELECT id FROM users LIMIT 100);

-- Result: All data in a single round trip
```

**Solution 2: ORM eager loading:**
```python
# Django ORM:
users = User.objects.prefetch_related('orders').all()[:100]
# 2 queries: 1 for users + 1 for all orders in bulk (WHERE user_id IN (1,2,...,100))

# SQLAlchemy:
users = session.query(User).options(joinedload(User.orders)).limit(100).all()

# Rails:
users = User.includes(:orders).limit(100)
```

**Solution 3: DataLoader pattern (GraphQL):**
```python
# DataLoader batches all N individual loads into ONE query:

class OrderLoader(DataLoader):
    def batch_load_fn(self, user_ids):
        # Called ONCE with all user_ids collected during execution
        orders = db.query("SELECT * FROM orders WHERE user_id IN (?)", user_ids)
        # Group by user_id and return in same order
        return [orders_by_user.get(uid, []) for uid in user_ids]

# GraphQL resolvers call loader.load(user.id) per user
# DataLoader batches all loads → single query
```

**Solution 4: Application-side JOIN:**
```python
# Fetch in two queries, join in application
user_ids = [u.id for u in users]
orders = db.query("SELECT * FROM orders WHERE user_id IN (?)", user_ids)  # 1 query

# Build lookup map in Python
orders_by_user = defaultdict(list)
for order in orders:
    orders_by_user[order.user_id].append(order)

for user in users:
    user.orders = orders_by_user[user.id]
# Total: 2 queries regardless of N
```

**Detection:**
```
Enable query logging:
  Django:  settings.LOGGING + django.db.backends at DEBUG level
  Rails:   Bullet gem (detects N+1 automatically)
  Node:    sequelize-log-query-count
  
Monitoring: Alert if any API endpoint makes > 20 DB queries per request
```

---

### Q9. Compare cursor-based vs offset pagination.

**Answer:**

Pagination controls how large datasets are split into pages for API responses. The two main approaches have very different performance characteristics at scale.

**Offset pagination:**
```sql
-- Page 1: OFFSET 0
SELECT * FROM posts ORDER BY created_at DESC LIMIT 20 OFFSET 0;

-- Page 2: OFFSET 20  
SELECT * FROM posts ORDER BY created_at DESC LIMIT 20 OFFSET 20;

-- Page 100: OFFSET 1980
SELECT * FROM posts ORDER BY created_at DESC LIMIT 20 OFFSET 1980;
```

**Problem with offset at scale:**
```
OFFSET 1980 → DB must:
  1. Sort ALL posts by created_at (or use index)
  2. Count through 1,980 rows and discard them
  3. Return rows 1,981–2,000

For OFFSET 1,000,000:
  DB traverses 1M rows and discards them = O(OFFSET) → very slow!
  
Practical limit: OFFSET > 10,000 → queries become noticeably slow
```

**Additional problem — data drift:**
```
User is on page 2 (OFFSET 20)
Between page 1 and page 2 requests, 3 new posts are inserted at the top

Now OFFSET 20 returns rows that the user ALREADY SAW on page 1
→ Duplicate items in pagination
```

**Cursor-based (keyset) pagination:**
```sql
-- Page 1: No cursor
SELECT * FROM posts ORDER BY created_at DESC, id DESC LIMIT 20;
-- Returns last item: created_at='2024-01-15 10:30:00', id=12345

-- Page 2: Use cursor (last seen values)
SELECT * FROM posts
WHERE (created_at, id) < ('2024-01-15 10:30:00', 12345)
ORDER BY created_at DESC, id DESC
LIMIT 20;

-- Page 3: Continue from new last item
```

**How it works:**
```
Cursor = serialized last-seen position (opaque to client)
  cursor = base64_encode("2024-01-15T10:30:00_12345")

API:
  GET /posts?cursor=<opaque_token>&limit=20
  Response:
  {
    "items": [...],
    "next_cursor": "<next_opaque_token>",  // null if last page
    "has_more": true
  }
```

**Performance:**
```
Keyset pagination uses an index efficiently:
  WHERE (created_at, id) < (cursor_timestamp, cursor_id)
  → Index scan starting from cursor position
  → O(log N + page_size) regardless of page depth
  → Page 1,000,000 as fast as page 1!
```

**Comparison:**

| Property | Offset | Cursor |
|---|---|---|
| Performance at large offsets | O(OFFSET) — very slow | O(log N) — always fast |
| Data drift (inserts during pagination) | Shows duplicates/gaps | No drift (cursor is position) |
| Random access (jump to page N) | Yes (OFFSET = N × limit) | No (must follow cursor chain) |
| Implementation complexity | Simple | Moderate |
| Sort flexibility | Any ORDER BY | Must sort by unique, indexed column(s) |
| Use case | Admin UIs with page jumps | Infinite scroll, API pagination |

**Recommendation:**
- Use cursor pagination for all high-traffic APIs and infinite scroll.
- Use offset only for admin pages where random page access is needed and dataset is small.

---

### Q10. What is CQRS and when should you use it?

**Answer:**

CQRS (Command Query Responsibility Segregation) separates the write model (commands that change state) from the read model (queries that read state) into distinct code paths, services, or databases.

**Traditional model (combined):**
```
Client → [API] → [Single Service] → [Single Database]
                    ↑
              Handles both writes and reads
              Same DB schema for both
              One service, one data model
```

**CQRS model (separated):**
```
WRITE PATH (Commands):
  Client → [Command API] → [Write Service] → [Write DB (PostgreSQL)]
                                                    │
                                              Emits domain events
                                                    │
                                            [Event Bus (Kafka)]
                                                    │
                                       ┌────────────┘
                                       ▼
READ PATH (Queries):           [Read Model Builder]
  Client → [Query API] →           │
  [Read Service] → [Read DB] ◄─────┘
                 (Elasticsearch,   Updates read model from events
                  Redis, Cassandra)
```

**Benefits:**

| Benefit | Details |
|---|---|
| Independent scaling | Read service scales independently from write service |
| Optimized data models | Write DB normalized; Read DB denormalized for fast queries |
| Different DB engines | Write: ACID SQL; Read: Elasticsearch, Redis, Cassandra |
| Event sourcing friendly | Commands produce events → read models derived from events |
| Clear separation | Easier to understand and test each path separately |

**Concrete example — E-commerce orders:**

```
Write path (Command):
  POST /orders
  → Validates, writes to PostgreSQL (ACID, normalized)
  → Emits OrderPlaced event to Kafka

Read path (Query):
  GET /orders?user_id=123
  → Queries DynamoDB (order summary table, denormalized)
  → GET /products/{id}/reviews → Queries Elasticsearch
  → GET /dashboard/sales → Queries ClickHouse (analytics)

Read model builder:
  Consumes OrderPlaced from Kafka
  Updates DynamoDB order summary table
  Updates ClickHouse sales aggregates
  Rebuilding read models: just replay Kafka events
```

**When NOT to use CQRS:**
- Simple CRUD applications (adding/removing blog posts).
- Small teams without capacity to maintain two data paths.
- When read and write models have almost identical structure.
- When eventual consistency between write and read is unacceptable.

**CQRS increases complexity significantly.** Use it when write-read scaling requirements diverge or when different read queries need fundamentally different data structures.

---

### Q11. What is database denormalization and when do you use it?

**Answer:**

Normalization stores data with no redundancy by splitting it into related tables. Denormalization deliberately introduces redundancy by merging tables or adding precomputed columns to improve read performance.

**Normalized schema (3NF):**
```sql
CREATE TABLE users (
    id INT PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(200)
);

CREATE TABLE products (
    id INT PRIMARY KEY,
    name VARCHAR(200),
    price DECIMAL(10,2)
);

CREATE TABLE orders (
    id INT PRIMARY KEY,
    user_id INT REFERENCES users(id),
    created_at TIMESTAMP
);

CREATE TABLE order_items (
    order_id INT REFERENCES orders(id),
    product_id INT REFERENCES products(id),
    quantity INT,
    price DECIMAL(10,2)  -- price at time of order
);
```

**Denormalized read model:**
```sql
-- Denormalized order summary table (for fast display)
CREATE TABLE order_summaries (
    order_id INT PRIMARY KEY,
    user_id INT,
    user_name VARCHAR(100),      -- duplicated from users
    user_email VARCHAR(200),     -- duplicated from users
    created_at TIMESTAMP,
    item_count INT,              -- precomputed
    total_amount DECIMAL(12,2),  -- precomputed
    product_names TEXT[]         -- duplicated from products
);

-- Query: "Show user's last 10 orders with details"
-- Normalized: 3-4 table JOINs
-- Denormalized: Single table scan → much faster
```

**Denormalization techniques:**

| Technique | Description |
|---|---|
| Column duplication | Copy frequently joined column into the table that queries it |
| Precomputed aggregates | Store COUNT, SUM as column (update on write) |
| Flattened nested data | Embed sub-items as JSON column instead of separate table |
| Materialized views | DB maintains precomputed query result |
| Summary/fact tables | Separate table for analytics aggregations |

**When to denormalize:**
```
Denormalize when:
  ✓ Read performance is critical and JOIN cost is high
  ✓ Data changes infrequently (low write maintenance cost)
  ✓ Access pattern is well-known (same query repeatedly)
  ✓ CQRS: Read model can be denormalized (write model stays normalized)

Do NOT denormalize when:
  ✗ Data changes frequently (every update must update all copies)
  ✗ Storage cost is a concern (duplicate data uses more space)
  ✗ Write consistency required (denormalized copies can diverge)
  ✗ Simple CRUD app (premature optimization)
```

---

### Q12. What is the outbox pattern, and how does it guarantee exactly-once message delivery?

**Answer:**

The outbox pattern solves the dual-write problem: when a service must update a database AND publish an event to a message broker atomically. Without the pattern, the two operations can be partially applied.

**The problem (dual write):**
```
Service:
  1. INSERT order INTO orders_table  ← succeeds
  2. PUBLISH "OrderPlaced" to Kafka  ← fails (Kafka down, network issue)

Result:
  Order saved in DB ✓
  Event NOT published to Kafka ✗
  
  Downstream services (Inventory, Email) never notified
  System is in inconsistent state
  
  Reverse failure:
  1. INSERT order → FAILS (DB down)
  2. PUBLISH → succeeds
  → Event published but no order in DB → consumers process non-existent order
```

**Outbox pattern solution:**
```
STEP 1: Within a SINGLE DB transaction:
  INSERT INTO orders (id, user_id, total) VALUES (...);
  INSERT INTO outbox_events (event_type, payload, status) 
    VALUES ('OrderPlaced', '{"orderId":123}', 'PENDING');
  COMMIT;  ← Both rows committed atomically!

STEP 2: Outbox processor (separate process/thread):
  Loop:
    events = SELECT * FROM outbox_events WHERE status='PENDING' LIMIT 100;
    for event in events:
      kafka.publish(event.event_type, event.payload)   ← may fail and retry
      UPDATE outbox_events SET status='SENT' WHERE id = event.id;

Guarantee:
  If DB transaction commits: BOTH order AND outbox event are saved
  If DB transaction fails: NEITHER order NOR outbox event is saved
  Kafka publish can be retried as many times as needed until success
```

**Flow diagram:**
```
App Server:
  ┌──── DB Transaction ──────────────────────┐
  │  INSERT orders                            │
  │  INSERT outbox_events (status=PENDING)    │
  └──── COMMIT (atomic!) ─────────────────────┘

Outbox Worker (polling):
  SELECT * FROM outbox_events WHERE status='PENDING'
       │
       ├── Publish to Kafka ──> "OrderPlaced" event
       └── UPDATE status='SENT'

Consumers:
  Inventory Service ← subscribes to OrderPlaced
  Email Service ← subscribes to OrderPlaced
```

**At-least-once vs exactly-once:**
```
The outbox pattern guarantees AT-LEAST-ONCE delivery:
  If worker crashes after Kafka publish but before status='SENT':
    Worker retries → sends duplicate event to Kafka

For exactly-once:
  Consumers must be idempotent:
  IF NOT EXISTS (SELECT 1 FROM processed_events WHERE event_id = X):
    process(event)
    INSERT INTO processed_events(event_id)
  
  Or use Kafka idempotent producers + transactions (exactly-once semantics)
```

---

### Q13. What are the approaches to distributed transactions?

**Answer:**

In a microservices architecture, a single business operation may need to update data across multiple services (and databases). Coordinating these atomically is a distributed transaction problem.

**Two-Phase Commit (2PC):**
```
Phase 1 — PREPARE:
  Coordinator → Service A: "Can you commit?"
  Coordinator → Service B: "Can you commit?"
  Service A → Coordinator: "YES, I'm prepared" (locks resources)
  Service B → Coordinator: "YES, I'm prepared"

Phase 2 — COMMIT:
  Coordinator → Service A: "COMMIT"
  Coordinator → Service B: "COMMIT"
  (or ROLLBACK if any service said NO in phase 1)

Database support:
  PostgreSQL, MySQL: Use XA transactions (distributed 2PC)
  Java: JTA (Java Transaction API) over JMS

Problems:
  - Blocking: Resources (rows, tables) LOCKED during entire protocol
  - Coordinator failure: If coordinator crashes between phases → STUCK (services hold locks)
  - Latency: Multiple network round trips; synchronous across services
  - CAP: Requires strong consistency → sacrifices availability during partition
  - Not recommended for microservices
```

**Saga Pattern:**
```
A saga is a sequence of local transactions, where each transaction publishes
an event that triggers the next step. If a step fails, compensating transactions
undo previous steps.

Order placement saga:
  1. Order Service:    Create order (status=PENDING)
  2. Inventory Service: Reserve items
  3. Payment Service:  Charge customer
  4. Shipping Service: Schedule shipment
  5. Order Service:    Mark order CONFIRMED

Compensating transactions on failure:
  Step 4 fails:
    Compensate Step 3: Refund payment
    Compensate Step 2: Release inventory reservation
    Compensate Step 1: Cancel order (status=CANCELLED)
```

**Saga implementation — Choreography:**
```
No central coordinator; services react to events

  [Order Service]
    Creates order → publishes "OrderCreated"
    
  [Inventory Service]
    Subscribes to "OrderCreated"
    Reserves items → publishes "ItemsReserved"
    
  [Payment Service]
    Subscribes to "ItemsReserved"
    Charges customer → publishes "PaymentCharged"
    On failure: publishes "PaymentFailed" → triggers compensations

  Pros: No central SPOF; loose coupling
  Cons: Hard to track global state; complex debugging; circular dependencies
```

**Saga implementation — Orchestration:**
```
Central orchestrator directs each step

  [Saga Orchestrator]
    → Command "ReserveItems" → [Inventory Service]
    ← Reply "ItemsReserved" ←
    → Command "ChargePayment" → [Payment Service]
    ← Reply "PaymentFailed" ←
    → Command "CancelReservation" → [Inventory Service]
    → Command "CancelOrder" → [Order Service]

  Pros: Single place to see full workflow; easier debugging
  Cons: Orchestrator is a SPOF; more coupling
```

**Comparison:**

| Property | 2PC | Saga (Choreography) | Saga (Orchestration) |
|---|---|---|---|
| Atomicity | True (all-or-nothing) | Eventual (compensations) | Eventual (compensations) |
| Availability | Low (locks held) | High | High |
| Data visibility | Consistent at all times | Temporarily inconsistent | Temporarily inconsistent |
| Complexity | Protocol complexity | Event chain complexity | Orchestrator complexity |
| Failure recovery | Coordinator crash = stuck | Compensations | Orchestrator restarts saga |
| Best for | Traditional distributed DB | Microservices event-driven | Microservices with clear workflow |

---

### Q14. What is zero-downtime schema migration?

**Answer:**

Schema migrations in production databases must not lock tables or cause downtime. Large table alterations without careful planning can lock the table for minutes or hours, causing outages.

**Problem — naive migration:**
```sql
-- ALTER TABLE on a 500M row table:
ALTER TABLE orders ADD COLUMN discount DECIMAL(10,2) DEFAULT 0;

On PostgreSQL:
  - Acquires ACCESS EXCLUSIVE LOCK on the table
  - All reads AND writes blocked for entire migration duration
  - 500M rows × rewrite time = 10-30 minutes of downtime!
```

**Zero-downtime migration strategies:**

**Expand-Contract Pattern (Parallel Change):**
```
PHASE 1 — EXPAND (backward compatible addition):
  Add new column as nullable (no lock held long):
  ALTER TABLE orders ADD COLUMN discount DECIMAL(10,2);
  -- Nullable: No rewrite; instant on PostgreSQL
  
  Deploy application code that:
  - Writes to BOTH old and new column
  - Reads from old column (new column may be null for old rows)

PHASE 2 — MIGRATE (background):
  UPDATE orders SET discount = 0 WHERE discount IS NULL;
  -- Do in batches to avoid lock:
  UPDATE orders SET discount = 0 WHERE id > 0 AND id <= 10000 AND discount IS NULL;
  -- Repeat for each batch; no long locks

PHASE 3 — CONTRACT (remove old):
  Deploy application to read from new column (backfill complete)
  Add NOT NULL constraint: ALTER TABLE orders ALTER COLUMN discount SET NOT NULL;
  -- Safe now: all rows have been backfilled
```

**Online DDL in MySQL (pt-online-schema-change):**
```
1. Create new table with desired schema
2. Add triggers on original table (INSERT/UPDATE/DELETE) → replicate to new table
3. Copy rows from original to new table in batches
4. Atomic table rename: RENAME TABLE orders TO orders_old, orders_new TO orders
5. Drop old table

Tools: pt-online-schema-change, gh-ost (GitHub)
MySQL 8.0+: Many DDL operations support ALGORITHM=INPLACE (no rewrite)
```

**PostgreSQL CONCURRENTLY:**
```sql
-- Index creation without blocking table:
CREATE INDEX CONCURRENTLY idx_orders_user_id ON orders(user_id);
-- Takes longer but allows reads/writes during build

-- Regular index creation:
CREATE INDEX idx_orders_user_id ON orders(user_id);
-- Blocks ALL writes for duration (hours on large tables!)
```

**Schema migration checklist:**
```
Before running:
  ✓ Test migration on staging with production data volume
  ✓ Measure migration duration
  ✓ Verify application works with BOTH old and new schema (expand-contract)
  ✓ Plan rollback: Can you roll back the application? The schema?
  
During:
  ✓ Monitor replication lag (migrations can cause lag on replicas)
  ✓ Monitor lock waits
  ✓ Batch large updates
  
Tooling:
  PostgreSQL: flyway, liquibase, or pgroll (online schema changes)
  MySQL:       gh-ost, pt-online-schema-change
  Rails:       strong_migrations gem (warns about unsafe migrations)
```

---

### Q15. What is the read replica pattern for scaling reads?

**Answer:**

Read replicas are asynchronous copies of the primary database that serve read queries, distributing read load and allowing the primary to focus on writes.

**Architecture:**
```
                    Writes
                     │
              ┌──────▼────────┐
              │   Primary DB   │  (accepts all writes)
              └──────┬────────┘
                     │ async replication
          ┌──────────┼──────────────┐
          ▼          ▼              ▼
    [Replica 1]  [Replica 2]  [Replica 3]
    (reads only) (reads only) (reads only)
    
    Read queries distributed across replicas
    Write queries always go to primary
```

**Routing reads to replicas:**

```python
class DatabaseRouter:
    def db_for_read(self, model, **hints):
        # Route reads to replicas
        return random.choice(['replica1', 'replica2', 'replica3'])
    
    def db_for_write(self, model, **hints):
        # All writes to primary
        return 'primary'

# Django example:
Product.objects.filter(category='electronics')  # → replica
Product.objects.create(name='iPhone')           # → primary
```

**Replication lag and its implications:**
```
Async replication lag: typically 10ms – a few seconds

Problem: Read-after-write consistency
  1. User updates email: PRIMARY confirms write at t=0
  2. User immediately views profile: reads from REPLICA
  3. Replica lags 200ms → old email shown!
  
Solutions:
  1. Read-your-writes: After a write, route user's reads to primary for 1-5 seconds
  2. Sticky primary: User always reads from primary after any write (sacrifices read scaling)
  3. Synchronous replication for critical paths only (high latency, use sparingly)
  4. Token-based: Write returns version number; reads request "at least version X"
```

**AWS RDS read replicas:**
```
Create up to 15 read replicas per RDS instance
Cross-region replicas for global read distribution
Replica promotion: Promote replica to standalone DB for read-write (for DR)

Use case:
  API reads → Route 53 weighted routing → 3 read replicas
  Admin writes → Always to primary
  Analytics queries → Dedicated analytics replica (long queries won't impact production)
```

**Read replica sizing:**
```
Scale:
  1 replica = ~2× read capacity
  5 replicas = ~6× read capacity (not 5×: some overhead per replica)
  
  Start with replicas equal in size to primary
  Can use smaller replicas for read-only analytics (read IOPS differ from write IOPS)
```

---

## HARD (Q16–Q20)

---

### Q16. What is eventual consistency and when is it acceptable?

**Answer:**

Eventual consistency is a model in which, given no new updates are made, all replicas will converge to the same value over time. It does not guarantee that all nodes see the same data at the same instant.

**How it works:**
```
t=0:   User A writes "price=100" to Node 1
t=0:   User B reads from Node 2: sees "price=80" (old value, Node 2 not yet updated)
t=0.2: Replication propagates to Node 2
t=0.2: User B reads from Node 2: now sees "price=100" (converged)

Eventual consistency says: "Eventually, all nodes will agree on price=100"
It does NOT say: "At any point in time, all nodes agree on the same value"
```

**Consistency spectrum:**

```
Weak consistency ─────────────────────────────── Strong consistency
       │                    │                           │
   "Best effort"     "Eventual"                  "Linearizable"
  UDP multicast      Cassandra (default)         ZooKeeper/etcd
  Video streaming    DynamoDB (default)          CockroachDB
  Cached reads       DNS TTL staleness           PostgreSQL (sync replica)
```

**When eventual consistency is acceptable:**

| Use Case | Acceptable? | Why |
|---|---|---|
| Social media likes count | YES | 1,000 vs 1,001 likes — user won't notice |
| Product view count | YES | Approximate count is fine |
| Product catalog details | YES | 1-2 second stale description is harmless |
| DNS records | YES | TTL provides bounded staleness |
| Shopping cart total | BORDERLINE | Show stale but recalculate at checkout |
| Account balance | NO | Must be exact; financial regulation |
| Inventory available | NO | Overselling is a business-critical error |
| Seat reservations | NO | Two people cannot book the same seat |
| Payment status | NO | "Payment succeeded" must be definitive |

**Bounded eventual consistency:**
```
Design guarantee: "Data will be consistent within X seconds"

Implementation:
  Cassandra consistency level QUORUM:
    Write: 2 of 3 nodes must confirm
    Read: 2 of 3 nodes must respond
    → Read sees write from a node with latest write
    → Bounded by replication lag (usually < 1 second)
  
  This is stronger than pure eventual but not fully linearizable
```

**Making users tolerant of eventual consistency:**
```
Optimistic UI (write locally before server confirmation):
  User clicks "Like" → UI immediately shows +1 like
  Server processes asynchronously → confirms or reverts
  
  User sees instant feedback (feels consistent) even if replica is stale
  
Read-your-writes guarantee:
  After user's own write → route reads to primary or wait for replication
  Other users may see stale data but the writing user always sees their own writes
```

---

### Q17. How do you handle hot partitions in a distributed database?

**Answer:**

A hot partition occurs when a disproportionate amount of traffic is concentrated on a single database partition (shard). It is the database equivalent of a hot cache key.

**Common causes:**
```
1. Poorly chosen partition key:
   Partition by user_id, but some users (celebrities, bots) generate 1000×
   more traffic than average users → their partition overwhelmed

2. Time-based partitioning:
   Cassandra partitioned by date → ALL new writes go to TODAY's partition
   Historical partitions get only reads; current partition gets all writes
   
3. Auto-increment ID sharding with range:
   New rows always have highest ID → always written to the LAST shard
   Last shard gets all writes; early shards get only reads
```

**Detection:**
```
DynamoDB:
  CloudWatch: ConsumedWriteCapacityUnits per partition
  Alert: Any single partition > 3× average → hot partition

Cassandra:
  nodetool cfstats: Compaction, read/write throughput per node
  nodetool tpstats: Thread pool statistics

PostgreSQL (sharded):
  Monitor: queries/second per shard host
  pg_stat_user_tables: seq_scan + idx_scan counts
```

**Solutions:**

**1. Better partition key selection:**
```
Problem: Partition by user_id for celebrity accounts
Solution: Composite partition key that distributes writes

Option A: Add random suffix (shard within a user):
  DynamoDB: PK = user_id#random(1-10)
  → 10 partitions per user → 10× write throughput
  
  Trade-off: Reading all posts by a user requires 10 queries (scatter-gather)

Option B: Hierarchical sharding:
  Shard 1: High-traffic users (user_id in [celeb1, celeb2, ...])
  Shards 2-10: Normal users (user_id by hash)
  Routing logic selects shard based on user tier
```

**2. Write-behind + aggregation for hot keys:**
```
High-frequency counter updates (like counts, view counts):

Instead of: UPDATE posts SET views = views + 1 WHERE id = 123  (hot row)
Use: 
  1. Write to Redis: INCR views:post:123  (in-memory, handles millions/sec)
  2. Periodic flush (every 60s): UPDATE posts SET views = redis_count
  
  Hot row gets one DB write per minute instead of millions
```

**3. DynamoDB adaptive capacity (automatic):**
```
DynamoDB automatically isolates hot partitions and reallocates capacity
  - Redistributes capacity from cold partitions to hot ones
  - Transparent to application
  - Handles temporary hot spots automatically

For sustained hot partitions: Still need design-level solution
```

**4. Cassandra write distribution techniques:**
```
Hot partition: time-series data partitioned by sensor_id + date
  "sensor:001:2024-01-15" → all today's writes go here

Solution: Add random bucket to partition key:
  "sensor:001:2024-01-15:bucket:3"  (bucket = random 1-5)
  
  Read ALL data for sensor 001 on 2024-01-15:
    SELECT * WHERE pk IN ("sensor:001:2024-01-15:bucket:1", 
                           "sensor:001:2024-01-15:bucket:2", ...5)
  
  Writes distributed across 5 partitions (5× write throughput)
  Reads require 5 parallel queries → small overhead acceptable
```

---

### Q18. Compare NewSQL vs traditional SQL vs NoSQL.

**Answer:**

NewSQL databases attempt to provide the horizontal scalability of NoSQL while maintaining the ACID guarantees and SQL interface of traditional RDBMS.

**Traditional SQL (PostgreSQL, MySQL):**
```
Architecture: Single primary, read replicas
Scaling: Vertical (mostly); horizontal reads via replicas; sharding requires app-level logic
ACID: Yes (fully)
Query model: Full SQL
Write scalability: ~50K-100K writes/sec (single primary bottleneck)
Consistency: Strong (ACID)

Limitations:
  - Horizontal write scaling requires manual application-level sharding
  - Sharding breaks cross-shard transactions and joins
  - Schema migrations can cause downtime on large tables
  
Best for: Most OLTP applications, < 10 TB data, < 50K writes/sec
```

**NoSQL (Cassandra, DynamoDB, MongoDB):**
```
Architecture: Leaderless or multi-primary horizontal sharding
Scaling: Linear horizontal write scaling (add nodes → add throughput)
ACID: Limited (single-partition only, or none)
Query model: Limited (NoSQL operators, no arbitrary JOINs)
Write scalability: Millions of writes/sec
Consistency: Eventual (mostly)

Limitations:
  - No cross-partition ACID transactions
  - Limited query flexibility (must design schema around access patterns)
  - Eventual consistency requires application-level handling
  
Best for: IoT, time-series, simple high-throughput access patterns
```

**NewSQL (CockroachDB, Google Spanner, TiDB, YugabyteDB):**
```
Architecture: Distributed, sharded with consensus replication (Raft/Paxos)
Scaling: Linear horizontal scaling for BOTH reads AND writes
ACID: Yes — full ACID transactions across shards
Query model: Full SQL (ANSI SQL compatible)
Write scalability: Millions of writes/sec (distributed)
Consistency: Strong (linearizable or serializable)

How it achieves distributed ACID:
  - Data sharded across nodes using consistent hashing
  - Raft consensus protocol per shard group → ensures durability + consistency
  - Two-phase locking for cross-shard transactions
  - Clock synchronization (TrueTime in Spanner, HLC in CockroachDB) for ordering
  
Best for: Global applications needing SQL+ACID+horizontal scale
```

**Comparison table:**

| Property | Traditional SQL | NoSQL (Cassandra) | NewSQL (CockroachDB) |
|---|---|---|---|
| Write scaling | Vertical/manual sharding | Linear horizontal | Linear horizontal |
| ACID transactions | Full (single node) | Single partition only | Full (cross-shard) |
| Query flexibility | Full SQL | Limited CQL | Full SQL |
| Consistency | Strong | Eventual (tunable) | Strong (linearizable) |
| Operational complexity | Low | High | Medium-High |
| Multi-region support | Complex | Native (AP) | Native (CP) |
| Cost | Low | Medium | Higher |
| Maturity | Very high | High | Medium |

**When to use NewSQL:**
```
Scenario: Global financial platform
  Requirements:
    - ACID transactions (cannot double-charge customers)
    - Global active-active (users in 3 regions → low latency everywhere)
    - Horizontal write scaling (10M transactions/day)
  
  Traditional SQL: Cross-region requires complex custom sharding + eventual consistency
  NoSQL: No ACID across operations
  NewSQL (Spanner/CockroachDB): Distributed ACID + geo-distribution → best fit
```

---

### Q19. What is the polyglot persistence pattern, and when should you use it?

**Answer:**

Polyglot persistence means using multiple different database technologies within one system, choosing the best database for each specific use case rather than forcing all data into a single database.

**The "one database for everything" anti-pattern:**
```
Using only PostgreSQL for all of:
  - User profiles (relational → good fit)
  - Product search (needs full-text → poor fit; workarounds needed)
  - Session tokens (key-value → wasteful of relational DB)
  - Analytics aggregations (columnar → OLAP, not OLTP)
  - Time-series metrics (specialized structure → poor fit for relational)
  - Activity feeds (append-only, time-ordered → poor fit)
  
  Result: Everything "works" but nothing works WELL
          Complex queries, slow performance, high cost
```

**Polyglot persistence architecture:**
```
                         ┌─────────────────────────────────────────┐
                         │             E-Commerce Platform          │
                         └──────────────────┬──────────────────────┘
                                            │
             ┌──────────────────────────────┼─────────────────────────────────┐
             │                             │                                  │
    ┌────────▼────────┐          ┌─────────▼──────────┐          ┌───────────▼────────┐
    │  PostgreSQL      │          │  Elasticsearch      │          │  Redis              │
    │  (relational)    │          │  (search)           │          │  (cache/sessions)   │
    │                  │          │                     │          │                     │
    │  users           │          │  product_index      │          │  session:abc123     │
    │  orders          │          │  (full-text search) │          │  cart:user456       │
    │  products        │          │  (faceted filters)  │          │  rate_limiter:key   │
    │  inventory       │          │  (autocomplete)     │          │  leaderboard (ZSet) │
    └────────┬─────────┘          └─────────────────────┘          └────────────────────┘
             │
    ┌────────▼─────────┐          ┌─────────────────────┐          ┌────────────────────┐
    │  Cassandra        │          │  ClickHouse          │          │  S3 / Object Store │
    │  (time-series)    │          │  (analytics)         │          │  (files/media)     │
    │                   │          │                      │          │                    │
    │  order_events     │          │  sales_aggregations  │          │  product images    │
    │  product_views    │          │  user_cohort_analysis│          │  order documents   │
    │  user_activity    │          │  funnel_reports      │          │  video content     │
    └───────────────────┘          └──────────────────────┘          └────────────────────┘
```

**Data synchronization:**
```
Master data in PostgreSQL → event-driven sync to specialized stores:

PostgreSQL (write) 
    │
    └──[Debezium CDC]──> [Kafka]
                              │
              ┌───────────────┼────────────────────┐
              ▼               ▼                    ▼
       Elasticsearch    ClickHouse             Cassandra
       (sync product    (sync sales             (sync activity
        catalog)         events)                 log)
```

**When polyglot persistence is appropriate:**
```
Use polyglot when:
  ✓ Different data types genuinely need different engines (search, analytics, sessions)
  ✓ Team has the operational expertise to manage multiple databases
  ✓ Traffic/scale justifies the complexity (high-traffic product)
  ✓ Clear domain boundaries between data types

Do NOT use polyglot when:
  ✗ Small team < 5 engineers (operational overhead too high)
  ✗ Low-traffic application (premature optimization)
  ✗ Strong ACID needed across all data (polyglot breaks transactions)
  ✗ No clear performance or feature need for second database
```

---

### Q20. Design a sharding strategy for a user database that grows to 1 billion users.

**Answer:**

At 1 billion users, a single database node cannot handle the write load, storage, or index size. A well-designed sharding strategy must handle growth, avoid hot spots, and enable operational manageability.

**Scale analysis:**
```
1 billion users:
  Average row size: 500 bytes (id, name, email, created_at, metadata)
  Total data: 1B × 500 bytes = 500 GB (just user table)
  
  With indexes + other user tables (preferences, settings, auth):
  Total storage per shard: 5 TB per database
  
  At 100M DAU, writes (updates + new registrations):
  ~100K writes/sec peak → single DB limit exceeded
  
  Target: 100 shards × 10M users = manageable per shard
```

**Sharding key selection:**

```
Option 1: user_id hash (recommended)
  shard_id = hash(user_id) % 100
  
  Pros:
    Uniform distribution (hash spreads evenly)
    All queries for a user go to one shard (user operations are user-scoped)
    Easy to add shards with consistent hashing
  
  Cons:
    Cross-user queries (friends, activity across users) need scatter-gather
    Cannot easily sort users by signup date across shards

Option 2: user_id range (not recommended)
  Shard 0: user_id 1–10M
  Shard 1: user_id 10M–20M
  
  Problem: New registrations always go to the LAST shard → hot spot
```

**Directory-based sharding (best for flexibility):**
```
[Routing Service / Shard Map]
  user_id: 123456  → shard_id: 47
  user_id: 789012  → shard_id: 12
  user_id: 345678  → shard_id: 83

Shard map stored in:
  Redis (fast lookups, < 1ms)
  Backed by PostgreSQL (durable source of truth)
  
Cached on each app server (reload on cache miss)
```

**Physical architecture:**
```
100 shards × (1 primary + 2 replicas) = 300 database nodes

Shard group:
  ┌────────────────────────────────────────┐
  │  Shard 0 (users 1-10M by directory)   │
  │  Primary: db-shard0.internal           │
  │  Replica1: db-shard0-r1.internal       │
  │  Replica2: db-shard0-r2.internal       │
  └────────────────────────────────────────┘
  (× 100 shard groups)

Writes → Primary of correct shard
Reads  → Replica of correct shard (or primary for read-after-write)
```

**Cross-shard operations:**

```
"Find all users in New York": (needs all shards)
  Send query to all 100 shards in parallel → merge results
  Called "scatter-gather" or "fan-out"
  
  Latency: max(latency of all shards) → not much worse than single shard
  Database load: 100 DB queries (resource intensive for analytics)
  
  Better approach:
    Keep users table sharded for writes
    Sync a denormalized copy to BigQuery/ClickHouse for analytics
    Never run analytics on production OLTP shards

"Get all friends of user 123":
  Friends are user:123 → user:456 → user:789 (different shards)
  Application-level join:
    1. Get friend_id list from user:123's shard
    2. Batch lookup user profiles across multiple shards
    3. Merge in application layer
```

**Shard rebalancing:**
```
Growing from 100 → 200 shards:

Option A: Consistent hashing (minimal remapping)
  Add 100 virtual nodes → only ~50% of keys remapped
  Background migration: Copy data to new shard, verify, update directory, cleanup old

Option B: Double sharding (simple)
  Create 200 new shards
  Each old shard splits into 2 new shards
  Old Shard 0 → New Shard 0 (first half) + New Shard 1 (second half)
  
  Migration:
  1. Start replicating old shard to two new shards
  2. Once in sync: redirect writes to new shards
  3. Complete remaining replication
  4. Decomission old shard
  5. Update shard directory
  
  Zero downtime with careful execution
```

**Key principles for this design:**
```
1. Shard on user_id for all user-owned data (orders, preferences, activity)
   → All data for user X on one shard → no cross-shard joins for user queries
   → Called "co-location" → critical for performance

2. Use directory-based routing for flexibility to rebalance

3. Never run cross-shard queries on OLTP — use separate analytics system

4. Design for growth: start with logical shard IDs (not physical DBs)
   10 physical DBs can each host multiple logical shards
   Scale by splitting logical shards to new physical DBs
```

---

## Quick Reference

### Database Selection Guide
| Use Case | Recommended DB | Reason |
|---|---|---|
| ACID transactions (finance, orders) | PostgreSQL | Full ACID, SQL |
| Document store, flexible schema | MongoDB | BSON documents |
| High-throughput time-series | Cassandra | AP, linear scaling |
| Serverless + AWS | DynamoDB | Auto-scaling, pay-per-use |
| Full-text search | Elasticsearch | Inverted index, relevance |
| Distributed ACID at scale | CockroachDB / Spanner | NewSQL |
| Key-value cache | Redis | In-memory, rich structures |
| Analytics OLAP | ClickHouse / Redshift | Columnar |

### CAP Theorem Quick Guide
| Database | CAP | Partition Behavior |
|---|---|---|
| PostgreSQL (replicated) | CP | Rejects writes to maintain consistency |
| Cassandra | AP | Accepts writes, reconciles later |
| DynamoDB (default) | AP | Eventual consistent reads |
| ZooKeeper / etcd | CP | Refuses requests without quorum |
| MongoDB (default) | CP | Reads from primary only |
| CockroachDB | CP | Strong consistency, distributed |

### Sharding Strategies
| Strategy | Best For | Hot Spot Risk |
|---|---|---|
| Hash sharding | Uniform random access | Low |
| Range sharding | Range queries | High (new writes to last range) |
| Directory-based | Maximum flexibility | Low (controlled by routing service) |
| Geographic | Compliance, latency | Medium (one region may grow faster) |

### Pagination Decision
| Scenario | Use |
|---|---|
| Infinite scroll, API pagination | Cursor-based (keyset) |
| Admin UI with page jumps | Offset (small datasets only) |
| > 10K rows deep pagination | Cursor-based only |

### Index Types
| Index | Supports | Not For |
|---|---|---|
| B-tree (default) | =, <, >, BETWEEN, LIKE 'x%' | Full-text, arrays |
| Composite | Multi-column (leftmost prefix) | Skip-column queries |
| Covering (INCLUDE) | Index-only scans | Heavy write tables |
| Partial (WHERE) | Sparse high-value subsets | Non-matching conditions |
| GIN | Arrays, JSONB, full-text | Simple equality |
| BRIN | Sequential large tables | Random access patterns |

### Distributed Transaction Patterns
| Pattern | Consistency | Complexity | Use When |
|---|---|---|---|
| 2PC | Strong ACID | High | Legacy systems, same DC |
| Saga (Choreography) | Eventual | Medium | Event-driven microservices |
| Saga (Orchestration) | Eventual | Medium-High | Complex workflow, clear steps |
| Outbox Pattern | Eventual (reliable) | Low-Medium | DB + message broker sync |
