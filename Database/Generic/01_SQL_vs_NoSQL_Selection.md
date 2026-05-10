# SQL vs NoSQL — Database Selection for System Design

## Easy (Q1–Q7)

---

**Q1. What are the fundamental differences between SQL and NoSQL databases?**

| Dimension | SQL (Relational) | NoSQL |
|---|---|---|
| Data model | Tables with fixed columns | Document, key-value, wide-column, graph |
| Schema | Enforced at write time | Flexible / schema-less |
| Query language | Standardized SQL | Database-specific API |
| ACID transactions | Built-in | Varies — often eventual consistency |
| Relationships | Foreign keys, JOINs | Denormalized / embedded |
| Horizontal scale | Difficult (sharding is manual) | Designed for horizontal scale |
| Examples | PostgreSQL, MySQL, SQL Server | MongoDB, Cassandra, DynamoDB, Redis |

SQL enforces data integrity and supports complex relational queries. NoSQL trades some of that rigidity for schema flexibility, horizontal scalability, and access-pattern-specific performance.

---

**Q2. What are the four main NoSQL database types and when is each appropriate?**

**1. Document stores (MongoDB, CouchDB, Firestore)**
- Store JSON/BSON documents; each document can have a different shape
- Use when: data is naturally nested or hierarchical, schema varies per record
- Examples: product catalog (each product type has different attributes), CMS, user profiles

**2. Key-value stores (Redis, DynamoDB, Memcached)**
- Simple hash map: key → value
- Use when: access is always by a known key, no range queries needed, extreme speed required
- Examples: session storage, caching, feature flags, rate limiting counters

**3. Wide-column stores (Apache Cassandra, HBase, Google Bigtable)**
- Rows keyed by partition key; columns vary per row; sorted by clustering key
- Use when: write-heavy, time-series or event data, known query patterns, massive scale
- Examples: IoT telemetry, activity logs, time-ordered events per user

**4. Graph databases (Neo4j, Amazon Neptune)**
- Nodes (entities) and edges (relationships) stored natively
- Use when: the query is "traverse relationships" — friends-of-friends, recommendations, fraud paths
- Examples: social networks, recommendation engines, fraud detection

---

**Q3. When should you choose SQL over NoSQL?**

Choose SQL when:

1. **Data has clear relationships** — users → orders → products → categories. JOINs are natural.
2. **ACID transactions are required** — bank transfers, order placement, inventory deduction. Partial failure is unacceptable.
3. **Schema is stable** — the data structure is well-understood and changes infrequently.
4. **Ad-hoc queries needed** — business analysts run arbitrary aggregations across the data. SQL is universally understood.
5. **Referential integrity matters** — the database must enforce that every order has a valid customer.
6. **You do not know your access patterns** — SQL handles any query without schema redesign.

```
Good SQL use cases:
  E-commerce (orders, payments, inventory)
  Banking and fintech
  ERP and CRM systems
  Healthcare records
  Any system where data integrity is non-negotiable
```

---

**Q4. When should you choose NoSQL over SQL?**

Choose NoSQL when:

1. **Schema varies per record** — a product catalog where shirts have color/size and laptops have RAM/CPU.
2. **Write throughput exceeds single-node SQL capacity** — millions of appends per second (Cassandra).
3. **Horizontal scale is a hard requirement** — data volume exceeds what any single server can hold.
4. **Access pattern is known and simple** — always `GET user:{id}`. No JOINs needed.
5. **Document model fits naturally** — deeply nested data, JSON APIs, variable structure.

```
Good NoSQL use cases:
  Session/cache (Redis)
  Product catalog with variable attributes (MongoDB)
  User activity feeds, timelines (Cassandra)
  Real-time leaderboards (Redis Sorted Sets)
  IoT sensor telemetry (Cassandra, InfluxDB)
```

---

**Q5. What is the CAP theorem and how does it guide database selection?**

CAP theorem states that a distributed database can guarantee at most **two** of:

| Property | Meaning |
|---|---|
| **Consistency (C)** | Every read returns the most recent committed write |
| **Availability (A)** | Every request receives a response (no timeout/error) |
| **Partition Tolerance (P)** | System operates despite network partitions |

Network partitions always happen in distributed systems, so P is unavoidable. The real choice is **CP vs AP**:

**CP (Consistency + Partition Tolerance):**
- On network partition: reject writes to maintain consistency
- Examples: HBase, MongoDB (majority write concern), Zookeeper, traditional SQL
- Use when: correctness is critical (financial systems, inventory)

**AP (Availability + Partition Tolerance):**
- On network partition: continue serving reads/writes, allow temporary inconsistency
- Examples: Cassandra, CouchDB, DynamoDB (default)
- Use when: availability is critical and stale data is acceptable (shopping cart, social feed)

**CA (single-node only):** PostgreSQL, MySQL — no partition to tolerate on a single server.

---

**Q6. What is ACID vs BASE and how do they affect database choice?**

**ACID (SQL databases):**
- **Atomicity** — all operations in a transaction succeed or all fail
- **Consistency** — database moves between valid states; constraints always satisfied
- **Isolation** — concurrent transactions do not interfere with each other
- **Durability** — committed data survives crashes

**BASE (many NoSQL databases):**
- **Basically Available** — system responds, even if data is stale
- **Soft state** — system state may change over time without input (convergence)
- **Eventually Consistent** — given no new writes, all nodes converge to the same value

```
ACID example (bank transfer):
  BEGIN;
  UPDATE accounts SET balance = balance - 500 WHERE id = 1;
  UPDATE accounts SET balance = balance + 500 WHERE id = 2;
  COMMIT;  -- both happen or neither happens

BASE example (Cassandra shopping cart):
  Write cart item to one node → confirmed
  Other nodes may return slightly stale cart for 100–500ms
  Eventually all nodes converge to same cart state
  → Acceptable: user rarely sees their own staleness
```

Choose ACID when correctness is non-negotiable. Choose BASE when availability and performance outweigh temporary inconsistency.

---

**Q7. When should you use a time-series database vs a general-purpose SQL database?**

**General-purpose SQL struggle with time-series because:**
- Millions of appends per second overwhelm B-tree indexes (random writes across pages)
- Range queries ("average CPU every 5 minutes over last 24 hours") require full scans without special structures
- Old data is rarely updated but needs automated deletion (retention policies)

**Time-series databases provide:**
- Columnar storage optimized for appending and time-range reads
- Native downsampling (resample to coarser granularity for older data)
- Automatic data retention and expiry
- Time-bucket functions (`time_bucket('5 minutes', ts)`)
- Compression by time chunk (delta encoding, RLE)

| Database | Type | When to use |
|---|---|---|
| TimescaleDB | PostgreSQL extension | Need SQL + time features in one system |
| InfluxDB | Purpose-built TSDB | Pure metrics, monitoring, IoT |
| ClickHouse | OLAP column store | Event data + analytics, very high ingestion |
| QuestDB | High-perf TSDB | Very high ingestion rate (> 1M rows/sec) |

**Rule:** Use TimescaleDB if you need JOINs with relational tables. Use a dedicated TSDB for purely append-heavy metric/event data exceeding ~100K writes/second.

---

## Medium (Q8–Q15)

---

**Q8. A product catalog has highly variable attributes — shirts have color/size, laptops have RAM/CPU/GPU. Design the database schema.**

**Option A: PostgreSQL with JSONB (recommended starting point)**
```sql
CREATE TABLE products (
    product_id   BIGSERIAL    PRIMARY KEY,
    name         TEXT         NOT NULL,
    category     TEXT         NOT NULL,
    base_price   NUMERIC(10,2),
    attributes   JSONB,                     -- {"color":"red","size":"M"} or {"ram_gb":16,"cpu":"i7"}
    created_at   TIMESTAMPTZ  DEFAULT NOW()
);

-- GIN index for attribute queries
CREATE INDEX idx_products_attrs ON products USING GIN (attributes);
CREATE INDEX idx_products_category ON products (category);

-- Query: all red shirts under $50
SELECT product_id, name, base_price
FROM products
WHERE category = 'shirt'
  AND attributes @> '{"color": "red"}'   -- contains operator
  AND base_price < 50;

-- Query: laptops with 16GB RAM
SELECT * FROM products
WHERE category = 'laptop'
  AND (attributes->>'ram_gb')::int >= 16;
```

**Option B: MongoDB (if attribute variety is extreme and schema truly unknown)**
```js
// Each category can have completely different fields — no schema migration needed
db.products.insertOne({
  name: "Ultra Laptop",
  category: "laptop",
  base_price: 999.99,
  attributes: { ram_gb: 16, cpu: "i7-12700H", storage: "512GB NVMe", gpu: "RTX 3060" }
})
db.products.createIndex({ "attributes.ram_gb": 1, "attributes.category": 1 })
```

**Option C: EAV (Entity-Attribute-Value) — avoid this**
```sql
-- Anti-pattern: terrible query performance, no type safety
CREATE TABLE product_attributes (
    product_id INT, attribute_name TEXT, attribute_value TEXT
);
-- Getting all attributes for 1000 products: 1000 × N rows to pivot → N+1 problem
```

**Recommendation:** Start with PostgreSQL + JSONB. It handles variable attributes, supports ACID for orders and payments, and avoids polyglot complexity. Migrate to MongoDB only if query patterns truly demand it.

---

**Q9. When would you use a graph database instead of modeling relationships in a SQL database?**

**When SQL struggles with relationship queries:**

```sql
-- Find all friends-of-friends-of-friends (3-hop traversal) in a 50M user social network
WITH RECURSIVE fof AS (
    SELECT friend_id AS uid, 1 AS depth FROM friendships WHERE user_id = 1
    UNION ALL
    SELECT f.friend_id, fof.depth + 1
    FROM friendships f JOIN fof ON f.user_id = fof.uid
    WHERE fof.depth < 3
)
SELECT DISTINCT uid FROM fof;
-- Table: 15 billion rows (50M users × 300 avg friends)
-- This query can take minutes and generates massive intermediate result sets
```

**Why graph databases win on traversal:**
```
Neo4j stores each relationship as a direct pointer:
  Node → [relationship pointer] → Node → [relationship pointer] → Node

Each hop = follow a pointer in memory (O(1) per hop, not O(log N) like a B-tree)
3-hop traversal on 50M users: Neo4j ≈ 200ms, PostgreSQL recursive CTE ≈ minutes
```

**Use graph databases when:**
- The query is "find paths" or "find connected nodes" N hops away
- Relationship traversal is the primary access pattern
- Relationships themselves have properties (friendship strength, trust score)
- Use cases: fraud detection (shared phone/address graphs), social recommendations, knowledge graphs

**Use SQL when:**
- You need aggregations across nodes (count friends per user → GROUP BY)
- The dataset fits in one server and relationship depth is shallow (≤ 2 hops)
- You want a single database for relational + graph data (PostgreSQL with recursive CTEs handles simple cases)

---

**Q10. Compare DynamoDB, Cassandra, and PostgreSQL for a write-heavy IoT telemetry system ingesting 1M events/second.**

**Requirements:**
- 1M writes/second, ~200 bytes per event
- Query: "last 1000 events for device X" and "all events for device X in last hour"
- 30-day retention, then delete
- Globally deployed devices

**PostgreSQL:**
```
Single node cap: ~50K–100K writes/second
Partitioned + multiple primaries: ~500K writes/second with effort
WAL + B-tree updates on every insert → high write amplification
Good for: queries, JOINs, aggregations
Verdict: Not suitable at 1M writes/second without extreme sharding complexity
```

**DynamoDB:**
```
Table design:
  Partition key: device_id
  Sort key: event_time (ISO format for lexicographic ordering)
  TTL attribute: expire_at (auto-delete after 30 days)

On-demand mode: scales to 1M writes/second automatically
Single-digit ms latency per write
No schema design needed for variable event payloads (JSON attribute)
Cost: ~$1.25 per million write request units → ~$108K/day at 1M/s (expensive!)
Verdict: Works technically, prohibitively expensive at this throughput
```

**Cassandra:**
```sql
CREATE TABLE telemetry (
    device_id   UUID,
    bucket      TEXT,        -- 'device_id:2024-01-15-14' (hour bucket to cap partition size)
    event_time  TIMESTAMP,
    payload     BLOB,
    PRIMARY KEY ((device_id, bucket), event_time)
) WITH CLUSTERING ORDER BY (event_time DESC)
  AND default_time_to_live = 2592000   -- 30 days in seconds
  AND compaction = { 'class': 'TimeWindowCompactionStrategy',
                     'compaction_window_size': 1,
                     'compaction_window_unit': 'HOURS' };

-- Linear scaling: add nodes to increase throughput
-- Multi-datacenter replication native
-- TWCS compaction perfect for time-series (seals old hour windows, no re-compaction)
Verdict: Best fit — designed exactly for this workload
```

**Winner: Cassandra** for this use case. Add Kafka in front to absorb burst writes.

---

**Q11. What is polyglot persistence and when should you adopt it?**

Polyglot persistence means using multiple different database technologies within one application, each chosen for the data it handles best.

**Example — e-commerce platform:**
```
User accounts + orders + payments  →  PostgreSQL
  (ACID transactions, relational integrity)

Product catalog with variable attrs →  MongoDB or PostgreSQL + JSONB
  (flexible schema per product type)

Session tokens, rate limit counters →  Redis
  (sub-millisecond access, TTL expiry)

Product + store search              →  Elasticsearch
  (full-text, faceted filters, fuzzy matching)

User activity events, clickstream   →  Cassandra or ClickHouse
  (high-write, time-ordered, analytics)
```

**When to adopt:**
- You have genuinely different data access patterns that a single database cannot serve efficiently
- Each data type has distinct scale requirements (session: 500K reads/sec; orders: 2K writes/sec)
- Your team has operational maturity to run, monitor, and back up multiple systems

**When NOT to adopt:**
- Early-stage product — operational overhead outweighs any benefit
- Single team — each database requires expertise, alerting, backup strategy, on-call runbooks
- PostgreSQL can solve the problem — JSONB handles variable schemas, GIN covers full-text basics, pg_trgm handles fuzzy search

**Rule:** Start with PostgreSQL for everything. Add a specialized database only when you hit a specific, measurable bottleneck that PostgreSQL provably cannot solve.

---

**Q12. How do you choose between embedding and referencing in a document database like MongoDB?**

This is the core schema design decision in MongoDB — it maps to normalization vs denormalization.

**Embed when (denormalize):**
- The child data is always accessed with the parent (order items always fetched with order)
- The child data does not change independently (product snapshot at order time)
- The child array has bounded size (an order has ≤ 100 items)
- You optimize for reads (one document = one read, no $lookup)

```js
// Embedded order with items (good)
{
  _id: ObjectId("..."),
  customer_id: 12345,
  status: "shipped",
  items: [
    { product_id: 1, name: "Laptop", qty: 1, price: 999.99 },
    { product_id: 2, name: "Mouse",  qty: 2, price: 29.99 }
  ]
}
// Single document read returns everything needed to display the order
```

**Reference when (normalize):**
- The child entity is shared across many parents (a product referenced by thousands of orders)
- The child changes independently (product price/description updates should not require updating all orders)
- The array could grow unboundedly (a user's all-time posts should not be embedded in the user document)
- The child is frequently accessed independently (product detail page — no need to fetch an order)

```js
// Referenced (separate collections)
orders:   { _id: ..., customer_id: 12345, item_ids: [ObjectId("prod1"), ...] }
products: { _id: ObjectId("prod1"), name: "Laptop", price: 999.99, stock: 45 }
// Use $lookup to join — but do this sparingly at scale
```

**Practical rule:** If you always read A with B, embed B in A. If B has its own lifecycle or appears in many As, reference it.

---

**Q13. How does Cassandra's data model differ from a relational model? How do you design a timeline table?**

**Cassandra design principles:**
1. **No JOINs ever** — denormalize everything; one query = one table
2. **Design tables around queries** — decide what queries you need first, then design the table
3. **Partition key** determines which node stores the data — all rows with the same partition key live on the same node(s) and are fast to read together
4. **Clustering key** determines sort order within a partition

**Twitter-like home timeline:**

```
Query: "Get the 50 most recent posts for user B's home feed"
Design: partition by follower_id (one partition = one user's entire feed)
         cluster by post_time DESC (most recent first within that partition)
```

```sql
CREATE TABLE home_feed (
    follower_id  UUID,
    post_time    TIMESTAMP,
    post_id      UUID,
    author_id    UUID,
    author_name  TEXT,
    content      TEXT,
    PRIMARY KEY (follower_id, post_time, post_id)
) WITH CLUSTERING ORDER BY (post_time DESC);

-- When user A (followed by B and C) posts:
INSERT INTO home_feed (follower_id, post_time, post_id, author_id, author_name, content)
VALUES (B_id, now(), post_uuid, A_id, 'Alice', 'Hello!');
INSERT INTO home_feed (follower_id, post_time, post_id, author_id, author_name, content)
VALUES (C_id, now(), post_uuid, A_id, 'Alice', 'Hello!');

-- Read B's feed (single partition → sequential read → very fast):
SELECT * FROM home_feed WHERE follower_id = B_id LIMIT 50;
```

**Trade-off:** Fan-out on write — one post by a user with 1M followers = 1M inserts. For celebrities (>1M followers), use a hybrid pull model to avoid this.

---

**Q14. What are NewSQL databases and when do you choose them over traditional SQL or NoSQL?**

NewSQL databases aim to combine horizontal scalability of NoSQL with ACID guarantees and SQL interface of relational databases.

| | Traditional SQL | NoSQL | NewSQL |
|---|---|---|---|
| SQL support | Full | None/limited | Full |
| ACID | Yes | Often not | Yes (distributed) |
| Horizontal scale | Limited | Yes | Yes (automatic) |
| Consistency | Strong | Often eventual | Strong (globally) |
| Latency | Low (local) | Low | Higher (consensus overhead) |
| Examples | PostgreSQL, MySQL | Cassandra, MongoDB | CockroachDB, Google Spanner, TiDB |

**How NewSQL achieves distributed ACID:**
- Consensus protocol (Raft/Paxos): replication without a single primary, survives node failures automatically
- Distributed MVCC: snapshot isolation across nodes using logical timestamps (HLC)
- Cross-shard transactions: 2PC over Raft groups

**When to choose NewSQL:**
- You need horizontal write scale beyond a single PostgreSQL instance
- You need ACID transactions that span shards (not possible in manually sharded MySQL)
- You need automatic geographic distribution with strong consistency
- Your team cannot manage manual sharding complexity

**Trade-off:** Distributed ACID is expensive — a simple single-row update requires 2–3 Raft consensus round trips. Latency is 10–50ms per write (vs <1ms for local PostgreSQL). Only justified when you've outgrown vertical scaling and sharding is too complex.

---

**Q15. How do you handle schema evolution differently in SQL vs NoSQL databases?**

**SQL — explicit, versioned migrations:**
```sql
-- Migration tool (Flyway, Liquibase, Alembic) tracks applied versions
-- V1: initial schema
CREATE TABLE users (id BIGINT PRIMARY KEY, email TEXT NOT NULL);

-- V2: add phone (zero-downtime approach)
ALTER TABLE users ADD COLUMN phone TEXT;   -- nullable, no default — fast on most DBs
-- Backfill in batches (don't lock the table for one large update):
UPDATE users SET phone = NULL WHERE id BETWEEN 1 AND 100000;  -- batch by PK range
-- Later: add NOT NULL when backfill complete

-- Zero-downtime pattern for NOT NULL column:
-- Phase 1: add nullable → Phase 2: dual-write (old+new) → Phase 3: backfill →
-- Phase 4: switch reads → Phase 5: add NOT NULL → Phase 6: drop old column
```

**NoSQL — implicit, application-managed:**
```python
# MongoDB: documents with different shapes coexist in the same collection
# Version 1 documents: { "name": "Alice", "contact": { "phone": "555-1234" } }
# Version 2 documents: { "name": "Bob",  "phone": "555-5678" }

# Application must handle both shapes:
def get_phone(user_doc):
    if "contact" in user_doc:                  # v1 format
        return user_doc["contact"].get("phone")
    return user_doc.get("phone")               # v2 format

# Background migration (lazy): on first read, normalize and re-save
def get_user(user_id):
    user = db.users.find_one({"_id": user_id})
    if "contact" in user:
        phone = user["contact"].get("phone")
        db.users.update_one({"_id": user_id},
                            {"$set": {"phone": phone}, "$unset": {"contact": ""}})
        user["phone"] = phone
    return user
```

**Key difference:** SQL schema changes are explicit, auditable, and reversible via migration tools. NoSQL "changes" are implicit — old and new document shapes silently coexist, requiring application code to handle all versions.

---

## Hard (Q16–Q20)

---

**Q16. You have a PostgreSQL database that is read-bottlenecked. Walk through the full escalation path before recommending a different database.**

**Step 1: Verify the bottleneck**
```sql
-- Check slow queries
SELECT query, calls, mean_exec_time, total_exec_time
FROM pg_stat_statements
ORDER BY total_exec_time DESC LIMIT 20;

-- Check table read patterns
SELECT relname, seq_scan, seq_tup_read, idx_scan, idx_tup_fetch
FROM pg_stat_user_tables
ORDER BY seq_tup_read DESC LIMIT 10;
-- High seq_scan with large seq_tup_read → missing indexes

-- Check cache hit rate
SELECT sum(heap_blks_hit) / (sum(heap_blks_hit) + sum(heap_blks_read)) AS cache_hit
FROM pg_statio_user_tables;
-- Target: > 99%
```

**Step 2: Index optimization (no infra changes)**
```sql
-- Add missing indexes for frequent query patterns
-- Use EXPLAIN (ANALYZE, BUFFERS) to identify seq scans
EXPLAIN (ANALYZE, BUFFERS) SELECT * FROM orders WHERE customer_id = 1234;
-- If Seq Scan: CREATE INDEX CONCURRENTLY ON orders (customer_id);

-- Use covering indexes to enable Index Only Scans
CREATE INDEX CONCURRENTLY ON orders (customer_id) INCLUDE (order_date, total);
```

**Step 3: Query optimization**
- Rewrite N+1 queries to use JOINs or batch IN()
- Replace correlated subqueries with window functions
- Add CTEs for repeated subexpressions
- Use materialized views for expensive aggregations

**Step 4: Vertical scaling**
- Increase shared_buffers (target: 25% of RAM)
- Add more RAM so hot data fits in buffer pool
- Move to NVMe SSDs if on spinning disks
- Add CPU cores for parallel query execution

**Step 5: Read replicas (horizontal read scale)**
```
Add 2–3 read replicas
Route read-only queries to replicas
Route writes and read-after-write to primary
Scales read throughput N× with no application logic changes (pgBouncer for routing)
```

**Step 6: Application-level caching (Redis)**
```python
# Cache frequent read-only queries in Redis
cached = redis.get(f"product:{product_id}")
if not cached:
    product = db.query("SELECT * FROM products WHERE id = %s", product_id)
    redis.setex(f"product:{product_id}", 300, serialize(product))
```

**Step 7: Only now consider a different database**
- If still bottlenecked after steps 1–6, examine the specific query type
- Full-text search → add Elasticsearch (don't replace PostgreSQL)
- Time-series metrics → add TimescaleDB (extension on same PostgreSQL)
- Simple K/V lookups → Redis already handles this (step 6)
- Most "read bottlenecks" are solved by steps 1–6; a different database is rarely the answer

---

**Q17. Design the database layer for a multi-region e-commerce platform. Justify each database technology chosen.**

**Services and their database needs:**

```
User/Auth Service
  Database: PostgreSQL (primary region) + read replicas in other regions
  Why: ACID for account creation, login history; read replicas for profile reads globally
  Schema: users, sessions, address_book

Order Service
  Database: PostgreSQL (per-region cluster with Patroni HA)
  Why: Orders need ACID — payment + inventory deduction + order record must be atomic
  Scale: Partition orders table by month; archive older partitions to cold storage
  Note: Orders are region-specific (user in EU places order on EU cluster)

Product Catalog Service
  Database: PostgreSQL + JSONB for variable attributes
  Why: Products have relationships (categories, variants), variable attributes; JSONB + GIN handles both
  Read scaling: Materialized view for catalog listings; Redis cache for individual product pages
  CDN: Product images and descriptions cached at CDN edge globally

Inventory Service
  Database: PostgreSQL with SELECT FOR UPDATE (pessimistic locking)
  Why: Inventory decrements must be atomic — oversell is worse than a slow response
  Scale: Redis for real-time stock level reads (cache with short TTL); PostgreSQL for authoritative stock

Search Service
  Database: Elasticsearch (separate from PostgreSQL)
  Why: Full-text search, faceted filtering, fuzzy matching not efficient in SQL
  Sync: CDC from PostgreSQL (Debezium → Kafka → Elasticsearch indexer)

Analytics / Reporting
  Database: ClickHouse (columnar, OLAP)
  Why: Aggregations over billions of order/event rows in seconds
  Pipeline: PostgreSQL → Kafka (CDC) → ClickHouse

Session / Rate Limiting
  Database: Redis
  Why: Sub-millisecond session reads; INCR for rate limiting counters; TTL for session expiry
```

**Regional architecture:**
```
EU region:  PostgreSQL primary + 1 sync replica + 1 async replica + Redis + Elasticsearch
NA region:  PostgreSQL primary + 1 sync replica + 1 async replica + Redis + Elasticsearch
APAC region: PostgreSQL primary + 1 sync replica + Redis (Elasticsearch NA replica)

Cross-region:
  Product catalog: replicated to all regions via logical replication (read-only in non-primary)
  Orders: owned by the region where the customer placed the order
  User profiles: home region owns; other regions get async replica for read performance
```

---

**Q18. How would you migrate a monolithic MySQL database to a service-oriented architecture where each service owns its data?**

**Starting state:**
```
Single MySQL database: users, orders, products, inventory, payments, reviews
All services (really: one monolith) query any table freely with JOINs
```

**Target state:**
```
User Service → users_db (PostgreSQL)
Order Service → orders_db (PostgreSQL)
Product Service → products_db (PostgreSQL + JSONB)
Inventory Service → inventory_db (PostgreSQL)
```

**Migration strategy: Strangler Fig + CDC**

**Phase 1: Identify service boundaries and cross-table JOINs (weeks 1–3)**
```sql
-- Find all queries that JOIN across future service boundaries
SELECT * FROM orders o JOIN users u ON o.customer_id = u.id JOIN products p ON ...
-- These JOINs must become: API calls or event-driven denormalization
-- Document every cross-boundary dependency
```

**Phase 2: Add CDC sync from monolith to new service DB (weeks 4–8)**
```
Debezium reads MySQL binlog → publishes to Kafka per-table topics
New Order Service DB subscribes to orders + order_items topics
Orders DB is initially a read-only replica of the monolith's tables
New Order Service reads from its own DB — validate output matches monolith
```

**Phase 3: Dual-write (weeks 9–12)**
```
New Order Service writes to both its own DB and the monolith (via old code path)
Run in parallel, compare outputs for discrepancies
Gradually shift read traffic: 5% → 25% → 50% → 100% to new service
```

**Phase 4: Cut over and clean up (week 13+)**
```
New service owns all reads and writes for its domain
Monolith calls new service API for cross-domain data (replaces JOIN with API call)
Remove tables from monolith schema (after confirming no direct queries remain)
Disable CDC sync (no longer needed)
```

**Handling cross-service queries post-migration:**
```
Old: SELECT o.id, u.email, p.name FROM orders o JOIN users u ... JOIN products p ...
New Option A: Order Service API calls User Service and Product Service, merges in memory
New Option B: Order Service stores denormalized copy of user email at order creation time
New Option C: Read model (CQRS) — event-driven view that pre-joins order + user + product
```

---

**Q19. Explain the trade-offs between strong consistency, eventual consistency, and causal consistency. When is each the right choice for a database?**

**Strong consistency (linearizability):**
```
Every read returns the most recent write, regardless of which node serves the read
Write → committed → ANY subsequent read from ANY node returns the new value

Implementation cost: synchronous coordination between nodes before returning write ACK
Latency: bounded by network RTT to quorum members

Use when:
  - Financial transactions (bank balance, stock trades)
  - Inventory counts (cannot oversell)
  - Auth tokens (login/logout must be immediately reflected everywhere)
  - Leader election (only one node must think it is leader)
```

**Eventual consistency:**
```
After a write, reads may return stale data for a window of time
Eventually (given no new writes), all nodes converge to the same value
Typical staleness window: milliseconds to seconds

Implementation cost: writes acknowledged locally (fast), async replication to other nodes
Latency: very low (no cross-node coordination on write)

Use when:
  - Social media like counts, follower counts (stale by 1 second is fine)
  - DNS records (designed for eventual consistency)
  - User preference settings (stale by a few seconds is acceptable)
  - Product view counts, page impressions
```

**Causal consistency:**
```
A read always reflects all writes that causally preceded it
"You always see your own writes"
Other users may still see stale data, but not in a way that violates causality

Example: User updates profile photo → immediately sees new photo (not old one)
         Other users may still see old photo for a few seconds → acceptable
         
Implementation: vector clocks or session tokens that carry version information

Use when:
  - User sees their own updates immediately (profile, settings, preferences)
  - Collaborative editing (operations must appear in causal order)
  - Comment threads (reply must appear after the original comment)
  - Most read-your-own-writes requirements are solved by causal consistency (cheaper than strong)
```

**Comparison:**
| Level | Latency | Throughput | Complexity | Use case |
|---|---|---|---|---|
| Strong | High (RTT) | Limited by coordination | High | Finance, inventory |
| Causal | Low | High | Medium | User-facing apps |
| Eventual | Very low | Very high | Low | Social metrics, analytics |

---

**Q20. A startup is choosing a database stack for a new platform. They expect to grow from 100 to 100M users over 3 years. What do you recommend and how does the stack evolve?**

**Guiding principle:** Start simple. Complexity has a carrying cost — every additional database is another system to operate, monitor, back up, and debug. Add complexity only when you have a measurable problem it solves.

**Phase 1: 0 → 100K users (months 1–12)**
```
Single database: PostgreSQL on managed cloud (RDS, Supabase, Neon)
  - One connection pool: PgBouncer or RDS Proxy
  - Redis: session storage + API response caching
  - No sharding, no replicas beyond the managed HA standby

Stack:
  PostgreSQL (8 vCPU, 32 GB RAM) — handles all data
  Redis (2 GB) — sessions, cache
  
Cost: ~$500/month
Operational overhead: minimal
```

**Phase 2: 100K → 5M users (year 1–2)**
```
PostgreSQL still handles everything with tuning:
  - Add read replica for analytics/reports (never query production for reporting)
  - Partition large tables by date (orders, events)
  - Add indexes based on pg_stat_statements
  - Add Redis cache for user profiles, product pages (reduce DB reads by 80%)
  - Upgrade instance size (32 vCPU, 128 GB RAM)

Optional: add Elasticsearch if full-text search is a core feature
  (Don't replace PostgreSQL for search — add Elasticsearch alongside it)

Cost: ~$3K/month
```

**Phase 3: 5M → 50M users (year 2–3)**
```
PostgreSQL primary + 3 read replicas + PgBouncer
  - Most read traffic goes to replicas
  - Writes still on single primary (if < 10K writes/sec, PostgreSQL handles it)

Redis Cluster (replace single Redis with cluster for HA + scale)

If write bottleneck hits:
  - Shard by user_id across 4 PostgreSQL primaries (application-level routing)
  - OR migrate to CockroachDB (automatic sharding, same SQL interface)

Add ClickHouse for analytics (separate from OLTP):
  - CDC pipeline: PostgreSQL → Kafka → ClickHouse
  - All dashboards and analytics run against ClickHouse, never PostgreSQL

Cost: ~$15K/month
```

**Phase 4: 50M → 100M+ users**
```
At this scale: likely 3–5 PostgreSQL shards + global read replicas
Redis Cluster: 6+ nodes
CDN + caching aggressively for static data
Consider managed distributed SQL (CockroachDB, Spanner) to avoid manual shard management

Specialist databases added only for proven bottlenecks:
  Search: Elasticsearch
  Time-series: TimescaleDB or ClickHouse
  Graph: Neo4j (if social features become central)
  Cache: Redis Cluster
```

**Summary:**
```
Year 1: PostgreSQL + Redis  (simple, fast to build)
Year 2: PostgreSQL + Redis + Read Replica + Elasticsearch (if needed)
Year 3: Sharded PostgreSQL + Redis Cluster + ClickHouse + Elasticsearch
Future: Evaluate CockroachDB to remove manual sharding burden
```

---

## Quick Reference

```
Database selection framework:
  1. What are the query patterns? (point lookup / range / JOIN / traversal / aggregate)
  2. What consistency do you need? (ACID / eventual / causal)
  3. What is the write throughput? (< 10K/s → SQL; > 100K/s → consider NoSQL)
  4. Is the schema stable or variable? (stable → SQL; variable → document or JSONB)
  5. Do you need horizontal write scale? (no → stay SQL; yes → NoSQL or NewSQL)

CAP choices:
  CP: HBase, MongoDB (w:majority), Zookeeper, traditional SQL
  AP: Cassandra, CouchDB, DynamoDB (default)
  CA: single-node PostgreSQL / MySQL

Document model rules:
  Embed: always accessed together, bounded size, child doesn't update independently
  Reference: shared across parents, unbounded growth, independent lifecycle

Consistency levels:
  Strong: finance, inventory, auth
  Causal: read-your-own-writes, collaborative apps
  Eventual: social metrics, analytics, feeds

Rule: Start with PostgreSQL. Add complexity only when you have a specific, measurable problem.
```
