# Database Sharding and Partitioning

## Easy (Q1–Q7)

---

**Q1. What is the difference between partitioning and sharding?**

Both split data into smaller pieces, but at different levels of the stack:

| Aspect | Partitioning | Sharding |
|---|---|---|
| Scope | Within one database instance | Across multiple database instances (servers) |
| Transparency | Query one table; DB handles routing internally | Application or middleware must know which shard |
| Transactions | Normal ACID within same DB | Cross-shard transactions are complex / expensive |
| Use case | Query performance, manageability, archival | Data volume / write throughput beyond one server |
| Setup | SQL DDL (`PARTITION BY`) | Application code or proxy (Vitess, Citus) |

```sql
-- Partitioning: single PostgreSQL instance, multiple storage files
CREATE TABLE orders (order_id BIGINT, order_date DATE, amount NUMERIC)
PARTITION BY RANGE (order_date);

CREATE TABLE orders_2024 PARTITION OF orders
    FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');

-- The DB automatically routes INSERT/SELECT to the right partition
-- One connection string, one database — transparent to the application
```

With sharding, application code contains logic like `shard_id = hash(user_id) % 4` and connects to different database servers based on the result.

---

**Q2. What are the three main sharding strategies?**

**1. Range-based sharding**
```
Shard 0: user_id 1 – 1,000,000
Shard 1: user_id 1,000,001 – 2,000,000
Shard 2: user_id 2,000,001 – 3,000,000

Pros: range queries on shard key hit one shard (efficient)
Cons: hot spots — new user IDs always go to the last shard ("append hot spot")
      imbalanced if distribution is uneven (some ranges have more active users)
```

**2. Hash-based sharding**
```
shard_id = hash(user_id) % num_shards

user_id=1234 → hash(1234) = 7381 → 7381 % 4 = 1 → Shard 1
user_id=1235 → hash(1235) = 2190 → 2190 % 4 = 2 → Shard 2

Pros: even distribution regardless of data characteristics
Cons: range queries must fan out to all shards (cannot prune)
      rebalancing when num_shards changes requires moving ~(N-1)/N keys
```

**3. Directory-based sharding (lookup table)**
```
Routing table: user_id → shard_id (stored in a fast KV store)
user_id=1234 → look up in Redis → "shard-3"
user_id=5678 → look up in Redis → "shard-1"

Pros: maximum flexibility, can move individual users between shards
Cons: routing table is a bottleneck / single point of failure; extra lookup per request
```

---

**Q3. What is a hot shard (hot spot) and how is it avoided?**

A hot shard is one that receives a disproportionate share of reads or writes, while other shards sit idle. This defeats the purpose of sharding.

**Causes:**

1. **Time-based shard key with range sharding:** All new writes always go to the shard for "today"
2. **Sequential primary key with range sharding:** New inserts always go to the highest shard
3. **Celebrity / power-user effect:** One user with 100M followers: all their writes on one shard
4. **Popular product / content:** One product ID is 50% of traffic → its shard is overwhelmed

**Solutions:**

```
1. Hash the shard key (most common fix):
   Old: shard = user_id % 4 (range-based → sequential inserts to last shard)
   New: shard = hash(user_id) % 4 → even distribution even for sequential IDs

2. Compound shard key:
   Shard by (user_id, date_bucket) so even one user's data spreads across shards
   But: range queries for one user now require all shards — trade-off

3. Dedicated shard for large entities:
   Celebrity users → dedicated shard + fan-out-on-read (pull model)
   Regular users → standard hash sharding

4. Add random salt to shard key:
   shard = hash(user_id + random_suffix) % (4 * N)
   Reads: must query all N sub-shards and merge (scatter-gather)
```

---

**Q4. What is consistent hashing and why is it used in distributed databases?**

**Problem with naive `hash(key) % N`:**
Adding or removing a shard changes N, causing almost all keys to remap:
```
N=3: key "alice" → hash(alice) % 3 = 1 → Shard 1
N=4: key "alice" → hash(alice) % 4 = 3 → Shard 3  ← moved!
~75% of all keys need to be migrated when adding just one shard
```

**Consistent hashing solution:**
```
1. Place shards on a conceptual ring (0 to 2³²) by hashing their identifiers
2. Place each key on the ring at hash(key)
3. Key belongs to the first shard clockwise from its ring position

Adding a new shard:
  New shard takes over a contiguous range from its clockwise neighbor
  Only ~1/N keys migrate → much less disruption

Ring example (simplified 0–100):
  Shard A at 10, Shard B at 40, Shard C at 70

  key hashes to 25 → clockwise neighbor = B → stored on B
  key hashes to 55 → clockwise neighbor = C → stored on C
  key hashes to 85 → wraps around → clockwise neighbor = A → stored on A

Adding Shard D at position 55:
  Previously: ring 41–70 was C's responsibility
  Now: ring 41–55 → D, ring 56–70 → C
  Only keys in range 41–55 need to migrate (~15% of all keys, not 75%)
```

**Virtual nodes (vnodes):** Each physical shard occupies multiple positions on the ring (e.g., 150 positions). This ensures even data distribution even when shard capacities differ, and allows fine-grained rebalancing.

Used by: Cassandra (vnodes), DynamoDB, Riak, Redis Cluster (hash slots).

---

**Q5. What are the types of table partitioning available in PostgreSQL and MySQL?**

**PostgreSQL declarative partitioning:**

```sql
-- Range partitioning (most common for time-series data)
CREATE TABLE orders (order_id BIGINT, order_date DATE, amount NUMERIC)
PARTITION BY RANGE (order_date);
CREATE TABLE orders_2024_q1 PARTITION OF orders FOR VALUES FROM ('2024-01-01') TO ('2024-04-01');
CREATE TABLE orders_2024_q2 PARTITION OF orders FOR VALUES FROM ('2024-04-01') TO ('2024-07-01');

-- Hash partitioning (even distribution, no range queries on partition key)
CREATE TABLE users (user_id BIGINT, name TEXT)
PARTITION BY HASH (user_id);
CREATE TABLE users_p0 PARTITION OF users FOR VALUES WITH (modulus 4, remainder 0);
CREATE TABLE users_p1 PARTITION OF users FOR VALUES WITH (modulus 4, remainder 1);
CREATE TABLE users_p2 PARTITION OF users FOR VALUES WITH (modulus 4, remainder 2);
CREATE TABLE users_p3 PARTITION OF users FOR VALUES WITH (modulus 4, remainder 3);

-- List partitioning (specific values per partition)
CREATE TABLE sales (sale_id BIGINT, region TEXT, amount NUMERIC)
PARTITION BY LIST (region);
CREATE TABLE sales_na PARTITION OF sales FOR VALUES IN ('US', 'CA', 'MX');
CREATE TABLE sales_eu PARTITION OF sales FOR VALUES IN ('DE', 'FR', 'GB', 'IT');
```

**MySQL partitioning:**
```sql
-- Range
CREATE TABLE orders (order_id INT, order_date DATE, amount DECIMAL(10,2))
PARTITION BY RANGE (YEAR(order_date)) (
    PARTITION p2022 VALUES LESS THAN (2023),
    PARTITION p2023 VALUES LESS THAN (2024),
    PARTITION p2024 VALUES LESS THAN (2025),
    PARTITION pmax  VALUES LESS THAN MAXVALUE
);

-- Hash
CREATE TABLE users (user_id INT, name VARCHAR(100))
PARTITION BY HASH (user_id)
PARTITIONS 8;
```

---

**Q6. What is partition pruning and how does it improve query performance?**

Partition pruning is the database's ability to skip partitions entirely when the query's WHERE clause can eliminate them based on the partition key.

```sql
-- Table partitioned by order_date (range, one partition per quarter)
SELECT SUM(amount)
FROM orders
WHERE order_date BETWEEN '2024-01-01' AND '2024-03-31';

-- Without pruning: scan all 8 partitions (2 years of data)
-- With pruning: only scan orders_2024_q1 partition → 4–8x faster

-- EXPLAIN shows partition pruning:
EXPLAIN SELECT SUM(amount) FROM orders WHERE order_date BETWEEN '2024-01-01' AND '2024-03-31';
-- Output: "Partitions: orders_2024_q1"  ← only one partition scanned
```

**Conditions for pruning to work:**
- WHERE clause must directly reference the partition key column
- The operator must be compatible with the partition type (range → `<`, `>`, `BETWEEN`; list → `=`, `IN`)
- Functions or type casts on the partition key **prevent** pruning:
  ```sql
  WHERE DATE_TRUNC('month', order_date) = '2024-01-01'  -- NO pruning (function wraps column)
  WHERE order_date >= '2024-01-01'                       -- YES pruning (direct column comparison)
  ```

---

**Q7. What problems does sharding introduce that do not exist in a single-database architecture?**

**1. Cross-shard queries / JOINs**
```
Old: SELECT o.*, u.email FROM orders o JOIN users u ON o.customer_id = u.id
     Works on one DB with one query

New with sharding by user_id:
  order may be on Shard 1, user may be on Shard 2
  Must: query Shard 1 for order → query Shard 2 for user → merge in application
  → N+1 queries across shards (scatter-gather)
```

**2. Cross-shard transactions**
```
Transfer $100 from user A (Shard 1) to user B (Shard 2):
  Old: BEGIN; UPDATE... COMMIT; — single ACID transaction
  New: requires distributed 2PC or compensating transactions (Saga)
```

**3. Global unique IDs**
```
Old: BIGSERIAL auto-increment — DB ensures uniqueness
New: each shard generates its own IDs — shard 1 and shard 2 both generate order_id = 1
Fix: use UUIDs, or Snowflake IDs (shard_id + timestamp + sequence), or a central ID generator
```

**4. Schema changes / DDL**
```
Old: ALTER TABLE orders ADD COLUMN ... — one operation
New: must run ALTER TABLE on all N shards simultaneously (or in rolling fashion)
     Tools like Vitess coordinate DDL across shards
```

**5. Rebalancing**
```
Old: add more disk / RAM to one server
New: adding a shard requires migrating data (potentially terabytes) across shards
     Live migration needed: move data + dual-write during transition
```

---

## Medium (Q8–Q15)

---

**Q8. Design the sharding strategy for a multi-tenant SaaS application where 10 customers generate 80% of the traffic.**

**Problem:** Standard hash sharding distributes rows evenly by key, but 10 large customers will each land on shards they share with other customers, and those shards become hot.

**Solution: Tiered sharding**

```
Architecture:
  Tier A — Dedicated shards (10 shards, one per large customer):
    Each large customer gets their own PostgreSQL shard with full resources
    Dedicated shard can be independently scaled (larger instance for larger customer)
    
  Tier B — Shared shards (8 shards for 9,990 small customers):
    hash(customer_id) % 8 → shard assignment
    Each small customer is too small to fill a shard alone
    
Routing table (stored in Redis or config DB):
  customer_id → shard_id
  "bigcorp"    → dedicated_shard_1   (explicit mapping)
  "midcompany" → dedicated_shard_5   (explicit mapping)
  "smallco"    → compute: hash("smallco") % 8 → shared_shard_3
```

**Migration path when a small customer grows large:**
```
1. Provision new dedicated shard
2. Set up logical replication from shared shard → dedicated shard (for this customer's tables)
3. Wait for replication to catch up
4. Brief dual-write period + validation
5. Atomic cutover: update routing table in Redis (single write, ~1ms)
6. Drop customer data from shared shard
Total user-visible downtime: < 1 second
```

**Schema enforcement:**
```sql
-- All tables include tenant_id as first column of PK to enforce data locality
CREATE TABLE orders (
    customer_id  INT  NOT NULL,
    order_id     BIGINT NOT NULL,
    ...
    PRIMARY KEY (customer_id, order_id)
);
-- Row-level security at DB level as additional safety:
ALTER TABLE orders ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON orders USING (customer_id = current_setting('app.customer_id')::INT);
```

---

**Q9. How does Cassandra's partitioning work and how do you design a partition key for a time-series workload?**

**Cassandra partitioning fundamentals:**
- The **partition key** determines which node(s) store the row (via consistent hashing on the partition key's token)
- All rows with the same partition key are stored together on the same node(s), sorted by the clustering key
- A partition is the unit of data locality — all reads and writes for one partition go to the same set of nodes
- Partitions must fit in available node memory for efficient reads; recommended max partition size: 100MB / 100K rows

**Designing for time-series:**

```sql
-- BAD design: single partition per device (partition grows unboundedly)
CREATE TABLE sensor_readings (
    device_id  UUID,
    read_time  TIMESTAMP,
    value      DOUBLE,
    PRIMARY KEY (device_id, read_time)
);
-- After 1 year × 1 reading/sec = 31M rows per device → partition too large → performance degrades

-- GOOD design: bucket by time period (limits partition size)
CREATE TABLE sensor_readings (
    device_id  UUID,
    time_bucket TEXT,           -- '2024-01-15' (day bucket) or '2024-01-15-14' (hour bucket)
    read_time  TIMESTAMP,
    value      DOUBLE,
    PRIMARY KEY ((device_id, time_bucket), read_time)
) WITH CLUSTERING ORDER BY (read_time DESC);

-- Query last hour of data for one device:
SELECT * FROM sensor_readings
WHERE device_id = ? AND time_bucket = '2024-01-15-14'
ORDER BY read_time DESC LIMIT 100;
-- One partition read — fast

-- Query last 24 hours (24 partition reads — acceptable):
SELECT * FROM sensor_readings
WHERE device_id = ? AND time_bucket IN ('2024-01-15-00', '2024-01-15-01', ..., '2024-01-15-23');
```

**Bucket sizing calculation:**
```
1 reading/sec × 3600 sec/hr × 200 bytes = 720 KB/hr per device
Hour bucket → 720 KB per partition → well within limits

For 1000 readings/sec per device:
  Hour bucket → 720 MB → too large → use minute bucket
  Minute bucket → 12 MB → acceptable
```

---

**Q10. What is a fanout query and how do you minimize it when using sharding?**

A fanout query (scatter-gather) is one that must be sent to multiple or all shards because the query predicate does not include the shard key. Each shard returns partial results; the application or middleware merges them.

```
Sharded by user_id (8 shards)

Query WITH shard key (no fanout):
  SELECT * FROM orders WHERE user_id = 1234 AND order_date > '2024-01-01'
  → hash(1234) % 8 = 2 → send only to Shard 2 → fast

Query WITHOUT shard key (full fanout):
  SELECT SUM(amount) FROM orders WHERE order_date > '2024-01-01'
  → Must query all 8 shards, get 8 partial sums, sum them in the application
  → 8× latency + 8× DB load compared to a single-shard query
```

**Strategies to minimize fanout:**

**1. Choose shard key based on most frequent query pattern:**
```
If 90% of queries filter by user_id → shard by user_id (eliminates fanout for 90%)
Remaining 10% (admin reports, analytics) can tolerate fanout
```

**2. Secondary index shards / routing shards:**
```
Main shards: by user_id
Separate "search shard" or Elasticsearch: indexed by order_date
  → "All orders from 2024" → query Elasticsearch → get list of (user_id, order_id)
  → use user_id to route follow-up queries to correct shard (N individual queries, each fast)
```

**3. Move analytics to separate OLAP system:**
```
Fanout queries are typically analytical
→ CDC pipeline: sharded OLTP DB → Kafka → ClickHouse
→ Analytics queries go to ClickHouse (no sharding complexity, columnar scan is fast)
→ OLTP shards serve only targeted lookups (no fanout)
```

**4. Vitess scatter-gather (MySQL sharding middleware):**
```
Vitess can automatically fan out a query to all shards and merge results
Useful for: aggregate queries, cross-shard JOINs
Trade-off: higher latency than single-shard, but transparent to the application
```

---

**Q11. How does partition maintenance work in practice? Walk through adding and removing partitions.**

**Adding a new partition (common: time-based partitions need regular new partitions):**

```sql
-- PostgreSQL: add future quarter partition before it starts receiving data
-- Best practice: create next partition in advance (don't wait until first INSERT)
CREATE TABLE orders_2025_q1 PARTITION OF orders
    FOR VALUES FROM ('2025-01-01') TO ('2025-04-01');

-- Automate with a cron job or pg_partman extension:
-- pg_partman: automatically creates future partitions and drops old ones
SELECT partman.create_parent('public.orders', 'order_date', 'native', 'quarterly');
-- Configure: premake = 4 (always have 4 future partitions ready)
```

**Detaching an old partition (archival / deletion):**
```sql
-- Detach without dropping (archive to cold storage):
ALTER TABLE orders DETACH PARTITION orders_2022;
-- After detach: orders_2022 is now a standalone table (not a partition)
-- Dump it to cold storage (S3 via pg_dump) and then drop:
\copy orders_2022 TO '/tmp/orders_2022.csv' CSV HEADER
DROP TABLE orders_2022;

-- Fast delete: dropping a partition is instant (drop the file, update catalog)
-- vs DELETE FROM orders WHERE order_date < '2023-01-01' which scans rows
ALTER TABLE orders DETACH PARTITION orders_2022;
DROP TABLE orders_2022;   -- instantaneous file-level delete
```

**ATTACH existing table as partition (bulk loading):**
```sql
-- Load data into a standalone table (no partition overhead):
CREATE TABLE orders_2025_q2 (LIKE orders INCLUDING ALL);
COPY orders_2025_q2 FROM '/data/orders_2025_q2.csv' CSV;
-- Validate constraints:
ALTER TABLE orders_2025_q2 ADD CONSTRAINT chk_date
    CHECK (order_date >= '2025-04-01' AND order_date < '2025-07-01') NOT VALID;
VALIDATE CONSTRAINT chk_date;
-- Attach (instant — PostgreSQL trusts the validated constraint):
ALTER TABLE orders ATTACH PARTITION orders_2025_q2
    FOR VALUES FROM ('2025-04-01') TO ('2025-07-01');
```

---

**Q12. What is the "write to multiple shards" problem during shard rebalancing and how is it handled?**

When adding a new shard (or rebalancing existing shards), data must be migrated from old shards to new ones. During migration, writes must go to the right place while a given key's data is in transit.

**The problem without a strategy:**
```
Migration running: moving user 1234's data from Shard 1 → Shard 3
  
  Write arrives for user 1234:
    Old routing: shard = hash(1234) % 3 = Shard 1  ← data being moved from here
    New routing: shard = hash(1234) % 4 = Shard 3  ← data moving here
    
  If routing switches before migration completes: writes go to Shard 3 but data isn't there yet
  If routing stays on Shard 1 during migration: new writes arrive on old shard, not being migrated
```

**Solution: Double-write during migration**

```
Phase 1: Begin background data copy
  Worker copies rows for user 1234 from Shard 1 to Shard 3
  Meanwhile, all writes still go to Shard 1 (routing unchanged)

Phase 2: Enable dual-write
  Routing layer routes writes for user 1234 to BOTH Shard 1 and Shard 3
  Reads still from Shard 1 (consistent)
  Shard 3 catches up to Shard 1 from the background copy

Phase 3: Switch reads to Shard 3
  Verify Shard 3 data matches Shard 1 (row count, checksum)
  Switch reads to Shard 3
  Continue dual-writes for a brief transition period

Phase 4: Stop writes to Shard 1
  Single-write to Shard 3 only
  Shard 1 data for user 1234 can be cleaned up

Total user impact: zero (no writes lost, reads always work)
```

**Tools that implement this:** Vitess (VReplication), Citus (live resharding), DynamoDB (automatic, managed).

---

**Q13. How does Citus (distributed PostgreSQL) handle sharding transparently?**

Citus is a PostgreSQL extension that transforms a single PostgreSQL cluster into a distributed database. It adds a coordinator node and multiple worker nodes.

```
Architecture:
  Coordinator node: receives all SQL queries, holds metadata about shard locations
  Worker nodes (N): each stores a subset of shards, runs full PostgreSQL

Query routing:
  Single-shard query (partition key in WHERE):
    Coordinator → routes to one worker → worker executes → returns result
    
  Multi-shard query (no partition key, GROUP BY, etc.):
    Coordinator → sends modified query to ALL workers in parallel
    Workers execute their partial query → return partial results
    Coordinator → merges results → returns to client
    
Application:
  Connects to coordinator only
  Uses normal PostgreSQL SQL
  No application changes needed for single-shard queries
```

**Distributed tables:**
```sql
-- Distribute a table by user_id (shard key):
SELECT create_distributed_table('orders', 'customer_id');

-- Citus creates N shards (default 32) and distributes them across workers
-- Each shard is a real PostgreSQL table: orders_102001, orders_102002, ...

-- Co-location: tables with the same distribution column and value on the same worker
-- customers and orders sharded by customer_id → same customer_id → same worker
-- JOINs between co-located tables execute locally on the worker (no cross-network JOIN)
SELECT create_distributed_table('customers', 'customer_id');
-- Now: JOIN orders + customers on customer_id → executes on each worker locally
```

**Reference tables (small, fully replicated):**
```sql
-- Tables that are joined frequently but don't need sharding:
SELECT create_reference_table('product_categories');
-- Full copy on every worker → JOINs with distributed tables don't need network hops
```

---

**Q14. What is a global secondary index in DynamoDB and why is it important for avoiding full-table scans?**

In DynamoDB, the primary table's access pattern is defined by its partition key (and optional sort key). Queries without the partition key require a full table scan (extremely expensive in DynamoDB).

**Global Secondary Index (GSI):** A separate, automatically maintained copy of part of the table, with a different partition key and sort key. Allows efficient access by a different attribute.

```
Primary table:
  Partition key: order_id
  Sort key: none
  Access: GET /orders/order_id  ← efficient (point lookup)
  Access: all orders for customer_id ← REQUIRES FULL SCAN (no customer_id partition key)

Without GSI:
  SELECT * FROM orders WHERE customer_id = 'C123'
  → Scans entire table → O(N) → expensive and slow

With GSI on customer_id:
  GSI partition key: customer_id
  GSI sort key: order_date
  DynamoDB maintains a separate index table automatically
  
  Query: get all orders for customer C123, sorted by date
  GSI Query: { IndexName: "customer_orders_gsi", KeyConditionExpression: "customer_id = :cid" }
  → O(log N) → fast
```

**Design rules:**
- Choose GSI partition key to have **high cardinality** (many distinct values) for even distribution
- Include projected attributes to avoid extra fetches (or use ALL projection)
- Each GSI consumes separate read/write capacity

```
Table: orders
  PK: order_id
  GSI 1: customer_id + order_date  → "all orders for a customer, newest first"
  GSI 2: status + created_at       → "all pending orders by creation time" (for processing queue)
  GSI 3: product_id + order_date   → "all orders containing a product"
```

**Local Secondary Index (LSI):** Same partition key, different sort key. Can only be created at table creation time. Limited to 10GB per partition key.

---

**Q15. How do you handle transactions that span multiple shards?**

This is the hardest problem in sharded databases. Several approaches exist, each with different trade-offs.

**Option A: Avoid cross-shard transactions by design (best)**
```
Design shard key so related data is always on the same shard
  Shard by customer_id: all of one customer's orders + payments → same shard
  → A customer placing an order + payment deduction = single-shard ACID transaction
  
Denormalize: store everything needed for a transaction in one shard
  Order table includes: customer_email, product_snapshot, shipping_address
  (denormalized at order creation time)
  → No need to look up customer or product during payment processing
```

**Option B: Two-Phase Commit (2PC) — strong consistency, high cost**
```
Coordinator sends PREPARE to Shard 1 and Shard 2
Both shards lock rows and respond PREPARED
Coordinator sends COMMIT to both
Both apply changes

Cost: 2 network round trips to each shard
      Locks held across network round trips
      If coordinator crashes between PREPARE and COMMIT: shards stuck indefinitely (in-doubt)
Use when: must have strong consistency and cannot redesign data model
```

**Option C: Saga pattern — eventual consistency, no locks**
```
Saga: sequence of local transactions with compensating actions on failure

Transfer $100 from Shard 1 (account A) to Shard 2 (account B):
  Step 1: Shard 1 — debit $100 from A, mark as "transfer_pending"  ← local ACID
  Step 2: Shard 2 — credit $100 to B                               ← local ACID
  If Step 2 fails:
    Compensation: Shard 1 — credit $100 back to A, mark "transfer_cancelled"

Consistency: brief window where A is debited but B is not yet credited
Use when: can tolerate brief inconsistency; don't need exact real-time balance
```

**Option D: NewSQL (CockroachDB, Spanner) — transparent distributed ACID**
```
Use a database that handles cross-shard transactions natively
No application-level coordination needed
Cost: higher write latency (consensus overhead per transaction)
```

---

## Hard (Q16–Q20)

---

**Q16. Design the complete sharding architecture for a payments platform processing 50,000 transactions per second globally.**

**Analysis:**
- Financial transactions → must be ACID
- 50K TPS globally → single PostgreSQL (~10K TPS) not sufficient
- Users are in NA, EU, APAC → geo-distribution needed
- Cross-user transactions (A→B): one "debit" + one "credit" operation

**Shard key choice: account_id**
```
account_id determines shard — all operations for one account go to one shard
Single-account transactions (balance check, deposit): always single-shard → ACID trivially

Cross-account transfer: two-shard problem
  account A on Shard 2, account B on Shard 7
  Solution: see transfer design below
```

**Architecture:**
```
Region: NA (US-East + US-West)
  8 PostgreSQL shards (each: primary + 1 sync standby)
  Shard assignment: hash(account_id) % 8 (among NA accounts)

Region: EU (Frankfurt + Amsterdam)
  6 PostgreSQL shards

Region: APAC (Singapore + Tokyo)
  4 PostgreSQL shards

Total: 18 shard primaries + 18 standbys = 36 PostgreSQL instances

Routing:
  account_id → region (from account_region field in a global routing DB)
  account_id → shard_id (hash(account_id) % shards_in_region)
  
  Global routing DB: tiny PostgreSQL cluster (just account_id → region mapping)
                     replicated globally with logical replication (read-only copies in each region)
```

**Transfer between accounts (cross-shard):**
```
Strategy: Use an "Outbox" pattern + eventual consistency for the split
  
  Step 1 (Shard A): Debit account A within one transaction:
    BEGIN;
    UPDATE accounts SET balance = balance - 100, status = 'pending_transfer' WHERE id = A;
    INSERT INTO outbox (transfer_id, from_account, to_account, amount, status)
           VALUES (uuid, A, B, 100, 'debit_complete');
    COMMIT;
    
  Step 2: Transfer processor reads outbox, credits account B:
    On Shard B:
    BEGIN;
    UPDATE accounts SET balance = balance + 100 WHERE id = B;
    INSERT INTO transfer_log (transfer_id, status) VALUES (uuid, 'complete');
    COMMIT;
    
    Update outbox on Shard A: mark transfer_id as 'complete'

  Failure handling:
    If Step 2 fails: retry with idempotency (transfer_id ensures deduplication)
    If Step 2 never succeeds: reconciliation job detects stuck outbox entries → refund / alert

  Consistency window: milliseconds to seconds (not zero — this is a trade-off)
  Alternative: if zero-latency strong consistency required across shards → CockroachDB
```

---

**Q17. How would you implement a live resharding process on a production database with zero downtime?**

**Scenario:** 4 shards at capacity → need to expand to 8 shards. Cannot take downtime.

**Phase 0: Preparation**
```
1. Provision 4 new shard servers (shards 4–7)
2. Initialize empty databases on new shards (same schema)
3. Plan new routing: hash(key) % 8 → new shard assignment
4. Identify which keys in existing shards need to move:
   Old shard 0 held keys where hash(key) % 4 = 0
   New routing: half of those go to shard 0, half to shard 4
```

**Phase 1: Background data copy**
```
For each key range moving to a new shard:
  1. Copy rows in batches (1000 rows at a time, with rate limiting to not overload source)
  2. Track the "high watermark" — which rows have been copied
  3. Track all writes to source shard during copy (via triggers or WAL)
```

**Phase 2: Dual-write for migrating keys**
```
Update routing layer to dual-write:
  Writes for key X → sent to both OLD shard AND NEW shard
  Reads still from OLD shard (authoritative)
  
  New shard catches up to old shard via background copy + dual-write
```

**Phase 3: Verify and switch reads**
```
Verification:
  Compare row counts: SELECT COUNT(*) on old and new shard for migrated key ranges
  Run checksum: compare hash of critical tables

Switch reads:
  Update routing layer: reads for migrated keys → NEW shard
  Brief (< 100ms) window: routing update propagation
```

**Phase 4: Stop writing to old shard for migrated keys**
```
Update routing: single-write to NEW shard only
Delete migrated data from old shard:
  DELETE FROM orders WHERE hash(customer_id) % 8 IN (4, 5, 6, 7) -- keys now on new shards
  (can be done lazily over hours to avoid I/O spike)
```

**Tools used for this in practice:**
- **Vitess (MySQL):** VReplication component automates this exact process
- **Citus (PostgreSQL):** `rebalance_table_shards()` does this automatically
- **DynamoDB:** Automatic transparent resharding (no user action needed)

---

**Q18. What are the consistency challenges when using read replicas alongside a sharded primary cluster?**

**Setup:**
```
Shard 0: Primary → Replica 0A, Replica 0B
Shard 1: Primary → Replica 1A, Replica 1B
Application: writes to primaries, reads from replicas (load balancing)
```

**Challenge 1: Read-after-write on the same shard**
```
Write to Shard 0 Primary: INSERT INTO orders (user_id=1, amount=100)
Read from Shard 0 Replica 0A: SELECT * FROM orders WHERE user_id=1
  → Replica 0A may be 200ms behind → user doesn't see their own order
  
Fix: route the user's reads to Shard 0 Primary for 1 second after any write
     redis.setex(f"primary_read:{user_id}", 1, shard_id)
```

**Challenge 2: Cross-shard read consistency**
```
Query: "Show me all orders for user 1 AND their loyalty points"
  Orders on Shard 0 Replica
  Loyalty points on Shard 3 Replica

Shard 0 Replica is 0ms behind (sync)
Shard 3 Replica is 500ms behind (high load on Shard 3)

Result: orders from T=10:00:00.500, loyalty points from T=10:00:00.000
→ Inconsistent snapshot across shards — not possible to prevent without distributed snapshots
  
Mitigation: accept this inconsistency (usually fine for display purposes)
            or: read loyalty points from Shard 3 Primary
            or: include read timestamp in response, show "(data as of 10:00:00)"
```

**Challenge 3: Replica used for shard routing decisions**
```
Global routing table (which account is on which shard) is replicated to all regions
If routing table replica lags:
  New account created → assigned to Shard 4
  Routing replica hasn't received the update
  Request for this new account → routing replica says "not found" → wrong shard
  
Fix: routing table reads MUST go to primary (or use synchronous replication for routing table)
```

---

**Q19. How does DynamoDB's automatic sharding differ from managing shards in PostgreSQL?**

**DynamoDB (managed sharding):**
```
Physical partitions are invisible to you
DynamoDB handles splitting and moving partitions automatically

Provisioned capacity:
  WCU (Write Capacity Units): 1 WCU = 1 write of ≤ 1KB/second
  RCU (Read Capacity Units):  1 RCU = 1 strongly consistent read of ≤ 4KB/second

Automatic partition management:
  Each partition handles max: 3000 RCU, 1000 WCU, or 10GB
  When a partition exceeds: DynamoDB automatically splits it
  No operator action needed, no data migration visible to you

Throttling (the main problem):
  If one partition key receives > 1000 WCU/sec: that partition is throttled
  → Hot partition → requests fail with ProvisionedThroughputExceededException
  → Fix: choose high-cardinality partition key or add random suffix (shard by 10 sub-keys)
```

**PostgreSQL (manual sharding):**
```
You control everything:
  Define shards (separate database instances)
  Implement routing logic in application code or via proxy (Citus, Vitess, PgPool)
  Monitor shard utilization manually
  Add shards and migrate data manually (or via tooling)
  Handle failover per shard independently

Advantages:
  Full control over shard boundaries
  Complex queries and JOINs within a shard
  No throttling limits — just hardware limits
  ACID transactions within a shard

Disadvantages:
  Operational complexity: N shards = N times the operational overhead
  Rebalancing requires manual work or specialized tooling
  No automatic hot spot detection or splitting
```

**Comparison:**
| | DynamoDB | PostgreSQL Sharding |
|---|---|---|
| Shard management | Automatic | Manual |
| Hot spot handling | Add RCU/WCU; design shard key | Manual shard splitting or rebalancing |
| JOINs | Limited (scatter-gather) | Full SQL within a shard |
| Operational overhead | Near zero | High |
| Cost at scale | High per request | Low (hardware only) |
| Transaction model | Single-item ACID; multi-item via transactions (extra cost) | Full ACID within shard |

---

**Q20. Describe a real-world scenario where choosing the wrong shard key destroyed performance and how it was fixed.**

**Scenario: Social platform — timeline sharded by post_id**

```
Initial design:
  Table: posts (post_id, author_id, content, created_at)
  Shard key: post_id
  Reasoning: "posts are the main entity, shard by their ID"

Working pattern:
  "Show me the last 50 posts by user A" → most common query
  
Query execution with post_id sharding:
  SELECT * FROM posts WHERE author_id = 'A' ORDER BY created_at DESC LIMIT 50
  
  post_id is random UUIDs → uniformly distributed across 8 shards
  User A's posts: randomly spread across all 8 shards
  
  Execution:
    Send query to all 8 shards
    Each shard scans its subset for author_id = 'A'
    8 partial result sets returned to coordinator
    Sort and merge 8 × 50 = 400 rows → return top 50
  
  Performance: 8× fanout on EVERY timeline read
  At 100K timeline reads/second: 800K shard queries/second → shards overloaded
```

**Root cause analysis:**
```
The access pattern (filter by author_id) doesn't match the shard key (post_id)
Every read requires a full scatter-gather across all shards
Indexes on author_id exist on each shard but cannot help eliminate shard fanout
```

**Fix: Re-shard by author_id**
```
New shard key: author_id
  hash(author_id) % 8 → shard assignment
  All posts by author_id = 'A' → same shard

New query execution:
  hash('A') % 8 = 3 → query only Shard 3
  Shard 3 has index on (author_id, created_at DESC) → fast index scan
  Return 50 rows → no merging needed
  
Performance: single-shard read, index scan → 100× faster than scatter-gather

Migration process (zero downtime):
  1. Build new shards alongside existing ones
  2. Background re-copy all posts to new sharding scheme (by author_id)
  3. Dual-write to both old (post_id shards) and new (author_id shards) during transition
  4. Switch reads to new shards (verify row counts match)
  5. Stop writes to old shards, clean up
```

**Lesson:** The shard key must match your most frequent and performance-critical query's primary filter. Shard by the attribute you most commonly query by, not by the entity's "natural" identifier.

---

## Quick Reference

```
Sharding strategies:
  Range:     hot spots possible, range queries efficient
  Hash:      even distribution, range queries require fanout
  Directory: maximum flexibility, routing table overhead

Hot spot prevention:
  Hash the shard key (don't use sequential IDs with range sharding)
  Dedicated shards for large tenants
  Time bucket + ID compound key for time-series

Partitioning types (within one DB):
  RANGE:  time-based data, partition pruning on date
  HASH:   even distribution within DB
  LIST:   explicit values (region, status)

Partition pruning requires:
  Direct column reference in WHERE (no functions wrapping the column)
  Operator compatible with partition type

Cross-shard transactions:
  Avoid by design (co-locate related data)
  2PC: strong consistency, high cost, coordinator risk
  Saga: eventual consistency, compensating transactions

Consistent hashing:
  Adding 1 shard → ~1/N keys migrate (vs ~(N-1)/N with modular hashing)
  Virtual nodes → even distribution with heterogeneous hardware

Rebalancing (zero downtime):
  Background copy → dual-write → switch reads → single-write → cleanup
  Tools: Vitess VReplication, Citus rebalance_table_shards
```
