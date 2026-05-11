# Database Design and Selection

## Table of Contents
1. [RDBMS Fundamentals](#rdbms-fundamentals)
2. [NoSQL Types](#nosql-types)
3. [Database Selection Criteria](#database-selection-criteria)
4. [CAP Theorem](#cap-theorem)
5. [BASE vs ACID](#base-vs-acid)
6. [Database Replication](#database-replication)
7. [Database Sharding](#database-sharding)
8. [Consistent Hashing for Database Sharding](#consistent-hashing-for-database-sharding)
9. [Hot Spots and Mitigation](#hot-spots-and-mitigation)
10. [Read Replicas and CQRS](#read-replicas-and-cqrs)
11. [Indexing Strategy](#indexing-strategy)
12. [Query Optimization](#query-optimization)
13. [Distributed Transactions](#distributed-transactions)
14. [NewSQL](#newsql)
15. [Database Migration Strategies](#database-migration-strategies)
16. [Polyglot Persistence](#polyglot-persistence)
17. [Quick Reference](#quick-reference)

---

## RDBMS Fundamentals

### ACID Properties

Every transaction in a relational database must satisfy ACID:

**Atomicity** — A transaction is all-or-nothing. Either all operations succeed, or none are applied.

```sql
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;  -- debit
UPDATE accounts SET balance = balance + 100 WHERE id = 2;  -- credit
COMMIT;  -- both succeed

-- If second UPDATE fails:
ROLLBACK;  -- first UPDATE is also undone
```

**Consistency** — A transaction brings the database from one valid state to another. All constraints, rules, and cascades are enforced.

```sql
-- Constraint: balance cannot go negative
ALTER TABLE accounts ADD CONSTRAINT chk_balance CHECK (balance >= 0);

-- Transaction violating this will fail entirely (atomicity)
BEGIN;
UPDATE accounts SET balance = balance - 1000 WHERE id = 1;  -- fails constraint
ROLLBACK;  -- balance unchanged
```

**Isolation** — Concurrent transactions execute as if they are serial. Isolation levels control the degree of visibility:

| Isolation Level | Dirty Read | Non-Repeatable Read | Phantom Read |
|---|---|---|---|
| READ UNCOMMITTED | Possible | Possible | Possible |
| READ COMMITTED | No | Possible | Possible |
| REPEATABLE READ | No | No | Possible (MySQL: No) |
| SERIALIZABLE | No | No | No |

```sql
-- PostgreSQL: set isolation level per transaction
BEGIN ISOLATION LEVEL REPEATABLE READ;
SELECT balance FROM accounts WHERE id = 1;
-- ... other operations ...
COMMIT;
```

**Durability** — Committed transactions are permanent. Data survives crashes (stored in WAL — Write-Ahead Log).

### Normalization

Normalization reduces data redundancy and ensures data integrity.

**First Normal Form (1NF):** Eliminate repeating groups; each column contains atomic values.

```sql
-- Violates 1NF (multiple values in one column)
user_id | hobbies
1       | "reading, hiking, coding"

-- 1NF compliant
user_id | hobby
1       | reading
1       | hiking
1       | coding
```

**Second Normal Form (2NF):** 1NF + every non-key attribute is fully dependent on the entire primary key (no partial dependency).

```sql
-- Violates 2NF: product_name depends only on product_id, not full key (order_id, product_id)
order_id | product_id | product_name | quantity

-- 2NF compliant: separate tables
orders:   order_id, product_id, quantity
products: product_id, product_name
```

**Third Normal Form (3NF):** 2NF + no transitive dependencies (non-key attribute depending on another non-key attribute).

```sql
-- Violates 3NF: zip_code -> city (non-key depends on non-key)
employee_id | zip_code | city

-- 3NF compliant:
employees: employee_id, zip_code
zip_codes:  zip_code, city
```

**BCNF (Boyce-Codd Normal Form):** Stronger version of 3NF. Every determinant is a candidate key.

**When to denormalize:** For read-heavy analytical workloads, denormalization (intentional redundancy) improves query performance by avoiding expensive JOINs.

### JOINs

```sql
-- INNER JOIN: only matching rows from both tables
SELECT u.name, o.total
FROM users u INNER JOIN orders o ON u.id = o.user_id;

-- LEFT JOIN: all users, even those with no orders
SELECT u.name, COALESCE(o.total, 0) as total
FROM users u LEFT JOIN orders o ON u.id = o.user_id;

-- RIGHT JOIN: all orders (rarely used; LEFT JOIN preferred)

-- FULL OUTER JOIN: all rows from both tables
SELECT u.name, o.total
FROM users u FULL OUTER JOIN orders o ON u.id = o.user_id;

-- CROSS JOIN: Cartesian product (every combination)
SELECT a.color, b.size FROM colors a CROSS JOIN sizes b;

-- SELF JOIN: table joined to itself
SELECT e.name AS employee, m.name AS manager
FROM employees e LEFT JOIN employees m ON e.manager_id = m.id;
```

### Indexes

An index is a data structure (usually B-tree) that speeds up data retrieval at the cost of additional write overhead and storage.

```sql
-- Single column index
CREATE INDEX idx_users_email ON users(email);

-- Composite index (order matters: leftmost prefix rule)
CREATE INDEX idx_orders_user_status ON orders(user_id, status);
-- This index is used for:
--   WHERE user_id = 123
--   WHERE user_id = 123 AND status = 'pending'
-- NOT used for:
--   WHERE status = 'pending' (only right side)

-- Unique index (enforces uniqueness constraint)
CREATE UNIQUE INDEX idx_users_email_unique ON users(email);

-- Partial index (only index rows matching a condition)
CREATE INDEX idx_orders_pending ON orders(user_id) WHERE status = 'pending';
-- Only indexes pending orders, not all orders

-- Covering index (includes extra columns to avoid table lookup)
CREATE INDEX idx_orders_covering ON orders(user_id) INCLUDE (total, created_at);
-- Query SELECT total, created_at FROM orders WHERE user_id = 123 uses index only
```

---

## NoSQL Types

### Document Databases (MongoDB, Firestore, CouchDB)

Store data as JSON/BSON documents. Schema-flexible.

```javascript
// MongoDB document
{
  _id: ObjectId("..."),
  userId: "user123",
  name: "Alice",
  address: {                    // embedded document (no JOIN needed)
    street: "123 Main St",
    city: "New York",
    zip: "10001"
  },
  orderHistory: [               // array (no separate table needed)
    { orderId: "ord1", total: 99.99, date: "2024-01-15" },
    { orderId: "ord2", total: 149.99, date: "2024-02-01" }
  ],
  tags: ["premium", "active"]
}
```

**Strengths:** Flexible schema, natural fit for hierarchical data, easy horizontal scaling.
**Weaknesses:** No JOINs (must denormalize or use `$lookup`), eventual consistency by default, no multi-document ACID without explicit transactions.

**Best for:** Content management, user profiles, product catalogs, event logging.

### Key-Value Stores (DynamoDB, Redis, Riak)

Simplest NoSQL model. Keys map to opaque values. All access is by primary key.

```python
# DynamoDB
table.put_item(Item={'user_id': '123', 'name': 'Alice', 'score': 9500})
table.get_item(Key={'user_id': '123'})

# DynamoDB with composite key
table.get_item(Key={'user_id': '123', 'timestamp': '2024-01-15T10:00:00Z'})
```

**Strengths:** Extremely fast O(1) reads/writes, massively scalable, simple data model.
**Weaknesses:** No queries beyond key lookup (no range queries unless using sort key), no aggregations.

**Best for:** Session storage, user preferences, real-time leaderboards, shopping carts.

### Wide-Column Stores (Cassandra, HBase, Google Bigtable)

Data is organized in rows and columns, but unlike RDBMS, each row can have a different set of columns. Optimized for time-series and write-heavy workloads.

```
Cassandra table structure:
Partition key: (user_id)  -- determines which node stores this row
Clustering key: (timestamp)  -- ordering within a partition

user_id | timestamp           | event_type | data
--------|---------------------|------------|-----
user1   | 2024-01-15 10:00:00 | click      | {...}
user1   | 2024-01-15 10:00:05 | purchase   | {...}
user2   | 2024-01-15 10:01:00 | view       | {...}
```

```sql
-- Cassandra CQL
CREATE TABLE user_events (
    user_id TEXT,
    timestamp TIMESTAMP,
    event_type TEXT,
    data TEXT,
    PRIMARY KEY (user_id, timestamp)
) WITH CLUSTERING ORDER BY (timestamp DESC);

-- Query (must include partition key)
SELECT * FROM user_events WHERE user_id = 'user1' LIMIT 100;
SELECT * FROM user_events WHERE user_id = 'user1' AND timestamp > '2024-01-01';
```

**Strengths:** Excellent write throughput, linear horizontal scaling, no single point of failure, multi-datacenter replication.
**Weaknesses:** Query patterns must be designed upfront (denormalize for each query), no JOINs, limited aggregation.

**Best for:** IoT telemetry, time-series data, audit logs, messaging systems, write-heavy workloads.

### Graph Databases (Neo4j, Amazon Neptune, JanusGraph)

Store data as nodes and edges (relationships). Optimized for traversing relationships.

```cypher
-- Neo4j Cypher query
// Create nodes and relationship
CREATE (alice:Person {name: "Alice"})
CREATE (bob:Person {name: "Bob"})
CREATE (alice)-[:FOLLOWS]->(bob)

// Find friends of friends (2 hops)
MATCH (user:Person {name: "Alice"})-[:FOLLOWS*2]->(fof:Person)
WHERE NOT (user)-[:FOLLOWS]->(fof)
RETURN fof.name

// Shortest path
MATCH path = shortestPath(
  (alice:Person {name: "Alice"})-[:FOLLOWS*]-(bob:Person {name: "Bob"})
)
RETURN path
```

**Strengths:** Extremely efficient for relationship queries (graph traversal), natural data model for networks.
**Weaknesses:** Does not scale horizontally as easily, expensive for non-relationship queries, less familiar query language.

**Best for:** Social networks (friends, followers), recommendation engines, fraud detection, knowledge graphs, access control.

### Time-Series Databases (InfluxDB, TimescaleDB, Prometheus)

Optimized for sequential, time-indexed data with high write throughput.

```sql
-- InfluxDB (InfluxQL)
-- Automatic time partitioning and compression
INSERT cpu_metrics,host=server1 value=72.5 1699999999000000000

SELECT mean(value) FROM cpu_metrics
WHERE host = 'server1' AND time > now() - 1h
GROUP BY time(5m)

-- TimescaleDB (PostgreSQL extension)
SELECT time_bucket('5 minutes', time) AS bucket, avg(value)
FROM cpu_metrics
WHERE host = 'server1' AND time > NOW() - INTERVAL '1 hour'
GROUP BY bucket ORDER BY bucket;
```

**Strengths:** Optimized storage (delta encoding, compression for sequential values), fast time-range queries, built-in downsampling.
**Weaknesses:** Limited query flexibility beyond time-based access, not suitable for general-purpose workloads.

**Best for:** Metrics, monitoring, IoT sensor data, financial tick data, system logs with timestamps.

---

## Database Selection Criteria

### Decision Framework

```
1. What is the data model?
   - Tabular with relationships -> RDBMS
   - Hierarchical/nested       -> Document DB
   - Key-value pairs           -> Key-Value store
   - Time-series               -> Time-series DB
   - Highly connected          -> Graph DB
   - Wide sparse rows          -> Wide-column

2. What are the access patterns?
   - Complex queries, arbitrary filters -> RDBMS
   - Access by primary key only         -> Key-value
   - Range queries on time              -> Time-series / Cassandra
   - Graph traversal                    -> Graph DB

3. What are the consistency requirements?
   - Strong consistency, ACID            -> RDBMS or NewSQL
   - Eventual consistency acceptable     -> Cassandra, DynamoDB
   - Strong + scale                      -> CockroachDB, Spanner

4. What is the expected scale?
   - < 10M rows, moderate traffic       -> RDBMS (PostgreSQL, MySQL)
   - Billions of rows, high write rate  -> Cassandra, DynamoDB
   - Petabytes                          -> HBase, Bigtable

5. What is the team's operational capability?
   - Managed service preferred           -> DynamoDB, Cloud SQL, Atlas
   - Self-managed, DBA expertise         -> PostgreSQL, MySQL
```

---

## CAP Theorem

### The Theorem

In a distributed data system, you can guarantee at most **2 of 3** properties:

- **C (Consistency):** Every read receives the most recent write or an error.
- **A (Availability):** Every request receives a response (not necessarily the most recent data).
- **P (Partition Tolerance):** The system continues operating despite network partitions.

**Key insight:** Network partitions WILL occur in any distributed system. Therefore, P is not optional — you always have P. The real choice is between **CP** (consistency + partition tolerance) and **AP** (availability + partition tolerance).

```
CAP Triangle:
          C
         / \
        /   \
       CA   CP
      /       \
     A----AP----P
```

**CA (Consistency + Availability without Partition Tolerance):**
Possible only in single-node systems. Any distributed system must tolerate network partitions. CA databases: single-node PostgreSQL, MySQL (not in cluster mode).

### CP Databases

During a network partition, a CP database refuses to answer queries (or returns errors) to guarantee consistency.

```
Partition occurs between nodes A and B.
CP behavior: Node A stops accepting writes until partition heals.
Result: Unavailability during partition, but no stale reads.
```

**Examples:** HBase, Zookeeper, etcd, Redis (in most configurations), MongoDB (with write concern majority).

### AP Databases

During a network partition, an AP database continues to answer queries but may return stale data.

```
Partition occurs between nodes A and B.
AP behavior: Both nodes accept reads and writes independently.
Result: After partition heals, conflicts must be resolved.
```

**Examples:** Cassandra, DynamoDB, CouchDB, Riak.

### Real-World CAP Examples

| Database | Type | Behavior During Partition |
|---|---|---|
| PostgreSQL (single node) | CA | N/A (not distributed) |
| PostgreSQL (multi-primary) | CP | Refuses write on minority partition |
| MySQL Group Replication | CP | Minority partition goes read-only |
| MongoDB | CP | Primary unavailable = no writes |
| Cassandra | AP | All nodes accept reads/writes |
| DynamoDB | AP (default) | Eventually consistent reads |
| DynamoDB (strong read) | CP | May fail during partition |
| HBase | CP | Master unavailable = no operations |
| CockroachDB | CP | Consensus-based, refuses on partition |
| Redis Cluster | CP | Minority shards go unavailable |
| Zookeeper | CP | Minority partition refuses operations |

### CAP is Not Binary

Modern databases often offer tunable consistency. Cassandra example:

```python
# Tunable consistency in Cassandra
from cassandra import ConsistencyLevel

# Strong consistency (like CP)
session.execute(query, consistency_level=ConsistencyLevel.QUORUM)

# Eventual consistency (like AP, higher availability)
session.execute(query, consistency_level=ConsistencyLevel.ONE)

# All (maximum consistency, minimum availability)
session.execute(query, consistency_level=ConsistencyLevel.ALL)
```

---

## BASE vs ACID

### BASE Properties (NoSQL alternative to ACID)

- **BA (Basically Available):** System guarantees availability (response, possibly stale).
- **S (Soft State):** State may change over time even without new input (due to eventual consistency).
- **E (Eventual Consistency):** The system will eventually become consistent, given no new updates.

```
Example (Cassandra):
  t=0:  Write to node A: user balance = 100
  t=1:  Read from node B: balance = 50 (old value - SOFT STATE)
  t=2:  Replication completes
  t=3:  Read from node B: balance = 100 (EVENTUAL CONSISTENCY achieved)
```

### When is Eventual Consistency Acceptable?

| Scenario | Eventual Consistency OK? |
|---|---|
| Bank balance transfer | NO — must be ACID |
| Social media "likes" count | YES — a few seconds delay is fine |
| Shopping cart item add | YES (with conflict resolution) |
| Inventory count (last 1 item) | NO — oversell risk |
| User profile bio update | YES |
| Ad impression counter | YES |
| Password change | NO — security requires immediate consistency |

---

## Database Replication

### Primary-Replica (Master-Slave) Replication

One primary node accepts all writes. One or more replica nodes receive a copy of all writes asynchronously.

```
                PRIMARY (read/write)
               /        |          \
         REPLICA 1   REPLICA 2   REPLICA 3
        (read only)  (read only)  (read only)
```

**Asynchronous replication (default in MySQL, PostgreSQL):**
```
Primary: writes to WAL, acknowledges client immediately
Replica: applies WAL in background (may lag by milliseconds to seconds)

Pros: Low write latency on primary
Cons: Replica may serve stale data, data loss if primary crashes before replication
```

**Synchronous replication:**
```
Primary: writes to WAL, waits for at least one replica to confirm
Primary: acknowledges client only after replica confirms

Pros: Zero data loss (at least one replica has the data)
Cons: Write latency increases by one RTT to replica
```

PostgreSQL: `synchronous_commit = on` + `synchronous_standby_names = 'replica1'`

### Replication Lag

The delay between a write on the primary and its appearance on the replica.

**Causes:** Network latency, replica hardware slower than primary, large transactions, DDL operations.

**Impact:**
- Reads from replica may return stale data.
- Reads-after-writes may be inconsistent (user writes then reads from replica, sees old data).

**Mitigation strategies:**
```python
# Read-your-writes: route reads after a write to primary temporarily
def update_user_profile(user_id, data):
    primary_db.execute("UPDATE users SET ...", user_id, data)
    # For the next 5 seconds, read this user from primary
    cache.set(f"read_from_primary:{user_id}", True, ttl=5)

def get_user_profile(user_id):
    if cache.get(f"read_from_primary:{user_id}"):
        return primary_db.query("SELECT * FROM users WHERE id = %s", user_id)
    return replica_db.query("SELECT * FROM users WHERE id = %s", user_id)
```

### Multi-Primary (Multi-Master) Replication

Multiple nodes accept writes simultaneously. Conflicts can occur.

```
    PRIMARY A          PRIMARY B
  (accepts writes)   (accepts writes)
       |                   |
   REPLICAS            REPLICAS

Conflict example:
  A: UPDATE users SET name='Alice' WHERE id=1
  B: UPDATE users SET name='Alicia' WHERE id=1 (same row, same time)
  -> Conflict! Which write wins?
```

**Conflict resolution strategies:**
- **Last-write-wins (LWW):** The write with the latest timestamp wins (risk: clock skew).
- **Application-level resolution:** Application defines merge logic.
- **CRDT (Conflict-free Replicated Data Types):** Data structures designed to merge without conflict (counters, sets).

**Practical use:** Geographically distributed databases (Cassandra, DynamoDB Global Tables, CockroachDB), where latency to a single primary would be too high.

---

## Database Sharding

Sharding partitions data across multiple database instances (shards) so that each shard holds only a subset of the total data.

### Range-Based Sharding

Partition data by a range of the shard key.

```
Shard 1: user_id 1 – 10,000,000
Shard 2: user_id 10,000,001 – 20,000,000
Shard 3: user_id 20,000,001 – 30,000,000

Query: user_id = 15,000,000 -> Shard 2 (exactly one shard)
Query: user_id BETWEEN 9,000,000 AND 11,000,000 -> Shard 1 + Shard 2
```

**Pros:** Good for range queries. Easy to understand. Sequential scans are efficient.
**Cons:** Uneven distribution if data is not uniformly distributed. New users always write to the "latest" shard (hot shard).

**Best for:** Time-series data (shard by date), sequential IDs with uniform distribution.

### Hash-Based Sharding

Apply a hash function to the shard key. The hash modulo the number of shards determines the target shard.

```
shard_id = hash(user_id) % num_shards

user_id = 123 -> hash = 789456 -> 789456 % 4 = 0 -> Shard 0
user_id = 456 -> hash = 234567 -> 234567 % 4 = 3 -> Shard 3
```

**Pros:** Even data distribution, eliminates hot spots for write-heavy workloads.
**Cons:** Range queries require all shards. Adding/removing shards requires re-hashing all data.

**Best for:** Even distribution requirements, high write throughput.

### Directory-Based Sharding

A lookup table (directory) maps each key to a shard.

```
Directory table:
  user_id 1-1000    -> Shard A
  user_id 1001-2000 -> Shard B
  user_id 2001      -> Shard C (moved due to rebalancing)

Query: Consult directory first, then route to correct shard.
```

**Pros:** Flexible — can move individual rows between shards. Easy to rebalance.
**Cons:** Directory is a single point of failure (must be replicated). Extra hop for every query (cache the directory).

### Sharding Challenges

**Cross-shard queries:**
```sql
-- Single shard (fast):
SELECT * FROM users WHERE user_id = 123  -- goes to one shard

-- Cross-shard (slow, complex):
SELECT u.name, COUNT(o.id)
FROM users u JOIN orders o ON u.id = o.user_id
GROUP BY u.name
-- Must query ALL shards and aggregate results
```

**Cross-shard transactions:**
```
Move $100 from user A (shard 1) to user B (shard 2)
Requires distributed transaction (2PC or Saga)
Complex, slow, and error-prone
```

**Rebalancing:** When adding shards, some data must move to new shards. During migration:
- Old shards must remain available.
- Double-write during migration window.
- Verify data integrity after migration.

---

## Consistent Hashing for Database Sharding

Standard hash-based sharding requires remapping all keys when shard count changes. Consistent hashing minimizes this.

### How It Works

Map both shards and keys to positions on a virtual ring (0 to 2^32).

```
Ring positions:
  Shard A: 90°
  Shard B: 210°
  Shard C: 330°

Key K -> hash(K) -> position on ring
        -> assign to the first shard clockwise

Adding Shard D at 150°:
  Only keys between 90° and 150° need to move from Shard B to Shard D
  All other keys remain unchanged
```

**Virtual nodes:** Each physical shard maps to multiple ring positions (vnodes). This prevents uneven load when the shard pool is small.

```python
class ConsistentHashRing:
    def __init__(self, vnodes_per_shard=150):
        self.vnodes_per_shard = vnodes_per_shard
        self.ring = {}  # position -> shard_id
        self.sorted_positions = []
    
    def add_shard(self, shard_id):
        for i in range(self.vnodes_per_shard):
            position = hash(f"{shard_id}:{i}") % (2**32)
            self.ring[position] = shard_id
        self.sorted_positions = sorted(self.ring.keys())
    
    def get_shard(self, key):
        position = hash(key) % (2**32)
        # Find first position >= key position (clockwise)
        import bisect
        idx = bisect.bisect_left(self.sorted_positions, position)
        if idx == len(self.sorted_positions):
            idx = 0  # wrap around
        return self.ring[self.sorted_positions[idx]]
```

---

## Hot Spots and Mitigation

### What Is a Hot Spot?

A hot spot occurs when a disproportionate amount of traffic is directed to a single shard or node, overloading it while others are idle.

**Causes:**
- Popular data (celebrity user with millions of followers)
- Sequential shard key (all new inserts go to the same shard)
- Poorly chosen partition key
- Time-based sharding with only the "current time" shard being written

### Mitigation Strategies

**1. Add a random suffix to the shard key (write spreading):**

```python
# Original: all writes for celebrity user go to one shard
shard_key = f"user:{celebrity_id}"

# Spread across 10 shards
import random
suffix = random.randint(0, 9)
shard_key = f"user:{celebrity_id}:{suffix}"  # 10 shards now share the load

# Reads: must query all 10 shards and merge
shard_keys = [f"user:{celebrity_id}:{i}" for i in range(10)]
```

**2. Dedicated hot-key caching:**

```python
# Hot keys served from cache, never hit the database
def get_celebrity_data(celebrity_id):
    # Check cache first (handles >99% of reads)
    cached = redis.get(f"celebrity:{celebrity_id}")
    if cached:
        return json.loads(cached)
    
    data = db.query(celebrity_id)
    redis.setex(f"celebrity:{celebrity_id}", 60, json.dumps(data))
    return data
```

**3. Monotonic key anti-pattern (use UUIDs instead of auto-increment for distributed):**

```sql
-- Bad: all inserts go to the "latest" range shard
user_id SERIAL PRIMARY KEY  -- 1, 2, 3, 4, 5...

-- Good: distributed across all shards
user_id UUID DEFAULT gen_random_uuid()  -- random distribution
-- Or: use time-ordered UUIDs (UUIDv7) for locality without hotspot
user_id UUID DEFAULT gen_uuidv7()
```

---

## Read Replicas and CQRS

### Read Replicas

Replicas that serve read-only queries, offloading the primary.

```
Primary (write + strong-read)
    |-- Replica 1 (reads: non-critical queries)
    |-- Replica 2 (reads: reporting, analytics)
    |-- Replica 3 (reads: search indexing)
```

```python
class DatabaseRouter:
    def get_connection(self, query_type: str):
        if query_type == "write" or query_type == "critical_read":
            return primary_connection
        elif query_type == "analytics":
            return analytics_replica_connection  # potentially larger lag
        else:
            return read_replica_connection
```

### CQRS (Command Query Responsibility Segregation)

Separate the write model (commands) from the read model (queries). Each is optimized for its use case.

```
Command side:                      Query side:
  Normalized database                Denormalized read models
  Optimized for writes               Optimized for reads
  ACID transactions                  Eventual consistency

User places order (COMMAND):
  -> Write to orders table (normalized)
  -> Publish OrderPlaced event
  -> Event handler updates read models:
       - order_summary_view (denormalized)
       - user_order_history_view
       - inventory_view
```

```python
# Command handler (write side)
class PlaceOrderCommand:
    def execute(self, order_data):
        order = Order.create(order_data)
        db.save(order)  # normalized, consistent write
        event_bus.publish(OrderPlaced(order))  # trigger read model update

# Query handler (read side - from denormalized projection)
class GetOrderHistoryQuery:
    def execute(self, user_id):
        # Read from a pre-built, denormalized view
        return read_db.query(
            "SELECT * FROM user_order_history_view WHERE user_id = %s ORDER BY created_at DESC LIMIT 20",
            user_id
        )
```

**Benefits of CQRS:**
- Read model can be a different database type (e.g., Elasticsearch for search).
- Each side scales independently.
- Read models are denormalized — no expensive JOINs on reads.
- Multiple read models optimized for different use cases.

**Costs of CQRS:**
- Eventual consistency between write and read models.
- More complex architecture.
- Need to handle read model projection failures.

---

## Indexing Strategy

### B-Tree Indexes (Default)

B-tree indexes support equality and range queries. Used for most indexes.

```sql
-- Effective for:
WHERE id = 123                    -- equality
WHERE age > 25                    -- range
WHERE name LIKE 'Alice%'          -- prefix (not %Alice%)
WHERE created_at BETWEEN x AND y  -- range
ORDER BY created_at               -- sorted access
```

### Composite Index — Column Order Matters

The leftmost prefix rule: a composite index (a, b, c) can be used for queries on (a), (a, b), or (a, b, c), but NOT for (b) or (c) alone.

```sql
CREATE INDEX idx_orders ON orders(user_id, status, created_at);

-- Uses index: ✓
SELECT * FROM orders WHERE user_id = 1;
SELECT * FROM orders WHERE user_id = 1 AND status = 'pending';
SELECT * FROM orders WHERE user_id = 1 AND status = 'pending' AND created_at > '2024-01-01';

-- Does NOT use index: ✗
SELECT * FROM orders WHERE status = 'pending';  -- user_id not included
SELECT * FROM orders WHERE created_at > '2024-01-01';  -- user_id not included

-- Rule: Put equality conditions before range conditions in composite index
CREATE INDEX idx_orders_optimal ON orders(user_id, status, created_at);
-- WHERE user_id = 1 AND status = 'pending' AND created_at > '2024-01-01'
-- -> user_id (equality), status (equality), created_at (range) -> all 3 columns used
```

### Covering Index

An index that includes all columns needed by a query, eliminating the need to read the actual table row.

```sql
-- Without covering index:
SELECT name, email FROM users WHERE age = 25;
-- Step 1: B-tree on age -> finds row pointers
-- Step 2: Follow each pointer to read name and email from table (expensive)

-- With covering index:
CREATE INDEX idx_users_age_covering ON users(age) INCLUDE (name, email);
-- Step 1: B-tree on age -> finds name and email directly in index
-- Step 2: No table read needed (index-only scan)
```

### Partial Index

Index only a subset of rows matching a condition. Smaller, faster, especially for low-cardinality filtered columns.

```sql
-- Without partial index: index ALL orders (millions of rows)
CREATE INDEX idx_orders_user ON orders(user_id);

-- With partial index: only index pending orders (< 1% of rows)
CREATE INDEX idx_orders_user_pending ON orders(user_id) WHERE status = 'pending';
-- Much smaller index, much faster for:
-- WHERE user_id = 123 AND status = 'pending'
```

### Functional Index

Index the result of an expression or function.

```sql
-- Case-insensitive search
CREATE INDEX idx_users_lower_email ON users(LOWER(email));
SELECT * FROM users WHERE LOWER(email) = 'alice@example.com';

-- Index on JSON field
CREATE INDEX idx_events_type ON events((data->>'event_type'));
```

---

## Query Optimization

### EXPLAIN ANALYZE

Always use EXPLAIN ANALYZE to understand how the query planner executes a query.

```sql
EXPLAIN ANALYZE
SELECT u.name, COUNT(o.id) as order_count
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE u.created_at > '2024-01-01'
GROUP BY u.id, u.name
ORDER BY order_count DESC
LIMIT 10;

-- Output shows:
-- Seq Scan vs Index Scan (want Index Scan for large tables)
-- actual rows vs estimated rows (large discrepancy = bad statistics, run ANALYZE)
-- actual time (find bottlenecks)
-- Hash Join vs Nested Loop vs Merge Join
```

### N+1 Query Problem

Fetching a list of records, then making one query per record to fetch related data.

```python
# N+1 Problem: 1 query for users + 1 query per user for their orders
users = db.query("SELECT * FROM users LIMIT 100")  # 1 query
for user in users:
    orders = db.query("SELECT * FROM orders WHERE user_id = %s", user.id)  # 100 queries!
# Total: 101 queries

# Solution: JOIN or IN clause
users_with_orders = db.query("""
    SELECT u.*, o.id AS order_id, o.total
    FROM users u
    LEFT JOIN orders o ON u.id = o.user_id
    LIMIT 100
""")  # 1 query, all data

# Or: batch fetch
user_ids = [u.id for u in users]
orders_by_user = defaultdict(list)
all_orders = db.query("SELECT * FROM orders WHERE user_id = ANY(%s)", user_ids)  # 1 query
for order in all_orders:
    orders_by_user[order.user_id].append(order)
```

### Pagination Strategies

**Offset-based pagination (simple but slow):**

```sql
SELECT * FROM posts ORDER BY created_at DESC LIMIT 20 OFFSET 1000;
-- Problem: DB must scan 1020 rows to return 20. Slow for large offsets.
-- Also: new inserts may cause duplicates/skips across pages
```

**Keyset/cursor-based pagination (preferred):**

```sql
-- First page:
SELECT * FROM posts ORDER BY created_at DESC, id DESC LIMIT 20;

-- Next page (pass last row's values as cursor):
SELECT * FROM posts
WHERE (created_at, id) < ('2024-01-15 10:00:00', 12345)  -- cursor
ORDER BY created_at DESC, id DESC
LIMIT 20;
-- DB uses index from the cursor position: O(log n + page_size)
-- No duplicates/skips regardless of concurrent inserts
```

---

## Distributed Transactions

### Two-Phase Commit (2PC)

A coordinator ensures all participants either commit or rollback.

```
Phase 1 — Prepare:
  Coordinator: "Can you commit transaction T?"
  Participant 1: "Yes, prepared"
  Participant 2: "Yes, prepared"

Phase 2 — Commit:
  Coordinator: "Commit transaction T"
  Participant 1: Commits
  Participant 2: Commits

If any participant says NO in Phase 1:
  Coordinator: "Rollback transaction T"
```

**Problems with 2PC:**
- **Blocking:** If coordinator crashes after Phase 1, participants are stuck in prepared state.
- **Performance:** Two round trips before commit; locks held during the entire process.
- **Availability:** Coordinator is a single point of failure.

**Used by:** PostgreSQL distributed extensions, XA transactions, Google Spanner (with Paxos for durability).

### Saga Pattern

Break a distributed transaction into a sequence of local transactions, each with a compensating transaction (rollback action).

```
Choreography-based Saga (event-driven):

Order Service: CreateOrder -> emit OrderCreated
Payment Service: on OrderCreated -> ProcessPayment -> emit PaymentProcessed
Inventory Service: on PaymentProcessed -> ReserveInventory -> emit InventoryReserved
Shipping Service: on InventoryReserved -> ScheduleShipment -> emit ShipmentScheduled

Compensation (if ScheduleShipment fails):
Shipping Service: emit ShipmentFailed
Inventory Service: on ShipmentFailed -> ReleaseInventory -> emit InventoryReleased
Payment Service: on InventoryReleased -> RefundPayment -> emit PaymentRefunded
Order Service: on PaymentRefunded -> CancelOrder
```

```
Orchestration-based Saga (centralized coordinator):

SagaOrchestrator:
  1. CreateOrder (Order Service)
  2. ProcessPayment (Payment Service)
     - On failure: compensate step 1 (CancelOrder)
  3. ReserveInventory (Inventory Service)
     - On failure: compensate step 2 (RefundPayment), then step 1
  4. ScheduleShipment (Shipping Service)
     - On failure: compensate step 3, 2, 1
```

**Pros:** No distributed locks, services are loosely coupled, each step uses local ACID transactions.
**Cons:** Eventual consistency — the system may be inconsistent between steps. Compensating transactions are complex and may fail. No isolation (other transactions may see intermediate state).

---

## NewSQL

NewSQL databases provide SQL semantics with horizontal scalability and distributed ACID transactions.

### CockroachDB

Distributed SQL database inspired by Google Spanner.

```
Architecture:
  - Multiple nodes, each with a subset of data (ranges)
  - Raft consensus for each range (3 replicas)
  - Serializable isolation by default
  - Standard PostgreSQL wire protocol

CRDB cluster:
  Node 1: Ranges [0, 1B] (3 replicas across nodes 1, 2, 3)
  Node 2: Ranges [1B, 2B]
  Node 3: Ranges [2B, 3B]
```

```sql
-- Standard SQL works on CockroachDB
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT;  -- Distributed ACID transaction across any nodes
```

**When to use:** Need SQL + horizontal scale + ACID, multi-region deployments with strong consistency.
**Not for:** Analytical queries (use a data warehouse), when latency of distributed consensus is unacceptable.

### Google Spanner

Google's globally distributed, externally consistent relational database.

- **TrueTime API:** Atomic clock + GPS receivers ensure bounded clock uncertainty.
- **External consistency:** Globally consistent reads and writes.
- **Automatically sharded:** Scales to petabytes.
- **SQL support:** ANSI SQL.

Available via **Cloud Spanner** on GCP.

---

## Database Migration Strategies

### Zero-Downtime Migration Principles

Never run destructive schema changes directly in production. Use the expand-contract pattern.

### Expand-Contract (Blue-Green Schema)

**Phase 1 — Expand (backward-compatible change):**
```sql
-- Add new column with default value (nullable or with default)
ALTER TABLE users ADD COLUMN full_name VARCHAR(255);
-- Application writes to BOTH old (first_name, last_name) and new (full_name)
```

**Phase 2 — Migrate data:**
```sql
-- Backfill in batches (avoid long-running transactions)
UPDATE users SET full_name = first_name || ' ' || last_name
WHERE full_name IS NULL
LIMIT 1000;  -- run in batches with LIMIT
-- Run this repeatedly until all rows have full_name
```

**Phase 3 — Deploy new application:**
```
Application now reads from full_name only
Application still writes to both old and new columns (for rollback safety)
```

**Phase 4 — Contract (cleanup):**
```sql
-- After verifying new code works for N days:
ALTER TABLE users DROP COLUMN first_name;
ALTER TABLE users DROP COLUMN last_name;
ALTER TABLE users ALTER COLUMN full_name SET NOT NULL;
```

### Online Schema Changes

For large tables, `ALTER TABLE` can lock the table for hours. Use tools:

```bash
# pt-online-schema-change (Percona Toolkit for MySQL)
pt-online-schema-change --alter "ADD INDEX idx_email (email)" \
  D=mydb,t=users --execute

# gh-ost (GitHub's online schema change for MySQL)
gh-ost --user="root" --password="pass" --host=127.0.0.1 \
  --database="mydb" --table="users" \
  --alter="ADD INDEX idx_email (email)" --execute

# PostgreSQL: pg_repack for large table rebuilds
pg_repack -d mydb -t users
```

### Blue-Green Database Deployments

Maintain two identical database environments. Switch traffic between them.

```
Current:  App -> Blue DB (v1 schema)
Migrate:  Copy data from Blue to Green DB
          Apply v2 schema to Green DB
          Sync ongoing changes (Blue -> Green CDC)
Switch:   App -> Green DB (v2 schema)
Rollback: App -> Blue DB (if issues arise within N hours)
```

---

## Polyglot Persistence

Using multiple database technologies in a single application, each optimized for its specific use case.

### Example: E-Commerce Platform

```
User Authentication:
  -> PostgreSQL (ACID, relational, user table)

Product Catalog:
  -> MongoDB (flexible schema, hierarchical product attributes, easy search)

Shopping Cart:
  -> Redis (fast in-memory, TTL-based session, low latency)

Order Processing:
  -> PostgreSQL (ACID transactions, financial data integrity)

Product Search:
  -> Elasticsearch (full-text search, faceted filtering)

Recommendations:
  -> Neo4j (graph: "users who bought X also bought Y")

Analytics / Reporting:
  -> BigQuery / Redshift (columnar, analytical queries on large datasets)

Metrics / Monitoring:
  -> InfluxDB / TimescaleDB (time-series metrics, dashboards)

User Activity Logs:
  -> Cassandra (write-heavy, append-only, time-ordered)
```

### When to Use Polyglot Persistence

- Different parts of your system have genuinely different data needs.
- Single-database performance has been proven insufficient for specific use cases.
- Team has expertise to operate multiple databases.
- The operational overhead is worth the performance/feature gains.

### When NOT to Use Polyglot Persistence

- Team is small — operational complexity may outweigh benefits.
- Adding a database for a minor use case that existing DB handles "good enough."
- Cross-database transactions are required (data consistency becomes very complex).
- Early-stage product — optimize for development speed, not perfect data model.

**Rule of thumb:** Start with PostgreSQL. Add specialized databases only when you have concrete evidence of a bottleneck or missing feature.

---

## Quick Reference

### SQL vs NoSQL Decision Matrix

| Requirement | Recommended DB |
|---|---|
| Complex queries with JOINs | RDBMS (PostgreSQL, MySQL) |
| ACID transactions required | RDBMS or NewSQL (CockroachDB) |
| Schema evolves frequently | Document DB (MongoDB) |
| Horizontal write scalability | Cassandra, DynamoDB |
| Sub-millisecond reads by key | Redis, DynamoDB |
| Full-text search | Elasticsearch |
| Relationship traversal | Neo4j |
| Time-series / metrics | InfluxDB, TimescaleDB |
| Global distribution + SQL | CockroachDB, Google Spanner |
| Multi-region, AP | Cassandra, DynamoDB |
| Multi-region, CP | CockroachDB, Spanner |

### CAP Examples Table

| Database | CAP Type | During Network Partition |
|---|---|---|
| PostgreSQL (single) | CA | N/A (not distributed) |
| CockroachDB | CP | Minority partitions reject writes |
| Google Spanner | CP | Uses TrueTime, partition-sensitive operations fail |
| MongoDB (majority write) | CP | Minority loses write capability |
| Cassandra | AP | All nodes accept reads/writes |
| DynamoDB (eventual) | AP | Continues operating, stale reads possible |
| DynamoDB (strong) | CP | May reject reads during partition |
| HBase | CP | Unavailable without HMaster |
| Redis Cluster | CP | Minority shards unavailable |
| Zookeeper | CP | Minority partition refuses all operations |
| CouchDB | AP | Multi-master, conflict resolution later |
| Riak | AP | All nodes accept reads/writes |

### Database Sharding Strategy Quick Guide

```
Access pattern is always by single key?
  YES -> Hash sharding (even distribution, no range queries needed)
  NO  ->
    Is the access pattern time-ordered?
      YES -> Range sharding by time (efficient time-range queries)
      NO  ->
        Need flexible rebalancing?
          YES -> Directory-based sharding
          NO  -> Hash sharding with consistent hashing (minimize reshuffle on scale)

Worried about hot spots on hash sharding?
  -> Use virtual nodes (150+ per physical shard) in consistent hashing
  -> Or: add random salt to popular keys (write spreading)
```

### Indexing Quick Rules

```
Always index:
  - Primary key (automatic)
  - Foreign keys used in JOINs
  - Columns used in WHERE clauses on large tables
  - Columns used in ORDER BY + LIMIT (avoid sort)

Composite index column order:
  1. Equality conditions first
  2. Range condition last
  (user_id, status, created_at) for WHERE user_id=X AND status=Y AND created_at>Z

Avoid over-indexing:
  - Every index slows writes (INSERT, UPDATE, DELETE)
  - Unused indexes waste memory and disk
  - Run pg_stat_user_indexes to find unused indexes

Use partial indexes for:
  - Status columns with one dominant value (status = 'active' for 95% of rows -> partial on 'inactive')
  - Soft-delete patterns (WHERE deleted_at IS NULL)

Use covering indexes when:
  - Same columns are repeatedly queried together
  - Can eliminate table heap access for hot queries
```
