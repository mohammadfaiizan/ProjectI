# HLD Interview Q&A — File 13: Scalability Patterns

---

## Easy Questions (Q1–Q7)

---

### Q1. How does design differ for read-heavy vs write-heavy systems?

**Answer:**

The fundamental bottleneck differs between read-heavy and write-heavy systems, which drives radically different architectural decisions.

**Read-heavy systems (e.g., social media feed, e-commerce catalog, news site):**

Characteristics: 100:1 or 1000:1 read-to-write ratio. Reads must be fast; slight staleness acceptable.

Design strategies:
- **Read replicas:** Route all reads to replicas, writes to primary.
- **Caching:** Add Redis/Memcached in front of DB. Cache hit rate > 95% is the goal.
- **CDN:** Static assets and even API responses cached at edge.
- **Denormalization:** Pre-join data so reads don't need expensive JOINs.
- **Search index:** Elasticsearch for complex query workloads.

**Write-heavy systems (e.g., metrics ingestion, logging, IoT sensor data, financial transactions):**

Characteristics: Writes arrive at high volume; read queries are less frequent but must be consistent.

Design strategies:
- **Write-optimized storage:** LSM-tree databases (Cassandra, RocksDB) instead of B-tree (MySQL, PostgreSQL).
- **Write buffering:** Batch small writes into larger writes (Kafka as write buffer, then batch insert).
- **Sharding:** Distribute writes across multiple nodes to avoid single-node bottleneck.
- **Async writes:** Queue writes to Kafka; write to DB asynchronously.
- **Time-series databases:** InfluxDB, TimescaleDB for append-only workloads.

```
Read-heavy architecture:
  Write → Primary DB
  Read  → Cache → Read Replica (cache miss)

Write-heavy architecture:
  Write → Kafka → Worker → DB (batched, sharded)
  Read  → DB (eventually consistent) or separate OLAP store
```

**Mixed workloads:** Use CQRS — separate read and write models, each optimized for its purpose.

---

### Q2. What is connection pooling, and how do you size pools?

**Answer:**

**Connection pooling** maintains a set of reusable database connections rather than opening and closing a new connection for every request. Opening a DB connection involves TCP handshake, authentication, and protocol negotiation — this takes 20–100ms. For a service handling thousands of requests per second, this overhead is catastrophic.

**Without pooling:**
```
Request 1 → Open connection → Query → Close connection (20ms overhead)
Request 2 → Open connection → Query → Close connection (20ms overhead)
At 1000 RPS: 1000 connection opens/sec → DB overwhelmed
```

**With pooling:**
```
Startup: Open 20 connections (pool)
Request 1 → Borrow connection → Query → Return to pool (< 1ms overhead)
Request 2 → Borrow connection → Query → Return to pool
```

**Pool sizing formula (Little's Law):**
```
Pool size = (Average request rate) × (Average DB query time)

Example:
  Service handles 500 RPS
  Average DB query time = 10ms = 0.01s
  Required connections = 500 × 0.01 = 5 concurrent connections
  
  Add buffer: 5 × 1.5 = ~10 connections (with 50% headroom)
```

**Common mistake:** Setting pool size too large.

```
Database: PostgreSQL, max_connections = 100
Services: 10 pods, each with pool_size = 50
Total connections at peak = 500 → EXCEEDS DB limit → connection errors

Fix: Use PgBouncer (connection pooler at DB level)
     Or: pool_size = 8 per pod → 80 total (safe)
```

**Pool parameters:**
```
min_pool_size:    Connections kept warm at idle (avoids cold start)
max_pool_size:    Hard limit (prevents DB exhaustion)
connection_timeout: How long to wait for a free connection before error
idle_timeout:     Close connections unused for N seconds (saves resources)
max_lifetime:     Recycle connections periodically (avoids stale state)
```

**Tools:** HikariCP (Java), pgbouncer (PostgreSQL), SQLAlchemy pool (Python), TypeORM/Sequelize pools (Node.js).

---

### Q3. How do you handle a thundering herd at system startup?

**Answer:**

A **thundering herd** occurs when many processes or threads simultaneously try to acquire the same resource, overwhelming it. At startup, this is common when:
- Multiple service instances start simultaneously after a deploy.
- All instances try to warm up caches from the database at the same time.
- A cache expires simultaneously and all requests hit the DB at once (also called cache stampede).

**Scenario: Cache stampede**
```
Redis TTL expires for popular key
→ 500 concurrent requests see cache miss
→ 500 requests hit database simultaneously
→ DB falls over
→ Cascading failures
```

**Solutions:**

**1. Jitter on TTL:**
```python
import random
base_ttl = 3600  # 1 hour
jitter = random.randint(0, 300)  # ±5 minutes
redis.setex(key, base_ttl + jitter, value)
# Different instances expire at different times
```

**2. Probabilistic early expiration (XFetch):**
```python
# Recompute cache slightly before expiry, probabilistically
# Avoids stampede by spreading recomputation over time
import math, random, time

def get_cached(key, recompute_fn, ttl):
    data, expiry = redis.get_with_expiry(key)
    delta = time.time() - expiry  # Negative = time until expiry
    
    # Probabilistically recompute early based on how close to expiry
    if random.random() * ttl < -delta:
        data = recompute_fn()
        redis.setex(key, ttl, data)
    return data
```

**3. Mutex lock (cache-aside with locking):**
```python
def get_with_lock(key, compute_fn, ttl):
    value = cache.get(key)
    if value: return value
    
    lock_key = f"lock:{key}"
    acquired = cache.setnx(lock_key, 1, ex=10)  # 10s lock
    if acquired:
        try:
            value = compute_fn()
            cache.setex(key, ttl, value)
        finally:
            cache.delete(lock_key)
    else:
        time.sleep(0.1)
        return get_with_lock(key, compute_fn, ttl)  # Retry
    return value
```

**4. Startup staggering:**
```
# In Kubernetes: rolling deploy with maxSurge and readinessProbe
# New pods become ready gradually, traffic shifts slowly
strategy:
  rollingUpdate:
    maxSurge: 25%
    maxUnavailable: 0
```

---

### Q4. What is the hot partition/hot key problem, and how do you solve it?

**Answer:**

A hot partition (or hot key) occurs when a disproportionate amount of traffic or data is concentrated on a single shard/partition/node, overwhelming it while others sit idle.

**How it happens:**
```
Sharding by user_id: users are distributed evenly. ✓
Sharding by product_id: one viral product gets 90% of traffic. ✗

Example: iPhone launch
  product_id: "iphone_15_pro" → shard 7
  Shard 7 receives 100,000 RPS
  Shards 1-6 receive 100 RPS each
  → Shard 7 is overwhelmed
```

**Solutions:**

**1. Key salting / hash spreading:**
```python
# Spread one hot key across N shards using suffix
def get_key(product_id, n_shards=10):
    salt = random.randint(0, n_shards - 1)  # Write
    return f"{product_id}#{salt}"

# On read: query all n_shards and aggregate
keys = [f"{product_id}#{i}" for i in range(n_shards)]
values = redis.mget(*keys)
result = sum(v for v in values if v)
```

**2. Local in-process caching for hot keys:**
```python
# Detect hot keys (count-min sketch or exact counter)
# For keys with > 1000 req/sec, cache in local memory
from cachetools import TTLCache
local_cache = TTLCache(maxsize=100, ttl=1)  # 1-second local cache

def get_product(product_id):
    if product_id in local_cache:
        return local_cache[product_id]
    result = redis.get(product_id)
    if is_hot(product_id):
        local_cache[product_id] = result
    return result
```

**3. Read replicas for hot data:**
Route reads of detected hot keys to a dedicated pool of replicas.

**4. Consistent hashing with virtual nodes:**
Virtual nodes allow redistribution of load without rehashing all keys. When a node is hot, add more virtual nodes to other servers.

**5. Application-level fan-out:**
For write-heavy hot keys (like a counter), use sharded counters:
```python
# Instead of one counter key:
# redis.incr("views:product_123")

# Use N sharded counters:
shard = user_id % 10
redis.incr(f"views:product_123:{shard}")

# Read: sum all shards
total = sum(redis.get(f"views:product_123:{i}") for i in range(10))
```

---

### Q5. What is data locality and why does it matter for performance?

**Answer:**

**Data locality** refers to organizing data so that frequently co-accessed data is physically close together — either in memory (spatial locality, temporal locality) or on disk (same shard, same node, same datacenter).

**Why it matters:** Memory access is 100x faster than disk I/O. Network calls across datacenters are 100x slower than within a datacenter. Moving computation to data instead of data to computation eliminates the most expensive operation.

**Types of locality:**

**1. Temporal locality:** Data accessed once is likely to be accessed again soon.
- Solution: LRU cache. Keep recently used data in memory.

**2. Spatial locality:** Data near recently accessed data will likely be accessed next.
- Solution: Sequential disk reads. Store related rows together (row-oriented for OLTP, column-oriented for OLAP). Prefetch adjacent pages.

**3. Geographic locality:** Users in eu-west-1 should be served from a datacenter in Europe, not us-east-1.
```
Without geo-locality:
  EU user → us-east-1 (100ms RTT)

With geo-locality:
  EU user → eu-west-1 (5ms RTT)
  EU user's data lives in eu-west-1 (compliance + speed)
```

**4. Shard-level locality:** Co-locate all of a user's data on the same shard.
```python
# Shard all user data by user_id
shard = user_id % 16

# Orders, payments, preferences — all on same shard
# A query spanning a user's data = single shard = fast
```

**5. Compute locality (Lambda/Spark):** Run processing code on the same machine as the data.
```
Spark: Scheduler assigns tasks to the worker node that holds the data partition
→ Avoids network transfer for large datasets
→ "Bring computation to data, not data to computation"
```

**Anti-pattern — data not local:**
```
User API calls User Service (us-east-1)
User Service calls Preferences Service (eu-west-1)
Preferences Service calls DB (us-east-1)
→ 3 cross-region calls = 300ms+
```

---

### Q6. What are the challenges of scaling a stateful service?

**Answer:**

Stateless services are easy to scale — just add more instances and load balance. Stateful services maintain session state, in-memory data, or persistent connections, which makes horizontal scaling complex.

**Core challenges:**

**1. Session affinity (sticky sessions):**
If a user's session lives on server A, subsequent requests must go to server A. If it goes to server B, the session is lost.
```
Problem: One server gets more load (uneven distribution)
         If server A crashes, session lost
Solution: Externalize session to Redis
          redis.setex(f"session:{session_id}", 3600, user_data)
          → Any server can now serve any user
```

**2. In-memory state:**
WebSocket connections, rate limiter counters, local caches — these are per-instance.
```
Problem: User connects to WebSocket on server 1
         Message sent to server 2 (wrong instance)
Solution: Pub/sub backbone (Redis Pub/Sub, Kafka)
          Server 1 publishes message to topic
          All servers subscribed receive it and route to connected clients
```

**3. Stateful data sharding:**
```
Problem: Resharding requires migrating data while serving traffic
Solution: Consistent hashing → only ~1/n keys need to move when adding a node
          Virtual nodes → finer-grained rebalancing
```

**4. Leader election:**
For services with a single leader (e.g., primary database, distributed lock manager):
```
Solution: Raft consensus (etcd, ZooKeeper) for leader election
          → Leader fails → Automatic re-election in seconds
```

**5. Graceful shutdown:**
A stateful service cannot be killed mid-operation.
```python
# Handle SIGTERM: drain connections before exit
signal.signal(signal.SIGTERM, lambda: begin_graceful_shutdown())

def begin_graceful_shutdown():
    server.stop_accepting_new_connections()
    wait_for_active_requests_to_complete(timeout=30)
    flush_pending_writes()
    close_database_connections()
    sys.exit(0)
```

---

### Q7. What is the CQRS pattern and how does it help with scaling?

**Answer:**

**CQRS (Command Query Responsibility Segregation)** separates the read model from the write model of an application. Commands (writes) and queries (reads) use different models, different services, and often different databases.

```
Traditional (single model):
  User → CRUD API → Single DB (reads and writes compete)

CQRS:
  User → Command API → Write DB (normalized, consistent)
                      ↓ (events/replication)
  User → Query API  → Read DB (denormalized, optimized for reads)
```

**Why it helps with scaling:**

1. **Different scale requirements:** Reads may be 100x writes. Scale read and write services independently.
2. **Different consistency requirements:** Writes need ACID transactions. Reads can be eventually consistent.
3. **Different data shapes:** Write DB is normalized (avoids anomalies). Read DB is denormalized (joins are pre-computed).
4. **Independent optimization:** Write DB uses row store for OLTP. Read DB uses column store or search index for queries.

**Example — e-commerce order system:**
```
Command side (write):
  POST /orders → Validate → Insert normalized row → Emit OrderCreated event

Query side (read):
  GET /orders/42 → Return denormalized view (order + items + user + shipping)

Sync: OrderCreated event → Consumer updates the read-optimized denormalized "order_view" table
```

```python
# Command handler
def create_order(command: CreateOrderCommand):
    order = Order(user_id=command.user_id, items=command.items)
    db.save(order)
    event_bus.publish(OrderCreated(order_id=order.id, ...))

# Event consumer updates read model
def on_order_created(event: OrderCreated):
    view = build_order_view(event)  # Joins user, items, shipping
    read_db.upsert("order_views", view)
```

**Trade-off:** Eventual consistency. The read model lags behind writes by milliseconds to seconds. For use cases requiring immediate read-after-write consistency, CQRS requires additional patterns (read-your-writes guarantee).

---

## Medium Questions (Q8–Q15)

---

### Q8. How do fan-out on write vs fan-out on read work, and when to use each?

**Answer:**

This is one of the most important system design trade-offs for feed/timeline systems (Twitter, Instagram, Facebook).

**Fan-out on Write (Push model):**
When a user posts, the post is immediately written to every follower's feed (inbox).
```
User A posts → Worker finds 500 followers
             → Writes post to 500 timelines in Redis
Read path: GET /timeline → Read from pre-computed Redis feed (O(1))
```

Pros: Read is extremely fast (pre-built feed).
Cons: Write is expensive (writing to 10M followers takes time). Storage cost is high. Celebrities are a problem — Justin Bieber has 100M followers; a single tweet causes 100M writes.

**Fan-out on Read (Pull model):**
No pre-computation. When a user reads their feed, the system fetches posts from all followed users and merges them.
```
User A requests timeline
  → Find 500 people user A follows
  → Query each person's posts (last 100)
  → Merge and sort by timestamp
  → Return top 20
```

Pros: Write is cheap (single write to user's own post table). No storage explosion.
Cons: Read is expensive. At 500 followees × DB query = slow for high follower counts.

**Twitter's hybrid approach (celebrity problem):**
```
Normal users (< 1M followers):  Fan-out on Write
  Post → Pre-computed into followers' feeds

Celebrities (>= 1M followers):  Fan-out on Read
  Post NOT pre-computed
  At read time: merge pre-computed feed (normal users) 
              + live-query celebrity posts

This limits write amplification while keeping reads fast.
```

**Decision matrix:**

| Scenario                     | Strategy         | Reason                           |
|------------------------------|------------------|----------------------------------|
| Read:Write ratio > 100:1     | Fan-out on Write | Pre-compute to make reads cheap  |
| Users follow many celebrities| Fan-out on Read  | Avoid write amplification        |
| Low follower counts          | Fan-out on Write | Manageable write amplification   |
| High follower counts         | Hybrid           | Balance read vs write cost       |

---

### Q9. How do you scale a database that has become a bottleneck?

**Answer:**

Database bottlenecks are the most common scaling challenge. The approach is layered — exhaust cheaper options before reaching for sharding.

**Layer 1: Optimize queries (no hardware change)**
```sql
-- Add missing index
EXPLAIN ANALYZE SELECT * FROM orders WHERE user_id = 42;
-- If Seq Scan appears → CREATE INDEX idx_orders_user_id ON orders(user_id);

-- Avoid N+1 queries
-- Rewrite: SELECT * FROM orders; then for each: SELECT user...
-- To:      SELECT o.*, u.name FROM orders o JOIN users u ON o.user_id = u.id
```

**Layer 2: Caching layer**
```
Add Redis in front of DB.
Cache reads for hot data (user profiles, product catalog, config).
Cache hit rate target: > 90%
Effect: DB load drops by 90%+ for read-heavy workloads.
```

**Layer 3: Read replicas (scale reads)**
```
Primary DB  ← All writes
Read Replica 1 ← Read queries (replication lag ~50ms)
Read Replica 2 ← Analytics queries
Read Replica 3 ← Reporting

Connection routing:
  writes → primary
  reads  → round-robin across replicas
```

**Layer 4: Vertical scaling (scale up)**
Upgrade to a larger instance (more CPU, memory, faster SSDs). Simple but has limits and is expensive.

**Layer 5: Table partitioning (within one node)**
```sql
-- Partition orders by month
CREATE TABLE orders_2024_01 PARTITION OF orders
  FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
-- Queries filtered by date only scan the relevant partition
```

**Layer 6: Horizontal sharding (scale out)**
Distribute data across multiple database nodes. Each shard holds a subset of rows.
```
Shard by user_id:
  user_id % 4 == 0 → Shard 0
  user_id % 4 == 1 → Shard 1
  user_id % 4 == 2 → Shard 2
  user_id % 4 == 3 → Shard 3
```

Cost: Cross-shard queries become complex. No ACID transactions across shards.

**Layer 7: NoSQL or specialized storage**
Migrate high-write workloads to Cassandra (LSM-tree, write-optimized).
Move time-series data to TimescaleDB or InfluxDB.
Move search to Elasticsearch.

**Recommended order:**
```
1. Optimize queries + add indexes
2. Add caching
3. Add read replicas
4. Vertical scale
5. Partition tables
6. Shard
7. Evaluate NoSQL
```

---

### Q10. How do you scale WebSocket connections horizontally?

**Answer:**

WebSocket connections are stateful — a client is persistently connected to a specific server. Unlike HTTP, you cannot simply add more servers and round-robin requests, because subsequent messages from the client must go to the same server (or the server must route them).

**The problem:**
```
Client A connected to Server 1
Client B connected to Server 2
Client A sends message to Client B

Server 1 doesn't have Client B's connection → Message lost
```

**Solution: Pub/Sub backbone**

```
Client A ──websocket──→ Server 1
Client B ──websocket──→ Server 2

Server 1 receives message for Client B:
  1. Publish to Redis Pub/Sub: channel "user:B", message: {...}

Server 2 is subscribed to "user:B" (because Client B is connected there):
  2. Receives published message
  3. Forwards to Client B's WebSocket connection
```

**Architecture:**
```
[Client A] ──WS──→ [Server 1] ──publish──→ [Redis Pub/Sub]
[Client B] ──WS──→ [Server 2] ──subscribe──→ [Redis Pub/Sub]
                                             ─ delivers to Server 2
                                             → Server 2 routes to Client B
```

**Connection registry:**
```python
# Server 2 registers Client B's connection on connect
redis.hset("ws_connections", "user:B", "server_2")

# Server 1 routes message to correct server
target_server = redis.hget("ws_connections", "user:B")
if target_server == "server_1":
    # local delivery
else:
    redis.publish(f"ws:user:B", json.dumps(message))
```

**Scaling numbers:**
- Each WebSocket connection uses a file descriptor + ~50KB memory.
- A single Node.js server can handle ~65,000 concurrent WebSocket connections (file descriptor limit).
- With 10 servers: 650,000 concurrent connections.
- For millions: Use dedicated WebSocket infrastructure (Socket.IO cluster, AWS API Gateway WebSocket API).

**Load balancing:** Use sticky sessions or consistent hashing to route reconnecting clients to the same server (reduces pub/sub traffic).

---

### Q11. How does event sourcing work and why does it help with scale and auditability?

**Answer:**

**Event sourcing** is a data storage pattern where, instead of storing the current state of an entity, you store the complete sequence of events that led to that state. The current state is derived by replaying all events.

```
Traditional (State-based storage):
  orders table: {id: 123, status: "shipped", total: 99.00}
  → Only current state, history lost

Event sourcing (Event-based storage):
  event_store:
    {seq: 1, type: "OrderCreated",   data: {items: [...], total: 99.00}}
    {seq: 2, type: "PaymentReceived", data: {amount: 99.00}}
    {seq: 3, type: "OrderShipped",   data: {tracking: "XYZ"}}
  → Full history preserved; current state = replay of events
```

**Why it helps with scale:**

1. **Append-only writes:** Event log is append-only. No updates, no deletes, no locking conflicts. Extremely high write throughput.
2. **Multiple read models:** The same event stream can build multiple read models, each optimized for different queries.
3. **Temporal queries:** "What was the state of order 123 at 2pm yesterday?" — replay events up to that timestamp.

**Why it helps with auditability:**
```
Financial services compliance: "Show me every change to this account and who made it."
With state storage: Impossible (state overwritten)
With event sourcing: Trivial — read the event log
```

**Read model (Projection) building:**
```python
def build_order_projection(order_id: str) -> Order:
    events = event_store.get_events(order_id)
    order = Order()
    for event in events:
        order.apply(event)  # Each event mutates the aggregate
    return order

class Order:
    def apply(self, event):
        if event.type == "OrderCreated":
            self.status = "pending"
            self.items = event.data["items"]
        elif event.type == "OrderShipped":
            self.status = "shipped"
            self.tracking = event.data["tracking"]
```

**Snapshot optimization:**
Replaying 10,000 events for every read is slow. Use snapshots:
```
Every 1000 events, store a snapshot of the current state
On read: load snapshot + replay only events after snapshot
```

**Trade-offs:**
- Complexity: Reasoning about state requires event replay.
- Schema evolution: Old events must be backward-compatible or migrated.
- Eventual consistency: Read models are updated asynchronously.

---

### Q12. What is the cell architecture pattern, and why do AWS and Netflix use it?

**Answer:**

**Cell architecture** (also called bulkhead architecture) divides a system into independent, isolated cells, each of which is a self-contained copy of the entire service stack. A fault in one cell is contained and does not propagate to others.

```
Traditional (shared infrastructure):
  [All users] → [Shared API Servers] → [Shared DB]
  A bug in shared DB = everyone is affected

Cell architecture:
  Cell 1: Users A-M → [API Servers 1] → [DB 1]
  Cell 2: Users N-Z → [API Servers 2] → [DB 2]
  Cell 3: VIP users → [API Servers 3] → [DB 3] (separate, better hardware)
  
  A bug in Cell 1 = only users A-M affected
```

**Why Netflix uses cells:**
Netflix divides its infrastructure into "regions" and within regions into cells. Streaming for users in cell 5 is completely independent of cell 6. A bad deployment only affects one cell's users (rolling updates at cell granularity).

**Why AWS uses cells (shuffle sharding):**
AWS Route 53 uses shuffle sharding — each customer's DNS resolution is served by a random subset of nameservers. A DDoS against one customer only affects the small set of nameservers in their cell, not all customers.

**Cell sizing considerations:**
```
Too small cells: Too many cells to manage, overhead per cell is high
Too large cells: Blast radius is too large
Ideal: Cell handles ~10–20% of traffic (5–10 cells total)
```

**Key properties of a cell:**
```
1. Self-contained: Has its own DB, cache, compute, config
2. Independently deployable: Can upgrade cell 1 without touching cell 2
3. Independently scalable: Scale cell 3 (VIP) without scaling others
4. Fault-isolated: A cascade failure in cell 1 cannot starve cell 2's resources
5. No cross-cell calls: Cells must not communicate with each other
```

**Limitation:** Cross-cell queries are impossible (users in cell 1 cannot see data from cell 2). Suitable for user data, not for global aggregations. A global control plane exists outside cells for routing (which cell is this user in?).

---

### Q13. How does database replication lag work, and what are solutions?

**Answer:**

**Replication lag** is the delay between a write being committed on the primary database and that write becoming visible on read replicas. In asynchronous replication (default for most databases), the primary does not wait for replicas to confirm before acknowledging the write.

```
Primary (us-east-1):
  t=0: Write order_status = "shipped"
  t=0: Acknowledge to client

Replica (us-west-2):
  t=50ms: Receives replication event
  t=50ms: Applies write

Between t=0 and t=50ms:
  Client reads from replica → gets "pending" (stale!)
```

**Real-world problems:**
- User updates profile picture, refreshes page → sees old picture (read from replica).
- User submits payment → reads order status from replica → order not found.
- Leader election/analytics reports based on stale data.

**Solutions:**

**1. Read-your-writes consistency:**
After a write, route the client's subsequent reads to the primary for N seconds.
```python
def update_profile(user_id, data):
    primary_db.update(user_id, data)
    session["read_primary_until"] = time.time() + 5  # 5 seconds

def get_profile(user_id):
    if time.time() < session.get("read_primary_until", 0):
        return primary_db.get(user_id)  # Use primary
    return replica_db.get(user_id)      # Use replica
```

**2. Synchronous replication (semi-sync):**
Primary waits for at least one replica to confirm before acknowledging. Adds write latency (50–100ms) but eliminates lag for that replica.

**3. Write to primary, read from primary for critical paths:**
Identify critical read paths (order status check after payment) and always route to primary.

**4. Monitor replication lag:**
```sql
-- PostgreSQL: check replication lag
SELECT client_addr, write_lag, replay_lag
FROM pg_stat_replication;

-- Alert if replay_lag > 5 seconds
```

**5. Application-level versioning:**
Include a version number with writes. On read, if the version is older than expected, retry from the primary.

---

### Q14. How do you design for 10x traffic with minimal code changes?

**Answer:**

Designing for 10x traffic means building a system where most of the scaling work is operational (adding nodes, adjusting config) rather than requiring architectural overhaul.

**The 10x design checklist:**

**1. Stateless application tier (horizontal scale with zero code changes):**
```
All application state lives in external stores (Redis, DB).
Scale from 5 pods to 50 pods: just change the replica count.
No code changes required.
```

**2. Externalized configuration:**
```
Connection pool sizes, cache TTLs, feature flags in env vars or config service.
At 10x: increase pool sizes without redeployment.
```

**3. Async processing for slow operations:**
```python
# Bad: Synchronous email sends during request processing
def checkout(order):
    process_payment(order)
    send_confirmation_email(order)  # Blocks for 500ms
    return order

# Good: Queue async work
def checkout(order):
    process_payment(order)
    queue.publish("send_email", {"order_id": order.id})  # Non-blocking
    return order

# At 10x: scale email workers independently, checkout API unchanged
```

**4. Cache aggressively:**
```
At 1x: DB handles all reads
At 10x: DB is overwhelmed. 
Add Redis cache. Popular data served from cache, DB shielded.
Code change: minimal (cache decorator pattern).
```

**5. Auto-scaling configuration:**
```yaml
# Kubernetes HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
spec:
  minReplicas: 5
  maxReplicas: 50
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 60
```

**6. Database read replicas:**
```
At 1x: Single DB handles reads and writes
At 10x: Route reads to replicas
Code change: Use read/write split in ORM config — often just a config file change
```

**7. CDN for static assets:**
```
Serving images from your servers at 1x: fine
At 10x: bandwidth exhaustion
Put CloudFront/Fastly in front: zero code changes
```

---

### Q15. What is graceful degradation under load (load shedding strategies)?

**Answer:**

**Graceful degradation** means that under extreme load, a system deliberately reduces functionality for some users rather than failing completely for all users. Load shedding is the deliberate dropping or deferring of requests to protect critical capacity.

**Without load shedding:**
```
10x traffic spike → All resources exhausted → P99 latency = 30 seconds
→ All users experience bad service simultaneously
```

**With load shedding:**
```
10x traffic spike → Shed 50% of non-critical requests
→ Critical users (paying customers) continue with normal latency
→ Non-critical users see degraded or delayed experience
```

**Strategies:**

**1. Priority queues:**
```python
# Critical path: checkout, payment
# Non-critical: recommendation updates, analytics events

queue.publish("checkout",     message, priority=HIGH)
queue.publish("reco_update",  message, priority=LOW)

# Under load: workers process HIGH priority first; LOW may wait or drop
```

**2. Circuit breakers:**
```python
# If dependent service is slow, stop calling it and use fallback
@circuit_breaker(failure_threshold=5, timeout=30)
def get_recommendations(user_id):
    return recommendations_service.get(user_id)

def get_page(user_id):
    try:
        recs = get_recommendations(user_id)
    except CircuitOpenError:
        recs = get_fallback_recommendations()  # Static popular items
    return render_page(recs)
```

**3. Adaptive timeout + fail fast:**
```python
# At 1x: allow 2s timeout for enrichment service
# Under load: reduce to 200ms; return partial response if it times out
timeout = 2.0 if current_load < 0.7 else 0.2
response = enrichment_service.call(timeout=timeout)
```

**4. Rate limiting with priority tiers:**
```
Tier 1 (unlimited): Core checkout, payment processing
Tier 2 (1000 RPS):  Search, browse
Tier 3 (100 RPS):   Analytics, recommendations
Tier 4 (10 RPS):    Background batch jobs
```

**5. Feature flags for degraded mode:**
```python
if feature_flags.get("degraded_mode"):
    # Disable personalization (expensive ML calls)
    # Return cached/static recommendations
    # Skip social features
    # Disable live inventory checks (use cached stock)
```

---

## Hard Questions (Q16–Q20)

---

### Q16. How did Twitter solve the celebrity/whale problem in timelines?

**Answer:**

Twitter's timeline system is the canonical example of the fan-out problem at extreme scale, and their evolution from pure fan-out-on-write to a hybrid model is a case study in practical systems design.

**Original design (fan-out on write, 2006–2010):**
```
User tweets → Write to own timeline
            → Find all followers
            → Write tweet ID to each follower's Redis sorted set (timeline)

Read: GET /timeline → Redis.zrange(user_timeline, 0, 20) → O(1), very fast
```

This worked until celebrities joined. Lady Gaga has 30M followers. A single tweet triggered 30M Redis writes. Justin Bieber's tweet processing time: 10–15 minutes.

**The celebrity (whale) problem:**
```
Celebrity tweets:
  30M followers × 1 write/follower = 30M Redis operations
  At 10 tweets/min = 300M operations/min
  This caused write storms that degraded the entire system
```

**Twitter's hybrid solution (circa 2012):**

Step 1: Define a threshold (e.g., > 1 million followers = "celebrity").

Step 2: For regular users, continue fan-out-on-write as before.

Step 3: For celebrities, do NOT fan-out. Tweet is stored only in celebrity's own sorted set.

Step 4: At read time, merge two sources:
```python
def get_timeline(user_id, limit=20):
    # Source 1: Pre-computed feed (from fan-out of normal users)
    precomputed_feed = redis.zrange(f"timeline:{user_id}", 0, 800)
    
    # Source 2: Live queries for each celebrity followed
    celebrity_ids = get_followed_celebrities(user_id)
    celebrity_tweets = []
    for cid in celebrity_ids:
        tweets = redis.zrange(f"user_tweets:{cid}", 0, 20)  # Recent tweets only
        celebrity_tweets.extend(tweets)
    
    # Merge and sort
    all_tweets = precomputed_feed + celebrity_tweets
    return sorted(all_tweets, key=lambda t: t.timestamp, reverse=True)[:limit]
```

**Result:**
- Write storm eliminated for celebrity tweets.
- Read cost: small number of celebrity queries merged in-memory (fast).
- Trade-off: Slightly more complex read path, but acceptable latency.

**Additional optimizations:**
- Pre-compute celebrity followers' timelines lazily (on first read, not on tweet).
- Cache merged timelines in Redis for a few seconds.
- Use sparse timeline entries (store tweet IDs only, not content — content fetched separately).

---

### Q17. How do you scale search at scale using Elasticsearch cluster design?

**Answer:**

Elasticsearch scales horizontally via sharding and replication. Understanding the internals is essential for designing a cluster that handles high write throughput and query volume simultaneously.

**Core concepts:**
```
Index: Logical collection of documents (like a DB table)
Shard: A physical Lucene index; the unit of parallelism
  → Primary shard: handles writes and reads
  → Replica shard: serves reads, provides failover
Node: A JVM process running Elasticsearch
Cluster: Multiple nodes

Document → Routed to shard: shard_id = hash(routing_key) % num_primary_shards
```

**Example cluster:**
```
Cluster: 3 nodes, 1 index, 3 primary shards, 1 replica each

Node 1: Primary Shard 0,  Replica Shard 1
Node 2: Primary Shard 1,  Replica Shard 2
Node 3: Primary Shard 2,  Replica Shard 0
```

**Scaling reads:**
Add more nodes → more replicas → more read throughput. Reads can be served by any shard copy.

**Scaling writes:**
Increase number of primary shards (more parallelism). Cannot change primary shard count after index creation — plan ahead.
```
Rule of thumb: 20–50 GB per shard, max 200M documents per shard
For 1 TB of data with 50GB shards: 20 primary shards
```

**High write throughput optimizations:**
```json
// Index settings for write-heavy workloads
{
  "settings": {
    "refresh_interval": "30s",       // Default 1s → index visible after 30s (less overhead)
    "number_of_replicas": 0,         // During initial bulk load, set to 0
    "translog.durability": "async",  // Async fsync (risk: data loss on crash)
    "codec": "best_compression"      // Smaller shards, less I/O
  }
}
```

**Index lifecycle management (ILM):**
```
Time-series data (logs, metrics): use rolling indices + ILM
  hot → warm → cold → delete

hot:  Current day's data, fast SSD, all primaries + 1 replica
warm: Last 7 days, slower disk, replica count reduced
cold: Last 30 days, object storage or frozen
delete: After 90 days
```

**Query optimization:**
```json
{
  "query": {
    "bool": {
      "filter": [{"term": {"status": "active"}}],  // filter: no scoring, cached
      "must":   [{"match": {"description": "query"}}]  // must: scores documents
    }
  },
  "routing": "user_42"  // Route to specific shard (avoids scatter-gather)
}
```

**Hardware sizing:**
```
JVM heap: 50% of RAM, max 32GB (avoids compressed OOPs threshold)
OS cache: Remaining 50% of RAM for Lucene file cache
CPU: 2x cores per node vs thread count
Storage: NVMe SSD for hot data
```

---

### Q18. How do you implement a real-time pub/sub backbone to scale real-time features?

**Answer:**

A pub/sub backbone decouples producers from consumers and enables real-time event distribution across horizontally scaled services.

**Use cases:** WebSocket message routing, live notifications, real-time collaborative editing, live leaderboards, chat systems.

**Architecture options:**

**Option 1: Redis Pub/Sub (simple, low latency)**
```
Publisher → PUBLISH channel "chat:room_42" message
Subscriber → SUBSCRIBE channel "chat:room_42"
             Receives message immediately
```

Pros: Sub-millisecond latency, simple API.
Cons: Fire-and-forget (no persistence, no consumer groups, messages lost if subscriber offline).

**Option 2: Kafka (durable, scalable)**
```
Publisher → Produce event to topic "notifications"
Consumer Group A (WebSocket servers) → Consumes events, routes to connections
Consumer Group B (DB writers) → Persists notifications to DB
Consumer Group C (Email service) → Sends email for offline users
```

Pros: Message durability, replay, multiple independent consumer groups, high throughput.
Cons: Higher latency (5–10ms), more operational complexity.

**Hybrid for real-time features:**
```
Event arrives → Kafka (durable storage + async processing)
             → Redis Pub/Sub (immediate real-time fanout to connected WebSocket servers)

Producer publishes to both:
  kafka.produce("notifications", event)          # Durable
  redis.publish("realtime:notifications", event) # Immediate
```

**Real-time notification system design:**
```python
# Notification producer
async def send_notification(user_id: str, notification: dict):
    msg = {"user_id": user_id, **notification, "ts": time.time()}
    
    # Durable path: for users who are offline
    await kafka.produce("notifications", key=user_id, value=msg)
    
    # Real-time path: for users who are online
    await redis.publish(f"user_notifications:{user_id}", json.dumps(msg))

# WebSocket server subscribes to Redis
async def websocket_handler(user_id: str, ws):
    async with redis.subscribe(f"user_notifications:{user_id}") as sub:
        async for message in sub:
            await ws.send(message.data)
```

**Scaling the Kafka backbone:**
```
Topic: 12 partitions (parallelism factor)
Consumer group: 12 instances (one per partition for max throughput)
Partitioning key: user_id (ensures order per user)
Retention: 7 days (allows replay for debugging + catchup)
```

---

### Q19. How does consistent hashing work, and why is it essential for distributed systems?

**Answer:**

**The problem with modular hashing:**
```
Simple hash: shard = hash(key) % N

If N = 3 (3 nodes):
  key "user_1" → hash % 3 = 1 → Node 1
  key "user_2" → hash % 3 = 2 → Node 2

Add a 4th node (N = 4):
  key "user_1" → hash % 4 = 3 → Node 3  ← CHANGED
  key "user_2" → hash % 4 = 2 → Node 2  ← unchanged

On re-hashing with N+1: ~N/(N+1) fraction of keys move = almost all keys
This causes a massive remapping when adding/removing nodes → cache invalidation storm
```

**Consistent hashing solution:**
Place both nodes and keys on a virtual ring (0 to 2^32). Each key maps to the first node clockwise from its position on the ring.

```
Ring (0 → 2^32):
     Node A (at 100)
         ↑
0 ──────────────────────── 2^32
         ↓           ↑
     Node C        Node B
     (at 300)     (at 200)

key_1 at position 150 → Node B (next clockwise from 150)
key_2 at position 250 → Node C (next clockwise from 250)
key_3 at position 350 → Node A (next clockwise, wrapping around)
```

**Adding a node:**
```
Add Node D at position 175:
  Keys between 150 and 175 now map to Node D instead of Node B
  Only ~1/N fraction of keys move (not all of them)
```

**Virtual nodes (vnodes):**
To avoid uneven distribution (especially important when nodes have different capacities), each physical node is represented by multiple virtual nodes on the ring.

```
Node A (high-capacity): 150 virtual nodes on ring
Node B (low-capacity):   50 virtual nodes on ring
→ Node A gets 3x the traffic (proportional to vnodes)
→ When Node A is removed: its 150 positions are distributed across all remaining nodes
   → Even redistribution
```

**Used by:** Cassandra (data partitioning), DynamoDB (data partitioning), Redis Cluster (hash slots — a variant), Memcached client-side (consistent hashing), Nginx upstream load balancing.

```python
import hashlib
import bisect

class ConsistentHashRing:
    def __init__(self, nodes, vnodes=150):
        self.ring = {}
        self.sorted_keys = []
        for node in nodes:
            for i in range(vnodes):
                key = hashlib.md5(f"{node}:{i}".encode()).hexdigest()
                self.ring[key] = node
                bisect.insort(self.sorted_keys, key)
    
    def get_node(self, key: str) -> str:
        h = hashlib.md5(key.encode()).hexdigest()
        idx = bisect.bisect_right(self.sorted_keys, h) % len(self.sorted_keys)
        return self.ring[self.sorted_keys[idx]]
```

---

### Q20. How do you design a content sharding vs user sharding vs geographic sharding strategy?

**Answer:**

Sharding strategy determines how data is divided across nodes. The choice of shard key has far-reaching consequences for query patterns, hotspots, and operational complexity.

**User Sharding:**
Data partitioned by `user_id`. All of a user's data lives on the same shard.

```
shard_id = hash(user_id) % N

User 42's orders, preferences, payments → all on Shard 3
User 99's data → Shard 7
```

Pros: User-centric queries are single-shard (fast, no scatter-gather). Natural isolation.
Cons: Popular users (celebrities, high-volume merchants) create hotspots. Cross-user aggregations require scatter-gather.

Best for: Social networks, e-commerce with per-user inventory, multi-tenant SaaS.

**Content Sharding:**
Data partitioned by `content_id` or `topic`. Used when content (posts, videos, products) is the primary access pattern.

```
shard_id = hash(content_id) % N

Post 12345 → Shard 2 (comments, likes, views all co-located with post)
```

Pros: Content-centric queries fast. Write throughput distributed evenly if content creation is uniform.
Cons: User-centric queries (all content by user X) require scatter-gather. Viral content creates hotspot.

Best for: Content delivery networks, media platforms, product catalogs.

**Geographic Sharding:**
Data partitioned by region/geography. All data for users in a given region lives in a datacenter in that region.

```
US users    → us-east-1 datacenter
EU users    → eu-west-1 datacenter
APAC users  → ap-southeast-1 datacenter
```

Pros: Low latency (data close to users). Regulatory compliance (GDPR data residency). Regional failure isolation.
Cons: Cross-region queries are expensive. Users who travel or change regions complicate routing.

Best for: Global consumer apps with data residency requirements (GDPR), real-time apps where latency matters.

**Hybrid sharding:**
Most large systems combine strategies:
```
Level 1: Geographic shard (data residency + latency)
  └── US shard
       ├── Level 2: User shard (user_id % 16)
       │     ├── Shard 0-3: users with high activity (hot shard mitigation)
       │     └── Shard 4-15: normal users
       └── Content index (Elasticsearch, content sharded by content_id)
```

**Shard key selection principles:**
```
1. High cardinality: Many unique values → even distribution
2. Immutable: Shard key should not change (would require data migration)
3. Evenly distributed: Avoid hot keys
4. Query-aligned: Most queries should be within a single shard
5. No cross-shard transactions: Shard key should prevent the need for distributed ACID
```

**Resharding:**
When shards become too large or too hot:
```
Strategy: Double the shards
  Before: 16 shards (shard_id = user_id % 16)
  After:  32 shards (shard_id = user_id % 32)
  
Migration: Each original shard splits into 2 child shards
           Use double-write during migration period
           Blue-green cutover at DNS level
```

---

## Quick Reference

```
READ VS WRITE HEAVY
  Read-heavy  → Caching + Read Replicas + CDN + Denormalize
  Write-heavy → LSM DB (Cassandra) + Kafka buffer + Sharding

CONNECTION POOL SIZING
  Pool size = RPS × avg_query_time (Little's Law)
  Example: 500 RPS × 10ms = 5 connections (+ buffer = 10)

THUNDERING HERD SOLUTIONS
  Jitter on TTL → spread expirations
  Mutex lock   → one recompute, rest wait
  Probabilistic early recompute (XFetch)

HOT KEY SOLUTIONS
  Key salting: product_123#shard → spread across N shards
  Local in-process cache for hot keys
  Sharded counters for write-heavy hot keys

FAN-OUT ON WRITE vs READ
  Write: Pre-compute into follower timelines → fast read
  Read:  Compute at read time → slow read, cheap write
  Hybrid: Write for normal users, Read for celebrities

DATABASE SCALING LAYERS
  1. Query optimization + indexes
  2. Redis cache
  3. Read replicas
  4. Vertical scale
  5. Table partitioning
  6. Horizontal sharding
  7. NoSQL migration

WEBSOCKET HORIZONTAL SCALING
  Redis Pub/Sub as backbone
  Connection registry → which server has which user
  Publish to channel → correct server delivers to client

CELL ARCHITECTURE
  Independent cells (own DB, cache, compute)
  Fault isolation: blast radius = 1 cell
  No cross-cell calls

CONSISTENT HASHING
  Keys + nodes on virtual ring
  Add/remove node: only ~1/N keys move
  Virtual nodes: proportional load distribution

SHARD KEY SELECTION
  High cardinality + Immutable + Even distribution
  Query-aligned + No cross-shard transactions needed

GRACEFUL DEGRADATION
  Priority queues → critical requests first
  Circuit breakers → fallback for slow dependencies
  Feature flags → disable expensive features under load
  Load shedding → drop non-critical tier requests
```

---

*File 13 of 15 — Scalability Patterns*
