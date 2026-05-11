# Performance Optimization — Interview Q&A

---

## Easy (Q1–Q7)

---

### Q1. What is the performance optimization process?

Performance optimization is not guesswork — it is a scientific process of measurement, diagnosis, and targeted intervention. The most dangerous optimization mistake is fixing the wrong thing based on intuition.

**The cycle:**
```
  ┌──────────────────────────────────────────┐
  │  1. Measure (establish baseline metrics)  │
  │         ↓                                │
  │  2. Identify Bottleneck (find the         │
  │     slowest/most constrained component)  │
  │         ↓                                │
  │  3. Fix (targeted change, one at a time) │
  │         ↓                                │
  │  4. Measure (verify improvement,         │
  │     ensure no regression elsewhere)      │
  │         ↓                                │
  │  Repeat until SLO is met                │
  └──────────────────────────────────────────┘
```

**Step 1 — Measure first:**
Before any optimization, capture baseline: p50, p95, p99 latency, CPU, memory, DB query times, cache hit rate. Use APM tools (Datadog, New Relic, Jaeger) to get distributed traces. A trace shows exactly where each millisecond is spent across services.

**Step 2 — Find the real bottleneck:**
Amdahl's Law: the speedup of a program is limited by its sequential portion. Optimize the most time-consuming component, not the easiest. If 90% of a request's time is in a DB query, optimizing the JSON serialization (5%) gives negligible improvement.

**Step 3 — Fix one thing at a time:**
Changing multiple things simultaneously makes it impossible to attribute the improvement or regression to a specific change.

**Step 4 — Measure again:**
A "fix" that improves one metric while degrading another (e.g., caching reduces latency but causes stale data bugs) may not be a net win. Always verify the trade-off is acceptable.

---

### Q2. What is the difference between latency and throughput, and what is Little's Law?

**Latency** is the time taken for a single request to complete — from when it is sent to when the response is received. It is measured in milliseconds.

**Throughput** is the number of requests a system can handle per unit of time — measured in requests per second (RPS) or transactions per second (TPS).

They are related but can move in opposite directions:
```
High throughput, high latency: batch processing (process 1M records, takes hours)
Low throughput, low latency: real-time single-key lookup (1ms, 100 RPS max)
High throughput, low latency: ideal (hard to achieve simultaneously)
```

**Little's Law:**
```
L = λ × W

L = average number of requests in the system (queue + service)
λ = average arrival rate (requests per second)
W = average time a request spends in the system (latency)

Example:
  API handles 1000 req/s (λ = 1000)
  Average latency = 50ms (W = 0.05s)
  Concurrent requests in flight: L = 1000 × 0.05 = 50

Implication:
  If latency spikes to 200ms: L = 1000 × 0.2 = 200 concurrent requests
  If system can only hold 100 in-flight: queues form, latency worsens further
  This is a positive feedback loop → system meltdown
```

Little's Law explains why latency spikes cause cascading failures: more in-flight requests accumulate, queues fill, latency worsens further. Reducing latency reduces queue depth, which reduces latency further — a virtuous cycle.

**Throughput vs latency trade-off:**
Batching increases throughput (fewer round trips) but increases per-item latency. Streaming decreases latency but may reduce throughput. The right trade-off depends on the use case — real-time vs. batch.

---

### Q3. What is the USE method for resource analysis?

The **USE Method** (by Brendan Gregg) is a systematic checklist for analyzing any resource in a system. For every resource, check three dimensions:

```
USE = Utilization + Saturation + Errors

Utilization:  How busy is the resource? (% of time doing work)
Saturation:   How much extra work is queued/waiting? (queue depth)
Errors:       Are there error events? (dropped packets, disk errors)
```

**Applying USE to common resources:**

| Resource   | Utilization               | Saturation             | Errors                    |
|------------|---------------------------|------------------------|---------------------------|
| CPU        | CPU% (top, htop)          | Load average > core ct | MCE errors (dmesg)        |
| Memory     | Used/Total RAM %          | Swap usage, paging     | OOM killer events         |
| Disk I/O   | iostat %util              | I/O wait time, queue   | smartctl disk errors      |
| Network    | NIC throughput % of max   | TX/RX drop counters    | ifconfig errors, drops    |
| DB Pool    | Active connections / max  | Waiting connections    | Connection refused errors |

**Example diagnosis:**
```
Problem: API latency increased from 20ms to 200ms

USE analysis:
  CPU utilization: 40% (normal)   → not CPU-bound
  Memory: 60% used, 0 swap        → not memory pressure
  Disk I/O: 95% util ← HIGH       → disk is saturated
  Disk saturation: 500ms I/O wait → writes are queuing

Conclusion: disk I/O is bottleneck
Fix options:
  1. Add more IOPS (upgrade EBS, add provisioned IOPS)
  2. Move to SSD if on HDD
  3. Reduce write amplification (batch writes)
  4. Enable write caching / write-behind
```

USE is resource-centric (vs. request-centric). Use it first to narrow down the resource bottleneck, then drill into that resource with deeper profiling tools.

---

### Q4. What is connection pooling and why does opening a new database connection cost 50–100ms?

A database connection is not a lightweight object. Creating one involves multiple round trips and authentication steps that add significant overhead.

**What happens when you open a new PostgreSQL connection:**
```
1. TCP SYN/ACK handshake (1 RTT)               ~ 1ms
2. SSL handshake (2-3 RTTs for TLS 1.2)        ~ 3ms
3. PostgreSQL authentication (MD5/SCRAM)        ~ 2ms
4. PostgreSQL backend process fork/start        ~ 30-80ms
5. Memory allocation (work_mem, etc.)           ~ 5ms

Total: ~50-100ms per new connection
(PostgreSQL spawns a new OS process per connection)
```

**Connection Pool behavior:**
```
Without pooling:
  Request1 → create connection (80ms) → query (5ms) → close → Total: 85ms
  Request2 → create connection (80ms) → query (5ms) → close → Total: 85ms

With pooling (PgBouncer/HikariCP):
  Pool startup: create 10 connections (80ms × 10 = 800ms, done once)
  
  Request1 → borrow connection from pool (< 1ms) → query (5ms) → return → Total: 6ms
  Request2 → borrow connection from pool (< 1ms) → query (5ms) → return → Total: 6ms
```

**Pool sizing guidance:**
```
PostgreSQL rule of thumb: pool_size ≈ (CPU_cores × 2) + disk_spindles
For a 4-core DB with SSDs: pool_size ≈ 10-20 connections

HikariCP (Java) recommended config:
  minimumIdle = 5
  maximumPoolSize = 20
  connectionTimeout = 30000  # 30s
  idleTimeout = 600000       # 10m
```

**PgBouncer modes:**
- **Session pooling:** one server connection per client session (least multiplexing)
- **Transaction pooling:** connection returned to pool after each transaction (recommended)
- **Statement pooling:** most aggressive, breaks transactions — avoid

Connection pools also protect the database from being overwhelmed by hundreds of app instances each trying to maintain many connections.

---

### Q5. What is the N+1 query problem and how do you solve it?

The N+1 problem is one of the most common ORM-related performance anti-patterns. It occurs when fetching a list of N records triggers N additional queries to load associated data.

**The problem:**
```python
# Fetching 100 orders and their customers (BAD)
orders = Order.query.all()          # 1 query: SELECT * FROM orders (100 rows)

for order in orders:
    print(order.customer.name)      # 100 queries: SELECT * FROM customers WHERE id=?
                                    # Each access triggers a separate query

Total: 1 + 100 = 101 queries
At 2ms per query: 202ms just for DB round trips
```

**Solution 1 — SQL JOIN:**
```python
# Load orders with customers in a single query (GOOD)
orders = db.session.query(Order).join(Customer).all()
# 1 query: SELECT orders.*, customers.* FROM orders JOIN customers ...
# Total: 1 query
```

**Solution 2 — Eager loading (ORM):**
```python
# SQLAlchemy eager load
orders = Order.query.options(joinedload(Order.customer)).all()
# or subquery load for collections:
orders = Order.query.options(subqueryload(Order.items)).all()
```

**Solution 3 — DataLoader (GraphQL / batch fetch):**
```javascript
// DataLoader batches all customer loads within one tick into a single query
const customerLoader = new DataLoader(async (customerIds) => {
  const customers = await Customer.findAll({
    where: { id: customerIds }      // IN (...) query
  });
  return customerIds.map(id => customers.find(c => c.id === id));
});

// Each resolver calls: customerLoader.load(order.customerId)
// DataLoader batches 100 loads into 1 query: SELECT * FROM customers WHERE id IN (...)
```

**Detection:** Enable query logging and count queries per request. Any page generating more than 5–10 queries likely has N+1 issues. APM tools like Datadog or New Relic can flag N+1 patterns automatically.

---

### Q6. What is CDN and HTTP caching and what is a good cache-hit ratio target?

A **CDN (Content Delivery Network)** distributes content to edge servers geographically close to users, reducing latency and offloading traffic from origin servers. A **cache-hit ratio** measures what percentage of requests are served from cache rather than origin.

**Cache-hit ratio target: > 95%** means only 5% of requests need to reach the origin server — 20x load reduction.

**HTTP caching headers:**
```
Cache-Control: public, max-age=86400        # Cache for 24 hours
Cache-Control: private, max-age=3600        # Client-only cache (user-specific)
Cache-Control: no-store                     # Never cache (auth pages)
ETag: "abc123"                              # Content hash for conditional GET
Last-Modified: Mon, 01 Jan 2025 00:00:00   # Timestamp for conditional GET

Conditional request:
  Client: GET /image.png  If-None-Match: "abc123"
  Server: 304 Not Modified  (no body sent, saves bandwidth)
```

**CDN architecture:**
```
User (Tokyo) → CDN Edge (Tokyo) → Cache HIT: respond immediately
                                 → Cache MISS: fetch from Origin (Virginia)
                                   Cache the response for next request
```

**What to cache vs not to cache:**
```
Cache (immutable / rarely changing):
  /static/app.v3.js          → max-age=31536000 (1 year), content-hashed filename
  /images/logo.png           → max-age=86400
  /api/products (public)     → max-age=300 (5 min TTL)

Do NOT cache:
  /api/orders/{user_id}      → private, user-specific
  /checkout                  → private, session-dependent
  /api/auth/*                → never cache
```

**Cache invalidation strategies:**
- **TTL expiry:** simple but accepts stale data up to TTL
- **Cache-busting with versioned filenames:** `app.v3.min.js` — old URL stays cached, new version gets new URL
- **CDN purge API:** explicitly invalidate on deploy (CloudFront `create_invalidation`)

---

### Q7. What is asynchronous processing and how does it remove latency from the critical path?

**Async processing** means deferring non-critical work to a background queue so the user receives a fast response without waiting for every operation to complete.

**The critical path** is the sequence of operations that the user must wait for. Any work not essential to generating the response should be moved off the critical path.

**Example — user registration:**
```
Synchronous (BAD):
  User submits form →
    1. Save user to DB (10ms)
    2. Send welcome email via SMTP (800ms)  ← blocks user
    3. Create Stripe customer (300ms)        ← blocks user
    4. Index user in Elasticsearch (200ms)  ← blocks user
    5. Send Slack notification (150ms)       ← blocks user
  Total user wait: ~1460ms

Asynchronous (GOOD):
  User submits form →
    1. Save user to DB (10ms)  ← critical path only
    2. Enqueue background jobs: [send_email, create_stripe, index_user, slack_notify]
  Total user wait: ~15ms (including queue publish)
  
  Background workers process jobs within seconds — user doesn't wait.
```

**Implementation with a job queue:**
```python
# Using Celery + Redis
@celery.task
def send_welcome_email(user_id):
    user = User.get(user_id)
    email_service.send(user.email, template="welcome")

# In registration handler:
def register_user(data):
    user = User.create(data)
    send_welcome_email.delay(user.id)   # non-blocking, enqueues job
    return {"user_id": user.id, "status": "created"}  # instant response
```

**What belongs on the critical path:**
- Writing the primary data record (required for consistency)
- Auth/authorization checks

**What to defer:**
- Email/SMS notifications
- Analytics events
- Search index updates
- Webhook deliveries to third parties
- Report generation
- Payment reconciliation jobs

The tradeoff: async work may fail independently. Implement retry logic, dead-letter queues, and idempotent job handlers.

---

## Medium (Q8–Q15)

---

### Q8. How do you optimize a slow API endpoint? Give a 5-step checklist.

When an endpoint is slow, the diagnosis process should be systematic rather than immediately adding caches or indices.

**5-Step Optimization Checklist:**

**Step 1 — Profile and get a trace:**
```
Use APM (Datadog, Jaeger) to get a distributed trace of the slow request.
Identify where time is spent:
  - 5ms: auth middleware
  - 850ms: DB query            ← investigate
  - 20ms: JSON serialization
  - 10ms: Redis lookup
```

**Step 2 — Analyze the database queries:**
```sql
-- Run EXPLAIN ANALYZE on slow queries
EXPLAIN ANALYZE
  SELECT u.*, o.* FROM users u
  JOIN orders o ON u.id = o.user_id
  WHERE o.status = 'pending' AND o.created_at > NOW() - INTERVAL '30 days';

-- Look for:
--   Seq Scan (should be Index Scan for large tables)
--   High actual_rows vs estimated rows (stale statistics → ANALYZE)
--   Nested Loop with large row counts
--   Hash Join spilling to disk
```

**Step 3 — Check for N+1 queries:**
Enable query logging for the endpoint. Count total queries. If > 5, look for ORM lazy loading. Fix with eager loading or batch fetching.

**Step 4 — Add appropriate caching:**
```python
# Cache expensive, read-heavy, infrequently changing data
@cache.cached(timeout=300, key_prefix=lambda: f"dashboard:{current_user.id}")
def get_dashboard_data():
    return compute_expensive_aggregation()
```

**Step 5 — Check concurrency and connection pool:**
```
Metrics to check:
  - DB connection pool wait time: if > 5ms, pool is undersized
  - Thread pool queue depth: if > 0, add workers
  - External API call timeout: add circuit breaker and timeout
  - Response payload size: if > 100KB, add pagination or field selection
```

**Quick wins by frequency:**
1. Missing DB index (most common)
2. N+1 query from ORM lazy loading
3. No caching on expensive read
4. Synchronous external API call (move to async)
5. Fetching entire rows when only 2 columns needed (SELECT *)

---

### Q9. How do database read replicas help performance and what are their limitations?

**Read replicas** are copies of the primary database that receive changes via replication and can serve read queries — offloading read traffic from the primary DB.

**Architecture:**
```
Writes → Primary DB ──async replication──► Replica 1 → Read queries
                    ──async replication──► Replica 2 → Read queries
                    ──async replication──► Replica 3 → Analytics queries

Application routing:
  db_write = connect(primary_host)
  db_read  = connect(replica_host, round_robin)

# Read-heavy workloads (80% reads, 20% writes) benefit greatly:
# Writes: 1 primary (no change)
# Reads: spread across N replicas → N× read throughput
```

**When to use read replicas:**
- Dashboard and reporting queries (heavy aggregations)
- Search and filtering operations
- Read-heavy API endpoints (product catalog, blog posts)
- Analytics and business intelligence queries

**Limitations and pitfalls:**

| Limitation               | Detail                                                  |
|--------------------------|---------------------------------------------------------|
| Replication lag          | Replica may be 100ms–seconds behind primary             |
| Stale reads              | Read-after-write may show old data if read hits replica |
| No write offloading      | Replicas handle reads only; writes still hit primary    |
| Lag under load           | Heavy write workloads increase replica lag              |
| Connection management    | App must implement read/write splitting logic           |

**Handling replication lag:**
```python
def get_user_profile(user_id, just_updated=False):
    # After a write, read from primary to avoid stale read
    if just_updated:
        return primary_db.query("SELECT * FROM users WHERE id = %s", user_id)
    else:
        return replica_db.query("SELECT * FROM users WHERE id = %s", user_id)
```

For strong consistency after writes, use "read your own writes" routing: for a short window after a write, route that user's reads to the primary. After the window expires (replication catches up), route back to replicas.

---

### Q10. What are pre-computation and materialized views and when do you use them?

**Pre-computation** means computing expensive aggregations in advance and storing the results, so queries retrieve pre-computed values rather than computing them on-demand.

**Materialized Views (Database level):**
```sql
-- Expensive query: compute total sales per product per day (runs on every dashboard load)
-- With 100M orders table: takes 30 seconds

-- Instead, create a materialized view:
CREATE MATERIALIZED VIEW daily_product_sales AS
  SELECT
    product_id,
    DATE(created_at) AS sale_date,
    SUM(amount) AS total_sales,
    COUNT(*) AS order_count
  FROM orders
  GROUP BY product_id, DATE(created_at);

-- Create index on materialized view
CREATE INDEX ON daily_product_sales(product_id, sale_date);

-- Dashboard query: instantaneous (reads pre-computed data)
SELECT * FROM daily_product_sales
WHERE product_id = 42 AND sale_date >= '2025-01-01';

-- Refresh the materialized view on a schedule:
-- Concurrent refresh (no read lock):
REFRESH MATERIALIZED VIEW CONCURRENTLY daily_product_sales;
-- Typically via cron: every hour, or triggered by data pipeline
```

**Application-level pre-computation (Redis):**
```python
# Background job runs every 5 minutes
def precompute_homepage_stats():
    stats = {
        "total_users": db.count_users(),
        "sales_today": db.sum_sales_today(),
        "trending_products": db.get_trending(limit=10)
    }
    redis.setex("homepage_stats", 300, json.dumps(stats))  # TTL 5 min

# Request handler: instant
def homepage():
    stats = json.loads(redis.get("homepage_stats"))
    return render(stats)
```

**When to pre-compute vs. compute on-demand:**

| Factor                  | Pre-compute              | On-demand                  |
|-------------------------|--------------------------|----------------------------|
| Query frequency         | High (> 100/min)         | Low (< 10/min)             |
| Data freshness required | Minutes acceptable       | Real-time required         |
| Computation cost        | Seconds–minutes          | Milliseconds               |
| User count              | All users see same result| User-specific results      |

Pre-computation trades storage and refresh complexity for dramatically better read performance. Use it aggressively for shared aggregations (leaderboards, dashboards, analytics) and avoid it for user-personalized or real-time data.

---

### Q11. What is write batching and how does it reduce I/O?

**Write batching** groups multiple write operations together and executes them in a single I/O operation, reducing round trips and amortizing per-operation overhead.

**Why individual writes are expensive:**
```
Single-row INSERT:
  App → (TCP) → DB → WAL write → fsync → (TCP) → App ACK
  Cost: 1 network RTT + 1 fsync (slow: ~10ms on spinning disk, ~1ms on SSD)

1000 single INSERTs = 1000 × 10ms = 10 seconds!
```

**Batch INSERT:**
```sql
-- Single batch of 1000 rows:
INSERT INTO events (user_id, event_type, created_at)
VALUES (1, 'click', NOW()),
       (2, 'view', NOW()),
       ... (1000 rows total)

-- Cost: 1 network RTT + 1 fsync = 10ms for all 1000 rows
-- Throughput: 100× better than individual inserts
```

**Application-level batching (buffer before flush):**
```python
class BatchWriter:
    def __init__(self, db, batch_size=500, flush_interval_ms=100):
        self.buffer = []
        self.batch_size = batch_size
        self.flush_interval = flush_interval_ms

    def write(self, record):
        self.buffer.append(record)
        if len(self.buffer) >= self.batch_size:
            self.flush()

    def flush(self):
        if self.buffer:
            db.execute_many("INSERT INTO events VALUES (%s, %s, %s)", self.buffer)
            self.buffer = []
```

**Kafka producer batching:**
```
kafka-producer config:
  linger.ms = 5          # Wait up to 5ms to accumulate records
  batch.size = 16384     # Batch up to 16KB before sending
  compression.type = lz4 # Compress batch

Effect:
  Without batching: 1000 messages → 1000 network requests
  With batching: 1000 messages → ~10-20 batches (100× fewer round trips)
```

**Trade-off:** Batching introduces latency (must wait for batch to fill). This is acceptable for non-interactive workloads (analytics, logging) but inappropriate for user-facing writes that need immediate confirmation.

---

### Q12. How does connection keep-alive and HTTP/2 multiplexing reduce latency?

**HTTP/1.1 Keep-Alive:**
Without keep-alive, every HTTP request requires a new TCP connection: TCP handshake (1 RTT) + TLS handshake (1-2 RTTs) before the first byte of data.

```
Without keep-alive:
  Request1: TCP SYN/ACK (1ms) + TLS (3ms) + HTTP (5ms) = 9ms
  Request2: TCP SYN/ACK (1ms) + TLS (3ms) + HTTP (5ms) = 9ms
  10 requests = ~90ms of connection overhead

With keep-alive:
  Request1: TCP SYN/ACK (1ms) + TLS (3ms) + HTTP (5ms) = 9ms
  Request2: reuse connection + HTTP (5ms) = 5ms
  Request3-10: 5ms each
  10 requests = 9ms + 9 × 5ms = 54ms  (40% reduction)
  
  Connection: keep-alive  (HTTP header)
```

**HTTP/2 Multiplexing:**
HTTP/1.1 is limited to ~6 parallel connections per domain (browser restriction) and requires head-of-line blocking (each connection handles one request at a time).

```
HTTP/1.1 (pipelining rarely used):
  Conn1: [GET /a] → [GET /b] (sequential on same connection)
  
  Browser opens 6 connections to handle parallel requests:
  Conn1: /a
  Conn2: /b
  Conn3: /c
  ... up to 6 parallel connections

HTTP/2 Multiplexing:
  Single connection with multiple streams:
  
  Stream 1: [GET /a header] ──────────────── [/a body]
  Stream 2:     [GET /b header] ────── [/b body]
  Stream 3:         [GET /c header] [/c body]
  
  All on ONE TCP connection, fully parallel, no head-of-line blocking
  Priority and dependency hints for critical resources
```

**HTTP/2 for API microservices:**
```
gRPC uses HTTP/2 by default:
  - Multiplexed streams: one connection handles thousands of concurrent RPCs
  - Header compression (HPACK): repeated headers (auth token) compressed after first request
  - Binary protocol: more efficient than HTTP/1.1 text
  - Server push: server can pre-send resources client will need
```

In practice, HTTP/2 reduces connection overhead by 60–80% for pages with many resources and eliminates the need for domain sharding (an HTTP/1.1 workaround).

---

### Q13. What is load testing methodology? Cover ramp-up, steady state, and spike testing.

Load testing verifies that a system meets performance requirements under expected and peak traffic. Without load testing, systems routinely fail at launch.

**Load test phases:**
```
Traffic:
  │          ┌──────────────────────────┐
  │         /│    Steady State          │\
  │        / │    (target: 1000 RPS)    │ \
  │       /  │                          │  \
  │Ramp-up   │                          │  Ramp-down
  │          │                          │
  └──────────────────────────────────────────────► Time
     0min   5min                      25min  30min
```

**Ramp-up:** Gradually increase load from 0 to target. This tests that auto-scaling triggers correctly and the system stabilizes before full load hits. A sudden jump to full traffic is unrealistic and hides the warming period.

**Steady state:** Maintain target load for 20–30 minutes. This validates that the system remains stable under sustained load — memory leaks, connection pool exhaustion, and GC pressure often only appear after minutes of sustained load.

**Spike test:** Sudden jump to 2–5× normal load. Validates auto-scaling speed and graceful degradation.

**Stress test:** Gradually increase load past the system's capacity to find the breaking point and confirm the failure mode is graceful (queuing, shedding) rather than a crash.

**Tool example (k6):**
```javascript
import http from 'k6/http';
import { sleep } from 'k6';

export let options = {
  stages: [
    { duration: '5m',  target: 100  },  // Ramp-up to 100 users
    { duration: '20m', target: 100  },  // Steady state
    { duration: '2m',  target: 500  },  // Spike
    { duration: '3m',  target: 100  },  // Recovery
    { duration: '5m',  target: 0    },  // Ramp-down
  ],
  thresholds: {
    http_req_duration: ['p95<200'],     // 95% of requests under 200ms
    http_req_failed: ['rate<0.01'],     // Error rate under 1%
  },
};

export default function() {
  http.get('https://api.example.com/products');
  sleep(1);
}
```

**What to measure during load test:**
- p50, p95, p99 latency at each load level
- Error rate (should be near 0 at target load)
- CPU, memory, DB connection pool utilization
- Auto-scaling trigger time (how long until new instances are serving traffic)

---

### Q14. How do you set performance SLOs and measure p50/p95/p99 latency?

**SLOs (Service Level Objectives)** are internal performance targets that define what "good enough" performance looks like. They drive engineering priorities and alert thresholds.

**Latency percentiles:**
```
Given 100 requests with these response times (ms):
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, ... 500ms (1 outlier)]

p50 (median):   50th percentile = 50% of requests are faster than this
p95:            95% of requests complete in under this time
p99:            99% of requests complete in under this time
p99.9 (0.1%):   The worst 0.1% of requests (1 in 1000)

Typical values for a healthy API:
  p50:   20ms    (typical case)
  p95:   80ms    (most users' experience)
  p99:   200ms   (occasional slow requests)
  p99.9: 1000ms  (rare worst case)
```

**Why percentiles beat averages:**
```
Example: 99 requests at 10ms, 1 request at 10000ms
  Average: (99×10 + 10000) / 100 = 109.9ms  ← looks OK
  p99:     10000ms                           ← reveals the problem
  p50:     10ms                              ← most users fine
```

**Setting SLOs — process:**
```
1. Measure current state: what is today's p95, p99?
2. Understand user experience: at what latency do users abandon?
   (Research shows: 100ms feels instant, 1s acceptable, 3s+ causes abandonment)
3. Set SLO slightly above current state:
   Current p99 = 150ms → SLO p99 < 200ms (error budget of 50ms)
4. Set alert threshold at 80% of SLO:
   Alert when p99 > 160ms (gives time to investigate before SLO breach)
```

**Error budget:**
```
99.9% SLO = 0.1% allowed to be slow
In a month with 10M requests: 10,000 can exceed SLO
Once error budget exhausted: freeze new feature work, focus on reliability
```

SLOs should be specific: "p99 latency < 200ms, measured over 5-minute windows, excluding planned maintenance windows."

---

### Q15. What is the "mechanical sympathy" principle?

**Mechanical sympathy** (coined by Martin Thompson) is the principle that software developers who understand how the underlying hardware works can write dramatically more efficient software. Named after racing driver Jackie Stewart: "You don't have to be an engineer to be a racing driver, but you do have to have mechanical sympathy."

**Key hardware properties that matter:**

**1. CPU Cache Hierarchy:**
```
L1 Cache: 4KB–64KB, ~1ns access   ← fits few hundred objects
L2 Cache: 256KB–1MB, ~5ns access
L3 Cache: 4MB–32MB, ~30ns access
Main RAM:                ~100ns access
Disk SSD:               ~100,000ns (100µs)

Implication: iterating a contiguous array is 10–100× faster than
following pointers (linked list) because arrays are cache-friendly.

// Cache-friendly (sequential memory):
int[] arr = new int[1000000];
for (int i = 0; i < arr.length; i++) sum += arr[i];  // Fast: cache lines prefetched

// Cache-unfriendly (pointer chasing):
Node head = linkedList.head;
while (head != null) { sum += head.value; head = head.next; }  // Slow: cache misses
```

**2. False Sharing (multi-threaded):**
```
Cache line = 64 bytes. If two threads modify different variables
that happen to share a cache line, each write invalidates the
other thread's cache entry → performance degrades to serial.

Fix: pad variables to 64-byte boundaries (Java: @Contended annotation)
```

**3. Branch Prediction:**
Modern CPUs predict which branch of an if/else will be taken and pre-execute it. Unpredictable branches (50/50 random) cause pipeline flushes.

**4. NUMA (Non-Uniform Memory Access):**
In multi-socket servers, RAM is physically closer to one CPU socket. Accessing remote RAM costs ~2× more. NUMA-aware allocation pins threads and memory to the same socket.

**Practical applications:**
- Use arrays over linked lists where possible
- Keep hot data structures small (fit in L1/L2 cache)
- Process data in cache-line-sized chunks
- Align network packet processing to avoid packet segmentation
- Disruptor pattern (LMAX) uses ring buffers for cache-line-optimized inter-thread communication

---

## Hard (Q16–Q20)

---

### Q16. How do CPU flame graphs and memory heap dumps work as profiling techniques?

**CPU Flame Graphs** (invented by Brendan Gregg) are visualizations of CPU profiling data that show which code paths are consuming CPU time.

**How flame graphs work:**
```
Profiler samples the call stack every 10ms:
  Sample 1:  main() → handleRequest() → parseJSON() → malloc()
  Sample 2:  main() → handleRequest() → dbQuery() → pgExecute()
  Sample 3:  main() → handleRequest() → dbQuery() → pgExecute()
  Sample 4:  main() → handleRequest() → dbQuery() → networkRead()
  ...

Flame graph aggregates samples:
  Width of each bar = % of total CPU samples in that function
  
  ┌────────────────────────────────────────────────┐
  │                  main()                        │ 100%
  ├─────────────────────────────────────────────────┤
  │              handleRequest()                   │ 98%
  ├─────────────────────┬──────────────────────────┤
  │    parseJSON() 20%  │    dbQuery() 78%          │
  ├──────────┬──────────┼───────────┬──────────────┤
  │malloc()  │ other    │pgExecute()│ networkRead() │
  │  12%     │  8%      │  60%      │  18%          │
  └──────────┴──────────┴───────────┴──────────────┘

Reading: Look for wide bars at the top = hot code paths
→ dbQuery() 78% of CPU time is the bottleneck
```

**Generating flame graphs:**
```bash
# Linux perf (for native apps)
perf record -F 99 -p <PID> -g -- sleep 30
perf script | stackcollapse-perf.pl | flamegraph.pl > flame.svg

# Java async-profiler
java -agentpath:libasyncProfiler.so=start,event=cpu,file=flame.html -jar app.jar

# Python (py-spy)
py-spy record -o flame.svg --pid <PID>
```

**Memory Heap Dumps:**
A heap dump is a snapshot of all objects in the JVM/Python heap at a point in time. Used to find memory leaks — objects that should be garbage collected but are still reachable.

```
Analysis process:
1. Take heap dump: jcmd <PID> GC.heap_dump /tmp/heap.hprof
2. Load in Eclipse MAT or VisualVM
3. Look for:
   - Largest objects by retained heap
   - Objects growing unboundedly (session cache, event listeners)
   - GC roots preventing collection

Common findings:
  - Session objects holding references to large data
  - Static collections growing without bound (event bus listeners)
  - Thread-local variables not cleaned up
  - Off-heap memory leaks (native libs)
```

**EXPLAIN ANALYZE for query profiling:**
```sql
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT * FROM orders WHERE customer_id = 42 AND status = 'pending';

-- Output shows:
-- Seq Scan (bad for large tables) vs Index Scan (good)
-- Actual vs Estimated rows (big difference → stale statistics → run ANALYZE)
-- Buffers hit (memory) vs read (disk) → low hit ratio → cache too small
-- Execution time broken down per node
```

---

### Q17. How do memory-mapped files and zero-copy techniques improve performance?

**Memory-Mapped Files (mmap):**
`mmap()` maps a file directly into the process's virtual address space. The OS manages loading pages on demand. Reads do not require an extra copy from kernel buffer to user space — the process reads file data directly from the page cache.

```
Traditional file read:
  Disk → Kernel buffer (page cache) → copy → User buffer → Process
         ↑ copy 1                  ↑ copy 2

Memory-mapped file:
  Disk → Kernel buffer (page cache) ← Process reads directly (same physical memory)
         No copy needed! Process accesses file as if it's an in-memory array.

mmap in C:
  fd = open("data.bin", O_RDONLY);
  void* data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  // Access data[offset] directly — OS handles paging from disk as needed
```

**Benefits:**
- No explicit `read()` system calls for random access
- OS page cache is shared across processes mapping the same file
- Sequential access triggers OS read-ahead (prefetching)
- Ideal for: databases (RocksDB, LMDB), search indexes (Lucene)

**Zero-Copy (sendfile):**
Traditional `send()` of a file over network requires 4 copies:
```
Traditional:
  1. Disk → Kernel read buffer (DMA)
  2. Kernel read buffer → User space app buffer (CPU copy)
  3. User space → Kernel socket buffer (CPU copy)
  4. Kernel socket buffer → NIC (DMA)

sendfile() system call (zero-copy):
  1. Disk → Kernel read buffer (DMA)
  2. Kernel read buffer → NIC (DMA, using DMA gather)
  No CPU copies at all!

Java NIO equivalent:
  FileChannel fileChannel = new FileInputStream(file).getChannel();
  SocketChannel socketChannel = ...;
  fileChannel.transferTo(0, fileChannel.size(), socketChannel);
  // Uses sendfile() under the hood — Kafka uses this for consumer fetches
```

**Real-world use:**
- **Kafka:** Uses zero-copy (`sendfile`) to transfer log segments from disk to consumer sockets — critical for high-throughput message delivery without CPU bottleneck
- **Nginx:** Uses `sendfile` for static file serving
- **Lucene (Elasticsearch):** Uses `mmap` for segment index access

For a server sending 10GB of files, zero-copy reduces CPU usage from ~30% to ~3% and doubles throughput.

---

### Q18. How do you design and tune a connection pool for maximum throughput?

Connection pool tuning is both art and science — undersized pools create queuing, oversized pools overwhelm the database.

**The database connection bottleneck:**
```
PostgreSQL with 100 connections:
  100 OS processes forked (each using ~5-10MB RAM)
  Scheduler context-switches between 100 processes
  Diminishing returns: 100 connections on 8 CPUs ≈ same throughput as 20
  
  PostgreSQL can max out CPU with ~(2 × CPUs) active connections doing queries
  Beyond that: more connections = more context switching = lower throughput
```

**Pool sizing formula (OLTP):**
```
Optimal pool size ≈ (CPU cores on DB host × 2) + effective_spindles

For RDS db.r6g.4xlarge (16 vCPUs, SSD):
  Pool size ≈ 16 × 2 + 1 = 33 ≈ 30-40 connections total
  
  With 10 app instances: 35 / 10 = 3-4 connections per app instance pool
```

**HikariCP (Java) production config:**
```java
HikariConfig config = new HikariConfig();
config.setMaximumPoolSize(10);           // Max connections per app instance
config.setMinimumIdle(5);               // Pre-warmed connections
config.setConnectionTimeout(30000);     // 30s: max wait to acquire connection
config.setIdleTimeout(600000);         // 10m: close idle connections
config.setMaxLifetime(1800000);        // 30m: recycle connections (avoids stale)
config.setKeepaliveTime(60000);        // 1m: send keepalive to prevent firewall timeout
config.setLeakDetectionThreshold(5000); // Warn if connection held > 5s (leak detection)
```

**PgBouncer between app and PostgreSQL:**
```
App instances (1000): each holds small pool (2-3 conns)
     ↓ 2000-3000 client connections
PgBouncer (transaction pooling):
     ↓ 30-50 actual server connections
PostgreSQL: healthy, never overwhelmed

PgBouncer absorbs connection storms
Without PgBouncer: deploy of 100 new instances creates 1000 new DB connections instantly
With PgBouncer: deploy creates 200-300 PgBouncer client connections → no impact on DB
```

**Monitoring pool health:**
```
Metrics to watch:
  pool_wait_time_ms: time waiting to acquire connection
  pool_pending_requests: queue depth (>0 means pool saturated)
  pool_active_connections: currently in use
  pool_idle_connections: available (should be > 0 at all times)
  
Alert thresholds:
  pool_wait_time_ms > 10ms → pool undersized or DB slow
  pool_pending_requests > 0 → immediate action needed
```

---

### Q19. How do you identify and resolve database index optimization issues?

Index optimization is one of the highest-leverage performance improvements — a query that takes 30 seconds with a table scan may take 1ms with the right index.

**Index types and use cases:**

```sql
-- B-Tree index (default): equality and range queries
CREATE INDEX idx_orders_customer ON orders(customer_id);
-- Useful for: WHERE customer_id = 42
--             WHERE customer_id IN (1,2,3)
--             ORDER BY customer_id

-- Composite index: multi-column queries
-- COLUMN ORDER MATTERS: most selective column first, then range/sort columns
CREATE INDEX idx_orders_status_created ON orders(status, created_at);
-- Useful for: WHERE status = 'pending' AND created_at > '2025-01-01'
-- NOT useful for: WHERE created_at > '2025-01-01' (skips leading column)

-- Covering index: include all columns needed by query (no heap fetch)
CREATE INDEX idx_orders_covering ON orders(customer_id) INCLUDE (total, status);
-- Query: SELECT total, status FROM orders WHERE customer_id = 42
-- Index contains all needed data → no row lookup → faster

-- Partial index: index only a subset of rows
CREATE INDEX idx_pending_orders ON orders(created_at) WHERE status = 'pending';
-- 10M orders, only 50K pending → tiny index → very fast
-- Useful when only querying a small predicate subset
```

**Detecting missing indexes:**
```sql
-- Find slow queries (pg_stat_statements):
SELECT query, mean_exec_time, calls, total_exec_time
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 20;

-- Find sequential scans on large tables:
SELECT relname, seq_scan, idx_scan,
       seq_scan / (seq_scan + idx_scan + 1) AS seq_ratio
FROM pg_stat_user_tables
WHERE seq_scan > 1000
ORDER BY seq_ratio DESC;

-- Missing index candidates (high seq_scan ratio on large tables = add index)
```

**Index maintenance:**
```sql
-- Bloated indexes (after many UPDATEs/DELETEs):
SELECT indexname, pg_size_pretty(pg_relation_size(indexname::regclass))
FROM pg_indexes WHERE tablename = 'orders';

-- Rebuild bloated index concurrently (no lock):
REINDEX INDEX CONCURRENTLY idx_orders_customer;

-- Unused indexes (waste write performance):
SELECT indexname FROM pg_stat_user_indexes
WHERE idx_scan = 0 AND indexrelname NOT LIKE 'pg_%';
-- Drop indexes that have never been used (after 30+ days)
```

**Index column order rule:**
```
For composite index (a, b, c):
  Query: WHERE a=? AND b=? → YES (leftmost prefix)
  Query: WHERE a=? AND c=? → Partial (uses a, filters c in-memory)
  Query: WHERE b=? AND c=? → NO (skips a, can't use index)
  
Rule: Put equality columns first, range columns last, high-cardinality before low.
```

---

### Q20. How do you optimize a system end-to-end? Walk through a case study of a slow e-commerce checkout.

**Scenario:** Checkout endpoint takes 3.2 seconds. Business impact: 40% cart abandonment above 2 seconds. SLO target: p99 < 500ms.

**Step 1 — Instrument and trace:**
```
Distributed trace for POST /checkout:

  [API Gateway]          10ms
  [Auth Service]         15ms
  [Inventory Service]   800ms  ← HIGH
    → 50 individual item lookups (N+1 problem!)
  [Price Calculation]    200ms
    → No cache, recomputes discount rules each time
  [Payment Gateway]      600ms  ← external, irreducible
  [Order DB Write]        20ms
  [Email Notification]   900ms  ← synchronous, can defer
  [Fraud Check]          500ms  ← synchronous, can async
  Total: 3,245ms
```

**Step 2 — Fix N+1 in Inventory Service:**
```python
# Before: 50 queries for 50 items
for item in cart.items:
    inventory = db.get_inventory(item.product_id)  # 50 queries

# After: 1 batch query
product_ids = [item.product_id for item in cart.items]
inventories = db.get_inventory_batch(product_ids)  # 1 query
# Improvement: 800ms → 25ms (32× faster)
```

**Step 3 — Cache price calculation:**
```python
# Discount rules change infrequently (updated by admin)
@cache.cached(timeout=300, key="discount_rules")
def get_discount_rules():
    return db.load_discount_rules()

# Per-user price can be cached for 60s
cache_key = f"cart_price:{user_id}:{cart_hash}"
cached_price = redis.get(cache_key)
if not cached_price:
    price = calculate_price(cart, get_discount_rules())
    redis.setex(cache_key, 60, price)
# Improvement: 200ms → 2ms (cache hit)
```

**Step 4 — Move non-critical work async:**
```python
def checkout(cart, user):
    # Critical path: validate + charge + create order
    validate_inventory(cart)              # 25ms (fixed)
    charge = payment_gateway.charge(...)  # 600ms (irreducible)
    order = db.create_order(cart, charge) # 20ms

    # Defer everything else to queue
    queue.publish("order_created", {
        "order_id": order.id,
        "user_id": user.id
    })
    # Email, fraud check, analytics processed in background

    return {"order_id": order.id}  # Respond immediately
```

**Step 5 — Results:**
```
Before optimization:
  p50: 2.8s  |  p95: 3.5s  |  p99: 4.2s

After optimization:
  API Gateway:        10ms (unchanged)
  Auth Service:       15ms (unchanged)
  Inventory:          25ms (was 800ms: N+1 fixed)
  Price Calculation:   2ms (was 200ms: cached)
  Payment Gateway:   600ms (irreducible: external API)
  Order DB Write:     20ms (unchanged)
  Background queue:   5ms  (was 1400ms: deferred)
  Total: ~677ms

  p50: 550ms  |  p95: 700ms  |  p99: 850ms

Further: add connection pool tuning, CDN for product images, read replica for catalog
After: p99 < 500ms SLO achieved
```

---

## Quick Reference

### The Optimization Cycle
`Measure → Identify Bottleneck → Fix → Measure`

### Little's Law
`L = λ × W`  (concurrent requests = arrival rate × average latency)

### USE Method
- **U**tilization: how busy is the resource?
- **S**aturation: how much is queued/waiting?
- **E**rrors: any error events?

### Latency Percentiles
- p50 = median (typical user)
- p95 = most users' experience
- p99 = worst 1 in 100
- p99.9 = worst 1 in 1000

### Connection Pool Sizing
`pool_size ≈ (DB CPU cores × 2) + spindles`

### N+1 Fix Options
| Method      | When to use                       |
|-------------|-----------------------------------|
| JOIN        | Simple parent-child relationships |
| Eager load  | ORM with known associations       |
| DataLoader  | GraphQL resolvers                 |
| Batch fetch | Custom repositories               |

### Caching Targets
- CDN cache-hit ratio: > 95%
- DB cache (buffer pool hit ratio): > 99%
- Redis cache-hit ratio: > 90%

### Zero-Downtime Checklist
1. Measure baseline (p50/p95/p99)
2. Profile with traces (find hotspot)
3. Fix N+1 queries first (highest ROI)
4. Cache shared read-heavy data
5. Move non-critical work to queues
6. Add read replicas for read-heavy load
7. Tune connection pools
8. Load test after each change
