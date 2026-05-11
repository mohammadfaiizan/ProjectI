# System Design Anti-Patterns — Interview Q&A

---

## Easy (Q1–Q7)

---

### Q1. What is a distributed monolith and how do you avoid it?

A **distributed monolith** is a system deployed as multiple services (looks like microservices) but behaves like a single monolith — every deployment requires coordinating all services simultaneously, and they are tightly coupled through shared databases, synchronous dependencies, or coordinated releases.

**How it happens:**
The team splits a monolith into separate deployable units but doesn't change the coupling patterns. Services call each other synchronously for almost every operation, share a single database, and cannot function without every other service being up.

```
Distributed Monolith (anti-pattern):

  Service A → Service B → Service C → Service D → Shared DB
  
  - A calls B synchronously for every request
  - B calls C synchronously for every request
  - All services share the same PostgreSQL database
  - Deploy service A requires deploying B, C, D in correct order
  - B is down → A fails → cascade failure
  
  This is NOT microservices. It's a monolith split across the network —
  all the complexity of distributed systems with none of the benefits.
```

**Signs you have a distributed monolith:**
- Services can't be deployed independently
- Services can't be scaled independently (they share a DB bottleneck)
- A single service failure cascades to bring down everything
- Teams can't release without coordinating with other teams
- The "deployment window" requires all services at once

**How to avoid it:**

| Distributed Monolith              | True Microservices                    |
|-----------------------------------|---------------------------------------|
| Shared database                   | Each service owns its own data store  |
| Synchronous chains for every call | Async events for non-critical paths   |
| Coordinated deploys               | Independent deployability             |
| Circular dependencies             | Dependency graph is a DAG             |
| Shared libraries with logic       | Logic in the service, not shared lib  |

True microservices: service A can be deployed without touching B, C, D. Service B can be down and service A degrades gracefully rather than failing hard.

---

### Q2. What is the "big ball of mud" and how do systems degrade into one?

A **Big Ball of Mud** (Brian Foote & Joseph Yoder, 1997) is a system with no discernible architecture — a tangled mass of code where everything depends on everything else, with no clear boundaries, modules, or separation of concerns.

**How systems degrade:**
```
Year 1 (Startup):
  Simple app: UserController → UserService → UserRepository → DB
  Clean, fast to build, no time for over-engineering

Year 2 (Growth):
  "Quick fix" — OrderController directly calls UserRepository
  "Just this once" — payment logic added to UserService
  Schema shared between orders and users (one table for both)

Year 3 (More Growth):
  Notification logic scattered across 5 services
  User schema has 80 columns including unrelated data
  Changing one field breaks 12 unrelated features
  No one understands how the whole system works

Year 4 (Big Ball of Mud):
  Every change breaks something unexpected
  New features take 10× longer than they should
  Bug fixes introduce new bugs
  Onboarding new engineers takes months
```

**Contributing factors:**
- **Pressure to ship:** shortcuts accumulate ("we'll clean it up later")
- **No enforcement:** no architecture review, no PR checks for boundary violations
- **Shared state:** global variables, shared mutable caches, shared databases
- **Copy-paste programming:** duplicated logic that diverges over time

**How to prevent degradation:**
- Define bounded contexts (Domain-Driven Design)
- Enforce module boundaries (ArchUnit for Java, dependency-cruiser for Node.js)
- Treat tech debt as a first-class backlog item
- Architecture decision records (ADRs) to document why decisions were made
- Modular monolith as a stepping stone: separate modules with clean interfaces even in a single deployable

The irony of the Big Ball of Mud: systems become it through many individually reasonable shortcuts. Preventing it requires discipline applied consistently, not a single rewrite.

---

### Q3. What is premature optimization and when should you actually optimize?

**Premature optimization** is spending engineering effort optimizing code or systems before there is evidence they are performance bottlenecks. Donald Knuth: "Premature optimization is the root of all evil."

**The anti-pattern:**
```python
# Engineer spends 3 days writing custom cache because "DB will be slow"
# Before writing any DB queries or measuring anything

class UltraFastCache:
    def __init__(self):
        self.memory = {}
        self._lock = threading.RLock()
        self._lru_queue = deque()
        # ... 200 lines of custom cache logic
    
# Actual DB query on the real dataset: 2ms
# Cache implementation: 3 days of engineering
# Actual user-facing impact: unmeasurable (2ms vs 1.9ms)
```

**The problem with premature optimization:**
1. You optimize the wrong thing (the bottleneck is elsewhere)
2. Optimized code is harder to read, maintain, and debug
3. Engineering time spent on optimization cannot be spent on features
4. The "slow" path may never actually be called in production

**The correct approach:**
```
1. Write the simplest correct solution first
2. Measure (deploy, observe in production or load test)
3. Identify actual bottlenecks (top 1-2 contributors to latency/cost)
4. Optimize only those with clear evidence of impact
5. Measure again to confirm improvement

"First make it work, then make it right, then make it fast." — Kent Beck
```

**When you SHOULD optimize:**
- You have measured data showing a specific component is the bottleneck
- You are designing for known scale (e.g., "we will have 1M users on launch day")
- Infrastructure costs are measurably too high (not "might be high")
- User-facing latency SLO is being violated
- The algorithm is provably O(n²) and n is expected to be large

The key distinction: optimization driven by measurements is engineering. Optimization driven by intuition is often wasted effort.

---

### Q4. What is a Single Point of Failure and why is it the most common system design mistake?

A **Single Point of Failure (SPOF)** is any component whose failure causes the entire system to become unavailable. It is the most common mistake because the first version of almost every system is designed with a SPOF — usually the database.

```
Classic SPOF architecture:
  Users → Load Balancer → App Server → PostgreSQL (single instance)
                                            ↑ SPOF
  
  DB crashes at 2 AM → entire system down
  DB disk fills up → entire system down
  DB CPU spike → entire system down
  
  Even if app servers, load balancer, network are all redundant:
  the single DB negates all that redundancy.
```

**Most common SPOFs and their frequency in real systems:**

| Component        | SPOF Pattern                              | Mitigation                             |
|------------------|-------------------------------------------|----------------------------------------|
| Database         | Single primary, no replica                | Primary + replica, Patroni failover    |
| Cache            | Single Redis instance                     | Redis Sentinel or Cluster              |
| Load Balancer    | Single LB process (not cloud-managed)     | HA LB pair (Keepalived) or cloud LB    |
| Message Queue    | Single RabbitMQ/Kafka broker              | Kafka cluster (3 brokers), MSK         |
| DNS              | Single DNS server                         | Managed DNS (Route 53, Cloudflare)     |
| Auth Service     | Single auth instance                      | Multiple replicas, no shared state     |
| API Gateway      | Single gateway with no replicas           | Multiple instances, cloud-managed      |
| Storage          | Local disk with no backup                 | Object storage (S3), replicated EBS    |

**Why it's so common:**
1. Development always starts with simple, single-node setup
2. Cost pressure: running two of everything costs 2×
3. "We'll add HA later" — later never comes until the outage
4. Systems appear fine with a SPOF until they suddenly aren't

**The 2 AM test:** For every critical component, ask: "If this fails at 2 AM when no engineer is awake, does the system continue serving users?" If the answer is no, you have a SPOF.

---

### Q5. What is over-engineering and how do you recognize it?

**Over-engineering** is building a more complex solution than the problem requires — adding abstractions, scalability, and features that are not needed now and may never be needed.

**Classic over-engineering examples:**

**Example 1 — Premature microservices:**
```
Problem: Build an MVP for a new SaaS product (expected: 100 users)

Over-engineered solution:
  14 microservices, Kubernetes, Kafka, Redis, Elasticsearch,
  service mesh (Istio), GitOps pipeline, 3 cloud regions
  
  Engineering time: 6 months to launch
  
Simple correct solution:
  1 monolith, 1 Postgres DB, 1 Redis, deployed on a single PaaS
  Engineering time: 3 weeks to launch
  
  When you actually have 10,000 users and need to scale: refactor then.
  3 months faster to market, earlier customer feedback, pivot if needed.
```

**Example 2 — Unnecessary abstractions:**
```python
# Over-engineered: abstract factory for one implementation
class AbstractDataProviderFactory(ABC):
    @abstractmethod
    def create_user_reader(self) -> AbstractUserReader: ...
    
    @abstractmethod
    def create_user_writer(self) -> AbstractUserWriter: ...

class PostgresDataProviderFactory(AbstractDataProviderFactory):
    def create_user_reader(self) -> PostgresUserReader:
        return PostgresUserReader()
    
    def create_user_writer(self) -> PostgresUserWriter:
        return PostgresUserWriter()

# Simple correct solution (you have one DB, one codebase):
def get_user(user_id: int) -> User:
    return db.execute("SELECT * FROM users WHERE id = %s", user_id)
```

**Signs of over-engineering:**
- More time spent on infrastructure than business logic
- Abstractions that are never instantiated more than once
- Solving problems you don't have yet
- Design documents thicker than product specs
- New team members need 2 weeks to understand the "Hello World" deploy

**The YAGNI principle:** "You Aren't Gonna Need It." Only build what is needed now. The cost of adding complexity early is higher than adding it later when the need is real and the requirements are better understood.

---

### Q6. What is the "database as a message queue" anti-pattern?

Using a relational database to implement a work queue — where producers INSERT rows and consumers SELECT + DELETE rows to process work — is a widespread anti-pattern that causes serious performance problems at scale.

**The pattern:**
```sql
-- Producer inserts a job
INSERT INTO job_queue (task_type, payload, status, created_at)
VALUES ('send_email', '{"user_id": 42}', 'pending', NOW());

-- Consumer polls for work
SELECT * FROM job_queue
WHERE status = 'pending'
ORDER BY created_at
LIMIT 10
FOR UPDATE SKIP LOCKED;

-- Consumer processes, then marks complete
UPDATE job_queue SET status = 'done' WHERE id = ?;
```

**Why it's problematic:**
```
1. Polling creates constant DB load:
   Consumer polls every second → 86,400 queries/day with no work to do
   10 consumers × 86,400 = 864,000 polling queries/day generating no value

2. Table bloat:
   "done" rows accumulate → table grows → queries slow down
   VACUUM required to reclaim space (blocks during large cleanups)

3. Coordination overhead:
   SELECT FOR UPDATE SKIP LOCKED works but creates lock contention
   Does not work well across multiple DB instances

4. No built-in features:
   No dead-letter queue (manual), no retry with backoff (manual),
   no message priority, no fan-out, no ordering guarantees

5. Mixes concerns:
   Your production database is also your job queue
   Heavy queue processing competes with OLTP queries for connections, CPU
```

**The correct solution:** Use a purpose-built message queue (SQS, RabbitMQ, Kafka, Redis Streams):
```python
# Producer
sqs.send_message(QueueUrl=QUEUE_URL, MessageBody=json.dumps(payload))

# Consumer (long-polling, no wasted queries)
messages = sqs.receive_message(QueueUrl=QUEUE_URL, WaitTimeSeconds=20)
# Built-in: visibility timeout, DLQ, retry, FIFO, at-least-once delivery
```

**When it is acceptable:** Very low volume jobs (< 100/day) for a simple application where adding a dedicated queue adds operational complexity that outweighs the benefits.

---

### Q7. What is cascading failure and how does one service failure bring down an entire system?

A **cascading failure** occurs when a failure in one component propagates through dependencies, causing failures in other components, until the entire system is down — even though only one small part originally failed.

**How it happens:**
```
Normal operation:
  Client → Service A → Service B → Service C → DB

Service C gets slow (DB overloaded):
  Service B waits for C (30s timeout)
  Service B's thread pool fills up (all threads waiting for C)
  Service A waits for B (which is now slow too)
  Service A's thread pool fills up
  Clients wait for A
  API Gateway times out → all users see errors

  One slow DB → entire system failure via thread pool exhaustion
```

**The thread pool exhaustion mechanism:**
```
Service B has 100 threads in its connection pool
Each request to C takes 30s (slow, not failing):
  t=0:   Request 1 uses thread 1 (waiting for C)
  t=1:   Request 2 uses thread 2 (waiting for C)
  ...
  t=100: All 100 threads occupied waiting for C
  t=101: Request 101 arrives → no thread available → rejected immediately
  
  Service B looks completely failed from Service A's perspective,
  but it's only slow because of C.
```

**Prevention:**

| Technique            | How it prevents cascading failure                          |
|----------------------|------------------------------------------------------------|
| Circuit Breaker      | After N failures, stop calling the service; fail fast      |
| Timeouts             | Never wait forever — fail at 500ms, not 30s               |
| Bulkheads            | Separate thread pools per dependency (C slow ≠ B's other calls slow) |
| Rate limiting        | Shed load before it overwhelms downstream                  |
| Retry with backoff   | Don't hammer a recovering service with retries             |
| Fallback             | Return cached/default response when dependency fails       |

```python
# Circuit breaker with fallback
@circuit_breaker(failure_threshold=5, timeout=30)
def get_recommendations(user_id):
    try:
        return recommendation_service.get(user_id, timeout=0.5)
    except (Timeout, CircuitBreakerOpen):
        return get_popular_items_fallback()  # Graceful degradation
```

---

## Medium (Q8–Q15)

---

### Q8. What is the "chatty services" anti-pattern in microservices?

**Chatty services** make excessive fine-grained synchronous calls between microservices to assemble a response. Each call adds network latency and creates tight coupling. The result is high latency and a system where one slow service makes everything slow.

**The problem:**
```
User profile page request flow (chatty):
  API Gateway → Profile Service
    → User Service.getUser(42)        [1 call, 5ms]
    → Auth Service.getPermissions(42) [1 call, 8ms]
    → Order Service.getOrders(42)     [1 call, 12ms]
    → Loyalty Service.getPoints(42)   [1 call, 7ms]
    → Notification Service.getPrefs(42) [1 call, 6ms]
    → Product Service.getWishlist(42) [1 call, 9ms]
  
  Total: 6 sequential calls = 47ms minimum
  Plus: if any single service is slow → whole response is slow
  Plus: if any service is down → whole response fails
```

**Solution 1 — Parallel calls:**
```python
# Call all services in parallel (async)
import asyncio

async def get_profile(user_id):
    user, permissions, orders, points, prefs, wishlist = await asyncio.gather(
        user_service.get(user_id),
        auth_service.get_permissions(user_id),
        order_service.get_orders(user_id),
        loyalty_service.get_points(user_id),
        notification_service.get_prefs(user_id),
        product_service.get_wishlist(user_id),
    )
    return build_profile(user, permissions, orders, points, prefs, wishlist)
# Total latency: max(5,8,12,7,6,9) = 12ms instead of 47ms
```

**Solution 2 — BFF (Backend For Frontend):**
```
Create a Profile Service that aggregates the data:
  Profile Service owns one call → returns complete profile
  Caches assembled profile for 60 seconds
  Internally manages the fan-out (or uses a data denormalization approach)
```

**Solution 3 — Event-driven denormalization:**
```
Each service publishes events on change:
  User Service: UserUpdated event → Profile Service updates its own copy
  Order Service: OrderPlaced event → Profile Service updates order summary
  
  Profile Service query: single DB read (no cross-service calls at all)
  Trade-off: eventual consistency (profile may be seconds behind)
```

**Solution 4 — GraphQL with DataLoader:**
Client specifies exactly what fields it needs, DataLoader batches sub-queries, eliminating N+1 and reducing over-fetching.

---

### Q9. What is the shared database anti-pattern in microservices?

**Shared database** occurs when multiple microservices access the same database schema — reading and writing each other's tables directly. This is one of the most common ways teams accidentally create a distributed monolith.

**The anti-pattern:**
```
Service A (Orders)     ──────────────────────┐
Service B (Inventory)  ─────────────────────►│ Shared PostgreSQL DB
Service C (Shipping)   ─────────────────────►│ (same schema)
Service D (Analytics)  ──────────────────────┘

Each service can:
  - Read any other service's tables
  - Write to any other service's tables
  - Break any other service by changing a column
```

**Why it creates problems:**

1. **Schema coupling:** Service A wants to rename a column → must check if B, C, D use it. Any schema change requires coordination of all teams.

2. **No encapsulation:** Service B can read Order.internalCostBasis (should be private to Orders service). Business logic leaks across boundaries.

3. **Scaling coupling:** Orders table needs sharding → can't shard without breaking Inventory, Shipping, Analytics queries.

4. **Deployment coupling:** Adding a column requires migration → all services must be compatible with both old and new schema → coordinated deploy required.

**The correct pattern — database per service:**
```
Service A (Orders) ──────► Orders DB (owns orders, order_items, payments)
Service B (Inventory) ───► Inventory DB (owns products, stock_levels)
Service C (Shipping) ────► Shipping DB (owns shipments, addresses)

Cross-service data access:
  NOT: Service B queries Orders DB directly
  YES: Service A publishes OrderPlaced event → Service B subscribes, updates stock
  YES: Service A calls Service B's API (GET /inventory/{productId}) — encapsulated
```

**Migration path from shared DB:**
1. Identify ownership: which service "owns" each table?
2. Make other services go through the owning service's API (strangler fig)
3. Once all direct DB access removed, physically separate the schema
4. Run as separate DB instances or schemas, then separate DB servers

---

### Q10. What is the back-pressure problem and how do you design a system that can slow down producers?

**Back-pressure** is the mechanism by which a consumer signals to a producer that it is overwhelmed and the producer should slow down. Without back-pressure, producers can overwhelm consumers, causing unbounded queue growth and eventual OOM or data loss.

**The problem without back-pressure:**
```
Producer: generates 10,000 events/second
Consumer: processes 1,000 events/second

Queue grows: +9,000 events/second
After 1 minute: 540,000 events in queue
After 10 minutes: 5,400,000 events in queue

Memory exhaustion → OOM crash → all queued events LOST
Or: queue fills → producer gets an error → producer also crashes
```

**Back-pressure solutions:**

**Solution 1 — Bounded queues with blocking:**
```java
// Java BlockingQueue: blocks producer when queue is full
BlockingQueue<Event> queue = new LinkedBlockingQueue<>(1000); // max 1000 items

// Producer blocks until queue has space
queue.put(event);  // Blocks if queue is full (back-pressure applied to producer)

// Consumer processes events
queue.take();      // Blocks if queue is empty
```

**Solution 2 — Reactive Streams (Project Reactor, RxJava):**
```java
// Flux with built-in back-pressure
Flux.range(1, 1_000_000)
    .onBackpressureBuffer(100)       // Buffer up to 100, then...
    .onBackpressureDrop()            // ...drop new items
    .subscribe(event -> process(event)); // Consumer controls demand
```

**Solution 3 — Kafka as a back-pressure buffer:**
```
Producer → Kafka topic (durable, large buffer)
Consumer → reads at its own pace (Kafka retains messages until consumed)

Kafka back-pressure: consumer simply reads more slowly
  - Consumer lag increases (monitor this metric)
  - Producer continues writing (Kafka absorbs the spike)
  - No data loss (Kafka is durable)
  - Alert when consumer lag > threshold → add more consumer instances
```

**Solution 4 — Rate limiting the producer at the source:**
```python
# Client-side rate limiter (Token bucket)
class RateLimiter:
    def __init__(self, rate):
        self.rate = rate        # tokens per second
        self.tokens = rate
        self.last_check = time.time()

    def allow(self):
        now = time.time()
        self.tokens += (now - self.last_check) * self.rate
        self.last_check = now
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False  # Back-pressure: tell producer to slow down
```

**Design principle:** Every queue in your system should be bounded. An unbounded queue is a future outage waiting to happen.

---

### Q11. What is the problem with ignoring idempotency and how do you design for it?

**Idempotency** means executing the same operation multiple times produces the same result as executing it once. Ignoring idempotency causes **duplicate processing** — charging users twice, sending duplicate emails, double-counting inventory.

**Why duplicates happen:**
```
Client sends POST /charge {amount: $100}
Server processes charge → responds 200
Network fails before client receives response

Client retries: POST /charge {amount: $100}
Server processes again → charges user $200 total!

This happens constantly in distributed systems:
- Client timeouts and retries
- Message queue at-least-once delivery (Kafka, SQS)
- Network partitions causing uncertain outcomes
- Dead letter queue reprocessing
```

**Idempotency key pattern:**
```python
# Client generates a unique ID for each logical operation
import uuid

def charge_user(user_id, amount):
    idempotency_key = str(uuid.uuid4())  # Generated once, reused on retry
    
    response = requests.post('/charge', json={
        "user_id": user_id,
        "amount": amount,
        "idempotency_key": idempotency_key
    })
    
    if response.status_code == 504:  # Timeout
        # Retry with SAME idempotency_key
        response = requests.post('/charge', json={
            "user_id": user_id,
            "amount": amount,
            "idempotency_key": idempotency_key  # Same key
        })
```

```python
# Server deduplication logic
def handle_charge(user_id, amount, idempotency_key):
    # Check if we've seen this key before (Redis with TTL)
    cached = redis.get(f"idempotent:{idempotency_key}")
    if cached:
        return json.loads(cached)  # Return same response as first execution
    
    # First time: process the charge
    charge = payment_provider.charge(user_id, amount)
    result = {"charge_id": charge.id, "status": "success"}
    
    # Cache result for 24 hours
    redis.setex(f"idempotent:{idempotency_key}", 86400, json.dumps(result))
    return result
```

**Database-level idempotency:**
```sql
-- Idempotent INSERT using ON CONFLICT DO NOTHING
INSERT INTO user_events (user_id, event_type, event_id, created_at)
VALUES (42, 'purchase', 'evt_xyz123', NOW())
ON CONFLICT (event_id) DO NOTHING;  -- event_id is unique, duplicate ignored

-- Idempotent order creation
INSERT INTO orders (order_id, user_id, total)
VALUES ('ord_abc', 42, 100.00)
ON CONFLICT (order_id) DO UPDATE SET updated_at = NOW();
-- Same order_id = no-op (idempotent)
```

Every message consumer (queue worker, event handler, webhook receiver) should be designed idempotent. The question is not "will we get duplicates?" — we will. The question is "are we ready to handle them safely?"

---

### Q12. What is the CAP theorem and what is the mistake of assuming you can have all three?

The **CAP theorem** (Eric Brewer, 2000) states that a distributed system can only guarantee two of the following three properties simultaneously during a network partition:

- **C (Consistency):** Every read receives the most recent write or an error
- **A (Availability):** Every request receives a response (success or failure, not timeout)
- **P (Partition Tolerance):** The system continues operating despite network partitions

```
CAP Triangle:
         Consistency
              /\
             /  \
            /    \
           / CP   \
          /        \
         /          \
        /____________\
  CA                  AP
  
  CA (Consistency + Availability, no Partition Tolerance):
    Single-node relational DB, no distribution → not viable for distributed systems
    
  CP (Consistency + Partition Tolerance, sacrifices Availability):
    HBase, ZooKeeper, etcd → prefers to reject requests than return stale data
    Banks: "I'd rather say 'service unavailable' than show wrong balance"
    
  AP (Availability + Partition Tolerance, sacrifices Consistency):
    Cassandra, DynamoDB, CouchDB → always responds, may return stale data
    Shopping carts: "Showing stale cart is better than cart unavailable"
```

**Why "all three" is impossible:**
```
Network partition occurs: Node A and Node B cannot communicate

Scenario: Client writes to Node A, then reads from Node B
  
  If you choose Consistency: 
    Node B must say "I don't know" (reject the read) → sacrifices Availability
    
  If you choose Availability:
    Node B returns its last known value (potentially stale) → sacrifices Consistency
    
  There is no third option that satisfies both simultaneously during the partition.
```

**The real trade-off (PACELC):** Even without partitions, there is a latency-consistency trade-off. PACELC extends CAP: "If Partition (P): A vs C; Else (E): Latency (L) vs Consistency (C)."

**Common mistake:** Designing a system that assumes strong consistency globally (all nodes always agree instantly). In reality, network partitions happen regularly in distributed systems. Design explicitly for your consistency requirements per operation type: financial transactions need CP; shopping carts can be AP.

---

### Q13. What is the problem with monitoring activity instead of symptoms?

This anti-pattern describes monitoring system internals (CPU, disk I/O, query counts) instead of monitoring the user-facing experience (error rates, latency, availability). Activity monitoring generates false positives and misses real problems.

**Activity monitoring (anti-pattern):**
```
Alerts configured:
  CPU > 80%  → PagerDuty alert
  Disk I/O > 1000 IOPS → PagerDuty alert
  DB connections > 80% of pool → PagerDuty alert
  GC pauses > 500ms → PagerDuty alert

Problem:
  CPU at 85%? Users might be completely fine — maybe running a scheduled job.
  Alert fires at 3 AM → engineer wakes up → users unaffected → alert noise.
  
  Over time: engineers start ignoring alerts (alert fatigue)
  Real outage happens: buried in noise, missed for 30 minutes.
```

**Symptom-based monitoring (correct approach — Google's 4 Golden Signals):**
```
1. Latency:    p99 request latency > 500ms → alert
2. Traffic:    Sudden 50% drop in request rate → alert (not a spike)
3. Errors:     HTTP error rate > 1% → alert
4. Saturation: Which resource is approaching limit AND causing degradation?

These directly measure user experience.
CPU at 85% with latency < 100ms and error rate 0%: NOT a problem, don't wake anyone up.
CPU at 30% with error rate 5%: DEFINITE problem, alert immediately.
```

**The USE method for resource metrics (not alerts):**
```
Resource metrics (CPU, memory, disk) go on dashboards for investigation.
They are clues, not symptoms.

Workflow:
  Alert fires (symptom: p99 > 500ms)
  Engineer investigates dashboard
  Finds: CPU 90% → identifies this as the likely cause
  Takes action: scale out, find CPU-hungry query
```

**SLO-based alerting:**
```
Better: Alert when error budget burn rate is too high

Error budget: 0.1% of requests can fail (99.9% SLO)
Monthly error budget: 0.001 × 30 × 24 × 60 × 60 = 2592 seconds

Alert when burn rate > 14.4× (will exhaust monthly budget in 2 hours)
  → Page on-call immediately

Alert when burn rate > 3× (will exhaust monthly budget in 10 days)
  → Create ticket, fix during business hours
```

---

### Q14. What is the problem with big-bang deploys and how do feature flags solve it?

**Big-bang deploys** release all new features to all users at once with no ability to roll back quickly. Any bug affects 100% of users immediately.

**The problem:**
```
Scenario: 3 new features developed over 6 weeks, deployed together

Deploy day:
  Feature 1: works fine
  Feature 2: works fine
  Feature 3: has a subtle bug causing 5% of users to experience checkout failures
  
  Impact: 5% × all users = significant revenue loss
  
  Options:
    A. Rollback: reverts Features 1 and 2 (which worked!) AND Feature 3
       → customers lose good features, PR pain
    B. Hotfix: need to find the bug in new code under production load
       → takes 30 minutes, damage done
```

**Feature flags (feature toggles):**
```python
# Configuration in LaunchDarkly / Unleash / custom Redis flag store
# Flag: "new_checkout_flow" = true/false per user segment

def handle_checkout(user, cart):
    if feature_flag.is_enabled("new_checkout_flow", user_id=user.id):
        return new_checkout_v2(user, cart)   # New code path
    else:
        return old_checkout_v1(user, cart)   # Old code path
```

**Progressive rollout with feature flags:**
```
Day 1:  Enable for 1% of users (catch obvious bugs)
Day 2:  Enable for 10% of users (monitor error rates)
Day 3:  Enable for 50% of users (confirm stability)
Day 4:  Enable for 100% of users

If bug found at any stage: disable flag → 0% users affected in seconds
No deploy required, no rollback of unrelated features.
```

**Feature flags enable:**
- **Dark launches:** deploy code that no one can see yet (feature disabled)
- **A/B testing:** route percentage to test variant, measure conversion
- **Kill switches:** disable a feature without a redeploy (critical for incidents)
- **Gradual rollouts:** canary for specific features independently of deployment

**Important:** Feature flags are technical debt. Flag cleanup is as important as flag creation. Old flags with dead code paths confuse engineers and hide complexity. Set a maximum lifetime for flags (e.g., 90 days) and enforce removal.

---

### Q15. What is the microservices tax and when should you NOT use microservices?

The **microservices tax** refers to the significant operational, development, and infrastructure overhead that microservices introduce. For small teams or simple applications, this tax exceeds the benefits.

**The real costs of microservices:**

| Cost Category           | Description                                                     |
|-------------------------|-----------------------------------------------------------------|
| Network overhead        | Every service call is a network hop (latency, failure modes)    |
| Operational complexity  | 20 services × CI/CD pipelines, Kubernetes, monitoring, tracing |
| Distributed tracing     | Requires Jaeger/Zipkin; debugging across services is hard       |
| Data consistency        | No transactions across services; sagas required                 |
| Service discovery       | Need service mesh or DNS-based discovery                        |
| Testing complexity      | Integration tests for 20 services are expensive to maintain     |
| Developer velocity      | Context-switching between repositories, deployment friction     |
| On-call burden          | Each service can fail independently; more alerts                |

**The microservices hype problem:**
```
Team size: 5 engineers
Chose microservices because: "Netflix does it"

Reality:
  Netflix has 700+ engineers and built microservices tooling for years.
  5 engineers spend 60% of time on infrastructure instead of features.
  Simple CRUD app has 8 microservices, a service mesh, 3 message queues.
  New engineer takes 3 months to understand the system.
```

**When microservices make sense:**
- Large teams (20+ engineers) that can own independent services
- Different scaling requirements per component (read vs write, ML vs API)
- Independent release cadences needed (payment team vs frontend team)
- Technology heterogeneity (one service in Go, another in Python)
- You have already built a well-structured monolith and need to scale specific parts

**When to use a monolith (or modular monolith):**
- < 20 engineers
- Early-stage product (requirements change rapidly)
- Tight team that deploys together
- Bounded contexts not yet clearly understood
- CRUD application with simple domain

**The right progression:**
```
Stage 1 (0-5 engineers): Monolith — fastest time to market
Stage 2 (5-15 engineers): Modular monolith — clean internal boundaries
Stage 3 (15+ engineers): Selective extraction — split only what causes bottlenecks
Stage 4 (50+ engineers): Full microservices — justified by team autonomy needs
```

---

## Hard (Q16–Q20)

---

### Q16. What is the over-reliance on distributed transactions (2PC in microservices) and what is the alternative?

**Two-Phase Commit (2PC)** is a distributed transaction protocol that achieves atomicity across multiple nodes. While it works, using it in microservices is an anti-pattern that creates tight coupling and availability problems.

**2PC mechanics:**
```
Phase 1 (Prepare):
  Coordinator → Service A: "Prepare to commit"
  Coordinator → Service B: "Prepare to commit"
  Coordinator → Service C: "Prepare to commit"
  All respond: "Ready" or "Abort"

Phase 2 (Commit):
  If all ready: Coordinator → A, B, C: "Commit"
  If any abort: Coordinator → A, B, C: "Rollback"
```

**Why 2PC is problematic in microservices:**

1. **Blocking protocol:** If the coordinator crashes after Phase 1, all participants are blocked (holding locks) indefinitely until coordinator recovers.

2. **Tight coupling:** Services must expose a transaction protocol — they cannot evolve independently.

3. **Availability:** 2PC availability = product of all participant availabilities. With 5 services at 99.9%: 0.999^5 = 99.5% — worse than each individual service.

4. **Performance:** Network round trips for Phase 1 + Phase 2 add latency to every transaction.

**The alternative — Saga Pattern:**
```
Choreography-based Saga (event-driven):

  Order Service: creates order, publishes OrderCreated event
       ↓
  Inventory Service: reserves items, publishes InventoryReserved event
       ↓
  Payment Service: processes payment, publishes PaymentProcessed event
       ↓
  Order Service: updates order to "confirmed"

On failure (payment fails):
  Payment Service: publishes PaymentFailed event
       ↑
  Inventory Service: listens for failure, releases reservation (compensating txn)
       ↑
  Order Service: listens for failure, marks order "cancelled" (compensating txn)
```

**Orchestration-based Saga (Step Functions / Conductor):**
```python
# AWS Step Functions orchestrates the saga
{
  "StartAt": "ReserveInventory",
  "States": {
    "ReserveInventory": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:inventory:reserve",
      "Next": "ProcessPayment",
      "Catch": [{"ErrorEquals": ["States.ALL"], "Next": "CancelOrder"}]
    },
    "ProcessPayment": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:payment:charge",
      "Next": "ConfirmOrder",
      "Catch": [{"ErrorEquals": ["States.ALL"], "Next": "ReleaseInventory"}]
    },
    ...
  }
}
```

The key difference: sagas use **compensating transactions** (undo actions) rather than rollback. They achieve eventual consistency rather than strict ACID atomicity.

---

### Q17. What is the "designing for average load" anti-pattern and why should you design for P99?

**Designing for average load** means sizing infrastructure and setting timeouts/retries based on typical (mean or p50) behavior. This leads to systems that work most of the time but fail spectacularly during spikes, GC pauses, cold caches, or end-of-month reports.

**The problem with averages:**
```
API latency distribution:
  p50: 20ms   (half of requests)
  p95: 80ms   (19 out of 20 are fine)
  p99: 500ms  (99 out of 100 are fine)
  p99.9: 3000ms (1 in 1000 takes 3 seconds)

If you design for p50 (20ms):
  - Set timeout at 50ms (2.5× average = "generous")
  - 5% of requests (p95: 80ms > 50ms) time out!
  - With 1000 RPS: 50 request failures per second
  - Users see errors constantly
```

**Tail latency amplification (the fan-out problem):**
```
Request requires 10 parallel sub-calls:
  Each sub-call: p99 = 500ms (1% chance of being slow)
  
  Probability that at least 1 of 10 is slow:
  P(at least one slow) = 1 - P(all fast)^10
  = 1 - 0.99^10 = 1 - 0.904 = 9.6%
  
  Nearly 10% of requests are slow — even though each service is only 1% slow!
  With 100 sub-calls: 1 - 0.99^100 = 63% of requests are slow.
  
  This is why Google/Amazon target p99.9 or p99.99 for individual services.
```

**Hedged requests (Google's tail latency mitigation):**
```python
# After 95th percentile latency, send a second request to another replica
# Return whichever responds first

async def hedged_request(service, request):
    # Send primary request
    primary = asyncio.create_task(service.call(request))
    
    # After p95 timeout, send hedge request
    try:
        await asyncio.wait_for(asyncio.shield(primary), timeout=0.08)  # 80ms = p95
        return primary.result()
    except asyncio.TimeoutError:
        # p95 exceeded — hedge
        hedge = asyncio.create_task(service.call(request))
        done, pending = await asyncio.wait([primary, hedge], return_when=asyncio.FIRST_COMPLETED)
        
        for task in pending:
            task.cancel()
        return done.pop().result()

# Cost: ~5% extra requests (only hedged when slow)
# Benefit: p99 drops from 500ms to ~90ms
```

**Load testing at peak, not average:**
```
Traffic profile:
  Average: 100 RPS (daily average)
  Business hours peak: 400 RPS
  Flash sale: 2000 RPS (20× average)
  
Design for: 2000 RPS with graceful degradation above that
Set alerts at: 80% of 2000 = 1600 RPS (time to scale before hitting limit)
```

---

### Q18. How does the "not designing for idempotency" anti-pattern cause real-world incidents?

While Q11 covers the basics, this question explores the deeper systemic failures caused by non-idempotent systems under operational conditions.

**Real-world failure modes:**

**Scenario 1 — Message queue redelivery:**
```
Order service publishes: OrderPlaced event to Kafka
Inventory service consumer: reads event, deducts stock, commits offset
  ← Consumer crashes between deducting stock and committing offset
  
On restart: Kafka redelivers the event (offset not committed)
Inventory service: deducts stock AGAIN → inventory goes negative
  
Root cause: inventory deduction not idempotent
Fix: track processed event IDs (Kafka partition + offset) in DB
     ON CONFLICT (event_id) DO NOTHING
```

**Scenario 2 — Network timeout retry storm:**
```
During a deployment, half the payment service pods restart simultaneously.
100 clients have in-flight payment requests.
All 100 time out after 30 seconds.
All 100 clients retry automatically.
95 original requests completed before restart (payment taken).
95 retries re-charge users.
5 retries that genuinely failed also retry → some charged correctly.

Result: 95 double-charges discovered the next morning.
Customer service nightmare + chargebacks.
  
Fix: idempotency keys, required for every payment endpoint.
     Validate: "Stripe requires idempotency keys for all POST requests" — it's that important.
```

**Scenario 3 — Deployment re-runs:**
```
CI/CD pipeline: step 3 (run DB migrations) → timeout → step 3 retried
Migration: ALTER TABLE users ADD COLUMN last_login timestamp;
First run: column added successfully
Second run (retry): "column already exists" error → migration fails → deploy fails

Fix: IF NOT EXISTS in migrations
  ALTER TABLE users ADD COLUMN IF NOT EXISTS last_login timestamp;
  — Idempotent migration, safe to retry
```

**Design checklist for idempotency:**
```
For every write endpoint (API, queue consumer, webhook):
  [ ] Does executing this operation twice cause harm?
  [ ] If yes, implement idempotency key deduplication
  [ ] Store result in cache with idempotency_key for 24 hours
  [ ] Return same response for duplicate requests
  
For DB migrations:
  [ ] Use IF NOT EXISTS, IF EXISTS for all DDL
  [ ] Make data migrations idempotent (UPDATE WHERE old_value IS NULL)

For queue consumers:
  [ ] Track processed message IDs in DB (event sourcing, dedup table)
  [ ] OR: design consumer logic to be naturally idempotent
      (INSERT ... ON CONFLICT DO NOTHING, UPDATE ... SET ... WHERE current_value = expected)
```

---

### Q19. What is the synchronous-everything anti-pattern and how does async improve resilience?

**Synchronous-everything** is building a system where every operation between services is a synchronous blocking call. The caller blocks waiting for the response. Every service in the chain must be up and fast for the user to get a response.

**The failure chain:**
```
Synchronous system (all blocking HTTP calls):

Client → API → Order Svc → Inventory Svc → Email Svc → [response]
                                            ↑ 
                              Email service goes down (SMTP issue)
                              
  Email Svc down → Order Svc blocks waiting → timeout after 30s
  Order Svc blocks → API Gateway blocks → Client waits 30s → timeout
  
  Result: Email service down → entire checkout broken.
  These are completely unrelated concerns!
```

**The async alternative:**
```
Async system (event-driven non-critical paths):

Client → API → Order Svc:
  1. Validate order (synchronous: business logic required immediately)
  2. Write order to DB (synchronous: required for consistency)
  3. Publish OrderPlaced to queue (fire and forget: ~2ms)
  4. Return 201 Created to client (fast!)
  
Background workers (independently):
  Email Worker: consumes OrderPlaced → sends confirmation email
  Inventory Worker: consumes OrderPlaced → deducts stock  
  Analytics Worker: consumes OrderPlaced → updates metrics
  
  Email Svc down → email queue builds up → emails sent when service recovers
  Order Svc: completely unaffected → users still checkout successfully
```

**Identifying what should be synchronous vs async:**
```
SYNCHRONOUS (user must wait):
  - Business validation (is inventory available? Is payment valid?)
  - Primary data write (creating the order record)
  - Auth/authorization
  - Any operation where the response determines next user action

ASYNCHRONOUS (user doesn't need to wait):
  - Notifications (email, SMS, push)
  - Audit logging
  - Analytics event recording
  - Search index updates
  - Webhook deliveries to third parties
  - Generating reports / PDFs
  - Image processing
  - Recommendation model updates
```

**Resilience improvements from async:**
- Service A failure cannot directly cause Service B failure
- Spikes are absorbed by queues (natural back-pressure)
- Operations can be retried without user re-action
- Services can be deployed independently without coupling release timing
- Each service's SLA is independent

---

### Q20. What is the correct framework for making system design trade-offs in an interview?

This meta-question tests whether candidates understand that system design is fundamentally about **deliberate trade-offs** rather than finding a single "correct" answer.

**The FAST Framework for trade-off analysis:**
```
F — Frame the requirements (functional + non-functional)
A — Articulate the constraints (scale, latency, cost, team size)
S — State the trade-offs explicitly (not just pick a solution)
T — Tailor to the specific problem (generic answers = no credit)
```

**Step 1 — Frame requirements precisely:**
```
BAD: "I'll design a scalable system"

GOOD: "Given the requirements:
  - 10M users, 1M DAU
  - 99.99% availability
  - p99 < 200ms API latency
  - $500K/year infra budget
  - 5-engineer platform team
  
  I'll make the following trade-offs..."
```

**Step 2 — Identify the key trade-off axes:**
```
Consistency vs Availability:
  "For the shopping cart, I'll choose availability (AP) because
   showing a slightly stale cart is better than cart unavailable."

Cost vs Complexity:
  "Multi-region active-active gives the best availability but costs 3×.
   Given 5 engineers, I'd start with multi-AZ and add multi-region
   when we have both the traffic and the team to justify it."

Build vs Buy:
  "Self-managed Kafka gives us full control and saves $30K/year,
   but MSK eliminates broker management. For a 5-person team,
   MSK is the right choice — the engineering time is more valuable."
```

**Step 3 — Anti-patterns to avoid in interviews:**

| Mistake                             | Correct Approach                              |
|-------------------------------------|-----------------------------------------------|
| Jumping to microservices immediately | Start monolith, justify split with scale      |
| Adding technology without justification | Justify each component ("we need Kafka because...") |
| Not acknowledging trade-offs         | Every choice has a downside — name it         |
| Over-engineering for day-1 MVP       | Design for current scale, plan for 10×        |
| Under-engineering (SPOF everywhere)  | Identify critical paths and make them HA      |
| Designing for average, not peak      | Ask about peak traffic patterns               |

**Step 4 — The answer template:**
```
"For [specific requirement], I would use [approach A] rather than [approach B] because:
  - Benefit: [specific advantage]
  - Trade-off: [specific cost or limitation]
  - Mitigation: [how I'd address the trade-off]
  - At this scale/stage, this trade-off is acceptable because [reason]"

Example:
"For session storage, I'd use Redis (external session store) rather than 
in-process session replication because:
- Benefit: app servers are fully stateless, enabling unlimited horizontal scale
- Trade-off: Redis is now a critical dependency; if Redis fails, all sessions fail
- Mitigation: Redis Sentinel/Cluster for HA; graceful fallback to re-auth
- At our scale (1M sessions), sticky sessions would make auto-scaling painful —
  the Redis dependency is worth accepting."
```

**Scoring what interviewers actually look for:**
```
Top signals:
  ✓ "It depends" — understanding that context changes the answer
  ✓ Naming trade-offs before being asked
  ✓ Asking clarifying questions (scale? consistency requirements? budget?)
  ✓ Using numbers (latency targets, data volumes, team size)
  ✓ Identifying the SPOF or bottleneck in your own design unprompted
  ✓ Knowing when NOT to use microservices, distributed transactions, caching

Red flags:
  ✗ "We'd use Kubernetes, microservices, Kafka, and distributed caching" (for a to-do app)
  ✗ Adding components without justification
  ✗ Not acknowledging failure modes
  ✗ Designing for average load only
  ✗ "There's only one right answer here" — there almost never is
```

---

## Quick Reference

### Anti-Pattern Checklist (what to watch for in design)

| Anti-Pattern               | Signal                                      | Fix                                          |
|----------------------------|---------------------------------------------|----------------------------------------------|
| Distributed Monolith       | Can't deploy independently                  | Database-per-service, async events           |
| Big Ball of Mud            | Fear of changing any code                   | Bounded contexts, modular design             |
| Premature Optimization     | Optimizing before measuring                 | Measure first, then target the bottleneck    |
| Single Point of Failure    | Any unredundant critical component          | Redundancy + automated failover everywhere   |
| Over-engineering           | Infrastructure > feature work for MVPs      | YAGNI, start simple, scale when needed       |
| DB as Message Queue        | Polling jobs table                          | Use SQS, Kafka, RabbitMQ                     |
| Cascading Failure          | One service down = everything down          | Circuit breakers, timeouts, bulkheads        |
| Chatty Services            | Many sync calls per request                 | Batch/parallel calls, BFF pattern, events    |
| Shared Database            | Multiple services, same schema              | Database per service + APIs + events         |
| No Back-pressure           | Queues growing unboundedly                  | Bounded queues, flow control                 |
| Non-idempotent ops         | Duplicate charges, double-sends on retry    | Idempotency keys, ON CONFLICT DO NOTHING     |
| CAP misconception          | "We need ACID + always-on + distributed"    | Choose CP or AP explicitly per use case      |
| Activity monitoring        | Alert on CPU but not error rate             | 4 Golden Signals: latency, traffic, errors, saturation |
| Big-bang deploys           | Full rollout with no kill switch            | Feature flags, canary deploys                |
| Microservices for CRUD     | 5 engineers, 15 microservices               | Monolith or modular monolith until justified |
| 2PC in microservices       | Cross-service distributed transactions      | Saga pattern (choreography or orchestration) |
| Average load design        | Timeouts set at p50                         | Design for p99, hedge requests               |
| Synchronous everything     | Email down = checkout down                  | Async all non-critical paths                 |

### Trade-off Axes for Interviews
- **Consistency vs Availability** (CAP: pick CP or AP)
- **Latency vs Throughput** (batching: more throughput, less latency)
- **Cost vs Complexity** (managed service costs money; self-managed costs engineering)
- **Simplicity vs Scale** (monolith is simple; microservices scale)
- **Build vs Buy** (build for differentiation; buy for commodities)

### The YAGNI + KISS Principle
- **YAGNI:** You Aren't Gonna Need It — don't build what isn't needed yet
- **KISS:** Keep It Simple, Stupid — the simplest solution that works is usually the best

### Key Questions to Ask in Any Design
1. What are the scale requirements? (users, RPS, data volume)
2. What is the consistency requirement? (strong vs eventual)
3. What is the availability SLO? (99.9% vs 99.99%)
4. What is the team size? (affects build vs buy, monolith vs microservices)
5. What is the budget constraint? (multi-cloud vs single-provider)
6. What is the latency target? (p50? p99? real-time vs batch?)
