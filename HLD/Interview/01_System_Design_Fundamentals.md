# System Design Fundamentals — Interview Q&A

> 20 questions | Easy: Q1–Q7 | Medium: Q8–Q15 | Hard: Q16–Q20

---

## EASY (Q1–Q7)

---

### Q1. How do you approach a system design interview?

**Answer:**

A system design interview tests your ability to translate ambiguous requirements into a concrete, scalable architecture. The best approach follows a structured framework:

**Step-by-step framework:**

```
1. CLARIFY (3-5 min)
   - Ask about scale: DAU, QPS, data size
   - Ask about features: must-have vs nice-to-have
   - Ask about constraints: latency SLA, consistency needs

2. ESTIMATE (2-3 min)
   - Back-of-envelope: storage, bandwidth, QPS
   - Identify bottlenecks before designing

3. HIGH-LEVEL DESIGN (10-15 min)
   - Draw major components: client, LB, API servers, DB, cache
   - Define data flow end-to-end

4. DEEP DIVE (10-15 min)
   - Pick the hardest parts: DB schema, caching strategy, scaling
   - Address bottlenecks identified earlier

5. TRADE-OFFS & WRAP-UP (3-5 min)
   - Discuss what you would do differently
   - Mention monitoring, failure handling, future extensions
```

**Key mindset tips:**
- Think aloud — interviewers evaluate reasoning, not just the final answer.
- Drive the conversation; do not wait to be led.
- Every design choice should be backed by a "why."
- Acknowledge trade-offs rather than presenting a single "correct" answer.
- Start simple, then layer complexity.

**Common mistakes to avoid:**
| Mistake | Better Approach |
|---|---|
| Jumping straight to technology | Gather requirements first |
| Over-engineering from the start | Start with the simplest correct solution |
| Ignoring failures | Always discuss what happens when a component fails |
| Forgetting non-functional requirements | Latency, availability, and consistency matter |

---

### Q2. What is the difference between functional and non-functional requirements?

**Answer:**

**Functional requirements** define *what* the system does — the specific behaviors and features it must support.

**Non-functional requirements (NFRs)** define *how well* the system performs those behaviors — quality attributes.

**Comparison table:**

| Dimension | Functional | Non-Functional |
|---|---|---|
| Definition | Features / behaviors | Quality attributes |
| Examples | User can post a tweet | 99.99% uptime |
| Testable by | Unit / integration tests | Load tests, SLA audits |
| Changes impact | Business logic | Architecture choices |

**Functional examples (Twitter-like system):**
- Users can post tweets (up to 280 characters)
- Users can follow/unfollow other users
- Users can see a home timeline feed
- Search tweets by hashtag

**Non-functional examples:**
- **Availability:** 99.99% uptime (< 52 min downtime/year)
- **Latency:** Timeline load < 200ms at p99
- **Throughput:** Handle 500K tweets/day, 150M reads/day
- **Durability:** No data loss (tweets persisted permanently)
- **Scalability:** Handle 10× traffic growth without redesign
- **Security:** Auth tokens expire in 24 hours, data encrypted at rest

**Why NFRs drive architecture:**
NFRs often have more architectural impact than functional requirements. A system needing 99.999% availability must eliminate single points of failure, use multi-region deployments, and implement circuit breakers — none of which are visible features.

---

### Q3. What is the formula to estimate QPS from DAU?

**Answer:**

Back-of-envelope estimation converts Daily Active Users (DAU) into Queries Per Second (QPS) — a core system design skill.

**Base formula:**

```
QPS = (DAU × requests_per_user_per_day) / 86,400 seconds

Peak QPS ≈ Average QPS × 2 to 3 (accounts for traffic spikes)
```

**Step-by-step example — Instagram-like system:**

```
Given:
  DAU = 500 million
  Average user actions per day = 20 (views, likes, uploads)

Average QPS = (500,000,000 × 20) / 86,400
            = 10,000,000,000 / 86,400
            ≈ 115,740 QPS

Peak QPS    ≈ 115,740 × 3
            ≈ ~350,000 QPS
```

**Read/write split (important for DB sizing):**

```
Assume 95% reads, 5% writes:
  Read QPS  = 350,000 × 0.95 = 332,500
  Write QPS = 350,000 × 0.05 =  17,500
```

**Storage estimation:**

```
Photos per day = 500M users × 1% upload rate = 5M photos/day
Avg photo size = 500 KB (after compression)
Daily storage  = 5,000,000 × 500 KB = 2.5 TB/day
5-year storage = 2.5 TB × 365 × 5  = ~4.5 PB
```

**Useful constants to memorize:**
| Unit | Value |
|---|---|
| Seconds/day | 86,400 |
| MB | 10^6 bytes |
| GB | 10^9 bytes |
| TB | 10^12 bytes |
| Million | 10^6 |
| Billion | 10^9 |

---

### Q4. What is the difference between latency and throughput?

**Answer:**

These are two of the most fundamental performance metrics in system design and are often confused.

**Latency** is the time taken for a single request to travel from the sender to the receiver and get a response. It measures *speed* of an individual operation.

**Throughput** is the number of operations the system can handle per unit of time. It measures *capacity* — how much work the system can do.

```
Latency:    [Request]---200ms--->[Response]
Throughput: 10,000 requests processed per second
```

**Analogy:** A highway
- Latency = time for one car to drive from city A to city B
- Throughput = number of cars crossing per hour

**Key relationship:**

```
Throughput = 1 / Latency   (only when there is ONE request in flight)

With concurrency (N parallel requests):
Throughput ≈ N / Latency
```

**Practical comparison:**

| Scenario | Latency | Throughput |
|---|---|---|
| Database query | 5ms | 10,000 QPS |
| File download | 2 seconds | 500 MB/s |
| Batch job | Not relevant | 1M records/hour |

**Trade-off:** Optimizing for one can hurt the other.
- Batching requests → increases latency (wait for batch to fill) but improves throughput.
- Aggressive caching → reduces latency but throughput is bounded by cache capacity.

**Latency percentiles (p50, p99, p999):**
Always measure latency at percentiles, not averages. A p99 of 500ms means 1% of users wait longer than 500ms. Averages hide tail latency which affects user experience.

---

### Q5. What is the difference between horizontal and vertical scaling?

**Answer:**

**Vertical scaling (Scale Up):** Adding more resources (CPU, RAM, disk) to an existing machine.

**Horizontal scaling (Scale Out):** Adding more machines to distribute load.

```
VERTICAL SCALING
  Before:  [Server: 4 CPU, 16 GB RAM]
  After:   [Server: 32 CPU, 256 GB RAM]

HORIZONTAL SCALING
  Before:  [Server 1]
  After:   [Server 1] [Server 2] [Server 3] [Server 4]
            <---- Load Balancer distributes traffic ---->
```

**Comparison table:**

| Dimension | Vertical | Horizontal |
|---|---|---|
| Cost curve | Exponential (large servers cost disproportionately more) | Linear |
| Downtime | Requires restart | No downtime (add nodes while running) |
| Upper limit | Hard hardware ceiling | Virtually unlimited |
| Complexity | Simple (no distribution logic) | Complex (need LB, distributed state) |
| Fault tolerance | Single point of failure | Redundant nodes |
| Best for | Databases (initially), low-traffic services | Stateless services, web tiers |

**When to use each:**

- **Vertical:** Early-stage products, relational databases (easier than horizontal sharding), services that are hard to distribute (e.g., legacy monoliths).
- **Horizontal:** Web/API servers, stateless microservices, any workload that can be parallelized.

**Practical rule:** Start vertical (simpler), migrate horizontal when you hit cost or hardware limits. Stateless services scale horizontally with almost no code changes. Stateful services (databases) require sharding/partitioning strategies.

---

### Q6. What is the difference between stateless and stateful services?

**Answer:**

**Stateless service:** Each request is self-contained. The server does not retain any session information between requests. All context needed to process the request is included in the request itself (headers, tokens, body).

**Stateful service:** The server maintains session state across requests for the same client. The server must remember who talked to it previously.

```
STATELESS
  Client sends: [Request + JWT token + all context]
  Server:       Processes request, returns response, forgets client
  Any server can handle any request

STATEFUL
  Client sends: [Request + session ID]
  Server:       Looks up session store → retrieves context → processes
  Client must reach the SAME server (or shared session store)
```

**Scaling comparison:**

| Dimension | Stateless | Stateful |
|---|---|---|
| Horizontal scaling | Trivial (any node handles any request) | Hard (sticky sessions or shared state) |
| Fault tolerance | Any node can fail, others take over | Node failure loses in-memory sessions |
| Load balancing | Round-robin works perfectly | Requires sticky sessions |
| Deployment | Rolling deploys are safe | Must drain sessions before restart |

**Architectural rule:** Push state to the edges (client-side tokens, external databases, distributed caches) and keep your application servers stateless. This is the foundation of cloud-native design.

**Example — JWT vs Sessions:**
- JWT (stateless): Token carries all claims, server validates signature only.
- Sessions (stateful): Server stores session data, client holds only a session ID.

---

### Q7. What is the difference between SLA, SLO, and SLI?

**Answer:**

These three terms form the reliability contract framework used by SRE teams and system designers.

**SLI (Service Level Indicator):** A quantitative *measurement* of service behavior. It is the raw metric you collect.

**SLO (Service Level Objective):** An *internal target* for an SLI. It is the goal your team commits to achieving.

**SLA (Service Level Agreement):** An *external contract* with customers that includes SLOs and the consequences (credits, penalties) of missing them.

```
SLI → what you measure
SLO → what you target internally
SLA → what you promise externally (with penalties)
```

**Example — API service:**

| Term | Example |
|---|---|
| SLI | 99.7% of requests in the last 30 days returned HTTP 2xx within 300ms |
| SLO | 99.9% of requests should succeed within 300ms |
| SLA | "We guarantee 99.9% availability; if we fall below, customers receive 10% service credit" |

**Error budget concept:**

```
Error Budget = 1 - SLO
Example:  SLO = 99.9%
          Error Budget = 0.1% = 43.8 minutes downtime per month allowed

If budget is consumed → freeze feature deployments, focus on reliability
If budget remains    → ship features, take calculated risks
```

**Design implication:** SLOs should be set slightly higher than SLAs to provide a safety buffer. If the SLA is 99.9%, the SLO might be 99.95% internally.

---

## MEDIUM (Q8–Q15)

---

### Q8. What are the "nines" of availability and what do they mean in practice?

**Answer:**

Availability is expressed as a percentage of uptime over a time period. Each additional "nine" dramatically reduces allowed downtime.

**Availability nines table:**

| Availability | Downtime/Year | Downtime/Month | Downtime/Week |
|---|---|---|---|
| 90% (1 nine) | 36.5 days | 72 hours | 16.8 hours |
| 99% (2 nines) | 3.65 days | 7.2 hours | 1.68 hours |
| 99.9% (3 nines) | 8.76 hours | 43.8 min | 10.1 min |
| 99.99% (4 nines) | 52.6 min | 4.38 min | 1.01 min |
| 99.999% (5 nines) | 5.26 min | 26.3 sec | 6.05 sec |
| 99.9999% (6 nines) | 31.5 sec | 2.63 sec | 0.6 sec |

**Compound availability — serial components:**

```
If system has multiple components A → B → C:
Total availability = Availability_A × Availability_B × Availability_C

Example:
  Web server:  99.99%
  Database:    99.9%
  Cache:       99.99%

Total = 0.9999 × 0.999 × 0.9999
      = 0.9988 ≈ 99.88%  (worse than any single component)
```

**Parallel redundancy improves availability:**

```
Two independent systems in parallel (either can serve):
Total = 1 - (1 - A1) × (1 - A2)

Example: Two 99.9% systems in parallel:
Total = 1 - (0.001 × 0.001) = 1 - 0.000001 = 99.9999%
```

**Design implications:**
- 99.9% is achievable with a single region, good monitoring, and fast rollbacks.
- 99.99% requires eliminating SPOFs, health checks, automated failover.
- 99.999% requires multi-region active-active with no single dependency.

**Cost vs availability:** Each additional nine roughly 10× the infrastructure and operational cost.

---

### Q9. What is a single point of failure (SPOF) and how do you eliminate it?

**Answer:**

A Single Point of Failure (SPOF) is any component whose failure causes the entire system to become unavailable. Identifying and eliminating SPOFs is a core responsibility in system design.

**Common SPOFs and mitigations:**

```
BEFORE (with SPOFs)              AFTER (HA design)
─────────────────────────────    ────────────────────────────────
Client                           Client
   │                                │
[Single LB]   ← SPOF            [LB Cluster: Active/Passive]
   │                                │
[Single Web Server] ← SPOF      [Web Server Pool] (N servers)
   │                                │
[Single DB]   ← SPOF            [Primary DB] + [Replica DB(s)]
                                        + automated failover
```

**SPOF elimination strategies:**

| Component | SPOF Risk | Mitigation |
|---|---|---|
| Load balancer | Yes — one LB fails | Active-passive LB pair (VRRP/floating IP) |
| Application server | Only if single instance | Horizontal pool behind LB |
| Database | Yes — one primary | Primary + replicas + automatic failover (e.g., RDS Multi-AZ) |
| DNS | Rarely, but possible | Redundant DNS servers (TTL + multiple providers) |
| Network switch | Yes in single-DC | Redundant switches, multi-path networking |
| Entire data center | Yes for single-DC | Multi-region or multi-AZ deployment |
| Third-party service | Dependency SPOF | Circuit breakers, fallback behavior |

**Key principle:** Every layer of the stack should have redundancy. When designing, ask for every component: "What happens if this fails?" If the answer is "everything goes down," you have a SPOF.

**SPOF in data:** A database with no replication is a SPOF for data durability. A single Kafka broker with no replication factor is a SPOF for messaging.

---

### Q10. What is graceful degradation, and how is it different from fault tolerance?

**Answer:**

**Fault tolerance** means a system continues operating *at full functionality* despite component failures. The failures are hidden from the user entirely.

**Graceful degradation** means a system continues operating *at reduced functionality* when components fail. Some features are unavailable, but the core experience remains intact.

```
E-commerce site example:

FAULT TOLERANT (ideal, expensive):
  Recommendation service fails → seamlessly fails over to replica
  User sees: exactly the same recommendations

GRACEFUL DEGRADATION (practical):
  Recommendation service fails → show static "popular items" list
  User sees: "You might also like" with generic items instead of
             personalized ones — but can still browse and buy
```

**Techniques for graceful degradation:**

1. **Feature flags:** Disable non-critical features under load.
2. **Fallback responses:** Return cached/default data when live data unavailable.
3. **Circuit breakers:** Stop calling a failing service, return fallback immediately.
4. **Read from cache:** If DB is down, serve stale but available cached data.
5. **Queue and retry:** Accept writes to a queue even if the primary DB is overloaded.

**Design example — news feed:**

```
Normal:         Personalized feed from ML service
Degraded L1:    Personalized feed from cache (slightly stale)
Degraded L2:    Generic trending feed (ML service unavailable)
Degraded L3:    Static "maintenance mode" page (all services down)
```

**Contrast with fault tolerance:**
| Property | Fault Tolerance | Graceful Degradation |
|---|---|---|
| Visibility to user | Zero impact | Reduced functionality |
| Cost | Higher (full redundancy) | Lower (partial fallbacks) |
| Complexity | Requires hot standby | Requires fallback logic |

---

### Q11. Explain CAP theorem. What does it mean for system design?

**Answer:**

CAP theorem (Brewer's theorem, 2000) states that a distributed system can only guarantee **two of three** properties simultaneously:

- **C — Consistency:** Every read receives the most recent write or an error. All nodes see the same data at the same time.
- **A — Availability:** Every request receives a response (not necessarily the most recent data). The system is always operational.
- **P — Partition Tolerance:** The system continues operating despite network partitions (nodes cannot communicate with each other).

**The catch:** Network partitions are unavoidable in distributed systems. You *must* choose P. So the real choice is **CP vs AP**:

```
         Consistency
              │
         CP   │   CA (not realistic in distributed systems)
              │
──────────────┼──────────────
              │
         AP   │
              │
    Availability ────── Partition Tolerance
```

**Real-world database classification:**

| Database | Category | Behavior during partition |
|---|---|---|
| PostgreSQL | CA → CP (when distributed) | Rejects writes to maintain consistency |
| Apache Cassandra | AP | Accepts writes on all nodes, reconciles later |
| HBase | CP | Becomes unavailable if region server fails |
| DynamoDB (default) | AP | Eventually consistent reads |
| DynamoDB (strong consistency) | CP | Higher latency, may reject in partition |
| Zookeeper / etcd | CP | Refuses reads/writes during partition |
| CouchDB | AP | Accepts writes, merges conflicts |

**Design implication:**

- **Financial systems, inventory, booking:** Use CP — a user should never see stale account balances or double-book a seat.
- **Social feeds, product recommendations, analytics:** Use AP — a user can tolerate seeing a slightly stale like count.

**Important nuance:** CAP theorem applies only during network partitions. Most of the time, modern systems are both consistent and available. CAP describes the *failure mode* behavior, not normal operation.

---

### Q12. What is the difference between ACID and BASE?

**Answer:**

**ACID** (Atomicity, Consistency, Isolation, Durability) is the traditional transactional model used by relational databases. It prioritizes correctness.

**BASE** (Basically Available, Soft state, Eventually consistent) is the model used by many NoSQL distributed systems. It prioritizes availability and performance over strict consistency.

**ACID explained:**

```
ATOMICITY:    All operations in a transaction succeed, or ALL are rolled back.
              (No partial updates — bank transfer: debit + credit both succeed or both fail)

CONSISTENCY:  A transaction takes DB from one valid state to another.
              (Constraints, foreign keys, and triggers are always satisfied)

ISOLATION:    Concurrent transactions produce results as if executed serially.
              (Isolation levels: Read Uncommitted → Repeatable Read → Serializable)

DURABILITY:   Once committed, data survives crashes (written to disk/WAL).
```

**BASE explained:**

```
BASICALLY AVAILABLE:   System guarantees availability (with partial failures allowed)
SOFT STATE:            State may change over time even without input (replication convergence)
EVENTUALLY CONSISTENT: Given no new updates, all replicas will eventually converge
```

**Comparison table:**

| Property | ACID | BASE |
|---|---|---|
| Consistency model | Strong (immediate) | Eventual |
| Availability | May sacrifice during partition | Always available |
| Performance | Lower (locking, 2PC overhead) | Higher (no global coordination) |
| Use cases | Banking, orders, inventory | Social, analytics, IoT, caching |
| Example DBs | PostgreSQL, MySQL, Oracle | Cassandra, DynamoDB, MongoDB |

**Practical guidance:**
- Use ACID for anything involving money, inventory, or legally auditable records.
- Use BASE for high-scale workloads where user-visible staleness of a few seconds is acceptable.

---

### Q13. What is the difference between synchronous and asynchronous design?

**Answer:**

**Synchronous design:** The caller waits (blocks) for the response before continuing. The operation completes end-to-end in a single request-response cycle.

**Asynchronous design:** The caller submits a request and continues processing. The response arrives later via a callback, polling, or event.

```
SYNCHRONOUS FLOW:
Client ──request──> Service A ──call──> Service B
Client ◄─────────────────────────────────────────── waits until done

ASYNCHRONOUS FLOW:
Client ──request──> Service A ──enqueue──> [Queue/Broker]
Client ◄── 202 Accepted (immediately)

[Queue/Broker] ──deliver──> Service B (processes later)
Service B ──result──> [Result Store or Webhook]
```

**Comparison:**

| Dimension | Synchronous | Asynchronous |
|---|---|---|
| Latency (caller perspective) | High (waits for full processing) | Low (immediate acknowledgment) |
| Throughput | Limited by slowest service | Decoupled; can absorb spikes |
| Coupling | Tight (caller needs receiver to be up) | Loose (queue buffers failures) |
| Error handling | Immediate error propagation | Errors need dead-letter queues, retries |
| Complexity | Simple to reason about | Requires queue infrastructure, idempotency |
| Use cases | Read queries, real-time responses | Email, notifications, video encoding, ETL |

**When to use async:**
- Long-running operations (> 500ms) where the user does not need an immediate result.
- Workloads with bursty traffic that exceed receiver capacity.
- Workflows across multiple services where partial failure should not block the caller.
- Fire-and-forget operations (logging, analytics, audit trails).

**Common async patterns:**
- **Message queues:** RabbitMQ, SQS — point-to-point.
- **Event streams:** Kafka, Kinesis — fan-out, replay.
- **Webhooks:** Service calls you back on completion.
- **Polling:** Client periodically checks a status endpoint.

---

### Q14. What is a trade-off analysis framework for system design decisions?

**Answer:**

Every system design decision involves trade-offs. A structured framework prevents the common mistake of presenting solutions without acknowledging costs.

**The COST framework:**

```
C — Consistency implications
O — Operational complexity introduced
S — Scalability impact (does this help or hinder growth?)
T — Trade-off summary (what are you giving up?)
```

**Decision matrix example — choosing a database:**

| Criterion | PostgreSQL | Cassandra | Redis |
|---|---|---|---|
| Consistency | Strong (ACID) | Eventual (tunable) | Strong (single node) |
| Write throughput | Medium | Very High | Extremely High |
| Query flexibility | Full SQL | Limited (partition key only) | Key-based only |
| Operational cost | Low | High (tuning needed) | Medium |
| Data size | GB–TB | TB–PB | GB (in-memory) |
| Best for | Transactions, joins | Time-series, wide column | Sessions, caching |

**Framing trade-offs in interviews:**

Always use the phrase: *"I would choose X over Y because of [reason], accepting the trade-off that [cost]."*

**Example statements:**
- "I'd use eventual consistency here, accepting that users might see a 5-second stale read, because this gives us 3× write throughput which matters at our scale."
- "I'd add a cache layer, accepting the added complexity of cache invalidation, because our read-to-write ratio is 100:1 and DB cost would be prohibitive otherwise."
- "I'd use a message queue here, accepting increased latency for the caller, because it decouples the services and lets us absorb traffic spikes without dropping requests."

**Key trade-off pairs to know:**
| Give up | Gain |
|---|---|
| Consistency | Availability, performance |
| Simplicity | Flexibility, scale |
| Latency | Throughput (via batching) |
| Durability | Write speed |
| Coupling | Developer ergonomics |

---

### Q15. When should you use microservices vs a monolith?

**Answer:**

This is one of the most debated questions in system design. The answer depends on team size, scale, and organizational maturity — not just technology preferences.

**Monolith characteristics:**

```
┌─────────────────────────────────────┐
│              Monolith               │
│  ┌─────────┐ ┌────────┐ ┌────────┐ │
│  │  Users  │ │ Orders │ │ Search │ │  ← All modules in one codebase
│  └─────────┘ └────────┘ └────────┘ │
│         Single database             │
│         Single deploy unit          │
└─────────────────────────────────────┘
```

**Microservices characteristics:**

```
[User Service]    [Order Service]    [Search Service]
      │                  │                  │
  [User DB]          [Order DB]         [Search Index]
      │                  │                  │
      └──────────[API Gateway]─────────────┘
```

**Decision framework:**

| Factor | Choose Monolith | Choose Microservices |
|---|---|---|
| Team size | < 15 engineers | > 50 engineers, multiple teams |
| Traffic | < 10K QPS | > 100K QPS, uneven load per service |
| Deployment cadence | Weekly releases | Independent deployments per service |
| Failure isolation | Not critical | Critical (one module should not crash others) |
| Technology diversity | Single stack preferred | Different services need different tech |
| Stage | Startup / MVP | Scale-up / mature product |

**Monolith pros:**
- Simple to develop, test, deploy, and debug.
- Lower operational overhead (one deployment pipeline).
- Easy cross-module transactions.
- No network call overhead between modules.

**Microservices pros:**
- Independent scaling of hot services.
- Independent deployment and release cycles.
- Fault isolation (one service crashing does not take down others).
- Technology independence per service.

**The pragmatic answer:** Start with a modular monolith. Extract microservices only when you have a clear, specific problem (a single module needing 10× more resources, a team ownership conflict, etc.). Premature decomposition is a common and costly mistake.

---

## HARD (Q16–Q20)

---

### Q16. What makes a system scalable, and what are the layers of scalability?

**Answer:**

Scalability is the ability of a system to handle increased load by adding resources, without requiring a redesign. A truly scalable system scales each layer independently.

**The scalability stack:**

```
Layer 1: DNS / Global Traffic Management
  └── Route users to nearest region; anycast routing

Layer 2: CDN (Content Delivery Network)
  └── Cache static assets at edge; reduces origin load by 80%+

Layer 3: Load Balancers
  └── Distribute requests across app servers; horizontal scaling entry point

Layer 4: Application / API Servers (Stateless)
  └── Easiest to scale — add instances behind LB; auto-scaling groups

Layer 5: Caching Layer (Redis / Memcached)
  └── Absorbs read load from DB; 100x faster than DB reads

Layer 6: Database
  └── Hardest to scale:
      Reads:  Add read replicas
      Writes: Sharding / partitioning
      Both:   Consider NoSQL for specific access patterns

Layer 7: Async Processing (Message Queues)
  └── Decouple write path; absorb spikes; enable horizontal worker pools

Layer 8: Storage
  └── Object storage (S3) scales infinitely; block storage scales with replicas
```

**Scalability anti-patterns:**

| Anti-Pattern | Problem | Solution |
|---|---|---|
| Shared mutable state | Creates write contention | Move state to dedicated store |
| Synchronous cross-service calls | Creates call chains that fail together | Async + circuit breakers |
| Fat application servers | Hard to autoscale specific bottlenecks | Split into smaller services |
| N+1 queries | DB hit per item in a list | Batch queries, eager loading |
| Unsharded write-heavy DB | Write bottleneck on single primary | Hash sharding |

**Measuring scalability:**
- **Linear scaling:** Doubling resources doubles throughput (ideal).
- **Sub-linear scaling:** Coordination overhead reduces returns.
- **Super-linear scaling:** Rare; caching effects can produce this.

**The key rule:** Stateless + horizontally scalable application tier + carefully designed data tier = a scalable system. The data tier is always the hardest and most expensive to scale.

---

### Q17. Explain the difference between reliability, availability, and maintainability (RAM).

**Answer:**

These three properties collectively describe how dependable a system is. They are related but measure different things.

**Reliability (R):** The probability that a system performs correctly for a specified time interval under given conditions. It is about *not failing*.

**Availability (A):** The fraction of time the system is operational and accessible. It includes recovery time.

**Maintainability (M):** How quickly a failed system can be restored to normal operation. It is measured as MTTR.

**Key metrics:**

```
MTTF  = Mean Time To Failure   (how long before something breaks)
MTTR  = Mean Time To Repair    (how long to fix it when it breaks)
MTBF  = Mean Time Between Failures = MTTF + MTTR

Availability = MTTF / (MTTF + MTTR)
             = MTTF / MTBF

Example:
  System fails every 1000 hours (MTTF = 1000h)
  Takes 1 hour to restore (MTTR = 1h)
  Availability = 1000 / (1000 + 1) = 99.9%

  Reduce MTTR to 6 min (0.1h):
  Availability = 1000 / (1000 + 0.1) = 99.99%
```

**RAM comparison table:**

| Property | Measures | Improved By | Metric |
|---|---|---|---|
| Reliability | Rate of failures | Redundancy, better code quality, testing | MTTF / MTBF |
| Availability | Uptime fraction | Redundancy + fast recovery | MTTF / (MTTF + MTTR) |
| Maintainability | Recovery speed | Automation, observability, runbooks | MTTR |

**Design strategies:**

**Improving Reliability:**
- Eliminate SPOFs through redundancy.
- Chaos engineering (Netflix's Chaos Monkey) to discover hidden failures.
- Rigorous testing: unit, integration, load, and chaos.

**Improving Availability:**
- Active-passive or active-active failover.
- Health checks with automatic traffic rerouting.
- Multi-AZ / multi-region deployments.

**Improving Maintainability:**
- Automated alerting and runbooks.
- Self-healing infrastructure (auto-restart, auto-scaling).
- Distributed tracing and observability (logs, metrics, traces).
- Blue-green / canary deployments for safe rollouts.

**Key insight:** High availability can be achieved with moderate reliability if you have excellent maintainability. A system that fails often but recovers in seconds (MTTR = 10s) can still achieve 99.99% availability.

---

### Q18. What is the back-of-envelope estimation methodology, and how do you apply it to a real system?

**Answer:**

Back-of-envelope (BOE) estimation is a rapid calculation technique to validate design feasibility before deep-diving into architecture. The goal is a correct *order of magnitude*, not exact numbers.

**Core methodology:**

```
Step 1: Establish user base and activity patterns
Step 2: Calculate QPS (read and write separately)
Step 3: Estimate storage requirements
Step 4: Estimate bandwidth
Step 5: Identify key constraints from these numbers
Step 6: Let constraints guide architecture choices
```

**Full worked example — Design YouTube:**

```
GIVEN:
  DAU        = 2 billion
  Watch/user = 5 videos/day
  Upload:    1 video per 1,000 users/day
  Avg video  = 500 MB (raw) → 300 MB stored (after compression, multi-res)
  Avg watch  = 300 MB transferred per video at 720p

─────────────────────────────────────────────────

STEP 1: QPS
  Video views/day  = 2B × 5 = 10B
  Read QPS         = 10B / 86,400 ≈ 115,000 QPS
  Peak read QPS    ≈ 345,000 QPS

  Uploads/day      = 2B / 1,000 = 2M videos/day
  Write QPS        = 2M / 86,400 ≈ 23 uploads/sec
  (Write QPS is low but each write is large)

STEP 2: STORAGE
  Daily upload storage = 2M × 300 MB = 600 TB/day
  5-year storage       = 600 TB × 365 × 5 ≈ 1.1 EB
  → Use object storage (S3/GCS), not block storage

STEP 3: BANDWIDTH
  Read bandwidth  = 115,000 QPS × 300 MB = impossible per-QPS
  (videos are streamed, not single requests)
  Realistic: avg bitrate 4 Mbps per viewer
  Peak bandwidth = 345,000 viewers × 4 Mbps = 1.38 Tbps
  → Requires CDN with global PoPs

STEP 4: METADATA (DB sizing)
  Video metadata  = 1 KB per video
  2M videos/day × 365 × 5 = 3.65B rows
  Storage for metadata = 3.65B × 1 KB ≈ 3.65 TB
  → Fits in a single large PostgreSQL cluster or sharded MySQL

─────────────────────────────────────────────────

ARCHITECTURAL IMPLICATIONS:
  • 1.38 Tbps egress → CDN is not optional, it is required
  • 600 TB/day writes → Distributed object storage required
  • 345K read QPS → Aggressive caching of video metadata
  • 23 uploads/sec → Async transcoding pipeline (not synchronous)
```

**Rules of thumb to memorize:**

| Operation | Approximate latency |
|---|---|
| L1 cache reference | 1 ns |
| RAM read | 100 ns |
| SSD random read | 100 μs |
| HDD seek | 10 ms |
| Network round trip (same DC) | 0.5 ms |
| Network round trip (US–EU) | 150 ms |
| Read 1 MB from SSD | 1 ms |
| Read 1 MB from network | 10 ms |

---

### Q19. How do you design for fault tolerance in a distributed system end-to-end?

**Answer:**

Fault tolerance in distributed systems requires defense-in-depth — multiple overlapping mechanisms at every layer, because any single mechanism will eventually fail.

**Failure taxonomy:**

```
Hardware failures:  Disk crash, NIC failure, power loss
Software failures:  Memory leaks, deadlocks, bugs
Network failures:   Packet loss, partition, high latency
Human errors:       Misconfiguration, bad deployments
Cascading failures: One service overloads, takes down dependents
```

**Fault tolerance mechanisms by layer:**

**Layer 1 — Infrastructure:**
```
Multi-AZ / Multi-Region deployment:
  Region A: [LB] → [App Cluster] → [DB Primary]
                                      ↕ replication
  Region B: [LB] → [App Cluster] → [DB Replica]
  
  DNS / GSLB: routes traffic away from failed region automatically
```

**Layer 2 — Application:**

```python
# Circuit Breaker implementation concept
class CircuitBreaker:
    states = ['CLOSED', 'OPEN', 'HALF_OPEN']
    
    # CLOSED: normal operation, pass requests through
    # OPEN:   service failed, reject immediately, return fallback
    # HALF_OPEN: probe with one request to check if service recovered
    
    def call(self, service_fn):
        if self.state == 'OPEN':
            if time_since_open > self.timeout:
                self.state = 'HALF_OPEN'
            else:
                return self.fallback()   # fast fail
        
        try:
            result = service_fn()
            self.on_success()
            return result
        except Exception:
            self.on_failure()
            raise
```

**Layer 3 — Data:**
```
Replication strategies:
  Sync replication:  Write confirmed on all replicas before ACK
                     → Zero data loss, higher write latency
  Async replication: Write confirmed on primary, replicas catch up
                     → Possible RPO > 0, lower write latency
  
  RPO = Recovery Point Objective (max acceptable data loss)
  RTO = Recovery Time Objective (max acceptable downtime)
```

**Layer 4 — Messaging:**
```
Message queue durability:
  At-most-once:  Fire and forget (no retries) — use for metrics
  At-least-once: Retry until ACK — use for orders (require idempotency)
  Exactly-once:  Transactional producers — use for financial records
```

**Layer 5 — Operational:**
- **Health checks:** Liveness (is the process alive?) + Readiness (can it serve traffic?).
- **Chaos engineering:** Intentionally inject failures (Chaos Monkey, Gremlin) to validate recovery mechanisms.
- **Bulkhead pattern:** Use separate thread pools/connection pools per downstream service so one slow service does not exhaust resources for all services.
- **Timeout + retry + jitter:** All service calls need timeouts. Retries need exponential backoff with jitter to prevent thundering herd during recovery.

```
Retry with exponential backoff + jitter:
  Attempt 1: wait 100ms   + random(0-50ms)
  Attempt 2: wait 200ms   + random(0-100ms)
  Attempt 3: wait 400ms   + random(0-200ms)
  Attempt 4: wait 800ms   + random(0-400ms)
  Max retries: 4, then dead-letter queue
```

---

### Q20. Walk through a complete system design trade-off analysis: choose between a microservices architecture and a modular monolith for a new e-commerce platform with 50K DAU growing to 5M DAU over 3 years.

**Answer:**

This question tests the ability to apply trade-off analysis to a realistic, time-bounded constraint. The answer must address both current and future state.

**Phase 1: Current state assessment (50K DAU)**

```
QPS Estimation (50K DAU):
  Avg requests/user/day = 30 (browse, search, cart, checkout)
  Avg QPS = (50,000 × 30) / 86,400 ≈ 17 QPS
  Peak QPS ≈ 50 QPS

This is trivially handleable by a single server.
```

**Phase 2: Future state assessment (5M DAU)**

```
QPS Estimation (5M DAU):
  Avg QPS = (5,000,000 × 30) / 86,400 ≈ 1,736 QPS
  Peak QPS ≈ 5,000 QPS

Breakdown by domain:
  Catalog/Search: 70% reads = ~3,500 read QPS at peak
  Cart/Orders:    10% writes = ~500 write QPS at peak
  Payments:        5% = ~250 QPS (requires strong consistency)
  User/Auth:      15% = ~750 QPS
```

**Architecture comparison:**

| Criterion | Modular Monolith | Microservices |
|---|---|---|
| Time-to-market (Y1) | Fast (single codebase) | Slow (infra setup, contracts) |
| Team at 50K DAU | 5–10 engineers: monolith wins | Overkill: 5 engineers cannot own 10 services |
| Operational cost (Y1) | $500/month (single cluster) | $3,000/month (per-service infra) |
| Scaling at 5M DAU | Requires manual vertical or selective extraction | Each service scales independently |
| Database transactions | Trivial within monolith | Distributed transactions are complex |
| Failure blast radius | Module bug can crash all | Isolated per service |
| Search scaling | Can extract just the search module | Search service scales independently |

**Recommended approach: Start with a modular monolith with extraction readiness:**

```
Year 1 (50K DAU):
  Modular Monolith:
  ┌──────────────────────────────────┐
  │  [Catalog Module] [Cart Module]  │
  │  [Order Module]  [User Module]   │
  │  [Payment Module]                │
  │  Single PostgreSQL + Redis cache │
  └──────────────────────────────────┘
  
  Key design rules:
  - Modules communicate via internal interfaces, NOT direct DB joins
  - Each module owns its own DB tables (logical separation)
  - Async communication via in-process event bus where possible
  
Year 2 (500K DAU):
  Extract first bottleneck: Catalog/Search
  ┌──────────────────────────────────┐   ┌─────────────────┐
  │  [Cart] [Order] [User] [Payment] │   │ [Search Service]│
  │  PostgreSQL + Redis              │   │ Elasticsearch   │
  └──────────────────────────────────┘   └─────────────────┘
  
Year 3 (5M DAU):
  Extract Payment (compliance isolation) and Orders (high write volume):
  [User+Auth] [Catalog] [Search] [Cart] [Order] [Payment]
  Each with own DB, connected via API Gateway + async events
```

**Trade-off summary:**

```
Decision: Modular Monolith → selective microservice extraction

GAIN:
  + Fast time-to-market in Year 1
  + Low operational overhead while team/traffic is small
  + Easy cross-domain transactions during growth
  + Clear extraction path to microservices when needed

COST:
  - Risk of architectural coupling if module boundaries not enforced
  - Year 2-3 extraction requires refactoring effort
  - Cannot scale individual modules without extracting them

MITIGATIONS:
  - Enforce module boundaries with linting rules / package structure
  - Use domain events internally from day one (extraction becomes easier)
  - Document API contracts between modules even within the monolith
```

This approach is used by Amazon (started as monolith), Shopify (still largely monolith at massive scale), and Stack Overflow (monolith serving 1B requests/month).

---

## Quick Reference

### Estimation Formulas
| Formula | Expression |
|---|---|
| Average QPS | `(DAU × requests/user/day) / 86,400` |
| Peak QPS | `Average QPS × 2-3` |
| Availability | `MTTF / (MTTF + MTTR)` |
| Compound availability (serial) | `A1 × A2 × A3` |
| Redundant availability (parallel) | `1 - (1-A1)(1-A2)` |

### Availability Nines
| Nines | Annual Downtime |
|---|---|
| 99% | 3.65 days |
| 99.9% | 8.76 hours |
| 99.99% | 52.6 minutes |
| 99.999% | 5.26 minutes |

### CAP Theorem Quick Guide
| Database | Type | Partition behavior |
|---|---|---|
| PostgreSQL | CP | Refuses writes to stay consistent |
| Cassandra | AP | Accepts writes, eventual consistency |
| Zookeeper | CP | Refuses requests during partition |
| DynamoDB | AP (tunable) | Eventually consistent by default |

### Design Decisions at a Glance
| Scenario | Choice |
|---|---|
| < 15 engineers, new product | Monolith |
| Read-heavy (100:1 ratio) | Add caching layer |
| Write-heavy, unstructured data | NoSQL (Cassandra/DynamoDB) |
| Financial transactions | ACID SQL database |
| Long-running tasks | Async with message queue |
| Global user base | CDN + multi-region |
| Fault isolation needed | Circuit breaker + bulkhead |

### Interview Framework Checklist
- [ ] Clarify functional requirements
- [ ] Clarify non-functional requirements (scale, latency, availability)
- [ ] Estimate QPS, storage, bandwidth
- [ ] Draw high-level architecture
- [ ] Identify bottlenecks
- [ ] Deep dive into hardest components
- [ ] Discuss trade-offs explicitly
- [ ] Address failure modes
