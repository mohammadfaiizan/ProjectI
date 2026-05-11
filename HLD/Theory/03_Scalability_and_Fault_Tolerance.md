# Scalability and Fault Tolerance

> How distributed systems scale to handle load and survive failures. This covers the patterns, techniques, and mental models required for senior-level system design discussions.

---

## Horizontal vs Vertical Scaling

### Vertical Scaling (Scale Up)

**Definition:** Upgrade the existing machine — more CPU, RAM, faster storage.

```
Before:          After:
[Server: 8 core] → [Server: 64 core]
[32 GB RAM]      → [512 GB RAM]
[1 TB SSD]       → [4 TB NVMe]
```

**Pros:**
- No application changes required
- Simpler architecture (still one machine)
- No distributed systems complexity (no network latency between components)
- Useful for databases with complex transactions (ACID)

**Cons:**
- Hard upper limit (largest server on the market)
- Single point of failure — one machine going down = total outage
- Downtime required for hardware upgrades
- Expensive: price/performance ratio worsens dramatically at high end
- Memory bandwidth becomes bottleneck beyond certain CPU counts

**Cost curve for vertical scaling:**
```
2x performance cost:
- CPU: roughly linear (2x CPUs ≈ 2x cost)
- RAM: slightly superlinear at high end
- Network: roughly linear
- But: single machine = single failure domain

10x performance via vertical: 
→ May not exist, or costs 20-50x more
```

**When to choose vertical scaling:**
- Early-stage products with simple architecture
- Databases with complex joins/transactions (ACID compliance)
- Applications that cannot be distributed easily (e.g., tightly coupled legacy systems)
- When the workload does not parallelize well

### Horizontal Scaling (Scale Out)

**Definition:** Add more machines to distribute load.

```
Before:          After:
[Server]    →    [Server 1]
                 [Server 2]
                 [Server 3]
                 [Load Balancer] → routes requests
```

**Pros:**
- Near-linear scalability (double machines ≈ double capacity)
- No upper hardware limit (in theory)
- No single point of failure (one server down ≠ outage)
- Can use commodity hardware (cheaper)
- Can scale incrementally as load grows

**Cons:**
- Application must be designed to run on multiple nodes (stateless)
- Network calls between nodes add latency
- Distributed systems complexity: consistency, consensus, partition
- Data sharding adds complexity
- Operational overhead of managing many machines

**When to choose horizontal scaling:**
- Stateless application servers (almost always)
- Read-heavy databases (read replicas)
- Cache layers (Redis Cluster)
- Web/API servers

### Scaling Decision Matrix

| Scenario | Recommendation |
|---|---|
| Web/API servers | Horizontal always |
| Primary database (write-heavy) | Vertical first, then shard |
| Cache servers | Horizontal (distributed cache) |
| Worker/job processing | Horizontal |
| GPU/ML inference | Vertical (large GPU instances) |
| Message brokers (Kafka) | Horizontal (partitions) |
| Search indexes | Horizontal (Elasticsearch shards) |

---

## Stateless Services — How to Make Services Stateless

### What Makes a Service Stateful?

A service is stateful when the server stores client-specific data between requests:
- In-memory session store
- Local file system writes
- In-process caches keyed per user
- Open database connections tied to specific logic

### Making Services Stateless

**Problem:** Session stored in application server memory
```
❌ Stateful Problem:
User logs in → Server A stores session in RAM
Next request → Load balancer routes to Server B
Result: Server B has no session → user is logged out!
```

**Solution 1: External Session Store (Redis)**
```
✓ Stateless with Redis:
User logs in → Server A writes session to Redis
Next request → Server B reads session from Redis (same data)
Result: Any server can handle any request

Implementation:
- Session ID in cookie or Authorization header
- Session data in Redis with TTL
- Key: session_id → Value: { user_id, permissions, preferences }
```

**Solution 2: JWT Tokens (Stateless by Design)**
```
User logs in → Server generates signed JWT token
Next request → Client sends JWT in Authorization header
Server validates JWT signature (no DB lookup!)
Result: Truly stateless — server doesn't store anything

JWT payload:
{
  "sub": "user_123",
  "roles": ["admin"],
  "exp": 1234567890
}

Trade-off: Can't invalidate JWT before expiry
Solution: Short-lived tokens (15 min) + refresh tokens (stored in Redis)
```

**Solution 3: Externalize File Storage**
```
❌ Stateful: App writes uploaded files to /var/uploads on server disk
✓ Stateless: App writes uploaded files to S3/GCS/Azure Blob

Any server can serve any file request → fully horizontally scalable
```

### Session Externalization Patterns

| Pattern | Latency | Scalability | Complexity |
|---|---|---|---|
| Server-side session (Redis) | ~1ms | High | Medium |
| JWT (client-side token) | 0ms (no lookup) | Infinite | Low (but token invalidation hard) |
| Database sessions | ~5-10ms | Medium | Low |
| Sticky sessions (L7 LB) | 0ms | Limited by server count | Low |

**Sticky sessions trade-off:**
```
Sticky sessions (route user to same server):
✓ Simple to implement
✗ Server failure loses all sticky sessions
✗ Uneven load distribution (chatty users hit same server)
✗ Limits autoscaling flexibility

Only use when: you cannot externalize state and need a quick fix
```

---

## Replication for Availability

### Active-Passive Replication (Primary-Replica)

```
          Writes
Client ──────────────→ [Primary DB]
                              │
                              │ Async replication
                              ↓
          Reads       [Replica 1]
Client ──────────────→ [Replica 2]
                       [Replica 3]
```

**How it works:**
- All writes go to primary
- Primary replicates changes to replicas (async or sync)
- Reads distributed across replicas
- On primary failure: manual or automatic promotion of a replica to primary

**Async vs Sync Replication:**

| | Async Replication | Sync Replication |
|---|---|---|
| Durability | Risk of data loss (replica lag) | No data loss |
| Write latency | Low (don't wait for replica) | Higher (wait for replica to confirm) |
| Availability | Higher | Lower (if replica fails, write fails) |
| Use case | Read scaling, near-durability | Financial data, strict consistency |

**Replication lag problem:**
```
Timeline:
T=0: Write to primary
T=0+lag: Replica receives write (lag can be 0-100s under load!)

Problem: User writes data, immediately reads → may get stale data from replica
Solution options:
1. Read-your-writes: after a write, send subsequent reads to primary for 1 second
2. Session stickiness to primary for the writing user
3. Sync replication for critical data
```

### Active-Active Replication

```
          Writes/Reads
Client A ──────────────→ [DB Node 1] ←──→ [DB Node 2] ←── Client B
                                 ↑                 ↓
                                  ←────────────────
                           (both accept writes, sync state)
```

**How it works:**
- Multiple primary nodes accept reads and writes
- Nodes sync changes to each other
- Conflict resolution required when same data written to two nodes simultaneously

**Conflict resolution strategies:**
- **Last Write Wins (LWW):** timestamp determines winner — risk of clock skew
- **Application-level:** application code resolves conflicts
- **CRDT (Conflict-free Replicated Data Type):** data structures designed to merge automatically
- **Operational transformation:** used by Google Docs for collaborative editing

**When to use active-active:**
- Multi-region deployments (write to nearest region)
- Need zero-downtime failover
- Examples: DynamoDB global tables, CockroachDB, Cassandra (multi-datacenter)

**Trade-off:** Complex conflict resolution vs active-passive simplicity.

---

## Redundancy Patterns

### N+1 Redundancy

**Definition:** Have N units required for full capacity + 1 spare.

```
If you need 3 servers to handle load:
N+1 = 4 servers deployed
→ One can fail and system still operates at full capacity

If you need 5 servers:
N+1 = 6 servers
→ One fails: 5 remaining handle load normally
```

**Where N+1 is applied:**
- Load balancers: active-passive pair
- Database primaries: primary + hot standby
- Power supplies in servers: dual PSU
- Network switches: redundant top-of-rack switches

### N+2 and 2N Redundancy

| Pattern | Description | Use Case |
|---|---|---|
| N+1 | 1 spare unit | Standard availability |
| N+2 | 2 spare units | Allows 2 simultaneous failures |
| 2N | Full duplication | Mission-critical (hospital systems) |
| 2N+1 | Two full sets + 1 | Extreme HA (tier 4 datacenter) |

### Geographic Redundancy

**Single Region, Multiple AZs:**
```
AWS Region (us-east-1)
├── AZ-1 (us-east-1a): App servers + DB primary
├── AZ-2 (us-east-1b): App servers + DB replica
└── AZ-3 (us-east-1c): App servers + DB replica

Load balancer spans all AZs
→ Survives single AZ failure (power outage, cooling failure)
→ Does NOT survive region failure
```

**Multi-Region Active-Passive:**
```
Primary Region (us-east-1): 100% traffic
Secondary Region (us-west-2): 0% traffic, data replicated

On primary region failure:
→ DNS failover (TTL matters!)
→ Secondary region takes over
→ RPO: depends on replication lag
→ RTO: DNS propagation + warmup (~1-10 minutes)
```

**Multi-Region Active-Active:**
```
Region A (us-east-1): 50% traffic (US East users)
Region B (eu-west-1): 30% traffic (European users)
Region C (ap-southeast-1): 20% traffic (Asian users)

Each region:
→ Has local app servers, caches, databases
→ Replicates asynchronously to other regions
→ Globally consistent via global DB (Spanner, CockroachDB) or eventual consistency

Benefits: Lowest latency (users served from nearest region)
          Survives regional outages
Challenges: Data consistency, conflict resolution, cost (3x infrastructure)
```

---

## Failure Modes

### Hardware Failures

```
Probability estimates (per year):
- Server failure: ~1-5%
- HDD failure: ~2-5%
- SSD failure: ~0.5-1%
- Network card failure: ~0.1-1%
- Switch failure: ~0.5%

With 1,000 servers: expect ~10-50 server failures per year
→ Hardware failure is expected, not exceptional
→ System must be designed to handle them automatically
```

**Mitigation:**
- RAID for disk redundancy (RAID 10 for performance + redundancy)
- Redundant PSU (dual power supplies)
- Redundant NICs
- ECC RAM (error-correcting code — detects and corrects single-bit errors)
- Automatic failover to healthy nodes

### Network Partitions

**Definition:** A network failure that splits the cluster into two groups that cannot communicate.

```
Before partition:        During partition:
[A]──[B]──[C]           [A]──[B]    [C]
  all connected          A,B can't reach C

Each partition may continue serving requests independently
→ Risk: A+B update data, C updates same data
→ When partition heals: conflict!
```

**Partition types:**
- **Split-brain:** Two nodes both think they are primary
- **Asymmetric partition:** A can reach B, B can reach A, but clients can't reach B
- **Full partition:** Node is completely isolated

**Mitigation:**
- Raft/Paxos consensus algorithms
- Leader election with quorum (majority must agree)
- Fencing tokens (prevent split-brain writes)
- Network redundancy (multiple switch paths)

### Cascading Failures

**Definition:** One failure causes overload on remaining components, which then fail, causing further overload.

```
Scenario: 10 servers handle 100K QPS (10K each)
Server 1 fails → 90K QPS across 9 servers (10K each) → still fine
Server 2 fails → 90K QPS across 8 servers (11.25K each) → above capacity!
Servers 3 & 4 fail due to overload → ...
→ Complete system collapse from one initial failure
```

**Prevention strategies:**
- **Circuit breaker:** Stop sending requests to a failing service
- **Load shedding:** Deliberately drop requests when overloaded
- **Bulkhead pattern:** Isolate components so one failure doesn't affect others
- **Autoscaling:** Automatically add capacity when load increases
- **Rate limiting:** Prevent any one client from overwhelming the system

---

## Circuit Breaker Pattern

### State Machine

```
         Request succeeds      Too many failures
            
    ┌──── CLOSED ──────────────────→ OPEN
    │         ↑                          │
    │         │ Some requests            │ Wait timeout
    │         │ succeed in half-open     ↓
    │         └───────────── HALF-OPEN ←─┘
    │                             │
    └─────────────────────────────┘
      All requests fail in half-open
```

### States Explained

**CLOSED (Normal Operation):**
- All requests pass through to the downstream service
- Circuit breaker monitors for failures
- If failures exceed threshold (e.g., 50% in last 30 seconds) → trips to OPEN

**OPEN (Failing Fast):**
- ALL requests are immediately rejected (fail fast)
- No requests sent to downstream service
- Returns error immediately (no waiting for timeout)
- After a configured sleep window (e.g., 60 seconds) → transition to HALF-OPEN

**HALF-OPEN (Probing Recovery):**
- Allow a limited number of test requests through
- If test requests succeed → reset to CLOSED
- If test requests fail → return to OPEN

### Circuit Breaker Configuration

```python
# Pseudocode - Circuit Breaker Configuration
CircuitBreaker(
    failure_threshold = 50,        # trips at 50% failure rate
    failure_count_window = 30,     # seconds
    minimum_requests = 10,         # at least 10 req before tripping
    sleep_window = 60,             # stay OPEN for 60 seconds
    half_open_max_requests = 3     # allow 3 test requests in HALF-OPEN
)
```

### When to Use Circuit Breaker

```
✓ Calls to external services (payment APIs, SMS providers)
✓ Database connection pools under stress
✓ Microservice-to-microservice calls
✓ Any remote network call with potential for cascading failure

❌ Don't use for: in-process function calls, reads from local cache
```

### Circuit Breaker vs Retry

```
Retry:          Good when failure is transient (network hiccup)
Circuit Breaker: Good when service is consistently failing

Use both together:
1. First: circuit breaker checks if open → fast fail
2. If closed: make request
3. If fails: retry 2-3 times with backoff
4. If still failing: circuit breaker tracks failure, may trip
```

---

## Bulkhead Pattern

**Concept:** From ship design — compartments prevent one hole from sinking the ship.

**In software:** Isolate resources for different services/consumers so one consumer's failure doesn't exhaust resources for others.

### Thread Pool Bulkhead

```
❌ Without bulkhead:
All API calls share one thread pool (100 threads total)
→ Slow database API calls consume all 100 threads
→ Fast, working API calls queue up and timeout
→ System appears down for all users

✓ With bulkhead (separate thread pools):
Database API calls: 30 threads (bulkhead)
Payment API calls:  20 threads (bulkhead)  
User API calls:     30 threads (bulkhead)
Other:              20 threads

→ Database API calls fill up their 30 threads
→ Payment and User API calls continue unaffected!
```

### Connection Pool Bulkhead

```
Without bulkhead:
All services share one DB connection pool (50 connections)
→ Reporting service runs heavy query, uses all 50
→ User login can't get a connection!

With bulkhead:
Reporting service:  5 connections (limited!)
Transaction service: 20 connections
Read service:       15 connections
Admin service:       5 connections
Overflow pool:       5 connections

→ Reporting can't starve critical services
```

### Bulkhead vs Circuit Breaker

| Pattern | Protects Against | Mechanism |
|---|---|---|
| Circuit Breaker | Cascading failure from slow/failing dependency | Stops sending requests to failing service |
| Bulkhead | Resource exhaustion from noisy neighbor | Limits resources allocated to each service |

---

## Retry with Exponential Backoff and Jitter

### Naive Retry (Bad)

```
❌ Simple retry (thundering herd problem):
Request fails at T=0
All clients retry at T=1 second simultaneously
→ Service gets 100x traffic spike at T=1!
→ Service still fails
→ All retry at T=2... T=3... 
→ Continuous spike prevents recovery
```

### Exponential Backoff

```
✓ Exponential backoff:
Attempt 1 fails → wait 1 second
Attempt 2 fails → wait 2 seconds
Attempt 3 fails → wait 4 seconds
Attempt 4 fails → wait 8 seconds
Attempt 5 fails → wait 16 seconds
→ Give up (or cap at max wait)

Formula: wait = min(cap, base * 2^attempt)
Example: min(32s, 1s * 2^attempt)
```

### Adding Jitter

```
Problem: All clients use same backoff → still spike at T=1, T=2, T=4...

✓ Full jitter:
wait = random_between(0, min(cap, base * 2^attempt))

✓ Equal jitter (better for latency-sensitive):
temp = min(cap, base * 2^attempt)
wait = temp/2 + random_between(0, temp/2)

✓ Decorrelated jitter (AWS recommendation):
wait = min(cap, random_between(base, previous_wait * 3))

```

### Retry Implementation

```python
def retry_with_backoff(func, max_attempts=5, base_delay=1.0, max_delay=32.0):
    for attempt in range(max_attempts):
        try:
            return func()
        except TransientException as e:
            if attempt == max_attempts - 1:
                raise  # last attempt, propagate error
            
            # Exponential backoff with full jitter
            delay = min(max_delay, base_delay * (2 ** attempt))
            sleep_time = random.uniform(0, delay)
            time.sleep(sleep_time)
        except PermanentException as e:
            raise  # Don't retry permanent errors (404, 400, etc.)
```

### What to Retry vs Not Retry

```
✓ RETRY these errors:
- 503 Service Unavailable (server overloaded)
- 504 Gateway Timeout
- 429 Too Many Requests (with Retry-After header)
- 500 Internal Server Error (sometimes transient)
- Network timeouts
- Connection refused (service restarting)

❌ DO NOT RETRY:
- 400 Bad Request (client error — retrying won't help)
- 401 Unauthorized (need new token first)
- 403 Forbidden (permissions issue)
- 404 Not Found (resource doesn't exist)
- 409 Conflict (business logic conflict)
- Idempotency note: never retry non-idempotent operations without idempotency key!
```

---

## Timeout Design

### Three-Layer Timeout Model

```
Client Browser (10 sec timeout)
    ↓
API Gateway (8 sec timeout)
    ↓
Service A (5 sec timeout)
    ↓
Database (3 sec timeout)

Rule: Inner timeouts < outer timeouts
→ Inner failure triggers before outer timeout
→ Allows graceful error propagation
```

### Types of Timeouts

**Connection Timeout:** Time to establish a TCP connection
```
connection_timeout = 2 seconds (typical)
→ If server doesn't respond to SYN within 2s, fail
```

**Read Timeout:** Time waiting for data after connection established
```
read_timeout = 5 seconds (typical)
→ If no data received for 5s after last byte, fail
```

**Write Timeout:** Time to complete sending a request
```
write_timeout = 5 seconds
→ Relevant for large file uploads
```

**End-to-End Timeout (Deadline):** Total time budget for an operation
```
# gRPC context deadline
context.WithTimeout(ctx, 10*time.Second)

# All downstream calls must complete within this budget
# Propagate deadline to all children calls
# Remaining time: 10s - 3s DB call - 2s cache call = 5s remaining
```

### Timeout Anti-Patterns

```
❌ No timeout: 
Thread hangs forever if downstream never responds
→ Thread pool exhaustion
→ Cascading failure

❌ Same timeout everywhere:
"Every call has 30s timeout"
→ Outer service can't respond in 10s if inner service uses 30s

❌ Timeout without circuit breaker:
Every request waits for full timeout before failing
→ Under high load: all threads blocked on timeout
→ Solution: circuit breaker fails fast before timeout

❌ Non-idempotent retries after timeout:
POST /payments timed out → retry
→ Payment charged twice!
→ Solution: idempotency keys for non-idempotent operations
```

---

## Graceful Degradation

**Definition:** System continues operating in a reduced-capability mode under failure or high load, rather than failing completely.

### Degradation Strategies

**Serve Stale Data:**
```
Normal: Cache miss → fetch from DB → update cache → return fresh data
Degraded: DB is down → serve stale cache data with "as of X minutes ago" indicator

Implementation:
- Cache entries have TTL = 60 minutes (normal freshness)
- On cache miss + DB failure: extend TTL 10x, serve stale
- Alert engineers but don't fail for users
```

**Feature Degradation:**
```
Normal mode: Full recommendation engine (ML model, personalized)
Degraded:    Simple trending content (no personalization)
Emergency:   Static curated list

Implement with feature flags:
if recommendation_service.is_healthy():
    show personalized recommendations
elif trending_service.is_healthy():
    show trending content
else:
    show static curated list
```

**Load Shedding:**
```
When CPU > 80%:
→ Start rejecting non-critical requests (analytics, batch jobs)
→ Prioritize user-facing reads over writes
→ Reject expensive search queries

Priority queues:
P1: Authentication, payments
P2: Core user actions (read/write)
P3: Analytics, recommendations
P4: Background jobs
→ Shed P4 first, then P3, then P2
```

**Read-Only Mode:**
```
Database primary down → switch to read-only (replica only)
→ Users can read their data
→ Writes fail with clear error: "System temporarily in read-only mode"
→ Better than complete outage

Examples:
- GitHub: "GitHub.com is currently read-only"
- Google Drive: "Changes cannot be saved, read-only mode"
```

---

## Health Checks: Liveness vs Readiness Probes

### Liveness Probe

**Purpose:** "Is the container/process alive?"
**Action if fails:** Restart the container/process

```
What to check:
- Is the process running?
- Is it stuck in an infinite loop?
- Is it deadlocked?

Example endpoint: GET /health/live
Response 200: "I am alive"
Response 500: "I am deadlocked/stuck"

Implementation:
- Simple: return 200 always (basic process alive check)
- Better: check if main thread is responsive
- Avoid: checking external dependencies (DB) — this should be readiness
```

### Readiness Probe

**Purpose:** "Is the service ready to receive traffic?"
**Action if fails:** Remove from load balancer rotation (but don't restart)

```
What to check:
- Can it connect to the database?
- Is the cache warm enough?
- Have all initialization tasks completed?
- Is the connection pool saturated?

Example endpoint: GET /health/ready
Response 200: "Ready to serve traffic"
Response 503: "Database connection failed / initializing"

Implementation:
- Check DB connection: SELECT 1
- Check Redis connection: PING
- Check required config loaded
- Check queue consumer connected
```

### Startup Probe

**Purpose:** "Has the application finished starting up?"
**Action if fails:** Restart; allows longer initial startup without false liveness failures.

```
Use when: Application takes a long time to start (e.g., warm up cache, 
          load ML model, run DB migrations)
Once startup probe succeeds: liveness and readiness probes take over
```

### Probe Configuration

```yaml
# Kubernetes example
livenessProbe:
  httpGet:
    path: /health/live
    port: 8080
  initialDelaySeconds: 30   # Wait 30s before first probe (startup time)
  periodSeconds: 10          # Check every 10s
  failureThreshold: 3        # Restart after 3 consecutive failures
  timeoutSeconds: 5          # Probe must respond within 5s

readinessProbe:
  httpGet:
    path: /health/ready
    port: 8080
  initialDelaySeconds: 10
  periodSeconds: 5
  failureThreshold: 2        # Remove from LB after 2 failures (faster)
  successThreshold: 2        # Requires 2 successes to add back to LB
```

---

## Chaos Engineering Principles

**Definition:** Deliberately injecting failures into production (or staging) to find weaknesses before they cause real incidents.

### Netflix Simian Army (Origin of Chaos Engineering)

```
Chaos Monkey:     Randomly terminates EC2 instances
Chaos Gorilla:    Simulates AZ outage
Chaos Kong:       Simulates region outage
Latency Monkey:   Injects artificial latency
Conformity Monkey: Finds instances not following best practices
Security Monkey:  Finds security vulnerabilities
```

### Chaos Engineering Process

```
Step 1: Define steady state
"System handles 100K QPS with P99 latency < 200ms"

Step 2: Hypothesize what will happen
"If we kill 30% of app servers, load balancer redistributes
 and P99 stays below 300ms"

Step 3: Design the experiment
"Kill 3 of 10 app servers, monitor for 5 minutes"

Step 4: Run in controlled blast radius
"Run in staging first, then prod off-peak, then prod peak"

Step 5: Verify hypothesis
"P99 went to 280ms — within acceptable range"
or
"P99 spiked to 2000ms — we found a weakness!"

Step 6: Fix the weakness
"Found: connection pool not sized for traffic on 7 servers
 Fix: increase pool size or add server"
```

### Common Chaos Experiments

| Experiment | What it Tests |
|---|---|
| Kill random service instances | Autoscaling, health check recovery |
| Introduce 500ms network latency | Timeout handling, circuit breakers |
| Fill disk to 90% | Disk space handling, log rotation |
| Kill primary database | Failover speed, RTO |
| Saturate CPU to 90% | Load shedding, autoscaling triggers |
| Inject packet loss (5%) | Retry logic, backoff |
| Kill DNS resolver | DNS caching, fallback |
| Rotate TLS certificates | Certificate renewal automation |

---

## SLA, SLO, and SLI Definitions

### The Hierarchy

```
SLA (Service Level Agreement)
  └── Legal contract between provider and customer
  └── Defines consequences of not meeting SLOs
  └── "If we miss 99.9% uptime, you get 10% bill credit"

  SLO (Service Level Objective)
    └── Internal target that drives SLAs
    └── Stricter than SLA (buffer for safety)
    └── "Our API will respond in < 200ms P99"

    SLI (Service Level Indicator)
      └── The actual measured metric
      └── "API P99 latency today: 156ms" (currently meeting SLO)
```

### Practical Examples

| SLI | SLO | SLA |
|---|---|---|
| API response time P99 | < 200ms | < 500ms (legal threshold) |
| Availability (uptime %) | > 99.95% | > 99.9% (in contract) |
| Error rate | < 0.1% | < 0.5% (in contract) |
| Data durability | 99.9999999% | 99.9999% |

**Why SLO is stricter than SLA:**
```
SLA: 99.9% (8.76 hours downtime/year)
SLO: 99.95% (4.38 hours downtime/year)

Buffer = 4.38 hours
→ Even if you miss your SLO, you're still within SLA
→ SLO violation triggers internal alert and action
→ SLA violation triggers customer credits and legal implications
```

### Error Budget

```
Error Budget = 100% - SLO

If SLO = 99.9%:
Error budget = 0.1% of time = 8.76 hours/year = 43.8 minutes/month

Uses of error budget:
✓ Planned maintenance
✓ Canary deployments (accept brief errors)
✓ Risky experiments

When budget exhausted:
→ Freeze risky deployments
→ Focus on reliability improvement
→ No new features until budget replenishes
```

---

## Calculating Composite Availability

### Systems in Series (All must work)

```
If any component fails, the system fails.

           Service A    Service B    Service C
Client → [99.9%] → [99.9%] → [99.9%]

Composite availability = A × B × C
= 0.999 × 0.999 × 0.999
= 0.997 = 99.7%

Adding more components in series always REDUCES availability!
10 services at 99.9% = 0.999^10 = 99.0%
```

### Systems in Parallel (Any can work)

```
           Service A1 [99.9%]
Client → (or)
           Service A2 [99.9%]

Composite availability = 1 - (1-A1) × (1-A2)
= 1 - (0.001 × 0.001)
= 1 - 0.000001
= 99.9999%

Two 99.9% services in parallel → 99.9999% composite!
```

### Real-World Composite Example

```
System: Web application with database

Components:
- Load Balancer:     99.99%
- App Server (2):    99.9% each (in parallel)
- Redis Cache:       99.9%
- Database Primary:  99.9%

Parallel app servers: 1 - (0.001 × 0.001) = 99.9999%

Series (LB → App → Cache → DB):
0.9999 × 0.999999 × 0.999 × 0.999
= 0.9979 ≈ 99.79%

Monthly downtime: 0.21% × 30 days × 24 hr × 60 min = 90 minutes/month

To meet 99.9% SLA: need to fix the DB and Cache SPOFs
→ Add DB replica with auto-failover: DB availability → 99.99%
→ Add Redis replica: Cache availability → 99.99%
→ New composite: ~99.97% ✓
```

---

## Quick Reference: Availability Nines Table

| Availability | Annual Downtime | Monthly Downtime | Weekly Downtime |
|---|---|---|---|
| 90% | 36.5 days | 72 hours | 16.8 hours |
| 95% | 18.25 days | 36 hours | 8.4 hours |
| 99% | 3.65 days | 7.2 hours | 1.68 hours |
| 99.5% | 1.83 days | 3.6 hours | 50.4 minutes |
| 99.9% | 8.76 hours | 43.8 minutes | 10.1 minutes |
| 99.95% | 4.38 hours | 21.9 minutes | 5 minutes |
| 99.99% | 52.6 minutes | 4.4 minutes | 1 minute |
| 99.999% | 5.26 minutes | 26.3 seconds | 6 seconds |
| 99.9999% | 31.5 seconds | 2.6 seconds | 0.6 seconds |

---

## Failure Taxonomy

| Failure Type | Examples | Detection | Mitigation |
|---|---|---|---|
| Hardware failure | Server crash, disk failure | Health check, monitoring | N+1 redundancy, auto-failover |
| Network failure | Packet loss, partition | Heartbeat, timeout | Retry, circuit breaker |
| Software bug | Memory leak, OOM | APM, crash reports | Liveness probe, canary deploy |
| Dependency failure | DB down, API down | Health check, circuit breaker | Circuit breaker, fallback |
| Resource exhaustion | CPU 100%, OOM, connection pool full | Metrics, alerts | Load shedding, autoscale |
| Data corruption | Bit flip, bad write | Checksums, data validation | Backups, data integrity checks |
| Cascading failure | Overload spreading | Latency spike, error rate | Bulkhead, load shedding |
| Human error | Wrong config, bad deploy | Change management | Canary releases, rollback |
| Security attack | DDoS, SQL injection | Rate limiting, WAF | DDoS protection, input validation |
| Byzantine failure | Partial, arbitrary failures | Difficult to detect | BFT consensus, checksums |

---

## Fault Tolerance Design Checklist

```
For every critical component, verify:

□ Is there a backup/replica for this component?
□ Does failover happen automatically (without human intervention)?
□ What is the RTO (recovery time objective)?
□ What is the RPO (recovery point objective)?
□ Is there a circuit breaker for calls to this component?
□ Are retries configured with backoff and jitter?
□ Are timeouts set correctly (inner < outer)?
□ Is there graceful degradation if this component fails?
□ Are there health checks (liveness + readiness)?
□ Has this failure been tested (chaos engineering)?
□ Are alerts set for approaching SLO thresholds?
□ Is the error budget being tracked?
```

---

*Reference: "Designing Data-Intensive Applications" by Martin Kleppmann, "Release It!" by Michael Nygard, Netflix Tech Blog*
