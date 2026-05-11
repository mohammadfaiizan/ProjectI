# Load Balancing and Rate Limiting — Interview Q&A

> 20 questions | Easy: Q1–Q7 | Medium: Q8–Q15 | Hard: Q16–Q20

---

## EASY (Q1–Q7)

---

### Q1. What are the main load balancing algorithms, and when do you use each?

**Answer:**

Load balancing algorithms determine how incoming requests are distributed across backend servers. The right choice depends on workload characteristics.

**Round Robin:**
```
Requests:  R1  R2  R3  R4  R5  R6
Servers:   S1  S2  S3  S1  S2  S3

Pros:  Simple, fair distribution, no state needed
Cons:  Ignores server load or capacity differences
Use:   Homogeneous servers with similar request costs
```

**Weighted Round Robin:**
```
Servers: S1 (weight=3), S2 (weight=1)
Distribution: S1 S1 S1 S2 S1 S1 S1 S2 ...

Use: Servers have different capacities (S1 is 3× stronger than S2)
```

**Least Connections:**
```
S1: 10 active connections
S2: 5 active connections   ← next request goes here
S3: 8 active connections

Pros:  Adapts to varying request durations; natural load equalization
Cons:  Requires tracking connection counts (stateful LB)
Use:   Long-lived connections (WebSockets, file uploads, DB connections)
```

**IP Hash (Sticky by client):**
```
hash(client_IP) % num_servers → server index

Client 10.0.0.5 always → S2
Client 10.0.0.7 always → S1

Pros:  Client always hits same server (session affinity without cookies)
Cons:  Uneven distribution if some IPs generate more traffic; fails on NAT
Use:   Sessions stored on server (legacy apps without shared session store)
```

**Least Response Time:**
```
Monitor average response time per server
Route next request to server with lowest response time

Pros:  Dynamically adapts to server performance
Cons:  More complex to implement; slight overhead in tracking
Use:   Heterogeneous servers with variable processing speed
```

**Random:**
```
Randomly select server for each request
Surprisingly effective at large scale (law of large numbers)
Use: Simple implementations; large number of backends
```

**URL/Content Hash:**
```
hash(request_URL) % num_servers → server

Same URL always goes to same server
Pros: Cache efficiency (server can cache responses for its URLs)
Use: Read-heavy cache-tier load balancing
```

**Summary table:**
| Algorithm | State | Best For |
|---|---|---|
| Round Robin | None | Uniform requests, homogeneous servers |
| Weighted RR | Weights only | Mixed-capacity servers |
| Least Connections | Connection counts | Long-lived connections |
| IP Hash | Hash table | Session affinity (legacy) |
| Least Response Time | Latency stats | Performance-sensitive routing |
| URL Hash | Hash table | Cache-efficient routing |

---

### Q2. What is consistent hashing, and why is it used for load balancing?

**Answer:**

Consistent hashing is a technique that minimizes the number of remappings (and thus cache invalidations or session disruptions) when servers are added or removed.

**Problem with naive modulo hashing:**
```
3 servers: hash(key) % 3

key "user123" → hash = 567 → 567 % 3 = 0 → Server 0
key "order99" → hash = 891 → 891 % 3 = 0 → Server 0

Add 4th server:  hash(key) % 4
key "user123" → 567 % 4 = 3 → Server 3  ← MOVED!
key "order99" → 891 % 4 = 3 → Server 3  ← MOVED!

Nearly ALL keys remapped → massive cache invalidation or session disruption
```

**Consistent hashing solution:**
```
Hash ring (0 to 2^32 - 1):

         0
      /     \
   S1         S2
  /               \
S4                 S3
  \               /
      \         /
        2^32 - 1

Algorithm:
1. Map each server to a point on the ring using hash(serverID)
2. Map each key to the ring using hash(key)
3. Walk clockwise from key's position to find its server

Effect:
  Remove a server: Only keys between prev_server and removed_server remapped
  Add a server:    Only keys between prev_server and new_server remapped
  Average remapping: n/N keys (n=keys on removed server, N=total servers)
  vs modulo: almost ALL keys remapped
```

**Virtual nodes (vnodes) for uniform distribution:**
```
Problem: Small number of servers → uneven spacing on ring
Solution: Each server gets K virtual nodes spread around the ring

Server 1: hash("S1-1"), hash("S1-2"), ..., hash("S1-150")
Server 2: hash("S2-1"), hash("S2-2"), ..., hash("S2-150")

Result: ~150 points per server → uniform distribution even with 3-5 servers
        Weighted capacity: S1 gets 200 vnodes (stronger), S2 gets 100
```

**Use cases:**
- Distributed caches (Redis Cluster, Memcached consistent hashing)
- Session routing in stateful load balancers
- Data partitioning in Cassandra, DynamoDB
- CDN PoP selection

---

### Q3. What are sticky sessions, and what are their trade-offs?

**Answer:**

Sticky sessions (session affinity) ensure a client's requests always reach the same backend server, typically using a cookie or IP hash.

**Implementation methods:**

**Cookie-based stickiness (most common):**
```
First request:
  Client ──> LB ──> Server A
  LB sets cookie: SERVERID=A; Path=/; HttpOnly
  
Subsequent requests:
  Client sends cookie SERVERID=A
  LB reads cookie → routes to Server A (bypasses balancing)
```

**IP-hash stickiness:**
```
hash(client_IP) % N → always same server
Less reliable: multiple users behind same NAT IP → all hit same server
```

**Trade-offs:**

| Trade-off | Pros | Cons |
|---|---|---|
| Session state | No shared session store needed (saves cost) | Sessions lost if server goes down |
| Load distribution | Slightly simpler routing | Uneven load if some users are more active |
| Failover | None needed for session data | Must re-login or re-process on server failure |
| Scalability | Easy initial setup | Adding/removing servers disrupts routed sessions |
| Cache efficiency | Application cache stays warm | Cannot scale by adding servers freely |

**When sticky sessions are acceptable:**
- Legacy applications that store session in server memory and cannot be refactored.
- Short-lived sessions (shopping cart, < 30 min) where disruption is minor.
- Read-heavy workloads where each server caches hot data.

**Better alternative:**
```
Externalize session state:
  Client → Any Server → Redis/Memcached (shared session store)
  
  Server reads/writes session from shared store → ANY server can handle any request
  Enables true stateless scaling → preferred for cloud-native apps
```

---

### Q4. What is the difference between active and passive health checks?

**Answer:**

Health checks determine whether a backend server should receive traffic. There are two strategies for detecting unhealthy servers.

**Active health checks:**
```
LB proactively sends periodic test requests to each backend:

Every 5 seconds:
  LB ──GET /health──> Server 1 → 200 OK ✓ (healthy)
  LB ──GET /health──> Server 2 → 503    ✗ (unhealthy, remove from pool)
  LB ──GET /health──> Server 3 → timeout ✗ (unhealthy, remove from pool)

Threshold: unhealthy after 3 consecutive failures
Recovery:  healthy after 2 consecutive successes
```

**Passive health checks (circuit breaker approach):**
```
LB monitors REAL traffic responses:

Client request → Server 2 → 500 Internal Server Error (failure #1)
Client request → Server 2 → 500 Internal Server Error (failure #2)
Client request → Server 2 → 500 Internal Server Error (failure #3)
  → Server 2 marked unhealthy, removed from pool

No artificial probe requests — uses production traffic as signal
```

**Comparison:**

| Property | Active | Passive |
|---|---|---|
| Detection speed | Proactive (before client impact) | Reactive (after real failures) |
| Resource overhead | Extra health-check traffic | No extra traffic |
| Accuracy | Tests specific endpoint (may not reflect all failures) | Tests real production path |
| Non-traffic servers | Can detect failures during idle periods | Cannot detect failures with no traffic |
| Database health | Can check actual DB connectivity in /health | Only detects via production query failures |

**Best practice:** Use both together.
- Active checks catch failures proactively and recover idle servers.
- Passive checks catch partial failures (slow responses, specific endpoint errors) that active checks might miss.

**Health check endpoint design:**
```python
# GET /health — deep health check
{
  "status": "healthy",
  "checks": {
    "database": "ok",        # Can query DB?
    "cache": "ok",           # Can reach Redis?
    "downstream_api": "ok"   # Critical dependencies healthy?
  },
  "latency_ms": 12
}

HTTP 200 → healthy
HTTP 503 → unhealthy (LB removes from pool)
```

---

### Q5. How do you prevent a load balancer from being a single point of failure?

**Answer:**

A load balancer that itself fails causes complete system outage — it is a critical SPOF. Multiple strategies exist to eliminate this.

**Active-Passive LB pair (most common):**
```
                      ┌─────────────────┐
                      │   Virtual IP    │
                      │  (Floating IP)  │
                      │  203.0.113.1    │
                      └────────┬────────┘
                               │
              ┌────────────────┴────────────────┐
              │                                 │
     [LB Primary] ◄── VRRP Heartbeat ──► [LB Secondary]
      (ACTIVE)                               (STANDBY)
       handles traffic                     ready to take over

On Primary failure:
  VRRP detects missed heartbeat (~2 seconds)
  Secondary claims the Virtual IP (gratuitous ARP)
  Traffic automatically flows to Secondary
  Failover time: 2-5 seconds
```

**Active-Active LB pair:**
```
DNS → LB1 IP and LB2 IP (DNS round-robin between two LBs)
Both LBs are active and handling traffic
If one fails, DNS TTL expires and clients stop using failed LB IP
Failover time: DNS TTL (typically 30-60 seconds)

Better: Use anycast — both LBs announce same IP via BGP
        Routing protocol automatically avoids failed LB
```

**AWS approach — managed LBs:**
```
AWS ALB/NLB are multi-AZ by design:
  LB nodes in each Availability Zone
  If one AZ fails, DNS stops resolving to that AZ's LB node
  No user action required

DNS name: my-lb-1234567890.us-east-1.elb.amazonaws.com
  → resolves to multiple IPs across multiple AZs
```

**DNS-level failover:**
```
Route 53 health checks on LB:
  Primary:   203.0.113.1 (LB primary) — health check passes → serves traffic
  Secondary: 198.51.100.1 (backup LB)  — only used if primary fails

TTL: 60s (fast failover)
```

**Cloud-native pattern:**
```
User → Route 53 (DNS failover) → AWS ALB (multi-AZ) → App servers
         ↑                            ↑
    Eliminates DNS SPOF         Eliminates LB SPOF
```

---

### Q6. What is the difference between rate limiting and throttling?

**Answer:**

These terms are related but have a precise distinction.

**Rate Limiting:** Enforces a hard cap on the number of requests a client can make within a time window. Requests exceeding the limit are rejected (HTTP 429 Too Many Requests).

**Throttling:** Slows down requests rather than rejecting them. When a client exceeds the threshold, requests are queued, delayed, or degraded — but not necessarily rejected.

```
RATE LIMITING (Hard reject):
  Client sends 1,001st request in 1-minute window
  Server returns: HTTP 429 Too Many Requests
  Client must wait until window resets
  No queueing

THROTTLING (Slow down):
  Client sends many requests
  Server processes them, but delays responses or reduces priority
  Client gets responses, just slower
  Good for: graceful degradation of service quality
```

**Comparison table:**

| Property | Rate Limiting | Throttling |
|---|---|---|
| Behavior on excess | Reject (429) | Delay / downgrade |
| Client experience | Hard error | Slow response |
| Protects against | Deliberate abuse, DDoS | Unintentional overload |
| Predictability | Predictable (client knows limit) | Unpredictable delay |
| Resource protection | Yes — drops excess load | Partial — still processes (just slower) |
| Use cases | API monetization, DDoS protection | Service quality management |

**Combined approach:**
```
Primary:   Rate limit at 1000 req/min per API key
Secondary: Throttle (queue + slow) at 800 req/min per API key

→ At 800-1000 req/min: responses slow down (warning zone)
→ Above 1000 req/min: hard reject with 429
```

---

### Q7. What is the token bucket algorithm for rate limiting?

**Answer:**

Token bucket is one of the most widely used rate limiting algorithms. It allows burst traffic up to a limit while enforcing an average rate.

**Mechanism:**
```
BUCKET STATE:
  capacity = 100 tokens
  fill_rate = 10 tokens/second
  current_tokens = 75

ON EACH REQUEST:
  1. Calculate tokens to add since last check:
     new_tokens = (current_time - last_check_time) × fill_rate
     current_tokens = min(capacity, current_tokens + new_tokens)
  
  2. Check if request can proceed:
     if current_tokens >= cost_of_request (usually 1):
         current_tokens -= cost_of_request
         ALLOW request
     else:
         REJECT request (429)

EXAMPLE SCENARIO:
  t=0:  tokens=100, client sends 100 requests in 1ms → all allowed
  t=0:  tokens=0,   client sends 1 more request → REJECTED
  t=1:  tokens=10  (10 filled), client sends 10 → all allowed
  t=2:  tokens=0,   burst again → 10 allowed, rest rejected
```

**Visual:**
```
Token Bucket:
  ┌──────────────────────────────┐
  │ Tokens added at fixed rate   │ ← Fill rate: 10/sec
  │  ○ ○ ○ ○ ○ ○ ○ ○ ○ ○        │
  │  ○ ○ ○ ○ ○ ○ ○ ○ ○ ○        │ ← Capacity: 100
  │  ...                         │
  └──────────────┬───────────────┘
                 │ Tokens consumed per request
                 ▼
         [Requests flow through]
```

**Key properties:**
| Property | Behavior |
|---|---|
| Burst allowance | Yes — entire bucket capacity can be consumed instantly |
| Average rate enforcement | Yes — long-term rate limited to fill rate |
| Bursty traffic handling | Excellent — clients can burst up to capacity |
| Memory per client | Small (just: token_count, last_check_time) |
| Implementation | Lazy token addition (calculate on each request) |

**Python implementation sketch:**
```python
class TokenBucket:
    def __init__(self, capacity, fill_rate):
        self.capacity = capacity        # max tokens
        self.fill_rate = fill_rate      # tokens per second
        self.tokens = capacity          # start full
        self.last_check = time.time()
    
    def allow(self, cost=1):
        now = time.time()
        elapsed = now - self.last_check
        self.tokens = min(
            self.capacity,
            self.tokens + elapsed * self.fill_rate
        )
        self.last_check = now
        
        if self.tokens >= cost:
            self.tokens -= cost
            return True  # Allow
        return False  # Reject (429)
```

---

## MEDIUM (Q8–Q15)

---

### Q8. How does the leaky bucket algorithm differ from token bucket?

**Answer:**

Both algorithms control request flow, but they handle bursts differently.

**Leaky Bucket:**
```
Requests enter a fixed-size queue (bucket)
Requests exit the bucket at a CONSTANT, fixed rate (leak rate)

  Requests arrive:    ─────────────────────────────────>
                      │  burst of 50 requests            │
                      ▼                                  │
              ┌────────────────────┐                     │
              │   Queue (bucket)   │  capacity=100       │
              │  [req][req][req]   │  ← if full → DROP   │
              └─────────┬──────────┘                     │
                        │ fixed rate: 10 req/sec         │
                        ▼                                │
              [Processing (exact rate)]

RESULT: Output is perfectly smooth — exactly 10 req/sec regardless of input
        Burst absorbed by queue; excess dropped if queue full
```

**Token Bucket vs Leaky Bucket:**

```
BURST BEHAVIOR:

Token Bucket:
  t=0: Client sends 100 requests (full bucket)
  t=0: ALL 100 processed immediately (burst allowed)
  t=0: Bucket empty; next request rejected until tokens refill
  
  Output rate: Bursty (up to capacity at once)

Leaky Bucket:
  t=0: Client sends 100 requests
  t=0: 100 queued (or 90 queued + 10 dropped if capacity=90)
  t=1 to t=10: Processed at 10/sec
  
  Output rate: Perfectly smooth (10/sec)
```

**Comparison:**

| Property | Token Bucket | Leaky Bucket |
|---|---|---|
| Burst handling | Allows controlled bursts | Smooths all bursts into constant rate |
| Output rate | Variable (up to fill_rate avg) | Constant (leak rate) |
| Dropped requests | Dropped if bucket empty at request time | Dropped if queue is full |
| Use case | APIs with bursty but fair usage | Network traffic shaping, QoS |
| Memory usage | O(1) per client | O(queue_size) per client |

**Practical choice:**
- **Token bucket:** API rate limiting (clients can burst on occasional heavy usage).
- **Leaky bucket:** Network packet shaping (smooth output prevents downstream overload).

---

### Q9. Compare sliding window vs fixed window rate limiting.

**Answer:**

The window strategy determines how time is divided when counting requests.

**Fixed Window:**
```
Window = 1 minute (reset at :00, :01, :02, ...)
Limit  = 100 requests/minute

PROBLEM — Boundary exploit:
  User sends 100 requests at 12:00:55  → allowed (window 12:00 - 12:01)
  User sends 100 requests at 12:01:05  → allowed (window 12:01 - 12:02)
  
  In the 10-second span (12:00:55 to 12:01:05):
  200 requests processed! — 2× the limit in 10 seconds
```

**Sliding Window (Log-based):**
```
Maintain log of request timestamps per user
On each request:
  1. Remove timestamps older than window_size (1 min) from log
  2. Count remaining entries
  3. If count < limit: allow, add current timestamp
  4. Else: reject

t=12:01:55 request:
  Log: [12:01:05, 12:01:15, 12:01:25, ..., 12:01:50] (last 60 seconds)
  Count: 50 → allow, add 12:01:55

RESULT: Exactly 100 requests in any 60-second window — no boundary abuse
```

**Sliding Window (Counter-based — memory efficient):**
```
Keep two windows: current + previous
Interpolate: current_rate = prev_window × ((60s - elapsed)/60s) + current_window

Example at 45 seconds into current window:
  prev_window_count = 80 requests
  curr_window_count = 30 requests
  weight_prev = (60-45)/60 = 0.25
  
  estimated_rate = 80 × 0.25 + 30 = 50 requests
  
  If 50 < limit (100) → allow
  
Memory: O(1) per user (just two counters) vs O(requests) for log-based
Accuracy: ~99% accurate with slight smoothing approximation
```

**Comparison:**

| Property | Fixed Window | Sliding Window Log | Sliding Window Counter |
|---|---|---|---|
| Accuracy | Low (boundary exploit) | Perfect | ~99% |
| Memory | O(1) | O(requests in window) | O(1) |
| Complexity | Very simple | Medium | Medium |
| Race conditions | Yes (need atomic ops) | Easier to make atomic | Yes (atomic incr) |
| Recommended? | Not for production | Small scale | Production choice |

**Implementation with Redis (sliding window counter):**
```
MULTI
ZREMRANGEBYSCORE user:123 0 (NOW - 60000)   # remove old entries
ZADD user:123 NOW NOW                         # add current timestamp
ZCARD user:123                                # count entries
EXPIRE user:123 60                            # cleanup TTL
EXEC
```

---

### Q10. Where should rate limiting be implemented — client, API gateway, or service level?

**Answer:**

Rate limiting can be applied at multiple layers. The right placement depends on what you are protecting and how precise you need the enforcement.

**Layer comparison:**

```
Internet → [Client SDK] → [API Gateway] → [Microservice] → [DB]
               L1              L2              L3
```

**Client-side rate limiting (L1):**
```
SDK/mobile app throttles itself before sending to server.

PROS:
  + Reduces network traffic to server
  + Improves UX (client knows immediately, no round-trip needed)
  + Good for SDK throttling to prevent API abuse by developers

CONS:
  - Trivially bypassed (attacker ignores SDK limits)
  - Cannot protect against server-side abuse
  
USE: Developer SDK experience improvement only; never as sole protection
```

**API Gateway rate limiting (L2):**
```
Central enforcement point — all traffic passes through.

PROS:
  + Single enforcement point — no per-service implementation
  + Can rate limit by API key, user ID, IP address
  + Fine-grained control (per endpoint, per method, per plan)
  + Protects ALL downstream services simultaneously

CONS:
  - Gateway can become bottleneck if not scaled
  - Cross-service rate limiting is coarse
  - Cannot enforce per-business-rule limits (e.g., "100 order attempts/day")

USE: Primary protection layer. 90% of rate limiting should happen here.
```

**Service-level rate limiting (L3):**
```
Each microservice enforces its own limits.

PROS:
  + Business logic aware (e.g., "User can make 3 failed payment attempts/hour")
  + Protects against internal callers (not just external via gateway)
  + Defense in depth — gateway may be bypassed internally

CONS:
  - Duplicated logic across services
  - Must distribute state (Redis) for accurate counts
  - Higher latency per request (Redis call on hot path)

USE: High-value operations only (payments, account changes, login attempts)
```

**Recommended architecture:**
```
Layer 1 (API Gateway): Rate limit by API key / IP / user
  → 1000 requests/minute per API key
  → 10,000 requests/minute per IP

Layer 2 (Service): Rate limit by business rule
  → 5 failed login attempts / 15 minutes (account lockout)
  → 3 order cancellations / day
  → 10 password reset emails / hour per user
```

---

### Q11. How does distributed rate limiting work with Redis?

**Answer:**

Single-server rate limiting is simple. Distributed rate limiting coordinates counters across multiple API gateway nodes using a shared store.

**Problem without distributed coordination:**
```
10 API gateway nodes
Limit: 100 requests/minute per user
Each node tracks its own counter in memory

User sends 100 requests equally across all nodes:
  Node 1: 10 requests (counter=10, allows all)
  Node 2: 10 requests (counter=10, allows all)
  ...
  Node 10: 10 requests (counter=10, allows all)
  
  Total: 100 requests — limit apparently not violated
  
  But if user targets ONE node: that node rejects at 100
  → Inconsistent enforcement across nodes
```

**Redis-based distributed rate limiting:**

**Simple counter approach:**
```python
def is_allowed(user_id: str, limit: int, window_secs: int) -> bool:
    key = f"rl:{user_id}:{int(time.time() // window_secs)}"
    
    pipe = redis.pipeline()
    pipe.incr(key)
    pipe.expire(key, window_secs * 2)  # cleanup
    count, _ = pipe.execute()
    
    return count <= limit

# All nodes share same Redis key → accurate global counter
```

**Lua script for atomic sliding window:**
```lua
-- Atomic sliding window in Redis Lua (executes as single transaction)
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])

-- Remove entries outside window
redis.call('ZREMRANGEBYSCORE', key, 0, now - window)

-- Count current entries
local count = redis.call('ZCARD', key)

if count < limit then
    -- Add current timestamp
    redis.call('ZADD', key, now, now .. math.random())
    redis.call('EXPIRE', key, window / 1000)
    return 1  -- Allow
end

return 0  -- Reject
```

**Redis cluster for high availability:**
```
Rate limiting Redis:
  ├── Redis Primary (writes + reads for rate limit checks)
  └── Redis Replica (failover)

With Redis Cluster (sharding):
  hash(user_id) → determines which Redis shard holds the counter
  Pros: Horizontal scaling of rate limit state
  Cons: Cross-slot transactions not supported (each user's key on one shard, which is fine)
```

**Approximate counting with Lua + expiry (production pattern):**
```
Fixed window counter (simplest production approach):
  Key: "rl:{user_id}:{minute_bucket}"
  e.g., "rl:user123:27342" (minute 27342 since epoch)
  
  INCR key              → atomic increment
  EXPIRE key 120        → auto-cleanup
  
  Compare result to limit → allow or reject
  
  Performance: < 1ms per check on Redis
  Accuracy: Fixed window (see Q9 boundary issue)
  For production: Add sliding window correction if precision needed
```

---

### Q12. What is the circuit breaker pattern, and what are its states?

**Answer:**

The circuit breaker pattern prevents a failing service from being continuously hammered with requests, allowing it time to recover. Named after an electrical circuit breaker.

**Problem without circuit breaker:**
```
Service A calls Service B
Service B is down/slow (taking 30s to timeout)

Without circuit breaker:
  - All requests to A block for 30 seconds waiting for B
  - Thread pool exhausted → Service A also becomes unavailable
  - Cascade failure: A → B failure cascades to all of A's callers
```

**Circuit breaker states:**

```
                  failure_threshold exceeded
    ┌─────────────────────────────────────────────────┐
    │                                                 ▼
[CLOSED]                                          [OPEN]
  │                                                   │
  │ Normal operation                                  │ Fast fail
  │ Requests pass through                             │ Return fallback immediately
  │ Track success/failure                             │ No calls to downstream
  │                                                   │
  │                    ◄─ timeout expires ────────────┤
  │                                                   │
  │              [HALF-OPEN]                          │
  │                   │                               │
  │                   │ Allow ONE probe request        │
  │                   │                               │
  │            success│          failure              │
  └─────◄─────────────┘            └─────────────────►┘
     Closes circuit                  Re-opens circuit
```

**State transitions:**

| State | Behavior | Transition |
|---|---|---|
| CLOSED | All requests pass through; count failures | → OPEN if failures ≥ threshold (e.g., 5 failures in 60s) |
| OPEN | All requests fast-fail; return fallback | → HALF-OPEN after timeout (e.g., 30 seconds) |
| HALF-OPEN | One probe request allowed | → CLOSED on success; → OPEN on failure |

**Implementation sketch:**
```python
class CircuitBreaker:
    def __init__(self, threshold=5, timeout=30, half_open_max=1):
        self.state = "CLOSED"
        self.failure_count = 0
        self.threshold = threshold
        self.timeout = timeout
        self.last_failure_time = None
    
    def call(self, func, fallback=None):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                return fallback() if fallback else raise CircuitOpenError()
        
        try:
            result = func()
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.threshold:
            self.state = "OPEN"
```

**Metrics to trigger circuit opening:**
- Error rate > threshold (e.g., > 50% errors in 60s).
- Response time > threshold (e.g., > 3 seconds average).
- Timeout rate > threshold (e.g., > 20% timeouts).

---

### Q13. What is the bulkhead pattern for failure isolation?

**Answer:**

The bulkhead pattern — named after watertight compartments in ships — isolates different parts of a system so one failure does not exhaust shared resources and cause a cascade failure.

**Problem without bulkheads:**
```
Service A has ONE thread pool (100 threads) for all operations:
  ├── Calls Service B (payments)
  ├── Calls Service C (recommendations)
  └── Calls Service D (inventory)

Service B becomes slow (each call takes 10 seconds):
  First 10 slow calls: 10 threads blocked waiting on B
  Next 10 more calls:  20 threads blocked
  After 100 requests:  ALL 100 threads blocked on B
  
  New requests for recommendations or inventory? → No threads available → Error
  Service A is completely down because of Service B!
```

**Bulkhead solution:**
```
Separate thread pools per downstream dependency:

Service A:
  ├── Payment Thread Pool (20 threads)
  │     → if payment service slow, ONLY these 20 threads affected
  ├── Recommendation Thread Pool (30 threads)
  │     → recommendations still work even if payments are slow
  └── Inventory Thread Pool (20 threads)
        → inventory still works

Total: 70 threads — less than original 100 but failures are ISOLATED
```

**Types of bulkheads:**

**Thread pool bulkhead:**
```
Separate thread pool per downstream service
Request to B goes to B's pool; if B's pool full → reject only B requests
```

**Semaphore bulkhead:**
```
Maximum concurrent calls per dependency
Lighter weight than thread pools (no new threads)
Semaphore count = max concurrent calls allowed
```

**Connection pool bulkhead:**
```
Separate database connection pools per use case:
  OLTP pool:    20 connections for user-facing queries
  Reporting pool: 5 connections for analytics queries
  Background jobs pool: 5 connections
  
  Heavy analytics query does not starve user-facing queries
```

**Combined circuit breaker + bulkhead:**
```
Request → [Bulkhead: max 20 concurrent] → [Circuit Breaker: 5 failures → open]
                                        → [Downstream Service]

Result: 
  - Bulkhead limits blast radius (max 20 threads consumed even if slow)
  - Circuit breaker stops calls entirely once failure threshold hit
  - Rest of system unaffected
```

**Hystrix/Resilience4j implement both patterns together.**

---

### Q14. What is backpressure and how do you handle it in distributed systems?

**Answer:**

Backpressure is a mechanism for a downstream component to signal to an upstream component to slow down its rate of message production, preventing the downstream from being overwhelmed.

**The problem:**
```
Producer ──────────────────────────────── Consumer
10,000 msg/sec ──→  [Queue growing...]  ←── 1,000 msg/sec

Without backpressure:
  Queue grows indefinitely → memory exhaustion → crash
  Or: consumer drops messages → data loss
```

**Backpressure strategies:**

**1. Blocking (synchronous):**
```
Producer blocks when queue is full
Producer ──send()──> [Full Queue] → BLOCKS until space available

Pros: No data loss; simple
Cons: Producer blocked → may cause its own upstream issues
Use: Simple producer-consumer within a single process
```

**2. Drop / Load shedding:**
```
Consumer accepts up to capacity; drops excess

if queue.size() >= MAX_QUEUE:
    drop_message()    # or reject with 429/503
    increment_dropped_counter()

Pros: Simple, fast
Cons: Data loss — only acceptable for non-critical data (metrics, logs)
Use: Analytics, logging, telemetry
```

**3. Signal back (reactive streams):**
```
Consumer tells producer its current capacity:
  Consumer: "I can handle 500 more messages"
  Producer: Slows production to 500/sec

Reactive Streams (Java): request(n) method in Subscriber
gRPC flow control: Window size propagated upstream
TCP: rwnd (receive window) is backpressure to TCP sender
```

**4. Queue-based buffering with consumer scaling:**
```
[Producer] → [Message Queue (Kafka/SQS)] → [Consumer Pool]
                      ↑
                 Buffer absorbs spikes

Auto-scale consumers when queue depth > threshold:
  queue.depth > 10,000 → scale consumer instances from 5 → 10
  
  Kafka consumer lag metric triggers auto-scaling
  SQS ApproximateNumberOfMessages metric triggers scaling
```

**Backpressure in HTTP:**
```
HTTP 429 Too Many Requests → client rate limiter (client-side backpressure)
HTTP 503 Service Unavailable → server overloaded, client should back off
Retry-After header: "Retry-After: 30" → tells client when to retry

Client exponential backoff is client-side backpressure response
```

---

### Q15. Compare Nginx, HAProxy, and AWS ALB as load balancers.

**Answer:**

Each has different strengths, typical deployment contexts, and feature profiles.

**Nginx:**
```
Origin: Web server that also functions as reverse proxy/LB
Written: C
Architecture: Event-driven, non-blocking (like Node.js model)
Primary use: Web server + static file serving + reverse proxy + LB

Strengths:
  - Excellent at serving static content (files, images)
  - Full web server (PHP, CGI, serve files directly)
  - Wide module ecosystem (Lua scripting, ModSecurity WAF)
  - SSL/TLS termination
  - HTTP caching (proxy_cache)
  - Very low memory footprint

Weaknesses:
  - L4 load balancing is limited (stream module exists but less mature)
  - Less granular health checking than HAProxy
  - Commercial features need Nginx Plus ($$$)

Config example:
  upstream backend {
      least_conn;
      server 10.0.0.1:8080 weight=3;
      server 10.0.0.2:8080;
      server 10.0.0.3:8080 backup;
  }
```

**HAProxy:**
```
Origin: Dedicated high-availability proxy and load balancer
Written: C
Architecture: Event-driven, single-process
Primary use: L4 and L7 load balancing, TCP proxying

Strengths:
  - Battle-tested for extreme reliability (Twitter, Reddit, GitHub, GitHub)
  - Rich L4 and L7 load balancing (more algorithms than Nginx)
  - Advanced health checking (TCP, HTTP, custom scripts)
  - Detailed statistics dashboard (built-in)
  - ACL-based routing rules are very powerful
  - Active-passive HA with VRRP keepalived pairing
  - Excellent TCP/SSL passthrough performance

Weaknesses:
  - Not a web server (cannot serve static files)
  - Steeper learning curve for complex configs
  - No built-in caching

Config example:
  backend web_servers
      balance leastconn
      option httpchk GET /health
      server web1 10.0.0.1:8080 check
      server web2 10.0.0.2:8080 check
```

**AWS ALB (Application Load Balancer):**
```
Origin: Managed cloud service (AWS)
Architecture: Fully managed, horizontally scaled by AWS
Primary use: Cloud-native L7 load balancing on AWS

Strengths:
  - Zero operational overhead (no servers to manage)
  - Automatic scaling (handles sudden traffic spikes)
  - Native AWS integrations (ACM for SSL, WAF, Cognito, ECS, EKS)
  - Content-based routing (path, host, headers, query params)
  - HTTP/2 and WebSocket support
  - Fixed connection draining (graceful shutdown)
  - Blue/green deployments via weighted target groups

Weaknesses:
  - AWS lock-in
  - More expensive than self-managed for sustained high traffic
  - Less control over low-level tuning
  - Latency slightly higher than self-managed (few ms)
```

**Decision table:**

| Scenario | Best Choice |
|---|---|
| On-prem or bare-metal deployment | HAProxy |
| Web server + LB combo (simpler setup) | Nginx |
| AWS cloud infrastructure | ALB |
| Extreme TCP performance tuning | HAProxy |
| Low operational overhead | ALB |
| Static file serving + LB | Nginx |
| Service mesh data plane | Envoy (not listed but worth mentioning) |

---

## HARD (Q16–Q20)

---

### Q16. Design a distributed rate limiter that handles 1 million requests per second across 100 API gateway nodes.

**Answer:**

At 1M RPS across 100 nodes, each node handles 10K RPS. The challenge is coordinating rate limits accurately without Redis becoming a bottleneck.

**Requirement analysis:**
```
Scale:        1M RPS total, 100 gateway nodes
Granularity:  Per user (10M users), per endpoint
Accuracy:     ~99% (slight approximation acceptable)
Latency SLA:  Rate limit check must add < 1ms
Availability: Rate limiter failure should not block all traffic
```

**Architecture options:**

**Option A: Centralized Redis (baseline):**
```
All 100 nodes → Redis Cluster (10 shards)

Redis Cluster:
  shard(user_id) → one of 10 Redis nodes
  Each shard handles: 1M / 10 = 100K ops/sec
  Redis single-node limit: ~1M simple ops/sec → 100K is fine

Latency: 0.5-1ms per check (same DC)
Accuracy: Perfect

Problem: Redis becomes bottleneck AND SPOF if latency spikes
         100K ops/sec per shard is workable but leaves little headroom
```

**Option B: Local + Sync (recommended for 1M RPS):**
```
Each gateway node maintains local counter (in-memory)
Periodically syncs with Redis (every 100ms)

Algorithm:
  1. Each node gets a "token allocation" from Redis
     e.g., User limit = 1000/min, 100 nodes
     → Each node gets allocation of 10 tokens (1000/100)
  
  2. Node decrements local counter (no Redis call per request)
  
  3. Every 100ms: sync with Redis
     - Report how many tokens consumed locally
     - Get new allocation based on remaining global budget
  
  Redis ops: 10M users × 10 syncs/sec = 100M ops/sec? Too high.
  
  Optimization: Only sync active users
  Active users at any second: 1M RPS with ~1000 unique users = 1K syncs/sec
  → very manageable
```

**Architecture diagram:**
```
[Gateway Node 1]                    [Redis Cluster]
├── Local Token Cache               ├── Shard 1: users A-M
│   user123: 45 tokens              ├── Shard 2: users N-Z
│   user456: 12 tokens              └── (3 replicas each)
│   ...                             
├── Sync Thread (every 100ms)
│   MULTI
│   INCRBY user123:consumed 55      # report consumption
│   GET user123:global_remaining    # get remaining budget
│   EXEC
│   → Recalculate local allocation
│
└── Request Handler (no Redis)
    check local_cache[user_id] > 0
    decrement → allow or reject

[Gateway Node 2] ... [Gateway Node 100] (same pattern)
```

**Failure handling:**
```
Redis unavailable:
  Option 1: Fail open (allow all requests) — recommended if availability > accuracy
  Option 2: Fall back to local-only limiting with reduced limits
  Option 3: Fail closed (reject all) — only if security is paramount

Implementation: Circuit breaker on Redis connection
  CLOSED → use Redis for accurate limiting
  OPEN   → use local counter with 80% of allocation (conservative)
```

**Redis Lua script for atomic allocation:**
```lua
-- Called by each node every 100ms
local key = KEYS[1]          -- user rate limit key
local consumed = ARGV[1]     -- tokens consumed since last sync
local window = ARGV[2]       -- window in seconds
local limit = ARGV[3]        -- global limit

-- Add consumed to global counter
local total = redis.call('INCRBY', key, consumed)
redis.call('EXPIRE', key, window)

-- Return remaining allocation
local remaining = limit - total
if remaining < 0 then remaining = 0 end
return remaining
```

---

### Q17. How does GSLB (Global Server Load Balancing) differ from local load balancing?

**Answer:**

Local load balancing operates within a single data center or region. GSLB operates globally, routing users to the most appropriate regional data center.

**Local LB:**
```
Data Center (us-east-1):
  User → [L7 ALB] → Server 1
                  → Server 2
                  → Server 3
  
  Scope: Distributes load within one region
  Decisions based on: server health, connection count, algorithm
  Latency scope: within one DC (<1ms routing decision)
```

**GSLB:**
```
Global:
  User in Tokyo  → DNS → GSLB → Route to ap-northeast-1
  User in London → DNS → GSLB → Route to eu-west-1
  User in NYC    → DNS → GSLB → Route to us-east-1
  
  ap-northeast-1 goes down → GSLB detects, reroutes Tokyo users to us-west-2
  
  Scope: Routes between multiple regions
  Decisions based on: geographic proximity, region health, latency, load, policy
```

**GSLB routing policies:**

| Policy | Description | Use Case |
|---|---|---|
| Geographic | Route to nearest region by client location | General latency reduction |
| Latency-based | Route to region with lowest measured latency | Precise performance optimization |
| Weighted | 70% to us-east, 30% to us-west | Gradual traffic migration, A/B testing |
| Failover | Primary region; failover to secondary on failure | Disaster recovery |
| Geofencing | EU traffic must stay in EU (GDPR) | Compliance requirements |

**DNS-based GSLB (AWS Route 53):**
```
Route 53 health checks each regional endpoint:
  GET https://api.us-east-1.example.com/health → 200 OK
  GET https://api.eu-west-1.example.com/health → 200 OK
  GET https://api.ap-ne-1.example.com/health   → 503 FAIL

Route 53 removes failing endpoint from DNS responses
TTL: 30-60 seconds for fast failover

Latency routing:
  Measures latency from Route 53 health checkers to each region
  Route user to lowest-latency region
```

**Anycast-based GSLB:**
```
Same IP announced from multiple regions via BGP
BGP routing automatically selects nearest region
No DNS TTL delay — routing at network level
Used by: Cloudflare, Fastly CDN

vs DNS GSLB:
  Anycast: Instant routing (no TTL), no DNS manipulation needed
  DNS GSLB: TTL-limited (30-60s failover), more flexible policies
```

**Architecture combining both:**
```
User → Anycast IP → Nearest CDN PoP (static assets served locally)
                  → GSLB DNS → Nearest API region (dynamic requests)
                  → Regional LB → Servers in region
```

---

### Q18. Explain SSL termination at the load balancer — pros, cons, and the security implications.

**Answer:**

SSL/TLS termination is the process of decrypting HTTPS traffic at the load balancer, then forwarding unencrypted (HTTP) traffic to backend servers.

**SSL termination at LB:**
```
Client ──[HTTPS encrypted]──> [Load Balancer] ──[HTTP plaintext]──> Backend
                               (decrypts here)                       Server

Flow:
  1. Client establishes TLS session with LB's certificate
  2. LB decrypts request payload
  3. LB forwards decrypted HTTP request to backend
  4. Backend responds with HTTP
  5. LB encrypts response and sends to client
```

**Pros:**

| Benefit | Details |
|---|---|
| CPU offload | TLS crypto is expensive; offloaded from many app servers to specialized LB hardware |
| Certificate management | One cert to manage (at LB), not N certs on N servers |
| Backend simplicity | App servers handle plain HTTP — simpler code, no TLS config |
| Session affinity | LB can inspect cookies for sticky sessions |
| L7 routing | LB can read URL paths, headers, cookies for routing decisions |
| Performance | Modern LBs use hardware acceleration (AES-NI, TLS acceleration cards) |

**Cons / Security implications:**

| Risk | Details |
|---|---|
| Plaintext in internal network | Traffic between LB and backends is unencrypted — risk if internal network is compromised |
| Trust boundary | LB becomes a high-value attack target (has the private key) |
| Compliance | Some standards (PCI-DSS, HIPAA) require end-to-end encryption |
| Private key exposure | Private key must be stored on LB — hardware HSM or Vault recommended |

**SSL passthrough (alternative):**
```
Client ──[HTTPS]──> [Load Balancer] ──[HTTPS]──> Backend
                    (does NOT decrypt)

LB routes based on SNI hostname (not full URL)
Backend decrypts and processes

Pros: End-to-end encryption, LB never sees plaintext
Cons: Cannot inspect HTTP content, no cookie-based routing, no WAF
```

**SSL bridging (best of both):**
```
Client ──[HTTPS cert A]──> [LB] ──[HTTPS cert B (internal)]──> Backend

LB terminates client TLS → forwards to backend re-encrypted with internal cert
Satisfies compliance (E2E encryption) while allowing L7 inspection
```

**AWS ALB approach:**
```
ACM (Certificate Manager): Free TLS certs, auto-renewal
ALB terminates TLS from client
ALB to EC2: HTTP (within VPC private subnet — considered secure)
ALB to EC2: HTTPS (optional, for strict compliance)

Security Group: EC2 accepts traffic ONLY from ALB → no direct internet access
```

**Private key security best practices:**
```
1. Use HSM (Hardware Security Module) to store private keys
2. Use cloud-managed keys (AWS ACM manages keys, never exports them)
3. Enable perfect forward secrecy (ECDHE cipher suites)
4. Rotate certificates before expiry (use automation)
5. Separate prod/staging certificates and key stores
```

---

### Q19. How do you handle a DDoS attack at the load balancer layer?

**Answer:**

DDoS (Distributed Denial of Service) attacks overwhelm a system's resources. The load balancer is the first line of defense for volumetric and protocol attacks.

**DDoS attack taxonomy:**

```
Volume-based:    Exhaust bandwidth (UDP flood, ICMP flood, amplification)
                 Measured in Gbps/Tbps
                 
Protocol-based:  Exhaust connection tables (SYN flood, Ping of Death)
                 Measured in pps (packets per second)
                 
Application L7:  Exhaust server resources (HTTP flood, slowloris)
                 Measured in RPS (requests per second)
                 Hard to distinguish from legitimate traffic
```

**Defense layers:**

**Layer 1 — ISP / Upstream (volume attacks):**
```
BGP blackholing: ISP drops all traffic to attacked IP
Scrubbing center: Traffic redirected to scrubbing center, attack traffic filtered
  
Limitation: Legitimate traffic also dropped until attack identified
```

**Layer 2 — CDN / Anycast (absorb volume):**
```
Anycast spreads 1 Tbps attack across 250 PoPs → 4 Gbps per PoP (manageable)
Cloudflare, Akamai absorb multi-Tbps attacks at edge

Architecture:
  All traffic → CDN edge → filter at edge → forward clean to origin
```

**Layer 3 — Load Balancer (protocol + application):**

**SYN flood protection:**
```
Problem: Attacker sends millions of SYN packets, never completes handshake
         Server allocates TCP state for each → fills connection table

Solution: SYN cookies
  LB does NOT allocate state on SYN
  LB returns SYN-ACK with cryptographic cookie encoding connection info
  Only allocates state when legitimate ACK arrives with valid cookie
  Invalid or missing ACKs → no state allocated
```

**Rate limiting by IP:**
```
# Nginx rate limiting by IP
limit_req_zone $binary_remote_addr zone=per_ip:10m rate=100r/s;
limit_req zone=per_ip burst=200 nodelay;

# 100 req/sec per IP, burst of 200 → drops excess with 429
```

**Challenge-based mitigation (L7):**
```
Suspicious IP or burst → issue JS challenge (CAPTCHA-like):
  1. Return HTTP 429 with challenge page
  2. Client must solve JS puzzle or CAPTCHA
  3. If solved → issue temporary token → allow through
  4. Bots typically cannot solve → filtered

Used by: Cloudflare, AWS Shield Advanced
```

**IP reputation and allow/blocklisting:**
```
Block known malicious IPs/CIDRs at LB level:
  - Threat intelligence feeds (commercial)
  - TOR exit nodes list
  - Known bot IP ranges
  
HAProxy ACL example:
  acl bad_ip src -f /etc/haproxy/blocklist.txt
  tcp-request connection reject if bad_ip
```

**Rate limiting by behavioral fingerprint:**
```
L7 fingerprints that indicate attack:
  - Same User-Agent string across many IPs
  - All requests to same path (/login, /search)
  - Unusual header combinations
  - Request timing patterns (perfectly regular = bot)
  
WAF (Web Application Firewall) applies these rules:
  AWS WAF, ModSecurity, Cloudflare WAF
```

**Architecture for DDoS-resilient system:**
```
Internet
    │
[Anycast + CDN Edge] ← absorb volumetric attacks
    │
[AWS Shield Advanced] ← L3/L4 DDoS protection
    │
[AWS WAF + ALB] ← L7 protection, IP reputation, rate limiting
    │
[Application Servers] ← rate limiting by user ID, circuit breakers
    │
[Database + Cache] ← protected from direct internet access (VPC)
```

---

### Q20. How do you implement auto-scaling trigger strategies, and what are the pitfalls?

**Answer:**

Auto-scaling automatically adjusts the number of server instances based on load metrics. Correctly configuring trigger strategies prevents both under-provisioning and over-provisioning.

**Scaling metrics and their implications:**

**CPU-based scaling (most common):**
```
Scale out: CPU > 70% for 3 consecutive minutes
Scale in:  CPU < 30% for 10 consecutive minutes

Problem (CPU pitfall):
  CPU may be high due to inefficient code, not legitimate load
  → Scaling out just runs inefficient code on more servers
  Better metric for CPU-intensive workloads, not I/O-bound workloads
  
  I/O-bound app (waiting on DB): CPU = 20% but threads all blocked
  → CPU metric says "scale in" but system is overloaded
```

**Request rate / QPS-based:**
```
Scale out: RPS > 5,000 for 2 consecutive minutes
Scale in:  RPS < 1,000 for 15 consecutive minutes

Pros: Directly measures actual load
Cons: QPS alone ignores response time (fast QPS ≠ no overload)
Use: Generally reliable for web/API tiers
```

**Response latency-based:**
```
Scale out: p99 latency > 500ms for 2 consecutive minutes
Scale in:  p99 latency < 100ms for 10 consecutive minutes

Pros: Directly measures user experience
Cons: Latency may be high due to downstream (DB), not compute — scaling app servers won't help
Best: Combine with CPU/QPS for accurate triggering
```

**Queue depth-based (for worker pools):**
```
Scale out: SQS queue depth > 1,000 messages for 1 minute
Scale in:  SQS queue depth < 100 messages for 5 minutes

Best for: Async worker pools processing from a queue
AWS: SQS → CloudWatch → Auto Scaling Group trigger
```

**Predictive scaling:**
```
ML model predicts traffic based on:
  - Time of day patterns (9am spike, lunch peak, evening ramp)
  - Day of week (Monday morning spike)
  - Historical data

AWS Predictive Scaling:
  Pre-scales 30 minutes before predicted peak
  Avoids cold-start delay during actual traffic spike
```

**Pitfalls and mitigations:**

| Pitfall | Description | Mitigation |
|---|---|---|
| Thrashing | Rapid scale out → scale in → scale out cycles | Use longer scale-in window (10-15 min); add cooldown period |
| Scale-in too aggressive | Removing instances during brief load dip causes latency spike | Conservative scale-in threshold (30% CPU, not 50%) |
| Metric lag | CloudWatch default: 1-minute metric intervals → too slow for spikes | Use detailed monitoring (10s), custom metrics |
| Warm-up time | New instances take 2-5 min to start, install app, connect to DB | Over-provision slightly; use pre-warmed AMI; connection pool pre-warming |
| DB connection explosion | 100 new instances × 50 DB connections = 5,000 new connections instantly | Use connection pooler (PgBouncer, RDS Proxy) |
| Wrong metric | CPU low but latency high (I/O bound workload) | Use multiple metrics (CPU + RPS + latency) with OR condition |

**Recommended multi-metric policy:**
```
Scale OUT if ANY of:
  CPU > 70% for 3 min
  RPS > 8,000 for 2 min
  p99 latency > 800ms for 2 min
  Queue depth > 5,000 for 1 min

Scale IN only if ALL of:
  CPU < 30% for 15 min
  RPS < 2,000 for 15 min
  p99 latency < 150ms for 15 min

Cooldown: 5 min between scale-out events
          15 min between scale-in events
```

---

## Quick Reference

### Load Balancing Algorithms
| Algorithm | State | Best For |
|---|---|---|
| Round Robin | None | Uniform workloads |
| Weighted RR | Weights | Mixed-capacity servers |
| Least Connections | Connection counts | Long-lived connections |
| IP Hash | Hash table | Sticky sessions (legacy) |
| Consistent Hashing | Hash ring | Cache-efficient routing |

### Rate Limiting Algorithms
| Algorithm | Burst? | Memory | Best For |
|---|---|---|---|
| Token Bucket | Yes | O(1) | API rate limiting |
| Leaky Bucket | No (smoothed) | O(queue) | Network traffic shaping |
| Fixed Window | Yes (boundary) | O(1) | Simple implementation |
| Sliding Window Log | No | O(requests) | Precise limiting |
| Sliding Window Counter | Approximate | O(1) | Production (best balance) |

### Circuit Breaker States
| State | Behavior | Transition |
|---|---|---|
| CLOSED | Normal operation | → OPEN on N failures |
| OPEN | Fast fail (return fallback) | → HALF-OPEN after timeout |
| HALF-OPEN | One probe request | → CLOSED on success; → OPEN on fail |

### Auto-Scaling Pitfalls
| Pitfall | Solution |
|---|---|
| Thrashing | Longer scale-in cooldown (15 min) |
| Wrong metric (I/O bound) | Multiple metrics OR trigger |
| DB connection explosion | PgBouncer / RDS Proxy |
| Slow instance warm-up | Predictive scaling + pre-warm AMIs |

### HTTP Status Codes for Rate Limiting
| Code | Meaning | When to use |
|---|---|---|
| 429 | Too Many Requests | Rate limit exceeded |
| 503 | Service Unavailable | Server overloaded |
| 503 + Retry-After | Rate limited | With cooldown hint |
