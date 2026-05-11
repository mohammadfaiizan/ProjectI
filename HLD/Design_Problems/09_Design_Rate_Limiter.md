# Design a Rate Limiter — High-Level Design

---

## 1. Problem Statement & Clarifying Questions

**Problem Statement:**
Design a rate limiter that controls the rate of requests allowed to a service or API. The rate limiter should prevent abuse, protect backend services from overload, enforce usage quotas, and handle distributed environments where multiple API gateway instances run simultaneously.

**Clarifying Questions:**
- Should the rate limiter work for a single server or distributed deployment?
- What is the rate limiting granularity: per user, per IP, per API key, per endpoint?
- Should it support different tiers (free: 100 req/min, paid: 1000 req/min)?
- What happens when the limit is exceeded — hard reject (429) or queue the request?
- Should limits be enforced globally across all regions or per-region?
- What consistency model is required (hard vs soft limits)?
- Does it need to support burst traffic (allow short spikes)?
- How should rate limit headers be exposed to clients?

**Assumptions:**
- Distributed rate limiter (multiple API gateway instances)
- Per user + per endpoint granularity
- Support multiple rate limit tiers
- Hard reject with 429 on limit exceeded
- Redis for distributed state
- Expose standard rate limit headers (X-RateLimit-*)

---

## 2. Functional & Non-Functional Requirements

### Functional Requirements
1. **Limit Requests:** Reject requests exceeding defined thresholds
2. **Multiple Algorithms:** Support Token Bucket, Fixed Window, Sliding Window
3. **Granular Limits:** Per user, per IP, per API key, per endpoint
4. **User Tiers:** Different limits for free/pro/enterprise users
5. **Rate Limit Headers:** Return remaining quota and retry time in headers
6. **Soft vs Hard Limits:** Option to warn vs reject
7. **Whitelist/Blacklist:** Bypass limit for trusted clients, block banned IPs
8. **Dynamic Configuration:** Update rate limits without redeployment

### Non-Functional Requirements
1. **Latency:** Rate limit check adds <1ms to request processing
2. **Accuracy:** No more than ±1% error on limit enforcement
3. **Availability:** Rate limiter failure should fail open (let requests through)
4. **Scalability:** Handle 1M requests/second across cluster
5. **Consistency:** Soft consistency acceptable (slightly over limit during network partition)
6. **Persistence:** Rate limit state survives single Redis node failure

---

## 3. Capacity Estimation

### Scale
- Total API requests: 1 Billion/day = 12K average QPS
- Peak QPS: 50K (during traffic spikes)
- Rate limit checks per request: 2-3 (user-level + endpoint-level)
- Redis operations per check: 2 (get + increment)
- Total Redis ops: 50K * 3 * 2 = 300K Redis ops/second

### Redis Sizing
- Each rate limit entry: key + counter = ~100 bytes
- Unique (user, endpoint) pairs active per second: 1M
- Memory for active entries: 1M * 100B = 100MB (tiny)
- With sliding window logs: 1M * 60 entries/min * 8B = 480MB
- Single Redis instance handles 300K ops/sec easily

### Latency Budget
- Total allowed overhead: 1ms
- Redis network round trip: ~0.3ms (same DC)
- Redis command execution: ~0.01ms
- Application processing: ~0.2ms
- Total: ~0.5ms per check (well within budget)

---

## 4. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                                     │
│           Mobile App       Web Browser       Partner API                │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      API GATEWAY CLUSTER                                 │
│                                                                          │
│  ┌────────────────┐   ┌────────────────┐   ┌────────────────┐          │
│  │  Gateway 1     │   │  Gateway 2     │   │  Gateway 3     │          │
│  │                │   │                │   │                │          │
│  │  Rate Limiter  │   │  Rate Limiter  │   │  Rate Limiter  │          │
│  │  Middleware    │   │  Middleware    │   │  Middleware    │          │
│  └───────┬────────┘   └───────┬────────┘   └───────┬────────┘          │
│          │                   │                     │                    │
│          └───────────────────┼─────────────────────┘                   │
│                              │ Redis EVAL (Lua atomic scripts)           │
└──────────────────────────────┼──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      REDIS CLUSTER (Rate Limit State)                    │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  rate:{user_id}:{endpoint}:{window}  →  counter                  │  │
│  │  token:{user_id}  →  {tokens: 95, last_refill: 1234567890.5}    │  │
│  │  log:{user_id}:{endpoint}  →  Sorted Set (timestamp, req_id)    │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  Primary Node       Replica Node       Redis Sentinel                   │
│  (write)           (read fallback)     (failover)                       │
└──────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       BACKEND SERVICES                                   │
│   Service A          Service B          Service C                       │
└─────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│                  CONFIGURATION & ADMIN                                  │
│  Rate Limit Rules DB (MySQL) → Config Service → Gateway Hot Reload     │
│  Rules: {user_tier, endpoint_pattern, algorithm, limit, window}        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Component Deep-Dive — The 5 Algorithms

### Algorithm 1: Token Bucket

```
Concept:
- Bucket holds up to `capacity` tokens
- Tokens added at rate `refill_rate` tokens/second
- Each request consumes 1 token
- If bucket empty → reject request

Properties:
- Allows burst traffic (up to `capacity` size)
- Smooth average rate controlled by `refill_rate`
- Most popular algorithm for APIs

Example:
- capacity = 10, refill_rate = 2/sec
- At t=0: bucket has 10 tokens
- 8 rapid requests → bucket has 2 tokens
- At t=2s: bucket refilled to 6 tokens
- At t=5s: bucket full again (10 tokens)

Implementation:
state = {tokens: current_tokens, last_refill: timestamp}
On request:
  elapsed = now - state.last_refill
  new_tokens = min(capacity, state.tokens + elapsed * refill_rate)
  if new_tokens >= 1:
      state.tokens = new_tokens - 1
      state.last_refill = now
      return ALLOW
  return DENY
```

### Algorithm 2: Leaky Bucket

```
Concept:
- Requests enter a queue (bucket)
- Requests processed at fixed rate `leak_rate` (e.g., 1 req/sec)
- If queue is full → reject incoming request
- Output is perfectly smooth regardless of input burst

Properties:
- No burst allowed — output is constant rate
- Queue absorbs small bursts, large bursts rejected
- Good for smoothing traffic before backend

Example:
- queue_size = 5, leak_rate = 1 req/sec
- 5 requests arrive simultaneously → 4 queued, processed 1/sec
- 6th request → queue full → rejected

Difference from Token Bucket:
- Token Bucket: burst sends immediately (up to bucket size)
- Leaky Bucket: burst queued and sent at fixed rate
```

### Algorithm 3: Fixed Window Counter

```
Concept:
- Divide time into fixed windows (e.g., 1-minute windows: [0:00, 1:00), [1:00, 2:00))
- Count requests per window per user
- If count exceeds limit → reject

Properties:
- Simple and memory efficient (one counter per window)
- PROBLEM: Boundary spike

Boundary Spike Problem:
- Limit: 100 req/min
- User sends 100 req at 0:59 → allowed (window 1)
- User sends 100 req at 1:00 → allowed (window 2)
- Effectively: 200 requests in 2 seconds at the window boundary!

Implementation (Redis):
key = f"rate:{user_id}:{int(now / window_size)}"
count = INCR(key)
if count == 1: EXPIRE(key, window_size)
if count > limit: REJECT else ALLOW
```

### Algorithm 4: Sliding Window Log

```
Concept:
- Store timestamp of every request in a sorted set
- On each request: remove timestamps outside the window
- Count remaining timestamps → current requests in window
- If count >= limit → reject

Properties:
- Most accurate (true sliding window, no boundary spike)
- Memory intensive: stores all timestamps per user per window
- O(N) cleanup where N = requests in window

Implementation (Redis sorted set):
key = f"log:{user_id}:{endpoint}"
now = time.time()
window_start = now - window_size

# Atomic Lua script:
# ZREMRANGEBYSCORE(key, 0, window_start)  ← remove old entries
# count = ZCARD(key)
# if count < limit:
#     ZADD(key, now, request_id)
#     EXPIRE(key, window_size)
#     return ALLOW
# return DENY
```

### Algorithm 5: Sliding Window Counter (Approximation)

```
Concept:
- Maintain counters for current and previous fixed windows
- Calculate weighted count using time position in current window
- approximate_count = prev_count * (1 - elapsed/window) + current_count

Properties:
- Memory efficient (2 counters per user)
- Approximation error: ≤ 10% in typical cases (worst case at boundary)
- Practical balance of accuracy vs efficiency

Formula:
  elapsed_in_window = now % window_size
  weight = elapsed_in_window / window_size
  approx_count = prev_window_count * (1 - weight) + current_window_count

Example:
- Window: 60s, Limit: 100 req/min
- Current window (t=45s in): 80 requests
- Previous window: 60 requests
- Weight = 45/60 = 0.75
- approx_count = 60 * (1 - 0.75) + 80 = 15 + 80 = 95 requests
- Under limit → ALLOW
```

---

## 6. Distributed Rate Limiting

### The Problem
- 3 API gateway instances, each checking rate limit independently
- User sends 100 req: 34 to gateway-1, 33 to gateway-2, 33 to gateway-3
- Each gateway allows 100 requests → 300 requests get through!

### Solution: Centralized Redis with Atomic Lua Scripts

```lua
-- Token Bucket in Redis Lua (atomic execution)
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local now = tonumber(ARGV[3])

local state = redis.call('HMGET', key, 'tokens', 'last_refill')
local tokens = tonumber(state[1]) or capacity
local last_refill = tonumber(state[2]) or now

local elapsed = now - last_refill
local new_tokens = math.min(capacity, tokens + elapsed * refill_rate)

if new_tokens >= 1 then
    redis.call('HMSET', key, 'tokens', new_tokens - 1, 'last_refill', now)
    redis.call('EXPIRE', key, 3600)
    return 1  -- ALLOW
else
    redis.call('HMSET', key, 'tokens', new_tokens, 'last_refill', now)
    return 0  -- DENY
end
```

Why Lua? Redis executes Lua scripts atomically — no race conditions, no WATCH/MULTI overhead.

### Rate Limit by Multiple Dimensions

```
Key hierarchy:
1. Global user limit:   rate:user:{user_id}:global
2. Endpoint limit:      rate:user:{user_id}:endpoint:{endpoint_hash}
3. IP limit:            rate:ip:{ip_address}:global
4. API key limit:       rate:apikey:{api_key}:global

Check order (AND logic — all must pass):
1. Is IP blacklisted?          → 403
2. Does API key exist?         → 401
3. Check IP rate limit         → 429 if exceeded
4. Check user rate limit       → 429 if exceeded
5. Check endpoint rate limit   → 429 if exceeded
→ Allow request
```

### Rate Limit Tiers

| Tier | Requests/min | Burst | Monthly quota |
|------|-------------|-------|---------------|
| Free | 60 | 10 | 10,000 |
| Pro | 1,000 | 100 | 1,000,000 |
| Enterprise | 10,000 | 1,000 | Unlimited |

---

## 7. API Design — Response Headers

### Allowed Request
```
HTTP/1.1 200 OK
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 847
X-RateLimit-Reset: 1704067200  (Unix timestamp when window resets)
X-RateLimit-Policy: 1000;w=60  (1000 per 60-second window)
```

### Rate Limited Request
```
HTTP/1.1 429 Too Many Requests
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1704067200
Retry-After: 43  (seconds until limit resets)
Content-Type: application/json

{
    "error": "rate_limit_exceeded",
    "message": "You have exceeded 1000 requests per minute",
    "limit": 1000,
    "remaining": 0,
    "retry_after": 43,
    "upgrade_url": "https://api.example.com/upgrade"
}
```

---

## 8. Database Design

### Rate Limit Rules (MySQL)

```sql
CREATE TABLE rate_limit_rules (
    rule_id         INT PRIMARY KEY AUTO_INCREMENT,
    name            VARCHAR(100) NOT NULL,
    user_tier       ENUM('free','pro','enterprise','custom') NOT NULL,
    endpoint_pattern VARCHAR(200),       -- NULL means global
    algorithm       ENUM('token_bucket','sliding_window','fixed_window'),
    limit_count     INT NOT NULL,
    window_seconds  INT NOT NULL,
    burst_size      INT,                 -- for token bucket
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMP DEFAULT NOW(),
    INDEX(user_tier, endpoint_pattern)
);

CREATE TABLE ip_blacklist (
    ip_cidr     VARCHAR(20) PRIMARY KEY,
    reason      VARCHAR(200),
    added_at    TIMESTAMP DEFAULT NOW(),
    expires_at  TIMESTAMP              -- NULL = permanent
);
```

### Redis Key Schema

```
# Token Bucket State
token:{user_id}:{endpoint}  →  Hash {tokens: float, last_refill: float}  TTL: 3600

# Fixed Window Counter
fw:{user_id}:{endpoint}:{window_ts}  →  Integer  TTL: window_size * 2

# Sliding Window Log
swl:{user_id}:{endpoint}  →  Sorted Set {score: timestamp, member: req_id}  TTL: window_size

# Sliding Window Counter (two counters)
swc:{user_id}:{endpoint}:prev  →  Integer  TTL: window_size * 2
swc:{user_id}:{endpoint}:curr  →  Integer  TTL: window_size * 2
```

---

## 9. Scalability & Bottlenecks

| Bottleneck | Problem | Solution |
|-----------|---------|----------|
| Redis single point | All rate limit checks go through Redis | Redis Cluster with sharding by user_id |
| Redis hot key | Popular user hammers same Redis slot | User-id sharding spreads load |
| Network latency | Redis round trip on every request | Local in-process cache for token state (sync every 100ms) |
| Rule evaluation | Complex rule matching per request | Rule compilation, fast path for simple cases |
| Config updates | Propagating new rules to all gateways | Pub/Sub from Redis, gateways subscribe to rule changes |

### Handling Redis Failure

**Fail-Open Strategy (preferred for API rate limiting):**
- If Redis is unavailable, allow all requests (fail open)
- Rationale: better to over-serve than to bring down the entire API
- Add circuit breaker: if Redis timeout > 50ms, fail open

**Local Fallback:**
- Each gateway maintains approximate local counter
- Falls back to local counter when Redis is down
- Risk: over-limit by factor of N_gateways during outage

---

## 10. Trade-offs & Design Decisions

### Sliding Window Log vs Sliding Window Counter
- **Log:** Exact accuracy, memory O(requests in window)
- **Counter:** ~10% max error, memory O(1) per user
- **Choice:** Sliding Window Counter for most cases; Log only for strict billing limits
- **Trade-off:** Memory vs accuracy

### Hard Limit vs Soft Limit
- **Hard Limit:** Strictly enforce — reject at exactly N requests
- **Soft Limit:** Allow up to N*1.1 requests (relaxed for UX)
- **Choice:** Hard limit for free tier, soft limit (+5%) for paid tier
- **Trade-off:** Fairness vs customer experience

### Centralized (Redis) vs Decentralized Rate Limiting
- **Centralized:** Accurate global limits, adds network round trip
- **Decentralized:** No network overhead, but limits enforced per node (not global)
- **Choice:** Centralized Redis for global accuracy
- **Trade-off:** Latency (+0.5ms) vs accuracy

### Sticky vs Non-Sticky Rate Limiting
- **Sticky:** User always routed to same gateway (consistent hashing)
  - Local counter is sufficient, no Redis needed
  - Problem: uneven load distribution
- **Non-Sticky:** User can hit any gateway
  - Requires centralized storage for accurate global limit
  - **Choice:** Non-sticky with Redis
  - **Trade-off:** Redis dependency vs simpler architecture

---

## Key Interview Talking Points

1. **Algorithm Trade-offs:** Know all 5 algorithms. Fixed window is simple but has boundary spike. Sliding window log is exact but memory-heavy. Token bucket is most common (allows burst). Be ready to draw timing diagrams.

2. **Boundary Spike Problem:** Fixed window flaw — demonstrate with example: 100 req/min limit, 100 req at :59 + 100 req at :00 = 200 req in 2 seconds. Sliding window solves this.

3. **Distributed Race Condition:** Two gateways checking simultaneously — both read count=99, both allow, both increment to 100 — race condition! Redis atomic Lua scripts solve this (EVAL is single-threaded in Redis).

4. **Redis Lua for Atomicity:** Explain why Lua scripts are atomic in Redis (single-threaded, no interleaving). Compare to MULTI/EXEC (optimistic locking, can fail). Lua is simpler and more performant.

5. **Response Headers:** X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset, Retry-After. These are critical for client experience — clients can self-throttle and retry intelligently.

6. **Fail-Open vs Fail-Closed:** Rate limiter should fail open when Redis is down. Rate limiting is a secondary protection — it's better to over-serve temporarily than to bring down the API for all users.

7. **User Tier Design:** Free tier uses token bucket (small bucket, small refill). Enterprise uses higher limits with same algorithm. Different endpoints may have different limits (write-heavy endpoints more restricted).

8. **Hot Key in Redis:** If one user sends 100K req/s, their Redis key becomes hot. Solution: shard the user's key by `user_id + random(0, 10)`, divide limit by 10 per shard.

9. **Client-Side Rate Limiting:** First line of defense is client-side throttling. SDKs implement exponential backoff. This reduces load on servers during spikes. 429 responses teach clients to back off.

10. **Back-of-Envelope:** 50K QPS * 3 checks/req * 2 Redis ops/check = 300K Redis ops/sec. Single Redis instance handles 1M ops/sec. One Redis cluster sufficient. P99 adds ~0.5ms overhead — acceptable.
