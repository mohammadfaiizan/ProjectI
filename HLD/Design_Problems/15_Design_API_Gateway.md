# System Design: API Gateway

---

## 1. Problem Statement & Clarifying Questions

### Problem Statement
Design an API Gateway that serves as the single entry point for all client requests to a microservices backend. It must handle 10 million requests per minute while providing authentication, rate limiting, routing, circuit breaking, caching, and request transformation.

### Clarifying Questions

**Scale:**
- How many requests per minute? *10M req/min (~167K req/sec)*
- How many downstream microservices? *~50 services*
- How many unique API consumers (clients/apps)? *~10K API keys*

**Functionality:**
- What authentication methods? *JWT, API keys, OAuth 2.0 introspection*
- Rate limiting: per user, per API key, per IP, or all three? *All three, with different limits*
- Does the gateway need to modify request/response bodies? *Yes — header injection, body rewrite*
- Protocol translation required? *Yes — REST to gRPC for some services*
- Response caching? *Yes — for GET requests with Cache-Control*

**Reliability:**
- What happens when a downstream service is down? *Circuit breaker returns 503*
- Can the gateway itself go down? *No — must be highly available; deploy as cluster*
- Latency budget for gateway overhead? *< 5ms added latency at p99*

---

## 2. Functional & Non-Functional Requirements

### Functional Requirements
1. Request routing (path-based, header-based, canary routing)
2. Authentication (JWT validation, API key lookup, OAuth introspection)
3. Rate limiting (per client, per endpoint, per IP; token bucket)
4. Circuit breaker per downstream service
5. Response caching (TTL-based, Cache-Control header compliance)
6. Request/response transformation (header injection, body rewrite)
7. SSL/TLS termination
8. Load balancing across service instances
9. Request aggregation (BFF pattern — combine multiple service calls)
10. Structured logging and distributed tracing
11. API versioning (route v1/v2 to different backends)
12. Health check aggregation

### Non-Functional Requirements
| Property | Target |
|---|---|
| Throughput | 10M req/min (167K req/sec) |
| Added Latency | < 5ms p99 gateway overhead |
| Availability | 99.999% (5 nines) |
| Scalability | Horizontally scalable; stateless nodes |
| Security | JWT validation, DDoS mitigation, OWASP protection |

---

## 3. Capacity Estimation

### Traffic
- 167K req/sec peak = **6 Gbps** assuming ~400 byte average request
- With 10ms average request duration: ~1.67M concurrent requests in-flight
- 50 downstream services × average 3 instances = **150 backend instances**

### Rate Limit State
- 10K API keys × counters per minute window = **10K Redis keys**
- Token bucket state per key: 32 bytes → 10K × 32 = **320 KB** (trivial)
- IP-based rate limit: up to 10M unique IPs/hour → **320 MB** in Redis

### Cache
- Average cached response: 2 KB
- Hot URLs (~1K entries): 1K × 2 KB = **2 MB** in memory
- Full response cache: tune based on TTL and traffic patterns; typically 1-10 GB

### Gateway Nodes
- Single node capacity: 20K req/sec (conservative estimate for auth + routing overhead)
- Nodes needed: 167K / 20K = **9 nodes minimum** → deploy 15 for headroom
- Each node: 32 vCPUs, 64 GB RAM

---

## 4. High-Level Architecture

```
                    ┌─────────────────────────────────────────┐
                    │     Internet / DNS Load Balancer         │
                    │     (AWS Route53 / Cloudflare)          │
                    └──────────────────┬──────────────────────┘
                                       │ HTTPS
                    ┌──────────────────▼──────────────────────┐
                    │           CDN / DDoS Shield              │
                    │       (Cloudflare / AWS CloudFront)     │
                    └──────────────────┬──────────────────────┘
                                       │
          ┌────────────────────────────▼────────────────────────────┐
          │              API Gateway Cluster (15 nodes)              │
          │                                                          │
          │  ┌──────────────────────────────────────────────────┐   │
          │  │         Middleware Pipeline (per request)         │   │
          │  │  Logging → Auth → RateLimit → Cache → Router     │   │
          │  │         → CircuitBreaker → Upstream Forward       │   │
          │  └──────────────────────────────────────────────────┘   │
          │                                                          │
          │  Shared State: Redis Cluster                             │
          │  (rate limit counters, distributed circuit breaker       │
          │   state, response cache, session tokens)                 │
          └──────────────────┬──────────────────────────────────────┘
                             │ Internal network (gRPC / HTTP/2)
          ┌──────────────────▼──────────────────────────────────────┐
          │              Service Mesh / Internal LB                  │
          └────┬───────────┬─────────────┬──────────────────────────┘
               │           │             │
    ┌──────────▼───┐  ┌────▼──────┐  ┌──▼──────────┐
    │ User Service │  │ Order Svc │  │ Product Svc │  ... (50 services)
    │ (3 instances)│  │           │  │             │
    └──────────────┘  └───────────┘  └─────────────┘

          ┌─────────────────────────────────────────────────────┐
          │               Observability Stack                    │
          │  Prometheus (metrics) + Jaeger (tracing)            │
          │  ELK Stack (logs) + Grafana (dashboards)            │
          └─────────────────────────────────────────────────────┘
```

---

## 5. Component Deep-Dive

### 5.1 Middleware Pipeline (Chain of Responsibility Pattern)

Every incoming request traverses a fixed middleware chain:

```
Request →  LoggingMiddleware      (start timer, assign trace ID)
        →  AuthMiddleware         (validate JWT/API key)
        →  RateLimitMiddleware    (token bucket check)
        →  CacheMiddleware        (return cached response if hit)
        →  RouterMiddleware       (path matching → select upstream)
        →  CircuitBreakerMW       (check circuit state)
        →  [Upstream HTTP call]
        →  CircuitBreakerMW       (record result — open/close)
        →  CacheMiddleware        (store response if cacheable)
        →  LoggingMiddleware      (log completion, latency)
        →  Response to client
```

Each middleware has a `handle(request, context, next)` signature. Calling `next()` passes control to the next middleware. This enables:
- Middleware reordering without changing code
- Easy A/B testing of middleware configurations
- Individual middleware enable/disable per route

### 5.2 Authentication at Gateway

**JWT Validation (stateless — no DB lookup):**
1. Extract `Authorization: Bearer <token>` header
2. Decode and verify signature using public key (RS256/ES256)
3. Check `exp` (expiry), `iss` (issuer), `aud` (audience) claims
4. Extract `sub` (user_id), `scope` claims → inject into downstream request headers
5. Public key cached in memory; refreshed from JWKS endpoint every hour

**API Key Validation (stateful — Redis lookup):**
1. Extract `X-API-Key` header or `?api_key=` query param
2. Hash key with SHA-256 → lookup in Redis hashmap `apikeys:{hash} → {client_id, rate_limit, scopes}`
3. Cache API key metadata in gateway memory (TTL=60s)

**OAuth 2.0 Token Introspection:**
1. For opaque tokens (not JWT): call auth server's `/introspect` endpoint
2. Cache introspection result for token TTL duration

### 5.3 Rate Limiting (Token Bucket Algorithm)

**Token Bucket Per Client:**
```
State (in Redis):
  key: ratelimit:{client_id}:{window}
  value: { tokens_remaining, last_refill_timestamp }

Algorithm:
  1. Compute tokens earned since last_refill:
     earned = (now - last_refill) × rate_per_second
  2. tokens = min(bucket_capacity, tokens_remaining + earned)
  3. If tokens >= 1: allow request, decrement tokens by 1
  4. Else: return 429 Too Many Requests
     Header: Retry-After: <seconds_until_next_token>
```

**Rate Limit Tiers:**
| Tier | Limit | Window |
|---|---|---|
| Free API key | 100 req/min | 60s |
| Pro API key | 10,000 req/min | 60s |
| Enterprise | 1M req/min | 60s |
| Per endpoint | Configurable | Configurable |
| Per IP (DDoS) | 1,000 req/min | 60s |

**Redis Lua Script (atomic token bucket):**
```lua
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local data = redis.call('HMGET', key, 'tokens', 'last_refill')
-- ... compute new tokens, return allowed/denied
```

### 5.4 Circuit Breaker (Per Downstream Service)

**States:** CLOSED → OPEN → HALF_OPEN → CLOSED

```
CLOSED:    Normal operation. Track failure rate.
           If failure_rate > threshold (e.g., 50%) in last N requests
           → trip to OPEN
OPEN:      Fast-fail all requests (return 503 immediately).
           After timeout (e.g., 30s) → transition to HALF_OPEN
HALF_OPEN: Allow 1 probe request.
           If success → CLOSED; if failure → back to OPEN
```

**Metrics tracked:**
- Success count, failure count, timeout count in a rolling window (last 100 requests)
- Response time percentiles (open if p99 > threshold)

**Why per-service, not global:**
- Service A being down should not trip the circuit for Service B
- Isolation prevents cascade failures

### 5.5 Request Routing

**Path-based routing:**
```
/api/users/**     → user-service
/api/orders/**    → order-service
/api/products/**  → product-service
```

**Header-based routing (canary):**
```
X-Canary: true → route to canary version (10% of traffic)
X-Region: EU   → route to EU data center services
```

**API versioning:**
```
/v1/users → user-service:v1 (legacy)
/v2/users → user-service:v2 (current)
/v3/users → user-service:v3 (beta, 5% canary)
```

**Load balancing algorithm:** Weighted Round Robin with health checks. Remove unhealthy instances from pool automatically.

### 5.6 Response Caching

**Cache keyed by:** `{method}:{path}:{query_params_sorted}:{vary_headers}`

**Respect HTTP cache semantics:**
- `Cache-Control: max-age=60` → cache for 60 seconds
- `Cache-Control: no-store` → never cache
- `Vary: Accept-Language` → separate cache entries per language
- `ETag` / `If-None-Match` → 304 Not Modified responses

**Cache invalidation:** `POST/PUT/DELETE` to `/users/{id}` invalidates `GET /users/{id}` cache entry.

### 5.7 Request/Response Transformation

**Header Injection (inbound → backend):**
- Strip `Authorization` header before forwarding (replace with decoded `X-User-Id`)
- Inject `X-Request-Id: <trace_id>` for distributed tracing
- Inject `X-Forwarded-For`, `X-Real-IP`
- Add `X-Gateway-Version` for debugging

**Response Transformation:**
- Remove internal headers (`X-Internal-Service`, `X-Pod-Name`)
- Inject CORS headers (`Access-Control-Allow-Origin`)
- Add security headers (`X-Content-Type-Options`, `Strict-Transport-Security`)

**Protocol Translation (REST → gRPC):**
- Route `/api/users/123` (REST GET) → `UserService.GetUser({id: "123"})` (gRPC)
- Map HTTP status codes ↔ gRPC status codes
- Handle Protobuf serialization/deserialization

---

## 6. Database Design

The gateway itself is largely stateless. Shared state lives in Redis:

```
Redis Data Structures:

# Rate limiting (token bucket)
HASH ratelimit:{client_id}
  tokens_remaining: "99.5"
  last_refill_ts: "1704067200.123"
  capacity: "100"
  refill_rate: "1.667"   # tokens per second = 100/min
  TTL: 3600 seconds (auto-expire idle clients)

# API key registry
HASH apikeys:{sha256_of_key}
  client_id: "client_123"
  name: "My App"
  tier: "pro"
  rate_limit: "10000"
  scopes: "read:users,write:orders"
  created_at: "1704067200"
  active: "1"

# Circuit breaker state (shared across gateway nodes)
HASH circuit:{service_name}
  state: "CLOSED"          # CLOSED, OPEN, HALF_OPEN
  failure_count: "3"
  success_count: "97"
  last_failure_ts: "1704067200.5"
  open_since_ts: "0"

# Response cache
STRING cache:{cache_key_hash}
  value: <serialized_response_body>
  TTL: as specified by Cache-Control

# Active JWT public keys (JWKS cache)
STRING jwks:public_key
  value: <JSON encoded JWKS>
  TTL: 3600 seconds
```

**Gateway Configuration (stored in a config service / etcd):**
```yaml
routes:
  - path_prefix: /api/users
    service: user-service
    instances: [10.0.1.1:8080, 10.0.1.2:8080]
    circuit_breaker:
      failure_threshold: 50%
      window: 100
      timeout: 30s

rate_limits:
  - tier: free
    limit: 100
    window: 60s
  - tier: pro
    limit: 10000
    window: 60s
```

---

## 7. API Design

The gateway exposes no data API itself — it proxies all business APIs. However, it provides a management API:

```
# Admin / Config API (internal only, mTLS required)
GET  /admin/routes                 # List all registered routes
POST /admin/routes                 # Register a new route
PUT  /admin/routes/{id}            # Update route config
DELETE /admin/routes/{id}          # Remove route

GET  /admin/circuit-breakers       # All circuit breaker states
POST /admin/circuit-breakers/{service}/reset  # Manually close circuit

GET  /admin/rate-limits            # Current rate limit config
PUT  /admin/rate-limits/{tier}     # Update limits

GET  /admin/cache/stats            # Cache hit rate, size
DELETE /admin/cache                # Flush entire cache
DELETE /admin/cache/{key}          # Invalidate specific key

GET  /health                       # Gateway health (Kubernetes liveness probe)
GET  /ready                        # Ready to receive traffic (Kubernetes readiness)
GET  /metrics                      # Prometheus metrics endpoint

# Standard proxy behavior for all other paths:
* /v1/**    → proxied to registered services
* /v2/**    → proxied to registered services
```

**Response Headers added by gateway:**
```
X-Request-Id: <uuid>             # for distributed tracing
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 97
X-RateLimit-Reset: 1704067260    # unix timestamp
X-Cache: HIT | MISS
X-Circuit-Breaker: CLOSED
X-Gateway-Latency-Ms: 2.3
```

---

## 8. Scalability & Bottlenecks

### Bottleneck 1: JWT Validation CPU Overhead
- RS256 signature verification is CPU-intensive (~1ms each)
- At 167K req/sec × 1ms = **167 cores** just for JWT validation
- **Solution:** Use asymmetric key caching (verify in memory); switch to Ed25519 (10× faster than RS256); use JWT session caching (validate once, cache result for 30s per `jti`)

### Bottleneck 2: Redis as Shared Rate Limit State
- Rate limit check = 2 Redis commands per request = 334K Redis ops/sec
- **Solution:** Redis Cluster with 10+ shards; Lua scripts for atomic operations; shard by `client_id` hash for locality; optionally use in-process sliding window for very hot keys (accept slight inaccuracy at cluster boundary)

### Bottleneck 3: Gateway as Hotspot (Single Point of Failure)
- Solution: **Cluster of 15 stateless nodes** behind an L4 load balancer
- DNS-level failover for region-level failures
- Kubernetes Horizontal Pod Autoscaler based on CPU/memory metrics

### Bottleneck 4: Circuit Breaker State Distribution
- Each gateway node has local circuit breaker state → split brain possible
- **Solution:** Store circuit state in Redis (shared); all nodes agree on circuit state; accept ~10ms staleness (Redis async reads)

### Bottleneck 5: Response Cache Invalidation
- Stale cache after write operations
- **Solution:** Write-through invalidation via Kafka: `order.updated` event → cache invalidation for `/api/orders/{id}/*`; short TTLs (30-60s) for frequently-changing data as a fallback

---

## 9. Trade-offs & Design Decisions

### Decision 1: Single Gateway vs Federated Gateways
- **Chosen:** Single logical gateway cluster with per-team route ownership
- **Alternative:** Each team owns a sub-gateway (Netflix Zuul approach)
- **Why single:** Simpler deployment, unified rate limiting and auth, easier monitoring
- **Trade-off:** Single team responsible for gateway uptime; changes require coordination

### Decision 2: Stateless Gateway vs Stateful Session
- **Chosen:** Stateless gateway (all state in Redis)
- **Why:** Any node can serve any request; horizontal scaling is trivial; no sticky sessions needed
- **Trade-off:** Every request hits Redis for rate limit + auth lookup; mitigated by in-process caching with short TTL

### Decision 3: Gateway vs Service Mesh (Istio/Linkerd)
- **Chosen:** API Gateway for north-south (external) traffic; Service Mesh for east-west (internal)
- **Why:** Gateway handles business concerns (auth, rate limiting, API versioning); service mesh handles infrastructure concerns (mTLS, service discovery, retries)
- They are complementary, not alternatives

### Decision 4: In-Process Cache vs Redis for Response Caching
- **Chosen:** Both — in-process LRU cache for hottest 1K URLs, Redis for broader cache
- **Why:** In-process cache is sub-microsecond; Redis adds 1-2ms; the hottest URLs justify in-process caching
- **Trade-off:** Cache inconsistency across nodes for in-process cache; TTL-bound staleness is acceptable

### Decision 5: Circuit Breaker Threshold Tuning
- Default: 50% failure rate over 100 requests, 30s open window
- Different per service: payment service (30% threshold — more sensitive), static content service (70% threshold — more tolerant)
- Why: Over-sensitive circuit breakers cause cascade failures from momentary blips; under-sensitive allow cascades

---

## 10. Key Interview Talking Points

1. **Gateway is the Cross-Cutting Concern Consolidator:** Rather than implementing auth, rate limiting, and logging in every microservice, the gateway centralizes these. This reduces code duplication and ensures consistent policy enforcement.

2. **Middleware Chain (Chain of Responsibility):** Each middleware is independent. Adding a new capability (e.g., A/B testing header injection) means adding a new middleware without touching existing ones. The chain is composable and testable in isolation.

3. **Token Bucket vs Leaky Bucket vs Sliding Window:**
   - Token bucket: allows short bursts (up to capacity), refills at steady rate
   - Leaky bucket: smooths traffic to a constant rate (no bursts)
   - Sliding window: most accurate, more expensive to compute
   - **Token bucket** is the industry standard (Stripe, AWS API Gateway) — allows bursting while enforcing average rate

4. **Circuit Breaker is Not Retry Logic:** Retries on a failing service add load; circuit breaker cuts the circuit immediately. The pattern: retry for transient errors (500, timeout), circuit break for systemic failure (all requests failing).

5. **JWT Validation is Stateless by Design:** The gateway doesn't call an auth service for every request — it validates the JWT signature locally using the cached public key. This is O(1) and adds < 1ms. The downside: revoked tokens are valid until expiry (mitigate with short-lived tokens + refresh token rotation).

6. **Kong vs AWS API Gateway vs Custom:**
   - Kong: plugin ecosystem, self-hosted or managed, Lua-extensible
   - AWS API Gateway: fully managed, tight AWS integration, higher latency
   - Custom (like this): full control, higher operational burden
   - **Interview answer:** Use managed if on cloud (faster time to market); custom if unique requirements (ultra-low latency, proprietary protocols)

7. **Canary Routing at Gateway:** Deploy new service version to 5% of traffic via header-based or percentage-based routing. Observe error rate in real-time. Gradually increase to 100%. Rollback = update route weight. No downtime.

8. **The BFF Pattern (Backend for Frontend):** Mobile app needs `GET /homepage` that returns user profile + recent orders + recommendations — all from different services. The gateway (or a dedicated BFF service behind it) fans out to 3 services in parallel, aggregates the results, and returns a single response. Reduces mobile round trips from 3 to 1.

9. **SSL Termination at Gateway:** Gateway decrypts TLS, forwards plain HTTP internally (or re-encrypts with internal cert for mTLS). This enables: single certificate management point, deep packet inspection for WAF, response caching (can't cache encrypted responses).

10. **Key Metrics to Monitor:**
    - Gateway latency overhead (p50/p99 of time added vs upstream latency)
    - Rate limit trigger rate per tier (alerts on spike)
    - Circuit breaker open count per service
    - Auth failure rate (spike = attack or cert rotation issue)
    - Cache hit rate (target > 40% for GET-heavy APIs)
    - Error rate per upstream service (routing decisions)
