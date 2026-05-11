# 39 — API Gateway and Service Mesh Deep Dive

## Easy (Q1–Q7)

---

### Q1. What is the difference between an API gateway, a reverse proxy, and a load balancer?

These three terms are frequently confused because modern products often combine multiple capabilities. Understanding the distinct purpose of each is critical.

```
LOAD BALANCER (Layer 4 / Layer 7)
──────────────────────────────────────────────────────────────
Purpose: Distribute traffic across multiple instances of the same service
Operates at: TCP/IP (L4) or HTTP (L7)
Intelligence: Routing algorithm (round-robin, least-connections, IP hash)
Example: AWS ALB/NLB, HAProxy, Nginx upstream

  Clients ──► Load Balancer ──► [Instance 1]
                           └──► [Instance 2]
                           └──► [Instance 3]

No awareness of API semantics, no auth, no rate limiting

REVERSE PROXY (Layer 7)
──────────────────────────────────────────────────────────────
Purpose: Sit in front of servers, forward requests, handle TLS termination
Operates at: HTTP/HTTPS (Layer 7)
Intelligence: URL routing, TLS termination, caching, compression
Example: Nginx, Caddy, Apache httpd

  Clients ──► Reverse Proxy ──► Backend Server(s)
  - TLS termination at proxy boundary
  - Caching of static content
  - Basic URL rewriting/routing

API GATEWAY (Application Layer)
──────────────────────────────────────────────────────────────
Purpose: API management — control, security, observability for APIs
Operates at: HTTP/HTTPS/gRPC (Application Layer)
Intelligence: ALL of the above + auth, rate limiting, transformation,
              API versioning, developer portal, analytics

  External Clients ──► API Gateway ──► Microservice A
                                  └──► Microservice B
                                  └──► Microservice C

  Additional responsibilities:
  - JWT / OAuth2 validation
  - Rate limiting per user/API key
  - Request/response transformation
  - API versioning (/v1, /v2)
  - Developer portal and API documentation
  - Analytics and billing
```

**Decision Guide:**
- Need to distribute load across replicas → **Load Balancer**
- Need TLS termination, caching, basic routing → **Reverse Proxy**
- Need auth, rate limiting, API management for external developers → **API Gateway**

---

### Q2. What are the cross-cutting concerns an API gateway handles?

An API gateway is the **enforcement point** for cross-cutting concerns — capabilities that every API needs but no individual service should implement independently.

```
API GATEWAY CROSS-CUTTING CONCERNS
──────────────────────────────────────────────────────────────
                    Incoming Request
                          │
                          ▼
               ┌─────────────────────┐
               │  1. Authentication  │ Verify identity (JWT, API key, OAuth)
               │  2. Authorization   │ Check permissions (scopes, RBAC)
               │  3. Rate Limiting   │ Per user/IP/key/endpoint
               │  4. SSL Termination │ Decrypt TLS, forward HTTP internally
               │  5. Routing         │ Path/header/weighted routing
               │  6. Load Balancing  │ Distribute across instances
               │  7. Transformation  │ Rewrite headers, body, protocol
               │  8. Caching         │ Response cache (GET requests)
               │  9. Logging         │ Access logs, correlation IDs
               │  10. Tracing        │ Inject trace headers
               │  11. Circuit Breaker│ Fail fast on unhealthy backends
               │  12. Compression    │ gzip/brotli response compression
               └─────────────────────┘
                          │
                    Backend Services
```

**Without an API gateway:** Each microservice must implement auth, rate limiting, logging independently. This leads to:
- Inconsistent enforcement (service A validates tokens strictly; service B doesn't)
- Duplicated code across 50 services
- Harder to change cross-cutting policy (update 50 services vs update gateway config)

**With an API gateway:** Cross-cutting logic is centralized. Services only implement business logic.

---

### Q3. How does JWT validation work at the API gateway?

When a client presents a JWT (JSON Web Token) as a Bearer token, the API gateway must validate it before forwarding the request to the backend. The gateway performs this validation on every request.

```
JWT STRUCTURE
──────────────────────────────────────────────────────────────
eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.        ← Header (base64)
eyJzdWIiOiJ1c2VyXzEyMyIsInNjb3BlcyI6W         ← Payload (base64)
  InJlYWQiLCJ3cml0ZSJdLCJleHAiOjE3M...}
.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV          ← Signature (base64)

Decoded Payload:
{
  "sub": "user_123",          ← Subject (user ID)
  "iss": "https://auth.example.com",  ← Issuer
  "aud": "https://api.example.com",   ← Audience
  "exp": 1714000000,          ← Expiry (Unix timestamp)
  "iat": 1713996400,          ← Issued at
  "scopes": ["read", "write"] ← Permissions
}
```

**Gateway Validation Steps:**
```python
import jwt
from cryptography.hazmat.primitives import serialization

class JWTValidator:
    def __init__(self, public_key_jwks_url: str):
        self.public_keys = self._fetch_jwks(public_key_jwks_url)

    def validate(self, token: str, required_scope: str) -> dict:
        try:
            # Step 1: Decode and verify signature (RS256)
            payload = jwt.decode(
                token,
                self.public_keys,
                algorithms=["RS256"],
                audience="https://api.example.com"  # validate aud
            )
        except jwt.ExpiredSignatureError:
            raise AuthError(401, "Token expired")
        except jwt.InvalidSignatureError:
            raise AuthError(401, "Invalid signature")

        # Step 2: Check expiry (jwt.decode handles this, but explicit check)
        if payload["exp"] < time.time():
            raise AuthError(401, "Token expired")

        # Step 3: Validate issuer
        if payload["iss"] != "https://auth.example.com":
            raise AuthError(401, "Invalid issuer")

        # Step 4: Check required scope
        if required_scope not in payload.get("scopes", []):
            raise AuthError(403, f"Missing scope: {required_scope}")

        return payload  # forward user context to backend
```

**Performance Optimization:** JWT validation is CPU-intensive (RSA signature verification). Gateways cache the JWKS (public keys) in memory and use in-process crypto libraries. A well-optimized gateway validates 10,000+ JWTs per second per CPU core.

---

### Q4. How does rate limiting work at the API gateway — what are the different limiting strategies?

Rate limiting protects backend services from being overwhelmed and prevents API abuse. The gateway applies rate limits at multiple granularities.

```
RATE LIMITING DIMENSIONS
──────────────────────────────────────────────────────────────
Per IP:        100 req/min — prevents DDoS from a single source
Per user:      1000 req/min — per authenticated user
Per API key:   10000 req/min — for registered API clients
Per endpoint:  POST /payment 10/min (expensive operation)
Per tier:      Free: 100/day, Pro: 10000/day, Enterprise: unlimited
```

**Fixed Window Algorithm:**
```python
def is_rate_limited_fixed_window(user_id: str, limit: int) -> bool:
    window_key = f"rate:{user_id}:{int(time.time() // 60)}"  # 1-minute window
    current = redis.incr(window_key)
    if current == 1:
        redis.expire(window_key, 60)
    return current > limit
# Problem: allows 2x limit at window boundary (burst at :59 + burst at :00)
```

**Sliding Window Algorithm (more accurate):**
```python
def is_rate_limited_sliding_window(user_id: str, limit: int,
                                    window_seconds: int = 60) -> bool:
    now = time.time()
    window_start = now - window_seconds
    key = f"rate_sw:{user_id}"

    with redis.pipeline() as pipe:
        # Remove entries outside the window
        pipe.zremrangebyscore(key, 0, window_start)
        # Count entries in window
        pipe.zcard(key)
        # Add current request
        pipe.zadd(key, {str(now): now})
        # Set expiry
        pipe.expire(key, window_seconds + 1)
        _, count, _, _ = pipe.execute()

    return count >= limit
```

**Token Bucket (allows bursting up to bucket capacity):**
```python
def is_allowed_token_bucket(user_id: str,
                             rate: float = 10.0,   # tokens/second
                             capacity: float = 100) -> bool:
    key = f"bucket:{user_id}"
    now = time.time()
    bucket = redis.hgetall(key)

    tokens = float(bucket.get("tokens", capacity))
    last_refill = float(bucket.get("last_refill", now))

    # Refill tokens based on elapsed time
    elapsed = now - last_refill
    tokens = min(capacity, tokens + elapsed * rate)

    if tokens >= 1:
        redis.hset(key, mapping={"tokens": tokens - 1, "last_refill": now})
        return True  # allow request
    return False     # rate limited
```

**Rate Limit Response Headers:**
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 743
X-RateLimit-Reset: 1714000060
Retry-After: 17           (when 429 is returned)
```

---

### Q5. How does the API gateway handle request and response transformation?

Transformation allows the gateway to modify requests before forwarding to backends and modify responses before returning to clients. This decouples client expectations from backend implementation.

```
TRANSFORMATION CAPABILITIES
──────────────────────────────────────────────────────────────
Request Transformation:
  Header injection:  Add X-User-Id, X-Request-Id, X-Forwarded-For
  Header removal:    Strip Authorization header before forwarding
  URL rewriting:     /v1/users/123 → /users/123 (version stripping)
  Body transformation: REST → gRPC, add/remove fields
  Query param modification: add/remove params

Response Transformation:
  Header modification: Add CORS headers, remove internal headers
  Body transformation: Add envelope {"data": ..., "meta": ...}
  Protocol translation: gRPC error → HTTP 4xx JSON
  Caching headers: Add Cache-Control, ETag
```

**Header Injection Example (Kong/Nginx configuration):**
```nginx
# Nginx API Gateway configuration
location /api/v1/users {
    # Inject user ID from JWT into header (backend gets user without parsing JWT)
    set $user_id $jwt_claim_sub;
    proxy_set_header X-User-Id $user_id;
    proxy_set_header X-Request-Id $request_id;
    proxy_set_header X-Forwarded-For $remote_addr;

    # Strip Authorization header from forwarded request
    # (backend doesn't need JWT after gateway validates it)
    proxy_set_header Authorization "";

    # URL rewrite: strip version prefix
    rewrite ^/api/v1/(.*) /$1 break;

    proxy_pass http://user-service-upstream;
}
```

**Protocol Translation (REST ↔ gRPC):**
```
Client sends:    POST /api/users (JSON body)
Gateway:         Translates to gRPC call → UserService.CreateUser(proto payload)
Backend:         Processes gRPC request, returns proto response
Gateway:         Translates proto response → JSON HTTP response
Client receives: 201 Created (JSON body)
```

This allows backends to use gRPC (efficient binary protocol) while clients use REST (more widely supported).

---

### Q6. How does API gateway routing work — path-based, header-based, and weighted routing?

The gateway's routing engine determines which backend service (and which instance/version) receives each request.

```
ROUTING TYPES
──────────────────────────────────────────────────────────────
PATH-BASED ROUTING (most common)
  /api/users/**     → user-service
  /api/orders/**    → order-service
  /api/payments/**  → payment-service
  /api/products/**  → product-service

HEADER-BASED ROUTING (for versioning, A/B, internal traffic)
  Accept-Version: v2      → user-service-v2
  X-Internal: true        → internal-user-service (higher rate limits)
  X-Beta-User: true       → experimental-feature-service

WEIGHTED ROUTING (canary deployment)
  /api/users/**:
    weight: 90 → user-service-v1 (stable)
    weight: 10 → user-service-v2 (canary)
```

**Weighted Routing for Canary (AWS API Gateway / Istio style):**
```yaml
# Istio VirtualService for canary routing
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: user-service
spec:
  http:
  - match:
    - headers:
        x-canary-user:
          exact: "true"
    route:
    - destination:
        host: user-service-v2
        weight: 100   # canary users always get v2
  - route:
    - destination:
        host: user-service-v1
        weight: 90    # 90% of traffic to v1
    - destination:
        host: user-service-v2
        weight: 10    # 10% of traffic to v2 (canary)
```

**Kong Gateway Route Configuration:**
```json
{
  "name": "users-route",
  "paths": ["/api/v1/users"],
  "methods": ["GET", "POST", "PUT"],
  "service": {"id": "user-service-id"},
  "plugins": [
    {"name": "jwt"},
    {"name": "rate-limiting", "config": {"minute": 1000}},
    {"name": "cors"}
  ]
}
```

---

### Q7. How do you make an API gateway highly available — avoiding single point of failure?

The API gateway sits in the critical path of every request. If it fails, the entire application is unreachable. Making it HA requires multiple instances, health checks, and geographic redundancy.

```
HA API GATEWAY ARCHITECTURE
──────────────────────────────────────────────────────────────
Internet
    │
    ▼ DNS (Route 53 / Cloud DNS)
    │  Multiple A records / geographic routing
    │
    ▼ Cloud Load Balancer (AWS ALB / GCP LB)
    │  Layer 4/7 HA, managed by cloud provider
    │
    ├──► API Gateway Instance 1 (AZ-1)
    ├──► API Gateway Instance 2 (AZ-2)
    └──► API Gateway Instance 3 (AZ-3)
         │  All instances are stateless
         │  Shared state: Redis cluster (rate limit counters)
         │
         ├──► Microservice A
         ├──► Microservice B
         └──► Microservice C
```

**Stateless Design (critical for HA):**
```
Each gateway instance must be stateless:
  Rate limit state      → Redis cluster (shared)
  JWT public keys       → In-memory cache (refreshed from JWKS endpoint)
  Route configuration   → Config server / K8s ConfigMap (refreshed periodically)
  Circuit breaker state → Local per-instance (acceptable: each instance
                          independently tracks failures)
```

**Health Check Configuration:**
```yaml
# Kubernetes Deployment with liveness/readiness probes
livenessProbe:
  httpGet:
    path: /health/live    # 200 = gateway is alive
    port: 8080
  initialDelaySeconds: 10
  periodSeconds: 5
  failureThreshold: 3     # 3 failures → restart instance

readinessProbe:
  httpGet:
    path: /health/ready   # 200 = gateway ready to serve traffic
    port: 8080
  periodSeconds: 5
  failureThreshold: 2     # 2 failures → remove from LB rotation
```

**Rolling Updates:** Deploy gateway updates as rolling deployments — never take all instances down simultaneously. This ensures zero-downtime configuration changes.

---

## Medium (Q8–Q15)

---

### Q8. How does a service mesh work — data plane vs control plane?

A service mesh is a dedicated infrastructure layer for handling service-to-service communication inside a cluster. It provides observability, security (mTLS), and traffic management without changing application code.

```
SERVICE MESH ARCHITECTURE
──────────────────────────────────────────────────────────────
CONTROL PLANE (Istiod in Istio)
  ┌─────────────────────────────────────────────────────────┐
  │  Pilot: Service discovery, traffic routing config       │
  │  Citadel: Certificate authority (issues mTLS certs)     │
  │  Galley: Config validation and distribution             │
  └─────────────────────────────────────────────────────────┘
              │  xDS protocol (gRPC streaming)
              │  Pushes config to all sidecars
              ▼
DATA PLANE (Envoy sidecars, one per pod)
  ┌──────────────────────────────────────────────────────────┐
  │  Pod A                        Pod B                     │
  │  ┌──────────┐                ┌──────────┐               │
  │  │ App A    │ ← All traffic  │ App B    │               │
  │  │          │   intercepted  │          │               │
  │  │──────────│   by sidecar   │──────────│               │
  │  │ Envoy    │────mTLS───────►│ Envoy    │               │
  │  │ Sidecar  │                │ Sidecar  │               │
  │  └──────────┘                └──────────┘               │
  └──────────────────────────────────────────────────────────┘

App code:     Connects to localhost (sidecar intercepts via iptables)
Sidecar:      Handles mTLS, load balancing, retry, circuit breaking, metrics
App doesn't know the mesh exists
```

**xDS API (configuration protocol):**
```
Control plane pushes to each Envoy sidecar:
  LDS: Listener Discovery Service  → what ports to listen on
  RDS: Route Discovery Service     → routing rules
  CDS: Cluster Discovery Service   → list of upstream endpoints
  EDS: Endpoint Discovery Service  → IPs/ports of endpoints
  SDS: Secret Discovery Service    → TLS certificates
```

**Traffic interception with iptables:**
```bash
# Init container runs before app container
# Redirects all inbound/outbound traffic through Envoy (port 15001)
iptables -t nat -A PREROUTING -p tcp -j REDIRECT --to-port 15001
iptables -t nat -A OUTPUT -p tcp -j REDIRECT --to-port 15001 \
  -m owner ! --uid-owner envoy  # avoid redirect loop
```

---

### Q9. How does mTLS work in a service mesh — automatic certificate rotation?

**mTLS (mutual TLS)** means both the client AND the server verify each other's identity with certificates. In a service mesh, this is automated — no application code changes needed.

```
MTLS HANDSHAKE
──────────────────────────────────────────────────────────────
Service A (client)              Service B (server)
Envoy sidecar                   Envoy sidecar
      │                               │
      │── ClientHello ───────────────►│
      │◄─ ServerHello + Certificate ──│  "I am service-b, signed by mesh CA"
      │   verify: cert signed by       │
      │   mesh CA? ✓                  │
      │── ClientCertificate ─────────►│  "I am service-a, signed by mesh CA"
      │                               │  verify: cert signed by mesh CA? ✓
      │◄──────── TLS session ─────────│  Both sides authenticated!
      │                               │
      Encrypted, authenticated communication
```

**Automatic Certificate Lifecycle (Istio Citadel/SPIFFE):**
```
Certificate lifespan: 24 hours (short-lived)
Rotation: automatic, 1 hour before expiry

Certificate contains SPIFFE SVID:
  spiffe://cluster.local/ns/production/sa/user-service
  ↑ standard format for workload identity
  ↑ encodes namespace + service account (not IP address)
  ↑ IP-based identity is unreliable in Kubernetes (pods restart)
```

**Zero-Config Encryption:**
```yaml
# Enable mTLS for entire mesh (PeerAuthentication)
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default
  namespace: istio-system  # mesh-wide policy
spec:
  mtls:
    mode: STRICT   # reject all non-mTLS traffic between services
```

With STRICT mTLS: even if an attacker gets onto the network, they cannot impersonate a service without a valid mesh certificate. This provides defense-in-depth beyond network segmentation.

---

### Q10. How does traffic management work in Istio — VirtualService and DestinationRule?

Istio provides fine-grained traffic control through two resources: `VirtualService` (routing logic) and `DestinationRule` (connection/load balancing policy for a destination).

```
VirtualService: "How to route to a destination"
DestinationRule: "How to connect to a destination"
```

**Canary Deployment with Traffic Split:**
```yaml
# DestinationRule: define subsets (versions) of the service
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: user-service
spec:
  host: user-service
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100
      http:
        h2UpgradePolicy: UPGRADE    # prefer HTTP/2
    loadBalancer:
      simple: ROUND_ROBIN
  subsets:
  - name: v1
    labels:
      version: v1        # matches Kubernetes pods with label version=v1
    trafficPolicy:
      connectionPool:
        http:
          maxRequestsPerConnection: 1
  - name: v2
    labels:
      version: v2
```

```yaml
# VirtualService: route 90% to v1, 10% to v2 (canary)
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: user-service
spec:
  hosts:
  - user-service
  http:
  - match:
    - headers:
        x-canary:
          exact: "true"
    route:
    - destination:
        host: user-service
        subset: v2    # canary header → always v2
  - route:
    - destination:
        host: user-service
        subset: v1
        weight: 90
    - destination:
        host: user-service
        subset: v2
        weight: 10
    timeout: 5s               # per-request timeout
    retries:
      attempts: 3
      perTryTimeout: 2s
      retryOn: "5xx,reset,connect-failure"
```

**Fault Injection (chaos testing in mesh):**
```yaml
# Inject 5% delay + 1% error to test resilience
http:
- fault:
    delay:
      percentage:
        value: 5
      fixedDelay: 500ms
    abort:
      percentage:
        value: 1
      httpStatus: 503
  route:
  - destination:
      host: user-service
```

---

### Q11. How does circuit breaking work at the service mesh layer?

Service mesh circuit breaking (implemented in Envoy) prevents cascading failures by ejecting unhealthy endpoints from the load balancing pool automatically.

```
CIRCUIT BREAKER AT MESH LAYER (Envoy Outlier Detection)
──────────────────────────────────────────────────────────────
Load balancing pool: [Pod1, Pod2, Pod3, Pod4, Pod5]

Pod3 starts returning 5xx errors:
  Request to Pod3 → 500 error (1st)
  Request to Pod3 → 500 error (2nd)
  Request to Pod3 → 500 error (3rd)
  5 consecutive failures in 30s → Envoy ejects Pod3 from pool

Pool: [Pod1, Pod2, Pod4, Pod5]  (Pod3 ejected for 30 seconds)
New requests go to healthy pods only.
After ejection interval: Pod3 re-admitted to pool
  If Pod3 returns 200: re-admitted permanently
  If Pod3 returns 500: ejection time doubled (exponential backoff)
```

**Istio DestinationRule Circuit Breaker Config:**
```yaml
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: user-service
spec:
  host: user-service
  trafficPolicy:
    # Connection pool limits (prevent overwhelming upstream)
    connectionPool:
      tcp:
        maxConnections: 50          # max TCP connections to service
      http:
        http1MaxPendingRequests: 20 # max queued requests before 503
        maxRequestsPerConnection: 10

    # Outlier detection (eject unhealthy pods)
    outlierDetection:
      consecutive5xxErrors: 5       # eject after 5 consecutive 5xx
      interval: 30s                 # check interval
      baseEjectionTime: 30s         # initial ejection duration
      maxEjectionPercent: 50        # never eject more than 50% of pool
      minHealthPercent: 30          # keep at least 30% of pool active
```

**Difference from application-level circuit breaker (Hystrix/Resilience4j):**
- App-level: each service implements its own circuit breaker in code
- Mesh-level: circuit breaking is at the proxy layer, no code changes needed
- Mesh circuit breaking is **per endpoint** (individual pod), not per service as a whole

---

### Q12. How does distributed tracing propagation work through a service mesh?

Distributed tracing reconstructs the full path of a request across multiple services. The service mesh automates trace header injection, but application code must propagate headers between incoming and outgoing calls.

```
DISTRIBUTED TRACE PROPAGATION
──────────────────────────────────────────────────────────────
Client → API Gateway → Service A → Service B → Database
                                         │
Each hop generates a SPAN (start time, end time, service name)
All spans for one request share a TRACE ID

HTTP Headers (B3 / W3C TraceContext format):
  X-B3-TraceId: abc123         ← same across entire request chain
  X-B3-SpanId: def456          ← unique per hop
  X-B3-ParentSpanId: 789abc    ← parent span
  X-B3-Sampled: 1              ← sample this trace
```

**Service Mesh Role (Envoy automatic injection):**
```
Envoy sidecar automatically:
  1. Generates new trace if none exists (entry point)
  2. Extracts trace headers from incoming request
  3. Injects trace headers into outgoing requests
  4. Reports span to tracing backend (Jaeger/Zipkin/Tempo)
```

**Application Responsibility (header propagation):**
```python
# Envoy handles span creation, but your app must propagate headers
# from incoming request to outgoing downstream calls

TRACE_HEADERS = [
    "x-request-id", "x-b3-traceid", "x-b3-spanid",
    "x-b3-parentspanid", "x-b3-flags", "x-b3-sampled",
    "b3"  # W3C format
]

def call_downstream_service(incoming_request, endpoint, body):
    # Propagate trace headers from incoming request to downstream call
    headers = {h: incoming_request.headers[h]
               for h in TRACE_HEADERS
               if h in incoming_request.headers}
    return http_client.post(endpoint, json=body, headers=headers)
```

**Without header propagation:** Each service starts a new root span — traces are fragmented and you can't see the full request path.

---

### Q13. What are the golden signals from a service mesh and how do you use them?

The **four golden signals** (from Google's SRE book) can be extracted automatically from the service mesh without any instrumentation in application code. The mesh observes all traffic between services.

```
FOUR GOLDEN SIGNALS FROM SERVICE MESH
──────────────────────────────────────────────────────────────
For EVERY service pair (source → destination):

1. LATENCY
   Envoy records: request_duration_milliseconds histogram
   Query (Prometheus):
     histogram_quantile(0.99,
       rate(istio_request_duration_milliseconds_bucket[5m]))

2. TRAFFIC
   Envoy records: total requests per second
   Query:
     rate(istio_requests_total[1m])

3. ERRORS
   Envoy records: response_code for every request
   Query (error rate):
     rate(istio_requests_total{response_code=~"5.."}[5m])
     / rate(istio_requests_total[5m])

4. SATURATION
   Envoy records: connection pool usage, pending requests
   Query:
     istio_tcp_connections_opened_total
     envoy_cluster_upstream_rq_pending_total
```

**Pre-built Grafana Dashboard (Istio):**
```
Per service-to-service edge:
  - Request rate (req/s)
  - Error rate (%)
  - Latency p50/p95/p99 (ms)
  - Bytes sent/received

Topology view (Kiali):
  Visual graph of all service-to-service connections
  Color coded: green (healthy) → yellow (degraded) → red (unhealthy)
  Click any edge: see golden signals for that specific service pair
```

**Alerting Rules:**
```yaml
# Prometheus alert: error rate > 5% for any service
- alert: HighErrorRate
  expr: |
    rate(istio_requests_total{response_code=~"5.."}[5m])
    / rate(istio_requests_total[5m]) > 0.05
  for: 2m
  labels:
    severity: warning
  annotations:
    summary: "High error rate: {{ $labels.destination_service_name }}"
```

---

### Q14. When is a service mesh overkill and when does it add genuine value?

Service meshes add operational complexity. For small or simple architectures, the overhead outweighs the benefits. Understanding the threshold helps you give the right architectural recommendation.

**Service Mesh Adds Genuine Value When:**
```
✓ 10+ microservices with complex inter-service communication
✓ Need zero-config mTLS between all services
✓ Multiple deployment environments (canary, blue-green) per week
✓ Compliance requirements: need encrypted service-to-service traffic
✓ Need automatic retry/timeout/circuit breaking without code changes
✓ Multiple language runtimes (Python, Go, Java) — mesh is language-agnostic
✓ Need detailed observability for hundreds of service-to-service paths
```

**Service Mesh Is Overkill When:**
```
✗ < 5 services: manual configuration is simpler
✗ Monolith moving toward microservices (premature optimization)
✗ Single programming language: use per-library solution (Resilience4j)
✗ Low traffic: mesh overhead (30-80MB per sidecar) doesn't justify
✗ Team doesn't have Kubernetes/Envoy expertise
✗ All services already share same process (don't need network-level mesh)
```

**Overhead (Istio with Envoy sidecar):**
```
Memory:  30–80 MB per pod (Envoy process)
Latency: 0.5–2ms added per hop (in practice, often < 1ms on fast hardware)
CPU:     0.2–0.5 vCPU per pod under load
         Negligible at low traffic; measurable at >1000 req/s per pod
```

**Alternative for small teams:** Use a service library (Resilience4j, go-kit, Dapr SDK) that provides circuit breaking, retries, and tracing in-process, without the operational overhead of a mesh.

---

### Q15. Compare Kong, AWS API Gateway, Nginx, and Envoy for different use cases.

```
COMPARISON TABLE
──────────────────────────────────────────────────────────────
                Kong        AWS API GW   Nginx         Envoy
────────────────────────────────────────────────────────────
Type            Self-hosted  Managed     Self-hosted   Self-hosted
                or managed              / embedded
Deployment      K8s, VM, DC  AWS only    VM, container Control plane
Protocol        HTTP, gRPC,  HTTP/REST,  HTTP, TCP,    HTTP/1.1, 2,
                WebSocket    WebSocket   UDP           gRPC, WebSocket
Admin API       REST API     AWS Console nginx.conf    xDS API
                                         (static)      (dynamic)
Plugins         500+         Lambda      Modules       Filters
                ecosystem    integrations (limited)    (extensible)
Performance     Very High    Managed     Very High     Highest
                                         (C-based)     (C++ envoy)
Rate Limiting   Built-in     Built-in    Nginx Plus    Via control
                                         only          plane
Circuit Breaking Plugin      Not native  Not native    Built-in
Observability   Prometheus   CloudWatch  Access logs   Prometheus,
                Grafana                               Zipkin, Jaeger
Cost            Open-source  Pay per     Free (nginx)  Free
                + enterprise request     + Nginx Plus
Best For        Multi-cloud  Pure AWS    Simple        Service mesh
                API mgmt     serverless  web serving   data plane
```

**Decision Guide:**
- **AWS-only startup:** AWS API Gateway (zero operational overhead)
- **Multi-cloud / self-hosted:** Kong (rich plugin ecosystem, supported)
- **High-performance web server with basic gateway:** Nginx
- **Service mesh data plane:** Envoy (used by Istio, Consul Connect)
- **Large enterprise with complex API products:** Kong or Apigee

---

## Hard (Q16–Q20)

---

### Q16. How do you design a global API gateway with regional failover?

A global API gateway must serve users with low latency worldwide, survive a full regional outage, and maintain consistent configuration across all regions.

```
GLOBAL API GATEWAY ARCHITECTURE
──────────────────────────────────────────────────────────────
                          Anycast DNS / GeoDNS
                         (Route53 / Cloud DNS)
                                  │
               ┌──────────────────┼──────────────────┐
               │                  │                  │
         US-EAST-1           EU-WEST-1          AP-EAST-1
         Gateway Cluster     Gateway Cluster    Gateway Cluster
         (3 instances)       (3 instances)      (3 instances)
               │                  │                  │
         Regional Backend   Regional Backend   Regional Backend
         Services           Services           Services
               │
         Shared State:
         - Rate limit counters: Redis Global (cross-region sync via CRDTs)
         - Config: Git → CI/CD pipeline → all regions simultaneously
         - JWT public keys: JWKS endpoint (cached per region)
```

**Active-Active Configuration:**
```
GeoDNS routing:
  US users → US-EAST-1 gateway cluster
  EU users → EU-WEST-1 gateway cluster
  AP users → AP-EAST-1 gateway cluster

Health checks:
  Route53 health check every 10s per region
  Unhealthy region → DNS TTL = 60s → traffic re-routed to next nearest

Failover time: ~60s (DNS TTL) for global users
               ~10s for users with short DNS cache
```

**Rate Limiting Across Regions:**
```
Challenge: User in US makes 800 requests to US gateway,
           then VPN to EU and makes 400 more requests.
           Total = 1200 but each region only saw 800 and 400.

Solution 1: Approximate global rate limiting
  Each region limits independently; allow some over-counting
  Use CRDT counters (conflict-free replicated data types)
  Bounded inaccuracy: 10-20% over limit acceptable for most use cases

Solution 2: Centralized Redis (higher latency)
  All gateways write to single Redis cluster
  Cross-region call: 50-150ms latency on rate limit check
  Use async write with local read: eventual consistency (10s lag)

Solution 3: Sticky region per user
  User always routes to home region via session affinity
  Breaks during regional failover (acceptable)
```

**Configuration Synchronization:**
```yaml
# GitOps: single source of truth for all regions
# CI/CD pipeline deploys config change to all regions sequentially:
deploy_config:
  strategy: rolling
  regions: [us-east-1, eu-west-1, ap-east-1]
  per_region_wait: 300s   # wait 5 minutes after each region
  rollback_on_error_spike: true
  error_threshold: 5%
```

---

### Q17. How does GraphQL federation work with Apollo Gateway aggregating multiple subgraphs?

**Apollo Federation** allows you to split a GraphQL schema across multiple services (subgraphs), each owned by a different team, while presenting a single unified GraphQL API to clients through a gateway.

```
APOLLO FEDERATION ARCHITECTURE
──────────────────────────────────────────────────────────────
                    Apollo Gateway (Router)
                    Receives: query { user(id: 1) { name orders { total } } }
                          │
          ┌───────────────┼───────────────┐
          │               │               │
   User Subgraph    Order Subgraph   Product Subgraph
   (User service)  (Order service)  (Product service)
   Owns: User type  Owns: Order type Owns: Product type
```

**Subgraph Schema (User Service):**
```graphql
# user-service schema
type User @key(fields: "id") {   # @key = primary key for federation
  id: ID!
  name: String!
  email: String!
}

type Query {
  user(id: ID!): User
}
```

**Subgraph Schema (Order Service — extends User):**
```graphql
# order-service schema
type User @key(fields: "id") @extends {
  id: ID! @external     # id comes from user-service
  orders: [Order!]      # orders service EXTENDS the User type
}

type Order {
  id: ID!
  total: Float!
  items: [OrderItem!]
}

type Query {
  order(id: ID!): Order
}
```

**Query Planning and Execution:**
```
Query: { user(id: 1) { name orders { total } } }

Apollo Gateway query plan:
  Step 1: Call user-service → { user(id: 1) { id name } }
  Step 2: Call order-service → { _entities(representations: [{__typename: "User", id: 1}]) { ... on User { orders { total } } } }
  Step 3: Merge results → { user: { name: "Alice", orders: [{ total: 99.99 }] } }

The @key directive allows gateway to fetch entities from other subgraphs
by passing the key field (id) as a representation.
```

**Benefits:**
| Benefit | Description |
|---|---|
| Team autonomy | Each team owns and deploys their subgraph independently |
| Unified API | Clients see one schema, not dozens of microservice APIs |
| Type extension | Order service can extend User type without changing user service |
| Independent scaling | Scale Order subgraph separately from User subgraph |

---

### Q18. How do you implement API versioning at the gateway — path, header, and content-type strategies?

API versioning allows you to evolve APIs without breaking existing clients. The gateway is the ideal enforcement point because versioning is a cross-cutting concern.

```
API VERSIONING STRATEGIES
──────────────────────────────────────────────────────────────
1. PATH VERSIONING (most common)
   GET /v1/users/123
   GET /v2/users/123
   ✓ Explicit, easy to read in logs
   ✓ Cacheable (distinct URLs)
   ✗ URL changes break links/bookmarks

2. HEADER VERSIONING
   GET /users/123
   Accept-Version: v2
   ✓ Clean URLs
   ✗ Not cacheable by CDN (requires Vary header)
   ✗ Less visible/discoverable

3. CONTENT-TYPE VERSIONING (Accept header)
   GET /users/123
   Accept: application/vnd.example.v2+json
   ✓ RESTful (content negotiation)
   ✗ Complex to implement and test

4. QUERY PARAMETER VERSIONING
   GET /users/123?version=2
   ✓ Easy to test in browser
   ✗ Cache pollution, easy to miss
```

**Gateway Routing with Path Versioning:**
```nginx
# Nginx gateway configuration
# v1 routes to old service
location ~ ^/v1/users {
    rewrite ^/v1/(.*)$ /$1 break;
    proxy_pass http://user-service-v1;
}

# v2 routes to new service
location ~ ^/v2/users {
    rewrite ^/v2/(.*)$ /$1 break;
    proxy_pass http://user-service-v2;
}

# Default (unversioned) routes to latest stable
location /users {
    proxy_pass http://user-service-v2;
}
```

**Versioning with Sunset Headers (deprecation):**
```python
# Middleware: add deprecation headers for v1 routes
def add_deprecation_headers(response, api_version: str):
    if api_version == "v1":
        response.headers["Deprecation"] = "true"
        response.headers["Sunset"] = "Sat, 01 Jan 2026 00:00:00 GMT"
        response.headers["Link"] = '</v2/users>; rel="successor-version"'
    return response
```

**Running Multiple Versions Simultaneously:**
- Keep v1 running for backward compatibility (ideally 6–12 months deprecation window)
- Gateway routes each version to its backend without the backends knowing about versions
- Monitor: track what % of traffic is still on v1 (helps decide when to sunset)

---

### Q19. How do you design API gateway security against common attacks?

The API gateway is the first line of defense and must protect against multiple attack vectors: DDoS, injection attacks, credential stuffing, and API-specific attacks.

```
ATTACK VECTORS AND GATEWAY DEFENSES
──────────────────────────────────────────────────────────────
Attack Type         Defense at Gateway
──────────────────────────────────────────────────────────────
DDoS                IP rate limiting, geo-blocking, CAPTCHA challenge
Credential stuffing Account rate limiting, CAPTCHA, device fingerprinting
SQL/Command inject  WAF (Web Application Firewall) + input validation
JWT attacks         Strict algorithm validation, short expiry, key rotation
SSRF                Whitelist allowed upstream URLs, block internal IPs
Path traversal      Normalize URL before routing, block /../ patterns
Large payload       Request size limits (e.g., max 10MB body)
Slow loris          Request timeout limits (e.g., 30s max)
Replay attacks      Timestamp in JWT + nonce (idempotency key)
```

**Defense-in-Depth Configuration:**
```python
class APIGatewaySecurityMiddleware:
    MAX_BODY_SIZE = 10 * 1024 * 1024   # 10MB
    REQUEST_TIMEOUT = 30               # seconds

    def validate_request(self, request):
        # 1. Block private IP ranges (SSRF prevention)
        if self._is_private_ip(request.target_host):
            raise SecurityError(403, "SSRF attempt blocked")

        # 2. Request size limit
        if int(request.headers.get("Content-Length", 0)) > self.MAX_BODY_SIZE:
            raise SecurityError(413, "Request too large")

        # 3. Block path traversal
        if "../" in request.path or "%2e%2e" in request.path.lower():
            raise SecurityError(400, "Invalid path")

        # 4. JWT replay prevention (iat must be recent)
        if request.jwt_payload:
            if time.time() - request.jwt_payload["iat"] > 86400:
                raise SecurityError(401, "Token too old")

        # 5. WAF rules (SQLi, XSS patterns in query params)
        for param, value in request.query_params.items():
            if self._matches_waf_rule(value):
                raise SecurityError(400, "Invalid input detected")
```

**Rate Limiting for Credential Stuffing:**
```python
# Progressive penalties: exponential backoff per account
def check_failed_login(account_id: str):
    key = f"failed_login:{account_id}"
    failures = int(redis.get(key) or 0) + 1
    redis.setex(key, 3600, failures)  # reset after 1 hour

    if failures >= 10:
        block_duration = 2 ** (failures - 10)  # exponential: 1s, 2s, 4s, 8s...
        redis.setex(f"blocked:{account_id}", block_duration, "1")
        raise SecurityError(429, f"Too many failures. Try again in {block_duration}s")
```

---

### Q20. How do you design a service mesh migration strategy — moving from direct service calls to mesh without downtime?

Migrating to a service mesh is a significant infrastructure change. Done incorrectly, it causes outages. A phased, non-disruptive migration is required.

```
SERVICE MESH MIGRATION PHASES
──────────────────────────────────────────────────────────────
Phase 0: Prerequisites (1–2 weeks)
  - Kubernetes cluster with Istio installed (permissive mode initially)
  - All services use HTTP/HTTPS (not just TCP)
  - Service accounts defined for each service
  - Monitoring: Prometheus + Grafana baseline established

Phase 1: Observability only (2–4 weeks)
  - Inject Envoy sidecars into non-production workloads first
  - Mode: PERMISSIVE (accepts both plain HTTP and mTLS)
  - No traffic policy changes; observe golden signals in Kiali
  - Goal: understand traffic patterns before enforcing policies

Phase 2: Traffic management (2–4 weeks)
  - Apply VirtualService for retry/timeout policies per service
  - Start with conservative values; tighten after observing behavior
  - Enable circuit breaking (outlier detection) gradually
  - Mode: still PERMISSIVE

Phase 3: mTLS (2–4 weeks)
  - Enable mTLS in PERMISSIVE mode (accepts both mTLS and plaintext)
  - Monitor: confirm all service-to-service calls are using mTLS
  - Switch namespace-by-namespace to STRICT mode
  - First: dev namespace, then staging, then production

Phase 4: Full mesh + cleanup (ongoing)
  - All namespaces in STRICT mTLS mode
  - Remove manual TLS implementations from application code
  - Remove per-service retry logic (handled by mesh)
  - Decommission old service-level logging (mesh provides it)
```

**Zero-Downtime Sidecar Injection:**
```bash
# Label namespace for automatic sidecar injection
kubectl label namespace production istio-injection=enabled

# New pods get sidecar automatically on next deployment
# Existing pods: rolling restart to inject sidecar
kubectl rollout restart deployment/user-service -n production
# Rolling restart: one pod at a time, old pods serve traffic while new ones start

# Verify sidecar injection
kubectl get pod -n production -l app=user-service -o jsonpath='{.items[*].spec.containers[*].name}'
# Should show: user-service istio-proxy (two containers per pod)
```

**Validation at Each Phase:**
```python
# Automated validation before advancing phase
def validate_mesh_phase(phase: int) -> bool:
    if phase == 1:  # observability
        return grafana.dashboards_populated() and \
               kiali.service_graph_complete()

    if phase == 2:  # traffic management
        return all(
            metrics.p99_latency(svc) < SLA_LATENCY and
            metrics.error_rate(svc) < 0.01
            for svc in critical_services
        )

    if phase == 3:  # mTLS
        return all(
            mesh.is_mtls_active(svc, namespace="production")
            for svc in all_services
        )
```

**Rollback Plan:**
- Phase 1–2 rollback: remove labels, sidecar injection disabled, traffic policies deleted
- Phase 3 rollback: switch mTLS mode back to PERMISSIVE (no service restart needed)
- Phase 4 rollback: revert to PERMISSIVE, restore manual TLS configs

---

## Quick Reference

```
API GATEWAY vs REVERSE PROXY vs LOAD BALANCER
──────────────────────────────────────────────────────────────
Load Balancer → distribute traffic across instances (L4/L7)
Reverse Proxy → TLS termination, caching, URL routing (L7)
API Gateway   → auth, rate limiting, transformation, analytics (API layer)

JWT VALIDATION ORDER
──────────────────────────────────────────────────────────────
1. Decode header → verify algorithm (RS256, not none!)
2. Verify signature against public key (JWKS)
3. Check expiry (exp claim)
4. Check issuer (iss claim)
5. Check audience (aud claim)
6. Check required scopes

RATE LIMITING ALGORITHMS
──────────────────────────────────────────────────────────────
Fixed window    → simple; burst at boundary (2x limit possible)
Sliding window  → accurate; Redis sorted set; higher memory
Token bucket    → allows burst up to capacity; smooth over time
Leaky bucket    → constant output rate; queues excess requests

SERVICE MESH COMPONENTS
──────────────────────────────────────────────────────────────
Data plane:     Envoy sidecar (one per pod) — handles all traffic
Control plane:  Istiod — manages config, certs, service discovery
xDS protocol:   LDS/RDS/CDS/EDS/SDS — control plane → sidecar config
mTLS:           SPIFFE SVID certificates; 24h lifetime; auto-rotation

ISTIO KEY RESOURCES
──────────────────────────────────────────────────────────────
VirtualService     → routing rules (weights, match conditions, retries, timeouts)
DestinationRule    → connection policy (pool size, outlier detection, subsets)
PeerAuthentication → mTLS mode (PERMISSIVE / STRICT)
AuthorizationPolicy→ allow/deny rules (source, destination, method)
Gateway            → ingress/egress configuration (L7 LB into mesh)

CIRCUIT BREAKER (Envoy Outlier Detection)
──────────────────────────────────────────────────────────────
consecutive5xxErrors: 5      → eject after 5 failures
baseEjectionTime: 30s        → ejection duration (doubles each time)
maxEjectionPercent: 50       → never eject more than half the pool

DISTRIBUTED TRACING HEADERS (B3 Format)
──────────────────────────────────────────────────────────────
x-b3-traceid     → same for entire request chain
x-b3-spanid      → unique per service hop
x-b3-parentspanid→ links to parent span
x-b3-sampled     → 1=sample this trace, 0=do not sample
App must propagate all these headers from incoming to outgoing calls!

TOOL COMPARISON SUMMARY
──────────────────────────────────────────────────────────────
Kong        → multi-cloud API management, rich plugins
AWS API GW  → serverless, AWS-only, pay-per-request
Nginx       → high-performance reverse proxy/web server
Envoy       → service mesh data plane, dynamic configuration
Istio       → full service mesh (Envoy + control plane)
Linkerd     → lightweight service mesh (Rust-based proxy)
Consul      → service mesh + service discovery

MIGRATION PHASE ORDER
──────────────────────────────────────────────────────────────
PERMISSIVE mode (accepts both plain + mTLS)
→ Observability only
→ Traffic policies (retry/timeout/circuit breaking)
→ mTLS per namespace (STRICT)
→ Full mesh operational
```
