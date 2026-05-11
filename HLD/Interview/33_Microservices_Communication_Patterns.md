# 33 — Microservices Communication Patterns

---

## Easy (Q1–Q7)

---

### Q1. What is the difference between synchronous and asynchronous communication in microservices?

**Answer:**

**Synchronous communication:** The caller sends a request and waits (blocks) for the response before proceeding. The caller and receiver must both be available at the same time.

**Asynchronous communication:** The caller sends a message and continues without waiting. The receiver processes the message independently, at its own pace.

```
Synchronous (REST/gRPC):
  Order Service ──── HTTP POST /payments ────► Payment Service
                ◄─── 200 OK { payment_id } ───
  (Order Service blocked for ~150ms)

Asynchronous (Kafka/RabbitMQ):
  Order Service ──── publish(order_placed) ──► Message Broker
                ◄─── ack (message stored) ────
  (Order Service free immediately)
                                        later...
  Message Broker ──── consume ──────────────► Payment Service
                                               (processes independently)
```

**When to use synchronous:**
- User-facing requests needing immediate response (user login, product search)
- Simple request-response with low latency requirement
- Query operations (GET requests)
- When the response is needed to continue processing (auth check)

**When to use asynchronous:**
- Long-running operations (video encoding, report generation)
- Fan-out to multiple services (send email + SMS + push notification)
- Decoupling throughput (payment processing can be slower than checkout)
- Resilience: downstream failures don't break the caller
- Event-driven workflows (order placed → payment → fulfillment → shipping)

**Trade-offs:**

| Dimension | Synchronous | Asynchronous |
|-----------|-------------|--------------|
| Latency | Adds up chain latency | Decoupled |
| Complexity | Simple | More complex (broker, consumers) |
| Failure coupling | Caller fails if receiver fails | Caller unaffected |
| Data consistency | Immediate | Eventual |
| Debugging | Simpler (request tracing) | Harder (event correlation) |
| Throughput | Limited by slowest service | Each service scales independently |

---

### Q2. Compare REST, gRPC, and GraphQL for inter-service communication.

**Answer:**

**REST (Representational State Transfer):**
```
Protocol: HTTP/1.1 or HTTP/2
Serialization: JSON (human-readable, larger)
Contract: OpenAPI/Swagger (optional)
Typical use: Public APIs, browser-to-server, simple CRUD

GET /users/42
POST /orders { "user_id": 42, "items": [...] }
```

**gRPC (Google Remote Procedure Call):**
```
Protocol: HTTP/2 (required)
Serialization: Protocol Buffers (binary, 5-10x smaller than JSON)
Contract: .proto files (mandatory, strongly typed)
Typical use: Internal service-to-service, high-throughput, polyglot

// users.proto
service UserService {
  rpc GetUser(GetUserRequest) returns (User);
  rpc StreamUsers(StreamRequest) returns (stream User);
}
```

**GraphQL:**
```
Protocol: HTTP/1.1 (usually)
Serialization: JSON
Contract: Schema + SDL (GraphQL SDL)
Typical use: BFF pattern, mobile clients with varying data needs

query {
  user(id: 42) {
    name
    email
    orders(last: 5) { total status }
  }
}
```

**Comparison table:**

| Property | REST | gRPC | GraphQL |
|----------|------|------|---------|
| Payload size | Large (JSON) | Small (Protobuf) | Medium (JSON) |
| Performance | Good | Best | Good |
| Browser support | Native | Limited (grpc-web) | Native |
| Streaming | No (HTTP/2 push) | Yes (bidirectional) | Subscriptions |
| Type safety | Optional (OpenAPI) | Enforced (.proto) | Enforced (SDL) |
| Versioning | URL /v2/ or headers | Backward compat Protobuf | Schema evolution |
| Over-fetching | Yes | Yes | No (client specifies fields) |
| Learning curve | Low | Medium | Medium-High |
| Best for | Public APIs, CRUD | Internal microservices | BFF, mobile |

**Recommendation:**
- Internal service-to-service: gRPC (performance, type safety, streaming)
- Public/external APIs: REST (simplicity, tooling, browser support)
- Mobile BFF or complex frontend queries: GraphQL

---

### Q3. What is the difference between request-response and publish-subscribe patterns?

**Answer:**

**Request-Response:**
```
Client ────────── request ──────────────► Server
Client ◄───────── response ─────────────── Server

Characteristics:
- One sender, one receiver
- Sender knows about and addresses receiver directly
- Synchronous (blocking) or async with callback
- Response goes back to original caller
- Used for: queries, commands needing acknowledgment
```

**Publish-Subscribe (Pub/Sub):**
```
Publisher ─── event: "order.placed" ──► Topic/Broker
                                              │
                          ┌───────────────────┼───────────────────┐
                          ▼                   ▼                   ▼
                   Payment Service    Email Service    Analytics Service
                   (subscriber)       (subscriber)     (subscriber)

Characteristics:
- One sender, many receivers (0 to N)
- Publisher has NO knowledge of subscribers
- Decoupled: adding a new subscriber requires no publisher change
- Used for: events, notifications, fan-out, audit logs
```

**Concrete example — order placed:**
```
Request-Response approach:
  Order Service → HTTP POST /payment-service/charge
  Order Service → HTTP POST /email-service/send
  Order Service → HTTP POST /inventory-service/reserve
  Problem: Order Service coupled to 3 services; one failure blocks the chain

Pub/Sub approach:
  Order Service publishes: { event: "order.placed", order_id: 42, ... } to Kafka
  Payment Service subscribes to "order.placed" topic
  Email Service subscribes to "order.placed" topic
  Inventory Service subscribes to "order.placed" topic
  Benefit: Order Service knows nothing about downstream; adding Analytics is zero-touch
```

**Message routing variations:**
```
Topic (Kafka, SNS): All subscribers receive all messages from topic
Queue (SQS, RabbitMQ): Competing consumers — only ONE consumer gets each message
Topic + Queue fan-out (SNS→SQS): All consumers get a copy via their own queue
```

---

### Q4. What is service discovery and why can't you hardcode IP addresses in microservices?

**Answer:**

Service discovery is the mechanism by which services find each other's network addresses at runtime without hardcoding them.

**Why hardcoding IPs fails:**
```
Hardcoded: payment-service → "http://10.0.1.45:8080"

Problems:
1. IP changes on restart:  Pod rescheduled to 10.0.2.78 → hardcoded IP dead
2. Scaling:    3 payment-service pods, which IP to use? Load balance how?
3. Environment differences: Dev=10.0.1.x, Staging=172.16.x.x, Prod=10.128.x.x
4. Rolling deploys: old pod on 10.0.1.45 replaced by new pod on 10.0.3.22
5. Cross-AZ:  Services span multiple data centers → IPs differ by region
```

**Service discovery patterns:**

**Pattern 1: Client-side discovery**
```
Service Registry (Consul/Eureka):
  payment-service: [10.0.1.45:8080, 10.0.2.78:8080, 10.0.3.22:8080]

Order Service (client):
  1. Query registry: "where is payment-service?"
  2. Receive list of healthy instances
  3. Client picks one (round-robin, least-connections)
  4. Call directly: 10.0.2.78:8080

Used by: Netflix Eureka + Ribbon (Spring Cloud)
```

**Pattern 2: Server-side discovery (Kubernetes native)**
```
Kubernetes DNS:
  payment-service.payments.svc.cluster.local → ClusterIP 10.96.45.23
  kube-proxy load-balances to any healthy pod

Order Service:
  Call: http://payment-service.payments/ (DNS-based)
  kube-proxy resolves and routes to a healthy pod automatically

No client code needed — just use the DNS name
```

**Pattern 3: Service mesh (Istio/Linkerd)**
```
Each pod has a sidecar proxy (Envoy)
Sidecar intercepts all traffic
Service registry embedded in control plane (Istiod)

Order Service → calls http://payment-service
  ↓ intercepted by Envoy sidecar
  Envoy queries control plane: healthy payment-service pods
  Envoy applies load balancing, retries, circuit breaking
  ↓ forwards to payment-service pod
```

**Health check integration:**
```yaml
# Kubernetes readiness probe — service only in registry when healthy
readinessProbe:
  httpGet:
    path: /health/ready
    port: 8080
  initialDelaySeconds: 5
  periodSeconds: 10
  failureThreshold: 3
```

Services not passing health checks are removed from discovery automatically.

---

### Q5. What is client-side load balancing vs server-side load balancing? When do you use each?

**Answer:**

**Server-side load balancing:**
```
Client ──────────► Load Balancer (ALB/Nginx/HAProxy) ──► Server A
                   (external, shared)                 ──► Server B
                                                      ──► Server C

Client knows only the LB address; LB makes routing decisions.
```

**Client-side load balancing:**
```
Client ────────────────────────────────────────────► Server A
(contains LB logic)  ──── or ─────────────────────► Server B
                          └────────────────────────► Server C

Client queries service registry, maintains server list, picks target itself.
```

**Comparison:**

| Property | Server-side LB | Client-side LB |
|----------|---------------|----------------|
| Complexity | Low (client is simple) | Higher (client has LB logic) |
| Single point of failure | Yes (LB itself) | No |
| Scalability | LB can become bottleneck | No bottleneck |
| Visibility | LB has all metrics | Client has metrics |
| Protocol support | HTTP/TCP | Any protocol |
| Sticky sessions | Built-in | Requires state |
| Typical use | HTTP APIs, external traffic | gRPC, service mesh clients |

**gRPC specifically requires client-side LB:**
```
gRPC uses long-lived HTTP/2 connections (multiplexing).
A traditional L7 LB sees one connection → routes ALL RPCs to one backend.

Solution: gRPC client-side LB
  Client resolves "payment-service" → [addr1, addr2, addr3]
  Client maintains a pool of connections (one per backend)
  Client applies round-robin per RPC (not per connection)

// Go gRPC client-side LB
conn, err := grpc.Dial(
    "dns:///payment-service:8080",
    grpc.WithDefaultServiceConfig(`{"loadBalancingPolicy":"round_robin"}`),
)
```

**Kubernetes + service mesh (best of both):**
```
Istio Envoy sidecar = client-side LB logic, centrally managed
  - Client code stays simple (just calls service name)
  - Envoy sidecar does actual LB, retries, circuit breaking
  - Centrally configured via Istio control plane
```

---

### Q6. How do you handle partial failures in synchronous call chains?

**Answer:**

In a synchronous chain (A → B → C → D), any service failure cascades upstream. The circuit breaker + fallback + timeout pattern prevents cascading failures.

**The cascade problem:**
```
Without protection:
  User → Order Service → Payment → Inventory → Shipping
         200ms wait     TIMEOUT    (waiting)   (waiting)
         
Payment times out after 30s → Order Service holds thread for 30s
1000 concurrent users = 1000 threads held for 30s = thread pool exhausted
Order Service now also fails → cascade to User → full outage
```

**Circuit Breaker pattern:**
```
States:
  CLOSED   → Normal operation, requests pass through
  OPEN     → Failure threshold exceeded, requests fail immediately (no call made)
  HALF_OPEN → Trial: allow one request to test if downstream recovered

State transitions:
  CLOSED: track error rate
    → error_rate > 50% in last 10s: OPEN
  OPEN: fail fast
    → after 30s cooldown: HALF_OPEN
  HALF_OPEN: send one probe request
    → success: CLOSED
    → failure: OPEN (reset timer)
```

**Implementation:**
```python
import time
from enum import Enum

class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=30):
        self.state = 'CLOSED'
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.last_failure_time = None
        self.timeout = timeout
    
    def call(self, func, *args, **kwargs):
        if self.state == 'OPEN':
            if time.time() - self.last_failure_time > self.timeout:
                self.state = 'HALF_OPEN'
            else:
                raise CircuitOpenError("Circuit breaker is OPEN — failing fast")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        self.failure_count = 0
        self.state = 'CLOSED'
    
    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
```

**Timeout + fallback:**
```python
# Layered defense
async def get_product_recommendations(user_id: int) -> list:
    try:
        # Timeout: don't wait more than 200ms
        async with asyncio.timeout(0.2):
            return await recommendation_service.get(user_id)
    except (TimeoutError, CircuitOpenError, ServiceUnavailableError):
        # Fallback: return static popular items
        return await get_cached_popular_items()

# Hystrix-style (Resilience4j)
@circuit_breaker(name="recommendation-service")
@time_limiter(timeout_duration=0.2)
@fallback(method="get_cached_popular_items")
async def get_recommendations(user_id: int) -> list:
    return await recommendation_service.get(user_id)
```

---

### Q7. What is the bulkhead pattern and how does it isolate failures between services?

**Answer:**

The bulkhead pattern (named after ship bulkheads that prevent flooding from spreading) isolates thread pools, connection pools, or resources per downstream service so that one slow dependency cannot exhaust resources for all others.

**Without bulkheads:**
```
Order Service has 100 shared threads
  - Normally: 30 threads for Inventory, 30 for Payment, 40 for other
  - Inventory goes slow: requests queue up
  - Soon: all 100 threads waiting on Inventory
  - Payment, Shipping calls also fail — they have no threads available
  - Complete outage despite only one dependency being slow
```

**With bulkheads (thread pool isolation):**
```
Order Service:
  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
  │ Inventory Pool   │  │ Payment Pool     │  │ Shipping Pool    │
  │ 20 threads       │  │ 30 threads       │  │ 20 threads       │
  │ queue: 10        │  │ queue: 15        │  │ queue: 10        │
  └──────────────────┘  └──────────────────┘  └──────────────────┘
  
  Inventory goes slow → Inventory pool exhausted (30 threads, queue full)
  → Inventory requests rejected immediately (fast fail)
  → Payment and Shipping pools UNAFFECTED
  → Order Service degrades gracefully (no inventory data, but payment works)
```

**Implementation with semaphore-based bulkhead:**
```python
import asyncio

class Bulkhead:
    """Limit concurrent calls per downstream service."""
    
    def __init__(self, max_concurrent: int, max_queue: int):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.max_queue = max_queue
        self.queue_depth = 0
    
    async def execute(self, func, *args):
        if self.queue_depth >= self.max_queue:
            raise BulkheadFullError("Bulkhead queue is full — rejecting request")
        
        self.queue_depth += 1
        try:
            async with self.semaphore:
                self.queue_depth -= 1
                return await func(*args)
        except Exception:
            self.queue_depth -= 1
            raise

# Per-service bulkheads
inventory_bulkhead = Bulkhead(max_concurrent=20, max_queue=10)
payment_bulkhead   = Bulkhead(max_concurrent=30, max_queue=15)

async def reserve_inventory(order_id):
    return await inventory_bulkhead.execute(
        inventory_service.reserve, order_id
    )
```

**Kubernetes resource limits (infrastructure-level bulkhead):**
```yaml
resources:
  limits:
    cpu: "500m"
    memory: "512Mi"
  requests:
    cpu: "200m"
    memory: "256Mi"
```

Bulkheads + circuit breakers + timeouts are the three pillars of resilient synchronous communication.

---

## Medium (Q8–Q15)

---

### Q8. How do you implement request correlation and trace ID propagation across microservices?

**Answer:**

Trace IDs link all log entries, spans, and events belonging to a single user request across all services, enabling end-to-end debugging.

**The problem without correlation:**
```
Order Service log:  [ERROR] Payment failed for order 42
Payment Service log: [ERROR] Card declined for user 99
Which error in Payment corresponds to which error in Order? Unknown.
```

**W3C Trace Context standard (recommended):**
```
HTTP Header: traceparent: 00-{trace-id}-{span-id}-{flags}
Example:     traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01
                             |trace-id (32 hex chars) |span-id (16)|sampling=01
```

**Implementation — middleware in each service:**
```python
import uuid
import contextvars

# Context variable for current trace (async-safe)
trace_context: contextvars.ContextVar = contextvars.ContextVar('trace_context')

class TraceMiddleware:
    async def __call__(self, request, call_next):
        # Extract incoming trace ID or generate new one
        traceparent = request.headers.get('traceparent')
        
        if traceparent:
            trace_id, parent_span_id = parse_traceparent(traceparent)
        else:
            trace_id = uuid.uuid4().hex  # Root of new trace
            parent_span_id = None
        
        span_id = uuid.uuid4().hex[:16]  # This service's span
        
        # Store in context (available to all code in this request)
        trace_context.set({
            'trace_id': trace_id,
            'span_id': span_id,
            'parent_span_id': parent_span_id
        })
        
        # Log with trace context
        logger.info(f"Request started trace_id={trace_id} span_id={span_id}")
        
        response = await call_next(request)
        
        logger.info(f"Request completed trace_id={trace_id} status={response.status_code}")
        return response

# HTTP client: propagate trace to downstream services
class TracingHTTPClient:
    async def post(self, url, data):
        ctx = trace_context.get()
        headers = {
            'traceparent': f"00-{ctx['trace_id']}-{ctx['span_id']}-01"
        }
        return await http_client.post(url, data=data, headers=headers)
```

**Structured logging with trace context:**
```python
import structlog

def get_logger():
    ctx = trace_context.get({})
    return structlog.get_logger().bind(
        trace_id=ctx.get('trace_id'),
        span_id=ctx.get('span_id'),
        service="order-service"
    )

# Usage
log = get_logger()
log.info("order_created", order_id=42, user_id=99)
# Output: {"event": "order_created", "order_id": 42, "trace_id": "4bf92...", "span_id": "a1b2..."}
```

**Querying correlated logs across services:**
```
Kibana/Grafana Loki query:
  {trace_id="4bf92f3577b34da6a3ce929d0e0e4736"}
  
Returns all log lines from ALL services for that single user request,
ordered by timestamp — instant root cause analysis.
```

**OpenTelemetry (standard library):**
```python
from opentelemetry import trace
from opentelemetry.propagate import inject, extract

tracer = trace.get_tracer("order-service")

async def process_order(request):
    # Extract context from incoming headers
    ctx = extract(request.headers)
    
    with tracer.start_as_current_span("process_order", context=ctx) as span:
        span.set_attribute("order.id", order_id)
        span.set_attribute("user.id", user_id)
        
        # Downstream call automatically propagates trace
        await payment_service.charge(order_id)
```

---

### Q9. Explain the fan-out aggregation pattern (BFF / API gateway fetching from N services in parallel).

**Answer:**

Fan-out aggregation fetches data from multiple services concurrently and combines the results, avoiding multiple round trips from the client.

**Problem without fan-out:**
```
Mobile client makes 5 sequential calls:
  1. GET /user-service/profile      → 80ms
  2. GET /order-service/orders      → 120ms
  3. GET /notification-service/feed → 100ms
  4. GET /product-service/wishlist  → 90ms
  5. GET /recommendation-service    → 200ms
  
Total: 590ms + 5 × network RTT from client (5 × 50ms = 250ms) = 840ms
```

**BFF (Backend For Frontend) fan-out:**
```
Mobile client → BFF (one call)
                    ↓ parallel
              ┌─────┼──────┬──────┬──────────────┐
              ▼     ▼      ▼      ▼              ▼
           User  Orders  Notifs  Wishlist  Recommendations
           (80ms)(120ms)(100ms) (90ms)    (200ms)
              └─────┴──────┴──────┴──────────────┘
              ↓ aggregate
              BFF returns combined response
              
Total: max(80,120,100,90,200) = 200ms + 1 × client RTT = 250ms
Speed-up: 840ms → 250ms = 3.4x faster
```

**Implementation:**
```python
import asyncio
from dataclasses import dataclass

@dataclass
class HomePageResponse:
    profile: dict | None
    recent_orders: list | None
    notifications: list | None
    recommendations: list | None

async def get_home_page(user_id: int) -> HomePageResponse:
    """Fan-out: fetch from all services in parallel."""
    
    # Launch all requests concurrently
    results = await asyncio.gather(
        user_service.get_profile(user_id),
        order_service.get_recent_orders(user_id, limit=5),
        notification_service.get_unread(user_id, limit=10),
        recommendation_service.get_for_user(user_id, limit=20),
        return_exceptions=True  # Don't fail all if one fails
    )
    
    profile, orders, notifications, recommendations = results
    
    # Graceful degradation: partial response if some services fail
    return HomePageResponse(
        profile=profile if not isinstance(profile, Exception) else None,
        recent_orders=orders if not isinstance(orders, Exception) else [],
        notifications=notifications if not isinstance(notifications, Exception) else [],
        recommendations=recommendations if not isinstance(recommendations, Exception) else []
    )
```

**With timeout per service:**
```python
async def get_with_timeout(coro, timeout_sec: float, default):
    try:
        return await asyncio.wait_for(coro, timeout=timeout_sec)
    except (asyncio.TimeoutError, Exception) as e:
        logger.warning(f"Fan-out call failed/timed out: {e}")
        return default

async def get_home_page(user_id: int) -> HomePageResponse:
    profile, orders, notifications, recommendations = await asyncio.gather(
        get_with_timeout(user_service.get_profile(user_id), 0.3, None),
        get_with_timeout(order_service.get_recent_orders(user_id), 0.5, []),
        get_with_timeout(notification_service.get_unread(user_id), 0.3, []),
        get_with_timeout(recommendation_service.get_for_user(user_id), 0.4, []),
    )
    return HomePageResponse(profile, orders, notifications, recommendations)
```

**GraphQL DataLoader (N+1 batching within fan-out):**
```javascript
// Without DataLoader: 1 query per order to get user info (N+1)
// With DataLoader: batch all user IDs into one query
const userLoader = new DataLoader(async (userIds) => {
    const users = await userService.getBatch(userIds);
    return userIds.map(id => users.find(u => u.id === id));
});
```

---

### Q10. How do you handle versioning of internal APIs between microservices?

**Answer:**

Internal API versioning prevents breaking changes when one service evolves and not all callers update simultaneously.

**The versioning problem:**
```
Order Service calls: POST /payments/v1/charge { user_id, amount }
Payment Service upgrades API: POST /payments/v2/charge { user_id, amount, currency }
  
v2 makes currency required → v1 callers break
Rolling deploy: half the Payment pods are v2, half v1 → 50% of requests fail
```

**Strategy 1: URL versioning**
```
/v1/payments/charge  → v1 handler
/v2/payments/charge  → v2 handler (new currency field)

Order Service continues calling /v1 until it's updated
Both versions run simultaneously during transition
v1 deprecated with notice period (6 weeks minimum)
```

**Strategy 2: Header versioning (cleaner URLs)**
```
POST /payments/charge
Headers: Accept: application/vnd.company.v2+json
         API-Version: 2025-05-01

Router dispatches based on version header
URL stays clean: /payments/charge (no v1/v2 in path)
```

**Strategy 3: Protocol Buffers backward compatibility (gRPC)**
```protobuf
// v1 message
message ChargeRequest {
    int64 user_id = 1;
    double amount = 2;
}

// v2 message (backward compatible — adds optional field)
message ChargeRequest {
    int64 user_id = 1;
    double amount = 2;
    string currency = 3;  // Optional — defaults to "" (treat as "USD")
    // NEVER reuse field numbers
    // NEVER remove fields from a published message
    // NEVER change field types
}
```

Protobuf rules:
- Old clients sending to new server: unknown fields (currency) are ignored by old clients; new server sees empty string → defaults to USD
- New clients sending to old server: unknown fields ignored safely

**Strategy 4: Event schema versioning**
```json
// Every event carries its schema version
{
  "event_type": "order.placed",
  "schema_version": "2",
  "event_id": "abc-123",
  "data": {
    "order_id": 42,
    "user_id": 99,
    "currency": "EUR"
  }
}

// Consumer handles multiple versions:
def handle_order_placed(event):
    version = event['schema_version']
    if version == '1':
        data = migrate_v1_to_v2(event['data'])
    else:
        data = event['data']
    process_order(data)
```

**API versioning lifecycle:**
```
1. Design new version (v2)
2. Deploy v2 alongside v1 (no callers use v2 yet)
3. Migrate callers to v2 (team by team, with validation)
4. Deprecate v1 (announce with timeline — 6-12 weeks)
5. Remove v1 (after all callers confirmed migrated)
```

---

### Q11. What is consumer-driven contract testing with Pact? How does it work?

**Answer:**

Consumer-driven contract testing (CDCT) lets service consumers define what they expect from a provider, then automatically verify the provider meets those expectations — without spinning up all services together.

**Traditional integration testing problem:**
```
Test environment: Deploy all 15 microservices + 5 databases
  Time to setup: 45 minutes
  Flakiness: One service fails → whole test suite fails
  Feedback loop: 1 hour per CI run
```

**Pact workflow:**
```
Step 1: Consumer writes a pact (contract)
  Order Service (consumer) says:
  "When I call POST /payments/charge with {user_id: 1, amount: 50},
   I expect a 200 response with {payment_id: string, status: 'approved'}"

Step 2: Pact generates a mock server
  Tests run against mock → fast, no real Payment Service needed

Step 3: Pact publishes the contract to Pact Broker
  Contract: "order-service v1.5 → payment-service v1.x"

Step 4: Provider verifies the contract
  Payment Service downloads the contract
  Pact replays the consumer's recorded request against the REAL Payment Service
  Checks: Does the real response match what Order Service expects?

Step 5: Pass/fail + can-i-deploy
  pact-broker can-i-deploy --pacticipant payment-service --version 2.0.0
  → FAIL: contract broken for order-service
```

**Consumer test (Pact Python):**
```python
import pytest
from pact import Consumer, Provider

pact = Consumer("OrderService").has_pact_with(Provider("PaymentService"))

def test_payment_charge():
    expected = {
        "payment_id": "abc-123",
        "status": "approved"
    }
    
    (pact
     .given("user 1 has a valid credit card")
     .upon_receiving("a charge request")
     .with_request("POST", "/payments/charge",
                   body={"user_id": 1, "amount": 50.00},
                   headers={"Content-Type": "application/json"})
     .will_respond_with(200, body=expected))
    
    with pact:
        # Order Service's actual code runs against Pact mock
        response = payment_client.charge(user_id=1, amount=50.00)
        assert response.status == "approved"

# Pact saves the interaction to: pacts/OrderService-PaymentService.json
```

**Provider verification:**
```python
# Payment Service CI: verify all consumer contracts
def test_payment_service_fulfills_contracts():
    pact = Verifier(
        provider="PaymentService",
        provider_base_url="http://localhost:8000"
    )
    
    output, _ = pact.verify_with_broker(
        broker_url="https://pact-broker.internal",
        provider_states_setup_url="http://localhost:8000/provider-states",
        publish_verification_results=True,
        provider_version="2.0.0"
    )
    assert output == 0, "Pact verification failed — consumer contract broken"
```

**Benefits over integration tests:**
- No other services needed for consumer tests
- Provider tests are fast (no external dependencies)
- Clear ownership: consumers own what they need, providers verify they deliver it
- Versioned contracts: know exactly which consumer version requires what

---

### Q12. What is the saga pattern for distributed transactions? Explain orchestration vs choreography.

**Answer:**

A saga is a sequence of local transactions, each publishing an event or message that triggers the next step. If a step fails, compensating transactions undo the previous steps.

**The distributed transaction problem:**
```
Order flow: Reserve Inventory → Charge Payment → Create Shipment

Two-Phase Commit (2PC) problems:
  - Blocking: all services locked during prepare phase
  - Single point of failure: coordinator crash leaves services stuck
  - Poor availability in microservices

Saga solution: Each step is its own local transaction with a compensating action
```

**Orchestration Saga (central coordinator):**
```
Saga Orchestrator (state machine):
  
  STARTED
    → INVENTORY_RESERVED
      → PAYMENT_CHARGED
        → SHIPMENT_CREATED → SUCCESS
        ↓ (fail)
      → COMPENSATE: Refund payment
      ← PAYMENT_REFUNDED
    → COMPENSATE: Release inventory
    ← INVENTORY_RELEASED
  → FAILED

Implementation:
  Orchestrator sends commands, receives replies
  State stored in DB (resumable after crash)
```

```python
class OrderSaga:
    def __init__(self, saga_id: str, order_data: dict):
        self.saga_id = saga_id
        self.state = 'STARTED'
        self.order_data = order_data
    
    async def execute(self):
        try:
            # Step 1: Reserve inventory
            self.state = 'RESERVING_INVENTORY'
            await inventory_service.reserve(self.order_data['items'])
            self.state = 'INVENTORY_RESERVED'
            
            # Step 2: Charge payment
            self.state = 'CHARGING_PAYMENT'
            payment = await payment_service.charge(self.order_data['payment'])
            self.state = 'PAYMENT_CHARGED'
            
            # Step 3: Create shipment
            self.state = 'CREATING_SHIPMENT'
            await shipment_service.create(self.order_data)
            self.state = 'COMPLETED'
            
        except PaymentFailed:
            # Compensate: release inventory
            await inventory_service.release(self.order_data['items'])
            self.state = 'FAILED'
        
        except ShipmentFailed:
            # Compensate both
            await payment_service.refund(payment.id)
            await inventory_service.release(self.order_data['items'])
            self.state = 'FAILED'
```

**Choreography Saga (event-driven, no central coordinator):**
```
Order Service      → publishes: order_created
                   
Inventory Service  → consumes: order_created
                   → publishes: inventory_reserved (or: inventory_failed)

Payment Service    → consumes: inventory_reserved
                   → publishes: payment_charged (or: payment_failed)

Shipment Service   → consumes: payment_charged
                   → publishes: shipment_created

If payment_failed:
  Inventory Service → consumes: payment_failed → publishes: inventory_released
```

**Orchestration vs Choreography:**

| Dimension | Orchestration | Choreography |
|-----------|--------------|--------------|
| Coupling | Services coupled to orchestrator | Services coupled only to events |
| Visibility | Centralized — easy to see state | Distributed — hard to track flow |
| Complexity | Concentrated in orchestrator | Distributed across services |
| Failure handling | Orchestrator manages compensation | Each service reacts to failure events |
| Testing | Test orchestrator as unit | Need end-to-end event testing |
| Best for | Complex, long-running sagas | Simple, few-step flows |

---

### Q13. What is request coalescing and how does the DataLoader pattern implement it?

**Answer:**

Request coalescing (batching) combines multiple individual requests that arrive in the same time window into one bulk request to the downstream service, reducing N+1 query problems.

**The N+1 problem:**
```
GraphQL query: Get 20 orders, each with user details
  
Without batching:
  Query 1: SELECT * FROM orders LIMIT 20;        → 20 order IDs
  Query 2: SELECT * FROM users WHERE id = 1;     → user for order 1
  Query 3: SELECT * FROM users WHERE id = 2;     → user for order 2
  ...
  Query 21: SELECT * FROM users WHERE id = 20;   → user for order 20
  
Total: 21 queries. For 100 orders: 101 queries. Terrible.

With batching (DataLoader):
  Query 1: SELECT * FROM orders LIMIT 20;
  Query 2: SELECT * FROM users WHERE id IN (1,2,3,...,20);  ← ONE query
  
Total: 2 queries. Always.
```

**DataLoader pattern (Facebook's implementation):**
```javascript
const DataLoader = require('dataloader');

// Batch function: receives array of IDs, returns array of values
const userLoader = new DataLoader(async (userIds) => {
    // One DB query for all IDs
    const users = await db.query(
        'SELECT * FROM users WHERE id = ANY($1)',
        [userIds]
    );
    
    // MUST return in same order as input IDs
    const userMap = new Map(users.map(u => [u.id, u]));
    return userIds.map(id => userMap.get(id) || null);
});

// Usage: each call schedules a load, DataLoader coalesces into one batch
// These three calls within the same event loop tick get batched:
const user1 = await userLoader.load(1);   // batch!
const user2 = await userLoader.load(2);   // batch!
const user3 = await userLoader.load(100); // batch!
// Under the hood: one call to batchFn([1, 2, 100])
```

**Python equivalent:**
```python
class UserDataLoader:
    def __init__(self):
        self._pending: dict[int, asyncio.Future] = {}
        self._scheduled = False
    
    async def load(self, user_id: int):
        future = asyncio.get_event_loop().create_future()
        self._pending[user_id] = future
        
        if not self._scheduled:
            self._scheduled = True
            asyncio.get_event_loop().call_soon(self._dispatch)
        
        return await future
    
    def _dispatch(self):
        ids = list(self._pending.keys())
        futures = list(self._pending.values())
        self._pending.clear()
        self._scheduled = False
        
        asyncio.ensure_future(self._batch_load(ids, futures))
    
    async def _batch_load(self, ids: list[int], futures: list):
        users = await db.fetch("SELECT * FROM users WHERE id = ANY($1)", ids)
        user_map = {u['id']: u for u in users}
        
        for user_id, future in zip(ids, futures):
            future.set_result(user_map.get(user_id))
```

**Cache layer on DataLoader:**
```javascript
// DataLoader caches results within a single request
// Second call to load(42) returns cached value from first call
// Per-request cache (not cross-request — no stale data)
const userLoader = new DataLoader(batchFn, {
    cache: true,           // Default: true
    cacheKeyFn: (key) => key.toString()
});
```

---

### Q14. How do you implement timeout budgets across a chain of service calls?

**Answer:**

A timeout budget ensures that downstream calls respect the remaining time available in an end-to-end request deadline, preventing timeouts that fire after the parent request has already timed out.

**Problem with fixed timeouts:**
```
User request timeout: 2 seconds total

Order Service → Payment Service (timeout: 3s) → Bank API (timeout: 5s)
                                                   
Bank API takes 4s → Payment times out at 5s → Order Service fails at 3s
User already failed at 2s → Payment and Bank wasted their resources for 3 extra seconds
```

**Deadline propagation (gRPC model):**
```python
# gRPC deadline — absolute time when request must complete
import grpc
from datetime import datetime, timedelta

def handle_order_request(user_request):
    # User has 2-second total budget
    deadline = datetime.utcnow() + timedelta(seconds=2)
    
    # Check if we have time to proceed
    remaining = (deadline - datetime.utcnow()).total_seconds()
    if remaining < 0.1:  # Less than 100ms remaining
        raise DeadlineExceededError("No time remaining in budget")
    
    # Pass deadline to downstream gRPC call
    channel = grpc.insecure_channel('payment-service:50051')
    stub = PaymentStub(channel)
    
    metadata = [('grpc-timeout', f'{int(remaining * 1000)}m')]  # milliseconds
    response = stub.Charge(request, timeout=remaining, metadata=metadata)
    
    return response
```

**HTTP header approach (custom):**
```python
class TimeoutBudget:
    """Tracks and propagates remaining request deadline."""
    
    def __init__(self, total_seconds: float):
        self.deadline = time.time() + total_seconds
    
    @property
    def remaining(self) -> float:
        return max(0, self.deadline - time.time())
    
    @property
    def is_expired(self) -> bool:
        return self.remaining <= 0
    
    def to_headers(self) -> dict:
        return {'X-Request-Deadline': str(self.deadline)}
    
    @classmethod
    def from_headers(cls, headers: dict) -> 'TimeoutBudget':
        deadline = float(headers.get('X-Request-Deadline', time.time() + 30))
        budget = cls(0)
        budget.deadline = deadline
        return budget

# In each service's middleware:
async def call_downstream(service, budget: TimeoutBudget):
    if budget.is_expired:
        raise TimeoutError("Deadline exceeded before downstream call")
    
    # Reserve 50ms for our own processing overhead
    downstream_timeout = budget.remaining - 0.05
    
    async with asyncio.timeout(downstream_timeout):
        response = await service.call(headers=budget.to_headers())
    
    return response
```

**Budget allocation strategy:**
```
Total budget: 2000ms
  ├── Auth middleware: 10ms
  ├── DB read (with cache): 50ms
  ├── Payment service call: 800ms (largest allocation to critical path)
  ├── Notification service call: 200ms (can fail without user impact)
  ├── Response serialization: 20ms
  └── Network buffer: 920ms (reserve for unexpected latency)

Rule: Sum of allocations < total budget
      Each call uses min(allocated_time, remaining_budget)
```

---

### Q15. How do you migrate from synchronous to asynchronous communication without downtime?

**Answer:**

Migrating from sync to async requires a transition period where both patterns coexist, ensuring no messages are lost and callers are not broken.

**Before state:**
```
Order Service ──── HTTP POST /email-service/send ──► Email Service
              ◄─── 200 OK ────────────────────────────
(synchronous, tightly coupled)
```

**Target state:**
```
Order Service ──── publish order.placed ──► Kafka ──► Email Service
(asynchronous, decoupled)
```

**Migration strategy: Strangler Fig + Dual Write**

**Phase 1: Add async path alongside sync (no service changes)**
```python
# Order Service: add async publish WITHOUT removing sync call
async def place_order(order: Order):
    # Existing sync call (still works)
    await email_service.send_confirmation_http(order)
    
    # NEW: also publish event (Email Service not yet consuming this)
    await kafka.publish('order.placed', {
        'order_id': order.id,
        'user_email': order.user.email
    })
    
    return order
```

**Phase 2: Email Service subscribes to events (dual consumption)**
```python
# Email Service: consume from Kafka AND keep HTTP endpoint running
class EmailConsumer:
    async def handle_order_placed(self, event):
        await self.send_confirmation_email(
            to=event['user_email'],
            order_id=event['order_id']
        )

# HTTP endpoint still active (sync callers still work)
@app.post("/email-service/send")
async def send_http(request):
    await send_confirmation_email(request.email, request.order_id)
    return {"status": "sent"}
```

**Phase 3: Validate async path under production traffic**
```
Monitoring for ~1 week:
  - Are Kafka events being consumed?
  - Email delivery rate same as before?
  - No duplicate emails? (implement idempotency key check)
  - Consumer lag acceptable?
  
# Idempotency in Email Service to prevent duplicates
async def handle_order_placed(event):
    idempotency_key = f"order_placed:{event['order_id']}"
    if await redis.set(idempotency_key, 1, nx=True, ex=86400):
        # Only process if key was newly set (not duplicate)
        await send_email(event)
```

**Phase 4: Remove sync call from Order Service**
```python
async def place_order(order: Order):
    # Sync call REMOVED
    # await email_service.send_confirmation_http(order)  ← deleted
    
    # Only async path remains
    await kafka.publish('order.placed', {
        'order_id': order.id,
        'user_email': order.user.email
    })
    
    return order
```

**Phase 5: Deprecate HTTP endpoint (after monitoring confirms nothing calls it)**
```
Check access logs for /email-service/send
Zero traffic for 2 weeks → safe to remove
Deploy Email Service without HTTP endpoint
```

**Rollback plan at each phase:**
- Phase 1: Remove kafka.publish line — zero impact
- Phase 2: Stop consumer — HTTP still working
- Phase 3: If dual emails: add dedup check
- Phase 4: Re-add sync call + drain event queue

---

## Hard (Q16–Q20)

---

### Q16. Design a service mesh architecture for 50 microservices with mTLS, traffic management, and observability.

**Answer:**

A service mesh provides cross-cutting concerns (security, observability, reliability) for microservices at the infrastructure level, without changing application code.

**Architecture:**
```
Control Plane (Istiod):
  ┌─────────────────────────────────────────────────────────┐
  │  Pilot (traffic routing config)                         │
  │  Citadel (certificate management / mTLS)                │
  │  Galley (config validation)                             │
  └────────────────────────┬────────────────────────────────┘
                           │ xDS API (push config)
Data Plane (Envoy sidecars):
  ┌─────────────────┐     ┌─────────────────┐
  │  Pod A          │     │  Pod B          │
  │  ┌───────────┐  │     │  ┌───────────┐  │
  │  │ App       │  │     │  │ App       │  │
  │  ├───────────┤  │     │  ├───────────┤  │
  │  │ Envoy     │◄─┼─────┼─►│ Envoy     │  │
  │  │ (sidecar) │  │mTLS │  │ (sidecar) │  │
  │  └───────────┘  │     │  └───────────┘  │
  └─────────────────┘     └─────────────────┘
```

**mTLS configuration (mutual TLS):**
```yaml
# Require mTLS for all services in the payments namespace
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: require-mtls
  namespace: payments
spec:
  mtls:
    mode: STRICT  # Reject all non-mTLS traffic

---
# Allow payments → order-service (authorization policy)
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: allow-payments-to-orders
  namespace: orders
spec:
  selector:
    matchLabels:
      app: order-service
  rules:
  - from:
    - source:
        principals: ["cluster.local/ns/payments/sa/payment-service"]
    to:
    - operation:
        methods: ["POST"]
        paths: ["/orders/*/fulfill"]
```

**Traffic management — canary release:**
```yaml
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: payment-service-routing
spec:
  hosts:
  - payment-service
  http:
  - match:
    - headers:
        x-canary:
          exact: "true"
    route:
    - destination:
        host: payment-service
        subset: v2
  - route:
    - destination:
        host: payment-service
        subset: v1
      weight: 95
    - destination:
        host: payment-service
        subset: v2
      weight: 5

---
apiVersion: networking.istio.io/v1alpha3
kind: DestinationRule
metadata:
  name: payment-service
spec:
  host: payment-service
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100
      http:
        http1MaxPendingRequests: 50
        maxRetries: 3
    outlierDetection:  # Circuit breaking via Envoy
      consecutiveGatewayErrors: 5
      interval: 30s
      baseEjectionTime: 30s
  subsets:
  - name: v1
    labels:
      version: v1
  - name: v2
    labels:
      version: v2
```

**Observability (automatic from sidecar):**
```
Every service automatically gets:
  Metrics: istio_request_total, istio_request_duration_milliseconds
  Traces:  Automatic Jaeger trace context injection (just add Zipkin headers)
  Access logs: All service-to-service traffic logged with trace ID

Grafana dashboard query:
  sum(rate(istio_requests_total{destination_service="payment-service",
           response_code=~"5.."}[5m]))
  /
  sum(rate(istio_requests_total{destination_service="payment-service"}[5m]))
  = error rate without any app-level code
```

---

### Q17. Design an event-driven architecture for an e-commerce system where services communicate only via events, with schema registry and backward compatibility.

**Answer:**

A fully event-driven architecture eliminates synchronous dependencies between services, improving resilience and scalability.

**Architecture:**
```
Services                Events                  Consumers
──────────              ──────────              ──────────
Order Service  ─┐       order.placed            Payment Service
               │──────► order.fulfilled  ──────► Shipment Service
               └──────► order.cancelled          Notification Service
                                                 Analytics Service
Payment Service ──────► payment.charged  ──────► Order Service
                        payment.failed           Notification Service

Schema Registry (Confluent/AWS Glue):
  Stores: All event schemas (Avro/Protobuf)
  Enforces: Compatibility rules (BACKWARD/FORWARD/FULL)
  Versions: Each schema change creates a new version
```

**Schema definition (Avro):**
```json
{
  "type": "record",
  "name": "OrderPlaced",
  "namespace": "com.company.orders.v1",
  "fields": [
    {"name": "event_id",   "type": "string"},
    {"name": "order_id",   "type": "long"},
    {"name": "user_id",    "type": "long"},
    {"name": "total",      "type": "double"},
    {"name": "items",      "type": {
      "type": "array",
      "items": {
        "type": "record",
        "name": "OrderItem",
        "fields": [
          {"name": "product_id", "type": "long"},
          {"name": "quantity",   "type": "int"},
          {"name": "price",      "type": "double"}
        ]
      }
    }},
    {"name": "created_at", "type": "long", "logicalType": "timestamp-millis"}
  ]
}
```

**Schema evolution (adding currency field — BACKWARD compatible):**
```json
{
  "name": "currency",
  "type": {"type": "string", "default": "USD"},
  "default": "USD"
}
```

Backward compatible: old consumers can read new events (they ignore currency).
Forward compatible: new consumers can read old events (they see default "USD").

**Producer with schema registry:**
```python
from confluent_kafka import SerializingProducer
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroSerializer

schema_registry = SchemaRegistryClient({"url": "https://schema-registry:8081"})

avro_serializer = AvroSerializer(
    schema_registry,
    schema_str=ORDER_PLACED_SCHEMA,
    conf={"auto.register.schemas": False}  # Don't auto-register; only from CI
)

producer = SerializingProducer({
    "bootstrap.servers": "kafka:9092",
    "value.serializer": avro_serializer
})

def publish_order_placed(order: Order):
    event = {
        "event_id": str(uuid.uuid4()),
        "order_id": order.id,
        "user_id": order.user_id,
        "total": float(order.total),
        "items": [{"product_id": i.product_id, "quantity": i.qty, "price": float(i.price)}
                  for i in order.items],
        "created_at": int(datetime.utcnow().timestamp() * 1000)
    }
    producer.produce(topic="order.placed", value=event,
                     key=str(order.id))
    producer.flush()
```

**Consumer with dead letter queue:**
```python
async def consume_order_placed():
    consumer = DeserializingConsumer({...})
    consumer.subscribe(["order.placed"])
    
    while True:
        msg = consumer.poll(timeout=1.0)
        if msg is None:
            continue
        
        try:
            event = msg.value()
            await process_order_placed(event)
            consumer.commit()  # Commit only after successful processing
            
        except ValidationError as e:
            # Schema error: send to DLQ
            await publish_to_dlq("order.placed.dlq", msg.value(), error=str(e))
            consumer.commit()
            
        except RetryableError as e:
            # Don't commit: will be retried on next poll
            logger.warning(f"Retryable error: {e}")
            await asyncio.sleep(5)
```

---

### Q18. Design a backward-compatible API change for a payment service with 20 consumer services.

**Answer:**

With 20 consumers, any breaking change creates a coordination nightmare. The solution is a fully backward-compatible rollout using the expand-contract pattern combined with dual support periods.

**Scenario:** Payment Service needs to change `amount` (float) to `{amount, currency}` object.

```
Current API:
  POST /charges
  { "user_id": 42, "amount": 99.99 }
  
Target API:
  POST /charges
  { "user_id": 42, "amount": { "value": 99.99, "currency": "USD" } }
```

**Phase 1: Accept BOTH formats (never break consumers)**
```python
from pydantic import BaseModel, validator
from typing import Union

class AmountV1(BaseModel):
    """Legacy format: flat float"""
    pass

class AmountV2(BaseModel):
    """New format: structured"""
    value: float
    currency: str = "USD"

class ChargeRequest(BaseModel):
    user_id: int
    # Accept both: float (v1) or dict (v2)
    amount: Union[float, AmountV2]
    
    @validator('amount', pre=True)
    def normalize_amount(cls, v):
        if isinstance(v, (int, float)):
            # Legacy v1: wrap in new structure
            return AmountV2(value=float(v), currency="USD")
        return v

@app.post("/charges")
async def create_charge(request: ChargeRequest):
    # Internal code always uses v2 structure
    charge = await payment_processor.charge(
        user_id=request.user_id,
        amount=request.amount.value,
        currency=request.amount.currency
    )
    return {"payment_id": charge.id, "status": charge.status}
```

**Phase 2: New response includes both formats**
```python
class ChargeResponse(BaseModel):
    payment_id: str
    status: str
    # Legacy field (keep for v1 consumers)
    amount: float             # Old consumers read this
    # New field (v2 consumers use this)
    charged_amount: AmountV2  # New consumers prefer this
    
# Both fields populated:
return ChargeResponse(
    payment_id=charge.id,
    status=charge.status,
    amount=float(charge.amount),                           # backward compat
    charged_amount=AmountV2(value=float(charge.amount),    # new
                            currency=charge.currency)
)
```

**Phase 3: Communicate migration to 20 consumers**
```
Migration communication:
  1. Publish to #api-changes Slack: "Payment API: amount field changing structure"
  2. Update OpenAPI docs with deprecation notice on flat amount field
  3. Add deprecation header to responses:
     Deprecation: Sat, 31 Dec 2025 23:59:59 GMT
     Sunset: Sat, 31 Dec 2025 23:59:59 GMT
     Link: <https://docs.internal/migration>; rel="deprecation"
  4. Dashboard: track how many consumers still use v1 format
     (log 'amount_format': 'v1' or 'v2' from request parsing)
```

**Phase 4: Monitor adoption**
```python
# Metrics on v1 vs v2 usage
@app.middleware("http")
async def track_api_version(request: Request, call_next):
    body = await request.json()
    version = 'v1' if isinstance(body.get('amount'), (int, float)) else 'v2'
    
    metrics.increment('charge_api_version', tags={'version': version,
                                                   'consumer': request.headers.get('X-Service-Name')})
    return await call_next(request)
```

**Phase 5: Hard cutoff (only after all 20 consumers migrated)**
```python
# Check: are all consumers using v2?
# Metrics show 0 v1 requests for 4 weeks → safe to remove
# Remove v1 support:
class ChargeRequest(BaseModel):
    user_id: int
    amount: AmountV2  # Only v2 accepted now
```

---

### Q19. How do you design a resilient microservice that handles 10x traffic spikes without SLO violations?

**Answer:**

Handling traffic spikes requires a combination of horizontal scaling, graceful degradation, load shedding, and caching — designed in layers.

**Spike scenario:**
```
Normal: 1,000 RPS
Spike:  10,000 RPS (flash sale, viral event)
Goal: No SLO violation (< 0.1% errors, P99 < 500ms)
```

**Layer 1: Autoscaling with pre-warming**
```yaml
# Kubernetes HPA with KEDA (queue-based scaling)
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: order-service-scaler
spec:
  scaleTargetRef:
    name: order-service
  minReplicaCount: 10   # Never go below 10 (warm baseline)
  maxReplicaCount: 100  # Up to 100 during spike
  triggers:
  - type: prometheus
    metadata:
      serverAddress: http://prometheus:9090
      metricName: http_requests_per_second
      threshold: "100"  # Scale out when > 100 RPS per pod
      query: |
        sum(rate(http_requests_total{service="order-service"}[1m]))
        / count(kube_pod_info{pod=~"order-service-.*"})
```

**Layer 2: Circuit breaker + load shedding in the service**
```python
class OrderService:
    def __init__(self):
        self.admission = AdaptiveAdmissionController()
        self.circuit_breakers = {
            'payment': CircuitBreaker(threshold=0.1, window=10),
            'inventory': CircuitBreaker(threshold=0.1, window=10)
        }
    
    async def handle_request(self, request: OrderRequest):
        # Admission control: shed load based on current CPU/queue
        if not await self.admission.should_admit(request.priority):
            raise TooManyRequestsError(
                retry_after=self.admission.estimated_wait_seconds()
            )
        
        # Process with circuit-broken downstream calls
        result = await self._process_with_fallbacks(request)
        return result
    
    async def _process_with_fallbacks(self, request):
        # Payment: critical, no fallback
        payment = await self.circuit_breakers['payment'].call(
            payment_service.charge, request.payment_details
        )
        
        # Inventory: can proceed with async reservation if service is slow
        try:
            inventory = await asyncio.wait_for(
                inventory_service.reserve(request.items), timeout=0.5
            )
        except asyncio.TimeoutError:
            # Fallback: async inventory reservation
            await kafka.publish('inventory.reserve', request.items)
            inventory = {"status": "pending"}
        
        return OrderResponse(payment=payment, inventory=inventory)
```

**Layer 3: Caching at multiple levels**
```
                                              Cache hit rate
Browser cache (assets):    CDN              99% (static)
API response cache:        Redis (5s TTL)   70% (product catalog)
DB query cache:            PgBouncer        60% (hot rows)
Compute cache:             Memoize          40% (pricing rules)

Spike scenario analysis:
  10,000 RPS × 70% cache hit = 7,000 served from cache (no DB)
  3,000 RPS hit DB (manageable)
```

**Layer 4: Queue-based smoothing for write-heavy operations**
```python
# Instead of synchronous DB write per request:
async def place_order(order: OrderRequest) -> PlaceOrderResponse:
    # Generate order ID immediately (for user response)
    order_id = generate_snowflake_id()
    
    # Return immediately — order processing is async
    await kafka.publish('order.submitted', {
        'order_id': order_id,
        'order_data': order.dict()
    })
    
    return PlaceOrderResponse(
        order_id=order_id,
        status='processing',
        check_status_url=f"/orders/{order_id}/status"
    )

# Consumer (separate service, scales independently):
async def process_order_submitted(event):
    order = await db.insert_order(event['order_data'])
    await kafka.publish('order.created', {'order_id': order.id})
```

**Spike readiness checklist:**
```
□ Autoscaling configured with appropriate min/max
□ Startup time < 10s (pre-warmed containers, connection pool)
□ Load shedding at 80% CPU (return 503 gracefully)
□ Circuit breakers on all downstream calls
□ Cache warmed before event
□ Feature flags to disable non-critical features
□ Capacity test run at 2x expected peak
□ Oncall briefed, runbooks updated, freeze on deploys
```

---

### Q20. Design the complete communication architecture for a ride-sharing platform (driver matching, real-time tracking, pricing, payments).

**Answer:**

Ride-sharing has distinct communication requirements: real-time (driver location), synchronous (pricing, booking), and async (payment, notifications).

**Domain services:**
```
Driver Service    — driver availability, location, status
Rider Service     — rider profile, trips history
Matching Service  — match driver to rider (algorithm)
Pricing Service   — surge pricing calculation
Trip Service      — trip lifecycle management
Payment Service   — charge rider, pay driver
Notification Svc  — push, SMS, email
Location Service  — real-time GPS tracking
Maps Service      — routing, ETA calculation
```

**Communication matrix:**

```
Interaction                        Pattern        Protocol   Latency Budget
───────────────────────────────────────────────────────────────────────────
Driver app → Location Service      Push            WebSocket  Real-time
Location Service → Matching Svc    Pub/Sub         Kafka      < 1s
Rider requests trip → Trip Svc     Request/Reply   REST/gRPC  < 500ms
Trip Svc → Matching Svc (find)     Sync            gRPC       < 200ms
Trip Svc → Pricing Svc             Sync            gRPC       < 100ms
Trip Svc → Payment Svc (charge)    Async + verify  Kafka+gRPC < 5s
Trip Svc → Notification Svc        Async (fire+forget) Kafka  < 3s
```

**Real-time driver location (WebSocket + Kafka):**
```python
# Location Service: receive GPS from driver apps
class LocationWebSocketHandler:
    connected_drivers: dict[str, WebSocket] = {}
    
    async def on_connect(self, driver_id: str, ws: WebSocket):
        self.connected_drivers[driver_id] = ws
        await ws.accept()
        
    async def on_message(self, driver_id: str, data: dict):
        location = DriverLocation(
            driver_id=driver_id,
            lat=data['lat'],
            lng=data['lng'],
            heading=data.get('heading'),
            timestamp=datetime.utcnow()
        )
        
        # Store in Redis (current position, O(1) lookup)
        await redis.geoadd("driver_locations", location.lng, location.lat, driver_id)
        await redis.setex(f"driver:{driver_id}:location", 30, location.json())
        
        # Publish to Kafka (subscribers: Matching, Tracking, Analytics)
        await kafka.publish("driver.location.updated", location.dict())
```

**Driver matching (synchronous, latency-critical):**
```python
# Trip Service calls Matching Service synchronously
# Matching Service must respond in < 200ms
async def request_trip(rider_id: int, pickup: Coords, destination: Coords):
    # Step 1: Get price (sync, < 100ms)
    price = await pricing_stub.CalculatePrice(
        PriceRequest(pickup=pickup, destination=destination),
        timeout=0.1
    )
    
    # Step 2: Find nearby drivers from Redis (< 10ms)
    nearby_drivers = await redis.georadius(
        "driver_locations", pickup.lng, pickup.lat, 5, "km",
        count=20, sort="ASC"
    )
    
    # Step 3: Match algorithm (< 50ms)
    match = await matching_stub.FindBestDriver(
        MatchRequest(
            rider_id=rider_id,
            pickup=pickup,
            candidate_drivers=nearby_drivers,
            max_eta_minutes=5
        ),
        timeout=0.15
    )
    
    # Step 4: Create trip (DB write)
    trip = await trip_repo.create(
        rider_id=rider_id,
        driver_id=match.driver_id,
        price=price,
        status='driver_assigned'
    )
    
    # Step 5: Notify driver (async, non-blocking)
    await kafka.publish("trip.driver_assigned", {
        "trip_id": trip.id,
        "driver_id": match.driver_id,
        "pickup": pickup.dict()
    })
    
    return TripResponse(trip_id=trip.id, driver=match.driver, price=price)
```

**Payment via saga:**
```
Trip completed event:
  Trip Service publishes: trip.completed { trip_id, rider_id, driver_id, amount }
  
Payment Saga Orchestrator:
  Step 1: Charge rider (Payment Service)
  Step 2: Create driver payout record (Driver Service)
  Step 3: Update trip status to paid (Trip Service)
  
  On failure at Step 1: Mark trip as payment_failed, retry 3x, then manual review
  On failure at Step 2: Refund rider charge, retry driver payout next batch
```

**Event topology:**
```
Kafka topics:
  driver.location.updated    (partitioned by driver_id, high throughput)
  trip.requested             (partitioned by rider_id)
  trip.driver_assigned       (partitioned by trip_id)
  trip.started               (partitioned by trip_id)
  trip.completed             (partitioned by trip_id)
  payment.charged            (partitioned by rider_id)
  notification.requested     (partitioned by user_id)
```

---

## Quick Reference

### Communication Pattern Decision Tree

```
Do you need an immediate response?
  YES → Do you need browser/mobile support?
          YES → REST
          NO  → gRPC (internal) or GraphQL (BFF)
  NO  → Do you need ordering/replay?
          YES → Kafka (event streaming)
          NO  → RabbitMQ/SQS (task queue)
```

### Resilience Patterns Summary

| Pattern | Problem Solved | Key Parameter |
|---------|---------------|---------------|
| Circuit Breaker | Cascading failure | Failure threshold, timeout |
| Bulkhead | Resource exhaustion | Thread pool size per service |
| Timeout | Slow downstream | Budget per service call |
| Retry | Transient failures | Max retries + backoff |
| Fallback | Graceful degradation | Default response |
| Load Shedding | Overload | Priority thresholds |

### REST vs gRPC vs GraphQL

| | REST | gRPC | GraphQL |
|-|------|------|---------|
| Best for | External/Public | Internal | BFF/Mobile |
| Payload | JSON (large) | Protobuf (small) | JSON |
| Streaming | No | Yes | Subscriptions |
| Type safety | Optional | Required | Required |
| Browser | Yes | grpc-web | Yes |

### Saga vs 2PC

| | 2PC | Saga |
|-|----|------|
| Blocking | Yes | No |
| Availability | Low | High |
| Consistency | Strong | Eventual |
| Complexity | Lower | Higher |
| Failure recovery | Coordinator crash = stuck | Compensating transactions |

### Service Discovery Options

| Option | Best For | Tool |
|--------|---------|------|
| DNS (Kubernetes) | K8s-native | CoreDNS + kube-proxy |
| Client-side | Spring Cloud, non-K8s | Eureka + Ribbon |
| Service mesh | mTLS, traffic management | Istio + Envoy |
| API Gateway | External traffic | Kong, AWS API GW |
