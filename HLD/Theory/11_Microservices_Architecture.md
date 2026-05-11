# 11 — Microservices Architecture

---

## Table of Contents
1. [Monolith vs Microservices](#1-monolith-vs-microservices)
2. [Service Decomposition](#2-service-decomposition)
3. [12-Factor App Principles](#3-12-factor-app-principles)
4. [Inter-Service Communication](#4-inter-service-communication)
5. [API Gateway Pattern](#5-api-gateway-pattern)
6. [Backend for Frontend (BFF)](#6-backend-for-frontend-bff)
7. [Service Mesh](#7-service-mesh)
8. [Service Registry and Discovery](#8-service-registry-and-discovery)
9. [Circuit Breaker and Bulkhead Patterns](#9-circuit-breaker-and-bulkhead-patterns)
10. [Distributed Tracing](#10-distributed-tracing)
11. [Sidecar and Ambassador Patterns](#11-sidecar-and-ambassador-patterns)
12. [Strangler Fig Pattern](#12-strangler-fig-pattern)
13. [Database per Service Pattern](#13-database-per-service-pattern)
14. [Saga Pattern for Distributed Transactions](#14-saga-pattern-for-distributed-transactions)
15. [Configuration Management](#15-configuration-management)
16. [Contract Testing](#16-contract-testing)
17. [Microservices Deployment](#17-microservices-deployment)
18. [Deployment Strategies](#18-deployment-strategies)
19. [Microservices Security](#19-microservices-security)
20. [Observability in Microservices](#20-observability-in-microservices)
21. [When Microservices Go Wrong](#21-when-microservices-go-wrong)
22. [Quick Reference](#22-quick-reference)

---

## 1. Monolith vs Microservices

### What is a Monolith?

A single deployable unit containing all business logic, data access, and UI.

```
Monolith:
┌────────────────────────────────────────┐
│                App Server              │
│  ┌──────────┐ ┌──────────┐ ┌───────┐  │
│  │  Orders  │ │Inventory │ │ Auth  │  │
│  └──────────┘ └──────────┘ └───────┘  │
│  ┌──────────┐ ┌──────────┐ ┌───────┐  │
│  │ Shipping │ │ Payment  │ │ Users │  │
│  └──────────┘ └──────────┘ └───────┘  │
│         Shared Database                │
└────────────────────────────────────────┘
```

### What are Microservices?

Collection of small, independently deployable services, each responsible for a specific business capability.

```
Microservices:
OrderService ──── Order DB
    │
PaymentService ── Payment DB
    │
InventoryService ─ Inventory DB
    │
ShippingService ── Shipping DB
    │
UserService ────── User DB

Each runs in its own process, deployed independently
```

### Monolith Advantages

- Simple to develop initially
- Easy debugging (single process, single log)
- No network overhead between components
- ACID transactions across all entities
- Simple deployment (one artifact)
- Lower operational complexity

### Monolith Disadvantages

- Tight coupling: change in one area risks all areas
- Cannot scale specific components independently
- Technology lock-in (entire codebase in one language)
- Long deployment cycles (whole app for small change)
- Team coordination: large team works on same codebase
- Reliability: one bug can crash entire app

### Microservices Advantages

- Independent deployment
- Technology heterogeneity (polyglot)
- Independent scaling
- Fault isolation
- Team autonomy (each team owns a service)
- Smaller codebase per service

### Microservices Disadvantages

- Distributed systems complexity
- Network latency between services
- Distributed transactions are hard
- Service discovery and load balancing needed
- Observability requires more tooling
- Higher operational overhead

### When to Choose Each

| Criterion | Choose Monolith | Choose Microservices |
|---|---|---|
| Team size | Small (< 10 engineers) | Large (multiple teams) |
| Complexity | Simple domain | Complex, multi-domain |
| Stage | Early product (MVP) | Established, scaling product |
| Scaling needs | Uniform scaling | Specific hot components |
| Release frequency | Slow | Frequent, independent |
| Technical maturity | Low (less DevOps) | High (K8s, CI/CD, observability) |
| Data boundaries | Shared, coupled | Well-defined per domain |

**Rule of thumb:** Start with a monolith; break into microservices when:
- Specific components need different scaling
- Teams become large enough to step on each other
- Independent deployments become critical

---

## 2. Service Decomposition

### By Business Capability

Align services with what the business does.

```
E-commerce platform capabilities:
  - Product Catalog
  - Order Management
  - Payment Processing
  - Inventory Management
  - User Management
  - Shipping and Fulfillment
  - Reviews and Ratings
```

Each capability becomes a service. Services map to business functions, not technical layers.

### By Bounded Context (Domain-Driven Design)

A **bounded context** is a linguistic boundary where a term has a specific, unambiguous meaning.

```
"Order" in:
  OrderService: {id, items, total, status}
  ShippingService: {tracking_number, address, weight}
  FinanceService: {invoice_number, tax, payment_method}

Same real-world concept (order) modeled differently per context
Each bounded context owns its model and data
```

**Strategic DDD Patterns:**

- **Core Domain:** Critical differentiator — build in-house (e.g., recommendation engine)
- **Supporting Domain:** Needed but not core — can outsource or buy (e.g., accounting)
- **Generic Domain:** No competitive advantage — use off-shelf SaaS (e.g., email sending)

### By Subdomain

Based on DDD subdomains:

```
Core subdomain:     OrderManagement (competitive advantage → invest heavily)
Supporting:         InventoryTracking (needed → build lean)
Generic:            EmailNotification (not differentiating → use SendGrid SaaS)
```

### Decomposition Anti-Patterns

**Functional decomposition** (bad):
```
Bad: Split by technical layer
  - database-service
  - frontend-service
  - api-service
  These are tightly coupled — any feature change requires all three
```

**Too fine-grained** (nano-services):
```
Bad: One service per entity
  - create-user-service
  - update-user-service
  - get-user-service
  Extreme network overhead, hard to maintain
```

---

## 3. 12-Factor App Principles

Methodology for building software-as-a-service apps that are portable, resilient, and scalable.

| Factor | Principle | Microservices Application |
|---|---|---|
| 1. Codebase | One codebase per service, tracked in VCS | Each service has its own repo |
| 2. Dependencies | Explicitly declare and isolate dependencies | Docker containers package dependencies |
| 3. Config | Store config in environment variables | ConfigMaps, Vault, Consul for env vars |
| 4. Backing services | Treat databases, queues as attached resources | Connection strings via env vars |
| 5. Build/release/run | Strictly separate build and run stages | CI/CD pipeline: build → test → deploy |
| 6. Processes | Execute app as stateless processes | No local state; use external stores |
| 7. Port binding | Export services via port binding | Each container exposes a port |
| 8. Concurrency | Scale out via the process model | Horizontal scaling with K8s replicas |
| 9. Disposability | Fast startup and graceful shutdown | Health checks, SIGTERM handling |
| 10. Dev/prod parity | Keep dev, staging, prod as similar as possible | Docker ensures environment parity |
| 11. Logs | Treat logs as event streams | stdout → aggregated by log agent |
| 12. Admin processes | Run admin tasks as one-off processes | K8s Jobs for migrations, batch |

### Key Implications for HLD

**Factor 3 (Config):** Never hardcode config; use environment injection:
```yaml
env:
  - name: DATABASE_URL
    valueFrom:
      secretKeyRef:
        name: db-secret
        key: url
```

**Factor 6 (Stateless):** Store session in Redis, files in S3 — never on local disk.

**Factor 9 (Disposability):** Containers must handle SIGTERM:
```python
import signal, sys

def graceful_shutdown(sig, frame):
    # Finish in-flight requests
    server.stop(grace_period=10)
    sys.exit(0)

signal.signal(signal.SIGTERM, graceful_shutdown)
```

---

## 4. Inter-Service Communication

### Synchronous: REST

- HTTP/HTTPS with JSON payloads
- Well-understood, easy debugging
- Stateless
- Higher latency than binary protocols

```
GET /orders/{id}
POST /orders
Content-Type: application/json
Authorization: Bearer <token>
```

### Synchronous: gRPC

- HTTP/2 binary protocol with Protobuf
- Strongly typed contracts via `.proto` files
- Bi-directional streaming
- ~5-10x faster than REST/JSON
- Better for internal service-to-service communication

```protobuf
service OrderService {
  rpc GetOrder(GetOrderRequest) returns (Order);
  rpc CreateOrder(CreateOrderRequest) returns (Order);
  rpc StreamOrders(StreamRequest) returns (stream Order);
}
```

### Asynchronous: Events/Messaging

- Services communicate via message queue or event bus
- Decoupled: sender doesn't wait for response
- Better resilience: if consumer down, messages queue up
- Eventual consistency

```
OrderService publishes → Kafka (order.placed) → InventoryService consumes
                                              → PaymentService consumes
                                              → EmailService consumes
```

### Communication Pattern Comparison

| Aspect | REST | gRPC | Async Messaging |
|---|---|---|---|
| Protocol | HTTP/1.1 | HTTP/2 | AMQP/Kafka protocol |
| Payload | JSON (text) | Protobuf (binary) | Binary/JSON |
| Latency | Medium | Low | Higher (indirect) |
| Coupling | Tight (direct call) | Tight (direct call) | Loose |
| Contract | OpenAPI | Proto file | Schema Registry |
| Streaming | Limited (SSE) | Full duplex | Native |
| Error handling | HTTP status codes | Status codes | DLQ, retries |
| Use case | External API, simple | Internal, performance | Async, decoupled |

### Choosing Communication Style

```
Real-time response needed?
  YES → Synchronous (REST or gRPC)
        Performance critical? → gRPC
        External/browser client? → REST
  NO  → Asynchronous (Kafka/RabbitMQ)
        Complex routing? → RabbitMQ
        Event streaming/replay? → Kafka
```

---

## 5. API Gateway Pattern

### What is an API Gateway?

Single entry point for all client requests. Acts as a reverse proxy, routing requests to appropriate services.

```
Client ──► [API Gateway] ──► UserService
                        ──► OrderService
                        ──► ProductService
                        ──► PaymentService
```

### Core Functions

**Routing:** Map URL paths to backend services
```
/api/users/...     → UserService
/api/orders/...    → OrderService
/api/products/...  → ProductService
```

**Authentication and Authorization:**
```
Gateway validates JWT/API key before forwarding
Services trust gateway (no need to validate tokens again)
```

**Rate Limiting:**
```
Per-user: 100 requests/minute
Per-IP: 1000 requests/minute
Per-endpoint: custom limits
```

**Request Aggregation (API Composition):**
```
Single client request → Gateway calls UserService + OrderService
                      → Aggregates responses → returns single response
```

**Protocol Translation:**
```
External: HTTPS/REST → Internal: gRPC
External: WebSocket  → Internal: REST + SSE
```

**SSL Termination:** Handles TLS; internal services use plain HTTP.

**Logging and Monitoring:** Single point for access logs, metrics.

### API Gateway Products

| Product | Type | Features |
|---|---|---|
| AWS API Gateway | Managed | REST, HTTP, WebSocket; Lambda integration |
| Kong | Open-source/Enterprise | Plugin ecosystem, DB-less mode |
| NGINX | Reverse proxy + gateway | High performance, custom Lua plugins |
| Envoy | Proxy | Service mesh integration |
| Traefik | Cloud-native | Auto-discovers K8s services |
| Apigee | Enterprise | Full lifecycle API management |

### API Gateway Anti-Patterns

- **Smart gateway:** Don't put business logic in gateway; keep it dumb/transparent
- **Single gateway for everything:** Consider separate gateways for different client types (BFF)

---

## 6. Backend for Frontend (BFF)

### The Problem

Different clients (mobile, web, third-party) have different data needs. Generic API forces clients to over-fetch or make multiple calls.

```
Mobile app needs: minimal data (bandwidth constraints)
Web app needs: rich data (fast network, large screen)
Third-party: standard API

One generic API = bad for all clients
```

### BFF Solution

Create dedicated API layer per client type.

```
Mobile App ──► [Mobile BFF] ──► UserService
                          ──► OrderService (only mobile-needed fields)

Web App ──────► [Web BFF]   ──► UserService
                          ──► OrderService (full data)
                          ──► RecommendationService (web only)

Third-party ──► [Public API Gateway] ──► Services (standard API)
```

### BFF Responsibilities

- Request aggregation (call 3 services, return one response)
- Response transformation (shape data for specific client)
- Client-specific caching
- Client-specific auth flows
- Client-specific error handling

### BFF Implementation

```javascript
// Mobile BFF: GET /api/mobile/orders/:id
app.get('/api/mobile/orders/:id', async (req, res) => {
  const [order, user, tracking] = await Promise.all([
    orderService.getOrder(req.params.id),
    userService.getUser(order.userId),
    shippingService.getTracking(order.id)
  ]);
  
  // Transform for mobile: only necessary fields
  res.json({
    order_id: order.id,
    status: order.status,
    total: order.total,
    eta: tracking.estimatedDelivery,
    user_name: user.firstName   // mobile only shows first name
  });
});
```

### BFF Trade-offs

| Benefit | Trade-off |
|---|---|
| Optimized per client | More services to maintain |
| Faster client rendering | Code duplication between BFFs |
| Independent client evolution | Team ownership complexity |
| Reduce over-fetching | Need good abstractions for shared logic |

---

## 7. Service Mesh

### The Problem

With hundreds of microservices, cross-cutting concerns repeat: mTLS, retries, circuit breakers, distributed tracing. Don't implement in each service.

### Service Mesh Architecture

A dedicated infrastructure layer that handles service-to-service communication.

```
Control Plane: [Istio Pilot] → pushes config to all proxies
               [Istio Citadel] → certificate management (mTLS)
               [Istio Galley] → configuration validation

Data Plane: Each pod has sidecar proxy (Envoy)
  Service A → [Envoy sidecar] ──network──► [Envoy sidecar] ← Service B
              (handles: TLS, retries, metrics, tracing)
```

### Data Plane vs Control Plane

**Data Plane (Envoy Proxy):**
- Intercepts all inbound/outbound traffic (as sidecar)
- Enforces mTLS
- Circuit breaking
- Load balancing
- Metrics collection per request

**Control Plane (Istio):**
- Distributes routing rules to proxies
- Manages certificates
- Collects and aggregates telemetry
- Provides policy enforcement

### Key Service Mesh Features

**mTLS (Mutual TLS):**
```
Service A ──► Envoy A ──[encrypted+mutual auth]──► Envoy B ──► Service B

Both sides verify each other's certificates
Automatic certificate rotation via Citadel
```

**Traffic Management:**
```yaml
# Canary release: 10% traffic to v2
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
spec:
  http:
  - route:
    - destination:
        host: orders
        subset: v1
      weight: 90
    - destination:
        host: orders
        subset: v2
      weight: 10
```

**Observability:**
- Automatic metrics (latency, error rate, throughput) without code changes
- Distributed tracing (Jaeger integration)
- Access logs per request

### Service Mesh Trade-offs

| Benefit | Trade-off |
|---|---|
| Uniform policy enforcement | Additional latency (~1ms per hop) |
| No code changes needed | Operational complexity |
| mTLS everywhere | Steep learning curve (Istio complexity) |
| Centralized visibility | More infrastructure to maintain |

---

## 8. Service Registry and Discovery

(Covered in depth in File 10 — brief recap here with microservices context)

### Consul for Microservices

```
Services register on startup:
  PUT /v1/agent/service/register
  {
    "ID": "order-service-1",
    "Name": "order-service",
    "Tags": ["v2"],
    "Port": 8080,
    "Check": {
      "HTTP": "http://localhost:8080/health",
      "Interval": "10s"
    }
  }

Clients discover:
  GET /v1/health/service/order-service?passing=true
  → returns only healthy instances
```

### Kubernetes Service Discovery

```yaml
# Kubernetes Service (ClusterIP)
apiVersion: v1
kind: Service
metadata:
  name: order-service
spec:
  selector:
    app: order-service
  ports:
  - port: 80
    targetPort: 8080
```

DNS: `order-service.production.svc.cluster.local`
kube-proxy routes to healthy pods automatically.

---

## 9. Circuit Breaker and Bulkhead Patterns

### Circuit Breaker

Prevents cascading failures when a downstream service is unhealthy.

```
States:
CLOSED → requests pass through
  5 failures in 10s → OPEN

OPEN → requests fail immediately (fail-fast, no waiting)
  After 30s timeout → HALF-OPEN

HALF-OPEN → allow 1 test request
  Success → CLOSED
  Failure → OPEN
```

**Hystrix (Netflix OSS, now maintenance mode):**
```java
@HystrixCommand(fallbackMethod = "getDefaultUser",
                commandProperties = {
    @HystrixProperty(name="circuitBreaker.errorThresholdPercentage", value="50"),
    @HystrixProperty(name="circuitBreaker.sleepWindowInMilliseconds", value="5000")
})
public User getUser(String userId) {
    return userService.getUser(userId);
}

public User getDefaultUser(String userId) {
    return new User("Unknown", "Guest");  // fallback
}
```

**Resilience4j (modern alternative):**
```java
CircuitBreaker cb = CircuitBreaker.of("userService", config);
Supplier<User> decorated = CircuitBreaker.decorateSupplier(cb, 
    () -> userService.getUser(userId));
User user = Try.of(decorated::get)
    .recover(throwable -> fallbackUser())
    .get();
```

### Bulkhead Pattern

Isolate failures by limiting resources per consumer.

**Analogy:** Ship bulkheads separate compartments — one flood doesn't sink the ship.

```
Without bulkhead:
  100 threads total
  PaymentService slow → all 100 threads wait → all services blocked

With bulkhead:
  20 threads for PaymentService
  20 threads for InventoryService
  20 threads for UserService
  20 threads for ShippingService
  20 threads for general
  
  PaymentService slow → only 20 threads blocked → other services unaffected
```

**Implementation in Resilience4j:**
```java
// Thread pool bulkhead
ThreadPoolBulkheadConfig config = ThreadPoolBulkheadConfig.custom()
    .maxThreadPoolSize(20)
    .coreThreadPoolSize(10)
    .queueCapacity(100)
    .build();

ThreadPoolBulkhead bulkhead = ThreadPoolBulkhead.of("payments", config);
```

---

## 10. Distributed Tracing

### The Problem

A single user request may touch 10+ microservices. When it's slow or fails, which service is responsible?

```
Browser → API Gateway → OrderService → InventoryService → Database
                                    → PaymentService → External Payment API
                                    → NotificationService → Email Provider

Total: 500ms. Where is the bottleneck?
```

### Trace and Span

**Trace:** Complete journey of a request (has unique trace ID)
**Span:** Individual unit of work within a trace (has span ID, parent span ID, start/end time)

```
Trace ID: abc-123

Span 1: API Gateway       [0ms ──────────────────── 500ms]
  Span 2: OrderService    [10ms ─────────────── 490ms]
    Span 3: InventoryService [15ms ── 100ms]
    Span 4: PaymentService  [110ms ─────── 300ms]  ← bottleneck
    Span 5: NotificationSvc [305ms ─ 350ms]
```

### Trace ID Propagation

Each service extracts and forwards trace context in headers:

```
HTTP Headers (OpenTelemetry W3C Trace Context):
  traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01
  tracestate: ...

Service extracts trace ID → creates child span → forwards in outbound calls
```

### OpenTelemetry

Standard for distributed tracing, metrics, and logs. Vendor-neutral.

```python
from opentelemetry import trace
from opentelemetry.instrumentation.flask import FlaskInstrumentor

# Auto-instrumentation
FlaskInstrumentor().instrument_app(app)

# Manual span creation
tracer = trace.get_tracer("order.service")
with tracer.start_as_current_span("process-payment") as span:
    span.set_attribute("payment.amount", amount)
    span.set_attribute("payment.currency", "USD")
    result = payment_service.charge(amount)
```

### Tracing Tools

| Tool | Type | Backend | Features |
|---|---|---|---|
| Jaeger (CNCF) | Open-source | Cassandra/Elasticsearch | Full tracing UI |
| Zipkin | Open-source | Cassandra/MySQL | Simple, lightweight |
| OpenTelemetry | Standard/SDK | Multiple backends | Vendor neutral |
| AWS X-Ray | Managed | AWS | Native AWS integration |
| Datadog APM | Commercial | Datadog | Full observability suite |

---

## 11. Sidecar and Ambassador Patterns

### Sidecar Pattern

Deploy a helper container alongside the main application container in the same pod.

```
Pod:
  ┌─────────────────────────────┐
  │  Main App Container         │
  │  (OrderService)             │
  │                             │
  │  Sidecar Container          │
  │  (Envoy proxy / log agent)  │
  └─────────────────────────────┘
  Shared: network namespace, localhost
```

**Use Cases:**
- Service mesh proxy (Envoy/Linkerd)
- Log collection (Fluentd sidecar ships logs to ELK)
- TLS termination
- Config watcher (Consul template refreshes config)
- Metrics exporter (Prometheus exporter)

**Advantages:**
- Main application doesn't need to implement cross-cutting concerns
- Technology-agnostic (sidecar works with any language)
- Independent updates of sidecar vs main app

### Ambassador Pattern

Proxy that handles outbound traffic from the application.

```
App → [Ambassador] → External Service (with retries, circuit breaking, TLS)
                   → Database (with connection pooling)
                   → Legacy Service (protocol translation)
```

The ambassador handles: retry logic, authentication, service discovery, load balancing.

**Example:** Envoy as ambassador for outbound calls; app just calls `localhost:9001`.

---

## 12. Strangler Fig Pattern

### The Problem

Migrate a monolith to microservices without a big-bang rewrite (high risk).

**Named after:** strangler fig tree grows around host tree and slowly replaces it.

### Migration Strategy

```
Phase 1: New requests for Feature X → Microservice X
         Old requests for Feature X → Monolith (via routing layer)
         
Phase 2: Microservice X handles all Feature X traffic
         Feature X removed from monolith
         
Phase 3: Repeat for Feature Y, Z...

Phase N: Monolith fully replaced
```

### Implementation with API Gateway

```
[API Gateway / Router]
  /feature-x  → New Microservice (strangler)
  /feature-y  → Monolith (not yet migrated)
  /feature-z  → Monolith (not yet migrated)
```

### Migration Sequence

1. Identify bounded context / feature to migrate
2. Create new microservice with same external interface
3. Route new traffic (or A/B traffic) to microservice
4. Once validated, route all traffic to microservice
5. Remove feature from monolith
6. Repeat

### Data Migration Challenge

The hardest part: monolith shares one database.

```
Approach 1: Dual write
  App writes to both old DB and new DB
  New service reads from new DB
  Validate consistency
  Cut over to new DB only

Approach 2: Database-per-service (new service gets new DB)
  Sync data via event stream from monolith
  Once synced, cut over
```

---

## 13. Database per Service Pattern

### Why Database per Service?

- Loose coupling: each service owns its data schema
- Independent scaling: scale DB with service
- Technology choice: choose best DB for each service (SQL, NoSQL, graph)
- Encapsulation: other services cannot directly query your DB

```
OrderService     → PostgreSQL (relational, ACID)
ProductCatalog   → Elasticsearch (full-text search)
UserSession      → Redis (key-value, TTL)
RecommendationSvc → Neo4j (graph DB for relationships)
ActivityFeed     → Cassandra (time-series, high write)
```

### Challenges

**Challenge 1: Distributed Queries / Joins**

Cannot join across service databases.

```sql
-- This works in monolith:
SELECT o.id, u.name, p.title
FROM orders o
JOIN users u ON o.user_id = u.id
JOIN products p ON o.product_id = p.id

-- In microservices: cannot join across DBs
```

Solutions:
1. **API Composition:** Call multiple services, join in application code
2. **CQRS Read Model:** Maintain a denormalized read view updated via events
3. **Shared database (compromise):** Some teams use one DB but different schemas per service (not ideal but pragmatic)

**Challenge 2: Distributed Transactions**

```
User buys item:
  1. OrderService: create order (in Order DB)
  2. InventoryService: decrement stock (in Inventory DB)
  3. PaymentService: charge user (in Payment DB)

If step 3 fails: must undo steps 1 and 2 → Saga pattern
```

**Challenge 3: Referential Integrity**

Cannot have foreign keys across service boundaries.

```
OrderService stores user_id but cannot have FK to UserService's users table
Solution: validate via API call (not FK constraint)
Accept eventual consistency (user deleted but old orders reference them)
```

### Data Ownership Rules

```
Rule 1: Only OrderService can write to Orders DB
Rule 2: Other services read order data via OrderService API
Rule 3: Never bypass service API to query another service's DB directly
```

---

## 14. Saga Pattern for Distributed Transactions

(Extended from File 09 — microservices-focused view)

### Choreography Saga — E-Commerce Example

```
1. OrderService: publishes OrderCreated
2. PaymentService: listens to OrderCreated
   → charges customer
   → publishes PaymentCompleted (or PaymentFailed)
3. InventoryService: listens to PaymentCompleted
   → reserves stock
   → publishes StockReserved (or StockInsufficient)
4. ShippingService: listens to StockReserved
   → creates shipment
   → publishes OrderFulfilled

Compensation on PaymentFailed:
  OrderService listens → marks order as Cancelled
```

### Orchestration Saga — E-Commerce Example

```
OrderSagaOrchestrator (stateful):
  State: PENDING_PAYMENT
  1. Send command to PaymentService: Charge($100)
     → PaymentCompleted → State: PENDING_INVENTORY
  2. Send command to InventoryService: Reserve(item123)
     → StockReserved → State: PENDING_SHIPMENT
  3. Send command to ShippingService: CreateShipment
     → ShipmentCreated → State: COMPLETED

  On PaymentFailed:
     → State: COMPENSATING
     → Send ReleaseInventory (if already reserved)
     → Mark order FAILED
```

### Saga State Machine

```
States: CREATED → PAYMENT_PENDING → STOCK_PENDING → FULFILLMENT_PENDING → COMPLETED
                        ↓ fail              ↓ fail
                  CANCELLING ←─────────────────────
                        ↓
                  CANCELLED
```

---

## 15. Configuration Management

### The Problem

100 microservices, each needs database URLs, API keys, feature flags. How to manage?

### Environment Variables (12-Factor)

Simple, works for most cases. Injected via deployment platform.

```yaml
# Kubernetes ConfigMap
apiVersion: v1
kind: ConfigMap
metadata:
  name: order-service-config
data:
  DATABASE_HOST: "postgres-orders.production"
  LOG_LEVEL: "info"
  MAX_CONNECTIONS: "100"

# Kubernetes Secret
apiVersion: v1
kind: Secret
data:
  DATABASE_PASSWORD: <base64-encoded>
  API_KEY: <base64-encoded>
```

### Consul Key-Value Store

```bash
# Store config
consul kv put service/order-service/db-url "postgres://..."

# Service watches for changes (hot reload without restart)
consul watch -type=key -key=service/order-service/db-url ./on-change.sh
```

### HashiCorp Vault (Secrets)

```
Vault provides:
  - Dynamic secrets: generate DB credentials on-demand, auto-expire
  - Secret rotation: rotate credentials automatically
  - Access control: services only get their secrets
  - Audit logging: who accessed what and when

Order service:
  → authenticate with Vault (via K8s service account)
  → GET /v1/secret/data/order-service/db-creds
  → Vault returns credentials (valid for 1 hour)
  → App uses credentials; Vault auto-rotates
```

### Spring Cloud Config

For Java/Spring ecosystem:
```yaml
# application.yml in config server (Git-backed)
order-service:
  db:
    url: jdbc:postgresql://...
  feature-flags:
    new-checkout: true
```

Services fetch config on startup; can refresh without restart.

### Feature Flags

```
LaunchDarkly / Split.io:
  - Toggle features without deployment
  - A/B testing
  - Gradual rollout

Code:
if featureFlags.isEnabled("new-checkout", userId):
    return newCheckoutFlow()
else:
    return oldCheckoutFlow()
```

---

## 16. Contract Testing

### The Problem

Service A (consumer) depends on Service B (provider). How to ensure B doesn't break A's assumptions?

**Integration tests:** Slow, require both services running simultaneously.

### Consumer-Driven Contract Testing (PACT)

Consumer defines the contract (what it expects from provider). Provider verifies it can fulfill the contract.

```
Step 1: Consumer writes contract test
  "I expect POST /orders to return {id, status, total}"
  Pact generates contract file: order-consumer → order-provider.json

Step 2: Contract shared with provider (via Pact Broker)

Step 3: Provider runs contract verification
  "Can I satisfy all contract expectations?"
  If yes → compatible. If no → build fails.
```

### Example

**Consumer test (JavaScript/Jest):**
```javascript
const { PactV3 } = require('@pact-foundation/pact');

describe('Order Service', () => {
  it('creates an order', async () => {
    await provider
      .addInteraction({
        given: 'user exists',
        uponReceiving: 'a request to create order',
        withRequest: {
          method: 'POST',
          path: '/orders',
          body: { userId: '123', items: [{ sku: 'ABC', qty: 2 }] }
        },
        willRespondWith: {
          status: 201,
          body: { id: like('uuid'), status: 'pending', total: like(99.99) }
        }
      });
    
    const order = await orderClient.createOrder({ userId: '123', items: [...] });
    expect(order.status).toBe('pending');
  });
});
```

**Provider verification:**
```javascript
new Verifier({
  providerBaseUrl: 'http://localhost:8080',
  pactUrls: ['./pacts/order-consumer-order-provider.json']
}).verifyProvider();
```

### Benefits of Contract Testing

- Catches breaking API changes before deployment
- No need to run both services simultaneously in tests
- Faster than full integration tests
- Documents inter-service API expectations

---

## 17. Microservices Deployment

### Docker Containers

```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY src/ ./src/
EXPOSE 3000
HEALTHCHECK --interval=30s CMD wget -qO- http://localhost:3000/health
CMD ["node", "src/index.js"]
```

**Why containers for microservices:**
- Packaging: all dependencies included
- Isolation: services don't interfere
- Environment parity: dev = staging = prod
- Fast startup: seconds, not minutes

### Kubernetes Orchestration

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: order-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: order-service
  template:
    spec:
      containers:
      - name: order-service
        image: company/order-service:v2.1.0
        resources:
          requests: { cpu: "100m", memory: "128Mi" }
          limits:   { cpu: "500m", memory: "512Mi" }
        livenessProbe:
          httpGet: { path: /health, port: 8080 }
          initialDelaySeconds: 30
        readinessProbe:
          httpGet: { path: /ready, port: 8080 }
```

### Kubernetes Benefits

- Auto-healing (restart failed pods)
- Horizontal pod autoscaling (HPA)
- Rolling updates
- Service discovery (DNS)
- ConfigMaps and Secrets
- Namespace isolation between teams

---

## 18. Deployment Strategies

### Blue-Green Deployment

Maintain two identical environments. Switch traffic instantly.

```
Before:                      After deployment:
Blue (v1): 100% traffic      Blue (v1): 0% traffic (standby)
Green (v2): 0%               Green (v2): 100% traffic

Rollback: instantly switch traffic back to Blue
Risk: requires double infrastructure cost
```

### Canary Release

Gradually route traffic to new version.

```
Week 1: v2 gets 5% of traffic
Week 2: v2 gets 20%
Week 3: v2 gets 50%
Week 4: v2 gets 100% → v1 retired

Monitor error rates and latency at each stage
Rollback: route all traffic back to v1
```

```yaml
# Istio canary routing
http:
- route:
  - destination: {host: order-service, subset: v1}
    weight: 80
  - destination: {host: order-service, subset: v2}
    weight: 20
```

### Feature Flags

Toggle features independently of deployment:
```python
# Gradual rollout: enable for 10% of users
if feature_flag_client.variation("new-algorithm", user_id, default=False):
    return new_recommendation_algorithm(user)
else:
    return old_recommendation_algorithm(user)
```

### Comparison

| Strategy | Rollback speed | Infrastructure cost | Risk |
|---|---|---|---|
| Recreate | Slow (downtime) | Low | High |
| Rolling update | Minutes | Low | Medium |
| Blue-Green | Instant | Double | Low |
| Canary | Minutes | Slightly more | Very low |
| Feature flags | Instant | None | Very low |

---

## 19. Microservices Security

### Service-to-Service Authentication

**mTLS (Mutual TLS):**
```
OrderService (client cert) ←→ PaymentService (server cert)
Both services verify each other's certificates
No shared secret; certificate-based identity
Handled automatically by service mesh (Istio)
```

**JWT between services:**
```
OrderService generates JWT with service identity:
  {
    "iss": "order-service",
    "sub": "service-account",
    "aud": "payment-service",
    "exp": 1700000000
  }
  Signed with private key → PaymentService verifies with public key
```

### API Gateway Auth

```
External clients → API Gateway → validates JWT/API key
                             → adds user context to header
                             → forwards to internal services

Internal services:
  Trust the gateway-validated headers
  No need to re-validate external tokens
  Only validate service-to-service tokens
```

### Zero Trust Security Model

```
Principles:
1. Never trust, always verify (even internal services)
2. Least privilege: services only get permissions they need
3. Assume breach: design for containment
4. Micro-segmentation: network policies between services
```

### Kubernetes Network Policies

```yaml
# Only allow order-service to call payment-service
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: payment-service-policy
spec:
  podSelector:
    matchLabels:
      app: payment-service
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: order-service
    ports:
    - protocol: TCP
      port: 8080
```

---

## 20. Observability in Microservices

### Three Pillars of Observability

**Logs:** What happened
**Metrics:** How much / how fast
**Traces:** Where time was spent

### Structured Logging

```json
{
  "timestamp": "2024-01-01T10:00:00Z",
  "level": "INFO",
  "service": "order-service",
  "version": "v2.1.0",
  "trace_id": "abc123",
  "span_id": "def456",
  "user_id": "user789",
  "order_id": "ord001",
  "event": "order.created",
  "duration_ms": 45,
  "message": "Order created successfully"
}
```

### Correlation IDs

```
Request enters API Gateway → Gateway generates correlation ID
Every service logs with correlation ID
Log aggregation: search by correlation ID → see full request path

X-Correlation-ID: e2e-uuid-12345
X-Trace-ID: jaeger-trace-id-67890
```

### Metrics with Prometheus

```python
from prometheus_client import Counter, Histogram

orders_total = Counter('orders_total', 'Total orders', ['status'])
order_duration = Histogram('order_duration_seconds', 'Order processing time')

@order_duration.time()
def process_order(order):
    # ...
    orders_total.labels(status='success').inc()
```

### Dashboards (Grafana)

```
Key metrics per service:
  - Request rate (RPS)
  - Error rate (%)
  - Latency (p50, p95, p99)
  - Saturation (CPU, memory)

USE Method: Utilization, Saturation, Errors
RED Method: Rate, Errors, Duration
```

---

## 21. When Microservices Go Wrong

### Distributed Monolith

All services deploy together and share a database. Worst of both worlds.

```
Symptom: OrderService cannot deploy without CoordinatedService being updated too
Cause: Direct database joins across services, shared libraries with business logic
Fix: Define bounded contexts, database per service, async communication
```

### Chatty Services

Too many synchronous calls between services.

```
Bad pattern:
  GetOrderDetails →
    → GetUser (1 call)
    → GetProduct × 5 items (5 calls)
    → GetInventory × 5 items (5 calls)
    → GetShipping (1 call)
    Total: 12 synchronous calls = 12 × 20ms = 240ms minimum

Fix: Batch APIs, caching, data denormalization, BFF aggregation
```

### Data Ownership Issues

Multiple services writing to the same database table.

```
Bad: OrderService and ShippingService both update orders table
Fix: Only OrderService owns orders; ShippingService calls OrderService API
```

### Cascading Failures

Service A → B → C → D fails → timeout propagates back.

```
Without circuit breaker:
  D is slow → C waits 30s → B waits 30s → A waits 30s
  All threads consumed → A crashes

Fix: Circuit breaker + bulkhead + timeouts at every hop
```

### Anti-Patterns Summary

| Anti-pattern | Symptom | Fix |
|---|---|---|
| Distributed monolith | Must deploy all together | True bounded contexts |
| Chatty services | Many sync calls per request | Async, caching, BFF |
| Shared database | Multiple services, one DB | Database per service |
| Hardcoded service URLs | Fragile after IP change | Service discovery |
| No distributed tracing | Can't debug latency | OpenTelemetry + Jaeger |
| No circuit breakers | Cascading failures | Resilience4j / Envoy |
| Nano-services | Tiny services with one method | Merge related services |

---

## 22. Quick Reference

### Monolith vs Microservices Decision Criteria

| Factor | Monolith | Microservices |
|---|---|---|
| Team size | < 10 engineers | Multiple teams |
| System maturity | Greenfield / MVP | Established product |
| Domain complexity | Simple | Complex, multi-domain |
| Scaling needs | Uniform | Heterogeneous |
| Release velocity | Low | High, independent |
| Operational maturity | Basic | DevOps, K8s, observability |
| Data boundaries | Fluid/coupled | Well-defined |

### Inter-Service Communication Comparison

| Aspect | REST | gRPC | Async (Kafka) |
|---|---|---|---|
| Latency | Medium | Low | Higher |
| Coupling | Tight | Tight | Loose |
| Contract | OpenAPI | Proto | Schema Registry |
| Streaming | Limited | Full | Native |
| Error propagation | HTTP codes | Status codes | DLQ, retry |
| Best for | External API | Internal perf | Decoupled events |

### Microservices Interview Cheat Sheet

1. **When not to use microservices:** Small team, early product, no DevOps maturity
2. **Service decomposition:** By bounded context (DDD); not by layer
3. **Database per service:** Enables independence; solve distributed queries with API composition or CQRS
4. **Saga vs 2PC:** Saga for eventual consistency with compensation; 2PC is blocking and fragile
5. **Circuit breaker:** Fail-fast to prevent cascading failures
6. **Service mesh:** Offload mTLS, retries, tracing to sidecar proxy
7. **Strangler fig:** Safe incremental migration from monolith
8. **BFF pattern:** Dedicated backend per client type for optimal data shaping
9. **Contract testing (PACT):** Verify compatibility without running both services
10. **Observability:** Logs + metrics + traces; correlation IDs across all services
```
