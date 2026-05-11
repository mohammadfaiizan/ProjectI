# Microservices Architecture — HLD Interview Q&A

---

## Easy (Q1–Q7)

---

### Q1. When should you use microservices vs a monolith?

**Answer:**

There is no universally correct answer — the right choice depends on team size, product maturity, and operational capabilities. Starting with a monolith and extracting microservices later (Strangler Fig) is often the pragmatic choice.

**Start with a monolith when:**
- Early-stage product with unclear domain boundaries.
- Small team (< 5-8 engineers) — coordination overhead of microservices exceeds benefits.
- Tight deadlines — monolith is faster to build and deploy.
- Domain complexity not yet understood (premature decomposition creates wrong boundaries).
- Limited DevOps maturity — microservices require CI/CD, container orchestration, distributed tracing.

**Move to microservices when:**
- Team has grown (Conway's Law: org structure drives architecture).
- Different components have wildly different scaling requirements.
- Independent deployability is needed (5 teams deploying 5 times/day to the same monolith is painful).
- Technology heterogeneity required (ML model in Python, core service in Go, frontend in Node).
- Fault isolation needed (one component crashing should not bring down the whole system).

**Decision criteria table:**

| Criterion | Monolith | Microservices |
|-----------|---------|---------------|
| Team size | < 10 | > 10, multiple teams |
| Domain clarity | Low | High |
| Deployment frequency | Low | High |
| Scaling requirements | Uniform | Heterogeneous |
| Operational maturity | Low | High |
| Time-to-market pressure | High | Lower (initially) |
| Fault isolation need | Low | High |

**The "microservices premium" (Fowler):** Microservices introduce operational complexity — distributed tracing, service discovery, network latency, partial failures. You pay this premium in exchange for independent scalability and deployability. If you don't need these benefits, the premium is all cost and no benefit.

**Rule of thumb:** A successful startup migrates *to* microservices; failed startups often die trying to launch *with* microservices.

---

### Q2. How do you decompose a monolith into microservices using Domain-Driven Design?

**Answer:**

**Domain-Driven Design (DDD)** provides the conceptual tools to identify meaningful service boundaries that align with business domains rather than technical layers.

**Key DDD concepts for decomposition:**

**Bounded Context:** A boundary within which a particular domain model is consistent. "Customer" in Sales means one thing; "Customer" in Shipping means another. Each bounded context is a candidate microservice.

```
E-Commerce Domain:

[Order Management]     [Inventory]      [User/Identity]
  - Order aggregate      - Product         - Account
  - OrderItem            - Stock level      - Profile
  - OrderStatus          - Warehouse        - Auth token

[Payment]              [Shipping]       [Notification]
  - Transaction          - Shipment        - Email
  - Refund               - TrackingEvent   - SMS
  - PaymentMethod        - Carrier         - Push
```

**Context Map:** Shows relationships between bounded contexts:
- **Upstream/Downstream:** Order service is upstream of Shipping (Shipping depends on Order events).
- **Shared Kernel:** Two contexts share a common model (dangerous coupling, avoid if possible).
- **Anti-Corruption Layer (ACL):** When integrating with a legacy system, translate between models at the boundary.

**Decomposition approaches:**

1. **By business capability** (preferred): Align services to business functions — Order Management, Inventory, Payments. Maps to org structure, stable over time.

2. **By subdomain:**
   - Core domain: competitive differentiator (custom-build: Recommendation Engine).
   - Supporting subdomain: necessary but not differentiating (buy or build: Order Management).
   - Generic subdomain: commodity (buy: Email, Auth).

3. **By data ownership:** Each service owns its data, no sharing. "If two services share a database table, they should probably be one service."

**Strangler Fig migration:**
```
Step 1: Identify bounded context in monolith (e.g., Inventory)
Step 2: Build new Inventory microservice
Step 3: Route new Inventory traffic to microservice (via API gateway)
Step 4: Migrate data from monolith DB to Inventory DB
Step 5: Remove Inventory code from monolith
Repeat for each bounded context.
```

**Warning signs of bad decomposition:**
- Services that must be deployed together (tight temporal coupling).
- Services that share a database.
- A single business transaction requires 10 service calls.
- "Nano-services" that do one thing each (too granular, coordination cost exceeds benefit).

---

### Q3. What are the trade-offs between synchronous and asynchronous inter-service communication?

**Answer:**

**Synchronous communication** (REST/gRPC): Service A calls Service B and waits for a response.
**Asynchronous communication** (Message queues/events): Service A publishes a message and continues; Service B processes it later.

**Synchronous (REST/gRPC):**
```
Client -> [Order Service] -> [Payment Service] (waits)
                          -> [Inventory Service] (waits)
                          -> [Shipping Service] (waits)
Total latency = sum of all downstream calls
```

Pros:
- Simple programming model (request/response).
- Immediate feedback (know if operation succeeded).
- Easier to debug and reason about.

Cons:
- **Temporal coupling:** Service A cannot function if Service B is down.
- **Cascading failures:** If Service B is slow, Service A threads block, causing resource exhaustion.
- **Latency multiplication:** Sequential calls add up; parallel calls add complexity.

**Asynchronous (Events/Messages):**
```
Client -> [Order Service] -> publishes "OrderCreated" event
  Order Service returns 202 Accepted immediately.
  
[Payment Service]  <- subscribes to OrderCreated, processes independently
[Inventory Service] <- subscribes to OrderCreated, processes independently
[Notification Service] <- subscribes to OrderCreated, sends email
```

Pros:
- **Temporal decoupling:** Payment Service being down doesn't block Order creation.
- **Resilience:** Events queue up; downstream services process when available.
- **Scalability:** Consumers can be scaled independently.
- **Easy fan-out:** New services can subscribe without changing producer.

Cons:
- **Complex programming model:** Eventual consistency, no immediate feedback.
- **Harder debugging:** Trace flows across events and queues.
- **Duplicate handling required:** At-least-once delivery means idempotency needed.
- **Order not guaranteed** (unless using Kafka partitioned by key).

**Decision guide:**

| Scenario | Use Sync | Use Async |
|----------|---------|----------|
| User needs immediate response | Yes | No |
| Fire-and-forget notifications | No | Yes |
| Query (read data) | Yes | No |
| Long-running workflows | No | Yes |
| High-availability requirement | No | Yes |
| Real-time validation | Yes | No |

**Hybrid approach:** Use sync for user-facing queries and critical real-time validations; use async for workflows, notifications, and data propagation.

---

### Q4. What are the responsibilities of an API Gateway?

**Answer:**

An **API Gateway** is the single entry point for all clients. It handles cross-cutting concerns that would otherwise be duplicated across every service.

```
[Mobile App]  [Web App]  [Third-party]
      |            |           |
      +------------+-----------+
                   |
            [API Gateway]
      /auth /routing /rate-limit /transform/
                   |
    +--------------+-----------+----------+
    |              |           |          |
[User Svc]  [Order Svc]  [Product Svc]  [Payment Svc]
```

**Core responsibilities:**

1. **Routing / Load balancing:** Route `GET /orders` to Order Service, `POST /payments` to Payment Service.

2. **Authentication & Authorization:** Validate JWT tokens, API keys. Reject unauthenticated requests before they reach services. Optionally translate from external to internal auth models.

3. **Rate limiting & Throttling:** Prevent abuse. Per-IP, per-user, or per-API-key limits.
   ```
   100 requests/second per API key
   1000 requests/minute per IP
   ```

4. **SSL Termination:** Handle TLS at the gateway; internal service communication can be plain HTTP or mTLS.

5. **Request/Response transformation:** Aggregate responses from multiple services (not ideal), transform formats (XML → JSON), inject/remove headers.

6. **Observability:** Centralized logging, metrics (latency, error rates), distributed trace ID injection.

7. **Caching:** Cache responses for idempotent reads to reduce downstream load.

8. **Circuit breaking:** Fail fast when downstream services are unavailable.

**What API Gateway should NOT do:**
- Business logic (becomes a logic dumping ground — "API Gateway anti-pattern").
- Data transformation that requires understanding of domain semantics.
- Become a bottleneck (deploy redundantly, scale horizontally).

**Popular solutions:** AWS API Gateway, Kong, Nginx, Traefik, Envoy (as edge proxy), Apigee.

---

### Q5. What is the Backend for Frontend (BFF) pattern, and when should you use it?

**Answer:**

The **Backend for Frontend (BFF)** pattern creates a dedicated backend per client type. Instead of one generic API Gateway serving all clients, each client has a tailored API layer optimized for its specific needs.

```
Without BFF:
[Mobile]  [Web]  [Smart TV]
    |        |        |
    +--------+--------+
             |
        [Generic API]  <- over-fetches for mobile, under-fetches for web
             |
    [Services behind]

With BFF:
[Mobile App]    [Web App]    [Smart TV]
     |               |            |
[Mobile BFF]   [Web BFF]   [TV BFF]
     |               |            |
     +---------------+------------+
                     |
              [Backend Services]
```

**Why different clients need different APIs:**

- **Mobile:** Bandwidth-constrained; needs aggregated, minimal responses; push notifications.
- **Web:** Rich interactions; server-side rendering; larger payloads acceptable.
- **Smart TV:** Simplified navigation; different content formats; slower input.
- **Third-party:** Standard REST/GraphQL; versioned; documented.

**Benefits:**
- Each BFF team optimizes for their client's specific needs.
- Mobile BFF can aggregate 3 service calls into 1 to save round trips.
- Schema changes in the BFF don't affect other clients.
- Separate deployability per client type.

**When to use BFF:**
- Multiple distinct client types with significantly different data needs.
- Client teams are separate from backend teams.
- Client performance is critical (mobile data costs, TV rendering).

**When NOT to use BFF:**
- Single client type — unnecessary overhead.
- Very small team — BFF duplication becomes maintenance burden.
- Client requirements are essentially the same.

**BFF vs GraphQL:** GraphQL is an alternative where clients can specify exactly what fields they need in a single API, reducing the need for multiple BFFs. Many companies use BFF with GraphQL as the BFF's query language.

---

### Q6. What is a service mesh, and what does it provide?

**Answer:**

A **service mesh** is a dedicated infrastructure layer for handling service-to-service communication. It moves network concerns (retries, mTLS, circuit breaking, observability) out of application code and into a separate sidecar proxy running alongside each service instance.

```
Without service mesh:
  [Service A code] -- retry/timeout/TLS logic embedded in code --> [Service B]

With service mesh (Istio/Envoy):
  [Service A code] --> [Envoy sidecar A] --> [Envoy sidecar B] --> [Service B code]
                              |                       |
                     [Istio Control Plane]
                     (Pilot, Citadel, Mixer)
                     
  All traffic flows through sidecars.
  Sidecars handle: mTLS, retries, circuit breaking, metrics.
  Application code is unaware.
```

**Key capabilities:**

| Capability | How Provided |
|------------|-------------|
| **mTLS** | Automatic certificate rotation, service identity verification |
| **Load balancing** | L7 load balancing (by path, header, weight) |
| **Circuit breaking** | Automatic open/close based on error rates |
| **Retries & timeouts** | Configurable per-route policies |
| **Observability** | Automatic metrics, logs, distributed traces per request |
| **Traffic management** | Canary deployments, A/B testing, fault injection |
| **Rate limiting** | Policy-based rate limits per service |

**Istio components:**
- **Envoy:** The sidecar proxy (data plane). High-performance C++ proxy.
- **Pilot:** Distributes routing configuration to Envoy sidecars.
- **Citadel:** Issues and rotates mTLS certificates.
- **Telemetry:** Collects metrics and traces from sidecars.

**Trade-offs:**
- Significant operational complexity.
- Latency overhead from sidecar hop (~1-2ms per call, but adds up).
- Large learning curve (Istio is infamous for complexity).
- Strong value in large organizations with many services.

**When a service mesh is justified:** >20 services, strong security requirements (zero-trust network), need for fine-grained traffic control (canary at 1% → 5% → 100%), dedicated platform team.

---

### Q7. How does the circuit breaker pattern work?

**Answer:**

The **Circuit Breaker** pattern prevents cascading failures in distributed systems. When a downstream service is failing, instead of waiting for timeouts on every request, the circuit breaker "trips" and fails fast, giving the downstream service time to recover.

**Named after electrical circuit breakers:** In electricity, a circuit breaker trips under overload to protect the circuit. In software, the circuit breaker trips under failure to protect the caller.

**Three states:**

```
CLOSED  --[failure threshold exceeded]--> OPEN
  |                                         |
  |                               [timeout period expires]
  |                                         |
  +--[success threshold met]---- HALF-OPEN <+
```

**State behavior:**

**CLOSED (Normal):**
- All requests pass through to the downstream service.
- Track failure/success counts in a sliding window.
- If failures exceed threshold (e.g., 50% in last 10 requests): trip to OPEN.

**OPEN (Tripped):**
- All requests fail immediately (no calls to downstream service).
- Returns cached response, default value, or error.
- After timeout (e.g., 30 seconds): transition to HALF-OPEN.

**HALF-OPEN (Testing recovery):**
- Allow a small number of test requests through.
- If test requests succeed: close the circuit (CLOSED).
- If test requests fail: return to OPEN (service not yet recovered).

```python
class CircuitBreaker:
    def call(self, service_fn):
        if self.state == OPEN:
            if time.now() > self.open_until:
                self.state = HALF_OPEN
            else:
                raise CircuitOpenException("Circuit is open")
        
        try:
            result = service_fn()
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise

    def on_failure(self):
        self.failure_count += 1
        if self.failure_count >= self.threshold:
            self.state = OPEN
            self.open_until = time.now() + self.timeout

    def on_success(self):
        if self.state == HALF_OPEN:
            self.state = CLOSED
        self.failure_count = 0
```

**Libraries:** Hystrix (Netflix, deprecated), Resilience4j (Java), Polly (.NET), PyBreaker (Python).

**Best practices:**
- Set thresholds based on measured normal error rates (don't trip on 1% errors if 2% is normal).
- Log every state transition (alert on OPEN).
- Provide meaningful fallbacks (cached data, degraded response).
- Combine with bulkhead pattern to isolate failures per dependency.

---

## Medium (Q8–Q15)

---

### Q8. What is the bulkhead pattern, and how does it provide failure isolation?

**Answer:**

The **Bulkhead** pattern is named after the watertight compartments in a ship's hull. If one compartment floods, the ship doesn't sink — other compartments remain intact. In microservices, bulkheads isolate failures to prevent one slow/failing dependency from consuming all resources and cascading to the entire system.

**The problem (without bulkheads):**
```
Service A uses a shared thread pool for all downstream calls.
Downstream Service B becomes slow (100ms -> 5000ms response time).

Thread pool (20 threads):
  Thread 1: waiting for Service B
  Thread 2: waiting for Service B
  ...
  Thread 20: waiting for Service B
  
New request for Service C (fast, healthy):
  No threads available! -> Request fails even though Service C is fine.
```

**Bulkhead solution — dedicated thread pools per dependency:**
```
Service A:
  Thread Pool for Service B: 5 threads (max)
  Thread Pool for Service C: 10 threads (max)
  Thread Pool for Service D: 5 threads (max)

Service B becomes slow:
  Thread Pool B fills up -> Service B requests fail fast (pool exhausted)
  Thread Pool C and D: unaffected -> Service C and D calls succeed normally
```

**Implementation patterns:**

1. **Thread pool isolation (Hystrix model):**
   Each downstream dependency gets its own thread pool. Overhead: context switching, thread creation.

2. **Semaphore isolation:**
   Limit concurrent requests to a dependency using a semaphore counter. Lower overhead but no timeout isolation.

```java
// Resilience4j Bulkhead
BulkheadConfig config = BulkheadConfig.custom()
    .maxConcurrentCalls(10)
    .maxWaitDuration(Duration.ofMillis(100))
    .build();

Bulkhead bulkhead = Bulkhead.of("serviceB", config);

Supplier<String> decoratedSupplier = Bulkhead.decorateSupplier(bulkhead, () ->
    serviceBClient.call());
```

3. **Connection pool bulkhead:**
   Separate DB connection pools per service role (read pool, write pool, analytics pool). Slow analytics queries don't starve transactional writes.

4. **Container/process-level bulkhead:**
   Deploy service instances that call Service B separately from those that don't. Kubernetes resource limits act as a bulkhead at the infrastructure level.

**Combined pattern:** Circuit Breaker + Bulkhead = resilient dependency isolation. Bulkhead limits blast radius; circuit breaker provides fast fail when that radius is exhausted.

---

### Q9. How does distributed tracing work, and how do correlation IDs propagate across services?

**Answer:**

**Distributed tracing** tracks a single request as it flows through multiple microservices, assembling the complete call tree with timing information for each hop. Essential for diagnosing latency problems and understanding system behavior.

**Core concepts:**

- **Trace:** The complete journey of a request from entry to exit. Identified by a `trace_id` (UUID).
- **Span:** A unit of work within a trace (one service handling part of the request). Has a `span_id`, `parent_span_id`, start time, duration, and tags.

```
Trace: trace_id=abc123
  Span 1: API Gateway  (duration: 250ms)
    Span 2: Order Service (duration: 200ms)
      Span 3: Payment Service (duration: 80ms)
      Span 4: Inventory Service (duration: 60ms)
        Span 5: DB query (duration: 30ms)
```

**Context propagation (how correlation IDs travel):**

HTTP headers carry trace context between services (W3C Trace Context standard):
```
HTTP Request headers:
  traceparent: 00-abc123def456-7890abcd-01
                  ^   trace_id   ^span_id^flags
  tracestate: vendor-specific data
```

**Instrumentation (automatic in most frameworks):**
```python
# With OpenTelemetry (auto-instrumented)
from opentelemetry.instrumentation.requests import RequestsInstrumentor
RequestsInstrumentor().instrument()

# Every outgoing HTTP call automatically:
# 1. Reads traceparent from incoming request headers
# 2. Creates a child span
# 3. Injects traceparent into outgoing request headers
# 4. Reports span to tracing backend on completion
```

**Propagation flow:**
```
Client -> API Gateway
  [API Gateway]: traceparent not present
    -> Create new trace_id=abc123, span_id=span1
    -> Log span1 (API Gateway operation)
    -> Add header: traceparent: 00-abc123-span1-01
    -> Forward to Order Service

Order Service receives request:
  -> Read trace_id=abc123 from header
  -> Create child span: span2 (parent=span1)
  -> Log span2 with ORDER_CREATED tag
  -> Call Payment Service with traceparent: 00-abc123-span2-01
  -> Call Inventory Service with traceparent: 00-abc123-span2-01

Collector (Jaeger/Zipkin/Tempo):
  Receives all spans async, assembles into trace tree by trace_id
```

**Sampling strategies:**
- **Always sample:** 100% of traces collected. High overhead in production.
- **Probabilistic:** Sample X% of traces (e.g., 1%). Low overhead but misses rare bugs.
- **Rate-limiting:** N traces/second max.
- **Tail-based sampling:** Collect all trace data, decide to keep/drop after seeing the full trace (keep slow or error traces). Expensive but effective.

**Tooling:** Jaeger, Zipkin, AWS X-Ray, Honeycomb, Datadog APM. Standard: OpenTelemetry.

---

### Q10. What are the challenges of the database-per-service pattern?

**Answer:**

The **database per service** pattern is a foundational microservices principle: each service has its own private database that no other service can access directly. Services communicate via APIs or events, never via shared database.

```
Correct:
  [Order Service] -- owns --> [Orders DB]
  [Inventory Service] -- owns --> [Inventory DB]
  [User Service] -- owns --> [User DB]

Incorrect (shared database anti-pattern):
  [Order Service] ----+
  [Inventory Service] +--> [Shared DB]  <- tight coupling, no service autonomy
  [User Service] -----+
```

**Benefits of database per service:**
- Independent schema evolution (change Order DB without coordinating with other teams).
- Independent scaling (Inventory DB can use Redis; Order DB uses PostgreSQL).
- Failure isolation (Inventory DB outage doesn't affect Order Service).
- Independent deployability.

**Challenges:**

**1. Joins across services:**
```sql
-- Traditional: single DB join
SELECT o.id, u.name FROM orders o JOIN users u ON o.user_id = u.id

-- Microservices: no cross-DB joins
-- Option A: API composition (order service calls user service)
-- Option B: Maintain denormalized data via event sync
```

**2. Data consistency:**
No cross-service ACID transactions. Must use Saga pattern for multi-service operations. Transient inconsistency is the norm.

**3. Query complexity:**
Reports and analytics that need data from multiple services require:
- API composition (N+1 problem risk).
- CQRS with a unified read model fed by events from all services.
- Data warehouse / data lake aggregation layer.

**4. Data duplication:**
Denormalization is necessary. Order service may store a copy of user name/address (as it was at order time) to avoid calling User Service for every historical order display.

**5. Referential integrity:**
Database foreign keys can't span services. Must enforce referential integrity at application level or via eventual consistency (e.g., soft-delete with grace period rather than immediate delete).

**6. Operational overhead:**
N services = N databases to manage, monitor, backup, and scale. Significant operational burden.

**Strategy for cross-service data needs:**
Use the **Event-Driven Data Sync** pattern: when User Service updates a profile, it publishes `UserUpdated` event. Order Service subscribes and updates its local copy of user data. Accepts eventual consistency.

---

### Q11. What is the Strangler Fig pattern, and how do you use it for monolith migration?

**Answer:**

The **Strangler Fig** pattern (Martin Fowler, inspired by the strangler fig tree that grows around and eventually replaces its host) is a migration strategy for incrementally replacing a monolith with microservices, without a "big bang" rewrite.

The fig tree starts growing on an existing tree, sending roots to the ground while using the host for support. Eventually it completely encases and replaces the host. Similarly, new microservices gradually "strangle" the monolith.

**Migration process:**
```
Phase 0: Monolith serves everything
  [All Traffic] -> [Monolith]

Phase 1: Introduce facade (API Gateway) in front of monolith
  [All Traffic] -> [API Gateway] -> [Monolith] (no behavior change)

Phase 2: Extract Inventory service
  [All Traffic] -> [API Gateway]
                      |           \
                  [Monolith]   [Inventory Service] <- /inventory/* routes here
                  (everything else)

Phase 3: Extract Payment service
  [API Gateway]
      |         \            \
  [Monolith]  [Inventory]  [Payment Service] <- /payments/* routes here

Phase N: Monolith is empty, decommissioned
  [API Gateway]
     /    |    \    \
 [Svc1] [Svc2] [Svc3] [Svc4]
```

**Detailed steps for extracting a service:**

1. **Identify the bounded context** to extract (e.g., Inventory).
2. **Build the new Inventory microservice** with its own database.
3. **Sync data:** Dual-write or use Change Data Capture (CDC) to keep both DBs in sync during transition.
4. **Route new traffic** to the microservice via API Gateway.
5. **Validate** the new service handles production traffic correctly.
6. **Stop dual-write to monolith** once confidence is high.
7. **Remove Inventory code** from the monolith.

**Key techniques:**

- **Facade pattern:** API Gateway as the routing facade. The client never knows which system is handling the request.
- **Branch by abstraction:** In the monolith, introduce an abstraction layer in front of the functionality being extracted. Switch implementations via feature flag.
- **Change Data Capture (CDC):** Debezium reads the monolith's DB transaction log and publishes events, allowing the new service to build its own DB state.

**Risks to manage:**
- Data consistency during the dual-write phase (monolith + new service DB both have the data temporarily).
- Feature parity: the new service must match monolith behavior exactly before cutover.
- Rollback plan: API Gateway can route traffic back to monolith quickly if issues arise.

---

### Q12. What is consumer-driven contract testing, and why is it important in microservices?

**Answer:**

**Contract testing** verifies that services can communicate with each other, without the overhead of running all dependent services together in an integration test. **Consumer-driven** means the consumer (the calling service) defines the contract (what they need from the provider).

**The problem without contract testing:**
```
Without contracts:
  Service A tests against a mock of Service B
  Service B changes its API (removes a field, changes types)
  Service A's mock is out of date -> tests pass but production breaks
```

**Consumer-Driven Contract testing (Pact framework):**

```
Step 1: Consumer writes a test defining what it expects from the provider.
  // Order Service (consumer) expects from User Service (provider):
  {
    "GET /users/123": {
      responseStatus: 200,
      responseBody: {
        id: "123",
        email: "user@example.com",
        name: "John Doe"
      }
    }
  }
  -> This generates a "pact file" (the contract)

Step 2: Consumer test runs against a mock provider
  Order Service tests run against Pact mock server
  Tests verify Order Service handles expected responses correctly

Step 3: Pact file published to Pact Broker (contract registry)

Step 4: Provider verification
  User Service pulls the pact file from the broker
  Runs the actual User Service against pact interactions
  Verifies: "Can I actually fulfill what Order Service needs?"

Step 5: Can-I-Deploy check
  Before deploying Order Service or User Service,
  verify in Pact Broker that contracts are all passing
```

**Benefits:**
- Catch breaking API changes before deployment, not in production.
- Enable independent deployment with confidence (if contracts pass, safe to deploy).
- Living documentation: pact files describe actual API usage, not just what's documented.
- Faster CI (no need to spin up full integration environment for every PR).

**What contracts cover:**
- Request format (method, path, headers, body schema).
- Response format (status code, body fields, types).
- NOT business logic or end-to-end workflows.

**Tools:** Pact (most popular, multi-language), Spring Cloud Contract (Java ecosystem).

**When to use:** Any organization with multiple teams where services are deployed independently. If you're always deploying all services together, traditional integration tests may suffice.

---

### Q13. What is the sidecar pattern vs the ambassador pattern?

**Answer:**

Both patterns deploy helper containers alongside the main service container, extending its functionality without modifying service code. They differ in their role.

**Sidecar Pattern**
A secondary container attached to the primary container, sharing its lifecycle, network, and storage. Extends or enhances the primary container's capabilities.

```
Pod (Kubernetes):
  +---------------------------+
  | [Main App Container]      |
  | (your service code)       |
  |                           |
  | [Sidecar Container]       |
  | (logging/proxy/agent)     |
  +---------------------------+
  Shared: network namespace, volumes

Sidecar examples:
  - Envoy proxy (service mesh): intercepts all in/out traffic for mTLS, metrics
  - Log shipper (Fluentd): reads app logs, forwards to Elasticsearch
  - Config watcher: watches for config changes, signals app to reload
  - Vault agent: fetches secrets, writes to shared volume for app to read
```

**Ambassador Pattern**
A specialized sidecar that acts as a proxy specifically for **outbound calls** to external services. It acts as an ambassador (representative) between the app and external systems.

```
Pod:
  +--------------------------------+
  | [Main App Container]           |
  | Makes HTTP calls to            |
  | localhost:6000 (ambassador)    |
  |                                |
  | [Ambassador Container]         |
  | Listens on :6000               |
  | Handles: connection pooling,   |
  |   retries, circuit breaking,   |
  |   service discovery,           |
  |   auth token injection         |
  | Forwards to actual remote svc  |
  +--------------------------------+
```

**Ambassador pattern use cases:**
- Legacy apps that can't be modified but need retry/circuit breaker logic.
- Service discovery abstraction (app talks to localhost, ambassador resolves actual endpoint).
- Protocol translation (app speaks HTTP/1.1, ambassador handles HTTP/2 or gRPC).

**Comparison:**

| Aspect | Sidecar | Ambassador |
|--------|---------|-----------|
| Primary purpose | Extend app capabilities | Proxy outbound traffic |
| Direction | Both in/outbound | Outbound only |
| Examples | Envoy mesh, log forwarder | Twemproxy (Redis proxy), Envoy as edge proxy |
| Coupling | Shares lifecycle | Shares lifecycle |

**Vs. Service Mesh:** A service mesh sidecar (Envoy) handles both inbound and outbound traffic with centralized control. The ambassador pattern is a simpler per-app deployment without a control plane.

---

### Q14. What is the difference between client-side and server-side service discovery?

**Answer:**

In microservices, services need to find each other's network addresses. Since services scale dynamically (IPs change), static configuration is insufficient. Service discovery solves this.

**Client-Side Service Discovery:**
The client (calling service) queries the service registry directly, gets the list of available instances, and picks one using its own load balancing algorithm.

```
[Order Service]
    |
    1. Query registry: "Where is Payment Service?"
    v
[Service Registry (Consul/Eureka)]
    |
    Returns: ["10.0.1.5:8080", "10.0.1.6:8080", "10.0.1.7:8080"]
    |
    v
[Order Service]
    |
    2. Apply load balancing (Round Robin, Least Connections)
    |
    3. Direct call to selected instance
    v
[Payment Service (10.0.1.6:8080)]
```

Pros: Simple, no proxy hop, client has full control of LB strategy.
Cons: Every client must implement service discovery + load balancing logic. Every language/framework needs a client library.

**Server-Side Service Discovery:**
The client sends the request to a load balancer or router. The router queries the registry and forwards to an available instance. The client is unaware of instance addresses.

```
[Order Service]
    |
    1. Call Payment Service via stable DNS: payment-service:8080
    v
[Load Balancer / API Gateway / Service Mesh (Envoy)]
    |
    2. Queries registry: "Where is Payment Service?"
    v
[Service Registry]
    |
    Returns: ["10.0.1.5:8080", "10.0.1.6:8080"]
    |
    v
[Load Balancer]
    |
    3. Forwards to selected instance
    v
[Payment Service (10.0.1.5:8080)]
```

Pros: Clients are simple (just call a DNS name). Any language/framework works without a library.
Cons: Load balancer is an additional network hop. Load balancer can be a SPOF (mitigate with redundancy).

**Comparison:**

| Aspect | Client-Side | Server-Side |
|--------|------------|-------------|
| Client complexity | High (LB logic in client) | Low (just call DNS) |
| Flexibility | High (custom LB) | Lower (LB strategy centralized) |
| Network hops | 1 (direct to instance) | 2 (via LB) |
| Language support | Needs SDK per language | Universal |
| Examples | Eureka + Ribbon (Netflix) | AWS ALB, Kubernetes Service, Istio |

**Kubernetes default:** Kubernetes Service is server-side — `kube-proxy` handles traffic routing to pods behind a stable ClusterIP. Service mesh (Istio/Linkerd) adds client-side awareness via sidecar proxies.

---

### Q15. What are blue-green, canary, and rolling deployment strategies?

**Answer:**

**Blue-Green Deployment:**
Two identical production environments (blue = current, green = new). Traffic is instantly switched from blue to green via load balancer or DNS change.

```
Before deployment:
  [Load Balancer] -> [Blue Env (v1)] (100% traffic)
  [Green Env] (v2 deployed, running, idle)

Deployment:
  [Load Balancer] -> [Green Env (v2)] (instant switch)
  [Blue Env (v1)] (kept warm for rollback)

Rollback: instantly flip back to Blue
```

Pros: Zero-downtime, instant rollback.
Cons: Requires double the infrastructure. DB schema changes must be backward compatible (both envs share DB or need migration strategy).

**Canary Deployment:**
New version deployed to a small subset of instances/users first. Traffic gradually shifts if metrics look good.

```
Phase 1: 5% canary
  [LB] -> [v2 (5%)] + [v1 (95%)]

Phase 2: 25% if error rates/latency acceptable
  [LB] -> [v2 (25%)] + [v1 (75%)]

Phase 3: 50% -> 100%
  [LB] -> [v2 (100%)]  v1 decommissioned
```

Pros: Low risk — issues caught early, blast radius small. Real production traffic validates new version.
Cons: Complex to implement properly. Requires robust monitoring to automate rollback on metric degradation.

**Rolling Deployment:**
Old instances are replaced one-by-one (or in small batches) with new instances. No idle environment needed.

```
Start: [v1][v1][v1][v1] (4 instances)
Step 1: [v2][v1][v1][v1] (replace 1 at a time)
Step 2: [v2][v2][v1][v1]
Step 3: [v2][v2][v2][v1]
Step 4: [v2][v2][v2][v2]
```

Pros: No double infrastructure cost. Gradual rollout.
Cons: During rollout, both v1 and v2 are serving traffic — API must be backward compatible. Rollback is slow (need to roll back one by one).

**Comparison:**

| Strategy | Infrastructure cost | Rollback speed | Risk | Complexity |
|----------|--------------------|--------------------|------|------------|
| Blue-Green | 2x | Instant | Low | Medium |
| Canary | ~1.1x | Fast (reroute) | Very Low | High |
| Rolling | 1x | Slow | Medium | Low |

**Kubernetes:**
- Rolling = default `Deployment` update strategy.
- Blue-Green = two `Deployments` + switch `Service` selector.
- Canary = two `Deployments` with weighted `Ingress` rules or Istio `VirtualService`.

---

## Hard (Q16–Q20)

---

### Q16. How do microservices handle security — mTLS, JWT, and zero-trust?

**Answer:**

Microservices introduce unique security challenges: many inter-service communication channels, no clear network perimeter, and potentially dozens of services with different trust requirements.

**Zero Trust Architecture principle:**
"Never trust, always verify." Every service must authenticate and authorize every request, even from internal services. The network perimeter doesn't exist.

```
Traditional (perimeter security):
  [Internet] --firewall--> [Trusted Internal Network]
  Inside the firewall: all services trust each other implicitly.
  Breach of one service = full lateral movement.

Zero Trust:
  Every service authenticates to every other service.
  Least-privilege access per service.
  Continuous verification.
```

**mTLS (Mutual TLS) for service-to-service auth:**
Standard TLS: client verifies server's certificate. mTLS: server ALSO verifies client's certificate.

```
Service A calls Service B:
1. TLS handshake begins
2. Service B presents its certificate (signed by internal CA)
3. Service A verifies Service B's certificate chain
4. Service A presents its own certificate
5. Service B verifies Service A is who it claims to be
6. Encrypted, mutually authenticated channel established

Identity is in the certificate:
  Subject: CN=order-service, O=my-company
  Service B policy: "Allow read access to order-service, deny others"
```

Service mesh (Istio) automates mTLS: Citadel issues certificates per service identity; Envoy enforces mTLS transparently; application code sees plain HTTP locally.

**JWT for user identity propagation:**
```
Client -> API Gateway: Authorization: Bearer <JWT>
API Gateway:
  1. Validates JWT signature (using public key from JWKS endpoint)
  2. Extracts claims: {sub: "user-123", roles: ["customer"], exp: ...}
  3. Forwards request to Order Service with validated claims in header
     X-User-Id: user-123
     X-User-Roles: customer

Order Service:
  4. Trusts the API Gateway (mTLS) — no need to re-validate JWT signature
  5. Uses X-User-Id for authorization decisions
```

**Layered security model:**
```
Layer 1: Edge (API Gateway)
  - TLS termination
  - JWT / API key validation
  - Rate limiting, DDoS protection

Layer 2: Service-to-service (Service Mesh)
  - mTLS identity verification
  - Network policy enforcement (Service A can call Service B, not Service C)
  - Authorization policy: RBAC per service identity

Layer 3: Application
  - Input validation
  - Business logic authorization ("Can user-123 access order-456?")
  - Audit logging
```

**Secret management:**
Never hardcode secrets in code or environment variables. Use HashiCorp Vault or AWS Secrets Manager with short-lived dynamically generated credentials:
```
Service starts -> authenticates to Vault with its mTLS identity
Vault: "Order Service identity, here are 5-minute DB credentials"
Service: uses credentials, renews before expiry
On service death: credentials expire automatically
```

---

### Q17. What is a distributed monolith anti-pattern, and how do you recognize it?

**Answer:**

A **distributed monolith** is the worst of both worlds: a system that has been split into multiple deployable services but still has the tight coupling of a monolith. You pay the operational costs of distributed systems without gaining their benefits.

**How to recognize a distributed monolith:**

**1. Services that must be deployed together:**
```
"We can't deploy the Order Service without also deploying the Payment Service
and the Inventory Service, because they share the same schema version."
-> Tight version coupling = distributed monolith
```

**2. Shared database:**
```
[Order Service]     \
[Payment Service]    +--> [Shared Database]  <- both read/write same tables
[Inventory Service] /
```
If one service changes the DB schema, all others must be updated simultaneously.

**3. Synchronous call chains (no async):**
```
Client -> Order Service -> Payment Service -> Fraud Service -> Scoring Service -> Risk DB
         (every service waits synchronously)
-> A failure anywhere brings down the whole chain
-> Latency = sum of all service latencies
```

**4. No independent deployability:**
Services are deployed on a release train — all must coordinate for any deployment.

**5. Shared code libraries with business logic:**
```
SharedOrderLibrary.jar v1.3.2
  -> Order Service depends on it
  -> Payment Service depends on it
  -> Inventory Service depends on it
  
Library update requires synchronized upgrade of all 3 services.
```

**6. Synchronous chatty communication:**
A single user request triggers 20+ synchronous service-to-service calls. One slow service latency multiplies.

**How to fix a distributed monolith:**

| Symptom | Fix |
|---------|-----|
| Shared DB | Separate databases, event-driven sync |
| Shared code with biz logic | Push logic into each service, share only pure utilities |
| Synchronous chains | Introduce async messaging, sagas |
| Coordinated deployments | Establish API versioning, backwards compatibility |
| Wrong boundaries | Re-evaluate bounded contexts (DDD) |

**Root cause:** Often caused by decomposing by technical layer (Controller service, Service service, Repository service) rather than by business domain. Or decomposing too granularly before understanding the domain.

---

### Q18. How do you implement the Saga pattern for a distributed checkout flow?

**Answer:**

Let's walk through a complete implementation of an order checkout saga that coordinates Order, Payment, Inventory, and Shipping services.

**The saga sequence:**
```
1. Create order (Order Service)
2. Reserve inventory (Inventory Service)
3. Process payment (Payment Service)
4. Schedule shipping (Shipping Service)

Compensating transactions (on failure):
  4 fails -> no compensation needed for 4 (nothing committed)
  3 fails -> cancel shipment (N/A) + cancel payment
  2 fails -> cancel inventory reservation + cancel order
  1 fails -> cancel order
```

**Orchestration-based Saga with a Saga Orchestrator:**

```python
class CheckoutSaga:
    def __init__(self, saga_id: str, order_data: dict):
        self.saga_id = saga_id
        self.order_data = order_data
        self.completed_steps = []
    
    def execute(self):
        try:
            # Step 1: Create order
            order_id = self.order_service.create_order(self.order_data)
            self.completed_steps.append(('order', order_id))
            
            # Step 2: Reserve inventory
            reservation_id = self.inventory_service.reserve(
                order_id, self.order_data['items']
            )
            self.completed_steps.append(('inventory', reservation_id))
            
            # Step 3: Process payment
            payment_id = self.payment_service.charge(
                order_id, self.order_data['amount']
            )
            self.completed_steps.append(('payment', payment_id))
            
            # Step 4: Schedule shipping
            shipment_id = self.shipping_service.schedule(order_id)
            self.completed_steps.append(('shipping', shipment_id))
            
            self.order_service.confirm(order_id)
            return {"status": "SUCCESS", "order_id": order_id}
            
        except PaymentFailedException as e:
            self.compensate()
            return {"status": "FAILED", "reason": "payment_failed"}
        except InventoryException as e:
            self.compensate()
            return {"status": "FAILED", "reason": "out_of_stock"}
    
    def compensate(self):
        # Compensate in reverse order
        for (step, resource_id) in reversed(self.completed_steps):
            if step == 'shipping':
                self.shipping_service.cancel(resource_id)
            elif step == 'payment':
                self.payment_service.refund(resource_id)
            elif step == 'inventory':
                self.inventory_service.cancel_reservation(resource_id)
            elif step == 'order':
                self.order_service.cancel(resource_id)
```

**Persistence and idempotency:**
The saga orchestrator must persist its state (completed steps) so it can resume after a crash:
```sql
CREATE TABLE saga_state (
    saga_id UUID PRIMARY KEY,
    current_step VARCHAR(50),
    status VARCHAR(20),  -- RUNNING, COMPLETED, COMPENSATING, FAILED
    completed_steps JSONB,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
```

**Handling partial failures — idempotent compensations:**
Each compensation must be idempotent. Refunding a payment twice must be safe:
```python
def refund(payment_id: str):
    payment = db.get(payment_id)
    if payment.status == 'REFUNDED':
        return  # Already done, safe to ignore
    payment.refund()
    db.update(payment)
```

**State machine view:**
```
PENDING -> INVENTORY_RESERVED -> PAYMENT_PROCESSED -> SHIPPED -> COMPLETED
    |               |                    |               |
    v               v                    v               v
CANCELLED    COMPENSATION-1         REFUNDING       CANCELLING
             (cancel order)      (cancel inv,    (cancel shipment,
                                  cancel order)   refund, cancel inv,
                                                  cancel order)
```

---

### Q19. What are the three pillars of observability applied to microservices?

**Answer:**

The three pillars of observability are **Metrics, Logs, and Traces**. In microservices, where a single request can touch 10 services, all three are required — each answers different questions.

**Pillar 1: Metrics (What is happening? How much?)**
Numerical measurements over time. Best for alerting and dashboards.

```
Key microservice metrics (USE + RED):

USE (infrastructure):
  - Utilization: CPU 70%, Memory 80%
  - Saturation: Request queue depth: 150
  - Errors: 5xx rate: 0.01%

RED (service-level):
  - Rate: Requests/second per service
  - Errors: Error rate per service per endpoint
  - Duration: P50/P95/P99 latency per service

Service Mesh metrics (from Envoy sidecar — no code changes):
  - istio_requests_total{source="order", destination="payment", response_code="200"}
  - istio_request_duration_milliseconds{...}
```

**Pillar 2: Logs (What happened exactly?)**
Structured log records with context. Best for debugging specific incidents.

```json
{
  "timestamp": "2026-05-11T10:30:00Z",
  "level": "ERROR",
  "service": "payment-service",
  "trace_id": "abc123",
  "span_id": "def456",
  "user_id": "user-789",
  "order_id": "order-123",
  "message": "Payment declined by provider",
  "error_code": "INSUFFICIENT_FUNDS",
  "duration_ms": 234
}
```

Key practices:
- **Structured logging** (JSON) over free-text — enables querying.
- Include `trace_id` in every log line — bridge between logs and traces.
- Log at service boundary entry/exit, not just errors.
- Centralize with ELK stack (Elasticsearch/Logstash/Kibana) or Loki/Grafana.

**Pillar 3: Traces (Where is the time going? What called what?)**
Distributed traces following a request across service boundaries. Best for latency investigation.

```
Trace for POST /checkout (total: 850ms):

API Gateway (10ms)
└── Order Service (800ms)
    ├── Validate order (5ms)
    ├── Call Payment Service (650ms)  <- SLOW! investigate here
    │   ├── Fraud check (200ms)
    │   ├── External payment API (400ms)  <- external API latency
    │   └── DB write (50ms)
    ├── Call Inventory Service (80ms)
    │   └── DB read (75ms)
    └── Publish event (15ms)
```

**Correlating all three:**
The `trace_id` is the key linking all three pillars:
1. Alert fires: payment_error_rate > 1% (Metrics).
2. Jump to Grafana, filter logs by service=payment, level=ERROR, time=alert_window (Logs).
3. Find `trace_id` in error log, jump to Jaeger to see full trace (Traces).
4. See that all errors share the same upstream external payment API span being slow.

**Observability tooling stack:**
```
Metrics:   Prometheus + Grafana (or Datadog)
Logs:      ELK Stack / Loki + Grafana
Traces:    Jaeger / Zipkin / Tempo + Grafana
Unified:   OpenTelemetry (collector + SDK) -> any backend
```

---

### Q20. How do configuration management and feature flags work in microservices?

**Answer:**

In a microservices environment, configuration must be managed externally (not baked into service code), dynamically updatable, and consistently distributed across potentially hundreds of service instances.

**Configuration management approaches:**

**1. Environment variables (12-factor app approach):**
```yaml
# Kubernetes Deployment
env:
  - name: DB_HOST
    valueFrom:
      secretKeyRef:
        name: db-credentials
        key: host
  - name: PAYMENT_TIMEOUT_MS
    value: "5000"
```
Simple, but requires pod restart to change. Not suitable for dynamic updates.

**2. Centralized config service:**
```
[Service Instance] -> polls/subscribes to [Config Service (Consul/Spring Cloud Config/AWS AppConfig)]
                              |
                     [Config Store (Git/DynamoDB/Consul KV)]

On config change:
  Config service notifies all subscribed instances
  Instances reload config without restart
```

**3. Secret management (separate from config):**
```
Regular config: feature flags, timeouts, URLs -> stored in Config Service
Secrets: DB passwords, API keys, certs -> stored in Vault/AWS Secrets Manager

Service startup:
  1. Authenticate to Vault using pod's IAM role / mTLS identity
  2. Fetch secrets with short TTL (e.g., 15 min DB credentials)
  3. Vault agent sidecar handles renewal transparently
```

**Feature flags in microservices:**

Feature flags decouple deployment from feature release. Code is deployed but feature is gated:

```python
# Feature flag check
if feature_flag_service.is_enabled("new-checkout-flow", user_id=user_id):
    return new_checkout_handler(request)
else:
    return old_checkout_handler(request)
```

**Feature flag use cases:**

| Use Case | Description |
|----------|-------------|
| Canary release | Enable for 5% of users initially |
| A/B testing | 50/50 split to measure impact |
| Kill switch | Instantly disable a broken feature |
| Dark launch | Ship code, enable for internal users only |
| Gradual rollout | 1% → 10% → 50% → 100% |

**Feature flag systems:**
- **LaunchDarkly:** Enterprise, real-time flag evaluation, targeting rules.
- **Unleash:** Open-source, self-hosted.
- **Flagr:** Open-source, API-driven.
- **ConfigCat:** Simple, cost-effective.

**Feature flag lifecycle:**
```
State 1: Dark launch (flag off for everyone, code in prod)
State 2: Internal testing (flag on for internal users only)
State 3: Canary (flag on for 5% of users)
State 4: Gradual rollout (5% -> 25% -> 100%)
State 5: GA (flag on for 100% — remove flag from code in next sprint)
State 6: Cleanup (remove flag entirely — flags are tech debt if left forever)
```

**Anti-pattern — flag sprawl:** Hundreds of old feature flags left in code become maintenance burden. Enforce TTLs on flags; auto-alert on flags older than X days.

**Combining config + feature flags:**
Use a unified platform (LaunchDarkly, Split.io) that handles both runtime configuration and feature targeting in one SDK call. This avoids N different config-fetching mechanisms in each service.

---

## Quick Reference

### Microservices Decision
| Use Microservices When | Use Monolith When |
|------------------------|-------------------|
| Teams > 10, multiple squads | Team < 10 |
| Independent scaling required | Uniform scaling |
| High deployment frequency | Batch/infrequent deploys |
| Domain well-understood | Domain unclear |
| High DevOps maturity | Limited ops capacity |

### Circuit Breaker States
```
CLOSED -> (failure threshold exceeded) -> OPEN
OPEN   -> (timeout period)             -> HALF-OPEN
HALF-OPEN -> (success)                 -> CLOSED
HALF-OPEN -> (failure)                 -> OPEN
```

### Deployment Strategies
| Strategy | Infra Cost | Rollback | Risk |
|----------|-----------|----------|------|
| Blue-Green | 2x | Instant | Low |
| Canary | ~1.1x | Fast | Very Low |
| Rolling | 1x | Slow | Medium |

### Communication Trade-offs
| Aspect | Sync (REST/gRPC) | Async (Events) |
|--------|-----------------|----------------|
| Coupling | Temporal | Decoupled |
| Failure isolation | Low | High |
| Latency | Low | Higher |
| Complexity | Low | High |

### Observability Pillars
| Pillar | Answers | Tools |
|--------|---------|-------|
| Metrics | What/How much | Prometheus, Datadog |
| Logs | What happened exactly | ELK, Loki |
| Traces | Where is time going | Jaeger, Zipkin, Tempo |

### API Gateway Responsibilities
- Authentication/Authorization
- Rate limiting
- SSL termination
- Routing / Load balancing
- Observability injection
- NOT business logic

### Saga Pattern
| Type | Coordinator | Visibility | Coupling |
|------|------------|------------|---------|
| Choreography | None | Low | Low |
| Orchestration | Central | High | Medium |
| 2PC | Central | High | Very High (blocking) |

### DDD Bounded Contexts
```
Each bounded context = candidate microservice
Core domain -> build custom
Supporting domain -> build or buy
Generic domain -> buy (SaaS)
```
