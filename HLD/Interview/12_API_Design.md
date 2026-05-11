# HLD Interview Q&A — File 12: API Design

---

## Easy Questions (Q1–Q7)

---

### Q1. What are REST API design principles? Explain resources, HTTP verbs, and status codes.

**Answer:**

REST (Representational State Transfer) is an architectural style for designing networked APIs. It is not a protocol — it is a set of constraints.

**Core REST principles:**
1. **Stateless:** Each request contains all information needed. No server-side session.
2. **Resource-based:** Model everything as nouns (resources), not verbs.
3. **Uniform interface:** Consistent conventions across the API.
4. **Client-Server separation:** Frontend and backend evolve independently.

**Resources (nouns, not verbs):**
```
Good (noun):         POST /orders          → create order
                     GET  /orders/123      → get order
Bad (verb):          POST /createOrder     → violates REST
                     GET  /getOrderById?id=123
```

**HTTP Verbs:**

| Verb   | Action          | Idempotent? | Safe? | Example                  |
|--------|----------------|-------------|-------|--------------------------|
| GET    | Read            | Yes         | Yes   | GET /users/42            |
| POST   | Create          | No          | No    | POST /orders             |
| PUT    | Full update     | Yes         | No    | PUT /orders/1 (full body)|
| PATCH  | Partial update  | No          | No    | PATCH /orders/1          |
| DELETE | Delete          | Yes         | No    | DELETE /orders/1         |

**Status Codes every engineer must know:**
```
2xx Success:
  200 OK                 → Standard success
  201 Created            → Resource created (with Location header)
  204 No Content         → Success, no body (DELETE, some PUT)
  202 Accepted           → Async job started

4xx Client Errors:
  400 Bad Request        → Invalid input, validation failed
  401 Unauthorized       → Not authenticated (missing/invalid token)
  403 Forbidden          → Authenticated but not authorized
  404 Not Found          → Resource doesn't exist
  409 Conflict           → Duplicate, version conflict
  422 Unprocessable      → Semantically invalid (correct format, wrong data)
  429 Too Many Requests  → Rate limited

5xx Server Errors:
  500 Internal Server Error → Unexpected server failure
  502 Bad Gateway           → Upstream service failed
  503 Service Unavailable   → Overloaded or down (retry-able)
  504 Gateway Timeout       → Upstream didn't respond in time
```

---

### Q2. What are API versioning strategies and their trade-offs?

**Answer:**

API versioning allows the server to evolve its contract without breaking existing clients. The three main strategies each have distinct trade-offs.

**Strategy 1: URL Path Versioning**
```
GET /v1/users/42
GET /v2/users/42
```
Pros: Obvious, easy to route, easy to test in browser, can be cached by CDN.
Cons: Pollutes the URL (URL should identify a resource, not its version). Breaks REST purity.
Best for: Public APIs, mobile APIs where client upgrades are slow.

**Strategy 2: Header Versioning**
```
GET /users/42
Accept: application/vnd.myapp.v2+json
```
Pros: Clean URLs, more REST-compliant, supports content negotiation.
Cons: Harder to test in browser, must set headers explicitly, less visible to developers.
Best for: Internal APIs, well-tooled teams.

**Strategy 3: Query Parameter Versioning**
```
GET /users/42?version=2
```
Pros: Easy to switch in browser/curl, backwards compatible (defaults to v1).
Cons: Breaks HTTP caching (version is often stripped from cache key), easy to misuse.
Best for: Optional backwards compatibility, rarely used as primary strategy.

**Comparison:**

| Strategy      | URL Clarity | Cache-Friendly | REST-Pure | Developer UX |
|---------------|-------------|----------------|-----------|--------------|
| URL path      | Low         | Yes            | No        | High         |
| Header        | High        | No             | Yes       | Medium       |
| Query param   | Medium      | No             | No        | High         |

**Industry reality:** URL path versioning dominates for public APIs (GitHub, Stripe, Twilio) because it is explicit and developer-friendly. Header versioning is preferred for internal APIs with OpenAPI-based tooling.

**Version lifecycle:**
```
v1 → Active (current)
v2 → Active (current)
v1 → Deprecated (announce 6-12 months ahead)
v1 → Sunset (return 410 Gone after deadline)
```

---

### Q3. What are the most important HTTP status codes every engineer must know?

**Answer:**

While there are over 70 HTTP status codes, roughly 15 cover the vast majority of real-world API design scenarios.

**2xx — Success:**
```
200 OK              The standard success response. Includes body.
201 Created         Resource was created. Include Location header with new resource URL.
                    Location: /orders/456
204 No Content      Success but no response body. Used for DELETE, some PUT/PATCH.
202 Accepted        Request accepted for async processing. Return job ID.
                    {"job_id": "abc123", "status_url": "/jobs/abc123"}
```

**3xx — Redirection:**
```
301 Moved Permanently   Old URL permanently gone. Clients should update.
304 Not Modified        Conditional GET — client cache is still fresh.
```

**4xx — Client Errors:**
```
400 Bad Request         Malformed request, missing required fields, bad JSON.
401 Unauthorized        Authentication required or invalid. Include WWW-Authenticate header.
403 Forbidden           Authenticated but lacks permission. Do NOT leak resource existence.
404 Not Found           Resource doesn't exist, or intentionally hidden (privacy).
405 Method Not Allowed  GET on a POST-only endpoint. Include Allow header.
409 Conflict            Duplicate unique field, optimistic locking conflict.
410 Gone                Resource permanently deleted (stronger signal than 404).
422 Unprocessable Entity Semantically invalid (well-formed JSON, but business rules violated).
429 Too Many Requests   Rate limited. Include Retry-After header.
```

**5xx — Server Errors:**
```
500 Internal Server Error  Bug or unexpected failure. Never expose stack traces to clients.
502 Bad Gateway            Reverse proxy got invalid response from upstream.
503 Service Unavailable    Server overloaded or in maintenance. Use Retry-After.
504 Gateway Timeout        Upstream service took too long.
```

**Design rule:** 4xx errors are the client's fault; include enough detail in the response body that the client can fix the request without reading docs. 5xx errors are your fault; include a request ID for correlation but never internal details.

```json
// Good 400 error body
{
  "error": "validation_failed",
  "message": "Request validation failed",
  "details": [
    {"field": "email", "issue": "invalid_format"},
    {"field": "age", "issue": "must_be_positive"}
  ],
  "request_id": "req_abc123"
}
```

---

### Q4. What is idempotency in API design, and how do idempotency keys work for POST?

**Answer:**

An operation is **idempotent** if calling it multiple times produces the same result as calling it once. This is critical for safe retries in distributed systems where network failures cause clients to be uncertain whether their request succeeded.

**HTTP idempotency by verb:**
```
GET    → Idempotent (reading doesn't change state)
PUT    → Idempotent (setting a value to X twice = same as once)
DELETE → Idempotent (deleting a deleted resource returns 404 but no side-effect)
POST   → NOT idempotent (creating an order twice creates two orders)
PATCH  → NOT idempotent (PATCH {increment: 1} twice = +2, not +1)
```

**The POST idempotency problem:**
```
Client → POST /payments  → Network timeout
Did it work? Client doesn't know. If it retries:
  - Payment not processed → Retry creates it (correct)
  - Payment WAS processed → Retry creates duplicate charge (catastrophic)
```

**Idempotency Keys (used by Stripe, Braintree):**
The client generates a unique UUID for each logical operation and sends it as a header. The server stores the result keyed by this ID and returns the cached response for duplicate requests.

```
Client request:
POST /payments
Idempotency-Key: 550e8400-e29b-41d4-a716-446655440000

Server behavior:
1. Check if Idempotency-Key exists in store
2. If yes: return cached response (do NOT re-execute)
3. If no: process, store result keyed by ID, return result
```

```python
# Server implementation sketch
def create_payment(request):
    key = request.headers.get("Idempotency-Key")
    if key:
        cached = redis.get(f"idem:{key}")
        if cached:
            return json.loads(cached)  # Return cached response
    
    result = process_payment(request.body)
    
    if key:
        redis.setex(f"idem:{key}", 86400, json.dumps(result))  # Cache 24h
    
    return result
```

**Key design rules:**
- Store idempotency results for 24–48 hours.
- The key must be scoped to the operation type (a key used for payment cannot be reused for refund).
- Return `409 Conflict` if the same key is used with different request body.

---

### Q5. What is OpenAPI/Swagger, and what are the benefits of contract-first design?

**Answer:**

**OpenAPI** (formerly Swagger) is a standard, language-agnostic specification format for describing REST APIs in YAML or JSON. It defines endpoints, request/response schemas, authentication, and documentation in a machine-readable file.

```yaml
# openapi.yaml (excerpt)
openapi: "3.0.3"
info:
  title: Orders API
  version: "1.0"
paths:
  /orders:
    post:
      summary: Create an order
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreateOrderRequest'
      responses:
        '201':
          description: Order created
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Order'
        '400':
          $ref: '#/components/responses/ValidationError'
```

**Contract-first vs code-first:**

| Approach      | Description                              | Best For                    |
|---------------|------------------------------------------|-----------------------------|
| Code-first    | Write code, generate spec from it        | Prototyping, small teams    |
| Contract-first| Write spec, generate code stubs from it  | Multi-team, client SDKs     |

**Benefits of contract-first:**
1. **Parallel development:** Frontend and backend teams can work simultaneously using mock servers generated from the spec.
2. **Client SDK generation:** Tools like `openapi-generator` produce typed clients in Go, Python, TypeScript, Java automatically.
3. **Early validation:** Catch API design mistakes before any code is written.
4. **Living documentation:** Swagger UI provides interactive docs that are always in sync with the actual API.
5. **Contract testing:** Consumer-driven contract tests (Pact) can be validated against the spec.

```bash
# Generate Python client from spec
openapi-generator generate -i openapi.yaml -g python -o ./sdk/python

# Generate server stubs
openapi-generator generate -i openapi.yaml -g python-fastapi -o ./server
```

---

### Q6. What is an API gateway, and when should you use a Backend for Frontend (BFF)?

**Answer:**

**API Gateway** is a single entry point that sits in front of your microservices. It handles cross-cutting concerns so individual services don't have to.

```
Mobile App ─────┐
Web App ─────────┤──→ [API Gateway] ──→ Service A
Partner API ─────┘                  ──→ Service B
                                    ──→ Service C
```

**API Gateway responsibilities:**
- Authentication / Authorization (validate JWT, check scopes)
- Rate limiting and throttling
- SSL termination
- Request/response transformation
- Load balancing and routing
- Request logging and tracing injection
- Caching of responses

Tools: Kong, AWS API Gateway, NGINX, Traefik, Apigee.

**Backend for Frontend (BFF):**
A BFF is a dedicated API layer per client type (mobile BFF, web BFF, partner BFF). Each BFF aggregates multiple microservice calls into responses optimized for its specific client.

```
Mobile App ──→ [Mobile BFF]  ──→ User Service
Web App    ──→ [Web BFF]     ──→ Order Service
                             ──→ Product Service
                             ──→ Payment Service
```

**When to use BFF:**
- Different clients need different data shapes (mobile needs compressed payloads, web needs full data).
- One client team needs to move fast without coordination with other client teams.
- N+1 API call problem on the client side (client makes 5 calls where 1 BFF call would do).

**API Gateway vs BFF:**

| Concern                     | API Gateway  | BFF                       |
|-----------------------------|--------------|---------------------------|
| Auth, rate limiting, routing| Yes          | Sometimes                 |
| Client-specific aggregation | No           | Yes                       |
| Response shaping per client | No           | Yes                       |
| Number of instances         | 1 shared     | 1 per client type         |
| Who owns it                 | Platform team| Client team               |

**Recommendation:** Use an API Gateway for infrastructure concerns (auth, rate limiting). Add BFFs when client-specific orchestration becomes complex.

---

### Q7. How do you design pagination? Compare offset, cursor, and keyset approaches.

**Answer:**

Pagination limits the number of records returned per response. As datasets grow, the choice of pagination strategy significantly affects performance and UX.

**Offset Pagination:**
```
GET /orders?offset=0&limit=20   → rows 1–20
GET /orders?offset=20&limit=20  → rows 21–40
```
Simple and familiar. Supported by all SQL databases.

Cons:
- **Inconsistency:** If a new record is inserted before the next page request, items shift and a record is skipped or duplicated.
- **Deep page performance:** `OFFSET 1000000 LIMIT 20` requires scanning 1,000,020 rows.
- Not suitable for real-time feeds.

**Cursor Pagination:**
```
GET /orders?limit=20
Response: {"data": [...], "next_cursor": "eyJpZCI6IDIwfQ=="}

GET /orders?cursor=eyJpZCI6IDIwfQ==&limit=20
```
The cursor encodes the position in the dataset (typically a base64-encoded last-seen ID or timestamp).

Pros: Stable under inserts/deletes, consistent across pages.
Cons: No random access (cannot jump to page 50), requires sorting by a consistent column.

**Keyset Pagination (Seek Method):**
```sql
-- Page 1
SELECT * FROM orders ORDER BY created_at DESC, id DESC LIMIT 20

-- Page 2 (using last seen values from page 1)
SELECT * FROM orders
WHERE (created_at, id) < ('2024-01-15T10:00:00', 1234)
ORDER BY created_at DESC, id DESC
LIMIT 20
```

This is the most performant strategy. An index on `(created_at, id)` makes each page O(log n) regardless of page depth.

**Comparison:**

| Criterion          | Offset     | Cursor     | Keyset        |
|--------------------|------------|------------|---------------|
| Deep page speed    | Slow       | Fast       | Fast          |
| Stable under change| No         | Yes        | Yes           |
| Random access      | Yes        | No         | No            |
| Implementation     | Simple     | Medium     | Complex       |
| Use case           | Admin UIs  | Feeds      | High-volume   |

**Best practice:** Use cursor/keyset for production APIs. Provide both `next_cursor` and `has_more` in responses.

```json
{
  "data": [...],
  "pagination": {
    "next_cursor": "eyJpZCI6IDQyfQ==",
    "has_more": true
  }
}
```

---

## Medium Questions (Q8–Q15)

---

### Q8. What are GraphQL's advantages and how does it solve the N+1 problem with DataLoader?

**Answer:**

**GraphQL** is a query language for APIs that lets clients specify exactly what data they need. It was created by Facebook in 2012 and open-sourced in 2015.

**Advantages over REST:**
1. **No over-fetching:** REST returns fixed shapes; GraphQL returns exactly what was asked.
2. **No under-fetching:** Instead of multiple REST calls, one GraphQL query fetches nested data.
3. **Strongly typed schema:** The schema is a contract and enables tooling (introspection, autocomplete, validation).
4. **Single endpoint:** All queries go to `/graphql`, simplifying routing.

```graphql
# Get user with their recent orders (would require 2 REST calls)
query {
  user(id: "42") {
    name
    email
    orders(limit: 5) {
      id
      total
      status
    }
  }
}
```

**The N+1 Problem:**
```
Query: Get 10 users and their orders.

Naive resolver:
  1 query: SELECT * FROM users LIMIT 10
  10 queries: SELECT * FROM orders WHERE user_id = ?  (once per user)
  = N+1 queries = 11 queries total

As N grows, this becomes catastrophic.
```

**DataLoader Solution:**
DataLoader batches individual load calls into a single batch query, and caches results within a request.

```javascript
// DataLoader in Node.js
const ordersLoader = new DataLoader(async (userIds) => {
  // Called once with all IDs, not once per user
  const orders = await db.query(
    `SELECT * FROM orders WHERE user_id = ANY($1)`, [userIds]
  );
  // Return orders grouped by user_id in same order as input
  return userIds.map(id => orders.filter(o => o.user_id === id));
});

// Resolver
const userResolver = {
  orders: (user) => ordersLoader.load(user.id)  // Batched automatically
};
```

**Result:** Instead of N+1 queries, DataLoader issues 1 batch query: `SELECT * FROM orders WHERE user_id IN (1,2,3,...,10)`.

DataLoader also deduplicates: if the same user ID appears in multiple parts of the query, it is only fetched once per request.

---

### Q9. gRPC vs REST vs GraphQL — when do you use each?

**Answer:**

Each protocol solves different problems and excels in different contexts.

**REST:**
- Stateless request/response over HTTP/1.1
- Returns JSON (or XML)
- Universally supported
Best for: Public APIs, external integrations, browser-accessible APIs.

**gRPC:**
- Uses Protocol Buffers (binary, compact, strongly typed)
- Runs over HTTP/2 (multiplexing, bidirectional streaming)
- Code generation for clients in 10+ languages
Best for: Internal microservice communication, high-throughput services, polyglot systems.

**GraphQL:**
- Query language for flexible data fetching
- Single endpoint, client-defined shape
- Requires understanding of the schema
Best for: Mobile/web frontend with complex data requirements, teams that want to consolidate many REST endpoints.

**Decision matrix:**

| Scenario                               | Use          | Reason                              |
|----------------------------------------|--------------|-------------------------------------|
| Public API / third-party consumers     | REST         | Universal compatibility             |
| Internal service-to-service calls      | gRPC         | Speed, schema enforcement, streaming|
| Mobile app with diverse data needs     | GraphQL      | Reduces over-fetching               |
| Real-time bidirectional streaming      | gRPC         | Native streaming support            |
| Simple CRUD with clear resources       | REST         | Simpler to implement                |
| Multiple clients with different shapes | GraphQL      | One query, client-defined fields    |
| Ultra-low latency (< 10ms budget)      | gRPC         | Binary protocol, HTTP/2             |

**Performance comparison (approximate):**
```
gRPC    : ~7x faster than REST for equivalent payload (binary vs JSON)
REST    : 1x baseline
GraphQL : Similar to REST (JSON overhead), but may reduce total requests
```

**Combination pattern:**
```
External clients → REST or GraphQL (at API Gateway)
Internal services → gRPC (between microservices)
```

---

### Q10. How does API rate limiting work? Where to implement and which algorithm to use?

**Answer:**

Rate limiting protects a service from being overwhelmed by too many requests, whether from abuse, misbehaving clients, or DDoS.

**Where to implement:**
```
Client → CDN → API Gateway → Load Balancer → Service

Rate limiting should be at the API Gateway layer (before traffic reaches your services).
For per-user limits, it must be after authentication (so you know who is asking).
For global protection, it can be at the CDN (IP-based).
```

**Common algorithms:**

**1. Fixed Window Counter:**
```
Window: 60 seconds
Limit: 100 requests
Counter resets at the start of each minute.
Problem: 100 requests at 00:59 + 100 at 01:01 = 200 requests in 2 seconds.
```

**2. Sliding Window Log:**
Store timestamps of all requests. Count requests in the last N seconds.
```
Accurate but memory-intensive: O(requests) per user.
```

**3. Sliding Window Counter (hybrid):**
Approximate sliding window using two fixed windows and linear interpolation.
```
current_count = current_window_count + (previous_window_count × overlap_fraction)
Memory-efficient, approximately accurate (< 1% error).
```

**4. Token Bucket:**
A bucket holds up to `capacity` tokens. Tokens are added at a fixed `refill_rate`. Each request consumes one token.
```
Allows bursting up to capacity, then throttles to refill_rate.
Flexible, commonly used (AWS, Stripe, GitHub).
```

**5. Leaky Bucket:**
Requests enter a queue (bucket); they leave at a fixed rate.
```
Smooths traffic spikes entirely. No bursting allowed.
Good for downstream services that need steady throughput.
```

```python
# Token bucket in Redis (Lua for atomicity)
local tokens = tonumber(redis.call('get', KEYS[1])) or ARGV[1]
if tokens >= 1 then
    redis.call('set', KEYS[1], tokens - 1)
    redis.call('expire', KEYS[1], ARGV[2])
    return 1  -- allowed
else
    return 0  -- rate limited
end
```

**Response when rate limited:**
```
HTTP 429 Too Many Requests
Retry-After: 30
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1705324800
```

---

### Q11. How do webhooks work? Cover delivery guarantees, retries, and signature verification.

**Answer:**

A webhook is an HTTP callback — instead of a client polling for updates, the server pushes events to a pre-registered URL when something happens. It is a push-based integration pattern.

```
Provider System                  Consumer System
      │                                │
      │  Event: payment.completed      │
      ├──── POST /webhooks/payments ──→│
      │     {event: ..., data: ...}    │
      │                                │
      │←── 200 OK ─────────────────────┤
```

**Delivery guarantees:**
Webhooks are typically "at-least-once" — the provider retries until it receives a 2xx response. This means consumers must handle duplicate events.

```python
# Consumer: idempotent webhook handling
def handle_payment_webhook(event):
    event_id = event["id"]
    if redis.setnx(f"webhook:{event_id}", "1"):
        redis.expire(f"webhook:{event_id}", 86400)
        process_payment(event["data"])  # Process exactly once
    # else: duplicate, ignore silently
```

**Retry strategy (provider side):**
```
Retry schedule (exponential backoff):
  Attempt 1: immediate
  Attempt 2: 1 minute later
  Attempt 3: 5 minutes later
  Attempt 4: 30 minutes later
  Attempt 5: 2 hours later
  Attempt 6: 24 hours later
  → Dead letter queue or email notification after all retries fail
```

**Signature verification (HMAC):**
Without verification, anyone can send fake webhook payloads to your endpoint.

```python
# Provider: signs the webhook payload
import hmac, hashlib

secret = "whsec_abc123"
payload = json.dumps(event_data)
signature = hmac.new(
    secret.encode(), payload.encode(), hashlib.sha256
).hexdigest()
# Send header: X-Signature-256: sha256=<signature>

# Consumer: verifies the signature
def verify_webhook(payload: str, signature_header: str, secret: str) -> bool:
    expected = hmac.new(
        secret.encode(), payload.encode(), hashlib.sha256
    ).hexdigest()
    received = signature_header.replace("sha256=", "")
    return hmac.compare_digest(expected, received)  # Constant-time comparison
```

Always use `hmac.compare_digest` (not `==`) to prevent timing attacks.

**Best practices:**
- Include event timestamp in signature to prevent replay attacks.
- Respond within 5 seconds; process asynchronously via queue.
- Implement a webhook dashboard so consumers can re-deliver failed events.

---

### Q12. How do you design an API for mobile clients? Cover payload optimization and offline sync.

**Answer:**

Mobile APIs have unique constraints: high latency connections, limited bandwidth, battery consumption, intermittent connectivity, and diverse device capabilities.

**Payload optimization:**

**1. Sparse fields (field selection):**
```
GET /users/42?fields=id,name,avatar_url
→ Returns only requested fields, not the full user object
```

**2. Response compression:**
```
Client sends:  Accept-Encoding: gzip
Server sends:  Content-Encoding: gzip
Typical savings: 60-80% reduction in payload size for JSON.
```

**3. Protocol efficiency:**
```
REST/JSON  : Human-readable, verbose
gRPC/Proto : Binary, ~3-5x smaller, but requires code gen
GraphQL    : Client-defined shape, reduces over-fetching
```

**4. Image optimization:**
```
Accept: image/webp       → Serve WebP (30% smaller than JPEG)
Use responsive images: ?width=300&format=webp
CDN-based image transformation (Cloudinary, Imgix)
```

**Offline sync design:**

**Conflict-free approach (append-only):**
```
Every change is stored locally with a timestamp.
On reconnect, sync by sending all changes since last_sync_time.
Server applies changes, returns server's changes.
```

**CRDTs (Conflict-free Replicated Data Types):**
For collaborative editing (like a note-taking app), CRDTs merge conflicting changes automatically.

**Delta sync:**
```
Client → GET /sync?since=1705324800
Server → Returns only changed records since that timestamp
```

```json
// Sync response
{
  "changes": [
    {"id": "item_1", "updated_at": "2024-01-15T10:00:00Z", "data": {...}},
    {"id": "item_2", "deleted": true, "updated_at": "2024-01-15T10:05:00Z"}
  ],
  "server_time": "2024-01-15T10:10:00Z"  // Store as next since= value
}
```

**Network error handling:**
Use exponential backoff with jitter for retries. Track in-flight request state to avoid duplicate submissions (idempotency keys for mutations).

---

### Q13. What is backward compatibility, and how do you handle API deprecation?

**Answer:**

**Backward compatibility** means that changes to an API do not break existing clients that were written against an older version. Maintaining it allows clients to upgrade at their own pace.

**Breaking vs non-breaking changes:**

| Change Type                           | Breaking? |
|---------------------------------------|-----------|
| Adding a new optional field           | No        |
| Removing a field                      | Yes       |
| Renaming a field                      | Yes       |
| Changing a field type                 | Yes       |
| Adding a new required field           | Yes       |
| Adding a new optional query param     | No        |
| Adding a new endpoint                 | No        |
| Changing HTTP method                  | Yes       |
| Changing error response structure     | Yes       |
| Tightening validation rules           | Yes       |
| Loosening validation rules            | No        |

**Robustness principle (Postel's Law):** "Be conservative in what you send, liberal in what you accept." Parse responses ignoring unknown fields.

**Deprecation lifecycle:**
```
Phase 1 — Announce (6-12 months before sunset):
  - Add Deprecation header to responses
  - Document migration guide
  - Email existing API consumers
  - Add sunset date to API docs

Phase 2 — Warn (active but deprecated):
  Deprecation: true
  Sunset: Sat, 01 Jan 2025 00:00:00 GMT
  Link: <https://api.example.com/migration-guide>; rel="deprecation"

Phase 3 — Sunset:
  Return 410 Gone with migration instructions:
  {
    "error": "api_version_sunset",
    "message": "v1 was sunset on 2025-01-01. Use v2.",
    "migration_guide": "https://docs.example.com/v1-to-v2"
  }
```

**Versioning strategy:** When a breaking change is unavoidable, create a new major version (v2). Run v1 and v2 in parallel behind the API gateway. Sunset v1 only after usage drops to zero (or near zero, with client notification).

**Tip:** Track per-client version usage via API key analytics so you can identify who still uses deprecated versions and reach out directly.

---

### Q14. How do you design a long-running operation API pattern?

**Answer:**

Some operations (video encoding, data export, ML inference, large batch jobs) take seconds to minutes. Synchronous APIs cannot hold the connection that long. The solution is an async job pattern.

**Pattern: Submit → Poll → Result**

```
Step 1: Client submits job
POST /exports
{
  "type": "csv",
  "filters": {"date_range": "2024-01"},
  "notify_email": "user@example.com"
}

Response (202 Accepted):
{
  "job_id": "job_abc123",
  "status": "pending",
  "status_url": "https://api.example.com/jobs/job_abc123",
  "estimated_completion": "2024-01-15T10:05:00Z"
}

Step 2: Client polls status
GET /jobs/job_abc123
{
  "job_id": "job_abc123",
  "status": "running",          ← pending | running | completed | failed
  "progress": 45,               ← optional
  "created_at": "...",
  "updated_at": "..."
}

Step 3: Result is ready
GET /jobs/job_abc123
{
  "job_id": "job_abc123",
  "status": "completed",
  "result_url": "https://storage.example.com/exports/job_abc123.csv",
  "expires_at": "2024-01-22T10:00:00Z"
}
```

**Notification alternatives to polling:**
1. **Webhook:** Client registers a callback URL; server POSTs when done.
2. **Server-Sent Events:** Client subscribes to a `/jobs/{id}/events` SSE stream.

```
GET /jobs/job_abc123/events
Content-Type: text/event-stream

event: progress
data: {"percent": 45}

event: completed
data: {"result_url": "https://..."}
```

**Implementation:**
```
POST /exports
  → Validate input
  → Create job record in DB (status: pending)
  → Enqueue job message to queue (SQS, RabbitMQ)
  → Return 202 with job_id

Worker picks up job:
  → Update status: running
  → Execute long task
  → Update status: completed, store result_url
  → Optionally: POST to webhook callback URL
```

**Expiry:** Long-running result artifacts must have an expiry time. Store them in object storage (S3) with a signed URL that expires after a defined period.

---

### Q15. What is API security? Cover OAuth scopes, JWT, API keys, and request signing.

**Answer:**

API security has multiple layers: identity (who are you?), authorization (what can you do?), and integrity (has the request been tampered with?).

**API Keys:**
Simple opaque tokens sent in headers. Good for server-to-server calls.
```
Authorization: ApiKey sk-live-abc123xyz
```
Cons: No expiry, no scopes, hard to rotate en masse, broad access.
Use for: Simple B2B integrations, internal services.

**OAuth 2.0 + Scopes:**
Delegated authorization — a user grants an app limited access to their resources.
```
Scopes define what the token can access:
  read:orders     → Can read orders
  write:orders    → Can create/update orders
  admin:all       → Full access (dangerous, minimize use)

Token with insufficient scope:
  → 403 Forbidden
  → {"error": "insufficient_scope", "required_scope": "write:orders"}
```

**JWT (JSON Web Token):**
A self-contained token carrying claims (user ID, scopes, expiry) signed by the server.
```
Header.Payload.Signature

Payload:
{
  "sub": "user_42",
  "scopes": ["read:orders", "write:orders"],
  "exp": 1705410000,    ← Expiry timestamp
  "iat": 1705323600,    ← Issued at
  "iss": "https://auth.example.com"
}
```
Pros: Stateless (no database lookup per request), carries user info.
Cons: Cannot be revoked until expiry (use short TTL + refresh tokens).

**Request Signing (HMAC):**
Used when the API is called server-to-server and you need to prove the request was not tampered with in transit.
```
Signature = HMAC-SHA256(
  method + "\n" + path + "\n" + body_hash + "\n" + timestamp,
  secret_key
)

Header:
Authorization: HMAC-SHA256 Credential=key_id/date,
               SignedHeaders=host;content-type,
               Signature=abc123
```
AWS Signature Version 4 uses this approach.

**Security checklist:**
```
✓ Always HTTPS (never HTTP for APIs)
✓ Short-lived JWTs (15-60 minutes) + refresh tokens
✓ Validate scopes on every protected endpoint
✓ Rate limit per API key / user
✓ Log all authentication events
✓ Rotate secrets regularly
✓ Never put secrets in URL parameters (they end up in logs)
```

---

## Hard Questions (Q16–Q20)

---

### Q16. How does GraphQL Federation work? Explain supergraph and subgraphs.

**Answer:**

GraphQL Federation (Apollo Federation) is an architecture for splitting a single large GraphQL schema across multiple independently deployed services (subgraphs), which are composed into a unified graph (supergraph) by a gateway.

**Problem it solves:**
A monolithic GraphQL schema becomes a bottleneck when:
- Multiple teams contribute to it.
- Different parts have different deployment cadences.
- Different services have different scaling needs.

**Architecture:**
```
Client
  │
  ▼
[Apollo Router / Gateway]    ← Supergraph: unified schema
  │         │         │
  ▼         ▼         ▼
[Users    [Orders   [Products
 Subgraph] Subgraph] Subgraph]
  │         │         │
  ▼         ▼         ▼
Users DB  Orders DB  Products DB
```

**How it works:**

**Step 1: Each subgraph defines its slice of the schema**
```graphql
# Users subgraph
type User @key(fields: "id") {
  id: ID!
  name: String!
  email: String!
}

# Orders subgraph — extends User with orders
extend type User @key(fields: "id") {
  id: ID! @external
  orders: [Order!]!
}

type Order {
  id: ID!
  total: Float!
  user: User!
}
```

**Step 2: The gateway composes the supergraph schema**
```
Supergraph sees:
type User {
  id: ID!
  name: String!       ← from Users subgraph
  email: String!      ← from Users subgraph
  orders: [Order!]!   ← from Orders subgraph
}
```

**Step 3: Query planning**
```graphql
# Client query
query {
  user(id: "42") {
    name        # Resolved by Users subgraph
    orders {    # Resolved by Orders subgraph
      total
    }
  }
}
```
The gateway creates a query plan:
1. Fetch `user(id:42)` from Users subgraph → get `{id, name}`.
2. Fetch `orders` for `user.id=42` from Orders subgraph → get `{total}`.
3. Merge and return to client.

**@key directive** is the foreign key mechanism — it tells the gateway how to identify an entity across subgraphs.

**Benefits:** Teams deploy subgraphs independently. Schema is versioned per subgraph. Single client-facing endpoint. Breaking a subgraph doesn't break others.

---

### Q17. What is SSE vs WebSocket at the API layer, and when do you use each?

**Answer:**

Both SSE and WebSockets enable real-time communication, but they have fundamentally different models.

**Server-Sent Events (SSE):**
A one-way, server-to-client push channel over HTTP. The client opens a persistent connection; the server streams events as `text/event-stream`.

```javascript
// Client
const stream = new EventSource('/api/notifications');
stream.addEventListener('order_update', (e) => {
  console.log(JSON.parse(e.data));
});

// Server (Node.js)
app.get('/api/notifications', (req, res) => {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  
  const send = (event, data) => {
    res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  };
  
  const interval = setInterval(() => send('heartbeat', {ts: Date.now()}), 30000);
  req.on('close', () => clearInterval(interval));
});
```

**WebSockets:**
A full-duplex, bidirectional persistent connection. Both client and server can send messages at any time after the initial HTTP upgrade.

```javascript
// Client
const ws = new WebSocket('wss://api.example.com/realtime');
ws.onmessage = (e) => console.log(JSON.parse(e.data));
ws.send(JSON.stringify({type: 'subscribe', channel: 'order_123'}));
```

**Comparison:**

| Dimension             | SSE                          | WebSocket                        |
|-----------------------|------------------------------|----------------------------------|
| Direction             | Server → Client only         | Bidirectional                    |
| Protocol              | HTTP/1.1 or HTTP/2           | WS (upgraded HTTP)               |
| Load balancer support | Excellent (standard HTTP)    | Needs sticky sessions or broker  |
| Reconnection          | Built-in (Last-Event-ID)     | Manual implementation            |
| Browser support       | All modern browsers          | All modern browsers              |
| Firewall traversal    | Easy (HTTP port 443)         | Sometimes blocked                |
| Horizontal scaling    | Easy (stateless)             | Hard (stateful connection)       |
| Use case              | Notifications, live dashboards, progress | Chat, gaming, collaborative editing |

**Decision guide:**
```
Do clients need to SEND data over the connection?
  YES → WebSocket
  NO  → SSE

Is this a notification/feed use case?
  YES → SSE (simpler, better infrastructure support)

Is this collaborative real-time editing or gaming?
  YES → WebSocket (bidirectional is essential)
```

**HTTP/2 advantage:** SSE over HTTP/2 multiplexes multiple event streams over a single TCP connection, eliminating the connection limit problem of HTTP/1.1 (where browsers limit to 6 connections per domain).

---

### Q18. What is HATEOAS and is it worth implementing?

**Answer:**

**HATEOAS** (Hypermedia As The Engine Of Application State) is the highest level of REST maturity (Richardson Maturity Model Level 3). It means API responses include links to related actions and resources, so clients can navigate the API dynamically without hardcoded URLs.

**Example:**
```json
// Regular REST response
{
  "id": "order_123",
  "status": "pending",
  "total": 99.00
}

// HATEOAS response
{
  "id": "order_123",
  "status": "pending",
  "total": 99.00,
  "_links": {
    "self":    {"href": "/orders/order_123",          "method": "GET"},
    "cancel":  {"href": "/orders/order_123/cancel",   "method": "POST"},
    "pay":     {"href": "/orders/order_123/payment",  "method": "POST"},
    "items":   {"href": "/orders/order_123/items",    "method": "GET"}
  }
}
```

**Richardson Maturity Model:**
```
Level 0: RPC over HTTP (one endpoint, POST everything)
Level 1: Multiple resources (/orders, /users)
Level 2: HTTP verbs + status codes
Level 3: HATEOAS (self-documenting, hypermedia-driven)
```

**The theoretical value:**
- Server can change URLs without breaking clients (clients follow links, not hardcode paths).
- Responses are self-documenting — available actions are explicit.
- State machine navigation: only show valid next actions (e.g., can't cancel a delivered order).

**Is it worth implementing?**

**Honest answer: Usually no, for most APIs.**

| Argument For HATEOAS            | Counter-Argument                              |
|---------------------------------|-----------------------------------------------|
| URL changes don't break clients | URL changes are rare and API versioning handles it |
| Self-documenting                | OpenAPI spec documents APIs better             |
| Dynamic state machine           | Worth it for complex state machine APIs        |
| Pure REST                       | Clients still need to understand link rel types |

**When it IS worth it:**
- Complex state machine APIs (loan origination, order fulfillment with many states).
- APIs consumed by generic hypermedia clients (rare but exists in banking/insurance).
- When the API surface is so large that "what can I do next?" is genuinely hard to reason about.

**Real-world adoption:** GitHub's REST API uses links. Stripe does not. Most production APIs skip HATEOAS — they rely on good documentation instead.

---

### Q19. How do you design a search API with filters, sorting, and facets?

**Answer:**

A well-designed search API is flexible, consistent, and efficient. It separates concerns: full-text search, structured filtering, sorting, pagination, and faceting.

**Request design:**
```
GET /products/search?
  q=running+shoes          ← Full-text query
  &category=footwear       ← Structured filter
  &price_min=50            ← Range filter
  &price_max=200
  &brand=nike,adidas       ← Multi-value filter
  &in_stock=true           ← Boolean filter
  &sort=price:asc          ← Sort field:direction
  &page_size=20
  &cursor=eyJpZCI6IDIwfQ== ← Cursor for pagination
  &facets=category,brand   ← Request facet aggregations
```

**Response design:**
```json
{
  "query": "running shoes",
  "total": 1234,
  "hits": [
    {
      "id": "prod_123",
      "name": "Nike Air Zoom Pegasus",
      "price": 130.00,
      "category": "footwear",
      "_score": 0.92      ← Relevance score (optional)
    }
  ],
  "facets": {
    "category": [
      {"value": "footwear",    "count": 980},
      {"value": "accessories", "count": 254}
    ],
    "brand": [
      {"value": "nike",    "count": 540},
      {"value": "adidas",  "count": 380}
    ],
    "price_range": [
      {"range": "0-50",    "count": 120},
      {"range": "50-100",  "count": 400},
      {"range": "100-200", "count": 460}
    ]
  },
  "pagination": {
    "next_cursor": "eyJpZCI6IDQyfQ==",
    "has_more": true
  }
}
```

**Filter expression language (for complex queries):**
```
GET /products/search?filter=category:footwear AND (price:[50 TO 200]) AND brand:(nike OR adidas)
```

**Elasticsearch backend mapping:**
```json
{
  "query": {
    "bool": {
      "must": [{"multi_match": {"query": "running shoes", "fields": ["name^2", "description"]}}],
      "filter": [
        {"term": {"category": "footwear"}},
        {"range": {"price": {"gte": 50, "lte": 200}}},
        {"terms": {"brand": ["nike", "adidas"]}}
      ]
    }
  },
  "aggs": {
    "categories": {"terms": {"field": "category.keyword"}},
    "brands": {"terms": {"field": "brand.keyword"}},
    "price_ranges": {"range": {"field": "price", "ranges": [...]}}
  },
  "sort": [{"price": "asc"}]
}
```

**Best practices:**
- Use `keyword` fields for filters/facets, `text` fields for full-text search.
- Implement `search_after` (keyset) pagination for deep pages.
- Cache facet results separately if they are expensive and change infrequently.
- Add request timeout to search queries (return partial results rather than timeout error).

---

### Q20. What is the difference between API throttling and rate limiting?

**Answer:**

These terms are often used interchangeably, but they represent distinct mechanisms with different goals.

**Rate Limiting:** A hard enforcement mechanism that restricts the number of requests a client can make within a time window. When exceeded, requests are rejected with `429 Too Many Requests`.

```
Rate limit: 1000 requests/hour
At request 1001: immediately return 429
Goal: Protect the server from excessive load and abuse
```

**Throttling:** A softer control mechanism that slows down or queues requests when demand exceeds capacity, rather than outright rejecting them. The goal is to degrade gracefully.

```
Throttle: Max 100 concurrent requests
At request 101: queue the request, add processing delay
Goal: Smooth demand spikes without dropping requests
```

**Comparison:**

| Dimension           | Rate Limiting              | Throttling                      |
|---------------------|----------------------------|---------------------------------|
| Response to excess  | Reject (429)               | Delay / queue                   |
| Client experience   | Hard failure               | Slower response                 |
| Goal                | Enforce fair use, prevent abuse | Smooth demand, protect capacity |
| Implementation      | Token bucket, sliding window | Queue depth, concurrency limits |
| User-facing         | Yes (public APIs)          | Often internal                  |

**Combined usage:**
Most production systems use both:
```
Layer 1 (Edge/CDN):        Rate limit by IP → 429 if > 10 req/sec (DDoS protection)
Layer 2 (API Gateway):     Rate limit by API key → 429 if > 1000 req/hour (fair use)
Layer 3 (Service):         Throttle by concurrency → queue if > 50 concurrent (overload protection)
Layer 4 (Database):        Connection pool throttling → queue if all connections busy
```

**Throttling implementation patterns:**
```python
# Token bucket for smooth throttling (allow bursts)
# Leaky bucket for strict rate smoothing (no bursts)
# Semaphore for concurrency limiting

import asyncio
semaphore = asyncio.Semaphore(50)  # Max 50 concurrent

async def handle_request(request):
    async with semaphore:        # Queues if 50 already running
        return await process(request)
```

**Circuit Breaker** is related but different: it completely stops sending requests to a downstream service that is failing, to prevent cascade failures. All three (rate limiting, throttling, circuit breaking) are layers of production traffic management.

---

## Quick Reference

```
HTTP VERBS + IDEMPOTENCY
  GET    → Safe + Idempotent (read only)
  PUT    → Idempotent (full replace)
  DELETE → Idempotent
  POST   → Not idempotent (use idempotency keys)
  PATCH  → Not idempotent

CRITICAL STATUS CODES
  200 OK | 201 Created | 202 Accepted | 204 No Content
  400 Bad Request | 401 Unauth | 403 Forbidden | 404 Not Found
  409 Conflict | 422 Unprocessable | 429 Too Many Requests
  500 Server Error | 502 Bad Gateway | 503 Unavailable | 504 Timeout

API VERSIONING
  URL path    → /v1/users (most common, CDN-friendly)
  Header      → Accept: application/vnd.app.v2+json (REST-pure)
  Query param → ?version=2 (not cacheable)

PAGINATION
  Offset  → Simple, inconsistent, slow at depth
  Cursor  → Stable, no random access, good for feeds
  Keyset  → Fastest, stable, use for production

RATE LIMITING ALGORITHMS
  Token Bucket  → Allows bursts, then rate-limits
  Leaky Bucket  → Strict smoothing, no bursts
  Sliding Window → Accurate, memory-intensive

IDEMPOTENCY KEY FLOW
  Client generates UUID → Sends as header →
  Server checks cache → If exists: return cached →
  If new: execute + cache result

WEBHOOK SECURITY
  HMAC-SHA256 signature → Timestamp to prevent replay →
  Idempotency check → Process async (queue)

GRAPHQL N+1 FIX
  DataLoader → batch + cache individual loads per request

gRPC vs REST vs GraphQL
  gRPC      → internal service-to-service (fast, binary, typed)
  REST      → public/external APIs (universal)
  GraphQL   → mobile/web with complex data needs

BFF vs API GATEWAY
  API Gateway → Auth, rate limiting, routing (shared)
  BFF         → Client-specific aggregation (per client team)

OAUTH SCOPES
  read:resource  → Read-only access
  write:resource → Create/update
  admin:resource → Full access (minimize)

JWT KEY PROPERTIES
  sub (user id), exp (expiry), scopes, iss (issuer)
  Short TTL (15-60 min) + refresh token pattern
```

---

*File 12 of 15 — API Design*
