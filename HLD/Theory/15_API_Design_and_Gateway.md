# 15. API Design and Gateway

## Table of Contents
1. [REST API Design Principles](#1-rest-api-design-principles)
2. [RESTful Naming and Versioning](#2-restful-naming-and-versioning)
3. [REST Best Practices: Pagination, Filtering, Sorting](#3-rest-best-practices-pagination-filtering-sorting)
4. [GraphQL](#4-graphql)
5. [GraphQL vs REST](#5-graphql-vs-rest)
6. [gRPC](#6-grpc)
7. [API Protocol Comparison](#7-api-protocol-comparison)
8. [API Gateway Responsibilities](#8-api-gateway-responsibilities)
9. [API Gateway Patterns](#9-api-gateway-patterns)
10. [Backend for Frontend (BFF)](#10-backend-for-frontend-bff)
11. [API Rate Limiting](#11-api-rate-limiting)
12. [API Versioning and Deprecation](#12-api-versioning-and-deprecation)
13. [API Security at Gateway](#13-api-security-at-gateway)
14. [Webhook Design](#14-webhook-design)
15. [WebSocket at Scale](#15-websocket-at-scale)
16. [Long Polling vs SSE vs WebSocket](#16-long-polling-vs-sse-vs-websocket)
17. [OpenAPI / Swagger](#17-openapi--swagger)
18. [API Design for Mobile](#18-api-design-for-mobile)
19. [GraphQL Federation](#19-graphql-federation)
20. [Quick Reference](#20-quick-reference)

---

## 1. REST API Design Principles

### What is REST?

Representational State Transfer — an architectural style for distributed hypermedia systems. It is not a protocol or standard.

### Six REST Constraints

1. **Client-Server**: UI and data storage are separated. Client doesn't know about server storage; server doesn't know about UI.
2. **Stateless**: Each request contains all information needed. No client session state on server.
3. **Cacheable**: Responses must define themselves as cacheable or non-cacheable.
4. **Uniform Interface**: Standardized method of communication (HTTP verbs, status codes, URIs).
5. **Layered System**: Client cannot tell if connected directly to server or intermediary (proxy, CDN).
6. **Code on Demand** (optional): Server can send executable code to client (JavaScript).

### Resources and HTTP Methods

```
Resource: A noun — a thing (user, order, product, comment)
Method:   What you do to the resource

HTTP Method → CRUD → SQL Equivalent
GET         → Read   → SELECT
POST        → Create → INSERT
PUT         → Replace (full update) → UPDATE (full)
PATCH       → Partial Update → UPDATE (partial)
DELETE      → Delete → DELETE
HEAD        → GET metadata only (no body)
OPTIONS     → List supported methods (CORS preflight)
```

### Resource-Oriented URLs

```
# Users resource
GET    /users              → List all users
POST   /users              → Create a new user
GET    /users/{id}         → Get user by ID
PUT    /users/{id}         → Replace user
PATCH  /users/{id}         → Partially update user
DELETE /users/{id}         → Delete user

# Nested resources (relationships)
GET    /users/{id}/orders          → List orders for user
POST   /users/{id}/orders          → Create order for user
GET    /users/{id}/orders/{oid}    → Get specific order for user

# Non-CRUD actions (use nouns for sub-resources)
POST   /users/{id}/activate        → Activate user
POST   /orders/{id}/cancel         → Cancel order
POST   /payments/{id}/refund       → Refund payment
```

### HTTP Status Codes

```
2xx — Success
  200 OK                  → Standard success
  201 Created             → Resource created (POST). Include Location header.
  202 Accepted            → Async processing started, will complete later
  204 No Content          → Success, no body (DELETE, PATCH with no response)
  206 Partial Content     → Range request (file download/streaming)

3xx — Redirection
  301 Moved Permanently   → URL permanently changed (update bookmarks)
  302 Found               → Temporary redirect
  304 Not Modified        → Cached response still valid (ETag/If-None-Match)
  307 Temporary Redirect  → Like 302, but preserve HTTP method
  308 Permanent Redirect  → Like 301, but preserve HTTP method

4xx — Client Error
  400 Bad Request         → Invalid request format/parameters
  401 Unauthorized        → Authentication required
  403 Forbidden           → Authenticated but not authorized
  404 Not Found           → Resource doesn't exist
  405 Method Not Allowed  → HTTP method not supported
  409 Conflict            → State conflict (duplicate resource, optimistic locking)
  410 Gone                → Resource permanently deleted (stronger than 404)
  422 Unprocessable Entity → Validation error (correct format, invalid semantics)
  429 Too Many Requests   → Rate limited
  
5xx — Server Error
  500 Internal Server Error → Generic server error
  502 Bad Gateway           → Upstream service returned invalid response
  503 Service Unavailable   → Server overloaded or in maintenance
  504 Gateway Timeout       → Upstream service timed out
```

### HATEOAS

Hypermedia As The Engine Of Application State — responses include links to possible next actions.

```json
// GET /orders/123
{
  "id": 123,
  "status": "pending",
  "total": 149.99,
  "_links": {
    "self": { "href": "/orders/123" },
    "cancel": { "href": "/orders/123/cancel", "method": "POST" },
    "pay": { "href": "/orders/123/payment", "method": "POST" },
    "user": { "href": "/users/42" }
  }
}
```

**Practical note**: Full HATEOAS is rarely implemented in practice (complexity vs benefit). Most teams implement the resource + links design but not the full hypermedia constraint.

---

## 2. RESTful Naming and Versioning

### Naming Conventions

```
Use nouns, not verbs:
  BAD:  GET /getUsers
        POST /createOrder
        DELETE /deleteComment
  GOOD: GET /users
        POST /orders
        DELETE /comments/{id}

Use lowercase with hyphens (kebab-case):
  BAD:  /userOrders
        /User_Orders
  GOOD: /user-orders

Plural for collections:
  BAD:  /user, /order
  GOOD: /users, /orders

Consistent nesting:
  /users/{userId}/orders/{orderId}/items
```

### Versioning Strategies

#### URL Path Versioning (Most Common)
```
/v1/users
/v2/users
/v1/orders
```
- Most visible and explicit
- Easy to route at CDN/gateway level
- Easy to test in browser
- Can be bookmarked
- Version in URL is "impure" (URL should identify resource, not its version)

#### Header Versioning
```http
GET /users
Accept: application/vnd.example.v2+json

# or custom header
GET /users
API-Version: 2
```
- Cleaner URLs
- Harder to test in browser
- Client must explicitly set header

#### Query Parameter Versioning
```
GET /users?version=2
GET /users?api-version=2024-01-01
```
- Easy to test in browser
- Can be forgotten/cached incorrectly

#### Content Negotiation (RFC-compliant)
```http
GET /users
Accept: application/vnd.example.users.v2+json
```
- Technically most RESTful
- Verbose and complex for clients

### Version Comparison

| Strategy | Pros | Cons | Caching | Popularity |
|---|---|---|---|---|
| URL path (/v2/) | Explicit, easy | URL semantics | Yes (by URL) | Most common |
| Header | Clean URLs | Hidden, hard to debug | Varies | Common in enterprise |
| Query param | Easy to test | Cache issues | Varies | Less common |
| Content type | RFC-compliant | Complex | Complex | Rare |

---

## 3. REST Best Practices: Pagination, Filtering, Sorting

### Pagination

#### Offset-Based Pagination
```
GET /orders?page=3&limit=20
GET /orders?offset=40&limit=20

Response:
{
  "data": [...],
  "pagination": {
    "total": 4527,
    "page": 3,
    "limit": 20,
    "pages": 227,
    "next": "/orders?page=4&limit=20",
    "prev": "/orders?page=2&limit=20"
  }
}
```

**Problems**:
- Page drift: if item inserted on page 1 while user is on page 3, they skip an item
- Slow for large offsets: `SELECT ... LIMIT 20 OFFSET 10000` scans 10,020 rows
- Good for: small datasets, non-real-time data, random page access

#### Cursor-Based Pagination
```
GET /orders?cursor=eyJpZCI6MTIzfQ==&limit=20

Response:
{
  "data": [...],
  "pagination": {
    "next_cursor": "eyJpZCI6MTQzfQ==",
    "has_next": true
  }
}
```

Cursor is typically encoded: `base64({"id": 123, "created_at": "2024-01-15T10:30:00Z"})`

SQL implementation:
```sql
-- Cursor: last seen id = 123
SELECT * FROM orders
WHERE id > 123
ORDER BY id ASC
LIMIT 20;
```

**Advantages**:
- No page drift (cursor is stable)
- Consistent performance regardless of offset
- Works well with real-time data

**Limitations**:
- No random page access
- Cursor must be opaque to client
- More complex implementation

#### Keyset Pagination (Seek Method)
Similar to cursor but uses actual column values:

```sql
-- Sort by (created_at DESC, id DESC), last seen: created_at='2024-01-15', id=456
SELECT * FROM orders
WHERE (created_at, id) < ('2024-01-15T10:30:00Z', 456)
ORDER BY created_at DESC, id DESC
LIMIT 20;
```

Requires a composite index on the sort columns.

### Filtering

```
GET /orders?status=pending
GET /orders?status=pending,completed      → OR filter
GET /products?price[gte]=10&price[lte]=50 → Range
GET /users?created_at[gte]=2024-01-01
GET /users?name[like]=alice              → Pattern match

# Complex filter (JSON)
GET /orders?filter={"status":"pending","user.country":"US"}
```

### Sorting

```
GET /orders?sort=created_at            → Ascending
GET /orders?sort=-created_at           → Descending (minus prefix)
GET /orders?sort=-created_at,total     → Multi-field: newest first, then by total

# Alternative
GET /orders?sort_by=created_at&sort_order=desc
```

### Partial Responses (Field Selection)

```
GET /users/42?fields=id,name,email
→ Returns only requested fields (reduce payload)

GET /orders?fields=id,status,total
→ Useful for list views where full object isn't needed
```

---

## 4. GraphQL

### Core Concepts

GraphQL is a query language for APIs and a runtime for executing those queries.

### Schema Definition

```graphql
type User {
  id: ID!
  name: String!
  email: String!
  orders(status: OrderStatus, limit: Int): [Order!]!
  createdAt: DateTime!
}

type Order {
  id: ID!
  status: OrderStatus!
  total: Float!
  items: [OrderItem!]!
  user: User!
}

enum OrderStatus {
  PENDING
  PROCESSING
  SHIPPED
  DELIVERED
  CANCELLED
}

type Query {
  user(id: ID!): User
  users(filter: UserFilter, first: Int, after: String): UserConnection!
  order(id: ID!): Order
}

type Mutation {
  createOrder(input: CreateOrderInput!): Order!
  cancelOrder(id: ID!): Order!
  updateUser(id: ID!, input: UpdateUserInput!): User!
}

type Subscription {
  orderStatusChanged(orderId: ID!): Order!
}
```

### Queries

```graphql
# Query — fetch specific fields
query GetUser {
  user(id: "42") {
    id
    name
    email
    orders(status: PENDING, limit: 5) {
      id
      total
      items {
        productName
        quantity
        price
      }
    }
  }
}

# Variables
query GetUserOrders($userId: ID!, $status: OrderStatus) {
  user(id: $userId) {
    orders(status: $status) {
      id
      total
    }
  }
}
# Variables: {"userId": "42", "status": "PENDING"}

# Fragments — reuse field selections
fragment OrderDetails on Order {
  id
  status
  total
  createdAt
}

query GetOrders {
  pendingOrders: orders(filter: {status: PENDING}) {
    ...OrderDetails
  }
  shippedOrders: orders(filter: {status: SHIPPED}) {
    ...OrderDetails
  }
}
```

### Mutations

```graphql
mutation CreateOrder($input: CreateOrderInput!) {
  createOrder(input: $input) {
    id
    status
    total
    items {
      productName
      quantity
    }
  }
}
# Variables: {"input": {"items": [{"productId": "1", "quantity": 2}]}}
```

### Subscriptions

```graphql
subscription OnOrderUpdate($orderId: ID!) {
  orderStatusChanged(orderId: $orderId) {
    id
    status
    updatedAt
  }
}
```

### N+1 Problem and DataLoader

```python
# N+1 Problem: fetching orders, then user for each order = N+1 DB queries
async def resolve_order_user(order, info):
    return await User.get(id=order.user_id)  # Called N times!

# DataLoader solution: batch and cache within a single request
from aiodataloader import DataLoader

class UserLoader(DataLoader):
    async def batch_load_fn(self, user_ids):
        # ONE query for all user IDs
        users = await User.filter(id__in=user_ids)
        user_map = {u.id: u for u in users}
        return [user_map.get(uid) for uid in user_ids]

user_loader = UserLoader()

async def resolve_order_user(order, info):
    return await user_loader.load(order.user_id)  # Batched!
```

DataLoader collects all calls within a tick, then fires ONE batched DB query.

---

## 5. GraphQL vs REST

### Feature Comparison

| Feature | REST | GraphQL |
|---|---|---|
| Data fetching | Fixed endpoints, fixed response | Client specifies exact fields needed |
| Over-fetching | Common (get entire object) | Eliminated |
| Under-fetching | Multiple requests needed | Single request possible |
| Versioning | URL versions (/v1, /v2) | Schema evolution with deprecations |
| Caching | Easy (HTTP caching by URL) | Harder (POST requests, no URL variance) |
| File uploads | Easy | Complex (multipart) |
| Real-time | SSE/WebSocket (external) | Subscriptions (built-in) |
| Learning curve | Low | Medium |
| Tooling | Extensive | Good (Apollo, Relay) |
| Type safety | OpenAPI/Swagger | Schema is the contract |

### When to Choose Each

**Choose REST when**:
- Public API (third parties, simple integrations)
- HTTP caching is critical
- File uploads/downloads
- Simple CRUD resources
- Team is new to APIs

**Choose GraphQL when**:
- Multiple client types with different data needs (mobile vs web)
- Aggregating data from multiple microservices
- Rapid frontend iteration without backend changes
- Complex interconnected data (social graphs)
- Bandwidth-constrained mobile clients

---

## 6. gRPC

### What is gRPC?

Google Remote Procedure Call. Uses Protocol Buffers (binary serialization) over HTTP/2.

### Protocol Buffers

```protobuf
// orders.proto
syntax = "proto3";
package orders.v1;

service OrderService {
  rpc GetOrder (GetOrderRequest) returns (Order);
  rpc ListOrders (ListOrdersRequest) returns (ListOrdersResponse);
  rpc CreateOrder (CreateOrderRequest) returns (Order);
  
  // Streaming
  rpc StreamOrderUpdates (StreamRequest) returns (stream OrderUpdate);  // Server streaming
  rpc UploadOrderItems (stream OrderItem) returns (UploadResponse);    // Client streaming
  rpc OrderChat (stream ChatMessage) returns (stream ChatMessage);     // Bidirectional
}

message Order {
  string id = 1;
  string status = 2;
  double total = 3;
  int64 created_at = 4;
  repeated OrderItem items = 5;
  User user = 6;
}

message OrderItem {
  string product_id = 1;
  string product_name = 2;
  int32 quantity = 3;
  double price = 4;
}

message GetOrderRequest {
  string order_id = 1;
}
```

### Code Generation

```bash
# Generate Python gRPC stubs
python -m grpc_tools.protoc \
  -I. \
  --python_out=. \
  --grpc_python_out=. \
  orders.proto
```

### Server Implementation (Python)

```python
import grpc
from concurrent import futures
import orders_pb2
import orders_pb2_grpc

class OrderServiceServicer(orders_pb2_grpc.OrderServiceServicer):
    def GetOrder(self, request, context):
        order = db.get_order(request.order_id)
        if not order:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Order {request.order_id} not found")
            return orders_pb2.Order()
        
        return orders_pb2.Order(
            id=order.id,
            status=order.status,
            total=order.total
        )
    
    def StreamOrderUpdates(self, request, context):
        # Server streaming — yields multiple responses
        while True:
            update = get_next_update(request.order_id)
            if update:
                yield orders_pb2.OrderUpdate(
                    order_id=update.order_id,
                    new_status=update.status
                )

server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
orders_pb2_grpc.add_OrderServiceServicer_to_server(
    OrderServiceServicer(), server
)
server.add_insecure_port('[::]:50051')
server.start()
```

### HTTP/2 Benefits for gRPC

```
HTTP/1.1: One request per connection (head-of-line blocking)
         Can use keep-alive + pipelining but limited

HTTP/2:
  - Multiplexing: Multiple streams over one connection simultaneously
  - Header compression (HPACK): Reduce header overhead
  - Server push: Server can proactively send resources
  - Binary framing: More efficient than text-based HTTP/1.1

gRPC benefits:
  - Multiple RPCs over single connection (no connection overhead per call)
  - Bidirectional streaming
  - Flow control per stream
```

### Streaming Patterns

| Pattern | Direction | Example |
|---|---|---|
| Unary | Client → single req → Server → single resp | Get user by ID |
| Server streaming | Client → req → Server → stream of resp | Live feed, file download |
| Client streaming | Client → stream → Server → single resp | Bulk upload, file upload |
| Bidirectional | Client ↔ Server streaming | Chat, real-time collaboration |

---

## 7. API Protocol Comparison

### Performance

```
Payload size for same data:
  JSON (REST/GraphQL): ~1000 bytes
  Protobuf (gRPC):     ~200 bytes (5x smaller)

Serialization speed:
  JSON parse:    ~100ms / 1M objects
  Protobuf:      ~15ms / 1M objects (7x faster)

Connection overhead:
  REST (HTTP/1.1): New TCP connection per request (or keep-alive)
  gRPC (HTTP/2):   Single connection, multiplexed streams
  GraphQL:         Single HTTP endpoint, batching supported
```

### Protocol Comparison Table

| Aspect | REST | GraphQL | gRPC |
|---|---|---|---|
| Protocol | HTTP/1.1 or HTTP/2 | HTTP/1.1 or HTTP/2 | HTTP/2 only |
| Format | JSON (text) | JSON (text) | Protobuf (binary) |
| Type safety | Via OpenAPI (optional) | Schema-enforced | Strong (Protobuf) |
| Browser support | Native | Native | Limited (grpc-web) |
| Streaming | SSE, WebSocket | Subscriptions | Native (4 modes) |
| Caching | HTTP caching | Harder | No HTTP caching |
| Learning curve | Low | Medium | Medium-High |
| Code generation | Optional | Optional | Required |
| Best for | Public APIs | Complex frontends | Internal microservices |

---

## 8. API Gateway Responsibilities

### What an API Gateway Does

```
Internet → [API Gateway] → Microservices

Gateway handles:
  1. Routing
  2. Authentication/Authorization
  3. Rate Limiting
  4. Request/Response Transformation
  5. Circuit Breaking
  6. Caching
  7. Logging and Monitoring
  8. SSL Termination
  9. Load Balancing
  10. Protocol Translation
```

### Routing

```yaml
# Kong route configuration
routes:
  - name: user-api
    paths: ["/v1/users"]
    strip_path: false
    service: user-service
  
  - name: order-api
    paths: ["/v1/orders"]
    methods: ["GET", "POST"]
    service: order-service
    
  - name: legacy-api
    paths: ["/api/old-endpoint"]
    service: new-service
    plugins:
      - name: request-transformer
        config:
          rename:
            uri: "/new-endpoint"
```

### Request/Response Transformation

```
Client sends:              API Gateway transforms:        Service receives:
POST /v1/orders            →  POST /orders                →  POST /orders
X-App-Token: abc123           Authorization: Bearer jwt       X-User-Id: 42
Body: {mobile_format}         Body: {service_format}          Body: {service_format}
```

```yaml
# Response transformation
plugins:
  - name: response-transformer
    config:
      remove:
        headers: ["X-Internal-Service-Id", "X-Debug-Info"]
      add:
        headers: ["X-Request-Id: ${request_id}"]
```

### Circuit Breaking at Gateway

```
Normal:  Client → Gateway → Service (200ms, success)
Degraded: Client → Gateway → Service (10s timeout, fail)
Open:    Client → Gateway → [Circuit Open] → Return 503 immediately
Half-open: Allow 1 request through → if success, close circuit
```

```yaml
# Envoy circuit breaker configuration
circuit_breakers:
  thresholds:
    - priority: DEFAULT
      max_connections: 1000
      max_pending_requests: 1000
      max_requests: 1000
      max_retries: 3
```

### Caching at Gateway

```yaml
# Cache GET /products for 60 seconds
plugins:
  - name: proxy-cache
    config:
      response_code: [200, 301, 404]
      request_method: ["GET", "HEAD"]
      content_type: ["application/json"]
      cache_ttl: 60
      strategy: memory
```

---

## 9. API Gateway Patterns

### Kong

Open-source, extensible, Lua-based plugins.

```yaml
# docker-compose.yml for Kong
services:
  kong:
    image: kong:3.4
    environment:
      KONG_DATABASE: postgres
      KONG_PG_HOST: kong-database
      KONG_PROXY_ACCESS_LOG: /dev/stdout
      KONG_ADMIN_ACCESS_LOG: /dev/stdout
    ports:
      - "8000:8000"   # Proxy
      - "8001:8001"   # Admin API
      - "8443:8443"   # Proxy SSL
```

```bash
# Register a service and route via Admin API
curl -X POST http://localhost:8001/services \
  -d name=user-service \
  -d url=http://user-service:3000

curl -X POST http://localhost:8001/services/user-service/routes \
  -d "paths[]=/v1/users"

# Add rate limiting plugin
curl -X POST http://localhost:8001/services/user-service/plugins \
  -d name=rate-limiting \
  -d "config.minute=100" \
  -d "config.hour=1000"
```

### AWS API Gateway

Managed service, integrates natively with Lambda, ECS, ALB.

```yaml
# Serverless Framework / SAM template
Resources:
  ApiGateway:
    Type: AWS::ApiGateway::RestApi
    Properties:
      Name: my-api
      
  UserResource:
    Type: AWS::ApiGateway::Resource
    Properties:
      ParentId: !GetAtt ApiGateway.RootResourceId
      PathPart: users
      RestApiId: !Ref ApiGateway

  GetUsersMethod:
    Type: AWS::ApiGateway::Method
    Properties:
      HttpMethod: GET
      AuthorizationType: COGNITO_USER_POOLS
      AuthorizerId: !Ref CognitoAuthorizer
      Integration:
        Type: AWS_PROXY
        IntegrationHttpMethod: POST
        Uri: !Sub "arn:aws:apigateway:${AWS::Region}:lambda:path/..."
```

### Envoy

High-performance proxy, backbone of Istio service mesh.

```yaml
# Envoy configuration
static_resources:
  listeners:
    - address:
        socket_address: { address: 0.0.0.0, port_value: 10000 }
      filter_chains:
        - filters:
            - name: envoy.filters.network.http_connection_manager
              typed_config:
                "@type": type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager
                route_config:
                  virtual_hosts:
                    - name: local_service
                      domains: ["*"]
                      routes:
                        - match: { prefix: "/v1/users" }
                          route: { cluster: user_service }
                        - match: { prefix: "/v1/orders" }
                          route: { cluster: order_service }
```

### Gateway Comparison

| Aspect | Kong | AWS API GW | NGINX | Envoy |
|---|---|---|---|---|
| Type | Open source | Managed | Open source | Open source |
| Plugin system | Lua plugins (rich) | Lambda authorizers | Modules | Filters |
| Service mesh | No (KongMesh) | No | No | Yes (Istio) |
| Protocol support | REST, gRPC, WebSocket | REST, WebSocket, HTTP/2 | All | All |
| Configuration | Admin API / YAML | Console / IaC | nginx.conf | xDS APIs |
| Best for | General purpose | AWS-native apps | Simple proxy/LB | K8s service mesh |

---

## 10. Backend for Frontend (BFF)

### Problem: One API, Many Clients

```
Mobile app needs:  Small payloads, offline support, compressed images
Web app needs:     Rich data, full images, complex filtering
IoT device needs:  Minimal payload, binary format, low power
```

### BFF Pattern

```
Mobile Client ────────► Mobile BFF ────────┐
Web Client   ────────► Web BFF    ──────────┤ → Backend Microservices
IoT Client   ────────► IoT BFF   ────────── ┘

Each BFF:
  - Tailored exactly to its client's needs
  - Owned by the frontend team (same team, same repo)
  - Aggregates multiple backend calls
  - Handles client-specific transformation
  - Different rate limits / auth flows per client
```

```javascript
// Mobile BFF — aggregates 3 backend calls into 1 mobile-optimized response
app.get('/mobile/dashboard', async (req, res) => {
  const [user, orders, recommendations] = await Promise.all([
    userService.getUser(req.userId),
    orderService.getRecentOrders(req.userId, { limit: 3 }),
    recommendationService.getPersonalized(req.userId, { limit: 5 })
  ]);
  
  // Mobile-optimized response (smaller payload)
  res.json({
    user: { id: user.id, name: user.firstName, avatar: user.avatarThumb },
    orders: orders.map(o => ({ id: o.id, status: o.status, total: o.total })),
    recommendations: recommendations.map(r => ({ id: r.id, title: r.name, price: r.price }))
  });
});
```

### BFF vs API Gateway

| Aspect | API Gateway | BFF |
|---|---|---|
| Ownership | Platform/Infra team | Frontend team |
| Logic | Cross-cutting (auth, rate limit) | Client-specific aggregation |
| Number | One (or few) | One per client type |
| Customization | Generic | Per-client optimized |

**Best practice**: Use both. API Gateway for cross-cutting concerns, BFF for client-specific aggregation.

---

## 11. API Rate Limiting

### Algorithms

#### Token Bucket

```python
class TokenBucketRateLimiter:
    """
    Tokens accumulated at constant rate up to capacity.
    Each request consumes tokens.
    Allows bursting up to capacity.
    """
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate  # tokens per second
    
    def is_allowed(self, key: str) -> bool:
        now = time.time()
        bucket = redis.hgetall(f"bucket:{key}")
        
        tokens = float(bucket.get('tokens', self.capacity))
        last_refill = float(bucket.get('last_refill', now))
        
        # Add tokens since last refill
        elapsed = now - last_refill
        tokens = min(self.capacity, tokens + elapsed * self.refill_rate)
        
        if tokens >= 1:
            redis.hset(f"bucket:{key}", {'tokens': tokens - 1, 'last_refill': now})
            redis.expire(f"bucket:{key}", 3600)
            return True
        return False
```

#### Fixed Window

```python
# Allow 100 requests per minute
def is_allowed_fixed_window(key: str) -> bool:
    window = int(time.time() / 60)  # Current minute bucket
    redis_key = f"ratelimit:{key}:{window}"
    
    current = redis.incr(redis_key)
    redis.expire(redis_key, 60)
    
    return current <= 100
    # Problem: boundary burst — 100 at 0:59, 100 at 1:00 = 200 in 2 seconds
```

#### Sliding Window Log

```python
# Accurate but memory-intensive for high-traffic
def is_allowed_sliding_window(key: str) -> bool:
    now = time.time()
    window_start = now - 60  # 60-second window
    redis_key = f"ratelimit:{key}"
    
    pipe = redis.pipeline()
    pipe.zremrangebyscore(redis_key, 0, window_start)  # Remove old requests
    pipe.zadd(redis_key, {str(uuid.uuid4()): now})      # Add current request
    pipe.zcard(redis_key)                               # Count requests in window
    pipe.expire(redis_key, 60)
    _, _, count, _ = pipe.execute()
    
    return count <= 100
```

#### Leaky Bucket

Queue-based. Requests processed at constant rate regardless of input burst. Good for smoothing spiky traffic.

### Rate Limiting Levels

| Level | Key | Use Case |
|---|---|---|
| Per IP | `ratelimit:ip:1.2.3.4` | Protect unauthenticated endpoints |
| Per user | `ratelimit:user:42` | Fair per-user quotas |
| Per API key | `ratelimit:key:abc123` | Tiered pricing plans |
| Per endpoint | `ratelimit:ip:1.2.3.4:/login` | Extra protection for auth endpoints |
| Global | `ratelimit:global` | Protect backend capacity |

### Rate Limit Response Headers

```http
HTTP/1.1 429 Too Many Requests
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1700003600
Retry-After: 60
Content-Type: application/json

{
  "error": "rate_limit_exceeded",
  "message": "Too many requests. Limit: 1000/hour. Reset at: 2024-01-15T11:00:00Z"
}
```

---

## 12. API Versioning and Deprecation

### Semantic Versioning for APIs

```
MAJOR.MINOR.PATCH
  MAJOR: Breaking change (remove field, change type, rename endpoint)
  MINOR: New feature, backward compatible (add field, new endpoint)
  PATCH: Bug fix, no interface change
```

### Deprecation Strategy

```http
# Deprecation headers (RFC 8594)
Deprecation: Sat, 01 Jan 2025 00:00:00 GMT
Sunset: Sat, 01 Jul 2025 00:00:00 GMT
Link: <https://docs.example.com/migration/v1-to-v2>; rel="deprecation"
```

```json
// Include deprecation notice in response body
{
  "data": {...},
  "_warnings": [
    {
      "code": "ENDPOINT_DEPRECATED",
      "message": "This endpoint will be removed on 2025-07-01. Use /v2/users instead.",
      "migration_guide": "https://docs.example.com/migration"
    }
  ]
}
```

### Deprecation Timeline

```
1. Announce deprecation (6-12 months before removal)
   - Email API consumers
   - Add deprecation headers
   - Update documentation
   
2. Sunset period (3-6 months)
   - Both old and new versions available
   - Monitor usage of deprecated endpoints
   - Provide migration support

3. Removal date
   - Return 410 Gone (not 404)
   - Body explains what happened and where to migrate
   
4. Sunset monitoring
   - Track which clients are still using deprecated endpoints
   - Proactively reach out to heavy users
```

---

## 13. API Security at Gateway

### OAuth 2.0 at Gateway

```yaml
# Kong JWT plugin (validate JWT from Authorization header)
plugins:
  - name: jwt
    config:
      key_claim_name: kid
      secret_is_base64: false
      claims_to_verify: ["exp", "nbf"]
      
# Kong OIDC plugin (validate with provider's JWKS endpoint)
plugins:
  - name: oidc
    config:
      client_id: "api-gateway"
      discovery: "https://auth.example.com/.well-known/openid-configuration"
      scope: "openid"
      bearer_only: "yes"
```

### API Key Management

```python
# API key format: prefix + random bytes
import secrets, hashlib

def generate_api_key():
    raw = secrets.token_urlsafe(32)
    key = f"sk_live_{raw}"
    
    # Store SHA-256 hash (not plaintext)
    key_hash = hashlib.sha256(key.encode()).hexdigest()
    
    db.insert_api_key({
        'hash': key_hash,
        'user_id': current_user.id,
        'scopes': ['read:orders'],
        'created_at': datetime.now()
    })
    
    return key  # Show once, never again

def validate_api_key(provided_key: str):
    key_hash = hashlib.sha256(provided_key.encode()).hexdigest()
    api_key = db.find_api_key_by_hash(key_hash)
    
    if not api_key or api_key.revoked:
        raise Unauthorized()
    
    return api_key
```

### AWS SigV4 Request Signing

```python
import hmac, hashlib, datetime

def sign_request(method, host, path, payload, access_key, secret_key, region, service):
    t = datetime.datetime.utcnow()
    amz_date = t.strftime('%Y%m%dT%H%M%SZ')
    date_stamp = t.strftime('%Y%m%d')
    
    # 1. Create canonical request
    payload_hash = hashlib.sha256(payload.encode()).hexdigest()
    canonical_headers = f"host:{host}\nx-amz-date:{amz_date}\n"
    signed_headers = "host;x-amz-date"
    canonical_request = f"{method}\n{path}\n\n{canonical_headers}\n{signed_headers}\n{payload_hash}"
    
    # 2. Create string to sign
    credential_scope = f"{date_stamp}/{region}/{service}/aws4_request"
    string_to_sign = f"AWS4-HMAC-SHA256\n{amz_date}\n{credential_scope}\n{hashlib.sha256(canonical_request.encode()).hexdigest()}"
    
    # 3. Calculate signing key
    def sign(key, msg):
        return hmac.new(key, msg.encode(), hashlib.sha256).digest()
    
    signing_key = sign(
        sign(sign(sign(f"AWS4{secret_key}".encode(), date_stamp), region), service),
        "aws4_request"
    )
    
    # 4. Create signature
    signature = hmac.new(signing_key, string_to_sign.encode(), hashlib.sha256).hexdigest()
    
    return f"AWS4-HMAC-SHA256 Credential={access_key}/{credential_scope}, SignedHeaders={signed_headers}, Signature={signature}"
```

---

## 14. Webhook Design

### Basic Webhook

```python
# Webhook delivery
import httpx, hmac, hashlib, json, time

async def deliver_webhook(subscription, event):
    payload = json.dumps({
        "id": event.id,
        "type": event.type,
        "created": int(time.time()),
        "data": event.data
    })
    
    # HMAC signature
    signature = hmac.new(
        subscription.secret.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()
    
    headers = {
        "Content-Type": "application/json",
        "X-Webhook-Signature": f"sha256={signature}",
        "X-Webhook-Id": event.id,
        "X-Webhook-Timestamp": str(int(time.time()))
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            subscription.url,
            content=payload,
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
```

### Signature Verification (Receiver Side)

```python
@app.post("/webhook")
async def receive_webhook(request: Request):
    payload = await request.body()
    signature = request.headers.get("X-Webhook-Signature")
    timestamp = request.headers.get("X-Webhook-Timestamp")
    
    # Verify timestamp to prevent replay attacks (reject > 5 min old)
    if abs(time.time() - int(timestamp)) > 300:
        raise HTTPException(400, "Webhook timestamp too old")
    
    # Verify signature
    expected = hmac.new(WEBHOOK_SECRET.encode(), payload, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(f"sha256={expected}", signature):
        raise HTTPException(401, "Invalid signature")
    
    # Process asynchronously (return 200 immediately)
    event = json.loads(payload)
    background_tasks.add_task(process_webhook_event, event)
    
    return {"status": "received"}
```

### Retry Logic

```python
# Exponential backoff with jitter
async def deliver_with_retry(subscription, event, max_attempts=5):
    for attempt in range(max_attempts):
        try:
            await deliver_webhook(subscription, event)
            return  # Success
        except Exception as e:
            if attempt == max_attempts - 1:
                # Dead letter queue
                await dead_letter_queue.push(subscription, event, str(e))
                return
            
            # Exponential backoff: 1s, 2s, 4s, 8s, 16s + jitter
            delay = (2 ** attempt) + random.uniform(0, 1)
            await asyncio.sleep(delay)
```

**Delivery guarantees**:
- At-least-once delivery (most practical): retry until success, consumer must be idempotent
- At-most-once: fire and forget (can lose events)
- Exactly-once: extremely hard, use idempotency keys

### Fanout

```
Single event → multiple subscribers

Event Bus (Kafka/SNS) → multiple webhook workers → parallel delivery to:
  Subscriber A endpoint
  Subscriber B endpoint
  Subscriber C endpoint
```

---

## 15. WebSocket at Scale

### Connection Architecture

```
Client A ──────────────────► WS Server 1 ──────────────────┐
Client B ──────────────────► WS Server 1 ──────────────────┤
Client C ──────────────────► WS Server 2 ──────────────────┤──► Redis Pub/Sub
Client D ──────────────────► WS Server 2 ──────────────────┘
                                                             │
                                               All servers subscribe to Redis
                                               Message sent to any server →
                                               Redis broadcasts to all servers →
                                               Each server forwards to connected clients
```

### WebSocket with Redis Pub/Sub

```python
# FastAPI + Redis pub/sub for WebSocket scaling
import asyncio
import redis.asyncio as aioredis
from fastapi import WebSocket
from typing import Dict, Set

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.redis = aioredis.from_url("redis://redis:6379")
    
    async def connect(self, websocket: WebSocket, channel: str):
        await websocket.accept()
        if channel not in self.active_connections:
            self.active_connections[channel] = set()
        self.active_connections[channel].add(websocket)
    
    async def broadcast_to_channel(self, channel: str, message: str):
        # Publish to Redis — other servers will also receive this
        await self.redis.publish(f"ws:{channel}", message)
    
    async def listen_redis(self):
        pubsub = self.redis.pubsub()
        await pubsub.psubscribe("ws:*")
        
        async for message in pubsub.listen():
            if message["type"] == "pmessage":
                channel = message["channel"].decode().removeprefix("ws:")
                data = message["data"].decode()
                
                # Forward to all local clients in this channel
                for ws in list(self.active_connections.get(channel, set())):
                    try:
                        await ws.send_text(data)
                    except Exception:
                        self.active_connections[channel].discard(ws)
```

### Horizontal Scaling Considerations

```
Challenge: WebSocket connections are sticky (long-lived, stateful)
  - Load balancer must use sticky sessions OR
  - All servers must share state via pub/sub

Sticky sessions (session affinity):
  - Client reconnects to same server
  - Simple but uneven load distribution
  - Server failure loses all its connections

Redis pub/sub (recommended):
  - Any server can receive messages for any client
  - Even load distribution
  - Server failure: clients reconnect to another server

Alternative: Use managed WebSocket service
  - Ably, Pusher, AWS API Gateway WebSocket API
  - Handles scaling automatically
  - Pay per connection per message
```

---

## 16. Long Polling vs SSE vs WebSocket

### Long Polling

```
Client                                    Server
  |--- GET /events?last_id=100 --------> |
  |                                       |
  |         (server holds connection)     |
  |         (server holds connection)     |
  |                                       |
  |<-- 200 [{event: "order_updated"}] ---|  (when event available)
  |                                       |
  |--- GET /events?last_id=101 --------> |  (immediately reconnect)
  |         (server holds again)          |
```

```javascript
async function longPoll(lastId) {
  while (true) {
    try {
      const response = await fetch(`/events?last_id=${lastId}&timeout=30`);
      const events = await response.json();
      
      for (const event of events) {
        handleEvent(event);
        lastId = event.id;
      }
    } catch (e) {
      await sleep(1000);  // Backoff on error
    }
  }
}
```

### Server-Sent Events (SSE)

```
Client                    Server
  |--- GET /events -----> |
  |<-- HTTP 200 ----------|
  |    Content-Type: text/event-stream
  |    Cache-Control: no-cache
  |    Connection: keep-alive
  |                        |
  |<-- data: {...}\n\n ----|  (server pushes events)
  |<-- data: {...}\n\n ----|
  |<-- :heartbeat\n\n -----|  (keep-alive comment)
```

```python
# FastAPI SSE
from fastapi.responses import StreamingResponse

async def event_stream(user_id: str):
    while True:
        events = await get_pending_events(user_id)
        for event in events:
            yield f"id: {event.id}\n"
            yield f"event: {event.type}\n"
            yield f"data: {json.dumps(event.data)}\n\n"
        
        # Heartbeat
        yield ": heartbeat\n\n"
        await asyncio.sleep(15)

@app.get("/events")
async def stream_events(user_id: str = Depends(get_current_user)):
    return StreamingResponse(
        event_stream(user_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )
```

### Decision Matrix

| Feature | Long Polling | SSE | WebSocket |
|---|---|---|---|
| Direction | Client-initiated only | Server → Client | Bidirectional |
| Protocol | HTTP | HTTP | WS (upgrade) |
| Browser support | Universal | Modern browsers (IE: no) | Modern browsers |
| Proxy/CDN support | Universal | Good | Varies |
| Reconnection | Manual | Built-in | Manual |
| Overhead | High (new req each time) | Low | Very low |
| Use case | Legacy compatibility | Notifications, feeds | Chat, gaming, collab |
| Server load | High | Medium | Low |

**When to choose**:
- **Long polling**: Broadest compatibility, legacy systems, event frequency < 1/minute
- **SSE**: Push notifications, live feeds, dashboards (unidirectional only)
- **WebSocket**: Chat, real-time collaboration, gaming, trading systems (bidirectional)

---

## 17. OpenAPI / Swagger

### OpenAPI 3.0 Specification

```yaml
openapi: 3.0.3
info:
  title: Orders API
  description: API for managing orders
  version: 1.0.0
  contact:
    email: api@example.com

servers:
  - url: https://api.example.com/v1
  - url: https://staging-api.example.com/v1

paths:
  /orders:
    get:
      summary: List orders
      operationId: listOrders
      tags: [Orders]
      security:
        - bearerAuth: []
      parameters:
        - name: status
          in: query
          schema:
            type: string
            enum: [pending, processing, shipped, delivered]
        - name: limit
          in: query
          schema:
            type: integer
            minimum: 1
            maximum: 100
            default: 20
        - name: cursor
          in: query
          schema:
            type: string
      responses:
        '200':
          description: List of orders
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/OrdersResponse'
        '401':
          $ref: '#/components/responses/Unauthorized'
        '429':
          $ref: '#/components/responses/RateLimited'

    post:
      summary: Create order
      operationId: createOrder
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreateOrderRequest'
      responses:
        '201':
          description: Order created
          headers:
            Location:
              schema:
                type: string
              description: URL of created order

components:
  schemas:
    Order:
      type: object
      required: [id, status, total]
      properties:
        id:
          type: string
          example: "ord_abc123"
        status:
          type: string
          enum: [pending, processing, shipped, delivered, cancelled]
        total:
          type: number
          format: float
          example: 149.99
        items:
          type: array
          items:
            $ref: '#/components/schemas/OrderItem'

  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT

  responses:
    Unauthorized:
      description: Authentication required
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
```

### Contract-First Development

```
1. Write OpenAPI spec first (business + frontend + backend agree)
2. Generate: server stubs, client SDKs, mock servers
3. Frontend uses mock server while backend builds
4. Validate implementation against spec in CI (openapi-validator)
5. Generate docs automatically (Swagger UI, Redoc)
```

---

## 18. API Design for Mobile

### Payload Optimization

```json
// Fat response (web app — can handle)
{
  "user": {
    "id": 42,
    "email": "alice@example.com",
    "full_name": "Alice Smith",
    "avatar_url_original": "https://cdn.example.com/avatars/42_original.jpg",
    "avatar_url_thumb": "https://cdn.example.com/avatars/42_thumb.jpg",
    "created_at": "2024-01-01T00:00:00Z",
    "preferences": { ... 50 fields ... },
    "address": { ... 10 fields ... },
    "billing": { ... 15 fields ... }
  }
}

// Optimized response (mobile app — only what's needed)
{
  "user": {
    "id": 42,
    "name": "Alice",
    "avatar": "https://cdn.example.com/avatars/42_thumb.jpg"
  }
}
```

### Offline Sync

```
Strategy: Optimistic UI + sync queue

Mobile App Architecture:
  ┌─────────────────────────────────┐
  │           Local SQLite          │
  │  (truth for reads while offline)│
  └──────────────┬──────────────────┘
                 │ sync
  ┌──────────────▼──────────────────┐
  │         Sync Queue              │
  │  (pending mutations)            │
  └──────────────┬──────────────────┘
                 │ when online
  ┌──────────────▼──────────────────┐
  │          Backend API            │
  └─────────────────────────────────┘
```

### Delta Updates

```
Full sync (expensive):
  GET /mobile/data → entire dataset

Delta sync (efficient):
  GET /mobile/data?since=1700000000 → only changes since timestamp

Response:
{
  "updated": [...items changed since timestamp...],
  "deleted": ["id1", "id2"],
  "server_timestamp": 1700003600
}
```

---

## 19. GraphQL Federation

### Problem: GraphQL Monolith at Scale

Single GraphQL schema maintained by all teams → merge conflicts, slow iteration, single point of failure.

### Apollo Federation Solution

```
                    ┌────────────────────────────────┐
Client              │         Apollo Gateway          │
  ──GraphQL Query──►│  (Supergraph — unified schema) │
                    └──────────────┬─────────────────┘
                                   │ query planning
                          ┌────────┴──────────┐
                          │                   │
                    ┌─────▼──────┐     ┌──────▼──────┐
                    │  Users     │     │  Orders      │
                    │  Subgraph  │     │  Subgraph    │
                    │  (team A)  │     │  (team B)    │
                    └────────────┘     └─────────────┘
```

### Subgraph Schema

```graphql
# Users subgraph
type User @key(fields: "id") {
  id: ID!
  name: String!
  email: String!
}

# Orders subgraph — extends User entity
extend type User @key(fields: "id") {
  id: ID! @external
  orders: [Order!]!
}

type Order {
  id: ID!
  status: String!
  user: User!
}
```

### Query Planning

```graphql
# Client query
query {
  user(id: "42") {
    name              # → fetched from Users subgraph
    orders {          # → fetched from Orders subgraph
      id
      status
    }
  }
}

# Gateway plan:
# 1. Send to Users subgraph: user(id: "42") { name }
# 2. Send to Orders subgraph: _entities(representations: [{__typename: "User", id: "42"}]) { ... on User { orders { id status } } }
# 3. Merge results
```

### Schema Stitching vs Apollo Federation

| Aspect | Schema Stitching | Apollo Federation |
|---|---|---|
| Type merging | Manual merge at gateway | Declarative with @key |
| Team ownership | Gateway team owns merge | Subgraph teams own types |
| Maturity | Older, more manual | Newer, more automated |
| Complexity | High | Medium |
| Error handling | Gateway-level | Better distributed |

---

## 20. Quick Reference

### HTTP Status Codes Cheat Sheet

```
200 OK              — Success
201 Created         — Resource created (POST)
204 No Content      — Success, no body (DELETE)
301 Moved           — Permanent redirect (update bookmarks)
304 Not Modified    — Cached version is valid
400 Bad Request     — Client sent invalid data
401 Unauthorized    — Authentication required
403 Forbidden       — No permission (even if authenticated)
404 Not Found       — Resource doesn't exist
405 Method Not Allowed — Wrong HTTP method for endpoint
409 Conflict        — Duplicate resource or version conflict
410 Gone            — Resource permanently deleted
422 Unprocessable   — Validation error
429 Too Many Requests — Rate limited
500 Internal Error  — Generic server error
502 Bad Gateway     — Upstream service failed
503 Unavailable     — Server down / overloaded
504 Gateway Timeout — Upstream service too slow
```

### API Protocol Comparison Table

| Feature | REST | GraphQL | gRPC |
|---|---|---|---|
| Protocol | HTTP/1.1+ | HTTP/1.1+ | HTTP/2 |
| Payload | JSON | JSON | Protobuf (binary) |
| Schema | Optional (OpenAPI) | Required | Required (proto) |
| Streaming | SSE / WebSocket | Subscriptions | 4 native modes |
| Browser native | Yes | Yes | No (grpc-web needed) |
| Performance | Medium | Medium | High |
| Cacheability | Excellent | Poor (POST) | N/A |
| Mobile friendly | Good | Good (no over-fetch) | Best (binary) |
| Code generation | Optional | Optional | Required |
| Best for | Public APIs | Complex clients | Internal microservices |

### Real-Time Protocol Selection

```
Need bidirectional?
  └── Yes → WebSocket
  
  └── No (server push only):
        High frequency (>1 msg/sec)?
          └── Yes → SSE
          
          └── No:
                Browser/proxy compatibility required?
                  └── Yes → Long polling
                  └── No  → SSE
```

### Pagination Selection Guide

```
Random page access needed?
  └── Yes → Offset pagination (accept performance trade-off)

Real-time or frequently updating data?
  └── Yes → Cursor-based pagination

Large dataset, performance critical?
  └── Yes → Keyset / cursor-based pagination

Simple admin panel with small dataset?
  └── Offset pagination is fine
```
