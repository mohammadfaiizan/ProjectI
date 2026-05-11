# Design Amazon — E-Commerce Platform

---

## 1. Problem Statement & Clarifying Questions

Design a large-scale e-commerce platform like Amazon that supports product browsing, search, shopping cart management, order placement, payment processing, and order fulfillment.

### Clarifying Questions

| Question | Assumption |
|---|---|
| Target user scale? | 500M registered users, 50M DAU |
| Product catalog size? | 300M products |
| Peak traffic events? | Black Friday: 5x normal traffic |
| Do we need a seller marketplace? | Yes — multi-seller per product |
| Do we need real-time inventory? | Yes — prevent oversell |
| Delivery tracking? | Yes — real-time order status updates |
| Product reviews? | Yes — verified purchase reviews with helpful votes |
| Recommendation system? | Yes — collaborative filtering |
| Payment processing? | Integrate with PSP (Stripe/Braintree), handle idempotency |
| International support? | Assume single region for scope |

---

## 2. Functional & Non-Functional Requirements

### Functional Requirements
1. **Product Catalog** — Browse, search, and filter products
2. **Search** — Full-text search with facets (category, price, rating, brand)
3. **Shopping Cart** — Add/remove items, persist across sessions, TTL for guest carts
4. **Inventory Management** — Real-time stock tracking, prevent oversell
5. **Order Management** — Place orders, state machine tracking
6. **Payment Processing** — Secure payment, idempotent retry, refunds
7. **Recommendations** — Personalized product recommendations
8. **Reviews** — Verified reviews, rating aggregation, helpful votes
9. **Seller Marketplace** — Multiple sellers per product, seller ratings
10. **Fulfillment Routing** — Route to nearest warehouse with available stock

### Non-Functional Requirements
1. **Availability** — 99.99% uptime; checkout must never go down
2. **Consistency** — Strong consistency for inventory and payments
3. **Latency** — Search < 100ms P95; product page < 200ms; checkout < 1s
4. **Scalability** — Handle 5x Black Friday spikes (50M DAU → 250M DAU equivalent)
5. **Durability** — No order or payment data loss

---

## 3. Capacity Estimation

### Traffic
- Product page views: 50M DAU × 10 pages = 500M/day = ~5,800/sec
- Search queries: 50M DAU × 3 searches = 150M/day = ~1,740/sec
- Orders: 50M DAU × 2% conversion = 1M orders/day = ~12 orders/sec
- Black Friday peak (5x): 29K page views/sec, 8.7K searches/sec, 60 orders/sec

### Storage
- Products: 300M × 10 KB = 3 TB (text metadata)
- Product images: 300M × 3 images × 500 KB = 450 TB (CDN-served)
- Orders: 1M/day × 2 KB = 2 GB/day → 730 GB/year
- Reviews: 1M/day × 500 B = 500 MB/day

### Cache
- Top 10K products (80% of views): 10K × 10 KB = 100 MB → fits in Redis entirely
- Product page cache hit rate target: 90%

### Search Index
- 300M products × 1 KB text = 300 GB
- Elasticsearch index (3x overhead) = ~900 GB
- 5 Elasticsearch nodes × 200 GB each = 1 TB capacity

---

## 4. High-Level Architecture

```
         ┌──────────────────────────────────────────────┐
         │              Client (Browser / App)           │
         └────────────────────┬─────────────────────────┘
                              │ HTTPS
         ┌────────────────────▼─────────────────────────┐
         │           API Gateway + CDN (CloudFront)      │
         │   (Static assets: S3+CDN; Dynamic: API GW)   │
         └────┬──────┬────────┬────────┬────────┬───────┘
              │      │        │        │        │
      ┌───────▼──┐ ┌─▼──────┐ ┌──────▼─┐ ┌────▼───┐ ┌───▼──────┐
      │ Product  │ │ Search │ │  Cart  │ │ Order  │ │ User /   │
      │ Service  │ │Service │ │Service │ │Service │ │ Auth Svc │
      └────┬─────┘ └───┬────┘ └───┬────┘ └───┬────┘ └──────────┘
           │           │          │          │
    ┌──────▼──┐  ┌──────▼──┐ ┌────▼──┐ ┌────▼──────────────────┐
    │ Postgres │  │Elastic  │ │Redis  │ │  Order DB (Postgres)  │
    │(products,│  │ Search  │ │(carts,│ │  + Kafka (events)     │
    │inventory │  │         │ │ cache)│ └──────┬────────────────┘
    │ reviews) │  └─────────┘ └───────┘        │
    └──────────┘                        ┌───────▼───────────────┐
                                        │   Payment Service     │
                                        │   (PSP Integration)   │
                                        └───────┬───────────────┘
                                                │
                                        ┌───────▼───────────────┐
                                        │  Fulfillment Service  │
                                        │  (Warehouse routing)  │
                                        └───────────────────────┘

         Supporting Services:
         ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
         │Recommendation│  │  Notification│  │  Analytics   │
         │   Engine     │  │   Service    │  │  (Kafka)     │
         └──────────────┘  └──────────────┘  └──────────────┘
```

---

## 5. Component Deep-Dive

### 5.1 Inventory Management — Preventing Oversell

This is one of the most critical challenges in e-commerce. Two users simultaneously adding the last item must not both succeed.

**Approach 1: Pessimistic Locking (SELECT FOR UPDATE)**
```sql
BEGIN;
SELECT stock FROM inventory WHERE product_id = ? AND warehouse_id = ?
FOR UPDATE;           -- Blocks other transactions until commit
-- If stock > 0: decrement
UPDATE inventory SET stock = stock - 1 WHERE product_id = ?;
COMMIT;
```
- Guarantees no oversell
- High contention for popular items → serialization bottleneck

**Approach 2: Optimistic Locking (Version-Based)**
```sql
-- Read current version
SELECT stock, version FROM inventory WHERE product_id = ?;
-- Attempt update only if version unchanged
UPDATE inventory
SET stock = stock - 1, version = version + 1
WHERE product_id = ? AND version = {read_version} AND stock > 0;
-- If 0 rows updated → conflict → retry
```
- Better throughput under low contention
- Retry logic required; high contention can cause retry storms

**Approach 3: Redis Atomic Decrement**
```lua
-- Lua script (atomic in Redis)
local stock = redis.call('GET', KEYS[1])
if tonumber(stock) > 0 then
    redis.call('DECR', KEYS[1])
    return 1   -- success
else
    return 0   -- out of stock
end
```
- Extremely fast for hot items
- Redis as inventory source of truth; periodic sync to PostgreSQL

**Choice:** Redis atomic operations for inventory reservation during checkout, backed by PostgreSQL for durability.

### 5.2 Order State Machine
```
PLACED → PAYMENT_PENDING → PAID → FULFILLMENT_ASSIGNED
  → PICKED → PACKED → SHIPPED → OUT_FOR_DELIVERY
  → DELIVERED
  (at any point) → CANCELLED
  (after delivery) → RETURN_REQUESTED → RETURNED
```

State transitions stored as events in Kafka; current state materialized in PostgreSQL.

### 5.3 Shopping Cart — Redis with TTL
```
HSET cart:{user_id} {product_id} quantity
EXPIRE cart:{user_id} 604800   -- 7-day TTL for guest carts
```

For authenticated users: persist cart to PostgreSQL on session end for cross-device access.

### 5.4 Search Architecture

Elasticsearch with faceted filtering:
- **Text fields:** title, description, brand (full-text analyzed)
- **Keyword fields:** category, brand_id, seller_id (exact match)
- **Numeric fields:** price, rating, review_count (range + sort)

**Query example:**
```json
{
  "query": { "match": { "title": "wireless headphones" } },
  "filter": [
    { "term": { "category": "Electronics" } },
    { "range": { "price": { "gte": 50, "lte": 200 } } },
    { "range": { "rating": { "gte": 4.0 } } }
  ],
  "sort": [{ "rating": "desc" }, { "_score": "desc" }]
}
```

### 5.5 Recommendation Engine

**Collaborative Filtering (User-Item Matrix):**
- Build user-item interaction matrix (views, purchases, ratings)
- Find similar users using cosine similarity
- Recommend items liked by similar users but not yet seen by target user

**"Customers Also Bought" (Item-Item CF):**
- Co-purchase matrix: products frequently bought together
- Displayed on product detail pages
- Updated daily from order history

**Real-Time Signals:** Combine offline batch recommendations with real-time "trending in your category" signals.

### 5.6 Fulfillment Routing

When an order is paid:
1. Query warehouses with sufficient stock for all items
2. Score each warehouse by: distance to delivery address + stock availability + warehouse load
3. Assign order to optimal warehouse (or split across warehouses if needed)
4. Reserve inventory at chosen warehouse

### 5.7 Dynamic Pricing Engine
- Base price from seller
- Demand multiplier (high demand → price increase within legal limits)
- Coupon/discount application
- Prime member discounts
- Flash sale overrides

---

## 6. Database Design

### 6.1 Products Table (PostgreSQL)
```sql
CREATE TABLE products (
    product_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title         VARCHAR(500) NOT NULL,
    description   TEXT,
    brand         VARCHAR(200),
    category_id   INT REFERENCES categories(category_id),
    seller_id     UUID REFERENCES sellers(seller_id),
    base_price    DECIMAL(10,2),
    rating_avg    DECIMAL(3,2) DEFAULT 0,
    review_count  INT DEFAULT 0,
    is_active     BOOLEAN DEFAULT TRUE,
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    INDEX idx_products_category (category_id),
    INDEX idx_products_brand (brand),
    INDEX idx_products_seller (seller_id)
);
```

### 6.2 Inventory Table
```sql
CREATE TABLE inventory (
    inventory_id   UUID PRIMARY KEY,
    product_id     UUID REFERENCES products(product_id),
    warehouse_id   UUID REFERENCES warehouses(warehouse_id),
    quantity       INT NOT NULL CHECK (quantity >= 0),
    reserved       INT DEFAULT 0,    -- Held for pending orders
    version        BIGINT DEFAULT 0, -- For optimistic locking
    updated_at     TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (product_id, warehouse_id)
);
```

### 6.3 Orders Table
```sql
CREATE TABLE orders (
    order_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id        UUID NOT NULL,
    status         VARCHAR(30) NOT NULL DEFAULT 'placed',
    subtotal       DECIMAL(10,2),
    shipping_cost  DECIMAL(10,2),
    tax            DECIMAL(10,2),
    total          DECIMAL(10,2),
    shipping_addr  JSONB,
    payment_id     UUID,
    warehouse_id   UUID,
    placed_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at     TIMESTAMPTZ DEFAULT NOW(),
    INDEX idx_orders_user (user_id),
    INDEX idx_orders_status (status)
);
```

### 6.4 Order Items Table
```sql
CREATE TABLE order_items (
    item_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    order_id       UUID REFERENCES orders(order_id),
    product_id     UUID REFERENCES products(product_id),
    seller_id      UUID,
    quantity       INT NOT NULL,
    unit_price     DECIMAL(10,2),
    discount       DECIMAL(10,2) DEFAULT 0
);
```

### 6.5 Reviews Table
```sql
CREATE TABLE reviews (
    review_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    product_id     UUID REFERENCES products(product_id),
    user_id        UUID NOT NULL,
    order_id       UUID,            -- Verified purchase link
    rating         INT CHECK (rating BETWEEN 1 AND 5),
    title          VARCHAR(200),
    body           TEXT,
    helpful_votes  INT DEFAULT 0,
    is_verified    BOOLEAN DEFAULT FALSE,
    created_at     TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (product_id, user_id)
);
```

---

## 7. API Design

### Search Products
```
GET /api/v1/products/search?q={query}&category={cat}&min_price={p}&max_price={p}&rating={r}&sort={field}&page={n}
Response: { products: [{product_id, title, price, rating, image_url}], total, facets }
```

### Add to Cart
```
POST /api/v1/cart/items
Body: { product_id, quantity }
Response: { cart_id, items: [...], subtotal }
```

### Place Order
```
POST /api/v1/orders
Body: { cart_id, shipping_address_id, payment_method_id, coupon_code }
Response: { order_id, status, estimated_delivery, total }
```

### Track Order
```
GET /api/v1/orders/{order_id}/tracking
Response: { order_id, status, events: [{status, timestamp, location}] }
```

### Submit Review
```
POST /api/v1/products/{product_id}/reviews
Body: { order_id, rating, title, body }
Response: { review_id, is_verified }
```

---

## 8. Scalability & Bottlenecks

### Bottleneck 1: Hot Product Inventory (Flash Sales)
- **Problem:** iPhone launch → 100K users try to buy simultaneously, all hitting inventory row
- **Solution:** Redis atomic DECR as pre-check. PostgreSQL stock is floor. Redis bucket approach: allocate 1000 units to Redis; when exhausted, refill from PostgreSQL.

### Bottleneck 2: Product Page Read Throughput
- **Problem:** 5,800 product page views/sec; 5x on Black Friday = 29K/sec
- **Solution:** CloudFront caches product pages (TTL 60s). CDN cache hit rate 90% → origin sees 2,900/sec. Product service handles with read replicas. Cache invalidation on price/stock change.

### Bottleneck 3: Search Performance Under Load
- **Problem:** 8,700 search queries/sec on Black Friday
- **Solution:** Elasticsearch cluster with 10 nodes, 3 shards/index, 1 replica. Query cache for popular queries. Pre-warm cache with popular searches before Black Friday.

### Bottleneck 4: Order Processing Bottleneck
- **Problem:** Payment + inventory reservation + order creation must be atomic
- **Solution:** Saga pattern. Each step is independent service with compensating transaction on failure.
  - Step 1: Reserve inventory (Redis atomic DECR)
  - Step 2: Process payment (PSP integration)
  - Step 3: Create order record
  - Compensating: if payment fails → release inventory reservation

### Bottleneck 5: Recommendation Computation
- **Problem:** Collaborative filtering over 500M users × 300M products is computationally expensive
- **Solution:** Offline batch job (Apache Spark) runs nightly. Pre-computed recommendations stored in DynamoDB (user_id → [product_ids]). Real-time fallback: popular items in viewed category.

---

## 9. Trade-offs & Design Decisions

### Decision 1: PostgreSQL vs. DynamoDB for Product Catalog
- **PostgreSQL:** Rich queries (join products + inventory + reviews), ACID, familiar
- **DynamoDB:** NoSQL, horizontal scale, simple key-value lookups fast
- **Choice:** PostgreSQL with read replicas for product catalog. Complex queries justify relational. Shard by category_id for scale.

### Decision 2: Optimistic vs. Pessimistic Locking for Inventory
- **Pessimistic:** Guarantees correctness, but SELECT FOR UPDATE serializes all writes to a row → bottleneck for popular items
- **Optimistic:** Higher throughput, requires retry logic. Under high contention (Black Friday) → retry storm risk
- **Choice:** Redis atomic decrement for fast path + PostgreSQL for durability. Redis is single-threaded → atomic by nature, no locking needed.

### Decision 3: Cart Storage — Redis vs. Database
- **Redis:** Sub-millisecond read/write, TTL support, simple data structure
- **Database:** Persistent, queryable, no TTL management needed
- **Choice:** Redis as primary cart store (fast, TTL for guest cleanup). Sync to PostgreSQL for authenticated users on checkout or session end.

### Decision 4: Synchronous vs. Asynchronous Inventory Update Post-Order
- **Synchronous:** Update inventory in the same transaction as order creation. Slow but consistent.
- **Asynchronous:** Order created first, inventory updated via Kafka consumer. Risk: oversell in edge cases.
- **Choice:** Synchronous reservation (Redis) at checkout time. Async confirmation write to PostgreSQL. This separates fast reservation from slow persistence.

### Decision 5: Flat vs. Hierarchical Product Categories
- **Flat:** Simple. Poor for navigation and faceted search.
- **Hierarchical:** Electronics → Headphones → Wireless. Better browse and drill-down.
- **Choice:** Nested set model in PostgreSQL for category tree. Elasticsearch flat category_path for fast faceting.

---

## 10. Key Interview Talking Points

1. **Inventory Oversell Prevention is the Core Challenge:** Two users, one item. The solution isn't just locking — it's choosing the right lock scope. Redis atomic DECR gives you serialized inventory check without database transaction overhead.

2. **Saga Pattern for Distributed Checkout:** Payment + inventory + order creation span multiple services. Use Saga: each step publishes an event; compensating transactions roll back on failure. No 2-phase commit across services.

3. **The Optimistic Lock Version Field:** Add `version BIGINT` to inventory. Read version, decrement, update WHERE version = {read_version}. If 0 rows affected, someone else changed it — retry. No explicit locks, great throughput for moderate contention.

4. **Elasticsearch Facets Enable Drill-Down Search:** Faceted search returns not just results but counts per filter value (e.g., "Brand: Nike (342), Adidas (218)"). Implemented via Elasticsearch aggregations, computed in the same search request as results.

5. **Order State Machine Drives the Business:** Every order state transition is a Kafka event. Downstream consumers (fulfillment, notifications, analytics) react to events. The state machine in PostgreSQL is the materialized view of the event log.

6. **Black Friday Capacity Planning:** 5x traffic. Product pages: CDN absorbs 90% → backend sees 1.5x normal. Search: Elasticsearch pre-warmed + query cache. Checkout: horizontal scale + Redis inventory reservation → no bottleneck. The goal is that the most critical path (checkout) scales independently.

7. **Review Authenticity — Verified Purchase:** Link review to order_id. If order exists and was delivered, review is_verified = true. Prominent display boost for verified reviews. Prevents fake reviews from non-buyers.

8. **Cart TTL for Abandonment:** Guest cart in Redis with 7-day TTL. On checkout, TTL extended. On login, merge guest cart with user's saved cart. 70% of carts are abandoned — TTL prevents Redis bloat.

9. **Recommendation Cold Start:** New user has no history. Fallback: bestsellers in browsed categories, geographically popular items, demographic-based recommendations. Collaborative filtering kicks in after 5+ purchases.

10. **Scale Numbers to Know:** 300M products, 500M users, 1M orders/day. 5,800 product page views/sec (normal), 29K/sec (Black Friday). Redis: 1M cart keys × 200 bytes = 200 MB — trivially fits in memory.
