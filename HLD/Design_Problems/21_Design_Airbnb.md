# System Design: Airbnb

## 1. Problem Statement & Clarifying Questions

### Problem Statement
Design a home-sharing marketplace platform like Airbnb where hosts can list their properties and guests can search, book, and review stays. The system must handle real-time availability, conflict-free bookings, dynamic pricing, and trust between strangers.

### Clarifying Questions
1. **Scale**: How many listings and users do we need to support? (7M listings, 100M users)
2. **Search**: What filters matter most — location, dates, price, amenities?
3. **Booking flow**: Instant book vs. host-approval required?
4. **Payments**: Do we handle payments in-platform or delegate to a third party?
5. **Reviews**: Are reviews mutual (both parties review each other)?
6. **Internationalization**: Multi-currency, multi-language support?
7. **Photos**: How many photos per listing on average? (~20-30)
8. **Messaging**: Real-time or async messaging between hosts and guests?
9. **Cancellation**: What cancellation policies exist (flexible, moderate, strict)?
10. **Availability**: How far in advance can guests book? (up to 12 months)

---

## 2. Functional & Non-Functional Requirements

### Functional Requirements
- Hosts can create, edit, and deactivate listings with photos, descriptions, amenities, pricing rules, and availability calendars
- Guests can search listings by location, date range, guest count, price, and amenities
- Guests can view listing detail pages with photos, calendars, and reviews
- Guests can create bookings; system prevents double-booking via conflict detection
- Payment flow: hold funds at booking, capture on check-in, release to host after stay
- Hosts and guests can message each other within the platform
- Both parties can submit reviews after a stay; mutual reveal (reviews hidden until both submitted or deadline passes)
- Hosts can set base price, seasonal pricing, and weekend premiums
- Dynamic pricing recommendations based on demand signals
- Guests can cancel bookings (refund governed by cancellation policy)
- Photo upload and storage with multiple sizes for web/mobile

### Non-Functional Requirements
- **Availability**: 99.99% uptime for search and booking flows
- **Consistency**: Strong consistency for booking (no double-bookings ever)
- **Latency**: Search results < 200ms p99; booking creation < 500ms p99
- **Scale**: 7M listings, 100M users, ~500K searches/day, ~1M bookings/month
- **Durability**: Zero loss of booking and payment records
- **Security**: PII encryption, PCI-DSS compliance for payment data
- **Geo-distributed**: Support global users across multiple regions

---

## 3. Capacity Estimation

### Traffic
- **DAU**: ~10M guests browsing, ~500K searches/hour at peak
- **Bookings**: ~1M/month → ~400/hour average, ~2000/hour peak
- **Messages**: ~5M messages/day
- **Photo uploads**: ~100K new photos/day (new listings + updates)

### Storage
- **Listings**: 7M × 2KB metadata = ~14GB
- **Availability records**: 7M listings × 365 days × ~20 bytes = ~51GB
- **Bookings**: 1M/month × 12 months × 500 bytes = ~6GB/year
- **Photos**: ~20 photos/listing × 7M listings × 500KB average = ~70TB
- **Messages**: 5M/day × 365 × 200 bytes = ~365GB/year
- **Reviews**: 2M reviews/year × 1KB = ~2GB/year

### Compute
- **Search cluster**: 20 Elasticsearch nodes (3 primary shards, 2 replicas)
- **Application servers**: ~50 horizontally scaled web servers
- **Database**: Primary PostgreSQL + 3 read replicas per region
- **Cache**: Redis cluster for session, availability cache, search results

---

## 4. High-Level Architecture

```
                         ┌──────────────────────────────────────────────────────┐
                         │                    Clients                            │
                         │         Web Browser / iOS App / Android App          │
                         └────────────────────────┬─────────────────────────────┘
                                                  │ HTTPS
                         ┌────────────────────────▼─────────────────────────────┐
                         │                 CDN (CloudFront)                      │
                         │         Static Assets + Photo Delivery                │
                         └────────────────────────┬─────────────────────────────┘
                                                  │
                         ┌────────────────────────▼─────────────────────────────┐
                         │               Load Balancer (L7)                      │
                         └──┬─────────────┬──────────────┬──────────────┬───────┘
                            │             │              │              │
              ┌─────────────▼──┐  ┌───────▼───────┐ ┌───▼──────────┐ ┌▼──────────────┐
              │  Search Service│  │Booking Service│ │User Service  │ │Messaging Svc  │
              │  (REST API)    │  │  (REST API)   │ │  (REST API)  │ │  (WebSocket)  │
              └───────┬────────┘  └───────┬───────┘ └──────┬───────┘ └───────┬───────┘
                      │                  │                  │                 │
        ┌─────────────▼──┐   ┌──────────▼──────┐  ┌───────▼───────┐  ┌──────▼──────┐
        │ Elasticsearch   │   │  PostgreSQL      │  │  PostgreSQL   │  │   Redis     │
        │ (Listings +     │   │  (Bookings +     │  │  (Users +     │  │  (Messages  │
        │  Geo Search)    │   │   Availability)  │  │   Reviews)    │  │   Cache)    │
        └─────────────────┘   └────────┬─────────┘  └───────────────┘  └─────────────┘
                                       │
                              ┌────────▼────────┐
                              │  Payment Service │
                              │  (Stripe/Adyen)  │
                              └────────┬─────────┘
                                       │
                              ┌────────▼────────┐
                              │   S3 + CDN       │
                              │ (Photo Storage)  │
                              └─────────────────┘

  CDC (Change Data Capture): PostgreSQL → Kafka → Elasticsearch Sync Pipeline
```

---

## 5. Component Deep-Dive

### 5.1 Search Service
The search service is the most read-heavy component. It uses Elasticsearch as the primary search backend with geo_point fields for location-based queries.

**Search Flow:**
1. Client sends search request: `{location, check_in, check_out, guests, price_min, price_max, amenities[]}`
2. Geocoding service converts location string to lat/lon coordinates
3. Elasticsearch query: geo_distance filter (bounding box) + availability filter + price range + amenity terms
4. Results ranked by: relevance score × rating × availability certainty
5. Pagination via search_after cursor (not offset-based, avoids deep pagination issues)

**Geo Filtering Strategy:**
- Use `geo_point` field type in Elasticsearch
- Bounding box query first (cheap), then geo_distance filter
- Geohash cells (precision 6 = ~1.2km cells) used for coarse bucket precomputation
- For "near me" searches, expand geohash radius until enough results

**Availability Pre-Filtering:**
- Maintain a separate `listing_availability_summary` table: `(listing_id, month, blocked_dates_bitmap)`
- Elasticsearch `available_months` field updated via CDC pipeline
- Detailed date-level conflict check happens only for final booking, not search

### 5.2 Availability Calendar
The availability system is the heart of booking conflict prevention.

**Schema Design:**
```sql
CREATE TABLE availability (
    listing_id    BIGINT NOT NULL,
    date          DATE NOT NULL,
    status        ENUM('available', 'blocked', 'booked') DEFAULT 'available',
    booking_id    BIGINT REFERENCES bookings(id),
    price         DECIMAL(10,2),
    PRIMARY KEY (listing_id, date)
);
```

**Conflict Prevention:**
- Approach 1 (Optimistic Locking): SELECT availability rows for date range, check all = 'available', then UPDATE in single transaction with version check
- Approach 2 (PostgreSQL Advisory Locks): `SELECT pg_advisory_xact_lock(listing_id)` before date range check
- Approach 3 (SELECT FOR UPDATE): Lock specific rows, preventing concurrent transactions on same listing+dates
- **Chosen**: SELECT FOR UPDATE with explicit transaction, combined with DB-level UNIQUE constraint on (listing_id, date, status='booked')

### 5.3 Booking Service
**Booking State Machine:**
```
PENDING_PAYMENT → CONFIRMED → CHECKED_IN → COMPLETED
                           ↓
                      CANCELLED (by guest or host)
                           ↓
                      REFUNDED
```

**Booking Creation Flow:**
1. Validate request (dates valid, guest count ≤ listing capacity)
2. BEGIN TRANSACTION
3. SELECT availability rows FOR UPDATE (locks rows)
4. Verify all dates have status='available'
5. Calculate total price (base + fees + taxes)
6. Create payment hold via Payment Service
7. INSERT booking record
8. UPDATE availability rows to status='booked'
9. COMMIT TRANSACTION
10. Send confirmation emails/push notifications asynchronously

### 5.4 Dynamic Pricing Engine
```
final_price = base_price × seasonal_factor × demand_multiplier × length_of_stay_discount

seasonal_factor:
  - Peak season (summer/holidays): 1.3–1.8
  - Shoulder season: 1.0–1.2
  - Low season: 0.7–0.9

demand_multiplier:
  - Computed from: (bookings in area last 7 days) / (avg bookings same period)
  - Scale: 0.8–2.0

length_of_stay_discount:
  - 7+ nights: 0.9 (10% off)
  - 28+ nights: 0.75 (25% off)
```

### 5.5 Review System (Mutual Reveal)
Both guest and host submit reviews independently. Neither can see the other's review until BOTH submit OR the review window closes (14 days after checkout).

**State Machine:**
```
REVIEW_PENDING → GUEST_SUBMITTED (host sees nothing)
             → HOST_SUBMITTED (guest sees nothing)
             → BOTH_SUBMITTED → REVEALED (both visible)
             → DEADLINE_PASSED → REVEALED (submitted ones visible)
```

**Implementation:**
- Store reviews with `revealed = false` initially
- Background job runs daily to flip `revealed = true` for expired windows
- On second submission: flip `revealed = true` immediately

### 5.6 Photo Storage Pipeline
1. Host uploads photo → API Gateway → Photo Service
2. Photo Service writes original to S3 with unique key
3. Triggers Lambda/worker: generate 4 sizes (thumbnail 150px, small 320px, medium 800px, large 1200px)
4. All sizes stored in S3 under `/listings/{id}/{photo_id}/{size}.jpg`
5. CDN (CloudFront) sits in front of S3 with long TTL (1 year for immutable photos)
6. Listing record stores `[{photo_id, sizes: {thumb: url, sm: url, md: url, lg: url}}]`

---

## 6. Database Design

### Users Table
```sql
CREATE TABLE users (
    id              BIGSERIAL PRIMARY KEY,
    email           VARCHAR(255) UNIQUE NOT NULL,
    phone           VARCHAR(20),
    first_name      VARCHAR(100),
    last_name       VARCHAR(100),
    profile_photo   VARCHAR(500),
    is_host         BOOLEAN DEFAULT false,
    verified_id     BOOLEAN DEFAULT false,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    rating_as_guest DECIMAL(3,2),
    rating_as_host  DECIMAL(3,2)
);
```

### Listings Table
```sql
CREATE TABLE listings (
    id              BIGSERIAL PRIMARY KEY,
    host_id         BIGINT REFERENCES users(id),
    title           VARCHAR(255),
    description     TEXT,
    property_type   VARCHAR(50),    -- apartment, house, cabin, etc.
    room_type       VARCHAR(50),    -- entire_place, private_room, shared_room
    lat             DECIMAL(9,6),
    lon             DECIMAL(9,6),
    city            VARCHAR(100),
    country         VARCHAR(100),
    max_guests      INT,
    bedrooms        INT,
    bathrooms       DECIMAL(3,1),
    base_price      DECIMAL(10,2),
    cleaning_fee    DECIMAL(10,2),
    amenities       JSONB,          -- {wifi: true, pool: false, ...}
    photos          JSONB,
    is_active       BOOLEAN DEFAULT true,
    instant_book    BOOLEAN DEFAULT false,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    avg_rating      DECIMAL(3,2),
    review_count    INT DEFAULT 0
);
CREATE INDEX idx_listings_geo ON listings USING GIST (point(lon, lat));
CREATE INDEX idx_listings_host ON listings(host_id);
CREATE INDEX idx_listings_price ON listings(base_price);
```

### Bookings Table
```sql
CREATE TABLE bookings (
    id              BIGSERIAL PRIMARY KEY,
    listing_id      BIGINT REFERENCES listings(id),
    guest_id        BIGINT REFERENCES users(id),
    check_in        DATE NOT NULL,
    check_out       DATE NOT NULL,
    guests          INT,
    status          VARCHAR(20) DEFAULT 'confirmed',
    total_price     DECIMAL(10,2),
    base_amount     DECIMAL(10,2),
    service_fee     DECIMAL(10,2),
    taxes           DECIMAL(10,2),
    payment_intent  VARCHAR(255),   -- Stripe payment intent ID
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT no_overlap EXCLUDE USING gist (
        listing_id WITH =,
        daterange(check_in, check_out) WITH &&
    ) WHERE (status NOT IN ('cancelled', 'refunded'))
);
```

### Reviews Table
```sql
CREATE TABLE reviews (
    id              BIGSERIAL PRIMARY KEY,
    booking_id      BIGINT REFERENCES bookings(id),
    reviewer_id     BIGINT REFERENCES users(id),
    reviewee_id     BIGINT REFERENCES users(id),
    listing_id      BIGINT REFERENCES listings(id),
    rating          INT CHECK (rating BETWEEN 1 AND 5),
    text            TEXT,
    review_type     VARCHAR(10),    -- 'guest_to_host', 'host_to_guest'
    revealed        BOOLEAN DEFAULT false,
    submitted_at    TIMESTAMPTZ,
    deadline        TIMESTAMPTZ,    -- 14 days after checkout
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
```

### Messages Table
```sql
CREATE TABLE messages (
    id              BIGSERIAL PRIMARY KEY,
    thread_id       BIGINT NOT NULL,
    sender_id       BIGINT REFERENCES users(id),
    recipient_id    BIGINT REFERENCES users(id),
    booking_id      BIGINT REFERENCES bookings(id),
    body            TEXT,
    sent_at         TIMESTAMPTZ DEFAULT NOW(),
    read_at         TIMESTAMPTZ
);
CREATE INDEX idx_messages_thread ON messages(thread_id, sent_at DESC);
```

---

## 7. API Design

### Search API
```
GET /api/v1/search
Query params:
  - location: string (city, address, or "lat,lon")
  - check_in: date (YYYY-MM-DD)
  - check_out: date (YYYY-MM-DD)
  - guests: int
  - price_min: decimal
  - price_max: decimal
  - amenities: comma-separated list
  - property_type: string
  - instant_book: boolean
  - sort: "relevance" | "price_asc" | "price_desc" | "rating"
  - cursor: string (for pagination)
  - limit: int (default 20, max 50)

Response: { listings: [...], next_cursor: "...", total_count: 1234 }
```

### Booking API
```
POST /api/v1/bookings
Body: { listing_id, check_in, check_out, guests, payment_method_id }
Response: { booking_id, status, total_price, payment_intent_id }

GET /api/v1/bookings/{booking_id}
Response: { booking details + listing summary + host info }

DELETE /api/v1/bookings/{booking_id}
Body: { reason }
Response: { refund_amount, refund_timeline }
```

### Listing API
```
POST /api/v1/listings
PUT /api/v1/listings/{listing_id}
GET /api/v1/listings/{listing_id}
GET /api/v1/listings/{listing_id}/availability?month=2024-06

POST /api/v1/listings/{listing_id}/photos
DELETE /api/v1/listings/{listing_id}/photos/{photo_id}
```

### Review API
```
POST /api/v1/reviews
Body: { booking_id, rating, text }
Response: { review_id, status: "submitted_awaiting_counterpart" }

GET /api/v1/listings/{listing_id}/reviews?page=1&limit=20
GET /api/v1/users/{user_id}/reviews?type=guest|host
```

---

## 8. Scalability & Bottlenecks

### Bottleneck 1: Search at Scale
- **Problem**: 500K searches/hour with geo + date filtering
- **Solution**: Elasticsearch cluster with dedicated search nodes; cache popular search queries (same city/dates) in Redis with 5-minute TTL; pre-warm cache for top 100 cities

### Bottleneck 2: Availability Calendar Contention
- **Problem**: Popular listings on peak dates get many concurrent booking attempts
- **Solution**: Use PostgreSQL row-level locking (SELECT FOR UPDATE); queue excess requests via Redis-based booking queue per listing; implement exponential backoff on conflict

### Bottleneck 3: Search Index Freshness
- **Problem**: New bookings must immediately block dates in search results
- **Solution**: CDC pipeline (Debezium) watches bookings table → publishes to Kafka → Elasticsearch consumer updates availability field within seconds; fallback to real-time check on listing page load

### Bottleneck 4: Photo Storage and Delivery
- **Problem**: 70TB photos, high CDN egress cost
- **Solution**: Lazy resize (only generate sizes on first request), aggressive CDN caching (1-year TTL for immutable photos), WebP format conversion (30-40% smaller), progressive JPEG for faster perceived load

### Bottleneck 5: Hot Listings
- **Problem**: Viral listings getting 10K concurrent viewers
- **Solution**: Cache listing detail pages in CDN; cache availability calendar in Redis (invalidate on booking); rate limit booking attempts per user per listing

---

## 9. Trade-offs & Design Decisions

### Decision 1: Elasticsearch vs PostgreSQL Full-Text Search
- **PostgreSQL FTS**: Simpler ops, ACID transactions, but geo queries are slow at scale
- **Elasticsearch**: Superior geo search, better relevance tuning, horizontal scaling
- **Choice**: Elasticsearch for search reads, PostgreSQL as source of truth; sync via CDC
- **Trade-off**: Eventual consistency between booking and search index (~seconds lag)

### Decision 2: Availability Table Design
- **Option A**: Sparse table (only blocked dates stored) — small table, but range queries need NOT EXISTS subquery
- **Option B**: Dense table (every date × listing) — fast queries but 7M × 365 = 2.5B rows
- **Option C**: Bitmap per week/month — complex but space-efficient
- **Choice**: Option A (sparse) with date range index, plus Redis bitmap cache for current month

### Decision 3: Payment Hold Strategy
- **Option A**: Charge immediately at booking — simpler, but guests complain about holds
- **Option B**: Hold at booking, capture at check-in — industry standard, protects both parties
- **Choice**: Option B; use Stripe payment intents with manual capture; cancel hold if booking cancelled within free cancellation window

### Decision 4: Monolith vs Microservices
- **Monolith**: Easier to start, ACID transactions across services, simpler deployment
- **Microservices**: Independent scaling, team autonomy, fault isolation
- **Choice**: Service-oriented architecture (middle ground): separate deployable services but NOT nano-services; Search, Booking, User, Messaging, Payment as distinct services sharing a database cluster initially, migrate to separate DBs as scale demands

### Decision 5: Review Mutual Reveal Timing
- **Option A**: Simultaneous reveal only when both submit (guest never writes if they see bad review first)
- **Option B**: Reveal after 14 days regardless
- **Choice**: Both: reveal immediately when both submit, OR after 14-day deadline, whichever is earlier

---

## 10. Key Interview Talking Points

### 1. The Double-Booking Problem
The hardest part of Airbnb design. Three approaches:
- **Optimistic locking**: Read availability, check, write — retry on conflict. High contention on popular dates.
- **Pessimistic locking (SELECT FOR UPDATE)**: Lock rows during transaction. Serializes concurrent requests, prevents conflicts at DB level.
- **PostgreSQL EXCLUDE constraint with daterange**: Database-level constraint prevents overlapping bookings for same listing. Zero-code conflict detection.
Best answer: Use SELECT FOR UPDATE inside a transaction + EXCLUDE constraint as final safety net.

### 2. Search Architecture
Explain why you CAN'T do geo search efficiently in PostgreSQL at 7M listings with 500K queries/hour. Elasticsearch geo_point + geo_distance queries are O(log n) with inverted index. Explain the CDC sync pipeline to keep search index fresh without sacrificing booking consistency.

### 3. Availability Calendar Design
The sparse vs. dense table trade-off. Most production systems use a hybrid: sparse table for source of truth + Redis bitmap for fast cache. Explain how you invalidate the cache when a booking is created.

### 4. Pricing Engine
Explain the formula: `base × seasonal × demand × length_discount`. The interesting part is demand_multiplier computed from real-time search/booking signals in the area. This is a ML model in production (Airbnb uses neural networks), but interview-level: use rolling 7-day booking rate ratio.

### 5. Review System Design
The mutual reveal pattern is a classic trust-and-safety design problem. The key insight: reviews should be stored but not revealed until the review window closes. This prevents strategic review behavior (writing a bad review because you got one).

### 6. Capacity Estimates to Know
- 7M listings, 100M users, 500K searches/hour, 1M bookings/month
- Elasticsearch needs ~14GB for listing metadata + 51GB for availability summaries
- Photos: 70TB is the big number — justify CDN + lazy resize + WebP conversion

### 7. CDC Pipeline
Explain how Change Data Capture (Debezium reading PostgreSQL WAL) enables you to keep Elasticsearch in sync without dual writes. WAL-based CDC is transactionally safe: the search index update only happens after the DB transaction commits.

### 8. Global Distribution
For global scale: region-based sharding of listings (US listings on US PostgreSQL cluster, EU on EU cluster). Search queries fan out to relevant region clusters. Booking data replicated globally for cross-region reads but written to home region.
