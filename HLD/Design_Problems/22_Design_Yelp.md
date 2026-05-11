# System Design: Yelp

## 1. Problem Statement & Clarifying Questions

### Problem Statement
Design a local business discovery and review platform like Yelp where users can search for businesses by location and category, read and write reviews, view photos, and get recommendations for nearby places.

### Clarifying Questions
1. **Scale**: How many businesses and users? (~150M businesses globally, 200M unique users/month)
2. **Read/Write ratio**: Primarily read-heavy? (Yes: ~1B reads/day, ~10M writes/day)
3. **Search**: Text search + proximity? Both name/category and geo?
4. **Reviews**: Can businesses respond to reviews?
5. **Photos**: User-uploaded photos per business?
6. **Real-time**: Is search result freshness critical? (eventual consistency fine for most)
7. **Personalization**: Should results be personalized per user? (yes, saved places signal)
8. **Check-ins**: Do we track user check-ins at businesses?
9. **Business owner features**: Can owners claim/update their listing?
10. **Moderation**: Auto-spam detection on reviews?

---

## 2. Functional & Non-Functional Requirements

### Functional Requirements
- Users can search for businesses by location (lat/lon or city), category, keywords
- Search results ranked by: distance + rating + review count + relevance
- Users can view full business detail: hours, address, photos, attributes, menu
- Users can write star-rated text reviews with photos
- Other users can vote reviews as "Useful", "Funny", "Cool"
- Business owners can claim listing, update info, respond to reviews
- Users can save businesses to lists (bookmarks)
- Check-in feature: user confirms they visited a business
- Tips: short-form posts (< 200 chars) about a business
- Trending: most checked-in / reviewed businesses in area

### Non-Functional Requirements
- **Availability**: 99.9% uptime
- **Read latency**: < 100ms p99 for nearby search queries
- **Write latency**: < 500ms for review submission
- **Scale**: 150M businesses, 200M MAU, 1B reads/day, 10M writes/day
- **Consistency**: Eventual consistency acceptable for search; strong for reviews
- **Geo accuracy**: Search results within correct radius
- **Storage**: ~10TB for business metadata, ~50TB for photos

---

## 3. Capacity Estimation

### Traffic
- **Reads**: 1B/day → ~11,600 RPS average, ~50K RPS peak
- **Writes**: 10M/day → ~116 WPS average, ~500 WPS peak
- **Search queries**: ~500M/day → ~5,800 QPS average
- **Photo views**: ~300M/day via CDN (mostly cache hits)

### Storage
- **Business records**: 150M × 500 bytes = ~75GB
- **Reviews**: ~500M total × 1KB = ~500GB
- **Photos**: ~5 photos/business × 150M × 300KB avg = ~225TB (with CDN caching)
- **Check-ins**: ~50M/day × 30 bytes × 365 = ~550GB/year
- **Search index (Elasticsearch)**: ~150GB for business names + geo + categories

### Caching
- Top 10K cities account for 80% of searches → cache nearby results per city + category
- Business detail pages: LRU cache, ~10M hot businesses, ~5KB/page = ~50GB cache

---

## 4. High-Level Architecture

```
                        ┌──────────────────────────────────────────────┐
                        │                 Clients                       │
                        │        Web / iOS / Android                    │
                        └──────────────────┬───────────────────────────┘
                                           │
                        ┌──────────────────▼───────────────────────────┐
                        │           CDN (CloudFront)                    │
                        │      Static assets + Photo delivery           │
                        └──────────────────┬───────────────────────────┘
                                           │
                        ┌──────────────────▼───────────────────────────┐
                        │           API Gateway / Load Balancer         │
                        └──┬──────────┬────────────┬───────────┬───────┘
                           │          │            │           │
              ┌────────────▼─┐  ┌─────▼──────┐ ┌──▼───────┐ ┌▼──────────┐
              │ Search Svc   │  │ Business   │ │ Review   │ │ User Svc  │
              │              │  │ Svc        │ │ Svc      │ │           │
              └──────┬───────┘  └─────┬──────┘ └──┬───────┘ └───────────┘
                     │                │            │
        ┌────────────▼──┐  ┌─────────▼──┐  ┌──────▼──────┐
        │ Elasticsearch  │  │ PostgreSQL  │  │ PostgreSQL  │
        │ (Geo + Text    │  │ (Business   │  │ (Reviews +  │
        │  Index)        │  │  Master DB) │  │  Votes)     │
        └───────────────┘  └────────────┘  └─────────────┘
                │                │
        ┌───────▼──────┐  ┌──────▼──────┐
        │  Redis Cache  │  │  S3 + CDN   │
        │ (Search TTL)  │  │  (Photos)   │
        └──────────────┘  └─────────────┘

  CDC: PostgreSQL → Kafka → Elasticsearch sync for near-real-time index updates
  QuadTree service: in-memory spatial index for dynamic density areas
```

---

## 5. Component Deep-Dive

### 5.1 Geo Search Strategy

**Approach 1: Geohash**
- Encode each business lat/lon as a geohash string (e.g., precision 6 = ~1.2km cells)
- Index geohash prefix in Elasticsearch (keyword field)
- Nearby search: compute geohash of center point + 8 neighboring cells
- Filter businesses where `geohash` starts with any of the 9 cell prefixes
- Pros: Simple prefix matching, cache-friendly (same geohash = same result set)
- Cons: Boundary effects at cell edges (neighbor expansion needed)

**Approach 2: QuadTree**
- Recursively divide 2D space into 4 quadrants
- Each leaf node holds ≤ K businesses (K=100)
- Search: traverse tree to find leaf containing query point, expand to neighbors
- Pros: Adapts to business density (sparse rural areas = large nodes, dense NYC = small nodes)
- Cons: Harder to distribute; updates require tree rebalancing

**Chosen**: Geohash for Elasticsearch (simpler, scales horizontally); QuadTree maintained in-memory on dedicated geo service for ultra-low-latency proximity queries.

**Haversine Formula:**
```
a = sin²(Δlat/2) + cos(lat1) × cos(lat2) × sin²(Δlon/2)
c = 2 × atan2(√a, √(1-a))
distance = R × c    (R = 6371 km)
```

### 5.2 Search Ranking
Final score = w1 × distance_score + w2 × rating_score + w3 × popularity_score + w4 × text_relevance

```
distance_score  = 1 / (1 + distance_km)      # closer = higher
rating_score    = bayesian_avg / 5.0          # normalized 0-1
popularity_score = log(review_count + 1) / 10 # log scale
text_relevance  = Elasticsearch BM25 score    # keyword match
```

Weights: w1=0.35, w2=0.30, w3=0.20, w4=0.15

### 5.3 Bayesian Rating
Raw average is misleading for businesses with few reviews.
Bayesian average: `(v × R + m × C) / (v + m)`
- v = number of user votes (review count)
- R = average rating for this business
- m = minimum votes required (e.g., 10)
- C = global average rating (~3.5 stars)

Result: new businesses with 2 reviews aren't unfairly ranked above established 500-review businesses.

### 5.4 Review System
- Reviews stored in PostgreSQL with `business_id` index
- Helpful votes stored separately in `review_votes` table (user_id, review_id, vote_type)
- Anti-spam: text similarity check using MinHash LSH to detect duplicate/copy-paste reviews
- Business owner can post one "Response" per review (stored in `review_responses`)
- Review flagging for moderation: threshold 3 flags → human review queue

### 5.5 Photo Management
1. User uploads → API validates format + size (max 10MB) → S3 presigned URL
2. Upload directly to S3 (bypass app server for large files)
3. S3 event triggers Lambda → generate 3 sizes: thumbnail (100px), medium (400px), large (1200px)
4. CloudFront CDN serves all photo requests
5. Moderation: AWS Rekognition NSFW detection on upload

### 5.6 Caching Strategy
- **L1**: CDN for photos + static pages (TTL: hours to days)
- **L2**: Redis for nearby business lists (TTL: 5 minutes per geohash cell)
- **L3**: Redis for individual business pages (TTL: 1 hour, invalidate on update)
- **Cache key**: `nearby:{geohash6}:{category}:{sort}` → list of business IDs
- **Cache invalidation**: Business update event → Kafka → cache invalidator service

---

## 6. Database Design

### Businesses Table
```sql
CREATE TABLE businesses (
    id              BIGSERIAL PRIMARY KEY,
    owner_id        BIGINT REFERENCES users(id),
    name            VARCHAR(255) NOT NULL,
    description     TEXT,
    phone           VARCHAR(20),
    website         VARCHAR(500),
    address         VARCHAR(500),
    city            VARCHAR(100),
    state           VARCHAR(50),
    country         VARCHAR(50),
    zip             VARCHAR(20),
    lat             DECIMAL(9,6),
    lon             DECIMAL(9,6),
    geohash6        CHAR(6),            -- for geo prefix queries
    categories      TEXT[],             -- ['restaurant', 'italian', 'pizza']
    attributes      JSONB,              -- {outdoor_seating: true, wifi: true}
    hours           JSONB,              -- {mon: "9-5", tue: "9-5", ...}
    price_range     SMALLINT,           -- 1=$, 2=$$, 3=$$$, 4=$$$$
    avg_rating      DECIMAL(3,2),
    review_count    INT DEFAULT 0,
    bayesian_rating DECIMAL(3,2),
    is_claimed      BOOLEAN DEFAULT false,
    is_active       BOOLEAN DEFAULT true,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_businesses_geohash ON businesses(geohash6);
CREATE INDEX idx_businesses_city    ON businesses(city, categories);
CREATE INDEX idx_businesses_rating  ON businesses(bayesian_rating DESC);
```

### Reviews Table
```sql
CREATE TABLE reviews (
    id              BIGSERIAL PRIMARY KEY,
    business_id     BIGINT REFERENCES businesses(id),
    user_id         BIGINT REFERENCES users(id),
    rating          SMALLINT CHECK (rating BETWEEN 1 AND 5),
    text            TEXT,
    useful_count    INT DEFAULT 0,
    funny_count     INT DEFAULT 0,
    cool_count      INT DEFAULT 0,
    photo_count     INT DEFAULT 0,
    is_flagged      BOOLEAN DEFAULT false,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (business_id, user_id)   -- one review per user per business
);
CREATE INDEX idx_reviews_business ON reviews(business_id, created_at DESC);
CREATE INDEX idx_reviews_user     ON reviews(user_id);
```

### Check-ins Table
```sql
CREATE TABLE check_ins (
    id          BIGSERIAL PRIMARY KEY,
    business_id BIGINT REFERENCES businesses(id),
    user_id     BIGINT REFERENCES users(id),
    checked_in_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_checkins_business ON check_ins(business_id, checked_in_at DESC);
```

### Photos Table
```sql
CREATE TABLE photos (
    id          BIGSERIAL PRIMARY KEY,
    business_id BIGINT REFERENCES businesses(id),
    user_id     BIGINT REFERENCES users(id),
    review_id   BIGINT REFERENCES reviews(id),
    s3_key      VARCHAR(500),
    caption     VARCHAR(200),
    is_flagged  BOOLEAN DEFAULT false,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
```

---

## 7. API Design

### Search API
```
GET /api/v1/businesses/search
Query params:
  - q: string (search term, optional)
  - lat: float
  - lon: float
  - radius: int (meters, default 5000, max 40000)
  - categories: comma-separated
  - price: "1,2,3" (price range filter)
  - open_now: boolean
  - sort: "distance" | "rating" | "review_count" | "relevance"
  - limit: int (default 20)
  - offset: int

Response: {
  businesses: [{id, name, rating, review_count, distance_m, price_range,
                address, categories, thumbnail_url, is_open}],
  total: 1250
}
```

### Business Detail API
```
GET /api/v1/businesses/{business_id}
Response: { full business object + top 3 reviews + photo URLs }

PUT /api/v1/businesses/{business_id}
(Owner-only: update hours, attributes, description)

POST /api/v1/businesses/{business_id}/check_in
POST /api/v1/businesses/{business_id}/tips
```

### Review API
```
POST /api/v1/businesses/{business_id}/reviews
Body: { rating, text, photo_ids[] }

GET /api/v1/businesses/{business_id}/reviews
Query: sort=recent|useful, page, limit

POST /api/v1/reviews/{review_id}/vote
Body: { vote_type: "useful"|"funny"|"cool" }

POST /api/v1/reviews/{review_id}/respond
(Owner-only: post response to review)
```

---

## 8. Scalability & Bottlenecks

### Bottleneck 1: Geo Search at 50K RPS
- Cache geohash-bucketed search results aggressively (80% cache hit rate target)
- Elasticsearch with geo_point and geohash_grid aggregation
- Read replicas for Elasticsearch: 3 replicas per shard
- Hot cells (Manhattan, downtown SF) get dedicated cache warming

### Bottleneck 2: Business Page Reads
- Business detail pages are read-heavy (100:1 read/write ratio)
- Solution: CDN-cacheable rendered business pages; cache invalidation on update
- Redis cluster caches hot business JSON objects (top 10M businesses)

### Bottleneck 3: Review Write Spam
- Rate limiting: max 5 reviews/day per user
- Content hashing to detect near-duplicate submissions
- Suspicious accounts flagged by ML anomaly detection

### Bottleneck 4: Photo Storage Growth
- ~225TB projected; use S3 Intelligent-Tiering (hot → warm → cold)
- Lazy deletion: flagged photos moved to "pending deletion" queue, batch deleted daily
- Deduplication: perceptual hash (pHash) to detect duplicate photo uploads

### Bottleneck 5: Review Count + Rating Aggregation
- Recomputing avg_rating on every write is expensive at scale
- Solution: Kafka stream of new reviews → streaming aggregator → update business counter
- De-normalized: `avg_rating` and `review_count` stored in businesses table (eventual consistency)

---

## 9. Trade-offs & Design Decisions

### Decision 1: Geohash vs QuadTree
- **Geohash**: String prefix matching — works natively in Elasticsearch, easy to cache
- **QuadTree**: Adapts to density, no boundary artifacts, but complex to distribute
- **Choice**: Geohash for primary search, QuadTree as in-memory service for exact proximity on low-latency path
- **Trade-off**: Geohash has boundary effects requiring 9-cell neighbor expansion

### Decision 2: Denormalized Ratings vs Real-time Aggregation
- **Real-time**: `SELECT AVG(rating) FROM reviews WHERE business_id=X` — slow at 500M reviews
- **Denormalized**: Store avg_rating in businesses table, update via triggers/events
- **Choice**: Denormalized with Kafka-driven async updates
- **Trade-off**: Rating can lag by seconds after new review; acceptable for this use case

### Decision 3: Search Result Caching Granularity
- **Per-query cache**: Too many combinations to cache effectively
- **Per-geohash-cell cache**: Fixed granularity cells; cache miss only when cell has recent update
- **Choice**: Cache at geohash precision-6 cell level; TTL 5 minutes
- **Trade-off**: New business won't appear in search for up to 5 minutes

### Decision 4: One Review Per User Per Business
- **Unlimited**: Users can game rating with multiple reviews
- **One review**: Fair but user can't update experience
- **Choice**: One review per user per business (UNIQUE constraint); allow edits to existing review
- **Trade-off**: Businesses can't get fresh ratings from returning customers

### Decision 5: Elasticsearch vs PostgreSQL Full-Text Search
- At 150M businesses and 50K RPS, PostgreSQL tsvector full-text search hits limits
- Elasticsearch handles combined geo + text + filter queries efficiently at this scale
- Maintain PostgreSQL as source of truth; sync to ES via Debezium CDC

---

## 10. Key Interview Talking Points

### 1. Geohash Spatial Indexing
The key insight: encode lat/lon as a base-32 string where common prefixes = geographic proximity. For search, compute the geohash of the center point, then query all businesses sharing the same geohash prefix. Include 8 neighboring cells to handle edge cases. In Elasticsearch: `geo_distance` query uses a Lucene spatial index under the hood, but explaining geohash shows you understand the algorithm.

### 2. Bayesian Rating vs Simple Average
Simple average is unfair to new businesses. Bayesian average shrinks ratings toward the global mean when review count is low. Formula: `(v×R + m×C) / (v+m)`. Use this when ranking search results. Mention Wilson score lower bound as an alternative for binary upvote/downvote systems.

### 3. Read-Heavy Optimization
1B reads vs 10M writes → aggressive caching at every layer:
- CDN for static content
- Redis for search result lists (geohash → business IDs)
- Redis for business detail objects
- CDN for photos
Key insight: cache the result lists (IDs), not the full objects, to reduce cache size and enable per-item invalidation.

### 4. Elasticsearch Sync Strategy
Dual-write risks inconsistency (write to PG succeeds, ES fails). CDC (Change Data Capture) via Debezium reads PostgreSQL WAL → Kafka → ES consumer. This guarantees at-least-once delivery and maintains PG as single source of truth.

### 5. QuadTree Deep Dive
If asked for an alternative to geohash: QuadTree recursively divides the map. Dense areas (NYC) get deep subdivision → small cells with few businesses each. Sparse areas (Montana) get shallow subdivision → large cells. Max K businesses per leaf (e.g., 100). Search: walk tree to find cell, if not enough results, expand to parent. Trade-off: harder to update on new business add (may trigger rebalance).

### 6. Scale Numbers
- 150M businesses, 200M MAU
- 1B reads/day = 11,600 RPS → 50K peak
- 10M writes/day = 116 WPS
- Photos: 225TB → must use tiered storage

### 7. Anti-Spam for Reviews
- Rate limit: 5 reviews/day per user
- Duplicate detection: MinHash / SimHash for near-duplicate text
- Behavioral signals: account age, past review history
- Flag threshold: auto-hide after N flags, human review queue
