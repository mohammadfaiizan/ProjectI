# System Design: URL Shortener (e.g., bit.ly, TinyURL)

---

## 1. Problem Statement

Design a URL shortening service that takes a long URL and returns a short, unique URL. When the short URL is accessed, it redirects users to the original long URL. The system must handle massive read traffic, support analytics, custom aliases, and URL expiry.

---

## 2. Clarifying Questions to Ask

- What is the expected scale? (DAU, URLs created per day)
- Should we support custom aliases (e.g., bit.ly/my-brand)?
- Do URLs expire? If so, who controls expiry — user-defined or system default?
- Do we need analytics? (click count, geographic data, referrer tracking)
- Should we support rate limiting per user to prevent abuse?
- Is 301 (permanent) or 302 (temporary) redirect preferred?
- Do we need user accounts, or is this anonymous?
- Should the same long URL always produce the same short URL (idempotency)?
- What is the maximum length of the short code?
- What is the SLA for redirect latency? (P99 < 10ms?)

---

## 3. Functional Requirements

1. Given a long URL, generate a unique short URL (7-character code by default).
2. When a short URL is accessed, redirect to the original long URL.
3. Support custom aliases provided by the user.
4. URLs can have an optional expiry time; expired URLs return 404.
5. Track click analytics: total clicks, unique clicks, timestamp, user-agent, referrer.
6. Rate limit URL creation per user (e.g., 100 URLs/day for free tier).
7. Allow users to delete their short URLs.
8. Return an error if custom alias is already taken.

---

## 4. Non-Functional Requirements

- **Availability**: 99.99% uptime (the system must always redirect)
- **Latency**: Redirect P99 < 10ms; creation P99 < 100ms
- **Durability**: No URL should be silently lost once created
- **Consistency**: Eventual consistency acceptable for analytics; strong for URL resolution
- **Scalability**: Handle 500M new URLs/day and 50B redirects/day
- **Security**: Prevent phishing/spam URLs; sanitize inputs
- **Read-heavy**: 100:1 read-to-write ratio

---

## 5. Capacity Estimation

### Users & Traffic
- Daily Active Users (DAU): 100M
- New URLs created per day: 500M
- URL creation QPS: 500M / 86,400 = ~5,800 writes/sec
- Peak write QPS (2x): ~11,600 writes/sec

### Read Traffic
- Read:Write ratio = 100:1
- Read QPS: 5,800 * 100 = 580,000 reads/sec
- Peak read QPS (2x): ~1.16M reads/sec

### Storage
- Average URL record size: ~500 bytes (long URL 200B + metadata 300B)
- Storage per day: 500M * 500B = 250 GB/day
- Storage for 5 years: 250 GB * 365 * 5 = ~456 TB

### Bandwidth
- Inbound (writes): 5,800 req/s * 500B = ~2.9 MB/s
- Outbound (reads): 580,000 req/s * 500B = ~290 MB/s

### Cache
- 80/20 rule: 20% of URLs drive 80% of traffic
- Hot URLs to cache: 500M * 0.20 = 100M URLs
- Cache storage: 100M * 500B = ~50 GB (fits in Redis cluster)

---

## 6. High-Level Architecture

```
Client
  |
  v
[Load Balancer]
  |           \
  v            v
[API Servers]   [API Servers]  (stateless, horizontally scalable)
  |     |
  v     v
[Cache Layer]    [Rate Limiter]
(Redis Cluster)  (Redis + Token Bucket)
  |
  v
[Database Layer]
  |              |
  v              v
[Primary DB]  [Read Replicas]
(PostgreSQL)  (x5 replicas)
  |
  v
[Analytics Service] --> [ClickHouse / Kafka]
  |
  v
[CDN / GeoDNS]
(Route to nearest PoP)
```

### Request Flow - URL Creation
```
POST /api/shorten
  --> Rate Limiter check
  --> Check if custom alias available (DB lookup)
  --> Generate base62 short code
  --> Write to DB (primary)
  --> Write to Cache (Redis)
  --> Return short URL
```

### Request Flow - URL Redirect
```
GET /{shortCode}
  --> Check Redis Cache (L1)
  --> Cache miss --> DB read replica lookup
  --> 301/302 redirect to long URL
  --> Async: log click event to Kafka
```

---

## 7. Component Deep-Dive

### 7.1 Short Code Generation

**Option A: MD5 Hash + Base62 Encoding**
- Take MD5(longURL + timestamp + userID)
- Take first 7 characters of base62-encoded hash
- Risk: collisions — must check DB and retry with incremented seed

**Option B: Auto-Increment Counter + Base62**
- Use a distributed counter (Zookeeper or Redis INCR)
- Encode counter as base62
- Pros: no collisions, predictable, sequential
- Cons: guessable, requires coordination service

**Option C: Pre-generated Code Pool**
- Background job pre-generates millions of codes
- Store unused codes in a "pool" table
- API takes one code from pool per request
- Pros: O(1) lookup, no coordination at request time
- Cons: wasted codes if URLs are deleted

**Chosen Approach**: Option B (counter + base62) with a distributed counter service. For 7 characters, base62^7 = 3.5 trillion unique codes — sufficient for decades.

### 7.2 Base62 Encoding

Characters: `0-9A-Za-z` (62 characters)
- Input: integer n
- Repeatedly divide by 62, collect remainders
- Map remainders to character set
- 7 characters supports up to 62^7 ≈ 3.5 trillion URLs

### 7.3 Collision Handling

For hash-based approach:
1. Generate hash code
2. Check DB/Cache if code exists
3. If collision: append salt (counter++) and rehash
4. Retry up to 5 times before failing
5. Use probabilistic data structure (Bloom Filter) to avoid DB lookup — fast membership check

### 7.4 Caching Strategy (Redis)

- **Cache pattern**: Cache-aside (lazy loading)
- **Key**: `url:{shortCode}` → `{longURL, expiry, userID}`
- **TTL**: 24 hours for regular URLs; never-expire for permanent URLs
- **Eviction**: LRU (Least Recently Used)
- **80/20 rule**: Cache the top 20% of URLs that get 80% of traffic
- **Cache invalidation**: On URL deletion, explicitly delete from cache
- **Cache warming**: Pre-load top URLs on startup

### 7.5 Redirect: 301 vs 302

| Factor | 301 (Permanent) | 302 (Temporary) |
|--------|----------------|-----------------|
| Browser caches | Yes | No |
| Server receives repeat requests | No | Yes |
| Analytics accuracy | Lower | Higher |
| CDN cacheability | High | Low |

**Decision**: Use 302 for analytics accuracy. 301 means browser handles redirect without hitting our server, so we lose click tracking. Use 301 only if user explicitly requests permanent redirect.

### 7.6 Rate Limiting

- **Algorithm**: Token Bucket per user per day
- **Storage**: Redis with key `rate:{userID}:{date}` → count
- **Free tier**: 100 URLs/day
- **Pro tier**: 10,000 URLs/day
- **IP-based limiting** for anonymous users: 10 URLs/hour per IP

### 7.7 Analytics Service

- Click events pushed to **Kafka** topic `url-clicks`
- Consumer writes to **ClickHouse** (columnar DB for analytics queries)
- Data captured: timestamp, shortCode, userAgent, referrer, IP (geo-resolved), country
- Aggregation job runs hourly to update dashboard metrics

### 7.8 Database Sharding

- **Primary key**: shortCode (string, 7 chars)
- **Sharding key**: first character of shortCode (62 shards possible)
- Each shard is a PostgreSQL primary with 2-3 read replicas
- **Cross-shard queries**: Not needed (shortCode is always the lookup key)

---

## 8. Database Design

### Table: short_codes
```sql
CREATE TABLE short_codes (
    short_code      VARCHAR(16) PRIMARY KEY,
    long_url        TEXT NOT NULL,
    user_id         BIGINT,
    created_at      TIMESTAMP DEFAULT NOW(),
    expires_at      TIMESTAMP,
    is_custom       BOOLEAN DEFAULT FALSE,
    is_active       BOOLEAN DEFAULT TRUE,
    click_count     BIGINT DEFAULT 0,
    title           VARCHAR(512),
    INDEX idx_user_id (user_id),
    INDEX idx_expires_at (expires_at),
    INDEX idx_created_at (created_at)
);
```

### Table: url_analytics
```sql
CREATE TABLE url_analytics (
    id              BIGSERIAL PRIMARY KEY,
    short_code      VARCHAR(16) NOT NULL,
    clicked_at      TIMESTAMP DEFAULT NOW(),
    user_agent      TEXT,
    referrer        TEXT,
    ip_address      INET,
    country         VARCHAR(2),
    city            VARCHAR(100),
    FOREIGN KEY (short_code) REFERENCES short_codes(short_code)
);
-- Partition by clicked_at (monthly partitions)
-- Use ClickHouse for analytics at scale
```

### Table: users
```sql
CREATE TABLE users (
    id              BIGSERIAL PRIMARY KEY,
    username        VARCHAR(50) UNIQUE NOT NULL,
    email           VARCHAR(255) UNIQUE NOT NULL,
    api_key         VARCHAR(64) UNIQUE,
    tier            VARCHAR(20) DEFAULT 'free',
    created_at      TIMESTAMP DEFAULT NOW(),
    daily_limit     INT DEFAULT 100
);
```

### Table: rate_limits (Redis — not SQL)
```
Key: "rl:{user_id}:{YYYY-MM-DD}"
Value: integer (count of URLs created today)
TTL: 86400 seconds (1 day)
```

---

## 9. API Design

### Create Short URL
```
POST /api/v1/shorten
Authorization: Bearer {api_key}

Request Body:
{
  "long_url": "https://www.example.com/very/long/path?param=value",
  "custom_alias": "my-alias",          // optional
  "expires_at": "2025-12-31T23:59:59Z" // optional
}

Response 200:
{
  "short_url": "https://sho.rt/abc1234",
  "short_code": "abc1234",
  "long_url": "https://www.example.com/very/long/path?param=value",
  "created_at": "2024-01-15T10:30:00Z",
  "expires_at": "2025-12-31T23:59:59Z"
}

Response 409 (alias taken):
{ "error": "custom alias already in use" }

Response 429 (rate limit):
{ "error": "daily limit exceeded", "limit": 100, "reset_at": "2024-01-16T00:00:00Z" }
```

### Redirect
```
GET /{shortCode}

Response 302:
Location: https://www.original-long-url.com/...
X-Redirect-To: https://www.original-long-url.com/...

Response 404:
{ "error": "short URL not found or expired" }
```

### Get Analytics
```
GET /api/v1/urls/{shortCode}/analytics
Authorization: Bearer {api_key}

Response 200:
{
  "short_code": "abc1234",
  "total_clicks": 15234,
  "unique_clicks": 8921,
  "clicks_by_day": [
    { "date": "2024-01-15", "count": 234 }
  ],
  "top_countries": [
    { "country": "US", "count": 5000 }
  ],
  "top_referrers": [
    { "referrer": "twitter.com", "count": 3000 }
  ]
}
```

### Delete URL
```
DELETE /api/v1/urls/{shortCode}
Authorization: Bearer {api_key}

Response 204: No Content
Response 403: { "error": "not authorized to delete this URL" }
```

### List User URLs
```
GET /api/v1/urls?page=1&limit=20
Authorization: Bearer {api_key}

Response 200:
{
  "urls": [...],
  "total": 1500,
  "page": 1,
  "limit": 20
}
```

---

## 10. Scalability & Bottlenecks

### Bottleneck 1: Read QPS (580K reads/sec)
- Solution: Redis cache cluster handles ~1M reads/sec per node
- Cache hit rate target: >99% (using LRU eviction)
- CDN layer caches redirect responses for popular URLs

### Bottleneck 2: Write QPS (5,800 writes/sec)
- Solution: Distributed counter (Redis INCR is atomic, ~100K ops/sec)
- Batch writes to DB using write-behind cache
- Master DB handles writes; replicas handle reads

### Bottleneck 3: URL Code Generation at Scale
- Pre-generation pool: background job generates codes ahead of time
- Each API server gets a range of 1000 codes from Zookeeper
- Uses codes locally without network calls

### Bottleneck 4: Analytics Write Load
- Decouple analytics from redirect path (async Kafka)
- ClickHouse handles billions of rows efficiently
- Batch inserts every 100ms

### Bottleneck 5: DB Storage Growth
- Partition `url_analytics` table by month, drop old partitions
- Archive expired URLs to cold storage (S3) after 30 days
- Use columnar compression for analytics data

---

## 11. Trade-offs & Design Decisions

### Hash vs Counter for Short Code
- **Hash**: Idempotent (same URL always same code), but collision-prone
- **Counter**: No collisions, but sequential codes are guessable
- **Decision**: Counter with base62 for production; add random salt if guessability is a concern

### 301 vs 302 Redirect
- **301**: Browser caches, no analytics, better for SEO, fewer server hits
- **302**: Every click hits server, accurate analytics
- **Decision**: 302 for analytics; let power users opt into 301

### SQL vs NoSQL for URL Storage
- **SQL (PostgreSQL)**: ACID compliance, complex queries, user ownership queries
- **NoSQL (DynamoDB/Cassandra)**: Better horizontal scale, simpler schema
- **Decision**: PostgreSQL with sharding for structured data + Redis for caching

### Synchronous vs Asynchronous Analytics
- **Synchronous**: Simpler but adds latency to redirect
- **Asynchronous (Kafka)**: Adds complexity but keeps redirect fast
- **Decision**: Async with Kafka; never block redirect on analytics write

### Expiry Check Strategy
- Check expiry in cache (store expires_at in cached value)
- Avoid DB call for expiry check on every redirect
- Background job cleans up expired records daily

---

## 12. Key Interview Talking Points

1. **Base62 math**: 62^7 = 3.5 trillion codes — justify the 7-character choice vs 6 (62^6 = 56B)

2. **Read-heavy design**: Always lead with cache. 99%+ cache hit rate means DB barely sees redirect traffic.

3. **301 vs 302 debate**: Shows you understand HTTP semantics AND product requirements (analytics).

4. **Counter vs hash**: Discuss trade-offs; interviewers love hearing you reason through this.

5. **Distributed counter**: Mention Zookeeper / Redis INCR as coordination mechanisms.

6. **Bloom filter**: Use to check code existence before DB lookup — reduces unnecessary DB reads.

7. **Sharding**: Explain that shortCode is the natural shard key — no cross-shard queries needed.

8. **Async analytics**: Demonstrates you know to decouple hot paths from slower operations.

9. **Rate limiting patterns**: Token bucket vs leaky bucket vs fixed window — know the differences.

10. **Capacity estimation**: Always do the math — 500M URLs/day = 5,800 writes/sec shows you can reason about scale.

11. **Geo-distribution**: CDN at the edge caches popular redirects, reducing latency globally.

12. **Idempotency**: What happens if user submits same URL twice? Return existing short URL or create new one?
