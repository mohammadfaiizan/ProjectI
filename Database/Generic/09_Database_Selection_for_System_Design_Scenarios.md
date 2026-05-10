# Database Selection for System Design Scenarios

---

## Easy (Q1–Q7)

---

**Q1. What are the primary criteria for choosing a database in a system design interview?**

Five dimensions drive database selection:

**1. Data model**
- Relational (tables, JOINs) → PostgreSQL, MySQL
- Document (nested JSON) → MongoDB, DynamoDB
- Wide-column (sparse, high write) → Cassandra, HBase
- Graph (nodes, edges, traversal) → Neo4j, Amazon Neptune
- Key-value (cache, session) → Redis, DynamoDB (simple key access)
- Time-series (append, rollup) → InfluxDB, TimescaleDB, ClickHouse

**2. Access patterns**
- Random reads by primary key → any DB
- Complex ad-hoc JOINs → relational
- High write throughput (millions/sec) → Cassandra, DynamoDB
- Full-text search → Elasticsearch, or PostgreSQL tsvector
- Aggregations on billions of rows → columnar (ClickHouse, Redshift, BigQuery)

**3. Consistency requirements**
- Strong consistency (financial, inventory) → RDBMS or NewSQL (CockroachDB)
- Eventual consistency acceptable (social feeds, analytics) → Cassandra, DynamoDB AP mode

**4. Scale**
- < 10M rows, moderate traffic → PostgreSQL with replicas
- Multi-region, global scale → Spanner, CockroachDB, DynamoDB global tables
- Write-heavy with known partition key → Cassandra, DynamoDB

**5. Operational complexity tolerance**
- Low ops team → managed cloud (RDS, Aurora, DynamoDB, Firestore)
- High control needed → self-managed PostgreSQL with Patroni

---

**Q2. What database would you use for a ride-sharing app like Uber?**

Different data has different requirements — polyglot persistence is appropriate here:

| Data | DB | Reason |
|------|----|--------|
| Driver/rider profiles, trips | **PostgreSQL** | Relational, ACID for payment records |
| Driver geolocation (real-time) | **Redis GEO** (GEOADD/GEORADIUS) | Sub-millisecond geo queries, Redis handles 1M writes/sec |
| Driver location history | **Cassandra** | Write-heavy time-series, append-only |
| Trip matching (active state) | **Redis** or **DynamoDB** | Low-latency, key-based access |
| Analytics / data warehouse | **ClickHouse** or **BigQuery** | Analytical queries on billions of trip records |
| Search (driver supply in area) | **Redis GEO** or **PostGIS** | Depends on latency requirement |
| Message/notification delivery | **Kafka** → **Cassandra** | High-throughput write, time-ordered per user |

**Interview tip:** Always explain *why* each choice. Saying "I'd use Redis for geo" without explaining that Redis GEORADIUS returns nearby sorted results in O(N+log(N)) demonstrates understanding.

---

**Q3. What database would you use for a social media platform like Twitter/Instagram?**

| Data | DB | Reason |
|------|----|--------|
| User profiles, follows | **PostgreSQL** | Relational, moderate size, ACID for follows |
| Tweets/posts | **PostgreSQL** or **Cassandra** | Cassandra if post volume is > 10K writes/sec |
| Timelines (fan-out) | **Redis** (sorted set per user) | Sub-millisecond feed reads for active users |
| Media metadata | **PostgreSQL** or **DynamoDB** | Key-based lookup by media ID |
| Media files (images, videos) | **S3 / object storage** | Not a database — binary large objects |
| Search (hashtags, user search) | **Elasticsearch** | Full-text, fuzzy search, real-time indexing |
| Analytics (trending, engagement) | **ClickHouse** or **BigQuery** | Aggregations on billions of events |
| Likes/counters | **Redis** (INCR) | Atomic counters, no locking |

**Key design choice — feed generation:**
- **Fan-out on write (push):** Write tweet to all followers' Redis feeds at post time. Fast reads, but expensive for users with millions of followers (celebrity problem).
- **Fan-out on read (pull):** Generate feed at read time from follow graph. Cheaper writes, slower reads.
- **Hybrid:** Pre-compute feeds for regular users (< 1M followers), fetch on-demand for celebrities.

---

**Q4. What database would you use for an e-commerce platform like Amazon?**

| Data | DB | Reason |
|------|----|--------|
| Products catalog | **PostgreSQL** + **Elasticsearch** | PG for source of truth, ES for search |
| Inventory (stock levels) | **PostgreSQL** with `SELECT FOR UPDATE` | ACID required — must not oversell |
| Orders, payments | **PostgreSQL** | ACID, foreign keys, transaction integrity |
| Shopping cart (active) | **Redis** or **DynamoDB** | Low-latency, key-value, TTL-based expiry |
| Sessions | **Redis** | Fast, TTL support |
| Product reviews | **PostgreSQL** or **DynamoDB** | Lookup by product_id |
| Recommendations | **Graph DB** or **ML feature store** | Collaborative filtering |
| Price history | **TimescaleDB** or **ClickHouse** | Time-series queries |
| Analytics / reporting | **Redshift** or **ClickHouse** | Columnar, OLAP |

**Critical inventory design:**
```sql
-- Prevent oversell with optimistic locking
UPDATE inventory
SET quantity = quantity - 1,
    version = version + 1
WHERE product_id = $1
  AND quantity >= 1
  AND version = $2;  -- fails if another transaction modified first
```

---

**Q5. What database would you choose for a URL shortener like bit.ly?**

| Data | DB | Reason |
|------|----|--------|
| Short code → long URL mapping | **Cassandra** or **DynamoDB** | High read throughput, simple key lookup, global scale |
| Click analytics | **ClickHouse** or **Cassandra** | Time-series writes, aggregation queries |
| User accounts | **PostgreSQL** | Small scale, relational |
| Caching hot URLs | **Redis** | > 99% of traffic hits < 1% of URLs (Pareto principle) |

**Why not PostgreSQL for the main mapping table?**
- A URL shortener at scale (millions of redirects/sec globally) needs multi-region writes and zero-downtime scaling
- Cassandra's consistent hashing scales horizontally without resharding downtime
- Access pattern is pure key-value: `GET long_url WHERE short_code = 'abc123'` — no JOINs needed

**Short code generation:**
- Encode a 64-bit auto-increment ID as base62 (0-9, a-z, A-Z) → 6 characters for up to 56 billion URLs
- Or use a UUID with hash truncated to 7 chars (collision probability very low at < 1B records)

---

**Q6. What database would you choose for a real-time chat application like WhatsApp or Slack?**

| Data | DB | Reason |
|------|----|--------|
| Messages (stored) | **Cassandra** | Append-only, high write throughput, time-ordered per conversation |
| Message delivery state | **Cassandra** or **DynamoDB** | Key-value per message, high write rate |
| User presence (online/offline) | **Redis** (pub/sub + hash) | Sub-second latency, ephemeral |
| Channel/group metadata | **PostgreSQL** | Low volume, relational |
| Message search | **Elasticsearch** | Full-text search on message content |
| Unread counts | **Redis** (INCR per user/channel) | Atomic, fast |
| Read receipts | **Cassandra** | High write rate (every read = write) |

**Cassandra schema for messages:**
```sql
CREATE TABLE messages (
    conversation_id UUID,
    message_id      TIMEUUID,        -- time-ordered UUID
    sender_id       UUID,
    content         TEXT,
    created_at      TIMESTAMP,
    PRIMARY KEY (conversation_id, message_id)
) WITH CLUSTERING ORDER BY (message_id DESC);
-- Retrieves last N messages efficiently: most recent first
```

**Partition key = conversation_id** → all messages for a conversation are co-located on one partition for fast retrieval.

---

**Q7. What database would you use for a metrics and analytics pipeline collecting IoT sensor data?**

**Write characteristics:** 10,000 devices × 1 write/second = 10,000 writes/sec continuous.
**Read characteristics:** "Show last 24 hours for device X", "Average temperature across region for last week."

| Option | Pros | Cons |
|--------|------|------|
| **TimescaleDB** (PostgreSQL extension) | Familiar SQL, hypertables with auto-partitioning, continuous aggregates | Single-node limits at extreme scale |
| **InfluxDB** | Purpose-built for time-series, built-in downsampling, line protocol ingest | Separate query language (Flux), less ecosystem |
| **Cassandra** | Massive write scale, multi-region | Time-series queries require careful partition design (time-bucketing) |
| **ClickHouse** | Best analytical query speed for large time ranges | Less optimal for high-cardinality individual device lookups |

**Recommended:** TimescaleDB for up to ~100K writes/sec with SQL familiarity, or InfluxDB for purpose-built time-series with downsampling. Add Kafka as a buffer between devices and DB to handle write bursts.

```sql
-- TimescaleDB hypertable
SELECT create_hypertable('sensor_readings', 'time', chunk_time_interval => INTERVAL '1 day');

-- Automatic downsampling with continuous aggregates
CREATE MATERIALIZED VIEW hourly_avg
WITH (timescaledb.continuous) AS
SELECT time_bucket('1 hour', time) AS hour, device_id, avg(temperature)
FROM sensor_readings
GROUP BY 1, 2;
```

---

## Medium (Q8–Q15)

---

**Q8. How would you design the database layer for a financial payment system?**

**Core requirements:**
- ACID transactions (double-entry bookkeeping)
- Idempotency (retries must not double-charge)
- Auditability (immutable ledger)
- High availability (zero downtime)

**Database choices:**

```
Primary: PostgreSQL (ACID, strong consistency)
    ├── Table: accounts (id, user_id, currency, balance)
    ├── Table: transactions (id, idempotency_key, from_account, to_account, 
    │                         amount, status, created_at)
    └── Table: ledger_entries (id, transaction_id, account_id, amount, 
                                 balance_after, entry_type)  -- append-only

Hot cache: Redis (account balance cache, rate limiting)
Analytics: ClickHouse (reporting, fraud pattern queries)
Audit log: Cassandra (append-only, immutable, time-ordered)
```

**Idempotency key pattern:**
```sql
-- Idempotency key prevents double-processing retries
INSERT INTO transactions (idempotency_key, from_account, to_account, amount, status)
VALUES ($idempotency_key, $from, $to, $amount, 'pending')
ON CONFLICT (idempotency_key) DO NOTHING
RETURNING id, status;
-- If conflict: previous attempt exists, return its result without processing again
```

**Double-entry ledger:**
```sql
-- Every transfer creates TWO ledger entries (debit + credit)
BEGIN;
    INSERT INTO ledger_entries (transaction_id, account_id, amount, entry_type)
    VALUES ($txn_id, $from_account, -$amount, 'debit');
    
    INSERT INTO ledger_entries (transaction_id, account_id, amount, entry_type)
    VALUES ($txn_id, $to_account, +$amount, 'credit');
    
    UPDATE transactions SET status = 'completed' WHERE id = $txn_id;
COMMIT;

-- Balance is always derivable from ledger (audit trail)
SELECT sum(amount) AS balance FROM ledger_entries WHERE account_id = $id;
```

**Why not Cassandra for payments:** Cassandra provides eventual consistency — not acceptable for financial data where balance accuracy is critical.

---

**Q9. How would you design the database layer for a search autocomplete / typeahead system?**

**Requirements:** Return suggestions in < 50ms for prefix queries, handle 10K queries/second.

**Option 1: Redis Sorted Sets (for simple use cases)**
```redis
# Score = frequency/relevance, member = suggestion string
ZADD autocomplete:prefix:jo 100 "john smith"
ZADD autocomplete:prefix:jo 95 "joe biden"

# For prefix "jo":
ZREVRANGEBYSCORE autocomplete:prefix:jo +inf -inf LIMIT 0 10
```
**Problem:** Stores a key per prefix → memory-expensive for long prefixes.

**Option 2: Elasticsearch (for complex use cases)**
```json
{
  "mappings": {
    "properties": {
      "suggest": { "type": "completion" },
      "name":    { "type": "keyword" }
    }
  }
}
```
- `completion` type uses a finite state transducer (FST) for O(1) prefix lookups
- Supports fuzzy matching, boosting by popularity
- Handles 10K+ QPS with a 3-node cluster

**Option 3: PostgreSQL with tsvector + trigram (for small-medium scale)**
```sql
-- Trigram extension for fast LIKE and similarity queries
CREATE EXTENSION pg_trgm;
CREATE INDEX idx_products_name_trgm ON products USING GIN (name gin_trgm_ops);

SELECT name FROM products
WHERE name ILIKE 'lap%'
ORDER BY similarity(name, 'lap') DESC
LIMIT 10;
```

**For interview:** Choose based on scale and complexity:
- < 1M items, moderate traffic → PostgreSQL + trigram + Redis cache
- Millions of items, complex boost logic → Elasticsearch
- Simple string autocomplete, massive scale → Redis sorted sets with prefix indexing

---

**Q10. How would you choose between PostgreSQL and Cassandra for a notification system?**

**Notification system characteristics:**
- High write rate: millions of notifications generated per hour
- Read pattern: "Get unread notifications for user X, newest first" (per-user lookup)
- Retention: keep 30 days, delete older
- Rarely updated after creation

**Analysis:**

| Criterion | PostgreSQL | Cassandra |
|-----------|-----------|----------|
| Write throughput | ~10K/s per node | ~100K/s per node |
| Read pattern | Index on user_id + created_at (works well) | Partition by user_id, cluster by created_at (ideal) |
| Deletion / TTL | VACUUM needed, no native TTL | Native TTL per row |
| Cross-user queries | Easy with SQL | Difficult (no cross-partition queries) |
| Operational complexity | Low | Higher |

**Verdict:** Cassandra is a better fit if:
- Write rate > 50K/s
- No need for complex queries (just per-user timeline)
- Need native TTL for automatic 30-day expiry

```sql
-- Cassandra schema
CREATE TABLE notifications (
    user_id     UUID,
    created_at  TIMEUUID,   -- time-ordered
    type        TEXT,
    payload     TEXT,
    read        BOOLEAN,
    PRIMARY KEY (user_id, created_at)
) WITH CLUSTERING ORDER BY (created_at DESC)
  AND default_time_to_live = 2592000;  -- 30 days TTL
```

**PostgreSQL is sufficient if:**
- Write rate < 20K/s
- Need to query notifications across users (admin dashboard)
- Team is PostgreSQL-familiar

---

**Q11. How would you design the database layer for a recommendation engine?**

Recommendation data has diverse storage needs:

**1. User-item interaction log (raw events)**
```
Database: Cassandra or ClickHouse
Schema: (user_id, item_id, event_type, event_time, context)
Why: Append-only, high write rate, time-series queries
```

**2. User feature vectors (for ML inference)**
```
Database: Redis (low-latency feature serving) + S3/Parquet (training data)
Schema: user_id → {embedding: [0.12, 0.84, ...], last_updated: ts}
Why: Sub-millisecond feature fetch during serving
```

**3. Item feature vectors**
```
Database: Redis or Elasticsearch (with dense_vector for ANN search)
Use case: Approximate Nearest Neighbor (ANN) search for similar items
Alternatives: Pinecone, Weaviate, pgvector (PostgreSQL extension)
```

**4. Precomputed recommendations cache**
```
Database: Redis or DynamoDB
Schema: user_id → [item_id_1, item_id_2, ..., item_id_20]  (TTL: 1 hour)
Why: Avoid recomputing recommendations for every request
```

**5. A/B test configuration and results**
```
Database: PostgreSQL
Why: Relational, small scale, complex queries for experiment analysis
```

**pgvector for similarity search:**
```sql
-- PostgreSQL with pgvector extension
CREATE EXTENSION vector;
CREATE TABLE items (id BIGINT, embedding vector(128));
CREATE INDEX ON items USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Find 10 most similar items to a query vector
SELECT id, 1 - (embedding <=> '[0.1, 0.2, ...]') AS similarity
FROM items
ORDER BY embedding <=> '[0.1, 0.2, ...]'
LIMIT 10;
```

---

**Q12. When should you use a graph database vs a relational database for relationship-heavy data?**

**Use a relational database when:**
- Relationships have a fixed, known depth (e.g., user → orders → items)
- Graph traversal depth is limited (1-2 hops)
- You need complex filtering across both nodes and edges
- Team knows SQL well and graph scale is moderate

```sql
-- Relational: works fine for shallow graphs
SELECT DISTINCT u2.id, u2.name
FROM users u1
JOIN follows f1 ON f1.follower_id = u1.id
JOIN users u2 ON u2.id = f1.followed_id
WHERE u1.id = $user_id;
```

**Use a graph database when:**
- Queries require variable-depth traversal ("friends of friends", paths of arbitrary length)
- Graph structure is the primary access pattern
- Need to find shortest paths, detect cycles, compute centrality
- Examples: fraud detection networks, knowledge graphs, social networks with recursive queries

```cypher
// Neo4j: find all friends within 3 hops
MATCH (u:User {id: $user_id})-[:FOLLOWS*1..3]->(friend:User)
WHERE NOT (u)-[:FOLLOWS]->(friend)  -- not already following
RETURN friend, count(*) AS mutual_connections
ORDER BY mutual_connections DESC
LIMIT 20;
```

**Hybrid approach (common in practice):**
- PostgreSQL with closure table or `WITH RECURSIVE` for shallow traversals
- Neo4j for graph-specific queries (fraud ring detection, recommendation paths)
- Keep PostgreSQL as system of record, sync to Neo4j for graph queries

**Performance comparison:**
| Hops | PostgreSQL recursive | Neo4j |
|------|---------------------|-------|
| 2 | ~10ms | ~5ms |
| 4 | ~500ms | ~15ms |
| 6 | ~30s+ | ~50ms |
Graph DBs win decisively at 4+ hops of traversal.

---

**Q13. How would you design the database layer for a multi-region global application?**

**Challenge:** Users in US, EU, and Asia. Need low-latency reads and writes for each region.

**Approach 1: Active-passive (primary in one region)**
```
US-EAST (primary write) → EU (read replica) → APAC (read replica)
Writes: all go to US-EAST → latency for EU/APAC writers
Reads: served locally (good)
RPO: ~seconds, RTO: ~minutes (promote replica)
Use case: write-light app, most users in US
```

**Approach 2: Multi-primary with conflict resolution**
```
US-EAST (primary) ←→ EU-WEST (primary) ←→ APAC (primary)
Writes: go to nearest region
Conflicts: resolved by LWW (last-write-wins) or application logic
Tools: Aurora Global Database, CockroachDB, YugabyteDB, Cassandra multi-DC
Use case: global writes, eventual consistency acceptable
```

**Approach 3: Geo-partitioned data**
```
User data: partitioned by home_region
EU users' data lives in EU-WEST PostgreSQL
US users' data lives in US-EAST PostgreSQL
Cross-region: routing layer (app or proxy)
Tools: Vitess geo-partitioning, CockroachDB PARTITION BY
GDPR benefit: EU user data never leaves EU
```

**CockroachDB for truly global SQL:**
```sql
-- Assign table rows to specific regions
ALTER TABLE users CONFIGURE ZONE USING
  constraints = '[+region=us-east1]';

-- Multi-region table (data replicated everywhere, writes go to home region)
ALTER TABLE global_config SET LOCALITY GLOBAL;
```

**Latency targets by approach:**
| Approach | Read latency | Write latency | Consistency |
|----------|-------------|---------------|-------------|
| Active-passive | 1-5ms (local) | 100-300ms (cross-region) | Strong |
| Multi-primary | 1-5ms | 1-5ms (local) | Eventual |
| Geo-partition | 1-5ms (own region) | 1-5ms (own region) | Strong (per region) |
| CockroachDB multi-region | 1-5ms (GLOBAL table) | 5-10ms (follower reads) | Serializable |

---

**Q14. How would you design the database layer for a booking/reservation system (hotel rooms, airline seats)?**

**Core challenge:** Prevent double-booking while handling high concurrency.

**Schema:**
```sql
CREATE TABLE rooms (
    id          BIGSERIAL PRIMARY KEY,
    hotel_id    BIGINT NOT NULL,
    room_number TEXT NOT NULL,
    room_type   TEXT NOT NULL,
    price       NUMERIC(10,2) NOT NULL
);

CREATE TABLE bookings (
    id          BIGSERIAL PRIMARY KEY,
    room_id     BIGINT NOT NULL REFERENCES rooms(id),
    user_id     BIGINT NOT NULL,
    check_in    DATE NOT NULL,
    check_out   DATE NOT NULL,
    status      TEXT NOT NULL DEFAULT 'confirmed',
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    -- Prevent overlapping bookings for the same room
    EXCLUDE USING gist (
        room_id WITH =,
        daterange(check_in, check_out, '[)') WITH &&
    ) WHERE (status != 'cancelled')
);
```

**The `EXCLUDE` constraint uses PostgreSQL's range exclusion operator:** prevents any two non-cancelled bookings for the same room with overlapping date ranges.

**Alternative with explicit lock:**
```sql
-- Check availability and book atomically
BEGIN;
  -- Lock room to prevent concurrent booking
  SELECT id FROM rooms WHERE id = $room_id FOR UPDATE;
  
  -- Check for overlaps
  SELECT count(*) FROM bookings
  WHERE room_id = $room_id
    AND status != 'cancelled'
    AND daterange(check_in, check_out, '[)') && daterange($in, $out, '[)');
  
  -- If 0 overlaps, insert booking
  INSERT INTO bookings (room_id, user_id, check_in, check_out)
  VALUES ($room_id, $user_id, $in, $out);
COMMIT;
```

**Caching availability:**
```
Redis: store available room IDs per {hotel_id, date} as a SET
Read: check Redis first (fast availability lookup)
Write: invalidate Redis on every booking/cancellation
TTL: 60 seconds (acceptable staleness for display; DB is authoritative for booking)
```

**Database choice:** PostgreSQL is ideal — ACID, range exclusion constraints, excellent for relational data. Not Cassandra (no range exclusion, no transactions for multiple rows).

---

**Q15. When would you use a columnar database vs a row-oriented database?**

**Row-oriented (PostgreSQL, MySQL, Oracle):**
- Stores all columns of a row together on disk
- Fast for: INSERT, UPDATE, DELETE, SELECT with few rows by primary key
- Reads entire row even when only 2 columns needed

**Columnar (ClickHouse, Redshift, BigQuery, Parquet files):**
- Stores all values of each column together on disk
- Fast for: SELECT with aggregations over millions/billions of rows
- Reads only the needed columns (I/O reduction: 10 columns of 100 = 10× less I/O)
- Excellent compression (same-type values compress together)

**Decision matrix:**

| Use case | Row-oriented | Columnar |
|---------|-------------|---------|
| OLTP (web app, transactions) | ✅ | ❌ |
| User profile lookup by ID | ✅ | ❌ |
| Analytics: avg revenue by country by month | ❌ slow | ✅ |
| Aggregations on billions of rows | ❌ very slow | ✅ |
| Frequent UPDATE/DELETE | ✅ | ❌ (expensive rewrite) |
| Time-series aggregation | ❌ | ✅ |

**Real-world example:**
```sql
-- Query on 1B row events table, selecting 3 of 50 columns
SELECT user_country, event_type, count(*) AS cnt, sum(revenue)
FROM events
WHERE event_time BETWEEN '2024-01-01' AND '2024-01-31'
GROUP BY 1, 2;

-- PostgreSQL (row-store): must read all 50 columns per row = ~500GB I/O
-- ClickHouse (column-store): reads only 4 columns = ~40GB I/O = 12× faster
```

**HTAP (Hybrid Transactional/Analytical):**
- Some databases (TiDB, SingleStore, AlloyDB) support both row and columnar storage
- Row store for OLTP, columnar store for analytics — same database

---

## Hard (Q16–Q20)

---

**Q16. Design the database layer for a system like Airbnb — property listings, search, bookings, and payments.**

**Scale assumptions:** 7M listings, 100M users, 500K bookings/day.

**Data stores:**

```
┌─────────────────────────────────────────────────────────────┐
│  PostgreSQL (primary OLTP)                                  │
│  Tables: users, hosts, listings, bookings, payments,        │
│          reviews, messages                                   │
│  Why: ACID (payments), relational JOINs, mature ecosystem   │
├─────────────────────────────────────────────────────────────┤
│  Elasticsearch (search)                                     │
│  Index: listings with geo_point, amenities, price, dates    │
│  Why: Full-text on description, geo bounding box, facets    │
├─────────────────────────────────────────────────────────────┤
│  Redis (caching + availability)                             │
│  Cache: listing details (TTL 5min), search results (TTL 1m) │
│  Calendar: listing_id → set of blocked dates (sorted set)  │
│  Why: < 1ms reads for hot listings                          │
├─────────────────────────────────────────────────────────────┤
│  ClickHouse (analytics)                                     │
│  Tables: booking_events, search_events, revenue_metrics     │
│  Why: Columnar, fast aggregation for host/business reports  │
└─────────────────────────────────────────────────────────────┘
```

**Critical: Preventing double-booking**
```sql
-- PostgreSQL range exclusion (same as hotel booking example)
ALTER TABLE bookings ADD CONSTRAINT no_overlap
  EXCLUDE USING gist (
    listing_id WITH =,
    daterange(check_in, check_out, '[)') WITH &&
  ) WHERE (status NOT IN ('cancelled', 'declined'));
```

**Search flow:**
```
1. User searches "NYC, 2 adults, Jan 5-10, wifi, < $200/night"
2. Query Elasticsearch:
   - geo_bounding_box filter (NYC coordinates)
   - term filter: amenities contains "wifi"
   - range filter: price_per_night <= 200
   - Exclude listings with blocked dates (pre-indexed availability bitmap)
3. Return listing IDs + ES-scored results
4. Fetch listing details from Redis cache (or PostgreSQL fallback)
5. Display to user
```

**Availability calendar (critical for performance):**
```sql
-- PostgreSQL: availability table indexed for range queries
CREATE TABLE availability (
    listing_id BIGINT NOT NULL,
    date DATE NOT NULL,
    is_available BOOLEAN NOT NULL DEFAULT true,
    price NUMERIC(10,2),
    PRIMARY KEY (listing_id, date)
);
-- Query: check 5-night availability
SELECT count(*) = 5 AS available
FROM availability
WHERE listing_id = $id
  AND date BETWEEN $check_in AND $check_out - 1
  AND is_available = true;
```

**Payment flow:**
```sql
-- Stripe handles the actual payment; PostgreSQL records the outcome
BEGIN;
  -- 1. Create booking in 'pending' state
  INSERT INTO bookings (listing_id, guest_id, check_in, check_out, status, idempotency_key)
  VALUES ($listing_id, $guest_id, $in, $out, 'pending', $idempotency_key);
  
  -- 2. Mark dates unavailable
  UPDATE availability SET is_available = false
  WHERE listing_id = $listing_id
    AND date BETWEEN $in AND $out - 1;
  
  -- 3. On payment success callback: set status = 'confirmed'
COMMIT;
-- On payment failure: rollback (dates remain available)
```

---

**Q17. Design the database layer for a YouTube-like video platform.**

**Scale:** 500 hours of video uploaded/minute, 1B active users, 1B+ daily views.

**Data stores:**

| Data | Store | Reason |
|------|-------|--------|
| Video metadata (title, description, tags, channel) | **PostgreSQL** | Relational, moderate write rate |
| Video files | **Object storage (S3)** | Binary blobs, not a DB concern |
| View counts, likes (approximate) | **Redis** (INCR) → async flush to PostgreSQL | Atomic counters, 1B+ events/day |
| Video search | **Elasticsearch** | Full-text on title/description/tags |
| Comments | **PostgreSQL** or **Cassandra** | PG for small-medium; Cassandra if 1B+ comments |
| Recommendations | **Redis** (precomputed per user) + **ML store** | Low-latency serving |
| User subscriptions | **PostgreSQL** or **Cassandra** | High cardinality but manageable |
| Watch history | **Cassandra** | Write-heavy, time-ordered per user |
| Analytics / trending | **ClickHouse** | Aggregate view/like events |
| Video processing queue | **Kafka** → **PostgreSQL** (job table) | Track transcoding jobs |

**View count at scale:**

```
Problem: 1B+ views/day = ~12,000 views/sec → can't UPDATE PostgreSQL per view

Solution:
1. Redis: INCR video:123:views (atomic, in-memory)
2. Background job every 30s: read Redis counts, batch UPDATE PostgreSQL
3. PostgreSQL stores "official" count (slightly delayed)
4. Client reads from Redis for real-time display, PostgreSQL for search ranking
```

**Video search indexing:**
```json
{
  "video_id": "abc123",
  "title": "Learn PostgreSQL in 1 hour",
  "description": "...",
  "tags": ["postgresql", "database", "tutorial"],
  "channel_id": "ch_456",
  "view_count": 1234567,
  "upload_date": "2024-01-15",
  "duration_seconds": 3600
}
```

**Comment threading:**
```sql
-- PostgreSQL: adjacency list for comment replies
CREATE TABLE comments (
    id          BIGSERIAL PRIMARY KEY,
    video_id    BIGINT NOT NULL,
    parent_id   BIGINT REFERENCES comments(id),  -- NULL = top-level
    user_id     BIGINT NOT NULL,
    content     TEXT NOT NULL,
    likes       INT DEFAULT 0,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_comments_video_parent ON comments(video_id, parent_id, created_at DESC);
-- Fetch top-level: WHERE video_id = $id AND parent_id IS NULL
-- Fetch replies: WHERE parent_id = $comment_id
```

---

**Q18. How would you design the database layer for a distributed rate limiter service?**

**Requirements:** Rate limit API calls per user (e.g., 1000 req/min). Must be accurate, fast (< 1ms overhead), and work across multiple application servers.

**Option 1: Redis with sliding window log**
```redis
# Key: rate_limit:{user_id}:{minute_bucket}
# Store request timestamps in sorted set
ZADD rate_limit:user123 {timestamp_ms} {request_id}
ZREMRANGEBYSCORE rate_limit:user123 0 {now_ms - 60000}  -- remove > 1 min old
ZCARD rate_limit:user123  -- current count
EXPIRE rate_limit:user123 120  -- auto-expire key
```

**Option 2: Redis with token bucket (Lua script for atomicity)**
```lua
-- Lua script runs atomically on Redis
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])  -- tokens per second
local now = tonumber(ARGV[3])

local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
local tokens = tonumber(bucket[1]) or capacity
local last_refill = tonumber(bucket[2]) or now

-- Refill tokens based on elapsed time
local elapsed = now - last_refill
local new_tokens = math.min(capacity, tokens + elapsed * rate)

if new_tokens >= 1 then
    redis.call('HMSET', key, 'tokens', new_tokens - 1, 'last_refill', now)
    redis.call('EXPIRE', key, math.ceil(capacity / rate) * 2)
    return 1  -- allowed
else
    return 0  -- rejected
end
```

**Option 3: PostgreSQL with advisory locks (for moderate scale)**
```sql
-- Not suitable for high-frequency rate limiting (too slow)
-- But useful for limiting heavy operations (exports, bulk downloads)
WITH rate_check AS (
    INSERT INTO rate_limit_log (user_id, request_at)
    VALUES ($user_id, NOW())
    RETURNING 1
)
SELECT count(*) < $limit AS allowed
FROM rate_limit_log
WHERE user_id = $user_id
  AND request_at > NOW() - INTERVAL '1 minute';
```

**Production recommendation:**
- **Redis** for real-time API rate limiting (sub-millisecond, atomic Lua scripts)
- **Redis Cluster** for distribution — shard by user_id hash
- **Database (PostgreSQL)** for persistent rate limit configuration (per-user/tier limits) and quota auditing
- **Fallback:** If Redis is unavailable, fail open (allow requests) and alert — never fail closed on rate limiter outage

---

**Q19. How would you design the database layer for a real-time collaborative document editor like Google Docs?**

**Challenges:**
- Multiple users editing the same document simultaneously
- Operational Transformation (OT) or CRDTs for conflict resolution
- Low-latency sync (< 100ms)
- Full revision history

**Data stores:**

```
┌─────────────────────────────────────────────────────────────┐
│  Redis (real-time collaboration state)                      │
│  Active sessions per document: hash of {user_id: cursor}   │
│  In-flight operations buffer: list of recent ops (60s TTL)  │
│  Presence pub/sub: who is online in each document           │
├─────────────────────────────────────────────────────────────┤
│  PostgreSQL (persistent storage)                            │
│  documents: (id, title, owner_id, created_at)              │
│  document_versions: (id, doc_id, content_snapshot, ver_num) │
│  operations: (id, doc_id, user_id, op_data, seq_num, ts)   │
│  permissions: (doc_id, user_id, role)                       │
├─────────────────────────────────────────────────────────────┤
│  Object storage (S3)                                        │
│  Full document snapshots every 100 operations               │
│  Reduces replay time for loading old documents              │
└─────────────────────────────────────────────────────────────┘
```

**Operation ordering (sequence numbers):**
```sql
-- Operations must be ordered globally per document
CREATE TABLE operations (
    id         BIGSERIAL PRIMARY KEY,
    doc_id     BIGINT NOT NULL,
    user_id    BIGINT NOT NULL,
    seq_num    BIGINT NOT NULL,       -- document-global sequence
    op_type    TEXT NOT NULL,         -- 'insert' | 'delete' | 'format'
    op_data    JSONB NOT NULL,        -- {pos: 42, text: "hello"}
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (doc_id, seq_num)          -- enforces ordering
);

-- Atomic sequence number assignment
WITH seq AS (
    SELECT COALESCE(MAX(seq_num), 0) + 1 AS next_seq
    FROM operations WHERE doc_id = $doc_id
    FOR UPDATE  -- prevents concurrent sequence gap
)
INSERT INTO operations (doc_id, user_id, seq_num, op_type, op_data)
SELECT $doc_id, $user_id, next_seq, $op_type, $op_data FROM seq
RETURNING seq_num;
```

**Snapshot strategy:**
```sql
-- Take snapshot every 100 operations to limit replay time on load
INSERT INTO document_versions (doc_id, content_snapshot, version_num, based_on_seq)
SELECT $doc_id, $full_content, version_num + 1, $current_seq
FROM document_versions
WHERE doc_id = $doc_id
ORDER BY version_num DESC LIMIT 1;
```

**Loading a document:**
1. Find latest snapshot (version) for document
2. Load all operations since that snapshot's seq_num
3. Replay operations to reconstruct current state
4. Subscribe to Redis pub/sub channel for live updates

---

**Q20. Given an existing monolithic PostgreSQL database for a ride-sharing app hitting limits, how do you decompose it into a multi-database architecture without downtime?**

**Current state:**
- Single PostgreSQL: rides, drivers, payments, notifications, analytics
- 50M rides/month, growing 20% MoM
- Bottlenecks: analytics queries slowing OLTP, payment table hot, notification writes causing lock contention

**Step 1: Identify decomposition boundaries**
```
Current monolith tables:
├── users, drivers (core identity)        → keep in PostgreSQL
├── rides, ride_events                    → keep in PostgreSQL (ACID needed)
├── payments, ledger_entries              → separate PostgreSQL (isolated ACID)
├── notifications                         → migrate to Cassandra
├── driver_locations (real-time)          → migrate to Redis
└── analytics events                      → migrate to ClickHouse
```

**Step 2: Extract analytics first (lowest risk)**
```
1. Set up ClickHouse cluster
2. Deploy Debezium CDC → Kafka → ClickHouse Kafka engine
3. Backfill historical data to ClickHouse
4. Switch analytics dashboards to read from ClickHouse
5. Drop analytics indexes from PostgreSQL (immediate relief)
```

**Step 3: Extract real-time driver locations**
```
1. Deploy Redis Cluster with Redis GEO
2. Application writes: dual-write to PostgreSQL AND Redis
3. Application reads: read from Redis only
4. After 2 weeks stable: stop writing to PostgreSQL driver_locations
5. Archive old data, drop PostgreSQL columns
```

**Step 4: Extract notifications**
```
1. Deploy Cassandra cluster with same schema
2. Dual-write: writes go to both PostgreSQL AND Cassandra
3. Reads: gradually shift traffic to Cassandra (canary → 10% → 50% → 100%)
4. Monitor: compare row counts and query results between databases
5. Cut over: 100% to Cassandra, deprecate PostgreSQL notifications tables
```

**Step 5: Extract payments (highest risk — requires careful planning)**
```
1. Deploy separate PostgreSQL for payments (same version)
2. Create payments service with its own DB connection
3. Migrate data with pg_dump --table=payments,ledger_entries
4. Set up logical replication slot from monolith → payments DB (sync)
5. Application: route payment writes to payments service
6. After replication is caught up and stable (1 week): stop replication
7. Drop payments tables from monolith
```

**Zero-downtime key principles:**
- Never delete from the source until you are 100% sure the destination is correct
- Use feature flags to control read/write routing
- Always dual-write during migration window
- Monitor row counts on both sides
- Keep rollback plan (re-enable reads from old DB) for minimum 2 weeks

**Result after decomposition:**
```
PostgreSQL (core): users, drivers, rides — leaner, faster
PostgreSQL (payments): isolated, dedicated resources, compliant
Cassandra: notifications at scale with TTL
Redis: real-time driver locations < 1ms
ClickHouse: analytics without competing with OLTP
```

---

**Quick Reference**

| System | Primary DB | Why | Supporting stores |
|--------|-----------|-----|------------------|
| E-commerce | PostgreSQL | ACID for inventory/orders | Redis (cart), Elasticsearch (search), ClickHouse (analytics) |
| Social media | PostgreSQL + Cassandra | PG for profiles, Cassandra for posts/feeds | Redis (timeline cache), Elasticsearch (search) |
| Chat app | Cassandra | Write-heavy, time-ordered, TTL | Redis (presence), Elasticsearch (search) |
| Payments | PostgreSQL | ACID, double-entry ledger | Redis (idempotency cache), ClickHouse (reports) |
| IoT / time-series | TimescaleDB or InfluxDB | Native time-series partitioning | Kafka (ingest buffer) |
| Ride-sharing | PostgreSQL + Redis | ACID for trips, Redis GEO for locations | Cassandra (history), ClickHouse (analytics) |
| Video platform | PostgreSQL + S3 | Metadata relational, files in object store | Redis (counts), Elasticsearch (search) |
| URL shortener | Cassandra or DynamoDB | High read throughput, simple key lookup | Redis (hot URL cache) |

| Pattern | When to apply |
|---------|--------------|
| Polyglot persistence | Different data has genuinely different requirements (not just "use more tech") |
| Read replicas | Read-heavy, write-light workloads; reporting queries |
| CDC + ClickHouse | Offload analytics from OLTP without application changes |
| Redis cache + DB write-through | Hot read path, acceptable staleness |
| Dual-write migration | Zero-downtime database extraction |
