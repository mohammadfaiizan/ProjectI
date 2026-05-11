# System Design: Twitter / X Feed

---

## 1. Problem Statement

Design Twitter's core functionality: users can post short text messages (tweets, up to 280 characters), follow other users, see a real-time feed of tweets from people they follow, retweet, use hashtags, and @mention other users. The system must handle hundreds of millions of daily active users with extremely high read throughput.

---

## 2. Clarifying Questions to Ask

- What is the expected DAU and tweet volume?
- Should the feed be strictly chronological or ranked?
- What is the maximum number of followers? (some accounts: 100M+)
- Do we need to support media (images/video) in tweets?
- Is retweet a direct copy or a reference to the original?
- Do we need DMs (Direct Messages)?
- How does search work — real-time or near-real-time?
- Should trending topics be global or personalized/regional?
- Do we need quote tweets (tweet within a tweet)?
- What is acceptable feed staleness? (seconds? minutes?)

---

## 3. Functional Requirements

1. Users can post tweets (up to 280 characters).
2. Users can follow/unfollow other users.
3. Users see a home timeline of tweets from accounts they follow.
4. Users can retweet (reshare) any tweet.
5. Users can @mention other users in tweets.
6. Users can use hashtags in tweets.
7. View trending topics (global and by region).
8. Search tweets by keyword or hashtag.
9. View another user's profile and their tweets.
10. Like (heart) any tweet.

---

## 4. Non-Functional Requirements

- **Availability**: 99.99% — timeline reads must always succeed
- **Latency**: Home timeline P99 < 300ms; tweet post P95 < 500ms
- **Consistency**: Eventual consistency acceptable — new tweet visible within 5 seconds
- **Scalability**: 300M DAU; 600 tweets/second; 600K timeline reads/second
- **Throughput**: Peak 1,200 tweets/sec; peak 1.2M reads/sec
- **Durability**: Tweets must never be lost once posted
- **Global**: Multi-region deployment with GeoDNS routing

---

## 5. Capacity Estimation

### Users & Activity
- DAU: 300M users
- Average tweets per user per day: 2
- Total tweets per day: 300M * 2 = 600M tweets/day
- Tweet write QPS: 600M / 86,400 = ~7,000 writes/sec
- Peak tweet write QPS: ~14,000 writes/sec

### Timeline Reads
- Average feed checks per DAU per day: 8 (every ~2 hours)
- Total feed reads per day: 300M * 8 = 2.4B reads/day
- Feed read QPS: 2.4B / 86,400 = ~27,800 reads/sec
- Peak feed read QPS (3x): ~83,400 reads/sec
- Timeline cache hit rate target: 95%+ (most timelines are pre-computed in Redis)

### Storage
- Average tweet size: 280 chars UTF-8 + metadata = ~500 bytes
- Tweets per day: 600M * 500 bytes = 300 GB/day
- Tweets per year: 300 GB * 365 = ~110 TB/year
- Index storage (Elasticsearch): ~3x raw = 330 TB/year
- Media: tweets with images/video stored separately in S3/CDN

### Bandwidth
- Write: 14,000 req/s * 500B = 7 MB/s
- Read: 83,400 req/s * ~10KB (20 tweets) = ~834 MB/s
- CDN offloads ~80% of media traffic

### Fan-out Math
- Average user follows 200 accounts
- Average non-celebrity user has 200 followers
- Per tweet fan-out: 200 Redis writes = 14,000 * 200 = 2.8M Redis ops/sec
- Redis can handle ~1M ops/sec per node → need 3+ nodes for fan-out

---

## 6. High-Level Architecture

```
              [Client]
                 |
        [GeoDNS / CDN]
                 |
        [Load Balancer]
          /    |     \
    [API]   [API]   [API]   (stateless API servers)
      |       |
   [Tweet Service]  [Timeline Service]
      |                    |
   [Fanout Service]    [Cache Layer]
      |                (Redis Cluster - user timelines)
   [Message Queue]
   (Kafka)
      |
   +--+------------------+
   |                     |
[Tweet Store]     [Search Indexer]
(Cassandra)        (Elasticsearch)
      |
[Follow Service]
(Redis + PostgreSQL)

[Trending Service] <-- reads from Kafka tweet stream
```

### Tweet Post Flow
```
1. POST /tweets  → API Server
2. API Server → Tweet Service (validate, store)
3. Tweet stored in Cassandra (tweet_id, user_id, content, timestamp)
4. Kafka event: { tweet_id, user_id, content, followers[] }
5. Fanout Service consumes Kafka:
   a. For each follower (non-celebrity): ZADD timeline:{follower_id} ts tweet_id
   b. For celebrities: skip push (followers pull on read)
6. Search Indexer consumes Kafka: index tweet in Elasticsearch
7. Trending Service: ZINCRBY trending:global 1 hashtag
```

### Timeline Read Flow
```
1. GET /timeline → API Server → Timeline Service
2. Timeline Service: ZREVRANGE timeline:{user_id} 0 19 (Redis)
3. Cache miss: merge followees' recent tweets in real-time
4. Fetch tweet details by tweet_id (Redis or Cassandra)
5. Return enriched timeline
```

---

## 7. Component Deep-Dive

### 7.1 Tweet Storage (Cassandra)

Why Cassandra for tweets?
- Write-heavy (millions of tweets/sec at Twitter scale)
- Tweets are immutable once posted (rarely updated/deleted)
- Wide-column model is perfect: partition by user_id, cluster by tweet_id
- Linear horizontal scalability (add nodes, no downtime)
- Eventual consistency acceptable for tweet reads
- No complex joins required (tweet lookup is always by tweet_id or user_id)

Cassandra data model:
```
Table: tweets_by_user
  Partition key: user_id
  Clustering key: tweet_id DESC (newest first)
  Columns: content, media_url, retweet_of, created_at, like_count

Table: tweets_by_id
  Partition key: tweet_id
  Columns: user_id, content, media_url, retweet_of, created_at
```

### 7.2 Snowflake ID Generator

Twitter's Snowflake generates 64-bit IDs that are:
- Globally unique across all machines
- Time-sortable (higher ID = newer tweet)
- Generated without coordination

**Bit layout**:
```
| 1 bit (0) | 41 bits (timestamp ms) | 10 bits (machine ID) | 12 bits (sequence) |
```
- 41 bits: ms since custom epoch (Jan 1, 2010) → ~69 years range
- 10 bits: up to 1,024 machines
- 12 bits: 4,096 IDs per ms per machine
- Max throughput: 4,096 * 1,024 = 4.2M IDs/sec system-wide

Benefit for timelines: sorting tweet_ids gives chronological order without a separate timestamp column.

### 7.3 Fan-out Service (Hybrid Push/Pull)

**Regular users (< 1M followers)**:
- Fan-out on write: push tweet_id to all follower timelines
- Timeline stored as Redis Sorted Set: `timeline:{user_id}` → score=tweet_id (snowflake is time-sortable)
- Max timeline size: 800 tweet_ids (trim oldest on insert)
- This pre-computation means feed reads are O(1)

**Celebrity users (≥ 1M followers)**:
- Fan-out on read: followers pull celebrity's tweets at read time
- Celebrity's tweet list: `user_tweets:{celebrity_id}` → Sorted Set
- Feed read merges pre-computed timeline + celebrity tweet sets
- On merge: N-way sorted merge of K celebrity tweet lists + 1 pre-computed list

**Fanout Service Architecture**:
- Reads from Kafka topic `tweets`
- Looks up follower list from Follow Service
- Batch Redis ZADD operations (pipeline 100 commands per round trip)
- Retries on failure with exponential backoff

### 7.4 Trending Topics

**Algorithm: Sliding Window Hashtag Counter**

- Every tweet event increments hashtag counters in Redis
- Key design: `trending:{hashtag}:{hour_bucket}` → count
- Sliding window: sum counts across last 3 hour buckets
- Top-K query: ZREVRANGE trending:global 0 9 (top 10 hashtags)
- Refresh every 15 minutes via scheduled job
- Regional trending: separate sorted sets per region

**Top-K algorithm**:
1. Every minute: aggregate hashtag counts from Kafka stream
2. Maintain a min-heap of size K (K=50 trending topics)
3. If a hashtag count exceeds heap minimum, replace it
4. Expose current top-K via API

**Decay factor**: older mentions count less than recent ones
- Score = count / (1 + hours_since_first_mention)

### 7.5 Search Service (Elasticsearch)

- Index: `tweets` with fields: text, user_id, created_at, hashtags[], mentions[]
- Near-real-time indexing: Kafka consumer → ES bulk index every 5 seconds
- Full-text search with relevance ranking
- Hashtag search: term query on hashtags field
- Recent tweets: filter by created_at + sort by recency
- Autocomplete: edge n-gram for @mention and hashtag suggestions

### 7.6 Follow Service

- **PostgreSQL**: source of truth for follow relationships
- **Redis**: `followers:{user_id}` sorted set (for fanout service)
  - Score = follow timestamp (allows "newest followers" queries)
  - Loaded lazily: cached when user comes online

### 7.7 Rate Limiting

- Per user: 300 tweets/day, 1000 API calls/hour
- Per IP: 100 requests/15 minutes (for unauthenticated)
- Algorithm: Sliding window counter in Redis
- Token bucket for burst allowance

---

## 8. Database Design

### Cassandra: tweets
```cql
CREATE TABLE tweets (
    tweet_id    BIGINT,           -- Snowflake ID
    user_id     BIGINT,
    content     TEXT,
    media_url   TEXT,
    retweet_of  BIGINT,           -- NULL if original tweet
    like_count  COUNTER,
    created_at  TIMESTAMP,
    PRIMARY KEY (tweet_id)
);

CREATE TABLE user_tweets (
    user_id     BIGINT,
    tweet_id    BIGINT,
    PRIMARY KEY (user_id, tweet_id)
) WITH CLUSTERING ORDER BY (tweet_id DESC);
```

### PostgreSQL: users
```sql
CREATE TABLE users (
    id              BIGSERIAL PRIMARY KEY,
    username        VARCHAR(50) UNIQUE NOT NULL,
    display_name    VARCHAR(100),
    bio             TEXT,
    follower_count  BIGINT DEFAULT 0,
    following_count BIGINT DEFAULT 0,
    tweet_count     BIGINT DEFAULT 0,
    verified        BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE TABLE follows (
    follower_id     BIGINT REFERENCES users(id),
    followee_id     BIGINT REFERENCES users(id),
    created_at      TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (follower_id, followee_id),
    INDEX idx_followee_id (followee_id)
);
```

### PostgreSQL: likes
```sql
CREATE TABLE likes (
    tweet_id        BIGINT NOT NULL,
    user_id         BIGINT NOT NULL REFERENCES users(id),
    created_at      TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (tweet_id, user_id)
);
```

### Redis Data Structures
```
timeline:{user_id}      → Sorted Set  { tweet_id: tweet_id }   (snowflake sorts by time)
user_tweets:{user_id}   → Sorted Set  { tweet_id: tweet_id }   (celebrity posts)
followers:{user_id}     → Sorted Set  { follower_id: follow_ts }
trending:global         → Sorted Set  { hashtag: score }
tweet:{tweet_id}        → Hash        { user_id, content, like_count, ... }
```

---

## 9. API Design

### Post Tweet
```
POST /api/v1/tweets
Authorization: Bearer {token}

Request:
{
  "content": "Hello world! #firsttweet @friend",
  "media_id": "optional-media-id",
  "reply_to": null
}

Response 201:
{
  "tweet_id": "1234567890123456789",
  "content": "Hello world! #firsttweet @friend",
  "created_at": "2024-01-15T10:30:00Z",
  "user": { "id": 123, "username": "alice" }
}
```

### Get Home Timeline
```
GET /api/v1/timeline?max_id=1234&limit=20
Authorization: Bearer {token}

Response 200:
{
  "tweets": [
    {
      "tweet_id": "123...",
      "user": { "id": 456, "username": "bob", "verified": false },
      "content": "...",
      "like_count": 1523,
      "retweet_count": 234,
      "liked_by_viewer": false,
      "retweeted_by_viewer": false,
      "created_at": "..."
    }
  ],
  "next_max_id": "...",
  "has_more": true
}
```

### Retweet
```
POST /api/v1/tweets/{tweet_id}/retweet
Authorization: Bearer {token}

Response 200:
{
  "retweet_id": "...",
  "original_tweet_id": "...",
  "retweeted": true
}
```

### Get Trending Topics
```
GET /api/v1/trending?region=US&limit=10

Response 200:
{
  "trends": [
    { "hashtag": "#SuperBowl", "tweet_count": 1250000, "trend_score": 9.8 },
    { "hashtag": "#AI", "tweet_count": 890000, "trend_score": 8.2 }
  ],
  "as_of": "2024-01-15T10:30:00Z"
}
```

### Search
```
GET /api/v1/search?q=hello+world&type=tweets&limit=20

Response 200:
{
  "tweets": [...],
  "users": [...],
  "hashtags": [...]
}
```

---

## 10. Scalability & Bottlenecks

### Bottleneck 1: Fan-out for Celebrities
- A user with 100M followers posting → 100M Redis writes
- Solution: Skip push for celebrities, pull on read
- Celebrity threshold: 1M followers
- This is Twitter's actual Hybrid Fan-out approach

### Bottleneck 2: Timeline Redis Memory
- 300M users * 800 tweet_ids * 8 bytes = ~1.9 TB just for timeline IDs
- Solution: Redis Cluster with 20+ nodes; LRU eviction for inactive users
- Cold user timelines rebuilt from Cassandra on demand

### Bottleneck 3: Cassandra Write Throughput
- 14,000 tweets/sec write QPS is manageable for Cassandra
- Each tweet requires 2 Cassandra writes (tweets + user_tweets table)
- Cassandra can handle 100K+ writes/sec per node → 5+ nodes sufficient

### Bottleneck 4: Trending Topic Accuracy
- Real-time counting in Redis is eventually consistent
- Solution: Kafka stream processing with 1-minute micro-batches
- Approximate counting acceptable (HyperLogLog for unique user counts)

### Bottleneck 5: Timeline Merge for Users Following Many Celebrities
- User follows 100 celebrities → 100 Redis sorted set reads + merge
- Solution: Pre-merge celebrity timelines in background for active users
- Use N-way merge (min-heap) for O(K log C) merge complexity

---

## 11. Trade-offs & Design Decisions

### Cassandra vs MySQL for Tweets
- MySQL: familiar, ACID, but doesn't scale horizontally for write-heavy workloads
- Cassandra: eventual consistency, excellent write throughput, natural time-series data model
- Decision: Cassandra for tweets (immutable, high volume, time-series pattern)

### Snowflake IDs vs UUID
- UUID: globally unique but not time-sortable, random, bad for range queries
- Snowflake: time-sortable, can be shard key, reveals tweet order
- Decision: Snowflake — time-sortability is critical for timeline ordering

### Push vs Pull Fan-out
- Push: fast reads, but celebrity writes create thundering herd
- Pull: slower reads, simple writes
- Decision: Hybrid — push for regular users, pull for celebrities

### Real-time vs Batch Trending
- Real-time (stream processing): accurate but complex (Flink/Kafka Streams)
- Batch (refresh every 15 min): simpler, slightly stale but acceptable
- Decision: 15-minute batch refresh with real-time stream for surge detection

### Timeline Max Length: 800 tweets
- More tweets: more Redis memory, slower reads
- Fewer tweets: users who don't check for a week see a truncated feed
- Decision: 800 is Twitter's documented number; rebuild from DB on overflow

---

## 12. Key Interview Talking Points

1. **Snowflake ID**: Time-sortable IDs eliminate the need for a timestamp sort — tweet_ids ARE the timeline ordering.

2. **Hybrid fan-out**: The celebrity problem is the hardest part. Push/pull hybrid is the production answer.

3. **Cassandra for tweets**: Write-heavy, immutable, time-series → Cassandra is a natural fit. Don't use MySQL.

4. **Timeline as Redis Sorted Set**: Key insight — score is tweet_id (Snowflake), gives you time ordering for free.

5. **Trending: sliding window**: Don't use a simple counter. Sliding window or decay functions give more relevant trending topics.

6. **Fan-out QPS math**: 7,000 tweets/sec * 200 followers avg = 1.4M Redis ops/sec. This justifies Redis Cluster.

7. **Timeline rebuild**: When a user's Redis timeline expires or overflows, rebuild from Cassandra `user_tweets` table of all followees.

8. **Kafka for decoupling**: Tweet posting should be fast (write to Cassandra + Kafka). Fan-out and search indexing are async consumers.

9. **Search latency**: Near-real-time (5s lag from tweet to searchable) via Kafka → ES pipeline is acceptable for Twitter.

10. **Cold start problem**: New user following many accounts — rebuild timeline from Cassandra in background, return partial results immediately.
