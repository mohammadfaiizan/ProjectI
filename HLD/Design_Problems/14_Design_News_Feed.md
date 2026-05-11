# System Design: News Feed (Social Media Feed)

---

## 1. Problem Statement & Clarifying Questions

### Problem Statement
Design a social media news feed system (like Facebook Feed, Twitter/X timeline, or Instagram Feed) that generates a personalized, ranked feed of posts for 500 million daily active users with a 100:1 read-to-write ratio.

### Clarifying Questions

**Scale:**
- DAU? *500 million*
- Posts per second? *~300 posts/second (write rate)*
- Average followers per user? *~200, but celebrities can have 100M+*
- Read:write ratio? *100:1 (feed reads dominate)*

**Functionality:**
- Is the feed ranked or chronological? *Ranked (engagement-based + time decay)*
- How deep is the feed paginated? *Up to 500 posts lookback per session*
- Do deleted posts disappear from feeds immediately? *Yes*
- Ads/sponsored posts included? *Yes — inject at positions 3, 7, 15...*
- Muting/blocking users? *Yes*

**Freshness:**
- How fresh should the feed be? *< 60 seconds for most users*
- Can celebrities' posts be slightly delayed? *Yes, up to a few minutes*

---

## 2. Functional & Non-Functional Requirements

### Functional Requirements
1. Create a post (text, images, links)
2. Follow/unfollow users
3. Get personalized feed (paginated with cursor)
4. Like, comment, share posts (engagement signals for ranking)
5. Delete posts (removed from all feeds)
6. Mute/block users (excluded from feed)
7. Inject sponsored posts at defined positions

### Non-Functional Requirements
| Property | Target |
|---|---|
| Availability | 99.99% |
| Feed read latency | < 100ms p99 |
| Post creation latency | < 500ms p99 |
| Feed freshness | < 60 seconds for regular users |
| Scale | 500M DAU, 300 posts/sec write, 30K reads/sec |
| Storage | Posts retained indefinitely; feed cache 7 days |

---

## 3. Capacity Estimation

### Traffic
- Posts written: 300/second = 25.9M/day
- Feed reads: 300 × 100 = 30,000/second (read-dominant)
- Follows per second: ~500/second

### Storage
- Post size: ~1 KB average (text + metadata)
- 300 posts/sec × 86400 sec = 25.9M posts/day × 1 KB = **25.9 GB/day**
- After 5 years: **~47 TB** of posts (manageable with tiered storage)
- Feed cache (Redis): 500M users × 200 post IDs × 8 bytes = **800 GB active feed cache**
  - Only cache feeds of active users (DAU); cold users' feeds are pull-generated

### Follow Graph
- 500M users × 200 avg follows × 8 bytes = **800 GB** adjacency list
- Stored in PostgreSQL + denormalized into Redis sets for hot users

---

## 4. High-Level Architecture

```
              ┌──────────────────────────────────────────────────┐
              │                   Clients                         │
              │          (iOS / Android / Web)                   │
              └──────────────────┬───────────────────────────────┘
                                 │
              ┌──────────────────▼───────────────────────────────┐
              │               API Gateway                         │
              │       (Auth, Rate Limiting, Routing)             │
              └────┬────────────────────────┬────────────────────┘
                   │                        │
       ┌───────────▼──────────┐  ┌──────────▼────────────────────┐
       │   Post Service        │  │   Feed Service                │
       │   create, delete,     │  │   get_feed (paginated)        │
       │   like, comment       │  │   cursor-based pagination     │
       └───────────┬──────────┘  └──────────┬────────────────────┘
                   │                        │
       ┌───────────▼──────────┐  ┌──────────▼───────────────────┐
       │   Fan-Out Service     │  │   Feed Cache (Redis)         │
       │   Kafka consumer      │  │   Sorted sets per user_id    │
       │   push to followers   │  │   score = rank_score         │
       │   (hybrid push/pull)  │  │   member = post_id           │
       └───────────────────────┘  └──────────────────────────────┘
                   │
       ┌───────────▼──────────────────────────────────────────────┐
       │                     Kafka                                 │
       │  Topics: post.created, post.deleted, user.followed       │
       └───────────────────────────────────────────────────────────┘

       ┌──────────────────────────────────────────────────────────┐
       │                     Data Layer                            │
       │  ┌──────────────┐  ┌─────────────────┐  ┌────────────┐  │
       │  │ PostgreSQL   │  │ Redis Cluster   │  │ Cassandra  │  │
       │  │ (posts,      │  │ (feed timelines │  │ (follow    │  │
       │  │  users,      │  │  follow graph   │  │  graph     │  │
       │  │  follows)    │  │  hot post cache)│  │  fallback) │  │
       │  └──────────────┘  └─────────────────┘  └────────────┘  │
       └──────────────────────────────────────────────────────────┘
```

---

## 5. Component Deep-Dive

### 5.1 Feed Generation Strategies

#### Strategy 1: Push (Fan-out on Write) — Precomputed Feeds
**How it works:**
- When user A posts → Fan-out service finds all of A's followers
- Pushes the post_id to each follower's feed timeline in Redis
- Feed read = simple sorted set range query (extremely fast)

**Pros:** Feed reads are O(1) from Redis cache
**Cons:** Celebrity with 100M followers → 100M Redis writes per post; write amplification

**When to use:** Regular users with < 5K followers

#### Strategy 2: Pull (Fan-out on Read) — Compute at Read Time
**How it works:**
- When user loads feed → fetch list of followed users → query their recent posts → merge and rank in memory

**Pros:** No write amplification; celebrities' posts are handled gracefully
**Cons:** Read-time fan-in can be expensive; latency proportional to follow count

**When to use:** Celebrity accounts (> 5K followers, configurable threshold)

#### Strategy 3: Hybrid (Instagram/Facebook approach)
- Regular users (< 5K followers): Push model — precompute feeds
- Celebrities (> 5K followers): Pull model — merge at read time
- At feed read: start with precomputed feed, merge in 5-10 celebrity posts, re-rank

```
Feed request for user U:
  1. Fetch precomputed feed from Redis (posts from non-celebrities)
  2. Identify followed celebrities (from Redis set: user:U:celeb_follows)
  3. Pull latest N posts from each celebrity (from post cache)
  4. Merge + re-rank the combined set
  5. Apply filters (mutes, blocks)
  6. Inject ads at positions 3, 7, 15
  7. Return paginated results via cursor
```

### 5.2 Ranking Algorithm (EdgeRank-style)

Feed score formula:
```
score = affinity_weight × content_type_weight × time_decay_factor

Where:
  affinity_weight   = interaction history between viewer and author
                      (likes, comments, DMs: 0.0 → 1.0)
  content_type_weight = video > image > link > text (engagement multipliers)
  time_decay_factor  = 1 / (1 + age_in_hours)
                       OR exponential: e^(-λ × age_hours)
```

**Redis Sorted Set Score:** Encode ranking score as the sorted set score.
Alternatively: store `(timestamp, post_id)` in sorted set; re-rank at read time using the formula above against the top-N candidates.

### 5.3 Redis Sorted Set Feed Cache

```
Key:   timeline:{user_id}
Type:  Sorted Set
Score: rank_score (higher = shown earlier)
Member: post_id

Operations:
  ZADD timeline:alice 98.5 "post:123"   # fan-out adds post
  ZREVRANGE timeline:alice 0 49          # get top 50 posts
  ZREMRANGEBYSCORE timeline:alice 0 <old_score  # trim old posts
  ZREM timeline:alice "post:123"         # delete propagation
  TTL: 7 days (inactive user feeds evicted)
```

### 5.4 Cursor-Based Pagination

**Why cursor, not offset:**
- `LIMIT 20 OFFSET 100` requires scanning 120 rows; slow and inconsistent (new posts shift offsets)
- Cursor: encode the last-seen post's rank score → `GET /feed?after=<score_cursor>`

```
First request:  GET /feed
Response: { posts: [...20 items], next_cursor: "98.3:post_456" }

Next request:   GET /feed?cursor=98.3:post_456
Query:          ZREVRANGEBYSCORE timeline:alice (98.3 -inf LIMIT 20
```

### 5.5 Fan-Out Service

```
Kafka consumer: post.created topic

For each new post event:
  1. Author has N followers → if N < CELEBRITY_THRESHOLD:
       batch write to all follower timelines (Redis ZADD pipeline)
     else:
       skip fan-out; mark author as celebrity; feeds will pull at read time

  2. Update author's post index: ZADD posts:{author_id} timestamp post_id

  3. Update engagement counters (Kafka to Counter Service)
```

### 5.6 Follow Graph Storage

- **PostgreSQL:** `follows(follower_id, followee_id, created_at)` — source of truth
- **Redis Sets:** `followers:{user_id}` = set of follower user_ids (for fan-out lookup)
- **Redis Sets:** `following:{user_id}` = set of users this user follows (for feed pull)
- For celebrities, followers set is too large for Redis → store follower list in Cassandra, iterate in batches for fan-out

---

## 6. Database Design

```sql
-- Posts
CREATE TABLE posts (
    post_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL,
    content     TEXT,
    media_urls  JSONB,  -- array of S3 URLs
    post_type   VARCHAR(20) DEFAULT 'TEXT',  -- TEXT, IMAGE, VIDEO, LINK
    like_count  INT DEFAULT 0,
    comment_count INT DEFAULT 0,
    share_count INT DEFAULT 0,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    deleted_at  TIMESTAMPTZ,  -- soft delete
    INDEX idx_user_created (user_id, created_at DESC),
    INDEX idx_created (created_at DESC)
) PARTITION BY RANGE (created_at);  -- monthly partitions

-- Follows (follow graph)
CREATE TABLE follows (
    follower_id UUID NOT NULL,
    followee_id UUID NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (follower_id, followee_id),
    INDEX idx_followee (followee_id, follower_id)  -- "who follows celebrity X"
);

-- Feed items (materialized feed — only for non-celebrities; overflow to Redis)
CREATE TABLE feed_items (
    user_id     UUID NOT NULL,
    post_id     UUID NOT NULL,
    score       FLOAT NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (user_id, post_id),
    INDEX idx_user_score (user_id, score DESC)
) -- Consider this only as fallback when Redis is down

-- Engagement (likes, shares) — denormalized counters in Cassandra for scale
CREATE TABLE likes (
    post_id     UUID NOT NULL,
    user_id     UUID NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (post_id, user_id)
);

-- Blocks/Mutes (for feed filtering)
CREATE TABLE mutes (
    user_id     UUID NOT NULL,  -- user who muted
    target_id   UUID NOT NULL,  -- user who was muted
    mute_type   VARCHAR(10),    -- MUTE, BLOCK
    PRIMARY KEY (user_id, target_id)
);
```

**Sharding Strategy:**
- `posts` sharded by `user_id` — all posts by one user on same shard
- `follows` sharded by `follower_id` for "who do I follow?" queries; secondary index by `followee_id` for "who follows me?"
- `feed_items` sharded by `user_id` — each user's feed on one shard

---

## 7. API Design

```
POST /v1/posts
Body: { content, media_urls?: [], post_type: "TEXT" }
Response 201: { post_id, created_at }

GET /v1/feed
Headers: Authorization: Bearer <jwt>
Query: ?limit=20&cursor=<opaque_cursor>
Response: {
  posts: [{ post_id, author, content, like_count, created_at, ... }],
  next_cursor: "eyJzY29yZSI6OTguMywicG9zdCI6IjEyMyJ9",
  has_more: true
}

GET /v1/posts/{post_id}

DELETE /v1/posts/{post_id}

POST /v1/posts/{post_id}/like
DELETE /v1/posts/{post_id}/like

POST /v1/users/{user_id}/follow
DELETE /v1/users/{user_id}/follow

GET /v1/users/{user_id}/followers?limit=50&cursor=xxx
GET /v1/users/{user_id}/following?limit=50&cursor=xxx

POST /v1/users/{user_id}/mute   # exclude from feed
POST /v1/users/{user_id}/block  # block + mute
```

---

## 8. Scalability & Bottlenecks

### Bottleneck 1: Celebrity Fan-Out
**Problem:** Lady Gaga posts → 100M fan-out writes → Redis overloaded for minutes.
**Solution:** Hybrid model — skip fan-out for users with > N followers. Pull-merge celebrity posts at read time. Limit: fetch latest 50 posts per celebrity followed; merge into cached feed.

### Bottleneck 2: Feed Cache Memory
**Problem:** 500M DAU × 200 post IDs × 8 bytes = 800 GB — expensive Redis cluster.
**Solution:** Only cache feeds of users active in last 24 hours (roughly 50% of DAU = 250M). Evict inactive feeds with TTL=7 days. Use Redis sorted sets with ZRANGEBYSCORE to cap feed depth at 500 posts.

### Bottleneck 3: Read Hotspots (Viral Posts)
**Problem:** A viral post gets millions of reads simultaneously — hammers the post DB.
**Solution:** Cache hot posts in Redis (`post:{post_id}` → JSON, TTL=1 hour). CDN caches images/videos. Decouple post metadata reads from feed reads.

### Bottleneck 4: Follow Graph Size
**Problem:** User with 50K following list → fan-in at read time is slow.
**Solution:** Cap follow pull to top 200 most-interacted accounts (affinity-ranked). Pre-compute "active following" list per user (updated daily). Celebrity pull limited to N=10 celebrities per feed refresh.

### Bottleneck 5: Real-time Engagement Counter Updates
**Problem:** A post going viral gets millions of likes/second — can't UPDATE the posts table row per like.
**Solution:** Buffer like counts in Redis counter (`INCR like_count:{post_id}`). Periodically sync to PostgreSQL in batches (every 60 seconds). This is the "eventual consistency" part — like count may be slightly stale.

---

## 9. Trade-offs & Design Decisions

### Decision 1: Push vs Pull vs Hybrid Fan-Out
- **Chosen:** Hybrid — push for regular users, pull for celebrities
- **Why:** Pure push fails at celebrity scale (100M writes); pure pull fails at high follow-count (50K DB reads per feed load)
- **Celebrity threshold:** Configurable, typically 5K-10K followers

### Decision 2: Pre-ranked Feed vs Rank at Read Time
- **Chosen:** Store post IDs in sorted set with pre-computed score; re-rank candidates at read time
- **Why:** Ranking signals (affinity, engagement) change dynamically; pre-ranking in fan-out uses stale signals
- **Approach:** Fan-out stores `(timestamp, post_id)` → at read time, fetch top 500 candidates, re-rank by formula, return top 20

### Decision 3: Cursor Pagination vs Offset
- **Chosen:** Cursor-based (encode last-seen score)
- **Why:** Offset pagination breaks when new posts are inserted; cursor is stable and efficient on sorted sets

### Decision 4: Redis Sorted Set vs Pre-Materialized DB Table
- **Chosen:** Redis as primary feed store, DB as fallback/source of truth
- **Why:** Redis sorted set operations (ZADD, ZREVRANGE) are O(log N) and run in memory — ideal for feed reads
- **DB fallback:** When Redis misses (cold user), compute feed from DB on the fly and warm cache

### Decision 5: Soft Deletes for Posts
- **Chosen:** Soft delete (`deleted_at` timestamp) + async propagation to remove from feeds
- **Why:** Hard delete has cascading complexity (remove from all follower timelines); soft delete is immediate from post service, async cleanup from feeds via Kafka event

---

## 10. Key Interview Talking Points

1. **Fan-Out on Write vs Read Trade-off:** Fan-out on write gives O(1) feed reads but O(followers) on write — unsustainable for celebrities. Fan-out on read gives O(1) writes but O(following_count) reads. The hybrid model draws the line at a configurable follower threshold.

2. **EdgeRank / Feed Ranking:** Facebook's original ranking formula: `Affinity × Weight × Time Decay`. Time decay prevents old posts from dominating. Affinity personalizes — you see more from people you interact with. Modern feeds use ML models (Transformer-based), but EdgeRank is the classic interview answer.

3. **Redis Sorted Sets:** `ZADD`, `ZREVRANGE`, `ZRANGEBYSCORE` — these are O(log N) operations. The score can encode ranking, timestamp, or a compound value. The member is a post ID. This is the canonical feed storage structure.

4. **Cursor Pagination:** Cursor encodes `(score, post_id)` → allows `ZREVRANGEBYSCORE timeline:user (cursor -inf LIMIT 20`. This is O(log N + K) where K is the result set — much faster than offset scan.

5. **Kafka Fan-Out:** Post creation publishes to `post.created` topic. Fan-out service is a consumer group. This decouples post creation latency from fan-out latency. Fan-out can lag 1-2 seconds without user impact.

6. **Memory Optimization:** Don't cache feeds for inactive users. 500M DAU doesn't mean 500M concurrent active feeds. With 7-day TTL eviction, you only hold feeds for recently active users. Redis cluster can be right-sized accordingly.

7. **Handling Deletes:** `post.deleted` Kafka event → Fan-out service sends `ZREM timeline:{follower_id} post_id` to all follower caches. For soft-deleted posts, the post service returns 410 Gone; feeds filter out deleted post IDs on the client side as well.

8. **Sponsored Posts Injection:** Ad service returns sponsored post IDs with ranking scores. Injected deterministically at positions 3, 7, 15 (standard mobile feed pattern). Separate ad impression tracking event fires when user scrolls past.

9. **Affinity Computation:** Affinity score = weighted sum of interactions (like=1pt, comment=3pts, share=5pts, DM=10pts) decayed by time, normalized to [0,1]. Computed offline (Spark job) daily and stored in Redis (`affinity:{viewer}:{author}`).

10. **Write Rate Back-of-Envelope:** 300 posts/sec × average 200 followers × push model = 60,000 Redis writes/second in the fan-out layer. At 100 bytes per Redis write, that's 6 MB/s — very manageable. The celebrity case is: 1 post × 100M followers = 100M writes → must use pull model.
