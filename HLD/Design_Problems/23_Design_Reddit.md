# System Design: Reddit

## 1. Problem Statement & Clarifying Questions

### Problem Statement
Design a social news aggregation and discussion platform like Reddit where users can submit posts (links or text), vote on content, comment in nested threads, and subscribe to topic-based communities (subreddits).

### Clarifying Questions
1. **Scale**: 1.5B monthly users, 430M posts — how many active subreddits? (~3M)
2. **Vote scale**: How many votes per day? (~500M votes/day)
3. **Feed types**: Hot, New, Top, Rising — all required?
4. **Comment depth**: Max nesting depth? (no hard limit, but display collapses beyond 7 levels)
5. **Content types**: Text, link, image, video?
6. **Moderation**: AutoModerator rules, shadow banning, content flagging?
7. **Search**: Full-text search across posts and comments?
8. **Consistency for votes**: Is exact vote count critical? (approximate OK for display, exact for karma)
9. **Cross-posting**: Can posts appear in multiple subreddits?
10. **Awards**: Reddit premium awards system?

---

## 2. Functional & Non-Functional Requirements

### Functional Requirements
- Users can create accounts with karma (post karma + comment karma)
- Users can create and subscribe to subreddits
- Users can submit posts (text, link, image, video) to subreddits
- Users can upvote/downvote posts and comments (one vote per user per item)
- Changing vote: upvote then downvote changes net score by -2
- Nested comment threads with path-based ordering
- Feed generation: hot/new/top/rising per subreddit
- Subreddit moderators can remove posts, ban users, set rules
- AutoModerator rules: spam filtering, keyword matching, karma requirements
- User profiles with post/comment history and karma display
- Cross-posting between subreddits
- Search: full-text across posts and comments (Elasticsearch)
- Awards: give awards to posts/comments (cosmetic + coins)

### Non-Functional Requirements
- **Scale**: 1.5B MAU, 430M posts, 500M votes/day
- **Availability**: 99.9% uptime
- **Feed latency**: < 150ms p99 for subreddit feed
- **Vote idempotency**: Double-clicking upvote should not double-count
- **Consistency**: Eventual consistency for vote counts; strong for karma
- **Read/Write ratio**: 95:5 (very read-heavy)
- **Storage**: 430M posts × ~2KB = ~860GB; comments ~5TB; votes ~50GB

---

## 3. Capacity Estimation

### Traffic
- **Page views**: ~5B/day → ~58K RPS average, 200K RPS peak
- **Votes**: 500M/day → ~5,800 VPS average, 20K VPS peak
- **Posts**: ~500K/day (new posts)
- **Comments**: ~5M/day

### Storage
- **Posts**: 430M × 2KB = ~860GB
- **Comments**: ~10B total × 500 bytes = ~5TB
- **Votes**: ~100B total (accumulated) × 20 bytes = ~2TB (use compressed bitmap)
- **User accounts**: 1.5B × 1KB = ~1.5TB
- **Subreddits**: 3M × 500 bytes = ~1.5GB

### Caching
- Hot post feeds cached in Redis (top 1000 posts per active subreddit, TTL 5 min)
- Hot post scores recomputed every minute via background job for top ~10K subreddits
- Vote counts: Redis INCR/DECR counters, flush to PostgreSQL every 30s

---

## 4. High-Level Architecture

```
                      ┌─────────────────────────────────────────────────────┐
                      │                    Clients                           │
                      │          Web / iOS / Android / API                   │
                      └──────────────────────┬──────────────────────────────┘
                                             │
                      ┌──────────────────────▼──────────────────────────────┐
                      │                Load Balancer                         │
                      └──┬────────────┬──────────────┬───────────┬──────────┘
                         │            │              │           │
             ┌───────────▼──┐  ┌──────▼──────┐ ┌────▼──────┐ ┌──▼──────────┐
             │  Feed Svc    │  │  Post Svc   │ │  Vote Svc │ │  User Svc   │
             └──────┬───────┘  └──────┬──────┘ └────┬──────┘ └─────────────┘
                    │                 │              │
          ┌─────────▼──────┐  ┌───────▼────┐  ┌──────▼────┐
          │  Redis (Sorted  │  │ PostgreSQL │  │  Redis    │
          │  Sets for feeds)│  │  (Posts +  │  │  Counters │
          │                 │  │  Comments) │  │  (Votes)  │
          └─────────────────┘  └───────┬────┘  └──────┬────┘
                                       │               │
                               ┌───────▼───────┐  ┌───▼──────┐
                               │ Elasticsearch │  │  Kafka   │
                               │ (Post/Comment │  │  (Vote   │
                               │  Search)      │  │  events) │
                               └───────────────┘  └──────────┘

  Sharding: posts and votes sharded by subreddit_id
  Hot subreddits (r/AskReddit, r/worldnews) get dedicated shards
```

---

## 5. Component Deep-Dive

### 5.1 Hot Score Algorithm
Reddit's actual hot ranking algorithm:

```python
import math
from datetime import datetime

def hot_score(ups: int, downs: int, post_time: datetime) -> float:
    score = ups - downs
    order = math.log(max(abs(score), 1), 10)
    sign  = 1 if score > 0 else (-1 if score < 0 else 0)
    seconds = post_time.timestamp() - 1134028003  # Reddit epoch
    return round(sign * order + seconds / 45000, 7)
```

Key insights:
- `log10` dampens the impact of very high vote counts (viral posts don't dominate forever)
- Time component `seconds/45000` decays post score after ~12.5 hours
- A post with 10K upvotes submitted 2 days ago scores lower than 100 upvotes submitted 30 min ago
- This creates natural content turnover in hot feeds

### 5.2 Vote Storage at Scale

**Challenge**: 500M votes/day, each vote needs to be idempotent (no double-counting), and users can change votes.

**Option 1: PostgreSQL votes table**
- `votes(user_id, post_id, vote_type)` with UNIQUE(user_id, post_id)
- Exact counts, strong consistency, but 500M WPS is too high for single DB

**Option 2: Redis counters + periodic flush**
- `HSET post:{id} ups N` in Redis; flush to PostgreSQL every 30s
- Vote deduplication: Redis SET `voted:{user_id}` contains post_ids they voted on
- Con: vote data lost on Redis crash before flush

**Option 3: Kafka vote events + stream processing**
- Each vote = Kafka event; stream processor deduplicates using Redis bloom filter
- Aggregate counters updated in Flink/Spark; flush to PostgreSQL hourly
- Vote log retained in Kafka for 7 days (audit trail)

**Chosen**: Option 2 (Redis counters) with WAL-based recovery + Kafka for karma tracking

### 5.3 Comment Tree Structure

**Path Encoding (materialized path)**:
Each comment stores its full ancestry path:
```
comment_id | parent_id | path
1          | NULL      | "/1/"
2          | 1         | "/1/2/"
3          | 1         | "/1/3/"
4          | 2         | "/1/2/4/"
```
- Fetch entire subtree: `WHERE path LIKE '/1/2/%'` — single index scan
- Sort by path = depth-first traversal order
- Prefix length = depth (for indentation rendering)

**Alternative: Closure Table**
- `(ancestor_id, descendant_id, depth)` for every ancestor-descendant pair
- Flexible queries but O(depth) storage per comment

**Chosen**: Materialized path for Reddit-style deep nesting (read-heavy, rarely restructured)

### 5.4 Feed Generation

**Approach: Pre-computed feeds (Push model)**
- For each active subreddit, maintain a Redis sorted set: `feed:{subreddit_id}:{sort}`
- Key = post_id, Score = hot_score (or timestamp for "new", or vote count for "top")
- Background job recomputes hot scores every 5 minutes for active subreddits
- On feed request: `ZREVRANGE feed:programming:hot 0 24` — O(log N) operation

**Why not pull-on-demand?**
- At 200K RPS, computing hot scores on every feed request would be too slow
- Pre-computation allows O(log N) Redis reads instead of O(M) score computations

**Fan-out challenge**: r/AskReddit has 40M subscribers. When a new post is created, do we update 40M user feeds? No — for large subreddits, use pull model: user requests feed → fetch from subreddit feed, not personal feed.

### 5.5 Karma System
- **Post karma**: upvotes - downvotes on your posts (capped contribution per post: max +1000 karma per post to prevent viral manipulation)
- **Comment karma**: upvotes - downvotes on your comments
- Karma updated asynchronously via Kafka vote events → karma service
- Anti-gaming: karma from self-votes excluded; vote manipulation detection

### 5.6 Content Moderation
**AutoModerator** (rule-based):
- Regex pattern matching on post titles/bodies
- Karma requirements: `author_karma < 100` → remove
- Account age requirements: `author_account_age < 30` → hold for review
- Domain blacklist for link posts

**Shadow banning**:
- Banned user can post/vote but their content invisible to others
- User sees their own content as normal (no indication of ban)
- Reduces ban evasion and harassment

---

## 6. Database Design

### Posts Table
```sql
CREATE TABLE posts (
    id              BIGSERIAL PRIMARY KEY,
    subreddit_id    BIGINT NOT NULL,
    author_id       BIGINT REFERENCES users(id),
    title           VARCHAR(300) NOT NULL,
    body            TEXT,
    url             VARCHAR(2000),
    post_type       VARCHAR(10),    -- text, link, image, video
    upvotes         INT DEFAULT 1,
    downvotes       INT DEFAULT 0,
    score           INT DEFAULT 1,  -- denormalized: upvotes - downvotes
    hot_score       FLOAT,
    comment_count   INT DEFAULT 0,
    is_nsfw         BOOLEAN DEFAULT false,
    is_removed      BOOLEAN DEFAULT false,
    is_shadow_removed BOOLEAN DEFAULT false,
    flair           VARCHAR(64),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_posts_subreddit_hot   ON posts(subreddit_id, hot_score DESC);
CREATE INDEX idx_posts_subreddit_new   ON posts(subreddit_id, created_at DESC);
CREATE INDEX idx_posts_subreddit_score ON posts(subreddit_id, score DESC);
```

### Comments Table
```sql
CREATE TABLE comments (
    id          BIGSERIAL PRIMARY KEY,
    post_id     BIGINT REFERENCES posts(id),
    author_id   BIGINT REFERENCES users(id),
    parent_id   BIGINT REFERENCES comments(id),
    path        VARCHAR(1000),      -- materialized path: /1/2/4/
    body        TEXT NOT NULL,
    score       INT DEFAULT 1,
    depth       INT DEFAULT 0,
    is_removed  BOOLEAN DEFAULT false,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_comments_post_path ON comments(post_id, path);
CREATE INDEX idx_comments_parent    ON comments(parent_id);
```

### Votes Table
```sql
CREATE TABLE votes (
    user_id     BIGINT NOT NULL,
    item_id     BIGINT NOT NULL,
    item_type   VARCHAR(10) NOT NULL,  -- post or comment
    vote_type   SMALLINT NOT NULL,      -- 1=up, -1=down
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (user_id, item_id, item_type)
);
```

### Subreddits Table
```sql
CREATE TABLE subreddits (
    id              BIGSERIAL PRIMARY KEY,
    name            VARCHAR(21) UNIQUE NOT NULL,  -- max 21 chars
    title           VARCHAR(100),
    description     TEXT,
    subscriber_count INT DEFAULT 0,
    is_nsfw         BOOLEAN DEFAULT false,
    is_private      BOOLEAN DEFAULT false,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
```

### User_Subreddits Table
```sql
CREATE TABLE user_subreddits (
    user_id         BIGINT REFERENCES users(id),
    subreddit_id    BIGINT REFERENCES subreddits(id),
    role            VARCHAR(10) DEFAULT 'member',  -- member, moderator
    joined_at       TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (user_id, subreddit_id)
);
```

---

## 7. API Design

### Post API
```
POST /api/v1/r/{subreddit}/submit
Body: { title, body|url, post_type, flair }
Response: { post_id, permalink }

GET /api/v1/r/{subreddit}?sort=hot&t=day&limit=25&after=post_id
Response: { posts: [...], after: "next_cursor" }

GET /api/v1/r/{subreddit}/comments/{post_id}
Response: { post, comments: [nested tree] }
```

### Vote API
```
POST /api/v1/vote
Body: { item_id, item_type: "post"|"comment", direction: 1|0|-1 }
(direction 0 = unvote)
Response: { new_score, user_vote_direction }
```

### Comment API
```
POST /api/v1/r/{subreddit}/comments/{post_id}
Body: { body, parent_id? }
Response: { comment_id, path, depth }

GET /api/v1/r/{subreddit}/comments/{post_id}/more
Body: { children: [comment_ids], link_id }
Response: { comments: [...] }
```

### Search API
```
GET /api/v1/search?q=query&type=post|comment|subreddit&sort=relevance|new|top&t=all|year|month|week|day
```

---

## 8. Scalability & Bottlenecks

### Bottleneck 1: Vote Handling at 20K VPS Peak
- Redis HINCRBY for per-post vote counters — O(1) per vote
- Vote deduplication: Redis SET per user for recently voted items
- Kafka vote events for durable processing; karma updates async

### Bottleneck 2: Hot Score Recomputation
- 430M posts, can't recompute all hot scores every minute
- Solution: only recompute for posts < 24 hours old (hot score decays, old posts irrelevant)
- Background worker processes ~1M active posts/minute — manageable

### Bottleneck 3: Comment Tree Loading for Viral Posts
- A viral post may have 50K comments
- Don't load all comments at once — load top-level comments sorted by score, lazy-load subtrees
- "Load more comments" API endpoint for deep branches

### Bottleneck 4: r/AskReddit Scale (40M subscribers)
- Subreddit feed is shared (not per-user for large subreddits)
- Redis sorted set for feed: one sorted set per active subreddit
- Large subreddits get dedicated Redis nodes

### Bottleneck 5: Search Indexing
- New posts must appear in search within seconds
- CDC pipeline: PostgreSQL → Kafka → Elasticsearch consumer (< 5 second lag)
- Elasticsearch dedicated cluster for search, not impacting main DB

---

## 9. Trade-offs & Design Decisions

### Decision 1: Vote Storage — Redis vs PostgreSQL
- **PostgreSQL**: Exact, durable, but 20K WPS exceeds single-node write throughput
- **Redis**: Fast, but requires flush mechanism for durability
- **Choice**: Redis as write buffer, async flush to PostgreSQL every 30 seconds via Kafka
- **Trade-off**: Up to 30 seconds of vote data at risk on crash (acceptable for vote counts)

### Decision 2: Feed Generation — Push vs Pull
- **Push (fanout on write)**: Write to all subscriber feeds on post creation. Fast reads but expensive for large subreddits.
- **Pull (fanout on read)**: Compute feed on request. Slower reads but simple writes.
- **Choice**: Pull model per subreddit (not per user) — subreddit feed is a shared sorted set. Personal home feed is a merge of subscribed subreddit feeds.

### Decision 3: Comment Storage — Adjacency List vs Materialized Path
- **Adjacency List**: Simple, but requires recursive CTE to fetch subtrees (N+1 query risk)
- **Materialized Path**: Single query for subtree, efficient range scan, easy depth calculation
- **Closure Table**: Flexible but O(depth²) storage
- **Choice**: Materialized path for Reddit's read-heavy comment loading

### Decision 4: Hot Score Caching Granularity
- **Per post**: Cache each post's hot score in Redis, invalidate on new vote
- **Per feed**: Cache entire sorted feed in Redis sorted set, recompute every N minutes
- **Choice**: Per-feed sorted set in Redis; recompute hot scores for active posts every 5 minutes
- **Trade-off**: Vote cast now may not reflect in feed for up to 5 minutes

### Decision 5: Sharding Strategy
- **By subreddit_id**: All posts in one subreddit on same shard — range queries fast, but hot subreddits (r/AskReddit) create hotspots
- **By post_id (hash)**: Even distribution but cross-shard queries for subreddit feeds
- **Choice**: Shard by subreddit_id for most subreddits; hot subreddits (>5M subscribers) get dedicated shards with further hash partitioning

---

## 10. Key Interview Talking Points

### 1. Reddit Hot Algorithm
Most candidates say "sort by votes" — stand out by knowing the actual algorithm. Key points:
- log10 scale dampens viral runaway (10K votes ≠ 10× better than 1K, more like 1.33×)
- Time decay: seconds since Reddit epoch divided by 45000 means a post ages out of hot in ~12 hours
- This combination creates natural content churn: fresh good content beats stale great content

### 2. Vote Idempotency
Critical: user votes must not double-count. Three layers:
1. Application-level check: Redis SET tracking user's recent votes
2. Database-level: UNIQUE constraint on (user_id, item_id, item_type)
3. Kafka deduplication: use post_id as Kafka message key → same vote to same partition = ordered, deduplicatable

### 3. Materialized Path for Comments
The key insight: Reddit comment trees are wide and deep but read far more than written. Materialized path makes `LIKE '/1/2/%'` a single range scan. Depth = count of `/` separators in path. Sorting by path gives depth-first traversal, which is the natural Reddit reading order.

### 4. Feed Pre-computation
At 200K RPS, you cannot compute hot scores on demand. The Redis sorted set pattern: ZADD with score=hot_score, ZREVRANGE to read top N. Background workers periodically update scores only for recent posts (past 24h). This scales horizontally: one worker per K subreddits.

### 5. Karma System Design
Karma is user-aggregate state. Computing it on demand (SELECT SUM of all votes on all posts) is O(career). Instead:
- Kafka stream of vote events
- Karma service consumes events, maintains running total in Redis + PostgreSQL
- Cap karma per post (max +1000) to prevent manipulation

### 6. Sharding for Scale
430M posts cannot fit on one PostgreSQL node. Shard by subreddit_id:
- 3M subreddits, ~143 posts/subreddit on average
- Top 1000 subreddits have 99% of traffic → dedicated shards
- Hash ring for small/medium subreddits
- This means subreddit feed queries (most common) never require cross-shard joins

### 7. Search at Scale
Elasticsearch for Reddit means: 430M posts × ~2KB = ~860GB of text data.
- Shard by subreddit for colocation (subreddit-scoped search hits one shard)
- For global search: scatter-gather across all shards
- CDC pipeline for near-real-time index updates (< 5s lag for new posts)
