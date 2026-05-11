# System Design: Instagram

---

## 1. Problem Statement

Design a photo and video sharing social network similar to Instagram. Users can upload photos/videos, follow other users, view a personalized feed of content from accounts they follow, like and comment on posts, and search for users/hashtags. The system must support hundreds of millions of users globally.

---

## 2. Clarifying Questions to Ask

- How many DAU are we targeting? (100M? 500M?)
- What is the read/write ratio for feeds?
- Do we need video support or only photos?
- What is the max photo/video size?
- How many followers can a celebrity have? (up to 100M)
- Should feed be real-time or can it be slightly stale (eventual consistency)?
- Do we need Stories (ephemeral content that expires in 24h)?
- Should the feed be ranked (ML-based) or purely chronological?
- Do we need DMs (Direct Messages)?
- What geographic regions must we support?

---

## 3. Functional Requirements

1. Users can create accounts, follow/unfollow other users.
2. Users can upload photos with captions, hashtags, and location tags.
3. Users can view a personalized feed of posts from accounts they follow.
4. Users can like and comment on posts.
5. Users can search by hashtag or username.
6. Users receive notifications (likes, comments, new followers).
7. Posts can be deleted by the owner.
8. User profile shows all their posts in a grid view.

---

## 4. Non-Functional Requirements

- **Availability**: 99.99% (4 nines) — feed reads must always work
- **Latency**: Feed load P99 < 200ms; photo upload P95 < 2s
- **Durability**: Photos must never be lost once upload confirmed
- **Consistency**: Eventual consistency acceptable for feeds (user can see new post within seconds)
- **Scalability**: 500M DAU, 100M photos uploaded daily
- **Throughput**: Handle 1.5M photo uploads/hour at peak
- **Storage**: Petabytes of photo storage
- **Global**: CDN for photo delivery worldwide

---

## 5. Capacity Estimation

### Users & Content
- DAU: 500M users
- Monthly Active Users (MAU): 1B
- Photos uploaded per day: 100M
- Average photo size (after compression): 500 KB
- Average photo with 3 sizes (original, medium, thumbnail): 1.5 MB total

### Storage
- Photo storage per day: 100M * 1.5 MB = 150 TB/day
- Storage per year: 150 TB * 365 = ~55 PB/year
- Metadata per post: ~2 KB (caption, tags, location, timestamp)
- Metadata storage per day: 100M * 2 KB = 200 GB/day

### Traffic
- Photo uploads per second: 100M / 86,400 = ~1,160 uploads/sec
- Peak upload QPS (3x): ~3,500 uploads/sec
- Feed reads per day: 500M DAU * 5 feed refreshes = 2.5B reads/day
- Feed read QPS: 2.5B / 86,400 = ~29,000 reads/sec
- Peak feed read QPS (3x): ~87,000 reads/sec

### Bandwidth
- Upload inbound: 1,160 * 1.5 MB = ~1.7 GB/s
- Feed outbound: 29,000 req/s * ~50 KB (10 thumbnails) = ~1.4 GB/s
- CDN offloads 90% of photo delivery

---

## 6. High-Level Architecture

```
                        [Client App / Browser]
                                |
                        [CDN  (CloudFront)]
                        /              \
               [Load Balancer]    [Media Load Balancer]
                /      |      \            |
        [API Server] [API]  [API]   [Upload Service]
              |                           |
      [Service Layer]              [Image Processor]
     /    |     |    \                (resize, filter)
 [Feed] [User] [Like] [Search]            |
Service Service Service  (ES)          [S3 / Object Store]
    |      |      |
 [Cache] [Cache] [Cache]
 (Redis) (Redis) (Redis)
    |      |
  [Primary DB]  [Read Replicas x5]
  (PostgreSQL)
    |
  [Follow Graph DB]
  (PostgreSQL or Redis Sets)
```

### Photo Upload Flow
```
1. Client requests pre-signed S3 URL from Upload Service
2. Client uploads photo directly to S3 (bypasses API servers)
3. S3 event triggers Image Processing Lambda/service
4. Image Processor creates: thumbnail (150x150), medium (720px), original
5. CDN URLs stored in DB
6. Post metadata saved to PostgreSQL
7. Feed fanout triggered (Kafka event → Fanout Service)
```

### Feed Read Flow
```
1. Client requests /feed?user_id=X&page=1
2. API Server checks Redis for cached timeline
3. Cache hit: return cached feed (list of post IDs)
4. Cache miss: query Feed Service to generate feed
5. Fetch post metadata + CDN URLs for each post ID
6. Return enriched feed to client
```

---

## 7. Component Deep-Dive

### 7.1 Photo Upload Pipeline

**Step 1: Pre-signed URL**
- Client calls `POST /api/upload/presign` to get a temporary S3 upload URL
- Client uploads directly to S3 — API servers never handle binary file data
- This offloads bandwidth from API tier

**Step 2: Image Processing**
- S3 triggers Lambda/container on new object
- Creates 3 sizes: thumbnail (150x150), medium (720px wide), HD (1080px wide)
- Uses libvips for fast image processing
- Stores all variants back in S3 under organized key:
  `photos/{user_id}/{year}/{month}/{post_id}/{size}.jpg`

**Step 3: CDN Distribution**
- CloudFront or Akamai sits in front of S3
- Photos cached at edge locations globally
- Cache TTL: 365 days (photos are immutable once processed)
- URL pattern: `https://cdn.instagram.com/photos/{user_id}/{post_id}/medium.jpg`

### 7.2 Feed Generation: Fan-out Strategy

**Fan-out on Write (Push Model)**
- When user X posts a photo, immediately push post_id to all X's followers' timelines
- Each follower's timeline is stored as a Redis Sorted Set: `timeline:{user_id}` → `{post_id: timestamp}`
- Pros: Feed reads are O(1), instant
- Cons: Celebrity with 10M followers → 10M Redis writes per post; high write amplification

**Fan-out on Read (Pull Model)**
- No pre-computation; when user opens feed, query all followed users' post lists and merge
- Pros: No write amplification, handles celebrities well
- Cons: High read latency, complex merge of N sorted lists

**Hybrid Approach (Production Choice)**
- Regular users (< 1M followers): Fan-out on write — push to followers' Redis timelines
- Celebrities (> 1M followers): Fan-out on read — followers pull from celebrity's post list
- Feed service merges pre-computed regular-user posts + on-demand celebrity posts
- This is approximately what Instagram/Twitter actually do

### 7.3 Follow Graph Storage

**Storage**: PostgreSQL `follows` table + Redis Sets for fast lookup

**PostgreSQL**: Source of truth for follow relationships
- Columns: follower_id, followee_id, created_at
- Query: "who does user X follow?" → SELECT followee_id WHERE follower_id = X
- Sharded by follower_id

**Redis Sets**:
- `followers:{user_id}` → Set of follower user_ids (for fanout)
- `following:{user_id}` → Set of followed user_ids (for feed generation)
- Cached for active users, evicted on inactivity

### 7.4 Like / Comment System

**Likes**:
- Stored in PostgreSQL: (post_id, user_id, timestamp)
- Shard by post_id
- Count cached in Redis: `likes:{post_id}` → integer
- Redis INCR/DECR for real-time count; periodic sync to DB

**Comments**:
- Stored in PostgreSQL with parent_comment_id for threading
- Paginated: load 20 comments at a time
- Comment count cached in Redis

### 7.5 Search (Elasticsearch)

**Hashtag Search**:
- Index: `hashtags` — field: tag (keyword), post_id, created_at
- When post is created, extract hashtags, index each to ES
- Query: `GET /hashtags/_search?q=sunset&sort=created_at:desc`

**User Search**:
- Index: `users` — fields: username, full_name, bio
- Use edge n-gram tokenizer for prefix search (autocomplete)

**Trending Hashtags**:
- Sliding window counter in Redis: `trending:{hashtag}:{hour}` → count
- Top-K query every 15 minutes

### 7.6 Notification System

- Event published to Kafka: `{event_type: "like", post_id, liker_id, post_owner_id}`
- Notification consumer:
  - Read Kafka events
  - Create notification record in DB
  - Push via WebSocket (for online users) or APNs/FCM (for mobile push)
- Notification fanout batched for efficiency

### 7.7 Sharding Strategy

- Primary shard key: **user_id** (most queries filter by user)
- Posts table: shard by user_id (creator's posts always on same shard)
- Follows table: shard by follower_id
- Challenge: "hot" celebrities on same shard — use consistent hashing + virtual nodes to rebalance

---

## 8. Database Design

### Table: users
```sql
CREATE TABLE users (
    id              BIGSERIAL PRIMARY KEY,
    username        VARCHAR(50) UNIQUE NOT NULL,
    email           VARCHAR(255) UNIQUE NOT NULL,
    full_name       VARCHAR(100),
    bio             TEXT,
    profile_pic_url TEXT,
    follower_count  BIGINT DEFAULT 0,
    following_count BIGINT DEFAULT 0,
    post_count      INT DEFAULT 0,
    created_at      TIMESTAMP DEFAULT NOW(),
    is_private      BOOLEAN DEFAULT FALSE,
    is_verified     BOOLEAN DEFAULT FALSE
);
```

### Table: posts
```sql
CREATE TABLE posts (
    id              BIGSERIAL PRIMARY KEY,
    user_id         BIGINT NOT NULL REFERENCES users(id),
    caption         TEXT,
    location        VARCHAR(255),
    thumbnail_url   TEXT,
    medium_url      TEXT,
    original_url    TEXT,
    like_count      INT DEFAULT 0,
    comment_count   INT DEFAULT 0,
    created_at      TIMESTAMP DEFAULT NOW(),
    is_deleted      BOOLEAN DEFAULT FALSE,
    INDEX idx_user_id_created (user_id, created_at DESC)
);
```

### Table: follows
```sql
CREATE TABLE follows (
    follower_id     BIGINT NOT NULL REFERENCES users(id),
    followee_id     BIGINT NOT NULL REFERENCES users(id),
    created_at      TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (follower_id, followee_id),
    INDEX idx_followee (followee_id)
);
```

### Table: likes
```sql
CREATE TABLE likes (
    post_id         BIGINT NOT NULL REFERENCES posts(id),
    user_id         BIGINT NOT NULL REFERENCES users(id),
    created_at      TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (post_id, user_id)
);
```

### Table: comments
```sql
CREATE TABLE comments (
    id              BIGSERIAL PRIMARY KEY,
    post_id         BIGINT NOT NULL REFERENCES posts(id),
    user_id         BIGINT NOT NULL REFERENCES users(id),
    parent_id       BIGINT REFERENCES comments(id),
    content         TEXT NOT NULL,
    created_at      TIMESTAMP DEFAULT NOW(),
    is_deleted      BOOLEAN DEFAULT FALSE,
    INDEX idx_post_id (post_id, created_at)
);
```

### Table: hashtags
```sql
CREATE TABLE post_hashtags (
    post_id         BIGINT REFERENCES posts(id),
    hashtag         VARCHAR(100),
    PRIMARY KEY (post_id, hashtag),
    INDEX idx_hashtag (hashtag)
);
```

### Redis Data Structures
```
timeline:{user_id}     → Sorted Set  { post_id: unix_timestamp }  (user's feed)
followers:{user_id}    → Set         { follower_user_ids }
following:{user_id}    → Set         { followee_user_ids }
likes:{post_id}        → Integer     (like count)
comments:{post_id}     → Integer     (comment count)
session:{token}        → Hash        { user_id, expires_at }
```

---

## 9. API Design

### Post a Photo
```
POST /api/v1/posts
Authorization: Bearer {token}
Content-Type: application/json

Request:
{
  "caption": "Beautiful sunset #nature #travel",
  "media_key": "s3-key-after-direct-upload",
  "location": "Malibu, CA",
  "hashtags": ["nature", "travel"]
}

Response 201:
{
  "post_id": 123456789,
  "thumbnail_url": "https://cdn.instagram.com/.../thumb.jpg",
  "created_at": "2024-01-15T18:30:00Z"
}
```

### Get Feed
```
GET /api/v1/feed?cursor=abc&limit=20
Authorization: Bearer {token}

Response 200:
{
  "posts": [
    {
      "post_id": 123,
      "user": { "id": 456, "username": "john_doe", "profile_pic": "..." },
      "caption": "...",
      "thumbnail_url": "...",
      "medium_url": "...",
      "like_count": 1523,
      "comment_count": 42,
      "liked_by_viewer": false,
      "created_at": "..."
    }
  ],
  "next_cursor": "xyz",
  "has_more": true
}
```

### Follow User
```
POST /api/v1/users/{user_id}/follow
Authorization: Bearer {token}

Response 200: { "following": true }
```

### Like a Post
```
POST /api/v1/posts/{post_id}/like
Authorization: Bearer {token}

Response 200: { "liked": true, "like_count": 1524 }
```

### Search
```
GET /api/v1/search?q=sunset&type=hashtag&limit=20

Response 200:
{
  "hashtags": [
    { "tag": "sunset", "post_count": 5000000 }
  ],
  "users": []
}
```

---

## 10. Scalability & Bottlenecks

### Bottleneck 1: Celebrity Fan-out
- Problem: 10M followers * 1 post = 10M Redis writes
- Solution: Hybrid fan-out (celebrities use pull model, skip push)

### Bottleneck 2: Photo Storage
- 150 TB/day requires distributed object storage (S3)
- CDN offloads 95% of read traffic from S3

### Bottleneck 3: Feed Read Latency
- Solution: Pre-computed timelines in Redis Sorted Sets
- Feed read = O(log N) Redis ZRANGE operation
- Target cache hit rate: 99%

### Bottleneck 4: Follow Graph at Scale
- Social graph is sparse but large
- Solution: Redis sets for hot users; PostgreSQL as source of truth
- Preload follower lists for online users

### Bottleneck 5: Like Count Accuracy
- Problem: Heavy concurrent likes on viral posts
- Solution: Redis INCR (atomic), periodic batch sync to DB every 30 seconds

---

## 11. Trade-offs & Design Decisions

### Fan-out on Write vs Read
- Write: Lower read latency, but write amplification for celebrities
- Read: Simple writes, but slow reads for users following many people
- Decision: Hybrid — gives best of both worlds

### SQL vs NoSQL for Posts
- SQL: Strong consistency, complex join queries (user + post + like in one query)
- NoSQL (Cassandra): Better horizontal scale, but no joins
- Decision: PostgreSQL sharded by user_id; Redis for counts and timelines

### Chronological vs Ranked Feed
- Chronological: Simple, predictable, no ML needed
- Ranked: Better engagement but requires ML infrastructure
- Decision: Chronological first; ranked feed is a separate feature iteration

### Strong vs Eventual Consistency for Likes
- Strong: Every read sees latest count — expensive at scale
- Eventual: Slight lag acceptable for like counts
- Decision: Eventual consistency via Redis cache + periodic DB sync

---

## 12. Key Interview Talking Points

1. **Photo upload pipeline**: Never upload to API server — use pre-signed S3 URLs for direct client-to-S3 upload to avoid bandwidth bottleneck.

2. **CDN for media**: Photos are read-heavy and globally distributed — CDN is non-negotiable; without it, S3 latency would be too high.

3. **Hybrid fan-out**: This is the Instagram-level insight. Regular users = push, celebrities = pull. Justify the threshold (1M followers).

4. **Redis Sorted Set for timelines**: Key insight — timeline is naturally ordered by timestamp, Sorted Set gives O(log N) insert and O(1) range read.

5. **Sharding by user_id**: Posts always queried by creator or feed consumer — user_id is the natural partition key. Avoids cross-shard joins.

6. **Eventual consistency for counts**: Like counts don't need to be real-time accurate; a 30-second lag is acceptable.

7. **Elasticsearch for search**: Full-text search, hashtag autocomplete — not possible efficiently in PostgreSQL at scale.

8. **Image processing pipeline**: Async processing (S3 event → Lambda) decouples upload from processing; user gets confirmation immediately.

9. **Follow graph in Redis**: For fanout service, we need follower list in memory — Redis Set allows O(1) membership check and O(N) iteration.

10. **Capacity math**: 100M photos/day * 1.5MB = 150 TB/day — leads naturally to CDN and distributed storage discussion.
