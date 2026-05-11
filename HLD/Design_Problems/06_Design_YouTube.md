# Design YouTube — High-Level Design

---

## 1. Problem Statement & Clarifying Questions

**Problem Statement:**
Design a large-scale video sharing platform (YouTube) that allows users to upload, process, store, and stream videos globally. The system must handle massive read traffic, complex video processing pipelines, personalized recommendations, search, and social features like comments, likes, and subscriptions.

**Clarifying Questions:**
- What is the expected scale? (2B registered users, 500M DAU)
- Do we need to support live streaming, or recorded videos only?
- What video resolutions should we support? (360p, 480p, 720p, 1080p, 4K)
- Should we design the recommendation engine in depth?
- Do we need to handle monetization (ads)?
- What is the geographic distribution of users?
- Should we support offline downloads?
- Do we need real-time comments or eventual consistency is fine?

**Assumptions:**
- Focus on recorded video upload, processing, and streaming
- Support adaptive bitrate streaming (HLS/DASH)
- Recommendations based on watch history and collaborative filtering
- Global CDN delivery required
- Comments, likes, subscriptions are core features

---

## 2. Functional & Non-Functional Requirements

### Functional Requirements
1. **Video Upload:** Users can upload videos (up to 10GB per video)
2. **Video Processing:** Transcode videos to multiple resolutions and bitrates
3. **Video Streaming:** Adaptive bitrate streaming based on network conditions
4. **Search:** Search videos by title, description, tags
5. **Recommendations:** Personalized video recommendations
6. **Social Features:** Like/dislike, comments, subscribe to channels
7. **View Count:** Real-time view count tracking
8. **Thumbnail Generation:** Auto-generate thumbnails during processing
9. **Subscription Feed:** Show latest videos from subscribed channels

### Non-Functional Requirements
1. **Availability:** 99.99% uptime for video streaming
2. **Latency:** Video starts playing within 2 seconds (P99)
3. **Consistency:** View counts can be eventually consistent (±1% accuracy)
4. **Durability:** Zero video loss after successful upload confirmation
5. **Scalability:** Handle 500M DAU, horizontal scaling
6. **Throughput:** 500 hours of video uploaded per minute
7. **Global Delivery:** <50ms latency for CDN-served content worldwide

---

## 3. Capacity Estimation

### Users
- Total Registered Users: 2 Billion
- Daily Active Users (DAU): 500 Million
- Monthly Active Users (MAU): 2 Billion

### Video Upload
- Upload rate: 500 hours of video per minute
- Average video size (raw): 1 hour ≈ 2GB → 500 * 2GB = 1TB raw video/minute
- Average processed size (all resolutions): ~5x compression with transcoding
- Storage per minute: ~200GB (after compression across all bitrates)
- Daily storage growth: 200GB * 60 * 24 ≈ 288TB/day
- Annual storage: ~100PB/year

### Video Watching
- Hours watched per day: 1 Billion hours
- Average video length: 7 minutes = 0.117 hours
- Videos watched per day: 1B / 0.117 ≈ 8.5 Billion video views/day
- Views per second: 8.5B / 86400 ≈ 100,000 views/second
- Peak QPS (3x average): ~300,000 views/second

### Bandwidth
- Average streaming bitrate: 3 Mbps (mix of 720p/1080p)
- Concurrent viewers at peak: ~10M
- Egress bandwidth: 10M * 3 Mbps = 30 Tbps
- CDN handles ~95% of traffic → origin bandwidth: 1.5 Tbps

### Metadata Storage
- Videos: 800M videos * 1KB metadata = 800GB
- Users: 2B users * 500B = 1TB
- Comments: 5B comments * 500B = 2.5TB
- Likes: 50B likes * 16B = 800GB
- Subscriptions: 10B subscriptions * 16B = 160GB

### QPS Breakdown
- Video search QPS: 500K/s
- Video view increment: 100K/s
- Comment writes: 10K/s
- Like operations: 50K/s

---

## 4. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           CLIENT LAYER                                   │
│           Web Browser / Mobile App / Smart TV / Embedded Player          │
└──────────────────────┬──────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        LOAD BALANCER / CDN EDGE                          │
│              Cloudflare / Akamai / AWS CloudFront (Global PoPs)          │
└──────┬───────────────┬──────────────────────┬───────────────────────────┘
       │               │                      │
       ▼               ▼                      ▼
┌──────────────┐ ┌──────────────┐   ┌────────────────────┐
│  Upload API  │ │  Stream API  │   │   Read/Metadata API │
│  Service     │ │  Service     │   │   Service           │
└──────┬───────┘ └──────┬───────┘   └────────┬───────────┘
       │               │                     │
       ▼               ▼                     ▼
┌──────────────────────────────────────────────────────────────┐
│                      MESSAGE QUEUE (Kafka)                    │
│   video.upload.raw | video.processing | view.events          │
│   comment.events  | notification.events | search.index       │
└──────────────────────┬───────────────────────────────────────┘
                       │
       ┌───────────────┼───────────────────────┐
       ▼               ▼                       ▼
┌─────────────┐  ┌──────────────┐    ┌──────────────────┐
│  Video      │  │  View Count  │    │  Search Indexer  │
│  Processing │  │  Aggregator  │    │  (Elasticsearch) │
│  Workers    │  │              │    │                  │
└──────┬──────┘  └──────┬───────┘    └──────────────────┘
       │               │
       ▼               ▼
┌─────────────┐  ┌──────────────┐
│  Object     │  │  Redis       │
│  Storage    │  │  Cluster     │
│  (S3/GCS)   │  │              │
└──────┬──────┘  └──────────────┘
       │
       ▼
┌─────────────────────┐
│   CDN Distribution  │
│   (HLS Segments)    │
└─────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    DATA LAYER                                │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐   │
│  │  PostgreSQL  │  │  Redis       │  │  Elasticsearch │   │
│  │  (Videos,    │  │  (Cache,     │  │  (Search)      │   │
│  │   Users,     │  │   Sessions,  │  │                │   │
│  │   Likes)     │  │   View Cnts) │  └────────────────┘   │
│  └──────────────┘  └──────────────┘                        │
│  ┌──────────────┐  ┌──────────────┐                        │
│  │  Cassandra   │  │  HBase       │                        │
│  │  (Comments,  │  │  (Watch      │                        │
│  │   Activity)  │  │   History)   │                        │
│  └──────────────┘  └──────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Component Deep-Dive

### 5.1 Video Upload Pipeline

**Chunked Upload Flow:**
1. Client requests upload URL from Upload API (multipart upload initialization)
2. Server generates pre-signed S3 URL and upload session ID
3. Client splits video into 5-10MB chunks
4. Each chunk uploaded directly to S3 (bypasses application servers)
5. Client signals completion — server verifies all chunks received
6. S3 triggers event → Kafka `video.upload.raw` topic
7. Video Processing Workers consume the event and begin transcoding

**Why Chunked Upload?**
- Resilient to network interruptions (resume from last chunk)
- Parallel chunk uploads improve throughput
- Client-side memory efficient
- Can validate chunks incrementally

### 5.2 Video Processing Pipeline (State Machine)

```
UPLOADED → QUEUED → TRANSCODING → THUMBNAIL_GEN → UPLOADING_CDN → READY
                                                                 ↓
                                                              FAILED (retry)
```

**Transcoding:**
- Input: Raw video (any format: .mov, .avi, .mkv, .mp4)
- Output profiles generated:
  - 360p @ 500 Kbps (H.264)
  - 480p @ 1 Mbps (H.264)
  - 720p @ 2.5 Mbps (H.264)
  - 1080p @ 5 Mbps (H.264/H.265)
  - 4K @ 15 Mbps (H.265/AV1)
- Tool: FFmpeg running in Docker containers on GPU instances

**HLS Segmentation:**
- Each resolution split into 10-second .ts segments
- Master playlist (m3u8) referencing all quality variants
- Segments uploaded to S3 → replicated to CDN edge nodes

**Thumbnail Generation:**
- Extract frames at 10%, 25%, 50%, 75% of video duration
- Run ML model to score "visual quality" and "interestingness"
- Store top 3 candidates → let creator choose or auto-select

### 5.3 Adaptive Bitrate Streaming (HLS/DASH)

- Player requests master playlist from CDN
- Player monitors download speed and buffer level
- Selects appropriate quality variant segment
- Seamlessly switches quality between segments
- Target buffer: 30 seconds ahead
- Stall threshold: <3 seconds buffer triggers quality downgrade

### 5.4 View Count at Scale

**Problem:** 100K view increments/second — direct DB writes would overload PostgreSQL.

**Solution: Redis INCR + Batch Flush**
```
Request → Redis INCR video:{video_id}:views (atomic O(1))
         ↓
Background Job (every 60s): 
  - Read all video view keys from Redis
  - Batch UPDATE PostgreSQL (views += redis_count)
  - Reset Redis counter (or use sliding window)
```

**Edge Cases:**
- Redis restart: use Redis persistence (AOF) or accept minor loss
- Video goes viral: Redis handles 100K+ INCR/s easily
- Consistency: show Redis count for real-time, DB for historical

### 5.5 Recommendation Engine

**Two-Phase Approach:**

**Phase 1: Candidate Generation**
- Collaborative Filtering: "Users who watched X also watched Y"
- Content-Based: Similar tags/category/channel
- Watch History: Videos from subscribed channels not yet watched
- Trending: High view velocity in last 24 hours

**Phase 2: Ranking**
- Feature vector per (user, video) pair:
  - CTR on thumbnail
  - Watch percentage (completion rate)
  - Time since publish (freshness)
  - User-channel affinity (past engagement)
- ML ranker (logistic regression / neural net) scores candidates
- Top-K served to user

**Storage for Recommendations:**
- Watch history: HBase (wide-column, time-series)
- Precomputed recommendations: Redis (TTL 1 hour)
- Item-item similarity matrix: offline computed, stored in S3/BigTable

### 5.6 Search

- **Indexing:** Elasticsearch cluster, indexed fields: title, description, tags, transcript (auto-generated)
- **Query Processing:** Tokenize → remove stop words → stem → multi-match query
- **Ranking factors:** Text relevance (BM25) + view count boost + recency + personalization
- **Autocomplete:** Elasticsearch completion suggester on title prefix
- **Index sharding:** Shard by video_id hash, 20 shards with 2 replicas each

### 5.7 Comment System

- **Storage:** Cassandra (high write throughput, time-series access)
- **Schema:** Partition key = video_id, clustering key = (timestamp DESC, comment_id)
- **Nested comments:** Store parent_comment_id, fetch top-level + paginate replies
- **Moderation:** Async ML classifier in Kafka pipeline flags inappropriate content

### 5.8 Subscription Feed Generation

**Fan-out on Write (for small channels):**
- When creator posts → write video_id to each subscriber's feed table
- Fast reads, expensive writes for mega-channels (100M subscribers)

**Fan-out on Read (for large channels):**
- Pull videos from subscribed channels at read time
- Merge-sort results by timestamp
- Cache aggregated feed in Redis for 5 minutes

**Hybrid Approach:**
- Channels < 1M subscribers: fan-out on write
- Channels >= 1M subscribers: fan-out on read
- Merge both sources at read time

---

## 6. Database Design

### PostgreSQL Schema

```sql
-- Videos table
CREATE TABLE videos (
    video_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES users(user_id),
    title           VARCHAR(100) NOT NULL,
    description     TEXT,
    status          ENUM('uploading','processing','ready','failed'),
    duration_secs   INTEGER,
    view_count      BIGINT DEFAULT 0,
    like_count      INTEGER DEFAULT 0,
    dislike_count   INTEGER DEFAULT 0,
    thumbnail_url   VARCHAR(500),
    s3_key          VARCHAR(500),
    hls_manifest    VARCHAR(500),
    tags            TEXT[],
    category_id     INTEGER,
    language        CHAR(5),
    created_at      TIMESTAMP DEFAULT NOW(),
    published_at    TIMESTAMP,
    INDEX(user_id),
    INDEX(created_at DESC),
    INDEX(view_count DESC)
);

-- Users table
CREATE TABLE users (
    user_id         UUID PRIMARY KEY,
    username        VARCHAR(50) UNIQUE NOT NULL,
    email           VARCHAR(255) UNIQUE NOT NULL,
    channel_name    VARCHAR(100),
    subscriber_count BIGINT DEFAULT 0,
    avatar_url      VARCHAR(500),
    created_at      TIMESTAMP DEFAULT NOW()
);

-- Subscriptions table
CREATE TABLE subscriptions (
    subscriber_id   UUID NOT NULL REFERENCES users(user_id),
    channel_id      UUID NOT NULL REFERENCES users(user_id),
    subscribed_at   TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (subscriber_id, channel_id),
    INDEX(channel_id)
);

-- Likes table (separate for scalability)
CREATE TABLE video_likes (
    user_id         UUID NOT NULL,
    video_id        UUID NOT NULL,
    is_like         BOOLEAN NOT NULL,  -- TRUE=like, FALSE=dislike
    created_at      TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (user_id, video_id)
);
```

### Cassandra Schema (Comments)

```
CREATE TABLE comments (
    video_id    UUID,
    created_at  TIMESTAMP,
    comment_id  UUID,
    user_id     UUID,
    content     TEXT,
    parent_id   UUID,  -- NULL for top-level
    like_count  INT,
    PRIMARY KEY (video_id, created_at, comment_id)
) WITH CLUSTERING ORDER BY (created_at DESC);
```

### Redis Keys

```
video:{video_id}:views          → Integer (pending view count)
video:{video_id}:metadata       → Hash (title, duration, status) TTL 1h
user:{user_id}:feed             → Sorted Set (score=timestamp, member=video_id)
user:{user_id}:watch_history    → List (video_ids, max 1000)
search:autocomplete:{prefix}    → Sorted Set (score=popularity, member=title)
processing:{video_id}:status    → String (state) TTL 24h
```

---

## 7. API Design

### Upload API
```
POST /api/v1/videos/upload/initiate
Request:  { title, description, file_size, content_type }
Response: { upload_id, upload_url, chunk_size }

PUT /api/v1/videos/upload/{upload_id}/chunk/{chunk_number}
Body: Binary chunk data
Response: { chunk_received: true, etag }

POST /api/v1/videos/upload/{upload_id}/complete
Request:  { parts: [{part_number, etag}] }
Response: { video_id, status: "processing" }
```

### Video API
```
GET /api/v1/videos/{video_id}
Response: { video_id, title, description, hls_manifest_url, thumbnail_url,
            view_count, like_count, channel_info, comments_url }

GET /api/v1/videos/{video_id}/stream
Query: ?quality=auto|360p|480p|720p|1080p
Response: Redirect to CDN HLS manifest URL

POST /api/v1/videos/{video_id}/view
Response: { view_count: 1234567 }

POST /api/v1/videos/{video_id}/like
Request:  { action: "like"|"dislike"|"remove" }
Response: { likes: 10000, dislikes: 500 }
```

### Comments API
```
GET /api/v1/videos/{video_id}/comments?page_token=xxx&limit=20
Response: { comments: [...], next_page_token }

POST /api/v1/videos/{video_id}/comments
Request:  { content, parent_comment_id? }
Response: { comment_id, created_at }
```

### Recommendations API
```
GET /api/v1/recommendations?limit=20
Response: { videos: [{video_id, title, thumbnail, channel, views, duration}] }
```

### Search API
```
GET /api/v1/search?q=python+tutorial&sort=relevance&filter=duration:short
Response: { results: [...], total: 1000000, next_page_token }
```

---

## 8. Scalability & Bottlenecks

### Identified Bottlenecks

| Component | Bottleneck | Solution |
|-----------|-----------|----------|
| Upload | Single upload endpoint | Direct S3 multipart, presigned URLs |
| Transcoding | CPU-intensive, slow | Horizontal scaling, GPU workers, parallel processing |
| View counting | 100K writes/sec to DB | Redis INCR + batch flush |
| CDN bandwidth | 30 Tbps egress | Multi-CDN, regional PoPs, cache everything |
| Search | High QPS on Elasticsearch | Read replicas, query caching in Redis |
| Recommendations | Expensive ML computation | Precompute hourly, serve from cache |
| Feed generation | Mega-channel fan-out | Hybrid fan-out strategy |

### Scaling Strategies

**Video Processing:**
- Auto-scaling processing worker pool (EC2 Spot Instances for cost)
- Priority queue: paid/partner channels get faster processing
- GPU instances for H.265/AV1 encoding (10x faster than CPU)

**Database Scaling:**
- PostgreSQL read replicas for video metadata reads (20:1 read/write ratio)
- Cassandra natural horizontal scaling (partition by video_id)
- Redis Cluster with consistent hashing

**CDN:**
- Multiple CDN providers (AWS CloudFront + Akamai) for redundancy
- Cache-Control headers: `max-age=31536000` for video segments (immutable)
- Origin shield to reduce S3 origin fetches

---

## 9. Trade-offs & Design Decisions

### Decision 1: HLS vs DASH for Adaptive Streaming
- **HLS:** Better iOS/Apple support, wider client compatibility
- **DASH:** Open standard, better flexibility, smaller overhead
- **Choice:** Support both HLS and DASH, serve HLS to iOS, DASH to Android/Web
- **Trade-off:** 2x storage for manifests, unified CDN delivery

### Decision 2: View Count Consistency
- **Strong consistency:** Direct DB write → too slow (100K/s)
- **Eventual consistency (Redis → DB):** Slight inaccuracy (~60s lag), vastly better performance
- **Choice:** Eventual consistency — viewers don't notice 60-second delay
- **Trade-off:** Minor inaccuracy vs massive scalability gain

### Decision 3: Fan-out Strategy for Subscriptions
- **Fan-out on write:** Fast reads, expensive writes for mega-channels
- **Fan-out on read:** Slow reads, cheap writes
- **Choice:** Hybrid based on channel subscriber count
- **Trade-off:** System complexity vs performance optimization

### Decision 4: SQL vs NoSQL for Comments
- **SQL (PostgreSQL):** ACID, complex queries, harder to scale writes
- **NoSQL (Cassandra):** High write throughput, time-series optimized, eventual consistency
- **Choice:** Cassandra — comments are write-heavy and time-series by nature
- **Trade-off:** No cross-video comment queries, eventual consistency

### Decision 5: Monolith vs Microservices
- **Choice:** Microservices — Upload, Processing, Streaming, Search, Recommendation services
- **Trade-off:** Operational complexity vs independent scaling and deployment

---

## 10. Key Interview Talking Points

1. **Video Upload Pipeline:** Explain chunked multipart upload to S3 with pre-signed URLs, why we bypass the application server for large binary uploads.

2. **Transcoding Workers:** Kafka decouples upload from processing. Workers are stateless and horizontally scalable. State machine tracks processing stages. Failed jobs go to dead-letter queue with retry.

3. **HLS Adaptive Streaming:** Client-driven quality selection based on bandwidth measurement. 10-second segments allow fine-grained quality switching. CDN serves segments — origin never touched during playback.

4. **View Count at Scale:** Redis INCR is O(1) and supports 1M+ ops/second per node. Batch flush every 60 seconds reduces DB writes by 6000x. Acceptable for use case — YouTube's view counts are also approximate.

5. **CDN Architecture:** Video segments are immutable (content-addressed), making CDN caching extremely effective. 95%+ cache hit rate. Push vs pull strategy — use pull (lazy caching) for long-tail videos.

6. **Recommendation System:** Two-phase: candidate generation (fast, approximate) then ranking (slower, personalized). Precompute offline, serve from cache. Key metrics: CTR, watch time, satisfaction signal.

7. **Search at Scale:** Elasticsearch with BM25 for text ranking. Personalization layer re-ranks results based on user history. Autocomplete uses prefix index with popularity scores.

8. **Database Choices:** PostgreSQL for structured metadata (ACID for billing-critical data), Cassandra for time-series comments (write-heavy), Redis for caching and real-time counts, HBase for watch history (wide-column, append-only).

9. **Fault Tolerance:** Every queue is persistent (Kafka with replication). S3 provides 11 nines durability. Processing failures trigger retry with exponential backoff. CDN failover between providers.

10. **Back-of-Envelope:** Always mention: 500 hours video/min upload → 288TB/day storage growth, 1B hours/day watching → 100K views/second, 30 Tbps CDN bandwidth. These numbers justify CDN-first architecture.
