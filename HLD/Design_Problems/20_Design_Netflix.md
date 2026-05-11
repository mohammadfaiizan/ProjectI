# Design Netflix — Video Streaming Platform

---

## 1. Problem Statement & Clarifying Questions

Design a large-scale video streaming service like Netflix that allows users to browse content, stream high-quality video across various devices, and receive personalized recommendations.

### Clarifying Questions

| Question | Assumption |
|---|---|
| Number of subscribers? | 220M paying subscribers globally |
| Concurrent streams at peak? | 15M concurrent streams |
| Video catalog size? | 15,000 titles (shows + movies) |
| Do we need live streaming? | No — on-demand streaming only |
| Device support? | TV, web browser, iOS, Android, Smart TVs |
| Do we need DRM? | Yes — Widevine (Android/Chrome), FairPlay (Apple) |
| Recommendation system? | Yes — personalized, A/B tested |
| Multiple profiles per account? | Yes — up to 5 profiles |
| Offline downloads? | Yes — for mobile apps |
| Resolution support? | 480p, 720p, 1080p, 4K HDR |

---

## 2. Functional & Non-Functional Requirements

### Functional Requirements
1. **Content Browsing** — Browse catalog by genre, trending, personalized rows
2. **Video Streaming** — Adaptive bitrate streaming with seamless quality switching
3. **Continue Watching** — Resume playback at the exact position across devices
4. **Search** — Search by title, cast, genre, description
5. **Recommendations** — Personalized content rows (10+ algorithms running simultaneously)
6. **User Profiles** — Up to 5 profiles per account with separate watch histories
7. **Like/Dislike** — Signal for recommendation tuning
8. **Download for Offline** — Mobile downloads with DRM-encrypted content
9. **A/B Testing** — Different recommendation algorithms, thumbnail images, UI layouts

### Non-Functional Requirements
1. **Availability** — 99.99% uptime (no interruption during peak viewing hours)
2. **Latency** — Video start time < 2 seconds; stream switch time < 500ms
3. **Scalability** — 15M concurrent streams; Netflix = 1/3 of US internet traffic at peak
4. **Durability** — No content loss; originals stored in multiple regions
5. **Consistency** — Watch position eventually consistent across devices (< 10 seconds)

---

## 3. Capacity Estimation

### Storage
- 15,000 titles × average 2 hours
- Each title encoded in 5 quality levels: 480p (0.7 Mbps), 720p (3 Mbps), 1080p (6 Mbps), 4K (15 Mbps), 4K HDR (20 Mbps)
- Average bitrate: 9 Mbps, 2 hours = 8.1 GB per title per quality
- 15,000 × 5 qualities × 8.1 GB = **608 TB** of encoded video content
- With audio tracks (5 languages) + subtitles: ~1.5 PB total

### Bandwidth
- 15M concurrent streams × 5 Mbps average = **75 Gbps** total egress
- Netflix accounts for ~34% of US downstream internet traffic at peak (8-11 PM EST)
- CDN delivers 99%+ of traffic; origin serves only cache misses

### Watch History
- 220M users × 200 watched titles = 44B records
- Each record: 200 bytes (title_id, user_id, position, timestamp) = **8.8 TB** total

### Metadata Storage
- Titles: 15,000 × 50 KB (full metadata, cast, images) = 750 MB
- Trivially small — fits in memory; cached globally

---

## 4. High-Level Architecture

```
          ┌─────────────────────────────────────────────┐
          │              Client Devices                  │
          │  (TV App / Browser / iOS / Android)          │
          └──────────────┬──────────────────────────────┘
                         │ HTTPS API + Video Streaming
          ┌──────────────▼──────────────────────────────┐
          │         API Gateway (AWS / Netflix Edge)     │
          └────┬───────┬──────────┬───────┬─────────────┘
               │       │          │       │
      ┌────────▼──┐ ┌──▼──────┐ ┌▼─────┐ ┌▼──────────────┐
      │ Catalog   │ │Recommend│ │Search│ │  Playback /   │
      │ Service   │ │  Engine │ │ Svc  │ │  Session Svc  │
      └────┬──────┘ └──┬──────┘ └──┬───┘ └──────┬────────┘
           │           │           │             │
   ┌───────▼──┐  ┌──────▼──┐  ┌────▼──┐  ┌──────▼────────┐
   │ Postgres │  │ Offline  │  │ Elastic│  │   Cassandra  │
   │(catalog, │  │ Reco     │  │ Search │  │  (watch hx,  │
   │ metadata)│  │ Store    │  │ Index  │  │   sessions)  │
   └──────────┘  │(DynamoDB)│  └────────┘  └──────────────┘
                 └──────────┘
                                     ┌────────────────────┐
                                     │   Redis Cluster    │
                                     │  (sessions, cache) │
                                     └────────────────────┘

  Video Delivery:
  ┌──────────────────────────────────────────────────────┐
  │                  Content Pipeline                     │
  │  S3 (raw) → Encoding Farm → S3 (encoded) → CDN       │
  │                                                      │
  │  Netflix Open Connect (ISP-embedded appliances)      │
  │  + Akamai (fallback)                                 │
  │  ──────────────────────────────────────────────      │
  │  Client requests manifest (m3u8/mpd) → CDN           │
  │  Client selects quality → fetches segments from CDN  │
  └──────────────────────────────────────────────────────┘
```

---

## 5. Component Deep-Dive

### 5.1 Video Encoding Pipeline

When content is uploaded (Netflix Originals) or licensed:

```
1. Ingest raw video → S3 (source store)
2. Encoding job scheduler creates jobs per quality/codec combination:
   - 480p H.264, 720p H.264, 1080p H.264
   - 1080p H.265 (better compression)
   - 4K AV1 (best compression, newer devices only)
3. Encoding workers (GPU-based, auto-scaling) transcode in parallel
4. Each video split into 4-second segments
5. Encoded segments uploaded to S3
6. CDN pre-populates from S3 (push-based for popular content)
```

**Netflix-specific optimization:** Per-title encoding (Netflix Originals). Analyze each scene's complexity, vary the bitrate dynamically within each quality tier. Result: 20% better visual quality at same file size.

### 5.2 Adaptive Bitrate Streaming (ABR)

ABR allows the client to switch video quality in real-time based on network conditions.

**HLS (HTTP Live Streaming) — Apple/iOS:**
```
Master Playlist (m3u8):
  #EXT-X-STREAM-INF:BANDWIDTH=800000,RESOLUTION=640x480
  480p/index.m3u8
  #EXT-X-STREAM-INF:BANDWIDTH=3000000,RESOLUTION=1280x720
  720p/index.m3u8
  #EXT-X-STREAM-INF:BANDWIDTH=6000000,RESOLUTION=1920x1080
  1080p/index.m3u8

Quality Playlist (720p/index.m3u8):
  #EXTINF:4.0
  seg_000.ts
  #EXTINF:4.0
  seg_001.ts
  ...
```

**ABR Algorithm (simplified buffer-based):**
```
if buffer_level > HIGH_WATERMARK:
    switch_up_quality()       # Buffer healthy, can afford higher bitrate
elif buffer_level < LOW_WATERMARK:
    switch_down_quality()     # Risk of stalling, reduce bitrate
else:
    maintain_current_quality()
```

### 5.3 Netflix Open Connect CDN

Netflix's private CDN embedded inside ISP networks:
- Open Connect Appliances (OCAs) — servers placed at ISP data centers
- ISPs host for free in exchange for reduced peering costs
- ~95% of traffic served from OCAs; ~5% from Akamai
- OCAs pre-populated nightly with top-N content for that ISP's users
- Result: content travels 0-1 hops to user (vs. 15-20 hops from origin)

**Edge Node Selection:**
1. Client requests streaming manifest from Netflix API
2. API returns CDN URL pointing to nearest OCA
3. OCA serves video segments directly
4. If OCA cache miss: fetch from S3 origin, cache locally

### 5.4 Recommendation System

Netflix runs multiple ML models simultaneously:
1. **Collaborative Filtering (Matrix Factorization):** User-item interaction matrix decomposed into latent factors. Pre-computed nightly with Spark ML.
2. **Content-Based Filtering:** Recommend similar content based on genre, cast, director.
3. **Trending:** Popular content in your country/region.
4. **Continue Watching:** Personalized re-engagement with incomplete content.
5. **Because You Watched X:** Item-item similarity.
6. **A/B Tested Ranking:** Multiple ranking algorithms compete; better CTR/completion rate wins.

**Thumbnail A/B Testing:** Netflix tests different thumbnail images per title per user segment. A user who watches action movies sees an action-heavy thumbnail; a user who watches romance sees a different frame.

### 5.5 Watch History & Continue Watching

Every 30 seconds during playback, client sends:
```json
{ "title_id": "breaking_bad_s01e01", "profile_id": "uuid", "position_sec": 1247 }
```

Stored in Cassandra: partition by profile_id, cluster by timestamp.
Continue Watching view: query Cassandra for all titles watched but not completed (< 90% watched), sorted by most recently watched.

### 5.6 DRM (Content Protection)

- **Widevine** (Google): Used on Android, Chrome, ChromeCast
- **FairPlay** (Apple): Used on iOS, macOS, tvOS, Safari
- **PlayReady** (Microsoft): Used on Edge, Xbox

Each client requests a license key from the DRM license server. License:
- Authorizes playback for this user session
- Specifies quality limits (SD/HD/4K) based on subscription tier
- Time-limited (must refresh periodically)

### 5.7 A/B Testing Infrastructure

Netflix runs 300+ experiments simultaneously:
- Every user is assigned to experiment buckets (hash of user_id + experiment_id)
- Assignment is stable: same user sees the same variant consistently
- Metrics collected: CTR, stream start time, completion rate, thumbs up/down
- Winning variant rolled out gradually (10% → 25% → 50% → 100%)

---

## 6. Database Design

### 6.1 Shows (PostgreSQL)
```sql
CREATE TABLE shows (
    show_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title         VARCHAR(500) NOT NULL,
    type          ENUM('movie','series') NOT NULL,
    description   TEXT,
    release_year  INT,
    maturity_rating VARCHAR(10),
    genres        TEXT[],
    cast_members  TEXT[],
    director      VARCHAR(200),
    country       VARCHAR(100),
    is_original   BOOLEAN DEFAULT FALSE,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);
```

### 6.2 Episodes (PostgreSQL)
```sql
CREATE TABLE episodes (
    episode_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    show_id       UUID REFERENCES shows(show_id),
    season_num    INT,
    episode_num   INT,
    title         VARCHAR(300),
    description   TEXT,
    duration_sec  INT,
    UNIQUE (show_id, season_num, episode_num)
);
```

### 6.3 Watch History (Cassandra)
```cql
CREATE TABLE watch_history (
    profile_id    UUID,
    watched_at    TIMESTAMP,
    show_id       UUID,
    episode_id    UUID,
    position_sec  INT,           -- Resume position
    duration_sec  INT,
    completed     BOOLEAN,
    PRIMARY KEY (profile_id, watched_at)
) WITH CLUSTERING ORDER BY (watched_at DESC);
```

### 6.4 User Profiles (PostgreSQL)
```sql
CREATE TABLE user_profiles (
    profile_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id    UUID NOT NULL,
    name          VARCHAR(100),
    avatar_url    VARCHAR(500),
    is_kids       BOOLEAN DEFAULT FALSE,
    language      VARCHAR(10) DEFAULT 'en',
    created_at    TIMESTAMPTZ DEFAULT NOW()
);
```

### 6.5 Ratings (PostgreSQL)
```sql
CREATE TABLE ratings (
    profile_id    UUID,
    show_id       UUID,
    rating        SMALLINT CHECK (rating IN (-1, 1)),   -- -1 = dislike, 1 = like
    rated_at      TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (profile_id, show_id)
);
```

### 6.6 Experiment Assignments (Redis / Cassandra)
```cql
CREATE TABLE experiment_assignments (
    profile_id    UUID,
    experiment_id TEXT,
    variant       TEXT,
    assigned_at   TIMESTAMP,
    PRIMARY KEY (profile_id, experiment_id)
);
```

---

## 7. API Design

### Stream Video
```
GET /api/v1/titles/{title_id}/playback-manifest
Response: {
  manifest_url: "https://cdn.netflix.com/title123/manifest.mpd",
  drm_license_url: "https://widevine.netflix.com/license",
  resume_position: 1247,
  session_id: "uuid"
}
```

### Record Watch Progress
```
PUT /api/v1/profiles/{profile_id}/watch-progress
Body: { title_id, episode_id, position_sec, duration_sec }
Response: { saved: true }
```

### Get Recommendations
```
GET /api/v1/profiles/{profile_id}/home
Response: {
  rows: [
    { title: "Continue Watching", items: [...] },
    { title: "Top 10 in Your Country", items: [...] },
    { title: "Because You Watched Breaking Bad", items: [...] }
  ]
}
```

### Search
```
GET /api/v1/search?q={query}&profile_id={id}
Response: { results: [{show_id, title, type, year, match_score}] }
```

### Like/Dislike
```
POST /api/v1/profiles/{profile_id}/ratings
Body: { show_id, rating: 1 | -1 }
Response: { show_id, rating }
```

---

## 8. Scalability & Bottlenecks

### Bottleneck 1: CDN Cache Miss for New Content
- **Problem:** New episode of Stranger Things — millions request it simultaneously; CDN has no cache
- **Solution:** Pre-warm CDN proactively. When a new episode is published, Netflix pushes content to OCAs before the release time. Scheduled pre-population from S3 to top-100 OCAs 30 minutes before release.

### Bottleneck 2: Watch Progress Write Storm
- **Problem:** 15M concurrent streams × 1 write/30 sec = 500K writes/sec to Cassandra
- **Solution:** Cassandra designed for this — write-optimized LSM tree, partition by profile_id distributes writes evenly. 20 Cassandra nodes × 25K writes/sec each = 500K writes/sec. Batched writes per stream.

### Bottleneck 3: Recommendation Personalization at Scale
- **Problem:** Pre-compute recommendations for 220M users × 15K titles is O(3.3T) computations
- **Solution:** Matrix factorization (ALS algorithm) reduces to lower-dimensional latent factors. Offline Spark job nightly. Online serving: pre-computed top-K recommendations stored in DynamoDB, served in < 10ms.

### Bottleneck 4: Search Performance
- **Problem:** 220M users doing catalog search
- **Solution:** Elasticsearch cluster. Catalog is only 15K titles — index is tiny (< 1 GB). Easy to replicate across all regions. All searches served from memory.

### Bottleneck 5: Startup Latency
- **Problem:** User hits play → video should start in < 2 seconds
- **Solution:**
  - DNS resolution → nearest OCA pre-selected
  - Manifest pre-fetched when user hovers/highlights title (pre-fetch on intent)
  - First segment (4 seconds of video) pre-buffered
  - Start at lowest quality (480p) → ramp up within 2-3 segments

---

## 9. Trade-offs & Design Decisions

### Decision 1: HLS vs. DASH
- **HLS:** Apple's format, native iOS/macOS/Safari support. Wide device support.
- **DASH (Dynamic Adaptive Streaming over HTTP):** Open standard, supported by Chrome, Android, Smart TVs. Better codec flexibility (VP9, AV1 support).
- **Choice:** Netflix supports both (DASH for most devices, HLS for Apple). Master manifest auto-negotiates format per device.

### Decision 2: PostgreSQL vs. Cassandra for Watch History
- **PostgreSQL:** ACID, but 500K writes/sec is challenging. Requires massive sharding.
- **Cassandra:** Write-optimized, partition by profile_id gives perfect write distribution. No ACID but eventual consistency acceptable (watch progress).
- **Choice:** Cassandra for watch history. The access pattern (write-heavy, query by profile_id) is Cassandra's sweet spot.

### Decision 3: Push vs. Pull CDN Pre-population
- **Pull (lazy):** Content pushed to CDN on first request. Simple, cold start latency.
- **Push (proactive):** Netflix pre-populates OCAs for popular content before release. Higher operational complexity.
- **Choice:** Push for Netflix Originals (known release time, high demand). Pull with long TTL for catalog tail content.

### Decision 4: Segment Size (2s vs. 4s vs. 10s)
- **Small segments (2s):** Fast quality switches, more HTTP requests overhead
- **Large segments (10s):** Fewer requests, slower quality adaptation
- **Choice:** 4-second segments. Balance between adaptation speed and HTTP overhead. Netflix uses 4s for most content.

### Decision 5: Offline Download DRM
- **No offline:** Simplest. But mobile users demand offline access.
- **Offline with DRM:** Complex license management. Time-limited download licenses (expire in 30 days or when subscription lapses).
- **Choice:** Offline download with time-limited Widevine/FairPlay licenses. Download count limited per title (varies by studio licensing deals).

---

## 10. Key Interview Talking Points

1. **Adaptive Bitrate is Client-Side Intelligence:** The server serves static segments. The client decides which quality to request based on its buffer level and measured throughput. No server-side logic needed per stream — this scales infinitely.

2. **Netflix Open Connect is a CDN Competitive Moat:** By placing hardware inside ISPs, Netflix achieves sub-millisecond network proximity to end users. It also shifts costs from Netflix (peering fees) to ISPs (who get lower transit costs in exchange). This is a business model innovation, not just technical.

3. **Encoding is a Multiplier, Not Storage:** Every title encoded 5 qualities × 3 codecs = 15 variants. But those 15 variants enable 15M concurrent streams without any per-stream computation. The work is front-loaded at upload time.

4. **Cassandra for Watch History — Perfect Fit:** Profile-scoped queries, append-only writes, eventual consistency acceptable. Partition by profile_id → 220M partitions → perfect write distribution. No hot partitions.

5. **Recommendation is Multiple Models in Parallel:** Netflix doesn't run "one recommendation algorithm." They run 10-20 simultaneously (collaborative filtering, content-based, trending, etc.) and use a ranking model to select and order the best results for each row.

6. **A/B Testing as Core Infrastructure:** Netflix makes almost no product decisions without A/B testing. Thumbnail A/B testing (different thumbnails for different user segments) is a major engagement driver. The infrastructure assigns users to stable experiment buckets via consistent hashing.

7. **DRM License Server as Critical Path:** When play is pressed, a DRM license must be fetched before decryption can start. This license server is single-point-of-failure risk. Netflix runs multiple license server replicas globally with aggressive caching.

8. **Continue Watching is Business-Critical:** "Continue Watching" drives re-engagement. It requires Cassandra to return all incomplete watches for a profile — requires a secondary index or dedicated partition. Netflix materializes this separately for fast access.

9. **Cold Start for New Content:** New originals don't appear in collaborative filter (no one has watched them). Solution: content-based similarity (same genre/cast as content user already likes), editorial curation, promotional placement, and popularity boost in ranking.

10. **Scale Numbers:** 220M subscribers, 15M concurrent streams, 75 Gbps egress, 34% of US internet traffic. 500K watch-progress writes/sec to Cassandra. 608 TB encoded video. CDN serves 99%+ → origin sees < 1% of traffic = ~0.75 Gbps from origin.
