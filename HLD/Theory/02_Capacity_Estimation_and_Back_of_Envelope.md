# Capacity Estimation and Back-of-Envelope Calculations

> The numbers every engineer must know cold. These estimates shape architectural decisions — whether you need caching, sharding, CDN, or a message queue is determined by these calculations.

---

## Latency Numbers Every Engineer Must Know

These numbers were originally from Jeff Dean's landmark talk at Google. Memorize the order-of-magnitude relationships.

### Core Latency Reference Table

| Operation | Latency | Relative to RAM |
|---|---|---|
| L1 cache reference | 0.5 ns | 1x |
| Branch misprediction | 5 ns | 10x |
| L2 cache reference | 7 ns | 14x |
| Mutex lock/unlock | 25 ns | 50x |
| Main memory (RAM) reference | 100 ns | 200x |
| Compress 1KB with Snappy | 3,000 ns (3 µs) | 6,000x |
| Send 1KB over 1 Gbps network | 10,000 ns (10 µs) | 20,000x |
| Read 4KB randomly from SSD | 150,000 ns (150 µs) | 300,000x |
| Read 1MB sequentially from RAM | 250,000 ns (250 µs) | 500,000x |
| Round-trip within same datacenter | 500,000 ns (0.5 ms) | 1,000,000x |
| Read 1MB sequentially from SSD | 1,000,000 ns (1 ms) | 2,000,000x |
| HDD seek | 10,000,000 ns (10 ms) | 20,000,000x |
| Read 1MB sequentially from HDD | 20,000,000 ns (20 ms) | 40,000,000x |
| Send packet CA → Netherlands → CA | 150,000,000 ns (150 ms) | 300,000,000x |

### Key Takeaways from Latency Numbers

```
Memory is 200x faster than SSD
SSD is 100x faster than HDD
RAM to RAM within datacenter: ~500µs
Cross-continent: ~150ms (300x slower than same-DC)

Implications:
✓ Cache in RAM — 200x improvement over SSD read
✓ Avoid HDD seeks — use SSD or memory
✓ Batch cross-datacenter calls — each call costs 150ms
✓ Compress before sending over network
```

### Modern Practical Latency (2024 Systems)

| Operation | Typical Latency |
|---|---|
| Redis GET | ~0.1ms (100µs) |
| Redis SET | ~0.1ms (100µs) |
| PostgreSQL query (indexed, cached) | ~1ms |
| PostgreSQL query (uncached, disk) | ~10-100ms |
| Cassandra read | ~1ms |
| Elasticsearch query | ~10-50ms |
| Same-region API call (AWS) | ~1-5ms |
| Cross-region API call | ~50-150ms |
| DNS resolution (cached) | ~0ms (local cache) |
| DNS resolution (full recursion) | ~50-100ms |
| CDN cache hit | ~5-30ms |
| CDN cache miss (origin fetch) | ~50-200ms |

---

## Powers of 2 and Storage Units

### Powers of 2 Reference

| Power | Value | Storage Unit | Approximation |
|---|---|---|---|
| 2^10 | 1,024 | 1 KB | ~1 thousand |
| 2^20 | 1,048,576 | 1 MB | ~1 million |
| 2^30 | 1,073,741,824 | 1 GB | ~1 billion |
| 2^40 | ~1.1 × 10^12 | 1 TB | ~1 trillion |
| 2^50 | ~1.1 × 10^15 | 1 PB | ~1 quadrillion |
| 2^60 | ~1.15 × 10^18 | 1 EB | ~1 quintillion |

### Storage Conversion Quick Reference

```
1 KB = 1,000 bytes (or 1,024 in binary)
1 MB = 1,000 KB = 1,000,000 bytes
1 GB = 1,000 MB = 1,000,000,000 bytes
1 TB = 1,000 GB = 10^12 bytes
1 PB = 1,000 TB = 10^15 bytes

Memory sizes (for estimation):
Average tweet: 280 chars ≈ 280 bytes ≈ 0.28 KB
Average photo (compressed): 300 KB - 3 MB
Average 1-minute video (720p): ~50 MB
Average 1-minute video (1080p): ~150 MB
Average 1-minute audio (MP3): ~1 MB
Typical user profile: ~1 KB
UUID: 16 bytes
Timestamp: 8 bytes
Integer: 4 bytes (32-bit), 8 bytes (64-bit)
```

### Time Conversions for Rate Calculations

```
1 day   = 86,400 seconds ≈ 10^5 seconds (use 100,000)
1 month = 30 days = 2,592,000 seconds ≈ 2.5 × 10^6
1 year  = 365 days = 31,536,000 seconds ≈ 3 × 10^7

Rounding for estimation:
10^5 seconds per day (86,400 ≈ 100,000 — overestimates by ~16%, fine for BOE)
```

---

## Traffic Estimation — DAU to QPS

### The Formula

```
QPS = DAU × requests_per_user_per_day / seconds_per_day

Peak QPS ≈ QPS × 2 to 5 (assume 2x for even distribution, 5x for spiky)
```

### Step-by-Step Calculation

**Example: Twitter-like system**

```
Given:
- 300M MAU (Monthly Active Users)
- 50% DAU/MAU ratio → 150M DAU
- Average user: reads 50 tweets, writes 1 tweet per day
- Tweet: 280 chars, metadata → ~500 bytes

Read QPS:
= 150M DAU × 50 reads/day / 100,000 seconds/day
= 7,500,000,000 / 100,000
= 75,000 QPS (reads)

Write QPS:
= 150M DAU × 1 write/day / 100,000 seconds/day
= 150,000,000 / 100,000
= 1,500 QPS (writes)

Read:Write ratio = 75,000 : 1,500 = 50:1

Peak read QPS (2x multiplier) = 150,000 QPS
Peak write QPS (2x multiplier) = 3,000 QPS
```

### DAU/MAU Ratio Benchmarks

| Product Type | DAU/MAU |
|---|---|
| Highly engaging (WhatsApp, TikTok) | 60-70% |
| Social media (Twitter, Instagram) | 40-50% |
| News/content apps | 20-30% |
| E-commerce | 10-20% |
| Tools (Dropbox, Evernote) | 10-20% |

### QPS Scale and Architecture Implications

| QPS | Architecture Required |
|---|---|
| < 1,000 | Single server, single DB fine |
| 1,000 – 10,000 | Load balancer + multiple app servers, read replicas |
| 10,000 – 100,000 | Caching layer required (Redis), DB optimization |
| 100,000 – 1M | Sharding, multiple cache clusters, CDN |
| > 1M | Distributed architecture, multi-region |

---

## Storage Estimation

### The Formula

```
Storage per day = objects_per_day × avg_size_per_object
Total storage   = storage_per_day × retention_days × replication_factor
```

### Storage Estimation for Twitter

```
New tweets per day:
= 1,500 QPS × 86,400 seconds = 129,600,000 ≈ 130M tweets/day

Storage per tweet:
- tweet_id:    8 bytes
- user_id:     8 bytes  
- content:   280 bytes
- metadata:  ~200 bytes (timestamp, like_count, retweet_count, etc.)
- Total:     ~500 bytes per tweet

Text storage per day:
= 130M × 500 bytes = 65 GB/day

Media (assume 10% of tweets have 1 photo, avg 300KB):
= 130M × 10% × 300 KB = 3.9 TB/day

Total storage per day (text + media):
≈ 3.9 TB/day

5-year storage (no replication):
= 3.9 TB/day × 365 days × 5 years = ~7.1 PB

With 3x replication:
= 7.1 PB × 3 = ~21 PB

Practical note: Twitter actual storage is larger — likes, follows,
search indexes, logs add significant overhead (typically 5-10x raw data)
```

### Storage for User Photos (Instagram-like)

```
Assumptions:
- 100M DAU
- 10% of users upload 1 photo/day = 10M photos/day
- Average photo (compressed): 1 MB
- Keep for 5 years
- 3 replicas

Raw storage/day = 10M × 1 MB = 10 TB/day
5-year storage  = 10 TB × 365 × 5 = 18,250 TB ≈ 18 PB
With replication = 18 PB × 3 = 54 PB

Photos are typically stored in 3 resolutions:
- Original: 1 MB
- Medium (feed): 300 KB  
- Thumbnail: 50 KB
Total per photo: ~1.35 MB

Revised: 54 PB × 1.35 = ~73 PB
```

### Data Size Reference for Estimation

| Data Type | Approximate Size |
|---|---|
| User profile (text) | 1 KB |
| Tweet / short post | 300-500 bytes |
| Email (text only) | 10-50 KB |
| Photo (compressed JPEG) | 200 KB – 2 MB |
| Photo (thumbnail) | 30-100 KB |
| Song (MP3, 3 min) | 3-5 MB |
| Video (1 min, 720p) | 30-50 MB |
| Video (1 min, 1080p) | 100-150 MB |
| Video (1 min, 4K) | 300-400 MB |
| PDF document | 100 KB – 10 MB |

---

## Bandwidth Estimation

### The Formula

```
Read bandwidth  = read_QPS × avg_response_size
Write bandwidth = write_QPS × avg_request_size
```

### Bandwidth for Twitter

```
Read bandwidth:
= 75,000 QPS × 500 bytes/tweet × 10 tweets/request
= 75,000 × 5,000 bytes
= 375,000,000 bytes/second
= 375 MB/s ≈ 3 Gbps

Write bandwidth:
= 1,500 QPS × 500 bytes/tweet
= 750,000 bytes/second
= 750 KB/s ≈ 0.006 Gbps

Note: Read dominates at 500x. CDN is essential for read-heavy systems.
```

### Network Bandwidth Reference

| Connection | Bandwidth |
|---|---|
| Home broadband | 25-100 Mbps |
| 4G LTE mobile | 10-50 Mbps |
| 5G mobile | 100-1000 Mbps |
| 1 GbE server NIC | 1 Gbps = 125 MB/s |
| 10 GbE server NIC | 10 Gbps = 1.25 GB/s |
| 25 GbE server NIC | 25 Gbps |
| AWS cross-AZ | ~25 Gbps per link |
| AWS cross-region | varies, typically limited by app design |
| CDN edge capacity | Terabits/s (aggregate) |

---

## Memory Estimation (Cache Sizing)

### The Formula

```
Cache size = hot_data_percentage × total_data_size

Typical hot data (Pareto principle):
- 20% of data accounts for 80% of reads
- Cache the top 20% = handle 80% of traffic

Cache entries estimate:
Cache size ÷ avg_object_size = number_of_objects_cached
```

### Cache Sizing for Twitter

```
Daily active data (tweets viewed today):
= 75,000 QPS read × 86,400 seconds × 500 bytes/tweet
= 3,240 GB/day ≈ 3.2 TB of tweet data read per day

Hot data (20% rule):
= 3.2 TB × 20% = 640 GB in cache

But we only need recent/popular tweets:
Top tweets (last 24h, high engagement): ~50 GB
Feed cache (per-user pre-computed): much larger

Practical Redis memory:
- Redis stores ~100M keys in 10 GB RAM
- Key overhead: ~50-100 bytes
- Value: depends on stored object

For tweet IDs in feed cache:
- 150M users × 200 tweet IDs × 8 bytes = 240 GB
- (Store only IDs, fetch tweet details on demand)
```

### Cache Hit Rate and Impact

```
Cache hit rate = cache_hits / (cache_hits + cache_misses)

If hit rate = 90%:
- 90% of requests served from cache (sub-millisecond)
- 10% go to database

Database load reduction:
- Without cache: 75,000 QPS to DB
- With 90% cache: 7,500 QPS to DB (10x reduction!)

Impact on cost:
- DB server: ~$1,000/month each
- Redis server: ~$500/month for large cache
- Typically 1 Redis replaces 5-10 DB read replicas
```

---

## Common System Estimates

### Twitter / X

```
Scale (estimated):
- 400M MAU, 200M DAU
- 500M tweets/day (including replies, retweets)
- 5,700 tweets/second average
- Peak: ~10,000-15,000 tweets/second

Storage:
- ~300 TB/day (tweets + media + metadata)
- Total corpus: multiple PB

Infrastructure estimate:
- ~1,500 app servers
- Large distributed cache (multiple TB of hot data)
- Multi-region database sharding
```

### YouTube

```
Scale (estimated):
- 2.7B MAU, 122M DAU
- 500 hours of video uploaded per minute
- 1 billion hours watched per day

Upload calculation:
= 500 hours/min × 60 min/hr × 24 hr/day = 720,000 hours/day
= 720,000 × 1 GB/hour (720p) = 720 TB/day raw
After transcoding to multiple resolutions (360p, 480p, 720p, 1080p, 4K):
≈ 720 TB × 4 = ~3 PB/day new storage

Watch bandwidth:
= 1B hours/day × 60 min/hr × 150 MB/min (720p) / 86,400 sec/day
= 1,000,000,000 × 3600 × 150 MB / 86,400
≈ 6.25 TB/s
= ~50 Tbps total outbound bandwidth

CDN is essential — impossible to serve from origin only.
```

### WhatsApp

```
Scale (estimated):
- 2.5B users, 500M DAU
- 100 billion messages/day
- 1.15M messages/second average
- Peak: ~2-3M messages/second

Message storage:
- Average message: 200 bytes (text) or 500 KB (media)
- Assume 95% text, 5% media
- Text/day: 95B × 200 bytes = 19 TB
- Media/day: 5B × 500 KB = 2.5 PB
- Total: ~2.5 PB/day

Server connections:
- 500M DAU, all maintaining persistent WebSocket connections
- Average connection: ~50 KB RAM
- 500M × 50 KB = 25 TB RAM for connections alone
- Need specialized connection servers (separate from message processing)

Connection servers required:
- Each server handles ~100,000 WebSocket connections (well-tuned)
- 500M / 100,000 = 5,000 connection servers
```

### Uber / Lyft

```
Scale (estimated):
- 130M MAU, 5M daily trips
- Driver location updates: every 4 seconds
- ~5M active drivers in peak

Location update QPS:
= 5M drivers × (1 update / 4 seconds) = 1.25M writes/second

Storage per location update:
- driver_id: 8 bytes
- latitude: 8 bytes
- longitude: 8 bytes
- timestamp: 8 bytes
= 32 bytes/update

Location storage per day:
= 1.25M × 86,400 × 32 bytes = 3.5 TB/day
(But we only need recent locations — can expire old ones)

Geospatial index requirement:
- Need to find nearby drivers within X km radius
- Use geohashing or quadtree
- Hot geohash regions (NYC, London) need aggressive caching
```

### Netflix

```
Scale (estimated):
- 260M subscribers, ~100M concurrent streams peak
- ~15% of global internet traffic (outbound)
- 1,000-3,000 content items added monthly

Streaming bandwidth:
- Average stream: 5 Mbps (1080p adaptive)
- 100M concurrent × 5 Mbps = 500 Tbps
- Netflix uses ~150 Tbps (actual reported) — shows real numbers

Content storage:
- 15,000 titles × average 2 hours
- 2 hours × 60 min × 150 MB/min (1080p) = 18 GB/title
- Multiple resolutions: 1080p, 720p, 480p, 360p, 4K HDR
- Each title stored in ~100 encoding variants
- 15,000 × 18 GB × 100 variants = 27 PB (rough estimate)
```

---

## Back-of-Envelope Worked Examples

### Example 1 — Design a URL Shortener (bit.ly)

```
Step 1: Requirements
- 100M URLs shortened per day (writes)
- 10:1 read:write ratio → 1B redirects/day (reads)
- Keep URLs for 5 years

Step 2: Calculate QPS
Write QPS = 100M / 86,400 ≈ 1,200 writes/second
Read QPS  = 1B / 86,400    ≈ 11,574 reads/second ≈ 12K QPS

Peak QPS (3x) ≈ 36K reads/second

Step 3: Storage
URL record = short_code (7 bytes) + original_url (200 bytes avg) + 
             created_at (8 bytes) + expiry (8 bytes) = ~300 bytes

Records per year = 100M/day × 365 = 36.5B records
5-year total     = 36.5B × 5 = 182.5B records
Storage          = 182.5B × 300 bytes = 54.75 TB

Step 4: Cache
Hot URLs (top 20% = 80% traffic):
= Total unique URLs read/day × 20%
= 1B reads × 200 bytes (store original URL) × 20%
≈ 40 GB cache

Conclusion:
- Single MySQL/PostgreSQL can handle 1,200 writes/sec
- Read replicas needed for 12K QPS (or Redis cache)
- 40 GB Redis cache for hot URLs
- Standard B-tree index on short_code for O(log n) lookup
```

### Example 2 — Design a Paste Service (Pastebin)

```
Given:
- 10M pastes created/day
- 10:1 read:write = 100M reads/day
- Max paste size: 10 MB (avg 10 KB)

QPS:
Write: 10M / 86,400 = 115 writes/second
Read:  100M / 86,400 = 1,157 reads/second

Storage:
10M pastes/day × 10 KB avg = 100 GB/day
10-year retention: 100 GB × 365 × 10 = 365 TB

Cache:
Popular pastes (20%): 1,157 reads/sec × 10 KB × 20% = 2.3 GB hot cache
→ Very manageable, single Redis instance sufficient

Conclusion: Simple system — can start with monolith + PostgreSQL + Redis
```

### Example 3 — Design a Rate Limiter

```
Given:
- 10M users
- Each user allowed 100 requests/minute
- Need to track request counts per user

Storage per user: user_id (8B) + count (4B) + window_start (8B) = 20 bytes
Total: 10M × 20 bytes = 200 MB

→ Fits entirely in Redis with room to spare
→ Redis INCR + EXPIRE for sliding window
→ No persistent storage needed (can reconstruct from logs if Redis fails)

Redis QPS:
If peak is 1M requests/second, Redis must handle 1M INCR/sec
→ Single Redis ~100K ops/sec, need Redis cluster or local in-process rate limiting
→ Solution: Approximate rate limiting with token bucket per app server
```

### Example 4 — Design a Search Engine Index

```
Given:
- Index 1 billion web pages
- Average page size: 100 KB
- Keep 10 words per page in index (stop words removed)
- Average word: 5 chars = 5 bytes

Raw page storage: 1B × 100 KB = 100 PB (far too much)
Indexed storage (doc_id + words only):
= 1B pages × 10 words × 5 bytes/word = 50 GB per document word store

Inverted index size:
- 1M unique English words
- Each word appears in avg 10M documents (common words more)
- word → [doc_ids] list
- Per entry: word (5B) + doc_ids (avg 10M × 8 bytes each) = 80 MB per word
- 1M words × 80 MB = 80 TB for posting lists

Sharding:
- Shard by term (each shard holds subset of words)
- 80 TB / 1 TB per shard = 80 shards minimum
- With replication (3x): 240 shard replicas
```

### Example 5 — Design a Notification System

```
Given:
- 500M users
- Each user receives avg 5 notifications/day
- Notification delivery in < 30 seconds (near real-time)

Volume:
= 500M × 5 = 2.5B notifications/day
= 2.5B / 86,400 = 28,900 notifications/second ≈ 29K/sec

Delivery method:
- Mobile: APNs (Apple) + FCM (Google)
- Web: WebSockets or SSE
- Email: SMTP gateway
- SMS: Twilio

Payload: 
notification_id (8B) + user_id (8B) + type (1B) + message (200B) + timestamp (8B) = ~225 bytes

Queue sizing:
At 29K/sec × 225 bytes = 6.5 MB/sec into queue
Daily retention (24h) in Kafka = 6.5 MB/sec × 86,400 = 562 GB/day

Fan-out consideration:
If broadcasting to all users (e.g., system alert):
= 500M messages in burst
= Need message queue with very high throughput
= Kafka with multiple partitions, multiple consumer groups
```

---

## When to Scale — Decision Thresholds

### Decision Points by QPS

```
QPS 0 → 1,000:
├── Single server is fine
├── PostgreSQL or MySQL handles this easily
└── Simple monolith, no special infrastructure

QPS 1,000 → 10,000:
├── Add a load balancer (nginx, HAProxy, or cloud LB)
├── 2-4 application servers
├── Add read replicas for DB
└── Consider Redis for frequently read data

QPS 10,000 → 100,000:
├── Caching is now mandatory (Redis cluster)
├── Database horizontal read scaling (5+ replicas)
├── Consider sharding for writes (if write-heavy)
├── CDN for static assets
└── Async processing for heavy operations (queues)

QPS 100,000 → 1,000,000:
├── Database sharding required
├── Multi-level caching (application cache + Redis + CDN)
├── Microservices architecture
├── Dedicated search cluster (Elasticsearch)
└── Multi-region if global

QPS > 1,000,000:
├── Multi-region active-active
├── Custom distributed infrastructure
├── Specialized hardware (custom chips, co-located infrastructure)
└── Examples: Google, Facebook, Amazon
```

### Storage Thresholds

```
< 1 TB:   Single PostgreSQL server (fine)
1-10 TB:  PostgreSQL with partitioning, or move to NoSQL
10-100 TB: Horizontal sharding required, consider Cassandra
100 TB+:  Distributed storage, HDFS, object storage (S3)
> 1 PB:   Specialized infrastructure, columnar stores for analytics
```

### Cache Sizing Rules of Thumb

```
Start with: 20% of working set in cache
Scale to:   Full working dataset if access is random

Redis memory:
- 1M small keys (100 bytes): ~100 MB RAM
- 1M large values (10 KB):   ~10 GB RAM

When to add more cache:
- Hit rate below 80% → add more cache capacity
- Hit rate above 95% → you're over-provisioned (save cost)
- Eviction rate high → working set exceeds cache size
```

---

## Quick Reference Tables

### Latency Numbers Cheat Sheet

```
ns = nanoseconds  (10^-9 seconds)
µs = microseconds (10^-6 seconds)
ms = milliseconds (10^-3 seconds)

L1 cache:          0.5 ns
L2 cache:          7   ns
RAM:             100   ns     (= 0.1 µs)
SSD random read: 150   µs     (= 0.15 ms)
HDD seek:         10   ms
Same-DC network:   0.5 ms
Cross-continent:  150  ms
```

### Storage Conversion Cheat Sheet

```
1 byte    = 8 bits
1 KB      = 1,000 bytes (or 1,024 for binary)
1 MB      = 1,000 KB
1 GB      = 1,000 MB
1 TB      = 1,000 GB
1 PB      = 1,000 TB
1 EB      = 1,000 PB

Useful: 1 GB = 10^9 bytes
        1 TB = 10^12 bytes
        1 PB = 10^15 bytes
```

### Quick QPS Calculations

```
1M requests/day   = 11.5/second  ≈ 12 QPS
10M requests/day  = 115/second   ≈ 100 QPS
100M requests/day = 1,157/second ≈ 1,000 QPS (1K QPS)
1B requests/day   = 11,574/second ≈ 10,000 QPS (10K QPS)
10B requests/day  ≈ 100,000 QPS (100K QPS)
100B requests/day ≈ 1,000,000 QPS (1M QPS)

Simple formula: N million req/day ≈ N × 12 QPS
```

### Estimation Shortcuts

```
Assume:
- 1 char ≈ 1 byte (ASCII), 2-4 bytes (Unicode)
- Image (compressed): 300 KB average
- 1 min video (streaming): 50-150 MB
- User profile: 1 KB
- 1 day ≈ 10^5 seconds (86,400 rounded up)
- 1 year ≈ 3 × 10^7 seconds (31,536,000)
- 1 GB RAM ≈ 10M objects (100-byte objects)

Replication multiplier: 3x (standard for HA)
Overhead multiplier: 2x (indexes, logs, temp files)
Growth buffer: 2-3x (plan for 2-3x growth)
```

---

## Estimation Methodology — Step by Step

```
STEP 1: Define base units
- What is 1 user action? (1 tweet, 1 message, 1 view)
- What data does 1 action generate?

STEP 2: Calculate daily volume
- DAU × actions/user/day = total daily actions

STEP 3: Calculate QPS
- Daily volume / 86,400 = average QPS
- Average QPS × 2-5 = peak QPS

STEP 4: Calculate storage
- Objects/day × object size × retention period × replication

STEP 5: Calculate bandwidth
- QPS × average response/request size

STEP 6: Draw architectural conclusions
- QPS > 10K → need caching
- QPS > 100K → need sharding
- Storage > 10TB → need distributed storage
- Bandwidth > 1 Gbps → need CDN

ALWAYS round generously and state your assumptions:
"I'll assume 100 bytes per tweet for simplicity.
The actual size might be 200-300 bytes, but this gives
us the right order of magnitude."
```

---

*Reference: Jeff Dean's "Numbers Every Engineer Should Know", Alex Xu "System Design Interview", Brendan Gregg's performance work*
