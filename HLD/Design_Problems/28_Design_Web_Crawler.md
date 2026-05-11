# Problem 28: Design a Web Crawler

---

## 1. Problem Statement & Clarifying Questions

### Problem Statement
Design a distributed web crawler that can crawl 1 billion web pages within 30 days, respecting robots.txt politeness rules, deduplicating content, and storing crawled data for a search index pipeline.

### Clarifying Questions
1. **Scale**: How many pages to crawl? How often re-crawl? (1B pages, re-crawl popular pages weekly)
2. **Purpose**: For search indexing, data mining, or archive? (Assume search indexing like Googlebot)
3. **Scope**: Only English content? Specific domains? (Assume global, all languages)
4. **JavaScript**: Handle JS-rendered content? (Assume yes, with headless Chrome for top sites)
5. **Politeness**: Honor robots.txt and crawl-delay strictly? (Yes — legal requirement)
6. **Freshness**: How to prioritize re-crawling? (Higher-PageRank pages re-crawled more frequently)
7. **Storage**: Raw HTML stored? Parsed content? Both? (Both: raw HTML in S3, parsed in Elasticsearch)
8. **Duplicate handling**: Exact duplicates or near-duplicates? (Both: SHA-256 for exact, SimHash for near-dups)
9. **Link depth**: Maximum crawl depth from seed? (No hard limit; priority-based BFS)
10. **Legal**: Handle GDPR/CCPA pages, login-required pages? (Skip login pages; respect noindex meta tags)

---

## 2. Functional & Non-Functional Requirements

### Functional Requirements
- Crawl web pages starting from seed URLs, following links discovered
- Respect robots.txt (crawl-delay, disallow rules, user-agent rules)
- Deduplicate URLs (exact URL dedup) and content (near-duplicate detection)
- Parse HTML to extract links, title, metadata, and text content
- Store raw HTML and parsed content for indexing pipeline
- Schedule re-crawls based on page change frequency and importance
- Handle redirects (301/302), canonicalization, rel=canonical
- DNS resolution with caching and TTL respect
- Support JavaScript-rendered pages via headless browser

### Non-Functional Requirements
- **Throughput**: Crawl 1B pages in 30 days = ~385 pages/second average; 10,000 pages/sec peak
- **Availability**: Crawler runs 24/7; failure of individual workers should not stop crawl
- **Storage**: 1B pages × avg 100 KB HTML = 100 TB raw storage
- **Politeness**: Maximum 1 request per domain per 10 seconds (unless crawl-delay specified)
- **Latency tolerance**: Slow response (> 30s) → timeout and reschedule
- **Scalability**: Add/remove crawler workers without system restart
- **Fault tolerance**: Worker failure recovers in < 5 minutes; no URL loss

---

## 3. Capacity Estimation

### Crawl Rate
- Target: 1B pages / 30 days = 385 pages/sec average
- With overhead (robots.txt, DNS, retries): need 500 pages/sec effective crawl
- 1000 workers × 10 pages/sec = 10,000 pages/sec capacity (10× headroom)

### Storage
- Raw HTML: 1B × 100 KB = 100 TB (compress with gzip → ~20 TB)
- Parsed content (ES): 1B × 10 KB = 10 TB
- URL frontier: 1B URLs × 100 bytes = 100 GB (fits in distributed memory + disk)
- Bloom filter (seen URLs): 1B entries × 10 bits (0.1% FPR) = ~1.25 GB RAM
- Link graph: 1B pages × avg 10 outlinks × 8 bytes = 80 GB

### Bandwidth
- Crawling: 10,000 pages/sec × 100 KB = 1 GB/s inbound bandwidth
- DNS queries: 10,000 URLs/sec (cached, so ~1,000 actual DNS/sec)
- robots.txt: 1 per new domain (cached) = negligible

### Workers
- 1,000 crawler workers, each handling 10 concurrent HTTP connections
- 10 parsing workers per crawler worker
- Total: 1,000 VMs in crawl farm (8 CPU, 16 GB RAM each)

---

## 4. High-Level Architecture (ASCII Diagram)

```
 ┌──────────────────────────────────────────────────────────────────────────┐
 │                           SEED URL INPUT                                  │
 │  Manual seeds │ Sitemap.xml feeds │ DNS zone transfers │ DMOZ-like lists  │
 └──────────────────────────────┬───────────────────────────────────────────┘
                                │
 ┌──────────────────────────────▼───────────────────────────────────────────┐
 │                        URL FRONTIER SERVICE                               │
 │  Priority Queue (score = PageRank + freshness_bonus + recency_penalty)   │
 │  Politeness queues: one queue per domain, rate-limited per domain         │
 │  URL dedup check via Bloom Filter before insertion                        │
 │  Persistent storage: Redis Sorted Set + disk-backed overflow (RocksDB)   │
 └──────────────────────────────┬───────────────────────────────────────────┘
                                │ URLs distributed to workers
 ┌──────────────────────────────▼───────────────────────────────────────────┐
 │                      CRAWLER WORKER POOL (1,000 workers)                  │
 │                                                                            │
 │  ┌─────────────────────────────────────────────────────────────────────┐  │
 │  │  For each URL:                                                       │  │
 │  │  1. DNS Cache lookup (TTL-based)                                    │  │
 │  │  2. robots.txt check (domain-cached)                                │  │
 │  │  3. HTTP GET (timeout 30s, max 10 redirects)                        │  │
 │  │  4. Content type check (only text/html, application/pdf, etc.)      │  │
 │  │  5. Content hash → check ContentHashStore (exact dedup)             │  │
 │  │  6. SimHash → check for near-duplicates                             │  │
 │  │  7. Store raw HTML to S3                                            │  │
 │  │  8. Send to Parser Queue                                            │  │
 │  └─────────────────────────────────────────────────────────────────────┘  │
 └──────────────────────────────┬───────────────────────────────────────────┘
                                │
         ┌──────────────────────▼─────────────────────────┐
         │              PARSING PIPELINE (Kafka-based)     │
         │  HTML Parser → Link Extractor → Text Extractor  │
         │  Metadata Extractor → Canonical URL resolver    │
         └──────┬──────────────────────────┬──────────────┘
                │                          │
  ┌─────────────▼──────────┐  ┌────────────▼────────────────────────────┐
  │  URL EXTRACTOR         │  │  CONTENT STORE                           │
  │  New URLs → deduplicate│  │  Raw HTML → S3 (20 TB compressed)       │
  │  → score → URL Frontier│  │  Parsed text → Elasticsearch (10 TB)    │
  └────────────────────────┘  │  Link graph → Neo4j / ArangoDB          │
                               └─────────────────────────────────────────┘
```

---

## 5. Component Deep-Dive

### 5.1 URL Frontier
The URL Frontier manages what to crawl next and in what order:

**Two-Level Queue Architecture:**
1. **Front queues** (priority): N queues ordered by priority (PageRank score × freshness)
2. **Back queues** (politeness): One queue per domain; worker takes from back queue, enforcing per-domain crawl delay

**Priority Scoring:**
```
priority = w1 × pagerank_score
         + w2 × (1 / days_since_last_crawl)
         + w3 × (historical_change_frequency)
         + w4 × (inlink_count / max_inlinks)
```

**Politeness Enforcement:**
- Track `last_crawl_time[domain]` and `crawl_delay[domain]` (from robots.txt)
- Worker checks: `now - last_crawl_time[domain] >= crawl_delay[domain]`
- Default crawl delay: 10 seconds if not specified in robots.txt
- Never crawl same domain on two workers simultaneously (domain-level locking)

### 5.2 URL Deduplication with Bloom Filter
- **Bloom filter**: Probabilistic set membership; O(k) insert and lookup where k = number of hash functions
- **Configuration**: 1B URLs, FPR = 0.1% → requires ~10 bits/element → 1.25 GB RAM
- **Implementation**: Use 3 independent hash functions (MurmurHash3)
- **Limitation**: False positives → occasionally miss crawling a valid new URL (acceptable)
- **False negatives**: None — never misidentifies seen URL as unseen
- **Distributed**: Bloom filter replicated across all workers; updates batched and gossiped

### 5.3 DNS Caching
- **Problem**: 10,000 URLs/sec → 10,000 DNS lookups/sec → DNS servers overwhelmed
- **Solution**: In-process DNS cache per worker with TTL from DNS response
- **Implementation**: LRU cache of size 100K entries; evict on TTL expiry
- **Hit rate**: ~95% for popular domains (news, social media dominate crawl)
- **Fallback**: Multiple DNS resolvers (8.8.8.8, 1.1.1.1) with round-robin for misses
- **Negative caching**: Cache NXDOMAIN for 5 minutes to avoid repeated lookups

### 5.4 robots.txt Compliance
```
User-agent: *
Disallow: /private/
Disallow: /admin/
Crawl-delay: 10
Allow: /public/

Sitemap: https://example.com/sitemap.xml
```
- Fetch and cache robots.txt per domain on first visit; refresh every 24 hours
- Parse disallow rules as prefix-match patterns
- Respect user-agent specific rules (use "Googlebot"-like agent identification)
- `Crawl-delay` sets minimum wait between requests to that domain
- Honor `noindex` in HTTP headers and meta tags (do not submit to search index)

### 5.5 Content Deduplication

**Exact Deduplication (SHA-256):**
- Hash raw HTML body (before any processing)
- Store hashes in distributed hash set (Redis SET or Bloom filter variant)
- If hash seen before: discard content, still record URL as crawled
- Covers exact mirrors and identical syndicated content

**Near-Duplicate Detection (SimHash):**
- Extract text tokens from HTML; compute SimHash (64-bit fingerprint)
- Two pages with Hamming distance ≤ 3 considered near-duplicates
- Store SimHash values in sorted table; query using locality-sensitive hashing (LSH)
- Covers: slight modifications (different ads, dates), scrapers republishing content

### 5.6 HTML Parsing and Link Extraction
- **Library**: Use lxml (Python) or Jsoup (Java) for HTML parsing
- **Link extraction**: Find all `<a href="...">` tags; handle relative URLs
- **Canonical URLs**: Prefer `<link rel="canonical">` over crawled URL
- **Link normalization**: Remove fragments (#section), sort query params, lowercase scheme/host
- **nofollow**: Respect `rel="nofollow"` — do not follow link, but may crawl it independently
- **Base URL**: Handle `<base href="...">` for relative URL resolution

### 5.7 JavaScript Rendering
- **Problem**: ~30% of web pages require JS execution to render content
- **Solution**: Headless Chrome (via Puppeteer/Playwright) for JS-heavy pages
- **Identification**: If initial HTML contains minimal content but JS scripts → render
- **Cost**: JS rendering 10× slower and 10× more CPU than static HTML
- **Scale**: Dedicate 10% of workers to JS rendering (100 workers); prioritize top-10K domains
- **Caching**: Cache rendered DOM for same URL within 1 hour

### 5.8 Freshness Scheduling
**Change frequency estimation:**
- Track `last_seen_content_hash` per URL
- On re-crawl: if hash changed → update change frequency estimate
- **Exponential moving average**: `change_freq = α × (hash_changed ? 1 : 0) + (1-α) × change_freq`
- **Re-crawl interval**: `interval = base_interval / change_frequency` (more frequent for changing pages)
- **Categories**: News sites: hourly; blogs: daily; static pages: monthly

---

## 6. Database Design

### Crawl State Store (HBase)
```
Row key: SHA-256(normalized_url) [8 bytes] + url [variable]
Columns:
  cf:url             → full URL string
  cf:last_crawl      → timestamp of last successful crawl
  cf:content_hash    → SHA-256 of last content
  cf:simhash         → 64-bit SimHash fingerprint
  cf:http_status     → last HTTP status code
  cf:crawl_depth     → depth from seed URL
  cf:pagerank        → estimated PageRank score
  cf:change_freq     → estimated change frequency (0.0-1.0)
  cf:next_crawl      → scheduled next crawl timestamp
```

### robots.txt Cache (Redis)
```
robots:{domain} → HASH {
    rules: JSON array of {path: string, allow: bool},
    crawl_delay: int (seconds),
    sitemaps: JSON array of strings,
    fetched_at: timestamp,
    ttl: 86400  (24 hours)
}
```

### Content Store (S3 + Elasticsearch)
```
S3 path: s3://crawl-bucket/{year}/{month}/{day}/{sha256_content_hash}.html.gz

Elasticsearch document:
{
  "url": "https://example.com/page",
  "canonical_url": "https://example.com/page",
  "title": "Page Title",
  "content": "Full text content...",
  "language": "en",
  "crawl_timestamp": "2024-01-15T10:30:00Z",
  "outlinks": ["https://example.com/other", ...],
  "inlink_count": 1500,
  "content_hash": "sha256hex...",
  "http_status": 200,
  "content_type": "text/html"
}
```

---

## 7. API Design

### Crawler Control API (Internal)
```
POST /v1/frontier/add
Body: { "urls": ["https://example.com"], "priority": 0.8 }

GET /v1/frontier/stats
Response: { "pending": 150000000, "active": 10000, "completed": 850000000 }

POST /v1/crawl/pause?domain=example.com    # Pause crawling a domain
POST /v1/crawl/resume?domain=example.com

GET /v1/url/status?url=https://example.com/page
Response: { "url": "...", "last_crawl": "...", "status": "CRAWLED", "content_hash": "..." }
```

### Content Retrieval API (for Indexer)
```
GET /v1/content/{content_hash}
→ Returns raw HTML or presigned S3 URL

GET /v1/content/search?domain=example.com&crawled_after=2024-01-01
→ Returns list of content hashes crawled from domain since date
```

---

## 8. Scalability & Bottlenecks

### Bottleneck 1: URL Frontier Size (1B URLs)
- Cannot fit entirely in RAM (100 GB)
- **Solution**: Hot URLs (top 10M by priority) in Redis; cold URLs in RocksDB on SSD

### Bottleneck 2: Bloom Filter Synchronization
- 1000 workers each need consistent view of seen URLs
- **Solution**: Centralized Bloom filter service with gRPC; workers cache locally and sync every 10s

### Bottleneck 3: Crawl Politeness at Scale
- 1000 workers, each respecting 1-req/10s per domain
- 10K workers can effectively crawl 1000 distinct domains simultaneously
- **Solution**: Domain-aware work distribution; consistent hash assigns domain to specific worker

### Bottleneck 4: S3 Write Throughput
- 10,000 pages/sec × 100 KB = 1 GB/s to S3
- **Solution**: Batch small files; use S3 multipart uploads; distribute across multiple S3 prefixes

### Bottleneck 5: DNS Amplification
- Single resolver overwhelmed by 10K/sec unique domain lookups
- **Solution**: Run local unbound DNS resolver per worker cluster; anycast DNS

---

## 9. Trade-offs & Design Decisions

### Decision 1: BFS vs DFS Crawl Order
- **BFS**: Discovers broad web quickly; finds high-quality pages faster (homepages before deep pages)
- **DFS**: Goes deep into one site; may miss important pages on other sites
- **Choice**: Priority-based BFS with freshness bonus; effectively BFS with quality weighting

### Decision 2: Distributed vs Centralized URL Frontier
- **Centralized**: Easy dedup; bottleneck at scale; single point of failure
- **Distributed**: Scalable; harder global dedup; potential for domain assignment skew
- **Choice**: Distributed frontier with centralized Bloom filter for dedup; consistent hashing for domain assignment

### Decision 3: Push vs Pull for Worker URL Distribution
- **Push**: Coordinator pushes URLs to workers; workers can be overloaded
- **Pull**: Workers pull URLs when ready; natural backpressure
- **Choice**: Pull model — workers request next URLs when their queue drops below threshold

### Decision 4: When to Re-crawl
- **Fixed schedule**: Simple; ignores content dynamics
- **Adaptive**: Re-crawl based on measured change frequency; more complex but efficient
- **Choice**: Adaptive re-crawl with exponential moving average of change frequency

### Decision 5: Handling Crawler Traps
- **Infinite URL generation**: Sites that generate infinite unique URLs (calendars, search results)
- **Detection**: URL pattern analysis; cap per-domain crawl depth to 5 levels
- **Protection**: Max URLs per domain = 1M; URL length limit = 2048 chars

---

## 10. Key Interview Talking Points

### 1. Why Bloom Filters for URL Dedup
Explain the math: 1B URLs × exact hash storage = 24 GB (URL strings) vs Bloom filter = 1.25 GB. The 0.1% false positive rate means we might skip ~1M valid URLs — acceptable. Never false negatives, so we never re-crawl already-seen URLs.

### 2. Politeness and the Two-Level Queue
Front queues = priority ordering. Back queues = politeness per domain. The distinction ensures we can be polite (max 1 req/10s per domain) while still crawling the highest-priority URLs available. Without back queues, all workers might hammer a single popular domain.

### 3. SimHash for Near-Duplicate Detection
Walk through the algorithm: tokenize text → hash each token → for each bit position, sum +1 for each hash with bit=1, -1 for bit=0 → sign the vector → 64-bit fingerprint. Two pages with Hamming distance ≤ 3 are near-duplicates. Handles scrapers, minor edits, localization.

### 4. The Freshness vs Politeness Tension
We want to re-crawl frequently-changing news sites every hour. But if a news site serves 10K pages and we have 1-req/10s politeness → 10K × 10s = 27 hours to re-crawl all pages. Solution: prioritize re-crawling the home page and most-linked articles; accept that deep pages re-crawl less frequently.

### 5. robots.txt Compliance
This is a legal/ethical requirement, not just best practice. Always cache robots.txt per domain; never fetching it for every URL is critical for politeness. Mention the `User-agent: *` wildcard vs specific crawlers.

### 6. JavaScript Rendering Challenges
Headless Chrome pool: each instance handles 1 page at a time, takes 2-3 seconds. At 10,000 pages/sec, need 20,000-30,000 Chrome instances — impractical. Solution: Render only top-10K domains (by PageRank); use static HTML for the rest; accept incomplete indexing of JS-heavy long-tail sites.

### 7. Handling Infinite Crawl Depth
Without depth limits, crawler could follow links forever. Solutions: max depth per domain, max URLs per domain, URL pattern deduplication (strip query parameters from discovered URLs).
