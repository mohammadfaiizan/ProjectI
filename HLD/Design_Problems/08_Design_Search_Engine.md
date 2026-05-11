# Design a Search Engine (like Google Web Search) — High-Level Design

---

## 1. Problem Statement & Clarifying Questions

**Problem Statement:**
Design a large-scale web search engine that crawls the internet, indexes web pages, and serves ranked search results with low latency. The system must handle billions of web pages, support relevance ranking, and process billions of queries per day.

**Clarifying Questions:**
- How many web pages should the system index? (10B pages)
- What is the expected query volume? (1B queries/day)
- Do we need to support real-time indexing or batch processing?
- Is personalization required?
- Do we need autocomplete/spelling correction?
- Should we support structured queries (site:, filetype:, etc.)?
- Do we need to index images/videos as well?
- What is the freshness requirement for index updates?

**Assumptions:**
- Focus on web document search (text-based)
- 10 Billion web pages indexed
- 1 Billion queries per day
- Index freshness: top pages re-crawled daily, others weekly/monthly
- Relevance ranking: BM25 + PageRank + freshness + authority
- Spelling correction and autocomplete included

---

## 2. Functional & Non-Functional Requirements

### Functional Requirements
1. **Web Crawling:** Discover and download web pages continuously
2. **Indexing:** Build inverted index from crawled content
3. **Search:** Return top-K relevant results for a query
4. **Ranking:** Combine text relevance + link authority (PageRank)
5. **Autocomplete:** Suggest query completions as user types
6. **Spelling Correction:** Handle typos and suggest corrections
7. **Snippet Generation:** Show relevant excerpt from each result
8. **Freshness:** Keep index up to date with web changes

### Non-Functional Requirements
1. **Latency:** Search results returned in <200ms (P99)
2. **Throughput:** 1B queries/day ≈ 12K QPS (peak ~50K QPS)
3. **Availability:** 99.99% uptime
4. **Index Size:** 10B pages, ~500 bytes average extracted text
5. **Crawl Rate:** Crawl the web in reasonable time (weeks for full crawl)
6. **Freshness:** Top 1B pages refreshed within 24 hours

---

## 3. Capacity Estimation

### Web Pages
- Total indexed pages: 10 Billion
- Average page size (raw HTML): 100KB
- Total raw storage: 10B * 100KB = 1 Petabyte
- Extracted text (10% of HTML): 10B * 10KB = 100TB
- Inverted index size: ~30% of text size = 30TB
- PageRank data: 10B pages * 24 bytes = 240GB

### Crawling
- Pages to recrawl daily (top 1B): 1B / 86400 ≈ 12K pages/second
- Full web recrawl (10B pages, weekly): 10B / (7 * 86400) ≈ 16K pages/second
- Total crawl rate: ~30K pages/second
- Bandwidth for crawling: 30K * 100KB = 3 GB/s ingress

### Query Processing
- Daily queries: 1 Billion
- Average QPS: 1B / 86400 ≈ 12K QPS
- Peak QPS: ~50K QPS (3-4x average during peak hours)
- Average query length: 3 words
- P99 latency target: 200ms

### Index Storage
- Inverted index: 30TB across index shards
- Each shard: ~1TB per index server
- 30 index server machines (primary) + 30 replicas = 60 machines
- Query fanout: query broadcast to all 30 shards simultaneously
- Results from each shard: top-10, then merge-sort globally

---

## 4. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           CRAWL SUBSYSTEM                                │
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
│  │  URL Frontier│    │  Crawler     │    │  Content Store (S3/GCS)  │  │
│  │  (Priority   │───▶│  Workers     │───▶│  Raw HTML pages          │  │
│  │   Queue)     │    │  (N machines)│    │  Deduplicated            │  │
│  │              │    │              │    │                          │  │
│  │  - robots.txt│    │  - Politeness│    │  Kafka: new.pages        │  │
│  │  - Bloom     │    │  - DNS cache │    │                          │  │
│  │    filter    │    │  - Timeout   │    └──────────────────────────┘  │
│  └──────────────┘    └──────────────┘                                   │
│          ▲ Extract new URLs                                              │
└──────────┼──────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        INDEXING PIPELINE                                 │
│                                                                          │
│  HTML Parser → Content Extractor → Tokenizer → Stop Words/Stemmer       │
│       ↓               ↓                                                  │
│  Link Extractor   Text Normalizer → TF-IDF Scorer → Index Merger        │
│       ↓                                                  ↓               │
│  URL Frontier                                    Inverted Index          │
│  (new URLs)                                      (Sharded)               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                         QUERY SERVING                                    │
│                                                                          │
│  User Query                                                              │
│     │                                                                    │
│     ▼                                                                    │
│  ┌──────────────────────────────────────────────────────────┐          │
│  │  Query Processor                                          │          │
│  │  ├── Tokenize → remove stop words → stem                 │          │
│  │  ├── Spell correction (edit distance / n-gram model)     │          │
│  │  ├── Query expansion (synonyms)                          │          │
│  │  └── Structured query parsing (site:, filetype:)        │          │
│  └──────────────────────────────────────────────────────────┘          │
│     │ Processed query                                                    │
│     ▼                                                                    │
│  ┌──────────────────────────────────────────────────────────┐          │
│  │  Index Scatter-Gather                                     │          │
│  │  Broadcasts to all 30 index shards in parallel           │          │
│  └──────────────────────────────────────────────────────────┘          │
│     │ Top-10 from each shard                                             │
│     ▼                                                                    │
│  ┌──────────────────────────────────────────────────────────┐          │
│  │  Result Ranker & Merger                                   │          │
│  │  ├── Merge top-10 from 30 shards = 300 candidates        │          │
│  │  ├── Re-rank using global PageRank + freshness           │          │
│  │  └── Personalization layer (if applicable)              │          │
│  └──────────────────────────────────────────────────────────┘          │
│     │                                                                    │
│     ▼                                                                    │
│  Snippet Generator → Final Result Rendering                              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                      AUXILIARY SYSTEMS                                   │
│  PageRank Computation    │  Query Cache (Redis)   │  Spell Corrector    │
│  (MapReduce / Spark,     │  Cache popular query   │  (n-gram model,     │
│   runs daily)            │  results (1h TTL)      │   edit distance)    │
└──────────────────────────┴────────────────────────┴─────────────────────┘
```

---

## 5. Component Deep-Dive

### 5.1 Web Crawler Architecture

**URL Frontier:**
- Priority queue of URLs to crawl
- Priority based on:
  - PageRank score (high-authority pages crawled more frequently)
  - Time since last crawl (freshness)
  - Domain diversity (politeness constraints)
- Implemented as distributed priority queue (Redis sorted sets or Kafka topics)

**Politeness Constraints:**
- Read `robots.txt` before crawling any domain
- Minimum delay between requests to same domain (1 req/sec default)
- Respect `Crawl-Delay` directive
- Store politeness state per domain: `{domain: last_crawl_time}`

**URL Deduplication with Bloom Filter:**
- Problem: Same URL discovered multiple times from different pages
- Bloom Filter: probabilistic data structure, O(1) lookup, no false negatives
- Size: 10B URLs * 10 bits/URL = 12.5GB (acceptable for RAM)
- False positive rate: ~1% (acceptable — just skip some new URLs)
- Fallback: periodic exact-match check against URL database

**BFS vs DFS Crawling:**
- BFS: discover all pages at depth 1, then depth 2, etc.
  - Pros: High-priority pages (linked from many places) found early
  - Cons: Memory for frontier grows proportionally to discovered URLs
- DFS: go deep into one domain before moving to next
  - Pros: More thorough per domain, better for site maps
  - Cons: May miss high-value pages on other domains
- **Choice:** BFS with domain diversity enforcement (politeness)

**Distributed Crawling:**
- Hash domain to crawler worker: `hash(domain) % N_workers`
- Each worker manages politeness for its assigned domains
- Central URL frontier distributes new URLs to workers

### 5.2 Document Processing Pipeline

```
Raw HTML
   ↓
HTML Parser (BeautifulSoup / custom)
   ↓ Extract text, title, meta, links
Content Normalizer
   ↓ Decode charset, strip HTML, remove boilerplate (nav/footer)
Language Detector
   ↓ Route non-English to language-specific pipeline
Tokenizer
   ↓ Split on whitespace/punctuation, lowercase
Stop Word Removal
   ↓ Remove "the", "a", "is", etc.
Stemmer (Porter/Snowball)
   ↓ "running" → "run", "searches" → "search"
Index Term Extractor
   ↓ Compute TF for each term in document
TF-IDF Scorer (partial: TF computed, IDF applied at query time)
   ↓
Inverted Index Writer
```

### 5.3 Inverted Index Construction

**Structure:**
```
term → postings list: [(doc_id, tf, [position1, position2, ...]), ...]

Example:
"python" → [(doc_1, tf=5, [12, 45, 78, 120, 200]),
            (doc_2, tf=3, [5, 60, 90]),
            ...]
```

**Index Sharding:**
- Shard by `hash(term) % 30` → each shard handles ~3B documents
- Term "python" always routes to shard 7 regardless of document
- Enables parallel query processing

**Index Format (like Lucene):**
- Term dictionary: sorted array of terms (binary search O(log N))
- Postings list: compressed with delta encoding + VByte compression
- Position lists: optional, needed for phrase queries

**Delta Encoding Example:**
- Raw doc IDs: [100, 345, 679, 1023]
- Deltas: [100, 245, 334, 344]
- VByte encode: smaller numbers use fewer bytes

### 5.4 TF-IDF Scoring

**Term Frequency (TF):**
```
TF(t, d) = count(t in d) / total_terms(d)
```

**Inverse Document Frequency (IDF):**
```
IDF(t) = log(N / df(t))
where N = total documents, df(t) = documents containing term t
```

**TF-IDF Score:**
```
TF-IDF(t, d) = TF(t, d) * IDF(t)
```

### 5.5 BM25 Ranking (Better than TF-IDF)

```
BM25(q, d) = Σ IDF(qi) * [TF(qi,d) * (k1+1)] / [TF(qi,d) + k1 * (1 - b + b * |d|/avgdl)]

Parameters:
- k1 = 1.5 (term frequency saturation)
- b = 0.75 (document length normalization)
- avgdl = average document length
```

BM25 advantages over TF-IDF:
- Saturates TF (diminishing returns for repeated terms)
- Normalizes for document length
- Standard in information retrieval

### 5.6 PageRank Computation

**Algorithm:**
```
PR(A) = (1 - d) + d * Σ [PR(T_i) / C(T_i)]

where:
- d = damping factor (0.85)
- T_i = pages that link to A
- C(T_i) = number of outbound links from T_i
```

**Iterative Computation:**
1. Initialize all pages with PR = 1/N
2. Iterate: new_PR(A) = formula above
3. Repeat until convergence (change < threshold)
4. Typically converges in 50-100 iterations for web-scale

**MapReduce / Spark Implementation:**
- Map: for each page P, emit (linked_page, PR(P)/outlinks(P)) for each link
- Reduce: sum contributions + add damping factor
- Run daily as batch job

### 5.7 Query Processing

**Query Pipeline:**
1. Tokenize query
2. Remove stop words (but keep in phrase queries "to be or not to be")
3. Spell correction: generate candidates with edit distance ≤ 2, score with n-gram language model
4. Query expansion: add synonyms from WordNet/learned embeddings
5. Identify query type: navigational, informational, transactional

**Result Ranking Formula:**
```
final_score(doc, query) = 
    α * BM25(query, doc) +
    β * log(PageRank(doc)) +
    γ * freshness_score(doc) +
    δ * authority_score(doc)

Where α, β, γ, δ are learned weights (ML ranking model)
```

### 5.8 Snippet Generation

- For each result, find the window of text with highest query term density
- Highlight query terms in bold
- Max snippet length: 160 characters
- Algorithm: sliding window over document text, score by term matches

---

## 6. Database Design

### Index Storage

```sql
-- Document metadata (PostgreSQL)
CREATE TABLE documents (
    doc_id      BIGINT PRIMARY KEY,
    url         VARCHAR(2000) UNIQUE NOT NULL,
    url_hash    CHAR(32) NOT NULL,         -- MD5 of URL
    title       VARCHAR(500),
    crawled_at  TIMESTAMP,
    content_hash CHAR(32),                 -- MD5 of content (dedup)
    page_rank   FLOAT DEFAULT 0.15,
    word_count  INTEGER,
    language    CHAR(5),
    is_indexed  BOOLEAN DEFAULT FALSE,
    INDEX(url_hash),
    INDEX(crawled_at)
);

-- Domain crawl state (Redis)
domain:{domain}:last_crawl → timestamp
domain:{domain}:crawl_count → integer
domain:{domain}:robots_txt → text (TTL 24h)
```

### Inverted Index (Custom Binary Format)

```
File: shard_{N}.idx

Header:
  - magic_bytes: 4 bytes
  - version: 2 bytes
  - num_terms: 8 bytes
  - created_at: 8 bytes

Term Dictionary Block:
  - [term_length: 2B][term: NB][postings_offset: 8B][postings_length: 4B]
  - Sorted alphabetically for binary search

Postings Block:
  - For each term: delta-encoded doc_ids + VByte compressed
  - [doc_id_delta: VByte][tf: VByte][positions_count: VByte][pos_deltas...]
```

---

## 7. API Design

### Search API
```
GET /search?q=python+tutorial&page=1&n=10&lang=en&sort=relevance
Response: {
    "query": "python tutorial",
    "corrected_query": null,
    "total_results": 5400000,
    "results": [
        {
            "url": "https://docs.python.org/tutorial",
            "title": "Python Tutorial — Python 3.11",
            "snippet": "...learn <b>Python</b> programming with this comprehensive <b>tutorial</b>...",
            "score": 0.95,
            "page_rank": 0.87,
            "cached_url": "/cache?q=...",
            "last_indexed": "2024-01-15"
        }
    ],
    "next_page_token": "eyJwYWdlIjozfQ=="
}
```

### Crawl Status API
```
GET /admin/crawl/status
Response: { "pages_crawled_today": 1234567, "queue_depth": 45000000, "errors_rate": 0.01 }
```

### Index Admin API
```
POST /admin/index/reindex?url=https://example.com
POST /admin/index/remove?url=https://spam.example.com
GET  /admin/index/stats → { "total_terms": 10B, "shards": 30, "index_size_gb": 30000 }
```

---

## 8. Scalability & Bottlenecks

| Component | Bottleneck | Solution |
|-----------|-----------|----------|
| Crawling | DNS resolution latency | Distributed DNS cache, batch DNS prefetch |
| Crawling | Robots.txt fetches | Cache robots.txt per domain (24h TTL) |
| Indexing | Single-machine index build | Distributed MapReduce index construction |
| Query serving | 30-shard fanout latency | Parallel scatter-gather, cut off stragglers at 150ms |
| PageRank | Full web graph computation | Daily Spark job, incremental updates for new pages |
| Result caching | Repeated popular queries | Redis query result cache (1h TTL, LRU eviction) |
| Autocomplete | High QPS, low latency | Trie/prefix index in Redis, pre-built suggestions |

---

## 9. Trade-offs & Design Decisions

### BFS vs Priority-Queue Crawling
- **BFS:** Simple, discovers all pages at each depth level
- **Priority Queue:** Crawl high-value pages more frequently, better freshness for important sites
- **Choice:** Priority queue based on PageRank + recency
- **Trade-off:** More complex scheduling vs better index quality

### Term-based vs Document-based Sharding
- **Term sharding:** All docs for a term on one shard → no scatter-gather for single-term queries
- **Document sharding:** Each shard has all terms for its documents → scatter-gather always needed
- **Choice:** Document sharding (used by most modern search engines)
- **Trade-off:** Query fanout vs simpler index management

### Real-time vs Batch Indexing
- **Real-time (Kafka + streaming):** Low latency index updates, higher complexity
- **Batch (MapReduce daily):** Simpler, higher throughput, but stale index
- **Choice:** Hybrid — real-time for high-priority URLs, batch for bulk index
- **Trade-off:** System complexity vs index freshness

### PageRank vs ML Ranking
- **PageRank:** Interpretable, offline computable, spam-resistant
- **ML Ranking (LambdaMART, BERT):** More accurate, personalized, but expensive
- **Choice:** Start with BM25 + PageRank, layer ML ranking on top
- **Trade-off:** Infrastructure cost vs ranking quality

---

## 10. Key Interview Talking Points

1. **Bloom Filter for URL Dedup:** O(1) lookup, memory-efficient (10 bits/URL), ~1% false positive rate acceptable. Never false negatives (we never skip a truly new URL). Periodically rebuild from exact URL store.

2. **Inverted Index Construction:** Show term → postings list structure. Explain delta encoding + VByte compression (reduces storage 4-8x). Shard by hash(term) for parallel query processing.

3. **BM25 vs TF-IDF:** BM25 is better because it handles term frequency saturation (TF doesn't grow linearly) and document length normalization. k1 and b are tunable parameters.

4. **PageRank Convergence:** Explain iterative computation, damping factor 0.85 represents 15% probability of random jump. Converges in ~50 iterations. Teleportation prevents "dangling node" problem.

5. **Scatter-Gather Pattern:** Query broadcast to all 30 shards simultaneously. Each returns top-10. Merge 300 candidates. Set 150ms timeout — accept results from shards that respond in time (tail latency cutoff).

6. **Index Freshness:** Crawl priority queue based on PageRank + time since last crawl. Top 1B pages daily, long tail weekly. Real-time indexing pipeline for breaking news sites.

7. **Spell Correction:** Edit distance candidates + n-gram language model for scoring. "Did you mean" shows highest-scoring alternative. Learned from query logs ("python toturial" → "python tutorial").

8. **Query Caching:** Top 1000 queries account for 20% of traffic (Zipf distribution). Cache full result sets in Redis. Stale queries invalidated when index updates contain those terms.

9. **Snippet Generation:** Sliding window algorithm to find most relevant excerpt. Query terms bolded. 160-character limit. Avoid cutting mid-sentence. Show first occurrence of query terms.

10. **Back-of-Envelope:** 10B pages * 10KB text = 100TB text. Inverted index ~30TB. 30 shards of 1TB each. 1B queries/day = 12K QPS. With query caching, only 80% hit index (2.4K QPS to index shards). Scatter to 30 shards = 72K shard QPS total, easily handled.
