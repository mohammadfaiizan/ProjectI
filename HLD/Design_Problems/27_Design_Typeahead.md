# Problem 27: Design a Typeahead / Autocomplete System

---

## 1. Problem Statement & Clarifying Questions

### Problem Statement
Design a real-time typeahead/autocomplete system that suggests the top-K most relevant search queries as a user types. The system must respond in under 100ms and handle Google-scale traffic of up to 10 million queries per second.

### Clarifying Questions
1. **Scale**: How many active users? (Assume 100M DAU, 10M QPS peak like Google Search)
2. **Suggestion count**: How many suggestions per keystroke? (Assume top 10)
3. **Freshness**: How quickly should trending queries appear in suggestions? (Minutes or hours?)
4. **Personalization**: Should suggestions be personalized per user, or global top-K only?
5. **Multilingual**: Support multiple languages and scripts? (Assume English first, extensible)
6. **Latency**: Strict < 100ms end-to-end including network? (Assume < 50ms server-side)
7. **Spell correction**: Should we correct typos before prefix lookup?
8. **Safe search**: Filter adult/offensive content from suggestions?
9. **Client debounce**: Handle debouncing client-side, or should API handle it?
10. **Logging**: Log every suggestion click for future frequency updates?

---

## 2. Functional & Non-Functional Requirements

### Functional Requirements
- Return top-10 query suggestions for any given prefix
- Rank suggestions by global popularity/frequency
- Update suggestions with trending queries within 10-15 minutes
- Support spell correction for common typos before prefix lookup
- Personalize suggestions by blending user history with global top-K
- Filter inappropriate content from suggestions
- Log query events for frequency aggregation pipeline

### Non-Functional Requirements
- **Latency**: < 100ms end-to-end; < 20ms P99 server processing time
- **Throughput**: 10M QPS at peak
- **Availability**: 99.99% uptime (< 1 hour/year downtime)
- **Consistency**: Eventual consistency acceptable; stale suggestions by < 15 minutes is fine
- **Storage**: Trie for top-500K queries; full index for 10B+ historical queries
- **Scalability**: Horizontally scalable; geographic distribution
- **Read-heavy**: 100:1 read-to-write ratio; optimize for reads

---

## 3. Capacity Estimation

### Traffic
- 100M DAU, each making ~20 searches/day = 2B searches/day
- Each search triggers ~5 prefix requests (debounced keystrokes) = 10B prefix requests/day
- Peak: 10M QPS (3× average)

### Data Size
- Unique queries in corpus: ~10B distinct queries
- Top queries serving suggestions: top 500K (cover ~80% of traffic)
- Average query length: 20 characters
- Trie storage: 500K queries × 20 chars × 16 bytes/node ≈ 160 MB (fits in RAM)
- Full trie for 10B queries: ~100 GB (distributed across machines)

### Query Log Volume
- 2B searches/day × 100 bytes/event = 200 GB/day of raw query logs
- After aggregation (top-1M queries + counts): ~50 MB/day delta updates

### Bandwidth
- Request: ~30 bytes per prefix query
- Response: ~500 bytes (10 suggestions × 50 chars each)
- 10M QPS × 530 bytes = 5.3 GB/s outbound (distributed across CDN)

---

## 4. High-Level Architecture (ASCII Diagram)

```
  ┌──────────────────────────────────────────────────────────────────────┐
  │                          CLIENT BROWSER / APP                        │
  │  Debounce 50-100ms │ Cancel prior request │ Show suggestions inline  │
  └───────────────────────────────┬──────────────────────────────────────┘
                                  │  GET /suggest?q=pre&uid=123
                          ┌───────▼──────────────────────────────────┐
                          │          CDN / EDGE CACHE                 │
                          │  Cache popular prefixes (TTL: 60s)        │
                          │  Geographic distribution                  │
                          └───────────────┬──────────────────────────┘
                                          │ Cache miss
                          ┌───────────────▼──────────────────────────┐
                          │         API GATEWAY / LOAD BALANCER       │
                          │  Rate limiting │ Auth │ Request routing   │
                          └──────────┬──────────────────┬─────────────┘
                                     │                  │
              ┌──────────────────────▼──────┐  ┌───────▼──────────────────┐
              │    SUGGESTION SERVICE        │  │  PERSONALIZATION SERVICE │
              │  Prefix → Top-K lookup       │  │  User history top-K      │
              │  Spell correction            │  │  Blend with global top-K │
              │  Safe search filter          │  └──────────────────────────┘
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────────────────────────────────┐
              │                   TRIE STORE (Redis Cluster)             │
              │  prefix → [suggestion_1, suggestion_2, ..., suggestion_10]│
              │  Sharded by first 2 characters of prefix                 │
              └──────────────┬──────────────────────────────────────────┘
                             │ Cache miss / rebuild
              ┌──────────────▼──────────────────────────────────────────┐
              │              IN-MEMORY TRIE SERVICE                      │
              │  Full trie loaded in RAM per shard                       │
              │  top-K at each node (min-heap size K)                    │
              └──────────────┬──────────────────────────────────────────┘
                             │
  ┌──────────────────────────▼────────────────────────────────────────────┐
  │                    OFFLINE PIPELINE (Trending Updates)                 │
  │                                                                        │
  │  Raw Query Logs → Kafka → Flink Aggregation → Top-K per prefix        │
  │       (Kafka)       (stream)    (5-min windows)   (trie update)        │
  │                                                                        │
  │  Daily full rebuild: MapReduce/Spark → new trie → blue-green deploy   │
  └────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Component Deep-Dive

### 5.1 Trie Data Structure
The core data structure for prefix lookup:

**TrieNode Structure:**
```
TrieNode {
    children: HashMap<char, TrieNode>  // 26 for English
    is_end: bool
    frequency: long
    top_k: MinHeap<(frequency, query)>  // size K=10
}
```

**Top-K Tracking per Node:**
- Each node maintains a min-heap of size K containing the K most frequent completions passing through it
- On insert: propagate up the trie — if new frequency > min in heap, replace and bubble down
- Space: O(N × K) where N = total nodes; for 500K queries × 10 = 5M entries

**Trie Operations:**
- `insert(query, frequency)`: O(L) where L = query length
- `search(prefix)`: O(L + K log K) — traverse to prefix node, return sorted top-K
- `update_frequency(query, delta)`: O(L × K) — update all ancestor nodes' heaps

**Why Trie over Inverted Index:**
- Trie: O(L) prefix traversal, no scoring overhead, perfect for autocomplete
- Inverted Index: Better for full-text search; overkill for prefix-only matching
- Elasticsearch completion suggester: Uses FST (Finite State Transducer) — space-efficient trie variant

### 5.2 Distributed Trie Partitioning
- **Strategy**: Partition by prefix range (a-f, g-m, n-z on 3 machines)
- **Replication**: Each partition replicated 3× for fault tolerance
- **Routing**: API gateway routes prefix request to correct shard based on first 2 chars
- **Hot spots**: Popular prefixes (like "the", "how") → consistent hash with virtual nodes
- **Trie size per shard**: ~60 MB (fits in L3 cache on modern CPUs)

### 5.3 Caching Layer (Redis)
- **What to cache**: Precomputed results for top-10K most popular prefixes
- **Cache key**: `suggest:{prefix}:{language}` → JSON array of suggestions
- **TTL**: 60 seconds for real-time freshness; 5 minutes for less popular prefixes
- **Hit rate**: ~80% cache hit for top-10K prefixes covering 90% of traffic
- **Eviction**: LRU with maxmemory-policy = allkeys-lru

### 5.4 Query Log Pipeline
Data collection → frequency update pipeline:
1. **Collection**: Every search query logged to Kafka topic `query-events`
2. **Stream aggregation**: Flink job consumes Kafka, aggregates counts in 5-minute tumbling windows
3. **Top-K computation**: For each active prefix in window, compute top-10 queries
4. **Delta updates**: Only update trie nodes where top-K has changed
5. **Full rebuild**: Nightly Spark job re-ranks all queries from complete logs; deploys new trie

### 5.5 Spell Correction
- **Algorithm**: BK-tree (Burkhard-Keller tree) for nearest-neighbor search in edit-distance space
- **Edit distance**: Levenshtein distance ≤ 2 for corrections
- **Noisy channel model**: P(intended|typed) ∝ P(typed|intended) × P(intended)
- **Implementation**: Check if prefix exists in trie; if not, find closest prefix within edit distance 1
- **Latency**: BK-tree lookup < 1ms for dictionary of 500K words

### 5.6 Personalization
- **User history**: Store last 30 days of user search queries in user profile store (Redis/Cassandra)
- **Blending**: `score = α × global_freq + (1-α) × user_freq` where α=0.7 by default
- **Cold start**: New users get global top-K only
- **Privacy**: User history kept client-side in browser localStorage for privacy-sensitive deployments

### 5.7 Client-Side Optimizations
- **Debouncing**: Wait 50-100ms after last keystroke before sending request
- **Request cancellation**: Cancel in-flight request when new keystroke arrives
- **Prefix caching**: Cache recent prefix results in browser memory (LRU, max 50 entries)
- **Speculative prefetching**: Preload top-3 likely next characters' suggestions

---

## 6. Database Design

### Query Frequency Store (HBase / BigTable)
```
Row key: query_string (normalized lowercase, trimmed)
Columns:
  cf:global_count    → long (total search count all time)
  cf:daily_count     → long (count today)
  cf:weekly_count    → long (count this week)
  cf:last_updated    → timestamp
  cf:language        → string
  cf:safe_search     → boolean (pre-filtered flag)
```

### User Query History (Cassandra)
```sql
CREATE TABLE user_query_history (
    user_id     UUID,
    queried_at  TIMESTAMP,
    query       TEXT,
    clicked_url TEXT,
    PRIMARY KEY (user_id, queried_at)
) WITH CLUSTERING ORDER BY (queried_at DESC)
  AND default_time_to_live = 2592000;  -- 30 days TTL
```

### Prefix Cache (Redis Cluster)
```
suggest:{prefix} → LIST of JSON strings
  [
    {"query": "how to cook pasta", "freq": 15000000},
    {"query": "how to lose weight", "freq": 12000000},
    ...
  ]
TTL: 60 seconds
```

---

## 7. API Design

### Suggestion API
```
GET /v1/suggest?q={prefix}&limit=10&uid={user_id}&lang=en

Response 200 OK:
{
  "prefix": "how to",
  "suggestions": [
    {"query": "how to cook pasta",     "score": 0.95},
    {"query": "how to lose weight",    "score": 0.91},
    {"query": "how to tie a tie",      "score": 0.87},
    {"query": "how to make money",     "score": 0.83},
    {"query": "how to get a passport", "score": 0.79}
  ],
  "source": "cache",
  "response_time_ms": 8
}

GET /v1/suggest?q=hwo+to  (typo)
→ spell-corrected to "how to" then same response
```

### Query Log API (internal)
```
POST /v1/query-log
Body: { "query": "how to cook pasta", "uid": "u123", "session": "s456", "ts": 1700000000 }
→ Fire-and-forget; async write to Kafka
```

---

## 8. Scalability & Bottlenecks

### Bottleneck 1: Trie Update Propagation
- Every frequency update requires updating O(L) nodes in trie
- **Solution**: Lazy propagation — batch updates every 5 minutes rather than per query

### Bottleneck 2: Hot Prefix Contention
- Prefix "the" → shared by millions of queries; lock contention on single trie node
- **Solution**: Immutable trie snapshots; updates via copy-on-write, atomic pointer swap

### Bottleneck 3: CDN Cache Invalidation
- Trending query appears but CDN still serves stale suggestions
- **Solution**: TTL-based expiry (60s); purge CDN for top-1000 prefixes on major trend events

### Bottleneck 4: Personalization Latency
- User history lookup adds 5-10ms to critical path
- **Solution**: Precompute personalized top-K asynchronously; serve pre-cached result

### Bottleneck 5: Memory for Full Trie
- 10B queries × avg 20 nodes × 64 bytes = ~12 TB RAM (infeasible single machine)
- **Solution**: Only keep top-500K queries in hot trie; remainder in Elasticsearch for long-tail

---

## 9. Trade-offs & Design Decisions

### Decision 1: Trie vs Elasticsearch Completion Suggester
- **Trie (custom)**: Full control, optimal performance, but complex distributed management
- **Elasticsearch FST**: Production-ready, handles 100K+ QPS easily, built-in scoring
- **Choice**: Custom trie for < 100ms latency requirement; Elasticsearch as fallback for long-tail

### Decision 2: Push vs Pull for Trie Updates
- **Push (real-time)**: Update trie on every query; stale by 0ms but high write load
- **Pull (batch)**: Aggregate 5-min windows; stale by 5 min but low write load
- **Choice**: Batch updates every 5 minutes; acceptable for non-breaking news freshness

### Decision 3: Global vs Per-Region Trie
- **Global**: Single source of truth; 100ms latency for cross-continent requests
- **Per-region**: < 20ms latency; but regional divergence in suggestions
- **Choice**: Per-region tries, synced from global pipeline every 5 minutes

### Decision 4: Personalization Depth
- **Deep personalization**: ML model scores; complex, latency-heavy
- **Simple blending**: Linear combination of user history and global; fast, interpretable
- **Choice**: Simple blending for < 10ms overhead; ML scoring for logged-in power users

### Decision 5: Safe Search Filtering
- **Pre-filter at index time**: Tag queries as safe/unsafe during trie build; zero runtime cost
- **Runtime filter**: Check each suggestion against blocklist; ~1ms overhead
- **Choice**: Pre-filter at build time; runtime blocklist for newly flagged queries

---

## 10. Key Interview Talking Points

### 1. Trie Top-K Propagation
Explain how maintaining a min-heap of size K at each trie node avoids doing a full subtree traversal on every prefix search. The trade-off is O(N×K) memory vs O(N) pure trie memory.

### 2. Frequency vs Recency Scoring
- Pure frequency: "chicken pox" ranked high from 2005 even if rarely searched now
- Solution: Exponential decay — `score = frequency × e^(-λ × days_since_last_search)`
- Trending boost: `score += trending_multiplier` if query volume doubled in last hour

### 3. The 10M QPS Problem
- No single server can handle 10M QPS
- CDN caches top 10K prefixes → covers ~70% of traffic
- Remaining 3M QPS → 100 suggestion service instances (30K QPS each → easy)
- Trie sharded across 50 machines → 200K QPS per shard

### 4. Trie Serialization
- Serialize trie to disk: DFS traversal, each node = (char, freq, num_children)
- Compressed representation: Patricia trie (compact common prefixes)
- Loading time: 160 MB trie loads in ~200ms on startup (acceptable)

### 5. Handling New vs Trending Queries
- New queries not in trie until next batch update
- Trending (e.g., breaking news): Flink real-time job detects 10× spike → immediate trie update
- Two-tier: hot path (real-time for trending), cold path (batch for stable queries)

### 6. Why Not Just Use a Database?
- SQL `WHERE query LIKE 'pre%'` with B-tree index: O(log N + K) but full scan of matching rows
- At 10M QPS, database cannot sustain; requires in-memory trie or equivalent

### 7. Client-Side Debouncing Impact
- Without debounce: 10M users × 10 keystrokes/second = 100M QPS → impossible
- With 100ms debounce: Average 3-4 requests per search session = 40× reduction
- Debounce is free latency optimization done on client with no server changes
