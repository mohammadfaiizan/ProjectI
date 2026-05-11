# HLD Interview Q&A — File 15: Search System Design

---

## Easy Questions (Q1–Q7)

---

### Q1. How does an inverted index work?

**Answer:**

An inverted index is the core data structure behind every search engine. It maps terms (words) to the list of documents that contain them — the inverse of a document's word list.

**Forward index (naive approach):**
```
Doc 1: "the quick brown fox"
Doc 2: "the lazy brown dog"
Doc 3: "quick brown cat"

To search for "brown": scan all 3 documents → O(N) where N = corpus size
```

**Inverted index:**
```
Term      │ Posting List (doc_id + position)
──────────┼──────────────────────────────────
the       │ [Doc1(pos:0), Doc2(pos:0)]
quick     │ [Doc1(pos:1), Doc3(pos:0)]
brown     │ [Doc1(pos:2), Doc2(pos:2), Doc3(pos:1)]
fox       │ [Doc1(pos:3)]
lazy      │ [Doc2(pos:1)]
dog       │ [Doc2(pos:3)]
cat       │ [Doc3(pos:2)]

Search for "brown": → look up "brown" → [Doc1, Doc2, Doc3] → O(1) lookup
```

**Building the index:**
```
Input document → Tokenization (split into terms)
              → Normalization (lowercase, remove punctuation)
              → Stop word removal (the, a, is)
              → Stemming / Lemmatization (running → run)
              → Update posting lists for each term
```

**Posting list contents:**
- Document ID (which document contains the term)
- Term frequency (how many times the term appears — used for TF-IDF scoring)
- Position list (where in the document — enables phrase matching)

**Boolean search with inverted index:**
```
Query: "brown AND fox"
1. Look up "brown" → {Doc1, Doc2, Doc3}
2. Look up "fox"   → {Doc1}
3. Intersection    → {Doc1}

Intersection of sorted posting lists: O(m+n) where m,n = posting list sizes
```

**Disk layout:** Lucene (the engine behind Elasticsearch) stores the inverted index in immutable segment files. New documents are added to new segments; segments are periodically merged.

---

### Q2. What is TF-IDF, and what is the intuition behind it?

**Answer:**

**TF-IDF (Term Frequency–Inverse Document Frequency)** is a ranking formula that scores how important a term is to a document within a collection. It solves the problem of common words ("the", "is") unfairly ranking high just because they appear frequently.

**Term Frequency (TF):**
How often does the term appear in this specific document?
```
TF(term, doc) = count(term in doc) / total_terms_in_doc

Document: "cat sat on the cat mat"
TF("cat", doc) = 2/6 = 0.333
TF("the", doc) = 1/6 = 0.167
```

**Inverse Document Frequency (IDF):**
How rare is the term across all documents? Rare terms are more informative.
```
IDF(term) = log(N / df(term))

Where: N = total documents, df = number of documents containing term

N = 1,000,000 documents
"cat"    appears in 50,000 docs: IDF = log(1M / 50K) = log(20)  = 2.99
"the"    appears in 999,000 docs: IDF = log(1M / 999K) ≈ 0.001
"hadoop" appears in 1,000 docs:   IDF = log(1M / 1K) = log(1000) = 6.9
```

**TF-IDF Score:**
```
TF-IDF(term, doc) = TF(term, doc) × IDF(term)

"cat" in our example doc:    0.333 × 2.99 = 0.997
"the" in our example doc:    0.167 × 0.001 = 0.000167
"hadoop" (if in doc once):   0.167 × 6.9  = 1.15

→ "hadoop" scores highest (very rare term, document is likely about Hadoop)
→ "the" scores near zero (common everywhere, not informative)
```

**Document ranking for a multi-term query:**
```
Query: "brown fox"
Score(doc) = TF-IDF("brown", doc) + TF-IDF("fox", doc)

Documents with both rare terms score higher.
```

**Intuition:** TF-IDF rewards documents that use a query term frequently (TF) when that term is rare across all documents (IDF). A document about "cryptography" scores high for "asymmetric" because asymmetric is rare; it scores low for "the" even if it contains many of them.

---

### Q3. What is BM25, and how does it improve over TF-IDF?

**Answer:**

**BM25 (Best Match 25)** is the ranking function used by Elasticsearch (and most modern search engines) by default. It addresses two weaknesses of raw TF-IDF.

**Problem 1: TF saturation**
Raw TF grows linearly with term frequency. A document mentioning "python" 100 times should not be 100x more relevant than one mentioning it 1 time.

BM25 applies a saturation function:
```
TF component = (k1 + 1) × tf / (k1 × (1 - b + b × |d|/avgdl) + tf)

Where:
  k1 = term frequency saturation (default 1.2)
  b  = field length normalization (default 0.75)
  |d| = document length
  avgdl = average document length

As tf → ∞: component approaches (k1 + 1) = 2.2 (saturates, does not grow infinitely)
```

**Problem 2: Document length normalization**
TF-IDF unfairly favors long documents (more chances to mention the query term). BM25 normalizes by document length.

```
Short doc: "Python is great" — "Python" = 1/3 of words
Long doc:  (1000 words essay about Python, mentions "Python" 5 times) — "Python" = 5/1000

BM25 recognizes the short doc is more focused on the topic.
```

**BM25 full formula:**
```
Score(q, d) = Σ IDF(qi) × [tf(qi,d) × (k1+1)] / [tf(qi,d) + k1 × (1 - b + b × |d|/avgdl)]
              i

Where the sum is over all query terms qi
```

**TF-IDF vs BM25 comparison:**

| Issue                    | TF-IDF          | BM25                        |
|--------------------------|-----------------|-----------------------------|
| TF saturation            | No (linear)     | Yes (asymptotic ceiling)    |
| Document length norm     | Basic           | Tunable (b parameter)       |
| Field length bias        | Yes             | Controlled                  |
| Tunable parameters       | None            | k1, b (default 1.2, 0.75)  |
| Accuracy                 | Baseline        | ~10-15% better in practice  |

BM25 is the current standard — Elasticsearch has used it as the default since version 5.0 (replacing TF-IDF).

---

### Q4. What is the Elasticsearch architecture? Explain indexes, shards, replicas, and nodes.

**Answer:**

Elasticsearch is a distributed search engine built on Apache Lucene. Understanding its architecture is essential for using it effectively.

**Hierarchy:**
```
Cluster
  └── Nodes (JVM processes, each hosts shards)
        └── Index (logical collection of documents, like a DB table)
              └── Shards (physical Lucene index, unit of parallelism)
                    └── Documents (JSON objects, the actual data)
```

**Node types:**
```
Master node:    Cluster metadata, index creation, node membership (not data)
Data node:      Stores shards, executes queries (most nodes are data nodes)
Coordinating:   Routes queries, aggregates results (no data storage)
Ingest node:    Pre-process documents before indexing (transform pipelines)
```

**Sharding:**
An index is divided into N primary shards. The number is fixed at index creation.
```
routing = hash(routing_key) % number_of_primary_shards
Default routing_key = document _id

Index: "products", 3 primary shards
product_id="abc" → hash % 3 = 1 → stored on Shard 1
product_id="xyz" → hash % 3 = 0 → stored on Shard 0
```

**Replication:**
Each primary shard has R replica shards on different nodes for fault tolerance and read scaling.
```
Cluster: 3 nodes, 1 index, 3 primary shards, 1 replica each (total 6 shards)

Node 1: P0 (Primary 0),  R1 (Replica of Primary 1)
Node 2: P1 (Primary 1),  R2 (Replica of Primary 2)
Node 3: P2 (Primary 2),  R0 (Replica of Primary 0)

Rule: Primary and its replica are NEVER on the same node
```

**Query execution:**
```
1. Client sends query to any node (becomes coordinating node)
2. Coordinating node broadcasts query to all relevant shards (primary or replica)
3. Each shard returns local top-N results
4. Coordinating node merges, re-scores, returns global top-N
```

**Near Real-Time (NRT) indexing:**
Documents are not immediately searchable. After indexing, Elasticsearch must refresh (create a new Lucene segment reader) before the document is visible. Default refresh interval: 1 second.

---

### Q5. What is fuzzy search, and how do edit distance and trigram indexing work?

**Answer:**

**Fuzzy search** finds documents even when the query contains typos or near-matches. It is implemented via edit distance (Levenshtein distance) and trigram indexing.

**Edit Distance (Levenshtein Distance):**
The minimum number of single-character edits (insertions, deletions, substitutions) to transform one string into another.
```
"color" → "colour": 1 edit (insert 'u')         distance = 1
"apple" → "aple":   1 edit (delete 'p')          distance = 1
"cat"   → "dog":    3 edits                      distance = 3
"kitten"→ "sitting":3 edits                      distance = 3
```

**Elasticsearch fuzzy query:**
```json
{
  "query": {
    "fuzzy": {
      "title": {
        "value": "recieve",    // Typo: should be "receive"
        "fuzziness": "AUTO",   // AUTO: distance 1 for len<6, 2 for len>=6
        "prefix_length": 2     // First 2 chars must match exactly (performance)
      }
    }
  }
}
```

**Problem with pure edit distance:** Computing edit distance between a query and every document term is O(query_len × term_len × vocabulary_size) — too slow.

**Trigram indexing (fast approximate matching):**
A trigram is a sequence of 3 consecutive characters. Index documents by their trigrams; queries that share trigrams are likely matches.
```
"color" → trigrams: [col, olo, lor]
"colour"→ trigrams: [col, olo, lou, our]

Shared trigrams: col, olo (2 of 3 from "color")
Similarity = 2/3 = 0.67 → likely match
```

**Trigram similarity in PostgreSQL:**
```sql
-- pg_trgm extension
SELECT name, similarity(name, 'recieve') AS sim
FROM products
WHERE name % 'recieve'  -- % operator uses trigram similarity threshold
ORDER BY sim DESC;

-- Create GIN index for fast trigram search
CREATE INDEX idx_products_trgm ON products USING GIN (name gin_trgm_ops);
```

**Elasticsearch approach:** Uses the `fuzziness` parameter which internally uses the Levenshtein automaton — a state machine that efficiently finds all terms within edit distance N without comparing against every term in the vocabulary.

---

### Q6. What is faceted search, and how is it implemented with aggregations?

**Answer:**

**Faceted search** provides users with a breakdown of search results by different categories (facets), allowing them to interactively narrow their results. It is the left-panel filter system on every e-commerce site.

```
Search results for "laptop":

[ ] Facets ─────────────────────
Brand:
  [✓] Apple (342)
  [ ] Dell (289)
  [ ] Lenovo (201)

Price:
  [ ] Under $500 (189)
  [✓] $500–$1000 (421)
  [ ] Over $1000 (222)

RAM:
  [ ] 8 GB (312)
  [✓] 16 GB (398)
  [ ] 32 GB (122)
─────────────────────────────────
Showing results filtered by: Apple, $500–$1000, 16GB
```

**Elasticsearch implementation:**
```json
{
  "query": {
    "bool": {
      "must": [{"match": {"name": "laptop"}}],
      "filter": [
        {"term": {"brand": "apple"}},
        {"range": {"price": {"gte": 500, "lte": 1000}}},
        {"term": {"ram_gb": 16}}
      ]
    }
  },
  "aggs": {
    "by_brand": {
      "terms": {"field": "brand.keyword", "size": 10}
    },
    "by_price_range": {
      "range": {
        "field": "price",
        "ranges": [
          {"to": 500},
          {"from": 500, "to": 1000},
          {"from": 1000}
        ]
      }
    },
    "by_ram": {
      "terms": {"field": "ram_gb", "size": 5}
    }
  }
}
```

**Important design consideration — post-filter vs filter:**
```
Scenario: User selects "Apple" brand filter. 
Should brand facet still show other brand counts?
  
  YES (multi-select facets): Use "post_filter" for filtering results,
       but aggregations run on unfiltered set.
  
  NO (single-select): Use "filter" context — aggregations are scoped to filtered set.
```

**Hierarchical facets:**
```
Category → Sub-category
Electronics → Laptops → Gaming Laptops

Implementation: nested terms aggregation
aggs: {
  "by_category": {
    "terms": {"field": "category"},
    "aggs": {
      "by_subcategory": {"terms": {"field": "subcategory"}}
    }
  }
}
```

---

### Q7. How does a typeahead/autocomplete system work? Compare trie vs Elasticsearch completion suggester.

**Answer:**

Autocomplete (typeahead) shows suggestions as a user types, completing their query before they finish. Low latency is critical — suggestions must appear within 100ms.

**Approach 1: Trie (Prefix Tree)**
A trie stores strings such that each path from root to leaf represents a complete term. Prefix search is O(prefix_length).

```
Trie containing: ["apple", "application", "apply", "apt"]

        root
        │
        a
        │
        p
       / \
      p   t
      |   |
      l   (apt)
     / \
    e   i
    |   |
  (apple) c
          ...
        (application)
```

**Pros:** Very fast prefix lookups (O(L) where L = prefix length), deterministic.
**Cons:** Memory-intensive (each character is a node), harder to rank by popularity, not distributed.

**Implementation with Redis Sorted Set (popular in interviews):**
```python
# Pre-populate: for each suggestion, add all prefixes
def add_suggestion(redis_client, term: str, score: int):
    for i in range(1, len(term) + 1):
        prefix = term[:i]
        redis_client.zadd("autocomplete", {term: score}, nx=False)

# Query: all terms starting with prefix
def get_suggestions(redis_client, prefix: str, limit: int = 10):
    # Sorted set rangebylex between prefix and prefix\xff
    return redis_client.zrangebylex("autocomplete", f"[{prefix}", f"[{prefix}\xff", 0, limit)
```

**Approach 2: Elasticsearch Completion Suggester**
Elasticsearch has a built-in completion suggester using an in-memory FST (Finite State Transducer) — a compressed trie.

```python
# Index mapping
PUT /search_suggestions
{
  "mappings": {
    "properties": {
      "suggest": {
        "type": "completion",
        "analyzer": "simple"
      },
      "popularity": {"type": "integer"}
    }
  }
}

# Index a suggestion
POST /search_suggestions/_doc
{
  "suggest": {"input": ["iphone", "iphone 15", "iphone pro"], "weight": 95}
}

# Query
POST /search_suggestions/_search
{
  "suggest": {
    "product_suggest": {
      "prefix": "ipho",
      "completion": {"field": "suggest", "size": 5, "skip_duplicates": true}
    }
  }
}
```

**Comparison:**

| Dimension          | Trie (custom)              | ES Completion Suggester    |
|--------------------|----------------------------|----------------------------|
| Latency            | < 1ms (in-memory)          | < 5ms                      |
| Ranking            | Custom (score field)       | Built-in weight            |
| Typo tolerance     | No                         | Yes (fuzziness param)      |
| Distribution       | Manual (Redis cluster)     | Built-in                   |
| Implementation     | More work                  | Out-of-the-box             |
| Best for           | Simple prefix, high volume | Rich autocomplete features |

---

## Medium Questions (Q8–Q15)

---

### Q8. How do you design search for high write throughput in Elasticsearch?

**Answer:**

Elasticsearch uses an immutable segment model (Lucene). Write throughput is limited by how quickly segments can be created and merged. High write throughput requires tuning the write path.

**The write path:**
```
Document indexed
    ↓
In-memory indexing buffer (heap)
    ↓
[refresh: every 1 second] → New Lucene segment (document visible)
    ↓
[flush: when buffer full or translog size] → Segment committed to disk + translog cleared
    ↓
[merge: background] → Small segments merged into larger segments
```

**Tuning for high write throughput:**

**1. Increase refresh interval:**
```json
PUT /my_index/_settings
{
  "refresh_interval": "30s"   // Default: 1s
}
// Documents not searchable for up to 30s, but massive write throughput gain
// Set to -1 during initial bulk load: completely disable refresh
```

**2. Use bulk API:**
```json
// Never index one document at a time
POST /_bulk
{"index": {"_index": "products", "_id": "1"}}
{"name": "Product A", "price": 99.0}
{"index": {"_index": "products", "_id": "2"}}
{"name": "Product B", "price": 49.0}
// ... hundreds more

// Optimal batch size: 5–15MB or 1000–5000 documents
```

**3. Reduce replicas during bulk load:**
```json
PUT /my_index/_settings
{"number_of_replicas": 0}   // No replication overhead during load
// After load: restore to production value
PUT /my_index/_settings
{"number_of_replicas": 1}
```

**4. Tune shard size and count:**
```
Rule: 20–50 GB per shard
Too many shards: overhead per shard (master state, memory)
Too few: limits parallelism and recovery speed
Target: Total shards ≤ 20× number of data nodes
```

**5. Increase indexing threadpool:**
```yaml
# elasticsearch.yml
thread_pool.write.queue_size: 1000  # Default 200
```

**6. Use async indexing (if consistency allows):**
```json
POST /products/_doc?wait_for_active_shards=1  // Don't wait for replicas
```

**Monitoring write throughput:**
```
Key metrics:
  indexing_rate (docs/sec)
  merge_time (high = bottleneck)
  indexing_throttle_time (high = I/O bound)
  segment_count (high = too many small segments, merge needed)
```

---

### Q9. What are search ranking signals beyond text relevance?

**Answer:**

Pure text relevance (BM25) is necessary but insufficient for a great search experience. Modern search systems combine text relevance with dozens of other signals.

**Categories of ranking signals:**

**Freshness signals:**
```
Recency bias: More recently published/updated content ranks higher.
  score += freshness_boost × exp(-decay_rate × (now - publish_date))
  
Time-sensitive queries: "iPhone launch" → recent results more relevant
Time-independent queries: "how to tie a tie" → freshness less important
```

**Authority signals:**
```
PageRank: Pages linked to by many high-authority pages rank higher.
Domain authority: Trust score for the domain.
In-network authority: For internal search (e-commerce), seller rating, product quality.
```

**Personalization signals:**
```
Click-through rate (CTR): Documents frequently clicked for this query rank higher.
Dwell time: If user spends 5 min on a result → strong relevance signal.
Purchase history: Show products similar to what user bought before.
Geographic location: "coffee shop" → nearby locations rank higher.
Language preference: Match user's preferred language.
```

**Quality signals:**
```
Spam/content quality score: Penalize thin content, duplicate content.
Image quality: Images with better resolution/clarity rank higher.
Bounce rate: High bounce rate → document not satisfying intent.
```

**Learning to Rank (LTR):**
Rather than hand-tuning weights, train an ML model to predict relevance given a feature vector of all signals.
```
Features: [bm25_score, freshness_days, ctr_last30d, authority_score, 
           user_past_clicks, price_match_budget, review_score, ...]

Training data: Human-labeled judgments (query, document, relevance_label)
               + Implicit signals (clicks, dwell time, purchases)

Model: Gradient Boosted Trees (XGBoost, LightGBM)
       Output: Relevance score [0, 1]
       
Output replaces or supplements BM25 as the primary ranking signal.
```

**Elasticsearch function score query:**
```json
{
  "query": {
    "function_score": {
      "query": {"match": {"title": "laptop"}},
      "functions": [
        {
          "exp": {
            "publish_date": {"origin": "now", "scale": "30d", "decay": 0.5}
          }
        },
        {"field_value_factor": {"field": "popularity_score", "factor": 0.1}}
      ],
      "score_mode": "sum",
      "boost_mode": "multiply"
    }
  }
}
```

---

### Q10. How does spell correction work? Explain the Noisy Channel Model.

**Answer:**

Spell correction in search must distinguish between a typo that should be corrected ("recieve" → "receive") and a valid but rare term ("Feynman" should not be corrected).

**Noisy Channel Model:**
The user intended a correct query Q, but their input Q' was produced by a "noisy channel" (the keyboard and human error). We want to find the Q that most likely produced Q'.

```
P(Q | Q') ∝ P(Q' | Q) × P(Q)
                ↑           ↑
          Error model    Language model
          (how likely    (how common
           is this        is this
           typo?)         query?)

Best correction = argmax_Q [P(Q' | Q) × P(Q)]
```

**Error model P(Q' | Q):**
Based on edit distance and keyboard layout. "r" → "t" (adjacent keys) is more likely than "r" → "z".

```
Confusion matrices built from typo datasets:
  P("recieve" | "receive") = 0.15  (common transposition: ie/ei)
  P("recieve" | "relieve") = 0.001 (different word structure)
```

**Language model P(Q):**
Frequency of the query in a large corpus. "receive" appears in billions of documents; "recieve" appears rarely (it is a misspelling).

```
From query logs or web corpus:
  P("receive") = high (common word)
  P("recieve") = very low (rare, mostly errors)
```

**Practical implementation:**
1. **Elasticsearch suggest API:** Uses the same inverted index, finds terms within edit distance N.
2. **SymSpell:** Very fast correction using pre-computed delete variants.
3. **Peter Norvig's spell corrector:** Simple, elegant, uses word frequency from corpus.

```python
# Norvig's approach
def edits1(word):
    """All strings 1 edit away"""
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in 'abcdefghijklmnopqrstuvwxyz']
    inserts = [L + c + R for L, R in splits for c in 'abcdefghijklmnopqrstuvwxyz']
    return set(deletes + transposes + replaces + inserts)

def correct(word, word_counts):
    candidates = edits1(word) | {word}
    valid = candidates & word_counts.keys()
    return max(valid, key=word_counts.get) if valid else word
```

**Context-aware correction:**
"I saw the dessert" vs "I walked through the desert" — single-word spelling is not enough. Use language model context (BERT-based models can correct in context).

---

### Q11. What is real-time search indexing pipeline from Kafka to Elasticsearch?

**Answer:**

Building a real-time indexing pipeline ensures that changes in a source database are reflected in the search index within seconds.

**Architecture:**
```
Source Database (PostgreSQL)
        │
        │  Debezium CDC (reads WAL)
        ▼
Kafka Topic: "product_changes"
  {op: "u", before: {...}, after: {"id":"123", "name":"New Nike Shoe", "price": 130}}
        │
        │  Elasticsearch Sink Connector (Kafka Connect)
        │  or custom consumer
        ▼
Elasticsearch Index: "products"
```

**Kafka Connect Elasticsearch Sink Connector:**
The simplest production approach — a managed connector that reads from Kafka and writes to Elasticsearch automatically.

```json
// Connector configuration
{
  "name": "es-products-sink",
  "config": {
    "connector.class": "io.confluent.connect.elasticsearch.ElasticsearchSinkConnector",
    "tasks.max": "4",
    "topics": "product_changes",
    "connection.url": "http://elasticsearch:9200",
    "type.name": "_doc",
    "key.ignore": "false",
    "schema.ignore": "true",
    "behavior.on.null.values": "delete",  // CDC delete → ES delete
    "batch.size": "2000",
    "max.buffered.records": "20000"
  }
}
```

**Custom consumer for transformation:**
```python
consumer = KafkaConsumer("product_changes", ...)
es = Elasticsearch(["http://elasticsearch:9200"])

def transform_for_search(change_event):
    if change_event["op"] == "d":  # Delete
        return None
    
    doc = change_event["after"]
    return {
        "_id": doc["id"],
        "name": doc["name"],
        "name_suggest": {"input": doc["name"].split(), "weight": doc["popularity"]},
        "price": doc["price"],
        "brand": doc["brand"],
        "full_text": f"{doc['name']} {doc['brand']} {doc['description']}",
        "indexed_at": datetime.utcnow().isoformat()
    }

bulk_buffer = []
for message in consumer:
    event = json.loads(message.value())
    doc = transform_for_search(event)
    
    if doc is None:
        bulk_buffer.append({"delete": {"_index": "products", "_id": event["before"]["id"]}})
    else:
        bulk_buffer.append({"index": {"_index": "products", "_id": doc["_id"]}})
        bulk_buffer.append(doc)
    
    if len(bulk_buffer) >= 500:
        es.bulk(body=bulk_buffer)
        bulk_buffer = []
```

**Latency budget:**
```
DB write → WAL → Debezium captures → Kafka produce → Consumer processes → ES indexes → Refresh
  ~0ms      ~10ms    ~50ms              ~5ms             ~10ms              ~50ms         ~1s
Total end-to-end: ~2 seconds
```

Reduce to < 500ms by decreasing `refresh_interval` and optimizing batch sizes.

---

### Q12. How do you perform a zero-downtime index rebuild strategy using alias swapping?

**Answer:**

Elasticsearch indices cannot have their mapping changed after creation (adding new fields is fine, but changing field types requires a full rebuild). A zero-downtime rebuild uses an alias pattern to switch traffic atomically.

**The problem:**
```
Old index: "products_v1" (type: "category" = keyword)
New requirement: "category" needs to be full-text AND keyword (for both search and aggregations)
→ Must rebuild the entire index with new mapping
→ Cannot stop search traffic during rebuild (may take hours for millions of documents)
```

**Solution: Alias pattern**
```
Step 1: Production traffic reads from alias "products" → points to "products_v1"

Step 2: Create new index with correct mapping
        PUT /products_v2 {correct mapping}

Step 3: Re-index all data from v1 to v2
        POST /_reindex
        {
          "source": {"index": "products_v1"},
          "dest": {"index": "products_v2"}
        }
        (This takes time; meanwhile v1 continues serving traffic)

Step 4: Replay changes that occurred during reindex
        (Track changes via "indexing_date" field or replay from Kafka)

Step 5: Atomic alias swap (zero downtime)
        POST /_aliases
        {
          "actions": [
            {"add":    {"index": "products_v2", "alias": "products"}},
            {"remove": {"index": "products_v1", "alias": "products"}}
          ]
        }
        (This is atomic — no moment where alias points to nothing)

Step 6: Verify v2 is serving traffic correctly
        Monitor error rates, result quality

Step 7: Delete old index
        DELETE /products_v1
```

**Write alias pattern (for real-time pipelines):**
```
Read alias:  "products"           → always points to current live index
Write alias: "products_write"     → points to current live index

During reindex:
  Write alias points to BOTH v1 and v2 (new documents go to both)
  This ensures v2 doesn't miss writes during the reindex window
  
After alias swap: write alias updated to point to v2 only
```

**Index lifecycle management (ILM) for time-series:**
For log/time-series indices that roll over daily:
```json
// Rollover: create new index when current exceeds size or age
PUT /_ilm/policy/logs_policy
{
  "phases": {
    "hot":  {"actions": {"rollover": {"max_size": "50gb", "max_age": "1d"}}},
    "warm": {"min_age": "7d",  "actions": {"allocate": {"number_of_replicas": 0}}},
    "cold": {"min_age": "30d", "actions": {"freeze": {}}},
    "delete": {"min_age": "90d", "actions": {"delete": {}}}
  }
}
```

---

### Q13. What is vector search and semantic search? Explain embeddings and ANN algorithms.

**Answer:**

**The problem with keyword search:**
Keyword search finds exact term matches. It misses semantic similarity:
```
Query: "automobile maintenance tips"
Document: "car repair guide"
BM25 score: ~0 (no word overlap)
But semantically: highly relevant!
```

**Vector search / semantic search** represents queries and documents as dense vectors (embeddings) in high-dimensional space. Semantically similar items have similar vectors (small cosine distance).

**Embeddings:**
Neural models (BERT, Sentence Transformers, OpenAI text-embedding-3) encode text into dense vectors.
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

doc_embedding  = model.encode("car repair guide")           # 384-dimensional vector
query_embedding = model.encode("automobile maintenance tips")  # 384-dimensional vector

cosine_similarity = dot(doc_embedding, query_embedding)  # ≈ 0.85 (very similar!)
```

**The ANN (Approximate Nearest Neighbor) problem:**
Finding the most similar vector to a query vector by brute-force is O(N × D) where N = corpus size, D = dimensions. For 1 billion 768-dimensional vectors: too slow.

**ANN algorithms:**

**HNSW (Hierarchical Navigable Small World):**
Builds a multi-layer graph where nodes have long-range connections at higher layers and local connections at lower layers. Search starts at the top layer (coarse) and refines at lower layers.
```
Layer 2:  A ────── E ────── I        (few long-range connections)
Layer 1:  A ─ B ─ C ─ D ─ E ─ F     (medium range)
Layer 0:  A─B─C─D─E─F─G─H─I─J      (all connections, dense)

Query: start at random top-layer node, greedily navigate to nearest neighbor
       descend to lower layers for refinement
```
Pros: Very high recall (> 99%), fast query, supports dynamic updates.
Cons: High memory usage (graph structure stored in RAM).

**IVF (Inverted File Index):**
Clusters vectors using k-means. At query time, search only the closest K clusters.
```
Build: K-means(vectors, n_clusters=1000) → 1000 cluster centroids
Index: vectors assigned to their nearest centroid

Query: 
  1. Find nearest M centroids to query (default M=1, can increase for better recall)
  2. Search only vectors in those M clusters
  3. Return top-k from those clusters
```
Pros: Memory efficient (centroids + cluster assignments).
Cons: Approximate (vectors near cluster boundaries may be missed), less recall than HNSW.

**Elasticsearch KNN:**
```json
POST /products/_search
{
  "knn": {
    "field": "description_embedding",
    "query_vector": [0.12, -0.34, 0.56, ...],  // 768-dim
    "k": 10,
    "num_candidates": 100
  }
}
```

---

### Q14. What is hybrid search and how does RRF (Reciprocal Rank Fusion) work?

**Answer:**

**Hybrid search** combines keyword search (BM25) and vector search (ANN) to get the best of both: exact term matching for specific queries and semantic similarity for conceptual queries.

**Why hybrid beats either alone:**
```
Query: "python tutorial for data science"

BM25 results: 
  1. "Python Tutorial: Complete Guide" (exact match on python, tutorial)
  2. "Python for Everybody" (term match)
  
Vector results:
  1. "Data Science with Pandas - beginner course" (semantic: data science, beginner)
  2. "Machine Learning with scikit-learn" (semantic: data science related)

Hybrid (best of both):
  1. "Python Tutorial: Complete Guide"           ← ranked by both
  2. "Data Science with Pandas - beginner course" ← semantic relevance
  3. "Python for Data Analysis - step by step"    ← combined signals
```

**RRF (Reciprocal Rank Fusion):**
A simple, effective method for combining rankings from multiple retrieval systems without requiring score normalization.

```
RRF_score(doc, rankings) = Σ  1 / (k + rank_i(doc))
                          i

Where:
  rank_i(doc) = position of document in ranking i
  k = constant (default 60) to dampen influence of very high ranks

Example:
  Doc A: BM25 rank = 1, Vector rank = 5
    RRF = 1/(60+1) + 1/(60+5) = 0.01639 + 0.01538 = 0.03177

  Doc B: BM25 rank = 3, Vector rank = 1  
    RRF = 1/(60+3) + 1/(60+1) = 0.01587 + 0.01639 = 0.03226

  Doc B wins: Better at vector, decent at BM25
```

**Elasticsearch hybrid search:**
```json
POST /products/_search
{
  "query": {
    "bool": {
      "should": [
        {
          "match": {"description": {"query": "python data science", "boost": 1.0}}
        }
      ]
    }
  },
  "knn": {
    "field": "embedding",
    "query_vector": [...],
    "k": 50,
    "num_candidates": 100
  },
  "rank": {
    "rrf": {"window_size": 100, "rank_constant": 60}
  }
}
```

**Alternative: Linear combination:**
```
final_score = α × bm25_score + (1-α) × cosine_similarity
α = tuned via offline evaluation on labeled query-document pairs
```

RRF is preferred because it does not require score normalization (BM25 and cosine similarity are on different scales), is simple, and works well in practice without tuning.

---

### Q15. How do you measure search quality? Explain precision, recall, NDCG, and MRR.

**Answer:**

Measuring search quality requires understanding both what was found and how well it was ranked.

**Precision and Recall:**
```
Precision = Relevant results returned / Total results returned
Recall    = Relevant results returned / Total relevant results in corpus

Example: 100 documents total, 20 are relevant for query.
  Search returns 15 results: 10 relevant, 5 not relevant.
  
  Precision = 10/15 = 0.667 (67% of returned results are relevant)
  Recall    = 10/20 = 0.50  (found 50% of all relevant docs)
```

Precision and recall trade off — returning all documents gives recall=1.0 but precision=0.2.

**Precision@K:** Precision computed at only the top K results.
```
P@1: Is the first result relevant?
P@5: Are at least 4 of the top 5 relevant?
P@10: Are at least 7 of the top 10 relevant?
```

**NDCG (Normalized Discounted Cumulative Gain):**
Accounts for graded relevance (not just binary relevant/not-relevant) and position (results ranked higher should be more relevant).

```
Relevance grades:
  3 = Highly relevant (perfect answer)
  2 = Relevant
  1 = Marginally relevant
  0 = Not relevant

DCG@5 = rel_1/log2(2) + rel_2/log2(3) + ... + rel_5/log2(6)

Ideal DCG (IDCG): DCG if results were perfectly ranked (3,3,2,2,1)

NDCG = DCG / IDCG  (normalized to 0-1 range)

Example results: [3, 1, 2, 0, 3]
DCG  = 3/1 + 1/1.585 + 2/2 + 0/2.322 + 3/2.585 = 3 + 0.63 + 1 + 0 + 1.16 = 5.79
IDCG = [3,3,2,1,0] → DCG = 3/1 + 3/1.585 + 2/2 + 1/2.322 + 0 = 3 + 1.89 + 1 + 0.43 = 6.32
NDCG = 5.79 / 6.32 = 0.916
```

**MRR (Mean Reciprocal Rank):**
Measures where the first relevant result appears. Used for navigational queries where there is one correct answer.

```
Query: "Python official docs"
  Run 1: relevant at position 1 → RR = 1/1 = 1.0
  Run 2: relevant at position 3 → RR = 1/3 = 0.33
  Run 3: relevant at position 2 → RR = 1/2 = 0.5

MRR = (1.0 + 0.33 + 0.5) / 3 = 0.611
```

**Which metric to use:**

| Scenario                             | Metric |
|--------------------------------------|--------|
| All results matter (research)        | NDCG   |
| First result matters (navigational)  | MRR    |
| Binary relevant/not relevant         | P@K    |
| Need to find all relevant docs       | Recall |
| Graded relevance + ranking quality   | NDCG   |

**Collecting evaluation data:**
1. Human labelers: slow, expensive, gold standard.
2. Click data: CTR as weak proxy for relevance (biased by position).
3. Dwell time: long engagement = relevant result.
4. A/B testing: compare NDCG across versions using real user behavior.

---

## Hard Questions (Q16–Q20)

---

### Q16. How does the PageRank algorithm work? Explain the random surfer model.

**Answer:**

PageRank was the core algorithm behind Google's original search engine (1998, Larry Page, Sergey Brin). It measures the authority of a web page based on the number and quality of links pointing to it.

**Random Surfer Model:**
Imagine a web surfer who randomly follows links. At each page, they either:
- Follow one of the outgoing links (with probability d, the damping factor, typically 0.85)
- Jump to a completely random page (with probability 1-d = 0.15)

**The PageRank of a page** = the probability that the random surfer is on that page at any given time (stationary distribution of the Markov chain).

```
PageRank(A) = (1 - d) / N  +  d × Σ [ PageRank(B) / OutDegree(B) ]
                                   B→A (pages linking to A)

Where:
  d = damping factor (0.85)
  N = total number of pages
  OutDegree(B) = number of links B has outgoing
```

**Example:**
```
Pages: A, B, C
Links: B→A, C→A, A→B, A→C  (A links to B and C; B and C link to A)

Iteration 0: PR(A)=PR(B)=PR(C) = 1/3

Iteration 1:
  PR(A) = 0.15/3 + 0.85 × [PR(B)/1 + PR(C)/1]  = 0.05 + 0.85 × (1/3 + 1/3) = 0.05 + 0.566 = 0.617
  PR(B) = 0.15/3 + 0.85 × [PR(A)/2]             = 0.05 + 0.85 × (1/6) = 0.05 + 0.141 = 0.191
  PR(C) = 0.15/3 + 0.85 × [PR(A)/2]             = 0.191

Iteration 2: (converges toward stable values)
  A has high PageRank because both B and C link to it.
  B and C have lower PageRank: only A links to them, and A splits its rank between both.
```

**Key insights:**
1. A link from a high-PageRank page is worth more than a link from a low-PageRank page.
2. A page with many outgoing links "dilutes" its PageRank across all of them.
3. The damping factor prevents rank sink (pages with no outgoing links hoarding all rank).

**Convergence:** PageRank is computed iteratively until stable (usually 50–100 iterations for the web).

**Modern usage:** Pure PageRank is rarely used alone in 2024. Google uses hundreds of signals. But the concept lives on in internal link graph analysis, citation networks (academic search), and social graph importance scoring.

---

### Q17. How do you design search personalization using user behavior signals?

**Answer:**

Personalization re-ranks search results based on individual user behavior, dramatically improving relevance for repeat users.

**Behavior signals to collect:**
```
Click signal:    User clicked result X for query Q → X is relevant for Q for this user
Dwell signal:    User spent 5min on result X → strong relevance signal
Bounce signal:   User returned to results immediately → X was not relevant
Purchase signal: User bought item X after searching Q → strongest signal
Skip signal:     User scrolled past X to click Y → X ranked too high
Save/bookmark:   User saved X → strong interest signal
```

**Architecture:**
```
User search behavior
        │
        ▼
[Event Collector (Kafka)]
        │
        ├── [Real-time processor (Flink)]
        │     └── Update user interest vectors (immediate, session-level)
        │
        └── [Batch processor (Spark)]
              └── Train/update personalization models (daily)
              └── Compute user-item affinity scores
              
At query time:
[Query] → [Base Ranking (BM25+Vectors)] → [Personalization Layer] → [Final Results]
```

**User interest modeling:**
```python
# Collaborative filtering: users with similar behavior have similar interests
# Content-based: user liked "Python books" → surface more programming books
# Session-based: within current session, user interested in "data science" → boost

# User embedding: represent user as vector based on interaction history
user_embedding = weighted_avg([item_embedding(i) * interaction_weight(i) 
                               for i in user_recent_interactions])

# At query time: compute cosine similarity between user_embedding and doc_embeddings
# Boost documents similar to user's interest vector
personalized_score = λ × base_score + (1-λ) × cosine_sim(user_vec, doc_vec)
```

**Query expansion with user history:**
```
User previously clicked: [Python data science, pandas tutorial, numpy docs]
Current query: "dataframe"
→ Expand with user context: "dataframe" + user's interest in "pandas"
→ Boost pandas documentation results
```

**Cold start problem:**
New users have no history. Solutions:
1. Use demographic or geographic defaults.
2. Ask for explicit preferences during onboarding.
3. Use popularity-based ranking until enough signals collected.
4. Collaborative filtering: find similar users based on initial signals (e.g., first search query).

**A/B testing personalization:**
```
Control:     Standard BM25 ranking
Treatment A: BM25 + user click history reranking
Treatment B: BM25 + user embedding similarity boost
Treatment C: Full LTR model with all personalization signals

Metrics: CTR@1, CTR@5, revenue per search, session depth
```

---

### Q18. How do you handle multi-language search?

**Answer:**

Multi-language search requires language-aware text processing at both index time and query time. The same word in different languages requires different analysis chains.

**Language-specific analysis pipeline:**
```
Raw text → Language Detection → Language-specific Analyzer → Indexed tokens

English "running":     stem → "run"     (Porter stemmer)
French  "courant":     stem → "cour"    (French stemmer)
German  "laufen":      stem → "lauf"    (German stemmer)
Arabic  "يجري":        stem → "جر"     (Arabic light stemmer)
Chinese "跑步":         segment → "跑","步" (ICU tokenizer, no spaces)
Japanese "走ること":     segment via Kuromoji
```

**Elasticsearch multi-language index design:**

**Option A: One index per language**
```
products_en: analyzed with english analyzer
products_fr: analyzed with french analyzer
products_de: analyzed with german analyzer

Query: search products_en + products_fr + products_de simultaneously
Pros:  Clean, simple per-language tuning
Cons:  Multiple indices to manage, cross-language queries harder
```

**Option B: Per-language fields in one index**
```json
PUT /products
{
  "mappings": {
    "properties": {
      "title_en": {"type": "text", "analyzer": "english"},
      "title_fr": {"type": "text", "analyzer": "french"},
      "title_de": {"type": "text", "analyzer": "german"},
      "title":    {"type": "text", "analyzer": "icu_analyzer"}  // Unicode-aware fallback
    }
  }
}

// At index time: populate all applicable language fields
// At query time: query only the field matching user's language
```

**Language detection:**
```python
from langdetect import detect

def detect_language(text: str) -> str:
    try:
        return detect(text)  # Returns 'en', 'fr', 'de', etc.
    except:
        return 'en'  # Default to English

def index_product(product):
    doc_lang = detect_language(product["description"])
    
    return {
        "id": product["id"],
        f"title_{doc_lang}": product["title"],
        f"description_{doc_lang}": product["description"],
        "detected_language": doc_lang
    }
```

**Cross-language search (multilingual models):**
For search where users might query in any language and find results in others, use multilingual embeddings (mBERT, multilingual-e5):

```python
# Multilingual embedding model
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

# Embed query in English: "blue running shoes"
# Embed document in French: "chaussures de course bleues"
# Cosine similarity ≈ 0.89 (semantically equivalent across languages)
```

**Stop words and stemming per language:**
```json
PUT /products
{
  "settings": {
    "analysis": {
      "analyzer": {
        "fr_custom": {
          "type": "custom",
          "tokenizer": "standard",
          "filter": ["lowercase", "french_stop", "french_stemmer", "asciifolding"]
        }
      },
      "filter": {
        "french_stop": {"type": "stop", "stopwords": "_french_"},
        "french_stemmer": {"type": "stemmer", "language": "light_french"}
      }
    }
  }
}
```

**ASCII folding:** Maps accented characters to their ASCII equivalents ("café" → "cafe") so queries with or without accents match.

---

### Q19. How do you design search for e-commerce with inventory-aware ranking?

**Answer:**

E-commerce search has unique requirements beyond pure text relevance: out-of-stock items should not rank high, price and promotion affect relevance, and real-time inventory status must be reflected.

**Unique challenges:**
```
1. Inventory changes in real-time (item can go from 1000 units to 0 in seconds during flash sale)
2. Promotions and discounts change hourly
3. Multiple variants (size, color) must be aggregated per product
4. Out-of-stock items should be demoted but not removed (user intent matters)
5. Personalization: user's size preference, budget history
6. Ranking = text relevance × business objectives (margin, promotion)
```

**Architecture:**
```
Product Service → CDC (Debezium) → Kafka → ES Indexing Consumer
Inventory Service                → Kafka → Inventory Cache (Redis) + ES update
Pricing Service                  → Kafka → ES update (price, discount_pct)

Search Query → [ES: text + vector scoring] 
             → [Inventory lookup: Redis (< 1ms)] 
             → [Re-rank: availability × price × relevance × personalization]
             → Results
```

**Index design:**
```json
{
  "id": "prod_123",
  "name": "Nike Air Max 270",
  "description": "Comfortable running shoe with Max Air cushioning",
  "brand": "Nike",
  "category": "footwear",
  "price": 150.00,
  "sale_price": 120.00,       // null if no sale
  "discount_pct": 20,
  "in_stock": true,
  "inventory_count": 45,
  "availability_score": 1.0,  // 0.0 = OOS, 0.5 = low stock, 1.0 = in stock
  "sizes_available": ["7", "8", "9", "10", "11"],
  "popularity_score": 0.85,   // Based on views, purchases, CTR
  "quality_score": 0.92,      // Based on reviews, returns rate
  "margin_score": 0.70        // Proprietary business metric
}
```

**Inventory-aware ranking query:**
```json
{
  "query": {
    "function_score": {
      "query": {
        "bool": {
          "must": [{"multi_match": {"query": "running shoes", "fields": ["name^2","description"]}}],
          "filter": [
            {"term": {"category": "footwear"}},
            {"terms": {"sizes_available": ["9"]}}  // Filter to user's size
          ]
        }
      },
      "functions": [
        {
          "field_value_factor": {
            "field": "availability_score",
            "factor": 2.0,              // Heavily penalize OOS
            "missing": 0               // Missing = OOS
          }
        },
        {
          "field_value_factor": {
            "field": "popularity_score",
            "factor": 0.5
          }
        },
        {
          "filter": {"range": {"discount_pct": {"gte": 20}}},
          "weight": 1.5                // Boost sale items
        }
      ],
      "score_mode": "multiply",
      "boost_mode": "multiply"
    }
  }
}
```

**Real-time inventory updates:**
Inventory changes too frequently (especially during flash sales) to re-index every change. Use a hybrid approach:
1. Elasticsearch stores base availability (in_stock: true/false) — updated via CDC, acceptable eventual consistency.
2. Real-time inventory count stored in Redis — checked at result rendering time to show exact stock.
3. For "filter by in-stock only" queries: accept slight staleness (5–30 seconds).

---

### Q20. What are the differences between Elasticsearch, Solr, and OpenSearch?

**Answer:**

All three are built on Apache Lucene and share core capabilities. The differences are governance, ecosystem, and feature direction.

**Apache Solr:**
The original open-source Lucene-based search platform. Developed at CNET, donated to Apache in 2006.
```
Architecture: Solr instances + SolrCloud (distributed mode using ZooKeeper)
Query language: SolrQL (XML/JSON config-heavy)
Schema: schema.xml (traditional), schemaless mode available
Primary use: Enterprise document search, traditional search applications
```

**Elasticsearch:**
Created by Shay Banon (Elastic, 2010). REST-first design, JSON-native, much easier to get started with.
```
Architecture: Native distributed (no ZooKeeper), Zen Discovery / ES-KNN
Query language: Query DSL (JSON), SQL via ES SQL plugin
Schema: Dynamic mapping (schema-free by default)
Primary use: Application search, log analytics (ELK), APM, SIEM
License change: Changed from Apache 2.0 to SSPL + Elastic License in 2021
```

**OpenSearch:**
AWS fork of Elasticsearch 7.10 when Elastic changed the license (2021). Fully Apache 2.0 licensed.
```
Architecture: Fork of ES, same core capabilities
Primary use: AWS customers, organizations requiring true open-source license
Query language: Same as ES (binary compatible with ES 7.x clients)
Active features: k-NN search, ML inference, anomaly detection built-in
```

**Feature comparison (2024):**

| Feature                  | Elasticsearch       | OpenSearch          | Solr               |
|--------------------------|---------------------|---------------------|--------------------|
| Vector/KNN search        | Yes (HNSW native)   | Yes (HNSW native)   | Yes (DenseVector)  |
| ML inference             | Elser, ELSER        | ML Commons          | Limited            |
| License                  | Elastic License 2.0 | Apache 2.0          | Apache 2.0         |
| Cloud offering           | Elastic Cloud       | AWS OpenSearch Svc  | SolrCloud          |
| Learning to Rank         | Yes (plugin)        | Yes (built-in)      | Yes (plugin)       |
| Real-time indexing       | Yes (1s default)    | Yes                 | Near real-time     |
| Community size           | Very large          | Growing             | Established        |
| REST API                 | Excellent           | Excellent           | Good               |
| Configuration complexity | Low                 | Low                 | High               |

**Choosing between them:**
```
Use Elasticsearch if:
  → On-prem or non-AWS cloud, enterprise features (Kibana, APM, SIEM)
  → Team familiar with ELK stack
  → Need Elastic's ML features (ELSER, vector models)

Use OpenSearch if:
  → On AWS (native integration with Kinesis, S3, CloudWatch)
  → Need strict Apache 2.0 license for compliance
  → Cost optimization on AWS

Use Solr if:
  → Legacy systems already on Solr
  → Complex NLP/facet requirements with SolrCloud maturity
  → Team has Solr expertise
```

---

## Quick Reference

```
INVERTED INDEX
  Term → Posting List [(doc_id, position, frequency)]
  Boolean ops: AND = intersect, OR = union, NOT = difference

TF-IDF
  TF(t,d) = count(t in d) / |d|
  IDF(t)  = log(N / df(t))
  TF-IDF  = TF × IDF  (high for rare terms in focused docs)

BM25 IMPROVEMENTS OVER TF-IDF
  TF saturation: term frequency capped (k1 parameter)
  Doc length normalization: shorter focused docs rewarded (b parameter)

ELASTICSEARCH ARCHITECTURE
  Cluster → Nodes → Index → Shards (Primary + Replica)
  routing = hash(_id) % num_primary_shards
  P and R always on different nodes
  Refresh: 1s default (document visible after refresh)

FUZZY SEARCH
  Edit distance (Levenshtein): min insertions/deletions/substitutions
  Elasticsearch AUTO fuzziness: 1 edit for short, 2 for longer words
  Trigrams: share 3-char sequences → likely similar strings

WINDOWS IN WINDOWED AGGREGATIONS
  Tumbling: non-overlapping fixed windows
  Sliding:  overlapping fixed windows
  Session:  gap-based dynamic windows

VECTOR SEARCH
  Embeddings: text → dense vector (384–1536 dims)
  Cosine similarity: semantic relevance
  ANN algorithms:
    HNSW: graph-based, fast, high recall, memory-intensive
    IVF:  cluster-based, memory-efficient, less recall

HYBRID SEARCH (BM25 + Vector)
  RRF: score = Σ 1/(60 + rank_i)
  No score normalization needed
  Better than either alone for mixed query types

SEARCH METRICS
  Precision@K: fraction of top-K that are relevant
  Recall:      fraction of all relevant docs found
  NDCG:        graded relevance + position discount
  MRR:         1/rank of first relevant result

PAGERANK
  Random surfer model
  PR(A) = (1-d)/N + d × Σ PR(B)/OutDegree(B)  for all B→A
  d = 0.85 (damping factor)

ELASTICSEARCH ALIAS SWAP (ZERO DOWNTIME REBUILD)
  1. Create products_v2 with new mapping
  2. Reindex from products_v1 to products_v2
  3. Atomic: add v2 to alias, remove v1
  4. Delete v1

FACETED SEARCH
  terms agg    → category counts
  range agg    → price distribution
  post_filter  → filter results but show all facet counts

MULTI-LANGUAGE
  One analyzer per language
  Language detection at index time
  Multilingual embeddings for cross-language semantic search

SPELL CORRECTION (NOISY CHANNEL)
  P(Q | Q') ∝ P(Q' | Q) × P(Q)
  Error model × Language model

E-COMMERCE RANKING
  Score = text_relevance × availability_score × popularity × business_objectives
  Real-time inventory: Redis (fast) + ES (eventual consistency)
```

---

*File 15 of 15 — Search System Design*
