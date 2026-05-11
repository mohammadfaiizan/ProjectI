# 17. Search System Design

## Table of Contents
1. [Search System Architecture Overview](#1-search-system-architecture-overview)
2. [Inverted Index](#2-inverted-index)
3. [TF-IDF Ranking](#3-tf-idf-ranking)
4. [BM25 Ranking](#4-bm25-ranking)
5. [PageRank Algorithm](#5-pagerank-algorithm)
6. [Elasticsearch Architecture](#6-elasticsearch-architecture)
7. [Elasticsearch Query DSL](#7-elasticsearch-query-dsl)
8. [Full-Text Search Features](#8-full-text-search-features)
9. [Typeahead and Autocomplete](#9-typeahead-and-autocomplete)
10. [Fuzzy Search](#10-fuzzy-search)
11. [Faceted Search](#11-faceted-search)
12. [Spell Correction](#12-spell-correction)
13. [Personalized Search](#13-personalized-search)
14. [Vector Search and Semantic Search](#14-vector-search-and-semantic-search)
15. [Search Result Caching](#15-search-result-caching)
16. [Scaling Search](#16-scaling-search)
17. [Search Relevance Tuning](#17-search-relevance-tuning)
18. [Real-Time Indexing Pipeline](#18-real-time-indexing-pipeline)
19. [Search for E-Commerce](#19-search-for-e-commerce)
20. [Quick Reference](#20-quick-reference)

---

## 1. Search System Architecture Overview

### Core Components

A production-grade search system consists of five major components working in a pipeline:

```
[Web / Data Sources]
        |
        v
  [Web Crawler]          -- Discovers and fetches documents
        |
        v
  [Document Store]       -- Raw content storage (S3, HDFS)
        |
        v
   [Indexer]             -- Processes and builds inverted index
        |
        v
  [Index Store]          -- Distributed index (Elasticsearch, Solr)
        |
        v
[Query Processor]        -- Parses query, applies analysis pipeline
        |
        v
   [Ranker]              -- Scores documents (TF-IDF, BM25, ML ranker)
        |
        v
[Serving Layer]          -- Returns results, applies business rules
        |
        v
  [Search UI / API]
```

### Web Crawler

The crawler is responsible for discovering and fetching content:

```
Crawler Architecture:
  - URL Frontier: Priority queue of URLs to crawl (politeness policy)
  - DNS Resolver: Cached DNS lookups (high volume)
  - Fetcher: HTTP/HTTPS downloader (respects robots.txt)
  - HTML Parser: Extracts links, text content, metadata
  - Content Deduplicator: SimHash to detect near-duplicate content
  - URL Normalizer: Canonical form (lowercase, remove fragments)
```

**Crawler Design Decisions:**
- Politeness: respect `Crawl-Delay` in robots.txt; 1 req/sec per domain default
- Priority: prioritize high-PageRank pages, recently updated pages
- Freshness: recrawl based on change frequency (sitemap.xml `changefreq`)
- Scale: Google crawls ~50 billion pages; needs distributed crawling

### Indexer Pipeline

```
Raw Document
    |
    v
[Content Extractor]    -- Strip HTML tags, extract metadata (title, URL, date)
    |
    v
[Language Detector]    -- Detect language for language-specific analysis
    |
    v
[Text Analyzer]        -- Tokenize, lowercase, remove stop words, stem/lemmatize
    |
    v
[Field Extractor]      -- title, body, anchor text, meta description
    |
    v
[Index Writer]         -- Append to inverted index posting lists
    |
    v
[Segment Merger]       -- Lucene-style segment merging for efficiency
```

### Query Processor

```python
class QueryProcessor:
    def process(self, raw_query: str) -> Query:
        # 1. Query parsing
        tokens = self.tokenize(raw_query)
        
        # 2. Query expansion
        expanded = self.expand_synonyms(tokens)
        
        # 3. Query classification
        intent = self.classify_intent(raw_query)  # navigational, informational, transactional
        
        # 4. Query rewriting
        rewritten = self.rewrite(expanded, intent)
        
        # 5. Build execution plan
        return self.build_query(rewritten)
```

### Serving Layer

Responsibilities:
- Query routing to appropriate shards
- Result merging and deduplication
- Business rule application (boost sponsored results, filter adult content)
- A/B test experiment assignment
- Logging for relevance feedback

---

## 2. Inverted Index

### Structure

An inverted index maps terms to the list of documents containing them:

```
Term          | Posting List (docID, frequency, positions)
-----------   | --------------------------------------------------
"apple"       | [(doc1, 3, [2,15,47]), (doc5, 1, [8]), (doc9, 2, [1,33])]
"banana"      | [(doc2, 1, [5]), (doc5, 2, [3,21])]
"cherry"      | [(doc1, 1, [9]), (doc3, 4, [1,2,3,4])]
```

**Posting List Entry Components:**
- `docID`: unique document identifier
- `term frequency (tf)`: how many times term appears in document
- `positions`: byte offsets for phrase queries and proximity scoring

### Index Storage Layers

```
Dictionary (In-Memory Hash or B-Tree):
  term → pointer to posting list on disk

Posting Lists (On Disk, Compressed):
  [docID delta-encoded] [tf] [positions delta-encoded]

Example for term "search":
  raw:       [5, 10, 15, 20, 25]
  deltas:    [5,  5,  5,  5,  5]  (much smaller, better compression)
```

### Compression Techniques

**Variable-Byte (VByte) Encoding:**
```
Number 214577:
  Binary: 110 1000101 1110001
  VByte:  00001101 | 00000101 | 01110001
  (continuation bit = 1 means more bytes follow)
```

**Gamma Encoding (Elias Gamma):**
- For small integers common in delta-encoded posting lists
- Encode n as: ⌊log₂(n)⌋ zeros followed by binary representation of n

**PForDelta (Patched Frame of Reference):**
- Used in Lucene/Elasticsearch for posting list blocks of 128 integers
- Determine frame size b = bits needed for 90th percentile values
- Pack most values in b bits, store outliers ("patches") separately

### Skip Pointers

Skip pointers accelerate AND queries on posting lists:

```
Posting list for "apple":  [2, 5, 8, 11, 14, 17, 20, 23, 26, ...]
Skip pointers (every k=3): 
  pos 0  → skip to pos 3  (value 11)
  pos 3  → skip to pos 6  (value 20)
  pos 6  → skip to pos 9  (value 29)

AND("apple", "banana"):
  apple:  2  → skip to 11 (since banana[0]=10, skip over 2,5,8)
  banana: 10 → advance to 11
  Merge:  apple[1]=11, banana[1]=11 → MATCH
```

**Skip pointer interval:** sqrt(n) is theoretically optimal for a posting list of length n.

### Index Segmentation (Lucene)

```
Lucene Index Structure:
  Segment 1 (immutable): docs 1-1000
  Segment 2 (immutable): docs 1001-2000
  Segment 3 (in-memory buffer, mutable): docs 2001-2050

Merge Policy:
  - TieredMergePolicy: merge segments of similar size
  - LogByteSizeMergePolicy: merge by byte size tiers
  
Benefits:
  - Write to memory buffer, flush periodically (near-real-time)
  - Immutable segments = no locking during reads
  - Merge in background without blocking searches
```

---

## 3. TF-IDF Ranking

### Intuition

TF-IDF (Term Frequency - Inverse Document Frequency) scores how relevant a term is to a document within a corpus:
- **TF**: terms appearing frequently in a document are more relevant
- **IDF**: terms appearing in fewer documents are more discriminative (rare = important)

### Formula

```
TF(t, d)  = count(t in d) / total_terms(d)
            OR log(1 + count(t in d))   [log-normalized to dampen effect]

IDF(t, D) = log( N / df(t) )
            OR log( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )  [smoothed]

  where:
    N     = total number of documents
    df(t) = number of documents containing term t

TF-IDF(t, d, D) = TF(t, d) × IDF(t, D)

Score for query Q in document d:
  score(Q, d) = Σ TF-IDF(t, d, D)  for each t in Q
```

### Example Calculation

```
Corpus: 1,000,000 documents
Query: "machine learning"

Document A: "machine learning is important in machine translation" (7 words)
  TF("machine", A) = 2/7 = 0.286
  TF("learning", A) = 1/7 = 0.143

  "machine"  appears in 100,000 docs → IDF = log(1M/100K) = log(10) = 2.303
  "learning" appears in 50,000 docs  → IDF = log(1M/50K)  = log(20) = 2.996

  TF-IDF("machine", A)  = 0.286 × 2.303 = 0.659
  TF-IDF("learning", A) = 0.143 × 2.996 = 0.428
  Score(A) = 0.659 + 0.428 = 1.087
```

### Limitations of TF-IDF

| Issue | Problem | BM25 Solution |
|-------|---------|---------------|
| TF saturation | Long docs get unfairly high TF | BM25 uses saturation function |
| Length normalization | Long docs score higher by default | BM25 has explicit length normalization |
| No term saturation | More occurrences always = higher score | BM25 k1 parameter caps TF contribution |

---

## 4. BM25 Ranking

### Formula

BM25 (Best Match 25) is the de facto standard for text retrieval:

```
BM25(Q, d) = Σ IDF(qi) × [ tf(qi,d) × (k1 + 1) ]
                           [ tf(qi,d) + k1 × (1 - b + b × |d|/avgdl) ]

Where:
  qi      = each query term
  tf(qi,d) = term frequency in document d
  |d|     = length of document d (in words)
  avgdl   = average document length in corpus
  k1      = term frequency saturation parameter (typical: 1.2 to 2.0)
  b       = length normalization parameter (typical: 0.75)
  
IDF(qi) = log( (N - n(qi) + 0.5) / (n(qi) + 0.5) + 1 )
  n(qi) = number of documents containing qi
```

### Parameter Tuning

```
k1 parameter (controls TF saturation):
  k1 = 0:  binary model (term present/absent, ignores frequency)
  k1 = 1.2: moderate saturation (Elasticsearch default)
  k1 = 2.0: higher TF weight, less saturation
  k1 → ∞:  equivalent to raw TF (no saturation)

b parameter (controls length normalization):
  b = 0:  no length normalization
  b = 0.75: default, balances length effect
  b = 1.0: full length normalization

Practical tuning:
  - Short queries (1-2 words): lower b to reduce length penalty
  - Long documents corpus: higher b to normalize
  - Keyword matching: higher k1
```

### BM25 vs TF-IDF Comparison

```
For document with tf=10 vs tf=1 (same length):
  TF-IDF ratio: 10/1 = 10x score
  BM25 ratio (k1=1.2): 
    high: 10×2.2 / (10 + 1.2) = 1.964
    low:  1×2.2  / (1 + 1.2)  = 1.0
    Ratio: 1.964x  ← much less sensitive to high TF
```

---

## 5. PageRank Algorithm

### Intuition: Random Surfer Model

PageRank simulates a user randomly clicking links:
- Start at random page
- With probability d (damping factor, ~0.85): follow random outbound link
- With probability 1-d: jump to completely random page in corpus
- PageRank of a page = probability of being on that page after infinite random walk

### Formula

```
PR(A) = (1 - d) / N + d × Σ [ PR(T) / C(T) ]
                           T → A

Where:
  PR(A)  = PageRank of page A
  d      = damping factor (typically 0.85)
  N      = total number of pages
  T      = pages that link to A
  C(T)   = number of outbound links from page T
  
Matrix form: PR = (1-d)/N × e + d × M × PR
  M[i][j] = 1/outdegree(j) if j links to i, else 0
  e = vector of ones
```

### Iterative Computation

```python
def pagerank(graph, damping=0.85, iterations=100, tolerance=1e-6):
    N = len(graph)
    pr = {node: 1.0/N for node in graph}
    
    for iteration in range(iterations):
        new_pr = {}
        for node in graph:
            rank = (1 - damping) / N
            for neighbor in graph.get_inbound(node):
                rank += damping * pr[neighbor] / graph.outdegree(neighbor)
            new_pr[node] = rank
        
        # Check convergence
        delta = sum(abs(new_pr[n] - pr[n]) for n in graph)
        pr = new_pr
        if delta < tolerance:
            break
    
    return pr
```

### Practical Considerations

- **Dangling nodes** (no outbound links): distribute their rank evenly to all pages
- **Spider traps**: subgraphs with no exit; damping factor prevents rank accumulation
- **TrustRank**: seed PageRank from trusted sites to fight spam
- **Topic-sensitive PageRank**: multiple random surfers, each preferring a topic

---

## 6. Elasticsearch Architecture

### Core Concepts

```
Cluster
  └── Nodes (physical/virtual machines)
        ├── Master Node: cluster state management, index/shard assignment
        ├── Data Node: stores shards, executes queries
        ├── Coordinating Node: routes requests, merges results
        ├── Ingest Node: pre-processing pipeline (transforms before indexing)
        └── ML Node: machine learning tasks

Index
  └── Shards (Lucene instances)
        ├── Primary Shard: accepts writes
        └── Replica Shard: serves reads, promoted on primary failure
```

### Shard Design

```
Index: "products" 
  Primary shards: 5
  Replica shards: 1 (per primary)
  
  Total shards: 5 × (1+1) = 10 shards across cluster

Routing:
  shard = hash(document_id) % number_of_primary_shards
  
Rule of thumb:
  - 20-40GB per shard for most use cases
  - Number of shards ≤ 3× number of nodes
  - Can't change primary shard count after creation (must reindex)
  - Can change replica count dynamically
```

### Write Path

```
Client → Coordinating Node
  → Route to Primary Shard (based on doc ID hash)
    → Write to primary shard's Lucene index (in-memory buffer)
    → Replicate to replica shards (parallel)
    → Flush to transaction log (translog) for durability
  ← Acknowledge to client (configurable: wait_for, 1, all)

Near-Real-Time (NRT):
  - Lucene refresh (memory buffer → searchable): default 1 second
  - Lucene flush (RAM → disk segment): based on translog size
  - fsync (OS buffer → disk): every 30 seconds or 512MB translog
```

### Read Path

```
Client → Coordinating Node
  → Scatter: broadcast query to all relevant shards (primary or replica, round-robin)
    └── Each shard executes query locally
    └── Returns top-k local results (docIDs + scores)
  → Gather: merge results from all shards, re-sort by score
  → Fetch: retrieve full documents for final top-k results
  ← Return to client

Two-phase approach:
  Phase 1 (Query): get docID + score from each shard
  Phase 2 (Fetch): get full document source for final results
```

### Cluster Settings

```json
{
  "settings": {
    "number_of_shards": 5,
    "number_of_replicas": 1,
    "refresh_interval": "1s",
    "index.routing.allocation.require.zone": "us-east-1a"
  },
  "mappings": {
    "properties": {
      "title": {"type": "text", "analyzer": "english"},
      "price": {"type": "double"},
      "category": {"type": "keyword"},
      "created_at": {"type": "date"}
    }
  }
}
```

---

## 7. Elasticsearch Query DSL

### Match Query (Full-Text)

```json
{
  "query": {
    "match": {
      "title": {
        "query": "quick brown fox",
        "operator": "or",
        "minimum_should_match": "75%",
        "fuzziness": "AUTO"
      }
    }
  }
}
```

### Term Query (Exact Match)

```json
{
  "query": {
    "term": {
      "status": {
        "value": "published"
      }
    }
  }
}
```

### Bool Query (Compound)

```json
{
  "query": {
    "bool": {
      "must": [
        {"match": {"title": "elasticsearch"}}
      ],
      "should": [
        {"match": {"body": "search"}},
        {"term":  {"tags": "open-source"}}
      ],
      "must_not": [
        {"term": {"status": "deleted"}}
      ],
      "filter": [
        {"range": {"price": {"gte": 10, "lte": 100}}},
        {"term":  {"in_stock": true}}
      ],
      "minimum_should_match": 1,
      "boost": 1.5
    }
  }
}
```

**Note:** `filter` context does not compute relevance score and is cached; `must/should` compute scores and are not cached.

### Range, Nested, Geo Queries

```json
// Range query
{
  "query": {
    "range": {
      "publish_date": {
        "gte": "2024-01-01",
        "lte": "now",
        "format": "yyyy-MM-dd"
      }
    }
  }
}

// Nested query (for nested object arrays)
{
  "query": {
    "nested": {
      "path": "reviews",
      "query": {
        "bool": {
          "must": [
            {"match":  {"reviews.text":   "great product"}},
            {"range":  {"reviews.rating": {"gte": 4}}}
          ]
        }
      },
      "score_mode": "avg"
    }
  }
}

// Geo distance query
{
  "query": {
    "geo_distance": {
      "distance": "10km",
      "location": {"lat": 40.715, "lon": -73.988}
    }
  }
}
```

### Function Score Query (Custom Scoring)

```json
{
  "query": {
    "function_score": {
      "query": {"match": {"title": "phone"}},
      "functions": [
        {
          "filter": {"term": {"is_featured": true}},
          "weight": 2.0
        },
        {
          "field_value_factor": {
            "field": "popularity_score",
            "factor": 1.2,
            "modifier": "log1p",
            "missing": 1
          }
        },
        {
          "gauss": {
            "price": {
              "origin": "500",
              "scale":  "200",
              "decay":  "0.5"
            }
          }
        }
      ],
      "score_mode": "multiply",
      "boost_mode": "multiply"
    }
  }
}
```

---

## 8. Full-Text Search Features

### Analysis Pipeline

```
Input text: "The Quick Brown Foxes jumped over the LAZY Dogs!"
                    |
              [Character Filters]
                    |
         "The Quick Brown Foxes jumped over the LAZY Dogs"  (strip punctuation)
                    |
               [Tokenizer]
                    |
         [The, Quick, Brown, Foxes, jumped, over, the, LAZY, Dogs]
                    |
              [Token Filters]
                    |
         Lowercase:   [the, quick, brown, foxes, jumped, over, the, lazy, dogs]
         Stop words:  [quick, brown, foxes, jumped, lazy, dogs]  (remove 'the', 'over')
         Stemmer:     [quick, brown, fox, jump, lazi, dog]
```

### Tokenizers

| Tokenizer | Description | Use Case |
|-----------|-------------|----------|
| `standard` | Unicode-aware word boundaries, lowercase | General English text |
| `whitespace` | Split on whitespace only | Code, URLs |
| `keyword` | No tokenization, single token | Exact match fields |
| `ngram` | Generate n-grams from tokens | Partial word matching |
| `edge_ngram` | N-grams from token start | Autocomplete |
| `pattern` | Regex-based splitting | Structured text |
| `uax_url_email` | Preserves URLs and emails | Web content |

### Stemming vs Lemmatization

```
Stemming (algorithmic, fast):
  "running" → "run"
  "studies" → "studi"    ← may not be a real word
  "better"  → "better"   ← Porter stemmer doesn't handle this
  
  Algorithms: Porter, Snowball, Lancaster

Lemmatization (dictionary-based, accurate):
  "running" → "run"
  "studies" → "study"
  "better"  → "good"     ← understands morphology
  
  Tools: WordNet, spaCy, Stanford NLP
```

### Synonyms

```json
// Elasticsearch synonym token filter
{
  "settings": {
    "analysis": {
      "filter": {
        "synonym_filter": {
          "type": "synonym",
          "synonyms": [
            "laptop, notebook, computer",
            "tv => television",
            "nyc, new york city"
          ]
        }
      },
      "analyzer": {
        "synonym_analyzer": {
          "tokenizer": "standard",
          "filter": ["lowercase", "synonym_filter"]
        }
      }
    }
  }
}
```

**Expansion vs Replacement:**
- `a, b, c` → expand: query for any triggers match on all
- `a => b` → replace: query for `a` becomes query for `b` only

### Custom Analyzer Example

```json
{
  "analysis": {
    "char_filter": {
      "html_strip": {"type": "html_strip"}
    },
    "tokenizer": {
      "standard": {"type": "standard"}
    },
    "filter": {
      "english_stop":   {"type": "stop", "stopwords": "_english_"},
      "english_stem":   {"type": "stemmer", "language": "english"},
      "english_possessive": {"type": "stemmer", "language": "possessive_english"}
    },
    "analyzer": {
      "english_custom": {
        "tokenizer": "standard",
        "char_filter": ["html_strip"],
        "filter": ["lowercase", "english_possessive", "english_stop", "english_stem"]
      }
    }
  }
}
```

---

## 9. Typeahead and Autocomplete

### Trie-Based Autocomplete

```
Trie for prefix matching:
         root
        /    \
       a      b
      / \      \
     p   n      a
    /     \      \
   p       d      n
  /         \      \
 l           r      a
 |           |      |
 e           o      n
             |      |
             i      a
             |
             d

prefix "app" → ["apple", "application", "applet", ...]
```

**Implementation for Top-K suggestions:**

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.top_k = []  # sorted list of (score, word) for this prefix
        self.is_end = False

class AutocompleteTrie:
    def insert(self, word: str, score: int):
        node = self.root
        for char in word:
            node = node.children.setdefault(char, TrieNode())
            # Maintain sorted top-k at each prefix node
            self._update_top_k(node.top_k, word, score)
    
    def search(self, prefix: str) -> List[str]:
        node = self._find_node(prefix)
        if not node:
            return []
        return [word for _, word in node.top_k[:10]]
```

**Scaling Trie:**
- Store trie in Redis (ZSET per prefix) or Cassandra
- Shard by first N characters of prefix
- Cache hot prefixes (top 1000) in L1 cache

### Elasticsearch Completion Suggester

```json
// Mapping
{
  "mappings": {
    "properties": {
      "suggest": {
        "type": "completion",
        "analyzer": "simple",
        "search_analyzer": "simple"
      }
    }
  }
}

// Indexing with weights
{
  "suggest": {
    "input": ["Nevermind", "Nirvana"],
    "weight": 34
  }
}

// Query
{
  "suggest": {
    "song-suggest": {
      "prefix": "nir",
      "completion": {
        "field": "suggest",
        "size": 5,
        "fuzzy": {
          "fuzziness": 1
        }
      }
    }
  }
}
```

**Completion suggester uses FST (Finite State Transducer) internally — extremely fast but not flexible.**

### Redis Sorted Set Autocomplete

```python
# Index: store each prefix → sorted set of completions
def index_word(redis, word: str, score: float):
    # Store all prefixes
    for i in range(1, len(word) + 1):
        prefix = word[:i]
        redis.zadd(f"autocomplete:{prefix}", {word: score})

# Query: O(log N + k) where k = number of results
def autocomplete(redis, prefix: str, limit: int = 10) -> List[str]:
    results = redis.zrevrange(f"autocomplete:{prefix}", 0, limit-1, withscores=True)
    return [word.decode() for word, _ in results]

# Storage estimate:
# 1M words × 10 avg chars per word × 20 bytes per entry ≈ 200MB
```

### Context-Aware Autocomplete

```json
{
  "suggest": {
    "place_suggest": {
      "prefix": "lon",
      "completion": {
        "field": "suggest",
        "contexts": {
          "place_type": ["city", "airport"]
        }
      }
    }
  }
}
```

---

## 10. Fuzzy Search

### Edit Distance (Levenshtein)

```
Edit operations: insert, delete, substitute (each costs 1)
Damerau-Levenshtein also includes transposition (swap adjacent chars)

distance("kitten", "sitting"):
   k→s (substitute): 1
   e→i (substitute): 1
   insert g at end:  1
   Total: 3

Dynamic Programming:
  dp[i][j] = edit distance between s1[0..i] and s2[0..j]
  dp[i][j] = dp[i-1][j-1] if s1[i] == s2[j]
           = 1 + min(dp[i-1][j],    // delete
                     dp[i][j-1],    // insert  
                     dp[i-1][j-1])  // substitute
```

### Elasticsearch Fuzzy Query

```json
{
  "query": {
    "fuzzy": {
      "title": {
        "value": "elastcsearch",
        "fuzziness": "AUTO",
        "prefix_length": 2,
        "max_expansions": 50,
        "transpositions": true
      }
    }
  }
}
```

**AUTO fuzziness:**
- 0-2 chars: exact
- 3-5 chars: fuzziness 1
- 6+ chars: fuzziness 2

### Trigram Indexing for Fuzzy Search

```
Word "hello" → trigrams: ["hel", "ell", "llo"]
Word "helo"  → trigrams: ["hel", "elo"]

Jaccard similarity: |intersection| / |union|
  = |{hel}| / |{hel, ell, llo, elo}| = 1/4 = 0.25

More similar:
"hello" → ["hel", "ell", "llo"]
"hellp" → ["hel", "ell", "llp"]
Jaccard = 2/4 = 0.5  (higher = more similar)
```

**PostgreSQL trigram index:**
```sql
CREATE EXTENSION pg_trgm;
CREATE INDEX products_name_trgm ON products USING GIN (name gin_trgm_ops);

SELECT * FROM products 
WHERE name % 'iphone'  -- trigram similarity operator
ORDER BY similarity(name, 'iphone') DESC
LIMIT 10;
```

---

## 11. Faceted Search

### Aggregations in Elasticsearch

```json
{
  "query": {
    "match": {"category": "electronics"}
  },
  "aggs": {
    "brands": {
      "terms": {
        "field": "brand.keyword",
        "size": 20
      }
    },
    "price_ranges": {
      "range": {
        "field": "price",
        "ranges": [
          {"to": 100},
          {"from": 100, "to": 500},
          {"from": 500}
        ]
      }
    },
    "avg_rating": {
      "avg": {"field": "rating"}
    },
    "rating_histogram": {
      "histogram": {
        "field": "rating",
        "interval": 1
      }
    }
  }
}
```

### Filter Aggregation (Post-Filter)

```json
{
  "query": {"match": {"category": "phones"}},
  "post_filter": {
    "term": {"brand.keyword": "Apple"}
  },
  "aggs": {
    "all_brands": {
      "terms": {"field": "brand.keyword"}
    }
  }
}
```

**Post-filter applies AFTER aggregation** — allows showing brand counts for all brands even when filtering by one brand (common UX pattern in faceted search).

### Filter Caching

```
Filter context queries are automatically cached by Elasticsearch:
  - Cached at shard level (per-segment cache)
  - Stored as bitsets (roaring bitmaps)
  - Cache key = query JSON
  - LRU eviction when cache exceeds 10% of heap

Queries cached: term, terms, range, exists, prefix, wildcard, geo_bounding_box
Queries NOT cached: match, match_phrase (scoring queries), script

Force caching: {"term": {"status": "active"}, "_cache": true}
```

---

## 12. Spell Correction

### Noisy Channel Model

```
P(intended | typed) ∝ P(typed | intended) × P(intended)

P(typed | intended): error model (how likely is a typo for this intended word?)
P(intended): language model (how likely is this word in context?)

"hte" → candidates: "the", "he", "she"
  P("the" | "hte") × P("the") = high × very high = best match
  P("he"  | "hte") × P("he")  = medium × high     = good match
  P("she" | "hte") × P("she") = low × medium       = worse
```

### Symmetric Delete Algorithm

```python
# Used in SymSpell - extremely fast spell correction
# Precompute: for each dictionary word, generate all deletes within edit distance k
# Query: generate all deletes of input within edit distance k
# Match: candidates that appear in both sets

def generate_deletes(word: str, max_distance: int) -> Set[str]:
    deletes = {word}
    current_level = {word}
    for _ in range(max_distance):
        next_level = set()
        for w in current_level:
            for i in range(len(w)):
                deleted = w[:i] + w[i+1:]
                if deleted not in deletes:
                    deletes.add(deleted)
                    next_level.add(deleted)
        current_level = next_level
    return deletes

# Index: {delete_variant → [original_words]}
# Query: generate deletes of input, look up in index
# O(1) lookup (hash table), precomputed index
```

### Elasticsearch Phrase Suggester

```json
{
  "suggest": {
    "text": "I am a roc strar",
    "phrase_suggestion": {
      "phrase": {
        "field": "title",
        "size": 3,
        "gram_size": 3,
        "direct_generator": [{
          "field": "title",
          "suggest_mode": "missing",
          "min_word_length": 3
        }],
        "highlight": {
          "pre_tag": "<em>",
          "post_tag": "</em>"
        }
      }
    }
  }
}
```

---

## 13. Personalized Search

### Query Understanding

```
Query: "python"
  Intent signals:
    - User is a developer (browsing history)
    - Time is daytime, weekday (work context)
    - Previous searches: "django", "flask" (programming context)
    
  Without personalization: Python language, Python snake
  With personalization:    Python programming language (reranked to top)
```

### User Behavior Signals

| Signal | Description | Weight |
|--------|-------------|--------|
| Click | User clicked result | High |
| Dwell time | Time spent on page after click | Very High |
| Skip | Result shown but not clicked | Negative |
| Reformulation | User reformulated query after clicking | Negative |
| Long click | Click + long dwell (>30s) | Very High |
| Bookmark | User bookmarked result | Very High |
| CTR | Click-through rate across users | High |

### Personalization Architecture

```
Query + User Context
        |
        v
[User Profile Service]
  - Interests (topic vector)
  - Historical queries
  - Click history
  - Location
        |
        v
[Query Rewriter]
  - Expand with user interest terms
  - Adjust boost based on user segment
        |
        v
[Elasticsearch]
  - function_score with user affinity boost
        |
        v
[Reranker ML Model]
  - Features: user affinity, recency, CTR, quality signals
  - Output: reranked result list
```

### A/B Testing Search

```
Treatment Assignment:
  - Hash(user_id + experiment_id) % 100 < traffic_percent → treatment
  
Metrics for search A/B tests:
  - MRR (Mean Reciprocal Rank): avg of 1/rank of first relevant result
  - NDCG (Normalized Discounted Cumulative Gain): graded relevance
  - CTR: click-through rate
  - Query abandonment rate
  - Zero-result rate

Statistical significance: minimum 2-week run for search experiments
```

---

## 14. Vector Search and Semantic Search

### Dense Vector Embeddings

```
Traditional keyword search: "car" does not match "automobile"
Semantic search: embed both in vector space, they are close together

Model: text → 768-dimensional float vector (BERT, sentence-transformers)

"car"        → [0.23, -0.15, 0.87, ...]  768 dims
"automobile" → [0.21, -0.14, 0.85, ...]  768 dims
"apple"      → [-0.42, 0.63, -0.11, ...] 768 dims

Cosine similarity("car", "automobile") ≈ 0.92 (very similar)
Cosine similarity("car", "apple")      ≈ 0.11 (unrelated)
```

### FAISS (Facebook AI Similarity Search)

```python
import faiss
import numpy as np

d = 768  # dimension
n = 1_000_000  # number of vectors

# Build index
index = faiss.IndexFlatL2(d)        # exact L2 search
# OR for approximate:
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist=1024)  # IVF
index.nprobe = 16  # number of clusters to search

index.train(vectors)  # train IVF
index.add(vectors)    # add all vectors

# Search
query_vector = np.array([[...]])  # 1 × 768
distances, indices = index.search(query_vector, k=10)
```

### Approximate Nearest Neighbor Algorithms

**HNSW (Hierarchical Navigable Small World):**
```
Build: Create layered graph where each node connects to M nearest neighbors
       Higher layers = fewer nodes, longer "jumps"
       Lower layers = more nodes, precise local search

Search: Start at top layer, greedily move toward query
        Descend through layers, narrowing candidate set
        
Parameters:
  M: connections per node (16-64); higher = better recall, more memory
  efConstruction: search depth during build (100-2000)
  ef: search depth during query (50-500); tune for recall vs speed

Memory: ~3.5 × M × 4 bytes per vector for graph structure
Recall: 99% recall at 0.1ms per query for 1M vectors
```

**IVF (Inverted File Index):**
```
Build: K-means cluster vectors into nlist clusters (centroids)
       Each vector assigned to nearest centroid

Search: Find nprobe nearest centroids to query
        Search only vectors in those clusters
        
nlist: sqrt(N) is a good starting point
nprobe: higher = better recall, slower query
```

### pgvector (PostgreSQL Extension)

```sql
CREATE EXTENSION vector;

CREATE TABLE documents (
  id    BIGINT PRIMARY KEY,
  text  TEXT,
  embedding vector(1536)  -- OpenAI ada-002 dimension
);

CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

-- Semantic search
SELECT id, text, 1 - (embedding <=> $1) AS similarity
FROM documents
ORDER BY embedding <=> $1  -- cosine distance operator
LIMIT 10;
```

### Hybrid Search (Keyword + Semantic)

```
score = α × BM25_score + (1 - α) × semantic_score

Problem: scores are on different scales
Solution: Reciprocal Rank Fusion (RRF)

RRF(d) = Σ 1 / (k + rank_i(d))
  k = 60 (constant to prevent high ranks dominating)
  rank_i = rank of document d in ranking i

Implementation:
  1. BM25 search → ranked list 1
  2. Vector search → ranked list 2
  3. Combine with RRF → final ranked list
```

---

## 15. Search Result Caching

### Query Cache

```
Cache key: (index_name, query_hash, filter_hash)
Cache value: list of (docID, score)

L1: Application-level cache (Redis, Memcached)
  - TTL: 5-60 seconds for real-time indexes
  - TTL: 1-24 hours for near-static content

L2: Elasticsearch query cache
  - Only caches filter context (not scoring)
  - Cache hit rate target: >80% for production

Cache invalidation:
  - On index refresh (automatic with short TTL)
  - Manual invalidation on bulk updates
  - Partial invalidation difficult → short TTL preferred
```

### Elasticsearch Shard Request Cache

```
Caches aggregation results and query hits count at shard level
Invalidated on each shard refresh

Configuration:
  index.requests.cache.enable: true
  indices.requests.cache.size: 1%  (of heap)

Most useful for:
  - Dashboards with time-range queries
  - Facet counts that change infrequently
```

### Result Precomputation

```
Popular queries (top 1% drive 80% of traffic):
  1. Identify top N queries from access logs
  2. Pre-run and cache results
  3. Warm cache on deployment

Implementation:
  - Cron job every 5 minutes: refresh top-1000 query cache
  - Cache-aside: serve stale, refresh asynchronously
  - Cache hit rate monitoring: alert if drops below 70%
```

---

## 16. Scaling Search

### Horizontal Sharding Strategies

**By Document ID (Default):**
```
shard = hash(doc_id) % num_shards
  + Even distribution
  + Simple routing
  - Cannot route by content type
  - Cannot easily delete old data
```

**By Time (Time-Based Sharding):**
```
Index: logs-2024-01, logs-2024-02, ..., logs-2024-12
  + Easy data lifecycle management (delete old index)
  + Hot/warm/cold architecture
  + Index aliases for seamless routing
  - Uneven distribution if queries focus on recent data
  - Need index aliases for cross-index search

Index lifecycle management (ILM):
  Hot:  active writes + reads (fast SSD)
  Warm: read-only, compressed (regular SSD)
  Cold: rare reads, highly compressed (HDD/S3)
  Delete: after retention period
```

**By Content Type:**
```
products_index: e-commerce products
articles_index: news/blog content
users_index:    user profiles

  + Optimized settings per type
  + Independent scaling
  - Cross-type queries require multi-index search
```

### Read Scaling with Replicas

```
Primary (1) + Replicas (2) = 3 copies per shard

Throughput: scales linearly with replicas for read-heavy workloads
Latency: replica serves nearest to client (zone-aware routing)
Failure: primary fails → promote replica (automatic)

Adaptive replica selection:
  ES chooses replica based on:
  - Response time (EWMA)
  - Queue size
  - Service time
```

### Index Aliases for Zero-Downtime Reindex

```python
# Zero-downtime reindex pattern:
# 1. Create new index with updated settings
PUT /products_v2

# 2. Reindex data from old to new
POST /_reindex
{
  "source": {"index": "products_v1"},
  "dest":   {"index": "products_v2"}
}

# 3. Atomic alias swap
POST /_aliases
{
  "actions": [
    {"remove": {"index": "products_v1", "alias": "products"}},
    {"add":    {"index": "products_v2", "alias": "products"}}
  ]
}

# 4. Delete old index
DELETE /products_v1
```

---

## 17. Search Relevance Tuning

### Click-Through Rate (CTR) Feedback

```
Signal collection:
  - Log: (query, position, docID, clicked, dwell_time)
  - Aggregate: CTR per (query, docID) pair
  
Position bias correction:
  - Position 1 gets 10x more clicks than position 10 even for same quality
  - Inverse propensity scoring: weight click by 1/P(click_at_position)
  - Use interleaving experiments to directly compare rankings

Learning:
  Positive signals: click with dwell > 30s
  Negative signals: skip (shown, not clicked, next result clicked)
```

### Learning to Rank (LTR)

```
Training data: (query, doc) → relevance label

Feature engineering:
  Query-doc features: BM25, TF-IDF, query term coverage
  Document features:  PageRank, freshness, content quality score
  Query features:     query length, query frequency in logs
  User features:      user affinity for topic, location
  
Models:
  Pointwise: predict relevance score per (query, doc) → regression/classification
  Pairwise:  predict which of (doc_a, doc_b) is better → RankSVM, RankNet
  Listwise:  optimize ranking metrics (NDCG) directly → LambdaMART, XGBoost

Elasticsearch LTR plugin:
  1. Upload trained model
  2. Use rescore query to apply LTR model to top-N BM25 results
  
  {
    "query": {"match": {"title": "laptop"}},
    "rescore": {
      "window_size": 100,
      "query": {
        "rescore_query": {
          "sltr": {
            "params": {"keywords": "laptop"},
            "model": "laptop_ltr_model"
          }
        }
      }
    }
  }
```

---

## 18. Real-Time Indexing Pipeline

### Kafka to Elasticsearch Pipeline

```
[Application DB] 
      |
      | (CDC via Debezium or binlog)
      v
[Kafka Topic: products_changes]
      |
      | (Kafka Connect Elasticsearch Sink OR custom consumer)
      v
[Elasticsearch Indexer Service]
  - Batch documents (bulk API, batch_size=500)
  - Handle retries with exponential backoff
  - Dead letter queue for failed documents
      |
      v
[Elasticsearch Cluster]

Throughput: 50,000 docs/sec per indexer instance
Latency: ~1 second end-to-end (DB change → searchable)
```

### Bulk Indexing API

```python
from elasticsearch import Elasticsearch, helpers

def bulk_index(es: Elasticsearch, documents: List[dict]):
    actions = [
        {
            "_index": "products",
            "_id": doc["id"],
            "_source": doc,
            "_op_type": "index"  # or "update", "delete"
        }
        for doc in documents
    ]
    
    success, failed = helpers.bulk(
        es,
        actions,
        chunk_size=500,
        request_timeout=30,
        raise_on_error=False,
        raise_on_exception=False
    )
    return success, failed
```

### Zero-Downtime Index Rebuild

```
Scenario: Need to change analyzer (requires full reindex)

Strategy: Blue-Green Index Deployment

1. Current state: alias "products" → index "products_v1"
2. Create "products_v2" with new mapping/settings
3. Enable dual-write: write to both v1 and v2
4. Reindex historical data into v2 (concurrent with dual-write)
5. Verify v2 has all data (compare doc counts)
6. Atomic alias swap: "products" → "products_v2"
7. Stop dual-write to v1
8. Delete v1 after monitoring period

Handling the gap:
  - Track reindex_start_time
  - After bulk reindex, catch up documents updated after reindex_start_time
  - Use Kafka offsets to replay recent changes
```

---

## 19. Search for E-Commerce

### Product Search Features

```
Query: "red nike running shoes size 10"

Query understanding:
  - Brand: Nike
  - Color: red
  - Category: running shoes
  - Size: 10

Mapping to search:
  - Full-text match on "running shoes"
  - Filter on brand = "Nike"
  - Filter on color = "red"
  - Filter on size = "10"
  - Sort by: inventory availability × relevance score
```

### Inventory-Aware Ranking

```json
{
  "query": {
    "function_score": {
      "query": {
        "bool": {
          "must":   [{"match": {"name": "running shoes"}}],
          "filter": [
            {"term": {"in_stock": true}},
            {"term": {"brand": "nike"}}
          ]
        }
      },
      "functions": [
        {
          "filter": {"range": {"inventory_count": {"gte": 10}}},
          "weight": 1.5
        },
        {
          "field_value_factor": {
            "field": "sales_velocity",
            "modifier": "log1p",
            "factor": 0.1
          }
        },
        {
          "gauss": {
            "margin_percent": {
              "origin": "40",
              "scale":  "20",
              "decay":  "0.5"
            }
          }
        }
      ]
    }
  }
}
```

### Category Filtering and Navigation

```
Product taxonomy:
  Electronics > Phones > Smartphones > iPhone

Elasticsearch approach:
  - Store path as keyword array: ["Electronics", "Electronics > Phones", ...]
  - Filter by prefix: all subcategories included
  
  {
    "filter": {
      "term": {"category_path": "Electronics > Phones"}
    }
  }

Alternative: Nested category aggregation for category tree navigation
```

### Synonyms and Query Expansion for E-Commerce

```
Brand synonyms:     "levi's" = "levis"
Product synonyms:   "tv" = "television" = "telly"
Size synonyms:      "xl" = "extra large" = "size 42"
Color normalization: "navy" → "dark blue", "royal blue" → "blue"
Spelling variants:  "colour" → "color"

Maintained in:
  - Config file (requires index rebuild on change)
  - Database-driven (allows hot reload via Elasticsearch /_analyze API)
```

---

## 20. Quick Reference

### Elasticsearch vs Solr vs OpenSearch

| Feature | Elasticsearch | Solr | OpenSearch |
|---------|---------------|------|------------|
| License | SSPL (7.x+) | Apache 2.0 | Apache 2.0 |
| Parent | Elastic | Apache | AWS |
| ML Features | Elasticsearch ML | Limited | AWS-integrated |
| KNN/Vector | Dense vector + kNN | Dense vector (8.x+) | k-NN plugin |
| Cloud | Elastic Cloud | Solr Cloud | AWS OpenSearch |
| Schema | Dynamic mapping | Schema.xml | Dynamic mapping |
| UI | Kibana | Solr Admin UI | OpenSearch Dashboards |
| Community | Large | Large | Growing |
| Best For | General + APM | Mature deployments | AWS ecosystem |

### Search Ranking Signals Table

| Signal Type | Examples | Update Frequency |
|-------------|----------|-----------------|
| Query-Document Relevance | BM25, TF-IDF | Per query |
| Authority | PageRank, backlinks | Weekly |
| Freshness | Publish date, update date | Per document |
| Quality | Readability, completeness | Weekly |
| Popularity | CTR, impression count | Daily |
| Personalization | User affinity, history | Per user session |
| Business | Margin, sponsored, featured | Per business rule |
| Engagement | Dwell time, bounce rate | Daily aggregation |

### Latency Targets for Search Systems

| Component | P50 Target | P99 Target |
|-----------|-----------|-----------|
| Autocomplete | 10ms | 50ms |
| Keyword search | 50ms | 200ms |
| Semantic search (exact) | 100ms | 500ms |
| Semantic search (ANN) | 20ms | 100ms |
| Faceted search | 100ms | 300ms |
| Total search API | 200ms | 500ms |

### Common Interview Questions

1. **How would you design autocomplete for Google Search?**
   - Trie for prefix matching, top-k per prefix node, Redis for hot prefixes, Elasticsearch completion suggester for flexible scoring, personalization layer

2. **How does Elasticsearch handle a search across 100 shards?**
   - Scatter: query all 100 shards in parallel; Gather: merge top-k from each (100 × k results); Fetch: retrieve full docs for final k; two-phase approach

3. **How would you implement semantic search at scale?**
   - Offline: encode all docs with sentence-transformers, store in FAISS or pgvector; Online: encode query, ANN search, re-rank with cross-encoder, combine with BM25 via RRF

4. **How do you handle zero-downtime mapping changes in Elasticsearch?**
   - Create new index with new mapping, dual-write, bulk reindex, atomic alias swap, verify, cleanup

5. **What is the problem with TF-IDF for long documents?**
   - Long documents get higher TF scores even if term density is the same. BM25 normalizes by document length using parameter `b`.
