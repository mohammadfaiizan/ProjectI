"""
SYSTEM DESIGN: WEB SEARCH ENGINE
==================================

Problem Statement:
Design a web-scale search engine that crawls the web, indexes pages,
and returns ranked results for a given query.

Functional Requirements:
  - Web crawling: discover and store web pages
  - Indexing: build inverted index from crawled pages
  - Search: return ranked results for a query
  - Autocomplete / query suggestions

Non-Functional Requirements:
  - Index 20B web pages
  - 100K searches/sec (Google handles ~100K)
  - Search latency < 500ms (< 200ms for top results)
  - Freshness: re-crawl popular pages every few hours

Core Components:

  1. Web Crawler:
     Distributed BFS/DFS over the web.
     Frontier: URL queue (Redis priority queue or Kafka).
     Politeness: obey robots.txt; rate limit per domain.
     DNS cache: avoid redundant lookups.
     Dedup: Bloom filter for seen URLs.
     URL normalization: https://www.example.com/path?a=1 → canonical form.

  2. Indexer:
     Parse HTML → extract text + links.
     Tokenize: lower-case, remove stop words, stem/lemmatize.
     Inverted index: {token → postings list [(doc_id, tf, positions)]}.
     Forward index: {doc_id → [tokens]}.
     Store in distributed store (BigTable/HBase).

  3. Ranking (PageRank + BM25):
     PageRank: iterative algorithm. PR(A) = (1-d) + d × Σ PR(B)/L(B).
               d=0.85 damping factor. B links to A. L(B) = outlinks of B.
     BM25:     TF-IDF variant with term saturation + doc length norm.
               score(D,Q) = Σ IDF(qi) × f(qi,D) × (k1+1) / (f(qi,D) + k1*(1-b+b*|D|/avgdl))
     Modern:   BERT embeddings for semantic search. BM25 for lexical.

  4. Autocomplete:
     Trie of popular queries.
     Sorted by frequency. Prefix search.
     Redis sorted set: ZADD queries 1000 "python tutorial" → ZRANGEBYLEX.

  5. Anti-Spam:
     Link farms: penalize domains with too many outlinks.
     Spam detection: ML classifier on features (link ratio, ad density).
     TrustRank: seed from authoritative sites; propagate trust.

Storage Scale:
  Raw HTML: 20B pages × 100KB avg = 2PB (S3).
  Inverted index: ~500TB (Bigtable/Spanner).
  PageRank graph: 20B nodes, 1T edges → ~100TB.
"""

from __future__ import annotations

import math
import time
import hashlib
import re
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse


# ─────────────────────────────────────────────
# BLOOM FILTER (for URL deduplication)
# ─────────────────────────────────────────────

class BloomFilter:
    """
    Probabilistic set membership. No false negatives; small false positive rate.
    Uses k hash functions over m bits.
    """

    def __init__(self, capacity: int = 1_000_000, fp_rate: float = 0.01):
        m     = int(-capacity * math.log(fp_rate) / math.log(2) ** 2)
        k     = int(m / capacity * math.log(2))
        self._m   = m
        self._k   = max(k, 1)
        self._bits = bytearray(m // 8 + 1)

    def _hashes(self, item: str) -> List[int]:
        h1 = int(hashlib.md5(item.encode()).hexdigest(), 16)
        h2 = int(hashlib.sha1(item.encode()).hexdigest(), 16)
        return [(h1 + i * h2) % self._m for i in range(self._k)]

    def add(self, item: str):
        for h in self._hashes(item):
            self._bits[h // 8] |= 1 << (h % 8)

    def __contains__(self, item: str) -> bool:
        return all(self._bits[h // 8] & (1 << (h % 8)) for h in self._hashes(item))


# ─────────────────────────────────────────────
# URL FRONTIER (crawl priority queue)
# ─────────────────────────────────────────────

@dataclass
class CrawlURL:
    url:       str
    priority:  float   # higher = crawl sooner
    depth:     int
    domain:    str

    def __lt__(self, other: "CrawlURL"):
        return self.priority > other.priority   # max priority first


class URLFrontier:
    """Per-domain bucketed URL queue (politeness)."""

    def __init__(self):
        # domain → [CrawlURL]
        self._buckets: Dict[str, List[CrawlURL]] = defaultdict(list)
        self._domain_last_fetch: Dict[str, float] = {}
        self._seen = BloomFilter(10_000_000)

    def add(self, url: str, priority: float = 1.0, depth: int = 0):
        if url in self._seen:
            return
        self._seen.add(url)
        domain = urlparse(url).netloc
        self._buckets[domain].append(CrawlURL(url, priority, depth, domain))

    def next(self, min_crawl_delay_s: float = 1.0) -> Optional[CrawlURL]:
        """Return next URL to crawl respecting politeness delay."""
        now = time.time()
        for domain, urls in self._buckets.items():
            if not urls:
                continue
            last = self._domain_last_fetch.get(domain, 0.0)
            if now - last >= min_crawl_delay_s:
                self._domain_last_fetch[domain] = now
                urls.sort(reverse=True)
                return urls.pop(0)
        return None

    def total(self) -> int:
        return sum(len(v) for v in self._buckets.values())


# ─────────────────────────────────────────────
# HTML PARSER / TEXT EXTRACTOR
# ─────────────────────────────────────────────

STOP_WORDS = {"the","a","an","is","it","in","on","at","to","for","of","and","or","with"}

def extract_tokens(text: str) -> List[str]:
    """Tokenize text: lower, remove punctuation, remove stop words."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]

def extract_links(html: str, base_url: str) -> List[str]:
    """Extract href links from simulated HTML."""
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', html)
    links = []
    for h in hrefs:
        if h.startswith("http"):
            links.append(h)
        elif h.startswith("/"):
            domain = urlparse(base_url).netloc
            links.append(f"https://{domain}{h}")
    return links


# ─────────────────────────────────────────────
# DOCUMENT
# ─────────────────────────────────────────────

@dataclass
class Document:
    doc_id:    int
    url:       str
    title:     str
    text:      str
    tokens:    List[str]
    outlinks:  List[str]
    crawled_at: float

    @property
    def token_count(self) -> int:
        return len(self.tokens)


# ─────────────────────────────────────────────
# INVERTED INDEX
# ─────────────────────────────────────────────

@dataclass
class Posting:
    doc_id:    int
    tf:        float   # term frequency (normalized)
    positions: List[int] = field(default_factory=list)


class InvertedIndex:
    """
    Maps token → postings list [(doc_id, tf, positions)].
    Stored per shard in production (BigTable / Elasticsearch).
    """

    def __init__(self):
        self._index:   Dict[str, List[Posting]] = defaultdict(list)
        self._docs:    Dict[int, Document]       = {}
        self._avg_dl:  float = 0.0
        self._doc_count: int = 0

    def add_document(self, doc: Document):
        self._docs[doc.doc_id] = doc
        self._doc_count += 1
        total_tokens = sum(len(d.tokens) for d in self._docs.values())
        self._avg_dl = total_tokens / self._doc_count

        # Build term frequency map
        tf_map: Dict[str, List[int]] = defaultdict(list)
        for pos, token in enumerate(doc.tokens):
            tf_map[token].append(pos)

        for token, positions in tf_map.items():
            tf  = len(positions) / max(doc.token_count, 1)
            posting = Posting(doc.doc_id, tf, positions)
            self._index[token].append(posting)

    def postings(self, token: str) -> List[Posting]:
        return self._index.get(token, [])

    def idf(self, token: str) -> float:
        """Inverse document frequency."""
        df = len(self._index.get(token, []))
        if df == 0:
            return 0.0
        return math.log((self._doc_count + 1) / (df + 1)) + 1

    def doc(self, doc_id: int) -> Optional[Document]:
        return self._docs.get(doc_id)


# ─────────────────────────────────────────────
# BM25 RANKER
# ─────────────────────────────────────────────

class BM25Ranker:
    """
    BM25 ranking: TF-IDF variant with saturation (k1) and length norm (b).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self._k1 = k1
        self._b  = b

    def score(self, index: InvertedIndex, query_tokens: List[str],
              doc_id: int) -> float:
        doc  = index.doc(doc_id)
        if not doc:
            return 0.0
        dl   = doc.token_count
        avdl = index._avg_dl
        tf_map = defaultdict(int)
        for tok in doc.tokens:
            tf_map[tok] += 1

        total = 0.0
        for tok in query_tokens:
            idf = index.idf(tok)
            tf  = tf_map.get(tok, 0)
            num = tf * (self._k1 + 1)
            den = tf + self._k1 * (1 - self._b + self._b * dl / max(avdl, 1))
            total += idf * num / max(den, 1e-9)
        return total


# ─────────────────────────────────────────────
# PAGERANK
# ─────────────────────────────────────────────

class PageRank:
    """
    Iterative PageRank algorithm.
    PR(A) = (1-d) + d × Σ PR(B) / L(B)
    d=0.85, iterate until convergence.
    """

    def __init__(self, damping: float = 0.85):
        self._d = damping

    def compute(self, docs: Dict[int, Document],
                iterations: int = 20) -> Dict[int, float]:
        n = len(docs)
        if n == 0:
            return {}

        # Build URL → doc_id map
        url_to_id = {d.url: d.doc_id for d in docs.values()}

        # Initial PR = 1/N
        pr = {doc_id: 1.0 / n for doc_id in docs}

        for _ in range(iterations):
            new_pr: Dict[int, float] = {}
            for doc_id, doc in docs.items():
                # Sum contributions from all pages linking to this one
                incoming = 0.0
                for src_id, src_doc in docs.items():
                    # Does src link to doc?
                    if doc.url in [l for l in src_doc.outlinks
                                   if url_to_id.get(l) == doc_id]:
                        outlinks = len(src_doc.outlinks)
                        incoming += pr[src_id] / max(outlinks, 1)
                new_pr[doc_id] = (1 - self._d) / n + self._d * incoming

            # Normalize
            total = sum(new_pr.values())
            pr    = {k: v / total for k, v in new_pr.items()}

        return pr


# ─────────────────────────────────────────────
# AUTOCOMPLETE TRIE
# ─────────────────────────────────────────────

class TrieNode:
    def __init__(self):
        self.children: Dict[str, "TrieNode"] = {}
        self.is_end:   bool = False
        self.frequency: int = 0


class AutocompleteTrie:
    def __init__(self):
        self._root = TrieNode()

    def insert(self, query: str, freq: int = 1):
        node = self._root
        for c in query.lower():
            node = node.children.setdefault(c, TrieNode())
            node.frequency = max(node.frequency, freq)
        node.is_end   = True
        node.frequency = freq

    def suggest(self, prefix: str, n: int = 5) -> List[str]:
        node = self._root
        for c in prefix.lower():
            if c not in node.children:
                return []
            node = node.children[c]

        results: List[Tuple[int, str]] = []
        self._dfs(node, prefix.lower(), results)
        results.sort(reverse=True)
        return [q for _, q in results[:n]]

    def _dfs(self, node: TrieNode, prefix: str,
             results: List[Tuple[int, str]]):
        if node.is_end:
            results.append((node.frequency, prefix))
        for c, child in node.children.items():
            self._dfs(child, prefix + c, results)


# ─────────────────────────────────────────────
# SEARCH ENGINE
# ─────────────────────────────────────────────

class SearchEngine:
    def __init__(self):
        self._index  = InvertedIndex()
        self._ranker = BM25Ranker()
        self._pr     = PageRank()
        self._autocomplete = AutocompleteTrie()
        self._pr_scores: Dict[int, float] = {}
        self._next_id = 0

    def index_document(self, url: str, title: str, text: str,
                       outlinks: List[str]):
        tokens = extract_tokens(title + " " + text)
        doc    = Document(self._next_id, url, title, text, tokens,
                          outlinks, time.time())
        self._next_id += 1
        self._index.add_document(doc)
        return doc

    def compute_pagerank(self):
        self._pr_scores = self._pr.compute(self._index._docs, iterations=5)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        tokens = extract_tokens(query)
        if not tokens:
            return []

        # Collect candidate docs from posting lists
        candidate_ids: Set[int] = set()
        for tok in tokens:
            for posting in self._index.postings(tok):
                candidate_ids.add(posting.doc_id)

        # Score candidates: BM25 + PageRank
        scored: List[Tuple[float, int]] = []
        for doc_id in candidate_ids:
            bm25    = self._ranker.score(self._index, tokens, doc_id)
            pr      = self._pr_scores.get(doc_id, 0.5)
            score   = bm25 * 0.7 + pr * 10 * 0.3   # weight BM25 70%, PR 30%
            scored.append((score, doc_id))

        scored.sort(reverse=True)
        results = []
        for score, doc_id in scored[:top_k]:
            doc = self._index.doc(doc_id)
            if doc:
                results.append((doc, round(score, 4)))
        return results

    def train_autocomplete(self, queries: List[Tuple[str, int]]):
        for q, freq in queries:
            self._autocomplete.insert(q, freq)

    def suggest(self, prefix: str) -> List[str]:
        return self._autocomplete.suggest(prefix)


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_search():
    print("=" * 65)
    print("SYSTEM DESIGN: WEB SEARCH ENGINE")
    print("=" * 65)

    engine = SearchEngine()

    # ── Index Documents ───────────────────────
    print("\n[1] INDEXING DOCUMENTS")
    print("─" * 55)

    corpus = [
        ("https://python.org",       "Python Programming Language",
         "Python is a high-level programming language for data science and web development.",
         ["https://pypi.org", "https://docs.python.org"]),
        ("https://docs.python.org",  "Python Documentation",
         "Complete reference for Python standard library functions and modules.",
         ["https://python.org"]),
        ("https://golang.org",       "Go Programming Language",
         "Go is a statically typed compiled programming language designed for systems.",
         ["https://pkg.go.dev"]),
        ("https://rust-lang.org",    "Rust Programming Language",
         "Rust is a systems programming language focused on safety performance and concurrency.",
         ["https://doc.rust-lang.org"]),
        ("https://scikit-learn.org", "Scikit-Learn Machine Learning",
         "Python machine learning library with algorithms for data science classification regression.",
         ["https://python.org", "https://numpy.org"]),
        ("https://tensorflow.org",   "TensorFlow Deep Learning",
         "Open source machine learning framework for deep learning neural network models.",
         ["https://python.org"]),
    ]

    docs = []
    for url, title, text, links in corpus:
        doc = engine.index_document(url, title, text, links)
        docs.append(doc)
        print(f"  Indexed: {url} ({doc.token_count} tokens)")

    # ── PageRank ──────────────────────────────
    print("\n[2] PAGERANK COMPUTATION")
    print("─" * 55)

    engine.compute_pagerank()
    pr_sorted = sorted(engine._pr_scores.items(), key=lambda x: -x[1])
    for doc_id, pr in pr_sorted:
        doc = engine._index.doc(doc_id)
        print(f"  PR={pr:.4f}  {doc.url}")

    # ── Search ────────────────────────────────
    print("\n[3] SEARCH RESULTS")
    print("─" * 55)

    queries = ["python programming", "machine learning deep learning", "systems programming safety"]
    for q in queries:
        print(f"\n  Query: '{q}'")
        results = engine.search(q, top_k=3)
        for rank, (doc, score) in enumerate(results, 1):
            print(f"    {rank}. [{score:.3f}] {doc.title[:45]}")
            print(f"         {doc.url}")

    # ── BM25 Scoring ──────────────────────────
    print("\n[4] BM25 TERM SCORES (for 'python machine learning')")
    print("─" * 55)

    tokens = extract_tokens("python machine learning")
    print(f"  Query tokens: {tokens}")
    print(f"  IDF scores:")
    for tok in tokens:
        idf = engine._index.idf(tok)
        df  = len(engine._index.postings(tok))
        print(f"    '{tok}': idf={idf:.3f}  df={df}")

    # ── Autocomplete ──────────────────────────
    print("\n[5] AUTOCOMPLETE TRIE")
    print("─" * 55)

    popular_queries = [
        ("python tutorial", 10000),
        ("python list comprehension", 5000),
        ("python decorator", 3000),
        ("python async await", 4000),
        ("programming languages 2024", 2000),
        ("program in rust", 1500),
    ]
    engine.train_autocomplete(popular_queries)

    for prefix in ["pyth", "prog", "x"]:
        suggestions = engine.suggest(prefix)
        print(f"  '{prefix}' → {suggestions}")

    # ── Bloom Filter (URL dedup) ───────────────
    print("\n[6] URL FRONTIER WITH BLOOM FILTER")
    print("─" * 55)

    frontier = URLFrontier()
    seed_urls = [
        "https://python.org",
        "https://python.org/docs",
        "https://golang.org",
        "https://python.org",  # duplicate
        "https://rust-lang.org",
    ]
    for url in seed_urls:
        frontier.add(url, priority=1.0)

    print(f"  Seeded {len(seed_urls)} URLs; unique in frontier: {frontier.total()}")
    print("  (duplicate python.org filtered by Bloom filter)")

    # ── Architecture ──────────────────────────
    print("\n[7] SEARCH ENGINE ARCHITECTURE")
    print("─" * 55)

    arch = [
        ("Crawler",          "Distributed BFS; per-domain rate limiting; robots.txt"),
        ("HTML store",       "S3: raw HTML; processed text in separate bucket"),
        ("Indexer",          "Hadoop/Spark: tokenize → inverted index → BigTable"),
        ("Inverted index",   "Sharded: CRC32(token) % N_shards → shard node"),
        ("PageRank",         "MapReduce: iterative PR computation over link graph"),
        ("BM25",             "Per-shard scoring → global merge top-K"),
        ("Serving",          "Query → tokenize → scatter to N shards → gather → rank"),
        ("Caching",          "Cache top-1000 queries in Redis (pre-computed results)"),
        ("Autocomplete",     "Redis sorted set per prefix; or distributed trie"),
        ("Freshness",        "Crawl priority: PageRank × recency score"),
    ]
    for component, detail in arch:
        print(f"  {component:<18} {detail}")


if __name__ == "__main__":
    demonstrate_search()
