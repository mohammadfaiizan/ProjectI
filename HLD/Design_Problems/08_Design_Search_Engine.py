"""
Search Engine System Design - Python Implementation
Demonstrates: web crawler with BFS + robots.txt + Bloom filter,
inverted index construction, TF-IDF and BM25 scoring, PageRank computation,
query processing with stop words/stemming, snippet generation.
No external dependencies - standard library only.
"""

import hashlib
import math
import re
import time
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────
# Bloom Filter for URL Deduplication
# ─────────────────────────────────────────────

class BloomFilter:
    """
    Probabilistic data structure for URL deduplication.
    - O(1) lookup, no false negatives
    - ~1% false positive rate with k=7 hash functions, m=10*n bits
    """

    def __init__(self, capacity: int = 10_000_000, error_rate: float = 0.01):
        self.capacity = capacity
        # Optimal bit array size: m = -n*ln(p) / (ln(2)^2)
        self.m = int(-capacity * math.log(error_rate) / (math.log(2) ** 2))
        # Optimal hash functions: k = (m/n) * ln(2)
        self.k = max(1, int((self.m / capacity) * math.log(2)))
        self._bits = [False] * self.m
        self._count = 0

    def add(self, item: str):
        for seed in range(self.k):
            idx = self._hash(item, seed)
            self._bits[idx] = True
        self._count += 1

    def contains(self, item: str) -> bool:
        return all(self._bits[self._hash(item, seed)] for seed in range(self.k))

    def _hash(self, item: str, seed: int) -> int:
        digest = hashlib.md5(f"{seed}:{item}".encode()).hexdigest()
        return int(digest, 16) % self.m

    @property
    def fill_ratio(self) -> float:
        return sum(self._bits) / self.m


# ─────────────────────────────────────────────
# Document & Posting
# ─────────────────────────────────────────────

@dataclass
class WebDocument:
    url: str
    title: str
    content: str
    links: list = field(default_factory=list)   # outbound URLs
    page_rank: float = 0.15
    crawled_at: float = field(default_factory=time.time)
    doc_id: int = 0
    language: str = "en"

    @property
    def full_text(self) -> str:
        return f"{self.title} {self.title} {self.content}"   # title weighted 2x


@dataclass
class Posting:
    doc_id: int
    tf: float           # term frequency in document
    positions: list     # character/word positions for phrase queries


# ─────────────────────────────────────────────
# Query Tokenizer
# ─────────────────────────────────────────────

class QueryTokenizer:
    """Tokenize, remove stop words, and apply simple stemming."""

    STOP_WORDS = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "shall",
        "should", "may", "might", "must", "can", "could", "to", "of", "in",
        "on", "at", "by", "for", "with", "about", "as", "into", "from",
        "and", "or", "but", "if", "then", "that", "this", "it", "i", "you",
        "he", "she", "we", "they", "what", "which", "who", "how", "when"
    }

    # Simple suffix rules for English stemming (Porter-lite)
    SUFFIXES = [
        ("ational", "ate"), ("tional", "tion"), ("enci", "ence"),
        ("anci", "ance"), ("izer", "ize"), ("ising", "ise"),
        ("izing", "ize"), ("ation", "ate"), ("ations", "ate"),
        ("ness", ""), ("ment", ""), ("ments", ""),
        ("ings", "ing"), ("ing", ""), ("edly", "ed"),
        ("ed", ""), ("ies", "y"), ("ier", "y"),
        ("ies", "ie"), ("ness", ""), ("ly", ""),
        ("er", ""), ("ers", ""), ("est", ""),
        ("es", ""), ("s", ""),
    ]

    def tokenize(self, text: str, remove_stops: bool = True,
                 apply_stemming: bool = True) -> list:
        # Lowercase and split on non-alphanumeric
        tokens = re.findall(r'[a-z0-9]+', text.lower())
        result = []
        for token in tokens:
            if remove_stops and token in self.STOP_WORDS:
                continue
            if len(token) < 2:
                continue
            if apply_stemming:
                token = self._stem(token)
            result.append(token)
        return result

    def _stem(self, word: str) -> str:
        """Simplified suffix stripping (not full Porter stemmer)."""
        if len(word) <= 4:
            return word
        for suffix, replacement in self.SUFFIXES:
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                return word[:-len(suffix)] + replacement
        return word


# ─────────────────────────────────────────────
# Inverted Index
# ─────────────────────────────────────────────

class InvertedIndex:
    """
    Core of a search engine.
    term -> list of Posting objects (doc_id, tf, positions)
    """

    def __init__(self):
        self._index: defaultdict = defaultdict(list)   # term -> [Posting]
        self._doc_count = 0
        self._doc_lengths: dict = {}     # doc_id -> word count (for BM25)
        self._avg_doc_length = 0.0
        self.tokenizer = QueryTokenizer()

    def add_document(self, doc: WebDocument) -> list:
        """Index a document. Returns list of extracted terms."""
        tokens = self.tokenizer.tokenize(doc.full_text)
        if not tokens:
            return []

        self._doc_lengths[doc.doc_id] = len(tokens)
        self._doc_count += 1
        # Recompute average
        self._avg_doc_length = sum(self._doc_lengths.values()) / self._doc_count

        # Compute TF and positions
        term_positions: defaultdict = defaultdict(list)
        for pos, token in enumerate(tokens):
            term_positions[token].append(pos)

        for term, positions in term_positions.items():
            tf = len(positions) / len(tokens)
            posting = Posting(doc_id=doc.doc_id, tf=tf, positions=positions)
            self._index[term].append(posting)

        return list(term_positions.keys())

    def get_postings(self, term: str) -> list:
        """Return postings list for a term (for boolean/ranked retrieval)."""
        stemmed = self.tokenizer._stem(term.lower())
        return self._index.get(stemmed, [])

    def document_frequency(self, term: str) -> int:
        stemmed = self.tokenizer._stem(term.lower())
        return len(self._index.get(stemmed, []))

    @property
    def total_documents(self) -> int:
        return self._doc_count


# ─────────────────────────────────────────────
# TF-IDF Scorer
# ─────────────────────────────────────────────

class TFIDFScorer:
    """Classic TF-IDF scoring with cosine similarity."""

    def score(self, query_terms: list, index: InvertedIndex,
              top_k: int = 10) -> list:
        """Returns [(doc_id, score)] sorted by score descending."""
        doc_scores: defaultdict = defaultdict(float)

        for term in query_terms:
            postings = index.get_postings(term)
            if not postings:
                continue
            df = len(postings)
            # IDF with smoothing
            idf = math.log((index.total_documents + 1) / (df + 1)) + 1
            for posting in postings:
                doc_scores[posting.doc_id] += posting.tf * idf

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:top_k]


# ─────────────────────────────────────────────
# BM25 Scorer (Better than TF-IDF)
# ─────────────────────────────────────────────

class BM25Scorer:
    """
    BM25 (Best Match 25) — industry standard for text ranking.
    Improves on TF-IDF:
      - TF saturation: diminishing returns for repeated terms (k1)
      - Document length normalization (b)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1    # Term frequency saturation parameter
        self.b = b      # Document length normalization

    def score(self, query_terms: list, index: InvertedIndex,
              top_k: int = 10) -> list:
        doc_scores: defaultdict = defaultdict(float)
        N = index.total_documents
        avgdl = index._avg_doc_length or 1

        for term in query_terms:
            postings = index.get_postings(term)
            if not postings:
                continue
            df = len(postings)
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)

            for posting in postings:
                doc_length = index._doc_lengths.get(posting.doc_id, avgdl)
                tf = posting.tf * doc_length   # Convert back to raw count
                # BM25 TF normalization
                bm25_tf = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * doc_length / avgdl)
                )
                doc_scores[posting.doc_id] += idf * bm25_tf

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:top_k]


# ─────────────────────────────────────────────
# PageRank
# ─────────────────────────────────────────────

class PageRankComputer:
    """
    Iterative PageRank computation.
    PR(A) = (1-d)/N + d * sum(PR(T_i)/C(T_i)) for all T_i linking to A
    d = damping factor (0.85)
    """

    def __init__(self, damping: float = 0.85, max_iterations: int = 50,
                 tolerance: float = 1e-6):
        self.damping = damping
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def compute(self, documents: dict) -> dict:
        """
        documents: dict of {doc_id: WebDocument}
        Returns: dict of {doc_id: pagerank_score}
        """
        if not documents:
            return {}

        # Build URL -> doc_id mapping
        url_to_id = {doc.url: doc.doc_id for doc in documents.values()}
        N = len(documents)

        # Initialize all pages with equal rank
        pr = {doc_id: 1.0 / N for doc_id in documents}

        # Build adjacency: incoming links per doc
        incoming: defaultdict = defaultdict(list)   # doc_id -> [source_doc_ids]
        outlinks: dict = {}                          # doc_id -> outbound count
        for doc_id, doc in documents.items():
            valid_links = [url_to_id[url] for url in doc.links if url in url_to_id]
            outlinks[doc_id] = len(valid_links) or 1  # Avoid division by zero
            for target_id in valid_links:
                incoming[target_id].append(doc_id)

        # Iterate until convergence
        for iteration in range(self.max_iterations):
            new_pr = {}
            for doc_id in documents:
                # Dangling node contribution (page with no outlinks)
                rank_sum = sum(
                    pr[src_id] / outlinks.get(src_id, 1)
                    for src_id in incoming.get(doc_id, [])
                )
                new_pr[doc_id] = (1 - self.damping) / N + self.damping * rank_sum

            # Check convergence
            delta = sum(abs(new_pr[doc_id] - pr[doc_id]) for doc_id in documents)
            pr = new_pr
            if delta < self.tolerance:
                print(f"  [PageRank] Converged in {iteration + 1} iterations "
                      f"(delta={delta:.2e})")
                break

        return pr


# ─────────────────────────────────────────────
# Snippet Generator
# ─────────────────────────────────────────────

class SnippetGenerator:
    """Find and return the most relevant excerpt from a document."""

    def generate(self, content: str, query_terms: list,
                 max_length: int = 160) -> str:
        if not content or not query_terms:
            return content[:max_length] + "..."

        words = content.split()
        if len(words) <= 30:
            return self._highlight(content, query_terms)[:max_length]

        # Sliding window: find window with highest query term density
        window_size = 30
        best_score = -1
        best_start = 0

        query_set = set(t.lower() for t in query_terms)
        for i in range(len(words) - window_size + 1):
            window = words[i:i + window_size]
            score = sum(1 for w in window
                        if re.sub(r'[^a-z]', '', w.lower()) in query_set)
            if score > best_score:
                best_score = score
                best_start = i

        snippet = " ".join(words[best_start:best_start + window_size])
        return "..." + self._highlight(snippet, query_terms)[:max_length] + "..."

    def _highlight(self, text: str, query_terms: list) -> str:
        """Wrap query terms in **bold** markers."""
        for term in query_terms:
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            text = pattern.sub(f"**{term}**", text)
        return text


# ─────────────────────────────────────────────
# BFS Web Crawler
# ─────────────────────────────────────────────

class WebCrawler:
    """
    BFS web crawler with:
    - Bloom filter for URL deduplication
    - Robots.txt cache
    - Politeness constraints
    """

    def __init__(self, max_pages: int = 50):
        self.max_pages = max_pages
        self.bloom = BloomFilter(capacity=max_pages * 10)
        self.robots_cache: dict = {}     # domain -> allowed_paths
        self.crawled_urls: list = []
        self.url_frontier: deque = deque()

    def crawl(self, seed_urls: list, fake_web: dict) -> list:
        """
        BFS crawl starting from seed URLs.
        fake_web: dict of {url: (title, content, links)}
        Returns: list of WebDocument objects
        """
        for url in seed_urls:
            self.url_frontier.append(url)

        documents = []
        doc_id_counter = 1

        while self.url_frontier and len(documents) < self.max_pages:
            url = self.url_frontier.popleft()

            # Deduplication check (Bloom filter first, then exact check)
            if self.bloom.contains(url):
                continue
            if url in self.crawled_urls:
                continue

            # Politeness: check robots.txt (simplified)
            domain = url.split("/")[0]
            if not self._is_allowed(domain, url):
                continue

            # Fetch and parse page
            page_data = fake_web.get(url)
            if page_data is None:
                continue

            title, content, links = page_data
            doc = WebDocument(
                url=url,
                title=title,
                content=content,
                links=links,
                doc_id=doc_id_counter
            )
            documents.append(doc)
            doc_id_counter += 1

            # Mark as visited
            self.bloom.add(url)
            self.crawled_urls.append(url)

            # Add discovered links to frontier
            for link in links:
                if not self.bloom.contains(link):
                    self.url_frontier.append(link)

        return documents

    def _is_allowed(self, domain: str, url: str) -> bool:
        """Check robots.txt (simulated: block /admin/ and /private/ paths)."""
        blocked_patterns = ["/admin/", "/private/", "/login/"]
        return not any(pattern in url for pattern in blocked_patterns)


# ─────────────────────────────────────────────
# Main Search Engine
# ─────────────────────────────────────────────

class SearchEngine:
    """
    Orchestrates crawling, indexing, and ranked search.
    Uses BM25 + PageRank for ranking.
    """

    def __init__(self):
        self.documents: dict = {}           # doc_id -> WebDocument
        self.url_to_doc: dict = {}          # url -> WebDocument
        self.index = InvertedIndex()
        self.bm25 = BM25Scorer(k1=1.5, b=0.75)
        self.tfidf = TFIDFScorer()
        self.pagerank_computer = PageRankComputer()
        self.snippet_gen = SnippetGenerator()
        self.tokenizer = QueryTokenizer()
        self.crawler = WebCrawler(max_pages=50)
        self._pageranks: dict = {}

    def crawl_and_index(self, seed_urls: list, fake_web: dict):
        """Crawl the web and build the search index."""
        print(f"  [Crawler] Starting BFS from {len(seed_urls)} seed URLs...")
        crawled = self.crawler.crawl(seed_urls, fake_web)
        print(f"  [Crawler] Crawled {len(crawled)} pages. "
              f"Bloom filter fill: {self.crawler.bloom.fill_ratio:.4f}")

        print(f"  [Indexer] Building inverted index...")
        for doc in crawled:
            terms = self.index.add_document(doc)
            self.documents[doc.doc_id] = doc
            self.url_to_doc[doc.url] = doc

        print(f"  [Indexer] Indexed {len(self.documents)} documents. "
              f"Unique terms: {len(self.index._index)}")

        # Compute PageRank
        print(f"  [PageRank] Computing PageRank...")
        self._pageranks = self.pagerank_computer.compute(self.documents)
        for doc_id, pr in self._pageranks.items():
            self.documents[doc_id].page_rank = pr

    def add_document(self, url: str, title: str, content: str,
                     links: list = None) -> WebDocument:
        """Manually add a single document to the index."""
        doc_id = len(self.documents) + 1
        doc = WebDocument(
            url=url, title=title, content=content,
            links=links or [], doc_id=doc_id
        )
        self.index.add_document(doc)
        self.documents[doc_id] = doc
        self.url_to_doc[url] = doc
        return doc

    def search(self, query: str, top_k: int = 10,
               scorer: str = "bm25") -> list:
        """
        Search with BM25 or TF-IDF, re-rank with PageRank.
        Returns list of result dicts with url, title, snippet, score.
        """
        query_terms = self.tokenizer.tokenize(query)
        if not query_terms:
            return []

        # Get text relevance scores
        if scorer == "bm25":
            scored_docs = self.bm25.score(query_terms, self.index,
                                           top_k=top_k * 3)
        else:
            scored_docs = self.tfidf.score(query_terms, self.index,
                                            top_k=top_k * 3)

        # Re-rank: combined_score = text_score + alpha * log(pagerank)
        alpha = 0.3
        final_scores = []
        for doc_id, text_score in scored_docs:
            doc = self.documents.get(doc_id)
            if not doc:
                continue
            pr = self._pageranks.get(doc_id, 0.15)
            pr_boost = alpha * math.log(pr + 1e-10 + 1)
            final_score = text_score + pr_boost
            final_scores.append((doc_id, final_score, text_score, pr))

        final_scores.sort(key=lambda x: x[1], reverse=True)

        # Build results
        results = []
        for doc_id, final_score, text_score, pr in final_scores[:top_k]:
            doc = self.documents[doc_id]
            snippet = self.snippet_gen.generate(doc.content, query_terms)
            results.append({
                "url": doc.url,
                "title": doc.title,
                "snippet": snippet,
                "final_score": round(final_score, 4),
                "text_score": round(text_score, 4),
                "page_rank": round(pr, 4),
            })

        return results

    def get_snippet(self, url: str, query: str) -> str:
        doc = self.url_to_doc.get(url)
        if not doc:
            return ""
        terms = self.tokenizer.tokenize(query)
        return self.snippet_gen.generate(doc.content, terms)


# ─────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────

def build_fake_web() -> tuple:
    """Simulate a mini web of interconnected pages."""
    web = {
        "python.org/tutorial": (
            "Python Tutorial",
            "Learn Python programming language from scratch. Python is an interpreted "
            "high-level general-purpose programming language. Variables, loops, "
            "functions, classes, and modules. Python syntax is clean and readable.",
            ["python.org/advanced", "python.org/docs", "realpython.com/tutorial"]
        ),
        "python.org/advanced": (
            "Advanced Python Programming",
            "Advanced Python concepts including decorators, metaclasses, generators, "
            "context managers, and async/await. Python performance optimization "
            "and profiling techniques for experienced developers.",
            ["python.org/tutorial", "python.org/docs"]
        ),
        "python.org/docs": (
            "Python Documentation",
            "Official Python documentation. Standard library reference, language "
            "reference, and Python Enhancement Proposals. Comprehensive guide "
            "for Python programmers.",
            ["python.org/tutorial"]
        ),
        "realpython.com/tutorial": (
            "Real Python Tutorial: Learn Python Step by Step",
            "Real Python provides Python tutorials for developers of all skill levels. "
            "Learn Python basics, data structures, object-oriented programming, "
            "web development with Django and Flask.",
            ["python.org/tutorial", "realpython.com/advanced"]
        ),
        "realpython.com/advanced": (
            "Advanced Python Techniques",
            "Advanced Python topics: type hints, dataclasses, protocol classes, "
            "Python internals, CPython bytecode, memory management and garbage collection.",
            ["realpython.com/tutorial"]
        ),
        "systemdesign.io/hld": (
            "System Design Interview Guide",
            "How to ace system design interviews. High-level design, low-level design, "
            "distributed systems, databases, caching, load balancing, microservices. "
            "Design YouTube, Twitter, WhatsApp, Uber, and more.",
            ["systemdesign.io/cache", "systemdesign.io/database"]
        ),
        "systemdesign.io/cache": (
            "Caching in Distributed Systems",
            "Cache design patterns: cache-aside, write-through, write-back. "
            "Redis vs Memcached. Consistent hashing, LRU eviction, "
            "cache stampede prevention, distributed cache clusters.",
            ["systemdesign.io/hld"]
        ),
        "systemdesign.io/database": (
            "Database Design for Scale",
            "SQL vs NoSQL databases. PostgreSQL, MySQL, MongoDB, Cassandra, Redis. "
            "Database sharding, replication, ACID properties, CAP theorem, "
            "eventual consistency.",
            ["systemdesign.io/hld", "systemdesign.io/cache"]
        ),
        "fastapi.tiangolo.com/tutorial": (
            "FastAPI Tutorial - Modern Python Web Framework",
            "Build production-ready REST APIs with FastAPI. Automatic documentation "
            "with OpenAPI. Request validation with Pydantic. Async support. "
            "Dependency injection. Authentication and security.",
            ["python.org/docs"]
        ),
        "machinelearning.org/python": (
            "Machine Learning with Python",
            "Python for machine learning. NumPy, Pandas, Scikit-learn, TensorFlow, "
            "PyTorch. Supervised learning, unsupervised learning, neural networks, "
            "deep learning, model training and evaluation.",
            ["python.org/tutorial"]
        ),
    }
    seeds = ["python.org/tutorial", "systemdesign.io/hld"]
    return seeds, web


def run_demo():
    print("=" * 60)
    print("SEARCH ENGINE SYSTEM DESIGN DEMO")
    print("=" * 60)

    # Build search engine
    print("\n--- Phase 1: Crawl and Index ---")
    engine = SearchEngine()
    seeds, fake_web = build_fake_web()
    engine.crawl_and_index(seeds, fake_web)

    # Show PageRank results
    print("\n--- PageRank Scores ---")
    pr_ranked = sorted(engine._pageranks.items(), key=lambda x: x[1], reverse=True)
    for doc_id, pr in pr_ranked:
        doc = engine.documents[doc_id]
        bar = "#" * int(pr * 100)
        print(f"  [{pr:.4f}] {bar} {doc.url}")

    # Queries
    print("\n--- Search Results ---")
    queries = [
        "python tutorial beginner",
        "system design distributed cache",
        "advanced python decorators",
        "REST API web framework",
    ]
    for query in queries:
        print(f"\nQuery: '{query}'")
        results = engine.search(query, top_k=3, scorer="bm25")
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['final_score']:.4f}] {r['title']}")
            print(f"     URL: {r['url']}")
            print(f"     {r['snippet'][:120]}...")

    # BM25 vs TF-IDF comparison
    print("\n--- BM25 vs TF-IDF Scoring Comparison ---")
    query = "python programming"
    bm25_results = engine.search(query, top_k=5, scorer="bm25")
    tfidf_results = engine.search(query, top_k=5, scorer="tfidf")
    print(f"BM25 top results for '{query}':")
    for r in bm25_results[:3]:
        print(f"  [{r['final_score']:.4f}] {r['title']}")
    print(f"TF-IDF top results for '{query}':")
    for r in tfidf_results[:3]:
        print(f"  [{r['final_score']:.4f}] {r['title']}")

    # Bloom filter stats
    print("\n--- Bloom Filter Statistics ---")
    bf = engine.crawler.bloom
    print(f"  Capacity: {bf.capacity:,}")
    print(f"  Bit array size: {bf.m:,} bits ({bf.m // 8 // 1024} KB)")
    print(f"  Hash functions (k): {bf.k}")
    print(f"  Items added: {bf._count}")
    print(f"  Fill ratio: {bf.fill_ratio:.4f}")
    print(f"  URL 'python.org/tutorial' seen: {bf.contains('python.org/tutorial')}")
    print(f"  URL 'unknown.com/page' seen: {bf.contains('unknown.com/page')}")

    # Tokenizer demo
    print("\n--- Query Tokenizer (Stop Words + Stemming) ---")
    tokenizer = QueryTokenizer()
    texts = [
        "The best Python tutorial for beginners",
        "advanced machine learning algorithms",
        "how to design distributed systems at scale",
    ]
    for text in texts:
        tokens = tokenizer.tokenize(text)
        print(f"  Input:  '{text}'")
        print(f"  Output: {tokens}\n")

    # Scale estimates
    print("--- Scale Estimates ---")
    stats = {
        "10B web pages": "100TB extracted text, 30TB inverted index",
        "Index shards": "30 shards, 1TB each (by hash(term) % 30)",
        "Query fanout": "broadcast to 30 shards, merge top-10 from each",
        "PageRank convergence": "50-100 iterations on full web graph (Spark)",
        "Daily recrawl": "1B priority pages = 12K pages/second",
        "Query cache": "Top 20% queries cover 80% traffic (Zipf), cache in Redis",
    }
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
