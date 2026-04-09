"""
SEARCH DATABASES AND ELASTICSEARCH
=====================================

Problem Statement:
SQL LIKE '%keyword%' scans the entire table and can't rank by relevance.
Full-text search requires an inverted index, tokenization, stemming, and
relevance scoring. Elasticsearch is the industry standard for this.

How Inverted Index Works:
  Document 1: "The quick brown fox"
  Document 2: "The lazy brown dog"
  Document 3: "The quick clever fox"

  Inverted Index:
    "quick"  → [doc1, doc3]
    "brown"  → [doc1, doc2]
    "fox"    → [doc1, doc3]
    "dog"    → [doc2]
    "lazy"   → [doc2]

  Query "quick fox" → intersect(doc1,doc3) ∩ intersect(doc1,doc3) = [doc1, doc3]

Elasticsearch Concepts:
  Index    : like a database
  Document : JSON object with _id
  Field    : typed attribute (text, keyword, date, geo_point)
  Shard    : horizontal partition of index (distributes storage)
  Replica  : copy of shard (high availability)
  Mapping  : schema definition for the index

Relevance Scoring (TF-IDF / BM25):
  TF  (Term Frequency): how often the term appears in the document
  IDF (Inverse Document Frequency): how rare the term is across all documents
  Score = TF × log(N / df)   [N = total docs, df = docs containing term]
  BM25: improved TF-IDF with saturation and field normalization (ES default)

Use Cases: full-text search, log analytics, product search, autocomplete
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
import math
import time
import re
import random
from collections import defaultdict


class FieldType(Enum):
    TEXT     = "text"      # analyzed: tokenized, lowercased, stemmed
    KEYWORD  = "keyword"   # exact match: no analysis
    INTEGER  = "integer"
    FLOAT    = "float"
    DATE     = "date"
    BOOL     = "boolean"
    GEO_POINT= "geo_point"


@dataclass
class SearchHit:
    doc_id  : str
    score   : float
    source  : Dict[str, Any]

    def __str__(self):
        return f"[score={self.score:.3f}] {self.source.get('title', self.doc_id)}"


@dataclass
class SearchResult:
    total     : int
    max_score : float
    hits      : List[SearchHit]
    took_ms   : float

    def show(self, n: int = 5):
        print(f"  Found {self.total} results in {self.took_ms:.2f}ms "
              f"(max_score={self.max_score:.3f}):")
        for h in self.hits[:n]:
            print(f"    {h}")


# ─────────────────────────────────────────────
# TEXT ANALYZER
# ─────────────────────────────────────────────

class TextAnalyzer:
    """
    Simulates Elasticsearch text analysis pipeline:
    1. Tokenizer (split on whitespace/punctuation)
    2. Token filters (lowercase, stop words, stemming)
    """

    STOP_WORDS = {"the", "a", "an", "and", "or", "but", "in", "on", "at",
                   "to", "for", "of", "is", "are", "was", "be", "with"}

    # Simplified stemmer: remove common suffixes
    STEMS = {
        "running": "run", "runs": "run", "runner": "run",
        "searching": "search", "searches": "search",
        "products": "product", "items": "item",
        "quickly": "quick", "fastest": "fast",
        "databases": "database", "indexing": "index",
    }

    def analyze(self, text: str) -> List[str]:
        """Returns list of analyzed tokens."""
        # Lowercase
        text = text.lower()
        # Tokenize (split on non-alphanumeric)
        tokens = re.findall(r"[a-z0-9]+", text)
        # Remove stop words
        tokens = [t for t in tokens if t not in self.STOP_WORDS]
        # Apply stemming
        tokens = [self.STEMS.get(t, t) for t in tokens]
        return tokens


# ─────────────────────────────────────────────
# INVERTED INDEX
# ─────────────────────────────────────────────

class InvertedIndex:
    """
    Core data structure of a search engine.
    Maps: term → list of (doc_id, term_frequency) tuples.
    """

    def __init__(self, analyzer: TextAnalyzer):
        self.analyzer  = analyzer
        self._index    : Dict[str, Dict[str, int]] = defaultdict(dict)  # term→{doc_id:tf}
        self._doc_count = 0

    def add_document(self, doc_id: str, text: str):
        self._doc_count += 1
        tokens = self.analyzer.analyze(text)
        # Count term frequency
        tf: Dict[str, int] = defaultdict(int)
        for token in tokens:
            tf[token] += 1
        # Add to inverted index
        for term, freq in tf.items():
            self._index[term][doc_id] = freq

    def search(self, term: str) -> Dict[str, int]:
        """Returns {doc_id: term_frequency} for a term."""
        tokens = self.analyzer.analyze(term)
        result = {}
        for t in tokens:
            result.update(self._index.get(t, {}))
        return result

    def df(self, term: str) -> int:
        """Document frequency: how many docs contain this term."""
        token = self.analyzer.analyze(term)
        if not token:
            return 0
        return len(self._index.get(token[0], {}))

    @property
    def doc_count(self) -> int:
        return self._doc_count


# ─────────────────────────────────────────────
# BM25 SCORER
# ─────────────────────────────────────────────

class BM25Scorer:
    """
    BM25 relevance scoring (Elasticsearch default).
    Improvement over TF-IDF: TF saturation + field length normalization.
    k1=1.2: TF saturation (1.2 = moderate saturation)
    b=0.75: length normalization (0.75 = moderate)
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b  = b

    def score(self, tf: int, df: int, N: int, field_len: int,
               avg_field_len: float) -> float:
        if df == 0 or N == 0:
            return 0.0
        idf   = math.log((N - df + 0.5) / (df + 0.5) + 1)
        tf_norm = (tf * (self.k1 + 1)) / (
            tf + self.k1 * (1 - self.b + self.b * field_len / max(1, avg_field_len))
        )
        return idf * tf_norm


# ─────────────────────────────────────────────
# SEARCH ENGINE
# ─────────────────────────────────────────────

class SearchEngine:
    """
    Simplified Elasticsearch-like search engine.
    Supports: full-text search, filters, scoring, fuzzy matching.
    """

    def __init__(self, index_name: str):
        self.index_name   = index_name
        self.analyzer     = TextAnalyzer()
        self.inv_index    = InvertedIndex(self.analyzer)
        self.scorer       = BM25Scorer()
        self._docs        : Dict[str, Dict] = {}
        self._field_lengths: Dict[str, int] = {}   # doc_id → analyzed token count
        self.search_count = 0

    def index_document(self, doc_id: str, doc: Dict):
        self._docs[doc_id] = doc
        # Analyze all text fields
        text_fields = [str(v) for v in doc.values()
                        if isinstance(v, str)]
        full_text = " ".join(text_fields)
        self.inv_index.add_document(doc_id, full_text)
        self._field_lengths[doc_id] = len(self.analyzer.analyze(full_text))

    def search(self, query: str, filters: Dict = None,
                size: int = 10) -> SearchResult:
        self.search_count += 1
        start = time.perf_counter()

        # Analyze query
        terms = self.analyzer.analyze(query)
        if not terms:
            return SearchResult(0, 0.0, [], 0.0)

        # Find candidate documents (union of all term posting lists)
        candidates: Dict[str, float] = defaultdict(float)
        N        = self.inv_index.doc_count
        avg_len  = sum(self._field_lengths.values()) / max(1, N)

        for term in terms:
            df      = self.inv_index.df(term)
            postings = self.inv_index.search(term)
            for doc_id, tf in postings.items():
                field_len = self._field_lengths.get(doc_id, avg_len)
                score     = self.scorer.score(tf, df, N, field_len, avg_len)
                candidates[doc_id] += score

        # Apply filters
        if filters:
            candidates = {
                doc_id: score for doc_id, score in candidates.items()
                if all(self._docs.get(doc_id, {}).get(k) == v
                       for k, v in filters.items())
            }

        # Sort by score, build hits
        sorted_hits = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        hits = [
            SearchHit(did, score, self._docs[did])
            for did, score in sorted_hits[:size]
            if did in self._docs
        ]
        max_score = hits[0].score if hits else 0.0
        latency   = (time.perf_counter() - start) * 1000 + 2.0

        return SearchResult(len(sorted_hits), max_score, hits, round(latency, 2))

    def suggest_completions(self, prefix: str, size: int = 5) -> List[str]:
        """Simple prefix-based autocomplete."""
        prefix_lower = prefix.lower()
        suggestions  = set()
        for doc in self._docs.values():
            title = doc.get("title", "")
            if title.lower().startswith(prefix_lower):
                suggestions.add(title)
        return sorted(suggestions)[:size]

    def fuzzy_search(self, query: str, max_edit_distance: int = 1) -> List[str]:
        """Find terms within edit distance from query term."""
        analyzed = self.analyzer.analyze(query)
        if not analyzed:
            return []
        target_term = analyzed[0]
        matches = []
        for term in self.inv_index._index:
            if self._edit_distance(target_term, term) <= max_edit_distance:
                matches.append(term)
        return matches

    @staticmethod
    def _edit_distance(s1: str, s2: str) -> int:
        """Levenshtein distance."""
        if abs(len(s1) - len(s2)) > 2:
            return 999
        m, n = len(s1), len(s2)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, n + 1):
                temp = dp[j]
                if s1[i-1] == s2[j-1]:
                    dp[j] = prev
                else:
                    dp[j] = 1 + min(dp[j], dp[j-1], prev)
                prev = temp
        return dp[n]


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_search_db():
    print("=" * 65)
    print("SEARCH DATABASES AND ELASTICSEARCH")
    print("=" * 65)

    # ── Build Search Index ────────────────────
    print("\n[1] INDEXING DOCUMENTS")
    print("─" * 55)
    engine = SearchEngine("products")

    products = [
        {"id": "p01", "title": "Apple MacBook Pro 14 M3",       "category": "laptop",    "price": 1999, "brand": "Apple"},
        {"id": "p02", "title": "Dell XPS 15 Laptop Intel",       "category": "laptop",    "price": 1499, "brand": "Dell"},
        {"id": "p03", "title": "Apple iPhone 15 Pro Smartphone", "category": "phone",     "price": 1099, "brand": "Apple"},
        {"id": "p04", "title": "Samsung Galaxy S24 Android",     "category": "phone",     "price": 899,  "brand": "Samsung"},
        {"id": "p05", "title": "Sony WH-1000XM5 Headphones",     "category": "audio",     "price": 349,  "brand": "Sony"},
        {"id": "p06", "title": "Apple AirPods Pro Wireless",     "category": "audio",     "price": 249,  "brand": "Apple"},
        {"id": "p07", "title": "Logitech MX Master Mouse",       "category": "accessory", "price": 99,   "brand": "Logitech"},
        {"id": "p08", "title": "MacBook Air 15 M2 Apple",        "category": "laptop",    "price": 1299, "brand": "Apple"},
        {"id": "p09", "title": "Quick Brown Fox Book",            "category": "book",      "price": 15,   "brand": "Scholastic"},
        {"id": "p10", "title": "Running Shoes Nike Quick",       "category": "sports",    "price": 120,  "brand": "Nike"},
    ]
    for p in products:
        engine.index_document(p["id"], p)
    print(f"  Indexed {len(products)} documents")

    # Show inverted index snippet
    print(f"\n  Inverted index for 'apple':")
    postings = engine.inv_index.search("apple")
    for doc_id, tf in list(postings.items())[:5]:
        print(f"    doc={doc_id}  tf={tf}  title={engine._docs[doc_id]['title']}")

    # ── Full-Text Search ──────────────────────
    print("\n\n[2] FULL-TEXT SEARCH WITH BM25 SCORING")
    print("─" * 55)
    queries = ["apple laptop", "wireless headphones", "quick"]
    for q in queries:
        result = engine.search(q)
        print(f"\n  Query: '{q}'")
        result.show(3)

    # ── Filtered Search ───────────────────────
    print("\n\n[3] FILTERED SEARCH (brand=Apple)")
    print("─" * 55)
    result = engine.search("laptop", filters={"brand": "Apple"})
    print(f"  Query: 'laptop' + filter brand=Apple")
    result.show()

    # ── Text Analyzer ─────────────────────────
    print("\n\n[4] TEXT ANALYSIS PIPELINE")
    print("─" * 55)
    analyzer = TextAnalyzer()
    texts = [
        "The quick brown fox quickly runs",
        "Searching databases and indexing products",
        "Apple MacBook Pro Laptops 14-inch",
    ]
    for text in texts:
        tokens = analyzer.analyze(text)
        print(f"  Input  : {text}")
        print(f"  Tokens : {tokens}")
        print()

    # ── Autocomplete ──────────────────────────
    print("\n\n[5] AUTOCOMPLETE (prefix search)")
    print("─" * 55)
    for prefix in ["App", "Mac", "S"]:
        suggestions = engine.suggest_completions(prefix)
        print(f"  Prefix '{prefix}' → {suggestions}")

    # ── Fuzzy Search ──────────────────────────
    print("\n\n[6] FUZZY SEARCH (typo tolerance)")
    print("─" * 55)
    typos = [("laptp", "laptop"), ("appl", "apple"), ("headphon", "headphones")]
    for typo, correct in typos:
        fuzzy_results = engine.fuzzy_search(typo)
        print(f"  Query '{typo}' (intended '{correct}') → fuzzy matches: {fuzzy_results[:3]}")

    # ── ES Architecture ───────────────────────
    print("\n\n[7] ELASTICSEARCH CLUSTER ARCHITECTURE")
    print("─" * 55)
    print("  Index: products")
    print("  ├── Primary shards: 5 (distribute data horizontally)")
    print("  │    Each shard is a standalone Lucene index")
    print("  └── Replica shards: 1 per primary (HA + read scalability)")
    print()
    print("  Nodes:")
    print("  ├── Master node: cluster coordination, shard allocation")
    print("  ├── Data nodes: store shards, execute queries")
    print("  └── Coordinating node: route queries, gather results")

    # ── Use Cases ─────────────────────────────
    print("\n\n[8] ELASTICSEARCH USE CASES")
    print("─" * 55)
    use_cases = [
        ("Product search",     "Faceted search, relevance ranking"),
        ("Log analytics",      "ELK stack: log aggregation, alerting"),
        ("Autocomplete",       "Completion suggester, edge n-gram tokenizer"),
        ("Geo search",         "Find restaurants within 5km (geo_point)"),
        ("Security/SIEM",      "Log correlation, anomaly detection"),
        ("E-commerce filters", "Aggregations: category, price ranges, brands"),
    ]
    for use_case, details in use_cases:
        print(f"  • {use_case:<25} {details}")

    print(f"\n  Total searches performed: {engine.search_count}")


if __name__ == "__main__":
    demonstrate_search_db()
