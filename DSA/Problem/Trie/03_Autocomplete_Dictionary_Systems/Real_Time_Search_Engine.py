"""
Real Time Search Engine - Multiple Approaches
Difficulty: Hard

Implement a real-time search engine that provides instant search results as users type.
Features include:
1. Real-time indexing and search
2. Ranked search results with relevance scoring
3. Fuzzy search with typo tolerance
4. Auto-complete and suggestion system
5. Search analytics and trending queries
6. Performance optimization for high throughput
"""

from typing import List, Dict, Set, Tuple, Optional, Any, Iterator
from collections import defaultdict, deque, Counter
import heapq
import time
import threading
import re
import math
from dataclasses import dataclass, field
from enum import Enum
import json

class SearchResultType(Enum):
    EXACT_MATCH = "exact_match"
    PREFIX_MATCH = "prefix_match"
    FUZZY_MATCH = "fuzzy_match"
    SUGGESTION = "suggestion"

@dataclass
class SearchResult:
    """Search result with metadata"""
    id: str
    title: str
    content: str
    score: float
    result_type: SearchResultType
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

@dataclass
class Document:
    """Document for indexing"""
    id: str
    title: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

class TrieNode:
    """Enhanced trie node for search engine"""
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.documents = set()  # Document IDs containing this term
        self.frequency = 0
        self.term_frequency = defaultdict(int)  # doc_id -> frequency in that doc

class RealTimeSearchEngine1:
    """
    Approach 1: Trie-based Real-time Search with TF-IDF Scoring
    
    Use trie for indexing with TF-IDF relevance scoring.
    """
    
    def __init__(self):
        self.trie = TrieNode()
        self.documents = {}  # doc_id -> Document
        self.document_count = 0
        self.query_log = deque(maxlen=10000)  # Recent queries
        self.trending_queries = Counter()
        self.lock = threading.RLock()
        
        # Search configuration
        self.max_results = 20
        self.fuzzy_threshold = 2  # Max edit distance for fuzzy search
    
    def index_document(self, document: Document) -> None:
        """
        Index a document in real-time.
        
        Time: O(sum of term lengths in document)
        Space: O(sum of term lengths in document)
        """
        with self.lock:
            # Store document
            self.documents[document.id] = document
            if document.id not in [doc.id for doc in self.documents.values()]:
                self.document_count += 1
            
            # Extract and index terms
            terms = self._extract_terms(document.title + " " + document.content)
            
            for term in terms:
                self._add_term_to_trie(term, document.id)
    
    def _extract_terms(self, text: str) -> List[str]:
        """Extract searchable terms from text"""
        # Simple tokenization - in production, use advanced NLP
        text = text.lower()
        # Remove punctuation and split
        terms = re.findall(r'\b[a-zA-Z]+\b', text)
        return [term for term in terms if len(term) >= 2]  # Filter short terms
    
    def _add_term_to_trie(self, term: str, doc_id: str) -> None:
        """Add term to trie with document reference"""
        node = self.trie
        
        for char in term:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        node.is_end = True
        node.documents.add(doc_id)
        node.frequency += 1
        node.term_frequency[doc_id] += 1
    
    def search(self, query: str, max_results: int = None) -> List[SearchResult]:
        """
        Perform real-time search.
        
        Time: O(|query| + matching_documents * log(max_results))
        Space: O(matching_documents)
        """
        with self.lock:
            if max_results is None:
                max_results = self.max_results
            
            query = query.strip().lower()
            if not query:
                return []
            
            # Log query for analytics
            self.query_log.append((query, time.time()))
            self.trending_queries[query] += 1
            
            # Get search results from different approaches
            exact_results = self._exact_search(query)
            prefix_results = self._prefix_search(query)
            fuzzy_results = self._fuzzy_search(query)
            
            # Combine and rank results
            all_results = {}  # doc_id -> SearchResult
            
            # Add exact matches (highest priority)
            for doc_id, score in exact_results:
                if doc_id in self.documents:
                    doc = self.documents[doc_id]
                    all_results[doc_id] = SearchResult(
                        id=doc_id,
                        title=doc.title,
                        content=doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                        score=score * 3.0,  # Boost exact matches
                        result_type=SearchResultType.EXACT_MATCH,
                        metadata={'relevance': 'exact'}
                    )
            
            # Add prefix matches
            for doc_id, score in prefix_results:
                if doc_id in self.documents and doc_id not in all_results:
                    doc = self.documents[doc_id]
                    all_results[doc_id] = SearchResult(
                        id=doc_id,
                        title=doc.title,
                        content=doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                        score=score * 2.0,  # Boost prefix matches
                        result_type=SearchResultType.PREFIX_MATCH,
                        metadata={'relevance': 'prefix'}
                    )
            
            # Add fuzzy matches
            for doc_id, score in fuzzy_results:
                if doc_id in self.documents and doc_id not in all_results:
                    doc = self.documents[doc_id]
                    all_results[doc_id] = SearchResult(
                        id=doc_id,
                        title=doc.title,
                        content=doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                        score=score,
                        result_type=SearchResultType.FUZZY_MATCH,
                        metadata={'relevance': 'fuzzy'}
                    )
            
            # Sort by score and return top results
            sorted_results = sorted(all_results.values(), key=lambda x: x.score, reverse=True)
            return sorted_results[:max_results]
    
    def _exact_search(self, query: str) -> List[Tuple[str, float]]:
        """Search for exact term matches"""
        terms = self._extract_terms(query)
        if not terms:
            return []
        
        # Find documents containing all terms
        term_docs = []
        for term in terms:
            docs = self._get_documents_for_term(term)
            if not docs:
                return []  # If any term is missing, no exact matches
            term_docs.append(docs)
        
        # Intersect document sets
        common_docs = set.intersection(*[set(docs.keys()) for docs in term_docs])
        
        # Calculate TF-IDF scores
        scored_results = []
        for doc_id in common_docs:
            score = self._calculate_tfidf_score(doc_id, terms)
            scored_results.append((doc_id, score))
        
        return scored_results
    
    def _prefix_search(self, query: str) -> List[Tuple[str, float]]:
        """Search for prefix matches"""
        if len(query) < 2:
            return []
        
        # Find terms that start with query
        matching_terms = self._find_prefix_terms(query)
        
        # Collect documents for matching terms
        doc_scores = defaultdict(float)
        
        for term in matching_terms:
            docs = self._get_documents_for_term(term)
            for doc_id, tf in docs.items():
                # Score based on term frequency and prefix match quality
                prefix_score = len(query) / len(term)  # Longer prefixes score higher
                doc_scores[doc_id] += tf * prefix_score
        
        return list(doc_scores.items())
    
    def _fuzzy_search(self, query: str) -> List[Tuple[str, float]]:
        """Search with fuzzy matching for typos"""
        if len(query) < 3:
            return []
        
        # Find terms within edit distance threshold
        fuzzy_terms = self._find_fuzzy_terms(query)
        
        # Collect documents for fuzzy terms
        doc_scores = defaultdict(float)
        
        for term, edit_distance in fuzzy_terms:
            docs = self._get_documents_for_term(term)
            for doc_id, tf in docs.items():
                # Score inversely proportional to edit distance
                fuzzy_score = 1.0 / (1 + edit_distance)
                doc_scores[doc_id] += tf * fuzzy_score
        
        return list(doc_scores.items())
    
    def _get_documents_for_term(self, term: str) -> Dict[str, int]:
        """Get documents containing term with term frequencies"""
        node = self.trie
        for char in term:
            if char not in node.children:
                return {}
            node = node.children[char]
        
        if node.is_end:
            return dict(node.term_frequency)
        return {}
    
    def _find_prefix_terms(self, prefix: str) -> List[str]:
        """Find all terms that start with prefix"""
        node = self.trie
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        
        # Collect all terms from this node
        terms = []
        self._collect_terms(node, prefix, terms)
        return terms
    
    def _collect_terms(self, node: TrieNode, prefix: str, terms: List[str]) -> None:
        """Collect all terms from trie node"""
        if node.is_end:
            terms.append(prefix)
        
        for char, child in node.children.items():
            self._collect_terms(child, prefix + char, terms)
    
    def _find_fuzzy_terms(self, query: str) -> List[Tuple[str, int]]:
        """Find terms within edit distance threshold"""
        fuzzy_terms = []
        
        # Simple approach: check all terms (in production, use more efficient method)
        all_terms = []
        self._collect_terms(self.trie, "", all_terms)
        
        for term in all_terms:
            if abs(len(term) - len(query)) <= self.fuzzy_threshold:
                edit_distance = self._edit_distance(query, term)
                if edit_distance <= self.fuzzy_threshold:
                    fuzzy_terms.append((term, edit_distance))
        
        return fuzzy_terms
    
    def _edit_distance(self, s1: str, s2: str) -> int:
        """Calculate edit distance between two strings"""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        return dp[m][n]
    
    def _calculate_tfidf_score(self, doc_id: str, terms: List[str]) -> float:
        """Calculate TF-IDF score for document"""
        score = 0.0
        
        for term in terms:
            # Term frequency in document
            tf = self._get_documents_for_term(term).get(doc_id, 0)
            
            # Document frequency (number of documents containing term)
            df = len(self._get_documents_for_term(term))
            
            # Inverse document frequency
            idf = math.log(self.document_count / (df + 1))
            
            # TF-IDF score
            score += tf * idf
        
        return score
    
    def get_suggestions(self, prefix: str, max_suggestions: int = 5) -> List[str]:
        """
        Get autocomplete suggestions.
        
        Time: O(|prefix| + suggestions)
        Space: O(suggestions)
        """
        if len(prefix) < 2:
            return []
        
        # Get recent trending queries that match prefix
        suggestions = []
        
        # From trending queries
        for query, count in self.trending_queries.most_common(100):
            if query.startswith(prefix.lower()) and query != prefix.lower():
                suggestions.append((query, count))
        
        # From indexed terms
        prefix_terms = self._find_prefix_terms(prefix.lower())
        for term in prefix_terms[:20]:  # Limit to avoid too many suggestions
            if term not in [s[0] for s in suggestions]:
                # Use term frequency as score
                docs = self._get_documents_for_term(term)
                total_freq = sum(docs.values())
                suggestions.append((term, total_freq))
        
        # Sort by frequency/popularity
        suggestions.sort(key=lambda x: x[1], reverse=True)
        
        return [suggestion[0] for suggestion in suggestions[:max_suggestions]]
    
    def get_trending_queries(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Get trending search queries"""
        return self.trending_queries.most_common(limit)
    
    def remove_document(self, doc_id: str) -> bool:
        """
        Remove document from index.
        
        Time: O(terms_in_document * average_term_length)
        Space: O(1)
        """
        with self.lock:
            if doc_id not in self.documents:
                return False
            
            document = self.documents[doc_id]
            terms = self._extract_terms(document.title + " " + document.content)
            
            # Remove from trie
            for term in terms:
                self._remove_term_from_trie(term, doc_id)
            
            # Remove document
            del self.documents[doc_id]
            self.document_count -= 1
            
            return True
    
    def _remove_term_from_trie(self, term: str, doc_id: str) -> None:
        """Remove document reference from term in trie"""
        node = self.trie
        for char in term:
            if char not in node.children:
                return
            node = node.children[char]
        
        if node.is_end and doc_id in node.documents:
            node.documents.remove(doc_id)
            if doc_id in node.term_frequency:
                del node.term_frequency[doc_id]
            node.frequency = sum(node.term_frequency.values())


class RealTimeSearchEngine2:
    """
    Approach 2: Inverted Index with Real-time Updates
    
    Use inverted index for efficient document retrieval.
    """
    
    def __init__(self):
        self.inverted_index = defaultdict(set)  # term -> set of doc_ids
        self.term_frequencies = defaultdict(lambda: defaultdict(int))  # term -> doc_id -> frequency
        self.documents = {}
        self.document_lengths = {}  # doc_id -> number of terms
        self.lock = threading.RLock()
    
    def index_document(self, document: Document) -> None:
        """
        Index document using inverted index.
        
        Time: O(terms_in_document)
        Space: O(terms_in_document)
        """
        with self.lock:
            doc_id = document.id
            self.documents[doc_id] = document
            
            # Extract terms
            terms = self._extract_terms(document.title + " " + document.content)
            self.document_lengths[doc_id] = len(terms)
            
            # Update inverted index
            term_counts = Counter(terms)
            
            for term, count in term_counts.items():
                self.inverted_index[term].add(doc_id)
                self.term_frequencies[term][doc_id] = count
    
    def _extract_terms(self, text: str) -> List[str]:
        """Extract terms from text"""
        text = text.lower()
        return re.findall(r'\b[a-zA-Z]+\b', text)
    
    def search(self, query: str, max_results: int = 20) -> List[SearchResult]:
        """
        Search using inverted index.
        
        Time: O(query_terms + matching_docs * log(max_results))
        Space: O(matching_docs)
        """
        with self.lock:
            query_terms = self._extract_terms(query)
            if not query_terms:
                return []
            
            # Get candidate documents
            candidate_docs = None
            
            for term in query_terms:
                term_docs = self.inverted_index.get(term, set())
                if candidate_docs is None:
                    candidate_docs = term_docs.copy()
                else:
                    candidate_docs &= term_docs  # Intersection for AND search
            
            if not candidate_docs:
                return []
            
            # Score documents
            scored_docs = []
            
            for doc_id in candidate_docs:
                score = self._calculate_bm25_score(doc_id, query_terms)
                scored_docs.append((doc_id, score))
            
            # Sort and return results
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            results = []
            for doc_id, score in scored_docs[:max_results]:
                doc = self.documents[doc_id]
                results.append(SearchResult(
                    id=doc_id,
                    title=doc.title,
                    content=doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                    score=score,
                    result_type=SearchResultType.EXACT_MATCH
                ))
            
            return results
    
    def _calculate_bm25_score(self, doc_id: str, query_terms: List[str]) -> float:
        """Calculate BM25 relevance score"""
        # BM25 parameters
        k1, b = 1.2, 0.75
        
        # Average document length
        avg_doc_length = sum(self.document_lengths.values()) / len(self.document_lengths)
        doc_length = self.document_lengths[doc_id]
        
        score = 0.0
        
        for term in query_terms:
            # Term frequency in document
            tf = self.term_frequencies[term].get(doc_id, 0)
            
            # Document frequency
            df = len(self.inverted_index[term])
            
            # IDF component
            idf = math.log((len(self.documents) - df + 0.5) / (df + 0.5))
            
            # BM25 formula
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_length / avg_doc_length))
            
            score += idf * (numerator / denominator)
        
        return score


class RealTimeSearchEngine3:
    """
    Approach 3: Distributed Search with Sharding
    
    Distribute index across multiple shards for scalability.
    """
    
    def __init__(self, num_shards: int = 4):
        self.num_shards = num_shards
        self.shards = [RealTimeSearchEngine1() for _ in range(num_shards)]
        self.shard_lock = threading.RLock()
    
    def _get_shard(self, doc_id: str) -> int:
        """Get shard index for document"""
        return hash(doc_id) % self.num_shards
    
    def index_document(self, document: Document) -> None:
        """
        Index document in appropriate shard.
        
        Time: O(terms_in_document)
        Space: O(terms_in_document)
        """
        shard_index = self._get_shard(document.id)
        self.shards[shard_index].index_document(document)
    
    def search(self, query: str, max_results: int = 20) -> List[SearchResult]:
        """
        Search across all shards and merge results.
        
        Time: O(shards * search_time + total_results * log(max_results))
        Space: O(total_results)
        """
        all_results = []
        
        # Search each shard in parallel (simplified - could use actual threading)
        for shard in self.shards:
            shard_results = shard.search(query, max_results)
            all_results.extend(shard_results)
        
        # Merge and sort results
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        return all_results[:max_results]
    
    def get_suggestions(self, prefix: str, max_suggestions: int = 5) -> List[str]:
        """Get suggestions from all shards"""
        all_suggestions = []
        
        for shard in self.shards:
            suggestions = shard.get_suggestions(prefix, max_suggestions)
            all_suggestions.extend(suggestions)
        
        # Remove duplicates and return top suggestions
        unique_suggestions = list(set(all_suggestions))
        return unique_suggestions[:max_suggestions]


class RealTimeSearchEngine4:
    """
    Approach 4: Memory-Optimized with Caching
    
    Add caching and memory optimization for high performance.
    """
    
    def __init__(self, cache_size: int = 1000):
        self.core_engine = RealTimeSearchEngine1()
        self.search_cache = {}
        self.suggestion_cache = {}
        self.cache_size = cache_size
        self.cache_lock = threading.RLock()
        
        # Performance metrics
        self.cache_hits = 0
        self.cache_misses = 0
    
    def index_document(self, document: Document) -> None:
        """Index document and invalidate relevant caches"""
        self.core_engine.index_document(document)
        
        # Invalidate caches that might be affected
        with self.cache_lock:
            self._invalidate_caches()
    
    def search(self, query: str, max_results: int = 20) -> List[SearchResult]:
        """
        Search with caching.
        
        Time: O(1) for cache hit, O(search_time) for cache miss
        Space: O(cache_size)
        """
        cache_key = (query.lower(), max_results)
        
        with self.cache_lock:
            # Check cache first
            if cache_key in self.search_cache:
                self.cache_hits += 1
                return self.search_cache[cache_key]
        
        # Cache miss - perform search
        self.cache_misses += 1
        results = self.core_engine.search(query, max_results)
        
        # Update cache
        with self.cache_lock:
            if len(self.search_cache) >= self.cache_size:
                # Remove oldest entry (simplified LRU)
                oldest_key = next(iter(self.search_cache))
                del self.search_cache[oldest_key]
            
            self.search_cache[cache_key] = results
        
        return results
    
    def get_suggestions(self, prefix: str, max_suggestions: int = 5) -> List[str]:
        """Get suggestions with caching"""
        cache_key = (prefix.lower(), max_suggestions)
        
        with self.cache_lock:
            if cache_key in self.suggestion_cache:
                return self.suggestion_cache[cache_key]
        
        # Cache miss
        suggestions = self.core_engine.get_suggestions(prefix, max_suggestions)
        
        # Update cache
        with self.cache_lock:
            if len(self.suggestion_cache) >= self.cache_size:
                oldest_key = next(iter(self.suggestion_cache))
                del self.suggestion_cache[oldest_key]
            
            self.suggestion_cache[cache_key] = suggestions
        
        return suggestions
    
    def _invalidate_caches(self) -> None:
        """Invalidate caches when index changes"""
        self.search_cache.clear()
        self.suggestion_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'search_cache_size': len(self.search_cache),
            'suggestion_cache_size': len(self.suggestion_cache)
        }


def test_search_engines():
    """Test all search engine implementations"""
    print("=== Testing Search Engine Implementations ===")
    
    # Create test documents
    documents = [
        Document("1", "Python Programming Guide", "Learn Python programming with examples and tutorials"),
        Document("2", "Machine Learning Basics", "Introduction to machine learning algorithms and concepts"),
        Document("3", "Data Science Tutorial", "Data analysis and visualization using Python"),
        Document("4", "Web Development", "Building web applications with modern frameworks"),
        Document("5", "Algorithm Design", "Advanced algorithms and data structures")
    ]
    
    engines = [
        ("Trie-based TF-IDF", RealTimeSearchEngine1()),
        ("Inverted Index BM25", RealTimeSearchEngine2()),
        ("Distributed Shards", RealTimeSearchEngine3()),
        ("Memory-Optimized", RealTimeSearchEngine4()),
    ]
    
    # Test each engine
    for name, engine in engines:
        print(f"\n{name}:")
        
        # Index documents
        for doc in documents:
            engine.index_document(doc)
        
        # Test searches
        test_queries = ["python", "machine learning", "data", "web"]
        
        for query in test_queries:
            results = engine.search(query, 3)
            print(f"  '{query}' -> {len(results)} results")
            
            for i, result in enumerate(results[:2]):  # Show top 2
                print(f"    {i+1}. {result.title} (score: {result.score:.2f})")


def demonstrate_real_time_features():
    """Demonstrate real-time search features"""
    print("\n=== Real-time Features Demo ===")
    
    engine = RealTimeSearchEngine1()
    
    print("1. Real-time Indexing:")
    
    # Add documents one by one
    docs = [
        Document("1", "Python Tutorial", "Learn Python programming"),
        Document("2", "Python Examples", "Python code examples"),
        Document("3", "Java Programming", "Java development guide")
    ]
    
    for doc in docs:
        engine.index_document(doc)
        print(f"   Indexed: '{doc.title}'")
        
        # Search immediately after indexing
        results = engine.search("python", 5)
        print(f"   Search 'python': {len(results)} results")
    
    print(f"\n2. Auto-complete Suggestions:")
    
    # Test suggestions
    prefixes = ["py", "pyt", "pyth", "pytho"]
    
    for prefix in prefixes:
        suggestions = engine.get_suggestions(prefix)
        print(f"   '{prefix}' -> {suggestions}")
    
    print(f"\n3. Trending Queries:")
    
    # Simulate searches to build trending data
    search_queries = ["python", "python", "java", "python", "machine learning", "python"]
    
    for query in search_queries:
        engine.search(query)
    
    trending = engine.get_trending_queries(5)
    print(f"   Trending: {trending}")
    
    print(f"\n4. Document Removal:")
    
    # Remove a document
    removed = engine.remove_document("2")
    print(f"   Removed document '2': {removed}")
    
    # Search again
    results = engine.search("python", 5)
    print(f"   Search 'python' after removal: {len(results)} results")


def demonstrate_fuzzy_search():
    """Demonstrate fuzzy search capabilities"""
    print("\n=== Fuzzy Search Demo ===")
    
    engine = RealTimeSearchEngine1()
    
    # Add documents
    docs = [
        Document("1", "Programming Tutorial", "Learn programming concepts"),
        Document("2", "Algorithm Design", "Algorithm design patterns"),
        Document("3", "Machine Learning", "ML algorithms and techniques")
    ]
    
    for doc in docs:
        engine.index_document(doc)
    
    # Test fuzzy searches (with typos)
    fuzzy_queries = [
        ("programing", "programming"),  # Missing 'm'
        ("algoritm", "algorithm"),      # Missing 'h'
        ("machne", "machine"),          # Missing 'i'
        ("learing", "learning"),        # Missing 'n'
    ]
    
    print("Fuzzy search results (with typos):")
    
    for typo_query, correct_term in fuzzy_queries:
        results = engine.search(typo_query, 3)
        print(f"\n   Query: '{typo_query}' (intended: '{correct_term}')")
        
        if results:
            print(f"   Found {len(results)} results:")
            for result in results:
                print(f"     - {result.title} (type: {result.result_type.value})")
        else:
            print(f"   No results found")


def benchmark_search_engines():
    """Benchmark search engine performance"""
    print("\n=== Benchmarking Search Engines ===")
    
    import random
    import string
    
    # Generate test data
    def generate_document(doc_id: int) -> Document:
        title_words = random.choices(['python', 'java', 'machine', 'learning', 'data', 'science', 'web', 'development'], k=3)
        content_words = random.choices(string.ascii_lowercase, k=50)
        
        return Document(
            id=str(doc_id),
            title=' '.join(title_words),
            content=' '.join(''.join(random.choices(string.ascii_lowercase, k=5)) for _ in range(20))
        )
    
    test_documents = [generate_document(i) for i in range(1000)]
    test_queries = ['python', 'java', 'machine', 'learning', 'data']
    
    engines = [
        ("Trie-based", RealTimeSearchEngine1()),
        ("Inverted Index", RealTimeSearchEngine2()),
        ("Memory-Optimized", RealTimeSearchEngine4()),
    ]
    
    for name, engine in engines:
        print(f"\n{name}:")
        
        # Measure indexing time
        start_time = time.time()
        for doc in test_documents:
            engine.index_document(doc)
        indexing_time = time.time() - start_time
        
        # Measure search time
        start_time = time.time()
        for query in test_queries * 20:  # 100 searches total
            engine.search(query, 10)
        search_time = time.time() - start_time
        
        print(f"  Index {len(test_documents)} docs: {indexing_time:.2f}s")
        print(f"  Search {len(test_queries) * 20} queries: {search_time:.2f}s")
        print(f"  Avg search time: {search_time / (len(test_queries) * 20) * 1000:.2f}ms")
        
        # Show cache stats if available
        if hasattr(engine, 'get_cache_stats'):
            stats = engine.get_cache_stats()
            print(f"  Cache hit rate: {stats['hit_rate']:.2%}")


def demonstrate_distributed_search():
    """Demonstrate distributed search capabilities"""
    print("\n=== Distributed Search Demo ===")
    
    distributed_engine = RealTimeSearchEngine3(num_shards=3)
    
    # Add documents (they'll be distributed across shards)
    docs = [
        Document("1", "Python Programming", "Python tutorial and examples"),
        Document("2", "Java Development", "Java programming guide"),
        Document("3", "Machine Learning", "ML algorithms and techniques"),
        Document("4", "Data Science", "Data analysis with Python"),
        Document("5", "Web Development", "Building web applications"),
        Document("6", "Algorithm Design", "Advanced algorithms")
    ]
    
    print("Distributing documents across shards:")
    
    for doc in docs:
        shard_index = distributed_engine._get_shard(doc.id)
        distributed_engine.index_document(doc)
        print(f"  Doc '{doc.id}' -> Shard {shard_index}: '{doc.title}'")
    
    print(f"\nSearching across all shards:")
    
    test_queries = ["python", "java", "machine"]
    
    for query in test_queries:
        results = distributed_engine.search(query, 5)
        print(f"  '{query}' -> {len(results)} results from all shards")
        
        for result in results[:2]:
            print(f"    - {result.title} (score: {result.score:.2f})")


if __name__ == "__main__":
    test_search_engines()
    demonstrate_real_time_features()
    demonstrate_fuzzy_search()
    benchmark_search_engines()
    demonstrate_distributed_search()

"""
Real Time Search Engine demonstrates comprehensive search engine implementations:

1. Trie-based with TF-IDF - Use trie for indexing with TF-IDF relevance scoring
2. Inverted Index with BM25 - Classic inverted index with BM25 ranking algorithm
3. Distributed with Sharding - Scale across multiple shards for high throughput
4. Memory-Optimized with Caching - Add caching and memory optimization

Key features implemented:
- Real-time document indexing and removal
- Multiple search modes (exact, prefix, fuzzy)
- Relevance scoring (TF-IDF, BM25)
- Auto-complete suggestions
- Search analytics and trending queries
- Performance optimization and caching
- Distributed architecture support

Each implementation showcases different aspects of modern search engines
from basic functionality to production-ready optimizations.
"""

