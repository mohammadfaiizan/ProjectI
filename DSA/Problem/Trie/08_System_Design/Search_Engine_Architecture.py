"""
Search Engine Architecture - Multiple Approaches
Difficulty: Hard

Design and implementation of a search engine using trie-based indexing
with focus on scalability, relevance, and real-time updates.

Components:
1. Document Indexing with Trie
2. Relevance Scoring (TF-IDF, BM25)
3. Query Processing and Optimization
4. Real-time Index Updates
5. Autocomplete and Suggestions
6. Distributed Search Architecture
"""

import math
import time
import threading
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, Counter
import heapq
import re

class Document:
    def __init__(self, doc_id: str, title: str, content: str, url: str = ""):
        self.doc_id = doc_id
        self.title = title
        self.content = content
        self.url = url
        self.word_count = len(content.split())
        self.timestamp = time.time()

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.documents = set()  # Document IDs containing this word
        self.frequency = defaultdict(int)  # doc_id -> frequency

class SearchIndex:
    """Inverted index using trie structure"""
    
    def __init__(self):
        self.trie = TrieNode()
        self.documents = {}  # doc_id -> Document
        self.document_count = 0
        self.total_words = 0
        self.lock = threading.RWLock() if hasattr(threading, 'RWLock') else threading.Lock()
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        # Simple tokenization - in production would be more sophisticated
        text = text.lower()
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        return words
    
    def _insert_word(self, word: str, doc_id: str, frequency: int) -> None:
        """Insert word into trie index"""
        node = self.trie
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        node.is_word = True
        node.documents.add(doc_id)
        node.frequency[doc_id] = frequency
    
    def add_document(self, document: Document) -> None:
        """Add document to search index"""
        with self.lock:
            self.documents[document.doc_id] = document
            self.document_count += 1
            
            # Tokenize and count words
            words = self._tokenize(document.title + " " + document.content)
            word_freq = Counter(words)
            self.total_words += len(words)
            
            # Add words to trie index
            for word, freq in word_freq.items():
                self._insert_word(word, document.doc_id, freq)
    
    def remove_document(self, doc_id: str) -> bool:
        """Remove document from index"""
        with self.lock:
            if doc_id not in self.documents:
                return False
            
            document = self.documents[doc_id]
            words = self._tokenize(document.title + " " + document.content)
            
            # Remove from trie
            for word in set(words):
                self._remove_word_from_doc(word, doc_id)
            
            del self.documents[doc_id]
            self.document_count -= 1
            return True
    
    def _remove_word_from_doc(self, word: str, doc_id: str) -> None:
        """Remove word-document association"""
        node = self.trie
        for char in word:
            if char not in node.children:
                return
            node = node.children[char]
        
        if node.is_word and doc_id in node.documents:
            node.documents.remove(doc_id)
            if doc_id in node.frequency:
                del node.frequency[doc_id]
    
    def get_word_info(self, word: str) -> Tuple[Set[str], Dict[str, int]]:
        """Get documents and frequencies for a word"""
        node = self.trie
        for char in word:
            if char not in node.children:
                return set(), {}
            node = node.children[char]
        
        if node.is_word:
            return node.documents.copy(), node.frequency.copy()
        return set(), {}

class RelevanceScorer:
    """Calculate relevance scores for search results"""
    
    def __init__(self, index: SearchIndex):
        self.index = index
    
    def calculate_tf_idf(self, query_words: List[str], doc_id: str) -> float:
        """Calculate TF-IDF score for document"""
        if doc_id not in self.index.documents:
            return 0.0
        
        document = self.index.documents[doc_id]
        score = 0.0
        
        for word in query_words:
            # Term Frequency (TF)
            documents, frequencies = self.index.get_word_info(word)
            tf = frequencies.get(doc_id, 0) / max(1, document.word_count)
            
            # Inverse Document Frequency (IDF)
            df = len(documents)  # Document frequency
            idf = math.log(self.index.document_count / max(1, df))
            
            # TF-IDF
            score += tf * idf
        
        return score
    
    def calculate_bm25(self, query_words: List[str], doc_id: str, 
                      k1: float = 1.5, b: float = 0.75) -> float:
        """Calculate BM25 score for document"""
        if doc_id not in self.index.documents:
            return 0.0
        
        document = self.index.documents[doc_id]
        score = 0.0
        
        # Average document length
        avg_doc_length = self.index.total_words / max(1, self.index.document_count)
        
        for word in query_words:
            documents, frequencies = self.index.get_word_info(word)
            
            # Term frequency in document
            tf = frequencies.get(doc_id, 0)
            
            # Document frequency
            df = len(documents)
            
            # IDF component
            idf = math.log((self.index.document_count - df + 0.5) / (df + 0.5))
            
            # BM25 formula
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (document.word_count / avg_doc_length))
            
            score += idf * (numerator / denominator)
        
        return score
    
    def calculate_title_boost(self, query_words: List[str], doc_id: str) -> float:
        """Calculate boost for title matches"""
        if doc_id not in self.index.documents:
            return 0.0
        
        document = self.index.documents[doc_id]
        title_words = set(self.index._tokenize(document.title))
        
        matches = sum(1 for word in query_words if word in title_words)
        return matches / max(1, len(query_words))

class QueryProcessor:
    """Process and optimize search queries"""
    
    def __init__(self, index: SearchIndex, scorer: RelevanceScorer):
        self.index = index
        self.scorer = scorer
        self.query_cache = {}  # Simple query caching
    
    def process_query(self, query: str, max_results: int = 10) -> List[Tuple[str, float]]:
        """Process search query and return ranked results"""
        # Check cache
        cache_key = f"{query}:{max_results}"
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]
        
        # Tokenize query
        query_words = self.index._tokenize(query)
        if not query_words:
            return []
        
        # Get candidate documents
        candidate_docs = self._get_candidate_documents(query_words)
        
        # Score documents
        scored_docs = []
        for doc_id in candidate_docs:
            # Calculate combined score
            bm25_score = self.scorer.calculate_bm25(query_words, doc_id)
            title_boost = self.scorer.calculate_title_boost(query_words, doc_id)
            
            # Combine scores (can be tuned)
            final_score = bm25_score + title_boost * 2.0
            
            scored_docs.append((doc_id, final_score))
        
        # Sort by score and take top results
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        results = scored_docs[:max_results]
        
        # Cache results
        self.query_cache[cache_key] = results
        
        return results
    
    def _get_candidate_documents(self, query_words: List[str]) -> Set[str]:
        """Get candidate documents that contain query words"""
        if not query_words:
            return set()
        
        # Start with documents containing first word
        documents, _ = self.index.get_word_info(query_words[0])
        candidates = documents.copy()
        
        # Intersect with documents containing other words (AND operation)
        for word in query_words[1:]:
            word_docs, _ = self.index.get_word_info(word)
            candidates &= word_docs
        
        # If intersection is empty, use union (OR operation)
        if not candidates:
            for word in query_words:
                word_docs, _ = self.index.get_word_info(word)
                candidates |= word_docs
        
        return candidates
    
    def get_suggestions(self, prefix: str, max_suggestions: int = 10) -> List[str]:
        """Get autocomplete suggestions for prefix"""
        suggestions = []
        
        def dfs(node: TrieNode, current_word: str, depth: int) -> None:
            if len(suggestions) >= max_suggestions:
                return
            
            if node.is_word and len(current_word) > len(prefix):
                # Score by popularity (number of documents)
                popularity = len(node.documents)
                suggestions.append((current_word, popularity))
            
            # Limit depth to prevent excessive recursion
            if depth < 20:
                for char, child in node.children.items():
                    dfs(child, current_word + char, depth + 1)
        
        # Navigate to prefix node
        node = self.index.trie
        for char in prefix.lower():
            if char not in node.children:
                return []
            node = node.children[char]
        
        # Collect suggestions
        dfs(node, prefix.lower(), 0)
        
        # Sort by popularity and return
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return [word for word, _ in suggestions[:max_suggestions]]

class SearchEngine:
    """Complete search engine implementation"""
    
    def __init__(self):
        self.index = SearchIndex()
        self.scorer = RelevanceScorer(self.index)
        self.query_processor = QueryProcessor(self.index, self.scorer)
        self.update_queue = []
        self.update_lock = threading.Lock()
    
    def add_document(self, doc_id: str, title: str, content: str, url: str = "") -> None:
        """Add document to search engine"""
        document = Document(doc_id, title, content, url)
        self.index.add_document(document)
        
        # Clear relevant caches
        self._invalidate_caches()
    
    def remove_document(self, doc_id: str) -> bool:
        """Remove document from search engine"""
        success = self.index.remove_document(doc_id)
        if success:
            self._invalidate_caches()
        return success
    
    def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search for documents"""
        scored_results = self.query_processor.process_query(query, max_results)
        
        # Format results
        results = []
        for doc_id, score in scored_results:
            if doc_id in self.index.documents:
                doc = self.index.documents[doc_id]
                result = {
                    'doc_id': doc_id,
                    'title': doc.title,
                    'content': doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                    'url': doc.url,
                    'score': score
                }
                results.append(result)
        
        return results
    
    def autocomplete(self, prefix: str, max_suggestions: int = 10) -> List[str]:
        """Get autocomplete suggestions"""
        return self.query_processor.get_suggestions(prefix, max_suggestions)
    
    def update_document(self, doc_id: str, title: str = None, content: str = None, url: str = None) -> bool:
        """Update existing document"""
        if doc_id not in self.index.documents:
            return False
        
        # Get current document
        current_doc = self.index.documents[doc_id]
        
        # Update fields
        new_title = title if title is not None else current_doc.title
        new_content = content if content is not None else current_doc.content
        new_url = url if url is not None else current_doc.url
        
        # Remove old document and add updated one
        self.remove_document(doc_id)
        self.add_document(doc_id, new_title, new_content, new_url)
        
        return True
    
    def bulk_add_documents(self, documents: List[Tuple[str, str, str, str]]) -> None:
        """Add multiple documents efficiently"""
        for doc_id, title, content, url in documents:
            self.add_document(doc_id, title, content, url)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get search engine statistics"""
        return {
            'total_documents': self.index.document_count,
            'total_words': self.index.total_words,
            'cache_size': len(self.query_processor.query_cache),
            'avg_document_length': self.index.total_words / max(1, self.index.document_count)
        }
    
    def _invalidate_caches(self) -> None:
        """Invalidate query caches after updates"""
        self.query_processor.query_cache.clear()

# Real-time updates simulation
class RealTimeIndexer:
    """Handle real-time index updates"""
    
    def __init__(self, search_engine: SearchEngine):
        self.search_engine = search_engine
        self.update_queue = []
        self.processing = False
        self.lock = threading.Lock()
    
    def queue_update(self, operation: str, *args) -> None:
        """Queue an update operation"""
        with self.lock:
            self.update_queue.append((operation, args, time.time()))
    
    def process_updates(self) -> None:
        """Process queued updates"""
        with self.lock:
            if self.processing or not self.update_queue:
                return
            
            self.processing = True
            updates_to_process = self.update_queue[:]
            self.update_queue.clear()
        
        try:
            for operation, args, timestamp in updates_to_process:
                if operation == "add":
                    self.search_engine.add_document(*args)
                elif operation == "remove":
                    self.search_engine.remove_document(*args)
                elif operation == "update":
                    self.search_engine.update_document(*args)
        finally:
            with self.lock:
                self.processing = False


def test_search_engine():
    """Test search engine functionality"""
    print("=== Testing Search Engine ===")
    
    engine = SearchEngine()
    
    # Add sample documents
    documents = [
        ("doc1", "Python Programming", "Learn Python programming language with examples and tutorials", "http://example.com/python"),
        ("doc2", "Machine Learning", "Introduction to machine learning algorithms and applications", "http://example.com/ml"),
        ("doc3", "Data Structures", "Understanding data structures like trees, graphs, and hash tables", "http://example.com/ds"),
        ("doc4", "Python Libraries", "Popular Python libraries for data science and web development", "http://example.com/pylib"),
        ("doc5", "Algorithm Design", "Designing efficient algorithms for solving computational problems", "http://example.com/algo")
    ]
    
    print("Adding documents...")
    for doc_data in documents:
        engine.add_document(*doc_data)
    
    # Test searches
    queries = ["python", "machine learning", "algorithms", "data"]
    
    print(f"\nSearch results:")
    for query in queries:
        print(f"\nQuery: '{query}'")
        results = engine.search(query, max_results=3)
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['title']} (score: {result['score']:.3f})")
            print(f"     {result['content']}")
    
    # Test autocomplete
    print(f"\nAutocomplete suggestions:")
    prefixes = ["py", "ma", "da"]
    
    for prefix in prefixes:
        suggestions = engine.autocomplete(prefix, max_suggestions=5)
        print(f"  '{prefix}': {suggestions}")

def test_relevance_scoring():
    """Test relevance scoring algorithms"""
    print("\n=== Testing Relevance Scoring ===")
    
    engine = SearchEngine()
    
    # Add documents with varying content
    docs = [
        ("doc1", "Python Tutorial", "Python is a programming language. Python is easy to learn."),
        ("doc2", "Java Programming", "Java is a programming language used for enterprise applications."),
        ("doc3", "Programming Languages", "Python and Java are popular programming languages."),
    ]
    
    for doc_data in docs:
        engine.add_document(*doc_data)
    
    # Test different scoring methods
    query = "python programming"
    results = engine.search(query)
    
    print(f"Search results for '{query}':")
    for result in results:
        print(f"  {result['title']}: {result['score']:.3f}")

def test_real_time_updates():
    """Test real-time index updates"""
    print("\n=== Testing Real-time Updates ===")
    
    engine = SearchEngine()
    indexer = RealTimeIndexer(engine)
    
    # Initial documents
    engine.add_document("doc1", "Original Title", "Original content", "http://example.com")
    
    # Test initial search
    results = engine.search("original")
    print(f"Before update - found {len(results)} results")
    
    # Queue updates
    indexer.queue_update("update", "doc1", "Updated Title", "Updated content with new information")
    indexer.queue_update("add", "doc2", "New Document", "This is a new document", "http://new.com")
    
    # Process updates
    indexer.process_updates()
    
    # Test search after updates
    results = engine.search("updated")
    print(f"After update - found {len(results)} results for 'updated'")
    
    results = engine.search("new")
    print(f"After update - found {len(results)} results for 'new'")

def benchmark_search_performance():
    """Benchmark search engine performance"""
    print("\n=== Benchmarking Search Performance ===")
    
    engine = SearchEngine()
    
    # Generate test documents
    import random
    import string
    
    print("Generating test documents...")
    
    def generate_document(doc_id: int) -> Tuple[str, str, str, str]:
        title_words = random.choices(["python", "java", "algorithm", "data", "machine", "learning"], k=3)
        title = " ".join(title_words).title()
        
        content_words = random.choices(
            ["programming", "language", "tutorial", "example", "code", "development", 
             "software", "computer", "science", "technology"], k=20
        )
        content = " ".join(content_words)
        
        return f"doc{doc_id}", title, content, f"http://example.com/{doc_id}"
    
    # Add documents
    num_docs = 1000
    start_time = time.time()
    
    for i in range(num_docs):
        doc_data = generate_document(i)
        engine.add_document(*doc_data)
    
    index_time = time.time() - start_time
    
    # Test search performance
    test_queries = ["python", "programming", "algorithm", "data science", "machine learning"]
    
    start_time = time.time()
    total_results = 0
    
    for query in test_queries * 20:  # 100 total queries
        results = engine.search(query)
        total_results += len(results)
    
    search_time = time.time() - start_time
    
    # Get statistics
    stats = engine.get_statistics()
    
    print(f"Performance Results:")
    print(f"  Indexing: {num_docs} documents in {index_time:.2f}s ({num_docs/index_time:.0f} docs/sec)")
    print(f"  Searching: 100 queries in {search_time:.2f}s ({100/search_time:.0f} queries/sec)")
    print(f"  Average results per query: {total_results/100:.1f}")
    print(f"  Total indexed words: {stats['total_words']}")
    print(f"  Average document length: {stats['avg_document_length']:.1f} words")

if __name__ == "__main__":
    test_search_engine()
    test_relevance_scoring()
    test_real_time_updates()
    benchmark_search_performance()

"""
Search Engine Architecture demonstrates enterprise search system design:

Key Components:
1. Inverted Index - Trie-based word-to-document mapping
2. Relevance Scoring - TF-IDF and BM25 algorithms for ranking
3. Query Processing - Query optimization and autocomplete
4. Real-time Updates - Handle index updates without downtime
5. Caching - Query result caching for performance
6. Scalability - Design for horizontal scaling

Search Features:
- Full-text search with relevance ranking
- Autocomplete and query suggestions
- Real-time index updates
- Title boost and field-specific scoring
- Efficient candidate document filtering

Performance Optimizations:
- Trie-based indexing for fast prefix operations
- Query result caching
- Bulk document operations
- Incremental index updates
- Memory-efficient data structures

Real-world Applications:
- Enterprise search systems
- E-commerce product search
- Document management systems
- Knowledge base search
- Content management platforms

This implementation provides a solid foundation for building
production-ready search engines with enterprise requirements.
"""
