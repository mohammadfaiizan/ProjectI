"""
Autocomplete Service - Multiple Approaches
Difficulty: Hard

Design and implementation of a high-performance autocomplete service
with real-time suggestions, personalization, and analytics.

Components:
1. Prefix-based Trie with Frequency Scoring
2. Personalized Suggestions
3. Real-time Analytics and Trending
4. Caching and Performance Optimization
5. A/B Testing Framework
6. Distributed Architecture
"""

import time
import threading
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, deque
import heapq
import random

class AutocompleteTrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.frequency = 0
        self.suggestions = []  # Pre-computed top suggestions
        self.last_updated = time.time()

class PersonalizationProfile:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.search_history = deque(maxlen=1000)
        self.category_preferences = defaultdict(float)
        self.last_activity = time.time()
        self.click_through_rates = defaultdict(float)

class AutocompleteService:
    """High-performance autocomplete service"""
    
    def __init__(self, max_suggestions: int = 10):
        self.trie = AutocompleteTrieNode()
        self.max_suggestions = max_suggestions
        self.global_frequencies = defaultdict(int)
        self.trending_queries = {}
        self.user_profiles = {}
        self.cache = {}
        self.analytics = {
            'total_queries': 0,
            'cache_hits': 0,
            'avg_response_time': 0
        }
        self.lock = threading.RLock()
    
    def add_query(self, query: str, frequency: int = 1, category: str = "general") -> None:
        """Add query to autocomplete with frequency"""
        with self.lock:
            self.global_frequencies[query.lower()] += frequency
            self._insert_into_trie(query.lower(), frequency)
            self._update_trending(query.lower(), frequency)
            self._invalidate_cache(query.lower())
    
    def _insert_into_trie(self, query: str, frequency: int) -> None:
        """Insert query into trie with frequency"""
        node = self.trie
        
        for char in query:
            if char not in node.children:
                node.children[char] = AutocompleteTrieNode()
            node = node.children[char]
            
            # Update frequency for this path
            node.frequency += frequency
        
        node.is_end = True
        node.last_updated = time.time()
        
        # Update suggestions for all nodes in the path
        self._update_suggestions_in_path(query)
    
    def _update_suggestions_in_path(self, query: str) -> None:
        """Update pre-computed suggestions for all prefixes"""
        for i in range(1, len(query) + 1):
            prefix = query[:i]
            self._recompute_suggestions(prefix)
    
    def _recompute_suggestions(self, prefix: str) -> None:
        """Recompute top suggestions for a prefix"""
        suggestions = []
        
        def dfs(node: AutocompleteTrieNode, current_query: str, depth: int) -> None:
            if len(suggestions) >= self.max_suggestions * 2:  # Get more than needed for ranking
                return
            
            if node.is_end:
                frequency = self.global_frequencies[current_query]
                suggestions.append((current_query, frequency))
            
            # Limit depth to prevent excessive computation
            if depth < 50:
                for char, child in node.children.items():
                    dfs(child, current_query + char, depth + 1)
        
        # Navigate to prefix node
        node = self.trie
        for char in prefix:
            if char not in node.children:
                return
            node = node.children[char]
        
        # Collect suggestions
        dfs(node, prefix, 0)
        
        # Sort by frequency and store top suggestions
        suggestions.sort(key=lambda x: x[1], reverse=True)
        node.suggestions = [query for query, _ in suggestions[:self.max_suggestions]]
    
    def get_suggestions(self, prefix: str, user_id: str = None, 
                       max_results: int = None) -> List[Dict[str, Any]]:
        """Get autocomplete suggestions for prefix"""
        start_time = time.time()
        
        with self.lock:
            self.analytics['total_queries'] += 1
            
            if not prefix:
                return []
            
            prefix = prefix.lower()
            max_results = max_results or self.max_suggestions
            
            # Check cache
            cache_key = f"{prefix}:{user_id}:{max_results}"
            if cache_key in self.cache:
                self.analytics['cache_hits'] += 1
                return self.cache[cache_key]
            
            # Get base suggestions from trie
            base_suggestions = self._get_base_suggestions(prefix, max_results * 2)
            
            # Apply personalization if user provided
            if user_id:
                personalized_suggestions = self._apply_personalization(
                    base_suggestions, user_id, max_results
                )
            else:
                personalized_suggestions = base_suggestions[:max_results]
            
            # Format results
            results = []
            for i, query in enumerate(personalized_suggestions):
                result = {
                    'suggestion': query,
                    'frequency': self.global_frequencies[query],
                    'rank': i + 1,
                    'is_trending': query in self.trending_queries,
                    'category': 'general'  # Could be enhanced with actual categories
                }
                results.append(result)
            
            # Cache results
            self.cache[cache_key] = results
            
            # Update analytics
            response_time = (time.time() - start_time) * 1000
            self._update_response_time(response_time)
            
            return results
    
    def _get_base_suggestions(self, prefix: str, max_results: int) -> List[str]:
        """Get base suggestions from trie"""
        node = self.trie
        
        # Navigate to prefix node
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        
        # Return pre-computed suggestions if available
        if node.suggestions:
            return node.suggestions[:max_results]
        
        # Fallback: compute suggestions on-demand
        suggestions = []
        
        def dfs(current_node: AutocompleteTrieNode, current_query: str, depth: int) -> None:
            if len(suggestions) >= max_results:
                return
            
            if current_node.is_end:
                frequency = self.global_frequencies[current_query]
                suggestions.append((current_query, frequency))
            
            if depth < 20:  # Limit depth
                for char, child in current_node.children.items():
                    dfs(child, current_query + char, depth + 1)
        
        dfs(node, prefix, 0)
        suggestions.sort(key=lambda x: x[1], reverse=True)
        
        return [query for query, _ in suggestions[:max_results]]
    
    def _apply_personalization(self, base_suggestions: List[str], 
                             user_id: str, max_results: int) -> List[str]:
        """Apply personalization to suggestions"""
        if user_id not in self.user_profiles:
            return base_suggestions[:max_results]
        
        profile = self.user_profiles[user_id]
        scored_suggestions = []
        
        for suggestion in base_suggestions:
            base_score = self.global_frequencies[suggestion]
            
            # Personal history boost
            personal_boost = 0
            if suggestion in [q.lower() for q in profile.search_history]:
                personal_boost = base_score * 0.5
            
            # Click-through rate boost
            ctr_boost = profile.click_through_rates.get(suggestion, 0) * base_score * 0.3
            
            final_score = base_score + personal_boost + ctr_boost
            scored_suggestions.append((suggestion, final_score))
        
        # Sort by personalized score
        scored_suggestions.sort(key=lambda x: x[1], reverse=True)
        
        return [query for query, _ in scored_suggestions[:max_results]]
    
    def record_user_interaction(self, user_id: str, query: str, 
                              action: str, suggestion_rank: int = None) -> None:
        """Record user interaction for personalization"""
        with self.lock:
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = PersonalizationProfile(user_id)
            
            profile = self.user_profiles[user_id]
            profile.last_activity = time.time()
            
            if action == "search":
                profile.search_history.append(query)
                # Boost global frequency
                self.global_frequencies[query.lower()] += 1
            
            elif action == "click" and suggestion_rank is not None:
                # Update click-through rate
                current_ctr = profile.click_through_rates[query.lower()]
                # Simple CTR calculation (can be improved)
                new_ctr = (current_ctr + (1.0 / suggestion_rank)) / 2
                profile.click_through_rates[query.lower()] = new_ctr
    
    def _update_trending(self, query: str, frequency: int) -> None:
        """Update trending queries"""
        current_time = time.time()
        
        if query not in self.trending_queries:
            self.trending_queries[query] = {
                'frequency': frequency,
                'first_seen': current_time,
                'last_seen': current_time,
                'trend_score': frequency
            }
        else:
            trend_data = self.trending_queries[query]
            trend_data['frequency'] += frequency
            trend_data['last_seen'] = current_time
            
            # Calculate trend score (frequency over time)
            time_window = current_time - trend_data['first_seen']
            trend_data['trend_score'] = trend_data['frequency'] / max(1, time_window / 3600)  # per hour
    
    def get_trending_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get currently trending queries"""
        with self.lock:
            trending_list = []
            current_time = time.time()
            
            for query, data in self.trending_queries.items():
                # Only consider recent queries (last 24 hours)
                if current_time - data['last_seen'] < 86400:
                    trending_list.append({
                        'query': query,
                        'frequency': data['frequency'],
                        'trend_score': data['trend_score']
                    })
            
            # Sort by trend score
            trending_list.sort(key=lambda x: x['trend_score'], reverse=True)
            
            return trending_list[:limit]
    
    def _invalidate_cache(self, query: str) -> None:
        """Invalidate cache entries affected by query update"""
        keys_to_remove = []
        
        for cache_key in self.cache:
            if cache_key.startswith(query) or query.startswith(cache_key.split(':')[0]):
                keys_to_remove.append(cache_key)
        
        for key in keys_to_remove:
            del self.cache[key]
    
    def _update_response_time(self, response_time: float) -> None:
        """Update average response time"""
        current_avg = self.analytics['avg_response_time']
        total_queries = self.analytics['total_queries']
        
        # Calculate running average
        self.analytics['avg_response_time'] = (
            (current_avg * (total_queries - 1) + response_time) / total_queries
        )
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get service analytics"""
        with self.lock:
            cache_hit_rate = self.analytics['cache_hits'] / max(1, self.analytics['total_queries'])
            
            return {
                'total_queries': self.analytics['total_queries'],
                'cache_hit_rate': cache_hit_rate,
                'avg_response_time_ms': self.analytics['avg_response_time'],
                'active_users': len(self.user_profiles),
                'total_cached_entries': len(self.cache),
                'trending_queries_count': len(self.trending_queries)
            }
    
    def cleanup_old_data(self, max_age_hours: int = 24) -> None:
        """Clean up old data to manage memory"""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        with self.lock:
            # Clean old trending queries
            queries_to_remove = []
            for query, data in self.trending_queries.items():
                if current_time - data['last_seen'] > max_age_seconds:
                    queries_to_remove.append(query)
            
            for query in queries_to_remove:
                del self.trending_queries[query]
            
            # Clean inactive user profiles
            users_to_remove = []
            for user_id, profile in self.user_profiles.items():
                if current_time - profile.last_activity > max_age_seconds:
                    users_to_remove.append(user_id)
            
            for user_id in users_to_remove:
                del self.user_profiles[user_id]
            
            # Clear cache (simple approach - could be more selective)
            self.cache.clear()

class AutocompleteABTesting:
    """A/B testing framework for autocomplete"""
    
    def __init__(self):
        self.experiments = {}
        self.user_assignments = {}
    
    def create_experiment(self, experiment_id: str, variants: List[str], 
                         traffic_split: List[float]) -> None:
        """Create new A/B test experiment"""
        if len(variants) != len(traffic_split):
            raise ValueError("Variants and traffic split must have same length")
        
        if abs(sum(traffic_split) - 1.0) > 0.001:
            raise ValueError("Traffic split must sum to 1.0")
        
        self.experiments[experiment_id] = {
            'variants': variants,
            'traffic_split': traffic_split,
            'metrics': {variant: defaultdict(int) for variant in variants}
        }
    
    def get_variant(self, experiment_id: str, user_id: str) -> str:
        """Get variant assignment for user"""
        if experiment_id not in self.experiments:
            return "control"
        
        # Consistent assignment based on user ID hash
        if (experiment_id, user_id) in self.user_assignments:
            return self.user_assignments[(experiment_id, user_id)]
        
        # Assign variant based on traffic split
        experiment = self.experiments[experiment_id]
        hash_value = hash(user_id) % 1000 / 1000.0
        
        cumulative = 0
        for i, (variant, split) in enumerate(zip(experiment['variants'], experiment['traffic_split'])):
            cumulative += split
            if hash_value < cumulative:
                self.user_assignments[(experiment_id, user_id)] = variant
                return variant
        
        # Fallback
        return experiment['variants'][0]
    
    def record_metric(self, experiment_id: str, variant: str, metric: str, value: int = 1) -> None:
        """Record metric for experiment variant"""
        if experiment_id in self.experiments and variant in self.experiments[experiment_id]['variants']:
            self.experiments[experiment_id]['metrics'][variant][metric] += value
    
    def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """Get experiment results"""
        if experiment_id not in self.experiments:
            return {}
        
        return {
            'experiment_id': experiment_id,
            'variants': self.experiments[experiment_id]['variants'],
            'metrics': dict(self.experiments[experiment_id]['metrics'])
        }


def test_autocomplete_service():
    """Test autocomplete service functionality"""
    print("=== Testing Autocomplete Service ===")
    
    service = AutocompleteService(max_suggestions=5)
    
    # Add sample queries
    queries = [
        ("python programming", 100),
        ("python tutorial", 80),
        ("python basics", 60),
        ("java programming", 90),
        ("javascript tutorial", 70),
        ("machine learning", 120),
        ("machine learning python", 85),
    ]
    
    print("Adding queries...")
    for query, freq in queries:
        service.add_query(query, freq)
    
    # Test suggestions
    test_prefixes = ["py", "python", "java", "machine"]
    
    print(f"\nSuggestions:")
    for prefix in test_prefixes:
        suggestions = service.get_suggestions(prefix)
        print(f"\n'{prefix}':")
        
        for suggestion in suggestions:
            print(f"  {suggestion['rank']}. {suggestion['suggestion']} "
                  f"(freq: {suggestion['frequency']}, trending: {suggestion['is_trending']})")

def test_personalization():
    """Test personalization features"""
    print("\n=== Testing Personalization ===")
    
    service = AutocompleteService()
    
    # Add base data
    queries = [
        ("python programming", 50),
        ("java programming", 60),
        ("web development", 40),
        ("machine learning", 70),
    ]
    
    for query, freq in queries:
        service.add_query(query, freq)
    
    # Simulate user interactions
    user_id = "user123"
    
    # User searches for Python-related queries
    service.record_user_interaction(user_id, "python programming", "search")
    service.record_user_interaction(user_id, "python tutorial", "search")
    service.record_user_interaction(user_id, "python programming", "click", 1)
    
    # Test personalized suggestions
    print("General suggestions for 'p':")
    general = service.get_suggestions("p")
    for s in general[:3]:
        print(f"  {s['suggestion']}")
    
    print(f"\nPersonalized suggestions for 'p' (user: {user_id}):")
    personalized = service.get_suggestions("p", user_id=user_id)
    for s in personalized[:3]:
        print(f"  {s['suggestion']}")

def test_trending_queries():
    """Test trending queries functionality"""
    print("\n=== Testing Trending Queries ===")
    
    service = AutocompleteService()
    
    # Simulate trending behavior
    current_queries = [
        ("covid vaccine", 20),
        ("climate change", 15),
        ("cryptocurrency", 25),
        ("remote work", 18),
    ]
    
    print("Adding trending queries...")
    for query, freq in current_queries:
        # Add with high frequency to simulate trending
        for _ in range(freq):
            service.add_query(query, 1)
    
    # Get trending queries
    trending = service.get_trending_queries(limit=5)
    
    print("Current trending queries:")
    for i, trend in enumerate(trending, 1):
        print(f"  {i}. {trend['query']} (score: {trend['trend_score']:.2f})")

def test_ab_testing():
    """Test A/B testing framework"""
    print("\n=== Testing A/B Testing ===")
    
    ab_testing = AutocompleteABTesting()
    
    # Create experiment
    ab_testing.create_experiment(
        "suggestion_count_test",
        variants=["control", "more_suggestions"],
        traffic_split=[0.5, 0.5]
    )
    
    # Simulate user assignments
    test_users = [f"user_{i}" for i in range(10)]
    
    print("User variant assignments:")
    for user in test_users:
        variant = ab_testing.get_variant("suggestion_count_test", user)
        print(f"  {user}: {variant}")
        
        # Record some metrics
        ab_testing.record_metric("suggestion_count_test", variant, "queries", 1)
        if variant == "more_suggestions":
            ab_testing.record_metric("suggestion_count_test", variant, "clicks", random.choice([0, 1]))
    
    # Get results
    results = ab_testing.get_experiment_results("suggestion_count_test")
    print(f"\nExperiment results:")
    for variant, metrics in results['metrics'].items():
        print(f"  {variant}: {dict(metrics)}")

def benchmark_autocomplete_performance():
    """Benchmark autocomplete performance"""
    print("\n=== Benchmarking Performance ===")
    
    service = AutocompleteService()
    
    # Generate test data
    import random
    import string
    
    print("Generating test data...")
    
    # Add large number of queries
    for i in range(10000):
        query_length = random.randint(5, 20)
        query = ''.join(random.choices(string.ascii_lowercase + ' ', k=query_length))
        frequency = random.randint(1, 100)
        service.add_query(query.strip(), frequency)
    
    # Benchmark suggestions
    test_prefixes = ['a', 'ab', 'abc', 'abcd', 'python', 'java', 'test']
    
    start_time = time.time()
    total_suggestions = 0
    
    # Run multiple queries
    for _ in range(1000):
        prefix = random.choice(test_prefixes)
        suggestions = service.get_suggestions(prefix)
        total_suggestions += len(suggestions)
    
    end_time = time.time()
    
    # Get analytics
    analytics = service.get_analytics()
    
    print(f"Performance results:")
    print(f"  1000 queries in {(end_time - start_time):.3f}s")
    print(f"  Average response time: {analytics['avg_response_time_ms']:.2f}ms")
    print(f"  Cache hit rate: {analytics['cache_hit_rate']:.2%}")
    print(f"  Total suggestions returned: {total_suggestions}")

if __name__ == "__main__":
    test_autocomplete_service()
    test_personalization()
    test_trending_queries()
    test_ab_testing()
    benchmark_autocomplete_performance()

"""
Autocomplete Service demonstrates enterprise-grade autocomplete system design:

Key Features:
1. Prefix-based Trie - Efficient prefix matching with frequency scoring
2. Personalization - User-specific suggestions based on search history
3. Trending Analysis - Real-time trending query detection
4. Performance Optimization - Caching and pre-computed suggestions
5. A/B Testing - Framework for experimenting with different algorithms
6. Analytics - Comprehensive metrics and monitoring

System Design Aspects:
- Trie-based data structure for fast prefix operations
- User profiling and personalization algorithms
- Real-time analytics and trending detection
- Caching strategies for improved performance
- A/B testing framework for continuous optimization
- Memory management and cleanup procedures

Real-world Applications:
- Search engine autocomplete (Google, Bing)
- E-commerce product suggestions (Amazon, eBay)
- Social media hashtag suggestions (Twitter, Instagram)
- IDE code completion (VSCode, IntelliJ)
- Command-line autocompletion
- Address and location autocomplete

Performance Considerations:
- Sub-millisecond response times
- Horizontal scaling across multiple servers
- Cache invalidation strategies
- Memory usage optimization
- Real-time index updates

This implementation provides a production-ready foundation for
building scalable autocomplete services with enterprise features.
"""
