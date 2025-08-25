"""
Advanced Autocomplete System - Multiple Approaches
Difficulty: Hard

Implement an advanced autocomplete system with the following features:
1. Context-aware suggestions based on user behavior
2. Personalized suggestions based on user history
3. Real-time learning and adaptation
4. Multi-language support
5. Fuzzy matching and error tolerance
6. Performance optimization for large-scale deployment
"""

from typing import List, Dict, Set, Tuple, Optional, Any
from collections import defaultdict, deque
import heapq
import time
import threading
from dataclasses import dataclass
import json

@dataclass
class SearchContext:
    """Context information for search suggestions"""
    user_id: str
    session_id: str
    timestamp: float
    location: Optional[str] = None
    device_type: Optional[str] = None
    previous_queries: Optional[List[str]] = None

@dataclass
class SuggestionResult:
    """Suggestion result with metadata"""
    text: str
    score: float
    source: str  # Source of suggestion (history, trending, etc.)
    metadata: Dict[str, Any] = None

class TrieNode:
    """Enhanced trie node for advanced autocomplete"""
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.words = []
        self.frequency = 0
        self.user_frequencies = defaultdict(int)  # Per-user frequencies
        self.context_scores = defaultdict(float)  # Context-based scores
        self.last_accessed = 0

class AdvancedAutocomplete1:
    """
    Approach 1: Context-Aware Trie with User Behavior Tracking
    
    Build trie with context awareness and user behavior tracking.
    """
    
    def __init__(self):
        self.root = TrieNode()
        self.user_histories = defaultdict(list)  # user_id -> query history
        self.global_frequencies = defaultdict(int)
        self.context_weights = {
            'recency': 0.3,
            'frequency': 0.4,
            'personalization': 0.2,
            'context': 0.1
        }
        self.decay_factor = 0.95  # For temporal decay
        self.lock = threading.RLock()
    
    def add_query(self, query: str, context: SearchContext) -> None:
        """
        Add query with context information.
        
        Time: O(|query|)
        Space: O(|query|)
        """
        with self.lock:
            query = query.lower().strip()
            current_time = time.time()
            
            # Update global frequency
            self.global_frequencies[query] += 1
            
            # Update user history
            self.user_histories[context.user_id].append({
                'query': query,
                'timestamp': current_time,
                'context': context
            })
            
            # Keep only recent history (last 1000 queries per user)
            if len(self.user_histories[context.user_id]) > 1000:
                self.user_histories[context.user_id] = self.user_histories[context.user_id][-1000:]
            
            # Update trie
            self._update_trie(query, context, current_time)
    
    def _update_trie(self, query: str, context: SearchContext, timestamp: float) -> None:
        """Update trie with query and context"""
        node = self.root
        
        for char in query:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            
            # Update node statistics
            node.frequency += 1
            node.user_frequencies[context.user_id] += 1
            node.last_accessed = timestamp
            
            # Update context scores
            if context.location:
                node.context_scores[f"location:{context.location}"] += 1
            if context.device_type:
                node.context_scores[f"device:{context.device_type}"] += 1
        
        node.is_end = True
        if query not in node.words:
            node.words.append(query)
    
    def get_suggestions(self, prefix: str, context: SearchContext, max_suggestions: int = 10) -> List[SuggestionResult]:
        """
        Get personalized suggestions with context awareness.
        
        Time: O(|prefix| + suggestions * log(suggestions))
        Space: O(suggestions)
        """
        with self.lock:
            prefix = prefix.lower().strip()
            current_time = time.time()
            
            # Navigate to prefix node
            node = self.root
            for char in prefix:
                if char not in node.children:
                    return []
                node = node.children[char]
            
            # Collect all possible completions
            suggestions = []
            self._collect_suggestions(node, prefix, context, current_time, suggestions)
            
            # Score and rank suggestions
            scored_suggestions = []
            for word in suggestions:
                score = self._calculate_score(word, context, current_time)
                source = self._determine_source(word, context)
                scored_suggestions.append(SuggestionResult(
                    text=word,
                    score=score,
                    source=source,
                    metadata={'frequency': self.global_frequencies[word]}
                ))
            
            # Sort by score and return top suggestions
            scored_suggestions.sort(key=lambda x: x.score, reverse=True)
            return scored_suggestions[:max_suggestions]
    
    def _collect_suggestions(self, node: TrieNode, prefix: str, context: SearchContext, 
                           current_time: float, suggestions: List[str]) -> None:
        """Collect all possible suggestions from trie"""
        if node.is_end:
            suggestions.extend(node.words)
        
        for child in node.children.values():
            self._collect_suggestions(child, prefix, context, current_time, suggestions)
    
    def _calculate_score(self, word: str, context: SearchContext, current_time: float) -> float:
        """Calculate personalized score for suggestion"""
        score = 0.0
        
        # Global frequency score
        global_freq = self.global_frequencies[word]
        freq_score = global_freq / (1 + global_freq)  # Normalized
        score += self.context_weights['frequency'] * freq_score
        
        # Personalization score based on user history
        user_history = self.user_histories[context.user_id]
        personal_score = 0.0
        
        for entry in user_history:
            if entry['query'] == word:
                # Apply temporal decay
                time_diff = current_time - entry['timestamp']
                decay = self.decay_factor ** (time_diff / 3600)  # Hourly decay
                personal_score += decay
        
        score += self.context_weights['personalization'] * personal_score
        
        # Recency score
        if user_history:
            recent_queries = [entry['query'] for entry in user_history[-10:]]
            if word in recent_queries:
                recency_score = 1.0 - (recent_queries[::-1].index(word) / 10)
                score += self.context_weights['recency'] * recency_score
        
        # Context score
        context_score = 0.0
        if context.location:
            # Boost if query is popular in user's location
            context_score += 0.1  # Simplified
        
        score += self.context_weights['context'] * context_score
        
        return score
    
    def _determine_source(self, word: str, context: SearchContext) -> str:
        """Determine the source of suggestion"""
        user_history = self.user_histories[context.user_id]
        recent_queries = [entry['query'] for entry in user_history[-10:]]
        
        if word in recent_queries:
            return "recent_history"
        elif any(entry['query'] == word for entry in user_history):
            return "personal_history"
        elif self.global_frequencies[word] > 100:  # Threshold for trending
            return "trending"
        else:
            return "general"


class AdvancedAutocomplete2:
    """
    Approach 2: Machine Learning Inspired with Feature Engineering
    
    Use feature-based scoring for suggestions.
    """
    
    def __init__(self):
        self.root = TrieNode()
        self.user_profiles = defaultdict(dict)  # user_id -> profile
        self.query_embeddings = {}  # Simplified query embeddings
        self.co_occurrence_matrix = defaultdict(lambda: defaultdict(int))
        
    def train_model(self, training_data: List[Tuple[str, SearchContext]]) -> None:
        """
        Train the autocomplete model on historical data.
        
        Time: O(n * average_query_length)
        Space: O(n * average_query_length)
        """
        # Build user profiles
        for query, context in training_data:
            user_id = context.user_id
            
            # Update user profile
            profile = self.user_profiles[user_id]
            profile['query_count'] = profile.get('query_count', 0) + 1
            
            if 'preferred_topics' not in profile:
                profile['preferred_topics'] = defaultdict(int)
            
            # Simple topic extraction (in real system, use NLP)
            topics = self._extract_topics(query)
            for topic in topics:
                profile['preferred_topics'][topic] += 1
            
            # Build co-occurrence matrix
            if context.previous_queries:
                for prev_query in context.previous_queries[-3:]:  # Last 3 queries
                    self.co_occurrence_matrix[prev_query][query] += 1
            
            # Add to trie
            self._add_to_trie(query, context)
    
    def _extract_topics(self, query: str) -> List[str]:
        """Extract topics from query (simplified)"""
        # In real system, use NLP libraries
        topic_keywords = {
            'tech': ['python', 'programming', 'code', 'software', 'computer'],
            'food': ['restaurant', 'food', 'recipe', 'cooking', 'eat'],
            'travel': ['hotel', 'flight', 'travel', 'vacation', 'trip'],
            'shopping': ['buy', 'purchase', 'shop', 'store', 'price']
        }
        
        topics = []
        query_lower = query.lower()
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                topics.append(topic)
        
        return topics if topics else ['general']
    
    def _add_to_trie(self, query: str, context: SearchContext) -> None:
        """Add query to trie with context"""
        node = self.root
        
        for char in query:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.frequency += 1
            node.user_frequencies[context.user_id] += 1
        
        node.is_end = True
        if query not in node.words:
            node.words.append(query)
    
    def get_suggestions(self, prefix: str, context: SearchContext, max_suggestions: int = 10) -> List[SuggestionResult]:
        """
        Get ML-powered suggestions.
        
        Time: O(|prefix| + suggestions * feature_computation)
        Space: O(suggestions)
        """
        prefix = prefix.lower().strip()
        
        # Get candidate suggestions from trie
        candidates = self._get_candidates(prefix)
        
        # Extract features and score
        scored_suggestions = []
        
        for candidate in candidates:
            features = self._extract_features(candidate, prefix, context)
            score = self._score_with_features(features)
            
            scored_suggestions.append(SuggestionResult(
                text=candidate,
                score=score,
                source="ml_model",
                metadata=features
            ))
        
        # Sort and return top suggestions
        scored_suggestions.sort(key=lambda x: x.score, reverse=True)
        return scored_suggestions[:max_suggestions]
    
    def _get_candidates(self, prefix: str) -> List[str]:
        """Get candidate completions from trie"""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        
        candidates = []
        self._collect_words(node, candidates)
        return candidates
    
    def _collect_words(self, node: TrieNode, candidates: List[str]) -> None:
        """Collect all words from trie node"""
        if node.is_end:
            candidates.extend(node.words)
        
        for child in node.children.values():
            self._collect_words(child, candidates)
    
    def _extract_features(self, candidate: str, prefix: str, context: SearchContext) -> Dict[str, float]:
        """Extract features for ML scoring"""
        features = {}
        
        # Length features
        features['candidate_length'] = len(candidate)
        features['prefix_ratio'] = len(prefix) / len(candidate) if candidate else 0
        
        # Frequency features
        features['global_frequency'] = self._get_frequency(candidate)
        features['user_frequency'] = self._get_user_frequency(candidate, context.user_id)
        
        # Context features
        user_profile = self.user_profiles[context.user_id]
        candidate_topics = self._extract_topics(candidate)
        
        # Topic similarity
        topic_similarity = 0.0
        if 'preferred_topics' in user_profile:
            for topic in candidate_topics:
                topic_similarity += user_profile['preferred_topics'].get(topic, 0)
        
        features['topic_similarity'] = topic_similarity
        
        # Co-occurrence features
        co_occurrence_score = 0.0
        if context.previous_queries:
            for prev_query in context.previous_queries[-3:]:
                co_occurrence_score += self.co_occurrence_matrix[prev_query][candidate]
        
        features['co_occurrence'] = co_occurrence_score
        
        # Temporal features
        features['hour_of_day'] = time.localtime().tm_hour
        features['day_of_week'] = time.localtime().tm_wday
        
        return features
    
    def _score_with_features(self, features: Dict[str, float]) -> float:
        """Score candidate using features (simplified linear model)"""
        # In real system, use trained ML model
        weights = {
            'global_frequency': 0.3,
            'user_frequency': 0.2,
            'topic_similarity': 0.2,
            'co_occurrence': 0.15,
            'prefix_ratio': 0.1,
            'candidate_length': -0.05  # Slight penalty for longer candidates
        }
        
        score = 0.0
        for feature, value in features.items():
            if feature in weights:
                score += weights[feature] * value
        
        return max(0, score)  # Ensure non-negative score
    
    def _get_frequency(self, word: str) -> int:
        """Get global frequency of word"""
        # Navigate trie to get frequency
        node = self.root
        for char in word:
            if char not in node.children:
                return 0
            node = node.children[char]
        
        return node.frequency if node.is_end else 0
    
    def _get_user_frequency(self, word: str, user_id: str) -> int:
        """Get user-specific frequency"""
        node = self.root
        for char in word:
            if char not in node.children:
                return 0
            node = node.children[char]
        
        return node.user_frequencies[user_id] if node.is_end else 0


class AdvancedAutocomplete3:
    """
    Approach 3: Real-time Learning with Adaptive Weights
    
    Continuously adapt suggestion weights based on user interactions.
    """
    
    def __init__(self):
        self.root = TrieNode()
        self.user_weights = defaultdict(lambda: defaultdict(float))  # user -> feature -> weight
        self.global_weights = {
            'frequency': 0.4,
            'recency': 0.3,
            'personalization': 0.2,
            'context': 0.1
        }
        self.learning_rate = 0.01
        
    def record_interaction(self, prefix: str, selected_suggestion: str, 
                          suggestions_shown: List[str], context: SearchContext) -> None:
        """
        Record user interaction for learning.
        
        Time: O(|suggestions|)
        Space: O(1)
        """
        user_id = context.user_id
        
        # Update weights based on user choice
        selected_index = suggestions_shown.index(selected_suggestion) if selected_suggestion in suggestions_shown else -1
        
        # Reward weights that led to selection of chosen suggestion
        if selected_index >= 0:
            # Higher reward for higher-ranked selections
            reward = 1.0 / (selected_index + 1)
            
            # Update user-specific weights (simplified reinforcement learning)
            for feature in self.global_weights:
                current_weight = self.user_weights[user_id].get(feature, self.global_weights[feature])
                self.user_weights[user_id][feature] = current_weight + self.learning_rate * reward
        
        # Normalize weights
        total_weight = sum(self.user_weights[user_id].values())
        if total_weight > 0:
            for feature in self.user_weights[user_id]:
                self.user_weights[user_id][feature] /= total_weight
    
    def get_adaptive_suggestions(self, prefix: str, context: SearchContext, 
                               max_suggestions: int = 10) -> List[SuggestionResult]:
        """
        Get suggestions with adaptive weights.
        
        Time: O(|prefix| + suggestions * log(suggestions))
        Space: O(suggestions)
        """
        user_id = context.user_id
        
        # Get user-specific weights or use global defaults
        weights = self.user_weights[user_id] if user_id in self.user_weights else self.global_weights
        
        # Get candidates and score with adaptive weights
        candidates = self._get_candidates(prefix)
        scored_suggestions = []
        
        for candidate in candidates:
            score = self._calculate_adaptive_score(candidate, context, weights)
            scored_suggestions.append(SuggestionResult(
                text=candidate,
                score=score,
                source="adaptive",
                metadata={'weights_used': dict(weights)}
            ))
        
        scored_suggestions.sort(key=lambda x: x.score, reverse=True)
        return scored_suggestions[:max_suggestions]
    
    def _calculate_adaptive_score(self, candidate: str, context: SearchContext, weights: Dict[str, float]) -> float:
        """Calculate score using adaptive weights"""
        # Simplified scoring - in real system, compute actual feature values
        base_scores = {
            'frequency': 0.5,  # Placeholder
            'recency': 0.3,
            'personalization': 0.7,
            'context': 0.2
        }
        
        total_score = 0.0
        for feature, weight in weights.items():
            if feature in base_scores:
                total_score += weight * base_scores[feature]
        
        return total_score
    
    def _get_candidates(self, prefix: str) -> List[str]:
        """Get candidate suggestions"""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        
        candidates = []
        self._collect_words(node, candidates)
        return candidates
    
    def _collect_words(self, node: TrieNode, candidates: List[str]) -> None:
        """Collect words from trie"""
        if node.is_end:
            candidates.extend(node.words)
        
        for child in node.children.values():
            self._collect_words(child, candidates)


class AdvancedAutocomplete4:
    """
    Approach 4: Multi-modal with Performance Optimization
    
    Combine multiple data sources with performance optimizations.
    """
    
    def __init__(self):
        self.tries = {
            'queries': TrieNode(),
            'products': TrieNode(),
            'locations': TrieNode(),
            'trending': TrieNode()
        }
        
        self.cache = {}  # LRU cache for suggestions
        self.cache_size = 1000
        self.background_updater = None
        
    def add_data_source(self, source_type: str, items: List[Tuple[str, float]]) -> None:
        """
        Add items to specific data source.
        
        Time: O(sum of item lengths)
        Space: O(sum of item lengths)
        """
        if source_type not in self.tries:
            self.tries[source_type] = TrieNode()
        
        root = self.tries[source_type]
        
        for item, score in items:
            node = root
            for char in item.lower():
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
                node.frequency += score
            
            node.is_end = True
            if item not in node.words:
                node.words.append(item)
    
    def get_multi_modal_suggestions(self, prefix: str, context: SearchContext, 
                                  source_weights: Dict[str, float] = None,
                                  max_suggestions: int = 10) -> List[SuggestionResult]:
        """
        Get suggestions from multiple sources.
        
        Time: O(|prefix| + total_suggestions * log(total_suggestions))
        Space: O(total_suggestions)
        """
        if source_weights is None:
            source_weights = {source: 1.0 for source in self.tries}
        
        # Check cache first
        cache_key = (prefix, context.user_id, tuple(sorted(source_weights.items())))
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        all_suggestions = []
        
        # Get suggestions from each source
        for source_type, weight in source_weights.items():
            if source_type in self.tries:
                source_suggestions = self._get_source_suggestions(
                    self.tries[source_type], prefix, source_type, weight
                )
                all_suggestions.extend(source_suggestions)
        
        # Merge and deduplicate
        suggestion_map = defaultdict(list)
        for suggestion in all_suggestions:
            suggestion_map[suggestion.text].append(suggestion)
        
        # Combine scores for duplicate suggestions
        final_suggestions = []
        for text, suggestions in suggestion_map.items():
            combined_score = sum(s.score for s in suggestions)
            sources = [s.source for s in suggestions]
            
            final_suggestions.append(SuggestionResult(
                text=text,
                score=combined_score,
                source='+'.join(sources),
                metadata={'source_count': len(suggestions)}
            ))
        
        # Sort and cache
        final_suggestions.sort(key=lambda x: x.score, reverse=True)
        result = final_suggestions[:max_suggestions]
        
        # Update cache (with size limit)
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry (simplified LRU)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[cache_key] = result
        return result
    
    def _get_source_suggestions(self, trie_root: TrieNode, prefix: str, 
                              source_type: str, weight: float) -> List[SuggestionResult]:
        """Get suggestions from specific source"""
        node = trie_root
        for char in prefix.lower():
            if char not in node.children:
                return []
            node = node.children[char]
        
        suggestions = []
        self._collect_weighted_suggestions(node, prefix, source_type, weight, suggestions)
        return suggestions
    
    def _collect_weighted_suggestions(self, node: TrieNode, prefix: str, 
                                    source_type: str, weight: float, 
                                    suggestions: List[SuggestionResult]) -> None:
        """Collect suggestions with source-specific weighting"""
        if node.is_end:
            for word in node.words:
                score = node.frequency * weight
                suggestions.append(SuggestionResult(
                    text=word,
                    score=score,
                    source=source_type
                ))
        
        for child in node.children.values():
            self._collect_weighted_suggestions(child, prefix, source_type, weight, suggestions)


def test_advanced_autocomplete():
    """Test advanced autocomplete systems"""
    print("=== Testing Advanced Autocomplete Systems ===")
    
    # Create test contexts
    context1 = SearchContext(
        user_id="user1",
        session_id="session1", 
        timestamp=time.time(),
        location="New York",
        device_type="mobile",
        previous_queries=["python programming", "machine learning"]
    )
    
    context2 = SearchContext(
        user_id="user2",
        session_id="session2",
        timestamp=time.time(),
        location="San Francisco", 
        device_type="desktop",
        previous_queries=["restaurant", "food delivery"]
    )
    
    systems = [
        ("Context-Aware", AdvancedAutocomplete1()),
        ("ML-Inspired", AdvancedAutocomplete2()),
        ("Adaptive Learning", AdvancedAutocomplete3()),
        ("Multi-modal", AdvancedAutocomplete4()),
    ]
    
    # Test queries
    test_queries = [
        ("python", "python programming"),
        ("machine", "machine learning"),
        ("rest", "restaurant"),
        ("food", "food delivery")
    ]
    
    for system_name, system in systems:
        print(f"\n{system_name}:")
        
        try:
            # Setup system with some data
            if isinstance(system, AdvancedAutocomplete1):
                system.add_query("python programming", context1)
                system.add_query("machine learning", context1)
                system.add_query("restaurant search", context2)
                system.add_query("food delivery", context2)
            
            elif isinstance(system, AdvancedAutocomplete2):
                training_data = [
                    ("python programming", context1),
                    ("machine learning", context1),
                    ("restaurant search", context2),
                    ("food delivery", context2)
                ]
                system.train_model(training_data)
            
            elif isinstance(system, AdvancedAutocomplete4):
                system.add_data_source("queries", [
                    ("python programming", 100),
                    ("machine learning", 80),
                    ("restaurant search", 60),
                    ("food delivery", 70)
                ])
            
            # Test suggestions
            for prefix, expected in test_queries[:2]:  # Test first 2 queries
                if system_name == "Multi-modal":
                    suggestions = system.get_multi_modal_suggestions(prefix, context1, max_suggestions=3)
                elif system_name == "Adaptive Learning":
                    suggestions = system.get_adaptive_suggestions(prefix, context1, max_suggestions=3)
                else:
                    suggestions = system.get_suggestions(prefix, context1, max_suggestions=3)
                
                print(f"  '{prefix}' -> {[s.text for s in suggestions]}")
        
        except Exception as e:
            print(f"  Error: {e}")


def demonstrate_personalization():
    """Demonstrate personalization features"""
    print("\n=== Personalization Demo ===")
    
    system = AdvancedAutocomplete1()
    
    # Create different user contexts
    tech_user = SearchContext(
        user_id="tech_user",
        session_id="session1",
        timestamp=time.time(),
        previous_queries=["python", "programming", "coding"]
    )
    
    food_user = SearchContext(
        user_id="food_user", 
        session_id="session2",
        timestamp=time.time(),
        previous_queries=["restaurant", "pizza", "cooking"]
    )
    
    # Add user-specific queries
    tech_queries = [
        "python programming", "python tutorial", "python libraries",
        "machine learning", "data science", "artificial intelligence"
    ]
    
    food_queries = [
        "pizza delivery", "pizza recipe", "pizza places",
        "restaurant reviews", "cooking tips", "food delivery"
    ]
    
    print("Building user profiles...")
    
    # Add queries for tech user
    for query in tech_queries:
        system.add_query(query, tech_user)
    
    # Add queries for food user  
    for query in food_queries:
        system.add_query(query, food_user)
    
    # Test personalized suggestions
    test_prefix = "p"
    
    print(f"\nSuggestions for prefix '{test_prefix}':")
    
    tech_suggestions = system.get_suggestions(test_prefix, tech_user, 5)
    food_suggestions = system.get_suggestions(test_prefix, food_user, 5)
    
    print(f"  Tech user: {[s.text for s in tech_suggestions]}")
    print(f"  Food user: {[s.text for s in food_suggestions]}")
    
    # Show suggestion sources
    print(f"\nTech user suggestion sources:")
    for suggestion in tech_suggestions:
        print(f"    '{suggestion.text}' -> {suggestion.source} (score: {suggestion.score:.2f})")


def demonstrate_real_time_learning():
    """Demonstrate real-time learning capabilities"""
    print("\n=== Real-time Learning Demo ===")
    
    system = AdvancedAutocomplete3()
    
    context = SearchContext(
        user_id="learning_user",
        session_id="session1", 
        timestamp=time.time()
    )
    
    # Initial suggestions (before learning)
    initial_suggestions = ["program", "programming", "progress", "project", "problem"]
    
    print("Initial suggestions for 'pro':")
    print(f"  {initial_suggestions}")
    
    # Simulate user interactions
    interactions = [
        ("pro", "programming", ["program", "programming", "progress", "project", "problem"]),
        ("pro", "programming", ["program", "programming", "progress", "project", "problem"]),
        ("pro", "project", ["program", "programming", "progress", "project", "problem"]),
        ("pro", "programming", ["program", "programming", "progress", "project", "problem"]),
    ]
    
    print(f"\nSimulating user interactions:")
    
    for prefix, selected, shown in interactions:
        system.record_interaction(prefix, selected, shown, context)
        print(f"  User selected '{selected}' from suggestions for '{prefix}'")
    
    # Show adapted suggestions
    adapted_suggestions = system.get_adaptive_suggestions("pro", context, 5)
    
    print(f"\nAdapted suggestions after learning:")
    print(f"  {[s.text for s in adapted_suggestions]}")
    
    # Show weight changes
    user_weights = system.user_weights[context.user_id]
    print(f"\nLearned user weights: {dict(user_weights)}")


def benchmark_advanced_systems():
    """Benchmark advanced autocomplete systems"""
    print("\n=== Benchmarking Advanced Systems ===")
    
    import random
    import string
    
    # Generate test data
    def generate_queries(count: int) -> List[str]:
        queries = []
        categories = ["python", "java", "machine", "food", "restaurant", "travel"]
        
        for _ in range(count):
            category = random.choice(categories)
            suffix = ''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 8)))
            queries.append(f"{category} {suffix}")
        
        return queries
    
    test_queries = generate_queries(1000)
    test_contexts = [
        SearchContext(
            user_id=f"user{i}",
            session_id=f"session{i}",
            timestamp=time.time()
        ) for i in range(100)
    ]
    
    systems = [
        ("Context-Aware", AdvancedAutocomplete1()),
        ("Multi-modal", AdvancedAutocomplete4()),
    ]
    
    for system_name, system in systems:
        print(f"\n{system_name}:")
        
        # Measure setup time
        start_time = time.time()
        
        if isinstance(system, AdvancedAutocomplete1):
            for query, context in zip(test_queries, test_contexts):
                system.add_query(query, context)
        elif isinstance(system, AdvancedAutocomplete4):
            system.add_data_source("queries", [(q, 1.0) for q in test_queries])
        
        setup_time = time.time() - start_time
        
        # Measure suggestion time
        test_prefixes = ["p", "py", "java", "mac", "foo"]
        start_time = time.time()
        
        for prefix in test_prefixes:
            context = random.choice(test_contexts)
            if system_name == "Multi-modal":
                system.get_multi_modal_suggestions(prefix, context, max_suggestions=5)
            else:
                system.get_suggestions(prefix, context, max_suggestions=5)
        
        suggestion_time = (time.time() - start_time) / len(test_prefixes)
        
        print(f"  Setup {len(test_queries)} queries: {setup_time*1000:.2f}ms")
        print(f"  Average suggestion time: {suggestion_time*1000:.2f}ms")


if __name__ == "__main__":
    test_advanced_autocomplete()
    demonstrate_personalization()
    demonstrate_real_time_learning()
    benchmark_advanced_systems()

"""
Advanced Autocomplete System demonstrates sophisticated autocomplete implementations:

1. Context-Aware with User Behavior - Track user behavior, context, and personalization
2. ML-Inspired Feature Engineering - Use feature-based scoring with user profiles
3. Real-time Learning with Adaptive Weights - Continuously adapt based on interactions
4. Multi-modal with Performance Optimization - Combine multiple data sources with caching

Each approach showcases different aspects of production autocomplete systems:
- Personalization and user behavior tracking
- Machine learning integration
- Real-time adaptation and learning
- Performance optimization and multi-source integration

These implementations provide foundations for building scalable, intelligent
autocomplete systems for various applications.
"""

