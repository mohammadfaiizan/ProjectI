"""
Spell Checker Implementation - Multiple Approaches
Difficulty: Medium

Implement a comprehensive spell checker that can:
1. Check if a word is spelled correctly
2. Suggest corrections for misspelled words
3. Handle different types of errors (insertion, deletion, substitution, transposition)
4. Provide ranked suggestions based on similarity and frequency
"""

from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, deque
import heapq
import re

class TrieNode:
    """Trie node for spell checker"""
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.word = ""
        self.frequency = 0

class SpellChecker1:
    """
    Approach 1: Basic Trie with Edit Distance
    
    Use trie for dictionary and compute edit distance for suggestions.
    """
    
    def __init__(self):
        self.root = TrieNode()
        self.word_frequencies = {}
    
    def build_dictionary(self, words: List[str], frequencies: List[int] = None) -> None:
        """
        Build dictionary from word list.
        
        Time: O(sum of word lengths)
        Space: O(sum of word lengths)
        """
        if frequencies is None:
            frequencies = [1] * len(words)
        
        for word, freq in zip(words, frequencies):
            self._add_word(word, freq)
    
    def _add_word(self, word: str, frequency: int) -> None:
        """Add word to trie"""
        word = word.lower()
        node = self.root
        
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        node.is_word = True
        node.word = word
        node.frequency = frequency
        self.word_frequencies[word] = frequency
    
    def is_correct(self, word: str) -> bool:
        """
        Check if word is spelled correctly.
        
        Time: O(|word|)
        Space: O(1)
        """
        word = word.lower()
        node = self.root
        
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        
        return node.is_word
    
    def suggest_corrections(self, word: str, max_suggestions: int = 5, max_distance: int = 2) -> List[Tuple[str, int]]:
        """
        Suggest corrections using edit distance.
        
        Time: O(|word| * dictionary_size * max_word_length)
        Space: O(max_suggestions)
        """
        word = word.lower()
        suggestions = []
        
        def dfs(node: TrieNode, current_word: str, remaining_word: str, distance: int):
            """DFS with edit distance calculation"""
            if distance > max_distance:
                return
            
            if not remaining_word:
                if node.is_word and distance > 0:  # Must have some edit distance
                    suggestions.append((node.word, distance, node.frequency))
                return
            
            char = remaining_word[0]
            rest = remaining_word[1:]
            
            # Exact match
            if char in node.children:
                dfs(node.children[char], current_word + char, rest, distance)
            
            # Substitution
            for child_char, child_node in node.children.items():
                if child_char != char:
                    dfs(child_node, current_word + child_char, rest, distance + 1)
            
            # Deletion (skip character in input)
            dfs(node, current_word, rest, distance + 1)
            
            # Insertion (add character)
            for child_char, child_node in node.children.items():
                dfs(child_node, current_word + child_char, remaining_word, distance + 1)
        
        dfs(self.root, "", word, 0)
        
        # Sort by distance, then by frequency
        suggestions.sort(key=lambda x: (x[1], -x[2]))
        
        return [(word, distance) for word, distance, _ in suggestions[:max_suggestions]]
    
    def suggest_by_prefix(self, prefix: str, max_suggestions: int = 5) -> List[str]:
        """
        Suggest words by prefix completion.
        
        Time: O(|prefix| + suggestions * max_word_length)
        Space: O(suggestions)
        """
        prefix = prefix.lower()
        node = self.root
        
        # Navigate to prefix
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        
        # Collect words with DFS
        suggestions = []
        
        def dfs(current_node: TrieNode, current_word: str):
            if len(suggestions) >= max_suggestions:
                return
            
            if current_node.is_word:
                suggestions.append((current_node.word, current_node.frequency))
            
            for char in sorted(current_node.children.keys()):
                dfs(current_node.children[char], current_word + char)
        
        dfs(node, prefix)
        
        # Sort by frequency and return words
        suggestions.sort(key=lambda x: -x[1])
        return [word for word, _ in suggestions]


class SpellChecker2:
    """
    Approach 2: N-gram Based Similarity
    
    Use character n-grams for similarity measurement.
    """
    
    def __init__(self, n: int = 3):
        self.n = n
        self.dictionary = set()
        self.word_frequencies = {}
        self.ngram_index = defaultdict(set)
    
    def build_dictionary(self, words: List[str], frequencies: List[int] = None) -> None:
        """
        Build n-gram index for words.
        
        Time: O(sum of word lengths * n)
        Space: O(sum of word lengths * n)
        """
        if frequencies is None:
            frequencies = [1] * len(words)
        
        for word, freq in zip(words, frequencies):
            word = word.lower()
            self.dictionary.add(word)
            self.word_frequencies[word] = freq
            
            # Build n-grams
            ngrams = self._get_ngrams(word)
            for ngram in ngrams:
                self.ngram_index[ngram].add(word)
    
    def _get_ngrams(self, word: str) -> List[str]:
        """Extract n-grams from word"""
        # Add padding
        padded = '$' * (self.n - 1) + word + '$' * (self.n - 1)
        return [padded[i:i+self.n] for i in range(len(padded) - self.n + 1)]
    
    def is_correct(self, word: str) -> bool:
        """Check if word is in dictionary"""
        return word.lower() in self.dictionary
    
    def suggest_corrections(self, word: str, max_suggestions: int = 5) -> List[Tuple[str, float]]:
        """
        Suggest corrections using n-gram similarity.
        
        Time: O(|word| * n + candidate_words * |word| * n)
        Space: O(candidate_words)
        """
        word = word.lower()
        word_ngrams = set(self._get_ngrams(word))
        
        # Get candidate words from n-gram index
        candidates = set()
        for ngram in word_ngrams:
            candidates.update(self.ngram_index[ngram])
        
        # Calculate similarity scores
        suggestions = []
        
        for candidate in candidates:
            if candidate != word:  # Don't suggest the same word
                candidate_ngrams = set(self._get_ngrams(candidate))
                
                # Jaccard similarity
                intersection = len(word_ngrams & candidate_ngrams)
                union = len(word_ngrams | candidate_ngrams)
                similarity = intersection / union if union > 0 else 0
                
                # Weight by frequency
                weighted_score = similarity * (1 + self.word_frequencies[candidate] / 100)
                
                suggestions.append((candidate, weighted_score))
        
        # Sort by similarity score
        suggestions.sort(key=lambda x: -x[1])
        
        return suggestions[:max_suggestions]


class SpellChecker3:
    """
    Approach 3: Phonetic Similarity (Soundex-like)
    
    Use phonetic encoding for similar-sounding word suggestions.
    """
    
    def __init__(self):
        self.dictionary = set()
        self.word_frequencies = {}
        self.phonetic_index = defaultdict(list)
    
    def build_dictionary(self, words: List[str], frequencies: List[int] = None) -> None:
        """
        Build phonetic index.
        
        Time: O(sum of word lengths)
        Space: O(sum of word lengths)
        """
        if frequencies is None:
            frequencies = [1] * len(words)
        
        for word, freq in zip(words, frequencies):
            word = word.lower()
            self.dictionary.add(word)
            self.word_frequencies[word] = freq
            
            # Create phonetic encoding
            phonetic = self._phonetic_encode(word)
            self.phonetic_index[phonetic].append(word)
    
    def _phonetic_encode(self, word: str) -> str:
        """Create simplified phonetic encoding"""
        # Simplified Soundex-like algorithm
        word = word.lower()
        
        # Remove vowels except first character
        if word:
            result = word[0]
            for char in word[1:]:
                if char not in 'aeiou':
                    result += char
        else:
            result = word
        
        # Replace similar consonants
        replacements = {
            'c': 'k', 'q': 'k', 'x': 'ks',
            'ph': 'f', 'gh': 'g',
            'ck': 'k', 'ch': 'k'
        }
        
        for old, new in replacements.items():
            result = result.replace(old, new)
        
        return result
    
    def is_correct(self, word: str) -> bool:
        """Check spelling"""
        return word.lower() in self.dictionary
    
    def suggest_corrections(self, word: str, max_suggestions: int = 5) -> List[str]:
        """
        Suggest phonetically similar words.
        
        Time: O(|word| + similar_words)
        Space: O(similar_words)
        """
        word = word.lower()
        phonetic = self._phonetic_encode(word)
        
        suggestions = []
        for candidate in self.phonetic_index[phonetic]:
            if candidate != word:
                suggestions.append((candidate, self.word_frequencies[candidate]))
        
        # Sort by frequency
        suggestions.sort(key=lambda x: -x[1])
        
        return [word for word, _ in suggestions[:max_suggestions]]


class SpellChecker4:
    """
    Approach 4: Machine Learning Inspired (Character-level Features)
    
    Use character-level features for similarity scoring.
    """
    
    def __init__(self):
        self.dictionary = set()
        self.word_frequencies = {}
        self.character_weights = {}
    
    def build_dictionary(self, words: List[str], frequencies: List[int] = None) -> None:
        """
        Build dictionary and learn character weights.
        
        Time: O(sum of word lengths)
        Space: O(sum of word lengths)
        """
        if frequencies is None:
            frequencies = [1] * len(words)
        
        # Build character frequency weights
        char_counts = defaultdict(int)
        total_chars = 0
        
        for word, freq in zip(words, frequencies):
            word = word.lower()
            self.dictionary.add(word)
            self.word_frequencies[word] = freq
            
            for char in word:
                char_counts[char] += freq
                total_chars += freq
        
        # Compute inverse frequency weights (rarer characters get higher weight)
        for char, count in char_counts.items():
            self.character_weights[char] = 1.0 / (count / total_chars)
    
    def is_correct(self, word: str) -> bool:
        """Check spelling"""
        return word.lower() in self.dictionary
    
    def suggest_corrections(self, word: str, max_suggestions: int = 5) -> List[Tuple[str, float]]:
        """
        Suggest corrections using character-level features.
        
        Time: O(dictionary_size * max_word_length)
        Space: O(max_suggestions)
        """
        word = word.lower()
        suggestions = []
        
        for candidate in self.dictionary:
            if candidate != word:
                similarity = self._calculate_similarity(word, candidate)
                suggestions.append((candidate, similarity))
        
        # Sort by similarity
        suggestions.sort(key=lambda x: -x[1])
        
        return suggestions[:max_suggestions]
    
    def _calculate_similarity(self, word1: str, word2: str) -> float:
        """Calculate weighted character similarity"""
        # Character-level cosine similarity with weights
        chars1 = defaultdict(int)
        chars2 = defaultdict(int)
        
        for char in word1:
            chars1[char] += 1
        for char in word2:
            chars2[char] += 1
        
        # Compute weighted dot product
        dot_product = 0
        norm1 = norm2 = 0
        
        all_chars = set(chars1.keys()) | set(chars2.keys())
        
        for char in all_chars:
            weight = self.character_weights.get(char, 1.0)
            count1 = chars1[char] * weight
            count2 = chars2[char] * weight
            
            dot_product += count1 * count2
            norm1 += count1 * count1
            norm2 += count2 * count2
        
        # Cosine similarity
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return dot_product / (norm1 ** 0.5 * norm2 ** 0.5)


class SpellChecker5:
    """
    Approach 5: Advanced Edit Distance with Costs
    
    Use weighted edit distance for different types of errors.
    """
    
    def __init__(self):
        self.dictionary = set()
        self.word_frequencies = {}
        
        # Error costs (can be tuned based on typical errors)
        self.substitution_cost = 1.0
        self.insertion_cost = 1.0
        self.deletion_cost = 1.0
        self.transposition_cost = 0.8  # Swapping adjacent characters
    
    def build_dictionary(self, words: List[str], frequencies: List[int] = None) -> None:
        """Build dictionary"""
        if frequencies is None:
            frequencies = [1] * len(words)
        
        for word, freq in zip(words, frequencies):
            word = word.lower()
            self.dictionary.add(word)
            self.word_frequencies[word] = freq
    
    def is_correct(self, word: str) -> bool:
        """Check spelling"""
        return word.lower() in self.dictionary
    
    def suggest_corrections(self, word: str, max_suggestions: int = 5, max_distance: float = 2.0) -> List[Tuple[str, float]]:
        """
        Suggest corrections using weighted edit distance.
        
        Time: O(dictionary_size * |word| * max_word_length)
        Space: O(|word| * max_word_length)
        """
        word = word.lower()
        suggestions = []
        
        for candidate in self.dictionary:
            if candidate != word:
                distance = self._weighted_edit_distance(word, candidate)
                if distance <= max_distance:
                    # Score combines distance and frequency
                    score = 1.0 / (1.0 + distance) * (1.0 + self.word_frequencies[candidate] / 100)
                    suggestions.append((candidate, score))
        
        # Sort by score
        suggestions.sort(key=lambda x: -x[1])
        
        return suggestions[:max_suggestions]
    
    def _weighted_edit_distance(self, word1: str, word2: str) -> float:
        """Calculate weighted edit distance with transposition"""
        m, n = len(word1), len(word2)
        
        # DP table
        dp = [[float('inf')] * (n + 1) for _ in range(m + 1)]
        
        # Base cases
        dp[0][0] = 0
        for i in range(1, m + 1):
            dp[i][0] = dp[i-1][0] + self.deletion_cost
        for j in range(1, n + 1):
            dp[0][j] = dp[0][j-1] + self.insertion_cost
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]  # No cost for match
                else:
                    # Substitution
                    dp[i][j] = min(dp[i][j], dp[i-1][j-1] + self.substitution_cost)
                
                # Insertion
                dp[i][j] = min(dp[i][j], dp[i][j-1] + self.insertion_cost)
                
                # Deletion
                dp[i][j] = min(dp[i][j], dp[i-1][j] + self.deletion_cost)
                
                # Transposition (swap adjacent characters)
                if (i > 1 and j > 1 and 
                    word1[i-1] == word2[j-2] and 
                    word1[i-2] == word2[j-1]):
                    dp[i][j] = min(dp[i][j], dp[i-2][j-2] + self.transposition_cost)
        
        return dp[m][n]


class ComprehensiveSpellChecker:
    """
    Comprehensive spell checker combining multiple approaches
    """
    
    def __init__(self):
        self.trie_checker = SpellChecker1()
        self.ngram_checker = SpellChecker2()
        self.phonetic_checker = SpellChecker3()
        self.advanced_checker = SpellChecker5()
    
    def build_dictionary(self, words: List[str], frequencies: List[int] = None) -> None:
        """Build all dictionaries"""
        self.trie_checker.build_dictionary(words, frequencies)
        self.ngram_checker.build_dictionary(words, frequencies)
        self.phonetic_checker.build_dictionary(words, frequencies)
        self.advanced_checker.build_dictionary(words, frequencies)
    
    def is_correct(self, word: str) -> bool:
        """Check spelling using trie (fastest)"""
        return self.trie_checker.is_correct(word)
    
    def suggest_corrections(self, word: str, max_suggestions: int = 5) -> List[Tuple[str, str]]:
        """
        Get suggestions from all approaches and combine.
        
        Returns list of (word, method) tuples.
        """
        all_suggestions = []
        
        # Get suggestions from each approach
        trie_suggestions = self.trie_checker.suggest_corrections(word, max_suggestions)
        ngram_suggestions = self.ngram_checker.suggest_corrections(word, max_suggestions)
        phonetic_suggestions = self.phonetic_checker.suggest_corrections(word, max_suggestions)
        advanced_suggestions = self.advanced_checker.suggest_corrections(word, max_suggestions)
        
        # Combine and rank suggestions
        suggestion_scores = defaultdict(list)
        
        for suggestion, score in trie_suggestions:
            suggestion_scores[suggestion].append(('edit_distance', score))
        
        for suggestion, score in ngram_suggestions:
            suggestion_scores[suggestion].append(('ngram', score))
        
        for suggestion in phonetic_suggestions:
            suggestion_scores[suggestion].append(('phonetic', 1.0))
        
        for suggestion, score in advanced_suggestions:
            suggestion_scores[suggestion].append(('advanced', score))
        
        # Calculate combined scores
        final_suggestions = []
        for word, scores in suggestion_scores.items():
            # Average score with method diversity bonus
            avg_score = sum(score for _, score in scores) / len(scores)
            diversity_bonus = len(scores) * 0.1  # Bonus for appearing in multiple methods
            final_score = avg_score + diversity_bonus
            
            methods = [method for method, _ in scores]
            final_suggestions.append((word, final_score, methods))
        
        # Sort by combined score
        final_suggestions.sort(key=lambda x: -x[1])
        
        return [(word, '+'.join(methods)) for word, _, methods in final_suggestions[:max_suggestions]]


def test_spell_checkers():
    """Test all spell checker implementations"""
    print("=== Testing Spell Checker Implementations ===")
    
    # Build test dictionary
    dictionary = [
        "hello", "world", "python", "programming", "computer", "science",
        "algorithm", "data", "structure", "machine", "learning", "artificial",
        "intelligence", "software", "development", "coding", "debug", "test"
    ]
    
    frequencies = [100, 90, 150, 120, 80, 70, 60, 85, 75, 95, 110, 65, 55, 130, 140, 125, 45, 88]
    
    # Test cases (misspelled words)
    test_cases = [
        ("helo", "hello"),           # Missing letter
        ("wrold", "world"),          # Transposition
        ("pythn", "python"),         # Missing letter
        ("programing", "programming"), # Missing letter
        ("compter", "computer"),     # Missing letter
        ("algortihm", "algorithm"),  # Transposition
    ]
    
    checkers = [
        ("Trie + Edit Distance", SpellChecker1()),
        ("N-gram Similarity", SpellChecker2()),
        ("Phonetic", SpellChecker3()),
        ("Character Features", SpellChecker4()),
        ("Advanced Edit Distance", SpellChecker5()),
    ]
    
    for name, checker in checkers:
        print(f"\n{name}:")
        checker.build_dictionary(dictionary, frequencies)
        
        for misspelled, expected in test_cases:
            is_correct = checker.is_correct(misspelled)
            suggestions = checker.suggest_corrections(misspelled, 3)
            
            print(f"  '{misspelled}' -> correct: {is_correct}")
            if suggestions:
                suggestion_words = [s[0] if isinstance(s, tuple) else s for s in suggestions]
                found_expected = expected in suggestion_words
                status = "✓" if found_expected else "✗"
                print(f"    Suggestions: {suggestion_words} {status}")
            else:
                print(f"    No suggestions ✗")


def demonstrate_comprehensive_checker():
    """Demonstrate comprehensive spell checker"""
    print("\n=== Comprehensive Spell Checker Demo ===")
    
    # Build comprehensive dictionary
    dictionary = [
        "hello", "world", "python", "programming", "computer", "science",
        "algorithm", "data", "structure", "machine", "learning", "development",
        "software", "engineering", "artificial", "intelligence", "coding", "debug"
    ]
    
    frequencies = [100] * len(dictionary)  # Equal frequencies for simplicity
    
    comprehensive = ComprehensiveSpellChecker()
    comprehensive.build_dictionary(dictionary, frequencies)
    
    test_words = ["helo", "wrold", "pythn", "sofware", "engeneering"]
    
    print("Comprehensive spell checking results:")
    
    for word in test_words:
        print(f"\nInput: '{word}'")
        
        is_correct = comprehensive.is_correct(word)
        print(f"  Correct: {is_correct}")
        
        if not is_correct:
            suggestions = comprehensive.suggest_corrections(word)
            print(f"  Suggestions:")
            for suggestion, methods in suggestions:
                print(f"    '{suggestion}' (detected by: {methods})")


def benchmark_spell_checkers():
    """Benchmark spell checker performance"""
    print("\n=== Benchmarking Spell Checkers ===")
    
    import time
    import random
    import string
    
    # Generate larger dictionary
    def generate_dictionary(size: int) -> List[str]:
        words = []
        for _ in range(size):
            length = random.randint(3, 12)
            word = ''.join(random.choices(string.ascii_lowercase, k=length))
            words.append(word)
        return list(set(words))
    
    dictionary = generate_dictionary(1000)
    frequencies = [random.randint(1, 100) for _ in range(len(dictionary))]
    
    # Generate test words (some correct, some misspelled)
    test_words = random.sample(dictionary, 50)  # Correct words
    misspelled = []
    
    for word in random.sample(dictionary, 20):  # Create misspellings
        if len(word) > 3:
            # Random character substitution
            pos = random.randint(0, len(word) - 1)
            new_char = random.choice(string.ascii_lowercase)
            misspelled_word = word[:pos] + new_char + word[pos+1:]
            misspelled.append(misspelled_word)
    
    test_words.extend(misspelled)
    
    checkers = [
        ("Trie + Edit Distance", SpellChecker1()),
        ("N-gram Similarity", SpellChecker2()),
        ("Phonetic", SpellChecker3()),
    ]
    
    for name, checker in checkers:
        print(f"\n{name}:")
        
        # Measure build time
        start_time = time.time()
        checker.build_dictionary(dictionary, frequencies)
        build_time = time.time() - start_time
        
        # Measure check time
        start_time = time.time()
        for word in test_words:
            checker.is_correct(word)
        check_time = time.time() - start_time
        
        # Measure suggestion time (for misspelled words only)
        start_time = time.time()
        for word in misspelled:
            checker.suggest_corrections(word, 3)
        suggest_time = time.time() - start_time
        
        print(f"  Build: {build_time*1000:.2f}ms")
        print(f"  Check {len(test_words)} words: {check_time*1000:.2f}ms")
        print(f"  Suggest for {len(misspelled)} words: {suggest_time*1000:.2f}ms")


def demonstrate_real_world_usage():
    """Demonstrate real-world spell checker usage"""
    print("\n=== Real-World Usage Demo ===")
    
    # Scenario 1: Text editor spell check
    print("1. Text Editor Spell Check:")
    
    spell_checker = SpellChecker1()
    
    # Common English words dictionary (simplified)
    common_words = [
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
        "hello", "world", "computer", "program", "write", "code", "test", "debug"
    ]
    
    frequencies = [1000, 900, 850, 800, 750] + [100] * (len(common_words) - 5)
    spell_checker.build_dictionary(common_words, frequencies)
    
    user_text = "Helo wrold! I am writting a compter progam to test spellng."
    words = re.findall(r'\b[a-zA-Z]+\b', user_text)
    
    print(f"   User text: '{user_text}'")
    print(f"   Spell check results:")
    
    for word in words:
        is_correct = spell_checker.is_correct(word)
        if not is_correct:
            suggestions = spell_checker.suggest_corrections(word, 3)
            print(f"     '{word}' -> MISSPELLED, suggestions: {[s[0] for s in suggestions]}")
        else:
            print(f"     '{word}' -> OK")
    
    # Scenario 2: Search query correction
    print(f"\n2. Search Query Correction:")
    
    search_checker = SpellChecker2()
    
    search_terms = [
        "python", "programming", "tutorial", "machine", "learning",
        "artificial", "intelligence", "computer", "science", "algorithm",
        "data", "structure", "software", "development", "coding"
    ]
    
    search_frequencies = [150, 120, 100, 110, 105, 80, 75, 140, 90, 85, 95, 88, 130, 125, 115]
    search_checker.build_dictionary(search_terms, search_frequencies)
    
    search_queries = ["pythom", "programing", "machne lerning", "algortihm"]
    
    print(f"   Search query corrections:")
    for query in search_queries:
        words = query.split()
        corrected_words = []
        
        for word in words:
            if search_checker.is_correct(word):
                corrected_words.append(word)
            else:
                suggestions = search_checker.suggest_corrections(word, 1)
                if suggestions:
                    corrected_words.append(suggestions[0][0])
                else:
                    corrected_words.append(word)
        
        corrected_query = ' '.join(corrected_words)
        print(f"     '{query}' -> '{corrected_query}'")


if __name__ == "__main__":
    test_spell_checkers()
    demonstrate_comprehensive_checker()
    benchmark_spell_checkers()
    demonstrate_real_world_usage()

"""
Spell Checker Implementation demonstrates comprehensive spelling correction approaches:

1. Basic Trie + Edit Distance - Standard approach using trie and dynamic programming
2. N-gram Similarity - Character n-gram based similarity for fuzzy matching
3. Phonetic Similarity - Soundex-like phonetic encoding for similar-sounding words
4. Character Features - Machine learning inspired character-level feature extraction
5. Advanced Edit Distance - Weighted edit distance with transposition support
6. Comprehensive Checker - Combines multiple approaches for best results

Each approach offers different strengths for various types of spelling errors
and use cases from simple typo correction to advanced linguistic analysis.
"""

