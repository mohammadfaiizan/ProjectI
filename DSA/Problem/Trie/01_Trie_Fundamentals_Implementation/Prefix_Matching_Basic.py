"""
Prefix Matching Basic - Multiple Approaches
Difficulty: Easy

Implement various prefix matching algorithms and data structures.
Focus on efficient prefix-based operations for different use cases.
"""

from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, deque
import bisect

class TrieNode:
    """Standard Trie Node for prefix matching"""
    def __init__(self):
        self.children: Dict[str, 'TrieNode'] = {}
        self.is_end_of_word: bool = False
        self.words_ending_here: List[str] = []  # Store actual words for retrieval

class PrefixMatcherTrie:
    """
    Approach 1: Trie-based Prefix Matching
    
    Classic trie implementation optimized for prefix operations.
    
    Time Complexity:
    - Insert: O(m) where m is word length
    - Search prefix: O(p) where p is prefix length
    - Get matches: O(k) where k is number of matches
    
    Space Complexity: O(ALPHABET_SIZE * N * M)
    """
    
    def __init__(self):
        self.root = TrieNode()
        self.word_count = 0
    
    def insert(self, word: str) -> None:
        """Insert a word into the trie"""
        node = self.root
        
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        if not node.is_end_of_word:
            self.word_count += 1
            node.words_ending_here.append(word)
        
        node.is_end_of_word = True
    
    def has_prefix(self, prefix: str) -> bool:
        """Check if any word starts with the given prefix"""
        node = self.root
        
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        
        return True
    
    def get_words_with_prefix(self, prefix: str) -> List[str]:
        """Get all words that start with the given prefix"""
        node = self.root
        
        # Navigate to prefix node
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        
        # Collect all words from this subtree
        words = []
        
        def dfs(node: TrieNode, current_prefix: str):
            if node.is_end_of_word:
                words.extend(node.words_ending_here)
            
            for char, child in node.children.items():
                dfs(child, current_prefix + char)
        
        dfs(node, prefix)
        return words
    
    def count_words_with_prefix(self, prefix: str) -> int:
        """Count how many words start with the given prefix"""
        return len(self.get_words_with_prefix(prefix))
    
    def find_shortest_unique_prefix(self, word: str) -> str:
        """Find the shortest prefix that uniquely identifies the word"""
        for i in range(1, len(word) + 1):
            prefix = word[:i]
            matches = self.get_words_with_prefix(prefix)
            if len(matches) == 1:
                return prefix
        return word
    
    def get_all_prefixes(self, word: str) -> List[str]:
        """Get all prefixes of a word that exist in the trie"""
        prefixes = []
        
        for i in range(1, len(word) + 1):
            prefix = word[:i]
            if self.has_prefix(prefix):
                prefixes.append(prefix)
        
        return prefixes

class PrefixMatcherSorted:
    """
    Approach 2: Sorted Array with Binary Search
    
    Keep words sorted and use binary search for prefix matching.
    
    Time Complexity:
    - Insert: O(n) for maintaining sorted order
    - Search prefix: O(log n + k) where k is number of matches
    
    Space Complexity: O(n * m) where n is number of words, m is average length
    """
    
    def __init__(self):
        self.words: List[str] = []
    
    def insert(self, word: str) -> None:
        """Insert word maintaining sorted order"""
        bisect.insort(self.words, word)
    
    def has_prefix(self, prefix: str) -> bool:
        """Check if any word starts with the given prefix"""
        return len(self.get_words_with_prefix(prefix)) > 0
    
    def get_words_with_prefix(self, prefix: str) -> List[str]:
        """Get all words with the given prefix using binary search"""
        # Find the first word >= prefix
        start_idx = bisect.bisect_left(self.words, prefix)
        
        # Find words that start with prefix
        matches = []
        for i in range(start_idx, len(self.words)):
            if self.words[i].startswith(prefix):
                matches.append(self.words[i])
            else:
                break  # Since array is sorted, no more matches
        
        return matches
    
    def count_words_with_prefix(self, prefix: str) -> int:
        """Count words with prefix efficiently"""
        # Find first position >= prefix
        start = bisect.bisect_left(self.words, prefix)
        
        # Find first position > prefix (by incrementing last char)
        if prefix:
            # Create upper bound by incrementing last character
            upper_bound = prefix[:-1] + chr(ord(prefix[-1]) + 1)
            end = bisect.bisect_left(self.words, upper_bound)
        else:
            end = len(self.words)
        
        # Count words in range that actually start with prefix
        count = 0
        for i in range(start, min(end, len(self.words))):
            if self.words[i].startswith(prefix):
                count += 1
            else:
                break
        
        return count

class PrefixMatcherHash:
    """
    Approach 3: Hash-based Prefix Storage
    
    Store all possible prefixes in a hash set for O(1) lookup.
    
    Time Complexity:
    - Insert: O(m²) where m is word length (all prefixes)
    - Search prefix: O(1)
    
    Space Complexity: O(n * m²) - stores all prefixes
    """
    
    def __init__(self):
        self.prefixes: Set[str] = set()
        self.prefix_to_words: Dict[str, List[str]] = defaultdict(list)
        self.words: Set[str] = set()
    
    def insert(self, word: str) -> None:
        """Insert word and all its prefixes"""
        if word in self.words:
            return
        
        self.words.add(word)
        
        # Add all prefixes
        for i in range(1, len(word) + 1):
            prefix = word[:i]
            self.prefixes.add(prefix)
            self.prefix_to_words[prefix].append(word)
    
    def has_prefix(self, prefix: str) -> bool:
        """O(1) prefix check"""
        return prefix in self.prefixes
    
    def get_words_with_prefix(self, prefix: str) -> List[str]:
        """Get words with prefix in O(1)"""
        return self.prefix_to_words.get(prefix, [])
    
    def count_words_with_prefix(self, prefix: str) -> int:
        """Count words with prefix in O(1)"""
        return len(self.prefix_to_words.get(prefix, []))

class PrefixMatcherSuffixArray:
    """
    Approach 4: Suffix Array based Prefix Matching
    
    Use suffix array data structure for advanced prefix operations.
    
    Time Complexity:
    - Build: O(n * m * log(n*m))
    - Search: O(log(n*m) + k)
    
    Space Complexity: O(n * m)
    """
    
    def __init__(self):
        self.words: List[str] = []
        self.suffix_array: List[Tuple[str, int, int]] = []  # (suffix, word_idx, pos)
        self.needs_rebuild = True
    
    def insert(self, word: str) -> None:
        """Insert word (requires rebuild of suffix array)"""
        self.words.append(word)
        self.needs_rebuild = True
    
    def _build_suffix_array(self) -> None:
        """Build suffix array from all words"""
        self.suffix_array = []
        
        for word_idx, word in enumerate(self.words):
            for pos in range(len(word)):
                suffix = word[pos:]
                self.suffix_array.append((suffix, word_idx, pos))
        
        # Sort by suffix
        self.suffix_array.sort(key=lambda x: x[0])
        self.needs_rebuild = False
    
    def has_prefix(self, prefix: str) -> bool:
        """Check if prefix exists using suffix array"""
        if self.needs_rebuild:
            self._build_suffix_array()
        
        # Binary search for prefix in suffix array
        left, right = 0, len(self.suffix_array)
        
        while left < right:
            mid = (left + right) // 2
            suffix = self.suffix_array[mid][0]
            
            if suffix.startswith(prefix):
                return True
            elif suffix < prefix:
                left = mid + 1
            else:
                right = mid
        
        return False
    
    def get_words_with_prefix(self, prefix: str) -> List[str]:
        """Get words with prefix using suffix array"""
        if self.needs_rebuild:
            self._build_suffix_array()
        
        words_found = set()
        
        for suffix, word_idx, pos in self.suffix_array:
            if suffix.startswith(prefix):
                # Only add if prefix starts at beginning of word
                if pos == 0:
                    words_found.add(self.words[word_idx])
            elif suffix > prefix:
                break  # Since array is sorted, no more matches
        
        return list(words_found)

class PrefixMatcherAutomaton:
    """
    Approach 5: Finite State Automaton for Pattern Matching
    
    Build an automaton for efficient multi-pattern prefix matching.
    
    Time Complexity:
    - Build: O(total length of all patterns)
    - Search: O(text length + matches)
    
    Space Complexity: O(total length * alphabet size)
    """
    
    def __init__(self):
        self.states = [{}]  # state -> {char -> next_state}
        self.outputs = [set()]  # state -> set of words ending at this state
        self.failure = [0]  # failure function for state transitions
        self.words = []
        self.needs_rebuild = True
    
    def insert(self, word: str) -> None:
        """Insert word into automaton"""
        self.words.append(word)
        self.needs_rebuild = True
    
    def _build_automaton(self) -> None:
        """Build the Aho-Corasick automaton"""
        self.states = [{}]
        self.outputs = [set()]
        self.failure = [0]
        
        # Build trie
        for word in self.words:
            state = 0
            for char in word:
                if char not in self.states[state]:
                    self.states.append({})
                    self.outputs.append(set())
                    self.failure.append(0)
                    self.states[state][char] = len(self.states) - 1
                state = self.states[state][char]
            self.outputs[state].add(word)
        
        # Build failure function
        queue = deque()
        for char, next_state in self.states[0].items():
            queue.append(next_state)
            self.failure[next_state] = 0
        
        while queue:
            state = queue.popleft()
            
            for char, next_state in self.states[state].items():
                queue.append(next_state)
                
                # Find failure state
                failure_state = self.failure[state]
                while failure_state != 0 and char not in self.states[failure_state]:
                    failure_state = self.failure[failure_state]
                
                if char in self.states[failure_state]:
                    self.failure[next_state] = self.states[failure_state][char]
                else:
                    self.failure[next_state] = 0
                
                # Add outputs from failure state
                self.outputs[next_state].update(self.outputs[self.failure[next_state]])
        
        self.needs_rebuild = False
    
    def has_prefix(self, prefix: str) -> bool:
        """Check if prefix matches any word"""
        if self.needs_rebuild:
            self._build_automaton()
        
        state = 0
        for char in prefix:
            while state != 0 and char not in self.states[state]:
                state = self.failure[state]
            
            if char in self.states[state]:
                state = self.states[state][char]
                if self.outputs[state]:  # Found a match
                    return True
        
        return False
    
    def get_words_with_prefix(self, prefix: str) -> List[str]:
        """Get all words that start with prefix"""
        if self.needs_rebuild:
            self._build_automaton()
        
        words_found = []
        
        for word in self.words:
            if word.startswith(prefix):
                words_found.append(word)
        
        return words_found


def test_basic_operations():
    """Test basic prefix matching operations"""
    print("=== Testing Basic Prefix Matching ===")
    
    # Test data
    words = ["apple", "app", "application", "apply", "banana", "band", "bandana", "can", "candy"]
    
    implementations = [
        ("Trie-based", PrefixMatcherTrie),
        ("Sorted Array", PrefixMatcherSorted),
        ("Hash-based", PrefixMatcherHash),
        ("Suffix Array", PrefixMatcherSuffixArray),
        ("Automaton", PrefixMatcherAutomaton),
    ]
    
    for name, MatcherClass in implementations:
        print(f"\n--- Testing {name} ---")
        
        matcher = MatcherClass()
        
        # Insert words
        for word in words:
            matcher.insert(word)
        
        # Test prefix queries
        test_prefixes = ["app", "ban", "can", "xyz", "a"]
        
        for prefix in test_prefixes:
            has_prefix = matcher.has_prefix(prefix)
            matches = matcher.get_words_with_prefix(prefix)
            count = matcher.count_words_with_prefix(prefix)
            
            print(f"  Prefix '{prefix}': exists={has_prefix}, count={count}, matches={matches}")


def test_advanced_operations():
    """Test advanced prefix operations"""
    print("\n=== Testing Advanced Operations ===")
    
    matcher = PrefixMatcherTrie()
    
    # Insert programming terms
    terms = [
        "python", "programming", "program", "project", "print",
        "java", "javascript", "json", "jupyter",
        "algorithm", "array", "api", "application"
    ]
    
    for term in terms:
        matcher.insert(term)
    
    print("Inserted programming terms:", terms)
    
    # Test unique prefix finding
    print(f"\nUnique prefixes:")
    for term in ["python", "programming", "java", "javascript"]:
        unique_prefix = matcher.find_shortest_unique_prefix(term)
        print(f"  '{term}' -> '{unique_prefix}'")
    
    # Test all prefixes of a word
    print(f"\nAll prefixes in trie:")
    for word in ["programming", "javascript"]:
        all_prefixes = matcher.get_all_prefixes(word)
        print(f"  '{word}' -> {all_prefixes}")


def benchmark_implementations():
    """Benchmark different prefix matching implementations"""
    print("\n=== Benchmarking Implementations ===")
    
    import time
    import random
    import string
    
    # Generate test data
    def generate_words(n: int, avg_length: int) -> List[str]:
        words = []
        for _ in range(n):
            length = max(1, avg_length + random.randint(-3, 3))
            word = ''.join(random.choices(string.ascii_lowercase, k=length))
            words.append(word)
        return list(set(words))  # Remove duplicates
    
    test_words = generate_words(1000, 8)
    test_prefixes = [word[:random.randint(1, len(word))] for word in test_words[:100]]
    
    implementations = [
        ("Trie", PrefixMatcherTrie),
        ("Sorted", PrefixMatcherSorted),
        ("Hash", PrefixMatcherHash),
    ]
    
    print(f"Testing with {len(test_words)} words, {len(test_prefixes)} prefix queries")
    
    for name, MatcherClass in implementations:
        start_time = time.time()
        
        matcher = MatcherClass()
        
        # Insert phase
        insert_start = time.time()
        for word in test_words:
            matcher.insert(word)
        insert_time = time.time() - insert_start
        
        # Query phase
        query_start = time.time()
        for prefix in test_prefixes:
            matcher.has_prefix(prefix)
            matcher.get_words_with_prefix(prefix)
        query_time = time.time() - query_start
        
        total_time = time.time() - start_time
        
        print(f"{name:10}: Total={total_time*1000:.1f}ms, Insert={insert_time*1000:.1f}ms, Query={query_time*1000:.1f}ms")


def demonstrate_real_world_applications():
    """Demonstrate real-world prefix matching applications"""
    print("\n=== Real-World Applications ===")
    
    # Application 1: Autocomplete System
    print("1. Autocomplete System:")
    autocomplete = PrefixMatcherTrie()
    
    # Programming language keywords
    keywords = [
        "def", "class", "import", "from", "if", "else", "elif", "for", "while",
        "try", "except", "finally", "with", "as", "return", "yield", "lambda"
    ]
    
    for keyword in keywords:
        autocomplete.insert(keyword)
    
    user_input = "de"
    suggestions = autocomplete.get_words_with_prefix(user_input)
    print(f"   User types '{user_input}' -> suggestions: {suggestions}")
    
    # Application 2: Command Line Completion
    print("\n2. Command Line Completion:")
    commands = PrefixMatcherHash()
    
    cli_commands = [
        "ls", "list", "cd", "copy", "cat", "grep", "find", "locate",
        "ps", "kill", "top", "htop", "df", "du", "mount", "umount"
    ]
    
    for cmd in cli_commands:
        commands.insert(cmd)
    
    partial_command = "l"
    matches = commands.get_words_with_prefix(partial_command)
    print(f"   Partial command '{partial_command}' -> completions: {matches}")
    
    # Application 3: DNS Lookup Optimization
    print("\n3. DNS Domain Matching:")
    dns_matcher = PrefixMatcherSorted()
    
    domains = [
        "google.com", "github.com", "stackoverflow.com", "python.org",
        "docs.python.org", "pypi.org", "numpy.org", "scipy.org"
    ]
    
    for domain in domains:
        dns_matcher.insert(domain)
    
    domain_prefix = "py"
    matching_domains = dns_matcher.get_words_with_prefix(domain_prefix)
    print(f"   Domain prefix '{domain_prefix}' -> matches: {matching_domains}")
    
    # Application 4: File Path Completion
    print("\n4. File Path Completion:")
    file_matcher = PrefixMatcherTrie()
    
    file_paths = [
        "/home/user/documents", "/home/user/downloads", "/home/user/desktop",
        "/usr/bin", "/usr/local/bin", "/var/log", "/etc/config"
    ]
    
    for path in file_paths:
        file_matcher.insert(path)
    
    path_prefix = "/home"
    matching_paths = file_matcher.get_words_with_prefix(path_prefix)
    print(f"   Path prefix '{path_prefix}' -> matches: {matching_paths}")


def test_edge_cases():
    """Test edge cases for prefix matching"""
    print("\n=== Testing Edge Cases ===")
    
    matcher = PrefixMatcherTrie()
    
    edge_cases = [
        "",  # Empty string
        "a",  # Single character
        "aa",  # Repeated characters
        "A",  # Uppercase
        "123",  # Numbers
        "a-b",  # Special characters
    ]
    
    print("Testing edge cases:")
    for word in edge_cases:
        matcher.insert(word)
        print(f"  Inserted: '{word}'")
    
    # Test queries
    test_queries = ["", "a", "A", "12", "xyz"]
    for query in test_queries:
        has = matcher.has_prefix(query)
        matches = matcher.get_words_with_prefix(query)
        print(f"  Query '{query}': has_prefix={has}, matches={matches}")


def analyze_space_complexity():
    """Analyze space complexity of different approaches"""
    print("\n=== Space Complexity Analysis ===")
    
    import sys
    
    # Create test data
    words = ["test", "testing", "tester", "tea", "team"]
    
    implementations = [
        ("Trie", PrefixMatcherTrie),
        ("Hash", PrefixMatcherHash),
        ("Sorted", PrefixMatcherSorted),
    ]
    
    for name, MatcherClass in implementations:
        matcher = MatcherClass()
        
        for word in words:
            matcher.insert(word)
        
        # Estimate memory usage (rough approximation)
        size = sys.getsizeof(matcher)
        print(f"{name:10}: Approximate size = {size} bytes")
        
        # Analyze characteristics
        if hasattr(matcher, 'prefixes'):
            print(f"           Hash approach stores {len(matcher.prefixes)} prefixes")
        elif hasattr(matcher, 'words'):
            print(f"           Sorted approach stores {len(matcher.words)} words")
        elif hasattr(matcher, 'root'):
            print(f"           Trie approach - structure analysis needed")


def demonstrate_pattern_analysis():
    """Demonstrate pattern analysis capabilities"""
    print("\n=== Pattern Analysis Demo ===")
    
    matcher = PrefixMatcherTrie()
    
    # Text processing terms
    text_terms = [
        "preprocessing", "tokenization", "stemming", "lemmatization",
        "parsing", "analysis", "classification", "clustering",
        "sentiment", "semantic", "syntactic", "morphological"
    ]
    
    for term in text_terms:
        matcher.insert(term)
    
    print("Analyzing text processing vocabulary:")
    
    # Find common prefixes
    prefix_analysis = {}
    for length in range(1, 6):  # Check prefixes of length 1-5
        for term in text_terms:
            if len(term) >= length:
                prefix = term[:length]
                count = matcher.count_words_with_prefix(prefix)
                if count > 1:  # Only interesting if shared
                    if prefix not in prefix_analysis:
                        prefix_analysis[prefix] = count
    
    print(f"Common prefixes found:")
    for prefix, count in sorted(prefix_analysis.items(), key=lambda x: x[1], reverse=True):
        words = matcher.get_words_with_prefix(prefix)
        print(f"  '{prefix}' ({count} words): {words}")


if __name__ == "__main__":
    test_basic_operations()
    test_advanced_operations()
    benchmark_implementations()
    demonstrate_real_world_applications()
    test_edge_cases()
    analyze_space_complexity()
    demonstrate_pattern_analysis()

"""
Prefix Matching Basic demonstrates comprehensive prefix matching techniques:

1. Trie-based: Classic tree structure for efficient prefix operations
2. Sorted Array: Binary search approach for space efficiency
3. Hash-based: Fast lookup with higher space usage
4. Suffix Array: Advanced structure for complex pattern matching
5. Automaton: Finite state machine for multi-pattern matching

Each approach offers different trade-offs suitable for various applications
from autocomplete systems to DNS lookup optimization.
"""
