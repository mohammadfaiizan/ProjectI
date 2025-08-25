"""
Multi-Pattern Search - Multiple Approaches
Difficulty: Medium

Implement efficient multi-pattern string searching algorithms.
Search for multiple patterns simultaneously in a given text.
"""

from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, deque

class TrieNode:
    """Enhanced Trie node for multi-pattern search"""
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.patterns = []  # Store patterns ending at this node
        self.failure_link = None  # For Aho-Corasick
        self.output_link = None   # For Aho-Corasick

class MultiPatternSearcher:
    
    def __init__(self):
        self.patterns = []
        self.text = ""
    
    def naive_multi_search(self, text: str, patterns: List[str]) -> Dict[str, List[int]]:
        """
        Approach 1: Naive Multi-pattern Search
        
        Search for each pattern individually using naive algorithm.
        
        Time: O(|text| * Σ|pattern_i|)
        Space: O(1)
        """
        results = defaultdict(list)
        
        for pattern in patterns:
            # Naive search for each pattern
            for i in range(len(text) - len(pattern) + 1):
                match = True
                for j in range(len(pattern)):
                    if text[i + j] != pattern[j]:
                        match = False
                        break
                
                if match:
                    results[pattern].append(i)
        
        return dict(results)
    
    def trie_based_search(self, text: str, patterns: List[str]) -> Dict[str, List[int]]:
        """
        Approach 2: Trie-based Multi-pattern Search
        
        Build trie from patterns and search text.
        
        Time: O(Σ|pattern_i| + |text| * max_pattern_length)
        Space: O(Σ|pattern_i|)
        """
        # Build trie
        root = TrieNode()
        
        for pattern in patterns:
            node = root
            for char in pattern:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_end = True
            node.patterns.append(pattern)
        
        results = defaultdict(list)
        
        # Search text starting from each position
        for i in range(len(text)):
            node = root
            j = i
            
            while j < len(text) and text[j] in node.children:
                node = node.children[text[j]]
                if node.is_end:
                    for pattern in node.patterns:
                        results[pattern].append(i)
                j += 1
        
        return dict(results)
    
    def aho_corasick_search(self, text: str, patterns: List[str]) -> Dict[str, List[int]]:
        """
        Approach 3: Aho-Corasick Algorithm
        
        Efficient multi-pattern search with failure function.
        
        Time: O(Σ|pattern_i| + |text| + total_matches)
        Space: O(Σ|pattern_i|)
        """
        if not patterns:
            return {}
        
        # Build trie
        root = TrieNode()
        
        for pattern in patterns:
            node = root
            for char in pattern:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_end = True
            node.patterns.append(pattern)
        
        # Build failure links (KMP-like)
        queue = deque()
        
        # Initialize failure links for depth 1
        for child in root.children.values():
            child.failure_link = root
            queue.append(child)
        
        # Build failure links for deeper levels
        while queue:
            current = queue.popleft()
            
            for char, child in current.children.items():
                queue.append(child)
                
                # Find failure link
                failure = current.failure_link
                while failure and char not in failure.children:
                    failure = failure.failure_link
                
                if failure and char in failure.children:
                    child.failure_link = failure.children[char]
                else:
                    child.failure_link = root
                
                # Set output link
                if child.failure_link.is_end:
                    child.output_link = child.failure_link
                else:
                    child.output_link = child.failure_link.output_link
        
        # Search text
        results = defaultdict(list)
        current = root
        
        for i, char in enumerate(text):
            # Follow failure links until we find a valid transition
            while current and char not in current.children:
                current = current.failure_link
            
            if current and char in current.children:
                current = current.children[char]
                
                # Check for matches at current node
                node = current
                while node:
                    if node.is_end:
                        for pattern in node.patterns:
                            start_pos = i - len(pattern) + 1
                            results[pattern].append(start_pos)
                    node = node.output_link
            else:
                current = root
        
        return dict(results)
    
    def suffix_array_search(self, text: str, patterns: List[str]) -> Dict[str, List[int]]:
        """
        Approach 4: Suffix Array based Search
        
        Build suffix array and search patterns using binary search.
        
        Time: O(|text|log|text| + Σ|pattern_i|*log|text|)
        Space: O(|text|)
        """
        # Build suffix array
        suffixes = [(text[i:], i) for i in range(len(text))]
        suffixes.sort()
        
        suffix_array = [pos for _, pos in suffixes]
        
        results = defaultdict(list)
        
        for pattern in patterns:
            # Binary search for pattern in suffix array
            positions = self._binary_search_pattern(text, suffix_array, pattern)
            results[pattern] = positions
        
        return dict(results)
    
    def _binary_search_pattern(self, text: str, suffix_array: List[int], pattern: str) -> List[int]:
        """Binary search for pattern in suffix array"""
        def starts_with_pattern(suffix_pos: int) -> bool:
            return text[suffix_pos:].startswith(pattern)
        
        # Find first occurrence
        left, right = 0, len(suffix_array)
        while left < right:
            mid = (left + right) // 2
            suffix = text[suffix_array[mid]:]
            if suffix < pattern:
                left = mid + 1
            else:
                right = mid
        
        first = left
        
        # Find last occurrence
        left, right = 0, len(suffix_array)
        while left < right:
            mid = (left + right) // 2
            suffix = text[suffix_array[mid]:]
            if suffix <= pattern or (suffix.startswith(pattern) and len(suffix) >= len(pattern)):
                left = mid + 1
            else:
                right = mid
        
        last = left
        
        # Collect all valid positions
        positions = []
        for i in range(first, last):
            if starts_with_pattern(suffix_array[i]):
                positions.append(suffix_array[i])
        
        return sorted(positions)
    
    def rolling_hash_search(self, text: str, patterns: List[str]) -> Dict[str, List[int]]:
        """
        Approach 5: Rolling Hash Multi-pattern Search
        
        Use rolling hash for efficient pattern matching.
        
        Time: O(|text| + Σ|pattern_i|)
        Space: O(number_of_patterns)
        """
        if not patterns:
            return {}
        
        base = 256
        mod = 10**9 + 7
        
        # Calculate pattern hashes
        pattern_hashes = {}
        for pattern in patterns:
            pattern_hash = 0
            for char in pattern:
                pattern_hash = (pattern_hash * base + ord(char)) % mod
            pattern_hashes[pattern] = pattern_hash
        
        results = defaultdict(list)
        
        # Group patterns by length for efficiency
        patterns_by_length = defaultdict(list)
        for pattern in patterns:
            patterns_by_length[len(pattern)].append(pattern)
        
        # Search for each length group
        for length, length_patterns in patterns_by_length.items():
            if length > len(text):
                continue
            
            # Calculate hash for first window
            text_hash = 0
            h = 1  # base^(length-1)
            
            for i in range(length - 1):
                h = (h * base) % mod
            
            for i in range(length):
                text_hash = (text_hash * base + ord(text[i])) % mod
            
            # Check first window
            for pattern in length_patterns:
                if text_hash == pattern_hashes[pattern] and text[:length] == pattern:
                    results[pattern].append(0)
            
            # Roll hash through text
            for i in range(length, len(text)):
                # Remove leading character
                text_hash = (text_hash - ord(text[i - length]) * h) % mod
                # Add trailing character
                text_hash = (text_hash * base + ord(text[i])) % mod
                
                # Handle negative hash
                if text_hash < 0:
                    text_hash += mod
                
                # Check for pattern matches
                start_pos = i - length + 1
                for pattern in length_patterns:
                    if (text_hash == pattern_hashes[pattern] and 
                        text[start_pos:start_pos + length] == pattern):
                        results[pattern].append(start_pos)
        
        return dict(results)
    
    def two_way_search(self, text: str, patterns: List[str]) -> Dict[str, List[int]]:
        """
        Approach 6: Two-Way Algorithm for Multiple Patterns
        
        Apply Two-Way string matching algorithm for each pattern.
        
        Time: O(|text| + Σ|pattern_i|)
        Space: O(max_pattern_length)
        """
        results = defaultdict(list)
        
        for pattern in patterns:
            # Apply Two-Way algorithm
            positions = self._two_way_single_pattern(text, pattern)
            results[pattern] = positions
        
        return dict(results)
    
    def _two_way_single_pattern(self, text: str, pattern: str) -> List[int]:
        """Two-Way algorithm for single pattern"""
        if not pattern:
            return []
        
        # Find critical factorization
        critical_pos = self._find_critical_factorization(pattern)
        
        # Search using Two-Way approach
        positions = []
        i = 0
        
        while i <= len(text) - len(pattern):
            # Check from critical position
            j = critical_pos
            while j < len(pattern) and text[i + j] == pattern[j]:
                j += 1
            
            if j == len(pattern):
                # Found match, now check left part
                k = critical_pos - 1
                while k >= 0 and text[i + k] == pattern[k]:
                    k -= 1
                
                if k < 0:
                    positions.append(i)
            
            # Advance position
            i += max(1, j - critical_pos)
        
        return positions
    
    def _find_critical_factorization(self, pattern: str) -> int:
        """Find critical factorization for Two-Way algorithm"""
        # Simplified implementation - return middle position
        return len(pattern) // 2


def test_multi_pattern_search():
    """Test multi-pattern search algorithms"""
    print("=== Testing Multi-Pattern Search ===")
    
    searcher = MultiPatternSearcher()
    
    test_cases = [
        # Basic case
        ("abcdefghijk", ["abc", "def", "ghi"], 
         {"abc": [0], "def": [3], "ghi": [6]}),
        
        # Overlapping patterns
        ("aaaa", ["aa", "aaa"], 
         {"aa": [0, 1, 2], "aaa": [0, 1]}),
        
        # No matches
        ("hello world", ["xyz", "123"], 
         {}),
        
        # Complex patterns
        ("abcabcabc", ["abc", "bca", "cab"], 
         {"abc": [0, 3, 6], "bca": [1, 4], "cab": [2, 5]}),
        
        # Single character patterns
        ("abcabc", ["a", "b", "c"], 
         {"a": [0, 3], "b": [1, 4], "c": [2, 5]}),
    ]
    
    algorithms = [
        ("Naive", searcher.naive_multi_search),
        ("Trie-based", searcher.trie_based_search),
        ("Aho-Corasick", searcher.aho_corasick_search),
        ("Suffix Array", searcher.suffix_array_search),
        ("Rolling Hash", searcher.rolling_hash_search),
        ("Two-Way", searcher.two_way_search),
    ]
    
    for i, (text, patterns, expected) in enumerate(test_cases):
        print(f"\nTest Case {i+1}: Text='{text}', Patterns={patterns}")
        print(f"Expected: {expected}")
        
        for name, algorithm in algorithms:
            try:
                result = algorithm(text, patterns)
                # Normalize empty results
                result = {k: v for k, v in result.items() if v}
                status = "✓" if result == expected else "✗"
                print(f"  {name:12}: {result} {status}")
            except Exception as e:
                print(f"  {name:12}: Error - {e}")


def demonstrate_aho_corasick():
    """Demonstrate Aho-Corasick algorithm construction"""
    print("\n=== Aho-Corasick Algorithm Demo ===")
    
    patterns = ["he", "she", "his", "hers"]
    text = "ushers"
    
    print(f"Patterns: {patterns}")
    print(f"Text: '{text}'")
    
    # Build trie
    root = TrieNode()
    print(f"\n1. Building Trie:")
    
    for pattern in patterns:
        print(f"   Inserting '{pattern}'")
        node = root
        for char in pattern:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
        node.patterns.append(pattern)
    
    # Build failure links
    print(f"\n2. Building Failure Links:")
    queue = deque()
    
    # Initialize first level
    for char, child in root.children.items():
        child.failure_link = root
        queue.append(child)
        print(f"   Node '{char}' -> failure link to root")
    
    # Build deeper levels
    while queue:
        current = queue.popleft()
        
        for char, child in current.children.items():
            queue.append(child)
            
            # Find failure link
            failure = current.failure_link
            while failure and char not in failure.children:
                failure = failure.failure_link
            
            if failure and char in failure.children:
                child.failure_link = failure.children[char]
                print(f"   Node path ending with '{char}' -> failure link found")
            else:
                child.failure_link = root
                print(f"   Node path ending with '{char}' -> failure link to root")
    
    # Search demonstration
    print(f"\n3. Searching Text:")
    searcher = MultiPatternSearcher()
    results = searcher.aho_corasick_search(text, patterns)
    
    for pattern, positions in results.items():
        print(f"   Pattern '{pattern}' found at positions: {positions}")


def benchmark_algorithms():
    """Benchmark multi-pattern search algorithms"""
    print("\n=== Benchmarking Algorithms ===")
    
    import time
    import random
    import string
    
    searcher = MultiPatternSearcher()
    
    # Generate test data
    def generate_text(length: int) -> str:
        return ''.join(random.choices(string.ascii_lowercase[:5], k=length))
    
    def generate_patterns(count: int, avg_length: int) -> List[str]:
        patterns = []
        for _ in range(count):
            length = max(1, avg_length + random.randint(-1, 1))
            pattern = ''.join(random.choices(string.ascii_lowercase[:5], k=length))
            patterns.append(pattern)
        return list(set(patterns))  # Remove duplicates
    
    test_scenarios = [
        ("Small", generate_text(100), generate_patterns(5, 3)),
        ("Medium", generate_text(1000), generate_patterns(20, 4)),
        ("Large", generate_text(5000), generate_patterns(50, 5)),
    ]
    
    algorithms = [
        ("Naive", searcher.naive_multi_search),
        ("Trie", searcher.trie_based_search),
        ("Aho-Corasick", searcher.aho_corasick_search),
        ("Rolling Hash", searcher.rolling_hash_search),
    ]
    
    for scenario_name, text, patterns in test_scenarios:
        print(f"\n--- {scenario_name} Dataset ---")
        print(f"Text length: {len(text)}, Patterns: {len(patterns)}, Avg pattern length: {sum(len(p) for p in patterns)/len(patterns):.1f}")
        
        for algorithm_name, algorithm in algorithms:
            start_time = time.time()
            
            # Run multiple times for better measurement
            iterations = 5 if scenario_name != "Large" else 1
            total_matches = 0
            
            for _ in range(iterations):
                result = algorithm(text, patterns)
                total_matches = sum(len(positions) for positions in result.values())
            
            end_time = time.time()
            avg_time = (end_time - start_time) / iterations
            
            print(f"  {algorithm_name:12}: {avg_time*1000:.2f}ms ({total_matches} total matches)")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    searcher = MultiPatternSearcher()
    
    # Application 1: Log analysis
    print("1. Log File Analysis:")
    
    log_content = """
    2024-01-01 10:30:15 ERROR: Database connection failed
    2024-01-01 10:30:16 WARNING: High memory usage detected
    2024-01-01 10:30:17 ERROR: Authentication failed for user
    2024-01-01 10:30:18 INFO: System recovery initiated
    2024-01-01 10:30:19 ERROR: Network timeout occurred
    """
    
    log_patterns = ["ERROR", "WARNING", "FAILED", "timeout"]
    
    print(f"   Searching for: {log_patterns}")
    
    log_results = searcher.aho_corasick_search(log_content, log_patterns)
    
    for pattern, positions in log_results.items():
        print(f"     '{pattern}': found {len(positions)} times")
    
    # Application 2: DNA sequence analysis
    print(f"\n2. DNA Sequence Analysis:")
    
    dna_sequence = "ATCGATCGATCGTAGCTAGCTAGCTACGATCGATCG"
    genetic_markers = ["ATCG", "TAGC", "CGAT", "GCTA"]
    
    print(f"   DNA sequence length: {len(dna_sequence)}")
    print(f"   Genetic markers: {genetic_markers}")
    
    dna_results = searcher.aho_corasick_search(dna_sequence, genetic_markers)
    
    for marker, positions in dna_results.items():
        print(f"     Marker '{marker}': positions {positions}")
    
    # Application 3: Keyword detection in text
    print(f"\n3. Keyword Detection:")
    
    document = """
    Machine learning and artificial intelligence are transforming
    technology. Deep learning models and neural networks enable
    advanced pattern recognition and natural language processing.
    """
    
    keywords = ["machine", "learning", "intelligence", "neural", "pattern"]
    
    print(f"   Document length: {len(document)} characters")
    print(f"   Keywords: {keywords}")
    
    keyword_results = searcher.aho_corasick_search(document.lower(), keywords)
    
    for keyword, positions in keyword_results.items():
        print(f"     '{keyword}': found at positions {positions}")
    
    # Application 4: Network intrusion detection
    print(f"\n4. Network Intrusion Detection:")
    
    network_traffic = "GET /admin/login.php?user=admin&pass=123456 HTTP/1.1"
    suspicious_patterns = ["admin", "login", "pass", "123", "php"]
    
    print(f"   Network request: {network_traffic}")
    print(f"   Suspicious patterns: {suspicious_patterns}")
    
    intrusion_results = searcher.aho_corasick_search(network_traffic.lower(), suspicious_patterns)
    
    risk_score = len(intrusion_results)
    print(f"   Risk patterns found: {len(intrusion_results)}")
    print(f"   Risk score: {risk_score}/10")
    
    for pattern, positions in intrusion_results.items():
        print(f"     Pattern '{pattern}': detected")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    searcher = MultiPatternSearcher()
    
    edge_cases = [
        # Empty cases
        ("", [], "Empty text and patterns"),
        ("hello", [], "Empty patterns"),
        ("", ["pattern"], "Empty text"),
        
        # Single character
        ("a", ["a"], "Single character match"),
        ("a", ["b"], "Single character no match"),
        
        # Overlapping patterns
        ("aaa", ["a", "aa", "aaa"], "Nested patterns"),
        
        # Duplicate patterns
        ("hello", ["hello", "hello"], "Duplicate patterns"),
        
        # Very long patterns
        ("short", ["verylongpattern"], "Pattern longer than text"),
        
        # Special characters
        ("a.b*c", [".", "*"], "Special characters"),
    ]
    
    for text, patterns, description in edge_cases:
        print(f"\n{description}:")
        print(f"  Text: '{text}'")
        print(f"  Patterns: {patterns}")
        
        try:
            result = searcher.aho_corasick_search(text, patterns)
            print(f"  Result: {result}")
        except Exception as e:
            print(f"  Error: {e}")


def analyze_complexity():
    """Analyze time and space complexity"""
    print("\n=== Complexity Analysis ===")
    
    complexity_analysis = [
        ("Naive Multi-Search",
         "Time: O(|text| * Σ|pattern_i|) - check each pattern at each position",
         "Space: O(1) - constant extra space"),
        
        ("Trie-based Search",
         "Time: O(Σ|pattern_i| + |text| * max_pattern_length)",
         "Space: O(Σ|pattern_i|) - trie storage"),
        
        ("Aho-Corasick",
         "Time: O(Σ|pattern_i| + |text| + total_matches)",
         "Space: O(Σ|pattern_i|) - automaton storage"),
        
        ("Suffix Array",
         "Time: O(|text|log|text| + Σ|pattern_i|*log|text|)",
         "Space: O(|text|) - suffix array storage"),
        
        ("Rolling Hash",
         "Time: O(|text| + Σ|pattern_i|) - linear scanning",
         "Space: O(number_of_patterns) - hash storage"),
        
        ("Two-Way",
         "Time: O(|text| + Σ|pattern_i|) - optimal for each pattern",
         "Space: O(max_pattern_length) - preprocessing space"),
    ]
    
    print("Algorithm Analysis:")
    for algorithm, time_complexity, space_complexity in complexity_analysis:
        print(f"\n{algorithm}:")
        print(f"  {time_complexity}")
        print(f"  {space_complexity}")
    
    print(f"\nRecommendations:")
    print(f"  • Use Aho-Corasick for many patterns (optimal)")
    print(f"  • Use Trie-based for moderate number of patterns")
    print(f"  • Use Rolling Hash for patterns of similar length")
    print(f"  • Use Naive for very few short patterns")


if __name__ == "__main__":
    test_multi_pattern_search()
    demonstrate_aho_corasick()
    benchmark_algorithms()
    demonstrate_real_world_applications()
    test_edge_cases()
    analyze_complexity()

"""
Multi-Pattern Search demonstrates comprehensive pattern matching algorithms:

1. Naive Multi-Search - Simple brute force approach for each pattern
2. Trie-based Search - Build pattern trie and search from each text position
3. Aho-Corasick - Optimal multi-pattern search with failure function
4. Suffix Array - Binary search approach using suffix array structure
5. Rolling Hash - Hash-based matching grouped by pattern length
6. Two-Way Algorithm - Linear time algorithm applied to each pattern

Each approach offers different trade-offs between preprocessing time,
search efficiency, and memory usage for various multi-pattern scenarios.
"""
