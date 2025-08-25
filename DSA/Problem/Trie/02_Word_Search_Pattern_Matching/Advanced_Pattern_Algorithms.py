"""
Advanced Pattern Algorithms - Multiple Approaches
Difficulty: Hard

Implementation of advanced string pattern matching algorithms including
Aho-Corasick, Suffix Trees, and other sophisticated pattern matching techniques.
"""

from typing import List, Dict, Set, Tuple, Optional, Union
from collections import defaultdict, deque
import heapq

class TrieNode:
    """Enhanced Trie node for advanced algorithms"""
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.patterns = []
        self.failure_link = None
        self.output_link = None
        self.depth = 0

class SuffixTreeNode:
    """Suffix tree node"""
    def __init__(self, start: int = -1, end: int = -1):
        self.children = {}
        self.start = start
        self.end = end
        self.suffix_link = None
        self.suffix_index = -1

class AdvancedPatternMatcher:
    
    def __init__(self):
        self.text = ""
        self.patterns = []
    
    def aho_corasick_advanced(self, text: str, patterns: List[str]) -> Dict[str, List[Tuple[int, int]]]:
        """
        Approach 1: Advanced Aho-Corasick with Position Tracking
        
        Enhanced Aho-Corasick that tracks start and end positions.
        
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
                node.depth = max(node.depth, len(pattern))
            node.is_end = True
            node.patterns.append(pattern)
        
        # Build failure links
        self._build_failure_links(root)
        
        # Search text with position tracking
        results = defaultdict(list)
        current = root
        
        for i, char in enumerate(text):
            # Follow failure links
            while current != root and char not in current.children:
                current = current.failure_link
            
            if char in current.children:
                current = current.children[char]
                
                # Check for pattern matches
                temp = current
                while temp:
                    if temp.is_end:
                        for pattern in temp.patterns:
                            start_pos = i - len(pattern) + 1
                            end_pos = i
                            results[pattern].append((start_pos, end_pos))
                    temp = temp.output_link
            
        return dict(results)
    
    def _build_failure_links(self, root: TrieNode) -> None:
        """Build failure links for Aho-Corasick automaton"""
        queue = deque()
        
        # Initialize first level
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
                while failure != root and char not in failure.children:
                    failure = failure.failure_link
                
                if char in failure.children:
                    child.failure_link = failure.children[char]
                else:
                    child.failure_link = root
                
                # Set output link
                if child.failure_link.is_end:
                    child.output_link = child.failure_link
                else:
                    child.output_link = child.failure_link.output_link
    
    def suffix_tree_search(self, text: str, patterns: List[str]) -> Dict[str, List[int]]:
        """
        Approach 2: Suffix Tree based Pattern Matching
        
        Build suffix tree and search for patterns.
        
        Time: O(|text|^2) construction + O(Σ|pattern_i| + matches)
        Space: O(|text|^2) worst case
        """
        # Build suffix tree (simplified implementation)
        suffix_tree = self._build_suffix_tree(text)
        
        results = defaultdict(list)
        
        for pattern in patterns:
            positions = self._search_in_suffix_tree(suffix_tree, text, pattern)
            results[pattern] = positions
        
        return dict(results)
    
    def _build_suffix_tree(self, text: str) -> SuffixTreeNode:
        """Build suffix tree (simplified Ukkonen's algorithm)"""
        root = SuffixTreeNode()
        
        # Add each suffix
        for i in range(len(text)):
            current = root
            for j in range(i, len(text)):
                char = text[j]
                
                if char not in current.children:
                    # Create new node
                    new_node = SuffixTreeNode(j, len(text) - 1)
                    new_node.suffix_index = i
                    current.children[char] = new_node
                    break
                else:
                    current = current.children[char]
        
        return root
    
    def _search_in_suffix_tree(self, root: SuffixTreeNode, text: str, pattern: str) -> List[int]:
        """Search pattern in suffix tree"""
        positions = []
        
        def dfs(node: SuffixTreeNode, depth: int):
            if depth >= len(pattern):
                # Found pattern, collect all suffixes
                if node.suffix_index != -1:
                    positions.append(node.suffix_index)
                
                for child in node.children.values():
                    dfs(child, depth)
                return
            
            # Check each child
            for char, child in node.children.items():
                if char == pattern[depth]:
                    dfs(child, depth + 1)
        
        dfs(root, 0)
        return sorted(positions)
    
    def z_algorithm_search(self, text: str, patterns: List[str]) -> Dict[str, List[int]]:
        """
        Approach 3: Z-Algorithm based Pattern Matching
        
        Use Z-algorithm for linear time pattern matching.
        
        Time: O(|text| + Σ|pattern_i|)
        Space: O(max_pattern_length)
        """
        results = defaultdict(list)
        
        for pattern in patterns:
            positions = self._z_algorithm_single_pattern(text, pattern)
            results[pattern] = positions
        
        return dict(results)
    
    def _z_algorithm_single_pattern(self, text: str, pattern: str) -> List[int]:
        """Z-algorithm for single pattern"""
        # Concatenate pattern + separator + text
        s = pattern + '$' + text
        n = len(s)
        
        # Build Z-array
        z = [0] * n
        l = r = 0
        
        for i in range(1, n):
            if i <= r:
                z[i] = min(r - i + 1, z[i - l])
            
            while i + z[i] < n and s[z[i]] == s[i + z[i]]:
                z[i] += 1
            
            if i + z[i] - 1 > r:
                l, r = i, i + z[i] - 1
        
        # Find pattern matches
        positions = []
        pattern_len = len(pattern)
        
        for i in range(pattern_len + 1, n):
            if z[i] == pattern_len:
                positions.append(i - pattern_len - 1)
        
        return positions
    
    def manacher_algorithm(self, text: str) -> List[Tuple[int, int]]:
        """
        Approach 4: Manacher's Algorithm for Palindrome Finding
        
        Find all palindromic substrings efficiently.
        
        Time: O(|text|)
        Space: O(|text|)
        """
        # Preprocess string: insert '#' between characters
        processed = '#'.join('^{}$'.format(text))
        n = len(processed)
        
        # Arrays to store palindrome information
        p = [0] * n  # Length of palindrome centered at i
        center = right = 0
        
        palindromes = []
        
        for i in range(1, n - 1):
            # Mirror of i with respect to center
            mirror = 2 * center - i
            
            if i < right:
                p[i] = min(right - i, p[mirror])
            
            # Try to expand palindrome centered at i
            while processed[i + p[i] + 1] == processed[i - p[i] - 1]:
                p[i] += 1
            
            # If palindrome centered at i extends past right, adjust center and right
            if i + p[i] > right:
                center, right = i, i + p[i]
            
            # Extract palindrome information
            if p[i] > 0:
                start = (i - p[i]) // 2
                length = p[i]
                palindromes.append((start, start + length - 1))
        
        return palindromes
    
    def kmp_multiple_patterns(self, text: str, patterns: List[str]) -> Dict[str, List[int]]:
        """
        Approach 5: Multiple KMP Pattern Matching
        
        Apply KMP algorithm for each pattern.
        
        Time: O(|text| * num_patterns + Σ|pattern_i|)
        Space: O(max_pattern_length)
        """
        results = defaultdict(list)
        
        for pattern in patterns:
            positions = self._kmp_single_pattern(text, pattern)
            results[pattern] = positions
        
        return dict(results)
    
    def _kmp_single_pattern(self, text: str, pattern: str) -> List[int]:
        """KMP algorithm for single pattern"""
        if not pattern:
            return []
        
        # Build failure function
        failure = self._build_kmp_failure_function(pattern)
        
        positions = []
        i = j = 0
        
        while i < len(text):
            if text[i] == pattern[j]:
                i += 1
                j += 1
            
            if j == len(pattern):
                positions.append(i - j)
                j = failure[j - 1]
            elif i < len(text) and text[i] != pattern[j]:
                if j != 0:
                    j = failure[j - 1]
                else:
                    i += 1
        
        return positions
    
    def _build_kmp_failure_function(self, pattern: str) -> List[int]:
        """Build KMP failure function"""
        m = len(pattern)
        failure = [0] * m
        j = 0
        
        for i in range(1, m):
            while j > 0 and pattern[i] != pattern[j]:
                j = failure[j - 1]
            
            if pattern[i] == pattern[j]:
                j += 1
            
            failure[i] = j
        
        return failure
    
    def suffix_array_lcp_search(self, text: str, patterns: List[str]) -> Dict[str, List[int]]:
        """
        Approach 6: Suffix Array with LCP for Pattern Matching
        
        Build suffix array with LCP array for efficient searching.
        
        Time: O(|text|log|text| + Σ|pattern_i|*log|text|)
        Space: O(|text|)
        """
        # Build suffix array
        suffix_array = self._build_suffix_array(text)
        lcp = self._build_lcp_array(text, suffix_array)
        
        results = defaultdict(list)
        
        for pattern in patterns:
            positions = self._search_with_suffix_array(text, suffix_array, pattern)
            results[pattern] = positions
        
        return dict(results)
    
    def _build_suffix_array(self, text: str) -> List[int]:
        """Build suffix array using simple sorting"""
        suffixes = [(text[i:], i) for i in range(len(text))]
        suffixes.sort()
        return [suffix[1] for suffix in suffixes]
    
    def _build_lcp_array(self, text: str, suffix_array: List[int]) -> List[int]:
        """Build LCP (Longest Common Prefix) array"""
        n = len(text)
        lcp = [0] * n
        rank = [0] * n
        
        # Build rank array
        for i in range(n):
            rank[suffix_array[i]] = i
        
        # Build LCP array
        h = 0
        for i in range(n):
            if rank[i] > 0:
                j = suffix_array[rank[i] - 1]
                while i + h < n and j + h < n and text[i + h] == text[j + h]:
                    h += 1
                lcp[rank[i]] = h
                if h > 0:
                    h -= 1
        
        return lcp
    
    def _search_with_suffix_array(self, text: str, suffix_array: List[int], pattern: str) -> List[int]:
        """Search pattern using suffix array and binary search"""
        positions = []
        
        # Binary search for first occurrence
        left, right = 0, len(suffix_array)
        while left < right:
            mid = (left + right) // 2
            suffix = text[suffix_array[mid]:]
            if suffix < pattern:
                left = mid + 1
            else:
                right = mid
        
        # Collect all matching positions
        start = left
        while start < len(suffix_array):
            suffix = text[suffix_array[start]:]
            if suffix.startswith(pattern):
                positions.append(suffix_array[start])
                start += 1
            else:
                break
        
        return sorted(positions)
    
    def approximate_string_matching(self, text: str, pattern: str, max_distance: int) -> List[Tuple[int, int]]:
        """
        Approach 7: Approximate String Matching with Edit Distance
        
        Find pattern matches allowing up to max_distance edits.
        
        Time: O(|text| * |pattern| * max_distance)
        Space: O(|pattern| * max_distance)
        """
        matches = []
        
        for i in range(len(text) - len(pattern) + max_distance + 1):
            # Extract window that could potentially match
            window_end = min(i + len(pattern) + max_distance, len(text))
            window = text[i:window_end]
            
            # Calculate edit distance
            distance = self._edit_distance(window, pattern, max_distance)
            
            if distance <= max_distance:
                matches.append((i, distance))
        
        return matches
    
    def _edit_distance(self, s1: str, s2: str, max_distance: int) -> int:
        """Calculate edit distance with early termination"""
        m, n = len(s1), len(s2)
        
        # Use space-optimized DP
        prev = list(range(n + 1))
        curr = [0] * (n + 1)
        
        for i in range(1, m + 1):
            curr[0] = i
            min_val = i
            
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    curr[j] = prev[j - 1]
                else:
                    curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
                
                min_val = min(min_val, curr[j])
            
            # Early termination if all values exceed max_distance
            if min_val > max_distance:
                return max_distance + 1
            
            prev, curr = curr, prev
        
        return prev[n]


def test_advanced_algorithms():
    """Test advanced pattern matching algorithms"""
    print("=== Testing Advanced Algorithms ===")
    
    matcher = AdvancedPatternMatcher()
    
    text = "abcabcabcabc"
    patterns = ["abc", "bca", "cab", "abcabc"]
    
    print(f"Text: '{text}'")
    print(f"Patterns: {patterns}")
    
    # Test different algorithms
    algorithms = [
        ("Aho-Corasick Advanced", matcher.aho_corasick_advanced),
        ("Suffix Tree", matcher.suffix_tree_search),
        ("Z-Algorithm", matcher.z_algorithm_search),
        ("Multiple KMP", matcher.kmp_multiple_patterns),
        ("Suffix Array LCP", matcher.suffix_array_lcp_search),
    ]
    
    for name, algorithm in algorithms:
        print(f"\n{name}:")
        try:
            result = algorithm(text, patterns)
            for pattern, positions in result.items():
                if isinstance(positions[0], tuple):
                    # Position ranges
                    print(f"  '{pattern}': {positions}")
                else:
                    # Simple positions
                    print(f"  '{pattern}': positions {positions}")
        except Exception as e:
            print(f"  Error: {e}")


def test_palindrome_detection():
    """Test palindrome detection using Manacher's algorithm"""
    print("\n=== Testing Palindrome Detection ===")
    
    matcher = AdvancedPatternMatcher()
    
    test_strings = [
        "racecar",
        "abcba",
        "aabaa",
        "abcdef",
        "aba",
    ]
    
    for text in test_strings:
        print(f"\nText: '{text}'")
        palindromes = matcher.manacher_algorithm(text)
        
        # Filter out single character palindromes for clarity
        significant_palindromes = [(start, end) for start, end in palindromes 
                                 if end - start + 1 > 1]
        
        if significant_palindromes:
            print(f"  Palindromes found:")
            for start, end in significant_palindromes:
                palindrome = text[start:end + 1]
                print(f"    '{palindrome}' at position {start}-{end}")
        else:
            print(f"  No palindromes found (length > 1)")


def test_approximate_matching():
    """Test approximate string matching"""
    print("\n=== Testing Approximate String Matching ===")
    
    matcher = AdvancedPatternMatcher()
    
    text = "programming"
    pattern = "program"
    max_distances = [0, 1, 2]
    
    print(f"Text: '{text}'")
    print(f"Pattern: '{pattern}'")
    
    for max_dist in max_distances:
        print(f"\nMax edit distance: {max_dist}")
        matches = matcher.approximate_string_matching(text, pattern, max_dist)
        
        if matches:
            for pos, distance in matches:
                end_pos = min(pos + len(pattern), len(text))
                matched_text = text[pos:end_pos]
                print(f"  Match at position {pos}: '{matched_text}' (distance: {distance})")
        else:
            print(f"  No matches found")


def demonstrate_suffix_tree():
    """Demonstrate suffix tree construction and search"""
    print("\n=== Suffix Tree Demo ===")
    
    text = "banana"
    patterns = ["ana", "nan", "ban"]
    
    print(f"Text: '{text}'")
    print(f"Building suffix tree and searching for: {patterns}")
    
    matcher = AdvancedPatternMatcher()
    results = matcher.suffix_tree_search(text, patterns)
    
    print(f"\nSuffix tree search results:")
    for pattern, positions in results.items():
        if positions:
            print(f"  Pattern '{pattern}' found at positions: {positions}")
            for pos in positions:
                print(f"    Position {pos}: '{text[pos:pos+len(pattern)]}'")
        else:
            print(f"  Pattern '{pattern}' not found")


def benchmark_advanced_algorithms():
    """Benchmark advanced pattern matching algorithms"""
    print("\n=== Benchmarking Advanced Algorithms ===")
    
    import time
    import random
    import string
    
    matcher = AdvancedPatternMatcher()
    
    # Generate test data
    def generate_text(length: int) -> str:
        return ''.join(random.choices(string.ascii_lowercase[:4], k=length))
    
    def generate_patterns(count: int, avg_length: int) -> List[str]:
        patterns = []
        for _ in range(count):
            length = max(1, avg_length + random.randint(-1, 1))
            pattern = ''.join(random.choices(string.ascii_lowercase[:4], k=length))
            patterns.append(pattern)
        return list(set(patterns))
    
    # Test scenarios
    text = generate_text(1000)
    patterns = generate_patterns(10, 5)
    
    algorithms = [
        ("Aho-Corasick", matcher.aho_corasick_advanced),
        ("Z-Algorithm", matcher.z_algorithm_search),
        ("Multiple KMP", matcher.kmp_multiple_patterns),
    ]
    
    print(f"Text length: {len(text)}")
    print(f"Number of patterns: {len(patterns)}")
    print(f"Average pattern length: {sum(len(p) for p in patterns) / len(patterns):.1f}")
    
    for name, algorithm in algorithms:
        start_time = time.time()
        
        result = algorithm(text, patterns)
        total_matches = sum(len(positions) for positions in result.values())
        
        end_time = time.time()
        duration = (end_time - start_time) * 1000
        
        print(f"\n{name:15}: {duration:.2f}ms ({total_matches} total matches)")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications of advanced algorithms"""
    print("\n=== Real-World Applications ===")
    
    matcher = AdvancedPatternMatcher()
    
    # Application 1: Bioinformatics - DNA sequence analysis
    print("1. DNA Sequence Analysis:")
    
    dna_sequence = "ATCGATCGATCGTAGCTAGCTAGCTACGATCGATCG"
    genetic_markers = ["ATCG", "TAGC", "CGAT", "GCTA", "TCGA"]
    
    print(f"   DNA sequence: {dna_sequence}")
    print(f"   Searching for genetic markers: {genetic_markers}")
    
    dna_results = matcher.aho_corasick_advanced(dna_sequence, genetic_markers)
    
    for marker, positions in dna_results.items():
        if positions:
            print(f"     Marker '{marker}': found {len(positions)} times")
            for start, end in positions:
                print(f"       Position {start}-{end}")
    
    # Application 2: Text processing - find all palindromes
    print(f"\n2. Palindrome Detection in Text:")
    
    text = "A man a plan a canal Panama"
    text_clean = ''.join(text.lower().split())  # Remove spaces and lowercase
    
    print(f"   Text: '{text}'")
    print(f"   Cleaned: '{text_clean}'")
    
    palindromes = matcher.manacher_algorithm(text_clean)
    significant_palindromes = [(s, e) for s, e in palindromes if e - s + 1 >= 3]
    
    print(f"   Palindromes found (length ≥ 3):")
    for start, end in significant_palindromes:
        palindrome = text_clean[start:end + 1]
        print(f"     '{palindrome}' at position {start}-{end}")
    
    # Application 3: Approximate search for spell checking
    print(f"\n3. Spell Checking with Approximate Matching:")
    
    dictionary_words = ["programming", "algorithm", "computer", "science"]
    misspelled_word = "progaming"  # Missing 'r'
    
    print(f"   Dictionary: {dictionary_words}")
    print(f"   Misspelled word: '{misspelled_word}'")
    
    suggestions = []
    for word in dictionary_words:
        # Check if misspelled word is approximately in the dictionary word
        matches = matcher.approximate_string_matching(word, misspelled_word, 2)
        if matches:
            suggestions.append((word, matches[0][1]))  # (word, edit_distance)
    
    suggestions.sort(key=lambda x: x[1])  # Sort by edit distance
    
    print(f"   Suggestions (by edit distance):")
    for word, distance in suggestions:
        print(f"     '{word}' (edit distance: {distance})")


def analyze_algorithm_complexity():
    """Analyze complexity of advanced algorithms"""
    print("\n=== Algorithm Complexity Analysis ===")
    
    complexity_analysis = [
        ("Aho-Corasick Advanced",
         "Time: O(Σ|pattern_i| + |text| + total_matches)",
         "Space: O(Σ|pattern_i|)",
         "Best for multiple pattern search"),
        
        ("Suffix Tree",
         "Time: O(|text|^2) construction + O(Σ|pattern_i| + matches)",
         "Space: O(|text|^2) worst case",
         "Good for repeated queries on same text"),
        
        ("Z-Algorithm",
         "Time: O(|text| + Σ|pattern_i|)",
         "Space: O(max_pattern_length)",
         "Linear time, good for single patterns"),
        
        ("Manacher's Algorithm",
         "Time: O(|text|)",
         "Space: O(|text|)",
         "Optimal for palindrome detection"),
        
        ("Suffix Array + LCP",
         "Time: O(|text|log|text| + Σ|pattern_i|*log|text|)",
         "Space: O(|text|)",
         "Good for range queries and repeated searches"),
        
        ("Approximate Matching",
         "Time: O(|text| * |pattern| * max_distance)",
         "Space: O(|pattern| * max_distance)",
         "For fuzzy string matching"),
    ]
    
    print("Algorithm Analysis:")
    for algorithm, time_complexity, space_complexity, use_case in complexity_analysis:
        print(f"\n{algorithm}:")
        print(f"  {time_complexity}")
        print(f"  {space_complexity}")
        print(f"  Use case: {use_case}")
    
    print(f"\nRecommendations:")
    print(f"  • Use Aho-Corasick for multiple exact pattern matching")
    print(f"  • Use Suffix Tree/Array for repeated queries on same text")
    print(f"  • Use Z-Algorithm for simple linear-time single pattern search")
    print(f"  • Use Manacher's for palindrome detection")
    print(f"  • Use Approximate Matching for spell checking and fuzzy search")


if __name__ == "__main__":
    test_advanced_algorithms()
    test_palindrome_detection()
    test_approximate_matching()
    demonstrate_suffix_tree()
    benchmark_advanced_algorithms()
    demonstrate_real_world_applications()
    analyze_algorithm_complexity()

"""
Advanced Pattern Algorithms demonstrates sophisticated string matching techniques:

1. Aho-Corasick Advanced - Enhanced multi-pattern search with position tracking
2. Suffix Tree Search - Tree-based approach for complex pattern queries
3. Z-Algorithm - Linear time pattern matching with Z-array
4. Manacher's Algorithm - Optimal palindrome detection in linear time
5. Multiple KMP - Apply KMP algorithm for multiple patterns
6. Suffix Array + LCP - Space-efficient suffix structure with LCP array
7. Approximate String Matching - Fuzzy matching with edit distance

Each algorithm targets specific advanced pattern matching scenarios with
optimal time complexity and specialized data structures.
"""
