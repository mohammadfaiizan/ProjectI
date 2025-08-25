"""
Simple Pattern Matching - Multiple Approaches
Difficulty: Easy

Implement various simple pattern matching algorithms and techniques.
Focus on basic string pattern matching without complex regular expressions.
"""

from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict

class TrieNode:
    """Simple trie node for pattern matching"""
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.patterns = []  # Store patterns ending at this node

class SimplePatternMatcher:
    
    def __init__(self):
        self.patterns = []
    
    def add_pattern(self, pattern: str) -> None:
        """Add a pattern to search for"""
        self.patterns.append(pattern)
    
    def naive_search(self, text: str, pattern: str) -> List[int]:
        """
        Approach 1: Naive String Matching
        
        Simple brute force pattern search.
        
        Time: O(n*m) where n=text length, m=pattern length
        Space: O(1)
        """
        matches = []
        n, m = len(text), len(pattern)
        
        for i in range(n - m + 1):
            # Check if pattern matches at position i
            match = True
            for j in range(m):
                if text[i + j] != pattern[j]:
                    match = False
                    break
            
            if match:
                matches.append(i)
        
        return matches
    
    def rabin_karp_search(self, text: str, pattern: str) -> List[int]:
        """
        Approach 2: Rabin-Karp Algorithm
        
        Rolling hash-based pattern matching.
        
        Time: O(n + m) average, O(n*m) worst case
        Space: O(1)
        """
        matches = []
        n, m = len(text), len(pattern)
        
        if m > n:
            return matches
        
        # Base for rolling hash
        base = 256
        mod = 10**9 + 7
        
        # Calculate hash of pattern and first window of text
        pattern_hash = 0
        text_hash = 0
        h = 1
        
        # h = base^(m-1) % mod
        for i in range(m - 1):
            h = (h * base) % mod
        
        # Calculate initial hashes
        for i in range(m):
            pattern_hash = (pattern_hash * base + ord(pattern[i])) % mod
            text_hash = (text_hash * base + ord(text[i])) % mod
        
        # Slide pattern over text
        for i in range(n - m + 1):
            # Check if hashes match
            if pattern_hash == text_hash:
                # Verify actual characters (handle hash collisions)
                if text[i:i+m] == pattern:
                    matches.append(i)
            
            # Calculate hash for next window
            if i < n - m:
                text_hash = (text_hash - ord(text[i]) * h) % mod
                text_hash = (text_hash * base + ord(text[i + m])) % mod
                
                # Handle negative hash
                if text_hash < 0:
                    text_hash += mod
        
        return matches
    
    def kmp_search(self, text: str, pattern: str) -> List[int]:
        """
        Approach 3: KMP (Knuth-Morris-Pratt) Algorithm
        
        Efficient pattern matching using failure function.
        
        Time: O(n + m)
        Space: O(m)
        """
        matches = []
        n, m = len(text), len(pattern)
        
        if m == 0:
            return matches
        
        # Build failure function
        failure = self._build_failure_function(pattern)
        
        i = j = 0
        while i < n:
            if text[i] == pattern[j]:
                i += 1
                j += 1
            
            if j == m:
                matches.append(i - j)
                j = failure[j - 1]
            elif i < n and text[i] != pattern[j]:
                if j != 0:
                    j = failure[j - 1]
                else:
                    i += 1
        
        return matches
    
    def _build_failure_function(self, pattern: str) -> List[int]:
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
    
    def boyer_moore_search(self, text: str, pattern: str) -> List[int]:
        """
        Approach 4: Boyer-Moore Algorithm (Simplified)
        
        Pattern matching with bad character rule.
        
        Time: O(n*m) worst case, O(n/m) best case
        Space: O(alphabet_size)
        """
        matches = []
        n, m = len(text), len(pattern)
        
        if m > n:
            return matches
        
        # Build bad character table
        bad_char = {}
        for i in range(m):
            bad_char[pattern[i]] = i
        
        shift = 0
        while shift <= n - m:
            j = m - 1
            
            # Compare pattern with text from right to left
            while j >= 0 and pattern[j] == text[shift + j]:
                j -= 1
            
            if j < 0:
                # Pattern found
                matches.append(shift)
                
                # Shift pattern to align with next possible match
                if shift + m < n:
                    shift += m - bad_char.get(text[shift + m], -1)
                else:
                    shift += 1
            else:
                # Shift pattern based on bad character rule
                shift += max(1, j - bad_char.get(text[shift + j], -1))
        
        return matches
    
    def trie_based_search(self, text: str, patterns: List[str]) -> Dict[str, List[int]]:
        """
        Approach 5: Trie-based Multi-pattern Search
        
        Search for multiple patterns simultaneously using trie.
        
        Time: O(total_pattern_length + n*max_pattern_length)
        Space: O(total_pattern_length)
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
        
        matches = defaultdict(list)
        
        # Search for patterns starting at each position
        for i in range(len(text)):
            node = root
            j = i
            
            while j < len(text) and text[j] in node.children:
                node = node.children[text[j]]
                if node.is_end:
                    for pattern in node.patterns:
                        matches[pattern].append(i)
                j += 1
        
        return dict(matches)
    
    def wildcard_match(self, text: str, pattern: str) -> bool:
        """
        Approach 6: Simple Wildcard Matching
        
        Pattern matching with '?' (single char) and '*' (any sequence).
        
        Time: O(n*m) with optimizations
        Space: O(n*m) for DP table
        """
        n, m = len(text), len(pattern)
        
        # DP table: dp[i][j] = can text[0:i] match pattern[0:j]
        dp = [[False] * (m + 1) for _ in range(n + 1)]
        
        # Empty pattern matches empty text
        dp[0][0] = True
        
        # Handle patterns with leading '*'
        for j in range(1, m + 1):
            if pattern[j - 1] == '*':
                dp[0][j] = dp[0][j - 1]
        
        # Fill DP table
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if pattern[j - 1] == '*':
                    # '*' can match empty sequence or any character
                    dp[i][j] = dp[i][j - 1] or dp[i - 1][j]
                elif pattern[j - 1] == '?' or pattern[j - 1] == text[i - 1]:
                    # '?' matches any character, or exact match
                    dp[i][j] = dp[i - 1][j - 1]
        
        return dp[n][m]
    
    def regex_like_match(self, text: str, pattern: str) -> bool:
        """
        Approach 7: Simple Regex-like Matching
        
        Support basic regex operators: '.', '*', '+', '?'
        
        Time: O(n*m)
        Space: O(n*m)
        """
        def match_helper(t_idx: int, p_idx: int, memo: Dict[Tuple[int, int], bool]) -> bool:
            if (t_idx, p_idx) in memo:
                return memo[(t_idx, p_idx)]
            
            # Base cases
            if p_idx >= len(pattern):
                result = t_idx >= len(text)
            elif t_idx >= len(text):
                # Check if remaining pattern can match empty string
                result = all(pattern[i] == '*' for i in range(p_idx, len(pattern)))
            else:
                char_match = (pattern[p_idx] == '.' or pattern[p_idx] == text[t_idx])
                
                # Look ahead for quantifiers
                if p_idx + 1 < len(pattern) and pattern[p_idx + 1] == '*':
                    # '*' means zero or more of preceding character
                    result = (match_helper(t_idx, p_idx + 2, memo) or  # Zero occurrences
                             (char_match and match_helper(t_idx + 1, p_idx, memo)))  # One or more
                elif p_idx + 1 < len(pattern) and pattern[p_idx + 1] == '+':
                    # '+' means one or more of preceding character
                    result = (char_match and 
                             (match_helper(t_idx + 1, p_idx + 2, memo) or  # Exactly one
                              match_helper(t_idx + 1, p_idx, memo)))       # More than one
                elif p_idx + 1 < len(pattern) and pattern[p_idx + 1] == '?':
                    # '?' means zero or one of preceding character
                    result = (match_helper(t_idx, p_idx + 2, memo) or  # Zero occurrences
                             (char_match and match_helper(t_idx + 1, p_idx + 2, memo)))  # One occurrence
                else:
                    # Regular character match
                    result = char_match and match_helper(t_idx + 1, p_idx + 1, memo)
            
            memo[(t_idx, p_idx)] = result
            return result
        
        return match_helper(0, 0, {})


def test_basic_pattern_matching():
    """Test basic pattern matching algorithms"""
    print("=== Testing Basic Pattern Matching ===")
    
    matcher = SimplePatternMatcher()
    
    test_cases = [
        ("abcdefg", "cde", [2]),
        ("aaaa", "aa", [0, 1, 2]),
        ("abcabcabc", "abc", [0, 3, 6]),
        ("hello world", "world", [6]),
        ("no match here", "xyz", []),
        ("", "pattern", []),
        ("text", "", []),
    ]
    
    algorithms = [
        ("Naive", matcher.naive_search),
        ("Rabin-Karp", matcher.rabin_karp_search),
        ("KMP", matcher.kmp_search),
        ("Boyer-Moore", matcher.boyer_moore_search),
    ]
    
    for i, (text, pattern, expected) in enumerate(test_cases):
        print(f"\nTest Case {i+1}: Text='{text}', Pattern='{pattern}'")
        print(f"Expected: {expected}")
        
        for name, algorithm in algorithms:
            try:
                result = algorithm(text, pattern)
                status = "✓" if result == expected else "✗"
                print(f"  {name:12}: {result} {status}")
            except Exception as e:
                print(f"  {name:12}: Error - {e}")


def test_multi_pattern_search():
    """Test multi-pattern search using trie"""
    print("\n=== Testing Multi-Pattern Search ===")
    
    matcher = SimplePatternMatcher()
    
    text = "she sells seashells by the seashore"
    patterns = ["she", "sea", "sell", "shore", "he"]
    
    print(f"Text: '{text}'")
    print(f"Patterns: {patterns}")
    
    result = matcher.trie_based_search(text, patterns)
    
    print(f"Results:")
    for pattern, positions in result.items():
        print(f"  '{pattern}': found at positions {positions}")


def test_wildcard_matching():
    """Test wildcard pattern matching"""
    print("\n=== Testing Wildcard Matching ===")
    
    matcher = SimplePatternMatcher()
    
    test_cases = [
        ("hello", "h*o", True),
        ("hello", "h?llo", True),
        ("hello", "h?l?o", True),
        ("hello", "h*", True),
        ("hello", "*o", True),
        ("hello", "world", False),
        ("abc", "a*c", True),
        ("abc", "a?c", False),  # ? matches exactly one char
        ("", "*", True),
        ("a", "*", True),
    ]
    
    print("Wildcard pattern matching ('?' = single char, '*' = any sequence):")
    
    for i, (text, pattern, expected) in enumerate(test_cases):
        result = matcher.wildcard_match(text, pattern)
        status = "✓" if result == expected else "✗"
        print(f"  '{text}' matches '{pattern}': {result} {status}")


def test_regex_like_matching():
    """Test regex-like pattern matching"""
    print("\n=== Testing Regex-like Matching ===")
    
    matcher = SimplePatternMatcher()
    
    test_cases = [
        ("hello", "h.llo", True),   # '.' matches any single char
        ("hello", "he.*o", True),  # '.*' matches any sequence
        ("hello", "hel+o", True),  # 'l+' matches one or more 'l'
        ("helo", "hel+o", False),  # Need at least one 'l'
        ("hello", "hel?o", False), # 'l?' means zero or one 'l'
        ("helo", "hel?o", True),   # Zero 'l' after 'he'
        ("abc", "a.c", True),
        ("ac", "a.c", False),      # '.' must match one char
    ]
    
    print("Regex-like matching ('.' = any char, '*' = zero or more, '+' = one or more, '?' = zero or one):")
    
    for i, (text, pattern, expected) in enumerate(test_cases):
        try:
            result = matcher.regex_like_match(text, pattern)
            status = "✓" if result == expected else "✗"
            print(f"  '{text}' matches '{pattern}': {result} {status}")
        except Exception as e:
            print(f"  '{text}' matches '{pattern}': Error - {e}")


def demonstrate_algorithm_steps():
    """Demonstrate algorithm steps for educational purposes"""
    print("\n=== Algorithm Steps Demo ===")
    
    text = "abaaaba"
    pattern = "aba"
    
    print(f"Text: '{text}'")
    print(f"Pattern: '{pattern}'")
    
    # Demonstrate naive search
    print(f"\n1. Naive Search Steps:")
    n, m = len(text), len(pattern)
    
    for i in range(n - m + 1):
        print(f"  Position {i}: ", end="")
        
        match = True
        for j in range(m):
            if text[i + j] != pattern[j]:
                print(f"Mismatch at {text[i + j]} != {pattern[j]}")
                match = False
                break
        
        if match:
            print("Match found!")
        elif j == 0:
            print(f"First char mismatch: {text[i]} != {pattern[0]}")
    
    # Demonstrate KMP preprocessing
    print(f"\n2. KMP Failure Function:")
    matcher = SimplePatternMatcher()
    failure = matcher._build_failure_function(pattern)
    
    print(f"  Pattern: {list(pattern)}")
    print(f"  Indices: {list(range(len(pattern)))}")
    print(f"  Failure: {failure}")
    
    for i, val in enumerate(failure):
        if val > 0:
            print(f"    Position {i}: can skip to position {val} on mismatch")


def benchmark_algorithms():
    """Benchmark different pattern matching algorithms"""
    print("\n=== Benchmarking Algorithms ===")
    
    import time
    import random
    import string
    
    matcher = SimplePatternMatcher()
    
    # Generate test data
    def generate_text(length: int) -> str:
        return ''.join(random.choices(string.ascii_lowercase[:5], k=length))  # Limited alphabet
    
    def generate_pattern(length: int) -> str:
        return ''.join(random.choices(string.ascii_lowercase[:5], k=length))
    
    test_scenarios = [
        ("Short", generate_text(100), generate_pattern(5)),
        ("Medium", generate_text(1000), generate_pattern(10)),
        ("Long", generate_text(10000), generate_pattern(20)),
    ]
    
    algorithms = [
        ("Naive", matcher.naive_search),
        ("Rabin-Karp", matcher.rabin_karp_search),
        ("KMP", matcher.kmp_search),
        ("Boyer-Moore", matcher.boyer_moore_search),
    ]
    
    for scenario_name, text, pattern in test_scenarios:
        print(f"\n--- {scenario_name} Dataset ---")
        print(f"Text length: {len(text)}, Pattern length: {len(pattern)}")
        
        for algorithm_name, algorithm in algorithms:
            start_time = time.time()
            
            # Run multiple times for better measurement
            for _ in range(10):
                result = algorithm(text, pattern)
            
            end_time = time.time()
            avg_time = (end_time - start_time) / 10
            
            print(f"  {algorithm_name:12}: {avg_time*1000:.2f}ms (found {len(result)} matches)")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    matcher = SimplePatternMatcher()
    
    # Application 1: Log file analysis
    print("1. Log File Analysis:")
    
    log_entries = [
        "2024-01-01 10:30:15 ERROR: Database connection failed",
        "2024-01-01 10:30:16 INFO: Retrying connection",
        "2024-01-01 10:30:17 ERROR: Authentication failed",
        "2024-01-01 10:30:18 WARNING: High memory usage",
        "2024-01-01 10:30:19 INFO: System recovered"
    ]
    
    error_patterns = ["ERROR", "WARNING", "FAILED"]
    
    print(f"   Log entries: {len(log_entries)}")
    print(f"   Searching for: {error_patterns}")
    
    for i, entry in enumerate(log_entries):
        matches = matcher.trie_based_search(entry, error_patterns)
        if matches:
            print(f"     Entry {i+1}: {matches}")
    
    # Application 2: DNA sequence analysis
    print(f"\n2. DNA Sequence Analysis:")
    
    dna_sequence = "ATCGATCGATCGTAGCTAGCTAGCT"
    genetic_markers = ["ATCG", "GCTA", "TAGC"]
    
    print(f"   DNA sequence: {dna_sequence}")
    print(f"   Genetic markers: {genetic_markers}")
    
    markers_found = matcher.trie_based_search(dna_sequence, genetic_markers)
    for marker, positions in markers_found.items():
        print(f"     Marker '{marker}' found at positions: {positions}")
    
    # Application 3: Text processing and highlighting
    print(f"\n3. Text Processing:")
    
    document = "Python is a programming language. Python developers love Python."
    keywords = ["Python", "programming", "language"]
    
    print(f"   Document: '{document}'")
    print(f"   Keywords to highlight: {keywords}")
    
    keyword_positions = matcher.trie_based_search(document, keywords)
    
    # Create highlighted version (simplified)
    highlighted = document
    for keyword, positions in keyword_positions.items():
        print(f"     '{keyword}' appears {len(positions)} times at positions {positions}")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    matcher = SimplePatternMatcher()
    
    edge_cases = [
        # Empty inputs
        ("", "", "Empty text and pattern"),
        ("text", "", "Empty pattern"),
        ("", "pattern", "Empty text"),
        
        # Single character
        ("a", "a", "Single character match"),
        ("a", "b", "Single character no match"),
        
        # Pattern longer than text
        ("hi", "hello", "Pattern longer than text"),
        
        # Repeated characters
        ("aaaa", "aa", "Repeated characters"),
        ("abcabc", "abc", "Repeated pattern"),
        
        # Pattern at boundaries
        ("pattern", "pat", "Pattern at start"),
        ("pattern", "ern", "Pattern at end"),
        
        # Case sensitivity
        ("Hello", "hello", "Case sensitivity"),
    ]
    
    for text, pattern, description in edge_cases:
        print(f"\n{description}:")
        print(f"  Text: '{text}', Pattern: '{pattern}'")
        
        try:
            result = matcher.naive_search(text, pattern)
            print(f"  Matches at: {result}")
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    test_basic_pattern_matching()
    test_multi_pattern_search()
    test_wildcard_matching()
    test_regex_like_matching()
    demonstrate_algorithm_steps()
    benchmark_algorithms()
    demonstrate_real_world_applications()
    test_edge_cases()

"""
Simple Pattern Matching demonstrates fundamental string searching algorithms:

1. Naive Search - Brute force O(n*m) approach for basic understanding
2. Rabin-Karp - Rolling hash technique for efficient searching
3. KMP Algorithm - Linear time pattern matching with failure function
4. Boyer-Moore - Right-to-left scanning with bad character heuristic
5. Trie-based - Multi-pattern search using trie data structure
6. Wildcard Matching - Support for '?' and '*' wildcards
7. Regex-like - Basic regex operators with dynamic programming

Each approach demonstrates different optimization strategies and use cases
from simple text search to advanced pattern matching with wildcards.
"""
