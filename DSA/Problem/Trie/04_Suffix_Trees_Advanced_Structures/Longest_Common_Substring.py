"""
Longest Common Substring - Multiple Approaches
Difficulty: Medium

Find the longest common substring between two or more strings. A common substring 
is a substring that appears in all given strings.

This is different from Longest Common Subsequence (LCS) - we need contiguous characters.

Problem Variations:
1. Longest Common Substring of two strings
2. Longest Common Substring of multiple strings  
3. All common substrings
4. Longest common substring with at most k mismatches

Applications:
- DNA sequence analysis
- Text similarity measurement
- Plagiarism detection
- String compression
"""

from typing import List, Set, Tuple, Dict, Optional
from collections import defaultdict
import time

class LongestCommonSubstring:
    
    def lcs_dynamic_programming(self, str1: str, str2: str) -> str:
        """
        Approach 1: Dynamic Programming
        
        Build DP table to find longest common substring.
        
        Time: O(m * n) where m, n are string lengths
        Space: O(m * n)
        """
        if not str1 or not str2:
            return ""
        
        m, n = len(str1), len(str2)
        
        # dp[i][j] = length of common substring ending at str1[i-1] and str2[j-1]
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        max_length = 0
        ending_pos = 0
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str1[i - 1] == str2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    
                    if dp[i][j] > max_length:
                        max_length = dp[i][j]
                        ending_pos = i
                else:
                    dp[i][j] = 0
        
        if max_length == 0:
            return ""
        
        return str1[ending_pos - max_length:ending_pos]
    
    def lcs_optimized_dp(self, str1: str, str2: str) -> str:
        """
        Approach 2: Space-Optimized Dynamic Programming
        
        Use only two rows instead of full DP table.
        
        Time: O(m * n)
        Space: O(min(m, n))
        """
        if not str1 or not str2:
            return ""
        
        # Ensure str1 is the shorter string for space optimization
        if len(str1) > len(str2):
            str1, str2 = str2, str1
        
        m, n = len(str1), len(str2)
        
        # Use only two rows
        prev_row = [0] * (m + 1)
        curr_row = [0] * (m + 1)
        
        max_length = 0
        ending_pos_str2 = 0
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if str2[i - 1] == str1[j - 1]:
                    curr_row[j] = prev_row[j - 1] + 1
                    
                    if curr_row[j] > max_length:
                        max_length = curr_row[j]
                        ending_pos_str2 = i
                else:
                    curr_row[j] = 0
            
            # Swap rows
            prev_row, curr_row = curr_row, prev_row
        
        if max_length == 0:
            return ""
        
        return str2[ending_pos_str2 - max_length:ending_pos_str2]
    
    def lcs_suffix_tree(self, str1: str, str2: str) -> str:
        """
        Approach 3: Generalized Suffix Tree
        
        Build suffix tree for both strings and find deepest internal node
        with suffixes from both strings.
        
        Time: O(m + n) for construction + O(m + n) for traversal
        Space: O(m + n)
        """
        class SuffixTreeNode:
            def __init__(self):
                self.children = {}
                self.is_end = False
                self.suffix_indices = []  # Store (string_id, suffix_start)
                self.string_ids = set()  # Which strings pass through this node
        
        if not str1 or not str2:
            return ""
        
        # Create generalized suffix tree
        # Combine strings with unique separators
        combined = str1 + '#' + str2 + '$'
        str1_length = len(str1)
        
        root = SuffixTreeNode()
        
        # Build suffix tree (simplified implementation)
        for i in range(len(combined)):
            current = root
            
            # Determine which string this suffix belongs to
            if i <= str1_length:
                string_id = 1
                actual_start = i
            else:
                string_id = 2
                actual_start = i - str1_length - 1
            
            for j in range(i, len(combined)):
                char = combined[j]
                
                if char == '#' or char == '$':
                    break
                
                if char not in current.children:
                    current.children[char] = SuffixTreeNode()
                
                current = current.children[char]
                current.string_ids.add(string_id)
                current.suffix_indices.append((string_id, actual_start))
        
        # Find deepest node with suffixes from both strings
        max_depth = 0
        lcs_result = ""
        
        def dfs(node: SuffixTreeNode, depth: int, path: str) -> None:
            nonlocal max_depth, lcs_result
            
            # Check if this node has suffixes from both strings
            if len(node.string_ids) >= 2 and depth > max_depth:
                max_depth = depth
                lcs_result = path
            
            # Continue DFS
            for char, child in node.children.items():
                dfs(child, depth + 1, path + char)
        
        dfs(root, 0, "")
        return lcs_result
    
    def lcs_rolling_hash(self, str1: str, str2: str) -> str:
        """
        Approach 4: Binary Search with Rolling Hash
        
        Binary search on answer length, use rolling hash to check existence.
        
        Time: O((m + n) * log(min(m, n)))
        Space: O(m + n)
        """
        if not str1 or not str2:
            return ""
        
        def has_common_substring(length: int) -> Optional[str]:
            """Check if there exists common substring of given length"""
            if length == 0:
                return ""
            
            # Rolling hash parameters
            base = 31
            mod = 10**9 + 7
            
            # Compute hash of first substring of str1
            hash1_set = set()
            
            # Hash all substrings of str1 with given length
            for i in range(len(str1) - length + 1):
                substring = str1[i:i + length]
                hash_val = 0
                for char in substring:
                    hash_val = (hash_val * base + ord(char)) % mod
                hash1_set.add((hash_val, substring))
            
            # Check substrings of str2
            for i in range(len(str2) - length + 1):
                substring = str2[i:i + length]
                hash_val = 0
                for char in substring:
                    hash_val = (hash_val * base + ord(char)) % mod
                
                # Check if this hash exists in str1
                for hash1_val, str1_substring in hash1_set:
                    if hash_val == hash1_val and substring == str1_substring:
                        return substring
            
            return None
        
        # Binary search on length
        left, right = 0, min(len(str1), len(str2))
        result = ""
        
        while left <= right:
            mid = (left + right) // 2
            common_substring = has_common_substring(mid)
            
            if common_substring is not None:
                result = common_substring
                left = mid + 1
            else:
                right = mid - 1
        
        return result
    
    def lcs_multiple_strings(self, strings: List[str]) -> str:
        """
        Approach 5: Multiple Strings using Generalized Suffix Tree
        
        Find longest common substring among multiple strings.
        
        Time: O(sum of string lengths)
        Space: O(sum of string lengths)
        """
        if not strings:
            return ""
        
        if len(strings) == 1:
            return strings[0]
        
        class GeneralizedTrieNode:
            def __init__(self):
                self.children = {}
                self.string_presence = set()  # Which strings have this prefix
        
        # Build generalized trie
        root = GeneralizedTrieNode()
        
        for string_id, string in enumerate(strings):
            for start in range(len(string)):
                current = root
                
                for pos in range(start, len(string)):
                    char = string[pos]
                    
                    if char not in current.children:
                        current.children[char] = GeneralizedTrieNode()
                    
                    current = current.children[char]
                    current.string_presence.add(string_id)
        
        # Find longest path where all strings are present
        max_length = 0
        lcs_result = ""
        
        def dfs(node: GeneralizedTrieNode, depth: int, path: str) -> None:
            nonlocal max_length, lcs_result
            
            # Check if all strings pass through this node
            if len(node.string_presence) == len(strings) and depth > max_length:
                max_length = depth
                lcs_result = path
            
            # Continue only if all strings still present
            if len(node.string_presence) == len(strings):
                for char, child in node.children.items():
                    dfs(child, depth + 1, path + char)
        
        dfs(root, 0, "")
        return lcs_result
    
    def lcs_with_mismatches(self, str1: str, str2: str, k: int) -> str:
        """
        Approach 6: Longest Common Substring with at most k mismatches
        
        Allow up to k character mismatches in the common substring.
        
        Time: O(m * n * k)
        Space: O(m * n * k)
        """
        if not str1 or not str2:
            return ""
        
        m, n = len(str1), len(str2)
        
        # dp[i][j][mismatches] = max length ending at i,j with 'mismatches' mismatches
        dp = [[[0 for _ in range(k + 1)] for _ in range(n + 1)] for _ in range(m + 1)]
        
        max_length = 0
        best_i, best_j, best_k = 0, 0, 0
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                for mismatches in range(k + 1):
                    if str1[i - 1] == str2[j - 1]:
                        # Characters match
                        dp[i][j][mismatches] = dp[i - 1][j - 1][mismatches] + 1
                    elif mismatches > 0:
                        # Characters don't match, use one mismatch
                        dp[i][j][mismatches] = dp[i - 1][j - 1][mismatches - 1] + 1
                    
                    # Update best result
                    if dp[i][j][mismatches] > max_length:
                        max_length = dp[i][j][mismatches]
                        best_i, best_j, best_k = i, j, mismatches
        
        if max_length == 0:
            return ""
        
        # Reconstruct the substring
        result = []
        i, j, mismatches = best_i, best_j, best_k
        
        while i > 0 and j > 0 and dp[i][j][mismatches] > 0:
            if str1[i - 1] == str2[j - 1]:
                result.append(str1[i - 1])
                i -= 1
                j -= 1
            elif mismatches > 0 and dp[i][j][mismatches] == dp[i - 1][j - 1][mismatches - 1] + 1:
                result.append(str1[i - 1])  # Or str2[j - 1], they're different
                i -= 1
                j -= 1
                mismatches -= 1
            else:
                break
        
        return ''.join(reversed(result))
    
    def all_common_substrings(self, str1: str, str2: str, min_length: int = 1) -> List[str]:
        """
        Approach 7: Find All Common Substrings
        
        Find all common substrings of minimum length.
        
        Time: O(m * n * min(m, n))
        Space: O(m * n)
        """
        if not str1 or not str2:
            return []
        
        # Use DP to find all common substrings
        m, n = len(str1), len(str2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Store all common substrings
        common_substrings = set()
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str1[i - 1] == str2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    
                    # Add all substrings ending at this position
                    length = dp[i][j]
                    if length >= min_length:
                        substring = str1[i - length:i]
                        common_substrings.add(substring)
                else:
                    dp[i][j] = 0
        
        return list(common_substrings)


def test_basic_lcs():
    """Test basic LCS functionality"""
    print("=== Testing Basic LCS ===")
    
    lcs = LongestCommonSubstring()
    
    test_cases = [
        # Basic cases
        ("ABABC", "BABCA", "BABC"),
        ("ABCDGH", "ACDGHR", "CDGH"),
        ("geeksforgeeks", "geeksquiz", "geeks"),
        
        # Edge cases
        ("", "ABC", ""),
        ("ABC", "", ""),
        ("ABC", "DEF", ""),
        ("A", "A", "A"),
        
        # Complex cases
        ("LCLC", "CLCL", "CLC"),
        ("programming", "programmer", "programm"),
        ("abcdefghijk", "defghijklmn", "defghijk"),
    ]
    
    approaches = [
        ("Dynamic Programming", lcs.lcs_dynamic_programming),
        ("Optimized DP", lcs.lcs_optimized_dp),
        ("Suffix Tree", lcs.lcs_suffix_tree),
        ("Rolling Hash", lcs.lcs_rolling_hash),
    ]
    
    for i, (str1, str2, expected) in enumerate(test_cases):
        print(f"\nTest Case {i+1}: '{str1}' & '{str2}'")
        print(f"Expected: '{expected}'")
        
        for name, method in approaches:
            try:
                result = method(str1, str2)
                # For LCS, there might be multiple valid answers of same length
                status = "✓" if len(result) == len(expected) else "✗"
                print(f"  {name:15}: '{result}' {status}")
            except Exception as e:
                print(f"  {name:15}: Error - {e}")


def test_multiple_strings():
    """Test LCS with multiple strings"""
    print("\n=== Testing Multiple Strings LCS ===")
    
    lcs = LongestCommonSubstring()
    
    test_cases = [
        (["ABCD", "ACBD", "AECD"], "ACD"),
        (["geeksforgeeks", "geeksquiz", "geekster"], "geeks"),
        (["programming", "programmer", "program"], "program"),
        (["ABC", "DEF", "GHI"], ""),
        (["AAAA", "AABB", "AACC"], "AA"),
    ]
    
    for i, (strings, expected_prefix) in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {strings}")
        
        result = lcs.lcs_multiple_strings(strings)
        print(f"Result: '{result}'")
        print(f"Expected prefix: '{expected_prefix}'")
        
        # Check if result is a valid common substring
        is_valid = all(result in s for s in strings) if result else True
        print(f"Valid: {is_valid}")


def test_lcs_with_mismatches():
    """Test LCS with allowed mismatches"""
    print("\n=== Testing LCS with Mismatches ===")
    
    lcs = LongestCommonSubstring()
    
    test_cases = [
        ("ABCD", "AXCD", 1, "should find ABCD/AXCD with 1 mismatch"),
        ("programming", "programing", 1, "should handle missing character"),
        ("HELLO", "HXLLO", 1, "single character mismatch"),
        ("ABCDE", "FGHIJ", 2, "many mismatches needed"),
    ]
    
    for str1, str2, k, description in test_cases:
        print(f"\nStrings: '{str1}' & '{str2}' with k={k}")
        print(f"Description: {description}")
        
        result = lcs.lcs_with_mismatches(str1, str2, k)
        print(f"Result: '{result}' (length: {len(result)})")


def test_all_common_substrings():
    """Test finding all common substrings"""
    print("\n=== Testing All Common Substrings ===")
    
    lcs = LongestCommonSubstring()
    
    test_cases = [
        ("ABCAB", "CABCA", 2),
        ("geeksforgeeks", "geeksquiz", 3),
        ("programming", "programmer", 4),
    ]
    
    for str1, str2, min_length in test_cases:
        print(f"\nStrings: '{str1}' & '{str2}' (min length: {min_length})")
        
        result = lcs.all_common_substrings(str1, str2, min_length)
        result.sort(key=lambda x: (-len(x), x))  # Sort by length desc, then alphabetically
        
        print(f"Common substrings: {result}")


def demonstrate_suffix_tree_approach():
    """Demonstrate suffix tree construction"""
    print("\n=== Suffix Tree Approach Demo ===")
    
    str1, str2 = "BANANA", "ANANAS"
    
    print(f"Finding LCS of '{str1}' and '{str2}' using suffix tree approach")
    
    # Manual demonstration of generalized suffix tree construction
    print(f"\nStep 1: Create combined string")
    combined = str1 + '#' + str2 + '$'
    print(f"Combined: '{combined}'")
    
    print(f"\nStep 2: Generate suffixes")
    str1_suffixes = [(i, str1[i:]) for i in range(len(str1))]
    str2_suffixes = [(i, str2[i:]) for i in range(len(str2))]
    
    print(f"String 1 suffixes:")
    for i, suffix in str1_suffixes:
        print(f"  {i}: '{suffix}'")
    
    print(f"String 2 suffixes:")
    for i, suffix in str2_suffixes:
        print(f"  {i}: '{suffix}'")
    
    # Find common prefixes manually
    print(f"\nStep 3: Find common prefixes")
    common_prefixes = set()
    
    for _, suf1 in str1_suffixes:
        for _, suf2 in str2_suffixes:
            # Find common prefix
            i = 0
            while i < min(len(suf1), len(suf2)) and suf1[i] == suf2[i]:
                i += 1
            
            if i > 0:
                common_prefix = suf1[:i]
                common_prefixes.add(common_prefix)
    
    # Find longest
    if common_prefixes:
        longest = max(common_prefixes, key=len)
        print(f"Longest common substring: '{longest}'")
    else:
        print(f"No common substring found")


def benchmark_approaches():
    """Benchmark different approaches"""
    print("\n=== Benchmarking Approaches ===")
    
    import random
    import string
    
    lcs = LongestCommonSubstring()
    
    # Generate test strings
    def generate_similar_strings(base_length: int, similarity: float) -> Tuple[str, str]:
        """Generate two strings with controlled similarity"""
        base = ''.join(random.choices(string.ascii_lowercase, k=base_length))
        
        # Create second string by modifying base
        str2_chars = []
        for char in base:
            if random.random() < similarity:
                str2_chars.append(char)  # Keep same
            else:
                str2_chars.append(random.choice(string.ascii_lowercase))  # Change
        
        # Add some extra characters
        extra_length = random.randint(0, base_length // 4)
        str2_chars.extend(random.choices(string.ascii_lowercase, k=extra_length))
        
        return base, ''.join(str2_chars)
    
    test_scenarios = [
        ("Small", 50, 0.8),
        ("Medium", 200, 0.7),
        ("Large", 500, 0.6),
    ]
    
    approaches = [
        ("Dynamic Programming", lcs.lcs_dynamic_programming),
        ("Optimized DP", lcs.lcs_optimized_dp),
        ("Rolling Hash", lcs.lcs_rolling_hash),
    ]
    
    for scenario_name, length, similarity in test_scenarios:
        str1, str2 = generate_similar_strings(length, similarity)
        
        print(f"\n--- {scenario_name} Strings ---")
        print(f"Length: {len(str1)}, {len(str2)}, Similarity: {similarity}")
        
        for approach_name, method in approaches:
            start_time = time.time()
            
            try:
                result = method(str1, str2)
                end_time = time.time()
                
                execution_time = (end_time - start_time) * 1000
                print(f"  {approach_name:18}: {execution_time:6.2f}ms (LCS length: {len(result)})")
            
            except Exception as e:
                print(f"  {approach_name:18}: Error - {str(e)[:30]}")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    lcs = LongestCommonSubstring()
    
    # Application 1: DNA sequence analysis
    print("1. DNA Sequence Analysis:")
    
    dna1 = "ATCGATCGATCG"
    dna2 = "TCGATCGATCGA"
    
    common_sequence = lcs.lcs_dynamic_programming(dna1, dna2)
    print(f"   DNA1: {dna1}")
    print(f"   DNA2: {dna2}")
    print(f"   Common sequence: {common_sequence}")
    print(f"   Similarity: {len(common_sequence) / max(len(dna1), len(dna2)) * 100:.1f}%")
    
    # Application 2: Plagiarism detection
    print(f"\n2. Plagiarism Detection:")
    
    doc1 = "the quick brown fox jumps over the lazy dog"
    doc2 = "a quick brown fox leaps over a lazy cat"
    
    # Remove spaces for better comparison
    doc1_clean = doc1.replace(" ", "")
    doc2_clean = doc2.replace(" ", "")
    
    common_text = lcs.lcs_dynamic_programming(doc1_clean, doc2_clean)
    print(f"   Document 1: {doc1}")
    print(f"   Document 2: {doc2}")
    print(f"   Common subsequence: {common_text}")
    print(f"   Potential plagiarism score: {len(common_text) / max(len(doc1_clean), len(doc2_clean)) * 100:.1f}%")
    
    # Application 3: Code similarity
    print(f"\n3. Code Similarity Detection:")
    
    code1 = "function(a,b){return a+b;}"
    code2 = "function(x,y){return x+y;}"
    
    common_code = lcs.lcs_dynamic_programming(code1, code2)
    print(f"   Code 1: {code1}")
    print(f"   Code 2: {code2}")
    print(f"   Common structure: {common_code}")
    print(f"   Code similarity: {len(common_code) / max(len(code1), len(code2)) * 100:.1f}%")
    
    # Application 4: Version control (diff)
    print(f"\n4. Version Control Diff:")
    
    old_version = "Hello World Program"
    new_version = "Hello Universe Program"
    
    common_parts = lcs.all_common_substrings(old_version, new_version, 2)
    unchanged_text = lcs.lcs_dynamic_programming(old_version, new_version)
    
    print(f"   Old: {old_version}")
    print(f"   New: {new_version}")
    print(f"   Unchanged: {unchanged_text}")
    print(f"   Common parts: {common_parts}")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    lcs = LongestCommonSubstring()
    
    edge_cases = [
        # Empty strings
        ("", "", "Both empty"),
        ("ABC", "", "One empty"),
        ("", "XYZ", "Other empty"),
        
        # Single characters
        ("A", "A", "Same single char"),
        ("A", "B", "Different single chars"),
        
        # No common substring
        ("ABC", "DEF", "No common chars"),
        ("ABCD", "EFGH", "Completely different"),
        
        # Entire strings are common
        ("HELLO", "HELLO", "Identical strings"),
        
        # One string is substring of other
        ("ABC", "XABCY", "Substring relationship"),
        
        # Repeated patterns
        ("AAAA", "AAAA", "All same character"),
        ("ABAB", "BABA", "Repeated patterns"),
        
        # Very long common substring
        ("A" * 100 + "B", "X" + "A" * 100, "Long common part"),
    ]
    
    for str1, str2, description in edge_cases:
        print(f"\n{description}:")
        print(f"  Strings: '{str1[:20]}{'...' if len(str1) > 20 else ''}' & '{str2[:20]}{'...' if len(str2) > 20 else ''}'")
        
        try:
            result = lcs.lcs_dynamic_programming(str1, str2)
            print(f"  LCS: '{result}' (length: {len(result)})")
            
            # Verify result is actually common
            if result and (result not in str1 or result not in str2):
                print(f"  ERROR: Result not found in both strings!")
        
        except Exception as e:
            print(f"  Error: {e}")


def analyze_complexity():
    """Analyze time and space complexity"""
    print("\n=== Complexity Analysis ===")
    
    complexity_analysis = [
        ("Dynamic Programming",
         "Time: O(m * n) where m, n are string lengths",
         "Space: O(m * n) for DP table"),
        
        ("Space-Optimized DP",
         "Time: O(m * n)",
         "Space: O(min(m, n)) using rolling arrays"),
        
        ("Generalized Suffix Tree",
         "Time: O(m + n) for construction + O(m + n) for traversal",
         "Space: O(m + n) for suffix tree"),
        
        ("Binary Search + Rolling Hash",
         "Time: O((m + n) * log(min(m, n)))",
         "Space: O(m + n) for hash sets"),
        
        ("Multiple Strings (k strings)",
         "Time: O(sum of lengths) for trie + O(sum of lengths) for traversal",
         "Space: O(sum of lengths)"),
        
        ("With k Mismatches",
         "Time: O(m * n * k)",
         "Space: O(m * n * k)"),
        
        ("All Common Substrings",
         "Time: O(m * n * min(m, n)) worst case",
         "Space: O(number of common substrings)"),
    ]
    
    print("Approach Analysis:")
    for approach, time_complexity, space_complexity in complexity_analysis:
        print(f"\n{approach}:")
        print(f"  {time_complexity}")
        print(f"  {space_complexity}")
    
    print(f"\nComparison with LCS (Longest Common Subsequence):")
    print(f"  • LCS allows non-contiguous characters")
    print(f"  • Longest Common Substring requires contiguous characters")
    print(f"  • Both have O(m*n) DP solutions")
    print(f"  • Substring problem often has suffix tree O(m+n) solutions")
    
    print(f"\nOptimization Strategies:")
    print(f"  • Use suffix trees for multiple queries on same strings")
    print(f"  • Use rolling hash for memory-constrained environments")
    print(f"  • Use space-optimized DP when only length is needed")
    print(f"  • Consider approximate algorithms for very large strings")
    
    print(f"\nRecommendations:")
    print(f"  • Use DP for general-purpose, moderate-sized strings")
    print(f"  • Use suffix trees for multiple LCS queries")
    print(f"  • Use rolling hash for very long strings")
    print(f"  • Use optimized DP for space-critical applications")


if __name__ == "__main__":
    test_basic_lcs()
    test_multiple_strings()
    test_lcs_with_mismatches()
    test_all_common_substrings()
    demonstrate_suffix_tree_approach()
    benchmark_approaches()
    demonstrate_real_world_applications()
    test_edge_cases()
    analyze_complexity()

"""
Longest Common Substring demonstrates comprehensive substring matching approaches:

1. Dynamic Programming - Classic O(m*n) solution with full DP table
2. Space-Optimized DP - Reduce space to O(min(m,n)) using rolling arrays
3. Generalized Suffix Tree - O(m+n) solution for repeated queries
4. Binary Search + Rolling Hash - O((m+n)*log(min(m,n))) with good constants
5. Multiple Strings - Extend to find LCS among k strings
6. With Mismatches - Allow up to k character differences
7. All Common Substrings - Find all common substrings above threshold

Key applications:
- DNA sequence analysis and bioinformatics
- Plagiarism detection and text similarity
- Code clone detection
- Version control systems (diff algorithms)
- Data compression and deduplication

Each approach offers different trade-offs between time complexity,
space usage, and flexibility for various substring matching scenarios.
"""
