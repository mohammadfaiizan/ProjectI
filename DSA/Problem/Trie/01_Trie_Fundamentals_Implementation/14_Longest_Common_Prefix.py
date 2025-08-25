"""
14. Longest Common Prefix - Multiple Approaches
Difficulty: Easy

Write a function to find the longest common prefix string amongst an array of strings.
If there is no common prefix, return an empty string "".

LeetCode Problem: https://leetcode.com/problems/longest-common-prefix/

Examples:
Input: strs = ["flower","flow","flight"]
Output: "fl"

Input: strs = ["dog","racecar","car"]
Output: ""
"""

from typing import List, Optional

class TrieNode:
    """Trie node for prefix-based approach"""
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.word_count = 0  # Track how many words pass through this node

class Solution:
    
    def longestCommonPrefix1(self, strs: List[str]) -> str:
        """
        Approach 1: Vertical Scanning
        
        Compare characters at each position across all strings.
        
        Time: O(S) where S is sum of all characters in all strings
        Space: O(1)
        """
        if not strs:
            return ""
        
        # Use first string as reference
        for i in range(len(strs[0])):
            char = strs[0][i]
            
            # Check if this character matches in all other strings
            for j in range(1, len(strs)):
                # Check bounds and character match
                if i >= len(strs[j]) or strs[j][i] != char:
                    return strs[0][:i]
        
        return strs[0]
    
    def longestCommonPrefix2(self, strs: List[str]) -> str:
        """
        Approach 2: Horizontal Scanning
        
        Find LCP between first two strings, then with third, and so on.
        
        Time: O(S) where S is sum of all characters
        Space: O(1)
        """
        if not strs:
            return ""
        
        prefix = strs[0]
        
        for i in range(1, len(strs)):
            # Find LCP between current prefix and strs[i]
            while strs[i][:len(prefix)] != prefix and prefix:
                prefix = prefix[:-1]
            
            if not prefix:
                break
        
        return prefix
    
    def longestCommonPrefix3(self, strs: List[str]) -> str:
        """
        Approach 3: Divide and Conquer
        
        Recursively divide the problem and combine results.
        
        Time: O(S) where S is sum of all characters
        Space: O(m * log n) where m is LCP length, n is number of strings
        """
        if not strs:
            return ""
        
        def lcp_of_two(str1: str, str2: str) -> str:
            """Find LCP of two strings"""
            min_len = min(len(str1), len(str2))
            for i in range(min_len):
                if str1[i] != str2[i]:
                    return str1[:i]
            return str1[:min_len]
        
        def divide_conquer(start: int, end: int) -> str:
            if start == end:
                return strs[start]
            
            mid = (start + end) // 2
            left_lcp = divide_conquer(start, mid)
            right_lcp = divide_conquer(mid + 1, end)
            
            return lcp_of_two(left_lcp, right_lcp)
        
        return divide_conquer(0, len(strs) - 1)
    
    def longestCommonPrefix4(self, strs: List[str]) -> str:
        """
        Approach 4: Binary Search
        
        Binary search on the length of the common prefix.
        
        Time: O(S * log m) where S is sum of characters, m is min string length
        Space: O(1)
        """
        if not strs:
            return ""
        
        def is_common_prefix(length: int) -> bool:
            """Check if first 'length' characters form a common prefix"""
            if length > len(strs[0]):
                return False
            
            prefix = strs[0][:length]
            for i in range(1, len(strs)):
                if not strs[i].startswith(prefix):
                    return False
            return True
        
        # Binary search on prefix length
        min_len = min(len(s) for s in strs)
        low, high = 0, min_len
        
        while low < high:
            mid = (low + high + 1) // 2  # Use upper mid to avoid infinite loop
            
            if is_common_prefix(mid):
                low = mid
            else:
                high = mid - 1
        
        return strs[0][:low]
    
    def longestCommonPrefix5(self, strs: List[str]) -> str:
        """
        Approach 5: Trie-based Solution
        
        Build a trie and find the path where all words converge.
        
        Time: O(S) where S is sum of all characters
        Space: O(S) for trie storage
        """
        if not strs:
            return ""
        
        # Build trie
        root = TrieNode()
        
        # Insert all strings
        for word in strs:
            node = root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
                node.word_count += 1  # Increment count for this node
            node.is_end_of_word = True
        
        # Find LCP by traversing trie
        lcp = ""
        node = root
        total_words = len(strs)
        
        while (len(node.children) == 1 and 
               not node.is_end_of_word and 
               node.word_count == total_words):
            
            char = next(iter(node.children.keys()))
            lcp += char
            node = node.children[char]
            
            # Stop if we reach a node where not all words pass through
            if node.word_count < total_words:
                break
        
        return lcp
    
    def longestCommonPrefix6(self, strs: List[str]) -> str:
        """
        Approach 6: Python Built-in Optimized
        
        Use Python's built-in functions for efficiency.
        
        Time: O(S) where S is sum of all characters
        Space: O(1)
        """
        if not strs:
            return ""
        
        # Use zip to get characters at each position
        # zip(*strs) transposes the strings character by character
        for i, chars in enumerate(zip(*strs)):
            if len(set(chars)) > 1:  # If characters are not all the same
                return strs[0][:i]
        
        # If we get here, one string is a prefix of all others
        return min(strs, key=len)
    
    def longestCommonPrefix7(self, strs: List[str]) -> str:
        """
        Approach 7: Character Frequency Approach
        
        Count character frequencies at each position.
        
        Time: O(S) where S is sum of all characters
        Space: O(k) where k is alphabet size
        """
        if not strs:
            return ""
        
        min_len = min(len(s) for s in strs)
        total_strings = len(strs)
        
        for i in range(min_len):
            char_count = {}
            
            # Count characters at position i
            for string in strs:
                char = string[i]
                char_count[char] = char_count.get(char, 0) + 1
            
            # If any character appears less than total_strings times,
            # then position i is where LCP ends
            if max(char_count.values()) < total_strings:
                return strs[0][:i]
        
        return strs[0][:min_len]


def test_basic_cases():
    """Test basic LCP cases"""
    print("=== Testing Basic Cases ===")
    
    solution = Solution()
    
    test_cases = [
        (["flower", "flow", "flight"], "fl"),
        (["dog", "racecar", "car"], ""),
        (["interspecies", "interstellar", "interstate"], "inters"),
        (["throne", "throne"], "throne"),
        ([""], ""),
        (["a"], "a"),
        (["ab", "a"], "a"),
        (["abab", "aba", "abc"], "ab"),
    ]
    
    approaches = [
        ("Vertical Scanning", solution.longestCommonPrefix1),
        ("Horizontal Scanning", solution.longestCommonPrefix2),
        ("Divide & Conquer", solution.longestCommonPrefix3),
        ("Binary Search", solution.longestCommonPrefix4),
        ("Trie-based", solution.longestCommonPrefix5),
        ("Python Optimized", solution.longestCommonPrefix6),
        ("Character Frequency", solution.longestCommonPrefix7),
    ]
    
    for i, (strs, expected) in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {strs}")
        print(f"Expected: '{expected}'")
        
        for name, method in approaches:
            try:
                result = method(strs)
                status = "✓" if result == expected else "✗"
                print(f"  {name:18}: '{result}' {status}")
            except Exception as e:
                print(f"  {name:18}: Error - {e}")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    solution = Solution()
    
    edge_cases = [
        ([], ""),  # Empty array
        ([""], ""),  # Single empty string
        (["", "abc"], ""),  # Empty string with non-empty
        (["abc", ""], ""),  # Non-empty with empty string
        (["a" * 1000], "a" * 1000),  # Very long single string
        (["a" * 1000, "a" * 999], "a" * 999),  # Long strings with difference
        (["same", "same", "same"], "same"),  # All identical strings
        (["ab", "abc", "abcd"], "ab"),  # Progressive lengths
    ]
    
    for i, (strs, expected) in enumerate(edge_cases):
        print(f"\nEdge Case {i+1}: {strs if len(str(strs)) < 50 else f'[{len(strs)} strings...]'}")
        print(f"Expected: '{expected if len(expected) < 20 else expected[:20] + '...'}'")
        
        # Test with a few approaches
        approaches = [
            ("Vertical", solution.longestCommonPrefix1),
            ("Trie", solution.longestCommonPrefix5),
            ("Python", solution.longestCommonPrefix6),
        ]
        
        for name, method in approaches:
            try:
                result = method(strs)
                status = "✓" if result == expected else "✗"
                result_display = result if len(result) < 20 else result[:20] + "..."
                print(f"  {name:10}: '{result_display}' {status}")
            except Exception as e:
                print(f"  {name:10}: Error - {e}")


def benchmark_approaches():
    """Benchmark different approaches"""
    print("\n=== Benchmarking Approaches ===")
    
    import time
    import random
    import string
    
    solution = Solution()
    
    # Generate test data
    def generate_test_data(n_strings: int, avg_length: int, common_prefix_len: int):
        common_prefix = ''.join(random.choices(string.ascii_lowercase, k=common_prefix_len))
        
        strings = []
        for _ in range(n_strings):
            suffix_len = max(0, avg_length - common_prefix_len + random.randint(-2, 2))
            suffix = ''.join(random.choices(string.ascii_lowercase, k=suffix_len))
            strings.append(common_prefix + suffix)
        
        return strings
    
    test_datasets = [
        ("Small strings", generate_test_data(10, 20, 5)),
        ("Medium strings", generate_test_data(100, 50, 10)),
        ("Many strings", generate_test_data(1000, 30, 8)),
        ("Long prefix", generate_test_data(50, 100, 50)),
    ]
    
    approaches = [
        ("Vertical", solution.longestCommonPrefix1),
        ("Horizontal", solution.longestCommonPrefix2),
        ("Divide&Conquer", solution.longestCommonPrefix3),
        ("Binary Search", solution.longestCommonPrefix4),
        ("Trie", solution.longestCommonPrefix5),
        ("Python Optimized", solution.longestCommonPrefix6),
    ]
    
    for dataset_name, test_data in test_datasets:
        print(f"\n--- {dataset_name} ({len(test_data)} strings) ---")
        
        for approach_name, method in approaches:
            start_time = time.time()
            
            # Run multiple times for better measurement
            for _ in range(10):
                result = method(test_data)
            
            end_time = time.time()
            avg_time = (end_time - start_time) / 10
            
            print(f"  {approach_name:15}: {avg_time*1000:.2f} ms (result: '{result[:10]}{'...' if len(result) > 10 else ''}')")


def analyze_complexity():
    """Analyze time and space complexity of each approach"""
    print("\n=== Complexity Analysis ===")
    
    complexity_analysis = [
        ("Vertical Scanning", 
         "Time: O(S) - S is sum of all characters",
         "Space: O(1) - constant extra space"),
        
        ("Horizontal Scanning",
         "Time: O(S) - in worst case, compares all characters",
         "Space: O(1) - constant extra space"),
        
        ("Divide and Conquer",
         "Time: O(S) - each character examined once",
         "Space: O(m*log n) - recursion stack, m=LCP length, n=number of strings"),
        
        ("Binary Search",
         "Time: O(S*log m) - S characters checked log m times, m=min string length", 
         "Space: O(1) - constant extra space"),
        
        ("Trie-based",
         "Time: O(S) - build trie then traverse",
         "Space: O(S) - trie storage"),
        
        ("Python Optimized",
         "Time: O(S) - zip and iteration",
         "Space: O(m) - storing character tuples, m=min string length"),
        
        ("Character Frequency",
         "Time: O(S) - count characters at each position",
         "Space: O(k*m) - k=alphabet size, m=min string length"),
    ]
    
    for approach, time_complexity, space_complexity in complexity_analysis:
        print(f"\n{approach}:")
        print(f"  {time_complexity}")
        print(f"  {space_complexity}")


def demonstrate_trie_visualization():
    """Visualize how trie-based approach works"""
    print("\n=== Trie Visualization Demo ===")
    
    strs = ["flower", "flow", "flight"]
    print(f"Finding LCP of: {strs}")
    
    # Build trie with detailed tracking
    root = TrieNode()
    
    print(f"\nBuilding Trie:")
    for word in strs:
        print(f"  Inserting '{word}'")
        node = root
        path = ""
        for char in word:
            path += char
            if char not in node.children:
                node.children[char] = TrieNode()
                print(f"    Created node for '{path}'")
            node = node.children[char]
            node.word_count += 1
            print(f"    Node '{path}' now has count: {node.word_count}")
        node.is_end_of_word = True
    
    # Traverse to find LCP
    print(f"\nTraversing for LCP:")
    lcp = ""
    node = root
    total_words = len(strs)
    
    while (len(node.children) == 1 and 
           not node.is_end_of_word and 
           node.word_count == total_words):
        
        char = next(iter(node.children.keys()))
        lcp += char
        node = node.children[char]
        
        print(f"  Added '{char}' to LCP: '{lcp}'")
        print(f"  Current node has {node.word_count} words, {len(node.children)} children")
        
        if node.word_count < total_words:
            print(f"  Stopping: not all words pass through this node")
            break
    
    print(f"\nFinal LCP: '{lcp}'")


def demonstrate_real_world_applications():
    """Show real-world applications of LCP"""
    print("\n=== Real-World Applications ===")
    
    solution = Solution()
    
    # Application 1: File path optimization
    print("1. File Path Optimization:")
    file_paths = [
        "/home/user/documents/projects/python/app.py",
        "/home/user/documents/projects/python/tests.py", 
        "/home/user/documents/projects/python/utils.py"
    ]
    
    common_path = solution.longestCommonPrefix1(file_paths)
    print(f"   Paths: {[path.split('/')[-1] for path in file_paths]}")
    print(f"   Common base path: '{common_path}'")
    
    # Application 2: URL routing optimization
    print("\n2. URL Routing:")
    urls = [
        "/api/v1/users/profile",
        "/api/v1/users/settings",
        "/api/v1/users/friends"
    ]
    
    common_route = solution.longestCommonPrefix1(urls)
    print(f"   URLs: {urls}")
    print(f"   Common route prefix: '{common_route}'")
    
    # Application 3: Database table prefixes
    print("\n3. Database Table Prefixes:")
    table_names = [
        "user_profiles",
        "user_settings", 
        "user_preferences",
        "user_sessions"
    ]
    
    common_prefix = solution.longestCommonPrefix1(table_names)
    print(f"   Tables: {table_names}")
    print(f"   Common prefix: '{common_prefix}'")
    
    # Application 4: String compression hint
    print("\n4. String Compression Hint:")
    strings = ["programming", "program", "programmer", "programmes"]
    
    lcp = solution.longestCommonPrefix1(strings)
    total_chars = sum(len(s) for s in strings)
    saved_chars = len(lcp) * (len(strings) - 1)
    
    print(f"   Strings: {strings}")
    print(f"   LCP: '{lcp}' (length: {len(lcp)})")
    print(f"   Potential compression: {saved_chars}/{total_chars} characters ({saved_chars/total_chars*100:.1f}%)")


if __name__ == "__main__":
    test_basic_cases()
    test_edge_cases()
    benchmark_approaches()
    analyze_complexity()
    demonstrate_trie_visualization()
    demonstrate_real_world_applications()

"""
14. Longest Common Prefix demonstrates multiple algorithmic approaches:

1. Vertical Scanning - Character-by-character comparison across all strings
2. Horizontal Scanning - Progressive LCP computation between pairs
3. Divide and Conquer - Recursive problem decomposition
4. Binary Search - Search space optimization on prefix length
5. Trie-based - Tree structure for prefix analysis
6. Python Optimized - Leveraging built-in functions
7. Character Frequency - Statistical approach to prefix detection

Each approach offers different trade-offs in terms of time complexity,
space usage, and implementation complexity, suitable for various scenarios.
"""
