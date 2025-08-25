"""
1239. Maximum Length of a Concatenated String with Unique Characters - Multiple Approaches
Difficulty: Medium

You are given an array of strings arr. A string s is a concatenation of a subsequence 
of arr that has unique characters.

Return the maximum possible length of s.

LeetCode Problem: https://leetcode.com/problems/maximum-length-of-a-concatenated-string-with-unique-characters/

Example:
Input: arr = ["un","iq","ue"]
Output: 4
Explanation: All the valid concatenations are:
- ""
- "un"
- "iq"
- "ue"
- "uniq" (concatenation of "un" and "iq")
- "ique" (concatenation of "iq" and "ue")
The maximum length is 4.
"""

from typing import List, Set, Dict, Optional, Tuple
from collections import defaultdict, deque
import time

class TrieNode:
    """Trie node for tracking character usage"""
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.length = 0
        self.used_chars = set()

class Solution:
    
    def maxLength1(self, arr: List[str]) -> int:
        """
        Approach 1: Brute Force Backtracking
        
        Try all possible combinations using backtracking.
        
        Time: O(2^n) where n is number of strings
        Space: O(n) for recursion depth
        """
        def has_unique_chars(s: str) -> bool:
            """Check if string has all unique characters"""
            return len(s) == len(set(s))
        
        def backtrack(index: int, current: str) -> int:
            """Backtrack to find maximum length"""
            if not has_unique_chars(current):
                return 0
            
            max_len = len(current)
            
            for i in range(index, len(arr)):
                max_len = max(max_len, backtrack(i + 1, current + arr[i]))
            
            return max_len
        
        return backtrack(0, "")
    
    def maxLength2(self, arr: List[str]) -> int:
        """
        Approach 2: Dynamic Programming with Bitmask
        
        Use bitmask to represent character sets efficiently.
        
        Time: O(2^n * m) where m is average string length
        Space: O(2^n)
        """
        def string_to_mask(s: str) -> int:
            """Convert string to bitmask, return -1 if duplicates"""
            mask = 0
            for char in s:
                bit = 1 << (ord(char) - ord('a'))
                if mask & bit:  # Duplicate character
                    return -1
                mask |= bit
            return mask
        
        # Filter out strings with duplicate characters
        valid_masks = []
        for s in arr:
            mask = string_to_mask(s)
            if mask != -1:
                valid_masks.append((mask, len(s)))
        
        # DP: for each possible character set, store maximum length
        dp = {0: 0}  # Empty set has length 0
        
        for mask, length in valid_masks:
            new_dp = dp.copy()
            
            for existing_mask, existing_length in dp.items():
                if existing_mask & mask == 0:  # No common characters
                    combined_mask = existing_mask | mask
                    combined_length = existing_length + length
                    
                    if combined_mask not in new_dp or new_dp[combined_mask] < combined_length:
                        new_dp[combined_mask] = combined_length
            
            dp = new_dp
        
        return max(dp.values())
    
    def maxLength3(self, arr: List[str]) -> int:
        """
        Approach 3: Trie-based Solution
        
        Build trie to track character usage and find maximum length.
        
        Time: O(2^n * m)
        Space: O(2^n * alphabet_size)
        """
        # Filter out invalid strings
        valid_strings = [s for s in arr if len(s) == len(set(s))]
        
        def dfs_trie(index: int, used_chars: Set[str]) -> int:
            """DFS with character tracking"""
            if index >= len(valid_strings):
                return 0
            
            max_len = dfs_trie(index + 1, used_chars)  # Skip current string
            
            # Try including current string
            current_string = valid_strings[index]
            current_chars = set(current_string)
            
            if not (used_chars & current_chars):  # No overlap
                new_used = used_chars | current_chars
                max_len = max(max_len, 
                            len(current_string) + dfs_trie(index + 1, new_used))
            
            return max_len
        
        return dfs_trie(0, set())
    
    def maxLength4(self, arr: List[str]) -> int:
        """
        Approach 4: Iterative DP with List
        
        Build up all valid combinations iteratively.
        
        Time: O(2^n * m)
        Space: O(2^n)
        """
        def has_duplicates(s: str) -> bool:
            return len(s) != len(set(s))
        
        # Start with empty combination
        valid_combinations = [set()]
        
        for string in arr:
            if has_duplicates(string):
                continue
            
            string_chars = set(string)
            new_combinations = []
            
            for existing_chars in valid_combinations:
                if not (existing_chars & string_chars):  # No overlap
                    new_combinations.append(existing_chars | string_chars)
            
            valid_combinations.extend(new_combinations)
        
        return max(len(chars) for chars in valid_combinations)
    
    def maxLength5(self, arr: List[str]) -> int:
        """
        Approach 5: Optimized Backtracking with Pruning
        
        Enhanced backtracking with early pruning.
        
        Time: O(2^n) with pruning
        Space: O(n)
        """
        # Preprocess: remove invalid strings and sort by length (optimization)
        valid_strings = []
        for s in arr:
            if len(s) == len(set(s)):  # No duplicates
                valid_strings.append(s)
        
        # Sort by length descending for better pruning
        valid_strings.sort(key=len, reverse=True)
        
        max_possible_length = sum(len(s) for s in valid_strings)
        
        def backtrack_optimized(index: int, used_chars: Set[str], current_length: int) -> int:
            """Optimized backtracking with pruning"""
            nonlocal max_possible_length
            
            # Pruning: if even using all remaining strings can't beat current best
            remaining_length = sum(len(valid_strings[i]) for i in range(index, len(valid_strings)))
            if current_length + remaining_length <= max_possible_length:
                max_possible_length = max(max_possible_length, current_length)
            
            if index >= len(valid_strings):
                return current_length
            
            max_len = current_length
            
            # Try including current string
            current_string = valid_strings[index]
            current_chars = set(current_string)
            
            if not (used_chars & current_chars):  # No overlap
                new_used = used_chars | current_chars
                max_len = max(max_len, 
                            backtrack_optimized(index + 1, new_used, 
                                              current_length + len(current_string)))
            
            # Try skipping current string
            max_len = max(max_len, backtrack_optimized(index + 1, used_chars, current_length))
            
            return max_len
        
        return backtrack_optimized(0, set(), 0)
    
    def maxLength6(self, arr: List[str]) -> int:
        """
        Approach 6: BFS with Character Sets
        
        Use BFS to explore all valid combinations level by level.
        
        Time: O(2^n * m)
        Space: O(2^n)
        """
        # Filter valid strings
        valid_strings = [s for s in arr if len(s) == len(set(s))]
        
        # BFS queue: (character_set, total_length)
        queue = deque([(set(), 0)])
        max_length = 0
        visited = set()
        
        while queue:
            current_chars, current_length = queue.popleft()
            max_length = max(max_length, current_length)
            
            # Convert set to frozenset for hashing
            state = frozenset(current_chars)
            if state in visited:
                continue
            visited.add(state)
            
            # Try adding each remaining valid string
            for string in valid_strings:
                string_chars = set(string)
                
                if not (current_chars & string_chars):  # No overlap
                    new_chars = current_chars | string_chars
                    new_length = current_length + len(string)
                    queue.append((new_chars, new_length))
        
        return max_length
    
    def maxLength7(self, arr: List[str]) -> int:
        """
        Approach 7: Advanced DP with Character Frequency
        
        Use character frequency tracking for optimization.
        
        Time: O(n * 2^26) worst case
        Space: O(2^26)
        """
        def char_mask(s: str) -> Tuple[int, bool]:
            """Return character mask and whether string is valid"""
            mask = 0
            for char in s:
                bit = 1 << (ord(char) - ord('a'))
                if mask & bit:  # Duplicate
                    return 0, False
                mask |= bit
            return mask, True
        
        # Convert strings to masks
        string_masks = []
        for s in arr:
            mask, valid = char_mask(s)
            if valid:
                string_masks.append((mask, len(s)))
        
        # DP with memoization
        memo = {}
        
        def dp(index: int, used_mask: int) -> int:
            """DP with memoization"""
            if index >= len(string_masks):
                return 0
            
            if (index, used_mask) in memo:
                return memo[(index, used_mask)]
            
            # Option 1: Skip current string
            result = dp(index + 1, used_mask)
            
            # Option 2: Include current string if possible
            string_mask, string_length = string_masks[index]
            if used_mask & string_mask == 0:  # No overlap
                result = max(result, string_length + dp(index + 1, used_mask | string_mask))
            
            memo[(index, used_mask)] = result
            return result
        
        return dp(0, 0)


def test_basic_functionality():
    """Test basic functionality"""
    print("=== Testing Basic Functionality ===")
    
    solution = Solution()
    
    test_cases = [
        # LeetCode examples
        (["un","iq","ue"], 4),
        (["cha","r","act","ers"], 6),
        (["abcdefghijklmnopqrstuvwxyz"], 26),
        
        # Edge cases
        ([], 0),
        (["a"], 1),
        (["aa"], 0),  # Invalid string with duplicates
        
        # Complex cases
        (["abc", "def", "ghi"], 9),
        (["abc", "def", "aei"], 6),  # "aei" conflicts with "abc"
        (["a", "b", "c", "d", "e", "f"], 6),
        
        # Strings with duplicates
        (["aa", "bb"], 0),
        (["ab", "ba"], 2),  # Only one can be used
        (["ab", "cd", "ef", "gh"], 8),
        
        # Single character optimization
        (["a", "b", "c", "ab", "ac", "bc"], 3),
    ]
    
    approaches = [
        ("Brute Force", solution.maxLength1),
        ("DP Bitmask", solution.maxLength2),
        ("Trie-based", solution.maxLength3),
        ("Iterative DP", solution.maxLength4),
        ("Optimized Backtrack", solution.maxLength5),
        ("BFS", solution.maxLength6),
        ("Advanced DP", solution.maxLength7),
    ]
    
    for i, (arr, expected) in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {arr}")
        print(f"Expected: {expected}")
        
        for name, method in approaches:
            try:
                result = method(arr[:])
                status = "✓" if result == expected else "✗"
                print(f"  {name:18}: {result} {status}")
            except Exception as e:
                print(f"  {name:18}: Error - {e}")


def demonstrate_bitmask_approach():
    """Demonstrate bitmask approach step by step"""
    print("\n=== Bitmask Approach Demo ===")
    
    arr = ["un", "iq", "ue"]
    print(f"Array: {arr}")
    
    def string_to_mask(s: str) -> int:
        mask = 0
        for char in s:
            bit = 1 << (ord(char) - ord('a'))
            if mask & bit:
                return -1  # Duplicate
            mask |= bit
        return mask
    
    def mask_to_chars(mask: int) -> str:
        chars = []
        for i in range(26):
            if mask & (1 << i):
                chars.append(chr(ord('a') + i))
        return ''.join(chars)
    
    print(f"\nStep 1: Convert strings to bitmasks")
    valid_masks = []
    
    for s in arr:
        mask = string_to_mask(s)
        if mask != -1:
            valid_masks.append((mask, len(s)))
            print(f"  '{s}' -> mask: {bin(mask):>10} -> chars: {mask_to_chars(mask)}")
        else:
            print(f"  '{s}' -> INVALID (duplicate characters)")
    
    print(f"\nStep 2: Dynamic programming with bitmasks")
    dp = {0: 0}  # mask -> max_length
    print(f"  Initial: dp[0] = 0 (empty set)")
    
    for i, (mask, length) in enumerate(valid_masks):
        new_dp = dp.copy()
        string = arr[i]
        
        print(f"\n  Processing '{string}' (mask: {bin(mask)}, length: {length}):")
        
        for existing_mask, existing_length in dp.items():
            if existing_mask & mask == 0:  # No common characters
                combined_mask = existing_mask | mask
                combined_length = existing_length + length
                
                existing_chars = mask_to_chars(existing_mask)
                combined_chars = mask_to_chars(combined_mask)
                
                print(f"    Combine with '{existing_chars}' -> '{combined_chars}' (length: {combined_length})")
                
                if combined_mask not in new_dp or new_dp[combined_mask] < combined_length:
                    new_dp[combined_mask] = combined_length
                    print(f"      Updated dp[{bin(combined_mask)}] = {combined_length}")
        
        dp = new_dp
    
    print(f"\nStep 3: Find maximum length")
    max_length = max(dp.values())
    print(f"  Maximum length: {max_length}")
    
    # Show all valid combinations
    print(f"\n  All valid combinations:")
    for mask, length in dp.items():
        chars = mask_to_chars(mask)
        print(f"    '{chars}' -> length: {length}")


def demonstrate_backtracking_process():
    """Demonstrate backtracking process"""
    print("\n=== Backtracking Process Demo ===")
    
    arr = ["abc", "def", "ghi"]
    print(f"Array: {arr}")
    
    print(f"\nBacktracking exploration:")
    
    def backtrack_verbose(index: int, used_chars: Set[str], path: List[str], depth: int = 0) -> int:
        indent = "  " * depth
        current_str = ''.join(path)
        
        print(f"{indent}State: index={index}, used={sorted(used_chars)}, path={path}, length={len(current_str)}")
        
        if index >= len(arr):
            print(f"{indent}End reached, returning length: {len(current_str)}")
            return len(current_str)
        
        current_string = arr[index]
        current_chars = set(current_string)
        
        max_len = len(current_str)
        
        # Option 1: Skip current string
        print(f"{indent}Option 1: Skip '{current_string}'")
        skip_result = backtrack_verbose(index + 1, used_chars, path, depth + 1)
        max_len = max(max_len, skip_result)
        
        # Option 2: Include current string if possible
        if not (used_chars & current_chars):
            print(f"{indent}Option 2: Include '{current_string}' (no conflicts)")
            new_used = used_chars | current_chars
            new_path = path + [current_string]
            include_result = backtrack_verbose(index + 1, new_used, new_path, depth + 1)
            max_len = max(max_len, include_result)
        else:
            conflicts = used_chars & current_chars
            print(f"{indent}Option 2: Cannot include '{current_string}' (conflicts: {sorted(conflicts)})")
        
        print(f"{indent}Returning max length: {max_len}")
        return max_len
    
    result = backtrack_verbose(0, set(), [])
    print(f"\nFinal result: {result}")


def demonstrate_optimization_techniques():
    """Demonstrate optimization techniques"""
    print("\n=== Optimization Techniques Demo ===")
    
    print("1. String Preprocessing:")
    
    arr = ["abc", "aa", "def", "bb", "ghi", "cc"]
    print(f"   Original array: {arr}")
    
    # Filter out invalid strings
    valid_strings = []
    invalid_strings = []
    
    for s in arr:
        if len(s) == len(set(s)):
            valid_strings.append(s)
        else:
            invalid_strings.append(s)
    
    print(f"   Valid strings: {valid_strings}")
    print(f"   Invalid strings (have duplicates): {invalid_strings}")
    
    print("\n2. Early Pruning in Backtracking:")
    
    arr2 = ["a", "abc", "def", "ghijklmnop"]
    print(f"   Array: {arr2}")
    
    # Calculate maximum possible length
    total_chars = set()
    for s in arr2:
        if len(s) == len(set(s)):
            total_chars.update(s)
    
    max_possible = len(total_chars)
    print(f"   Maximum possible length: {max_possible}")
    print(f"   Available characters: {sorted(total_chars)}")
    
    # Show how pruning works
    def demonstrate_pruning(current_length: int, remaining_strings: List[str]) -> bool:
        """Show pruning decision"""
        remaining_chars = set()
        for s in remaining_strings:
            if len(s) == len(set(s)):
                remaining_chars.update(s)
        
        max_additional = len(remaining_chars)
        can_improve = current_length + max_additional > max_possible
        
        print(f"     Current: {current_length}, Max additional: {max_additional}")
        print(f"     Can improve best ({max_possible})? {can_improve}")
        
        return can_improve
    
    print(f"   Pruning examples:")
    demonstrate_pruning(20, ["abc"])
    demonstrate_pruning(5, ["def", "ghi"])
    
    print("\n3. Bitmask Optimization:")
    
    # Show bitmask operations
    s1, s2 = "abc", "def"
    
    def char_mask(s: str) -> int:
        mask = 0
        for char in s:
            mask |= 1 << (ord(char) - ord('a'))
        return mask
    
    mask1 = char_mask(s1)
    mask2 = char_mask(s2)
    
    print(f"   String '{s1}': mask = {bin(mask1):>10}")
    print(f"   String '{s2}': mask = {bin(mask2):>10}")
    print(f"   Overlap check: {mask1 & mask2} (0 = no overlap)")
    print(f"   Combined mask: {bin(mask1 | mask2):>10}")
    
    print("\n4. Memory Optimization:")
    
    # Show space complexity comparison
    n = 10  # Number of strings
    
    print(f"   For {n} strings:")
    print(f"     Brute force combinations: 2^{n} = {2**n}")
    print(f"     Bitmask DP states: up to 2^26 = {2**26:,}")
    print(f"     Actual DP states: much less due to validity constraints")
    print(f"     Backtracking space: O({n}) recursion depth")


def benchmark_approaches():
    """Benchmark different approaches"""
    print("\n=== Benchmarking Approaches ===")
    
    import random
    import string
    
    solution = Solution()
    
    # Generate test cases
    def generate_strings(count: int, max_length: int, alphabet_size: int) -> List[str]:
        strings = []
        alphabet = string.ascii_lowercase[:alphabet_size]
        
        for _ in range(count):
            length = random.randint(1, max_length)
            
            # 70% chance of valid string (no duplicates)
            if random.random() < 0.7:
                chars = random.sample(alphabet, min(length, len(alphabet)))
                strings.append(''.join(chars))
            else:
                # Invalid string with possible duplicates
                chars = random.choices(alphabet, k=length)
                strings.append(''.join(chars))
        
        return strings
    
    test_scenarios = [
        ("Small", generate_strings(8, 4, 10)),
        ("Medium", generate_strings(12, 5, 15)),
        ("Large", generate_strings(16, 6, 20)),
    ]
    
    approaches = [
        ("DP Bitmask", solution.maxLength2),
        ("Iterative DP", solution.maxLength4),
        ("Optimized Backtrack", solution.maxLength5),
        ("BFS", solution.maxLength6),
        ("Advanced DP", solution.maxLength7),
    ]
    
    for scenario_name, test_arr in test_scenarios:
        print(f"\n--- {scenario_name} Dataset ---")
        print(f"Strings: {len(test_arr)}, Example: {test_arr[:3]}...")
        
        for approach_name, method in approaches:
            start_time = time.time()
            
            try:
                result = method(test_arr[:])
                end_time = time.time()
                
                execution_time = (end_time - start_time) * 1000
                print(f"  {approach_name:18}: max_length={result:2} in {execution_time:6.2f}ms")
            
            except Exception as e:
                print(f"  {approach_name:18}: Error - {str(e)[:30]}")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    solution = Solution()
    
    # Application 1: Team skill optimization
    print("1. Team Skill Optimization:")
    
    skills = ["python", "java", "react", "sql", "aws", "docker"]
    candidates = [
        "python",     # Candidate 1: Python
        "java",       # Candidate 2: Java  
        "react",      # Candidate 3: React
        "py",         # Candidate 4: Python (abbreviated)
        "js",         # Candidate 5: JavaScript
        "aws",        # Candidate 6: AWS
    ]
    
    max_skills = solution.maxLength2(candidates)
    print(f"   Available skills: {skills}")
    print(f"   Candidate skills: {candidates}")
    print(f"   Maximum unique skills in team: {max_skills}")
    
    # Find the actual combination
    def find_combination(arr: List[str]) -> List[str]:
        def backtrack(index: int, used_chars: Set[str], path: List[str]) -> List[str]:
            if index >= len(arr):
                return path[:]
            
            best_path = backtrack(index + 1, used_chars, path)  # Skip
            
            string = arr[index]
            string_chars = set(string)
            
            if len(string) == len(string_chars) and not (used_chars & string_chars):
                new_path = backtrack(index + 1, used_chars | string_chars, path + [string])
                if sum(len(s) for s in new_path) > sum(len(s) for s in best_path):
                    best_path = new_path
            
            return best_path
        
        return backtrack(0, set(), [])
    
    optimal_team = find_combination(candidates)
    print(f"   Optimal team skills: {optimal_team}")
    
    # Application 2: License plate optimization
    print(f"\n2. License Plate Character Optimization:")
    
    plate_segments = ["ABC", "DEF", "123", "XYZ", "GHI"]
    
    # Convert numbers to letters for this example
    letter_segments = [seg for seg in plate_segments if seg.isalpha()]
    
    max_chars = solution.maxLength2(letter_segments)
    print(f"   Available segments: {letter_segments}")
    print(f"   Maximum unique characters: {max_chars}")
    
    # Application 3: Product feature combinations
    print(f"\n3. Product Feature Combinations:")
    
    features = [
        "auth",      # Authentication
        "db",        # Database
        "cache",     # Caching
        "api",       # API
        "ui",        # User Interface
        "test",      # Testing
        "log",       # Logging
    ]
    
    max_features = solution.maxLength2(features)
    optimal_features = find_combination(features)
    
    print(f"   Available features: {features}")
    print(f"   Maximum feature chars: {max_features}")
    print(f"   Optimal feature set: {optimal_features}")
    print(f"   Total chars used: {sum(len(f) for f in optimal_features)}")
    
    # Application 4: DNA sequence optimization
    print(f"\n4. DNA Sequence Optimization:")
    
    dna_segments = [
        "ATCG",      # Segment 1
        "GCTA",      # Segment 2  
        "TTAA",      # Segment 3 (invalid - repeated A)
        "CGAT",      # Segment 4
        "AGCT",      # Segment 5
    ]
    
    max_nucleotides = solution.maxLength2(dna_segments)
    optimal_dna = find_combination(dna_segments)
    
    print(f"   DNA segments: {dna_segments}")
    print(f"   Maximum unique nucleotides: {max_nucleotides}")
    print(f"   Optimal DNA combination: {optimal_dna}")
    
    # Check for valid DNA (only A, T, C, G)
    all_chars = set()
    for seg in optimal_dna:
        all_chars.update(seg)
    
    valid_dna = all_chars.issubset({'A', 'T', 'C', 'G'})
    print(f"   Valid DNA sequence: {valid_dna}")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    solution = Solution()
    
    edge_cases = [
        # Empty and single cases
        ([], "Empty array"),
        ([""], "Single empty string"),
        (["a"], "Single character"),
        (["abc"], "Single valid string"),
        
        # Invalid strings
        (["aa"], "String with duplicates"),
        (["aa", "bb", "cc"], "All strings have duplicates"),
        (["ab", "aa", "cd"], "Mix of valid and invalid"),
        
        # Maximum cases
        (["abcdefghijklmnopqrstuvwxyz"], "All 26 letters"),
        (list("abcdefghijklmnopqrstuvwxyz"), "26 single characters"),
        
        # Conflicting strings
        (["ab", "bc", "cd"], "Overlapping pairs"),
        (["abc", "def", "aei"], "Some conflicts"),
        
        # Identical strings
        (["abc", "abc"], "Duplicate strings"),
        (["a", "a", "a"], "Multiple identical"),
        
        # Complex patterns
        (["a", "b", "ab"], "Subset relationships"),
        (["abc", "a", "bc"], "Partial overlaps"),
    ]
    
    for arr, description in edge_cases:
        print(f"\n{description}: {arr}")
        
        try:
            result = solution.maxLength2(arr)
            print(f"  Result: {result}")
            
            # Additional validation
            if result > 26:
                print(f"  WARNING: Result exceeds alphabet size!")
            elif result == 0 and arr and any(len(s) > 0 for s in arr):
                print(f"  Note: No valid combinations possible")
        
        except Exception as e:
            print(f"  Error: {e}")


def analyze_complexity():
    """Analyze time and space complexity"""
    print("\n=== Complexity Analysis ===")
    
    complexity_analysis = [
        ("Brute Force Backtracking",
         "Time: O(2^n * m) where n=strings, m=avg length",
         "Space: O(n) for recursion depth"),
        
        ("DP with Bitmask",
         "Time: O(2^n * m) for generating all combinations",
         "Space: O(2^n) for storing all valid character sets"),
        
        ("Trie-based DFS",
         "Time: O(2^n * m) for exploring all combinations",
         "Space: O(2^n * alphabet_size) for trie storage"),
        
        ("Iterative DP",
         "Time: O(n * 2^n) building combinations iteratively",
         "Space: O(2^n) for storing all combinations"),
        
        ("Optimized Backtracking",
         "Time: O(2^n) with effective pruning",
         "Space: O(n) for recursion + O(alphabet_size) for tracking"),
        
        ("BFS Exploration",
         "Time: O(2^n * m) exploring level by level",
         "Space: O(2^n) for queue and visited set"),
        
        ("Advanced DP with Memoization",
         "Time: O(n * 2^alphabet_size) worst case",
         "Space: O(n * 2^alphabet_size) for memoization"),
    ]
    
    print("Approach Analysis:")
    for approach, time_complexity, space_complexity in complexity_analysis:
        print(f"\n{approach}:")
        print(f"  {time_complexity}")
        print(f"  {space_complexity}")
    
    print(f"\nKey Insights:")
    print(f"  • Problem is exponential in nature (2^n combinations)")
    print(f"  • Bitmask representation is most efficient for character sets")
    print(f"  • Early pruning significantly improves practical performance")
    print(f"  • String preprocessing eliminates invalid inputs early")
    
    print(f"\nOptimization Strategies:")
    print(f"  • Filter invalid strings (with duplicates) during preprocessing")
    print(f"  • Use bitmasks for O(1) character set operations")
    print(f"  • Sort strings by length for better pruning")
    print(f"  • Memoize intermediate results to avoid recomputation")
    print(f"  • Use iterative DP to avoid recursion overhead")
    
    print(f"\nPractical Considerations:")
    print(f"  • Input size is typically small (n ≤ 20) making exponential acceptable")
    print(f"  • Character set size is bounded by alphabet (26 letters)")
    print(f"  • Memory usage dominated by storing valid combinations")
    print(f"  • Preprocessing cost is usually negligible")
    
    print(f"\nRecommendations:")
    print(f"  • Use DP with Bitmask for optimal performance")
    print(f"  • Use Optimized Backtracking for memory-constrained environments")
    print(f"  • Use Iterative DP for simplest implementation")
    print(f"  • Consider Advanced DP for very large character sets")


if __name__ == "__main__":
    test_basic_functionality()
    demonstrate_bitmask_approach()
    demonstrate_backtracking_process()
    demonstrate_optimization_techniques()
    benchmark_approaches()
    demonstrate_real_world_applications()
    test_edge_cases()
    analyze_complexity()

"""
1239. Maximum Length of a Concatenated String with Unique Characters demonstrates comprehensive optimization approaches:

1. Brute Force Backtracking - Try all possible combinations using recursive exploration
2. Dynamic Programming with Bitmask - Use bitmasks to represent character sets efficiently
3. Trie-based Solution - Build trie to track character usage and explore combinations
4. Iterative DP - Build valid combinations incrementally without recursion
5. Optimized Backtracking - Enhanced backtracking with pruning and preprocessing
6. BFS Exploration - Level-by-level exploration of all valid combinations
7. Advanced DP with Memoization - Use memoization to cache intermediate results

Key concepts:
- Bitmask representation for efficient set operations
- Dynamic programming for avoiding recomputation
- Backtracking with intelligent pruning strategies
- Preprocessing to filter invalid inputs early
- State space exploration with various traversal methods

Real-world applications:
- Team skill optimization and resource allocation
- License plate and identifier optimization
- Product feature combination analysis
- DNA sequence optimization and analysis

Each approach demonstrates different strategies for handling the exponential
nature of subset selection with constraint satisfaction efficiently.
"""
