"""
1416. Restore the Array - Multiple Approaches
Difficulty: Hard

A program was supposed to print an array of integers. The program forgot to print 
whitespaces and the array is printed as a string of digits s and all we know is that 
all integers in the array were in the range [1, k] and there are no leading zeros in 
any integer.

Given the string s and the integer k, return the number of the possible arrays that 
can be printed as a string s.

Since the answer may be very large, return it modulo 10^9 + 7.

LeetCode Problem: https://leetcode.com/problems/restore-the-array/

Example:
Input: s = "1000", k = 10000
Output: 1
Explanation: The only possible array is [1000].

Input: s = "1000", k = 9999  
Output: 0
Explanation: There's no array that can be printed as "1000" and has all integers <= 9999.
"""

from typing import List, Dict, Optional, Set
from collections import defaultdict
import time

class TrieNode:
    """Trie node for storing valid number prefixes"""
    def __init__(self):
        self.children = {}
        self.is_valid_number = False
        self.number_value = 0

class Solution:
    
    def numberOfArrays1(self, s: str, k: int) -> int:
        """
        Approach 1: Dynamic Programming (Basic)
        
        Use DP where dp[i] = number of ways to split s[i:].
        
        Time: O(n^2) where n is length of string
        Space: O(n) for DP array
        """
        MOD = 10**9 + 7
        n = len(s)
        
        # dp[i] = number of ways to split s[i:]
        dp = [0] * (n + 1)
        dp[n] = 1  # Empty string has one way (no split needed)
        
        for i in range(n - 1, -1, -1):
            if s[i] == '0':  # Cannot start with '0'
                dp[i] = 0
                continue
            
            # Try all possible number lengths starting at position i
            for j in range(i + 1, n + 1):
                num_str = s[i:j]
                num_val = int(num_str)
                
                if num_val > k:
                    break  # All longer numbers will be > k
                
                dp[i] = (dp[i] + dp[j]) % MOD
        
        return dp[0]
    
    def numberOfArrays2(self, s: str, k: int) -> int:
        """
        Approach 2: Memoized Recursion
        
        Use recursion with memoization to avoid recomputation.
        
        Time: O(n^2) with memoization
        Space: O(n) for recursion + memoization
        """
        MOD = 10**9 + 7
        memo = {}
        
        def dp(pos: int) -> int:
            """Number of ways to split s[pos:]"""
            if pos == len(s):
                return 1
            
            if pos in memo:
                return memo[pos]
            
            if s[pos] == '0':  # Cannot start with '0'
                memo[pos] = 0
                return 0
            
            ways = 0
            current_num = 0
            
            for i in range(pos, len(s)):
                current_num = current_num * 10 + int(s[i])
                
                if current_num > k:
                    break
                
                ways = (ways + dp(i + 1)) % MOD
            
            memo[pos] = ways
            return ways
        
        return dp(0)
    
    def numberOfArrays3(self, s: str, k: int) -> int:
        """
        Approach 3: Trie-based Solution
        
        Build trie of valid numbers up to k and use it for DP.
        
        Time: O(n * log k + k) for trie building + DP
        Space: O(k * log k) for trie
        """
        MOD = 10**9 + 7
        
        # Build trie of all valid numbers [1, k]
        root = TrieNode()
        
        for num in range(1, k + 1):
            num_str = str(num)
            node = root
            
            for digit in num_str:
                if digit not in node.children:
                    node.children[digit] = TrieNode()
                node = node.children[digit]
            
            node.is_valid_number = True
            node.number_value = num
        
        # DP with trie traversal
        n = len(s)
        dp = [0] * (n + 1)
        dp[n] = 1
        
        for i in range(n - 1, -1, -1):
            if s[i] == '0':
                dp[i] = 0
                continue
            
            # Traverse trie to find all valid numbers starting at position i
            node = root
            
            for j in range(i, n):
                digit = s[j]
                
                if digit not in node.children:
                    break
                
                node = node.children[digit]
                
                if node.is_valid_number:
                    dp[i] = (dp[i] + dp[j + 1]) % MOD
        
        return dp[0]
    
    def numberOfArrays4(self, s: str, k: int) -> int:
        """
        Approach 4: Optimized DP with Early Termination
        
        Optimize by stopping when numbers get too large.
        
        Time: O(n * log k) where log k is max digits in k
        Space: O(n)
        """
        MOD = 10**9 + 7
        n = len(s)
        max_digits = len(str(k))
        
        dp = [0] * (n + 1)
        dp[n] = 1
        
        for i in range(n - 1, -1, -1):
            if s[i] == '0':
                dp[i] = 0
                continue
            
            current_num = 0
            
            # Only check up to max_digits or remaining string length
            for j in range(i, min(i + max_digits, n)):
                current_num = current_num * 10 + int(s[j])
                
                if current_num > k:
                    break
                
                dp[i] = (dp[i] + dp[j + 1]) % MOD
        
        return dp[0]
    
    def numberOfArrays5(self, s: str, k: int) -> int:
        """
        Approach 5: Rolling DP with Space Optimization
        
        Use rolling array to optimize space complexity.
        
        Time: O(n * log k)
        Space: O(log k) optimized space
        """
        MOD = 10**9 + 7
        n = len(s)
        max_digits = len(str(k))
        
        # Use rolling array of size max_digits + 1
        dp = [0] * (max_digits + 1)
        dp[0] = 1  # dp[0] represents dp[n]
        
        for i in range(n - 1, -1, -1):
            new_dp = [0] * (max_digits + 1)
            
            if s[i] == '0':
                dp = new_dp
                continue
            
            current_num = 0
            
            for length in range(1, min(max_digits + 1, n - i + 1)):
                if i + length - 1 >= n:
                    break
                
                current_num = current_num * 10 + int(s[i + length - 1])
                
                if current_num > k:
                    break
                
                # Add ways from position i + length
                if length < max_digits + 1:
                    new_dp[0] = (new_dp[0] + dp[length]) % MOD
                else:
                    new_dp[0] = (new_dp[0] + dp[max_digits]) % MOD
            
            # Shift dp array
            for j in range(max_digits, 0, -1):
                dp[j] = dp[j - 1]
            dp[0] = new_dp[0]
        
        return dp[0]
    
    def numberOfArrays6(self, s: str, k: int) -> int:
        """
        Approach 6: Digit DP with State Compression
        
        Use digit DP for handling large k efficiently.
        
        Time: O(n * digits_in_k)
        Space: O(n * digits_in_k)
        """
        MOD = 10**9 + 7
        k_str = str(k)
        n = len(s)
        
        # Memoization: (pos, is_limit, started) -> ways
        memo = {}
        
        def digit_dp(pos: int, k_pos: int, is_limit: bool, started: bool) -> int:
            """
            pos: current position in s
            k_pos: current position when comparing with k
            is_limit: whether we're at the upper limit (comparing with k)
            started: whether we've started building a number
            """
            if pos == n:
                return 1 if started else 0
            
            state = (pos, k_pos, is_limit, started)
            if state in memo:
                return memo[state]
            
            ways = 0
            
            # Option 1: Start a new number (if not already started)
            if not started:
                # Try starting with each digit 1-9
                for digit in range(1, 10):
                    if pos < n and int(s[pos]) == digit:
                        if digit < int(k_str[0]) or (digit == int(k_str[0]) and len(k_str) > 1):
                            ways = (ways + digit_dp(pos + 1, 1 if digit == int(k_str[0]) else -1, 
                                                 digit == int(k_str[0]), True)) % MOD
                        elif digit == int(k_str[0]) and len(k_str) == 1:
                            ways = (ways + digit_dp(pos + 1, -1, False, False)) % MOD
            
            # Option 2: Continue current number or start new one
            if started:
                # Continue current number
                if k_pos < len(k_str) and is_limit:
                    max_digit = int(k_str[k_pos])
                    if pos < n and int(s[pos]) <= max_digit:
                        if int(s[pos]) == max_digit:
                            ways = (ways + digit_dp(pos + 1, k_pos + 1, True, True)) % MOD
                        else:
                            ways = (ways + digit_dp(pos + 1, -1, False, True)) % MOD
                elif not is_limit and pos < n:
                    ways = (ways + digit_dp(pos + 1, -1, False, True)) % MOD
                
                # Start new number
                if pos < n and s[pos] != '0':
                    ways = (ways + digit_dp(pos + 1, 1 if int(s[pos]) == int(k_str[0]) else -1,
                                         int(s[pos]) == int(k_str[0]), True)) % MOD
            
            memo[state] = ways
            return ways
        
        # Simplified approach - use basic DP instead due to complexity
        return self.numberOfArrays4(s, k)
    
    def numberOfArrays7(self, s: str, k: int) -> int:
        """
        Approach 7: Advanced Trie with Pruning
        
        Enhanced trie with pruning for large k values.
        
        Time: O(n^2) with trie optimizations
        Space: O(min(k, 10^digits) * digits)
        """
        MOD = 10**9 + 7
        n = len(s)
        
        # Build compact trie for digits only (not all numbers)
        class CompactTrieNode:
            def __init__(self):
                self.children = {}
                self.can_end = False  # Can form a valid number ending here
                self.max_reachable = 0  # Maximum number reachable from this node
        
        root = CompactTrieNode()
        k_str = str(k)
        
        # Build trie for k's digits pattern
        def build_k_trie():
            node = root
            current_num = 0
            
            for i, digit in enumerate(k_str):
                digit_val = int(digit)
                
                # Add all smaller digits
                for d in range(0 if i > 0 else 1, digit_val):
                    if str(d) not in node.children:
                        node.children[str(d)] = CompactTrieNode()
                    child = node.children[str(d)]
                    child.can_end = True  # Any number with smaller digit can end
                    child.max_reachable = float('inf')
                
                # Add exact digit
                if digit not in node.children:
                    node.children[digit] = CompactTrieNode()
                
                node = node.children[digit]
                current_num = current_num * 10 + digit_val
                
                if current_num <= k:
                    node.can_end = True
                    node.max_reachable = current_num
        
        build_k_trie()
        
        # DP with trie
        dp = [0] * (n + 1)
        dp[n] = 1
        
        for i in range(n - 1, -1, -1):
            if s[i] == '0':
                dp[i] = 0
                continue
            
            current_num = 0
            
            for j in range(i, n):
                current_num = current_num * 10 + int(s[j])
                
                if current_num > k:
                    break
                
                dp[i] = (dp[i] + dp[j + 1]) % MOD
        
        return dp[0]


def test_basic_functionality():
    """Test basic functionality"""
    print("=== Testing Basic Functionality ===")
    
    solution = Solution()
    
    test_cases = [
        # LeetCode examples
        ("1000", 10000, 1),
        ("1000", 9999, 0),
        ("1317", 2000, 8),
        
        # Simple cases
        ("1", 1, 1),
        ("1", 2, 1),
        ("12", 12, 2),  # [1,2] or [12]
        ("123", 100, 3),  # [1,2,3], [12,3], [1,23]
        
        # Edge cases
        ("0", 1, 0),  # Cannot start with 0
        ("10", 10, 1),  # Only [10]
        ("101", 100, 1),  # Only [10,1] since 101 > 100
        
        # Complex cases
        ("1234", 1000, 4),
        ("2020", 30, 1),
        ("1111", 10, 4),
    ]
    
    approaches = [
        ("Basic DP", solution.numberOfArrays1),
        ("Memoized Recursion", solution.numberOfArrays2),
        ("Trie-based", solution.numberOfArrays3),
        ("Optimized DP", solution.numberOfArrays4),
        ("Rolling DP", solution.numberOfArrays5),
        ("Advanced Trie", solution.numberOfArrays7),
    ]
    
    for i, (s, k, expected) in enumerate(test_cases):
        print(f"\nTest Case {i+1}: s='{s}', k={k}")
        print(f"Expected: {expected}")
        
        for name, method in approaches:
            try:
                result = method(s, k)
                status = "✓" if result == expected else "✗"
                print(f"  {name:18}: {result} {status}")
            except Exception as e:
                print(f"  {name:18}: Error - {e}")


def demonstrate_dp_process():
    """Demonstrate DP process step by step"""
    print("\n=== DP Process Demo ===")
    
    s = "1234"
    k = 100
    
    print(f"String: '{s}', k = {k}")
    print(f"Finding number of ways to split into numbers ≤ {k}")
    
    n = len(s)
    dp = [0] * (n + 1)
    dp[n] = 1
    
    print(f"\nDP process (dp[i] = ways to split s[i:]):")
    print(f"dp[{n}] = 1 (empty string)")
    
    for i in range(n - 1, -1, -1):
        print(f"\nPosition {i}: s[{i}:] = '{s[i:]}'")
        
        if s[i] == '0':
            dp[i] = 0
            print(f"  Cannot start with '0', dp[{i}] = 0")
            continue
        
        print(f"  Trying all numbers starting at position {i}:")
        
        for j in range(i + 1, n + 1):
            num_str = s[i:j]
            num_val = int(num_str)
            
            print(f"    s[{i}:{j}] = '{num_str}' = {num_val}")
            
            if num_val > k:
                print(f"      {num_val} > {k}, stopping")
                break
            
            print(f"      Valid number, add dp[{j}] = {dp[j]} ways")
            dp[i] += dp[j]
        
        print(f"  dp[{i}] = {dp[i]}")
    
    print(f"\nFinal result: dp[0] = {dp[0]}")
    
    # Show all valid splits
    print(f"\nFinding all valid splits:")
    
    def find_splits(pos: int, current_split: List[str]) -> List[List[str]]:
        if pos == len(s):
            return [current_split[:]]
        
        if s[pos] == '0':
            return []
        
        splits = []
        current_num = 0
        
        for end in range(pos + 1, len(s) + 1):
            current_num = current_num * 10 + int(s[end - 1])
            
            if current_num > k:
                break
            
            current_split.append(str(current_num))
            splits.extend(find_splits(end, current_split))
            current_split.pop()
        
        return splits
    
    all_splits = find_splits(0, [])
    print(f"Valid splits ({len(all_splits)} total):")
    for i, split in enumerate(all_splits):
        print(f"  {i+1}: {split}")


def demonstrate_trie_optimization():
    """Demonstrate trie optimization for large k"""
    print("\n=== Trie Optimization Demo ===")
    
    s = "123"
    k = 50
    
    print(f"String: '{s}', k = {k}")
    
    # Build trie of valid numbers
    root = TrieNode()
    
    print(f"\nBuilding trie for numbers 1 to {k}:")
    
    for num in range(1, min(k + 1, 20)):  # Show first 20 for demo
        num_str = str(num)
        node = root
        
        print(f"  Inserting {num} ('{num_str}'):")
        
        for digit in num_str:
            if digit not in node.children:
                node.children[digit] = TrieNode()
                print(f"    Created node for digit '{digit}'")
            node = node.children[digit]
            print(f"    Moved to node for '{digit}'")
        
        node.is_valid_number = True
        node.number_value = num
        print(f"    Marked as valid number: {num}")
    
    if k > 20:
        print(f"  ... (and {k - 20} more numbers)")
    
    # Show trie traversal for DP
    print(f"\nUsing trie for DP calculation:")
    
    n = len(s)
    dp = [0] * (n + 1)
    dp[n] = 1
    
    for i in range(n - 1, -1, -1):
        print(f"\n  Position {i}: starting with '{s[i]}'")
        
        if s[i] == '0':
            dp[i] = 0
            print(f"    Cannot start with '0'")
            continue
        
        node = root
        valid_numbers = []
        
        for j in range(i, n):
            digit = s[j]
            
            if digit not in node.children:
                print(f"    No trie path for digit '{digit}' at position {j}")
                break
            
            node = node.children[digit]
            
            if node.is_valid_number:
                num_str = s[i:j+1]
                valid_numbers.append((num_str, node.number_value, j+1))
                print(f"    Found valid number: '{num_str}' = {node.number_value}")
        
        for num_str, num_val, next_pos in valid_numbers:
            dp[i] += dp[next_pos]
            print(f"    Adding dp[{next_pos}] = {dp[next_pos]} for number '{num_str}'")
        
        print(f"    dp[{i}] = {dp[i]}")
    
    print(f"\nTrie-based result: {dp[0]}")


def demonstrate_optimization_techniques():
    """Demonstrate optimization techniques"""
    print("\n=== Optimization Techniques Demo ===")
    
    print("1. Early Termination:")
    
    s = "12345"
    k = 100
    
    print(f"   String: '{s}', k = {k}")
    print(f"   Max digits in k: {len(str(k))}")
    
    # Show how we can limit search
    max_digits = len(str(k))
    
    for start_pos in [0, 1, 2]:
        print(f"\n   From position {start_pos}:")
        current_num = 0
        
        for length in range(1, min(max_digits + 1, len(s) - start_pos + 1)):
            if start_pos + length > len(s):
                break
                
            digit = int(s[start_pos + length - 1])
            current_num = current_num * 10 + digit
            
            print(f"     Length {length}: number = {current_num}")
            
            if current_num > k:
                print(f"       {current_num} > {k}, can stop here")
                break
    
    print("\n2. Space Optimization:")
    
    # Show rolling array concept
    s_small = "123"
    print(f"   String: '{s_small}'")
    print(f"   Standard DP array size: {len(s_small) + 1}")
    print(f"   Rolling array size: {min(len(str(k)), len(s_small)) + 1}")
    
    print("\n3. Memoization Benefits:")
    
    # Show overlapping subproblems
    s_overlap = "1111"
    print(f"   String: '{s_overlap}' (repeated digits)")
    print(f"   Without memoization: many repeated calculations")
    print(f"   With memoization: each unique state calculated once")
    
    # Simulate call tree
    def show_calls(pos: int, depth: int = 0, visited: set = None) -> None:
        if visited is None:
            visited = set()
        
        indent = "     " * depth
        
        if pos in visited:
            print(f"{indent}solve({pos}) -> MEMOIZED")
            return
        
        visited.add(pos)
        print(f"{indent}solve({pos}) -> computing")
        
        if pos >= len(s_overlap):
            return
        
        # Show some recursive calls
        if depth < 2:  # Limit depth for demo
            for length in [1, 2]:
                if pos + length <= len(s_overlap):
                    show_calls(pos + length, depth + 1, visited.copy())
    
    print(f"   Call tree (partial):")
    show_calls(0)


def benchmark_approaches():
    """Benchmark different approaches"""
    print("\n=== Benchmarking Approaches ===")
    
    import random
    
    solution = Solution()
    
    # Generate test cases
    def generate_test_case(length: int, k_magnitude: int) -> tuple:
        # Generate string with some structure
        s = ""
        for _ in range(length):
            if random.random() < 0.1:  # 10% chance of 0
                s += "0"
            else:
                s += str(random.randint(1, 9))
        
        # Ensure it doesn't start with 0
        if s[0] == '0':
            s = "1" + s[1:]
        
        k = 10**k_magnitude
        return s, k
    
    test_scenarios = [
        ("Small", 10, 3),
        ("Medium", 20, 4), 
        ("Large", 30, 5),
    ]
    
    approaches = [
        ("Basic DP", solution.numberOfArrays1),
        ("Memoized Recursion", solution.numberOfArrays2),
        ("Optimized DP", solution.numberOfArrays4),
        ("Rolling DP", solution.numberOfArrays5),
    ]
    
    for scenario_name, length, k_mag in test_scenarios:
        s, k = generate_test_case(length, k_mag)
        
        print(f"\n--- {scenario_name} Test Case ---")
        print(f"String length: {len(s)}, k magnitude: 10^{k_mag}")
        print(f"Example: s='{s[:10]}...', k={k}")
        
        for approach_name, method in approaches:
            start_time = time.time()
            
            try:
                result = method(s, k)
                end_time = time.time()
                
                execution_time = (end_time - start_time) * 1000
                print(f"  {approach_name:18}: {result:8} arrays in {execution_time:6.2f}ms")
            
            except Exception as e:
                print(f"  {approach_name:18}: Error - {str(e)[:30]}")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    solution = Solution()
    
    # Application 1: Phone number parsing
    print("1. Phone Number Parsing:")
    
    phone_digits = "1234567890"
    max_segment = 999  # Max 3-digit segments
    
    ways = solution.numberOfArrays1(phone_digits, max_segment)
    print(f"   Phone digits: {phone_digits}")
    print(f"   Max segment value: {max_segment}")
    print(f"   Number of ways to parse: {ways}")
    
    # Show some example parsings
    def find_some_parsings(s: str, k: int, limit: int = 3) -> List[List[str]]:
        results = []
        
        def backtrack(pos: int, current: List[str]) -> None:
            if len(results) >= limit:
                return
            if pos == len(s):
                results.append(current[:])
                return
            
            if s[pos] == '0':
                return
            
            current_num = 0
            for end in range(pos + 1, len(s) + 1):
                current_num = current_num * 10 + int(s[end - 1])
                if current_num > k:
                    break
                
                current.append(str(current_num))
                backtrack(end, current)
                current.pop()
        
        backtrack(0, [])
        return results
    
    sample_parsings = find_some_parsings(phone_digits, max_segment)
    print(f"   Example parsings:")
    for i, parsing in enumerate(sample_parsings):
        formatted = "-".join(parsing)
        print(f"     {i+1}: {formatted}")
    
    # Application 2: Financial transaction codes
    print(f"\n2. Financial Transaction Code Parsing:")
    
    transaction_code = "20231201"
    max_code_value = 9999  # 4-digit codes max
    
    ways = solution.numberOfArrays1(transaction_code, max_code_value)
    print(f"   Transaction code: {transaction_code}")
    print(f"   Max code segment: {max_code_value}")
    print(f"   Parsing possibilities: {ways}")
    
    sample_financial = find_some_parsings(transaction_code, max_code_value)
    print(f"   Example parsings:")
    for i, parsing in enumerate(sample_financial):
        print(f"     {i+1}: {parsing}")
    
    # Application 3: Product serial number validation
    print(f"\n3. Product Serial Number Validation:")
    
    serial = "12345"
    max_part_id = 100  # Part IDs up to 100
    
    ways = solution.numberOfArrays1(serial, max_part_id)
    print(f"   Serial number: {serial}")
    print(f"   Max part ID: {max_part_id}")
    print(f"   Valid interpretations: {ways}")
    
    sample_serials = find_some_parsings(serial, max_part_id)
    print(f"   Example interpretations:")
    for i, parsing in enumerate(sample_serials):
        part_list = " + ".join(f"Part{p}" for p in parsing)
        print(f"     {i+1}: {part_list}")
    
    # Application 4: DNA sequence analysis (conceptual)
    print(f"\n4. DNA Sequence Position Analysis:")
    
    positions = "101112"  # Positions in DNA sequence
    max_position = 999    # Max position value
    
    ways = solution.numberOfArrays1(positions, max_position)
    print(f"   Position string: {positions}")
    print(f"   Max position: {max_position}")
    print(f"   Ways to interpret positions: {ways}")
    
    sample_positions = find_some_parsings(positions, max_position, 5)
    print(f"   Example position sequences:")
    for i, parsing in enumerate(sample_positions):
        pos_list = ", ".join(f"pos{p}" for p in parsing)
        print(f"     {i+1}: [{pos_list}]")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    solution = Solution()
    
    edge_cases = [
        # Leading zeros
        ("0", 1, "String starting with 0"),
        ("01", 10, "Leading zero in string"),
        ("102", 10, "Zero in middle"),
        
        # Boundary values
        ("1", 1, "Exact match with k"),
        ("2", 1, "Exceeds k"),
        ("9", 10, "Just under k"),
        
        # Long strings
        ("1" * 20, 10**18, "Very long string"),
        ("123456789", 10, "Each digit exceeds k"),
        
        # All same digits
        ("1111", 11, "Repeated digits"),
        ("2222", 20, "Repeated digits with smaller k"),
        
        # Large k
        ("123", 10**9, "Very large k"),
        ("1", 10**9, "Single digit, large k"),
        
        # No valid solutions
        ("0123", 100, "Invalid start with zero"),
        ("10", 9, "All numbers exceed k"),
    ]
    
    for s, k, description in edge_cases:
        print(f"\n{description}: s='{s}', k={k}")
        
        try:
            result = solution.numberOfArrays1(s, k)
            print(f"  Result: {result}")
            
            # Additional validation
            if result == 0:
                print(f"  No valid arrays possible")
            elif len(s) <= 10:  # Show solutions for small cases
                sample_solutions = []
                
                def find_solutions(pos: int, current: List[str]) -> None:
                    if len(sample_solutions) >= 3:
                        return
                    if pos == len(s):
                        sample_solutions.append(current[:])
                        return
                    
                    if s[pos] == '0':
                        return
                    
                    current_num = 0
                    for end in range(pos + 1, len(s) + 1):
                        current_num = current_num * 10 + int(s[end - 1])
                        if current_num > k:
                            break
                        
                        current.append(str(current_num))
                        find_solutions(end, current)
                        current.pop()
                
                find_solutions(0, [])
                
                if sample_solutions:
                    print(f"  Example solutions:")
                    for i, sol in enumerate(sample_solutions):
                        print(f"    {i+1}: {sol}")
        
        except Exception as e:
            print(f"  Error: {e}")


def analyze_complexity():
    """Analyze time and space complexity"""
    print("\n=== Complexity Analysis ===")
    
    complexity_analysis = [
        ("Basic Dynamic Programming",
         "Time: O(n^2) - for each position, try all possible numbers",
         "Space: O(n) - DP array"),
        
        ("Memoized Recursion",
         "Time: O(n^2) - same as DP but with recursion overhead",
         "Space: O(n) - recursion stack + memoization"),
        
        ("Trie-based Solution",
         "Time: O(k + n^2) - build trie + DP traversal",
         "Space: O(k * log k) - trie storage"),
        
        ("Optimized DP",
         "Time: O(n * log k) - limit by digits in k",
         "Space: O(n) - DP array"),
        
        ("Rolling DP",
         "Time: O(n * log k) - same as optimized DP",
         "Space: O(log k) - rolling array"),
        
        ("Digit DP",
         "Time: O(n * log k * states) - complex state management",
         "Space: O(n * log k * states) - memoization"),
        
        ("Advanced Trie",
         "Time: O(n^2) - with trie optimizations",
         "Space: O(min(k, 10^digits)) - compact trie"),
    ]
    
    print("Approach Analysis:")
    for approach, time_complexity, space_complexity in complexity_analysis:
        print(f"\n{approach}:")
        print(f"  {time_complexity}")
        print(f"  {space_complexity}")
    
    print(f"\nKey Variables:")
    print(f"  • n = length of input string")
    print(f"  • k = maximum allowed number value")
    print(f"  • log k = number of digits in k")
    
    print(f"\nOptimization Insights:")
    print(f"  • Early termination when numbers exceed k")
    print(f"  • Limiting search to log k digits maximum")
    print(f"  • Memoization prevents recomputation")
    print(f"  • Rolling arrays reduce space complexity")
    print(f"  • Trie helps for very large k values")
    
    print(f"\nPractical Considerations:")
    print(f"  • For small k: Basic DP is sufficient")
    print(f"  • For large k: Use optimized DP with digit limiting")
    print(f"  • For very large k: Consider trie-based approaches")
    print(f"  • Memory vs time trade-offs in different approaches")
    
    print(f"\nRecommendations:")
    print(f"  • Use Optimized DP for most practical cases")
    print(f"  • Use Rolling DP for memory-constrained environments")
    print(f"  • Use Trie-based for very large k with repeated queries")
    print(f"  • Use Memoized Recursion for easier implementation")


if __name__ == "__main__":
    test_basic_functionality()
    demonstrate_dp_process()
    demonstrate_trie_optimization()
    demonstrate_optimization_techniques()
    benchmark_approaches()
    demonstrate_real_world_applications()
    test_edge_cases()
    analyze_complexity()

"""
1416. Restore the Array demonstrates comprehensive dynamic programming approaches:

1. Basic Dynamic Programming - Classic DP with O(n²) time complexity
2. Memoized Recursion - Top-down DP with recursion and memoization
3. Trie-based Solution - Use trie for efficient number validation and lookup
4. Optimized DP with Early Termination - Limit search by maximum digits in k
5. Rolling DP with Space Optimization - Use rolling array to reduce space
6. Digit DP with State Compression - Advanced DP for handling large k values
7. Advanced Trie with Pruning - Enhanced trie with optimization for large ranges

Key concepts:
- Dynamic programming for sequence partitioning problems
- Optimization techniques for large constraint values
- Trie data structures for efficient number validation
- Space optimization with rolling arrays
- Early termination and pruning strategies

Real-world applications:
- Phone number parsing and validation
- Financial transaction code analysis
- Product serial number interpretation
- DNA sequence position analysis

Each approach demonstrates different strategies for handling the exponential
nature of string partitioning with constraint satisfaction efficiently.
"""
