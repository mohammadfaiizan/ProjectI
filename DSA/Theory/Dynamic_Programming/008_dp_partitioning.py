"""
Dynamic Programming - Partitioning Patterns
This module implements various DP problems involving optimal partitioning including
matrix chain multiplication, palindrome partitioning, boolean parenthesization, and interval DP.
"""

from typing import List, Dict, Tuple, Optional
import time
import math

# ==================== PALINDROME PARTITIONING ====================

class PalindromePartitioning:
    """
    Palindrome Partitioning Problems
    
    Find optimal ways to partition a string into palindromic substrings
    with various optimization criteria.
    """
    
    def min_cuts_palindrome_partition(self, s: str) -> int:
        """
        Find minimum cuts to partition string into palindromes
        
        LeetCode 132 - Palindrome Partitioning II
        
        Time Complexity: O(n²)
        Space Complexity: O(n²)
        
        Args:
            s: Input string
        
        Returns:
            Minimum number of cuts needed
        """
        n = len(s)
        if n <= 1:
            return 0
        
        # is_palindrome[i][j] = True if s[i:j+1] is palindrome
        is_palindrome = [[False] * n for _ in range(n)]
        
        # Every single character is palindrome
        for i in range(n):
            is_palindrome[i][i] = True
        
        # Check for 2-character palindromes
        for i in range(n - 1):
            is_palindrome[i][i + 1] = (s[i] == s[i + 1])
        
        # Check for palindromes of length 3 and more
        for length in range(3, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                is_palindrome[i][j] = (s[i] == s[j] and is_palindrome[i + 1][j - 1])
        
        # dp[i] = minimum cuts needed for s[0:i+1]
        dp = [0] * n
        
        for i in range(n):
            if is_palindrome[0][i]:
                dp[i] = 0  # Entire substring is palindrome
            else:
                dp[i] = float('inf')
                for j in range(i):
                    if is_palindrome[j + 1][i]:
                        dp[i] = min(dp[i], dp[j] + 1)
        
        return dp[n - 1]
    
    def min_cuts_optimized(self, s: str) -> int:
        """
        Optimized version with Manacher's algorithm inspiration
        
        Time Complexity: O(n²)
        Space Complexity: O(n)
        """
        n = len(s)
        if n <= 1:
            return 0
        
        # dp[i] = min cuts for s[0:i]
        dp = list(range(n))  # Worst case: every character is a cut
        
        for center in range(n):
            # Odd length palindromes
            left = right = center
            while left >= 0 and right < n and s[left] == s[right]:
                if left == 0:
                    dp[right] = 0
                else:
                    dp[right] = min(dp[right], dp[left - 1] + 1)
                left -= 1
                right += 1
            
            # Even length palindromes
            left, right = center, center + 1
            while left >= 0 and right < n and s[left] == s[right]:
                if left == 0:
                    dp[right] = 0
                else:
                    dp[right] = min(dp[right], dp[left - 1] + 1)
                left -= 1
                right += 1
        
        return dp[n - 1]
    
    def all_palindrome_partitions(self, s: str) -> List[List[str]]:
        """
        Find all possible palindrome partitions
        
        LeetCode 131 - Palindrome Partitioning
        
        Time Complexity: O(n * 2^n)
        Space Complexity: O(n²)
        """
        n = len(s)
        if not s:
            return [[]]
        
        # Precompute palindrome check
        is_palindrome = [[False] * n for _ in range(n)]
        
        for i in range(n):
            is_palindrome[i][i] = True
        
        for i in range(n - 1):
            is_palindrome[i][i + 1] = (s[i] == s[i + 1])
        
        for length in range(3, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                is_palindrome[i][j] = (s[i] == s[j] and is_palindrome[i + 1][j - 1])
        
        result = []
        
        def backtrack(start: int, current_partition: List[str]):
            if start == n:
                result.append(current_partition[:])
                return
            
            for end in range(start, n):
                if is_palindrome[start][end]:
                    current_partition.append(s[start:end + 1])
                    backtrack(end + 1, current_partition)
                    current_partition.pop()
        
        backtrack(0, [])
        return result
    
    def min_insertions_palindrome(self, s: str) -> int:
        """
        Minimum insertions to make entire string palindrome
        
        This is different from partitioning - we want one big palindrome
        
        Time Complexity: O(n²)
        Space Complexity: O(n²)
        """
        n = len(s)
        
        # dp[i][j] = min insertions to make s[i:j+1] palindrome
        dp = [[0] * n for _ in range(n)]
        
        # Length 2 substrings
        for i in range(n - 1):
            if s[i] != s[i + 1]:
                dp[i][i + 1] = 1
        
        # Length 3 and more
        for length in range(3, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                
                if s[i] == s[j]:
                    dp[i][j] = dp[i + 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i + 1][j], dp[i][j - 1])
        
        return dp[0][n - 1]

# ==================== MATRIX CHAIN MULTIPLICATION ====================

class MatrixChainMultiplication:
    """
    Matrix Chain Multiplication Problems
    
    Find optimal order to multiply a chain of matrices to minimize
    scalar multiplications.
    """
    
    def matrix_chain_order(self, dimensions: List[int]) -> int:
        """
        Find minimum scalar multiplications for matrix chain
        
        Classic MCM problem
        
        Time Complexity: O(n³)
        Space Complexity: O(n²)
        
        Args:
            dimensions: Array where matrix i has dimensions dimensions[i-1] x dimensions[i]
        
        Returns:
            Minimum number of scalar multiplications
        """
        n = len(dimensions) - 1  # Number of matrices
        if n <= 1:
            return 0
        
        # dp[i][j] = minimum multiplications for matrices from i to j
        dp = [[0] * n for _ in range(n)]
        
        # l is chain length
        for l in range(2, n + 1):  # Chain length from 2 to n
            for i in range(n - l + 1):
                j = i + l - 1
                dp[i][j] = float('inf')
                
                for k in range(i, j):
                    cost = (dp[i][k] + dp[k + 1][j] + 
                           dimensions[i] * dimensions[k + 1] * dimensions[j + 1])
                    dp[i][j] = min(dp[i][j], cost)
        
        return dp[0][n - 1]
    
    def matrix_chain_order_with_parentheses(self, dimensions: List[int]) -> Tuple[int, str]:
        """
        Find minimum cost and optimal parenthesization
        
        Args:
            dimensions: Matrix dimensions array
        
        Returns:
            Tuple of (min_cost, parenthesization_string)
        """
        n = len(dimensions) - 1
        if n <= 1:
            return 0, "M0" if n == 1 else ""
        
        dp = [[0] * n for _ in range(n)]
        split = [[0] * n for _ in range(n)]
        
        for l in range(2, n + 1):
            for i in range(n - l + 1):
                j = i + l - 1
                dp[i][j] = float('inf')
                
                for k in range(i, j):
                    cost = (dp[i][k] + dp[k + 1][j] + 
                           dimensions[i] * dimensions[k + 1] * dimensions[j + 1])
                    
                    if cost < dp[i][j]:
                        dp[i][j] = cost
                        split[i][j] = k
        
        def construct_parentheses(i: int, j: int) -> str:
            if i == j:
                return f"M{i}"
            
            k = split[i][j]
            left = construct_parentheses(i, k)
            right = construct_parentheses(k + 1, j)
            return f"({left} * {right})"
        
        return dp[0][n - 1], construct_parentheses(0, n - 1)
    
    def matrix_chain_memoization(self, dimensions: List[int]) -> int:
        """
        Top-down memoization approach
        
        Time Complexity: O(n³)
        Space Complexity: O(n²)
        """
        n = len(dimensions) - 1
        memo = {}
        
        def mcm(i: int, j: int) -> int:
            if i == j:
                return 0
            
            if (i, j) in memo:
                return memo[(i, j)]
            
            min_cost = float('inf')
            for k in range(i, j):
                cost = (mcm(i, k) + mcm(k + 1, j) + 
                       dimensions[i] * dimensions[k + 1] * dimensions[j + 1])
                min_cost = min(min_cost, cost)
            
            memo[(i, j)] = min_cost
            return min_cost
        
        return mcm(0, n - 1)
    
    def matrix_chain_with_cache_optimization(self, dimensions: List[int], 
                                           cache_size: int) -> int:
        """
        MCM with limited cache (additional constraint)
        
        This variant considers memory constraints in computation
        """
        n = len(dimensions) - 1
        
        # Basic MCM with penalty for cache misses
        dp = [[0] * n for _ in range(n)]
        
        for l in range(2, n + 1):
            for i in range(n - l + 1):
                j = i + l - 1
                dp[i][j] = float('inf')
                
                for k in range(i, j):
                    # Basic cost
                    basic_cost = (dp[i][k] + dp[k + 1][j] + 
                                dimensions[i] * dimensions[k + 1] * dimensions[j + 1])
                    
                    # Cache penalty (simplified model)
                    matrices_in_scope = j - i + 1
                    cache_penalty = max(0, matrices_in_scope - cache_size) * 1000
                    
                    total_cost = basic_cost + cache_penalty
                    dp[i][j] = min(dp[i][j], total_cost)
        
        return dp[0][n - 1]

# ==================== BOOLEAN PARENTHESIZATION ====================

class BooleanParenthesization:
    """
    Boolean Parenthesization Problems
    
    Count ways to parenthesize boolean expression to get desired result.
    """
    
    def count_ways_to_evaluate_true(self, symbols: str, operators: str) -> int:
        """
        Count ways to parenthesize to get True result
        
        Time Complexity: O(n³)
        Space Complexity: O(n²)
        
        Args:
            symbols: String of T/F symbols
            operators: String of &/|/^ operators
        
        Returns:
            Number of ways to get True result
        """
        n = len(symbols)
        if n == 0:
            return 0
        
        # dp_true[i][j] = ways to get True from symbols[i:j+1]
        # dp_false[i][j] = ways to get False from symbols[i:j+1]
        dp_true = [[0] * n for _ in range(n)]
        dp_false = [[0] * n for _ in range(n)]
        
        # Base case: single symbols
        for i in range(n):
            if symbols[i] == 'T':
                dp_true[i][i] = 1
                dp_false[i][i] = 0
            else:
                dp_true[i][i] = 0
                dp_false[i][i] = 1
        
        # Fill for subexpressions of length 2, 3, ...
        for length in range(3, n + 1, 2):  # Only odd lengths are valid
            for i in range(n - length + 1):
                j = i + length - 1
                
                # Try all possible split points (at operators)
                for k in range(i + 1, j, 2):
                    operator = operators[k // 2]
                    
                    left_true = dp_true[i][k - 1]
                    left_false = dp_false[i][k - 1]
                    right_true = dp_true[k + 1][j]
                    right_false = dp_false[k + 1][j]
                    
                    if operator == '&':
                        dp_true[i][j] += left_true * right_true
                        dp_false[i][j] += (left_true * right_false + 
                                         left_false * right_true + 
                                         left_false * right_false)
                    elif operator == '|':
                        dp_true[i][j] += (left_true * right_true + 
                                        left_true * right_false + 
                                        left_false * right_true)
                        dp_false[i][j] += left_false * right_false
                    elif operator == '^':
                        dp_true[i][j] += (left_true * right_false + 
                                        left_false * right_true)
                        dp_false[i][j] += (left_true * right_true + 
                                         left_false * right_false)
        
        return dp_true[0][n - 1]
    
    def count_ways_memoization(self, expression: str, target: bool) -> int:
        """
        Memoization approach for boolean parenthesization
        
        Args:
            expression: Full expression string like "T&F|T"
            target: Desired boolean result
        
        Returns:
            Number of ways to achieve target
        """
        memo = {}
        
        def solve(expr: str, is_true: bool) -> int:
            if len(expr) == 1:
                if is_true:
                    return 1 if expr == 'T' else 0
                else:
                    return 1 if expr == 'F' else 0
            
            if (expr, is_true) in memo:
                return memo[(expr, is_true)]
            
            result = 0
            
            # Try splitting at each operator
            for i in range(1, len(expr), 2):
                operator = expr[i]
                left_expr = expr[:i]
                right_expr = expr[i + 1:]
                
                if operator == '&':
                    if is_true:
                        result += (solve(left_expr, True) * solve(right_expr, True))
                    else:
                        result += (solve(left_expr, True) * solve(right_expr, False) +
                                 solve(left_expr, False) * solve(right_expr, True) +
                                 solve(left_expr, False) * solve(right_expr, False))
                
                elif operator == '|':
                    if is_true:
                        result += (solve(left_expr, True) * solve(right_expr, True) +
                                 solve(left_expr, True) * solve(right_expr, False) +
                                 solve(left_expr, False) * solve(right_expr, True))
                    else:
                        result += (solve(left_expr, False) * solve(right_expr, False))
                
                elif operator == '^':
                    if is_true:
                        result += (solve(left_expr, True) * solve(right_expr, False) +
                                 solve(left_expr, False) * solve(right_expr, True))
                    else:
                        result += (solve(left_expr, True) * solve(right_expr, True) +
                                 solve(left_expr, False) * solve(right_expr, False))
            
            memo[(expr, is_true)] = result
            return result
        
        return solve(expression, target)

# ==================== BURST BALLOONS ====================

class BurstBalloons:
    """
    Burst Balloons Problem
    
    LeetCode 312 - Burst Balloons
    Find maximum coins by bursting balloons optimally.
    """
    
    def max_coins(self, nums: List[int]) -> int:
        """
        Find maximum coins from bursting balloons
        
        Time Complexity: O(n³)
        Space Complexity: O(n²)
        
        Key insight: Think of which balloon to burst LAST in a range
        
        Args:
            nums: Array of balloon values
        
        Returns:
            Maximum coins obtainable
        """
        if not nums:
            return 0
        
        # Add dummy balloons with value 1 at both ends
        balloons = [1] + nums + [1]
        n = len(balloons)
        
        # dp[i][j] = max coins from bursting balloons between i and j (exclusive)
        dp = [[0] * n for _ in range(n)]
        
        # Length of gap between i and j
        for length in range(2, n):
            for i in range(n - length):
                j = i + length
                
                # Try bursting each balloon k last in range (i, j)
                for k in range(i + 1, j):
                    coins = (balloons[i] * balloons[k] * balloons[j] + 
                           dp[i][k] + dp[k][j])
                    dp[i][j] = max(dp[i][j], coins)
        
        return dp[0][n - 1]
    
    def max_coins_memoization(self, nums: List[int]) -> int:
        """
        Memoization approach for burst balloons
        """
        if not nums:
            return 0
        
        balloons = [1] + nums + [1]
        memo = {}
        
        def burst(left: int, right: int) -> int:
            if left + 1 == right:
                return 0
            
            if (left, right) in memo:
                return memo[(left, right)]
            
            max_coins = 0
            for k in range(left + 1, right):
                coins = (balloons[left] * balloons[k] * balloons[right] + 
                        burst(left, k) + burst(k, right))
                max_coins = max(max_coins, coins)
            
            memo[(left, right)] = max_coins
            return max_coins
        
        return burst(0, len(balloons) - 1)

# ==================== EVALUATE EXPRESSION TO TRUE ====================

class EvaluateExpression:
    """
    Advanced expression evaluation problems
    """
    
    def different_ways_add_parentheses(self, expression: str) -> List[int]:
        """
        Find all possible results from adding parentheses
        
        LeetCode 241 - Different Ways to Add Parentheses
        
        Time Complexity: O(4^n / √n) - Catalan number
        Space Complexity: O(4^n / √n)
        """
        memo = {}
        
        def compute(expr: str) -> List[int]:
            if expr in memo:
                return memo[expr]
            
            results = []
            
            # If expression is just a number
            if expr.isdigit():
                results.append(int(expr))
                memo[expr] = results
                return results
            
            # Try splitting at each operator
            for i, char in enumerate(expr):
                if char in '+-*':
                    left_results = compute(expr[:i])
                    right_results = compute(expr[i + 1:])
                    
                    for left in left_results:
                        for right in right_results:
                            if char == '+':
                                results.append(left + right)
                            elif char == '-':
                                results.append(left - right)
                            elif char == '*':
                                results.append(left * right)
            
            memo[expr] = results
            return results
        
        return compute(expression)
    
    def expression_add_operators(self, num: str, target: int) -> List[str]:
        """
        Add operators to make expression equal target
        
        LeetCode 282 - Expression Add Operators
        
        Args:
            num: String of digits
            target: Target value
        
        Returns:
            List of valid expressions
        """
        result = []
        
        def backtrack(index: int, current_expr: str, current_value: int, prev_value: int):
            if index == len(num):
                if current_value == target:
                    result.append(current_expr)
                return
            
            for i in range(index, len(num)):
                num_str = num[index:i + 1]
                
                # Skip numbers with leading zeros (except single zero)
                if len(num_str) > 1 and num_str[0] == '0':
                    break
                
                num_val = int(num_str)
                
                if index == 0:
                    # First number, no operator needed
                    backtrack(i + 1, num_str, num_val, num_val)
                else:
                    # Try addition
                    backtrack(i + 1, current_expr + '+' + num_str, 
                            current_value + num_val, num_val)
                    
                    # Try subtraction
                    backtrack(i + 1, current_expr + '-' + num_str, 
                            current_value - num_val, -num_val)
                    
                    # Try multiplication
                    backtrack(i + 1, current_expr + '*' + num_str, 
                            current_value - prev_value + prev_value * num_val, 
                            prev_value * num_val)
        
        backtrack(0, "", 0, 0)
        return result

# ==================== ADVANCED INTERVAL PARTITIONING ====================

class AdvancedIntervalPartitioning:
    """
    Advanced interval and partitioning problems
    """
    
    def minimum_cost_tree_from_leaf_values(self, arr: List[int]) -> int:
        """
        Minimum cost to build tree from leaf values
        
        LeetCode 1130 - Minimum Cost Tree From Leaf Values
        
        Time Complexity: O(n³)
        Space Complexity: O(n²)
        """
        n = len(arr)
        if n <= 1:
            return 0
        
        # dp[i][j] = min cost for subarray arr[i:j+1]
        dp = [[0] * n for _ in range(n)]
        # max_val[i][j] = max value in arr[i:j+1]
        max_val = [[0] * n for _ in range(n)]
        
        # Initialize max values
        for i in range(n):
            max_val[i][i] = arr[i]
            for j in range(i + 1, n):
                max_val[i][j] = max(max_val[i][j - 1], arr[j])
        
        # Fill DP table
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                dp[i][j] = float('inf')
                
                for k in range(i, j):
                    cost = (dp[i][k] + dp[k + 1][j] + 
                           max_val[i][k] * max_val[k + 1][j])
                    dp[i][j] = min(dp[i][j], cost)
        
        return dp[0][n - 1]
    
    def strange_printer(self, s: str) -> int:
        """
        Minimum turns to print string with strange printer
        
        LeetCode 664 - Strange Printer
        
        Time Complexity: O(n³)
        Space Complexity: O(n²)
        """
        if not s:
            return 0
        
        n = len(s)
        # dp[i][j] = min turns to print s[i:j+1]
        dp = [[0] * n for _ in range(n)]
        
        # Base case: single characters
        for i in range(n):
            dp[i][i] = 1
        
        # Fill for substrings of length 2, 3, ...
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                dp[i][j] = length  # Worst case: print each character separately
                
                # If first and last characters are same
                if s[i] == s[j]:
                    dp[i][j] = dp[i][j - 1]
                else:
                    # Try all possible split points
                    for k in range(i, j):
                        dp[i][j] = min(dp[i][j], dp[i][k] + dp[k + 1][j])
        
        return dp[0][n - 1]
    
    def optimal_binary_search_tree(self, keys: List[int], frequencies: List[int]) -> int:
        """
        Construct optimal binary search tree
        
        Time Complexity: O(n³)
        Space Complexity: O(n²)
        
        Args:
            keys: Sorted array of keys
            frequencies: Frequency of each key
        
        Returns:
            Minimum expected search cost
        """
        n = len(keys)
        if n != len(frequencies):
            raise ValueError("Keys and frequencies must have same length")
        
        # dp[i][j] = min cost for keys[i:j+1]
        dp = [[0] * n for _ in range(n)]
        # freq_sum[i][j] = sum of frequencies from i to j
        freq_sum = [[0] * n for _ in range(n)]
        
        # Initialize frequency sums
        for i in range(n):
            freq_sum[i][i] = frequencies[i]
            for j in range(i + 1, n):
                freq_sum[i][j] = freq_sum[i][j - 1] + frequencies[j]
        
        # Base case: single keys
        for i in range(n):
            dp[i][i] = frequencies[i]
        
        # Fill for subsets of length 2, 3, ...
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                dp[i][j] = float('inf')
                
                # Try each key as root
                for k in range(i, j + 1):
                    left_cost = dp[i][k - 1] if k > i else 0
                    right_cost = dp[k + 1][j] if k < j else 0
                    cost = left_cost + right_cost + freq_sum[i][j]
                    dp[i][j] = min(dp[i][j], cost)
        
        return dp[0][n - 1]

# ==================== PERFORMANCE ANALYSIS ====================

def performance_comparison():
    """Compare performance of different partitioning approaches"""
    print("=== Partitioning DP Performance Analysis ===\n")
    
    # Test Matrix Chain Multiplication
    mcm = MatrixChainMultiplication()
    
    test_cases = [
        [40, 20, 30, 10, 30],  # Small case
        [5, 4, 6, 2, 7, 3, 4, 5, 2, 6],  # Medium case
    ]
    
    for i, dimensions in enumerate(test_cases):
        print(f"MCM Test Case {i + 1}: {len(dimensions) - 1} matrices")
        
        # Tabulation
        start_time = time.time()
        tab_result = mcm.matrix_chain_order(dimensions)
        tab_time = time.time() - start_time
        
        # Memoization
        start_time = time.time()
        memo_result = mcm.matrix_chain_memoization(dimensions)
        memo_time = time.time() - start_time
        
        print(f"  Tabulation: {tab_result} ({tab_time:.6f}s)")
        print(f"  Memoization: {memo_result} ({memo_time:.6f}s)")
        print(f"  Results match: {tab_result == memo_result}")
        print()

# ==================== EXAMPLE USAGE AND TESTING ====================

if __name__ == "__main__":
    print("=== Partitioning DP Demo ===\n")
    
    # Palindrome Partitioning
    print("1. Palindrome Partitioning:")
    pal_partition = PalindromePartitioning()
    
    test_strings = ["aab", "raceacar", "abccba"]
    
    for s in test_strings:
        min_cuts = pal_partition.min_cuts_palindrome_partition(s)
        min_cuts_opt = pal_partition.min_cuts_optimized(s)
        
        print(f"  String: '{s}'")
        print(f"    Minimum cuts: {min_cuts}")
        print(f"    Optimized cuts: {min_cuts_opt}")
        
        if len(s) <= 10:  # Only show all partitions for short strings
            all_partitions = pal_partition.all_palindrome_partitions(s)
            print(f"    All partitions: {all_partitions}")
        
        min_insertions = pal_partition.min_insertions_palindrome(s)
        print(f"    Min insertions for full palindrome: {min_insertions}")
        print()
    
    # Matrix Chain Multiplication
    print("2. Matrix Chain Multiplication:")
    mcm = MatrixChainMultiplication()
    
    dimensions = [40, 20, 30, 10, 30]
    min_mults = mcm.matrix_chain_order(dimensions)
    min_mults_with_parens, parentheses = mcm.matrix_chain_order_with_parentheses(dimensions)
    
    print(f"  Matrix dimensions: {dimensions}")
    print(f"  Matrices: {len(dimensions) - 1}")
    print(f"  Minimum scalar multiplications: {min_mults}")
    print(f"  Optimal parenthesization: {parentheses}")
    
    # With cache optimization
    cache_optimized = mcm.matrix_chain_with_cache_optimization(dimensions, 2)
    print(f"  With cache constraint (size=2): {cache_optimized}")
    print()
    
    # Boolean Parenthesization
    print("3. Boolean Parenthesization:")
    bool_paren = BooleanParenthesization()
    
    symbols = "TTFT"
    operators = "&|^"
    
    ways_true = bool_paren.count_ways_to_evaluate_true(symbols, operators)
    expression = "T&F|T^F"
    ways_true_memo = bool_paren.count_ways_memoization(expression, True)
    ways_false_memo = bool_paren.count_ways_memoization(expression, False)
    
    print(f"  Symbols: {symbols}")
    print(f"  Operators: {operators}")
    print(f"  Ways to get True: {ways_true}")
    print(f"  Expression '{expression}':")
    print(f"    Ways to get True: {ways_true_memo}")
    print(f"    Ways to get False: {ways_false_memo}")
    print()
    
    # Burst Balloons
    print("4. Burst Balloons:")
    burst = BurstBalloons()
    
    balloons = [3, 1, 5, 8]
    max_coins_tab = burst.max_coins(balloons)
    max_coins_memo = burst.max_coins_memoization(balloons)
    
    print(f"  Balloons: {balloons}")
    print(f"  Maximum coins (tabulation): {max_coins_tab}")
    print(f"  Maximum coins (memoization): {max_coins_memo}")
    print()
    
    # Evaluate Expression
    print("5. Expression Evaluation:")
    eval_expr = EvaluateExpression()
    
    expression = "2*3-1"
    all_results = eval_expr.different_ways_add_parentheses(expression)
    print(f"  Expression: {expression}")
    print(f"  All possible results: {sorted(all_results)}")
    
    # Add operators
    num_str = "123"
    target = 6
    valid_expressions = eval_expr.expression_add_operators(num_str, target)
    print(f"  Add operators to '{num_str}' to get {target}: {valid_expressions[:5]}...")  # Show first 5
    print()
    
    # Advanced Interval Problems
    print("6. Advanced Interval Problems:")
    advanced = AdvancedIntervalPartitioning()
    
    # Minimum cost tree
    leaf_values = [6, 2, 4]
    min_cost_tree = advanced.minimum_cost_tree_from_leaf_values(leaf_values)
    print(f"  Min cost tree from leaves {leaf_values}: {min_cost_tree}")
    
    # Strange printer
    string_to_print = "aba"
    min_turns = advanced.strange_printer(string_to_print)
    print(f"  Min turns to print '{string_to_print}': {min_turns}")
    
    # Optimal BST
    keys = [10, 20, 30]
    frequencies = [34, 8, 50]
    optimal_cost = advanced.optimal_binary_search_tree(keys, frequencies)
    print(f"  Optimal BST cost for keys {keys} with freq {frequencies}: {optimal_cost}")
    print()
    
    # Performance comparison
    performance_comparison()
    
    # Pattern Recognition Guide
    print("=== Partitioning DP Pattern Recognition ===")
    print("Common Partitioning DP Patterns:")
    print("  1. Interval DP: dp[i][j] for range [i, j]")
    print("  2. Split Point: Try all k where i ≤ k < j")
    print("  3. Optimal Substructure: Combine optimal solutions of subproblems")
    print("  4. Cost Function: Usually additive with some combination rule")
    
    print("\nKey Problem Types:")
    print("  1. Matrix Chain: Minimize matrix multiplication cost")
    print("  2. Palindrome Partition: Minimize cuts or count ways")
    print("  3. Boolean Expression: Count evaluation ways")
    print("  4. Burst Balloons: Think of what to do LAST")
    print("  5. Expression Parsing: All possible interpretations")
    
    print("\nState Transition Pattern:")
    print("  for length in range(2, n+1):")
    print("    for i in range(n-length+1):")
    print("      j = i + length - 1")
    print("      for k in range(i, j):  # Split point")
    print("        dp[i][j] = optimize(dp[i][k], dp[k+1][j], cost(i,k,j))")
    
    print("\nOptimization Techniques:")
    print("  1. Memoization vs Tabulation trade-offs")
    print("  2. Knuth-Yao optimization for quadrangle inequality")
    print("  3. Divide and conquer optimization")
    print("  4. Monotonic queue/stack for special cases")
    
    print("\nReal-world Applications:")
    print("  1. Compiler optimization (expression parsing)")
    print("  2. Parallel computing (task scheduling)")
    print("  3. Database query optimization")
    print("  4. Network routing and load balancing")
    print("  5. Game theory and decision trees")
    print("  6. Bioinformatics (sequence alignment)")
    
    print("\n=== Demo Complete ===") 