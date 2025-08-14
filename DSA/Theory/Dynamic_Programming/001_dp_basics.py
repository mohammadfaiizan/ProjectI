"""
Dynamic Programming Basics - Fundamental Concepts and Techniques
This module covers the core concepts of Dynamic Programming including memoization, tabulation,
state definition, and transition patterns with detailed examples.
"""

from typing import List, Dict, Tuple, Optional
import time
import sys
from functools import lru_cache, wraps
from abc import ABC, abstractmethod

# ==================== DYNAMIC PROGRAMMING FUNDAMENTALS ====================

class DPBasics:
    """
    Core Dynamic Programming concepts and techniques
    
    Dynamic Programming is an algorithmic technique that solves complex problems
    by breaking them down into simpler subproblems and storing the solutions
    to avoid redundant calculations.
    
    Key Principles:
    1. Optimal Substructure: Problem can be broken into optimal subproblems
    2. Overlapping Subproblems: Same subproblems are solved multiple times
    3. Memoization: Store solutions to avoid recomputation
    4. Bottom-up Construction: Build solutions from base cases
    """
    
    def __init__(self):
        """Initialize DP basics with memoization support"""
        self.memo_cache = {}
        self.call_count = 0
        self.cache_hits = 0
    
    # ==================== WHAT IS DYNAMIC PROGRAMMING? ====================
    
    def fibonacci_naive(self, n: int) -> int:
        """
        Naive Fibonacci implementation without DP
        
        This demonstrates the problem DP solves - exponential time due to
        redundant calculations of the same subproblems.
        
        Time Complexity: O(2^n) - exponential
        Space Complexity: O(n) - recursion stack
        
        Args:
            n: The nth Fibonacci number to calculate
        
        Returns:
            The nth Fibonacci number
        """
        self.call_count += 1
        
        if n <= 1:
            return n
        
        return self.fibonacci_naive(n - 1) + self.fibonacci_naive(n - 2)
    
    def demonstrate_dp_need(self, n: int = 10) -> Dict[str, any]:
        """
        Demonstrate why Dynamic Programming is needed
        
        Shows the exponential growth of function calls in naive recursion
        vs optimized DP approaches.
        """
        print(f"=== Demonstrating DP Need with Fibonacci({n}) ===")
        
        # Reset counters
        self.call_count = 0
        
        start_time = time.time()
        naive_result = self.fibonacci_naive(n)
        naive_time = time.time() - start_time
        naive_calls = self.call_count
        
        print(f"Naive Approach:")
        print(f"  Result: {naive_result}")
        print(f"  Time: {naive_time:.6f} seconds")
        print(f"  Function calls: {naive_calls}")
        print(f"  Time Complexity: O(2^n)")
        
        # Reset for DP approach
        self.memo_cache.clear()
        self.call_count = 0
        self.cache_hits = 0
        
        start_time = time.time()
        dp_result = self.fibonacci_memoization(n)
        dp_time = time.time() - start_time
        dp_calls = self.call_count
        
        print(f"\nDP Memoization Approach:")
        print(f"  Result: {dp_result}")
        print(f"  Time: {dp_time:.6f} seconds")
        print(f"  Function calls: {dp_calls}")
        print(f"  Cache hits: {self.cache_hits}")
        print(f"  Time Complexity: O(n)")
        
        if naive_time > 0:
            speedup = naive_time / dp_time
            call_reduction = naive_calls / dp_calls if dp_calls > 0 else float('inf')
            print(f"\nImprovement:")
            print(f"  Speedup: {speedup:.2f}x")
            print(f"  Call reduction: {call_reduction:.2f}x")
        
        return {
            "naive_time": naive_time,
            "dp_time": dp_time,
            "naive_calls": naive_calls,
            "dp_calls": dp_calls,
            "speedup": naive_time / dp_time if dp_time > 0 else float('inf')
        }
    
    # ==================== MEMOIZATION (TOP-DOWN) ====================
    
    def fibonacci_memoization(self, n: int) -> int:
        """
        Fibonacci using Memoization (Top-Down DP)
        
        Memoization is a top-down approach where we solve the problem recursively
        but store the results of subproblems to avoid recomputation.
        
        Time Complexity: O(n)
        Space Complexity: O(n) - for memoization cache and recursion stack
        
        Args:
            n: The nth Fibonacci number to calculate
        
        Returns:
            The nth Fibonacci number
        """
        self.call_count += 1
        
        # Check if already computed
        if n in self.memo_cache:
            self.cache_hits += 1
            return self.memo_cache[n]
        
        # Base cases
        if n <= 1:
            result = n
        else:
            result = self.fibonacci_memoization(n - 1) + self.fibonacci_memoization(n - 2)
        
        # Store result for future use
        self.memo_cache[n] = result
        return result
    
    @lru_cache(maxsize=None)
    def fibonacci_lru_cache(self, n: int) -> int:
        """
        Fibonacci using Python's built-in LRU cache decorator
        
        This is Python's optimized way to implement memoization.
        The @lru_cache decorator automatically handles caching.
        
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if n <= 1:
            return n
        
        return self.fibonacci_lru_cache(n - 1) + self.fibonacci_lru_cache(n - 2)
    
    def memoization_decorator(self, func):
        """
        Custom memoization decorator
        
        This demonstrates how to create a reusable memoization decorator
        that can be applied to any function.
        """
        cache = {}
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create a key from arguments
            key = str(args) + str(sorted(kwargs.items()))
            
            if key in cache:
                return cache[key]
            
            result = func(*args, **kwargs)
            cache[key] = result
            return result
        
        wrapper.cache = cache
        wrapper.cache_clear = lambda: cache.clear()
        return wrapper
    
    # ==================== TABULATION (BOTTOM-UP) ====================
    
    def fibonacci_tabulation(self, n: int) -> int:
        """
        Fibonacci using Tabulation (Bottom-Up DP)
        
        Tabulation is a bottom-up approach where we solve smaller subproblems
        first and build up to the solution iteratively.
        
        Time Complexity: O(n)
        Space Complexity: O(n) - for the DP table
        
        Args:
            n: The nth Fibonacci number to calculate
        
        Returns:
            The nth Fibonacci number
        """
        if n <= 1:
            return n
        
        # Create DP table
        dp = [0] * (n + 1)
        
        # Base cases
        dp[0] = 0
        dp[1] = 1
        
        # Fill table bottom-up
        for i in range(2, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]
        
        return dp[n]
    
    def fibonacci_space_optimized(self, n: int) -> int:
        """
        Space-optimized Fibonacci using only two variables
        
        Since Fibonacci only depends on the previous two values,
        we don't need to store the entire DP table.
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Args:
            n: The nth Fibonacci number to calculate
        
        Returns:
            The nth Fibonacci number
        """
        if n <= 1:
            return n
        
        prev2, prev1 = 0, 1
        
        for i in range(2, n + 1):
            current = prev1 + prev2
            prev2, prev1 = prev1, current
        
        return prev1
    
    # ==================== STATE DEFINITION & TRANSITION ====================
    
    def demonstrate_state_definition(self) -> None:
        """
        Demonstrate how to define states and transitions in DP
        
        State Definition:
        - What does dp[i] represent?
        - What are the dimensions needed?
        - What parameters change in subproblems?
        
        State Transition:
        - How to move from one state to another?
        - What decisions lead to subproblems?
        - How to combine subproblem solutions?
        """
        print("=== State Definition & Transition ===")
        
        print("\n1. Fibonacci State Definition:")
        print("   State: dp[i] = ith Fibonacci number")
        print("   Transition: dp[i] = dp[i-1] + dp[i-2]")
        print("   Base: dp[0] = 0, dp[1] = 1")
        
        print("\n2. Min Cost Climbing Stairs:")
        print("   State: dp[i] = minimum cost to reach step i")
        print("   Transition: dp[i] = cost[i] + min(dp[i-1], dp[i-2])")
        print("   Base: dp[0] = cost[0], dp[1] = cost[1]")
        
        print("\n3. Longest Increasing Subsequence:")
        print("   State: dp[i] = length of LIS ending at index i")
        print("   Transition: dp[i] = max(dp[j] + 1) for all j < i where arr[j] < arr[i]")
        print("   Base: dp[i] = 1 for all i")
    
    def min_cost_climbing_stairs(self, cost: List[int]) -> int:
        """
        Example: Minimum Cost Climbing Stairs
        
        Demonstrates state definition and transition with a practical example.
        You can start from step 0 or 1, and from each step you can climb 1 or 2 steps.
        
        State: dp[i] = minimum cost to reach step i
        Transition: dp[i] = cost[i] + min(dp[i-1], dp[i-2])
        
        Time Complexity: O(n)
        Space Complexity: O(n)
        
        Args:
            cost: Cost array where cost[i] is cost of stepping on ith step
        
        Returns:
            Minimum cost to reach the top (beyond last step)
        """
        n = len(cost)
        if n <= 2:
            return min(cost)
        
        # dp[i] represents minimum cost to reach step i
        dp = [0] * n
        
        # Base cases - can start from step 0 or 1
        dp[0] = cost[0]
        dp[1] = cost[1]
        
        # Fill DP table
        for i in range(2, n):
            dp[i] = cost[i] + min(dp[i - 1], dp[i - 2])
        
        # Can reach top from either last or second last step
        return min(dp[n - 1], dp[n - 2])
    
    def min_cost_climbing_stairs_optimized(self, cost: List[int]) -> int:
        """
        Space-optimized version of min cost climbing stairs
        
        Since we only need previous two values, optimize space usage.
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        n = len(cost)
        if n <= 2:
            return min(cost)
        
        prev2 = cost[0]
        prev1 = cost[1]
        
        for i in range(2, n):
            current = cost[i] + min(prev1, prev2)
            prev2, prev1 = prev1, current
        
        return min(prev1, prev2)
    
    # ==================== BASE CASE & RECURRENCE RELATION ====================
    
    def demonstrate_base_cases(self) -> None:
        """
        Demonstrate importance of base cases and recurrence relations
        
        Base Cases:
        - Smallest subproblems that can be solved directly
        - Usually when problem size is 0, 1, or some small value
        - Critical for correctness of the solution
        
        Recurrence Relation:
        - Mathematical expression relating problem to subproblems
        - Defines how to combine subproblem solutions
        - Must lead to base cases
        """
        print("=== Base Cases & Recurrence Relations ===")
        
        print("\n1. Factorial:")
        print("   Base: fact(0) = 1, fact(1) = 1")
        print("   Recurrence: fact(n) = n * fact(n-1)")
        
        print("\n2. Fibonacci:")
        print("   Base: fib(0) = 0, fib(1) = 1")
        print("   Recurrence: fib(n) = fib(n-1) + fib(n-2)")
        
        print("\n3. Coin Change (min coins):")
        print("   Base: dp[0] = 0 (0 coins needed for amount 0)")
        print("   Recurrence: dp[amount] = min(dp[amount - coin] + 1) for all coins")
        
        print("\n4. Longest Common Subsequence:")
        print("   Base: dp[i][0] = 0, dp[0][j] = 0")
        print("   Recurrence: dp[i][j] = dp[i-1][j-1] + 1 if s1[i-1] == s2[j-1]")
        print("              else max(dp[i-1][j], dp[i][j-1])")
    
    def coin_change_min_coins(self, coins: List[int], amount: int) -> int:
        """
        Example: Coin Change - Minimum Coins
        
        Demonstrates base cases and recurrence relation with a classic DP problem.
        
        Base Case: dp[0] = 0 (need 0 coins for amount 0)
        Recurrence: dp[i] = min(dp[i - coin] + 1) for all valid coins
        
        Time Complexity: O(amount * len(coins))
        Space Complexity: O(amount)
        
        Args:
            coins: Available coin denominations
            amount: Target amount to make
        
        Returns:
            Minimum number of coins needed, -1 if impossible
        """
        if amount == 0:
            return 0
        
        # dp[i] represents minimum coins needed for amount i
        dp = [float('inf')] * (amount + 1)
        
        # Base case
        dp[0] = 0
        
        # Fill DP table
        for i in range(1, amount + 1):
            for coin in coins:
                if coin <= i:
                    dp[i] = min(dp[i], dp[i - coin] + 1)
        
        return dp[amount] if dp[amount] != float('inf') else -1
    
    def longest_common_subsequence(self, text1: str, text2: str) -> int:
        """
        Example: Longest Common Subsequence
        
        Demonstrates 2D DP with proper base cases and recurrence.
        
        Base Cases: dp[i][0] = 0, dp[0][j] = 0
        Recurrence: 
        - If text1[i-1] == text2[j-1]: dp[i][j] = dp[i-1][j-1] + 1
        - Else: dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        Time Complexity: O(m * n)
        Space Complexity: O(m * n)
        """
        m, n = len(text1), len(text2)
        
        # Create DP table with base cases (0s)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        return dp[m][n]

# ==================== DP PROBLEM PATTERNS ====================

class DPPatterns:
    """
    Common Dynamic Programming patterns and their identification
    
    Helps recognize when and how to apply DP to different problem types.
    """
    
    @staticmethod
    def identify_dp_problem(problem_description: str) -> Dict[str, any]:
        """
        Guide to identify if a problem can be solved using DP
        
        Returns analysis of whether DP is applicable and which pattern to use.
        """
        indicators = {
            "optimal_substructure": [
                "minimum", "maximum", "optimal", "best", "shortest", "longest"
            ],
            "overlapping_subproblems": [
                "recursive", "subproblems", "repetitive", "recompute"
            ],
            "counting": [
                "number of ways", "count", "how many", "combinations"
            ],
            "decision": [
                "choose", "select", "pick", "include", "exclude"
            ]
        }
        
        problem_lower = problem_description.lower()
        
        pattern_scores = {}
        for pattern, keywords in indicators.items():
            score = sum(1 for keyword in keywords if keyword in problem_lower)
            pattern_scores[pattern] = score
        
        # Determine likely DP applicability
        total_score = sum(pattern_scores.values())
        is_dp_suitable = total_score >= 2
        
        # Suggest pattern
        suggested_patterns = []
        if any(word in problem_lower for word in ["knapsack", "subset", "partition"]):
            suggested_patterns.append("Knapsack Pattern")
        if any(word in problem_lower for word in ["sequence", "subsequence", "string"]):
            suggested_patterns.append("Sequence DP")
        if any(word in problem_lower for word in ["grid", "matrix", "path"]):
            suggested_patterns.append("Grid DP")
        if any(word in problem_lower for word in ["tree", "binary"]):
            suggested_patterns.append("Tree DP")
        
        return {
            "is_dp_suitable": is_dp_suitable,
            "confidence": min(total_score / 4.0, 1.0),
            "pattern_scores": pattern_scores,
            "suggested_patterns": suggested_patterns,
            "recommendations": _get_dp_recommendations(pattern_scores, suggested_patterns)
        }

def _get_dp_recommendations(scores: Dict[str, int], patterns: List[str]) -> List[str]:
    """Generate recommendations based on pattern analysis"""
    recommendations = []
    
    if scores["optimal_substructure"] >= 2:
        recommendations.append("Strong indication of optimal substructure - consider DP")
    
    if scores["counting"] >= 1:
        recommendations.append("Counting problem - likely uses bottom-up DP")
    
    if scores["decision"] >= 1:
        recommendations.append("Decision problem - consider memoization with recursion")
    
    if patterns:
        recommendations.append(f"Consider these patterns: {', '.join(patterns)}")
    
    if not recommendations:
        recommendations.append("DP may not be the best approach for this problem")
    
    return recommendations

# ==================== EXAMPLE USAGE AND TESTING ====================

def demonstrate_dp_comparison():
    """Demonstrate different DP approaches with performance comparison"""
    print("=== Dynamic Programming Approaches Comparison ===\n")
    
    dp_basics = DPBasics()
    
    # Test different approaches
    test_values = [10, 20, 25]
    
    for n in test_values:
        print(f"Computing Fibonacci({n}):")
        
        # Memoization approach
        dp_basics.memo_cache.clear()
        start_time = time.time()
        memo_result = dp_basics.fibonacci_memoization(n)
        memo_time = time.time() - start_time
        
        # Tabulation approach
        start_time = time.time()
        tab_result = dp_basics.fibonacci_tabulation(n)
        tab_time = time.time() - start_time
        
        # Space-optimized approach
        start_time = time.time()
        opt_result = dp_basics.fibonacci_space_optimized(n)
        opt_time = time.time() - start_time
        
        print(f"  Memoization: {memo_result} ({memo_time:.6f}s)")
        print(f"  Tabulation: {tab_result} ({tab_time:.6f}s)")
        print(f"  Optimized: {opt_result} ({opt_time:.6f}s)")
        print(f"  All results match: {memo_result == tab_result == opt_result}")
        print()

if __name__ == "__main__":
    print("=== Dynamic Programming Basics Demo ===\n")
    
    dp_basics = DPBasics()
    
    # Demonstrate why DP is needed
    dp_basics.demonstrate_dp_need(15)
    print()
    
    # Demonstrate state definition
    dp_basics.demonstrate_state_definition()
    print()
    
    # Demonstrate base cases
    dp_basics.demonstrate_base_cases()
    print()
    
    # Test practical examples
    print("=== Practical DP Examples ===")
    
    # Min Cost Climbing Stairs
    cost = [10, 15, 20]
    min_cost = dp_basics.min_cost_climbing_stairs(cost)
    min_cost_opt = dp_basics.min_cost_climbing_stairs_optimized(cost)
    print(f"Min Cost Climbing Stairs {cost}: {min_cost} (optimized: {min_cost_opt})")
    
    # Coin Change
    coins = [1, 3, 4]
    amount = 6
    min_coins = dp_basics.coin_change_min_coins(coins, amount)
    print(f"Coin Change - coins {coins}, amount {amount}: {min_coins} coins")
    
    # Longest Common Subsequence
    text1, text2 = "abcde", "ace"
    lcs_length = dp_basics.longest_common_subsequence(text1, text2)
    print(f"LCS of '{text1}' and '{text2}': {lcs_length}")
    print()
    
    # DP Pattern Recognition
    print("=== DP Pattern Recognition ===")
    
    sample_problems = [
        "Find the minimum number of coins to make a given amount",
        "Count the number of ways to climb stairs",
        "Find the longest increasing subsequence in an array",
        "Sort an array using quicksort"
    ]
    
    patterns = DPPatterns()
    
    for problem in sample_problems:
        analysis = patterns.identify_dp_problem(problem)
        print(f"Problem: {problem}")
        print(f"  DP Suitable: {analysis['is_dp_suitable']}")
        print(f"  Confidence: {analysis['confidence']:.2f}")
        print(f"  Suggested Patterns: {analysis['suggested_patterns']}")
        print(f"  Recommendations: {analysis['recommendations'][0] if analysis['recommendations'] else 'None'}")
        print()
    
    # Performance comparison
    demonstrate_dp_comparison()
    
    # Memory usage analysis
    print("=== Memory Usage Analysis ===")
    print("Memoization (Top-Down):")
    print("  - Pros: Natural recursive thinking, only computes needed subproblems")
    print("  - Cons: Function call overhead, stack space usage")
    print("  - Use when: Problem naturally recursive, not all subproblems needed")
    
    print("\nTabulation (Bottom-Up):")
    print("  - Pros: No recursion overhead, guaranteed order, better cache locality")
    print("  - Cons: May compute unnecessary subproblems, harder to implement")
    print("  - Use when: All subproblems needed, iterative solution clear")
    
    print("\nSpace Optimization:")
    print("  - Reduce space from O(n) to O(1) when possible")
    print("  - Identify which previous states are actually needed")
    print("  - Use rolling arrays or variables for better memory efficiency")
    
    print("\n=== Demo Complete ===") 