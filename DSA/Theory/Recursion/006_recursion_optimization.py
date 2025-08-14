"""
Recursion Optimization Techniques
=================================

Topics: Memoization, tail recursion, iterative conversion, space optimization
Companies: All companies test optimization understanding
Difficulty: Medium to Hard
Time Complexity: Optimization reduces from exponential to polynomial
Space Complexity: Can reduce from O(n) to O(1) with proper techniques
"""

from typing import List, Dict, Optional, Any, Callable
from functools import lru_cache, wraps
import time
import sys

class RecursionOptimization:
    
    def __init__(self):
        """Initialize with performance tracking"""
        self.call_count = 0
        self.cache_hits = 0
        self.memo_cache = {}
    
    # ==========================================
    # 1. MEMOIZATION TECHNIQUES
    # ==========================================
    
    def fibonacci_naive(self, n: int) -> int:
        """
        Naive Fibonacci - demonstrates the problem
        
        Time: O(2^n), Space: O(n)
        """
        self.call_count += 1
        
        if n <= 1:
            return n
        
        return self.fibonacci_naive(n - 1) + self.fibonacci_naive(n - 2)
    
    def fibonacci_memoized_manual(self, n: int, memo: Dict[int, int] = None) -> int:
        """
        Fibonacci with manual memoization
        
        Time: O(n), Space: O(n)
        """
        if memo is None:
            memo = {}
        
        self.call_count += 1
        
        # Check if result is cached
        if n in memo:
            self.cache_hits += 1
            return memo[n]
        
        # Base cases
        if n <= 1:
            result = n
        else:
            result = (self.fibonacci_memoized_manual(n - 1, memo) + 
                     self.fibonacci_memoized_manual(n - 2, memo))
        
        # Cache the result
        memo[n] = result
        return result
    
    @lru_cache(maxsize=None)
    def fibonacci_lru_cache(self, n: int) -> int:
        """
        Fibonacci with @lru_cache decorator
        
        Time: O(n), Space: O(n)
        """
        self.call_count += 1
        
        if n <= 1:
            return n
        
        return self.fibonacci_lru_cache(n - 1) + self.fibonacci_lru_cache(n - 2)
    
    def create_memoization_decorator(self) -> Callable:
        """
        Create a custom memoization decorator
        """
        def memoize(func):
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
            
            # Add cache inspection methods
            wrapper.cache = cache
            wrapper.cache_clear = lambda: cache.clear()
            wrapper.cache_info = lambda: f"Cache size: {len(cache)}"
            
            return wrapper
        return memoize
    
    def demonstrate_memoization_impact(self, n: int = 35) -> None:
        """
        Demonstrate the impact of memoization on performance
        """
        print(f"=== MEMOIZATION IMPACT ANALYSIS (n={n}) ===")
        
        # Test naive approach (use smaller n to avoid timeout)
        test_n = min(n, 35)
        print(f"Testing with n={test_n}")
        
        # Naive Fibonacci
        self.call_count = 0
        start_time = time.time()
        naive_result = self.fibonacci_naive(test_n)
        naive_time = time.time() - start_time
        naive_calls = self.call_count
        
        # Memoized Fibonacci
        self.call_count = 0
        self.cache_hits = 0
        start_time = time.time()
        memo_result = self.fibonacci_memoized_manual(test_n)
        memo_time = time.time() - start_time
        memo_calls = self.call_count
        memo_hits = self.cache_hits
        
        # LRU Cache Fibonacci
        self.fibonacci_lru_cache.cache_clear()
        self.call_count = 0
        start_time = time.time()
        lru_result = self.fibonacci_lru_cache(test_n)
        lru_time = time.time() - start_time
        lru_calls = self.call_count
        
        print(f"\nResults for Fibonacci({test_n}):")
        print(f"Naive approach:")
        print(f"  Result: {naive_result}")
        print(f"  Time: {naive_time:.6f} seconds")
        print(f"  Function calls: {naive_calls}")
        
        print(f"\nMemoized approach:")
        print(f"  Result: {memo_result}")
        print(f"  Time: {memo_time:.6f} seconds")
        print(f"  Function calls: {memo_calls}")
        print(f"  Cache hits: {memo_hits}")
        
        print(f"\nLRU Cache approach:")
        print(f"  Result: {lru_result}")
        print(f"  Time: {lru_time:.6f} seconds")
        print(f"  Function calls: {lru_calls}")
        print(f"  Cache info: {self.fibonacci_lru_cache.cache_info()}")
        
        if naive_time > 0:
            speedup_memo = naive_time / memo_time if memo_time > 0 else float('inf')
            speedup_lru = naive_time / lru_time if lru_time > 0 else float('inf')
            
            print(f"\nPerformance improvement:")
            print(f"  Memoization: {speedup_memo:.1f}x faster")
            print(f"  LRU Cache: {speedup_lru:.1f}x faster")
            print(f"  Call reduction: {naive_calls / memo_calls:.1f}x fewer calls")
    
    # ==========================================
    # 2. TAIL RECURSION OPTIMIZATION
    # ==========================================
    
    def factorial_regular(self, n: int) -> int:
        """
        Regular factorial recursion
        
        Time: O(n), Space: O(n)
        """
        if n <= 1:
            return 1
        
        return n * self.factorial_regular(n - 1)
    
    def factorial_tail_recursive(self, n: int, accumulator: int = 1) -> int:
        """
        Tail recursive factorial
        
        Time: O(n), Space: O(n) in Python (not optimized)
        But can be optimized to O(1) space in other languages
        """
        print(f"factorial_tail({n}, {accumulator})")
        
        if n <= 1:
            return accumulator
        
        # Tail recursive call - last operation
        return self.factorial_tail_recursive(n - 1, n * accumulator)
    
    def factorial_iterative(self, n: int) -> int:
        """
        Iterative equivalent of tail recursive factorial
        
        Time: O(n), Space: O(1)
        """
        accumulator = 1
        while n > 1:
            accumulator = n * accumulator
            n = n - 1
        return accumulator
    
    def fibonacci_tail_recursive(self, n: int, a: int = 0, b: int = 1) -> int:
        """
        Tail recursive Fibonacci
        
        Time: O(n), Space: O(n) in Python
        """
        print(f"fib_tail({n}, {a}, {b})")
        
        if n == 0:
            return a
        
        # Tail recursive call
        return self.fibonacci_tail_recursive(n - 1, b, a + b)
    
    def fibonacci_iterative(self, n: int) -> int:
        """
        Iterative Fibonacci (converted from tail recursion)
        
        Time: O(n), Space: O(1)
        """
        a, b = 0, 1
        
        for _ in range(n):
            a, b = b, a + b
        
        return a
    
    def demonstrate_tail_recursion(self, n: int = 10) -> None:
        """
        Demonstrate tail recursion and its conversion to iteration
        """
        print(f"=== TAIL RECURSION DEMONSTRATION (n={n}) ===")
        
        print("Tail recursive factorial:")
        tail_result = self.factorial_tail_recursive(n)
        print(f"Result: {tail_result}")
        
        print(f"\nIterative equivalent:")
        iter_result = self.factorial_iterative(n)
        print(f"Result: {iter_result}")
        
        print(f"Results match: {tail_result == iter_result}")
        
        print(f"\nTail recursive Fibonacci:")
        fib_tail_result = self.fibonacci_tail_recursive(n)
        print(f"Result: {fib_tail_result}")
        
        print(f"Iterative Fibonacci:")
        fib_iter_result = self.fibonacci_iterative(n)
        print(f"Result: {fib_iter_result}")
        
        print(f"Results match: {fib_tail_result == fib_iter_result}")
    
    # ==========================================
    # 3. RECURSION TO ITERATION CONVERSION
    # ==========================================
    
    def tree_traversal_recursive(self, root, result=None):
        """
        Recursive tree traversal (inorder)
        """
        if result is None:
            result = []
        
        if root:
            self.tree_traversal_recursive(root.left, result)
            result.append(root.val)
            self.tree_traversal_recursive(root.right, result)
        
        return result
    
    def tree_traversal_iterative(self, root):
        """
        Iterative tree traversal using explicit stack
        
        Converts recursion to iteration by simulating call stack
        """
        if not root:
            return []
        
        result = []
        stack = []
        current = root
        
        while stack or current:
            # Go to leftmost node
            while current:
                stack.append(current)
                current = current.left
            
            # Process current node
            current = stack.pop()
            result.append(current.val)
            
            # Move to right subtree
            current = current.right
        
        return result
    
    def binary_search_recursive(self, arr: List[int], target: int, left: int = 0, right: int = None) -> int:
        """
        Recursive binary search
        
        Time: O(log n), Space: O(log n)
        """
        if right is None:
            right = len(arr) - 1
        
        if left > right:
            return -1
        
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] > target:
            return self.binary_search_recursive(arr, target, left, mid - 1)
        else:
            return self.binary_search_recursive(arr, target, mid + 1, right)
    
    def binary_search_iterative(self, arr: List[int], target: int) -> int:
        """
        Iterative binary search
        
        Time: O(log n), Space: O(1)
        """
        left, right = 0, len(arr) - 1
        
        while left <= right:
            mid = (left + right) // 2
            
            if arr[mid] == target:
                return mid
            elif arr[mid] > target:
                right = mid - 1
            else:
                left = mid + 1
        
        return -1
    
    def demonstrate_recursion_to_iteration(self) -> None:
        """
        Demonstrate conversion patterns from recursion to iteration
        """
        print("=== RECURSION TO ITERATION CONVERSION ===")
        
        # Binary search comparison
        arr = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
        target = 7
        
        recursive_result = self.binary_search_recursive(arr, target)
        iterative_result = self.binary_search_iterative(arr, target)
        
        print(f"Binary search for {target} in {arr}:")
        print(f"  Recursive result: {recursive_result}")
        print(f"  Iterative result: {iterative_result}")
        print(f"  Results match: {recursive_result == iterative_result}")
        
        print(f"\nConversion techniques:")
        print("1. Use explicit stack to simulate call stack")
        print("2. Convert tail recursion to loops")
        print("3. Use state variables instead of parameters")
        print("4. Identify iteration patterns in the recursion")
    
    # ==========================================
    # 4. SPACE OPTIMIZATION TECHNIQUES
    # ==========================================
    
    def longest_common_subsequence_naive(self, text1: str, text2: str, i: int = 0, j: int = 0) -> int:
        """
        Naive LCS recursion
        
        Time: O(2^(m+n)), Space: O(m+n)
        """
        if i >= len(text1) or j >= len(text2):
            return 0
        
        if text1[i] == text2[j]:
            return 1 + self.longest_common_subsequence_naive(text1, text2, i + 1, j + 1)
        else:
            return max(
                self.longest_common_subsequence_naive(text1, text2, i + 1, j),
                self.longest_common_subsequence_naive(text1, text2, i, j + 1)
            )
    
    def longest_common_subsequence_memo(self, text1: str, text2: str) -> int:
        """
        LCS with memoization
        
        Time: O(m*n), Space: O(m*n)
        """
        memo = {}
        
        def lcs_helper(i: int, j: int) -> int:
            if i >= len(text1) or j >= len(text2):
                return 0
            
            if (i, j) in memo:
                return memo[(i, j)]
            
            if text1[i] == text2[j]:
                result = 1 + lcs_helper(i + 1, j + 1)
            else:
                result = max(lcs_helper(i + 1, j), lcs_helper(i, j + 1))
            
            memo[(i, j)] = result
            return result
        
        return lcs_helper(0, 0)
    
    def longest_common_subsequence_dp(self, text1: str, text2: str) -> int:
        """
        LCS with bottom-up DP (converted from recursion)
        
        Time: O(m*n), Space: O(m*n)
        """
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = 1 + dp[i - 1][j - 1]
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        return dp[m][n]
    
    def longest_common_subsequence_optimized(self, text1: str, text2: str) -> int:
        """
        Space-optimized LCS using only two rows
        
        Time: O(m*n), Space: O(min(m,n))
        """
        # Make text1 the shorter string for space optimization
        if len(text1) > len(text2):
            text1, text2 = text2, text1
        
        m, n = len(text1), len(text2)
        prev = [0] * (m + 1)
        curr = [0] * (m + 1)
        
        for j in range(1, n + 1):
            for i in range(1, m + 1):
                if text1[i - 1] == text2[j - 1]:
                    curr[i] = 1 + prev[i - 1]
                else:
                    curr[i] = max(prev[i], curr[i - 1])
            prev, curr = curr, prev
        
        return prev[m]
    
    def demonstrate_space_optimization(self) -> None:
        """
        Demonstrate space optimization techniques
        """
        print("=== SPACE OPTIMIZATION DEMONSTRATION ===")
        
        text1, text2 = "abcde", "ace"
        
        print(f"Finding LCS of '{text1}' and '{text2}':")
        
        # Memoized version
        memo_result = self.longest_common_subsequence_memo(text1, text2)
        print(f"  Memoized result: {memo_result}")
        
        # DP version
        dp_result = self.longest_common_subsequence_dp(text1, text2)
        print(f"  DP result: {dp_result}")
        
        # Space-optimized version
        optimized_result = self.longest_common_subsequence_optimized(text1, text2)
        print(f"  Space-optimized result: {optimized_result}")
        
        print(f"  All results match: {memo_result == dp_result == optimized_result}")
        
        print(f"\nSpace complexity comparison:")
        print(f"  Naive recursion: O(m+n) call stack")
        print(f"  Memoization: O(m*n) cache + O(m+n) call stack")
        print(f"  DP table: O(m*n) table")
        print(f"  Optimized: O(min(m,n)) using rolling array")
    
    # ==========================================
    # 5. ADVANCED OPTIMIZATION TECHNIQUES
    # ==========================================
    
    def demonstrate_advanced_optimizations(self) -> None:
        """
        Demonstrate advanced optimization techniques
        """
        print("=== ADVANCED OPTIMIZATION TECHNIQUES ===")
        
        print("1. Memoization patterns:")
        print("   - Top-down: Store results as you compute them")
        print("   - Use hash maps for complex state spaces")
        print("   - Consider @lru_cache for simple cases")
        
        print("\n2. Tail recursion conversion:")
        print("   - Add accumulator parameters")
        print("   - Make recursive call the last operation")
        print("   - Convert to while loop for O(1) space")
        
        print("\n3. Space optimization:")
        print("   - Rolling arrays for DP problems")
        print("   - In-place algorithms when possible")
        print("   - Reuse data structures")
        
        print("\n4. Stack simulation:")
        print("   - Use explicit stack for deep recursion")
        print("   - Avoid stack overflow issues")
        print("   - Iterative deepening for large search spaces")
        
        print("\n5. Early termination:")
        print("   - Pruning impossible branches")
        print("   - Short-circuit evaluation")
        print("   - Boundary condition checking")

# ==========================================
# DEMONSTRATION AND TESTING
# ==========================================

def demonstrate_recursion_optimization():
    """Demonstrate all recursion optimization techniques"""
    print("=== RECURSION OPTIMIZATION DEMONSTRATION ===\n")
    
    optimizer = RecursionOptimization()
    
    # 1. Memoization impact
    optimizer.demonstrate_memoization_impact(30)
    print("\n" + "="*60 + "\n")
    
    # 2. Tail recursion
    optimizer.demonstrate_tail_recursion(5)
    print("\n" + "="*60 + "\n")
    
    # 3. Recursion to iteration
    optimizer.demonstrate_recursion_to_iteration()
    print("\n" + "="*60 + "\n")
    
    # 4. Space optimization
    optimizer.demonstrate_space_optimization()
    print("\n" + "="*60 + "\n")
    
    # 5. Advanced techniques
    optimizer.demonstrate_advanced_optimizations()
    
    # 6. Custom memoization decorator example
    print("\n=== CUSTOM MEMOIZATION DECORATOR ===")
    memoize = optimizer.create_memoization_decorator()
    
    @memoize
    def expensive_function(n):
        print(f"Computing expensive_function({n})")
        if n <= 1:
            return n
        return expensive_function(n-1) + expensive_function(n-2)
    
    print("First call:")
    result1 = expensive_function(10)
    print(f"Result: {result1}")
    
    print(f"\nCache info: {expensive_function.cache_info()}")
    
    print("\nSecond call (should use cache):")
    result2 = expensive_function(10)
    print(f"Result: {result2}")

if __name__ == "__main__":
    # Increase recursion limit for demonstration
    sys.setrecursionlimit(10000)
    
    demonstrate_recursion_optimization()
    
    print("\n=== OPTIMIZATION STRATEGY GUIDE ===")
    print("🚀 When to use each technique:")
    
    print("\n1. Memoization:")
    print("   ✅ Overlapping subproblems")
    print("   ✅ Pure functions (no side effects)")
    print("   ✅ Expensive computations")
    print("   ❌ Problems with large state spaces")
    
    print("\n2. Tail Recursion:")
    print("   ✅ Linear recursion patterns")
    print("   ✅ Accumulator-friendly problems")
    print("   ✅ When stack overflow is a concern")
    print("   ❌ Tree/graph traversals")
    
    print("\n3. Iteration Conversion:")
    print("   ✅ Deep recursion causing stack overflow")
    print("   ✅ Performance-critical code")
    print("   ✅ Simple recursion patterns")
    print("   ❌ Complex tree/backtracking algorithms")
    
    print("\n4. Space Optimization:")
    print("   ✅ DP problems with large tables")
    print("   ✅ When memory is constrained")
    print("   ✅ Rolling array patterns possible")
    print("   ❌ When you need to reconstruct solutions")
    
    print("\n🎯 General optimization principles:")
    print("   • Profile first - identify actual bottlenecks")
    print("   • Start with correct solution, then optimize")
    print("   • Consider trade-offs between time and space")
    print("   • Test optimized versions thoroughly")
    print("   • Document optimization reasoning")
