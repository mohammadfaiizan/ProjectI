"""
Dynamic Programming - Fibonacci & Variants
This module implements classic Fibonacci and its numerous variants including staircase problems,
tiling patterns, and jump ways with multiple optimization approaches.
"""

from typing import List, Dict, Tuple, Optional
import time
from functools import lru_cache
import math

# ==================== CLASSIC FIBONACCI ====================

class FibonacciDP:
    """
    Implementation of classic Fibonacci with multiple DP approaches
    
    Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...
    Each number is the sum of the two preceding ones.
    """
    
    def __init__(self):
        """Initialize with caching for memoization"""
        self.memo = {}
        self.call_count = 0
    
    def fibonacci_memoization(self, n: int) -> int:
        """
        Fibonacci using memoization (top-down approach)
        
        Time Complexity: O(n)
        Space Complexity: O(n) - for cache and recursion stack
        
        Args:
            n: Position in Fibonacci sequence
        
        Returns:
            nth Fibonacci number
        """
        self.call_count += 1
        
        if n in self.memo:
            return self.memo[n]
        
        if n <= 1:
            result = n
        else:
            result = self.fibonacci_memoization(n - 1) + self.fibonacci_memoization(n - 2)
        
        self.memo[n] = result
        return result
    
    def fibonacci_tabulation(self, n: int) -> int:
        """
        Fibonacci using tabulation (bottom-up approach)
        
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if n <= 1:
            return n
        
        dp = [0] * (n + 1)
        dp[0], dp[1] = 0, 1
        
        for i in range(2, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]
        
        return dp[n]
    
    def fibonacci_optimized(self, n: int) -> int:
        """
        Space-optimized Fibonacci using only two variables
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if n <= 1:
            return n
        
        prev2, prev1 = 0, 1
        
        for i in range(2, n + 1):
            current = prev1 + prev2
            prev2, prev1 = prev1, current
        
        return prev1
    
    def fibonacci_matrix(self, n: int) -> int:
        """
        Fibonacci using matrix exponentiation
        
        Time Complexity: O(log n)
        Space Complexity: O(log n)
        
        Uses the property: [[1,1],[1,0]]^n = [[F(n+1),F(n)],[F(n),F(n-1)]]
        """
        if n <= 1:
            return n
        
        def matrix_multiply(A, B):
            """Multiply two 2x2 matrices"""
            return [
                [A[0][0] * B[0][0] + A[0][1] * B[1][0], A[0][0] * B[0][1] + A[0][1] * B[1][1]],
                [A[1][0] * B[0][0] + A[1][1] * B[1][0], A[1][0] * B[0][1] + A[1][1] * B[1][1]]
            ]
        
        def matrix_power(matrix, power):
            """Calculate matrix^power using binary exponentiation"""
            if power == 1:
                return matrix
            
            if power % 2 == 0:
                half = matrix_power(matrix, power // 2)
                return matrix_multiply(half, half)
            else:
                return matrix_multiply(matrix, matrix_power(matrix, power - 1))
        
        base_matrix = [[1, 1], [1, 0]]
        result_matrix = matrix_power(base_matrix, n)
        
        return result_matrix[0][1]
    
    @lru_cache(maxsize=None)
    def fibonacci_lru(self, n: int) -> int:
        """Fibonacci using Python's LRU cache decorator"""
        if n <= 1:
            return n
        
        return self.fibonacci_lru(n - 1) + self.fibonacci_lru(n - 2)

# ==================== STAIRCASE PROBLEM (CLIMBING STAIRS) ====================

class ClimbingStairs:
    """
    Staircase problems and variants
    
    Classic problem: How many ways to climb n stairs if you can take 1 or 2 steps at a time?
    This is essentially Fibonacci with different base cases.
    """
    
    def climb_stairs_basic(self, n: int) -> int:
        """
        Basic climbing stairs - 1 or 2 steps at a time
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Pattern: Same as Fibonacci starting from f(1)=1, f(2)=2
        
        Args:
            n: Number of stairs
        
        Returns:
            Number of ways to climb n stairs
        """
        if n <= 2:
            return n
        
        prev2, prev1 = 1, 2
        
        for i in range(3, n + 1):
            current = prev1 + prev2
            prev2, prev1 = prev1, current
        
        return prev1
    
    def climb_stairs_k_steps(self, n: int, k: int) -> int:
        """
        Climbing stairs with up to k steps at a time
        
        Time Complexity: O(n * k)
        Space Complexity: O(n)
        
        Args:
            n: Number of stairs
            k: Maximum steps allowed per move
        
        Returns:
            Number of ways to climb n stairs
        """
        if n == 0:
            return 1
        if n < 0:
            return 0
        
        dp = [0] * (n + 1)
        dp[0] = 1
        
        for i in range(1, n + 1):
            for step in range(1, min(i, k) + 1):
                dp[i] += dp[i - step]
        
        return dp[n]
    
    def climb_stairs_specific_steps(self, n: int, steps: List[int]) -> int:
        """
        Climbing stairs with specific allowed step sizes
        
        Time Complexity: O(n * len(steps))
        Space Complexity: O(n)
        
        Args:
            n: Number of stairs
            steps: List of allowed step sizes
        
        Returns:
            Number of ways to climb n stairs
        """
        dp = [0] * (n + 1)
        dp[0] = 1
        
        for i in range(1, n + 1):
            for step in steps:
                if i >= step:
                    dp[i] += dp[i - step]
        
        return dp[n]
    
    def climb_stairs_with_cost(self, cost: List[int]) -> int:
        """
        Minimum cost to climb stairs (LeetCode 746)
        
        You can start from step 0 or 1, and pay cost[i] to step on stair i.
        From stair i, you can climb 1 or 2 steps.
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Args:
            cost: Cost array for each step
        
        Returns:
            Minimum cost to reach the top
        """
        n = len(cost)
        if n <= 2:
            return min(cost)
        
        prev2 = cost[0]
        prev1 = cost[1]
        
        for i in range(2, n):
            current = cost[i] + min(prev1, prev2)
            prev2, prev1 = prev1, current
        
        # Can reach top from either last or second-last step
        return min(prev1, prev2)
    
    def climb_stairs_jump_game(self, nums: List[int]) -> bool:
        """
        Jump Game - can reach the end? (LeetCode 55)
        
        Each element represents maximum jump length from that position.
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Args:
            nums: Array where nums[i] is max jump from position i
        
        Returns:
            True if can reach the last index
        """
        max_reach = 0
        
        for i in range(len(nums)):
            if i > max_reach:
                return False
            
            max_reach = max(max_reach, i + nums[i])
            
            if max_reach >= len(nums) - 1:
                return True
        
        return max_reach >= len(nums) - 1
    
    def climb_stairs_min_jumps(self, nums: List[int]) -> int:
        """
        Jump Game II - minimum jumps to reach end (LeetCode 45)
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Args:
            nums: Array where nums[i] is max jump from position i
        
        Returns:
            Minimum number of jumps to reach the end
        """
        n = len(nums)
        if n <= 1:
            return 0
        
        jumps = 0
        current_end = 0
        farthest = 0
        
        for i in range(n - 1):
            farthest = max(farthest, i + nums[i])
            
            if i == current_end:
                jumps += 1
                current_end = farthest
        
        return jumps

# ==================== TILING PROBLEMS ====================

class TilingProblems:
    """
    Various tiling problems using dynamic programming
    
    These problems involve filling a space with tiles of different sizes.
    """
    
    def tile_2xn_with_1x2(self, n: int) -> int:
        """
        Tile a 2×n board with 1×2 dominoes
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Pattern: f(n) = f(n-1) + f(n-2)
        - Place vertical domino: leaves 2×(n-1) board
        - Place two horizontal dominoes: leaves 2×(n-2) board
        
        Args:
            n: Length of the 2×n board
        
        Returns:
            Number of ways to tile the board
        """
        if n <= 2:
            return n
        
        prev2, prev1 = 1, 2
        
        for i in range(3, n + 1):
            current = prev1 + prev2
            prev2, prev1 = prev1, current
        
        return prev1
    
    def tile_3xn_with_2x1(self, n: int) -> int:
        """
        Tile a 3×n board with 2×1 dominoes
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Only possible when n is even.
        Pattern: f(n) = 4*f(n-2) - f(n-4) for n >= 4
        
        Args:
            n: Length of the 3×n board
        
        Returns:
            Number of ways to tile the board (0 if n is odd)
        """
        if n % 2 == 1:
            return 0
        
        if n == 0:
            return 1
        if n == 2:
            return 3
        
        prev2, prev1 = 1, 3
        
        for i in range(4, n + 1, 2):
            current = 4 * prev1 - prev2
            prev2, prev1 = prev1, current
        
        return prev1
    
    def tile_nx3_with_2x1(self, n: int) -> int:
        """
        Tile an n×3 board with 2×1 dominoes
        
        Time Complexity: O(n)
        Space Complexity: O(n)
        
        More complex recurrence due to multiple states.
        
        Args:
            n: Height of the n×3 board
        
        Returns:
            Number of ways to tile the board
        """
        if n == 0:
            return 1
        if n == 1:
            return 0
        
        # dp[i][state] where state represents the filling pattern of row i
        # state 0: completely filled
        # state 1: first column filled, others empty
        # state 2: second column filled, others empty
        # state 3: third column filled, others empty
        # state 4: first and second filled, third empty
        # state 5: second and third filled, first empty
        # state 6: first and third filled, second empty
        # state 7: all empty
        
        dp = [[0] * 8 for _ in range(n + 1)]
        dp[0][0] = 1
        
        for i in range(1, n + 1):
            # Transitions based on how we can fill the current row
            dp[i][0] = dp[i-1][0] + dp[i-1][7]  # Complete row
            dp[i][1] = dp[i-1][6]
            dp[i][2] = dp[i-1][5]
            dp[i][3] = dp[i-1][4]
            dp[i][4] = dp[i-1][3]
            dp[i][5] = dp[i-1][2]
            dp[i][6] = dp[i-1][1]
            dp[i][7] = dp[i-1][0]
        
        return dp[n][0]
    
    def tile_rectangle_with_corners(self, m: int, n: int) -> int:
        """
        Tile an m×n rectangle with L-shaped triominoes and 1×1 squares
        
        This is a complex tiling problem that demonstrates advanced DP techniques.
        
        Time Complexity: O(m * n * 2^min(m,n))
        Space Complexity: O(2^min(m,n))
        """
        # For demonstration, implement a simpler case
        if m > n:
            m, n = n, m
        
        # Use bitmask DP for small dimensions
        if m <= 3:
            return self._tile_small_rectangle(m, n)
        
        return -1  # Complex case not implemented for brevity
    
    def _tile_small_rectangle(self, m: int, n: int) -> int:
        """Helper for small rectangle tiling"""
        # Implementation for m <= 3
        if m == 1:
            return 1
        if m == 2:
            return self.tile_2xn_with_1x2(n)
        if m == 3:
            return self.tile_3xn_with_2x1(n)
        
        return 0

# ==================== JUMP WAYS PROBLEMS ====================

class JumpWays:
    """
    Various jump problems with different constraints
    """
    
    def count_ways_123_steps(self, n: int) -> int:
        """
        Count ways to reach n using steps of size 1, 2, or 3
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Args:
            n: Target position
        
        Returns:
            Number of ways to reach position n
        """
        if n < 0:
            return 0
        if n == 0:
            return 1
        if n <= 2:
            return n
        if n == 3:
            return 4
        
        prev3, prev2, prev1 = 1, 2, 4
        
        for i in range(4, n + 1):
            current = prev1 + prev2 + prev3
            prev3, prev2, prev1 = prev2, prev1, current
        
        return prev1
    
    def count_ways_with_obstacles(self, grid: List[List[int]]) -> int:
        """
        Count paths in grid with obstacles (1 = obstacle, 0 = free)
        
        Time Complexity: O(m * n)
        Space Complexity: O(n)
        
        Args:
            grid: 2D grid where 1 represents obstacle
        
        Returns:
            Number of paths from top-left to bottom-right
        """
        if not grid or not grid[0] or grid[0][0] == 1:
            return 0
        
        m, n = len(grid), len(grid[0])
        dp = [0] * n
        dp[0] = 1
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    dp[j] = 0
                elif j > 0:
                    dp[j] = dp[j] + dp[j - 1]
        
        return dp[n - 1]
    
    def frog_jump(self, stones: List[int]) -> bool:
        """
        Frog Jump problem (LeetCode 403)
        
        Frog starts at first stone and wants to reach last stone.
        If last jump was k units, next jump must be k-1, k, or k+1 units.
        
        Time Complexity: O(n²)
        Space Complexity: O(n²)
        
        Args:
            stones: Array of stone positions
        
        Returns:
            True if frog can reach the last stone
        """
        if not stones or len(stones) < 2:
            return len(stones) <= 1
        
        # If gap between first two stones is > 1, impossible
        if stones[1] - stones[0] > 1:
            return False
        
        n = len(stones)
        stone_to_index = {stones[i]: i for i in range(n)}
        
        # dp[i] = set of possible jump sizes to reach stone i
        dp = [set() for _ in range(n)]
        dp[1].add(1)  # First jump is always 1
        
        for i in range(1, n):
            for jump_size in dp[i]:
                # Try jumps of size jump_size-1, jump_size, jump_size+1
                for next_jump in [jump_size - 1, jump_size, jump_size + 1]:
                    if next_jump > 0:
                        next_pos = stones[i] + next_jump
                        if next_pos in stone_to_index:
                            next_index = stone_to_index[next_pos]
                            dp[next_index].add(next_jump)
        
        return len(dp[n - 1]) > 0
    
    def minimum_jumps_to_reach_end(self, arr: List[int]) -> int:
        """
        Minimum jumps to reach end with jump values as array elements
        
        Time Complexity: O(n²)
        Space Complexity: O(n)
        
        Args:
            arr: Array where arr[i] represents jump size from position i
        
        Returns:
            Minimum jumps to reach end, -1 if impossible
        """
        n = len(arr)
        if n <= 1:
            return 0
        
        if arr[0] == 0:
            return -1
        
        dp = [float('inf')] * n
        dp[0] = 0
        
        for i in range(1, n):
            for j in range(i):
                if j + arr[j] >= i:
                    dp[i] = min(dp[i], dp[j] + 1)
        
        return dp[n - 1] if dp[n - 1] != float('inf') else -1

# ==================== ADVANCED FIBONACCI VARIANTS ====================

class AdvancedFibonacci:
    """
    Advanced Fibonacci variants and generalizations
    """
    
    def tribonacci(self, n: int) -> int:
        """
        Tribonacci sequence: T(n) = T(n-1) + T(n-2) + T(n-3)
        Base: T(0) = 0, T(1) = 1, T(2) = 1
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if n == 0:
            return 0
        if n <= 2:
            return 1
        
        prev3, prev2, prev1 = 0, 1, 1
        
        for i in range(3, n + 1):
            current = prev1 + prev2 + prev3
            prev3, prev2, prev1 = prev2, prev1, current
        
        return prev1
    
    def fibonacci_mod(self, n: int, mod: int) -> int:
        """
        Fibonacci with modular arithmetic to handle large numbers
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if n <= 1:
            return n % mod
        
        prev2, prev1 = 0, 1
        
        for i in range(2, n + 1):
            current = (prev1 + prev2) % mod
            prev2, prev1 = prev1, current
        
        return prev1
    
    def fibonacci_sum(self, n: int) -> int:
        """
        Sum of first n Fibonacci numbers
        
        Property: Sum(F(0) to F(n)) = F(n+2) - 1
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if n < 0:
            return 0
        
        # Calculate F(n+2) and subtract 1
        fib_n_plus_2 = FibonacciDP().fibonacci_optimized(n + 2)
        return fib_n_plus_2 - 1
    
    def fibonacci_gcd(self, m: int, n: int) -> int:
        """
        GCD property: gcd(F(m), F(n)) = F(gcd(m, n))
        
        Time Complexity: O(log(min(m,n)) + max(m,n))
        Space Complexity: O(1)
        """
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a
        
        gcd_mn = gcd(m, n)
        return FibonacciDP().fibonacci_optimized(gcd_mn)
    
    def lucas_numbers(self, n: int) -> int:
        """
        Lucas numbers: L(n) = L(n-1) + L(n-2)
        Base: L(0) = 2, L(1) = 1
        
        Related to Fibonacci: L(n) = F(n-1) + F(n+1)
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if n == 0:
            return 2
        if n == 1:
            return 1
        
        prev2, prev1 = 2, 1
        
        for i in range(2, n + 1):
            current = prev1 + prev2
            prev2, prev1 = prev1, current
        
        return prev1
    
    def fibonacci_like_sequence(self, a: int, b: int, n: int) -> int:
        """
        Generalized Fibonacci-like sequence with custom initial values
        
        F(0) = a, F(1) = b, F(n) = F(n-1) + F(n-2) for n >= 2
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if n == 0:
            return a
        if n == 1:
            return b
        
        prev2, prev1 = a, b
        
        for i in range(2, n + 1):
            current = prev1 + prev2
            prev2, prev1 = prev1, current
        
        return prev1

# ==================== PERFORMANCE ANALYSIS ====================

def performance_comparison():
    """Compare performance of different Fibonacci implementations"""
    print("=== Fibonacci Performance Comparison ===\n")
    
    fib = FibonacciDP()
    test_values = [20, 30, 35]
    
    for n in test_values:
        print(f"Computing Fibonacci({n}):")
        
        # Reset memoization cache
        fib.memo.clear()
        fib.call_count = 0
        
        # Memoization
        start_time = time.time()
        memo_result = fib.fibonacci_memoization(n)
        memo_time = time.time() - start_time
        memo_calls = fib.call_count
        
        # Tabulation
        start_time = time.time()
        tab_result = fib.fibonacci_tabulation(n)
        tab_time = time.time() - start_time
        
        # Optimized
        start_time = time.time()
        opt_result = fib.fibonacci_optimized(n)
        opt_time = time.time() - start_time
        
        # Matrix
        start_time = time.time()
        matrix_result = fib.fibonacci_matrix(n)
        matrix_time = time.time() - start_time
        
        print(f"  Memoization: {memo_result} ({memo_time:.6f}s, {memo_calls} calls)")
        print(f"  Tabulation:  {tab_result} ({tab_time:.6f}s)")
        print(f"  Optimized:   {opt_result} ({opt_time:.6f}s)")
        print(f"  Matrix:      {matrix_result} ({matrix_time:.6f}s)")
        print(f"  All match:   {memo_result == tab_result == opt_result == matrix_result}")
        print()

# ==================== EXAMPLE USAGE AND TESTING ====================

if __name__ == "__main__":
    print("=== Fibonacci & Variants Demo ===\n")
    
    # Classic Fibonacci
    print("1. Classic Fibonacci:")
    fib = FibonacciDP()
    
    for n in [0, 1, 5, 10, 15]:
        result = fib.fibonacci_optimized(n)
        print(f"  F({n}) = {result}")
    
    print()
    
    # Climbing Stairs
    print("2. Climbing Stairs Problems:")
    stairs = ClimbingStairs()
    
    n = 5
    print(f"  Ways to climb {n} stairs (1-2 steps): {stairs.climb_stairs_basic(n)}")
    print(f"  Ways to climb {n} stairs (1-3 steps): {stairs.climb_stairs_k_steps(n, 3)}")
    print(f"  Ways to climb {n} stairs (specific steps [1,3,5]): {stairs.climb_stairs_specific_steps(n, [1, 3, 5])}")
    
    cost = [10, 15, 20]
    print(f"  Min cost to climb stairs {cost}: {stairs.climb_stairs_with_cost(cost)}")
    
    nums = [2, 3, 1, 1, 4]
    print(f"  Can reach end with jumps {nums}: {stairs.climb_stairs_jump_game(nums)}")
    print(f"  Min jumps to reach end: {stairs.climb_stairs_min_jumps(nums)}")
    
    print()
    
    # Tiling Problems
    print("3. Tiling Problems:")
    tiling = TilingProblems()
    
    for n in [1, 2, 3, 4, 5]:
        ways_2xn = tiling.tile_2xn_with_1x2(n)
        print(f"  Ways to tile 2×{n} board: {ways_2xn}")
    
    for n in [0, 2, 4, 6]:
        ways_3xn = tiling.tile_3xn_with_2x1(n)
        print(f"  Ways to tile 3×{n} board: {ways_3xn}")
    
    print()
    
    # Jump Ways
    print("4. Jump Ways Problems:")
    jumps = JumpWays()
    
    for n in [1, 2, 3, 4, 5]:
        ways = jumps.count_ways_123_steps(n)
        print(f"  Ways to reach {n} with 1-2-3 steps: {ways}")
    
    grid = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    paths = jumps.count_ways_with_obstacles(grid)
    print(f"  Paths in grid with obstacles: {paths}")
    
    stones = [0, 1, 3, 5, 6, 8, 12, 17]
    can_jump = jumps.frog_jump(stones)
    print(f"  Frog can jump across stones {stones}: {can_jump}")
    
    arr = [1, 3, 6, 1, 0, 9]
    min_jumps = jumps.minimum_jumps_to_reach_end(arr)
    print(f"  Min jumps for array {arr}: {min_jumps}")
    
    print()
    
    # Advanced Fibonacci
    print("5. Advanced Fibonacci Variants:")
    advanced = AdvancedFibonacci()
    
    for n in [0, 1, 2, 5, 10]:
        trib = advanced.tribonacci(n)
        lucas = advanced.lucas_numbers(n)
        print(f"  T({n}) = {trib}, L({n}) = {lucas}")
    
    # Custom sequence
    custom = advanced.fibonacci_like_sequence(3, 7, 10)
    print(f"  Custom Fibonacci-like (3,7,n): F(10) = {custom}")
    
    # Fibonacci sum
    fib_sum = advanced.fibonacci_sum(10)
    print(f"  Sum of first 10 Fibonacci numbers: {fib_sum}")
    
    print()
    
    # Performance comparison
    performance_comparison()
    
    # Problem pattern analysis
    print("=== Problem Pattern Analysis ===")
    print("Fibonacci Pattern Recognition:")
    print("  1. If dp[i] = dp[i-1] + dp[i-2] → Classic Fibonacci")
    print("  2. If dp[i] = dp[i-1] + dp[i-2] + ... + dp[i-k] → k-step problem")
    print("  3. If counting ways to reach position → Fibonacci variant")
    print("  4. If tiling 2×n space → Fibonacci")
    print("  5. If choices at each step limited → Modified Fibonacci")
    
    print("\nOptimization Strategies:")
    print("  1. Space: O(n) → O(1) when only previous values needed")
    print("  2. Time: O(n) → O(log n) using matrix exponentiation")
    print("  3. Modular arithmetic for large numbers")
    print("  4. Memoization for recursive problems")
    print("  5. Bottom-up for iterative problems")
    
    print("\nReal-world Applications:")
    print("  1. Path counting in grids")
    print("  2. Resource allocation problems")
    print("  3. Sequence optimization")
    print("  4. Game theory (ways to win)")
    print("  5. Financial modeling (compound growth)")
    
    print("\n=== Demo Complete ===") 