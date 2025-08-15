"""
LeetCode 279: Perfect Squares
Difficulty: Medium
Category: Fibonacci & Linear DP (Unbounded Knapsack variant)

PROBLEM DESCRIPTION:
===================
Given an integer n, return the least number of perfect square numbers that sum to n.

A perfect square is an integer that is the square of an integer; in other words, it is the 
product of some integer with itself. For example, 1, 4, 9, and 16 are perfect squares while 
3 and 11 are not.

Example 1:
Input: n = 12
Output: 3
Explanation: 12 = 4 + 4 + 4.

Example 2:
Input: n = 13
Output: 2
Explanation: 13 = 4 + 9.

Constraints:
- 1 <= n <= 10^4
"""

def num_squares_bruteforce(n):
    """
    BRUTE FORCE APPROACH - RECURSION:
    ================================
    Try all possible perfect squares at each step.
    
    Time Complexity: O(n^(n/2)) - extremely high due to overlapping subproblems
    Space Complexity: O(sqrt(n)) - recursion stack depth
    """
    import math
    
    def min_squares(remaining):
        if remaining == 0:
            return 0
        if remaining < 0:
            return float('inf')
        
        min_count = float('inf')
        
        # Try all perfect squares <= remaining
        for i in range(1, int(math.sqrt(remaining)) + 1):
            square = i * i
            if square <= remaining:
                count = 1 + min_squares(remaining - square)
                min_count = min(min_count, count)
        
        return min_count
    
    return min_squares(n)


def num_squares_memoization(n):
    """
    DYNAMIC PROGRAMMING - TOP DOWN (MEMOIZATION):
    ============================================
    Use memoization to avoid recalculating same subproblems.
    
    Time Complexity: O(n * sqrt(n)) - n states, sqrt(n) transitions each
    Space Complexity: O(n) - memoization table + recursion stack
    """
    import math
    
    memo = {}
    
    def min_squares(remaining):
        if remaining == 0:
            return 0
        if remaining < 0:
            return float('inf')
        
        if remaining in memo:
            return memo[remaining]
        
        min_count = float('inf')
        
        for i in range(1, int(math.sqrt(remaining)) + 1):
            square = i * i
            if square <= remaining:
                count = 1 + min_squares(remaining - square)
                min_count = min(min_count, count)
        
        memo[remaining] = min_count
        return min_count
    
    return min_squares(n)


def num_squares_tabulation(n):
    """
    DYNAMIC PROGRAMMING - BOTTOM UP (TABULATION):
    =============================================
    Build solution from bottom up using DP array.
    dp[i] = minimum perfect squares that sum to i
    
    Time Complexity: O(n * sqrt(n)) - nested loops
    Space Complexity: O(n) - DP array
    """
    import math
    
    # dp[i] = minimum number of perfect squares that sum to i
    dp = [float('inf')] * (n + 1)
    dp[0] = 0  # Base case: 0 requires 0 perfect squares
    
    # For each number from 1 to n
    for i in range(1, n + 1):
        # Try all perfect squares <= i
        for j in range(1, int(math.sqrt(i)) + 1):
            square = j * j
            if square <= i:
                dp[i] = min(dp[i], dp[i - square] + 1)
    
    return dp[n]


def num_squares_optimized_dp(n):
    """
    OPTIMIZED DP WITH EARLY TERMINATION:
    ===================================
    Add optimizations for better performance.
    
    Time Complexity: O(n * sqrt(n)) - worst case, often better
    Space Complexity: O(n) - DP array
    """
    import math
    
    # Precompute perfect squares up to n
    squares = []
    i = 1
    while i * i <= n:
        squares.append(i * i)
        i += 1
    
    dp = [float('inf')] * (n + 1)
    dp[0] = 0
    
    for i in range(1, n + 1):
        for square in squares:
            if square > i:
                break
            dp[i] = min(dp[i], dp[i - square] + 1)
            
            # Early termination: if we achieve minimum possible (1)
            if dp[i] == 1:
                break
    
    return dp[n]


def num_squares_bfs(n):
    """
    BFS APPROACH:
    ============
    Use BFS to find minimum number of steps to reach n.
    Each step uses a perfect square.
    
    Time Complexity: O(n * sqrt(n)) - similar to DP
    Space Complexity: O(n) - queue and visited set
    """
    import math
    from collections import deque
    
    if n <= 3:
        return n
    
    # Precompute perfect squares
    squares = []
    i = 1
    while i * i <= n:
        squares.append(i * i)
        i += 1
    
    # BFS
    queue = deque([n])
    visited = {n}
    level = 0
    
    while queue:
        level += 1
        size = len(queue)
        
        for _ in range(size):
            current = queue.popleft()
            
            for square in squares:
                next_val = current - square
                
                if next_val == 0:
                    return level
                
                if next_val > 0 and next_val not in visited:
                    visited.add(next_val)
                    queue.append(next_val)
    
    return level


def num_squares_mathematical(n):
    """
    MATHEMATICAL APPROACH - LAGRANGE'S FOUR SQUARE THEOREM:
    ======================================================
    Use mathematical properties to solve efficiently.
    
    Time Complexity: O(sqrt(n)) - checking perfect squares
    Space Complexity: O(1) - constant space
    """
    import math
    
    def is_perfect_square(num):
        root = int(math.sqrt(num))
        return root * root == num
    
    # Check if n is a perfect square (answer = 1)
    if is_perfect_square(n):
        return 1
    
    # Check if n can be represented as sum of 2 perfect squares
    for i in range(1, int(math.sqrt(n)) + 1):
        if is_perfect_square(n - i * i):
            return 2
    
    # Check if n can be represented as sum of 4 perfect squares
    # According to Legendre's three-square theorem:
    # n can be represented as sum of 3 squares iff n ≠ 4^a(8b + 7)
    temp = n
    while temp % 4 == 0:
        temp //= 4
    if temp % 8 == 7:
        return 4
    
    # Otherwise, answer is 3
    return 3


def num_squares_static_dp():
    """
    STATIC DP WITH PRECOMPUTATION:
    ==============================
    Precompute results for reuse across multiple queries.
    """
    class PerfectSquaresCalculator:
        def __init__(self):
            self.dp = [0]  # dp[0] = 0
            self.squares = [1]  # List of perfect squares
        
        def num_squares(self, n):
            import math
            
            # Extend dp array if needed
            while len(self.dp) <= n:
                self._extend_dp()
            
            return self.dp[n]
        
        def _extend_dp(self):
            current_length = len(self.dp)
            
            # Add new perfect squares if needed
            next_square_root = int(math.sqrt(self.squares[-1])) + 1
            next_square = next_square_root * next_square_root
            if next_square <= current_length:
                self.squares.append(next_square)
            
            # Calculate dp for next number
            min_count = float('inf')
            for square in self.squares:
                if square > current_length:
                    break
                min_count = min(min_count, self.dp[current_length - square] + 1)
            
            self.dp.append(min_count)
    
    return PerfectSquaresCalculator()


def num_squares_space_optimized_bfs(n):
    """
    SPACE OPTIMIZED BFS:
    ===================
    BFS with optimized space usage.
    
    Time Complexity: O(n * sqrt(n))
    Space Complexity: O(sqrt(n)) - reduced space usage
    """
    import math
    from collections import deque
    
    if n <= 3:
        return n
    
    # Check if n is perfect square first
    sqrt_n = int(math.sqrt(n))
    if sqrt_n * sqrt_n == n:
        return 1
    
    # Precompute perfect squares
    squares = [i * i for i in range(1, sqrt_n + 1)]
    
    # BFS with level-by-level processing
    current_level = {n}
    level = 0
    
    while current_level:
        level += 1
        next_level = set()
        
        for num in current_level:
            for square in squares:
                if square > num:
                    break
                
                next_num = num - square
                
                if next_num == 0:
                    return level
                
                if next_num > 0:
                    next_level.add(next_num)
        
        current_level = next_level
    
    return level


def num_squares_with_composition(n):
    """
    FIND ACTUAL PERFECT SQUARES COMPOSITION:
    =======================================
    Return both minimum count and actual perfect squares used.
    
    Time Complexity: O(n * sqrt(n)) - DP + composition reconstruction
    Space Complexity: O(n) - DP array and parent tracking
    """
    import math
    
    dp = [float('inf')] * (n + 1)
    parent = [-1] * (n + 1)
    dp[0] = 0
    
    for i in range(1, n + 1):
        for j in range(1, int(math.sqrt(i)) + 1):
            square = j * j
            if square <= i and dp[i - square] + 1 < dp[i]:
                dp[i] = dp[i - square] + 1
                parent[i] = square
    
    # Reconstruct composition
    composition = []
    current = n
    while current > 0:
        square_used = parent[current]
        composition.append(square_used)
        current -= square_used
    
    return dp[n], composition


# Test cases
def test_num_squares():
    """Test all implementations with various inputs"""
    test_cases = [
        (1, 1),
        (4, 1),
        (12, 3),
        (13, 2),
        (7, 4),
        (9, 1),
        (10, 2),
        (25, 1),
        (26, 2),
        (100, 1),
        (123, 3),
        (9999, 4)
    ]
    
    print("Testing Perfect Squares Solutions:")
    print("=" * 70)
    
    for i, (n, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: n = {n}")
        print(f"Expected: {expected}")
        
        # Test approaches (skip brute force for large n)
        if n <= 15:
            brute = num_squares_bruteforce(n)
            print(f"Brute Force:      {brute:>3} {'✓' if brute == expected else '✗'}")
        
        memo = num_squares_memoization(n)
        tab = num_squares_tabulation(n)
        opt_dp = num_squares_optimized_dp(n)
        bfs = num_squares_bfs(n)
        math_approach = num_squares_mathematical(n)
        space_opt_bfs = num_squares_space_optimized_bfs(n)
        
        print(f"Memoization:      {memo:>3} {'✓' if memo == expected else '✗'}")
        print(f"Tabulation:       {tab:>3} {'✓' if tab == expected else '✗'}")
        print(f"Optimized DP:     {opt_dp:>3} {'✓' if opt_dp == expected else '✗'}")
        print(f"BFS:              {bfs:>3} {'✓' if bfs == expected else '✗'}")
        print(f"Mathematical:     {math_approach:>3} {'✓' if math_approach == expected else '✗'}")
        print(f"Space Opt BFS:    {space_opt_bfs:>3} {'✓' if space_opt_bfs == expected else '✗'}")
        
        # Show composition for small numbers
        if n <= 30:
            count, composition = num_squares_with_composition(n)
            print(f"Composition: {' + '.join(map(str, composition))} = {sum(composition)}")
    
    # Test static calculator
    print(f"\nStatic Calculator Test:")
    calc = num_squares_static_dp()
    for test_n in [12, 13, 25, 26]:
        result = calc.num_squares(test_n)
        expected = next(exp for n, exp in test_cases if n == test_n)
        print(f"Static calc({test_n}) = {result} {'✓' if result == expected else '✗'}")
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Brute Force:      Time: O(n^(n/2)),   Space: O(sqrt(n))")
    print("Memoization:      Time: O(n*sqrt(n)), Space: O(n)")
    print("Tabulation:       Time: O(n*sqrt(n)), Space: O(n)")
    print("Optimized DP:     Time: O(n*sqrt(n)), Space: O(n)")
    print("BFS:              Time: O(n*sqrt(n)), Space: O(n)")
    print("Mathematical:     Time: O(sqrt(n)),   Space: O(1)")
    print("Space Opt BFS:    Time: O(n*sqrt(n)), Space: O(sqrt(n))")


if __name__ == "__main__":
    test_num_squares()


"""
PATTERN RECOGNITION:
==================
This is an Unbounded Knapsack variant:
- Unlimited use of perfect squares (1, 4, 9, 16, ...)
- Minimize number of perfect squares used
- Similar to Coin Change but with perfect squares as "coins"

KEY INSIGHTS:
============
1. Perfect squares are the "coins": 1², 2², 3², ..., √n²
2. We want minimum number of "coins" to make sum n
3. Each perfect square can be used unlimited times
4. This is exactly the Coin Change problem!

STATE DEFINITION:
================
dp[i] = minimum number of perfect squares that sum to i

RECURRENCE RELATION:
===================
dp[i] = min(dp[i], dp[i - j²] + 1) for all j where j² ≤ i
Base case: dp[0] = 0

MATHEMATICAL OPTIMIZATION:
=========================
Lagrange's Four Square Theorem: Every positive integer can be represented 
as sum of at most 4 perfect squares.

Legendre's Three Square Theorem: A positive integer can be represented as 
sum of 3 perfect squares iff it's not of the form 4ᵃ(8b + 7).

Algorithm:
1. If n is perfect square → return 1
2. If n = a² + b² for some a,b → return 2  
3. If n is of form 4ᵃ(8b + 7) → return 4
4. Otherwise → return 3

OPTIMIZATION TECHNIQUES:
=======================
1. Precompute perfect squares up to √n
2. BFS approach: find minimum steps to reach 0
3. Mathematical approach: O(√n) using number theory
4. Early termination in DP when dp[i] = 1

VARIANTS TO PRACTICE:
====================
- Coin Change (322) - same pattern with different "coins"
- Coin Change 2 (518) - count ways instead of minimum coins
- Combination Sum IV (377) - count combinations
- Minimum Cost For Tickets (983) - similar optimization problem

INTERVIEW TIPS:
==============
1. Recognize as Coin Change variant with perfect squares
2. Show DP solution first: O(n√n) time
3. Mention BFS as alternative approach
4. For advanced discussion: mathematical approach using number theory
5. Handle edge cases (n = 1, perfect squares)
6. Discuss space optimization possibilities
7. Compare with regular Coin Change problem
"""
