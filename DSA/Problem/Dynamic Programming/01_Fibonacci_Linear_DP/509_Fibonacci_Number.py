"""
LeetCode 509: Fibonacci Number
Difficulty: Easy
Category: Fibonacci & Linear DP

PROBLEM DESCRIPTION:
===================
The Fibonacci numbers, commonly denoted F(n) form a sequence, called the Fibonacci sequence, 
such that each number is the sum of the two preceding ones, starting from 0 and 1. That is,

F(0) = 0, F(1) = 1
F(n) = F(n - 1) + F(n - 2), for n > 1.

Given n, calculate F(n).

Example 1:
Input: n = 2
Output: 1
Explanation: F(2) = F(1) + F(0) = 1 + 0 = 1.

Example 2:
Input: n = 3
Output: 2
Explanation: F(3) = F(2) + F(1) = 1 + 1 = 2.

Example 3:
Input: n = 4
Output: 3
Explanation: F(4) = F(3) + F(2) = 2 + 1 = 3.

Constraints:
- 0 <= n <= 30
"""

def fibonacci_recursive(n):
    """
    NAIVE RECURSIVE APPROACH:
    ========================
    Direct implementation of Fibonacci definition.
    
    Time Complexity: O(2^n) - exponential due to overlapping subproblems
    Space Complexity: O(n) - recursion stack depth
    """
    if n <= 1:
        return n
    
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)


def fibonacci_memoization(n):
    """
    DYNAMIC PROGRAMMING - TOP DOWN (MEMOIZATION):
    ============================================
    Use memoization to avoid recalculating same subproblems.
    
    Time Complexity: O(n) - each subproblem calculated once
    Space Complexity: O(n) - memoization table + recursion stack
    """
    memo = {}
    
    def fib_helper(n):
        if n <= 1:
            return n
        
        if n in memo:
            return memo[n]
        
        memo[n] = fib_helper(n - 1) + fib_helper(n - 2)
        return memo[n]
    
    return fib_helper(n)


def fibonacci_tabulation(n):
    """
    DYNAMIC PROGRAMMING - BOTTOM UP (TABULATION):
    =============================================
    Build solution from bottom up using array.
    
    Time Complexity: O(n) - single pass through array
    Space Complexity: O(n) - DP array
    """
    if n <= 1:
        return n
    
    dp = [0] * (n + 1)
    dp[0] = 0
    dp[1] = 1
    
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    
    return dp[n]


def fibonacci_space_optimized(n):
    """
    SPACE OPTIMIZED DYNAMIC PROGRAMMING:
    ===================================
    Since we only need previous two values, use variables instead of array.
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(1) - constant space
    """
    if n <= 1:
        return n
    
    prev2 = 0  # F(0)
    prev1 = 1  # F(1)
    
    for i in range(2, n + 1):
        current = prev1 + prev2
        prev2 = prev1
        prev1 = current
    
    return prev1


def fibonacci_matrix_exponentiation(n):
    """
    MATRIX EXPONENTIATION APPROACH:
    ==============================
    Use matrix exponentiation to compute Fibonacci in O(log n).
    Based on: [[F(n+1), F(n)], [F(n), F(n-1)]] = [[1,1], [1,0]]^n
    
    Time Complexity: O(log n) - matrix exponentiation
    Space Complexity: O(log n) - recursion stack for exponentiation
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
        """Compute matrix^power using fast exponentiation"""
        if power == 1:
            return matrix
        
        if power % 2 == 0:
            half = matrix_power(matrix, power // 2)
            return matrix_multiply(half, half)
        else:
            return matrix_multiply(matrix, matrix_power(matrix, power - 1))
    
    # Base matrix for Fibonacci: [[1, 1], [1, 0]]
    base_matrix = [[1, 1], [1, 0]]
    result_matrix = matrix_power(base_matrix, n)
    
    # F(n) is at position [0][1] in the result matrix
    return result_matrix[0][1]


def fibonacci_golden_ratio(n):
    """
    BINET'S FORMULA (GOLDEN RATIO):
    ==============================
    Direct formula using golden ratio: φ = (1 + √5) / 2
    F(n) = (φ^n - ψ^n) / √5, where ψ = (1 - √5) / 2
    
    Time Complexity: O(1) - constant time calculation
    Space Complexity: O(1) - constant space
    
    Note: May have floating point precision issues for large n
    """
    if n <= 1:
        return n
    
    import math
    
    sqrt5 = math.sqrt(5)
    phi = (1 + sqrt5) / 2  # Golden ratio
    psi = (1 - sqrt5) / 2  # Conjugate of golden ratio
    
    # Binet's formula
    result = (phi ** n - psi ** n) / sqrt5
    
    return round(result)


def fibonacci_iterative_fast(n):
    """
    FAST ITERATIVE DOUBLING:
    ========================
    Use the fast doubling method based on:
    F(2k) = F(k) * (2*F(k+1) - F(k))
    F(2k+1) = F(k+1)^2 + F(k)^2
    
    Time Complexity: O(log n) - process bits of n
    Space Complexity: O(1) - constant space
    """
    if n <= 1:
        return n
    
    def fib_pair(k):
        """Returns (F(k), F(k+1))"""
        if k == 0:
            return (0, 1)
        
        m = k // 2
        f_m, f_m1 = fib_pair(m)
        
        c = f_m * (2 * f_m1 - f_m)
        d = f_m * f_m + f_m1 * f_m1
        
        if k % 2 == 0:
            return (c, d)
        else:
            return (d, c + d)
    
    return fib_pair(n)[0]


def fibonacci_with_memoization_class():
    """
    CLASS-BASED MEMOIZATION:
    =======================
    Fibonacci calculator with persistent memoization.
    """
    class FibonacciCalculator:
        def __init__(self):
            self.memo = {0: 0, 1: 1}
        
        def calculate(self, n):
            if n in self.memo:
                return self.memo[n]
            
            self.memo[n] = self.calculate(n - 1) + self.calculate(n - 2)
            return self.memo[n]
        
        def get_sequence(self, n):
            """Get Fibonacci sequence up to F(n)"""
            return [self.calculate(i) for i in range(n + 1)]
    
    return FibonacciCalculator()


def fibonacci_sequence_generator(n):
    """
    GENERATOR APPROACH:
    ==================
    Generate Fibonacci sequence up to F(n) using generator.
    
    Time Complexity: O(n) - generate n numbers
    Space Complexity: O(1) - constant space (generator)
    """
    def fib_generator():
        a, b = 0, 1
        while True:
            yield a
            a, b = b, a + b
    
    gen = fib_generator()
    result = 0
    for i in range(n + 1):
        result = next(gen)
    
    return result


# Test cases
def test_fibonacci():
    """Test all implementations with various inputs"""
    test_cases = [
        (0, 0),
        (1, 1),
        (2, 1),
        (3, 2),
        (4, 3),
        (5, 5),
        (6, 8),
        (10, 55),
        (15, 610),
        (20, 6765),
        (25, 75025),
        (30, 832040)
    ]
    
    print("Testing Fibonacci Number Solutions:")
    print("=" * 70)
    
    for i, (n, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: F({n})")
        print(f"Expected: {expected}")
        
        # Test approaches (skip recursive for large n)
        if n <= 20:
            recursive = fibonacci_recursive(n)
            print(f"Recursive:        {recursive:>8} {'✓' if recursive == expected else '✗'}")
        
        memo = fibonacci_memoization(n)
        tab = fibonacci_tabulation(n)
        space_opt = fibonacci_space_optimized(n)
        matrix = fibonacci_matrix_exponentiation(n)
        golden = fibonacci_golden_ratio(n)
        fast_iter = fibonacci_iterative_fast(n)
        generator = fibonacci_sequence_generator(n)
        
        print(f"Memoization:      {memo:>8} {'✓' if memo == expected else '✗'}")
        print(f"Tabulation:       {tab:>8} {'✓' if tab == expected else '✗'}")
        print(f"Space Optimized:  {space_opt:>8} {'✓' if space_opt == expected else '✗'}")
        print(f"Matrix Exp:       {matrix:>8} {'✓' if matrix == expected else '✗'}")
        print(f"Golden Ratio:     {golden:>8} {'✓' if golden == expected else '✗'}")
        print(f"Fast Iterative:   {fast_iter:>8} {'✓' if fast_iter == expected else '✗'}")
        print(f"Generator:        {generator:>8} {'✓' if generator == expected else '✗'}")
    
    # Demonstrate class-based calculator
    print(f"\nClass-based Calculator:")
    calc = fibonacci_with_memoization_class()
    print(f"F(10) = {calc.calculate(10)}")
    print(f"Sequence F(0) to F(10): {calc.get_sequence(10)}")
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Recursive:        Time: O(2^n),    Space: O(n)")
    print("Memoization:      Time: O(n),      Space: O(n)")
    print("Tabulation:       Time: O(n),      Space: O(n)")
    print("Space Optimized:  Time: O(n),      Space: O(1)")
    print("Matrix Exp:       Time: O(log n),  Space: O(log n)")
    print("Golden Ratio:     Time: O(1),      Space: O(1)")
    print("Fast Iterative:   Time: O(log n),  Space: O(1)")
    print("Generator:        Time: O(n),      Space: O(1)")


if __name__ == "__main__":
    test_fibonacci()


"""
PATTERN RECOGNITION:
==================
This is the classic Fibonacci sequence problem:
- Base cases: F(0) = 0, F(1) = 1
- Recurrence: F(n) = F(n-1) + F(n-2)
- Foundation for many DP problems

KEY INSIGHTS:
============
1. Classic example of overlapping subproblems
2. Exponential recursion can be optimized with memoization
3. Can be solved iteratively with O(1) space
4. Multiple advanced approaches exist for very large n

MATHEMATICAL PROPERTIES:
=======================
1. Golden Ratio: φ = (1 + √5) / 2 ≈ 1.618
2. Binet's Formula: F(n) = (φ^n - ψ^n) / √5
3. Matrix Form: [[F(n+1), F(n)], [F(n), F(n-1)]] = [[1,1], [1,0]]^n
4. Fast Doubling: F(2k) = F(k)(2F(k+1) - F(k)), F(2k+1) = F(k+1)² + F(k)²

RECURRENCE RELATION:
===================
F(n) = F(n-1) + F(n-2)
Base cases: F(0) = 0, F(1) = 1

OPTIMIZATION TECHNIQUES:
=======================
1. Memoization: O(n) time, O(n) space
2. Tabulation: O(n) time, O(n) space
3. Space optimization: O(n) time, O(1) space
4. Matrix exponentiation: O(log n) time
5. Golden ratio formula: O(1) time (with precision limits)
6. Fast doubling: O(log n) time, O(1) space

VARIANTS TO PRACTICE:
====================
- Climbing Stairs (70) - same recurrence relation
- Tribonacci Number (1137) - three-term recurrence
- House Robber (198) - Fibonacci with constraints
- Decode Ways (91) - Fibonacci-like counting

INTERVIEW TIPS:
==============
1. Start with naive recursive solution
2. Identify overlapping subproblems for memoization
3. Show iterative DP solution
4. Optimize space from O(n) to O(1)
5. For advanced discussion: mention matrix exponentiation
6. Discuss golden ratio formula for mathematical insight
7. Handle edge cases (n = 0, n = 1)
8. Mention applications (nature patterns, algorithms)
"""
