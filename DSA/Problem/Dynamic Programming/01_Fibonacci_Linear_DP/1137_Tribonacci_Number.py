"""
LeetCode 1137: N-th Tribonacci Number
Difficulty: Easy
Category: Fibonacci & Linear DP

PROBLEM DESCRIPTION:
===================
The Tribonacci sequence Tn is defined as follows:

T0 = 0, T1 = 1, T2 = 1, and Tn+3 = Tn + Tn+1 + Tn+2 for n >= 0.

Given n, return the value of Tn.

Example 1:
Input: n = 4
Output: 4
Explanation:
T_0 = 0
T_1 = 1
T_2 = 1
T_3 = 0 + 1 + 1 = 2
T_4 = 1 + 1 + 2 = 4

Example 2:
Input: n = 25
Output: 1389537

Constraints:
- 0 <= n <= 37
"""

def tribonacci_recursive(n):
    """
    NAIVE RECURSIVE APPROACH:
    ========================
    Direct implementation of Tribonacci definition.
    
    Time Complexity: O(3^n) - exponential due to overlapping subproblems
    Space Complexity: O(n) - recursion stack depth
    """
    if n == 0:
        return 0
    if n == 1 or n == 2:
        return 1
    
    return (tribonacci_recursive(n - 1) + 
            tribonacci_recursive(n - 2) + 
            tribonacci_recursive(n - 3))


def tribonacci_memoization(n):
    """
    DYNAMIC PROGRAMMING - TOP DOWN (MEMOIZATION):
    ============================================
    Use memoization to avoid recalculating same subproblems.
    
    Time Complexity: O(n) - each subproblem calculated once
    Space Complexity: O(n) - memoization table + recursion stack
    """
    memo = {}
    
    def trib_helper(n):
        if n == 0:
            return 0
        if n == 1 or n == 2:
            return 1
        
        if n in memo:
            return memo[n]
        
        memo[n] = (trib_helper(n - 1) + 
                   trib_helper(n - 2) + 
                   trib_helper(n - 3))
        return memo[n]
    
    return trib_helper(n)


def tribonacci_tabulation(n):
    """
    DYNAMIC PROGRAMMING - BOTTOM UP (TABULATION):
    =============================================
    Build solution from bottom up using array.
    
    Time Complexity: O(n) - single pass through array
    Space Complexity: O(n) - DP array
    """
    if n == 0:
        return 0
    if n == 1 or n == 2:
        return 1
    
    dp = [0] * (n + 1)
    dp[0] = 0
    dp[1] = 1
    dp[2] = 1
    
    for i in range(3, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2] + dp[i - 3]
    
    return dp[n]


def tribonacci_space_optimized(n):
    """
    SPACE OPTIMIZED DYNAMIC PROGRAMMING:
    ===================================
    Since we only need previous three values, use variables instead of array.
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(1) - constant space
    """
    if n == 0:
        return 0
    if n == 1 or n == 2:
        return 1
    
    prev3 = 0  # T(n-3)
    prev2 = 1  # T(n-2)
    prev1 = 1  # T(n-1)
    
    for i in range(3, n + 1):
        current = prev1 + prev2 + prev3
        prev3 = prev2
        prev2 = prev1
        prev1 = current
    
    return prev1


def tribonacci_matrix_exponentiation(n):
    """
    MATRIX EXPONENTIATION APPROACH:
    ==============================
    Use matrix exponentiation to compute Tribonacci in O(log n).
    
    Time Complexity: O(log n) - matrix exponentiation
    Space Complexity: O(1) - constant space for 3x3 matrices
    """
    if n == 0:
        return 0
    if n == 1 or n == 2:
        return 1
    
    def matrix_multiply(A, B):
        """Multiply two 3x3 matrices"""
        result = [[0] * 3 for _ in range(3)]
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    result[i][j] += A[i][k] * B[k][j]
        return result
    
    def matrix_power(matrix, power):
        """Compute matrix^power using fast exponentiation"""
        if power == 1:
            return matrix
        
        if power % 2 == 0:
            half = matrix_power(matrix, power // 2)
            return matrix_multiply(half, half)
        else:
            return matrix_multiply(matrix, matrix_power(matrix, power - 1))
    
    # Transformation matrix for Tribonacci
    # [T(n+1)]   [1 1 1] [T(n)  ]
    # [T(n)  ] = [1 0 0] [T(n-1)]
    # [T(n-1)]   [0 1 0] [T(n-2)]
    base_matrix = [
        [1, 1, 1],
        [1, 0, 0],
        [0, 1, 0]
    ]
    
    result_matrix = matrix_power(base_matrix, n - 2)
    
    # [T(3)]   [result_matrix] [T(2)]   [result_matrix] [1]
    # [T(2)] = [            ] [T(1)] = [            ] [1]
    # [T(1)]                  [T(0)]                  [0]
    
    return (result_matrix[0][0] * 1 +  # T(2) * coefficient
            result_matrix[0][1] * 1 +  # T(1) * coefficient  
            result_matrix[0][2] * 0)   # T(0) * coefficient


def tribonacci_closed_form():
    """
    CLOSED FORM FORMULA:
    ===================
    Tribonacci has a closed form using roots of x³ - x² - x - 1 = 0
    More complex than Fibonacci's golden ratio formula.
    
    Time Complexity: O(1) - constant time
    Space Complexity: O(1) - constant space
    
    Note: Complex to implement and may have precision issues.
    """
    import cmath
    import math
    
    def tribonacci_closed(n):
        if n == 0:
            return 0
        if n == 1 or n == 2:
            return 1
        
        # The characteristic polynomial is x³ - x² - x - 1 = 0
        # This has one real root and two complex conjugate roots
        
        # Real root (approximately 1.839)
        # For exact computation, we'd need to solve the cubic equation
        # This is a simplified approximation
        
        # Using the fact that for large n, T(n) ≈ c * α^n where α is the largest root
        alpha = 1.839286755214161  # Approximate largest root
        
        # This is an approximation - exact closed form is more complex
        return round(alpha ** n / 3.678)
    
    return tribonacci_closed


def tribonacci_iterative_optimized(n):
    """
    OPTIMIZED ITERATIVE APPROACH:
    =============================
    Highly optimized iterative solution with minimal operations.
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(1) - constant space
    """
    if n < 3:
        return 1 if n else 0
    
    a, b, c = 0, 1, 1
    
    for _ in range(n - 2):
        a, b, c = b, c, a + b + c
    
    return c


def tribonacci_with_sequence(n):
    """
    GENERATE TRIBONACCI SEQUENCE:
    =============================
    Generate and return the entire sequence up to T(n).
    
    Time Complexity: O(n) - generate n numbers
    Space Complexity: O(n) - store entire sequence
    """
    if n == 0:
        return [0]
    if n == 1:
        return [0, 1]
    if n == 2:
        return [0, 1, 1]
    
    sequence = [0, 1, 1]
    
    for i in range(3, n + 1):
        next_val = sequence[i - 1] + sequence[i - 2] + sequence[i - 3]
        sequence.append(next_val)
    
    return sequence


def tribonacci_generator():
    """
    GENERATOR APPROACH:
    ==================
    Generate Tribonacci numbers using generator.
    
    Time Complexity: O(1) per number generated
    Space Complexity: O(1) - constant space
    """
    def trib_generator():
        a, b, c = 0, 1, 1
        yield a
        yield b
        yield c
        
        while True:
            a, b, c = b, c, a + b + c
            yield c
    
    return trib_generator()


class TribonacciCalculator:
    """
    CLASS-BASED CALCULATOR WITH CACHING:
    ===================================
    Tribonacci calculator with persistent memoization.
    """
    def __init__(self):
        self.cache = {0: 0, 1: 1, 2: 1}
        self.max_computed = 2
    
    def get(self, n):
        """Get T(n) with caching"""
        if n <= self.max_computed:
            return self.cache[n]
        
        # Extend cache up to n
        for i in range(self.max_computed + 1, n + 1):
            self.cache[i] = (self.cache[i - 1] + 
                           self.cache[i - 2] + 
                           self.cache[i - 3])
        
        self.max_computed = n
        return self.cache[n]
    
    def get_sequence(self, n):
        """Get sequence [T(0), T(1), ..., T(n)]"""
        self.get(n)  # Ensure cache is populated
        return [self.cache[i] for i in range(n + 1)]


# Test cases
def test_tribonacci():
    """Test all implementations with various inputs"""
    test_cases = [
        (0, 0),
        (1, 1),
        (2, 1),
        (3, 2),
        (4, 4),
        (5, 7),
        (6, 13),
        (10, 149),
        (15, 3136),
        (25, 1389537),
        (30, 53798080),
        (37, 2082876103)
    ]
    
    print("Testing Tribonacci Number Solutions:")
    print("=" * 70)
    
    for i, (n, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: T({n})")
        print(f"Expected: {expected}")
        
        # Test approaches (skip recursive for large n)
        if n <= 15:
            recursive = tribonacci_recursive(n)
            print(f"Recursive:        {recursive:>10} {'✓' if recursive == expected else '✗'}")
        
        memo = tribonacci_memoization(n)
        tab = tribonacci_tabulation(n)
        space_opt = tribonacci_space_optimized(n)
        iterative_opt = tribonacci_iterative_optimized(n)
        
        print(f"Memoization:      {memo:>10} {'✓' if memo == expected else '✗'}")
        print(f"Tabulation:       {tab:>10} {'✓' if tab == expected else '✗'}")
        print(f"Space Optimized:  {space_opt:>10} {'✓' if space_opt == expected else '✗'}")
        print(f"Iterative Opt:    {iterative_opt:>10} {'✓' if iterative_opt == expected else '✗'}")
        
        if n <= 20:
            matrix = tribonacci_matrix_exponentiation(n)
            print(f"Matrix Exp:       {matrix:>10} {'✓' if matrix == expected else '✗'}")
    
    # Test sequence generation
    print(f"\nSequence Generation:")
    sequence = tribonacci_with_sequence(10)
    print(f"T(0) to T(10): {sequence}")
    
    # Test generator
    print(f"\nGenerator Test:")
    gen = tribonacci_generator()
    gen_sequence = [next(gen) for _ in range(11)]
    print(f"Generator T(0) to T(10): {gen_sequence}")
    
    # Test class-based calculator
    print(f"\nClass-based Calculator:")
    calc = TribonacciCalculator()
    print(f"T(10) = {calc.get(10)}")
    print(f"T(15) = {calc.get(15)}")
    print(f"Cached sequence T(0) to T(6): {calc.get_sequence(6)}")
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Recursive:        Time: O(3^n),    Space: O(n)")
    print("Memoization:      Time: O(n),      Space: O(n)")
    print("Tabulation:       Time: O(n),      Space: O(n)")
    print("Space Optimized:  Time: O(n),      Space: O(1)")
    print("Matrix Exp:       Time: O(log n),  Space: O(1)")
    print("Iterative Opt:    Time: O(n),      Space: O(1)")
    print("Generator:        Time: O(1)/num,  Space: O(1)")


if __name__ == "__main__":
    test_tribonacci()


"""
PATTERN RECOGNITION:
==================
This is a generalization of the Fibonacci sequence:
- Base cases: T(0) = 0, T(1) = 1, T(2) = 1
- Recurrence: T(n) = T(n-1) + T(n-2) + T(n-3)
- Similar optimization techniques as Fibonacci

KEY INSIGHTS:
============
1. Three-term recurrence instead of two-term
2. Need to track three previous values instead of two
3. Same DP optimization principles apply
4. Matrix exponentiation uses 3x3 matrix instead of 2x2

RECURRENCE RELATION:
===================
T(n) = T(n-1) + T(n-2) + T(n-3) for n >= 3
Base cases: T(0) = 0, T(1) = 1, T(2) = 1

OPTIMIZATION TECHNIQUES:
=======================
1. Memoization: O(n) time, O(n) space
2. Tabulation: O(n) time, O(n) space  
3. Space optimization: O(n) time, O(1) space (track 3 variables)
4. Matrix exponentiation: O(log n) time, O(1) space
5. Closed form: O(1) time (complex to derive)

MATRIX FORM:
===========
[T(n+1)]   [1 1 1] [T(n)  ]
[T(n)  ] = [1 0 0] [T(n-1)]
[T(n-1)]   [0 1 0] [T(n-2)]

COMPARISON WITH FIBONACCI:
=========================
- Fibonacci: F(n) = F(n-1) + F(n-2), needs 2 variables
- Tribonacci: T(n) = T(n-1) + T(n-2) + T(n-3), needs 3 variables
- Growth rate: Tribonacci grows faster than Fibonacci

VARIANTS TO PRACTICE:
====================
- Fibonacci Number (509) - two-term version
- Climbing Stairs (70) - same recurrence as Fibonacci
- N-th Tribonacci Number (1137) - this problem
- House Robber (198) - similar pattern with constraints

INTERVIEW TIPS:
==============
1. Identify as generalized Fibonacci problem
2. Show progression from recursive to optimized solutions
3. Explain why we need 3 variables for space optimization
4. Compare with Fibonacci (2 variables vs 3 variables)
5. Mention matrix exponentiation for O(log n) solution
6. Handle edge cases (n = 0, 1, 2)
7. Discuss growth rate compared to Fibonacci
"""
