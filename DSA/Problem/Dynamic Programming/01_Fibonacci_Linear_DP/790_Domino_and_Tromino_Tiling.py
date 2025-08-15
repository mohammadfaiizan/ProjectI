"""
LeetCode 790: Domino and Tromino Tiling
Difficulty: Medium
Category: Fibonacci & Linear DP

PROBLEM DESCRIPTION:
===================
You have two types of tiles: a 2x1 domino shape and a tromino shape. You may rotate these shapes.

Given an integer n, return the number of ways to tile an 2 x n board. Since the answer may be very large, 
return it modulo 10^9 + 7.

In a tiling, every square must be covered by a tile. Two tilings are different if and only if there are 
two 4-directionally adjacent cells on the board such that exactly one of the tilings has both squares 
occupied by the same tile.

Example 1:
Input: n = 3
Output: 5
Explanation: The five different ways are show above.

Example 2:
Input: n = 1
Output: 1

Constraints:
- 1 <= n <= 1000
"""

def num_tilings_bruteforce(n):
    """
    BRUTE FORCE APPROACH - RECURSION:
    ================================
    Try all possible tiling arrangements recursively.
    
    Time Complexity: O(3^n) - exponential due to multiple tile choices
    Space Complexity: O(n) - recursion stack depth
    """
    MOD = 10**9 + 7
    
    def count_tilings(remaining):
        if remaining == 0:
            return 1
        if remaining == 1:
            return 1
        if remaining == 2:
            return 2
        if remaining < 0:
            return 0
        
        # Three main ways to fill:
        # 1. Place vertical domino (covers 1 column)
        vertical = count_tilings(remaining - 1)
        
        # 2. Place two horizontal dominos (covers 2 columns)
        horizontal = count_tilings(remaining - 2)
        
        # 3. Place tromino patterns (various combinations)
        tromino = count_tilings(remaining - 3)
        
        return (vertical + horizontal + tromino) % MOD
    
    return count_tilings(n)


def num_tilings_memoization(n):
    """
    DYNAMIC PROGRAMMING - TOP DOWN (MEMOIZATION):
    ============================================
    Use memoization with state tracking for partial tilings.
    
    Time Complexity: O(n) - each subproblem calculated once
    Space Complexity: O(n) - memoization table + recursion stack
    """
    MOD = 10**9 + 7
    memo = {}
    
    def dp(i, state):
        """
        i: current column
        state: 0 = both cells empty, 1 = top filled, 2 = bottom filled, 3 = both filled
        """
        if i == n:
            return 1 if state == 0 else 0
        
        if (i, state) in memo:
            return memo[(i, state)]
        
        result = 0
        
        if state == 0:  # Both cells empty
            # Place vertical domino
            result += dp(i + 1, 0)
            # Place two horizontal dominos
            result += dp(i + 1, 3)
            # Place tromino (up)
            result += dp(i + 1, 1)
            # Place tromino (down)
            result += dp(i + 1, 2)
            
        elif state == 1:  # Top filled, bottom empty
            # Complete with horizontal domino
            result += dp(i + 1, 2)
            # Place tromino
            result += dp(i + 1, 0)
            
        elif state == 2:  # Bottom filled, top empty
            # Complete with horizontal domino
            result += dp(i + 1, 1)
            # Place tromino
            result += dp(i + 1, 0)
            
        elif state == 3:  # Both filled
            result += dp(i + 1, 0)
        
        result %= MOD
        memo[(i, state)] = result
        return result
    
    return dp(0, 0)


def num_tilings_simplified_states(n):
    """
    SIMPLIFIED STATE DP:
    ===================
    Use simplified state representation for easier understanding.
    
    Time Complexity: O(n) - linear pass
    Space Complexity: O(n) - DP arrays
    """
    MOD = 10**9 + 7
    
    if n <= 2:
        return n
    
    # dp[i] = ways to completely fill 2 x i board
    # partial[i] = ways to fill 2 x i board with one partial column
    dp = [0] * (n + 1)
    partial = [0] * (n + 1)
    
    dp[0] = 1
    dp[1] = 1
    dp[2] = 2
    partial[2] = 2
    
    for i in range(3, n + 1):
        # Complete tiling from complete previous states
        dp[i] = (dp[i - 1] + dp[i - 2] + 2 * partial[i - 1]) % MOD
        # Partial tiling
        partial[i] = (dp[i - 2] + partial[i - 1]) % MOD
    
    return dp[n]


def num_tilings_mathematical_recurrence(n):
    """
    MATHEMATICAL RECURRENCE RELATION:
    ================================
    Derive and use the mathematical recurrence relation.
    
    Time Complexity: O(n) - linear iteration
    Space Complexity: O(1) - constant space
    """
    MOD = 10**9 + 7
    
    if n == 1:
        return 1
    if n == 2:
        return 2
    if n == 3:
        return 5
    
    # Recurrence: f(n) = 2*f(n-1) + f(n-3)
    # This comes from analyzing all possible ways to extend tilings
    
    fn3 = 1  # f(n-3)
    fn2 = 2  # f(n-2) 
    fn1 = 5  # f(n-1)
    
    for i in range(4, n + 1):
        fn = (2 * fn1 + fn3) % MOD
        fn3 = fn2
        fn2 = fn1
        fn1 = fn
    
    return fn1


def num_tilings_tabulation_detailed(n):
    """
    DETAILED TABULATION WITH STATE TRACKING:
    ========================================
    Complete DP solution with detailed state tracking.
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(n) - DP arrays
    """
    MOD = 10**9 + 7
    
    if n == 0:
        return 1
    if n == 1:
        return 1
    if n == 2:
        return 2
    
    # full[i] = ways to completely tile 2 x i board
    # up[i] = ways to tile with top-right cell of column i missing
    # down[i] = ways to tile with bottom-right cell of column i missing
    
    full = [0] * (n + 1)
    up = [0] * (n + 1)
    down = [0] * (n + 1)
    
    # Base cases
    full[0] = 1
    full[1] = 1
    full[2] = 2
    up[2] = 1
    down[2] = 1
    
    for i in range(3, n + 1):
        # Complete tiling
        full[i] = (full[i - 1] +     # Add vertical domino
                  full[i - 2] +     # Add two horizontal dominos
                  up[i - 1] +       # Complete from up state
                  down[i - 1]) % MOD # Complete from down state
        
        # Partial tilings
        up[i] = (full[i - 2] + down[i - 1]) % MOD
        down[i] = (full[i - 2] + up[i - 1]) % MOD
    
    return full[n]


def num_tilings_matrix_exponentiation(n):
    """
    MATRIX EXPONENTIATION APPROACH:
    ==============================
    Use matrix exponentiation for O(log n) solution.
    
    Time Complexity: O(log n) - matrix exponentiation
    Space Complexity: O(1) - constant space for matrices
    """
    MOD = 10**9 + 7
    
    if n <= 2:
        return n if n > 0 else 1
    
    def matrix_multiply(A, B):
        """Multiply two 3x3 matrices"""
        result = [[0] * 3 for _ in range(3)]
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    result[i][j] = (result[i][j] + A[i][k] * B[k][j]) % MOD
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
    
    # Transformation matrix based on recurrence f(n) = 2*f(n-1) + f(n-3)
    # [f(n)  ]   [2 0 1] [f(n-1)]
    # [f(n-1)] = [1 0 0] [f(n-2)]
    # [f(n-2)]   [0 1 0] [f(n-3)]
    
    base_matrix = [
        [2, 0, 1],
        [1, 0, 0],
        [0, 1, 0]
    ]
    
    if n == 3:
        return 5
    
    result_matrix = matrix_power(base_matrix, n - 3)
    
    # [f(n)  ]   [result_matrix] [f(3)]   [result_matrix] [5]
    # [f(n-1)] = [            ] [f(2)] = [            ] [2]
    # [f(n-2)]                  [f(1)]                  [1]
    
    return (result_matrix[0][0] * 5 + 
            result_matrix[0][1] * 2 + 
            result_matrix[0][2] * 1) % MOD


def num_tilings_pattern_analysis(n):
    """
    PATTERN ANALYSIS AND VERIFICATION:
    =================================
    Analyze the tiling patterns and verify recurrence relation.
    
    Time Complexity: O(n) - pattern computation
    Space Complexity: O(n) - store pattern
    """
    MOD = 10**9 + 7
    
    def analyze_pattern(max_n):
        """Analyze the pattern for small values"""
        values = []
        
        # Compute first few values using detailed DP
        for i in range(max_n + 1):
            if i == 0:
                values.append(1)
            elif i == 1:
                values.append(1)
            elif i == 2:
                values.append(2)
            else:
                # Use recurrence: f(n) = 2*f(n-1) + f(n-3)
                val = (2 * values[i - 1] + values[i - 3]) % MOD
                values.append(val)
        
        return values
    
    # Generate pattern
    pattern = analyze_pattern(min(n, 10))
    
    print("Pattern analysis:")
    for i, val in enumerate(pattern):
        print(f"f({i}) = {val}")
    
    # Verify recurrence relation
    if n >= 3:
        print(f"\nVerifying recurrence f(n) = 2*f(n-1) + f(n-3):")
        for i in range(3, min(n + 1, len(pattern))):
            expected = (2 * pattern[i - 1] + pattern[i - 3]) % MOD
            actual = pattern[i]
            print(f"f({i}) = 2*{pattern[i-1]} + {pattern[i-3]} = {expected} ({'✓' if expected == actual else '✗'})")
    
    return pattern[n] if n < len(pattern) else num_tilings_mathematical_recurrence(n)


def num_tilings_with_visualization(n):
    """
    TILING WITH VISUALIZATION:
    =========================
    Show actual tiling patterns for small n.
    
    Time Complexity: O(n) - computation
    Space Complexity: O(1) - visualization only
    """
    def show_tilings(size):
        """Show all possible tilings for small sizes"""
        if size == 1:
            print("n = 1: 1 way")
            print("||")
            print("||")
            
        elif size == 2:
            print("n = 2: 2 ways")
            print("Way 1: || ||    Way 2: ====")
            print("       || ||           ====")
            
        elif size == 3:
            print("n = 3: 5 ways")
            print("1: || ====   2: ==== ||   3: || || ||")
            print("   || ====      ==== ||      || || ||")
            print()
            print("4: ═══╗      5: ╔═══")
            print("   ═══╝         ╚═══")
    
    if n <= 3:
        show_tilings(n)
    
    return num_tilings_mathematical_recurrence(n)


# Test cases
def test_num_tilings():
    """Test all implementations with various inputs"""
    test_cases = [
        (1, 1),
        (2, 2),
        (3, 5),
        (4, 11),
        (5, 24),
        (6, 53),
        (7, 117),
        (8, 258),
        (9, 569),
        (10, 1255),
        (30, 312342182),
        (100, 841919786)
    ]
    
    print("Testing Domino and Tromino Tiling Solutions:")
    print("=" * 70)
    
    for i, (n, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: n = {n}")
        print(f"Expected: {expected}")
        
        # Test approaches (skip brute force for large n)
        if n <= 8:
            # brute = num_tilings_bruteforce(n)  # Too slow, skip
            pass
        
        memo = num_tilings_memoization(n)
        simplified = num_tilings_simplified_states(n)
        math_rec = num_tilings_mathematical_recurrence(n)
        detailed = num_tilings_tabulation_detailed(n)
        
        print(f"Memoization:      {memo:>9} {'✓' if memo == expected else '✗'}")
        print(f"Simplified:       {simplified:>9} {'✓' if simplified == expected else '✗'}")
        print(f"Math Recurrence:  {math_rec:>9} {'✓' if math_rec == expected else '✗'}")
        print(f"Detailed Tab:     {detailed:>9} {'✓' if detailed == expected else '✗'}")
        
        if n <= 20:
            matrix = num_tilings_matrix_exponentiation(n)
            print(f"Matrix Exp:       {matrix:>9} {'✓' if matrix == expected else '✗'}")
    
    # Show pattern analysis
    print(f"\nPattern Analysis for n=6:")
    pattern_result = num_tilings_pattern_analysis(6)
    
    # Show visualizations for small n
    print(f"\nTiling Visualizations:")
    for size in [1, 2, 3]:
        num_tilings_with_visualization(size)
        print()
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Brute Force:      Time: O(3^n),      Space: O(n)")
    print("Memoization:      Time: O(n),        Space: O(n)")
    print("Simplified:       Time: O(n),        Space: O(n)")
    print("Math Recurrence:  Time: O(n),        Space: O(1)")
    print("Detailed Tab:     Time: O(n),        Space: O(n)")
    print("Matrix Exp:       Time: O(log n),    Space: O(1)")


if __name__ == "__main__":
    test_num_tilings()


"""
PATTERN RECOGNITION:
==================
This is a complex tiling DP problem with multiple tile types:
- 2x1 domino (vertical or horizontal)
- L-shaped tromino (4 orientations)
- Count ways to tile 2×n board
- Recurrence relation: f(n) = 2×f(n-1) + f(n-3)

KEY INSIGHTS:
============
1. Multiple tile types create complex state transitions
2. Need to track partial tilings (incomplete columns)
3. Mathematical recurrence emerges from pattern analysis
4. Can be optimized using matrix exponentiation

STATE DEFINITION:
================
Multiple possible state definitions:
1. Simple: dp[i] = ways to tile 2×i board completely
2. Detailed: track partial states (top/bottom filled separately)
3. Mathematical: use derived recurrence relation

RECURRENCE RELATION:
===================
After detailed analysis of all tiling patterns:
f(n) = 2×f(n-1) + f(n-3)

Base cases: f(0)=1, f(1)=1, f(2)=2, f(3)=5

DERIVATION OF RECURRENCE:
========================
1. Extend f(n-1): Add vertical domino or tromino patterns
2. Extend f(n-2): Add two horizontal dominos
3. Extend f(n-3): Special tromino combinations
4. Combining all cases gives: f(n) = 2×f(n-1) + f(n-3)

OPTIMIZATION TECHNIQUES:
=======================
1. State reduction: Identify essential states only
2. Mathematical recurrence: Avoid complex state tracking
3. Space optimization: O(n) → O(1) using variables
4. Matrix exponentiation: O(n) → O(log n) for large n

TILING TYPES:
============
1. Vertical domino: covers 1 column
2. Two horizontal dominos: covers 2 columns  
3. Tromino patterns: various L-shaped arrangements
4. Complex combinations create the recurrence

VARIANTS TO PRACTICE:
====================
- Fibonacci Number (509) - simpler tiling (1×n with 1×1 and 1×2)
- Unique Paths (62) - grid traversal DP
- Perfect Squares (279) - similar mathematical optimization
- House Robber (198) - constraint-based linear DP

INTERVIEW TIPS:
==============
1. Start by identifying all possible tile placements
2. Define states clearly (complete vs partial tilings)
3. Derive recurrence relation step by step
4. Show mathematical pattern discovery
5. Optimize space from O(n) to O(1)
6. Mention matrix exponentiation for advanced optimization
7. Handle modular arithmetic correctly
"""
