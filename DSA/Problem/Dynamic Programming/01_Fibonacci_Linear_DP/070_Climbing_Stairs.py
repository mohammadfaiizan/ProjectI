"""
LeetCode 70: Climbing Stairs
Difficulty: Easy
Category: Fibonacci & Linear DP

PROBLEM DESCRIPTION:
===================
You are climbing a staircase. It takes n steps to reach the top.
Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

Example 1:
Input: n = 2
Output: 2
Explanation: There are two ways to climb to the top.
1. 1 step + 1 step
2. 2 steps

Example 2:
Input: n = 3
Output: 3
Explanation: There are three ways to climb to the top.
1. 1 step + 1 step + 1 step
2. 1 step + 2 steps
3. 2 steps + 1 step

Constraints:
- 1 <= n <= 45
"""

def climbing_stairs_bruteforce(n):
    """
    BRUTE FORCE APPROACH:
    ====================
    Use recursion to explore all possible ways.
    At each step, we can either take 1 step or 2 steps.
    
    Time Complexity: O(2^n) - exponential due to overlapping subproblems
    Space Complexity: O(n) - recursion stack depth
    """
    if n <= 2:
        return n
    
    return climbing_stairs_bruteforce(n - 1) + climbing_stairs_bruteforce(n - 2)


def climbing_stairs_memoization(n):
    """
    DYNAMIC PROGRAMMING - TOP DOWN (MEMOIZATION):
    ============================================
    Use memoization to avoid recalculating same subproblems.
    
    Time Complexity: O(n) - each subproblem calculated once
    Space Complexity: O(n) - memoization table + recursion stack
    """
    memo = {}
    
    def dp(n):
        if n <= 2:
            return n
        
        if n in memo:
            return memo[n]
        
        memo[n] = dp(n - 1) + dp(n - 2)
        return memo[n]
    
    return dp(n)


def climbing_stairs_tabulation(n):
    """
    DYNAMIC PROGRAMMING - BOTTOM UP (TABULATION):
    =============================================
    Build solution from bottom up using iteration.
    
    Time Complexity: O(n) - single pass through all values
    Space Complexity: O(n) - DP table
    """
    if n <= 2:
        return n
    
    dp = [0] * (n + 1)
    dp[1] = 1
    dp[2] = 2
    
    for i in range(3, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    
    return dp[n]


def climbing_stairs_optimized(n):
    """
    SPACE OPTIMIZED DYNAMIC PROGRAMMING:
    ===================================
    Since we only need previous two values, use variables instead of array.
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(1) - constant space
    """
    if n <= 2:
        return n
    
    prev2 = 1  # ways to reach step 1
    prev1 = 2  # ways to reach step 2
    
    for i in range(3, n + 1):
        current = prev1 + prev2
        prev2 = prev1
        prev1 = current
    
    return prev1


def climbing_stairs_matrix_exponentiation(n):
    """
    MATRIX EXPONENTIATION APPROACH:
    ==============================
    For very large n, use matrix exponentiation to compute Fibonacci in O(log n).
    
    Time Complexity: O(log n) - matrix exponentiation
    Space Complexity: O(1) - constant space
    """
    if n <= 2:
        return n
    
    def matrix_multiply(A, B):
        return [
            [A[0][0] * B[0][0] + A[0][1] * B[1][0], A[0][0] * B[0][1] + A[0][1] * B[1][1]],
            [A[1][0] * B[0][0] + A[1][1] * B[1][0], A[1][0] * B[0][1] + A[1][1] * B[1][1]]
        ]
    
    def matrix_power(matrix, power):
        if power == 1:
            return matrix
        
        if power % 2 == 0:
            half = matrix_power(matrix, power // 2)
            return matrix_multiply(half, half)
        else:
            return matrix_multiply(matrix, matrix_power(matrix, power - 1))
    
    # Base matrix for Fibonacci: [[1, 1], [1, 0]]
    base_matrix = [[1, 1], [1, 0]]
    result_matrix = matrix_power(base_matrix, n - 1)
    
    # F(n) = result_matrix[0][0] * F(2) + result_matrix[0][1] * F(1)
    return result_matrix[0][0] * 2 + result_matrix[0][1] * 1


# Test cases
def test_climbing_stairs():
    """Test all implementations with various inputs"""
    test_cases = [1, 2, 3, 4, 5, 10, 20]
    
    print("Testing Climbing Stairs Solutions:")
    print("=" * 50)
    
    for n in test_cases:
        brute = climbing_stairs_bruteforce(n) if n <= 10 else "Skip (too slow)"
        memo = climbing_stairs_memoization(n)
        tab = climbing_stairs_tabulation(n)
        opt = climbing_stairs_optimized(n)
        matrix = climbing_stairs_matrix_exponentiation(n)
        
        print(f"n = {n:2d}: Brute: {str(brute):>12}, Memo: {memo:>6}, Tab: {tab:>6}, Opt: {opt:>6}, Matrix: {matrix:>6}")
    
    print("\nComplexity Analysis:")
    print("Brute Force:     Time: O(2^n),   Space: O(n)")
    print("Memoization:     Time: O(n),     Space: O(n)")
    print("Tabulation:      Time: O(n),     Space: O(n)")
    print("Optimized:       Time: O(n),     Space: O(1)")
    print("Matrix Exp:      Time: O(log n), Space: O(1)")


if __name__ == "__main__":
    test_climbing_stairs()


"""
PATTERN RECOGNITION:
==================
This is a classic Fibonacci sequence problem:
- F(n) = F(n-1) + F(n-2)
- Base cases: F(1) = 1, F(2) = 2

KEY INSIGHTS:
============
1. Each step can be reached from either (step-1) or (step-2)
2. Total ways = ways to reach (step-1) + ways to reach (step-2)
3. This creates the Fibonacci recurrence relation
4. Space optimization is possible since we only need last 2 values

VARIANTS TO PRACTICE:
====================
- Min Cost Climbing Stairs (746)
- House Robber (198) - similar pattern with constraints
- Tribonacci (1137) - 3-step variant
- Fibonacci Number (509) - direct implementation

INTERVIEW TIPS:
==============
1. Start with brute force to show understanding
2. Identify overlapping subproblems for DP
3. Always optimize space when possible
4. Mention matrix exponentiation for very large inputs
5. Discuss integer overflow for large n values
"""
