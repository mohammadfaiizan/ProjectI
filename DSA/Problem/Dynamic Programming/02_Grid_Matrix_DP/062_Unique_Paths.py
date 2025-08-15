"""
LeetCode 62: Unique Paths
Difficulty: Medium
Category: Grid/Matrix DP

PROBLEM DESCRIPTION:
===================
There is a robot on an m x n grid. The robot is initially located at the top-left corner 
(i.e., grid[0][0]). The robot tries to move to the bottom-right corner (i.e., grid[m-1][n-1]). 
The robot can only move either down or right at any point in time.

Given the two integers m and n, return the number of possible unique paths that the robot 
can take to reach the bottom-right corner.

Example 1:
Input: m = 3, n = 7
Output: 28

Example 2:
Input: m = 3, n = 2
Output: 3
Explanation: From the top-left corner, there are a total of 3 ways to reach the bottom-right corner:
1. Right -> Down -> Down
2. Down -> Down -> Right
3. Down -> Right -> Down

Constraints:
- 1 <= m, n <= 100
"""

def unique_paths_bruteforce(m, n):
    """
    BRUTE FORCE APPROACH - RECURSION:
    ================================
    Use recursion to explore all possible paths.
    At each cell, we can either go right or down.
    
    Time Complexity: O(2^(m+n)) - exponential due to overlapping subproblems
    Space Complexity: O(m+n) - recursion stack depth
    """
    def count_paths(row, col):
        # Base case: reached destination
        if row == m - 1 and col == n - 1:
            return 1
        
        # Out of bounds
        if row >= m or col >= n:
            return 0
        
        # Recursive calls: go right or down
        return count_paths(row + 1, col) + count_paths(row, col + 1)
    
    return count_paths(0, 0)


def unique_paths_memoization(m, n):
    """
    DYNAMIC PROGRAMMING - TOP DOWN (MEMOIZATION):
    ============================================
    Use memoization to avoid recalculating same subproblems.
    
    Time Complexity: O(m * n) - each subproblem calculated once
    Space Complexity: O(m * n) - memoization table + recursion stack
    """
    memo = {}
    
    def count_paths(row, col):
        if row == m - 1 and col == n - 1:
            return 1
        
        if row >= m or col >= n:
            return 0
        
        if (row, col) in memo:
            return memo[(row, col)]
        
        memo[(row, col)] = count_paths(row + 1, col) + count_paths(row, col + 1)
        return memo[(row, col)]
    
    return count_paths(0, 0)


def unique_paths_tabulation(m, n):
    """
    DYNAMIC PROGRAMMING - BOTTOM UP (TABULATION):
    =============================================
    Build solution from bottom up using 2D DP table.
    dp[i][j] = number of unique paths to reach cell (i, j)
    
    Time Complexity: O(m * n) - fill entire DP table
    Space Complexity: O(m * n) - 2D DP table
    """
    # Create DP table
    dp = [[0] * n for _ in range(m)]
    
    # Initialize first row and first column
    # There's only one way to reach any cell in first row or first column
    for i in range(m):
        dp[i][0] = 1
    for j in range(n):
        dp[0][j] = 1
    
    # Fill the DP table
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
    
    return dp[m - 1][n - 1]


def unique_paths_space_optimized(m, n):
    """
    SPACE OPTIMIZED DYNAMIC PROGRAMMING:
    ===================================
    Since we only need the previous row, use 1D array instead of 2D.
    
    Time Complexity: O(m * n) - same number of operations
    Space Complexity: O(n) - only store one row
    """
    # Use 1D array to represent current row
    dp = [1] * n  # Initialize with 1s (first row)
    
    # For each subsequent row
    for i in range(1, m):
        for j in range(1, n):
            # dp[j] represents current cell
            # dp[j-1] represents cell to the left
            # dp[j] (before update) represents cell above
            dp[j] = dp[j] + dp[j - 1]
    
    return dp[n - 1]


def unique_paths_combinatorics(m, n):
    """
    MATHEMATICAL APPROACH - COMBINATORICS:
    =====================================
    Total moves needed: (m-1) down + (n-1) right = (m+n-2) moves
    Choose (m-1) positions for down moves out of (m+n-2) total positions.
    This is C(m+n-2, m-1) = C(m+n-2, n-1)
    
    Time Complexity: O(min(m, n)) - computing combinations
    Space Complexity: O(1) - constant space
    """
    # Total moves needed
    total_moves = m + n - 2
    down_moves = m - 1
    
    # Calculate C(total_moves, down_moves)
    # Use the identity C(n, k) = C(n, n-k) to minimize computation
    k = min(down_moves, n - 1)
    
    result = 1
    for i in range(k):
        result = result * (total_moves - i) // (i + 1)
    
    return result


def unique_paths_iterative_combinatorics(m, n):
    """
    ITERATIVE COMBINATORICS WITH OVERFLOW PROTECTION:
    ================================================
    Calculate C(m+n-2, m-1) iteratively to avoid overflow.
    
    Time Complexity: O(min(m, n)) - computing combinations
    Space Complexity: O(1) - constant space
    """
    if m == 1 or n == 1:
        return 1
    
    # Calculate C(m+n-2, min(m-1, n-1))
    total = m + n - 2
    k = min(m - 1, n - 1)
    
    # Calculate C(total, k) = total! / (k! * (total-k)!)
    # Iteratively: C(total, k) = (total * (total-1) * ... * (total-k+1)) / (k!)
    numerator = 1
    denominator = 1
    
    for i in range(k):
        numerator *= (total - i)
        denominator *= (i + 1)
    
    return numerator // denominator


# Test cases
def test_unique_paths():
    """Test all implementations with various inputs"""
    test_cases = [
        (3, 7, 28),
        (3, 2, 3),
        (1, 1, 1),
        (1, 10, 1),
        (10, 1, 1),
        (3, 3, 6),
        (4, 4, 20),
        (5, 5, 70),
        (10, 10, 48620)
    ]
    
    print("Testing Unique Paths Solutions:")
    print("=" * 70)
    
    for i, (m, n, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: m = {m}, n = {n}")
        print(f"Expected: {expected}")
        
        # Test all approaches (skip brute force for large inputs)
        if m + n <= 10:
            brute = unique_paths_bruteforce(m, n)
            print(f"Brute Force:      {brute:>8} {'✓' if brute == expected else '✗'}")
        
        memo = unique_paths_memoization(m, n)
        tab = unique_paths_tabulation(m, n)
        space_opt = unique_paths_space_optimized(m, n)
        comb = unique_paths_combinatorics(m, n)
        iter_comb = unique_paths_iterative_combinatorics(m, n)
        
        print(f"Memoization:      {memo:>8} {'✓' if memo == expected else '✗'}")
        print(f"Tabulation:       {tab:>8} {'✓' if tab == expected else '✗'}")
        print(f"Space Optimized:  {space_opt:>8} {'✓' if space_opt == expected else '✗'}")
        print(f"Combinatorics:    {comb:>8} {'✓' if comb == expected else '✗'}")
        print(f"Iter Comb:        {iter_comb:>8} {'✓' if iter_comb == expected else '✗'}")
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Brute Force:      Time: O(2^(m+n)),    Space: O(m+n)")
    print("Memoization:      Time: O(m*n),        Space: O(m*n)")
    print("Tabulation:       Time: O(m*n),        Space: O(m*n)")
    print("Space Optimized:  Time: O(m*n),        Space: O(n)")
    print("Combinatorics:    Time: O(min(m,n)),   Space: O(1)")
    print("Iter Comb:        Time: O(min(m,n)),   Space: O(1)")


if __name__ == "__main__":
    test_unique_paths()


"""
PATTERN RECOGNITION:
==================
This is a classic 2D grid DP problem:
- Can only move right or down
- Count number of paths from top-left to bottom-right
- dp[i][j] = dp[i-1][j] + dp[i][j-1]

KEY INSIGHTS:
============
1. Each cell can be reached from the cell above or cell to the left
2. Number of paths to current cell = sum of paths to previous cells
3. First row and first column have only one path (boundary conditions)
4. This is equivalent to choosing positions for right/down moves

STATE DEFINITION:
================
dp[i][j] = number of unique paths to reach cell (i, j) from (0, 0)

RECURRENCE RELATION:
===================
dp[i][j] = dp[i-1][j] + dp[i][j-1]
Base cases: dp[0][j] = 1, dp[i][0] = 1

MATHEMATICAL INSIGHT:
====================
Total moves = (m-1) down + (n-1) right = (m+n-2)
Problem reduces to: Choose (m-1) positions out of (m+n-2) for down moves
Answer = C(m+n-2, m-1) = C(m+n-2, n-1)

SPACE OPTIMIZATION:
==================
Since we only need previous row, we can use O(n) space instead of O(m*n).

VARIANTS TO PRACTICE:
====================
- Unique Paths II (63) - with obstacles
- Minimum Path Sum (64) - find minimum cost path
- Dungeon Game (174) - reverse DP with health constraints
- Cherry Pickup (741) - bidirectional path optimization

INTERVIEW TIPS:
==============
1. Start with recursive solution to show understanding
2. Identify overlapping subproblems for DP optimization
3. Draw small examples to verify recurrence relation
4. Discuss space optimization using 1D array
5. Mention combinatorial solution for mathematical insight
6. Handle edge cases (m=1 or n=1)
"""
