"""
LeetCode 64: Minimum Path Sum
Difficulty: Medium
Category: Grid/Matrix DP

PROBLEM DESCRIPTION:
===================
Given a m x n grid filled with non-negative numbers, find a path from top left to bottom 
right, which minimizes the sum of all numbers along its path.

Note: You can only move either down or right at any point in time.

Example 1:
Input: grid = [[1,3,1],[1,5,1],[4,2,1]]
Output: 7
Explanation: Because the path 1 → 3 → 1 → 1 → 1 minimizes the sum.

Example 2:
Input: grid = [[1,2,3],[4,5,6]]
Output: 12

Constraints:
- m == grid.length
- n == grid[i].length
- 1 <= m, n <= 200
- 0 <= grid[i][j] <= 100
"""

def min_path_sum_bruteforce(grid):
    """
    BRUTE FORCE APPROACH - RECURSION:
    ================================
    Use recursion to explore all possible paths.
    At each cell, try going right or down and take minimum.
    
    Time Complexity: O(2^(m+n)) - exponential due to overlapping subproblems
    Space Complexity: O(m+n) - recursion stack depth
    """
    m, n = len(grid), len(grid[0])
    
    def min_path(row, col):
        # Base case: reached destination
        if row == m - 1 and col == n - 1:
            return grid[row][col]
        
        # Out of bounds
        if row >= m or col >= n:
            return float('inf')
        
        # Recursive calls: go right or down
        right = min_path(row, col + 1)
        down = min_path(row + 1, col)
        
        return grid[row][col] + min(right, down)
    
    return min_path(0, 0)


def min_path_sum_memoization(grid):
    """
    DYNAMIC PROGRAMMING - TOP DOWN (MEMOIZATION):
    ============================================
    Use memoization to avoid recalculating same subproblems.
    
    Time Complexity: O(m * n) - each subproblem calculated once
    Space Complexity: O(m * n) - memoization table + recursion stack
    """
    m, n = len(grid), len(grid[0])
    memo = {}
    
    def min_path(row, col):
        if row == m - 1 and col == n - 1:
            return grid[row][col]
        
        if row >= m or col >= n:
            return float('inf')
        
        if (row, col) in memo:
            return memo[(row, col)]
        
        right = min_path(row, col + 1)
        down = min_path(row + 1, col)
        
        memo[(row, col)] = grid[row][col] + min(right, down)
        return memo[(row, col)]
    
    return min_path(0, 0)


def min_path_sum_tabulation(grid):
    """
    DYNAMIC PROGRAMMING - BOTTOM UP (TABULATION):
    =============================================
    Build solution from bottom up using 2D DP table.
    dp[i][j] = minimum path sum to reach cell (i, j)
    
    Time Complexity: O(m * n) - fill entire DP table
    Space Complexity: O(m * n) - 2D DP table
    """
    m, n = len(grid), len(grid[0])
    
    # Create DP table and initialize with grid values
    dp = [[0] * n for _ in range(m)]
    dp[0][0] = grid[0][0]
    
    # Initialize first row
    for j in range(1, n):
        dp[0][j] = dp[0][j - 1] + grid[0][j]
    
    # Initialize first column
    for i in range(1, m):
        dp[i][0] = dp[i - 1][0] + grid[i][0]
    
    # Fill the DP table
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = grid[i][j] + min(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m - 1][n - 1]


def min_path_sum_space_optimized(grid):
    """
    SPACE OPTIMIZED DYNAMIC PROGRAMMING:
    ===================================
    Since we only need the previous row, use 1D array instead of 2D.
    
    Time Complexity: O(m * n) - same number of operations
    Space Complexity: O(n) - only store one row
    """
    m, n = len(grid), len(grid[0])
    
    # Initialize DP array with first row
    dp = [0] * n
    dp[0] = grid[0][0]
    
    # Fill first row
    for j in range(1, n):
        dp[j] = dp[j - 1] + grid[0][j]
    
    # Process remaining rows
    for i in range(1, m):
        # Update first element of current row
        dp[0] += grid[i][0]
        
        # Update remaining elements
        for j in range(1, n):
            dp[j] = grid[i][j] + min(dp[j], dp[j - 1])
    
    return dp[n - 1]


def min_path_sum_inplace(grid):
    """
    IN-PLACE DYNAMIC PROGRAMMING:
    ============================
    Modify the input grid itself to store DP values.
    Most space-efficient but modifies input.
    
    Time Complexity: O(m * n) - single pass through grid
    Space Complexity: O(1) - no extra space (excluding input)
    """
    m, n = len(grid), len(grid[0])
    
    # Initialize first row
    for j in range(1, n):
        grid[0][j] += grid[0][j - 1]
    
    # Initialize first column
    for i in range(1, m):
        grid[i][0] += grid[i - 1][0]
    
    # Fill the rest of the grid
    for i in range(1, m):
        for j in range(1, n):
            grid[i][j] += min(grid[i - 1][j], grid[i][j - 1])
    
    return grid[m - 1][n - 1]


def min_path_sum_with_path(grid):
    """
    DP WITH PATH RECONSTRUCTION:
    ===========================
    Find minimum path sum and reconstruct the actual path.
    
    Time Complexity: O(m * n) - DP + path reconstruction
    Space Complexity: O(m * n) - DP table + path storage
    """
    m, n = len(grid), len(grid[0])
    
    # Create DP table
    dp = [[0] * n for _ in range(m)]
    dp[0][0] = grid[0][0]
    
    # Initialize first row
    for j in range(1, n):
        dp[0][j] = dp[0][j - 1] + grid[0][j]
    
    # Initialize first column
    for i in range(1, m):
        dp[i][0] = dp[i - 1][0] + grid[i][0]
    
    # Fill DP table
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = grid[i][j] + min(dp[i - 1][j], dp[i][j - 1])
    
    # Reconstruct path
    path = []
    i, j = m - 1, n - 1
    
    while i > 0 or j > 0:
        path.append((i, j))
        
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            # Choose the direction that led to current minimum
            if dp[i - 1][j] < dp[i][j - 1]:
                i -= 1
            else:
                j -= 1
    
    path.append((0, 0))
    path.reverse()
    
    return dp[m - 1][n - 1], path


# Test cases
def test_min_path_sum():
    """Test all implementations with various inputs"""
    test_cases = [
        ([[1, 3, 1], [1, 5, 1], [4, 2, 1]], 7),
        ([[1, 2, 3], [4, 5, 6]], 12),
        ([[1]], 1),
        ([[1, 2], [3, 4]], 8),
        ([[5, 0, 1, 3], [2, 4, 2, 1], [1, 2, 3, 1]], 9),
        ([[1, 4, 8, 6, 2, 2, 1, 7], [4, 7, 3, 1, 4, 5, 5, 1]], 23)
    ]
    
    print("Testing Minimum Path Sum Solutions:")
    print("=" * 70)
    
    for i, (grid_input, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Grid: {grid_input}")
        print(f"Expected: {expected}")
        
        # Test all approaches (skip brute force for large inputs)
        if len(grid_input) * len(grid_input[0]) <= 9:
            grid_copy = [row[:] for row in grid_input]
            brute = min_path_sum_bruteforce(grid_copy)
            print(f"Brute Force:      {brute:>3} {'✓' if brute == expected else '✗'}")
        
        # Test other approaches with fresh copies
        grid_copy = [row[:] for row in grid_input]
        memo = min_path_sum_memoization(grid_copy)
        
        grid_copy = [row[:] for row in grid_input]
        tab = min_path_sum_tabulation(grid_copy)
        
        grid_copy = [row[:] for row in grid_input]
        space_opt = min_path_sum_space_optimized(grid_copy)
        
        grid_copy = [row[:] for row in grid_input]
        inplace = min_path_sum_inplace(grid_copy)
        
        grid_copy = [row[:] for row in grid_input]
        with_path, path = min_path_sum_with_path(grid_copy)
        
        print(f"Memoization:      {memo:>3} {'✓' if memo == expected else '✗'}")
        print(f"Tabulation:       {tab:>3} {'✓' if tab == expected else '✗'}")
        print(f"Space Optimized:  {space_opt:>3} {'✓' if space_opt == expected else '✗'}")
        print(f"In-place:         {inplace:>3} {'✓' if inplace == expected else '✗'}")
        print(f"With Path:        {with_path:>3} {'✓' if with_path == expected else '✗'}")
        
        # Show path for small grids
        if len(grid_input) <= 3 and len(grid_input[0]) <= 4:
            print(f"Optimal Path: {' -> '.join(map(str, path))}")
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Brute Force:      Time: O(2^(m+n)),    Space: O(m+n)")
    print("Memoization:      Time: O(m*n),        Space: O(m*n)")
    print("Tabulation:       Time: O(m*n),        Space: O(m*n)")
    print("Space Optimized:  Time: O(m*n),        Space: O(n)")
    print("In-place:         Time: O(m*n),        Space: O(1)")
    print("With Path:        Time: O(m*n),        Space: O(m*n)")


if __name__ == "__main__":
    test_min_path_sum()


"""
PATTERN RECOGNITION:
==================
This is a classic 2D grid optimization DP problem:
- Can only move right or down
- Find minimum sum path from top-left to bottom-right
- dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]

KEY INSIGHTS:
============
1. Each cell can be reached from the cell above or cell to the left
2. Minimum path sum to current cell = current value + minimum of previous cells
3. First row can only be reached from left, first column from above
4. Greedy approach won't work - need to consider all paths

STATE DEFINITION:
================
dp[i][j] = minimum path sum to reach cell (i, j) from (0, 0)

RECURRENCE RELATION:
===================
dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])
Base cases: 
- dp[0][0] = grid[0][0]
- dp[0][j] = dp[0][j-1] + grid[0][j]
- dp[i][0] = dp[i-1][0] + grid[i][0]

SPACE OPTIMIZATION:
==================
1. Use 1D array: O(n) space instead of O(m*n)
2. In-place modification: O(1) extra space

PATH RECONSTRUCTION:
===================
Work backwards from destination, choosing the direction that gave minimum value.

VARIANTS TO PRACTICE:
====================
- Unique Paths (62) - count paths instead of minimum sum
- Unique Paths II (63) - with obstacles
- Triangle (120) - variable width grid
- Dungeon Game (174) - maximize minimum health
- Cherry Pickup (741) - bidirectional optimization

INTERVIEW TIPS:
==============
1. Identify this as an optimization problem on 2D grid
2. Start with recursive solution, then add memoization
3. Show how to derive recurrence relation step by step
4. Discuss space optimization techniques
5. Mention path reconstruction if asked
6. Handle edge cases (single cell, single row/column)
7. Consider in-place modification vs preserving input
"""
