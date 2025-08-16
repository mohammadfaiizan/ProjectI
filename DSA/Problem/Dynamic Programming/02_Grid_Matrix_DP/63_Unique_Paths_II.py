"""
LeetCode 63: Unique Paths II
Difficulty: Medium
Category: Grid/Matrix DP

PROBLEM DESCRIPTION:
===================
You are given an m x n integer array grid. There is a robot initially located at the top-left corner 
(i.e., grid[0][0]). The robot tries to move to the bottom-right corner (i.e., grid[m - 1][n - 1]). 
The robot can only move either down or right at any point in time.

An obstacle and space are marked as 1 and 0 respectively in grid. A path that the robot takes cannot 
include any square that is an obstacle.

Return the number of unique paths that the robot can take to reach the bottom-right corner.

The testcases are generated so that the answer will be at most 2 * 10^9.

Example 1:
Input: obstacleGrid = [[0,0,0],[0,1,0],[0,0,0]]
Output: 2
Explanation: There is one obstacle in the middle of the 3x3 grid.
There are two ways to reach the bottom-right corner:
1. Right -> Right -> Down -> Down
2. Down -> Down -> Right -> Right

Example 2:
Input: obstacleGrid = [[0,1],[0,0]]
Output: 1

Constraints:
- m == obstacleGrid.length
- n == obstacleGrid[i].length
- 1 <= m, n <= 100
- obstacleGrid[i][j] is 0 or 1.
"""

def unique_paths_with_obstacles_brute_force(obstacleGrid):
    """
    BRUTE FORCE APPROACH:
    ====================
    Try all possible paths from top-left to bottom-right.
    
    Time Complexity: O(2^(m+n)) - exponential paths
    Space Complexity: O(m+n) - recursion depth
    """
    if not obstacleGrid or not obstacleGrid[0] or obstacleGrid[0][0] == 1:
        return 0
    
    m, n = len(obstacleGrid), len(obstacleGrid[0])
    
    def dfs(row, col):
        # Base cases
        if row >= m or col >= n or obstacleGrid[row][col] == 1:
            return 0
        
        if row == m - 1 and col == n - 1:
            return 1
        
        # Try going down and right
        down = dfs(row + 1, col)
        right = dfs(row, col + 1)
        
        return down + right
    
    return dfs(0, 0)


def unique_paths_with_obstacles_memoization(obstacleGrid):
    """
    MEMOIZATION APPROACH:
    ====================
    Use memoization to avoid recomputing same subproblems.
    
    Time Complexity: O(m*n) - each cell computed once
    Space Complexity: O(m*n) - memoization table
    """
    if not obstacleGrid or not obstacleGrid[0] or obstacleGrid[0][0] == 1:
        return 0
    
    m, n = len(obstacleGrid), len(obstacleGrid[0])
    memo = {}
    
    def dfs(row, col):
        # Base cases
        if row >= m or col >= n or obstacleGrid[row][col] == 1:
            return 0
        
        if row == m - 1 and col == n - 1:
            return 1
        
        if (row, col) in memo:
            return memo[(row, col)]
        
        # Try going down and right
        down = dfs(row + 1, col)
        right = dfs(row, col + 1)
        
        result = down + right
        memo[(row, col)] = result
        return result
    
    return dfs(0, 0)


def unique_paths_with_obstacles_dp(obstacleGrid):
    """
    DYNAMIC PROGRAMMING APPROACH:
    ============================
    Build solution bottom-up using 2D DP table.
    
    Time Complexity: O(m*n) - process each cell once
    Space Complexity: O(m*n) - DP table
    """
    if not obstacleGrid or not obstacleGrid[0] or obstacleGrid[0][0] == 1:
        return 0
    
    m, n = len(obstacleGrid), len(obstacleGrid[0])
    
    # DP table
    dp = [[0] * n for _ in range(m)]
    
    # Initialize starting position
    dp[0][0] = 1
    
    # Fill first row
    for j in range(1, n):
        if obstacleGrid[0][j] == 0:
            dp[0][j] = dp[0][j-1]
        else:
            dp[0][j] = 0
    
    # Fill first column
    for i in range(1, m):
        if obstacleGrid[i][0] == 0:
            dp[i][0] = dp[i-1][0]
        else:
            dp[i][0] = 0
    
    # Fill rest of the table
    for i in range(1, m):
        for j in range(1, n):
            if obstacleGrid[i][j] == 0:
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
            else:
                dp[i][j] = 0
    
    return dp[m-1][n-1]


def unique_paths_with_obstacles_space_optimized(obstacleGrid):
    """
    SPACE OPTIMIZED APPROACH:
    =========================
    Use only O(n) space by processing row by row.
    
    Time Complexity: O(m*n) - process each cell once
    Space Complexity: O(n) - single row array
    """
    if not obstacleGrid or not obstacleGrid[0] or obstacleGrid[0][0] == 1:
        return 0
    
    m, n = len(obstacleGrid), len(obstacleGrid[0])
    
    # Use single array to represent current row
    dp = [0] * n
    dp[0] = 1
    
    for i in range(m):
        for j in range(n):
            if obstacleGrid[i][j] == 1:
                dp[j] = 0
            elif j > 0:
                dp[j] += dp[j-1]
    
    return dp[n-1]


def unique_paths_with_obstacles_in_place(obstacleGrid):
    """
    IN-PLACE APPROACH:
    =================
    Modify the input grid itself to save space.
    
    Time Complexity: O(m*n) - process each cell once
    Space Complexity: O(1) - no extra space
    """
    if not obstacleGrid or not obstacleGrid[0] or obstacleGrid[0][0] == 1:
        return 0
    
    m, n = len(obstacleGrid), len(obstacleGrid[0])
    
    # Mark obstacles as -1 to distinguish from 0 paths
    for i in range(m):
        for j in range(n):
            if obstacleGrid[i][j] == 1:
                obstacleGrid[i][j] = -1
    
    # Initialize starting position
    obstacleGrid[0][0] = 1
    
    # Fill first row
    for j in range(1, n):
        if obstacleGrid[0][j] != -1:
            obstacleGrid[0][j] = obstacleGrid[0][j-1] if obstacleGrid[0][j-1] != -1 else 0
        else:
            obstacleGrid[0][j] = 0
    
    # Fill first column
    for i in range(1, m):
        if obstacleGrid[i][0] != -1:
            obstacleGrid[i][0] = obstacleGrid[i-1][0] if obstacleGrid[i-1][0] != -1 else 0
        else:
            obstacleGrid[i][0] = 0
    
    # Fill rest of the grid
    for i in range(1, m):
        for j in range(1, n):
            if obstacleGrid[i][j] != -1:
                left = obstacleGrid[i][j-1] if obstacleGrid[i][j-1] != -1 else 0
                top = obstacleGrid[i-1][j] if obstacleGrid[i-1][j] != -1 else 0
                obstacleGrid[i][j] = left + top
            else:
                obstacleGrid[i][j] = 0
    
    return obstacleGrid[m-1][n-1]


def unique_paths_with_obstacles_with_paths(obstacleGrid):
    """
    FIND ACTUAL PATHS:
    =================
    Return count and actual paths from start to end.
    
    Time Complexity: O(m*n + k*path_length) - DP + path enumeration
    Space Complexity: O(m*n + k*path_length) - DP table + paths storage
    """
    if not obstacleGrid or not obstacleGrid[0] or obstacleGrid[0][0] == 1:
        return 0, []
    
    m, n = len(obstacleGrid), len(obstacleGrid[0])
    
    # First get the count using standard DP
    count = unique_paths_with_obstacles_dp([row[:] for row in obstacleGrid])
    
    if count == 0:
        return 0, []
    
    # Find all paths using backtracking
    all_paths = []
    
    def find_paths(row, col, current_path):
        # Add current position to path
        current_path.append((row, col))
        
        # If reached destination
        if row == m - 1 and col == n - 1:
            all_paths.append(current_path[:])
            current_path.pop()
            return
        
        # Try going down
        if row + 1 < m and obstacleGrid[row + 1][col] == 0:
            find_paths(row + 1, col, current_path)
        
        # Try going right
        if col + 1 < n and obstacleGrid[row][col + 1] == 0:
            find_paths(row, col + 1, current_path)
        
        current_path.pop()
    
    find_paths(0, 0, [])
    
    return count, all_paths


def unique_paths_with_obstacles_analysis(obstacleGrid):
    """
    DETAILED ANALYSIS:
    =================
    Show step-by-step DP computation with obstacle handling.
    
    Time Complexity: O(m*n) - analysis computation
    Space Complexity: O(m*n) - temporary tables
    """
    if not obstacleGrid or not obstacleGrid[0]:
        return 0
    
    m, n = len(obstacleGrid), len(obstacleGrid[0])
    
    print("Obstacle Grid:")
    for i, row in enumerate(obstacleGrid):
        print(f"  Row {i}: {row}")
    
    if obstacleGrid[0][0] == 1:
        print("Starting position blocked! No paths possible.")
        return 0
    
    print(f"\nDP Computation:")
    
    # Create DP table for visualization
    dp = [[0] * n for _ in range(m)]
    dp[0][0] = 1
    
    print(f"Initialize start: dp[0][0] = 1")
    
    # Fill first row
    print(f"\nFilling first row:")
    for j in range(1, n):
        if obstacleGrid[0][j] == 0:
            dp[0][j] = dp[0][j-1]
            print(f"  dp[0][{j}] = dp[0][{j-1}] = {dp[0][j]}")
        else:
            dp[0][j] = 0
            print(f"  dp[0][{j}] = 0 (obstacle)")
    
    # Fill first column
    print(f"\nFilling first column:")
    for i in range(1, m):
        if obstacleGrid[i][0] == 0:
            dp[i][0] = dp[i-1][0]
            print(f"  dp[{i}][0] = dp[{i-1}][0] = {dp[i][0]}")
        else:
            dp[i][0] = 0
            print(f"  dp[{i}][0] = 0 (obstacle)")
    
    # Fill rest of the table
    print(f"\nFilling rest of grid:")
    for i in range(1, m):
        for j in range(1, n):
            if obstacleGrid[i][j] == 0:
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
                print(f"  dp[{i}][{j}] = dp[{i-1}][{j}] + dp[{i}][{j-1}] = {dp[i-1][j]} + {dp[i][j-1]} = {dp[i][j]}")
            else:
                dp[i][j] = 0
                print(f"  dp[{i}][{j}] = 0 (obstacle)")
    
    print(f"\nFinal DP table:")
    for i, row in enumerate(dp):
        print(f"  Row {i}: {row}")
    
    result = dp[m-1][n-1]
    print(f"\nTotal unique paths: {result}")
    
    return result


def unique_paths_with_obstacles_edge_cases():
    """
    TEST EDGE CASES:
    ===============
    Test various edge cases and special scenarios.
    """
    test_cases = [
        # Basic cases
        ([[0,0,0],[0,1,0],[0,0,0]], 2, "Standard case with obstacle"),
        ([[0,1],[0,0]], 1, "Small grid with obstacle"),
        
        # Edge cases
        ([[1]], 0, "Single cell blocked"),
        ([[0]], 1, "Single cell open"),
        ([[1,0],[0,0]], 0, "Start blocked"),
        ([[0,0],[0,1]], 0, "End blocked"),
        
        # Blocked paths
        ([[0,0],[1,0]], 1, "Partial block"),
        ([[0,1,0],[0,1,0],[0,0,0]], 1, "Wall in middle"),
        ([[1,1,1],[1,1,1],[1,1,0]], 0, "Almost all blocked"),
        
        # No obstacles
        ([[0,0,0],[0,0,0],[0,0,0]], 6, "No obstacles"),
        ([[0,0],[0,0]], 2, "Small no obstacles"),
        
        # Complex patterns
        ([[0,0,1,0],[0,1,0,0],[1,0,0,0],[0,0,0,0]], 4, "Zigzag obstacles")
    ]
    
    print("Testing Edge Cases:")
    print("=" * 60)
    
    for i, (grid, expected, description) in enumerate(test_cases):
        print(f"\nTest {i+1}: {description}")
        print(f"Grid: {grid}")
        print(f"Expected: {expected}")
        
        result = unique_paths_with_obstacles_dp([row[:] for row in grid])
        print(f"Result: {result} {'✓' if result == expected else '✗'}")
        
        if len(grid) <= 3 and len(grid[0]) <= 3:
            count, paths = unique_paths_with_obstacles_with_paths([row[:] for row in grid])
            if paths:
                print(f"Sample paths: {paths[:2]}{'...' if len(paths) > 2 else ''}")


# Test cases
def test_unique_paths_with_obstacles():
    """Test all implementations with various inputs"""
    test_cases = [
        ([[0,0,0],[0,1,0],[0,0,0]], 2),
        ([[0,1],[0,0]], 1),
        ([[1]], 0),
        ([[0]], 1),
        ([[0,0],[1,0]], 1),
        ([[0,0],[0,0]], 2),
        ([[0,1,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,0]], 4),
        ([[1,0]], 0),
        ([[0,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,0]], 7)
    ]
    
    print("Testing Unique Paths with Obstacles Solutions:")
    print("=" * 70)
    
    for i, (obstacleGrid, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Grid: {obstacleGrid}")
        print(f"Expected: {expected}")
        
        # Test approaches (skip expensive ones for large inputs)
        if len(obstacleGrid) <= 4 and len(obstacleGrid[0]) <= 4:
            try:
                brute = unique_paths_with_obstacles_brute_force([row[:] for row in obstacleGrid])
                print(f"Brute Force:      {brute:>5} {'✓' if brute == expected else '✗'}")
            except:
                print(f"Brute Force:      Timeout")
        
        memo = unique_paths_with_obstacles_memoization([row[:] for row in obstacleGrid])
        dp_result = unique_paths_with_obstacles_dp([row[:] for row in obstacleGrid])
        space_opt = unique_paths_with_obstacles_space_optimized([row[:] for row in obstacleGrid])
        
        print(f"Memoization:      {memo:>5} {'✓' if memo == expected else '✗'}")
        print(f"DP (2D):          {dp_result:>5} {'✓' if dp_result == expected else '✗'}")
        print(f"Space Optimized:  {space_opt:>5} {'✓' if space_opt == expected else '✗'}")
        
        # Test in-place (modifies input)
        grid_copy = [row[:] for row in obstacleGrid]
        in_place = unique_paths_with_obstacles_in_place(grid_copy)
        print(f"In-place:         {in_place:>5} {'✓' if in_place == expected else '✗'}")
        
        # Show paths for small cases
        if len(obstacleGrid) <= 3 and len(obstacleGrid[0]) <= 3 and expected > 0:
            count, paths = unique_paths_with_obstacles_with_paths([row[:] for row in obstacleGrid])
            if paths:
                print(f"Sample paths: {paths[:2]}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    unique_paths_with_obstacles_analysis([[0,0,0],[0,1,0],[0,0,0]])
    
    # Edge cases testing
    print(f"\n" + "=" * 70)
    unique_paths_with_obstacles_edge_cases()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. OBSTACLE HANDLING: Treat obstacles as 0 paths")
    print("2. BOUNDARY CONDITIONS: Start/end obstacles make result 0")
    print("3. DP MODIFICATION: Standard unique paths + obstacle checks")
    print("4. SPACE OPTIMIZATION: Same techniques as original problem")
    print("5. EDGE CASES: Single cell, blocked start/end, no obstacles")
    
    print("\n" + "=" * 70)
    print("Algorithm Comparison:")
    print("Brute Force:      Try all possible paths")
    print("Memoization:      Top-down with caching")
    print("DP (2D):          Bottom-up with 2D table")
    print("Space Optimized:  Row-by-row processing")
    print("In-place:         Modify input grid")
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Brute Force:      Time: O(2^(m+n)), Space: O(m+n)")
    print("Memoization:      Time: O(m*n),     Space: O(m*n)")
    print("DP (2D):          Time: O(m*n),     Space: O(m*n)")
    print("Space Optimized:  Time: O(m*n),     Space: O(n)")
    print("In-place:         Time: O(m*n),     Space: O(1)")


if __name__ == "__main__":
    test_unique_paths_with_obstacles()


"""
PATTERN RECOGNITION:
==================
Unique Paths II is the classic Unique Paths problem with obstacles:
- Same movement rules: only right and down
- Additional constraint: cannot pass through obstacles (marked as 1)
- Core DP logic remains the same with obstacle handling
- Excellent example of problem variation and constraint addition

KEY INSIGHT - OBSTACLE INTEGRATION:
==================================
**Standard Unique Paths**: dp[i][j] = dp[i-1][j] + dp[i][j-1]

**With Obstacles**: 
```python
if obstacle[i][j] == 1:
    dp[i][j] = 0  # No paths through obstacles
else:
    dp[i][j] = dp[i-1][j] + dp[i][j-1]
```

**Critical Cases**:
- Start position blocked → return 0
- End position blocked → return 0  
- Intermediate obstacles → set paths = 0

ALGORITHM APPROACHES:
====================

1. **2D DP (Standard)**: O(m×n) time, O(m×n) space
   - Build complete DP table
   - Handle obstacles by setting dp[i][j] = 0
   - Most straightforward approach

2. **Space Optimized**: O(m×n) time, O(n) space
   - Process row by row
   - Update single array in-place
   - Optimal space complexity

3. **In-Place**: O(m×n) time, O(1) space
   - Modify input grid directly
   - Use obstacles as markers
   - Ultimate space optimization

4. **Memoization**: O(m×n) time, O(m×n) space
   - Top-down recursive approach
   - Natural handling of obstacles

OBSTACLE HANDLING STRATEGIES:
============================

**Method 1 - Direct Check**:
```python
if obstacleGrid[i][j] == 1:
    dp[i][j] = 0
else:
    dp[i][j] = dp[i-1][j] + dp[i][j-1]
```

**Method 2 - Preprocessing**:
```python
# Mark obstacles as -1, then process normally
for i in range(m):
    for j in range(n):
        if obstacleGrid[i][j] == 1:
            obstacleGrid[i][j] = -1
```

**Method 3 - Conditional Addition**:
```python
dp[i][j] = 0
if obstacleGrid[i][j] == 0:
    if i > 0 and obstacleGrid[i-1][j] == 0:
        dp[i][j] += dp[i-1][j]
    if j > 0 and obstacleGrid[i][j-1] == 0:
        dp[i][j] += dp[i][j-1]
```

EDGE CASE ANALYSIS:
==================

**Critical Edge Cases**:
1. **Start blocked**: obstacleGrid[0][0] == 1 → return 0
2. **End blocked**: obstacleGrid[m-1][n-1] == 1 → return 0
3. **Single cell**: Handle 1×1 grids specially
4. **Complete blockage**: No path possible
5. **No obstacles**: Reduces to original Unique Paths

**Boundary Initialization**:
- First row: Stop at first obstacle, rest are 0
- First column: Stop at first obstacle, rest are 0
- Handle obstacles in boundaries carefully

SPACE OPTIMIZATION DETAILS:
==========================

**1D Array Approach**:
```python
dp = [0] * n
dp[0] = 1 if obstacleGrid[0][0] == 0 else 0

for i in range(m):
    for j in range(n):
        if obstacleGrid[i][j] == 1:
            dp[j] = 0
        elif j > 0:
            dp[j] += dp[j-1]
```

**Key insight**: Process left-to-right, carry forward valid paths

MATHEMATICAL PROPERTIES:
=======================
- **Without obstacles**: C(m+n-2, m-1) paths
- **With obstacles**: No closed-form solution
- **Complexity**: Still O(m×n) due to obstacle dependencies
- **Optimization**: Cannot improve time complexity

PATH RECONSTRUCTION:
===================
To find actual paths:
1. Use backtracking from start to end
2. At each position, choose valid next moves
3. Filter out paths through obstacles
4. Count matches DP result

APPLICATIONS:
============
1. **Robot Navigation**: Autonomous vehicle path planning
2. **Game AI**: Character movement in grid-based games
3. **Network Routing**: Packet routing with failed nodes
4. **Maze Solving**: Count all solutions in maze

VARIANTS TO PRACTICE:
====================
- Unique Paths (62) - original problem without obstacles
- Minimum Path Sum (64) - minimize cost instead of count paths
- Dungeon Game (174) - similar grid traversal with health
- Unique Paths III (980) - visit all empty squares

INTERVIEW TIPS:
==============
1. **Start with base case**: Recognize as Unique Paths variation
2. **Handle obstacles early**: Check for blocked start/end
3. **Show DP recurrence**: How obstacles affect transitions
4. **Demonstrate optimizations**: 2D → 1D → in-place
5. **Consider edge cases**: Single cell, all blocked, no obstacles
6. **Discuss alternatives**: Top-down vs bottom-up
7. **Path enumeration**: How to find actual paths
8. **Real applications**: Robot navigation, game AI
9. **Complexity analysis**: Why O(m×n) is necessary
10. **Follow-up questions**: What if diagonal moves allowed?

OPTIMIZATION TECHNIQUES:
=======================
1. **Early termination**: If start/end blocked
2. **Sparse obstacle handling**: Skip empty regions
3. **Memory access optimization**: Cache-friendly traversal
4. **Parallel computation**: Independent subproblems

The problem beautifully demonstrates how constraints (obstacles) 
modify classic DP problems while preserving core algorithmic structure.
"""
