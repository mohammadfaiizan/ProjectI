"""
LeetCode 120: Triangle
Difficulty: Medium
Category: Grid/Matrix DP

PROBLEM DESCRIPTION:
===================
Given a triangle array, return the minimum path sum from top to bottom.

For each step, you may move to an adjacent number of the row below. More formally, if you are on index i 
on the current row, you may move to either index i or index i + 1 on the next row.

Example 1:
Input: triangle = [[2],[3,4],[6,5,7],[4,1,8,3]]
Output: 11
Explanation: The triangle looks like:
   2
  3 4
 6 5 7
4 1 8 3
The minimum path sum from top to bottom is 11 (i.e., 2 + 3 + 5 + 1 = 11).

Example 2:
Input: triangle = [[-10]]
Output: -10

Constraints:
- 1 <= triangle.length <= 200
- triangle[0].length == 1
- triangle[i].length == triangle[i - 1].length + 1
- -10^4 <= triangle[i][j] <= 10^4

Follow up: Could you do this using only O(n) extra space, where n is the total number of rows in the triangle?
"""

def minimum_total_brute_force(triangle):
    """
    BRUTE FORCE APPROACH:
    ====================
    Try all possible paths from top to bottom.
    
    Time Complexity: O(2^n) - exponential paths
    Space Complexity: O(n) - recursion depth
    """
    def dfs(row, col):
        if row >= len(triangle):
            return 0
        
        current = triangle[row][col]
        
        # Try both adjacent positions in next row
        left = dfs(row + 1, col)
        right = dfs(row + 1, col + 1)
        
        return current + min(left, right)
    
    return dfs(0, 0)


def minimum_total_memoization(triangle):
    """
    MEMOIZATION APPROACH:
    ====================
    Use memoization to avoid recomputing same subproblems.
    
    Time Complexity: O(n^2) - n^2 states to compute
    Space Complexity: O(n^2) - memoization table
    """
    memo = {}
    
    def dfs(row, col):
        if row >= len(triangle):
            return 0
        
        if (row, col) in memo:
            return memo[(row, col)]
        
        current = triangle[row][col]
        
        # Try both adjacent positions in next row
        left = dfs(row + 1, col)
        right = dfs(row + 1, col + 1)
        
        result = current + min(left, right)
        memo[(row, col)] = result
        return result
    
    return dfs(0, 0)


def minimum_total_tabulation(triangle):
    """
    TABULATION APPROACH (BOTTOM-UP):
    ================================
    Build solution from bottom to top.
    
    Time Complexity: O(n^2) - process all elements
    Space Complexity: O(n^2) - DP table
    """
    n = len(triangle)
    
    # Create DP table
    dp = [[0] * len(triangle[i]) for i in range(n)]
    
    # Initialize last row
    for j in range(len(triangle[n-1])):
        dp[n-1][j] = triangle[n-1][j]
    
    # Fill from bottom to top
    for i in range(n - 2, -1, -1):
        for j in range(len(triangle[i])):
            dp[i][j] = triangle[i][j] + min(dp[i+1][j], dp[i+1][j+1])
    
    return dp[0][0]


def minimum_total_space_optimized(triangle):
    """
    SPACE OPTIMIZED APPROACH:
    =========================
    Use only O(n) space by reusing the last row.
    
    Time Complexity: O(n^2) - process all elements
    Space Complexity: O(n) - single array
    """
    n = len(triangle)
    
    # Use the last row as starting point
    dp = triangle[-1][:]
    
    # Process from second last row to top
    for i in range(n - 2, -1, -1):
        for j in range(len(triangle[i])):
            dp[j] = triangle[i][j] + min(dp[j], dp[j + 1])
    
    return dp[0]


def minimum_total_in_place(triangle):
    """
    IN-PLACE APPROACH:
    =================
    Modify the triangle itself to save space.
    
    Time Complexity: O(n^2) - process all elements
    Space Complexity: O(1) - no extra space
    """
    n = len(triangle)
    
    # Process from second last row to top
    for i in range(n - 2, -1, -1):
        for j in range(len(triangle[i])):
            triangle[i][j] += min(triangle[i+1][j], triangle[i+1][j+1])
    
    return triangle[0][0]


def minimum_total_with_path(triangle):
    """
    FIND ACTUAL MINIMUM PATH:
    =========================
    Return both minimum sum and the actual path.
    
    Time Complexity: O(n^2) - DP + path reconstruction
    Space Complexity: O(n^2) - DP table + path tracking
    """
    n = len(triangle)
    
    # DP table for minimum sums
    dp = [[float('inf')] * len(triangle[i]) for i in range(n)]
    # Parent tracking for path reconstruction
    parent = [[-1] * len(triangle[i]) for i in range(n)]
    
    # Initialize first element
    dp[0][0] = triangle[0][0]
    
    # Fill DP table top to bottom
    for i in range(n - 1):
        for j in range(len(triangle[i])):
            if dp[i][j] == float('inf'):
                continue
            
            # Move to position (i+1, j)
            new_sum = dp[i][j] + triangle[i+1][j]
            if new_sum < dp[i+1][j]:
                dp[i+1][j] = new_sum
                parent[i+1][j] = j
            
            # Move to position (i+1, j+1)
            new_sum = dp[i][j] + triangle[i+1][j+1]
            if new_sum < dp[i+1][j+1]:
                dp[i+1][j+1] = new_sum
                parent[i+1][j+1] = j
    
    # Find minimum in last row
    min_sum = min(dp[n-1])
    min_col = dp[n-1].index(min_sum)
    
    # Reconstruct path
    path = []
    row, col = n - 1, min_col
    
    while row >= 0:
        path.append((row, col))
        if row > 0:
            col = parent[row][col]
        row -= 1
    
    path.reverse()
    path_values = [triangle[r][c] for r, c in path]
    
    return min_sum, path, path_values


def minimum_total_analysis(triangle):
    """
    DETAILED ANALYSIS:
    =================
    Show step-by-step computation and path analysis.
    
    Time Complexity: O(n^2) - analysis computation
    Space Complexity: O(n^2) - temporary tables
    """
    n = len(triangle)
    
    print("Triangle:")
    for i, row in enumerate(triangle):
        spaces = " " * (n - i - 1)
        print(f"{spaces}{' '.join(map(str, row))}")
    
    print(f"\nDP Computation (Bottom-Up):")
    
    # Create DP table for visualization
    dp = [[0] * len(triangle[i]) for i in range(n)]
    
    # Initialize last row
    print(f"Initialize last row:")
    for j in range(len(triangle[n-1])):
        dp[n-1][j] = triangle[n-1][j]
        print(f"  dp[{n-1}][{j}] = {dp[n-1][j]}")
    
    # Fill from bottom to top
    for i in range(n - 2, -1, -1):
        print(f"\nProcessing row {i}:")
        for j in range(len(triangle[i])):
            left_child = dp[i+1][j]
            right_child = dp[i+1][j+1]
            dp[i][j] = triangle[i][j] + min(left_child, right_child)
            print(f"  dp[{i}][{j}] = {triangle[i][j]} + min({left_child}, {right_child}) = {dp[i][j]}")
    
    print(f"\nFinal DP table:")
    for i, row in enumerate(dp):
        spaces = " " * (n - i - 1)
        print(f"{spaces}{' '.join(map(str, row))}")
    
    print(f"\nMinimum path sum: {dp[0][0]}")
    
    # Show actual path
    min_sum, path, path_values = minimum_total_with_path([row[:] for row in triangle])
    print(f"Minimum path: {path}")
    print(f"Path values: {path_values}")
    print(f"Sum verification: {' + '.join(map(str, path_values))} = {sum(path_values)}")
    
    return dp[0][0]


def minimum_total_all_paths(triangle):
    """
    FIND ALL OPTIMAL PATHS:
    =======================
    Find all paths that achieve the minimum sum.
    
    Time Complexity: O(n^2 + k) - DP + k optimal paths
    Space Complexity: O(n^2) - DP table
    """
    n = len(triangle)
    
    # Standard DP to find minimum sum
    dp = [[float('inf')] * len(triangle[i]) for i in range(n)]
    dp[0][0] = triangle[0][0]
    
    for i in range(n - 1):
        for j in range(len(triangle[i])):
            if dp[i][j] == float('inf'):
                continue
            
            # Update adjacent positions
            dp[i+1][j] = min(dp[i+1][j], dp[i][j] + triangle[i+1][j])
            dp[i+1][j+1] = min(dp[i+1][j+1], dp[i][j] + triangle[i+1][j+1])
    
    min_sum = min(dp[n-1])
    
    # Find all paths with minimum sum
    def find_all_paths(row, col, current_path, current_sum):
        current_path.append((row, col))
        current_sum += triangle[row][col]
        
        if row == n - 1:
            if current_sum == min_sum:
                return [current_path[:]]
            return []
        
        all_paths = []
        
        # Try both children if they lead to optimal solution
        if dp[row+1][col] + current_sum == min_sum:
            all_paths.extend(find_all_paths(row + 1, col, current_path, current_sum))
        
        if dp[row+1][col+1] + current_sum == min_sum:
            all_paths.extend(find_all_paths(row + 1, col + 1, current_path, current_sum))
        
        current_path.pop()
        return all_paths
    
    optimal_paths = find_all_paths(0, 0, [], 0)
    return min_sum, optimal_paths


# Test cases
def test_minimum_total():
    """Test all implementations with various inputs"""
    test_cases = [
        ([[2],[3,4],[6,5,7],[4,1,8,3]], 11),
        ([[-10]], -10),
        ([[1],[2,3]], 3),
        ([[1],[2,3],[4,5,6]], 6),
        ([[-1],[2,3],[1,-1,-3]], -1),
        ([[2],[3,4],[6,5,7],[4,1,8,3]], 11),
        ([[5],[7,1],[2,3,4]], 8),
        ([[1],[1,1],[1,1,1]], 3),
        ([[-7],[-2,1],[-5,-8,3]], -14)
    ]
    
    print("Testing Triangle Minimum Path Sum Solutions:")
    print("=" * 70)
    
    for i, (triangle, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Triangle: {triangle}")
        print(f"Expected: {expected}")
        
        # Test approaches (skip expensive ones for large inputs)
        if len(triangle) <= 5:
            try:
                brute = minimum_total_brute_force([row[:] for row in triangle])
                print(f"Brute Force:      {brute:>5} {'✓' if brute == expected else '✗'}")
            except:
                print(f"Brute Force:      Timeout")
        
        memo = minimum_total_memoization([row[:] for row in triangle])
        tabulation = minimum_total_tabulation([row[:] for row in triangle])
        space_opt = minimum_total_space_optimized([row[:] for row in triangle])
        
        print(f"Memoization:      {memo:>5} {'✓' if memo == expected else '✗'}")
        print(f"Tabulation:       {tabulation:>5} {'✓' if tabulation == expected else '✗'}")
        print(f"Space Optimized:  {space_opt:>5} {'✓' if space_opt == expected else '✗'}")
        
        # Test in-place (modifies input)
        triangle_copy = [row[:] for row in triangle]
        in_place = minimum_total_in_place(triangle_copy)
        print(f"In-place:         {in_place:>5} {'✓' if in_place == expected else '✗'}")
        
        # Show path for small cases
        if len(triangle) <= 4:
            min_sum, path, path_values = minimum_total_with_path([row[:] for row in triangle])
            print(f"Path: {path}")
            print(f"Values: {path_values}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    minimum_total_analysis([[2],[3,4],[6,5,7],[4,1,8,3]])
    
    # Show all optimal paths example
    print(f"\n" + "=" * 70)
    print("ALL OPTIMAL PATHS EXAMPLE:")
    print("-" * 40)
    min_sum, all_paths = minimum_total_all_paths([[1],[1,1],[1,1,1]])
    print(f"Triangle: [[1],[1,1],[1,1,1]]")
    print(f"Minimum sum: {min_sum}")
    print(f"All optimal paths: {all_paths}")
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. TRIANGLE STRUCTURE: Adjacent means same column or column + 1")
    print("2. BOTTOM-UP DP: Start from bottom, work up to minimize space")
    print("3. SPACE OPTIMIZATION: Can reuse last row array")
    print("4. IN-PLACE: Modify input triangle for O(1) space")
    print("5. PATH RECONSTRUCTION: Track parent choices for actual path")
    
    print("\n" + "=" * 70)
    print("Algorithm Comparison:")
    print("Brute Force:      Try all possible paths")
    print("Memoization:      Top-down with caching")
    print("Tabulation:       Bottom-up DP")
    print("Space Optimized:  O(n) space optimization")
    print("In-place:         O(1) space by modifying input")
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Brute Force:      Time: O(2^n),   Space: O(n)")
    print("Memoization:      Time: O(n²),    Space: O(n²)")
    print("Tabulation:       Time: O(n²),    Space: O(n²)")
    print("Space Optimized:  Time: O(n²),    Space: O(n)")
    print("In-place:         Time: O(n²),    Space: O(1)")


if __name__ == "__main__":
    test_minimum_total()


"""
PATTERN RECOGNITION:
==================
Triangle is a classic Path Sum problem with unique structure:
- Each row has one more element than previous row
- Can only move to adjacent positions (same col or col+1)
- Find minimum path sum from top to bottom
- Excellent introduction to 2D DP optimization

KEY INSIGHT - TRIANGLE ADJACENCY:
================================
**Adjacent Definition**: From position (i,j), can move to:
- (i+1, j) - directly below
- (i+1, j+1) - diagonally below right

**Why Bottom-Up is Natural**:
- Each position depends only on two positions below
- Natural to start from known values (bottom row)
- Eliminates boundary checking

ALGORITHM APPROACHES:
====================

1. **Bottom-Up DP (Standard)**: O(n²) time, O(n²) space
   - Start from bottom row
   - Each cell = current + min(two children below)
   - Most intuitive approach

2. **Space Optimized**: O(n²) time, O(n) space  
   - Reuse array representing current/next row
   - Process bottom to top, update in-place

3. **In-Place**: O(n²) time, O(1) space
   - Modify input triangle directly
   - Ultimate space optimization

4. **Top-Down Memoization**: O(n²) time, O(n²) space
   - Recursive with memoization
   - Natural problem decomposition

DP STATE DEFINITION:
===================
dp[i][j] = minimum path sum from position (i,j) to bottom

RECURRENCE RELATION:
===================
```
Bottom-up:
dp[i][j] = triangle[i][j] + min(dp[i+1][j], dp[i+1][j+1])

Top-down:
dp[i][j] = triangle[i][j] + min(dfs(i+1,j), dfs(i+1,j+1))
```

Base case: dp[n-1][j] = triangle[n-1][j] (last row)

SPACE OPTIMIZATION TECHNIQUES:
=============================

**Level 1 - Rolling Array**: O(n) space
```python
dp = triangle[-1][:]  # Start with last row
for i in range(n-2, -1, -1):
    for j in range(len(triangle[i])):
        dp[j] = triangle[i][j] + min(dp[j], dp[j+1])
```

**Level 2 - In-Place**: O(1) space
```python
for i in range(n-2, -1, -1):
    for j in range(len(triangle[i])):
        triangle[i][j] += min(triangle[i+1][j], triangle[i+1][j+1])
```

PATH RECONSTRUCTION:
===================
To find actual minimum path:
1. Use standard DP to find minimum sum
2. Track parent choices during DP
3. Backtrack from top using parent pointers
4. Or forward track by choosing optimal at each step

MATHEMATICAL PROPERTIES:
=======================
**Triangle Numbers**: Row i has i+1 elements
**Total Elements**: 1 + 2 + ... + n = n(n+1)/2
**Space Efficiency**: Triangle uses exactly n(n+1)/2 space

EDGE CASES:
==========
1. **Single element**: Return that element
2. **Negative numbers**: Handle correctly (don't assume positive)
3. **All same values**: Multiple optimal paths
4. **Large values**: Integer overflow considerations

OPTIMIZATION OPPORTUNITIES:
==========================
1. **Early termination**: If all paths converge
2. **Pruning**: Skip clearly suboptimal branches
3. **Parallel computation**: Independent columns can be computed in parallel
4. **Memory access patterns**: Cache-friendly traversal

APPLICATIONS:
============
1. **Game Trees**: Decision trees with branching factor 2
2. **Investment Planning**: Multi-stage investment decisions
3. **Resource Allocation**: Hierarchical resource distribution
4. **Path Planning**: Navigation with restricted movement

VARIANTS TO PRACTICE:
====================
- Minimum Path Sum (64) - rectangular grid version
- Unique Paths (62) - counting instead of minimizing  
- Maximum Path Sum in Binary Tree (124) - tree version
- Path Sum III (437) - tree path counting

INTERVIEW TIPS:
==============
1. **Recognize triangle structure**: Adjacency rules are key
2. **Choose bottom-up**: More natural than top-down
3. **Show space optimization**: O(n²) → O(n) → O(1)
4. **Handle edge cases**: Single element, negative values
5. **Explain recurrence**: Why min of two children
6. **Discuss alternatives**: Top-down vs bottom-up
7. **Path reconstruction**: How to find actual optimal path
8. **Complexity analysis**: Why O(n²) is optimal
9. **Real applications**: Decision trees, game theory
10. **Follow-up optimizations**: In-place modification, parallel computation

MATHEMATICAL INSIGHT:
====================
Triangle demonstrates the principle of **optimal substructure**:
- Optimal path to bottom = current value + optimal path from next row
- Local optimality (choosing min of two children) leads to global optimality
- Classic example of **greedy choice property** in DP context

The space optimization showcases how understanding data dependencies 
enables dramatic space improvements without affecting time complexity.
"""
