"""
LeetCode 931: Minimum Falling Path Sum
Difficulty: Medium
Category: Grid/Matrix DP

PROBLEM DESCRIPTION:
===================
Given an n x n array of integers matrix, return the minimum sum of any falling path through matrix.

A falling path starts at any element in the first row and chooses the element in the next row that is 
either directly below or diagonally below left or diagonally below right. Specifically, the next element 
from position (row, col) will be (row + 1, col - 1), (row + 1, col), or (row + 1, col + 1).

Example 1:
Input: matrix = [[2,1,3],[6,5,4],[7,8,9]]
Output: 13
Explanation: There are two falling paths with a minimum sum as shown.

Example 2:
Input: matrix = [[-19,57],[-40,-5]]
Output: -59

Constraints:
- n == matrix.length == matrix[i].length
- 1 <= n <= 100
- -100 <= matrix[i][j] <= 100
"""

def min_falling_path_sum_brute_force(matrix):
    """
    BRUTE FORCE APPROACH:
    ====================
    Try all possible falling paths from each starting position.
    
    Time Complexity: O(3^n) - three choices at each level
    Space Complexity: O(n) - recursion depth
    """
    if not matrix or not matrix[0]:
        return 0
    
    n = len(matrix)
    min_sum = float('inf')
    
    def dfs(row, col):
        if row >= n:
            return 0
        
        if col < 0 or col >= n:
            return float('inf')
        
        current = matrix[row][col]
        
        # Try three possible moves: down-left, down, down-right
        down_left = dfs(row + 1, col - 1)
        down = dfs(row + 1, col)
        down_right = dfs(row + 1, col + 1)
        
        return current + min(down_left, down, down_right)
    
    # Try starting from each position in first row
    for start_col in range(n):
        path_sum = dfs(0, start_col)
        min_sum = min(min_sum, path_sum)
    
    return min_sum


def min_falling_path_sum_memoization(matrix):
    """
    MEMOIZATION APPROACH:
    ====================
    Use memoization to avoid recomputing same subproblems.
    
    Time Complexity: O(n^2) - each cell computed once
    Space Complexity: O(n^2) - memoization table
    """
    if not matrix or not matrix[0]:
        return 0
    
    n = len(matrix)
    memo = {}
    
    def dfs(row, col):
        if row >= n:
            return 0
        
        if col < 0 or col >= n:
            return float('inf')
        
        if (row, col) in memo:
            return memo[(row, col)]
        
        current = matrix[row][col]
        
        # Try three possible moves
        down_left = dfs(row + 1, col - 1)
        down = dfs(row + 1, col)
        down_right = dfs(row + 1, col + 1)
        
        result = current + min(down_left, down, down_right)
        memo[(row, col)] = result
        return result
    
    # Try starting from each position in first row
    min_sum = float('inf')
    for start_col in range(n):
        path_sum = dfs(0, start_col)
        min_sum = min(min_sum, path_sum)
    
    return min_sum


def min_falling_path_sum_dp(matrix):
    """
    DYNAMIC PROGRAMMING APPROACH:
    ============================
    Build solution bottom-up using 2D DP table.
    
    Time Complexity: O(n^2) - process each cell once
    Space Complexity: O(n^2) - DP table
    """
    if not matrix or not matrix[0]:
        return 0
    
    n = len(matrix)
    
    # DP table: dp[i][j] = minimum falling path sum to reach position (i,j)
    dp = [[float('inf')] * n for _ in range(n)]
    
    # Initialize first row
    for j in range(n):
        dp[0][j] = matrix[0][j]
    
    # Fill DP table row by row
    for i in range(1, n):
        for j in range(n):
            # Consider three possible previous positions
            # From directly above
            dp[i][j] = min(dp[i][j], dp[i-1][j] + matrix[i][j])
            
            # From diagonally above-left
            if j > 0:
                dp[i][j] = min(dp[i][j], dp[i-1][j-1] + matrix[i][j])
            
            # From diagonally above-right
            if j < n - 1:
                dp[i][j] = min(dp[i][j], dp[i-1][j+1] + matrix[i][j])
    
    # Return minimum value in last row
    return min(dp[n-1])


def min_falling_path_sum_space_optimized(matrix):
    """
    SPACE OPTIMIZED APPROACH:
    =========================
    Use only O(n) space by processing row by row.
    
    Time Complexity: O(n^2) - process each cell once
    Space Complexity: O(n) - single row array
    """
    if not matrix or not matrix[0]:
        return 0
    
    n = len(matrix)
    
    # Use single array to represent previous row
    prev = matrix[0][:]
    
    for i in range(1, n):
        curr = [float('inf')] * n
        
        for j in range(n):
            # From directly above
            curr[j] = min(curr[j], prev[j] + matrix[i][j])
            
            # From diagonally above-left
            if j > 0:
                curr[j] = min(curr[j], prev[j-1] + matrix[i][j])
            
            # From diagonally above-right
            if j < n - 1:
                curr[j] = min(curr[j], prev[j+1] + matrix[i][j])
        
        prev = curr
    
    return min(prev)


def min_falling_path_sum_in_place(matrix):
    """
    IN-PLACE APPROACH:
    =================
    Modify the input matrix itself to save space.
    
    Time Complexity: O(n^2) - process each cell once
    Space Complexity: O(1) - no extra space
    """
    if not matrix or not matrix[0]:
        return 0
    
    n = len(matrix)
    
    # Process from second row to last row
    for i in range(1, n):
        for j in range(n):
            # Find minimum from three possible previous positions
            min_prev = matrix[i-1][j]  # From directly above
            
            if j > 0:
                min_prev = min(min_prev, matrix[i-1][j-1])  # From above-left
            
            if j < n - 1:
                min_prev = min(min_prev, matrix[i-1][j+1])  # From above-right
            
            matrix[i][j] += min_prev
    
    # Return minimum value in last row
    return min(matrix[n-1])


def min_falling_path_sum_with_path(matrix):
    """
    FIND ACTUAL MINIMUM PATH:
    =========================
    Return minimum sum and the actual optimal path.
    
    Time Complexity: O(n^2) - DP + path reconstruction
    Space Complexity: O(n^2) - DP table + parent tracking
    """
    if not matrix or not matrix[0]:
        return 0, []
    
    n = len(matrix)
    
    # DP table and parent tracking
    dp = [[float('inf')] * n for _ in range(n)]
    parent = [[-1] * n for _ in range(n)]
    
    # Initialize first row
    for j in range(n):
        dp[0][j] = matrix[0][j]
    
    # Fill DP table with parent tracking
    for i in range(1, n):
        for j in range(n):
            # Check three possible previous positions
            candidates = []
            
            # From directly above
            candidates.append((dp[i-1][j], j))
            
            # From diagonally above-left
            if j > 0:
                candidates.append((dp[i-1][j-1], j-1))
            
            # From diagonally above-right
            if j < n - 1:
                candidates.append((dp[i-1][j+1], j+1))
            
            # Choose best candidate
            min_sum, best_prev_col = min(candidates)
            dp[i][j] = min_sum + matrix[i][j]
            parent[i][j] = best_prev_col
    
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
    path_values = [matrix[r][c] for r, c in path]
    
    return min_sum, path, path_values


def min_falling_path_sum_analysis(matrix):
    """
    DETAILED ANALYSIS:
    =================
    Show step-by-step DP computation and path analysis.
    
    Time Complexity: O(n^2) - analysis computation
    Space Complexity: O(n^2) - temporary tables
    """
    if not matrix or not matrix[0]:
        print("Empty matrix!")
        return 0
    
    n = len(matrix)
    
    print("Input Matrix:")
    for i, row in enumerate(matrix):
        print(f"  Row {i}: {row}")
    
    print(f"\nDP Computation:")
    
    # Create DP table for visualization
    dp = [[float('inf')] * n for _ in range(n)]
    
    # Initialize first row
    print(f"Initialize first row:")
    for j in range(n):
        dp[0][j] = matrix[0][j]
        print(f"  dp[0][{j}] = {dp[0][j]}")
    
    # Fill DP table row by row
    for i in range(1, n):
        print(f"\nProcessing row {i}:")
        for j in range(n):
            candidates = []
            
            # From directly above
            candidates.append((dp[i-1][j], f"from ({i-1},{j})"))
            
            # From diagonally above-left
            if j > 0:
                candidates.append((dp[i-1][j-1], f"from ({i-1},{j-1})"))
            
            # From diagonally above-right
            if j < n - 1:
                candidates.append((dp[i-1][j+1], f"from ({i-1},{j+1})"))
            
            min_prev, source = min(candidates)
            dp[i][j] = min_prev + matrix[i][j]
            
            print(f"  dp[{i}][{j}] = {min_prev} + {matrix[i][j]} = {dp[i][j]} ({source})")
    
    print(f"\nFinal DP table:")
    for i, row in enumerate(dp):
        print(f"  Row {i}: {row}")
    
    result = min(dp[n-1])
    min_col = dp[n-1].index(result)
    print(f"\nMinimum falling path sum: {result}")
    print(f"Optimal ending position: ({n-1}, {min_col})")
    
    # Show actual path
    min_sum, path, path_values = min_falling_path_sum_with_path([row[:] for row in matrix])
    print(f"Optimal path: {path}")
    print(f"Path values: {path_values}")
    print(f"Sum verification: {' + '.join(map(str, path_values))} = {sum(path_values)}")
    
    return result


def min_falling_path_sum_all_paths(matrix):
    """
    FIND ALL OPTIMAL PATHS:
    =======================
    Find all paths that achieve the minimum sum.
    
    Time Complexity: O(n^2 + k) - DP + k optimal paths
    Space Complexity: O(n^2) - DP table
    """
    if not matrix or not matrix[0]:
        return 0, []
    
    n = len(matrix)
    
    # Standard DP to find minimum sum
    dp = [[float('inf')] * n for _ in range(n)]
    
    # Initialize first row
    for j in range(n):
        dp[0][j] = matrix[0][j]
    
    # Fill DP table
    for i in range(1, n):
        for j in range(n):
            # From directly above
            dp[i][j] = min(dp[i][j], dp[i-1][j] + matrix[i][j])
            
            # From diagonally above-left
            if j > 0:
                dp[i][j] = min(dp[i][j], dp[i-1][j-1] + matrix[i][j])
            
            # From diagonally above-right
            if j < n - 1:
                dp[i][j] = min(dp[i][j], dp[i-1][j+1] + matrix[i][j])
    
    min_sum = min(dp[n-1])
    
    # Find all paths with minimum sum
    def find_all_paths(row, col, current_path, current_sum):
        current_path.append((row, col))
        current_sum += matrix[row][col]
        
        if row == n - 1:
            if current_sum == min_sum:
                return [current_path[:]]
            return []
        
        all_paths = []
        
        # Try three possible next moves
        for next_col in [col - 1, col, col + 1]:
            if 0 <= next_col < n and dp[row+1][next_col] + current_sum == min_sum:
                all_paths.extend(find_all_paths(row + 1, next_col, current_path, current_sum))
        
        current_path.pop()
        return all_paths
    
    # Find all optimal starting positions
    optimal_paths = []
    for start_col in range(n):
        if dp[n-1][start_col] == min_sum:
            # Work backwards to find paths ending here
            optimal_paths.extend(find_all_paths(0, start_col, [], 0))
    
    return min_sum, optimal_paths


# Test cases
def test_min_falling_path_sum():
    """Test all implementations with various inputs"""
    test_cases = [
        ([[2,1,3],[6,5,4],[7,8,9]], 13),
        ([[-19,57],[-40,-5]], -59),
        ([[1]], 1),
        ([[7,6,2]], 2),
        ([[1,2,3],[4,5,6],[7,8,9]], 12),
        ([[-1,-2,-3],[-4,-5,-6],[-7,-8,-9]], -21),
        ([[100,-42,-46,-41],[31,97,10,-10],[-58,-51,82,89],[51,81,69,-51]], -36),
        ([[1,2],[3,4]], 5),
        ([[2,1,3],[6,5,4],[7,8,9]], 13)
    ]
    
    print("Testing Minimum Falling Path Sum Solutions:")
    print("=" * 70)
    
    for i, (matrix, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Matrix: {matrix}")
        print(f"Expected: {expected}")
        
        # Test approaches (skip expensive ones for large inputs)
        if len(matrix) <= 4:
            try:
                brute = min_falling_path_sum_brute_force([row[:] for row in matrix])
                print(f"Brute Force:      {brute:>5} {'✓' if brute == expected else '✗'}")
            except:
                print(f"Brute Force:      Timeout")
        
        memo = min_falling_path_sum_memoization([row[:] for row in matrix])
        dp_result = min_falling_path_sum_dp([row[:] for row in matrix])
        space_opt = min_falling_path_sum_space_optimized([row[:] for row in matrix])
        
        print(f"Memoization:      {memo:>5} {'✓' if memo == expected else '✗'}")
        print(f"DP (2D):          {dp_result:>5} {'✓' if dp_result == expected else '✗'}")
        print(f"Space Optimized:  {space_opt:>5} {'✓' if space_opt == expected else '✗'}")
        
        # Test in-place (modifies input)
        matrix_copy = [row[:] for row in matrix]
        in_place = min_falling_path_sum_in_place(matrix_copy)
        print(f"In-place:         {in_place:>5} {'✓' if in_place == expected else '✗'}")
        
        # Show path for small cases
        if len(matrix) <= 4:
            min_sum, path, path_values = min_falling_path_sum_with_path([row[:] for row in matrix])
            print(f"Optimal path: {path}")
            print(f"Path values: {path_values}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    min_falling_path_sum_analysis([[2,1,3],[6,5,4],[7,8,9]])
    
    # Show all optimal paths example
    print(f"\n" + "=" * 70)
    print("ALL OPTIMAL PATHS EXAMPLE:")
    print("-" * 40)
    min_sum, all_paths = min_falling_path_sum_all_paths([[1,2,3],[4,5,6],[7,8,9]])
    print(f"Matrix: [[1,2,3],[4,5,6],[7,8,9]]")
    print(f"Minimum sum: {min_sum}")
    print(f"All optimal paths: {all_paths}")
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. MOVEMENT RULES: Can move to 3 positions in next row (left, center, right)")
    print("2. BOUNDARY HANDLING: Edge columns have fewer movement options")
    print("3. DP RECURRENCE: dp[i][j] = matrix[i][j] + min(dp[i-1][j-1:j+2])")
    print("4. SPACE OPTIMIZATION: Can reduce to O(n) space using rolling array")
    print("5. MULTIPLE STARTS: Try all starting positions in first row")
    
    print("\n" + "=" * 70)
    print("Algorithm Comparison:")
    print("Brute Force:      Try all possible falling paths")
    print("Memoization:      Top-down with caching")
    print("DP (2D):          Bottom-up with 2D table")
    print("Space Optimized:  Row-by-row processing")
    print("In-place:         Modify input matrix")
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Brute Force:      Time: O(3^n),   Space: O(n)")
    print("Memoization:      Time: O(n²),    Space: O(n²)")
    print("DP (2D):          Time: O(n²),    Space: O(n²)")
    print("Space Optimized:  Time: O(n²),    Space: O(n)")
    print("In-place:         Time: O(n²),    Space: O(1)")


if __name__ == "__main__":
    test_min_falling_path_sum()


"""
PATTERN RECOGNITION:
==================
Minimum Falling Path Sum is a classic path optimization problem:
- Find optimal path through matrix with movement constraints
- Can move to 3 adjacent positions in next row
- Similar to triangle problem but with fixed width and diagonal moves
- Foundation for many path optimization problems in grids

KEY INSIGHT - MOVEMENT CONSTRAINTS:
==================================
**Movement Rules**: From position (i,j), can move to:
- (i+1, j-1) - diagonally down-left
- (i+1, j) - directly down  
- (i+1, j+1) - diagonally down-right

**Boundary Conditions**:
- Left edge (j=0): Can't move to j-1
- Right edge (j=n-1): Can't move to j+1
- Middle positions: All three moves available

**Starting Positions**: Can start from any position in first row

ALGORITHM APPROACHES:
====================

1. **2D DP (Standard)**: O(n²) time, O(n²) space
   - Build complete DP table row by row
   - Consider all valid previous positions
   - Most intuitive approach

2. **Space Optimized**: O(n²) time, O(n) space
   - Use rolling arrays (previous row, current row)
   - Reduce space complexity significantly

3. **In-Place**: O(n²) time, O(1) space
   - Modify input matrix directly
   - Ultimate space optimization

4. **Memoization**: O(n²) time, O(n²) space
   - Top-down recursive approach with caching
   - Natural problem decomposition

DP RECURRENCE RELATION:
======================
```
dp[i][j] = matrix[i][j] + min(valid_previous_positions)

where valid_previous_positions are:
- dp[i-1][j-1] if j > 0
- dp[i-1][j]
- dp[i-1][j+1] if j < n-1
```

**Base Case**: dp[0][j] = matrix[0][j] for all j (first row)

**Goal**: min(dp[n-1][j]) for all j (minimum in last row)

SPACE OPTIMIZATION DETAILS:
===========================

**Level 1 - Rolling Array**: O(n) space
```python
prev = matrix[0][:]  # Initialize with first row
for i in range(1, n):
    curr = [float('inf')] * n
    for j in range(n):
        # Compute curr[j] using prev array
    prev = curr
```

**Level 2 - In-Place**: O(1) space
```python
for i in range(1, n):
    for j in range(n):
        min_prev = matrix[i-1][j]
        if j > 0: min_prev = min(min_prev, matrix[i-1][j-1])
        if j < n-1: min_prev = min(min_prev, matrix[i-1][j+1])
        matrix[i][j] += min_prev
```

BOUNDARY HANDLING:
==================
**Edge Case Management**:
- **Left boundary** (j=0): Only consider moves from (i-1,0) and (i-1,1)
- **Right boundary** (j=n-1): Only consider moves from (i-1,n-2) and (i-1,n-1)  
- **Interior positions**: Consider all three possible previous positions

**Implementation Pattern**:
```python
candidates = [dp[i-1][j]]  # Always include directly above
if j > 0: candidates.append(dp[i-1][j-1])  # Add left diagonal if valid
if j < n-1: candidates.append(dp[i-1][j+1])  # Add right diagonal if valid
dp[i][j] = matrix[i][j] + min(candidates)
```

PATH RECONSTRUCTION:
===================
To find actual optimal path:
1. **Forward pass**: Compute DP table with parent tracking
2. **Find optimal ending**: Minimum value in last row
3. **Backward reconstruction**: Follow parent pointers to reconstruct path

```python
# During DP computation
min_val, best_prev_col = min((dp[i-1][k], k) for k in valid_prev_cols)
dp[i][j] = matrix[i][j] + min_val
parent[i][j] = best_prev_col

# Reconstruction
path = []
col = min_ending_column
for row in range(n-1, -1, -1):
    path.append((row, col))
    if row > 0: col = parent[row][col]
path.reverse()
```

EDGE CASES:
==========
1. **Single element**: Return matrix[0][0]
2. **Single row**: Return minimum element in row
3. **Single column**: Return sum of all elements
4. **All negative**: Still find minimum path
5. **All positive**: Greedy approach would work, but DP is general

MATHEMATICAL PROPERTIES:
=======================
- **Optimal substructure**: Optimal path contains optimal subpaths
- **Overlapping subproblems**: Same positions reached via different paths
- **No negative cycles**: Problem well-defined (acyclic graph)
- **Multiple optima**: May have multiple paths with same minimum sum

APPLICATIONS:
============
1. **Game Development**: Character movement with terrain costs
2. **Robotics**: Path planning with movement constraints
3. **Economics**: Cost optimization with limited choices
4. **Image Processing**: Seam carving algorithms
5. **Network Routing**: Multi-hop path optimization

VARIANTS TO PRACTICE:
====================
- Triangle (120) - variable width version
- Unique Paths (62/63) - counting instead of optimization
- Dungeon Game (174) - backward DP with constraints
- Cherry Pickup (741) - bidirectional path optimization

INTERVIEW TIPS:
==============
1. **Clarify movement rules**: Understand exactly which moves are allowed
2. **Handle boundaries carefully**: Edge cases for j=0 and j=n-1
3. **Show space optimization**: Demonstrate O(n²) → O(n) → O(1) progression
4. **Trace small example**: Walk through DP computation step by step
5. **Discuss starting positions**: Why we try all positions in first row
6. **Path reconstruction**: Explain how to find actual optimal path
7. **Edge cases**: Single row, single column, all negative values
8. **Real applications**: Game development, robotics, economics
9. **Complexity analysis**: Why O(n²) time is necessary
10. **Alternative approaches**: Greedy vs DP, when each works

OPTIMIZATION OPPORTUNITIES:
==========================
1. **Early termination**: If all remaining paths clearly suboptimal
2. **Preprocessing**: Identify obviously bad regions
3. **Parallel computation**: Independent column computations within row
4. **Memory access optimization**: Cache-friendly traversal patterns

MATHEMATICAL INSIGHT:
====================
This problem demonstrates **constrained optimization** in dynamic programming:
- **Movement constraints** limit available transitions
- **Boundary conditions** require special handling
- **Multiple starting points** require global optimization

The three-way choice at each step creates a **dependency structure** that makes 
DP natural while preventing greedy approaches from working in general.
"""
