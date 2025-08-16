"""
LeetCode 1594: Maximum Non Negative Product in a Matrix
Difficulty: Medium
Category: Grid/Matrix DP

PROBLEM DESCRIPTION:
===================
You are given a m x n matrix grid. Initially, you are located at the top-left corner (0, 0), and in each step, 
you can only move right or down in the matrix.

Among all possible paths to reach the bottom-right corner (m - 1, n - 1), your task is to find the path with 
the maximum product. The product of a path is the product of all the elements along the path.

Note that the product may be very large, so the result should be taken modulo 10^9 + 7.

If the maximum product is negative, return -1.

Example 1:
Input: grid = [[-1,-2,-3],[-2,-3,-3],[-3,-3,-2]]
Output: -1
Explanation: It's not possible to get non-negative product in the path from (0, 0) to (2, 2), so return -1.

Example 2:
Input: grid = [[1,-2,1],[1,-2,1],[3,-4,5]]
Output: 8
Explanation: Maximum non-negative product is shown (1 * 1 * -2 * -4 * 5 = 40).

Example 3:
Input: grid = [[1,3],[0,-4]]
Output: 0
Explanation: Maximum non-negative product is shown (1 * 0 * -4 = 0).

Constraints:
- m == grid.length
- n == grid[i].length
- 1 <= m, n <= 15
- -4 <= grid[i][j] <= 4
"""

def max_product_path_brute_force(grid):
    """
    BRUTE FORCE APPROACH:
    ====================
    Try all possible paths from top-left to bottom-right.
    
    Time Complexity: O(2^(m+n)) - exponential paths
    Space Complexity: O(m+n) - recursion depth
    """
    if not grid or not grid[0]:
        return -1
    
    m, n = len(grid), len(grid[0])
    MOD = 10**9 + 7
    max_product = float('-inf')
    
    def dfs(row, col, current_product):
        nonlocal max_product
        
        # Update current product
        current_product *= grid[row][col]
        
        # If reached destination
        if row == m - 1 and col == n - 1:
            max_product = max(max_product, current_product)
            return
        
        # Try going down
        if row + 1 < m:
            dfs(row + 1, col, current_product)
        
        # Try going right
        if col + 1 < n:
            dfs(row, col + 1, current_product)
    
    dfs(0, 0, 1)
    
    if max_product < 0:
        return -1
    
    return max_product % MOD


def max_product_path_memoization(grid):
    """
    MEMOIZATION APPROACH:
    ====================
    Use memoization to track both maximum and minimum products.
    
    Time Complexity: O(m*n) - each cell computed once
    Space Complexity: O(m*n) - memoization table
    """
    if not grid or not grid[0]:
        return -1
    
    m, n = len(grid), len(grid[0])
    MOD = 10**9 + 7
    memo = {}
    
    def dfs(row, col):
        if row >= m or col >= n:
            return (float('-inf'), float('inf'))  # (max, min)
        
        if row == m - 1 and col == n - 1:
            return (grid[row][col], grid[row][col])
        
        if (row, col) in memo:
            return memo[(row, col)]
        
        current = grid[row][col]
        
        # Get max/min from both directions
        down_max, down_min = dfs(row + 1, col)
        right_max, right_min = dfs(row, col + 1)
        
        # Calculate candidates
        candidates = []
        if down_max != float('-inf'):
            candidates.extend([current * down_max, current * down_min])
        if right_max != float('-inf'):
            candidates.extend([current * right_max, current * right_min])
        
        if not candidates:
            result = (float('-inf'), float('inf'))
        else:
            result = (max(candidates), min(candidates))
        
        memo[(row, col)] = result
        return result
    
    max_product, _ = dfs(0, 0)
    
    if max_product < 0:
        return -1
    
    return max_product % MOD


def max_product_path_dp(grid):
    """
    DYNAMIC PROGRAMMING APPROACH:
    ============================
    Track both maximum and minimum products at each position.
    
    Time Complexity: O(m*n) - process each cell once
    Space Complexity: O(m*n) - DP tables
    """
    if not grid or not grid[0]:
        return -1
    
    m, n = len(grid), len(grid[0])
    MOD = 10**9 + 7
    
    # DP tables for maximum and minimum products
    dp_max = [[float('-inf')] * n for _ in range(m)]
    dp_min = [[float('inf')] * n for _ in range(m)]
    
    # Initialize starting position
    dp_max[0][0] = dp_min[0][0] = grid[0][0]
    
    # Fill first row
    for j in range(1, n):
        dp_max[0][j] = dp_max[0][j-1] * grid[0][j]
        dp_min[0][j] = dp_min[0][j-1] * grid[0][j]
    
    # Fill first column
    for i in range(1, m):
        dp_max[i][0] = dp_max[i-1][0] * grid[i][0]
        dp_min[i][0] = dp_min[i-1][0] * grid[i][0]
    
    # Fill rest of the table
    for i in range(1, m):
        for j in range(1, n):
            current = grid[i][j]
            
            # Calculate candidates from both directions
            candidates = [
                current * dp_max[i-1][j],  # From top (max)
                current * dp_min[i-1][j],  # From top (min)
                current * dp_max[i][j-1],  # From left (max)
                current * dp_min[i][j-1]   # From left (min)
            ]
            
            dp_max[i][j] = max(candidates)
            dp_min[i][j] = min(candidates)
    
    max_product = dp_max[m-1][n-1]
    
    if max_product < 0:
        return -1
    
    return max_product % MOD


def max_product_path_space_optimized(grid):
    """
    SPACE OPTIMIZED APPROACH:
    =========================
    Use only O(n) space by processing row by row.
    
    Time Complexity: O(m*n) - process each cell once
    Space Complexity: O(n) - single row arrays
    """
    if not grid or not grid[0]:
        return -1
    
    m, n = len(grid), len(grid[0])
    MOD = 10**9 + 7
    
    # Use arrays to represent current row
    max_prev = [0] * n
    min_prev = [0] * n
    
    # Initialize first row
    max_prev[0] = min_prev[0] = grid[0][0]
    for j in range(1, n):
        max_prev[j] = max_prev[j-1] * grid[0][j]
        min_prev[j] = min_prev[j-1] * grid[0][j]
    
    # Process each subsequent row
    for i in range(1, m):
        max_curr = [0] * n
        min_curr = [0] * n
        
        # First column
        max_curr[0] = max_prev[0] * grid[i][0]
        min_curr[0] = min_prev[0] * grid[i][0]
        
        # Rest of the row
        for j in range(1, n):
            current = grid[i][j]
            
            candidates = [
                current * max_prev[j],  # From top (max)
                current * min_prev[j],  # From top (min)
                current * max_curr[j-1],  # From left (max)
                current * min_curr[j-1]   # From left (min)
            ]
            
            max_curr[j] = max(candidates)
            min_curr[j] = min(candidates)
        
        max_prev, min_prev = max_curr, min_curr
    
    max_product = max_prev[n-1]
    
    if max_product < 0:
        return -1
    
    return max_product % MOD


def max_product_path_with_path(grid):
    """
    FIND ACTUAL MAXIMUM PATH:
    =========================
    Return maximum product and the actual optimal path.
    
    Time Complexity: O(m*n) - DP + path reconstruction
    Space Complexity: O(m*n) - DP tables + parent tracking
    """
    if not grid or not grid[0]:
        return -1, []
    
    m, n = len(grid), len(grid[0])
    MOD = 10**9 + 7
    
    # DP tables and parent tracking
    dp_max = [[float('-inf')] * n for _ in range(m)]
    dp_min = [[float('inf')] * n for _ in range(m)]
    parent = [[None] * n for _ in range(m)]
    
    # Initialize starting position
    dp_max[0][0] = dp_min[0][0] = grid[0][0]
    
    # Fill first row
    for j in range(1, n):
        dp_max[0][j] = dp_max[0][j-1] * grid[0][j]
        dp_min[0][j] = dp_min[0][j-1] * grid[0][j]
        parent[0][j] = (0, j-1)
    
    # Fill first column
    for i in range(1, m):
        dp_max[i][0] = dp_max[i-1][0] * grid[i][0]
        dp_min[i][0] = dp_min[i-1][0] * grid[i][0]
        parent[i][0] = (i-1, 0)
    
    # Fill rest of the table
    for i in range(1, m):
        for j in range(1, n):
            current = grid[i][j]
            
            # Calculate candidates with sources
            candidates = [
                (current * dp_max[i-1][j], (i-1, j)),  # From top (max)
                (current * dp_min[i-1][j], (i-1, j)),  # From top (min)
                (current * dp_max[i][j-1], (i, j-1)),  # From left (max)
                (current * dp_min[i][j-1], (i, j-1))   # From left (min)
            ]
            
            # Find best for maximum
            max_val, max_source = max(candidates)
            dp_max[i][j] = max_val
            
            # Find best for minimum
            min_val, _ = min(candidates)
            dp_min[i][j] = min_val
            
            # Track parent for maximum path
            parent[i][j] = max_source
    
    max_product = dp_max[m-1][n-1]
    
    if max_product < 0:
        return -1, []
    
    # Reconstruct path
    path = []
    current = (m-1, n-1)
    
    while current is not None:
        path.append(current)
        current = parent[current[0]][current[1]]
    
    path.reverse()
    
    return max_product % MOD, path


def max_product_path_analysis(grid):
    """
    DETAILED ANALYSIS:
    =================
    Show step-by-step DP computation and product tracking.
    
    Time Complexity: O(m*n) - analysis computation
    Space Complexity: O(m*n) - temporary tables
    """
    if not grid or not grid[0]:
        print("Empty grid!")
        return -1
    
    m, n = len(grid), len(grid[0])
    MOD = 10**9 + 7
    
    print("Input Grid:")
    for i, row in enumerate(grid):
        print(f"  Row {i}: {row}")
    
    print(f"\nDP Computation (tracking max and min products):")
    
    # Create DP tables for visualization
    dp_max = [[float('-inf')] * n for _ in range(m)]
    dp_min = [[float('inf')] * n for _ in range(m)]
    
    # Initialize starting position
    dp_max[0][0] = dp_min[0][0] = grid[0][0]
    print(f"Initialize start: dp_max[0][0] = dp_min[0][0] = {grid[0][0]}")
    
    # Fill first row
    print(f"\nFilling first row:")
    for j in range(1, n):
        dp_max[0][j] = dp_max[0][j-1] * grid[0][j]
        dp_min[0][j] = dp_min[0][j-1] * grid[0][j]
        print(f"  dp_max[0][{j}] = {dp_max[0][j-1]} * {grid[0][j]} = {dp_max[0][j]}")
        print(f"  dp_min[0][{j}] = {dp_min[0][j-1]} * {grid[0][j]} = {dp_min[0][j]}")
    
    # Fill first column
    print(f"\nFilling first column:")
    for i in range(1, m):
        dp_max[i][0] = dp_max[i-1][0] * grid[i][0]
        dp_min[i][0] = dp_min[i-1][0] * grid[i][0]
        print(f"  dp_max[{i}][0] = {dp_max[i-1][0]} * {grid[i][0]} = {dp_max[i][0]}")
        print(f"  dp_min[{i}][0] = {dp_min[i-1][0]} * {grid[i][0]} = {dp_min[i][0]}")
    
    # Fill rest of the table
    print(f"\nFilling rest of grid:")
    for i in range(1, m):
        for j in range(1, n):
            current = grid[i][j]
            
            candidates = [
                current * dp_max[i-1][j],  # From top (max)
                current * dp_min[i-1][j],  # From top (min)
                current * dp_max[i][j-1],  # From left (max)
                current * dp_min[i][j-1]   # From left (min)
            ]
            
            dp_max[i][j] = max(candidates)
            dp_min[i][j] = min(candidates)
            
            print(f"  dp_max[{i}][{j}] = max({candidates}) = {dp_max[i][j]}")
            print(f"  dp_min[{i}][{j}] = min({candidates}) = {dp_min[i][j]}")
    
    print(f"\nFinal DP tables:")
    print("Max products:")
    for i, row in enumerate(dp_max):
        print(f"  Row {i}: {row}")
    
    print("Min products:")
    for i, row in enumerate(dp_min):
        print(f"  Row {i}: {row}")
    
    max_product = dp_max[m-1][n-1]
    print(f"\nMaximum product: {max_product}")
    
    if max_product < 0:
        print("Result: -1 (negative product)")
        return -1
    else:
        result = max_product % MOD
        print(f"Result: {result} (mod {MOD})")
        
        # Show actual path
        _, path = max_product_path_with_path([row[:] for row in grid])
        if path:
            print(f"Optimal path: {path}")
            path_values = [grid[r][c] for r, c in path]
            print(f"Path values: {path_values}")
            product = 1
            for val in path_values:
                product *= val
            print(f"Product verification: {' * '.join(map(str, path_values))} = {product}")
        
        return result


# Test cases
def test_max_product_path():
    """Test all implementations with various inputs"""
    test_cases = [
        ([[-1,-2,-3],[-2,-3,-3],[-3,-3,-2]], -1),
        ([[1,-2,1],[1,-2,1],[3,-4,5]], 8),
        ([[1,3],[0,-4]], 0),
        ([[1,4,4,0],[2,3,1,1],[2,3,1,1]], 16),
        ([[-1,-2,-3]], -1),
        ([[2,-3,3],[1,4,-2]], 24),
        ([[1]], 1),
        ([[0]], 0),
        ([[-1]], -1),
        ([[2,3],[1,4]], 24),
        ([[1,-2,3],[-4,5,-6],[7,-8,9]], 432)
    ]
    
    print("Testing Maximum Non Negative Product Solutions:")
    print("=" * 70)
    
    for i, (grid, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Grid: {grid}")
        print(f"Expected: {expected}")
        
        # Test approaches (skip expensive ones for large inputs)
        if len(grid) <= 3 and len(grid[0]) <= 3:
            try:
                brute = max_product_path_brute_force([row[:] for row in grid])
                print(f"Brute Force:      {brute:>8} {'✓' if brute == expected else '✗'}")
            except:
                print(f"Brute Force:      Timeout")
        
        memo = max_product_path_memoization([row[:] for row in grid])
        dp_result = max_product_path_dp([row[:] for row in grid])
        space_opt = max_product_path_space_optimized([row[:] for row in grid])
        
        print(f"Memoization:      {memo:>8} {'✓' if memo == expected else '✗'}")
        print(f"DP (2D):          {dp_result:>8} {'✓' if dp_result == expected else '✗'}")
        print(f"Space Optimized:  {space_opt:>8} {'✓' if space_opt == expected else '✗'}")
        
        # Show path for small cases
        if len(grid) <= 3 and len(grid[0]) <= 3:
            result, path = max_product_path_with_path([row[:] for row in grid])
            if path:
                print(f"Optimal path: {path}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    max_product_path_analysis([[1,-2,1],[1,-2,1],[3,-4,5]])
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. TRACK BOTH MAX AND MIN: Negative numbers can turn min into max")
    print("2. MULTIPLICATION EFFECTS: Sign changes affect optimal choices")
    print("3. ZERO HANDLING: Zeros can reset products advantageously")
    print("4. MODULO ARITHMETIC: Large products need modular reduction")
    print("5. NEGATIVE RESULT: Return -1 if maximum product is negative")
    
    print("\n" + "=" * 70)
    print("Algorithm Comparison:")
    print("Brute Force:      Try all possible paths")
    print("Memoization:      Top-down with max/min tracking")
    print("DP (2D):          Bottom-up with dual DP tables")
    print("Space Optimized:  Row-by-row processing")
    print("With Path:        DP + path reconstruction")
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Brute Force:      Time: O(2^(m+n)), Space: O(m+n)")
    print("Memoization:      Time: O(m*n),     Space: O(m*n)")
    print("DP (2D):          Time: O(m*n),     Space: O(m*n)")
    print("Space Optimized:  Time: O(m*n),     Space: O(n)")
    print("With Path:        Time: O(m*n),     Space: O(m*n)")


if __name__ == "__main__":
    test_max_product_path()


"""
PATTERN RECOGNITION:
==================
Maximum Non Negative Product is a sophisticated path optimization problem:
- Combines path traversal with product optimization
- Requires tracking both maximum AND minimum products (negative numbers!)
- Demonstrates importance of sign consideration in optimization
- Shows modular arithmetic application in competitive programming

KEY INSIGHT - DUAL TRACKING NECESSITY:
======================================
**Why Track Both Max and Min?**
- Multiplication by negative number flips the relationship
- Current minimum × negative = potential new maximum
- Current maximum × negative = potential new minimum
- Must consider all possibilities at each step

**Example**: 
```
max = 10, min = -20, current = -3
new_max = max(-3*10, -3*(-20)) = max(-30, 60) = 60
new_min = min(-3*10, -3*(-20)) = min(-30, 60) = -30
```

ALGORITHM APPROACHES:
====================

1. **Dual DP Tables**: O(m×n) time, O(m×n) space
   - Maintain separate tables for maximum and minimum products
   - Consider all four combinations at each step
   - Most systematic approach

2. **Space Optimized**: O(m×n) time, O(n) space
   - Use rolling arrays for max/min products
   - Process row by row to reduce space

3. **Memoization**: O(m×n) time, O(m×n) space
   - Top-down approach returning (max, min) tuples
   - Natural recursive decomposition

4. **Brute Force**: O(2^(m+n)) time
   - Try all possible paths
   - Exponential complexity

DP STATE DEFINITION:
===================
```
dp_max[i][j] = maximum product of path ending at (i,j)
dp_min[i][j] = minimum product of path ending at (i,j)
```

RECURRENCE RELATION:
===================
```
current = grid[i][j]

candidates = [
    current * dp_max[i-1][j],  # From top (using max)
    current * dp_min[i-1][j],  # From top (using min)
    current * dp_max[i][j-1],  # From left (using max)
    current * dp_min[i][j-1]   # From left (using min)
]

dp_max[i][j] = max(candidates)
dp_min[i][j] = min(candidates)
```

**Why Four Candidates?**
- If current > 0: prefer max from previous positions
- If current < 0: prefer min from previous positions (becomes max)
- If current = 0: product becomes 0 regardless

SIGN ANALYSIS:
=============
**Current Value Effects**:
- **Positive**: Preserves relative ordering (max stays max, min stays min)
- **Negative**: Flips relative ordering (max becomes min, min becomes max)
- **Zero**: Resets both max and min to 0

**Strategic Implications**:
- Negative numbers can be beneficial if multiplied by negative product
- Zeros can be helpful to "reset" from bad negative products
- Even count of negatives in path gives positive product

MODULAR ARITHMETIC:
==================
**Why MOD 10^9 + 7?**
- Products can become extremely large (up to 4^225 for max grid)
- Need to prevent integer overflow
- Standard competitive programming modulus

**Implementation**:
```python
MOD = 10**9 + 7
return max_product % MOD if max_product >= 0 else -1
```

EDGE CASES:
==========
1. **All negative path**: Return -1 (no non-negative product possible)
2. **Zero in path**: Can give product 0 (non-negative)
3. **Single element**: Return element if non-negative, else -1
4. **Even negative count**: Can achieve positive product
5. **Odd negative count**: Best case is negative (return -1)

SPACE OPTIMIZATION:
==================
**Rolling Arrays**: Reduce O(m×n) to O(n)
```python
max_prev = [0] * n  # Previous row max products
min_prev = [0] * n  # Previous row min products

for each row:
    max_curr = [0] * n  # Current row max products
    min_curr = [0] * n  # Current row min products
    # Compute current row using previous row
    max_prev, min_prev = max_curr, min_curr
```

PATH RECONSTRUCTION:
===================
To find actual optimal path:
1. **Track parent pointers** during DP computation
2. **Choose parent** that led to maximum product
3. **Reconstruct backwards** from destination to start

```python
# During DP
max_val, best_source = max(candidates_with_sources)
dp_max[i][j] = max_val
parent[i][j] = best_source

# Reconstruction
path = []
current = (m-1, n-1)
while current:
    path.append(current)
    current = parent[current[0]][current[1]]
path.reverse()
```

APPLICATIONS:
============
1. **Financial Modeling**: Portfolio optimization with leverage
2. **Signal Processing**: Maximum likelihood path estimation
3. **Game Theory**: Optimal strategy with multiplicative rewards
4. **Physics**: Path integrals with complex coefficients

VARIANTS TO PRACTICE:
====================
- Maximum Product Subarray (152) - 1D version
- Minimum Path Sum (64) - additive instead of multiplicative
- Unique Paths (62) - counting instead of optimization
- Cherry Pickup (741) - bidirectional path optimization

INTERVIEW TIPS:
==============
1. **Recognize dual tracking need**: Key insight for negative numbers
2. **Explain sign effects**: How multiplication changes relationships
3. **Handle modular arithmetic**: Large number considerations
4. **Show space optimization**: 2D → 1D reduction technique
5. **Trace negative example**: Demonstrate max/min flipping
6. **Discuss zero handling**: Special case analysis
7. **Path reconstruction**: How to find actual optimal path
8. **Edge cases**: All negative, single element, zeros
9. **Complexity analysis**: Why O(mn) with dual tracking
10. **Real applications**: Finance, signal processing, game theory

MATHEMATICAL INSIGHT:
====================
This problem demonstrates **sign-sensitive optimization**:
- **Multiplication** fundamentally different from **addition**
- **Negative values** create **non-monotonic behavior**
- **Dual state tracking** necessary for **complete solution space coverage**

The need to track both maximum and minimum reflects the mathematical reality 
that in multiplicative contexts, extremes can interchange based on sign.
"""
