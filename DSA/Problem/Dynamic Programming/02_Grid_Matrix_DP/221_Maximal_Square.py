"""
LeetCode 221: Maximal Square
Difficulty: Medium
Category: Grid/Matrix DP

PROBLEM DESCRIPTION:
===================
Given an m x n binary matrix filled with 0's and 1's, find the largest square containing only 1's 
and return its area.

Example 1:
Input: matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
Output: 4
Explanation: The largest square has side length 2, so area = 4.

Example 2:
Input: matrix = [["0","1"],["1","0"]]
Output: 1

Example 3:
Input: matrix = [["0"]]
Output: 0

Constraints:
- m == matrix.length
- n == matrix[i].length
- 1 <= m, n <= 300
- matrix[i][j] is '0' or '1'.
"""

def maximal_square_brute_force(matrix):
    """
    BRUTE FORCE APPROACH:
    ====================
    Check all possible squares starting from each position.
    
    Time Complexity: O(m*n*min(m,n)^2) - check all positions and sizes
    Space Complexity: O(1) - constant extra space
    """
    if not matrix or not matrix[0]:
        return 0
    
    m, n = len(matrix), len(matrix[0])
    max_side = 0
    
    def is_valid_square(row, col, side):
        """Check if square of given side starting at (row,col) contains only 1s"""
        if row + side > m or col + side > n:
            return False
        
        for i in range(row, row + side):
            for j in range(col, col + side):
                if matrix[i][j] == '0':
                    return False
        return True
    
    # Try all positions as top-left corner
    for i in range(m):
        for j in range(n):
            if matrix[i][j] == '1':
                # Try increasing square sizes
                side = 1
                while is_valid_square(i, j, side):
                    max_side = max(max_side, side)
                    side += 1
    
    return max_side * max_side


def maximal_square_dp(matrix):
    """
    DYNAMIC PROGRAMMING APPROACH:
    ============================
    Use DP to find largest square ending at each position.
    
    Time Complexity: O(m*n) - process each cell once
    Space Complexity: O(m*n) - DP table
    """
    if not matrix or not matrix[0]:
        return 0
    
    m, n = len(matrix), len(matrix[0])
    
    # dp[i][j] = side length of largest square with bottom-right corner at (i,j)
    dp = [[0] * n for _ in range(m)]
    max_side = 0
    
    # Fill DP table
    for i in range(m):
        for j in range(n):
            if matrix[i][j] == '1':
                if i == 0 or j == 0:
                    # First row or column - can only form 1x1 square
                    dp[i][j] = 1
                else:
                    # Take minimum of three neighbors and add 1
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
                
                max_side = max(max_side, dp[i][j])
    
    return max_side * max_side


def maximal_square_space_optimized(matrix):
    """
    SPACE OPTIMIZED APPROACH:
    =========================
    Use only O(n) space by processing row by row.
    
    Time Complexity: O(m*n) - process each cell once
    Space Complexity: O(n) - single row array
    """
    if not matrix or not matrix[0]:
        return 0
    
    m, n = len(matrix), len(matrix[0])
    
    # Use two arrays: previous row and current row
    prev = [0] * n
    curr = [0] * n
    max_side = 0
    
    for i in range(m):
        for j in range(n):
            if matrix[i][j] == '1':
                if i == 0 or j == 0:
                    curr[j] = 1
                else:
                    curr[j] = min(prev[j], curr[j-1], prev[j-1]) + 1
                
                max_side = max(max_side, curr[j])
            else:
                curr[j] = 0
        
        # Swap arrays for next iteration
        prev, curr = curr, prev
    
    return max_side * max_side


def maximal_square_optimized_1d(matrix):
    """
    FURTHER OPTIMIZED APPROACH:
    ===========================
    Use only O(n) space with single array and variable.
    
    Time Complexity: O(m*n) - process each cell once
    Space Complexity: O(n) - single array
    """
    if not matrix or not matrix[0]:
        return 0
    
    m, n = len(matrix), len(matrix[0])
    
    # Single array to represent previous row
    dp = [0] * (n + 1)  # Extra element to avoid boundary checks
    max_side = 0
    prev = 0  # To store dp[i-1][j-1]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            temp = dp[j]  # Store current dp[j] before updating
            
            if matrix[i-1][j-1] == '1':
                dp[j] = min(dp[j], dp[j-1], prev) + 1
                max_side = max(max_side, dp[j])
            else:
                dp[j] = 0
            
            prev = temp  # Update prev for next iteration
    
    return max_side * max_side


def maximal_square_with_details(matrix):
    """
    FIND SQUARE WITH DETAILS:
    =========================
    Return area, side length, and position of largest square.
    
    Time Complexity: O(m*n) - DP computation
    Space Complexity: O(m*n) - DP table
    """
    if not matrix or not matrix[0]:
        return 0, 0, (-1, -1)
    
    m, n = len(matrix), len(matrix[0])
    
    dp = [[0] * n for _ in range(m)]
    max_side = 0
    max_pos = (-1, -1)
    
    for i in range(m):
        for j in range(n):
            if matrix[i][j] == '1':
                if i == 0 or j == 0:
                    dp[i][j] = 1
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
                
                if dp[i][j] > max_side:
                    max_side = dp[i][j]
                    max_pos = (i, j)  # Bottom-right corner
    
    area = max_side * max_side
    return area, max_side, max_pos


def maximal_square_all_squares(matrix):
    """
    FIND ALL MAXIMAL SQUARES:
    =========================
    Find all squares with maximum side length.
    
    Time Complexity: O(m*n) - DP computation
    Space Complexity: O(m*n) - DP table + squares storage
    """
    if not matrix or not matrix[0]:
        return 0, []
    
    m, n = len(matrix), len(matrix[0])
    
    dp = [[0] * n for _ in range(m)]
    max_side = 0
    
    # Build DP table
    for i in range(m):
        for j in range(n):
            if matrix[i][j] == '1':
                if i == 0 or j == 0:
                    dp[i][j] = 1
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
                
                max_side = max(max_side, dp[i][j])
    
    # Find all squares with maximum side length
    max_squares = []
    for i in range(m):
        for j in range(n):
            if dp[i][j] == max_side:
                # Calculate top-left corner
                top_left = (i - max_side + 1, j - max_side + 1)
                bottom_right = (i, j)
                max_squares.append((top_left, bottom_right))
    
    return max_side * max_side, max_squares


def maximal_square_analysis(matrix):
    """
    DETAILED ANALYSIS:
    =================
    Show step-by-step DP computation and square identification.
    
    Time Complexity: O(m*n) - analysis computation
    Space Complexity: O(m*n) - temporary tables
    """
    if not matrix or not matrix[0]:
        print("Empty matrix!")
        return 0
    
    m, n = len(matrix), len(matrix[0])
    
    print("Input Matrix:")
    for i, row in enumerate(matrix):
        print(f"  Row {i}: {row}")
    
    print(f"\nDP Computation:")
    
    dp = [[0] * n for _ in range(m)]
    max_side = 0
    max_positions = []
    
    for i in range(m):
        for j in range(n):
            if matrix[i][j] == '1':
                if i == 0 or j == 0:
                    dp[i][j] = 1
                    print(f"  dp[{i}][{j}] = 1 (border)")
                else:
                    left = dp[i][j-1]
                    top = dp[i-1][j]
                    diag = dp[i-1][j-1]
                    dp[i][j] = min(left, top, diag) + 1
                    print(f"  dp[{i}][{j}] = min({left}, {top}, {diag}) + 1 = {dp[i][j]}")
                
                if dp[i][j] > max_side:
                    max_side = dp[i][j]
                    max_positions = [(i, j)]
                elif dp[i][j] == max_side and max_side > 0:
                    max_positions.append((i, j))
            else:
                dp[i][j] = 0
                print(f"  dp[{i}][{j}] = 0 (matrix value is '0')")
    
    print(f"\nFinal DP table:")
    for i, row in enumerate(dp):
        print(f"  Row {i}: {row}")
    
    print(f"\nResults:")
    print(f"  Maximum side length: {max_side}")
    print(f"  Maximum area: {max_side * max_side}")
    print(f"  Positions (bottom-right corners): {max_positions}")
    
    # Show actual squares
    if max_side > 0:
        print(f"\nMaximal squares:")
        for pos in max_positions:
            i, j = pos
            top_left = (i - max_side + 1, j - max_side + 1)
            print(f"  Square at {top_left} to {pos} (size {max_side}x{max_side})")
    
    return max_side * max_side


def maximal_square_visualize(matrix):
    """
    VISUALIZE SOLUTION:
    ==================
    Show the matrix with largest square highlighted.
    """
    if not matrix or not matrix[0]:
        return 0
    
    area, side, max_pos = maximal_square_with_details(matrix)
    
    if side == 0:
        print("No square found!")
        return 0
    
    m, n = len(matrix), len(matrix[0])
    
    # Create visualization matrix
    visual = [['.' if matrix[i][j] == '0' else '1' for j in range(n)] for i in range(m)]
    
    # Highlight the maximal square
    bottom_right_i, bottom_right_j = max_pos
    top_left_i = bottom_right_i - side + 1
    top_left_j = bottom_right_j - side + 1
    
    for i in range(top_left_i, bottom_right_i + 1):
        for j in range(top_left_j, bottom_right_j + 1):
            visual[i][j] = 'X'
    
    print("Matrix with maximal square highlighted (X):")
    for i, row in enumerate(visual):
        print(f"  Row {i}: {' '.join(row)}")
    
    print(f"Square: top-left ({top_left_i},{top_left_j}) to bottom-right ({bottom_right_i},{bottom_right_j})")
    print(f"Side length: {side}, Area: {area}")
    
    return area


# Test cases
def test_maximal_square():
    """Test all implementations with various inputs"""
    test_cases = [
        ([["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]], 4),
        ([["0","1"],["1","0"]], 1),
        ([["0"]], 0),
        ([["1"]], 1),
        ([["1","1"],["1","1"]], 4),
        ([["0","0","0"],["0","0","0"],["0","0","0"]], 0),
        ([["1","1","1"],["1","1","1"],["1","1","1"]], 9),
        ([["1","0","1","1","1"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]], 9),
        ([["1","1","1","1"],["1","1","1","1"],["1","1","1","1"]], 16),
        ([["0","1","1","0","1"],["1","1","0","1","0"],["0","1","1","1","0"],["1","1","1","1","0"],["1","1","1","1","1"],["0","0","0","0","0"]], 9)
    ]
    
    print("Testing Maximal Square Solutions:")
    print("=" * 70)
    
    for i, (matrix, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Matrix: {matrix}")
        print(f"Expected: {expected}")
        
        # Test approaches (skip expensive ones for large inputs)
        if len(matrix) <= 4 and len(matrix[0]) <= 4:
            try:
                brute = maximal_square_brute_force([row[:] for row in matrix])
                print(f"Brute Force:      {brute:>5} {'✓' if brute == expected else '✗'}")
            except:
                print(f"Brute Force:      Timeout")
        
        dp_result = maximal_square_dp([row[:] for row in matrix])
        space_opt = maximal_square_space_optimized([row[:] for row in matrix])
        optimized_1d = maximal_square_optimized_1d([row[:] for row in matrix])
        
        print(f"DP (2D):          {dp_result:>5} {'✓' if dp_result == expected else '✗'}")
        print(f"Space Optimized:  {space_opt:>5} {'✓' if space_opt == expected else '✗'}")
        print(f"Optimized 1D:     {optimized_1d:>5} {'✓' if optimized_1d == expected else '✗'}")
        
        # Show details for small cases
        if len(matrix) <= 4 and len(matrix[0]) <= 4:
            area, side, pos = maximal_square_with_details([row[:] for row in matrix])
            if side > 0:
                print(f"Details: side={side}, position={pos}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    maximal_square_analysis([["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]])
    
    # Visualization example
    print(f"\n" + "=" * 70)
    print("VISUALIZATION EXAMPLE:")
    print("-" * 40)
    maximal_square_visualize([["1","1","1"],["1","1","1"],["1","1","1"]])
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. DP STATE: dp[i][j] = side length of largest square ending at (i,j)")
    print("2. RECURRENCE: dp[i][j] = min(left, top, diagonal) + 1")
    print("3. SQUARE PROPERTY: All three neighbors must form squares")
    print("4. SPACE OPTIMIZATION: Can reduce to O(n) space")
    print("5. BOUNDARY: First row/column can only form 1x1 squares")
    
    print("\n" + "=" * 70)
    print("Algorithm Comparison:")
    print("Brute Force:      Check all possible squares")
    print("DP (2D):          Bottom-up with 2D table")
    print("Space Optimized:  Use two arrays (prev/curr)")
    print("Optimized 1D:     Single array with variable")
    print("With Details:     DP + position tracking")
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Brute Force:      Time: O(mn(min(m,n))²), Space: O(1)")
    print("DP (2D):          Time: O(mn),             Space: O(mn)")
    print("Space Optimized:  Time: O(mn),             Space: O(n)")
    print("Optimized 1D:     Time: O(mn),             Space: O(n)")
    print("With Details:     Time: O(mn),             Space: O(mn)")


if __name__ == "__main__":
    test_maximal_square()


"""
PATTERN RECOGNITION:
==================
Maximal Square is a classic 2D DP problem with geometric constraints:
- Find largest square of 1s in binary matrix
- Key insight: square can only be formed if all three neighbors are squares
- Classic example of reducing complex geometric problem to simple recurrence
- Foundation for histogram-based rectangle problems

KEY INSIGHT - SQUARE FORMATION:
==============================
**Square Property**: A square of side k can be formed at position (i,j) if and only if:
- Current cell is '1'
- Left neighbor has square of side ≥ k-1
- Top neighbor has square of side ≥ k-1  
- Diagonal neighbor has square of side ≥ k-1

**Mathematical Formulation**:
```
dp[i][j] = side length of largest square with bottom-right corner at (i,j)

if matrix[i][j] == '1':
    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
else:
    dp[i][j] = 0
```

ALGORITHM APPROACHES:
====================

1. **2D DP (Standard)**: O(m×n) time, O(m×n) space
   - Build complete DP table
   - Track maximum side length seen
   - Most intuitive approach

2. **Space Optimized**: O(m×n) time, O(n) space  
   - Use two arrays: previous row and current row
   - Reduce space complexity significantly

3. **1D Optimized**: O(m×n) time, O(n) space
   - Single array with careful variable management
   - Ultimate space optimization

4. **Brute Force**: O(m×n×min(m,n)²) time
   - Check all positions and all possible square sizes
   - Exponential in square size

DP RECURRENCE EXPLANATION:
=========================

**Why min(left, top, diagonal) + 1?**

Consider position (i,j) with value '1':
- To form square of side k, need squares of side k-1 at:
  - (i-1, j) - directly above
  - (i, j-1) - directly left  
  - (i-1, j-1) - diagonally above-left

**Intuition**: 
- If any neighbor has smaller square, we're limited by that
- Taking minimum ensures all constraints satisfied
- +1 because current cell extends the square

SPACE OPTIMIZATION TECHNIQUES:
=============================

**Level 1 - Two Arrays**: O(2n) = O(n) space
```python
prev = [0] * n  # Previous row
curr = [0] * n  # Current row
# Process row by row, swap arrays
```

**Level 2 - Single Array**: O(n) space
```python
dp = [0] * (n + 1)  # Extra element avoids boundary checks
prev = 0  # Variable to store diagonal value
# Update array in-place with careful variable management
```

**Level 3 - In-place**: Not possible (would destroy input)

BOUNDARY CONDITIONS:
===================
- **First row**: dp[0][j] = 1 if matrix[0][j] == '1', else 0
- **First column**: dp[i][0] = 1 if matrix[i][0] == '1', else 0
- **General case**: Use recurrence relation

EDGE CASES:
==========
1. **Empty matrix**: Return 0
2. **All zeros**: Return 0  
3. **All ones**: Return min(m,n)²
4. **Single cell**: Return 1 if '1', else 0
5. **Single row/column**: Return 1 if any '1' exists

MATHEMATICAL PROPERTIES:
=======================
- **Maximum possible**: min(m,n)² when entire matrix is '1'
- **Monotonicity**: Larger squares require smaller squares as building blocks
- **Optimal substructure**: Optimal square contains optimal sub-squares

APPLICATIONS:
============
1. **Computer Graphics**: Rectangle/square detection in images
2. **VLSI Design**: Chip layout optimization
3. **Game Development**: Collision detection, area queries
4. **Data Compression**: Finding repeated patterns

VARIANTS TO PRACTICE:
====================
- Largest Rectangle in Histogram (84) - 1D version
- Maximal Rectangle (85) - general rectangle version  
- Count Square Submatrices with All Ones (1277) - counting version
- Number of Submatrices That Sum to Target (1074) - sum constraint

INTERVIEW TIPS:
==============
1. **Start with recurrence**: Explain the min() logic clearly
2. **Show space optimization**: 2D → 1D progression
3. **Handle edge cases**: Empty matrix, single cell, all zeros
4. **Trace small example**: Show DP table construction
5. **Discuss alternatives**: Brute force vs DP trade-offs
6. **Real applications**: Graphics, VLSI, game development
7. **Follow-up variations**: Rectangles, counting, different constraints
8. **Complexity analysis**: Why O(mn) is optimal
9. **Implementation details**: Boundary handling, indexing
10. **Mathematical insight**: Why geometric constraint leads to min()

OPTIMIZATION OPPORTUNITIES:
==========================
1. **Early termination**: If max possible area already found
2. **Preprocessing**: Skip rows/columns with all zeros
3. **Parallel computation**: Independent subproblems
4. **Cache optimization**: Memory-friendly traversal patterns

MATHEMATICAL INSIGHT:
====================
The problem demonstrates how **geometric constraints** can be elegantly 
encoded in DP recurrences. The min() operation captures the essence of 
square formation: **all constituent parts must be valid squares**.

This principle extends to many geometric optimization problems where 
local validity constraints propagate to global optimization.
"""
