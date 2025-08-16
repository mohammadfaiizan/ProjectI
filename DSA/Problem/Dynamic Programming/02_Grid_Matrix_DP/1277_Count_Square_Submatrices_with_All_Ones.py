"""
LeetCode 1277: Count Square Submatrices with All Ones
Difficulty: Medium
Category: Grid/Matrix DP

PROBLEM DESCRIPTION:
===================
Given a m * n matrix of ones and zeros, return how many square submatrices have all ones.

Example 1:
Input: matrix = [
  [0,1,1,1],
  [1,1,1,1],
  [0,1,1,1]
]
Output: 15
Explanation: 
There are 10 squares of side 1.
There are 4 squares of side 2.
There are 1 square of side 3.
Total number of squares = 10 + 4 + 1 = 15.

Example 2:
Input: matrix = [
  [1,0,1],
  [1,1,0],
  [1,1,0]
]
Output: 7
Explanation: 
There are 6 squares of side 1.  
There are 1 square of side 2.
Total number of squares = 6 + 1 = 7.

Constraints:
- 1 <= arr.length <= 300
- 1 <= arr[0].length <= 300
- 0 <= arr[i][j] <= 1
"""

def count_squares_brute_force(matrix):
    """
    BRUTE FORCE APPROACH:
    ====================
    Check all possible squares in the matrix.
    
    Time Complexity: O(m*n*min(m,n)^3) - check all positions, sizes, and validate
    Space Complexity: O(1) - constant extra space
    """
    if not matrix or not matrix[0]:
        return 0
    
    m, n = len(matrix), len(matrix[0])
    count = 0
    
    def is_valid_square(row, col, size):
        """Check if square of given size starting at (row,col) contains only 1s"""
        if row + size > m or col + size > n:
            return False
        
        for i in range(row, row + size):
            for j in range(col, col + size):
                if matrix[i][j] == 0:
                    return False
        return True
    
    # Try all positions as top-left corner
    for i in range(m):
        for j in range(n):
            if matrix[i][j] == 1:
                # Try all possible square sizes
                max_size = min(m - i, n - j)
                for size in range(1, max_size + 1):
                    if is_valid_square(i, j, size):
                        count += 1
                    else:
                        break  # No larger squares possible
    
    return count


def count_squares_dp(matrix):
    """
    DYNAMIC PROGRAMMING APPROACH:
    ============================
    Use DP to count squares ending at each position.
    
    Time Complexity: O(m*n) - process each cell once
    Space Complexity: O(m*n) - DP table
    """
    if not matrix or not matrix[0]:
        return 0
    
    m, n = len(matrix), len(matrix[0])
    
    # dp[i][j] = side length of largest square with bottom-right corner at (i,j)
    dp = [[0] * n for _ in range(m)]
    count = 0
    
    for i in range(m):
        for j in range(n):
            if matrix[i][j] == 1:
                if i == 0 or j == 0:
                    # First row or column - can only form 1x1 square
                    dp[i][j] = 1
                else:
                    # Take minimum of three neighbors and add 1
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
                
                # Add all squares ending at this position
                count += dp[i][j]
    
    return count


def count_squares_space_optimized(matrix):
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
    
    # Use single array to represent previous row
    prev = [0] * n
    count = 0
    
    for i in range(m):
        curr = [0] * n
        for j in range(n):
            if matrix[i][j] == 1:
                if i == 0 or j == 0:
                    curr[j] = 1
                else:
                    curr[j] = min(prev[j], curr[j-1], prev[j-1]) + 1
                
                count += curr[j]
        
        prev = curr
    
    return count


def count_squares_in_place(matrix):
    """
    IN-PLACE APPROACH:
    =================
    Modify the input matrix itself to save space.
    
    Time Complexity: O(m*n) - process each cell once
    Space Complexity: O(1) - no extra space
    """
    if not matrix or not matrix[0]:
        return 0
    
    m, n = len(matrix), len(matrix[0])
    count = 0
    
    for i in range(m):
        for j in range(n):
            if matrix[i][j] == 1:
                if i == 0 or j == 0:
                    # First row or column - keep as 1
                    pass
                else:
                    # Update with minimum of neighbors + 1
                    matrix[i][j] = min(matrix[i-1][j], matrix[i][j-1], matrix[i-1][j-1]) + 1
                
                count += matrix[i][j]
    
    return count


def count_squares_with_breakdown(matrix):
    """
    COUNT WITH SIZE BREAKDOWN:
    =========================
    Return total count and breakdown by square size.
    
    Time Complexity: O(m*n) - DP computation
    Space Complexity: O(m*n + min(m,n)) - DP table + size counts
    """
    if not matrix or not matrix[0]:
        return 0, {}
    
    m, n = len(matrix), len(matrix[0])
    max_possible_size = min(m, n)
    
    dp = [[0] * n for _ in range(m)]
    size_counts = {}
    total_count = 0
    
    for i in range(m):
        for j in range(n):
            if matrix[i][j] == 1:
                if i == 0 or j == 0:
                    dp[i][j] = 1
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
                
                # Count squares of each size ending at this position
                max_size = dp[i][j]
                for size in range(1, max_size + 1):
                    size_counts[size] = size_counts.get(size, 0) + 1
                    total_count += 1
    
    return total_count, size_counts


def count_squares_all_positions(matrix):
    """
    FIND ALL SQUARE POSITIONS:
    ==========================
    Return count and all square positions by size.
    
    Time Complexity: O(m*n*total_squares) - DP + enumeration
    Space Complexity: O(m*n + total_squares) - DP table + positions
    """
    if not matrix or not matrix[0]:
        return 0, {}
    
    m, n = len(matrix), len(matrix[0])
    
    dp = [[0] * n for _ in range(m)]
    squares_by_size = {}
    total_count = 0
    
    for i in range(m):
        for j in range(n):
            if matrix[i][j] == 1:
                if i == 0 or j == 0:
                    dp[i][j] = 1
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
                
                # Record all squares ending at this position
                max_size = dp[i][j]
                for size in range(1, max_size + 1):
                    if size not in squares_by_size:
                        squares_by_size[size] = []
                    
                    # Calculate top-left corner
                    top_left = (i - size + 1, j - size + 1)
                    bottom_right = (i, j)
                    squares_by_size[size].append((top_left, bottom_right))
                    total_count += 1
    
    return total_count, squares_by_size


def count_squares_analysis(matrix):
    """
    DETAILED ANALYSIS:
    =================
    Show step-by-step DP computation and square counting.
    
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
    running_count = 0
    
    for i in range(m):
        for j in range(n):
            if matrix[i][j] == 1:
                if i == 0 or j == 0:
                    dp[i][j] = 1
                    print(f"  dp[{i}][{j}] = 1 (border cell)")
                else:
                    left = dp[i][j-1]
                    top = dp[i-1][j]
                    diag = dp[i-1][j-1]
                    dp[i][j] = min(left, top, diag) + 1
                    print(f"  dp[{i}][{j}] = min({left}, {top}, {diag}) + 1 = {dp[i][j]}")
                
                squares_here = dp[i][j]
                running_count += squares_here
                print(f"    Squares ending at ({i},{j}): {squares_here}, Total so far: {running_count}")
            else:
                dp[i][j] = 0
                print(f"  dp[{i}][{j}] = 0 (matrix value is 0)")
    
    print(f"\nFinal DP table:")
    for i, row in enumerate(dp):
        print(f"  Row {i}: {row}")
    
    # Show breakdown by size
    total_count, size_breakdown = count_squares_with_breakdown([row[:] for row in matrix])
    print(f"\nSquare count breakdown:")
    for size in sorted(size_breakdown.keys()):
        count = size_breakdown[size]
        print(f"  Size {size}x{size}: {count} squares")
    
    print(f"\nTotal squares: {total_count}")
    
    return total_count


def count_squares_visualize(matrix):
    """
    VISUALIZE SOLUTION:
    ==================
    Show the matrix with DP values and highlight squares.
    """
    if not matrix or not matrix[0]:
        return 0
    
    m, n = len(matrix), len(matrix[0])
    
    # Calculate DP values
    dp = [[0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            if matrix[i][j] == 1:
                if i == 0 or j == 0:
                    dp[i][j] = 1
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
    
    print("Original Matrix:")
    for i, row in enumerate(matrix):
        print(f"  Row {i}: {row}")
    
    print("\nDP Matrix (max square size ending at each position):")
    for i, row in enumerate(dp):
        print(f"  Row {i}: {row}")
    
    # Show some example squares
    total_count, squares_by_size = count_squares_all_positions([row[:] for row in matrix])
    
    print(f"\nExample squares by size:")
    for size in sorted(squares_by_size.keys()):
        squares = squares_by_size[size]
        print(f"  Size {size}x{size}: {len(squares)} squares")
        # Show first few squares of each size
        for i, (top_left, bottom_right) in enumerate(squares[:3]):
            print(f"    Square {i+1}: {top_left} to {bottom_right}")
        if len(squares) > 3:
            print(f"    ... and {len(squares) - 3} more")
    
    return total_count


# Test cases
def test_count_squares():
    """Test all implementations with various inputs"""
    test_cases = [
        ([[0,1,1,1],[1,1,1,1],[0,1,1,1]], 15),
        ([[1,0,1],[1,1,0],[1,1,0]], 7),
        ([[0,1,1,1],[1,1,1,1],[0,1,1,1]], 15),
        ([[1]], 1),
        ([[0]], 0),
        ([[1,1],[1,1]], 6),
        ([[0,0],[0,0]], 0),
        ([[1,1,1],[1,1,1],[1,1,1]], 14),
        ([[1,0,1,1,1],[1,0,1,1,1],[1,1,1,1,1],[1,0,0,1,0]], 13),
        ([[0,1,1,0,1],[1,1,0,1,0],[0,1,1,1,0],[1,1,1,1,0],[1,1,1,1,1]], 21)
    ]
    
    print("Testing Count Square Submatrices Solutions:")
    print("=" * 70)
    
    for i, (matrix, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Matrix: {matrix}")
        print(f"Expected: {expected}")
        
        # Test approaches (skip expensive ones for large inputs)
        if len(matrix) <= 4 and len(matrix[0]) <= 4:
            try:
                brute = count_squares_brute_force([row[:] for row in matrix])
                print(f"Brute Force:      {brute:>5} {'✓' if brute == expected else '✗'}")
            except:
                print(f"Brute Force:      Timeout")
        
        dp_result = count_squares_dp([row[:] for row in matrix])
        space_opt = count_squares_space_optimized([row[:] for row in matrix])
        
        print(f"DP (2D):          {dp_result:>5} {'✓' if dp_result == expected else '✗'}")
        print(f"Space Optimized:  {space_opt:>5} {'✓' if space_opt == expected else '✗'}")
        
        # Test in-place (modifies input)
        matrix_copy = [row[:] for row in matrix]
        in_place = count_squares_in_place(matrix_copy)
        print(f"In-place:         {in_place:>5} {'✓' if in_place == expected else '✗'}")
        
        # Show breakdown for small cases
        if len(matrix) <= 4 and len(matrix[0]) <= 4:
            total, breakdown = count_squares_with_breakdown([row[:] for row in matrix])
            print(f"Breakdown: {breakdown}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    count_squares_analysis([[0,1,1,1],[1,1,1,1],[0,1,1,1]])
    
    # Visualization example
    print(f"\n" + "=" * 70)
    print("VISUALIZATION EXAMPLE:")
    print("-" * 40)
    count_squares_visualize([[1,1,1],[1,1,1],[1,1,1]])
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. DP EXTENSION: Similar to Maximal Square but COUNT all squares")
    print("2. CONTRIBUTION: Each position contributes dp[i][j] squares")
    print("3. NESTED SQUARES: Square of size k contains squares of sizes 1,2,...,k")
    print("4. OPTIMAL SUBSTRUCTURE: Larger squares built from smaller ones")
    print("5. SPACE OPTIMIZATION: Same techniques as Maximal Square")
    
    print("\n" + "=" * 70)
    print("Algorithm Comparison:")
    print("Brute Force:      Check all possible squares")
    print("DP (2D):          Count squares ending at each position")
    print("Space Optimized:  Use single row array")
    print("In-place:         Modify input matrix")
    print("With Breakdown:   DP + size analysis")
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Brute Force:      Time: O(mn(min(m,n))³), Space: O(1)")
    print("DP (2D):          Time: O(mn),             Space: O(mn)")
    print("Space Optimized:  Time: O(mn),             Space: O(n)")
    print("In-place:         Time: O(mn),             Space: O(1)")
    print("With Breakdown:   Time: O(mn),             Space: O(mn)")


if __name__ == "__main__":
    test_count_squares()


"""
PATTERN RECOGNITION:
==================
Count Square Submatrices is an extension of Maximal Square problem:
- Instead of finding largest square, count ALL squares
- Same DP recurrence but different interpretation
- Each DP value represents contribution to total count
- Demonstrates how slight problem variation changes solution focus

KEY INSIGHT - COUNTING vs MAXIMIZING:
====================================
**Maximal Square**: dp[i][j] = side length of largest square ending at (i,j)
**Count Squares**: dp[i][j] = side length of largest square ending at (i,j)

**Critical Difference**: In counting version, dp[i][j] also represents the NUMBER of squares ending at (i,j)

**Why This Works**:
- If largest square at (i,j) has side k, then squares of sizes 1,2,...,k all end at (i,j)
- So dp[i][j] = k means k squares end at this position
- Total count = sum of all dp[i][j] values

MATHEMATICAL FORMULATION:
========================
**DP Recurrence** (same as Maximal Square):
```
if matrix[i][j] == 1:
    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
else:
    dp[i][j] = 0
```

**Counting Formula**:
```
total_squares = sum(dp[i][j] for all i,j)
```

**Intuition**: If dp[i][j] = k, then squares of sizes 1×1, 2×2, ..., k×k all have bottom-right corner at (i,j)

ALGORITHM APPROACHES:
====================

1. **2D DP (Standard)**: O(m×n) time, O(m×n) space
   - Build complete DP table
   - Sum all DP values for total count
   - Most straightforward approach

2. **Space Optimized**: O(m×n) time, O(n) space
   - Process row by row
   - Accumulate count while computing DP
   - Same optimization as Maximal Square

3. **In-Place**: O(m×n) time, O(1) space
   - Modify input matrix directly
   - Ultimate space optimization

4. **Brute Force**: O(m×n×min(m,n)³) time
   - Check all possible squares
   - Exponential in square size

COUNTING BREAKDOWN ANALYSIS:
===========================
**Size Distribution**: For dp[i][j] = k, position (i,j) contributes:
- 1 square of size k×k
- 1 square of size (k-1)×(k-1)
- ...
- 1 square of size 1×1
- Total: k squares

**Example**: dp[2][2] = 3 means position (2,2) contributes 3 squares:
- 1×1 square: (2,2) to (2,2)
- 2×2 square: (1,1) to (2,2)  
- 3×3 square: (0,0) to (2,2)

SPACE OPTIMIZATION TECHNIQUES:
=============================
**Level 1 - Rolling Array**: O(n) space
```python
prev = [0] * n
for i in range(m):
    curr = [0] * n
    for j in range(n):
        # Compute curr[j] using prev
        count += curr[j]
    prev = curr
```

**Level 2 - In-Place**: O(1) space
```python
for i in range(m):
    for j in range(n):
        if matrix[i][j] == 1 and i > 0 and j > 0:
            matrix[i][j] = min(matrix[i-1][j], matrix[i][j-1], matrix[i-1][j-1]) + 1
        count += matrix[i][j]
```

EDGE CASES:
==========
1. **Empty matrix**: Return 0
2. **All zeros**: Return 0
3. **All ones**: Return sum of triangle numbers
4. **Single cell**: Return 1 if matrix[0][0] == 1, else 0
5. **Single row/column**: Count of 1s in the row/column

MATHEMATICAL PROPERTIES:
=======================
**Maximum possible squares**: For m×n matrix of all 1s:
```
total = sum(min(i+1, j+1, m-i, n-j) for i in range(m) for j in range(n))
```

**Triangle number connection**: Each position contributes based on how many complete squares can fit ending at that position.

APPLICATIONS:
============
1. **Computer Vision**: Count feature patterns in images
2. **Game Development**: Count valid placement areas
3. **VLSI Design**: Count possible component placements
4. **Pattern Analysis**: Statistical analysis of square patterns

VARIANTS TO PRACTICE:
====================
- Maximal Square (221) - find largest instead of count
- Count Square Submatrices with All Equal Elements - generalized version
- Number of Submatrices That Sum to Target (1074) - sum constraint
- Count Corner Rectangles (750) - different geometric constraint

INTERVIEW TIPS:
==============
1. **Connect to Maximal Square**: Show relationship clearly
2. **Explain counting logic**: Why dp[i][j] equals number of squares
3. **Show space optimization**: 2D → 1D → in-place progression
4. **Handle edge cases**: Empty, all zeros, single cell
5. **Trace small example**: Demonstrate DP computation
6. **Discuss alternatives**: Brute force vs DP trade-offs
7. **Real applications**: Image processing, game development
8. **Mathematical insight**: Triangle numbers and combinatorics
9. **Complexity analysis**: Why O(mn) is optimal
10. **Follow-up questions**: Different shapes, 3D version

OPTIMIZATION OPPORTUNITIES:
==========================
1. **Early termination**: If remaining cells can't improve count
2. **Preprocessing**: Skip rows/columns with all zeros
3. **Parallel computation**: Independent row computations
4. **Memory access optimization**: Cache-friendly traversal

MATHEMATICAL INSIGHT:
====================
This problem demonstrates how **interpretation changes** can lead to different algorithms on the same core computation:

- **Same recurrence relation** as Maximal Square
- **Different aggregation** (sum vs max)
- **Different semantic meaning** (count vs size)

The DP value serves dual purposes:
1. **Size of largest square** ending at position
2. **Count of all squares** ending at position

This duality is a powerful pattern in DP problems where one computation serves multiple purposes.
"""
