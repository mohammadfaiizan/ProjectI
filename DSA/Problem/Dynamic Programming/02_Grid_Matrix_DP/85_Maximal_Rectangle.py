"""
LeetCode 85: Maximal Rectangle
Difficulty: Hard
Category: Grid/Matrix DP

PROBLEM DESCRIPTION:
===================
Given a rows x cols binary matrix filled with 0's and 1's, find the largest rectangle containing 
only 1's and return its area.

Example 1:
Input: matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
Output: 6
Explanation: The maximal rectangle is shown in the above picture.

Example 2:
Input: matrix = [["0"]]
Output: 0

Example 3:
Input: matrix = [["1"]]
Output: 1

Constraints:
- rows == matrix.length
- cols == matrix[i].length
- 1 <= rows, cols <= 200
- matrix[i][j] is '0' or '1'.
"""

def maximal_rectangle_brute_force(matrix):
    """
    BRUTE FORCE APPROACH:
    ====================
    Check all possible rectangles in the matrix.
    
    Time Complexity: O(m^2 * n^2 * m * n) - six nested loops
    Space Complexity: O(1) - constant extra space
    """
    if not matrix or not matrix[0]:
        return 0
    
    m, n = len(matrix), len(matrix[0])
    max_area = 0
    
    # Try all possible top-left corners
    for r1 in range(m):
        for c1 in range(n):
            # Try all possible bottom-right corners
            for r2 in range(r1, m):
                for c2 in range(c1, n):
                    # Check if rectangle contains only 1s
                    valid = True
                    for i in range(r1, r2 + 1):
                        for j in range(c1, c2 + 1):
                            if matrix[i][j] == '0':
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        area = (r2 - r1 + 1) * (c2 - c1 + 1)
                        max_area = max(max_area, area)
    
    return max_area


def maximal_rectangle_histogram_dp(matrix):
    """
    HISTOGRAM DP APPROACH:
    =====================
    Convert each row to histogram and find largest rectangle in histogram.
    
    Time Complexity: O(m * n) - process each cell once
    Space Complexity: O(n) - height array
    """
    if not matrix or not matrix[0]:
        return 0
    
    m, n = len(matrix), len(matrix[0])
    heights = [0] * n  # Heights of histogram bars
    max_area = 0
    
    def largest_rectangle_in_histogram(heights):
        """Find largest rectangle in histogram using stack"""
        stack = []
        max_area = 0
        
        for i in range(len(heights)):
            while stack and heights[i] < heights[stack[-1]]:
                h = heights[stack.pop()]
                w = i if not stack else i - stack[-1] - 1
                max_area = max(max_area, h * w)
            stack.append(i)
        
        while stack:
            h = heights[stack.pop()]
            w = len(heights) if not stack else len(heights) - stack[-1] - 1
            max_area = max(max_area, h * w)
        
        return max_area
    
    # Process each row
    for i in range(m):
        for j in range(n):
            if matrix[i][j] == '1':
                heights[j] += 1
            else:
                heights[j] = 0
        
        # Find largest rectangle in current histogram
        area = largest_rectangle_in_histogram(heights)
        max_area = max(max_area, area)
    
    return max_area


def maximal_rectangle_dp_optimized(matrix):
    """
    OPTIMIZED DP APPROACH:
    =====================
    Use DP arrays to track left, right boundaries and heights.
    
    Time Complexity: O(m * n) - process each cell once
    Space Complexity: O(n) - DP arrays
    """
    if not matrix or not matrix[0]:
        return 0
    
    m, n = len(matrix), len(matrix[0])
    
    # DP arrays
    heights = [0] * n    # Height of consecutive 1s ending at current row
    lefts = [0] * n      # Left boundary of rectangle ending at current position
    rights = [n] * n     # Right boundary of rectangle ending at current position
    
    max_area = 0
    
    for i in range(m):
        # Update heights
        for j in range(n):
            if matrix[i][j] == '1':
                heights[j] += 1
            else:
                heights[j] = 0
        
        # Update left boundaries
        cur_left = 0
        for j in range(n):
            if matrix[i][j] == '1':
                lefts[j] = max(lefts[j], cur_left)
            else:
                lefts[j] = 0
                cur_left = j + 1
        
        # Update right boundaries
        cur_right = n
        for j in range(n - 1, -1, -1):
            if matrix[i][j] == '1':
                rights[j] = min(rights[j], cur_right)
            else:
                rights[j] = n
                cur_right = j
        
        # Calculate max area for current row
        for j in range(n):
            if heights[j] > 0:
                width = rights[j] - lefts[j]
                area = heights[j] * width
                max_area = max(max_area, area)
    
    return max_area


def maximal_rectangle_stack_optimized(matrix):
    """
    STACK OPTIMIZED APPROACH:
    =========================
    Optimized version using monotonic stack for each row.
    
    Time Complexity: O(m * n) - each element pushed/popped once per row
    Space Complexity: O(n) - stack and height array
    """
    if not matrix or not matrix[0]:
        return 0
    
    m, n = len(matrix), len(matrix[0])
    heights = [0] * n
    max_area = 0
    
    def largest_rectangle_optimized(heights):
        stack = [-1]  # Sentinel for easier width calculation
        max_area = 0
        
        for i in range(len(heights)):
            while len(stack) > 1 and heights[i] < heights[stack[-1]]:
                h = heights[stack.pop()]
                w = i - stack[-1] - 1
                max_area = max(max_area, h * w)
            stack.append(i)
        
        while len(stack) > 1:
            h = heights[stack.pop()]
            w = len(heights) - stack[-1] - 1
            max_area = max(max_area, h * w)
        
        return max_area
    
    for i in range(m):
        # Update heights for current row
        for j in range(n):
            heights[j] = heights[j] + 1 if matrix[i][j] == '1' else 0
        
        # Find largest rectangle in current histogram
        area = largest_rectangle_optimized(heights)
        max_area = max(max_area, area)
    
    return max_area


def maximal_rectangle_with_details(matrix):
    """
    FIND RECTANGLE WITH DETAILS:
    ============================
    Return area and coordinates of largest rectangle.
    
    Time Complexity: O(m * n) - DP computation
    Space Complexity: O(n) - DP arrays
    """
    if not matrix or not matrix[0]:
        return 0, (-1, -1, -1, -1)
    
    m, n = len(matrix), len(matrix[0])
    
    heights = [0] * n
    max_area = 0
    best_coords = (-1, -1, -1, -1)
    
    def largest_rectangle_with_coords(heights, row):
        stack = []
        max_area = 0
        best_coords = (-1, -1, -1, -1)
        
        for i in range(len(heights)):
            while stack and heights[i] < heights[stack[-1]]:
                h = heights[stack.pop()]
                w = i if not stack else i - stack[-1] - 1
                area = h * w
                
                if area > max_area:
                    max_area = area
                    left = 0 if not stack else stack[-1] + 1
                    right = i - 1
                    top = row - h + 1
                    bottom = row
                    best_coords = (top, left, bottom, right)
            
            stack.append(i)
        
        while stack:
            h = heights[stack.pop()]
            w = len(heights) if not stack else len(heights) - stack[-1] - 1
            area = h * w
            
            if area > max_area:
                max_area = area
                left = 0 if not stack else stack[-1] + 1
                right = len(heights) - 1
                top = row - h + 1
                bottom = row
                best_coords = (top, left, bottom, right)
        
        return max_area, best_coords
    
    for i in range(m):
        # Update heights
        for j in range(n):
            heights[j] = heights[j] + 1 if matrix[i][j] == '1' else 0
        
        # Find largest rectangle in current histogram
        area, coords = largest_rectangle_with_coords(heights, i)
        
        if area > max_area:
            max_area = area
            best_coords = coords
    
    return max_area, best_coords


def maximal_rectangle_analysis(matrix):
    """
    DETAILED ANALYSIS:
    =================
    Show step-by-step computation and rectangle identification.
    
    Time Complexity: O(m * n) - analysis computation
    Space Complexity: O(n) - temporary arrays
    """
    if not matrix or not matrix[0]:
        print("Empty matrix!")
        return 0
    
    m, n = len(matrix), len(matrix[0])
    
    print("Input Matrix:")
    for i, row in enumerate(matrix):
        print(f"  Row {i}: {row}")
    
    print(f"\nHistogram Heights Analysis:")
    
    heights = [0] * n
    max_area = 0
    best_row = -1
    
    for i in range(m):
        # Update heights
        print(f"\nRow {i}:")
        for j in range(n):
            if matrix[i][j] == '1':
                heights[j] += 1
            else:
                heights[j] = 0
        
        print(f"  Heights: {heights}")
        
        # Calculate area for this histogram
        def largest_rectangle_debug(heights):
            stack = []
            max_area = 0
            operations = []
            
            for i in range(len(heights)):
                while stack and heights[i] < heights[stack[-1]]:
                    h = heights[stack.pop()]
                    w = i if not stack else i - stack[-1] - 1
                    area = h * w
                    max_area = max(max_area, area)
                    operations.append(f"    Pop height {h}, width {w}, area {area}")
                
                stack.append(i)
                operations.append(f"    Push index {i} (height {heights[i]})")
            
            while stack:
                h = heights[stack.pop()]
                w = len(heights) if not stack else len(heights) - stack[-1] - 1
                area = h * w
                max_area = max(max_area, area)
                operations.append(f"    Final: height {h}, width {w}, area {area}")
            
            return max_area, operations
        
        area, operations = largest_rectangle_debug(heights)
        print(f"  Histogram processing:")
        for op in operations:
            print(op)
        print(f"  Max area in this row: {area}")
        
        if area > max_area:
            max_area = area
            best_row = i
    
    print(f"\nOverall Results:")
    print(f"  Maximum area: {max_area}")
    print(f"  Best row: {best_row}")
    
    # Show the actual rectangle
    area, coords = maximal_rectangle_with_details([row[:] for row in matrix])
    if coords != (-1, -1, -1, -1):
        top, left, bottom, right = coords
        print(f"  Rectangle coordinates: ({top},{left}) to ({bottom},{right})")
        print(f"  Rectangle size: {bottom-top+1} × {right-left+1}")
    
    return max_area


def maximal_rectangle_visualize(matrix):
    """
    VISUALIZE SOLUTION:
    ==================
    Show the matrix with largest rectangle highlighted.
    """
    if not matrix or not matrix[0]:
        return 0
    
    area, coords = maximal_rectangle_with_details([row[:] for row in matrix])
    
    if coords == (-1, -1, -1, -1):
        print("No rectangle found!")
        return 0
    
    m, n = len(matrix), len(matrix[0])
    top, left, bottom, right = coords
    
    # Create visualization matrix
    visual = [['.' if matrix[i][j] == '0' else '1' for j in range(n)] for i in range(m)]
    
    # Highlight the maximal rectangle
    for i in range(top, bottom + 1):
        for j in range(left, right + 1):
            visual[i][j] = 'X'
    
    print("Matrix with maximal rectangle highlighted (X):")
    for i, row in enumerate(visual):
        print(f"  Row {i}: {' '.join(row)}")
    
    print(f"Rectangle: ({top},{left}) to ({bottom},{right})")
    print(f"Size: {bottom-top+1} × {right-left+1} = {area}")
    
    return area


# Test cases
def test_maximal_rectangle():
    """Test all implementations with various inputs"""
    test_cases = [
        ([["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]], 6),
        ([["0"]], 0),
        ([["1"]], 1),
        ([["1","1"],["1","1"]], 4),
        ([["0","0"],["0","0"]], 0),
        ([["1","1","1"],["1","1","1"]], 6),
        ([["1","0","1","1","1"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]], 8),
        ([["1","1","1","1"],["1","1","1","1"],["1","1","1","1"]], 12),
        ([["0","1","1","0","1"],["1","1","0","1","0"],["0","1","1","1","0"],["1","1","1","1","0"],["1","1","1","1","1"]], 6)
    ]
    
    print("Testing Maximal Rectangle Solutions:")
    print("=" * 70)
    
    for i, (matrix, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Matrix: {matrix}")
        print(f"Expected: {expected}")
        
        # Test approaches (skip expensive ones for large inputs)
        if len(matrix) <= 3 and len(matrix[0]) <= 3:
            try:
                brute = maximal_rectangle_brute_force([row[:] for row in matrix])
                print(f"Brute Force:      {brute:>5} {'✓' if brute == expected else '✗'}")
            except:
                print(f"Brute Force:      Timeout")
        
        histogram_dp = maximal_rectangle_histogram_dp([row[:] for row in matrix])
        dp_opt = maximal_rectangle_dp_optimized([row[:] for row in matrix])
        stack_opt = maximal_rectangle_stack_optimized([row[:] for row in matrix])
        
        print(f"Histogram DP:     {histogram_dp:>5} {'✓' if histogram_dp == expected else '✗'}")
        print(f"DP Optimized:     {dp_opt:>5} {'✓' if dp_opt == expected else '✗'}")
        print(f"Stack Optimized:  {stack_opt:>5} {'✓' if stack_opt == expected else '✗'}")
        
        # Show details for small cases
        if len(matrix) <= 4 and len(matrix[0]) <= 5:
            area, coords = maximal_rectangle_with_details([row[:] for row in matrix])
            if coords != (-1, -1, -1, -1):
                print(f"Rectangle: {coords}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    maximal_rectangle_analysis([["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]])
    
    # Visualization example
    print(f"\n" + "=" * 70)
    print("VISUALIZATION EXAMPLE:")
    print("-" * 40)
    maximal_rectangle_visualize([["1","1","1"],["1","1","1"],["1","1","0"]])
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. HISTOGRAM REDUCTION: Convert 2D problem to multiple 1D problems")
    print("2. MONOTONIC STACK: Efficient largest rectangle in histogram")
    print("3. HEIGHT TRACKING: Accumulate heights row by row")
    print("4. DP OPTIMIZATION: Track left/right boundaries efficiently")
    print("5. AMORTIZED ANALYSIS: Each element processed constant times")
    
    print("\n" + "=" * 70)
    print("Algorithm Comparison:")
    print("Brute Force:      Check all possible rectangles")
    print("Histogram DP:     Row-wise histogram + stack")
    print("DP Optimized:     Track boundaries with DP arrays")
    print("Stack Optimized:  Improved stack implementation")
    print("With Details:     DP + coordinate tracking")
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Brute Force:      Time: O(m²n²mn),   Space: O(1)")
    print("Histogram DP:     Time: O(mn),       Space: O(n)")
    print("DP Optimized:     Time: O(mn),       Space: O(n)")
    print("Stack Optimized:  Time: O(mn),       Space: O(n)")
    print("With Details:     Time: O(mn),       Space: O(n)")


if __name__ == "__main__":
    test_maximal_rectangle()


"""
PATTERN RECOGNITION:
==================
Maximal Rectangle is a sophisticated 2D optimization problem:
- Reduces 2D rectangle problem to multiple 1D histogram problems
- Combines dynamic programming with stack-based algorithms
- Demonstrates problem decomposition and reduction techniques
- Foundation for computational geometry and image processing

KEY INSIGHT - HISTOGRAM REDUCTION:
=================================
**Problem Transformation**:
- For each row, treat it as base of histogram
- Heights = consecutive 1s ending at current row
- Find largest rectangle in each histogram
- Take maximum across all rows

**Why This Works**:
- Any rectangle in 2D matrix corresponds to rectangle in some histogram
- Histogram heights capture vertical extent at each column
- Largest rectangle in histogram gives optimal width×height trade-off

ALGORITHM APPROACHES:
====================

1. **Histogram + Stack (Optimal)**: O(m×n) time, O(n) space
   - Convert each row to histogram problem
   - Use monotonic stack for O(n) histogram solution
   - Most elegant and efficient approach

2. **DP with Boundaries**: O(m×n) time, O(n) space
   - Track heights, left boundaries, right boundaries
   - Update boundaries dynamically for each row
   - Alternative O(mn) solution

3. **Brute Force**: O(m²×n²×m×n) time, O(1) space
   - Check all possible rectangles
   - Exponential complexity, only for small inputs

HISTOGRAM ALGORITHM DETAILS:
===========================

**Height Calculation**:
```python
for j in range(n):
    if matrix[i][j] == '1':
        heights[j] += 1
    else:
        heights[j] = 0
```

**Largest Rectangle in Histogram (Stack)**:
```python
def largest_rectangle(heights):
    stack = []
    max_area = 0
    
    for i in range(len(heights)):
        while stack and heights[i] < heights[stack[-1]]:
            h = heights[stack.pop()]
            w = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, h * w)
        stack.append(i)
    
    # Process remaining elements
    while stack:
        h = heights[stack.pop()]
        w = len(heights) if not stack else len(heights) - stack[-1] - 1
        max_area = max(max_area, h * w)
    
    return max_area
```

MONOTONIC STACK TECHNIQUE:
=========================
**Stack Invariant**: Indices in stack correspond to increasing heights

**Key Operations**:
1. **Push**: When current height ≥ stack top height
2. **Pop**: When current height < stack top height
   - Calculate rectangle with popped height
   - Width = distance between current position and new stack top

**Width Calculation Logic**:
- If stack empty after pop: width = current_index
- Otherwise: width = current_index - stack_top - 1

DP BOUNDARY TRACKING:
====================
**Alternative Approach**: Track boundaries explicitly

```python
heights[j] = heights from previous row + 1 if matrix[i][j] == '1' else 0
lefts[j] = leftmost column that can extend to current height
rights[j] = rightmost column that can extend to current height

area = heights[j] * (rights[j] - lefts[j])
```

**Boundary Update Rules**:
- Left boundary: max(previous_left[j], current_left_boundary)
- Right boundary: min(previous_right[j], current_right_boundary)

COMPLEXITY ANALYSIS:
===================
**Time Complexity**: O(m×n)
- Each matrix element processed exactly once
- Each histogram bar pushed/popped exactly once per row
- Amortized O(1) per element

**Space Complexity**: O(n)
- Heights array: O(n)
- Stack: O(n) worst case
- No additional space proportional to m

EDGE CASES:
==========
1. **Empty matrix**: Return 0
2. **All zeros**: Return 0
3. **All ones**: Return m×n
4. **Single row/column**: Reduces to 1D problem
5. **Single cell**: Return 1 if '1', else 0

APPLICATIONS:
============
1. **Computer Vision**: Object detection, region analysis
2. **VLSI Design**: Chip layout optimization
3. **Image Processing**: Rectangle detection algorithms
4. **Computational Geometry**: Largest empty rectangle
5. **Database Systems**: Range query optimization

VARIANTS TO PRACTICE:
====================
- Largest Rectangle in Histogram (84) - 1D foundation
- Maximal Square (221) - constrained to squares only
- Largest Rectangle in Binary Matrix - direct variant
- Maximum Rectangle with Cost - optimization with weights

INTERVIEW TIPS:
==============
1. **Start with histogram insight**: Key reduction technique
2. **Explain stack algorithm**: Why monotonic stack works
3. **Show height calculation**: How to build histogram per row
4. **Trace small example**: Demonstrate stack operations
5. **Discuss alternatives**: DP boundary tracking approach
6. **Handle edge cases**: Empty matrix, all zeros/ones
7. **Complexity analysis**: Why O(mn) is optimal
8. **Real applications**: Computer vision, VLSI design
9. **Problem reduction**: 2D → multiple 1D problems
10. **Follow-up questions**: Maximal square, 3D version

OPTIMIZATION OPPORTUNITIES:
==========================
1. **Early termination**: If current max area equals m×n
2. **Row preprocessing**: Skip rows with all zeros
3. **Memory optimization**: Reuse arrays between iterations
4. **Parallel processing**: Independent histogram computations

MATHEMATICAL INSIGHT:
====================
This problem beautifully demonstrates **problem reduction**:
- Complex 2D optimization → Series of 1D optimizations
- Geometric intuition → Algorithmic technique
- **Divide and conquer** in the spatial dimension

The monotonic stack technique captures the essence of rectangle formation:
maintaining increasing heights while efficiently computing areas when constraints are violated.
"""
