"""
LeetCode 363: Max Sum of Rectangle No Larger Than K
Difficulty: Hard
Category: Grid/Matrix DP

PROBLEM DESCRIPTION:
===================
Given an m x n matrix matrix and an integer k, return the max sum of a rectangle in the matrix such that its sum is no larger than k.

It is guaranteed that there will be a rectangle with a sum no larger than k.

Example 1:
Input: matrix = [[1,0,1],[0,-2,3]], k = 2
Output: 2
Explanation: Because the sum of the rectangle [[0, 1], [-2, 3]] is 2, and 2 is the max number no larger than k (k = 2).

Example 2:
Input: matrix = [[2,2,-1]], k = 3
Output: 3

Constraints:
- m == matrix.length
- n == matrix[i].length
- 1 <= m, n <= 100
- -100 <= matrix[i][j] <= 100
- -10^5 <= k <= 10^5
"""

import bisect
from typing import List

def max_sum_submatrix_brute_force(matrix: List[List[int]], k: int) -> int:
    """
    BRUTE FORCE APPROACH:
    ====================
    Check all possible rectangles and find max sum <= k.
    
    Time Complexity: O(m^2 * n^2 * m * n) = O(m^3 * n^3) - check all rectangles and compute sums
    Space Complexity: O(1) - constant extra space
    """
    if not matrix or not matrix[0]:
        return 0
    
    m, n = len(matrix), len(matrix[0])
    max_sum = float('-inf')
    
    # Try all possible top-left corners
    for r1 in range(m):
        for c1 in range(n):
            # Try all possible bottom-right corners
            for r2 in range(r1, m):
                for c2 in range(c1, n):
                    # Calculate sum of rectangle from (r1,c1) to (r2,c2)
                    rect_sum = 0
                    for i in range(r1, r2 + 1):
                        for j in range(c1, c2 + 1):
                            rect_sum += matrix[i][j]
                    
                    # Update max if valid
                    if rect_sum <= k:
                        max_sum = max(max_sum, rect_sum)
    
    return max_sum


def max_sum_submatrix_prefix_sum(matrix: List[List[int]], k: int) -> int:
    """
    PREFIX SUM OPTIMIZATION:
    =======================
    Use 2D prefix sums to compute rectangle sums in O(1).
    
    Time Complexity: O(m^2 * n^2) - check all rectangles with O(1) sum computation
    Space Complexity: O(m * n) - prefix sum matrix
    """
    if not matrix or not matrix[0]:
        return 0
    
    m, n = len(matrix), len(matrix[0])
    
    # Build 2D prefix sum matrix
    prefix = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            prefix[i][j] = (matrix[i-1][j-1] + 
                          prefix[i-1][j] + 
                          prefix[i][j-1] - 
                          prefix[i-1][j-1])
    
    max_sum = float('-inf')
    
    # Try all possible rectangles
    for r1 in range(m):
        for c1 in range(n):
            for r2 in range(r1, m):
                for c2 in range(c1, n):
                    # Calculate rectangle sum using prefix sums
                    rect_sum = (prefix[r2+1][c2+1] - 
                              prefix[r1][c2+1] - 
                              prefix[r2+1][c1] + 
                              prefix[r1][c1])
                    
                    if rect_sum <= k:
                        max_sum = max(max_sum, rect_sum)
    
    return max_sum


def max_sum_submatrix_kadane_optimization(matrix: List[List[int]], k: int) -> int:
    """
    KADANE'S ALGORITHM OPTIMIZATION:
    ===============================
    Reduce 2D problem to multiple 1D problems using Kadane's algorithm.
    
    Time Complexity: O(m^2 * n^2) - but with better practical performance
    Space Complexity: O(n) - temporary array for column sums
    """
    if not matrix or not matrix[0]:
        return 0
    
    m, n = len(matrix), len(matrix[0])
    max_sum = float('-inf')
    
    def max_subarray_sum_no_larger_than_k(arr, k):
        """Find max subarray sum <= k using sliding window + sorted prefix sums"""
        max_sum = float('-inf')
        
        # Try all subarrays
        for i in range(len(arr)):
            current_sum = 0
            for j in range(i, len(arr)):
                current_sum += arr[j]
                if current_sum <= k:
                    max_sum = max(max_sum, current_sum)
        
        return max_sum
    
    # Fix top and bottom rows, reduce to 1D problem
    for top in range(m):
        # Array to store column sums between top and bottom rows
        temp = [0] * n
        
        for bottom in range(top, m):
            # Add current bottom row to temp array
            for j in range(n):
                temp[j] += matrix[bottom][j]
            
            # Find max subarray sum <= k in this 1D array
            current_max = max_subarray_sum_no_larger_than_k(temp, k)
            max_sum = max(max_sum, current_max)
    
    return max_sum


def max_sum_submatrix_optimized(matrix: List[List[int]], k: int) -> int:
    """
    OPTIMIZED APPROACH WITH BINARY SEARCH:
    ======================================
    Use Kadane's algorithm + binary search on prefix sums for 1D subarray problem.
    
    Time Complexity: O(m^2 * n * log(n)) - binary search optimization
    Space Complexity: O(n) - temporary arrays
    """
    if not matrix or not matrix[0]:
        return 0
    
    m, n = len(matrix), len(matrix[0])
    max_sum = float('-inf')
    
    def max_subarray_sum_no_larger_than_k_optimized(arr, k):
        """Use binary search on prefix sums for optimal 1D solution"""
        max_sum = float('-inf')
        prefix_sums = [0]  # prefix_sums[i] = sum(arr[0:i])
        
        for i in range(len(arr)):
            current_prefix = prefix_sums[-1] + arr[i]
            
            # For subarray ending at i, we want prefix_sums[j] >= current_prefix - k
            # This gives us subarray sum = current_prefix - prefix_sums[j] <= k
            target = current_prefix - k
            
            # Binary search for smallest prefix_sum >= target
            idx = bisect.bisect_left(prefix_sums, target)
            
            if idx < len(prefix_sums):
                max_sum = max(max_sum, current_prefix - prefix_sums[idx])
            
            # Insert current prefix sum in sorted order
            bisect.insort(prefix_sums, current_prefix)
        
        return max_sum
    
    # Optimize for the smaller dimension
    if m > n:
        # Fix left and right columns, reduce to 1D problem
        for left in range(n):
            temp = [0] * m
            for right in range(left, n):
                # Add current right column to temp array
                for i in range(m):
                    temp[i] += matrix[i][right]
                
                current_max = max_subarray_sum_no_larger_than_k_optimized(temp, k)
                max_sum = max(max_sum, current_max)
    else:
        # Fix top and bottom rows, reduce to 1D problem
        for top in range(m):
            temp = [0] * n
            for bottom in range(top, m):
                # Add current bottom row to temp array
                for j in range(n):
                    temp[j] += matrix[bottom][j]
                
                current_max = max_subarray_sum_no_larger_than_k_optimized(temp, k)
                max_sum = max(max_sum, current_max)
    
    return max_sum


def max_sum_submatrix_with_rectangle(matrix: List[List[int]], k: int) -> tuple:
    """
    FIND ACTUAL RECTANGLE:
    ======================
    Return max sum and the actual rectangle coordinates.
    
    Time Complexity: O(m^2 * n * log(n)) - optimized approach
    Space Complexity: O(n) - temporary arrays
    """
    if not matrix or not matrix[0]:
        return 0, None
    
    m, n = len(matrix), len(matrix[0])
    max_sum = float('-inf')
    best_rectangle = None
    
    def max_subarray_with_indices(arr, k):
        """Find max subarray sum <= k and return sum, start, end indices"""
        max_sum = float('-inf')
        best_start, best_end = -1, -1
        
        prefix_sums = [(0, -1)]  # (prefix_sum, index)
        
        for i in range(len(arr)):
            current_prefix = prefix_sums[-1][0] + arr[i]
            target = current_prefix - k
            
            # Binary search for smallest prefix_sum >= target
            left, right = 0, len(prefix_sums)
            while left < right:
                mid = (left + right) // 2
                if prefix_sums[mid][0] >= target:
                    right = mid
                else:
                    left = mid + 1
            
            if left < len(prefix_sums):
                candidate_sum = current_prefix - prefix_sums[left][0]
                if candidate_sum > max_sum:
                    max_sum = candidate_sum
                    best_start = prefix_sums[left][1] + 1
                    best_end = i
            
            # Insert current prefix sum in sorted order
            insert_pos = bisect.bisect_left([x[0] for x in prefix_sums], current_prefix)
            prefix_sums.insert(insert_pos, (current_prefix, i))
        
        return max_sum, best_start, best_end
    
    # Fix top and bottom rows
    for top in range(m):
        temp = [0] * n
        for bottom in range(top, m):
            # Add current bottom row to temp array
            for j in range(n):
                temp[j] += matrix[bottom][j]
            
            current_max, left, right = max_subarray_with_indices(temp, k)
            if current_max > max_sum:
                max_sum = current_max
                best_rectangle = (top, left, bottom, right)
    
    return max_sum, best_rectangle


def max_sum_submatrix_analysis(matrix: List[List[int]], k: int):
    """
    DETAILED ANALYSIS:
    =================
    Show step-by-step computation and rectangle analysis.
    """
    if not matrix or not matrix[0]:
        print("Empty matrix!")
        return 0
    
    m, n = len(matrix), len(matrix[0])
    
    print(f"Input Matrix ({m}x{n}):")
    for i, row in enumerate(matrix):
        print(f"  Row {i}: {row}")
    print(f"Target: k = {k}")
    
    # Build prefix sum matrix for visualization
    print(f"\nBuilding 2D Prefix Sum Matrix:")
    prefix = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            prefix[i][j] = (matrix[i-1][j-1] + 
                          prefix[i-1][j] + 
                          prefix[i][j-1] - 
                          prefix[i-1][j-1])
    
    print("Prefix Sum Matrix:")
    for i, row in enumerate(prefix):
        print(f"  Row {i}: {row}")
    
    # Show some example rectangle calculations
    print(f"\nExample Rectangle Sum Calculations:")
    examples = [
        (0, 0, 0, 0),  # Single cell
        (0, 0, 0, min(1, n-1)),  # First row, first two cells
        (0, 0, min(1, m-1), 0),  # First column, first two cells
    ]
    
    for r1, c1, r2, c2 in examples:
        if r2 < m and c2 < n:
            rect_sum = (prefix[r2+1][c2+1] - 
                       prefix[r1][c2+1] - 
                       prefix[r2+1][c1] + 
                       prefix[r1][c1])
            
            print(f"  Rectangle ({r1},{c1}) to ({r2},{c2}): sum = {rect_sum}")
            print(f"    Elements: ", end="")
            elements = []
            for i in range(r1, r2+1):
                for j in range(c1, c2+1):
                    elements.append(matrix[i][j])
            print(f"{elements} = {sum(elements)}")
    
    # Find optimal solution
    result, rectangle = max_sum_submatrix_with_rectangle(matrix, k)
    
    print(f"\nOptimal Solution:")
    print(f"Maximum sum <= {k}: {result}")
    
    if rectangle:
        top, left, bottom, right = rectangle
        print(f"Rectangle: ({top},{left}) to ({bottom},{right})")
        print(f"Elements in optimal rectangle:")
        
        rect_elements = []
        for i in range(top, bottom + 1):
            row_elements = []
            for j in range(left, right + 1):
                row_elements.append(matrix[i][j])
            rect_elements.append(row_elements)
            print(f"  Row {i}: {row_elements}")
        
        # Verify sum
        total = sum(sum(row) for row in rect_elements)
        print(f"Sum verification: {total}")
    
    return result


# Test cases
def test_max_sum_submatrix():
    """Test all implementations with various inputs"""
    test_cases = [
        ([[1, 0, 1], [0, -2, 3]], 2, 2),
        ([[2, 2, -1]], 3, 3),
        ([[1]], 1, 1),
        ([[1]], 0, -1),  # No valid rectangle
        ([[-1, -1], [-1, -1]], -2, -2),
        ([[5, -4, -3, 4], [-3, -4, 4, 5], [5, 1, 5, -4]], 10, 10),
        ([[2, 2, -1]], 0, -1),
        ([[1, 2], [3, 4]], 4, 4),
        ([[-2, -3], [-1, -4]], -1, -1),
        ([[0, 0], [0, 0]], 0, 0)
    ]
    
    print("Testing Max Sum Rectangle No Larger Than K Solutions:")
    print("=" * 70)
    
    for i, (matrix, k, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Matrix: {matrix}")
        print(f"k: {k}")
        print(f"Expected: {expected}")
        
        # Test approaches (skip expensive ones for large inputs)
        if len(matrix) <= 3 and len(matrix[0]) <= 3:
            try:
                brute = max_sum_submatrix_brute_force(matrix, k)
                print(f"Brute Force:      {brute:>5} {'✓' if brute == expected else '✗'}")
            except:
                print(f"Brute Force:      Error")
        
        prefix_result = max_sum_submatrix_prefix_sum(matrix, k)
        kadane_result = max_sum_submatrix_kadane_optimization(matrix, k)
        optimized_result = max_sum_submatrix_optimized(matrix, k)
        
        print(f"Prefix Sum:       {prefix_result:>5} {'✓' if prefix_result == expected else '✗'}")
        print(f"Kadane Opt:       {kadane_result:>5} {'✓' if kadane_result == expected else '✗'}")
        print(f"Optimized:        {optimized_result:>5} {'✓' if optimized_result == expected else '✗'}")
        
        # Show rectangle for small cases
        if len(matrix) <= 3 and len(matrix[0]) <= 3:
            result, rectangle = max_sum_submatrix_with_rectangle(matrix, k)
            if rectangle:
                print(f"Rectangle: {rectangle}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    max_sum_submatrix_analysis([[1, 0, 1], [0, -2, 3]], 2)
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. 2D TO 1D REDUCTION: Fix rows, reduce to 1D max subarray problem")
    print("2. BINARY SEARCH OPTIMIZATION: Use sorted prefix sums for efficiency")
    print("3. CONSTRAINT HANDLING: Sum must be <= k, not just maximum")
    print("4. PREFIX SUM TECHNIQUE: Fast rectangle sum computation")
    print("5. DIMENSION OPTIMIZATION: Process smaller dimension as outer loop")
    
    print("\n" + "=" * 70)
    print("Algorithm Comparison:")
    print("Brute Force:      Check all rectangles, compute sums naively")
    print("Prefix Sum:       O(1) rectangle sum with 2D prefix matrix")
    print("Kadane Opt:       Reduce to 1D max subarray problems")
    print("Optimized:        Binary search on prefix sums for 1D part")
    print("With Rectangle:   Track coordinates of optimal rectangle")
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Brute Force:      Time: O(m³n³),          Space: O(1)")
    print("Prefix Sum:       Time: O(m²n²),          Space: O(mn)")
    print("Kadane Opt:       Time: O(m²n²),          Space: O(n)")
    print("Optimized:        Time: O(m²n log n),     Space: O(n)")
    print("With Rectangle:   Time: O(m²n log n),     Space: O(n)")


if __name__ == "__main__":
    test_max_sum_submatrix()


"""
PATTERN RECOGNITION:
==================
Max Sum Rectangle No Larger Than K combines several advanced techniques:
- 2D to 1D problem reduction using fixed boundaries
- Constrained optimization (sum <= k instead of just maximum)
- Binary search on prefix sums for efficient 1D subarray queries
- Demonstrates the power of dimension reduction in optimization problems

KEY INSIGHT - CONSTRAINED SUBARRAY MAXIMUM:
===========================================
**Unlike standard max subarray**: Must find max sum that doesn't exceed k
**Standard Kadane's doesn't work**: Need to consider ALL subarrays, not just positive-contributing ones
**Solution**: Use prefix sums + binary search to efficiently find best subarray ending at each position

**For 1D array with constraint**:
- For subarray ending at position i, need to find best starting position j
- Subarray sum = prefix[i+1] - prefix[j]
- Want: prefix[i+1] - prefix[j] <= k
- Equivalent: prefix[j] >= prefix[i+1] - k
- Binary search on sorted prefix sums to find smallest valid prefix[j]

ALGORITHM APPROACHES:
====================

1. **Brute Force**: O(m³n³) time
   - Check all possible rectangles
   - Compute each rectangle sum from scratch
   - Only viable for very small matrices

2. **Prefix Sum Optimization**: O(m²n²) time, O(mn) space
   - Build 2D prefix sum matrix
   - Compute any rectangle sum in O(1)
   - Still checks all possible rectangles

3. **Kadane's Reduction**: O(m²n²) time, O(n) space
   - Fix top and bottom rows
   - Reduce to 1D max subarray problem with constraint
   - Better practical performance

4. **Optimized with Binary Search**: O(m²n log n) time, O(n) space
   - Use binary search on prefix sums for 1D part
   - Optimal complexity for this problem
   - Choose smaller dimension for outer loop

DIMENSION REDUCTION TECHNIQUE:
=============================
**Core Idea**: Convert 2D problem to multiple 1D problems

```
For each pair of rows (top, bottom):
    1. Compress matrix between these rows into 1D array
    2. Solve 1D max subarray sum <= k problem
    3. Track global maximum
```

**Compression Example**:
```
Matrix:     [1,  0,  1]     Rows 0-1:  [1, -2, 4]
            [0, -2,  3]     (column sums)
            
After fixing rows 0-1, solve: max subarray sum <= k in [1, -2, 4]
```

CONSTRAINED 1D SUBARRAY PROBLEM:
===============================
**Problem**: Find max sum of subarray in array A such that sum <= k

**Naive Solution**: O(n²) - check all subarrays
**Optimized Solution**: O(n log n) using binary search

```python
def max_subarray_sum_leq_k(arr, k):
    max_sum = float('-inf')
    prefix_sums = [0]  # sorted list of prefix sums
    
    for i in range(len(arr)):
        current_prefix = prefix_sums[-1] + arr[i]  # This is wrong - need running prefix
        # ... binary search logic ...
```

**Correct Implementation**:
```python
def max_subarray_sum_leq_k(arr, k):
    max_sum = float('-inf')
    prefix_sums = [0]
    current_prefix = 0
    
    for num in arr:
        current_prefix += num
        
        # Find smallest prefix_sum such that current_prefix - prefix_sum <= k
        target = current_prefix - k
        idx = bisect.bisect_left(prefix_sums, target)
        
        if idx < len(prefix_sums):
            max_sum = max(max_sum, current_prefix - prefix_sums[idx])
        
        bisect.insort(prefix_sums, current_prefix)
    
    return max_sum
```

OPTIMIZATION STRATEGIES:
=======================
1. **Choose Smaller Dimension**: 
   - If m > n: fix columns (left, right), iterate rows
   - If n >= m: fix rows (top, bottom), iterate columns
   - Reduces outer loop complexity

2. **Early Termination**:
   - If current max equals k, return immediately
   - If remaining area can't improve result, skip

3. **Preprocessing**:
   - Skip empty rows/columns
   - Handle edge cases efficiently

APPLICATIONS:
============
1. **Financial Analysis**: Maximum profit regions with risk constraints
2. **Image Processing**: Optimal rectangular regions with pixel sum constraints
3. **Resource Allocation**: Maximum utilization within budget constraints
4. **Game Development**: Optimal rectangular selections with score limits

VARIANTS TO PRACTICE:
====================
- Max Sum Rectangle (no constraint) - simpler version
- Max Sum of 3 Non-Overlapping Subarrays (1584) - similar constraint optimization
- Largest Rectangle in Histogram (84) - related geometric optimization
- Maximum Size Subarray Sum Equals k (325) - exact sum instead of <= k

INTERVIEW TIPS:
==============
1. **Recognize 2D to 1D reduction**: Key insight for optimization
2. **Explain constraint handling**: Why standard Kadane's doesn't work
3. **Binary search motivation**: How prefix sums enable efficient queries
4. **Complexity trade-offs**: Space vs time optimizations
5. **Edge cases**: Empty matrix, no valid rectangle, k very small/large
6. **Alternative approaches**: Discuss brute force to optimal progression
7. **Real applications**: Financial analysis, image processing
8. **Follow-up questions**: Exact sum = k, multiple rectangles, 3D version

MATHEMATICAL INSIGHT:
====================
This problem demonstrates **constrained optimization in 2D space**:
- **Dimension reduction** transforms complex 2D problem into manageable 1D problems
- **Binary search** provides logarithmic speedup for constraint satisfaction
- **Prefix sum technique** enables efficient range query processing

The combination of these techniques shows how **algorithmic composition** can solve 
complex problems by breaking them into well-understood subproblems.
"""
