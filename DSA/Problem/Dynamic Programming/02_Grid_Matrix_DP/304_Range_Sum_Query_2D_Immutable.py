"""
LeetCode 304: Range Sum Query 2D - Immutable
Difficulty: Medium
Category: Grid/Matrix DP

PROBLEM DESCRIPTION:
===================
Given a 2D matrix matrix, handle multiple queries of the following type:

Calculate the sum of the elements of matrix inside the rectangle defined by its upper left corner 
(row1, col1) and lower right corner (row2, col2).

Implement the NumMatrix class:
- NumMatrix(int[][] matrix) Initializes the object with the integer matrix matrix.
- int sumRegion(int row1, int col1, int row2, int col2) Returns the sum of the elements of matrix 
  inside the rectangle defined by its upper left corner (row1, col1) and lower right corner (row2, col2).

Example 1:
Input
["NumMatrix", "sumRegion", "sumRegion", "sumRegion"]
[[[[3, 0, 1, 4, 2], [5, 6, 3, 2, 1], [1, 2, 0, 1, 5], [4, 1, 0, 1, 7], [1, 0, 3, 0, 5]]], [2, 1, 4, 3], [1, 1, 2, 2], [1, 2, 2, 4]]
Output
[null, 8, 11, 12]

Explanation
NumMatrix numMatrix = new NumMatrix([[3, 0, 1, 4, 2], [5, 6, 3, 2, 1], [1, 2, 0, 1, 5], [4, 1, 0, 1, 7], [1, 0, 3, 0, 5]]);
numMatrix.sumRegion(2, 1, 4, 3); // return 8 (i.e sum of the red rectangle)
numMatrix.sumRegion(1, 1, 2, 2); // return 11 (i.e sum of the green rectangle)
numMatrix.sumRegion(1, 2, 2, 4); // return 12 (i.e sum of the blue rectangle)

Constraints:
- m == matrix.length
- n == matrix[i].length
- 1 <= m, n <= 200
- -10^5 <= matrix[i][j] <= 10^5
- 0 <= row1 <= row2 < m
- 0 <= col1 <= col2 < n
- At most 10^4 calls will be made to sumRegion.
"""

class NumMatrix_Naive:
    """
    NAIVE APPROACH:
    ==============
    Calculate sum for each query by iterating through the rectangle.
    
    Initialization: O(1)
    Query: O((row2-row1+1) * (col2-col1+1)) - iterate through rectangle
    Space: O(1) - no extra space
    """
    
    def __init__(self, matrix):
        self.matrix = matrix
        self.m = len(matrix)
        self.n = len(matrix[0]) if matrix else 0
    
    def sumRegion(self, row1, col1, row2, col2):
        total = 0
        for i in range(row1, row2 + 1):
            for j in range(col1, col2 + 1):
                total += self.matrix[i][j]
        return total


class NumMatrix_RowPrefix:
    """
    ROW PREFIX SUM APPROACH:
    =======================
    Precompute prefix sums for each row.
    
    Initialization: O(m*n) - compute all row prefix sums
    Query: O(row2-row1+1) - sum across rows
    Space: O(m*n) - prefix sum matrix
    """
    
    def __init__(self, matrix):
        if not matrix or not matrix[0]:
            return
        
        self.m = len(matrix)
        self.n = len(matrix[0])
        
        # Build row prefix sums
        self.row_prefix = [[0] * (self.n + 1) for _ in range(self.m)]
        
        for i in range(self.m):
            for j in range(self.n):
                self.row_prefix[i][j + 1] = self.row_prefix[i][j] + matrix[i][j]
    
    def sumRegion(self, row1, col1, row2, col2):
        total = 0
        for i in range(row1, row2 + 1):
            # Sum of elements in row i from col1 to col2
            row_sum = self.row_prefix[i][col2 + 1] - self.row_prefix[i][col1]
            total += row_sum
        return total


class NumMatrix_2DPrefix:
    """
    2D PREFIX SUM APPROACH (OPTIMAL):
    ================================
    Precompute 2D prefix sums for O(1) query time.
    
    Initialization: O(m*n) - compute 2D prefix sums
    Query: O(1) - constant time using inclusion-exclusion
    Space: O(m*n) - 2D prefix sum matrix
    """
    
    def __init__(self, matrix):
        if not matrix or not matrix[0]:
            return
        
        self.m = len(matrix)
        self.n = len(matrix[0])
        
        # Build 2D prefix sum matrix
        # prefix[i][j] = sum of rectangle from (0,0) to (i-1,j-1)
        self.prefix = [[0] * (self.n + 1) for _ in range(self.m + 1)]
        
        for i in range(1, self.m + 1):
            for j in range(1, self.n + 1):
                self.prefix[i][j] = (matrix[i-1][j-1] + 
                                   self.prefix[i-1][j] + 
                                   self.prefix[i][j-1] - 
                                   self.prefix[i-1][j-1])
    
    def sumRegion(self, row1, col1, row2, col2):
        # Use inclusion-exclusion principle
        return (self.prefix[row2 + 1][col2 + 1] - 
                self.prefix[row1][col2 + 1] - 
                self.prefix[row2 + 1][col1] + 
                self.prefix[row1][col1])


class NumMatrix_SpaceOptimized:
    """
    SPACE OPTIMIZED APPROACH:
    =========================
    Optimize space usage while maintaining good query performance.
    
    Initialization: O(m*n)
    Query: O(min(rows, cols)) - choose better dimension
    Space: O(m*n) - same as 2D prefix but with optimizations
    """
    
    def __init__(self, matrix):
        if not matrix or not matrix[0]:
            return
        
        self.matrix = matrix
        self.m = len(matrix)
        self.n = len(matrix[0])
        
        # Choose whether to optimize by rows or columns
        if self.m <= self.n:
            # Fewer rows: use row prefix sums
            self._build_row_prefix()
            self.query_method = "row"
        else:
            # Fewer columns: use column prefix sums
            self._build_col_prefix()
            self.query_method = "col"
    
    def _build_row_prefix(self):
        self.row_prefix = [[0] * (self.n + 1) for _ in range(self.m)]
        for i in range(self.m):
            for j in range(self.n):
                self.row_prefix[i][j + 1] = self.row_prefix[i][j] + self.matrix[i][j]
    
    def _build_col_prefix(self):
        self.col_prefix = [[0] * self.n for _ in range(self.m + 1)]
        for j in range(self.n):
            for i in range(self.m):
                self.col_prefix[i + 1][j] = self.col_prefix[i][j] + self.matrix[i][j]
    
    def sumRegion(self, row1, col1, row2, col2):
        if self.query_method == "row":
            total = 0
            for i in range(row1, row2 + 1):
                total += self.row_prefix[i][col2 + 1] - self.row_prefix[i][col1]
            return total
        else:
            total = 0
            for j in range(col1, col2 + 1):
                total += self.col_prefix[row2 + 1][j] - self.col_prefix[row1][j]
            return total


class NumMatrix_Mutable:
    """
    MUTABLE VERSION:
    ===============
    Support both range sum queries and updates.
    
    Initialization: O(m*n)
    Query: O(1)
    Update: O(m*n) - need to rebuild prefix sums
    Space: O(m*n)
    """
    
    def __init__(self, matrix):
        if not matrix or not matrix[0]:
            return
        
        self.matrix = [row[:] for row in matrix]  # Deep copy
        self.m = len(matrix)
        self.n = len(matrix[0])
        self._build_prefix()
    
    def _build_prefix(self):
        self.prefix = [[0] * (self.n + 1) for _ in range(self.m + 1)]
        
        for i in range(1, self.m + 1):
            for j in range(1, self.n + 1):
                self.prefix[i][j] = (self.matrix[i-1][j-1] + 
                                   self.prefix[i-1][j] + 
                                   self.prefix[i][j-1] - 
                                   self.prefix[i-1][j-1])
    
    def update(self, row, col, val):
        """Update matrix[row][col] to val"""
        self.matrix[row][col] = val
        self._build_prefix()  # Rebuild prefix sums
    
    def sumRegion(self, row1, col1, row2, col2):
        return (self.prefix[row2 + 1][col2 + 1] - 
                self.prefix[row1][col2 + 1] - 
                self.prefix[row2 + 1][col1] + 
                self.prefix[row1][col1])


class NumMatrix_Fenwick:
    """
    FENWICK TREE (BINARY INDEXED TREE) APPROACH:
    ===========================================
    Support efficient updates and queries using 2D Fenwick tree.
    
    Initialization: O(m*n*log(m)*log(n))
    Query: O(log(m)*log(n))
    Update: O(log(m)*log(n))
    Space: O(m*n)
    """
    
    def __init__(self, matrix):
        if not matrix or not matrix[0]:
            return
        
        self.matrix = [[0] * len(matrix[0]) for _ in range(len(matrix))]
        self.m = len(matrix)
        self.n = len(matrix[0])
        
        # Initialize Fenwick tree
        self.tree = [[0] * (self.n + 1) for _ in range(self.m + 1)]
        
        # Build tree by updating each element
        for i in range(self.m):
            for j in range(self.n):
                self.update(i, j, matrix[i][j])
    
    def update(self, row, col, val):
        """Update matrix[row][col] to val"""
        diff = val - self.matrix[row][col]
        self.matrix[row][col] = val
        
        # Update Fenwick tree
        i = row + 1
        while i <= self.m:
            j = col + 1
            while j <= self.n:
                self.tree[i][j] += diff
                j += j & (-j)  # Add lowest set bit
            i += i & (-i)  # Add lowest set bit
    
    def _query(self, row, col):
        """Get prefix sum from (0,0) to (row,col)"""
        total = 0
        i = row + 1
        while i > 0:
            j = col + 1
            while j > 0:
                total += self.tree[i][j]
                j -= j & (-j)  # Remove lowest set bit
            i -= i & (-i)  # Remove lowest set bit
        return total
    
    def sumRegion(self, row1, col1, row2, col2):
        # Use inclusion-exclusion principle
        return (self._query(row2, col2) - 
                self._query(row1 - 1, col2) - 
                self._query(row2, col1 - 1) + 
                self._query(row1 - 1, col1 - 1))


def analyze_2d_prefix_construction(matrix):
    """
    ANALYZE 2D PREFIX CONSTRUCTION:
    ==============================
    Show step-by-step construction of 2D prefix sum matrix.
    """
    if not matrix or not matrix[0]:
        return
    
    m, n = len(matrix), len(matrix[0])
    
    print("Original Matrix:")
    for i, row in enumerate(matrix):
        print(f"  Row {i}: {row}")
    
    print(f"\n2D Prefix Sum Construction:")
    
    # Build prefix sum step by step
    prefix = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            current = matrix[i-1][j-1]
            left = prefix[i][j-1]
            top = prefix[i-1][j]
            diag = prefix[i-1][j-1]
            
            prefix[i][j] = current + left + top - diag
            
            print(f"  prefix[{i}][{j}] = {current} + {left} + {top} - {diag} = {prefix[i][j]}")
    
    print(f"\nFinal Prefix Sum Matrix:")
    for i, row in enumerate(prefix):
        print(f"  Row {i}: {row}")
    
    return prefix


def test_query_examples(matrix, queries):
    """Test various query examples with different approaches"""
    
    print("Testing Query Examples:")
    print("=" * 50)
    
    # Initialize all approaches
    approaches = {
        "Naive": NumMatrix_Naive(matrix),
        "Row Prefix": NumMatrix_RowPrefix(matrix),
        "2D Prefix": NumMatrix_2DPrefix(matrix),
        "Space Optimized": NumMatrix_SpaceOptimized(matrix)
    }
    
    for i, (row1, col1, row2, col2) in enumerate(queries):
        print(f"\nQuery {i+1}: sumRegion({row1}, {col1}, {row2}, {col2})")
        
        results = {}
        for name, num_matrix in approaches.items():
            result = num_matrix.sumRegion(row1, col1, row2, col2)
            results[name] = result
            print(f"  {name:15}: {result}")
        
        # Verify all results are the same
        unique_results = set(results.values())
        if len(unique_results) == 1:
            print(f"  ✓ All approaches agree: {unique_results.pop()}")
        else:
            print(f"  ✗ Disagreement: {results}")


# Test cases
def test_num_matrix():
    """Test all implementations with various inputs"""
    
    # Test matrix from the problem
    matrix1 = [
        [3, 0, 1, 4, 2],
        [5, 6, 3, 2, 1],
        [1, 2, 0, 1, 5],
        [4, 1, 0, 1, 7],
        [1, 0, 3, 0, 5]
    ]
    
    queries1 = [
        (2, 1, 4, 3),  # Expected: 8
        (1, 1, 2, 2),  # Expected: 11
        (1, 2, 2, 4)   # Expected: 12
    ]
    
    print("Testing 2D Range Sum Query Solutions:")
    print("=" * 70)
    
    # Show 2D prefix construction
    print("2D PREFIX CONSTRUCTION ANALYSIS:")
    print("-" * 40)
    analyze_2d_prefix_construction(matrix1)
    
    # Test queries
    print(f"\n" + "=" * 70)
    test_query_examples(matrix1, queries1)
    
    # Test edge cases
    print(f"\n" + "=" * 70)
    print("EDGE CASES:")
    print("-" * 20)
    
    edge_cases = [
        # Single element
        ([[5]], [(0, 0, 0, 0)], [5]),
        
        # Single row
        ([[1, 2, 3, 4]], [(0, 1, 0, 2)], [5]),
        
        # Single column  
        ([[1], [2], [3]], [(0, 0, 2, 0)], [6]),
        
        # Negative numbers
        ([[-1, 2], [3, -4]], [(0, 0, 1, 1)], [0]),
        
        # All zeros
        ([[0, 0], [0, 0]], [(0, 0, 1, 1)], [0])
    ]
    
    for i, (matrix, queries, expected) in enumerate(edge_cases):
        print(f"\nEdge Case {i+1}: {matrix}")
        num_matrix = NumMatrix_2DPrefix(matrix)
        
        for j, (r1, c1, r2, c2) in enumerate(queries):
            result = num_matrix.sumRegion(r1, c1, r2, c2)
            exp = expected[j]
            print(f"  Query ({r1},{c1},{r2},{c2}): {result} {'✓' if result == exp else '✗'}")
    
    # Performance comparison
    print(f"\n" + "=" * 70)
    print("PERFORMANCE COMPARISON:")
    print("-" * 30)
    
    import time
    
    # Large matrix for performance testing
    large_matrix = [[i * j for j in range(100)] for i in range(100)]
    large_queries = [(i, i, i+10, i+10) for i in range(0, 80, 10)]
    
    approaches = [
        ("Naive", NumMatrix_Naive),
        ("Row Prefix", NumMatrix_RowPrefix), 
        ("2D Prefix", NumMatrix_2DPrefix)
    ]
    
    for name, cls in approaches:
        start_time = time.time()
        
        # Initialization
        num_matrix = cls(large_matrix)
        init_time = time.time() - start_time
        
        # Queries
        query_start = time.time()
        for query in large_queries:
            num_matrix.sumRegion(*query)
        query_time = time.time() - query_start
        
        print(f"{name:15}: Init={init_time:.4f}s, Queries={query_time:.4f}s")
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. 2D PREFIX OPTIMAL: O(1) queries after O(mn) preprocessing")
    print("2. INCLUSION-EXCLUSION: sum = total - left - top + diagonal")
    print("3. SPACE-TIME TRADEOFF: More space for faster queries")
    print("4. IMMUTABLE ASSUMPTION: Preprocessing cost amortized over many queries")
    print("5. BOUNDARY HANDLING: Pad with zeros to avoid index checks")
    
    print("\n" + "=" * 70)
    print("Algorithm Comparison:")
    print("Naive:            O(1) space, O(area) per query")
    print("Row Prefix:       O(mn) space, O(rows) per query")
    print("2D Prefix:        O(mn) space, O(1) per query")
    print("Space Optimized:  O(mn) space, O(min(rows,cols)) per query")
    print("Fenwick Tree:     O(mn) space, O(log m log n) per query/update")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("- Image processing: Sum of pixel regions")
    print("- Database queries: Aggregate functions over ranges")
    print("- Computer graphics: Texture sampling")
    print("- Scientific computing: Integration over 2D domains")
    print("- Game development: Area-of-effect calculations")


if __name__ == "__main__":
    test_num_matrix()


"""
PATTERN RECOGNITION:
==================
2D Range Sum Query is a classic preprocessing + query optimization problem:
- Precompute information to answer queries efficiently
- 2D extension of 1D prefix sum technique
- Demonstrates inclusion-exclusion principle in 2D
- Foundation for many computational geometry problems

KEY INSIGHT - 2D PREFIX SUMS:
============================
**1D Prefix Sum**: prefix[i] = sum of elements from 0 to i-1
**2D Prefix Sum**: prefix[i][j] = sum of rectangle from (0,0) to (i-1,j-1)

**Construction**:
```
prefix[i][j] = matrix[i-1][j-1] + prefix[i-1][j] + prefix[i][j-1] - prefix[i-1][j-1]
```

**Query using Inclusion-Exclusion**:
```
sum(r1,c1,r2,c2) = prefix[r2+1][c2+1] - prefix[r1][c2+1] - prefix[r2+1][c1] + prefix[r1][c1]
```

INCLUSION-EXCLUSION PRINCIPLE:
=============================
To find sum of rectangle (r1,c1) to (r2,c2):

1. **Include**: Total sum from (0,0) to (r2,c2)
2. **Exclude**: Sum from (0,0) to (r1-1,c2) [above rectangle]
3. **Exclude**: Sum from (0,0) to (r2,c1-1) [left of rectangle]  
4. **Include**: Sum from (0,0) to (r1-1,c1-1) [double-excluded diagonal]

**Visual**:
```
+-------+-------+
|   A   |   B   |
+-------+-------+
|   C   | Target|
+-------+-------+

Target = Total - A - (A+C) + A = Total - A - A - C + A = Total - A - C
```

ALGORITHM APPROACHES:
====================

1. **2D Prefix Sum (Optimal)**: 
   - Initialization: O(m×n), Query: O(1), Space: O(m×n)
   - Best for multiple queries on immutable matrix

2. **Row Prefix Sum**:
   - Initialization: O(m×n), Query: O(rows), Space: O(m×n)
   - Good compromise when columns >> rows

3. **Naive Iteration**:
   - Initialization: O(1), Query: O(area), Space: O(1)
   - Best for few queries or space-constrained environments

4. **Fenwick Tree (2D BIT)**:
   - Initialization: O(m×n×log m×log n), Query/Update: O(log m×log n)
   - Best for dynamic matrices with updates

SPACE OPTIMIZATION TECHNIQUES:
=============================

**Boundary Padding**: Add extra row/column of zeros
```python
prefix = [[0] * (n + 1) for _ in range(m + 1)]
# Avoids boundary checks: prefix[i-1][j-1] is always valid
```

**Adaptive Strategy**: Choose optimization based on matrix dimensions
```python
if m <= n:
    use_row_prefix()  # Fewer rows
else:
    use_column_prefix()  # Fewer columns
```

TIME COMPLEXITY ANALYSIS:
========================
For Q queries on m×n matrix:

- **Naive**: O(Q × area) where area ≤ m×n
- **Row Prefix**: O(m×n + Q×m) 
- **2D Prefix**: O(m×n + Q) ← Optimal for Q >> 1
- **Fenwick**: O(m×n×log m×log n + Q×log m×log n)

SPACE COMPLEXITY:
================
- **Naive**: O(1)
- **Row/Column Prefix**: O(m×n)
- **2D Prefix**: O(m×n)
- **Fenwick**: O(m×n)

EDGE CASES:
==========
1. **Empty matrix**: Handle gracefully
2. **Single element**: prefix[1][1] = matrix[0][0]
3. **Single row/column**: Reduces to 1D prefix sum
4. **Negative numbers**: Algorithm works unchanged
5. **Large coordinates**: Ensure no integer overflow

APPLICATIONS:
============
1. **Image Processing**: Sum of pixel intensities in regions
2. **Database Systems**: Range aggregate queries
3. **Computer Graphics**: Texture filtering, mip-mapping
4. **Scientific Computing**: Integration over 2D domains
5. **Game Development**: Area damage calculations
6. **GIS Systems**: Spatial range queries

VARIANTS TO PRACTICE:
====================
- Range Sum Query - Mutable (307) - 1D version with updates
- Range Sum Query 2D - Mutable (308) - 2D version with updates
- Submatrix Sum Queries - generalized versions
- Maximum Sum Rectangle - optimization variant

INTERVIEW TIPS:
==============
1. **Start with 1D**: Explain 1D prefix sum first
2. **Show 2D extension**: How to extend to 2D naturally
3. **Draw inclusion-exclusion**: Visual explanation crucial
4. **Handle boundaries**: Explain padding technique
5. **Discuss trade-offs**: Space vs time for different approaches
6. **Consider updates**: How to handle mutable matrices
7. **Real applications**: Image processing, databases
8. **Edge cases**: Single element, negative numbers
9. **Complexity analysis**: Why O(1) queries after O(mn) preprocessing
10. **Follow-up questions**: 3D version, sparse matrices

MATHEMATICAL INSIGHT:
====================
The problem demonstrates the power of **preprocessing** in algorithm design:
- Spend time upfront to save time later
- Transform expensive repeated operations into cheap lookups
- 2D inclusion-exclusion principle generalizes to higher dimensions

This technique is fundamental in computational geometry and database query optimization.
"""
