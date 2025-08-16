"""
LeetCode 1039: Minimum Score Triangulation of Polygon
Difficulty: Medium
Category: Interval DP - Geometric Optimization

PROBLEM DESCRIPTION:
===================
You have a convex n-sided polygon where each vertex has an integer value. 
You are given an integer array values where values[i] is the value of the ith vertex (in order).

You will triangulate the polygon into n - 2 triangles. For each triangle, the value of that triangle is the product of the values of its vertices, and the total score of the triangulation is the sum of these values over all n - 2 triangles.

Return the minimum possible total score that you can achieve with some triangulation of the polygon.

Example 1:
Input: values = [1,2,3]
Output: 6
Explanation: The polygon is already a triangle, so the score is 1 * 2 * 3 = 6.

Example 2:
Input: values = [3,7,4,5]
Output: 144
Explanation: There are two triangulations:
The first, with diagonals (0,2) and (2,3): triangles (0,1,2) and (0,2,3) which gives score 3*7*4 + 3*4*5 = 84 + 60 = 144.
The second, with diagonal (1,3): triangles (0,1,3) and (1,2,3) which gives score 3*7*5 + 7*4*5 = 105 + 140 = 245.

Example 3:
Input: values = [1,3,1,4,1,5]
Output: 13
Explanation: The minimum score triangulation has score 1*1*3 + 1*1*4 + 1*1*5 + 1*1*1 = 3 + 4 + 5 + 1 = 13.

Constraints:
- n == values.length
- 3 <= n <= 50
- 1 <= values[i] <= 100
"""

def min_score_triangulation_brute_force(values):
    """
    BRUTE FORCE APPROACH:
    ====================
    Try all possible triangulations recursively.
    
    Time Complexity: O(C_n) - Catalan number, approximately 4^n/n^1.5
    Space Complexity: O(n) - recursion depth
    """
    def triangulate(start, end):
        # Base case: less than 3 vertices, no triangle can be formed
        if end - start < 2:
            return 0
        
        min_score = float('inf')
        
        # Try each vertex k as the third vertex of triangle (start, k, end)
        for k in range(start + 1, end):
            # Score of triangle (start, k, end)
            triangle_score = values[start] * values[k] * values[end]
            
            # Recursively triangulate the two sub-polygons
            left_score = triangulate(start, k)
            right_score = triangulate(k, end)
            
            total_score = triangle_score + left_score + right_score
            min_score = min(min_score, total_score)
        
        return min_score
    
    n = len(values)
    return triangulate(0, n - 1)


def min_score_triangulation_memoization(values):
    """
    MEMOIZATION APPROACH:
    ====================
    Use memoization to avoid recomputing same sub-polygons.
    
    Time Complexity: O(n^3) - each interval computed once
    Space Complexity: O(n^2) - memoization table
    """
    n = len(values)
    memo = {}
    
    def triangulate(start, end):
        if end - start < 2:
            return 0
        
        if (start, end) in memo:
            return memo[(start, end)]
        
        min_score = float('inf')
        
        for k in range(start + 1, end):
            triangle_score = values[start] * values[k] * values[end]
            left_score = triangulate(start, k)
            right_score = triangulate(k, end)
            
            total_score = triangle_score + left_score + right_score
            min_score = min(min_score, total_score)
        
        memo[(start, end)] = min_score
        return min_score
    
    return triangulate(0, n - 1)


def min_score_triangulation_dp(values):
    """
    BOTTOM-UP DP APPROACH:
    ======================
    Build solution from smaller to larger intervals.
    
    Time Complexity: O(n^3) - three nested loops
    Space Complexity: O(n^2) - DP table
    """
    n = len(values)
    
    # dp[i][j] = minimum score to triangulate sub-polygon from i to j
    dp = [[0] * n for _ in range(n)]
    
    # Process intervals by length
    for length in range(3, n + 1):  # At least 3 vertices needed for triangle
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            
            # Try each vertex k as the middle vertex of triangle (i, k, j)
            for k in range(i + 1, j):
                triangle_score = values[i] * values[k] * values[j]
                total_score = dp[i][k] + triangle_score + dp[k][j]
                dp[i][j] = min(dp[i][j], total_score)
    
    return dp[0][n - 1]


def min_score_triangulation_with_path(values):
    """
    TRACK TRIANGULATION PATH:
    ========================
    Return minimum score and the actual triangulation.
    
    Time Complexity: O(n^3) - DP computation + reconstruction
    Space Complexity: O(n^2) - DP table + path tracking
    """
    n = len(values)
    dp = [[0] * n for _ in range(n)]
    split = [[0] * n for _ in range(n)]  # Track optimal split point
    
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            
            for k in range(i + 1, j):
                triangle_score = values[i] * values[k] * values[j]
                total_score = dp[i][k] + triangle_score + dp[k][j]
                
                if total_score < dp[i][j]:
                    dp[i][j] = total_score
                    split[i][j] = k
    
    # Reconstruct triangulation
    def get_triangles(start, end):
        if end - start < 2:
            return []
        
        k = split[start][end]
        triangles = [(start, k, end)]
        triangles.extend(get_triangles(start, k))
        triangles.extend(get_triangles(k, end))
        return triangles
    
    triangles = get_triangles(0, n - 1)
    return dp[0][n - 1], triangles


def min_score_triangulation_analysis(values):
    """
    DETAILED ANALYSIS:
    =================
    Show step-by-step DP computation and triangulation process.
    """
    print(f"Polygon Triangulation Analysis:")
    print(f"Vertices: {values}")
    print(f"Number of vertices: {len(values)}")
    
    n = len(values)
    
    # Visualize the polygon
    print(f"\nPolygon visualization (vertices in order):")
    for i in range(n):
        print(f"  Vertex {i}: {values[i]}")
    
    # Build DP table with detailed logging
    dp = [[0] * n for _ in range(n)]
    split = [[0] * n for _ in range(n)]
    
    print(f"\nDP Table Construction:")
    print(f"dp[i][j] = minimum score to triangulate polygon from vertex i to j")
    
    for length in range(3, n + 1):
        print(f"\nLength {length} intervals:")
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            
            print(f"  Polygon [{i}, {j}] with vertices: {[values[x] for x in range(i, j+1)]}")
            
            for k in range(i + 1, j):
                triangle_score = values[i] * values[k] * values[j]
                total_score = dp[i][k] + triangle_score + dp[k][j]
                
                print(f"    Split at vertex {k} (value {values[k]}):")
                print(f"      Triangle ({i},{k},{j}): {values[i]} * {values[k]} * {values[j]} = {triangle_score}")
                print(f"      Left polygon [{i},{k}]: {dp[i][k]}")
                print(f"      Right polygon [{k},{j}]: {dp[k][j]}")
                print(f"      Total: {dp[i][k]} + {triangle_score} + {dp[k][j]} = {total_score}")
                
                if total_score < dp[i][j]:
                    dp[i][j] = total_score
                    split[i][j] = k
                    print(f"      *** New minimum for [{i},{j}]")
            
            print(f"    Best for [{i},{j}]: {dp[i][j]} (split at {split[i][j]})")
    
    print(f"\nFinal DP Table:")
    print("   ", end="")
    for j in range(n):
        print(f"{j:6}", end="")
    print()
    
    for i in range(n):
        print(f"{i:2}: ", end="")
        for j in range(n):
            if j >= i + 2:  # Only meaningful for intervals of length >= 3
                print(f"{dp[i][j]:6}", end="")
            else:
                print(f"{'':6}", end="")
        print()
    
    print(f"\nMinimum triangulation score: {dp[0][n-1]}")
    
    # Show optimal triangulation
    min_score, triangles = min_score_triangulation_with_path(values)
    print(f"\nOptimal triangulation:")
    total_check = 0
    for i, (a, b, c) in enumerate(triangles):
        score = values[a] * values[b] * values[c]
        total_check += score
        print(f"  Triangle {i+1}: ({a},{b},{c}) = {values[a]} * {values[b]} * {values[c]} = {score}")
    
    print(f"Total score verification: {total_check}")
    
    return dp[0][n-1]


def min_score_triangulation_variants():
    """
    TRIANGULATION VARIANTS:
    ======================
    Different scenarios and modifications.
    """
    
    def max_score_triangulation(values):
        """Find maximum score triangulation"""
        n = len(values)
        dp = [[0] * n for _ in range(n)]
        
        for length in range(3, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                
                for k in range(i + 1, j):
                    triangle_score = values[i] * values[k] * values[j]
                    total_score = dp[i][k] + triangle_score + dp[k][j]
                    dp[i][j] = max(dp[i][j], total_score)
        
        return dp[0][n - 1]
    
    def count_triangulations(n):
        """Count number of possible triangulations (Catalan number)"""
        if n < 3:
            return 0
        
        # C(n-2) where C(k) is the k-th Catalan number
        # C(k) = (2k)! / ((k+1)! * k!)
        def catalan(k):
            if k <= 1:
                return 1
            
            result = 1
            for i in range(k):
                result = result * (k + 1 + i) // (i + 1)
            return result // (k + 1)
        
        return catalan(n - 2)
    
    def triangulation_with_weights(values, weights):
        """Triangulation where each vertex has a weight affecting the score"""
        n = len(values)
        dp = [[0] * n for _ in range(n)]
        
        for length in range(3, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                dp[i][j] = float('inf')
                
                for k in range(i + 1, j):
                    # Include weights in the calculation
                    triangle_score = (values[i] * values[k] * values[j] * 
                                    weights[i] * weights[k] * weights[j])
                    total_score = dp[i][k] + triangle_score + dp[k][j]
                    dp[i][j] = min(dp[i][j], total_score)
        
        return dp[0][n - 1]
    
    def min_triangulation_with_forbidden(values, forbidden_edges):
        """Triangulation avoiding certain edges"""
        n = len(values)
        forbidden_set = set(forbidden_edges)
        dp = [[float('inf')] * n for _ in range(n)]
        
        # Base case
        for i in range(n - 2):
            dp[i][i + 2] = 0
        
        for length in range(3, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                
                for k in range(i + 1, j):
                    # Check if any edge is forbidden
                    if ((i, k) in forbidden_set or (k, i) in forbidden_set or
                        (k, j) in forbidden_set or (j, k) in forbidden_set or
                        (i, j) in forbidden_set or (j, i) in forbidden_set):
                        continue
                    
                    triangle_score = values[i] * values[k] * values[j]
                    total_score = dp[i][k] + triangle_score + dp[k][j]
                    dp[i][j] = min(dp[i][j], total_score)
        
        return dp[0][n - 1] if dp[0][n - 1] != float('inf') else -1
    
    # Test variants
    test_cases = [
        [1, 2, 3],
        [3, 7, 4, 5],
        [1, 3, 1, 4, 1, 5],
        [2, 3, 1, 4, 2]
    ]
    
    print("Polygon Triangulation Variants:")
    print("=" * 50)
    
    for values in test_cases:
        print(f"\nVertices: {values}")
        
        min_score = min_score_triangulation_dp(values)
        max_score = max_score_triangulation(values)
        count = count_triangulations(len(values))
        
        print(f"Min score: {min_score}")
        print(f"Max score: {max_score}")
        print(f"Possible triangulations: {count}")
        
        # With weights
        weights = [1.2] * len(values)  # 20% increase
        weighted_score = triangulation_with_weights(values, weights)
        print(f"With 1.2x weights: {weighted_score:.1f}")
        
        # With forbidden edges (example: forbid edge between vertices 0 and 2)
        if len(values) > 3:
            forbidden = [(0, 2)]
            forbidden_score = min_triangulation_with_forbidden(values, forbidden)
            print(f"Forbidding edge (0,2): {forbidden_score}")


# Test cases
def test_min_score_triangulation():
    """Test all implementations with various inputs"""
    test_cases = [
        ([1, 2, 3], 6),
        ([3, 7, 4, 5], 144),
        ([1, 3, 1, 4, 1, 5], 13),
        ([2, 3, 1, 4, 2], 16),
        ([1, 2, 1, 4, 1, 3], 8),
        ([5, 2, 4, 1, 3], 28)
    ]
    
    print("Testing Minimum Score Triangulation Solutions:")
    print("=" * 70)
    
    for i, (values, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Values: {values}")
        print(f"Expected: {expected}")
        
        # Skip brute force for large inputs
        if len(values) <= 5:
            try:
                brute_force = min_score_triangulation_brute_force(values)
                print(f"Brute Force:      {brute_force:>4} {'✓' if brute_force == expected else '✗'}")
            except:
                print(f"Brute Force:      Timeout")
        
        memoization = min_score_triangulation_memoization(values)
        dp_result = min_score_triangulation_dp(values)
        
        print(f"Memoization:      {memoization:>4} {'✓' if memoization == expected else '✗'}")
        print(f"Bottom-up DP:     {dp_result:>4} {'✓' if dp_result == expected else '✗'}")
        
        # Show triangulation for small cases
        if len(values) <= 6:
            min_score, triangles = min_score_triangulation_with_path(values)
            print(f"Triangulation: {triangles}")
            print(f"Triangle scores: {[values[a]*values[b]*values[c] for a,b,c in triangles]}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    min_score_triangulation_analysis([3, 7, 4, 5])
    
    # Variants demonstration
    print(f"\n" + "=" * 70)
    min_score_triangulation_variants()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. INTERVAL DP: Split polygon optimally at each vertex")
    print("2. TRIANGULATION: Every n-gon has exactly n-2 triangles")
    print("3. CATALAN NUMBERS: Number of triangulations = C(n-2)")
    print("4. OPTIMAL SUBSTRUCTURE: Optimal triangulation has optimal sub-triangulations")
    print("5. O(N^3) COMPLEXITY: Try all split points for all intervals")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Computational Geometry: Polygon decomposition")
    print("• Computer Graphics: Mesh generation and optimization")
    print("• Game Development: Collision detection optimization")
    print("• CAD Systems: Surface triangulation")
    print("• Algorithm Design: Geometric optimization problems")


if __name__ == "__main__":
    test_min_score_triangulation()


"""
MINIMUM SCORE TRIANGULATION OF POLYGON - GEOMETRIC INTERVAL DP:
===============================================================

This problem demonstrates interval DP applied to geometric optimization:
- Triangulate a convex polygon to minimize total score
- Each triangle contributes the product of its three vertex values
- Must partition polygon into exactly n-2 triangles
- Classic application of interval DP to computational geometry

KEY INSIGHTS:
============
1. **TRIANGULATION PROPERTY**: Every n-sided polygon has exactly n-2 triangles
2. **INTERVAL DECOMPOSITION**: Choosing diagonal splits polygon into subproblems
3. **CONVEX POLYGON**: All diagonals are valid, simplifying the problem
4. **OPTIMAL SUBSTRUCTURE**: Optimal triangulation contains optimal sub-triangulations
5. **CATALAN NUMBERS**: Number of possible triangulations is the (n-2)th Catalan number

ALGORITHM APPROACHES:
====================

1. **Brute Force**: O(C_n) time, O(n) space
   - Try all possible triangulations
   - Catalan number complexity ~4^n/n^1.5

2. **Memoization**: O(n³) time, O(n²) space
   - Cache results for sub-polygons
   - Top-down approach

3. **Bottom-up DP**: O(n³) time, O(n²) space
   - Build from smaller to larger sub-polygons
   - Standard interval DP approach

4. **Space Optimized**: O(n³) time, O(n) space
   - Possible with careful implementation

CORE INTERVAL DP ALGORITHM:
==========================
```python
# dp[i][j] = minimum score to triangulate sub-polygon from vertex i to j
dp = [[0] * n for _ in range(n)]

for length in range(3, n + 1):           # Sub-polygon size
    for i in range(n - length + 1):      # Start vertex
        j = i + length - 1               # End vertex
        dp[i][j] = float('inf')
        
        for k in range(i + 1, j):        # Split vertex
            triangle_score = values[i] * values[k] * values[j]
            total_score = dp[i][k] + triangle_score + dp[k][j]
            dp[i][j] = min(dp[i][j], total_score)

return dp[0][n-1]
```

TRIANGULATION PROPERTIES:
=========================
**Triangle Count**: For an n-sided polygon:
- Number of triangles = n - 2
- Number of diagonals = n - 3
- Each diagonal connects two non-adjacent vertices

**Catalan Numbers**: Number of triangulations of (n+2)-gon:
- C_n = (2n)! / ((n+1)! × n!)
- C_0=1, C_1=1, C_2=2, C_3=5, C_4=14, ...
- Recurrence: C_n = Σ(C_i × C_(n-1-i)) for i=0 to n-1

RECURRENCE RELATION:
===================
```
dp[i][j] = min(dp[i][k] + values[i] * values[k] * values[j] + dp[k][j])
           for all k in (i+1, j-1)

Base case: dp[i][j] = 0 if j - i < 2 (less than 3 vertices)
```

**Intuition**: To triangulate polygon from vertex i to j:
- Choose intermediate vertex k to form triangle (i, k, j)
- Recursively triangulate sub-polygons [i, k] and [k, j]
- Add the cost of triangle (i, k, j)

GEOMETRIC INTERPRETATION:
========================
**Diagonal Selection**: Each choice of k corresponds to:
- Drawing diagonal from vertex i to vertex k
- Drawing diagonal from vertex k to vertex j
- Creating triangle with vertices i, k, j

**Sub-polygon Independence**: After choosing vertex k:
- Left sub-polygon: vertices i, i+1, ..., k
- Right sub-polygon: vertices k, k+1, ..., j
- These can be triangulated independently

COMPLEXITY ANALYSIS:
===================
- **Time**: O(n³) - three nested loops
- **Space**: O(n²) - DP table
- **States**: O(n²) - all possible sub-polygons
- **Transitions**: O(n) - try each intermediate vertex

SOLUTION RECONSTRUCTION:
=======================
To find the actual triangulation:
```python
def get_triangles(start, end, split):
    if end - start < 2:
        return []
    
    k = split[start][end]
    triangles = [(start, k, end)]
    triangles.extend(get_triangles(start, k, split))
    triangles.extend(get_triangles(k, end, split))
    return triangles
```

APPLICATIONS:
============
- **Computational Geometry**: Polygon decomposition algorithms
- **Computer Graphics**: Mesh generation and surface triangulation
- **Game Development**: Collision detection and rendering optimization
- **CAD Systems**: Surface representation and manipulation
- **Geographic Information Systems**: Terrain modeling

RELATED PROBLEMS:
================
- **Burst Balloons (312)**: Same interval DP pattern
- **Matrix Chain Multiplication**: Classic interval DP
- **Optimal Binary Search Trees**: Weighted interval optimization
- **Minimum Cost to Merge Stones**: Generalized merging problem

GEOMETRIC EXTENSIONS:
====================
- **Non-convex polygons**: More complex constraint handling
- **Weighted vertices**: Different scoring functions
- **3D triangulation**: Extend to higher dimensions
- **Constrained triangulation**: Forbidden edges or required diagonals

MATHEMATICAL PROPERTIES:
========================
- **Optimal Substructure**: Essential for DP approach
- **Overlapping Subproblems**: Same sub-polygons appear multiple times
- **Monotonicity**: Larger polygons have at least as many options
- **Convexity**: Ensures all diagonals are valid

OPTIMIZATION TECHNIQUES:
=======================
- **Pruning**: Early termination for suboptimal branches
- **Space Optimization**: Reduce memory usage for large polygons
- **Parallel Processing**: Independent subproblems can be computed in parallel
- **Approximation**: Heuristic methods for very large polygons

EDGE CASES:
==========
- **Triangle (n=3)**: Only one possible triangulation
- **Quadrilateral (n=4)**: Two possible triangulations
- **Regular polygon**: Symmetric optimization opportunities
- **Degenerate cases**: Vertices with value 0

This problem beautifully combines geometric intuition with algorithmic
optimization, serving as an excellent introduction to computational geometry
applications of dynamic programming.
"""
