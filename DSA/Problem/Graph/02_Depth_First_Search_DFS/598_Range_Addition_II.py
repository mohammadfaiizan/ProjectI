"""
598. Range Addition II
Difficulty: Easy

Problem:
You are given an m x n matrix M initialized with all 0's and several update operations.
Operations are represented by a 2D array, and each operation is represented by an array 
with two positive integers a and b, which means M[i][j] should be incremented by one for 
all 0 <= i < a and 0 <= j < b.

You need to count and return the number of maximum integers in the matrix after performing 
all the operations.

Examples:
Input: m = 3, n = 3, operations = [[2,2],[3,3]]
Output: 4

Input: m = 3, n = 3, operations = []
Output: 9

Constraints:
- 1 <= m, n <= 4 * 10^4
- 0 <= operations.length <= 10^4
- operations[i].length == 2
- 1 <= ai <= m
- 1 <= bi <= n
"""

from typing import List

class Solution:
    def maxCount_approach1_intersection_analysis(self, m: int, n: int, operations: List[List[int]]) -> int:
        """
        Approach 1: Intersection Analysis (Optimal)
        
        The maximum value will be in the intersection of all operation rectangles.
        This intersection is determined by the minimum dimensions across all operations.
        
        Time: O(K) where K = len(operations)
        Space: O(1)
        """
        if not operations:
            return m * n  # All cells have value 0 (maximum)
        
        # Find intersection of all rectangles
        min_row = m
        min_col = n
        
        for a, b in operations:
            min_row = min(min_row, a)
            min_col = min(min_col, b)
        
        return min_row * min_col
    
    def maxCount_approach2_simulation(self, m: int, n: int, operations: List[List[int]]) -> int:
        """
        Approach 2: Direct Simulation (Brute Force)
        
        Actually perform all operations and count maximum values.
        Only for educational purposes - inefficient for large inputs.
        
        Time: O(K * M * N) where K = len(operations)
        Space: O(M * N)
        """
        # Initialize matrix
        matrix = [[0] * n for _ in range(m)]
        
        # Apply all operations
        for a, b in operations:
            for i in range(a):
                for j in range(b):
                    matrix[i][j] += 1
        
        # Find maximum value
        max_val = 0
        for i in range(m):
            for j in range(n):
                max_val = max(max_val, matrix[i][j])
        
        # Count cells with maximum value
        count = 0
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == max_val:
                    count += 1
        
        return count
    
    def maxCount_approach3_rectangle_overlap(self, m: int, n: int, operations: List[List[int]]) -> int:
        """
        Approach 3: Rectangle Overlap Analysis
        
        View as finding area of intersection of multiple rectangles.
        All rectangles start from (0,0).
        
        Time: O(K)
        Space: O(1)
        """
        if not operations:
            return m * n
        
        # Each operation defines a rectangle from (0,0) to (a-1, b-1)
        # Intersection is from (0,0) to (min_a-1, min_b-1)
        
        intersection_height = m
        intersection_width = n
        
        for height, width in operations:
            intersection_height = min(intersection_height, height)
            intersection_width = min(intersection_width, width)
        
        return intersection_height * intersection_width
    
    def maxCount_approach4_coordinate_compression(self, m: int, n: int, operations: List[List[int]]) -> int:
        """
        Approach 4: Coordinate Compression Perspective
        
        Think of operations as defining important coordinates.
        The answer is the area of the bottom-left most intersection.
        
        Time: O(K)
        Space: O(1)
        """
        if not operations:
            return m * n
        
        # Find the smallest rectangle that's covered by all operations
        max_covered_rows = min(op[0] for op in operations)
        max_covered_cols = min(op[1] for op in operations)
        
        # Ensure we don't exceed matrix bounds
        max_covered_rows = min(max_covered_rows, m)
        max_covered_cols = min(max_covered_cols, n)
        
        return max_covered_rows * max_covered_cols
    
    def maxCount_approach5_incremental_analysis(self, m: int, n: int, operations: List[List[int]]) -> int:
        """
        Approach 5: Incremental Analysis
        
        Build up the intersection incrementally.
        
        Time: O(K)
        Space: O(1)
        """
        if not operations:
            return m * n
        
        current_rows = m
        current_cols = n
        
        for a, b in operations:
            # Update intersection bounds
            current_rows = min(current_rows, a)
            current_cols = min(current_cols, b)
        
        return current_rows * current_cols

def test_max_count():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (m, n, operations, expected)
        (3, 3, [[2,2],[3,3]], 4),
        (3, 3, [], 9),
        (1, 1, [[1,1]], 1),
        (4, 4, [[2,2],[3,3],[1,1]], 1),
        (5, 5, [[3,4],[4,3],[2,5],[5,2]], 6),
        (3, 3, [[1,1],[2,2],[3,3]], 1),
        (2, 2, [[2,2],[2,2]], 4),
    ]
    
    approaches = [
        ("Intersection Analysis", solution.maxCount_approach1_intersection_analysis),
        ("Direct Simulation", solution.maxCount_approach2_simulation),
        ("Rectangle Overlap", solution.maxCount_approach3_rectangle_overlap),
        ("Coordinate Compression", solution.maxCount_approach4_coordinate_compression),
        ("Incremental Analysis", solution.maxCount_approach5_incremental_analysis),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (m, n, operations, expected) in enumerate(test_cases):
            result = func(m, n, operations)
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} m={m}, n={n}, operations={operations}")
            print(f"         Expected: {expected}, Got: {result}")

def demonstrate_intersection_concept():
    """Demonstrate the rectangle intersection concept"""
    print("\n=== Rectangle Intersection Demo ===")
    
    m, n = 4, 4
    operations = [[3,3],[2,4],[4,2]]
    
    print(f"Matrix size: {m} x {n}")
    print(f"Operations: {operations}")
    
    # Visualize each operation
    print(f"\nEach operation affects a rectangle from (0,0):")
    for i, (a, b) in enumerate(operations):
        print(f"  Operation {i+1}: Rectangle (0,0) to ({a-1},{b-1})")
        print(f"    Affects {a} rows and {b} columns")
    
    # Find intersection
    min_rows = min(op[0] for op in operations)
    min_cols = min(op[1] for op in operations)
    
    print(f"\nIntersection analysis:")
    print(f"  Minimum rows affected by all operations: {min_rows}")
    print(f"  Minimum columns affected by all operations: {min_cols}")
    print(f"  Intersection area: {min_rows} x {min_cols} = {min_rows * min_cols}")
    
    # Simulate to verify
    matrix = [[0] * n for _ in range(m)]
    
    for a, b in operations:
        for i in range(a):
            for j in range(b):
                matrix[i][j] += 1
    
    print(f"\nMatrix after all operations:")
    for i, row in enumerate(matrix):
        print(f"  Row {i}: {row}")
    
    max_val = max(max(row) for row in matrix)
    count = sum(1 for i in range(m) for j in range(n) if matrix[i][j] == max_val)
    
    print(f"\nMaximum value: {max_val}")
    print(f"Count of maximum values: {count}")

def visualize_operations():
    """Create visual representation of operations"""
    print("\n=== Operations Visualization ===")
    
    examples = [
        ("Simple Case", 3, 3, [[2,2],[3,3]]),
        ("Single Operation", 3, 3, [[2,2]]),
        ("Nested Rectangles", 4, 4, [[4,4],[3,3],[2,2],[1,1]]),
        ("Partial Overlap", 3, 4, [[2,3],[3,2]]),
    ]
    
    for name, m, n, operations in examples:
        print(f"\n{name}: {m}x{n} matrix, operations: {operations}")
        
        # Create matrix and apply operations
        matrix = [[0] * n for _ in range(m)]
        
        for a, b in operations:
            for i in range(min(a, m)):
                for j in range(min(b, n)):
                    matrix[i][j] += 1
        
        # Display matrix
        print("  Matrix after operations:")
        for i, row in enumerate(matrix):
            display_row = []
            for val in row:
                display_row.append(f"{val:2d}")
            print(f"    {' '.join(display_row)}")
        
        # Find intersection analytically
        if operations:
            min_a = min(op[0] for op in operations)
            min_b = min(op[1] for op in operations)
            analytical_result = min_a * min_b
        else:
            analytical_result = m * n
        
        # Count maximum values
        max_val = max(max(row) for row in matrix)
        count = sum(1 for row in matrix for val in row if val == max_val)
        
        print(f"  Maximum value: {max_val}")
        print(f"  Count (simulation): {count}")
        print(f"  Count (analytical): {analytical_result}")

def analyze_algorithm_efficiency():
    """Analyze efficiency of different approaches"""
    print("\n=== Algorithm Efficiency Analysis ===")
    
    approaches = [
        ("Intersection Analysis", "O(K)", "O(1)", "Optimal mathematical insight"),
        ("Direct Simulation", "O(K*M*N)", "O(M*N)", "Straightforward but inefficient"),
        ("Rectangle Overlap", "O(K)", "O(1)", "Geometric interpretation"),
        ("Coordinate Compression", "O(K)", "O(1)", "Spatial analysis perspective"),
        ("Incremental Analysis", "O(K)", "O(1)", "Step-by-step intersection building"),
    ]
    
    print(f"{'Approach':<22} {'Time':<12} {'Space':<8} {'Key Insight'}")
    print("-" * 70)
    
    for approach, time_comp, space_comp, insight in approaches:
        print(f"{approach:<22} {time_comp:<12} {space_comp:<8} {insight}")
    
    print(f"\nKey Mathematical Insight:")
    print("The maximum value occurs in cells affected by ALL operations.")
    print("This region is the intersection of all operation rectangles.")
    print("Since all rectangles start at (0,0), intersection = min dimensions.")
    
    print(f"\nWhy this works:")
    print("- Each operation increments a rectangle starting from (0,0)")
    print("- Cells in intersection get incremented by ALL operations")
    print("- Cells outside intersection miss some operations")
    print("- Therefore, intersection cells have maximum value")

def edge_cases_analysis():
    """Analyze edge cases and special scenarios"""
    print("\n=== Edge Cases Analysis ===")
    
    edge_cases = [
        ("No operations", 3, 3, [], "All cells have value 0"),
        ("Single operation", 3, 3, [[2,2]], "4 cells with value 1"),
        ("Full matrix", 3, 3, [[3,3]], "All 9 cells with value 1"),
        ("Point operation", 5, 5, [[1,1]], "Single cell with value 1"),
        ("Identical ops", 3, 3, [[2,2],[2,2]], "4 cells with value 2"),
        ("Nested sequence", 4, 4, [[4,4],[3,3],[2,2],[1,1]], "1 cell with value 4"),
    ]
    
    solution = Solution()
    
    for case_name, m, n, operations, description in edge_cases:
        result = solution.maxCount_approach1_intersection_analysis(m, n, operations)
        print(f"{case_name:<18}: {result:2d} cells  ({description})")

if __name__ == "__main__":
    test_max_count()
    demonstrate_intersection_concept()
    visualize_operations()
    analyze_algorithm_efficiency()
    edge_cases_analysis()

"""
Graph Theory Concepts:
1. Rectangle Intersection in 2D Space
2. Geometric Algorithm Optimization
3. Coordinate Compression Techniques
4. Spatial Range Query Analysis

Key Mathematical Insights:
- All operations start from origin (0,0)
- Maximum value occurs in intersection of all rectangles
- Intersection area = min(all heights) × min(all widths)
- No need to simulate - pure mathematical analysis suffices

Problem Pattern Recognition:
- Range update operations on 2D grid
- Finding maximum overlap region
- Rectangle intersection problem
- Optimization through mathematical analysis vs simulation

Algorithm Design Lessons:
- Sometimes simulation is unnecessary
- Mathematical analysis can provide O(1) space solutions
- Geometric intuition often leads to optimal algorithms
- Understanding problem structure enables significant optimizations

Real-world Applications:
- Heat map analysis (overlapping influence regions)
- Resource allocation (overlapping coverage areas)
- Image processing (overlapping filters)
- Sensor coverage analysis
- Marketing reach analysis (overlapping campaigns)

This problem demonstrates how geometric insight can transform
an O(M*N*K) simulation into an O(K) mathematical analysis.
"""
