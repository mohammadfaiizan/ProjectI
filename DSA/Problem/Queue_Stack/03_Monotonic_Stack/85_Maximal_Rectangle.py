"""
85. Maximal Rectangle - Multiple Approaches
Difficulty: Hard

Given a rows x cols binary matrix filled with 0's and 1's, find the largest rectangle containing only 1's and return its area.
"""

from typing import List

class MaximalRectangle:
    """Multiple approaches to find maximal rectangle in binary matrix"""
    
    def maximalRectangle_histogram_approach(self, matrix: List[List[str]]) -> int:
        """
        Approach 1: Convert to Histogram Problem (Optimal)
        
        Convert each row to histogram and find largest rectangle.
        
        Time: O(m * n), Space: O(n)
        """
        if not matrix or not matrix[0]:
            return 0
        
        rows, cols = len(matrix), len(matrix[0])
        heights = [0] * cols
        max_area = 0
        
        def largest_rectangle_in_histogram(heights: List[int]) -> int:
            """Find largest rectangle in histogram using stack"""
            stack = []
            max_area = 0
            
            for i in range(len(heights)):
                while stack and heights[stack[-1]] > heights[i]:
                    height = heights[stack.pop()]
                    width = i if not stack else i - stack[-1] - 1
                    max_area = max(max_area, height * width)
                stack.append(i)
            
            while stack:
                height = heights[stack.pop()]
                width = len(heights) if not stack else len(heights) - stack[-1] - 1
                max_area = max(max_area, height * width)
            
            return max_area
        
        # Process each row
        for row in matrix:
            # Update heights array
            for j in range(cols):
                if row[j] == '1':
                    heights[j] += 1
                else:
                    heights[j] = 0
            
            # Find max rectangle in current histogram
            area = largest_rectangle_in_histogram(heights)
            max_area = max(max_area, area)
        
        return max_area
    
    def maximalRectangle_dp_approach(self, matrix: List[List[str]]) -> int:
        """
        Approach 2: Dynamic Programming
        
        Use DP to track heights, left bounds, and right bounds.
        
        Time: O(m * n), Space: O(n)
        """
        if not matrix or not matrix[0]:
            return 0
        
        rows, cols = len(matrix), len(matrix[0])
        heights = [0] * cols
        left_bounds = [0] * cols
        right_bounds = [cols] * cols
        max_area = 0
        
        for row in matrix:
            # Update heights
            for j in range(cols):
                if row[j] == '1':
                    heights[j] += 1
                else:
                    heights[j] = 0
            
            # Update left bounds
            current_left = 0
            for j in range(cols):
                if row[j] == '1':
                    left_bounds[j] = max(left_bounds[j], current_left)
                else:
                    left_bounds[j] = 0
                    current_left = j + 1
            
            # Update right bounds
            current_right = cols
            for j in range(cols - 1, -1, -1):
                if row[j] == '1':
                    right_bounds[j] = min(right_bounds[j], current_right)
                else:
                    right_bounds[j] = cols
                    current_right = j
            
            # Calculate max area for current row
            for j in range(cols):
                area = heights[j] * (right_bounds[j] - left_bounds[j])
                max_area = max(max_area, area)
        
        return max_area
    
    def maximalRectangle_brute_force(self, matrix: List[List[str]]) -> int:
        """
        Approach 3: Brute Force
        
        Check all possible rectangles.
        
        Time: O(m² * n²), Space: O(1)
        """
        if not matrix or not matrix[0]:
            return 0
        
        rows, cols = len(matrix), len(matrix[0])
        max_area = 0
        
        for i in range(rows):
            for j in range(cols):
                if matrix[i][j] == '1':
                    # Try all rectangles starting at (i, j)
                    min_width = cols
                    
                    for k in range(i, rows):
                        if matrix[k][j] == '0':
                            break
                        
                        # Find width for current height
                        width = 0
                        for l in range(j, cols):
                            if matrix[k][l] == '1':
                                width += 1
                            else:
                                break
                        
                        min_width = min(min_width, width)
                        height = k - i + 1
                        area = height * min_width
                        max_area = max(max_area, area)
        
        return max_area
    
    def maximalRectangle_stack_optimized(self, matrix: List[List[str]]) -> int:
        """
        Approach 4: Stack Optimized with Preprocessing
        
        Preprocess matrix and use optimized stack algorithm.
        
        Time: O(m * n), Space: O(n)
        """
        if not matrix or not matrix[0]:
            return 0
        
        rows, cols = len(matrix), len(matrix[0])
        heights = [0] * (cols + 1)  # Add sentinel
        max_area = 0
        
        for row in matrix:
            # Update heights
            for j in range(cols):
                heights[j] = heights[j] + 1 if row[j] == '1' else 0
            
            # Find max rectangle using stack with sentinel
            stack = []
            for i in range(len(heights)):
                while stack and heights[stack[-1]] > heights[i]:
                    height = heights[stack.pop()]
                    width = i if not stack else i - stack[-1] - 1
                    max_area = max(max_area, height * width)
                stack.append(i)
        
        return max_area
    
    def maximalRectangle_divide_conquer(self, matrix: List[List[str]]) -> int:
        """
        Approach 5: Divide and Conquer
        
        Use divide and conquer for each histogram.
        
        Time: O(m * n log n), Space: O(n)
        """
        if not matrix or not matrix[0]:
            return 0
        
        rows, cols = len(matrix), len(matrix[0])
        heights = [0] * cols
        max_area = 0
        
        def largest_rectangle_dc(heights: List[int], left: int, right: int) -> int:
            """Divide and conquer for largest rectangle"""
            if left > right:
                return 0
            
            # Find minimum height and its index
            min_height = float('inf')
            min_idx = left
            
            for i in range(left, right + 1):
                if heights[i] < min_height:
                    min_height = heights[i]
                    min_idx = i
            
            # Calculate area with minimum height
            area_with_min = min_height * (right - left + 1)
            
            # Recursively find max in left and right parts
            left_max = largest_rectangle_dc(heights, left, min_idx - 1)
            right_max = largest_rectangle_dc(heights, min_idx + 1, right)
            
            return max(area_with_min, left_max, right_max)
        
        for row in matrix:
            # Update heights
            for j in range(cols):
                heights[j] = heights[j] + 1 if row[j] == '1' else 0
            
            # Find max rectangle using divide and conquer
            area = largest_rectangle_dc(heights, 0, cols - 1)
            max_area = max(max_area, area)
        
        return max_area
    
    def maximalRectangle_segment_tree(self, matrix: List[List[str]]) -> int:
        """
        Approach 6: Segment Tree for Range Minimum Query
        
        Use segment tree to find minimum in ranges efficiently.
        
        Time: O(m * n log n), Space: O(n)
        """
        if not matrix or not matrix[0]:
            return 0
        
        rows, cols = len(matrix), len(matrix[0])
        heights = [0] * cols
        max_area = 0
        
        def build_segment_tree(arr: List[int]) -> List[int]:
            """Build segment tree for range minimum query"""
            n = len(arr)
            tree = [0] * (4 * n)
            
            def build(node: int, start: int, end: int) -> None:
                if start == end:
                    tree[node] = start
                else:
                    mid = (start + end) // 2
                    build(2 * node, start, mid)
                    build(2 * node + 1, mid + 1, end)
                    
                    left_idx = tree[2 * node]
                    right_idx = tree[2 * node + 1]
                    
                    if arr[left_idx] <= arr[right_idx]:
                        tree[node] = left_idx
                    else:
                        tree[node] = right_idx
            
            build(1, 0, n - 1)
            return tree
        
        def query_min_idx(tree: List[int], arr: List[int], node: int, start: int, end: int, l: int, r: int) -> int:
            """Query minimum index in range [l, r]"""
            if r < start or end < l:
                return -1
            
            if l <= start and end <= r:
                return tree[node]
            
            mid = (start + end) // 2
            left_idx = query_min_idx(tree, arr, 2 * node, start, mid, l, r)
            right_idx = query_min_idx(tree, arr, 2 * node + 1, mid + 1, end, l, r)
            
            if left_idx == -1:
                return right_idx
            if right_idx == -1:
                return left_idx
            
            return left_idx if arr[left_idx] <= arr[right_idx] else right_idx
        
        def largest_rectangle_st(heights: List[int]) -> int:
            """Find largest rectangle using segment tree"""
            if not heights:
                return 0
            
            tree = build_segment_tree(heights)
            
            def find_max_area(left: int, right: int) -> int:
                if left > right:
                    return 0
                
                min_idx = query_min_idx(tree, heights, 1, 0, len(heights) - 1, left, right)
                area_with_min = heights[min_idx] * (right - left + 1)
                
                left_max = find_max_area(left, min_idx - 1)
                right_max = find_max_area(min_idx + 1, right)
                
                return max(area_with_min, left_max, right_max)
            
            return find_max_area(0, len(heights) - 1)
        
        for row in matrix:
            # Update heights
            for j in range(cols):
                heights[j] = heights[j] + 1 if row[j] == '1' else 0
            
            # Find max rectangle using segment tree
            area = largest_rectangle_st(heights)
            max_area = max(max_area, area)
        
        return max_area


def test_maximal_rectangle():
    """Test maximal rectangle algorithms"""
    solver = MaximalRectangle()
    
    test_cases = [
        ([["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]], 6, "Example 1"),
        ([["0"]], 0, "Single 0"),
        ([["1"]], 1, "Single 1"),
        ([["0","0"],["0","0"]], 0, "All zeros"),
        ([["1","1"],["1","1"]], 4, "All ones 2x2"),
        ([["1","1","1"],["1","1","1"]], 6, "All ones 2x3"),
        ([["1","0","1"],["1","1","1"],["1","1","1"]], 4, "L-shape"),
        ([["1","1","0","1"],["1","1","0","1"],["1","1","1","1"]], 4, "Complex pattern"),
        ([["0","1","1","0","1"],["1","1","0","1","0"],["0","1","1","1","0"],["1","1","1","1","0"],["1","1","1","1","1"],["0","0","0","0","0"]], 9, "Large matrix"),
    ]
    
    algorithms = [
        ("Histogram Approach", solver.maximalRectangle_histogram_approach),
        ("DP Approach", solver.maximalRectangle_dp_approach),
        ("Brute Force", solver.maximalRectangle_brute_force),
        ("Stack Optimized", solver.maximalRectangle_stack_optimized),
        ("Divide Conquer", solver.maximalRectangle_divide_conquer),
    ]
    
    print("=== Testing Maximal Rectangle ===")
    
    for matrix, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Matrix: {matrix}")
        print(f"Expected: {expected}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(matrix)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:20} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:40]}")


def demonstrate_histogram_approach():
    """Demonstrate histogram approach step by step"""
    print("\n=== Histogram Approach Step-by-Step Demo ===")
    
    matrix = [
        ["1","0","1","0","0"],
        ["1","0","1","1","1"],
        ["1","1","1","1","1"],
        ["1","0","0","1","0"]
    ]
    
    print("Matrix:")
    for row in matrix:
        print("  " + " ".join(row))
    
    print("\nConverting each row to histogram:")
    
    heights = [0] * len(matrix[0])
    max_area = 0
    
    def largest_rectangle_in_histogram(heights: List[int]) -> int:
        stack = []
        max_area = 0
        
        for i in range(len(heights)):
            while stack and heights[stack[-1]] > heights[i]:
                height = heights[stack.pop()]
                width = i if not stack else i - stack[-1] - 1
                max_area = max(max_area, height * width)
            stack.append(i)
        
        while stack:
            height = heights[stack.pop()]
            width = len(heights) if not stack else len(heights) - stack[-1] - 1
            max_area = max(max_area, height * width)
        
        return max_area
    
    for i, row in enumerate(matrix):
        print(f"\nRow {i}: {row}")
        
        # Update heights
        for j in range(len(row)):
            if row[j] == '1':
                heights[j] += 1
            else:
                heights[j] = 0
        
        print(f"  Heights: {heights}")
        
        # Find max rectangle in histogram
        area = largest_rectangle_in_histogram(heights)
        max_area = max(max_area, area)
        
        print(f"  Max rectangle area in this histogram: {area}")
        print(f"  Overall max area so far: {max_area}")
    
    print(f"\nFinal maximum rectangle area: {max_area}")


def visualize_matrix_rectangles():
    """Visualize rectangles in matrix"""
    print("\n=== Matrix Rectangle Visualization ===")
    
    matrix = [
        ["1","0","1","0","0"],
        ["1","0","1","1","1"],
        ["1","1","1","1","1"],
        ["1","0","0","1","0"]
    ]
    
    print("Original matrix:")
    for i, row in enumerate(matrix):
        print(f"Row {i}: " + " ".join(row))
    
    print("\nHeight matrix (cumulative heights):")
    
    heights_matrix = []
    heights = [0] * len(matrix[0])
    
    for row in matrix:
        for j in range(len(row)):
            if row[j] == '1':
                heights[j] += 1
            else:
                heights[j] = 0
        heights_matrix.append(heights[:])
    
    for i, heights_row in enumerate(heights_matrix):
        print(f"Row {i}: " + " ".join(f"{h:2}" for h in heights_row))
    
    print("\nFinding rectangles in each histogram:")
    
    solver = MaximalRectangle()
    max_area = 0
    
    for i, heights_row in enumerate(heights_matrix):
        print(f"\nHistogram {i}: {heights_row}")
        
        # Visualize histogram
        max_height = max(heights_row) if heights_row else 0
        for level in range(max_height, 0, -1):
            line = f"{level} |"
            for height in heights_row:
                if height >= level:
                    line += "██"
                else:
                    line += "  "
            print(line)
        
        print("  +" + "--" * len(heights_row))
        
        # Find largest rectangle
        area = solver.maximalRectangle_histogram_approach([matrix[i]])[1] if i == 0 else 0
        # Recalculate properly
        stack = []
        current_max = 0
        
        for j in range(len(heights_row)):
            while stack and heights_row[stack[-1]] > heights_row[j]:
                height = heights_row[stack.pop()]
                width = j if not stack else j - stack[-1] - 1
                current_max = max(current_max, height * width)
            stack.append(j)
        
        while stack:
            height = heights_row[stack.pop()]
            width = len(heights_row) if not stack else len(heights_row) - stack[-1] - 1
            current_max = max(current_max, height * width)
        
        max_area = max(max_area, current_max)
        print(f"  Max rectangle area: {current_max}")
    
    print(f"\nOverall maximum rectangle area: {max_area}")


def benchmark_maximal_rectangle():
    """Benchmark different approaches"""
    import time
    import random
    
    algorithms = [
        ("Histogram Approach", MaximalRectangle().maximalRectangle_histogram_approach),
        ("DP Approach", MaximalRectangle().maximalRectangle_dp_approach),
        ("Stack Optimized", MaximalRectangle().maximalRectangle_stack_optimized),
    ]
    
    # Generate test matrices of different sizes
    def generate_matrix(rows: int, cols: int) -> List[List[str]]:
        return [[str(random.randint(0, 1)) for _ in range(cols)] for _ in range(rows)]
    
    test_sizes = [(10, 10), (20, 20), (50, 50)]
    
    print("\n=== Maximal Rectangle Performance Benchmark ===")
    
    for rows, cols in test_sizes:
        print(f"\n--- Matrix Size: {rows}x{cols} ---")
        
        matrix = generate_matrix(rows, cols)
        
        for alg_name, alg_func in algorithms:
            start_time = time.time()
            
            try:
                result = alg_func(matrix)
                end_time = time.time()
                print(f"{alg_name:20} | Time: {end_time - start_time:.4f}s | Result: {result}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:30]}")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    solver = MaximalRectangle()
    
    edge_cases = [
        ([], 0, "Empty matrix"),
        ([[]], 0, "Empty row"),
        ([["0"]], 0, "Single 0"),
        ([["1"]], 1, "Single 1"),
        ([["0","0","0"]], 0, "All zeros row"),
        ([["1","1","1"]], 3, "All ones row"),
        ([["0"],["0"],["0"]], 0, "All zeros column"),
        ([["1"],["1"],["1"]], 3, "All ones column"),
        ([["1","0"],["0","1"]], 1, "Diagonal pattern"),
        ([["0","1"],["1","0"]], 1, "Anti-diagonal pattern"),
    ]
    
    for matrix, expected, description in edge_cases:
        try:
            result = solver.maximalRectangle_histogram_approach(matrix)
            status = "✓" if result == expected else "✗"
            print(f"{description:25} | {status} | Matrix: {matrix} -> {result}")
        except Exception as e:
            print(f"{description:25} | ERROR: {str(e)[:30]}")


def compare_approaches():
    """Compare different approaches"""
    print("\n=== Approach Comparison ===")
    
    test_matrices = [
        [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]],
        [["1","1"],["1","1"]],
        [["0","1","1","0"],["1","1","1","1"],["1","1","1","1"],["0","1","1","0"]],
    ]
    
    solver = MaximalRectangle()
    
    approaches = [
        ("Histogram", solver.maximalRectangle_histogram_approach),
        ("DP", solver.maximalRectangle_dp_approach),
        ("Brute Force", solver.maximalRectangle_brute_force),
        ("Stack Optimized", solver.maximalRectangle_stack_optimized),
    ]
    
    for i, matrix in enumerate(test_matrices):
        print(f"\nTest case {i+1}:")
        for row in matrix:
            print("  " + " ".join(row))
        
        results = {}
        
        for name, func in approaches:
            try:
                result = func(matrix)
                results[name] = result
                print(f"{name:15} | Result: {result}")
            except Exception as e:
                print(f"{name:15} | ERROR: {str(e)[:40]}")
        
        # Check consistency
        if results:
            first_result = list(results.values())[0]
            all_same = all(result == first_result for result in results.values())
            print(f"All approaches agree: {'✓' if all_same else '✗'}")


def analyze_time_complexity():
    """Analyze time complexity of different approaches"""
    print("\n=== Time Complexity Analysis ===")
    
    approaches = [
        ("Histogram Approach", "O(m * n)", "O(n)", "Convert each row to histogram problem"),
        ("DP Approach", "O(m * n)", "O(n)", "Track heights, left, right bounds"),
        ("Brute Force", "O(m² * n²)", "O(1)", "Check all possible rectangles"),
        ("Stack Optimized", "O(m * n)", "O(n)", "Optimized stack with sentinel"),
        ("Divide Conquer", "O(m * n log n)", "O(n)", "D&C for each histogram"),
        ("Segment Tree", "O(m * n log n)", "O(n)", "Range minimum queries"),
    ]
    
    print(f"{'Approach':<20} | {'Time':<15} | {'Space':<8} | {'Notes'}")
    print("-" * 75)
    
    for approach, time_comp, space_comp, notes in approaches:
        print(f"{approach:<20} | {time_comp:<15} | {space_comp:<8} | {notes}")


def demonstrate_dp_approach():
    """Demonstrate DP approach step by step"""
    print("\n=== DP Approach Step-by-Step Demo ===")
    
    matrix = [
        ["1","0","1"],
        ["1","1","1"],
        ["1","1","1"]
    ]
    
    print("Matrix:")
    for row in matrix:
        print("  " + " ".join(row))
    
    cols = len(matrix[0])
    heights = [0] * cols
    left_bounds = [0] * cols
    right_bounds = [cols] * cols
    max_area = 0
    
    print(f"\nProcessing each row:")
    
    for i, row in enumerate(matrix):
        print(f"\nRow {i}: {row}")
        
        # Update heights
        for j in range(cols):
            if row[j] == '1':
                heights[j] += 1
            else:
                heights[j] = 0
        
        print(f"  Heights: {heights}")
        
        # Update left bounds
        current_left = 0
        for j in range(cols):
            if row[j] == '1':
                left_bounds[j] = max(left_bounds[j], current_left)
            else:
                left_bounds[j] = 0
                current_left = j + 1
        
        print(f"  Left bounds: {left_bounds}")
        
        # Update right bounds
        current_right = cols
        for j in range(cols - 1, -1, -1):
            if row[j] == '1':
                right_bounds[j] = min(right_bounds[j], current_right)
            else:
                right_bounds[j] = cols
                current_right = j
        
        print(f"  Right bounds: {right_bounds}")
        
        # Calculate areas
        areas = []
        for j in range(cols):
            area = heights[j] * (right_bounds[j] - left_bounds[j])
            areas.append(area)
            max_area = max(max_area, area)
        
        print(f"  Areas: {areas}")
        print(f"  Max area so far: {max_area}")
    
    print(f"\nFinal maximum area: {max_area}")


if __name__ == "__main__":
    test_maximal_rectangle()
    demonstrate_histogram_approach()
    visualize_matrix_rectangles()
    demonstrate_dp_approach()
    test_edge_cases()
    compare_approaches()
    analyze_time_complexity()
    benchmark_maximal_rectangle()

"""
Maximal Rectangle demonstrates advanced applications of monotonic stack
for 2D geometric problems, including histogram conversion, dynamic programming,
and multiple optimization strategies for finding largest rectangular areas.
"""
