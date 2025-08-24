"""
1504. Count Submatrices With All Ones - Multiple Approaches
Difficulty: Medium

Given a rows x cols matrix mat, where mat[i][j] is either 0 or 1, return the number of submatrices that have all ones.
"""

from typing import List

class CountSubmatricesWithAllOnes:
    """Multiple approaches to count submatrices with all ones"""
    
    def numSubmat_histogram_approach(self, mat: List[List[int]]) -> int:
        """
        Approach 1: Histogram-based with Stack (Optimal)
        
        Convert each row to histogram and count rectangles.
        
        Time: O(m * n), Space: O(n)
        """
        if not mat or not mat[0]:
            return 0
        
        rows, cols = len(mat), len(mat[0])
        heights = [0] * cols
        total_count = 0
        
        def count_rectangles_in_histogram(heights: List[int]) -> int:
            """Count all rectangles in histogram using stack"""
            stack = []
            count = 0
            
            for i in range(len(heights)):
                while stack and heights[stack[-1]] > heights[i]:
                    h = heights[stack.pop()]
                    w = i if not stack else i - stack[-1] - 1
                    
                    # Count all rectangles with height h and width <= w
                    count += h * w * (w + 1) // 2
                
                stack.append(i)
            
            # Process remaining elements in stack
            while stack:
                h = heights[stack.pop()]
                w = len(heights) if not stack else len(heights) - stack[-1] - 1
                count += h * w * (w + 1) // 2
            
            return count
        
        # Process each row
        for row in mat:
            # Update heights array
            for j in range(cols):
                if row[j] == 1:
                    heights[j] += 1
                else:
                    heights[j] = 0
            
            # Count rectangles in current histogram
            total_count += count_rectangles_in_histogram(heights)
        
        return total_count
    
    def numSubmat_dp_approach(self, mat: List[List[int]]) -> int:
        """
        Approach 2: Dynamic Programming
        
        Use DP to count submatrices ending at each position.
        
        Time: O(m * n²), Space: O(n)
        """
        if not mat or not mat[0]:
            return 0
        
        rows, cols = len(mat), len(mat[0])
        total_count = 0
        
        # For each row, calculate heights
        heights = [0] * cols
        
        for i in range(rows):
            # Update heights for current row
            for j in range(cols):
                if mat[i][j] == 1:
                    heights[j] += 1
                else:
                    heights[j] = 0
            
            # Count submatrices ending at row i
            for j in range(cols):
                if heights[j] > 0:
                    min_height = heights[j]
                    
                    # Extend to the left
                    for k in range(j, -1, -1):
                        if heights[k] == 0:
                            break
                        
                        min_height = min(min_height, heights[k])
                        width = j - k + 1
                        
                        # Add rectangles with this width and height <= min_height
                        total_count += min_height
        
        return total_count
    
    def numSubmat_brute_force(self, mat: List[List[int]]) -> int:
        """
        Approach 3: Brute Force
        
        Check all possible submatrices.
        
        Time: O(m² * n²), Space: O(1)
        """
        if not mat or not mat[0]:
            return 0
        
        rows, cols = len(mat), len(mat[0])
        count = 0
        
        # Try all possible top-left corners
        for i1 in range(rows):
            for j1 in range(cols):
                # Try all possible bottom-right corners
                for i2 in range(i1, rows):
                    for j2 in range(j1, cols):
                        # Check if submatrix has all ones
                        all_ones = True
                        
                        for i in range(i1, i2 + 1):
                            for j in range(j1, j2 + 1):
                                if mat[i][j] == 0:
                                    all_ones = False
                                    break
                            if not all_ones:
                                break
                        
                        if all_ones:
                            count += 1
        
        return count
    
    def numSubmat_optimized_brute_force(self, mat: List[List[int]]) -> int:
        """
        Approach 4: Optimized Brute Force
        
        Use early termination and row-wise checking.
        
        Time: O(m² * n²), Space: O(1)
        """
        if not mat or not mat[0]:
            return 0
        
        rows, cols = len(mat), len(mat[0])
        count = 0
        
        for i1 in range(rows):
            for j1 in range(cols):
                if mat[i1][j1] == 0:
                    continue
                
                # Find maximum width for current starting position
                max_width = cols
                
                for i2 in range(i1, rows):
                    # Find width of all-ones row starting at (i2, j1)
                    width = 0
                    for j in range(j1, min(j1 + max_width, cols)):
                        if mat[i2][j] == 1:
                            width += 1
                        else:
                            break
                    
                    if width == 0:
                        break
                    
                    max_width = min(max_width, width)
                    
                    # Add all submatrices with height (i2 - i1 + 1) and width <= max_width
                    count += max_width
        
        return count
    
    def numSubmat_stack_optimized(self, mat: List[List[int]]) -> int:
        """
        Approach 5: Stack Optimized with Contribution Counting
        
        Use stack to efficiently count contributions.
        
        Time: O(m * n), Space: O(n)
        """
        if not mat or not mat[0]:
            return 0
        
        rows, cols = len(mat), len(mat[0])
        heights = [0] * cols
        total_count = 0
        
        def count_submatrices_with_stack(heights: List[int]) -> int:
            """Count submatrices using stack with contribution method"""
            stack = []
            count = 0
            
            for i in range(len(heights)):
                while stack and heights[stack[-1]] > heights[i]:
                    j = stack.pop()
                    h = heights[j]
                    w = i if not stack else i - stack[-1] - 1
                    
                    # Count contribution of rectangles with height h
                    count += h * w * (w + 1) // 2
                
                stack.append(i)
            
            # Process remaining elements
            while stack:
                j = stack.pop()
                h = heights[j]
                w = len(heights) if not stack else len(heights) - stack[-1] - 1
                count += h * w * (w + 1) // 2
            
            return count
        
        for row in mat:
            # Update heights
            for j in range(cols):
                heights[j] = heights[j] + 1 if row[j] == 1 else 0
            
            # Count submatrices in current histogram
            total_count += count_submatrices_with_stack(heights)
        
        return total_count
    
    def numSubmat_contribution_method(self, mat: List[List[int]]) -> int:
        """
        Approach 6: Contribution Method
        
        Calculate contribution of each cell to total count.
        
        Time: O(m * n²), Space: O(n)
        """
        if not mat or not mat[0]:
            return 0
        
        rows, cols = len(mat), len(mat[0])
        total_count = 0
        
        # For each starting row
        for start_row in range(rows):
            # Track consecutive ones in each column
            consecutive = [0] * cols
            
            # Extend downward from start_row
            for end_row in range(start_row, rows):
                # Update consecutive ones for current row
                for j in range(cols):
                    if mat[end_row][j] == 1:
                        consecutive[j] += 1
                    else:
                        consecutive[j] = 0
                
                # Count submatrices in current row range
                total_count += self._count_subarrays_with_min_height(consecutive, end_row - start_row + 1)
        
        return total_count
    
    def _count_subarrays_with_min_height(self, heights: List[int], min_height: int) -> int:
        """Count subarrays where all elements >= min_height"""
        count = 0
        length = 0
        
        for height in heights:
            if height >= min_height:
                length += 1
                count += length
            else:
                length = 0
        
        return count


def test_count_submatrices_with_all_ones():
    """Test count submatrices with all ones algorithms"""
    solver = CountSubmatricesWithAllOnes()
    
    test_cases = [
        ([[1,0,1],[1,1,0],[1,1,0]], 13, "Example 1"),
        ([[0,1,1,0],[0,1,1,1],[1,1,1,0]], 24, "Example 2"),
        ([[1,1,1,1,1,1]], 21, "Single row all ones"),
        ([[1],[1],[0],[1],[1]], 9, "Single column"),
        ([[0,0],[0,0]], 0, "All zeros"),
        ([[1,1],[1,1]], 9, "All ones 2x2"),
        ([[1]], 1, "Single cell"),
        ([[0]], 0, "Single zero"),
        ([[1,0,1,1,1],[1,0,1,1,1],[1,1,1,1,1]], 39, "Complex pattern"),
    ]
    
    algorithms = [
        ("Histogram Approach", solver.numSubmat_histogram_approach),
        ("DP Approach", solver.numSubmat_dp_approach),
        ("Brute Force", solver.numSubmat_brute_force),
        ("Optimized Brute Force", solver.numSubmat_optimized_brute_force),
        ("Stack Optimized", solver.numSubmat_stack_optimized),
    ]
    
    print("=== Testing Count Submatrices With All Ones ===")
    
    for mat, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print("Matrix:")
        for row in mat:
            print(f"  {row}")
        print(f"Expected: {expected}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(mat)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:20} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:40]}")


def demonstrate_histogram_approach():
    """Demonstrate histogram approach step by step"""
    print("\n=== Histogram Approach Step-by-Step Demo ===")
    
    mat = [
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 0]
    ]
    
    print("Matrix:")
    for i, row in enumerate(mat):
        print(f"Row {i}: {row}")
    
    print("\nConverting to histogram problem:")
    
    heights = [0] * len(mat[0])
    total_count = 0
    
    def count_rectangles_in_histogram(heights: List[int]) -> int:
        """Count rectangles in histogram with detailed output"""
        stack = []
        count = 0
        
        print(f"    Processing histogram: {heights}")
        
        for i in range(len(heights)):
            while stack and heights[stack[-1]] > heights[i]:
                j = stack.pop()
                h = heights[j]
                w = i if not stack else i - stack[-1] - 1
                
                # Count rectangles with height h and width <= w
                rect_count = h * w * (w + 1) // 2
                count += rect_count
                
                print(f"      Height {h}, Width {w}: {rect_count} rectangles")
            
            stack.append(i)
        
        # Process remaining elements
        while stack:
            j = stack.pop()
            h = heights[j]
            w = len(heights) if not stack else len(heights) - stack[-1] - 1
            rect_count = h * w * (w + 1) // 2
            count += rect_count
            
            print(f"      Height {h}, Width {w}: {rect_count} rectangles")
        
        print(f"    Total rectangles in this histogram: {count}")
        return count
    
    for i, row in enumerate(mat):
        print(f"\nProcessing row {i}: {row}")
        
        # Update heights
        for j in range(len(row)):
            if row[j] == 1:
                heights[j] += 1
            else:
                heights[j] = 0
        
        print(f"  Updated heights: {heights}")
        
        # Count rectangles in current histogram
        row_count = count_rectangles_in_histogram(heights)
        total_count += row_count
        
        print(f"  Total count so far: {total_count}")
    
    print(f"\nFinal result: {total_count}")


def demonstrate_rectangle_counting():
    """Demonstrate rectangle counting in histogram"""
    print("\n=== Rectangle Counting in Histogram ===")
    
    heights = [3, 1, 3, 2, 2]
    print(f"Histogram heights: {heights}")
    
    print("\nVisualizing histogram:")
    max_height = max(heights)
    for level in range(max_height, 0, -1):
        line = f"{level} |"
        for height in heights:
            if height >= level:
                line += "██"
            else:
                line += "  "
        print(line)
    
    print("  +" + "--" * len(heights))
    print("   " + "".join(f"{i:2}" for i in range(len(heights))))
    
    print("\nCounting rectangles:")
    print("For each height h and width w, number of rectangles = h * w * (w + 1) / 2")
    
    # Manual calculation for demonstration
    total = 0
    
    # Height 1 rectangles
    width_1 = 5  # All positions have height >= 1
    count_1 = 1 * width_1 * (width_1 + 1) // 2
    total += count_1
    print(f"Height 1, Width {width_1}: {count_1} rectangles")
    
    # Height 2 rectangles (positions 0, 3, 4)
    # Need to find continuous segments
    segments_2 = [(0, 1), (3, 5)]  # [start, end)
    count_2 = 0
    for start, end in segments_2:
        w = end - start
        c = 1 * w * (w + 1) // 2  # height = 2 - 1 = 1 additional
        count_2 += c
        print(f"Height 2, segment [{start}:{end}), width {w}: {c} additional rectangles")
    total += count_2
    
    # Height 3 rectangles (positions 0, 2)
    segments_3 = [(0, 1), (2, 3)]
    count_3 = 0
    for start, end in segments_3:
        w = end - start
        c = 1 * w * (w + 1) // 2  # height = 3 - 2 = 1 additional
        count_3 += c
        print(f"Height 3, segment [{start}:{end}), width {w}: {c} additional rectangles")
    total += count_3
    
    print(f"\nTotal rectangles: {total}")


def visualize_submatrix_counting():
    """Visualize submatrix counting"""
    print("\n=== Submatrix Counting Visualization ===")
    
    mat = [
        [1, 1],
        [1, 1]
    ]
    
    print("Matrix:")
    for row in mat:
        print("  " + " ".join(map(str, row)))
    
    print("\nAll possible submatrices:")
    
    rows, cols = len(mat), len(mat[0])
    count = 0
    
    for i1 in range(rows):
        for j1 in range(cols):
            for i2 in range(i1, rows):
                for j2 in range(j1, cols):
                    # Extract submatrix
                    submat = []
                    all_ones = True
                    
                    for i in range(i1, i2 + 1):
                        row = []
                        for j in range(j1, j2 + 1):
                            row.append(mat[i][j])
                            if mat[i][j] == 0:
                                all_ones = False
                        submat.append(row)
                    
                    if all_ones:
                        count += 1
                        print(f"  Submatrix [{i1},{j1}] to [{i2},{j2}]: {submat}")
    
    print(f"\nTotal count: {count}")


def benchmark_count_submatrices():
    """Benchmark different approaches"""
    import time
    import random
    
    algorithms = [
        ("Histogram Approach", CountSubmatricesWithAllOnes().numSubmat_histogram_approach),
        ("DP Approach", CountSubmatricesWithAllOnes().numSubmat_dp_approach),
        ("Optimized Brute Force", CountSubmatricesWithAllOnes().numSubmat_optimized_brute_force),
        ("Stack Optimized", CountSubmatricesWithAllOnes().numSubmat_stack_optimized),
    ]
    
    # Generate test matrices
    def generate_matrix(rows: int, cols: int) -> List[List[int]]:
        return [[random.randint(0, 1) for _ in range(cols)] for _ in range(rows)]
    
    test_sizes = [(10, 10), (20, 20), (30, 30)]
    
    print("\n=== Count Submatrices Performance Benchmark ===")
    
    for rows, cols in test_sizes:
        print(f"\n--- Matrix Size: {rows}x{cols} ---")
        
        mat = generate_matrix(rows, cols)
        
        for alg_name, alg_func in algorithms:
            start_time = time.time()
            
            try:
                result = alg_func(mat)
                end_time = time.time()
                print(f"{alg_name:20} | Time: {end_time - start_time:.4f}s | Result: {result}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:30]}")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    solver = CountSubmatricesWithAllOnes()
    
    edge_cases = [
        ([], 0, "Empty matrix"),
        ([[]], 0, "Empty row"),
        ([[0]], 0, "Single 0"),
        ([[1]], 1, "Single 1"),
        ([[0, 0], [0, 0]], 0, "All zeros"),
        ([[1, 1], [1, 1]], 9, "All ones 2x2"),
        ([[1, 0, 1]], 2, "Single row with zeros"),
        ([[1], [0], [1]], 2, "Single column with zeros"),
        ([[1, 1, 1]], 6, "Single row all ones"),
        ([[1], [1], [1]], 6, "Single column all ones"),
    ]
    
    for mat, expected, description in edge_cases:
        try:
            result = solver.numSubmat_histogram_approach(mat)
            status = "✓" if result == expected else "✗"
            print(f"{description:25} | {status} | Matrix: {mat} -> {result}")
        except Exception as e:
            print(f"{description:25} | ERROR: {str(e)[:30]}")


def compare_approaches():
    """Compare different approaches"""
    print("\n=== Approach Comparison ===")
    
    test_matrices = [
        [[1, 0, 1], [1, 1, 0], [1, 1, 0]],
        [[1, 1], [1, 1]],
        [[1, 1, 1], [1, 1, 1]],
        [[0, 1, 1, 0], [0, 1, 1, 1], [1, 1, 1, 0]],
    ]
    
    solver = CountSubmatricesWithAllOnes()
    
    approaches = [
        ("Histogram", solver.numSubmat_histogram_approach),
        ("DP", solver.numSubmat_dp_approach),
        ("Brute Force", solver.numSubmat_brute_force),
        ("Stack Optimized", solver.numSubmat_stack_optimized),
    ]
    
    for i, mat in enumerate(test_matrices):
        print(f"\nTest case {i+1}:")
        for row in mat:
            print("  " + " ".join(map(str, row)))
        
        results = {}
        
        for name, func in approaches:
            try:
                result = func(mat)
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
        ("Histogram Approach", "O(m * n)", "O(n)", "Convert to histogram + stack processing"),
        ("DP Approach", "O(m * n²)", "O(n)", "DP with width extension"),
        ("Brute Force", "O(m² * n²)", "O(1)", "Check all possible submatrices"),
        ("Optimized Brute Force", "O(m² * n²)", "O(1)", "Early termination optimization"),
        ("Stack Optimized", "O(m * n)", "O(n)", "Stack with contribution counting"),
        ("Contribution Method", "O(m * n²)", "O(n)", "Calculate each cell's contribution"),
    ]
    
    print(f"{'Approach':<25} | {'Time':<12} | {'Space':<8} | {'Notes'}")
    print("-" * 75)
    
    for approach, time_comp, space_comp, notes in approaches:
        print(f"{approach:<25} | {time_comp:<12} | {space_comp:<8} | {notes}")


def demonstrate_contribution_counting():
    """Demonstrate contribution counting method"""
    print("\n=== Contribution Counting Method ===")
    
    heights = [2, 1, 3]
    print(f"Histogram heights: {heights}")
    
    print("\nCounting rectangles by contribution:")
    print("Each bar contributes to rectangles where it's the minimum height")
    
    total = 0
    
    for i, h in enumerate(heights):
        print(f"\nBar {i} (height {h}):")
        
        # Find left boundary (first bar to the left with height < h)
        left = i
        while left > 0 and heights[left - 1] >= h:
            left -= 1
        
        # Find right boundary (first bar to the right with height < h)
        right = i
        while right < len(heights) - 1 and heights[right + 1] >= h:
            right += 1
        
        width = right - left + 1
        contribution = h * width * (width + 1) // 2
        total += contribution
        
        print(f"  Can extend from index {left} to {right} (width {width})")
        print(f"  Contribution: {h} * {width} * {width + 1} / 2 = {contribution}")
    
    print(f"\nTotal rectangles: {total}")


if __name__ == "__main__":
    test_count_submatrices_with_all_ones()
    demonstrate_histogram_approach()
    demonstrate_rectangle_counting()
    visualize_submatrix_counting()
    demonstrate_contribution_counting()
    test_edge_cases()
    compare_approaches()
    analyze_time_complexity()
    benchmark_count_submatrices()

"""
Count Submatrices With All Ones demonstrates advanced histogram-based
algorithms for 2D counting problems, including stack optimization and
contribution methods for efficient submatrix enumeration.
"""
