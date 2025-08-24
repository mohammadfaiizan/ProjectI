"""
84. Largest Rectangle in Histogram - Multiple Approaches
Difficulty: Hard

Given an array of integers heights representing the histogram's bar height where the width of each bar is 1, return the area of the largest rectangle in the histogram.
"""

from typing import List

class LargestRectangleInHistogram:
    """Multiple approaches to find largest rectangle in histogram"""
    
    def largestRectangleArea_stack_approach(self, heights: List[int]) -> int:
        """
        Approach 1: Monotonic Stack (Optimal)
        
        Use stack to find left and right boundaries for each bar.
        
        Time: O(n), Space: O(n)
        """
        stack = []  # Store indices
        max_area = 0
        
        for i in range(len(heights)):
            # Pop bars taller than current bar
            while stack and heights[stack[-1]] > heights[i]:
                height = heights[stack.pop()]
                width = i if not stack else i - stack[-1] - 1
                max_area = max(max_area, height * width)
            
            stack.append(i)
        
        # Process remaining bars in stack
        while stack:
            height = heights[stack.pop()]
            width = len(heights) if not stack else len(heights) - stack[-1] - 1
            max_area = max(max_area, height * width)
        
        return max_area
    
    def largestRectangleArea_brute_force(self, heights: List[int]) -> int:
        """
        Approach 2: Brute Force
        
        For each bar, find maximum rectangle with that bar as minimum height.
        
        Time: O(n²), Space: O(1)
        """
        max_area = 0
        n = len(heights)
        
        for i in range(n):
            min_height = heights[i]
            
            for j in range(i, n):
                min_height = min(min_height, heights[j])
                width = j - i + 1
                area = min_height * width
                max_area = max(max_area, area)
        
        return max_area
    
    def largestRectangleArea_divide_conquer(self, heights: List[int]) -> int:
        """
        Approach 3: Divide and Conquer
        
        Recursively find maximum in left, right, and across minimum.
        
        Time: O(n log n) average, O(n²) worst case, Space: O(log n)
        """
        def find_max_area(left: int, right: int) -> int:
            if left > right:
                return 0
            
            # Find minimum height and its index
            min_height = float('inf')
            min_idx = left
            
            for i in range(left, right + 1):
                if heights[i] < min_height:
                    min_height = heights[i]
                    min_idx = i
            
            # Calculate area with minimum height spanning entire range
            area_with_min = min_height * (right - left + 1)
            
            # Recursively find maximum in left and right parts
            left_max = find_max_area(left, min_idx - 1)
            right_max = find_max_area(min_idx + 1, right)
            
            return max(area_with_min, left_max, right_max)
        
        return find_max_area(0, len(heights) - 1)
    
    def largestRectangleArea_left_right_arrays(self, heights: List[int]) -> int:
        """
        Approach 4: Precompute Left and Right Boundaries
        
        Use arrays to store left and right boundaries for each bar.
        
        Time: O(n), Space: O(n)
        """
        n = len(heights)
        if n == 0:
            return 0
        
        # Arrays to store left and right boundaries
        left = [0] * n   # left[i] = leftmost index where height >= heights[i]
        right = [0] * n  # right[i] = rightmost index where height >= heights[i]
        
        # Fill left array
        left[0] = 0
        for i in range(1, n):
            p = i - 1
            while p >= 0 and heights[p] >= heights[i]:
                p = left[p] - 1
            left[i] = p + 1
        
        # Fill right array
        right[n - 1] = n - 1
        for i in range(n - 2, -1, -1):
            p = i + 1
            while p < n and heights[p] >= heights[i]:
                p = right[p] + 1
            right[i] = p - 1
        
        # Calculate maximum area
        max_area = 0
        for i in range(n):
            width = right[i] - left[i] + 1
            area = heights[i] * width
            max_area = max(max_area, area)
        
        return max_area
    
    def largestRectangleArea_optimized_stack(self, heights: List[int]) -> int:
        """
        Approach 5: Optimized Stack with Sentinel
        
        Add sentinel values to simplify edge cases.
        
        Time: O(n), Space: O(n)
        """
        # Add sentinel values
        heights = [0] + heights + [0]
        stack = []
        max_area = 0
        
        for i in range(len(heights)):
            while stack and heights[stack[-1]] > heights[i]:
                height = heights[stack.pop()]
                width = i - stack[-1] - 1
                max_area = max(max_area, height * width)
            
            stack.append(i)
        
        return max_area
    
    def largestRectangleArea_segment_tree(self, heights: List[int]) -> int:
        """
        Approach 6: Segment Tree for Range Minimum Query
        
        Use segment tree to find minimum in ranges efficiently.
        
        Time: O(n log n), Space: O(n)
        """
        n = len(heights)
        if n == 0:
            return 0
        
        # Build segment tree for range minimum query
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
                
                if heights[left_idx] <= heights[right_idx]:
                    tree[node] = left_idx
                else:
                    tree[node] = right_idx
        
        def query(node: int, start: int, end: int, l: int, r: int) -> int:
            if r < start or end < l:
                return -1
            
            if l <= start and end <= r:
                return tree[node]
            
            mid = (start + end) // 2
            left_idx = query(2 * node, start, mid, l, r)
            right_idx = query(2 * node + 1, mid + 1, end, l, r)
            
            if left_idx == -1:
                return right_idx
            if right_idx == -1:
                return left_idx
            
            return left_idx if heights[left_idx] <= heights[right_idx] else right_idx
        
        def find_max_area(left: int, right: int) -> int:
            if left > right:
                return 0
            
            min_idx = query(1, 0, n - 1, left, right)
            area_with_min = heights[min_idx] * (right - left + 1)
            
            left_max = find_max_area(left, min_idx - 1)
            right_max = find_max_area(min_idx + 1, right)
            
            return max(area_with_min, left_max, right_max)
        
        build(1, 0, n - 1)
        return find_max_area(0, n - 1)


def test_largest_rectangle_in_histogram():
    """Test largest rectangle in histogram algorithms"""
    solver = LargestRectangleInHistogram()
    
    test_cases = [
        ([2,1,5,6,2,3], 10, "Example 1"),
        ([2,4], 4, "Example 2"),
        ([1,1], 2, "All same height"),
        ([2,1,2], 3, "Valley shape"),
        ([1,2,3,4,5], 9, "Increasing heights"),
        ([5,4,3,2,1], 9, "Decreasing heights"),
        ([0,2,0], 2, "With zero heights"),
        ([1], 1, "Single bar"),
        ([], 0, "Empty array"),
        ([2,0,2], 2, "Zero in middle"),
        ([6,7,5,2,4,5,9,3], 16, "Complex case"),
    ]
    
    algorithms = [
        ("Stack Approach", solver.largestRectangleArea_stack_approach),
        ("Brute Force", solver.largestRectangleArea_brute_force),
        ("Divide Conquer", solver.largestRectangleArea_divide_conquer),
        ("Left Right Arrays", solver.largestRectangleArea_left_right_arrays),
        ("Optimized Stack", solver.largestRectangleArea_optimized_stack),
    ]
    
    print("=== Testing Largest Rectangle in Histogram ===")
    
    for heights, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Heights: {heights}")
        print(f"Expected: {expected}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(heights)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:20} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:40]}")


def demonstrate_stack_approach():
    """Demonstrate stack approach step by step"""
    print("\n=== Stack Approach Step-by-Step Demo ===")
    
    heights = [2, 1, 5, 6, 2, 3]
    print(f"Heights: {heights}")
    print("Processing with monotonic stack:")
    
    stack = []
    max_area = 0
    
    for i in range(len(heights)):
        print(f"\nStep {i+1}: Processing height {heights[i]} at index {i}")
        print(f"  Stack before: {[f'{idx}({heights[idx]})' for idx in stack]}")
        
        # Pop bars taller than current
        while stack and heights[stack[-1]] > heights[i]:
            height_idx = stack.pop()
            height = heights[height_idx]
            width = i if not stack else i - stack[-1] - 1
            area = height * width
            max_area = max(max_area, area)
            
            print(f"  Popped index {height_idx} (height {height})")
            print(f"    Width calculation: {i} - {stack[-1] if stack else 'start'} - 1 = {width}")
            print(f"    Area: {height} × {width} = {area}")
            print(f"    Max area so far: {max_area}")
        
        stack.append(i)
        print(f"  Stack after: {[f'{idx}({heights[idx]})' for idx in stack]}")
    
    # Process remaining bars
    print(f"\nProcessing remaining bars in stack:")
    while stack:
        height_idx = stack.pop()
        height = heights[height_idx]
        width = len(heights) if not stack else len(heights) - stack[-1] - 1
        area = height * width
        max_area = max(max_area, area)
        
        print(f"  Popped index {height_idx} (height {height})")
        print(f"    Width: {width}, Area: {area}")
        print(f"    Max area so far: {max_area}")
    
    print(f"\nFinal maximum area: {max_area}")


def visualize_histogram():
    """Visualize histogram and largest rectangle"""
    print("\n=== Histogram Visualization ===")
    
    heights = [2, 1, 5, 6, 2, 3]
    print(f"Heights: {heights}")
    
    # Find the largest rectangle
    solver = LargestRectangleInHistogram()
    max_area = solver.largestRectangleArea_stack_approach(heights)
    
    # Create visual representation
    max_height = max(heights) if heights else 0
    
    print("\nHistogram visualization:")
    for level in range(max_height, 0, -1):
        line = ""
        for height in heights:
            if height >= level:
                line += "██ "
            else:
                line += "   "
        print(f"{level} |{line}")
    
    # Print base
    print("  +" + "---" * len(heights))
    print("   " + "".join(f"{i:3}" for i in range(len(heights))))
    
    print(f"\nLargest rectangle area: {max_area}")


def demonstrate_rectangle_calculation():
    """Demonstrate how rectangles are calculated"""
    print("\n=== Rectangle Calculation Demonstration ===")
    
    heights = [2, 1, 5, 6, 2, 3]
    print(f"Heights: {heights}")
    print("\nFor each bar, find the largest rectangle with that bar as the minimum height:")
    
    for i, height in enumerate(heights):
        # Find left boundary
        left = i
        while left > 0 and heights[left - 1] >= height:
            left -= 1
        
        # Find right boundary
        right = i
        while right < len(heights) - 1 and heights[right + 1] >= height:
            right += 1
        
        width = right - left + 1
        area = height * width
        
        print(f"\nBar {i} (height {height}):")
        print(f"  Left boundary: {left}, Right boundary: {right}")
        print(f"  Width: {width}, Area: {height} × {width} = {area}")
        
        # Show the rectangle
        rect_heights = heights[left:right+1]
        print(f"  Rectangle spans: {rect_heights}")


def benchmark_largest_rectangle():
    """Benchmark different approaches"""
    import time
    import random
    
    algorithms = [
        ("Stack Approach", LargestRectangleInHistogram().largestRectangleArea_stack_approach),
        ("Brute Force", LargestRectangleInHistogram().largestRectangleArea_brute_force),
        ("Divide Conquer", LargestRectangleInHistogram().largestRectangleArea_divide_conquer),
        ("Left Right Arrays", LargestRectangleInHistogram().largestRectangleArea_left_right_arrays),
    ]
    
    # Test with different array sizes
    sizes = [100, 1000, 5000]
    
    print("\n=== Largest Rectangle Performance Benchmark ===")
    
    for size in sizes:
        print(f"\n--- Array Size: {size} ---")
        
        # Generate random heights
        heights = [random.randint(1, 100) for _ in range(size)]
        
        for alg_name, alg_func in algorithms:
            start_time = time.time()
            
            try:
                result = alg_func(heights)
                end_time = time.time()
                print(f"{alg_name:20} | Time: {end_time - start_time:.4f}s | Result: {result}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:30]}")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    solver = LargestRectangleInHistogram()
    
    edge_cases = [
        ([], 0, "Empty array"),
        ([0], 0, "Single zero"),
        ([5], 5, "Single bar"),
        ([1, 1, 1], 3, "All same height"),
        ([1, 2, 3], 4, "Increasing"),
        ([3, 2, 1], 4, "Decreasing"),
        ([0, 1, 0], 1, "Zero boundaries"),
        ([1, 0, 1], 1, "Zero in middle"),
        ([100], 100, "Large single bar"),
        ([1, 100, 1], 100, "Peak in middle"),
    ]
    
    for heights, expected, description in edge_cases:
        try:
            result = solver.largestRectangleArea_stack_approach(heights)
            status = "✓" if result == expected else "✗"
            print(f"{description:20} | {status} | heights: {heights} -> {result}")
        except Exception as e:
            print(f"{description:20} | ERROR: {str(e)[:30]}")


def compare_stack_vs_brute_force():
    """Compare stack vs brute force approaches"""
    print("\n=== Stack vs Brute Force Comparison ===")
    
    test_cases = [
        [2, 1, 5, 6, 2, 3],
        [1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1],
        [3, 3, 3, 3],
    ]
    
    solver = LargestRectangleInHistogram()
    
    for i, heights in enumerate(test_cases):
        print(f"\nTest case {i+1}: {heights}")
        
        # Stack approach
        result_stack = solver.largestRectangleArea_stack_approach(heights)
        print(f"Stack approach:  {result_stack}")
        
        # Brute force approach
        result_brute = solver.largestRectangleArea_brute_force(heights)
        print(f"Brute force:     {result_brute}")
        
        # Check consistency
        consistent = result_stack == result_brute
        print(f"Results match:   {'✓' if consistent else '✗'}")


def analyze_time_complexity():
    """Analyze time complexity of different approaches"""
    print("\n=== Time Complexity Analysis ===")
    
    approaches = [
        ("Stack Approach", "O(n)", "O(n)", "Optimal - each element pushed/popped once"),
        ("Brute Force", "O(n²)", "O(1)", "Check all possible rectangles"),
        ("Divide Conquer", "O(n log n)", "O(log n)", "Average case, O(n²) worst case"),
        ("Left Right Arrays", "O(n)", "O(n)", "Three passes through array"),
        ("Optimized Stack", "O(n)", "O(n)", "Stack with sentinel values"),
        ("Segment Tree", "O(n log n)", "O(n)", "Range minimum queries"),
    ]
    
    print(f"{'Approach':<20} | {'Time':<12} | {'Space':<10} | {'Notes'}")
    print("-" * 75)
    
    for approach, time_comp, space_comp, notes in approaches:
        print(f"{approach:<20} | {time_comp:<12} | {space_comp:<10} | {notes}")


def demonstrate_monotonic_stack_property():
    """Demonstrate monotonic stack property"""
    print("\n=== Monotonic Stack Property Demonstration ===")
    
    heights = [6, 7, 5, 2, 4, 5, 9, 3]
    print(f"Processing heights: {heights}")
    print("Stack maintains increasing order (indices of increasing heights):")
    
    stack = []
    
    for i, height in enumerate(heights):
        print(f"\nStep {i+1}: Processing height {height} at index {i}")
        print(f"  Stack before: {[f'{idx}({heights[idx]})' for idx in stack]}")
        
        # Show what gets popped and why
        popped = []
        while stack and heights[stack[-1]] > height:
            popped_idx = stack.pop()
            popped.append(f'{popped_idx}({heights[popped_idx]})')
        
        if popped:
            print(f"  Popped: {popped} (heights > {height})")
        
        stack.append(i)
        print(f"  Stack after: {[f'{idx}({heights[idx]})' for idx in stack]}")
        
        # Verify monotonic property
        stack_heights = [heights[idx] for idx in stack]
        is_increasing = all(stack_heights[j] <= stack_heights[j+1] for j in range(len(stack_heights)-1))
        print(f"  Monotonic property: {'✓' if is_increasing else '✗'} (heights: {stack_heights})")


if __name__ == "__main__":
    test_largest_rectangle_in_histogram()
    demonstrate_stack_approach()
    visualize_histogram()
    demonstrate_rectangle_calculation()
    demonstrate_monotonic_stack_property()
    test_edge_cases()
    compare_stack_vs_brute_force()
    analyze_time_complexity()
    benchmark_largest_rectangle()

"""
Largest Rectangle in Histogram demonstrates advanced monotonic stack
techniques for geometric problems, including multiple approaches for
finding maximum rectangular areas with comprehensive analysis.
"""
