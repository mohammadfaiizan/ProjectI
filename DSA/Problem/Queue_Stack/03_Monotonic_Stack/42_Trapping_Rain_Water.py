"""
42. Trapping Rain Water - Multiple Approaches
Difficulty: Hard

Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it can trap after raining.
"""

from typing import List

class TrappingRainWater:
    """Multiple approaches to solve trapping rain water problem"""
    
    def trap_two_pointers(self, height: List[int]) -> int:
        """
        Approach 1: Two Pointers (Optimal)
        
        Use two pointers moving towards each other.
        
        Time: O(n), Space: O(1)
        """
        if not height or len(height) < 3:
            return 0
        
        left, right = 0, len(height) - 1
        left_max, right_max = 0, 0
        water = 0
        
        while left < right:
            if height[left] < height[right]:
                if height[left] >= left_max:
                    left_max = height[left]
                else:
                    water += left_max - height[left]
                left += 1
            else:
                if height[right] >= right_max:
                    right_max = height[right]
                else:
                    water += right_max - height[right]
                right -= 1
        
        return water
    
    def trap_stack_approach(self, height: List[int]) -> int:
        """
        Approach 2: Monotonic Stack
        
        Use stack to find water trapped between bars.
        
        Time: O(n), Space: O(n)
        """
        stack = []  # Store indices
        water = 0
        
        for i in range(len(height)):
            # Pop bars shorter than current bar
            while stack and height[stack[-1]] < height[i]:
                bottom = stack.pop()
                
                if not stack:
                    break
                
                # Calculate trapped water
                distance = i - stack[-1] - 1
                bounded_height = min(height[i], height[stack[-1]]) - height[bottom]
                water += distance * bounded_height
            
            stack.append(i)
        
        return water
    
    def trap_dp_approach(self, height: List[int]) -> int:
        """
        Approach 3: Dynamic Programming
        
        Precompute left and right maximum heights.
        
        Time: O(n), Space: O(n)
        """
        if not height:
            return 0
        
        n = len(height)
        left_max = [0] * n
        right_max = [0] * n
        
        # Fill left_max array
        left_max[0] = height[0]
        for i in range(1, n):
            left_max[i] = max(left_max[i-1], height[i])
        
        # Fill right_max array
        right_max[n-1] = height[n-1]
        for i in range(n-2, -1, -1):
            right_max[i] = max(right_max[i+1], height[i])
        
        # Calculate trapped water
        water = 0
        for i in range(n):
            water += min(left_max[i], right_max[i]) - height[i]
        
        return water
    
    def trap_brute_force(self, height: List[int]) -> int:
        """
        Approach 4: Brute Force
        
        For each position, find left and right maximum.
        
        Time: O(n²), Space: O(1)
        """
        water = 0
        n = len(height)
        
        for i in range(1, n-1):  # Skip first and last positions
            # Find maximum height to the left
            left_max = 0
            for j in range(i):
                left_max = max(left_max, height[j])
            
            # Find maximum height to the right
            right_max = 0
            for j in range(i+1, n):
                right_max = max(right_max, height[j])
            
            # Calculate water at current position
            water_level = min(left_max, right_max)
            if water_level > height[i]:
                water += water_level - height[i]
        
        return water
    
    def trap_divide_conquer(self, height: List[int]) -> int:
        """
        Approach 5: Divide and Conquer
        
        Recursively solve for left and right parts.
        
        Time: O(n log n), Space: O(log n)
        """
        def trap_recursive(left: int, right: int, left_max: int, right_max: int) -> int:
            if left >= right:
                return 0
            
            if left_max <= right_max:
                # Process from left
                if height[left] < left_max:
                    return left_max - height[left] + trap_recursive(left + 1, right, left_max, right_max)
                else:
                    return trap_recursive(left + 1, right, height[left], right_max)
            else:
                # Process from right
                if height[right] < right_max:
                    return right_max - height[right] + trap_recursive(left, right - 1, left_max, right_max)
                else:
                    return trap_recursive(left, right - 1, left_max, height[right])
        
        if not height or len(height) < 3:
            return 0
        
        return trap_recursive(0, len(height) - 1, 0, 0)
    
    def trap_segment_approach(self, height: List[int]) -> int:
        """
        Approach 6: Segment-based Processing
        
        Process array in segments for better cache locality.
        
        Time: O(n), Space: O(1)
        """
        if not height or len(height) < 3:
            return 0
        
        total_water = 0
        segment_size = min(len(height), 1000)
        
        for start in range(0, len(height), segment_size):
            end = min(start + segment_size, len(height))
            
            # Use two pointers for current segment
            left, right = start, end - 1
            left_max, right_max = 0, 0
            segment_water = 0
            
            while left < right:
                if height[left] < height[right]:
                    if height[left] >= left_max:
                        left_max = height[left]
                    else:
                        segment_water += left_max - height[left]
                    left += 1
                else:
                    if height[right] >= right_max:
                        right_max = height[right]
                    else:
                        segment_water += right_max - height[right]
                    right -= 1
            
            total_water += segment_water
        
        return total_water


def test_trapping_rain_water():
    """Test trapping rain water algorithms"""
    solver = TrappingRainWater()
    
    test_cases = [
        ([0,1,0,2,1,0,1,3,2,1,2,1], 6, "Example 1"),
        ([4,2,0,3,2,5], 9, "Example 2"),
        ([3,0,2,0,4], 7, "Valley pattern"),
        ([0,1,0], 0, "Single peak"),
        ([1,0,1], 1, "Simple valley"),
        ([2,1,2], 1, "Symmetric valley"),
        ([1,2,3], 0, "Increasing"),
        ([3,2,1], 0, "Decreasing"),
        ([1], 0, "Single element"),
        ([], 0, "Empty array"),
        ([5,4,1,2], 1, "Complex pattern"),
        ([0,2,0,4,0,3,0,1], 7, "Multiple valleys"),
    ]
    
    algorithms = [
        ("Two Pointers", solver.trap_two_pointers),
        ("Stack Approach", solver.trap_stack_approach),
        ("DP Approach", solver.trap_dp_approach),
        ("Brute Force", solver.trap_brute_force),
        ("Divide Conquer", solver.trap_divide_conquer),
    ]
    
    print("=== Testing Trapping Rain Water ===")
    
    for height, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Height: {height}")
        print(f"Expected: {expected}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(height)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:15} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:15} | ERROR: {str(e)[:40]}")


def demonstrate_two_pointers_approach():
    """Demonstrate two pointers approach step by step"""
    print("\n=== Two Pointers Approach Step-by-Step Demo ===")
    
    height = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]
    print(f"Height array: {height}")
    
    left, right = 0, len(height) - 1
    left_max, right_max = 0, 0
    water = 0
    
    step = 1
    
    while left < right:
        print(f"\nStep {step}: left={left}, right={right}")
        print(f"  height[left]={height[left]}, height[right]={height[right]}")
        print(f"  left_max={left_max}, right_max={right_max}")
        
        if height[left] < height[right]:
            if height[left] >= left_max:
                left_max = height[left]
                print(f"  Updated left_max to {left_max}")
            else:
                trapped = left_max - height[left]
                water += trapped
                print(f"  Trapped {trapped} units at position {left}")
            left += 1
            print(f"  Moved left pointer to {left}")
        else:
            if height[right] >= right_max:
                right_max = height[right]
                print(f"  Updated right_max to {right_max}")
            else:
                trapped = right_max - height[right]
                water += trapped
                print(f"  Trapped {trapped} units at position {right}")
            right -= 1
            print(f"  Moved right pointer to {right}")
        
        print(f"  Total water so far: {water}")
        step += 1
    
    print(f"\nFinal result: {water} units of water trapped")


def demonstrate_stack_approach():
    """Demonstrate stack approach step by step"""
    print("\n=== Stack Approach Step-by-Step Demo ===")
    
    height = [3, 0, 2, 0, 4]
    print(f"Height array: {height}")
    
    stack = []
    water = 0
    
    for i in range(len(height)):
        print(f"\nStep {i+1}: Processing height {height[i]} at index {i}")
        print(f"  Stack before: {[f'{idx}({height[idx]})' for idx in stack]}")
        
        while stack and height[stack[-1]] < height[i]:
            bottom = stack.pop()
            print(f"  Popped index {bottom} (height {height[bottom]}) as bottom")
            
            if not stack:
                print(f"  No left boundary, cannot trap water")
                break
            
            distance = i - stack[-1] - 1
            bounded_height = min(height[i], height[stack[-1]]) - height[bottom]
            trapped = distance * bounded_height
            water += trapped
            
            print(f"  Left boundary: index {stack[-1]} (height {height[stack[-1]]})")
            print(f"  Right boundary: index {i} (height {height[i]})")
            print(f"  Distance: {i} - {stack[-1]} - 1 = {distance}")
            print(f"  Water height: min({height[i]}, {height[stack[-1]]}) - {height[bottom]} = {bounded_height}")
            print(f"  Water trapped: {distance} × {bounded_height} = {trapped}")
            print(f"  Total water: {water}")
        
        stack.append(i)
        print(f"  Stack after: {[f'{idx}({height[idx]})' for idx in stack]}")
    
    print(f"\nFinal result: {water} units of water trapped")


def visualize_water_trapping():
    """Visualize water trapping"""
    print("\n=== Water Trapping Visualization ===")
    
    height = [3, 0, 2, 0, 4]
    print(f"Height array: {height}")
    
    # Calculate trapped water using DP approach for visualization
    solver = TrappingRainWater()
    n = len(height)
    
    # Calculate left and right max arrays
    left_max = [0] * n
    right_max = [0] * n
    
    left_max[0] = height[0]
    for i in range(1, n):
        left_max[i] = max(left_max[i-1], height[i])
    
    right_max[n-1] = height[n-1]
    for i in range(n-2, -1, -1):
        right_max[i] = max(right_max[i+1], height[i])
    
    # Calculate water at each position
    water_levels = []
    for i in range(n):
        water_level = min(left_max[i], right_max[i])
        water_levels.append(water_level)
    
    # Create visualization
    max_height = max(max(height), max(water_levels))
    
    print("\nVisualization (█ = ground, ≈ = water, · = air):")
    for level in range(max_height, 0, -1):
        line = f"{level} |"
        for i in range(n):
            if height[i] >= level:
                line += "██"
            elif water_levels[i] >= level:
                line += "≈≈"
            else:
                line += "··"
        print(line)
    
    # Print base
    print("  +" + "--" * n)
    print("   " + "".join(f"{i:2}" for i in range(n)))
    
    # Show water calculation
    total_water = 0
    print(f"\nWater calculation:")
    for i in range(n):
        trapped = max(0, water_levels[i] - height[i])
        total_water += trapped
        print(f"Position {i}: min({left_max[i]}, {right_max[i]}) - {height[i]} = {trapped}")
    
    print(f"Total water trapped: {total_water}")


def benchmark_trapping_rain_water():
    """Benchmark different approaches"""
    import time
    import random
    
    algorithms = [
        ("Two Pointers", TrappingRainWater().trap_two_pointers),
        ("Stack Approach", TrappingRainWater().trap_stack_approach),
        ("DP Approach", TrappingRainWater().trap_dp_approach),
        ("Brute Force", TrappingRainWater().trap_brute_force),
    ]
    
    # Test with different array sizes
    sizes = [100, 1000, 5000]
    
    print("\n=== Trapping Rain Water Performance Benchmark ===")
    
    for size in sizes:
        print(f"\n--- Array Size: {size} ---")
        
        # Generate random heights
        height = [random.randint(0, 100) for _ in range(size)]
        
        for alg_name, alg_func in algorithms:
            start_time = time.time()
            
            try:
                result = alg_func(height)
                end_time = time.time()
                print(f"{alg_name:15} | Time: {end_time - start_time:.4f}s | Result: {result}")
            except Exception as e:
                print(f"{alg_name:15} | ERROR: {str(e)[:30]}")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    solver = TrappingRainWater()
    
    edge_cases = [
        ([], 0, "Empty array"),
        ([1], 0, "Single element"),
        ([1, 2], 0, "Two elements"),
        ([0, 0, 0], 0, "All zeros"),
        ([5, 5, 5], 0, "All same height"),
        ([1, 0, 1], 1, "Simple valley"),
        ([2, 0, 2], 2, "Deeper valley"),
        ([1, 2, 3, 4], 0, "Strictly increasing"),
        ([4, 3, 2, 1], 0, "Strictly decreasing"),
        ([0, 1, 0, 1, 0], 1, "Alternating pattern"),
    ]
    
    for height, expected, description in edge_cases:
        try:
            result = solver.trap_two_pointers(height)
            status = "✓" if result == expected else "✗"
            print(f"{description:20} | {status} | height: {height} -> {result}")
        except Exception as e:
            print(f"{description:20} | ERROR: {str(e)[:30]}")


def compare_approaches():
    """Compare different approaches"""
    print("\n=== Approach Comparison ===")
    
    test_cases = [
        [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1],
        [4, 2, 0, 3, 2, 5],
        [3, 0, 2, 0, 4],
    ]
    
    solver = TrappingRainWater()
    
    approaches = [
        ("Two Pointers", solver.trap_two_pointers),
        ("Stack", solver.trap_stack_approach),
        ("DP", solver.trap_dp_approach),
        ("Brute Force", solver.trap_brute_force),
    ]
    
    for i, height in enumerate(test_cases):
        print(f"\nTest case {i+1}: {height}")
        
        results = {}
        
        for name, func in approaches:
            try:
                result = func(height)
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
        ("Two Pointers", "O(n)", "O(1)", "Optimal - single pass with constant space"),
        ("Stack Approach", "O(n)", "O(n)", "Single pass but uses stack space"),
        ("DP Approach", "O(n)", "O(n)", "Three passes with extra arrays"),
        ("Brute Force", "O(n²)", "O(1)", "For each position, scan left and right"),
        ("Divide Conquer", "O(n log n)", "O(log n)", "Recursive approach"),
        ("Segment Approach", "O(n)", "O(1)", "Process in segments"),
    ]
    
    print(f"{'Approach':<15} | {'Time':<12} | {'Space':<8} | {'Notes'}")
    print("-" * 65)
    
    for approach, time_comp, space_comp, notes in approaches:
        print(f"{approach:<15} | {time_comp:<12} | {space_comp:<8} | {notes}")


def demonstrate_water_physics():
    """Demonstrate water physics concept"""
    print("\n=== Water Physics Demonstration ===")
    
    height = [3, 0, 2, 0, 4]
    print(f"Elevation map: {height}")
    print("\nWater physics rules:")
    print("1. Water flows to lower elevations")
    print("2. Water is trapped between higher elevations")
    print("3. Water level at any point = min(max_left, max_right)")
    
    print(f"\nAnalyzing each position:")
    
    for i in range(len(height)):
        # Find max height to the left
        max_left = max(height[:i+1])
        
        # Find max height to the right
        max_right = max(height[i:])
        
        # Water level at this position
        water_level = min(max_left, max_right)
        
        # Trapped water
        trapped = max(0, water_level - height[i])
        
        print(f"Position {i} (elevation {height[i]}):")
        print(f"  Max left: {max_left}, Max right: {max_right}")
        print(f"  Water level: min({max_left}, {max_right}) = {water_level}")
        print(f"  Trapped water: {water_level} - {height[i]} = {trapped}")
        print()


def demonstrate_stack_water_calculation():
    """Demonstrate how stack calculates water between bars"""
    print("\n=== Stack Water Calculation Demonstration ===")
    
    height = [4, 2, 0, 3, 2, 5]
    print(f"Height array: {height}")
    print("Stack approach calculates water horizontally (layer by layer):")
    
    stack = []
    water = 0
    
    for i in range(len(height)):
        print(f"\nProcessing bar {i} (height {height[i]}):")
        
        layer = 1
        while stack and height[stack[-1]] < height[i]:
            bottom_idx = stack.pop()
            bottom_height = height[bottom_idx]
            
            print(f"  Layer {layer}: Bottom at index {bottom_idx} (height {bottom_height})")
            
            if not stack:
                print(f"    No left wall, water flows away")
                break
            
            left_idx = stack[-1]
            left_height = height[left_idx]
            right_height = height[i]
            
            width = i - left_idx - 1
            water_height = min(left_height, right_height) - bottom_height
            layer_water = width * water_height
            water += layer_water
            
            print(f"    Left wall: index {left_idx} (height {left_height})")
            print(f"    Right wall: index {i} (height {right_height})")
            print(f"    Width: {i} - {left_idx} - 1 = {width}")
            print(f"    Water height: min({left_height}, {right_height}) - {bottom_height} = {water_height}")
            print(f"    Layer water: {width} × {water_height} = {layer_water}")
            print(f"    Total water: {water}")
            
            layer += 1
        
        stack.append(i)
        print(f"  Added bar {i} to stack: {stack}")
    
    print(f"\nTotal water trapped: {water}")


if __name__ == "__main__":
    test_trapping_rain_water()
    demonstrate_two_pointers_approach()
    demonstrate_stack_approach()
    visualize_water_trapping()
    demonstrate_water_physics()
    demonstrate_stack_water_calculation()
    test_edge_cases()
    compare_approaches()
    analyze_time_complexity()
    benchmark_trapping_rain_water()

"""
Trapping Rain Water demonstrates multiple advanced algorithms for
geometric water simulation including two pointers, monotonic stack,
dynamic programming, and divide-and-conquer approaches with visualization.
"""
