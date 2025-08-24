"""
1762. Buildings With an Ocean View - Multiple Approaches
Difficulty: Medium

There are n buildings in a line. You are given an integer array heights of size n that represents the heights of the buildings in the line.

The ocean is to the right of the buildings. A building has an ocean view if the building can see the ocean without being blocked by another building. Formally, a building has an ocean view if all the buildings to its right have a strictly smaller height.

Return a list of indices (0-indexed) of buildings that have an ocean view, sorted in increasing order.
"""

from typing import List

class BuildingsWithOceanView:
    """Multiple approaches to find buildings with ocean view"""
    
    def findBuildings_right_to_left(self, heights: List[int]) -> List[int]:
        """
        Approach 1: Right to Left Traversal (Optimal)
        
        Traverse from right to left, track maximum height seen so far.
        
        Time: O(n), Space: O(1) excluding output
        """
        result = []
        max_height = 0
        
        # Traverse from right to left
        for i in range(len(heights) - 1, -1, -1):
            if heights[i] > max_height:
                result.append(i)
                max_height = heights[i]
        
        # Reverse to get increasing order
        return result[::-1]
    
    def findBuildings_stack(self, heights: List[int]) -> List[int]:
        """
        Approach 2: Monotonic Stack
        
        Use monotonic decreasing stack to find buildings with ocean view.
        
        Time: O(n), Space: O(n)
        """
        stack = []
        
        for i, height in enumerate(heights):
            # Remove buildings that are blocked by current building
            while stack and heights[stack[-1]] <= height:
                stack.pop()
            
            stack.append(i)
        
        return stack
    
    def findBuildings_brute_force(self, heights: List[int]) -> List[int]:
        """
        Approach 3: Brute Force
        
        For each building, check if all buildings to its right are shorter.
        
        Time: O(n²), Space: O(1) excluding output
        """
        result = []
        n = len(heights)
        
        for i in range(n):
            has_ocean_view = True
            
            # Check all buildings to the right
            for j in range(i + 1, n):
                if heights[j] >= heights[i]:
                    has_ocean_view = False
                    break
            
            if has_ocean_view:
                result.append(i)
        
        return result
    
    def findBuildings_suffix_maximum(self, heights: List[int]) -> List[int]:
        """
        Approach 4: Suffix Maximum Array
        
        Precompute suffix maximum array and use it to find ocean view buildings.
        
        Time: O(n), Space: O(n)
        """
        n = len(heights)
        if n == 0:
            return []
        
        # Build suffix maximum array
        suffix_max = [0] * n
        suffix_max[-1] = heights[-1]
        
        for i in range(n - 2, -1, -1):
            suffix_max[i] = max(heights[i], suffix_max[i + 1])
        
        result = []
        for i in range(n):
            # Building i has ocean view if it's taller than all buildings to its right
            if i == n - 1 or heights[i] > suffix_max[i + 1]:
                result.append(i)
        
        return result
    
    def findBuildings_divide_conquer(self, heights: List[int]) -> List[int]:
        """
        Approach 5: Divide and Conquer
        
        Use divide and conquer to find buildings with ocean view.
        
        Time: O(n log n), Space: O(log n) due to recursion
        """
        def solve(left: int, right: int) -> List[int]:
            if left > right:
                return []
            
            if left == right:
                return [left]
            
            mid = (left + right) // 2
            
            # Get results from left and right parts
            left_result = solve(left, mid)
            right_result = solve(mid + 1, right)
            
            # Merge results
            result = []
            
            # Add buildings from left part that are not blocked by right part
            right_max = max(heights[mid + 1:right + 1]) if mid + 1 <= right else 0
            
            for idx in left_result:
                if heights[idx] > right_max:
                    result.append(idx)
            
            # Add all buildings from right part (they can't be blocked by left part)
            result.extend(right_result)
            
            return result
        
        return solve(0, len(heights) - 1)


def test_buildings_with_ocean_view():
    """Test buildings with ocean view algorithms"""
    solver = BuildingsWithOceanView()
    
    test_cases = [
        ([4,2,3,1], [0,2,3], "Example 1"),
        ([4,3,2,1], [0,1,2,3], "Example 2"),
        ([1,3,2,4], [3], "Example 3"),
        ([1], [0], "Single building"),
        ([1,2,3,4,5], [4], "Increasing heights"),
        ([5,4,3,2,1], [0,1,2,3,4], "Decreasing heights"),
        ([2,2,2,2], [3], "All same heights"),
        ([1,2,1,3,1], [1,3,4], "Mixed heights"),
        ([5,3,8,2,6,1], [0,2,4,5], "Complex case"),
        ([10,5,15,3,12,1], [0,2,4,5], "Another complex"),
        ([1,2,3,2,1], [2,3,4], "Peak in middle"),
        ([3,1,4,1,5], [0,2,4], "Irregular pattern"),
    ]
    
    algorithms = [
        ("Right to Left", solver.findBuildings_right_to_left),
        ("Stack", solver.findBuildings_stack),
        ("Brute Force", solver.findBuildings_brute_force),
        ("Suffix Maximum", solver.findBuildings_suffix_maximum),
        ("Divide Conquer", solver.findBuildings_divide_conquer),
    ]
    
    print("=== Testing Buildings With Ocean View ===")
    
    for heights, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Heights: {heights}")
        print(f"Expected: {expected}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(heights[:])  # Copy to avoid modification
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:20} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:40]}")


def demonstrate_right_to_left_approach():
    """Demonstrate right to left approach step by step"""
    print("\n=== Right to Left Approach Step-by-Step Demo ===")
    
    heights = [4, 2, 3, 1]
    
    print(f"Heights: {heights}")
    print("Strategy: Traverse from right to left, track maximum height")
    print("A building has ocean view if it's taller than all buildings to its right")
    
    result = []
    max_height = 0
    
    print(f"\nStep-by-step processing (right to left):")
    
    for i in range(len(heights) - 1, -1, -1):
        print(f"\nStep {len(heights) - i}: Building at index {i}, height {heights[i]}")
        print(f"  Current max height from right: {max_height}")
        
        if heights[i] > max_height:
            result.append(i)
            max_height = heights[i]
            print(f"  Building {i} has ocean view (height {heights[i]} > {max_height - heights[i]})")
            print(f"  Update max height to {max_height}")
        else:
            print(f"  Building {i} blocked (height {heights[i]} <= {max_height})")
        
        print(f"  Current result: {result}")
    
    # Reverse to get increasing order
    result.reverse()
    print(f"\nFinal result (sorted): {result}")


def visualize_ocean_view():
    """Visualize buildings and ocean view"""
    print("\n=== Ocean View Visualization ===")
    
    heights = [4, 2, 3, 1]
    solver = BuildingsWithOceanView()
    ocean_view = solver.findBuildings_right_to_left(heights)
    
    print(f"Heights: {heights}")
    print(f"Buildings with ocean view: {ocean_view}")
    
    # Create visual representation
    max_height = max(heights)
    
    print(f"\nVisual representation (ocean is to the right):")
    
    for level in range(max_height, 0, -1):
        line = ""
        for i, height in enumerate(heights):
            if height >= level:
                if i in ocean_view:
                    line += "█ "  # Building with ocean view
                else:
                    line += "▓ "  # Building without ocean view
            else:
                line += "  "  # Empty space
        line += "~ OCEAN"
        print(f"Level {level}: {line}")
    
    # Show indices
    indices = "Indices: " + " ".join(f"{i}" for i in range(len(heights))) + "   "
    print(indices)
    
    # Show which have ocean view
    view_line = "Ocean:   "
    for i in range(len(heights)):
        if i in ocean_view:
            view_line += "✓ "
        else:
            view_line += "✗ "
    print(view_line)


def demonstrate_stack_approach():
    """Demonstrate stack approach"""
    print("\n=== Stack Approach Demo ===")
    
    heights = [4, 2, 3, 1]
    
    print(f"Heights: {heights}")
    print("Strategy: Use monotonic decreasing stack")
    print("Remove buildings that are blocked by current building")
    
    stack = []
    
    print(f"\nStep-by-step processing:")
    
    for i, height in enumerate(heights):
        print(f"\nStep {i+1}: Processing building {i} with height {height}")
        print(f"  Current stack: {stack}")
        
        # Remove buildings that are blocked
        removed = []
        while stack and heights[stack[-1]] <= height:
            removed.append(stack.pop())
        
        if removed:
            print(f"  Removed blocked buildings: {removed}")
        else:
            print(f"  No buildings to remove")
        
        stack.append(i)
        print(f"  Added building {i} to stack")
        print(f"  Stack after: {stack}")
    
    print(f"\nFinal result: {stack}")


def demonstrate_competitive_programming_patterns():
    """Demonstrate competitive programming patterns"""
    print("\n=== Competitive Programming Patterns ===")
    
    solver = BuildingsWithOceanView()
    
    # Pattern 1: Right to left traversal
    print("1. Right to Left Traversal:")
    print("   Process elements from right to left")
    print("   Maintain running maximum/minimum")
    
    example1 = [4, 2, 3, 1]
    result1 = solver.findBuildings_right_to_left(example1)
    print(f"   {example1} -> {result1}")
    
    # Pattern 2: Monotonic stack
    print(f"\n2. Monotonic Stack:")
    print("   Maintain stack in monotonic order")
    print("   Remove elements that don't satisfy condition")
    
    example2 = [5, 4, 3, 2, 1]
    result2 = solver.findBuildings_stack(example2)
    print(f"   {example2} -> {result2}")
    
    # Pattern 3: Suffix processing
    print(f"\n3. Suffix Processing:")
    print("   Precompute suffix information")
    print("   Use suffix data for O(1) queries")
    
    # Pattern 4: Visibility problems
    print(f"\n4. Visibility Problems:")
    print("   Common pattern: can element see beyond others?")
    print("   Applications: line of sight, next greater element")


def analyze_time_complexity():
    """Analyze time complexity of different approaches"""
    print("\n=== Time Complexity Analysis ===")
    
    approaches = [
        ("Right to Left", "O(n)", "O(1)", "Single pass with constant space"),
        ("Stack", "O(n)", "O(n)", "Each element pushed/popped once"),
        ("Brute Force", "O(n²)", "O(1)", "Nested loops for each building"),
        ("Suffix Maximum", "O(n)", "O(n)", "Two passes with extra array"),
        ("Divide Conquer", "O(n log n)", "O(log n)", "Divide and conquer overhead"),
    ]
    
    print(f"{'Approach':<20} | {'Time':<10} | {'Space':<8} | {'Notes'}")
    print("-" * 65)
    
    for approach, time_comp, space_comp, notes in approaches:
        print(f"{approach:<20} | {time_comp:<10} | {space_comp:<8} | {notes}")
    
    print(f"\nRight to Left approach is optimal for competitive programming")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    solver = BuildingsWithOceanView()
    
    edge_cases = [
        ([], [], "Empty array"),
        ([1], [0], "Single building"),
        ([1, 1], [1], "Two same heights"),
        ([1, 2], [1], "Increasing pair"),
        ([2, 1], [0, 1], "Decreasing pair"),
        ([1, 2, 3], [2], "Strictly increasing"),
        ([3, 2, 1], [0, 1, 2], "Strictly decreasing"),
        ([2, 2, 2], [2], "All same heights"),
        ([1, 3, 2, 4], [3], "Peak at end"),
        ([4, 2, 3, 1], [0, 2, 3], "Mixed pattern"),
        ([5, 5, 5, 5], [3], "All equal"),
        ([1, 1000000, 1], [1, 2], "Large height difference"),
    ]
    
    for heights, expected, description in edge_cases:
        try:
            result = solver.findBuildings_right_to_left(heights[:])
            status = "✓" if result == expected else "✗"
            print(f"{description:25} | {status} | {heights} -> {result}")
        except Exception as e:
            print(f"{description:25} | ERROR: {str(e)[:30]}")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    solver = BuildingsWithOceanView()
    
    # Application 1: City planning
    print("1. City Planning - Ocean View Properties:")
    print("   Determine which buildings have unobstructed ocean views")
    print("   Important for property valuation and zoning")
    
    building_heights = [15, 8, 12, 6, 20, 3]  # Heights in floors
    ocean_view_buildings = solver.findBuildings_right_to_left(building_heights)
    
    print(f"   Building heights (floors): {building_heights}")
    print(f"   Buildings with ocean view: {ocean_view_buildings}")
    
    for i, height in enumerate(building_heights):
        view_status = "Ocean View" if i in ocean_view_buildings else "Blocked"
        print(f"     Building {i}: {height} floors - {view_status}")
    
    # Application 2: Antenna placement
    print(f"\n2. Antenna/Tower Placement:")
    print("   Determine which antennas have clear line of sight")
    print("   Important for wireless communication networks")
    
    antenna_heights = [50, 30, 45, 25, 60, 20]  # Heights in meters
    clear_antennas = solver.findBuildings_right_to_left(antenna_heights)
    
    print(f"   Antenna heights (meters): {antenna_heights}")
    print(f"   Antennas with clear eastward signal: {clear_antennas}")
    
    # Application 3: Solar panel efficiency
    print(f"\n3. Solar Panel Efficiency:")
    print("   Determine which panels won't be shaded by taller structures")
    print("   Important for solar energy optimization")
    
    panel_heights = [3, 2, 4, 1, 5, 2]  # Heights in meters
    unshaded_panels = solver.findBuildings_right_to_left(panel_heights)
    
    print(f"   Panel heights (meters): {panel_heights}")
    print(f"   Unshaded panels (afternoon sun): {unshaded_panels}")
    
    # Application 4: Surveillance systems
    print(f"\n4. Surveillance Camera Coverage:")
    print("   Determine which cameras have unobstructed view of target area")
    print("   Important for security system design")
    
    camera_heights = [8, 5, 7, 4, 10, 3]  # Heights in meters
    clear_cameras = solver.findBuildings_right_to_left(camera_heights)
    
    print(f"   Camera heights (meters): {camera_heights}")
    print(f"   Cameras with clear view: {clear_cameras}")


def benchmark_approaches():
    """Benchmark different approaches"""
    import time
    import random
    
    approaches = [
        ("Right to Left", BuildingsWithOceanView().findBuildings_right_to_left),
        ("Stack", BuildingsWithOceanView().findBuildings_stack),
        ("Suffix Maximum", BuildingsWithOceanView().findBuildings_suffix_maximum),
    ]
    
    # Generate test data
    test_sizes = [1000, 5000, 10000]
    
    print(f"\n=== Performance Benchmark ===")
    
    for size in test_sizes:
        print(f"\nArray size: {size}")
        
        # Generate random heights
        heights = [random.randint(1, 1000) for _ in range(size)]
        
        for name, func in approaches:
            try:
                start_time = time.time()
                
                # Run multiple times for better measurement
                for _ in range(10):
                    func(heights[:])
                
                end_time = time.time()
                avg_time = (end_time - start_time) / 10
                
                print(f"  {name:20} | Avg time: {avg_time:.6f}s")
                
            except Exception as e:
                print(f"  {name:20} | ERROR: {str(e)[:30]}")


if __name__ == "__main__":
    test_buildings_with_ocean_view()
    demonstrate_right_to_left_approach()
    visualize_ocean_view()
    demonstrate_stack_approach()
    demonstrate_competitive_programming_patterns()
    analyze_time_complexity()
    test_edge_cases()
    demonstrate_real_world_applications()
    benchmark_approaches()

"""
Buildings With an Ocean View demonstrates competitive programming
patterns with right-to-left traversal, monotonic stack, and visibility
problems for efficient line-of-sight calculations.
"""
