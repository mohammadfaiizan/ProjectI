"""
LeetCode 42: Trapping Rain Water
Difficulty: Hard
Category: Fibonacci & Linear DP / Array

PROBLEM DESCRIPTION:
===================
Given n non-negative integers representing an elevation map where the width of each bar is 1, 
compute how much water it can trap after raining.

Example 1:
Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6
Explanation: The above elevation map (black section) is represented by array [0,1,0,2,1,0,1,3,2,1,2,1]. 
In this case, 6 units of rain water (blue section) are being trapped.

Example 2:
Input: height = [4,2,0,3,2,5]
Output: 9

Constraints:
- n == height.length
- 1 <= n <= 2 * 10^4
- 0 <= height[i] <= 3 * 10^4
"""

def trap_water_bruteforce(height):
    """
    BRUTE FORCE APPROACH:
    ====================
    For each position, find water level = min(max_left, max_right).
    
    Time Complexity: O(n^2) - for each position, scan left and right
    Space Complexity: O(1) - constant space
    """
    if not height:
        return 0
    
    n = len(height)
    total_water = 0
    
    for i in range(n):
        # Find maximum height to the left
        max_left = 0
        for j in range(i):
            max_left = max(max_left, height[j])
        
        # Find maximum height to the right
        max_right = 0
        for j in range(i + 1, n):
            max_right = max(max_right, height[j])
        
        # Water level at position i
        water_level = min(max_left, max_right)
        
        # Water trapped = water_level - current_height (if positive)
        if water_level > height[i]:
            total_water += water_level - height[i]
    
    return total_water


def trap_water_dp_precompute(height):
    """
    DYNAMIC PROGRAMMING - PRECOMPUTE MAX ARRAYS:
    ===========================================
    Precompute max_left and max_right arrays.
    
    Time Complexity: O(n) - three passes through array
    Space Complexity: O(n) - two additional arrays
    """
    if not height or len(height) < 3:
        return 0
    
    n = len(height)
    
    # Precompute maximum heights to the left
    max_left = [0] * n
    max_left[0] = height[0]
    for i in range(1, n):
        max_left[i] = max(max_left[i - 1], height[i])
    
    # Precompute maximum heights to the right
    max_right = [0] * n
    max_right[n - 1] = height[n - 1]
    for i in range(n - 2, -1, -1):
        max_right[i] = max(max_right[i + 1], height[i])
    
    # Calculate trapped water
    total_water = 0
    for i in range(n):
        water_level = min(max_left[i], max_right[i])
        if water_level > height[i]:
            total_water += water_level - height[i]
    
    return total_water


def trap_water_two_pointers(height):
    """
    TWO POINTERS APPROACH:
    =====================
    Use two pointers moving towards each other.
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(1) - constant space
    """
    if not height or len(height) < 3:
        return 0
    
    left = 0
    right = len(height) - 1
    left_max = 0
    right_max = 0
    total_water = 0
    
    while left < right:
        if height[left] < height[right]:
            if height[left] >= left_max:
                left_max = height[left]
            else:
                total_water += left_max - height[left]
            left += 1
        else:
            if height[right] >= right_max:
                right_max = height[right]
            else:
                total_water += right_max - height[right]
            right -= 1
    
    return total_water


def trap_water_stack(height):
    """
    STACK APPROACH:
    ==============
    Use stack to track decreasing heights and calculate water when increasing.
    
    Time Complexity: O(n) - each element pushed/popped once
    Space Complexity: O(n) - stack storage
    """
    if not height:
        return 0
    
    stack = []
    total_water = 0
    
    for i, h in enumerate(height):
        # While current height is greater than stack top
        while stack and height[stack[-1]] < h:
            # Pop the top (this will be filled with water)
            top = stack.pop()
            
            if not stack:
                break
            
            # Calculate distance and bounded height
            distance = i - stack[-1] - 1
            bounded_height = min(height[i], height[stack[-1]]) - height[top]
            
            total_water += distance * bounded_height
        
        stack.append(i)
    
    return total_water


def trap_water_divide_conquer(height):
    """
    DIVIDE AND CONQUER APPROACH:
    ===========================
    Find maximum element and recursively solve left and right parts.
    
    Time Complexity: O(n log n) - average case, O(n^2) worst case
    Space Complexity: O(log n) - recursion stack
    """
    if not height or len(height) < 3:
        return 0
    
    def trap_recursive(left, right, left_max, right_max):
        if left >= right:
            return 0
        
        # Find maximum element in current range
        max_idx = left
        for i in range(left + 1, right + 1):
            if height[i] > height[max_idx]:
                max_idx = i
        
        water = 0
        
        # Add water in left part
        current_max = left_max
        for i in range(left, max_idx):
            current_max = max(current_max, height[i])
            water_level = min(current_max, height[max_idx])
            if water_level > height[i]:
                water += water_level - height[i]
        
        # Add water in right part
        current_max = right_max
        for i in range(right, max_idx, -1):
            current_max = max(current_max, height[i])
            water_level = min(height[max_idx], current_max)
            if water_level > height[i]:
                water += water_level - height[i]
        
        # Recursively solve left and right
        water += trap_recursive(left, max_idx - 1, left_max, height[max_idx])
        water += trap_recursive(max_idx + 1, right, height[max_idx], right_max)
        
        return water
    
    return trap_recursive(0, len(height) - 1, 0, 0)


def trap_water_segment_tree(height):
    """
    SEGMENT TREE APPROACH:
    =====================
    Use segment tree for range maximum queries (overkill for this problem).
    
    Time Complexity: O(n log n) - build tree + queries
    Space Complexity: O(n) - segment tree storage
    """
    if not height or len(height) < 3:
        return 0
    
    n = len(height)
    
    # Build segment tree for range maximum query
    class SegmentTree:
        def __init__(self, arr):
            self.n = len(arr)
            self.tree = [0] * (4 * self.n)
            self.build(arr, 0, 0, self.n - 1)
        
        def build(self, arr, node, start, end):
            if start == end:
                self.tree[node] = arr[start]
            else:
                mid = (start + end) // 2
                self.build(arr, 2 * node + 1, start, mid)
                self.build(arr, 2 * node + 2, mid + 1, end)
                self.tree[node] = max(self.tree[2 * node + 1], self.tree[2 * node + 2])
        
        def query(self, node, start, end, l, r):
            if r < start or end < l:
                return 0
            if l <= start and end <= r:
                return self.tree[node]
            
            mid = (start + end) // 2
            left_max = self.query(2 * node + 1, start, mid, l, r)
            right_max = self.query(2 * node + 2, mid + 1, end, l, r)
            return max(left_max, right_max)
        
        def range_max(self, l, r):
            return self.query(0, 0, self.n - 1, l, r)
    
    seg_tree = SegmentTree(height)
    total_water = 0
    
    for i in range(n):
        # Get max to left and right using segment tree
        max_left = seg_tree.range_max(0, i) if i > 0 else 0
        max_right = seg_tree.range_max(i, n - 1) if i < n - 1 else 0
        
        water_level = min(max_left, max_right)
        if water_level > height[i]:
            total_water += water_level - height[i]
    
    return total_water


def trap_water_with_visualization(height):
    """
    TRAP WATER WITH VISUALIZATION:
    ==============================
    Calculate trapped water and show visual representation.
    
    Time Complexity: O(n) - linear solution + visualization
    Space Complexity: O(n) - for visualization
    """
    if not height:
        return 0, []
    
    n = len(height)
    
    # Use DP approach to get water levels
    max_left = [0] * n
    max_right = [0] * n
    
    max_left[0] = height[0]
    for i in range(1, n):
        max_left[i] = max(max_left[i - 1], height[i])
    
    max_right[n - 1] = height[n - 1]
    for i in range(n - 2, -1, -1):
        max_right[i] = max(max_right[i + 1], height[i])
    
    # Calculate water and create visualization
    water_levels = []
    total_water = 0
    
    for i in range(n):
        water_level = min(max_left[i], max_right[i])
        trapped = max(0, water_level - height[i])
        water_levels.append(water_level)
        total_water += trapped
    
    # Create visual representation
    if n <= 20:  # Only for small arrays
        max_height = max(max(height), max(water_levels))
        
        print("Visual representation (H=height, W=water, .=air):")
        for level in range(max_height, 0, -1):
            line = ""
            for i in range(n):
                if height[i] >= level:
                    line += "H"
                elif water_levels[i] >= level:
                    line += "W"
                else:
                    line += "."
            print(f"{level:2d} |{line}|")
        
        print("   " + "-" * (n + 2))
        print("    " + "".join(str(i % 10) for i in range(n)))
        print(f"Heights: {height}")
        print(f"Water:   {[water_levels[i] - height[i] for i in range(n)]}")
    
    return total_water, water_levels


# Test cases
def test_trap_water():
    """Test all implementations with various inputs"""
    test_cases = [
        ([0,1,0,2,1,0,1,3,2,1,2,1], 6),
        ([4,2,0,3,2,5], 9),
        ([3,0,2,0,4], 7),
        ([0,1,0], 0),
        ([2,0,2], 2),
        ([3,2,0,4], 7),
        ([1,2,3,4,5], 0),
        ([5,4,3,2,1], 0),
        ([0,0,0], 0),
        ([1], 0),
        ([], 0)
    ]
    
    print("Testing Trapping Rain Water Solutions:")
    print("=" * 70)
    
    for i, (height, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: height = {height}")
        print(f"Expected: {expected}")
        
        # Test approaches (skip expensive ones for large inputs)
        if len(height) <= 15:
            brute = trap_water_bruteforce(height.copy())
            print(f"Brute Force:      {brute:>3} {'✓' if brute == expected else '✗'}")
        
        dp_precompute = trap_water_dp_precompute(height.copy())
        two_ptr = trap_water_two_pointers(height.copy())
        stack = trap_water_stack(height.copy())
        
        print(f"DP Precompute:    {dp_precompute:>3} {'✓' if dp_precompute == expected else '✗'}")
        print(f"Two Pointers:     {two_ptr:>3} {'✓' if two_ptr == expected else '✗'}")
        print(f"Stack:            {stack:>3} {'✓' if stack == expected else '✗'}")
        
        if len(height) <= 10:
            divide = trap_water_divide_conquer(height.copy())
            print(f"Divide Conquer:   {divide:>3} {'✓' if divide == expected else '✗'}")
    
    # Show visualization for interesting cases
    print(f"\nVisualization Examples:")
    
    test_visual = [0,1,0,2,1,0,1,3,2,1,2,1]
    print(f"\nExample 1: {test_visual}")
    water, levels = trap_water_with_visualization(test_visual)
    print(f"Total water trapped: {water}")
    
    test_visual = [4,2,0,3,2,5]
    print(f"\nExample 2: {test_visual}")
    water, levels = trap_water_with_visualization(test_visual)
    print(f"Total water trapped: {water}")
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Brute Force:      Time: O(n^2),      Space: O(1)")
    print("DP Precompute:    Time: O(n),        Space: O(n)")
    print("Two Pointers:     Time: O(n),        Space: O(1)")
    print("Stack:            Time: O(n),        Space: O(n)")
    print("Divide Conquer:   Time: O(n log n),  Space: O(log n)")
    print("Segment Tree:     Time: O(n log n),  Space: O(n)")


if __name__ == "__main__":
    test_trap_water()


"""
PATTERN RECOGNITION:
==================
This is a classic array DP problem with multiple optimal solutions:
- Water trapped = min(max_left, max_right) - current_height
- Can be solved with DP precomputation or two pointers
- Stack approach provides different perspective
- Multiple techniques demonstrate algorithm versatility

KEY INSIGHT - WATER LEVEL:
==========================
Water level at position i = min(max_height_left, max_height_right)
Water trapped at i = max(0, water_level - height[i])

MULTIPLE OPTIMAL APPROACHES:
===========================

1. **DP Precomputation**: O(n) time, O(n) space
   - Precompute max_left[] and max_right[] arrays
   - Calculate water for each position

2. **Two Pointers**: O(n) time, O(1) space  
   - Move pointers from both ends
   - Process side with smaller maximum first

3. **Stack**: O(n) time, O(n) space
   - Use stack to track decreasing heights
   - Calculate water when finding taller bar

ALGORITHM COMPARISON:
====================
- **Brute Force**: O(n²) - check left/right max for each position
- **DP**: O(n) time, O(n) space - precompute maximums
- **Two Pointers**: O(n) time, O(1) space - optimal solution
- **Stack**: O(n) time, O(n) space - different perspective
- **Divide & Conquer**: O(n log n) - educational approach

STATE DEFINITION (DP):
=====================
max_left[i] = maximum height from 0 to i
max_right[i] = maximum height from i to n-1
water[i] = min(max_left[i], max_right[i]) - height[i]

TWO POINTERS INSIGHT:
====================
Key insight: If height[left] < height[right], then max_right >= height[right] > height[left]
So water level at left is determined by max_left only.

VARIANTS TO PRACTICE:
====================
- Container With Most Water (11) - similar two pointers
- Largest Rectangle in Histogram (84) - stack approach
- Rain Water Trapping II (407) - 2D version
- Product of Array Except Self (238) - similar precomputation

INTERVIEW TIPS:
==============
1. Start with brute force approach
2. Optimize with DP precomputation
3. Show two pointers as optimal solution
4. Explain stack approach for variety
5. Draw visual examples for clarity
6. Handle edge cases (empty, single element)
7. Discuss space/time trade-offs
8. Mention real-world applications
"""
