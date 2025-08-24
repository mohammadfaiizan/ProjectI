"""
2104. Sum of Subarray Ranges - Multiple Approaches
Difficulty: Medium

You are given an integer array nums. The range of a subarray of nums is the difference between the largest and smallest element in the subarray.

Return the sum of all subarray ranges of nums.

A subarray is a contiguous non-empty sequence of elements within an array.
"""

from typing import List

class SumOfSubarrayRanges:
    """Multiple approaches to calculate sum of subarray ranges"""
    
    def subArrayRanges_brute_force(self, nums: List[int]) -> int:
        """
        Approach 1: Brute Force
        
        Check all subarrays and calculate their ranges.
        
        Time: O(n³), Space: O(1)
        """
        n = len(nums)
        total_sum = 0
        
        for i in range(n):
            for j in range(i, n):
                # Find min and max in subarray nums[i:j+1]
                min_val = max_val = nums[i]
                
                for k in range(i, j + 1):
                    min_val = min(min_val, nums[k])
                    max_val = max(max_val, nums[k])
                
                total_sum += max_val - min_val
        
        return total_sum
    
    def subArrayRanges_optimized_brute_force(self, nums: List[int]) -> int:
        """
        Approach 2: Optimized Brute Force
        
        Maintain min/max while extending subarray.
        
        Time: O(n²), Space: O(1)
        """
        n = len(nums)
        total_sum = 0
        
        for i in range(n):
            min_val = max_val = nums[i]
            
            for j in range(i, n):
                min_val = min(min_val, nums[j])
                max_val = max(max_val, nums[j])
                
                total_sum += max_val - min_val
        
        return total_sum
    
    def subArrayRanges_monotonic_stack(self, nums: List[int]) -> int:
        """
        Approach 3: Monotonic Stack (Optimal)
        
        Use monotonic stack to calculate contribution of each element.
        
        Time: O(n), Space: O(n)
        """
        def sum_of_maximums(arr: List[int]) -> int:
            """Calculate sum of maximum elements in all subarrays"""
            n = len(arr)
            stack = []
            result = 0
            
            for i in range(n + 1):
                while stack and (i == n or arr[stack[-1]] <= arr[i]):
                    mid = stack.pop()
                    left = stack[-1] if stack else -1
                    right = i
                    
                    # Number of subarrays where arr[mid] is maximum
                    count = (mid - left) * (right - mid)
                    result += arr[mid] * count
                
                stack.append(i)
            
            return result
        
        def sum_of_minimums(arr: List[int]) -> int:
            """Calculate sum of minimum elements in all subarrays"""
            n = len(arr)
            stack = []
            result = 0
            
            for i in range(n + 1):
                while stack and (i == n or arr[stack[-1]] >= arr[i]):
                    mid = stack.pop()
                    left = stack[-1] if stack else -1
                    right = i
                    
                    # Number of subarrays where arr[mid] is minimum
                    count = (mid - left) * (right - mid)
                    result += arr[mid] * count
                
                stack.append(i)
            
            return result
        
        return sum_of_maximums(nums) - sum_of_minimums(nums)
    
    def subArrayRanges_contribution_technique(self, nums: List[int]) -> int:
        """
        Approach 4: Contribution Technique
        
        Calculate contribution of each element as max and min.
        
        Time: O(n), Space: O(n)
        """
        n = len(nums)
        
        # For each element, find how many subarrays it's the maximum/minimum
        def calculate_contributions():
            # Left smaller/greater elements
            left_smaller = [-1] * n
            left_greater = [-1] * n
            right_smaller = [n] * n
            right_greater = [n] * n
            
            # Monotonic stack for left smaller
            stack = []
            for i in range(n):
                while stack and nums[stack[-1]] >= nums[i]:
                    stack.pop()
                if stack:
                    left_smaller[i] = stack[-1]
                stack.append(i)
            
            # Monotonic stack for left greater
            stack = []
            for i in range(n):
                while stack and nums[stack[-1]] <= nums[i]:
                    stack.pop()
                if stack:
                    left_greater[i] = stack[-1]
                stack.append(i)
            
            # Monotonic stack for right smaller
            stack = []
            for i in range(n - 1, -1, -1):
                while stack and nums[stack[-1]] > nums[i]:
                    stack.pop()
                if stack:
                    right_smaller[i] = stack[-1]
                stack.append(i)
            
            # Monotonic stack for right greater
            stack = []
            for i in range(n - 1, -1, -1):
                while stack and nums[stack[-1]] < nums[i]:
                    stack.pop()
                if stack:
                    right_greater[i] = stack[-1]
                stack.append(i)
            
            return left_smaller, left_greater, right_smaller, right_greater
        
        left_smaller, left_greater, right_smaller, right_greater = calculate_contributions()
        
        total_sum = 0
        for i in range(n):
            # Contribution as maximum
            max_contribution = (i - left_greater[i]) * (right_greater[i] - i) * nums[i]
            
            # Contribution as minimum
            min_contribution = (i - left_smaller[i]) * (right_smaller[i] - i) * nums[i]
            
            total_sum += max_contribution - min_contribution
        
        return total_sum
    
    def subArrayRanges_divide_conquer(self, nums: List[int]) -> int:
        """
        Approach 5: Divide and Conquer
        
        Use divide and conquer to calculate ranges.
        
        Time: O(n log n) average, O(n²) worst case, Space: O(log n)
        """
        def solve(left: int, right: int) -> int:
            if left == right:
                return 0
            
            if right - left == 1:
                return abs(nums[right] - nums[left])
            
            mid = (left + right) // 2
            
            # Recursively solve left and right parts
            result = solve(left, mid) + solve(mid + 1, right)
            
            # Add contribution of subarrays crossing the middle
            for i in range(left, mid + 1):
                min_val = max_val = nums[i]
                
                for j in range(i, right + 1):
                    if j <= mid:
                        min_val = min(min_val, nums[j])
                        max_val = max(max_val, nums[j])
                    else:
                        min_val = min(min_val, nums[j])
                        max_val = max(max_val, nums[j])
                        result += max_val - min_val
            
            return result
        
        return solve(0, len(nums) - 1)


def test_sum_of_subarray_ranges():
    """Test sum of subarray ranges algorithms"""
    solver = SumOfSubarrayRanges()
    
    test_cases = [
        ([1,2,3], 4, "Example 1"),
        ([1,3,3], 4, "Example 2"),
        ([4,-2,-3,4,1], 59, "Example 3"),
        ([1], 0, "Single element"),
        ([1,1,1], 0, "All same"),
        ([1,2], 1, "Two elements"),
        ([3,1,2], 4, "Three elements"),
        ([1,4,2,3], 13, "Four elements"),
        ([5,1,3,2,4], 26, "Five elements"),
        ([2,1,3], 4, "Small case"),
        ([1,2,3,4,5], 20, "Increasing"),
        ([5,4,3,2,1], 20, "Decreasing"),
    ]
    
    algorithms = [
        ("Brute Force", solver.subArrayRanges_brute_force),
        ("Optimized BF", solver.subArrayRanges_optimized_brute_force),
        ("Monotonic Stack", solver.subArrayRanges_monotonic_stack),
        ("Contribution", solver.subArrayRanges_contribution_technique),
        ("Divide Conquer", solver.subArrayRanges_divide_conquer),
    ]
    
    print("=== Testing Sum of Subarray Ranges ===")
    
    for nums, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Input: {nums}")
        print(f"Expected: {expected}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(nums[:])  # Copy to avoid modification
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:20} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:40]}")


def demonstrate_monotonic_stack_approach():
    """Demonstrate monotonic stack approach step by step"""
    print("\n=== Monotonic Stack Approach Step-by-Step Demo ===")
    
    nums = [1, 3, 2]
    
    print(f"Input: {nums}")
    print("Strategy: Calculate sum of maximums - sum of minimums")
    print("Use monotonic stack to find contribution of each element")
    
    def demonstrate_sum_of_maximums(arr):
        print(f"\nCalculating sum of maximums:")
        n = len(arr)
        stack = []
        result = 0
        
        for i in range(n + 1):
            print(f"\nStep {i+1}: i={i}")
            print(f"  Stack: {stack}")
            
            while stack and (i == n or arr[stack[-1]] <= arr[i]):
                mid = stack.pop()
                left = stack[-1] if stack else -1
                right = i
                
                print(f"    Processing element at index {mid} (value={arr[mid]})")
                print(f"    Left boundary: {left}, Right boundary: {right}")
                
                count = (mid - left) * (right - mid)
                contribution = arr[mid] * count
                
                print(f"    Subarrays where {arr[mid]} is maximum: {count}")
                print(f"    Contribution: {arr[mid]} * {count} = {contribution}")
                
                result += contribution
                print(f"    Running sum: {result}")
            
            if i < n:
                stack.append(i)
                print(f"  Added index {i} to stack: {stack}")
        
        print(f"\nSum of maximums: {result}")
        return result
    
    def demonstrate_sum_of_minimums(arr):
        print(f"\nCalculating sum of minimums:")
        n = len(arr)
        stack = []
        result = 0
        
        for i in range(n + 1):
            print(f"\nStep {i+1}: i={i}")
            print(f"  Stack: {stack}")
            
            while stack and (i == n or arr[stack[-1]] >= arr[i]):
                mid = stack.pop()
                left = stack[-1] if stack else -1
                right = i
                
                print(f"    Processing element at index {mid} (value={arr[mid]})")
                print(f"    Left boundary: {left}, Right boundary: {right}")
                
                count = (mid - left) * (right - mid)
                contribution = arr[mid] * count
                
                print(f"    Subarrays where {arr[mid]} is minimum: {count}")
                print(f"    Contribution: {arr[mid]} * {count} = {contribution}")
                
                result += contribution
                print(f"    Running sum: {result}")
            
            if i < n:
                stack.append(i)
                print(f"  Added index {i} to stack: {stack}")
        
        print(f"\nSum of minimums: {result}")
        return result
    
    sum_max = demonstrate_sum_of_maximums(nums)
    sum_min = demonstrate_sum_of_minimums(nums)
    
    print(f"\nFinal calculation:")
    print(f"Sum of maximums: {sum_max}")
    print(f"Sum of minimums: {sum_min}")
    print(f"Sum of ranges: {sum_max} - {sum_min} = {sum_max - sum_min}")


def visualize_subarray_ranges():
    """Visualize all subarray ranges"""
    print("\n=== Subarray Ranges Visualization ===")
    
    nums = [1, 3, 2]
    n = len(nums)
    
    print(f"Input: {nums}")
    print("All subarrays and their ranges:")
    
    total_sum = 0
    subarray_count = 0
    
    for i in range(n):
        for j in range(i, n):
            subarray = nums[i:j+1]
            min_val = min(subarray)
            max_val = max(subarray)
            range_val = max_val - min_val
            
            subarray_count += 1
            total_sum += range_val
            
            print(f"  Subarray {subarray_count}: {subarray}")
            print(f"    Min: {min_val}, Max: {max_val}, Range: {range_val}")
    
    print(f"\nTotal sum of ranges: {total_sum}")
    print(f"Number of subarrays: {subarray_count}")


def demonstrate_competitive_programming_patterns():
    """Demonstrate competitive programming patterns"""
    print("\n=== Competitive Programming Patterns ===")
    
    solver = SumOfSubarrayRanges()
    
    # Pattern 1: Monotonic stack for contribution
    print("1. Monotonic Stack for Element Contribution:")
    print("   Calculate how many subarrays each element affects as min/max")
    print("   Use monotonic stack to find boundaries efficiently")
    
    example1 = [1, 3, 2]
    result1 = solver.subArrayRanges_monotonic_stack(example1)
    print(f"   {example1} -> {result1}")
    
    # Pattern 2: Contribution technique
    print(f"\n2. Contribution Technique:")
    print("   Sum of (max contributions) - Sum of (min contributions)")
    print("   Each element contributes to multiple subarrays")
    
    # Pattern 3: Problem decomposition
    print(f"\n3. Problem Decomposition:")
    print("   Range = Max - Min")
    print("   Sum of ranges = Sum of maxes - Sum of mins")
    
    # Pattern 4: Boundary calculation
    print(f"\n4. Boundary Calculation:")
    print("   For each element, find left/right boundaries")
    print("   Count subarrays where element is min/max")


def analyze_time_complexity():
    """Analyze time complexity of different approaches"""
    print("\n=== Time Complexity Analysis ===")
    
    approaches = [
        ("Brute Force", "O(n³)", "O(1)", "Triple nested loops"),
        ("Optimized BF", "O(n²)", "O(1)", "Maintain min/max while extending"),
        ("Monotonic Stack", "O(n)", "O(n)", "Each element processed once"),
        ("Contribution", "O(n)", "O(n)", "Four monotonic stack passes"),
        ("Divide Conquer", "O(n log n)", "O(log n)", "Divide and conquer approach"),
    ]
    
    print(f"{'Approach':<20} | {'Time':<12} | {'Space':<8} | {'Notes'}")
    print("-" * 70)
    
    for approach, time_comp, space_comp, notes in approaches:
        print(f"{approach:<20} | {time_comp:<12} | {space_comp:<8} | {notes}")
    
    print(f"\nMonotonic Stack approach is optimal for competitive programming")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    solver = SumOfSubarrayRanges()
    
    edge_cases = [
        ([1], 0, "Single element"),
        ([1, 1], 0, "Two same elements"),
        ([1, 1, 1], 0, "All same elements"),
        ([1, 2], 1, "Two different elements"),
        ([2, 1], 1, "Two elements reverse"),
        ([1, 2, 3], 4, "Increasing sequence"),
        ([3, 2, 1], 4, "Decreasing sequence"),
        ([1, 3, 1], 4, "Peak in middle"),
        ([3, 1, 3], 8, "Valley in middle"),
        ([1, 2, 1, 2], 6, "Alternating pattern"),
        ([5, 5, 5, 5], 0, "All identical"),
        ([1, 100, 1], 198, "Large difference"),
    ]
    
    for nums, expected, description in edge_cases:
        try:
            result = solver.subArrayRanges_monotonic_stack(nums[:])
            status = "✓" if result == expected else "✗"
            print(f"{description:25} | {status} | {nums} -> {result}")
        except Exception as e:
            print(f"{description:25} | ERROR: {str(e)[:30]}")


def demonstrate_contribution_calculation():
    """Demonstrate contribution calculation"""
    print("\n=== Contribution Calculation Demo ===")
    
    nums = [1, 3, 2]
    n = len(nums)
    
    print(f"Input: {nums}")
    print("Calculating contribution of each element:")
    
    # Manual calculation for demonstration
    contributions = []
    
    for i in range(n):
        print(f"\nElement {nums[i]} at index {i}:")
        
        # Count subarrays where this element is maximum
        max_count = 0
        max_subarrays = []
        
        # Count subarrays where this element is minimum
        min_count = 0
        min_subarrays = []
        
        for start in range(n):
            for end in range(start, n):
                subarray = nums[start:end+1]
                
                if start <= i <= end:  # Element is in this subarray
                    if max(subarray) == nums[i]:
                        max_count += 1
                        max_subarrays.append(subarray)
                    
                    if min(subarray) == nums[i]:
                        min_count += 1
                        min_subarrays.append(subarray)
        
        max_contribution = nums[i] * max_count
        min_contribution = nums[i] * min_count
        net_contribution = max_contribution - min_contribution
        
        print(f"  Maximum in {max_count} subarrays: {max_subarrays}")
        print(f"  Minimum in {min_count} subarrays: {min_subarrays}")
        print(f"  Max contribution: {nums[i]} * {max_count} = {max_contribution}")
        print(f"  Min contribution: {nums[i]} * {min_count} = {min_contribution}")
        print(f"  Net contribution: {max_contribution} - {min_contribution} = {net_contribution}")
        
        contributions.append(net_contribution)
    
    total = sum(contributions)
    print(f"\nTotal sum of contributions: {contributions} = {total}")


if __name__ == "__main__":
    test_sum_of_subarray_ranges()
    demonstrate_monotonic_stack_approach()
    visualize_subarray_ranges()
    demonstrate_competitive_programming_patterns()
    analyze_time_complexity()
    test_edge_cases()
    demonstrate_contribution_calculation()

"""
Sum of Subarray Ranges demonstrates competitive programming patterns
with monotonic stack optimization, contribution technique, and efficient
calculation of element contributions across multiple subarrays.
"""
