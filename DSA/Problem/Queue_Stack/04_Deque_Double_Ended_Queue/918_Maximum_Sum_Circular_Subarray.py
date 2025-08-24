"""
918. Maximum Sum Circular Subarray - Multiple Approaches
Difficulty: Medium

Given a circular integer array nums of length n, return the maximum possible sum of a non-empty subarray of nums.

A circular array means the end of the array connects to the beginning of the array. Formally, the next element of nums[i] is nums[(i + 1) % n] and the previous element of nums[i] is nums[(i - 1 + n) % n].

A subarray may only include each element of the circular array at most once.
"""

from typing import List, Deque
from collections import deque

class MaximumSumCircularSubarray:
    """Multiple approaches to find maximum sum circular subarray"""
    
    def maxSubarraySumCircular_kadane_approach(self, nums: List[int]) -> int:
        """
        Approach 1: Kadane's Algorithm with Circular Logic (Optimal)
        
        Consider two cases: max subarray is non-circular or circular.
        
        Time: O(n), Space: O(1)
        """
        def kadane_max(arr: List[int]) -> int:
            """Standard Kadane's algorithm for maximum subarray"""
            max_ending_here = max_so_far = arr[0]
            
            for i in range(1, len(arr)):
                max_ending_here = max(arr[i], max_ending_here + arr[i])
                max_so_far = max(max_so_far, max_ending_here)
            
            return max_so_far
        
        def kadane_min(arr: List[int]) -> int:
            """Kadane's algorithm for minimum subarray"""
            min_ending_here = min_so_far = arr[0]
            
            for i in range(1, len(arr)):
                min_ending_here = min(arr[i], min_ending_here + arr[i])
                min_so_far = min(min_so_far, min_ending_here)
            
            return min_so_far
        
        # Case 1: Maximum subarray is non-circular
        max_kadane = kadane_max(nums)
        
        # Case 2: Maximum subarray is circular
        # This means we take total sum minus minimum subarray
        total_sum = sum(nums)
        min_kadane = kadane_min(nums)
        max_circular = total_sum - min_kadane
        
        # Edge case: if all elements are negative, max_circular would be 0
        # but we need at least one element, so return max_kadane
        if max_circular == 0:
            return max_kadane
        
        return max(max_kadane, max_circular)
    
    def maxSubarraySumCircular_deque_approach(self, nums: List[int]) -> int:
        """
        Approach 2: Deque with Prefix Sums
        
        Use deque to find maximum subarray sum in circular array.
        
        Time: O(n), Space: O(n)
        """
        n = len(nums)
        
        # Create extended array (original + original)
        extended = nums + nums
        
        # Calculate prefix sums
        prefix = [0] * (2 * n + 1)
        for i in range(2 * n):
            prefix[i + 1] = prefix[i] + extended[i]
        
        max_sum = float('-inf')
        dq: Deque[int] = deque()
        
        # Process each position
        for i in range(2 * n):
            # Remove indices outside window of size n
            while dq and dq[0] < i - n + 1:
                dq.popleft()
            
            # Calculate maximum subarray ending at i
            if dq:
                max_sum = max(max_sum, prefix[i + 1] - prefix[dq[0]])
            
            # Maintain increasing order of prefix sums
            while dq and prefix[dq[-1]] >= prefix[i + 1]:
                dq.pop()
            
            dq.append(i + 1)
            
            # For the first n elements, also consider subarray starting from 0
            if i < n:
                max_sum = max(max_sum, prefix[i + 1])
        
        return max_sum
    
    def maxSubarraySumCircular_brute_force(self, nums: List[int]) -> int:
        """
        Approach 3: Brute Force
        
        Check all possible subarrays in circular array.
        
        Time: O(n²), Space: O(1)
        """
        n = len(nums)
        max_sum = float('-inf')
        
        # Try all possible starting positions
        for i in range(n):
            current_sum = 0
            
            # Try all possible lengths (1 to n)
            for length in range(1, n + 1):
                j = (i + length - 1) % n
                current_sum += nums[j]
                max_sum = max(max_sum, current_sum)
        
        return max_sum
    
    def maxSubarraySumCircular_prefix_approach(self, nums: List[int]) -> int:
        """
        Approach 4: Prefix Sum with Circular Logic
        
        Use prefix sums to handle circular nature.
        
        Time: O(n²), Space: O(n)
        """
        n = len(nums)
        
        # Calculate prefix sums
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
        
        max_sum = float('-inf')
        
        # Case 1: Non-circular subarrays
        for i in range(n):
            for j in range(i, n):
                subarray_sum = prefix[j + 1] - prefix[i]
                max_sum = max(max_sum, subarray_sum)
        
        # Case 2: Circular subarrays
        total_sum = prefix[n]
        
        # Find minimum subarray sum (excluding full array)
        min_sum = float('inf')
        for i in range(n):
            for j in range(i, n):
                if i == 0 and j == n - 1:  # Skip full array
                    continue
                subarray_sum = prefix[j + 1] - prefix[i]
                min_sum = min(min_sum, subarray_sum)
        
        if min_sum != float('inf'):
            max_sum = max(max_sum, total_sum - min_sum)
        
        return max_sum
    
    def maxSubarraySumCircular_dp_approach(self, nums: List[int]) -> int:
        """
        Approach 5: Dynamic Programming
        
        Use DP to track maximum and minimum subarrays.
        
        Time: O(n), Space: O(1)
        """
        n = len(nums)
        
        # DP for maximum subarray (Kadane's)
        max_ending_here = max_so_far = nums[0]
        
        # DP for minimum subarray
        min_ending_here = min_so_far = nums[0]
        
        total_sum = nums[0]
        
        for i in range(1, n):
            total_sum += nums[i]
            
            # Update maximum subarray
            max_ending_here = max(nums[i], max_ending_here + nums[i])
            max_so_far = max(max_so_far, max_ending_here)
            
            # Update minimum subarray
            min_ending_here = min(nums[i], min_ending_here + nums[i])
            min_so_far = min(min_so_far, min_ending_here)
        
        # Case 1: Maximum is non-circular
        result = max_so_far
        
        # Case 2: Maximum is circular (total - minimum)
        if min_so_far < total_sum:  # Ensure we don't take empty subarray
            result = max(result, total_sum - min_so_far)
        
        return result
    
    def maxSubarraySumCircular_sliding_window(self, nums: List[int]) -> int:
        """
        Approach 6: Sliding Window with Deque
        
        Use sliding window maximum technique.
        
        Time: O(n), Space: O(n)
        """
        n = len(nums)
        
        # Calculate prefix sums for extended array
        extended = nums + nums
        prefix = [0] * (2 * n + 1)
        
        for i in range(2 * n):
            prefix[i + 1] = prefix[i] + extended[i]
        
        # Use deque to maintain minimum prefix sum in sliding window
        dq: Deque[int] = deque([0])
        max_sum = nums[0]  # At least one element
        
        for i in range(1, 2 * n + 1):
            # Remove elements outside window of size n
            while dq and dq[0] < i - n:
                dq.popleft()
            
            # Calculate maximum subarray ending at position i-1
            if dq:
                max_sum = max(max_sum, prefix[i] - prefix[dq[0]])
            
            # Maintain increasing order of prefix sums
            while dq and prefix[dq[-1]] >= prefix[i]:
                dq.pop()
            
            dq.append(i)
        
        return max_sum
    
    def maxSubarraySumCircular_two_pass(self, nums: List[int]) -> int:
        """
        Approach 7: Two Pass Algorithm
        
        First pass for max subarray, second for circular case.
        
        Time: O(n), Space: O(1)
        """
        n = len(nums)
        
        # First pass: Find maximum subarray sum (non-circular)
        def max_subarray_sum(arr: List[int]) -> int:
            max_ending = max_so_far = arr[0]
            for i in range(1, len(arr)):
                max_ending = max(arr[i], max_ending + arr[i])
                max_so_far = max(max_so_far, max_ending)
            return max_so_far
        
        # Second pass: Find minimum subarray sum
        def min_subarray_sum(arr: List[int]) -> int:
            min_ending = min_so_far = arr[0]
            for i in range(1, len(arr)):
                min_ending = min(arr[i], min_ending + arr[i])
                min_so_far = min(min_so_far, min_ending)
            return min_so_far
        
        # Case 1: Maximum subarray is non-circular
        max_normal = max_subarray_sum(nums)
        
        # Case 2: Maximum subarray is circular
        total_sum = sum(nums)
        min_subarray = min_subarray_sum(nums)
        max_circular = total_sum - min_subarray
        
        # Handle edge case where all elements are negative
        if max_circular == 0:
            return max_normal
        
        return max(max_normal, max_circular)


def test_maximum_sum_circular_subarray():
    """Test maximum sum circular subarray algorithms"""
    solver = MaximumSumCircularSubarray()
    
    test_cases = [
        ([1,-2,3,-2], 3, "Example 1"),
        ([5,-3,5], 10, "Example 2"),
        ([-3,-2,-3], -2, "Example 3"),
        ([1], 1, "Single element"),
        ([1, 2], 3, "Two elements"),
        ([3, -1, 2, -1], 4, "Circular case"),
        ([2, -3, -1, 5, -4], 5, "Mixed values"),
        ([-2, -3, -1], -1, "All negative"),
        ([5, 5, 5], 15, "All positive"),
        ([1, -2, 3, -2, 5], 6, "Complex case"),
    ]
    
    algorithms = [
        ("Kadane Approach", solver.maxSubarraySumCircular_kadane_approach),
        ("Deque Approach", solver.maxSubarraySumCircular_deque_approach),
        ("Brute Force", solver.maxSubarraySumCircular_brute_force),
        ("Prefix Approach", solver.maxSubarraySumCircular_prefix_approach),
        ("DP Approach", solver.maxSubarraySumCircular_dp_approach),
        ("Sliding Window", solver.maxSubarraySumCircular_sliding_window),
        ("Two Pass", solver.maxSubarraySumCircular_two_pass),
    ]
    
    print("=== Testing Maximum Sum Circular Subarray ===")
    
    for nums, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Array: {nums}")
        print(f"Expected: {expected}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(nums)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:20} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:40]}")


def demonstrate_kadane_approach():
    """Demonstrate Kadane's approach step by step"""
    print("\n=== Kadane's Approach Step-by-Step Demo ===")
    
    nums = [5, -3, 5]
    print(f"Array: {nums}")
    
    # Case 1: Non-circular maximum subarray
    print("\nCase 1: Non-circular maximum subarray (Standard Kadane's)")
    
    max_ending_here = max_so_far = nums[0]
    print(f"Initial: max_ending_here = max_so_far = {nums[0]}")
    
    for i in range(1, len(nums)):
        max_ending_here = max(nums[i], max_ending_here + nums[i])
        max_so_far = max(max_so_far, max_ending_here)
        
        print(f"Step {i}: nums[{i}] = {nums[i]}")
        print(f"  max_ending_here = max({nums[i]}, {max_ending_here - nums[i] if i > 0 else 0} + {nums[i]}) = {max_ending_here}")
        print(f"  max_so_far = {max_so_far}")
    
    max_kadane = max_so_far
    print(f"Non-circular maximum: {max_kadane}")
    
    # Case 2: Circular maximum subarray
    print(f"\nCase 2: Circular maximum subarray")
    print("Idea: Total sum - minimum subarray = maximum circular subarray")
    
    total_sum = sum(nums)
    print(f"Total sum: {total_sum}")
    
    # Find minimum subarray using Kadane's
    print(f"\nFinding minimum subarray:")
    min_ending_here = min_so_far = nums[0]
    print(f"Initial: min_ending_here = min_so_far = {nums[0]}")
    
    for i in range(1, len(nums)):
        min_ending_here = min(nums[i], min_ending_here + nums[i])
        min_so_far = min(min_so_far, min_ending_here)
        
        print(f"Step {i}: nums[{i}] = {nums[i]}")
        print(f"  min_ending_here = min({nums[i]}, {min_ending_here - nums[i] if i > 0 else 0} + {nums[i]}) = {min_ending_here}")
        print(f"  min_so_far = {min_so_far}")
    
    min_kadane = min_so_far
    max_circular = total_sum - min_kadane
    
    print(f"Minimum subarray sum: {min_kadane}")
    print(f"Circular maximum: {total_sum} - ({min_kadane}) = {max_circular}")
    
    # Final result
    result = max(max_kadane, max_circular) if max_circular != 0 else max_kadane
    print(f"\nFinal result: max({max_kadane}, {max_circular}) = {result}")


def visualize_circular_array():
    """Visualize circular array concept"""
    print("\n=== Circular Array Visualization ===")
    
    nums = [3, -1, 2, -1]
    print(f"Array: {nums}")
    
    print("\nCircular array representation:")
    print("Index:  0   1   2   3   0   1   2   3")
    print("Value:  3  -1   2  -1   3  -1   2  -1")
    print("        ^               ^")
    print("        |               |")
    print("        Connected in circular fashion")
    
    print("\nPossible subarrays:")
    n = len(nums)
    
    # Non-circular subarrays
    print("Non-circular subarrays:")
    for i in range(n):
        for j in range(i, n):
            subarray = nums[i:j+1]
            subarray_sum = sum(subarray)
            print(f"  [{i}:{j+1}] = {subarray} -> sum = {subarray_sum}")
    
    # Circular subarrays
    print("\nCircular subarrays:")
    for i in range(n):
        for j in range(i):  # j < i means circular
            # Circular subarray from i to end, then from 0 to j
            subarray = nums[i:] + nums[:j+1]
            subarray_sum = sum(subarray)
            indices = list(range(i, n)) + list(range(j + 1))
            print(f"  indices {indices} = {subarray} -> sum = {subarray_sum}")
    
    # Find maximum
    all_sums = []
    
    # Non-circular
    for i in range(n):
        for j in range(i, n):
            all_sums.append(sum(nums[i:j+1]))
    
    # Circular
    for i in range(n):
        for j in range(i):
            all_sums.append(sum(nums[i:] + nums[:j+1]))
    
    print(f"\nAll possible sums: {sorted(set(all_sums))}")
    print(f"Maximum sum: {max(all_sums)}")


def demonstrate_circular_vs_non_circular():
    """Demonstrate difference between circular and non-circular cases"""
    print("\n=== Circular vs Non-Circular Cases ===")
    
    test_cases = [
        ([1, -2, 3, -2], "Non-circular wins"),
        ([5, -3, 5], "Circular wins"),
        ([-3, -2, -3], "All negative"),
        ([2, -1, 2, -1], "Circular case"),
    ]
    
    solver = MaximumSumCircularSubarray()
    
    for nums, description in test_cases:
        print(f"\n{description}: {nums}")
        
        # Non-circular maximum (standard Kadane's)
        def kadane_max(arr):
            max_ending = max_so_far = arr[0]
            for i in range(1, len(arr)):
                max_ending = max(arr[i], max_ending + arr[i])
                max_so_far = max(max_so_far, max_ending)
            return max_so_far
        
        # Minimum subarray
        def kadane_min(arr):
            min_ending = min_so_far = arr[0]
            for i in range(1, len(arr)):
                min_ending = min(arr[i], min_ending + arr[i])
                min_so_far = min(min_so_far, min_ending)
            return min_so_far
        
        max_non_circular = kadane_max(nums)
        total_sum = sum(nums)
        min_subarray = kadane_min(nums)
        max_circular = total_sum - min_subarray
        
        print(f"  Non-circular maximum: {max_non_circular}")
        print(f"  Total sum: {total_sum}")
        print(f"  Minimum subarray: {min_subarray}")
        print(f"  Circular maximum: {total_sum} - ({min_subarray}) = {max_circular}")
        
        if max_circular == 0:
            result = max_non_circular
            print(f"  Edge case: circular = 0, use non-circular")
        else:
            result = max(max_non_circular, max_circular)
            winner = "circular" if max_circular > max_non_circular else "non-circular"
            print(f"  Winner: {winner}")
        
        print(f"  Final result: {result}")


def benchmark_circular_subarray():
    """Benchmark different approaches"""
    import time
    import random
    
    algorithms = [
        ("Kadane Approach", MaximumSumCircularSubarray().maxSubarraySumCircular_kadane_approach),
        ("Deque Approach", MaximumSumCircularSubarray().maxSubarraySumCircular_deque_approach),
        ("Brute Force", MaximumSumCircularSubarray().maxSubarraySumCircular_brute_force),
        ("DP Approach", MaximumSumCircularSubarray().maxSubarraySumCircular_dp_approach),
        ("Two Pass", MaximumSumCircularSubarray().maxSubarraySumCircular_two_pass),
    ]
    
    # Test with different array sizes
    test_sizes = [100, 1000, 5000]
    
    print("\n=== Maximum Sum Circular Subarray Performance Benchmark ===")
    
    for size in test_sizes:
        print(f"\n--- Array Size: {size} ---")
        
        # Generate random array
        nums = [random.randint(-10, 10) for _ in range(size)]
        
        for alg_name, alg_func in algorithms:
            start_time = time.time()
            
            try:
                result = alg_func(nums)
                end_time = time.time()
                print(f"{alg_name:20} | Time: {end_time - start_time:.4f}s | Result: {result}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:30]}")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    solver = MaximumSumCircularSubarray()
    
    edge_cases = [
        ([1], 1, "Single element"),
        ([-1], -1, "Single negative"),
        ([1, 2], 3, "Two positive"),
        ([-1, -2], -1, "Two negative"),
        ([0], 0, "Single zero"),
        ([0, 0], 0, "All zeros"),
        ([5, -5], 5, "Sum to zero"),
        ([1, -1, 1], 2, "Alternating"),
        ([10, -1, 10], 20, "Circular better"),
        ([-1, 2, -1], 2, "Middle element best"),
    ]
    
    for nums, expected, description in edge_cases:
        try:
            result = solver.maxSubarraySumCircular_kadane_approach(nums)
            status = "✓" if result == expected else "✗"
            print(f"{description:25} | {status} | nums: {nums} -> {result}")
        except Exception as e:
            print(f"{description:25} | ERROR: {str(e)[:30]}")


def compare_approaches():
    """Compare different approaches"""
    print("\n=== Approach Comparison ===")
    
    test_cases = [
        [1, -2, 3, -2],
        [5, -3, 5],
        [-3, -2, -3],
        [3, -1, 2, -1],
    ]
    
    solver = MaximumSumCircularSubarray()
    
    approaches = [
        ("Kadane", solver.maxSubarraySumCircular_kadane_approach),
        ("Deque", solver.maxSubarraySumCircular_deque_approach),
        ("Brute Force", solver.maxSubarraySumCircular_brute_force),
        ("DP", solver.maxSubarraySumCircular_dp_approach),
        ("Two Pass", solver.maxSubarraySumCircular_two_pass),
    ]
    
    for i, nums in enumerate(test_cases):
        print(f"\nTest case {i+1}: {nums}")
        
        results = {}
        
        for name, func in approaches:
            try:
                result = func(nums)
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
        ("Kadane Approach", "O(n)", "O(1)", "Two passes of Kadane's algorithm"),
        ("Deque Approach", "O(n)", "O(n)", "Deque with prefix sums"),
        ("Brute Force", "O(n²)", "O(1)", "Check all possible subarrays"),
        ("Prefix Approach", "O(n²)", "O(n)", "Prefix sums with nested loops"),
        ("DP Approach", "O(n)", "O(1)", "Single pass DP"),
        ("Sliding Window", "O(n)", "O(n)", "Deque-based sliding window"),
        ("Two Pass", "O(n)", "O(1)", "Separate max and min passes"),
    ]
    
    print(f"{'Approach':<20} | {'Time':<8} | {'Space':<8} | {'Notes'}")
    print("-" * 65)
    
    for approach, time_comp, space_comp, notes in approaches:
        print(f"{approach:<20} | {time_comp:<8} | {space_comp:<8} | {notes}")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    # Application 1: Circular shift work scheduling
    print("1. Shift Work Scheduling - Maximum productivity in circular schedule:")
    productivity = [8, -2, 6, -1, 7, -3]  # Productivity scores for each shift
    
    solver = MaximumSumCircularSubarray()
    max_productivity = solver.maxSubarraySumCircular_kadane_approach(productivity)
    
    print(f"  Shift productivity: {productivity}")
    print(f"  Maximum total productivity: {max_productivity}")
    print("  (Can wrap around from last shift to first shift)")
    
    # Application 2: Circular buffer optimization
    print("\n2. Circular Buffer - Maximum sum of consecutive elements:")
    buffer_values = [5, -1, 3, -2, 4, -1]  # Values in circular buffer
    
    max_buffer_sum = solver.maxSubarraySumCircular_kadane_approach(buffer_values)
    
    print(f"  Buffer values: {buffer_values}")
    print(f"  Maximum consecutive sum: {max_buffer_sum}")
    print("  (Buffer wraps around from end to beginning)")
    
    # Application 3: Circular route profit optimization
    print("\n3. Delivery Route - Maximum profit on circular route:")
    route_profits = [10, -5, 8, -3, 12, -7]  # Profit at each stop
    
    max_route_profit = solver.maxSubarraySumCircular_kadane_approach(route_profits)
    
    print(f"  Route profits: {route_profits}")
    print(f"  Maximum consecutive profit: {max_route_profit}")
    print("  (Route is circular - can continue from last stop to first)")


def demonstrate_why_circular_works():
    """Demonstrate why circular approach works"""
    print("\n=== Why Circular Approach Works ===")
    
    print("Key insight: Maximum circular subarray = Total sum - Minimum subarray")
    print("\nWhy this works:")
    print("1. If maximum subarray wraps around, it uses elements from both ends")
    print("2. The elements NOT used form a contiguous subarray in the middle")
    print("3. To maximize the circular subarray, minimize the middle subarray")
    print("4. So: max_circular = total_sum - min_subarray")
    
    nums = [5, -3, 5]
    print(f"\nExample: {nums}")
    
    print("\nAll possible subarrays:")
    n = len(nums)
    
    # Non-circular
    print("Non-circular:")
    for i in range(n):
        for j in range(i, n):
            subarray = nums[i:j+1]
            print(f"  {subarray} -> sum = {sum(subarray)}")
    
    # Circular (complement of middle subarrays)
    print("\nCircular (total - middle):")
    total = sum(nums)
    
    for i in range(n):
        for j in range(i, n):
            if i == 0 and j == n - 1:  # Skip full array
                continue
            middle = nums[i:j+1]
            circular_sum = total - sum(middle)
            remaining = nums[:i] + nums[j+1:]
            print(f"  total - {middle} = {total} - {sum(middle)} = {circular_sum}")
            print(f"    (equivalent to {remaining})")
    
    print(f"\nMaximum non-circular: {max(sum(nums[i:j+1]) for i in range(n) for j in range(i, n))}")
    
    # Find minimum middle subarray (excluding full array)
    min_middle = float('inf')
    for i in range(n):
        for j in range(i, n):
            if i == 0 and j == n - 1:
                continue
            min_middle = min(min_middle, sum(nums[i:j+1]))
    
    max_circular = total - min_middle if min_middle != float('inf') else 0
    print(f"Maximum circular: {total} - {min_middle} = {max_circular}")


if __name__ == "__main__":
    test_maximum_sum_circular_subarray()
    demonstrate_kadane_approach()
    visualize_circular_array()
    demonstrate_circular_vs_non_circular()
    demonstrate_why_circular_works()
    demonstrate_real_world_applications()
    test_edge_cases()
    compare_approaches()
    analyze_time_complexity()
    benchmark_circular_subarray()

"""
Maximum Sum Circular Subarray demonstrates advanced applications of Kadane's
algorithm for circular arrays, including deque-based approaches and multiple
optimization strategies for handling wrap-around subarrays.
"""
