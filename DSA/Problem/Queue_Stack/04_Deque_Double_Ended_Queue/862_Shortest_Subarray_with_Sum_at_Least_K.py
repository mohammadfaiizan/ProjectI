"""
862. Shortest Subarray with Sum at Least K - Multiple Approaches
Difficulty: Hard (but categorized as Medium in deque context)

Given an integer array nums and an integer k, return the length of the shortest non-empty subarray of nums with a sum of at least k. If there is no such subarray, return -1.

A subarray is a contiguous part of an array.
"""

from typing import List, Deque
from collections import deque
import heapq

class ShortestSubarrayWithSumAtLeastK:
    """Multiple approaches to find shortest subarray with sum at least K"""
    
    def shortestSubarray_deque_approach(self, nums: List[int], k: int) -> int:
        """
        Approach 1: Monotonic Deque with Prefix Sum (Optimal)
        
        Use deque to maintain increasing prefix sums for optimal subarray finding.
        
        Time: O(n), Space: O(n)
        """
        n = len(nums)
        # Calculate prefix sums
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
        
        dq: Deque[int] = deque()  # Store indices
        min_length = float('inf')
        
        for i in range(n + 1):
            # Check if we can form a valid subarray ending at i-1
            while dq and prefix[i] - prefix[dq[0]] >= k:
                min_length = min(min_length, i - dq.popleft())
            
            # Maintain increasing order of prefix sums
            while dq and prefix[dq[-1]] >= prefix[i]:
                dq.pop()
            
            dq.append(i)
        
        return min_length if min_length != float('inf') else -1
    
    def shortestSubarray_brute_force(self, nums: List[int], k: int) -> int:
        """
        Approach 2: Brute Force
        
        Check all possible subarrays.
        
        Time: O(n²), Space: O(1)
        """
        n = len(nums)
        min_length = float('inf')
        
        for i in range(n):
            current_sum = 0
            for j in range(i, n):
                current_sum += nums[j]
                if current_sum >= k:
                    min_length = min(min_length, j - i + 1)
                    break  # Found shortest starting at i
        
        return min_length if min_length != float('inf') else -1
    
    def shortestSubarray_prefix_sum_map(self, nums: List[int], k: int) -> int:
        """
        Approach 3: Prefix Sum with HashMap
        
        Use hashmap to store prefix sums and find valid subarrays.
        
        Time: O(n²), Space: O(n)
        """
        n = len(nums)
        prefix = [0] * (n + 1)
        
        # Calculate prefix sums
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
        
        min_length = float('inf')
        
        # For each ending position
        for j in range(1, n + 1):
            # Find the rightmost starting position
            for i in range(j):
                if prefix[j] - prefix[i] >= k:
                    min_length = min(min_length, j - i)
                    break
        
        return min_length if min_length != float('inf') else -1
    
    def shortestSubarray_sliding_window_modified(self, nums: List[int], k: int) -> int:
        """
        Approach 4: Modified Sliding Window
        
        Use sliding window with careful handling of negative numbers.
        
        Time: O(n²), Space: O(1)
        """
        n = len(nums)
        min_length = float('inf')
        
        for start in range(n):
            current_sum = 0
            for end in range(start, n):
                current_sum += nums[end]
                
                if current_sum >= k:
                    min_length = min(min_length, end - start + 1)
                    break  # Found shortest starting at start
        
        return min_length if min_length != float('inf') else -1
    
    def shortestSubarray_heap_approach(self, nums: List[int], k: int) -> int:
        """
        Approach 5: Min Heap with Prefix Sums
        
        Use min heap to efficiently find minimum prefix sum.
        
        Time: O(n log n), Space: O(n)
        """
        n = len(nums)
        prefix = [0] * (n + 1)
        
        # Calculate prefix sums
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
        
        heap = [(0, 0)]  # (prefix_sum, index)
        min_length = float('inf')
        
        for i in range(1, n + 1):
            # Remove invalid entries (where we can't form valid subarray)
            while heap and prefix[i] - heap[0][0] >= k:
                _, idx = heapq.heappop(heap)
                min_length = min(min_length, i - idx)
            
            heapq.heappush(heap, (prefix[i], i))
        
        return min_length if min_length != float('inf') else -1
    
    def shortestSubarray_segment_tree(self, nums: List[int], k: int) -> int:
        """
        Approach 6: Segment Tree for Range Minimum Query
        
        Use segment tree to find minimum prefix sum in ranges.
        
        Time: O(n log n), Space: O(n)
        """
        n = len(nums)
        prefix = [0] * (n + 1)
        
        # Calculate prefix sums
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
        
        # Coordinate compression for segment tree
        sorted_prefix = sorted(set(prefix))
        coord_map = {val: i for i, val in enumerate(sorted_prefix)}
        
        # Build segment tree for minimum index
        tree_size = len(sorted_prefix)
        tree = [float('inf')] * (4 * tree_size)
        
        def update(node: int, start: int, end: int, idx: int, val: int) -> None:
            if start == end:
                tree[node] = min(tree[node], val)
            else:
                mid = (start + end) // 2
                if idx <= mid:
                    update(2 * node, start, mid, idx, val)
                else:
                    update(2 * node + 1, mid + 1, end, idx, val)
                tree[node] = min(tree[2 * node], tree[2 * node + 1])
        
        def query_min(node: int, start: int, end: int, l: int, r: int) -> int:
            if r < start or end < l:
                return float('inf')
            if l <= start and end <= r:
                return tree[node]
            
            mid = (start + end) // 2
            left_min = query_min(2 * node, start, mid, l, r)
            right_min = query_min(2 * node + 1, mid + 1, end, l, r)
            return min(left_min, right_min)
        
        min_length = float('inf')
        update(1, 0, tree_size - 1, coord_map[prefix[0]], 0)
        
        for i in range(1, n + 1):
            # Find minimum prefix sum that makes current sum >= k
            target = prefix[i] - k
            
            # Binary search for the range
            left = 0
            right = len(sorted_prefix) - 1
            valid_idx = -1
            
            while left <= right:
                mid = (left + right) // 2
                if sorted_prefix[mid] <= target:
                    valid_idx = mid
                    left = mid + 1
                else:
                    right = mid - 1
            
            if valid_idx != -1:
                min_idx = query_min(1, 0, tree_size - 1, 0, valid_idx)
                if min_idx != float('inf'):
                    min_length = min(min_length, i - min_idx)
            
            update(1, 0, tree_size - 1, coord_map[prefix[i]], i)
        
        return min_length if min_length != float('inf') else -1
    
    def shortestSubarray_two_pointers_optimized(self, nums: List[int], k: int) -> int:
        """
        Approach 7: Two Pointers with Optimization
        
        Use two pointers with careful handling of negative numbers.
        
        Time: O(n²), Space: O(n)
        """
        n = len(nums)
        prefix = [0] * (n + 1)
        
        # Calculate prefix sums
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
        
        min_length = float('inf')
        
        # Try all possible starting positions
        for i in range(n + 1):
            # Use binary search to find the smallest j > i such that prefix[j] - prefix[i] >= k
            left, right = i + 1, n
            
            while left <= right:
                mid = (left + right) // 2
                if prefix[mid] - prefix[i] >= k:
                    min_length = min(min_length, mid - i)
                    right = mid - 1
                else:
                    left = mid + 1
        
        return min_length if min_length != float('inf') else -1


def test_shortest_subarray_with_sum_at_least_k():
    """Test shortest subarray with sum at least K algorithms"""
    solver = ShortestSubarrayWithSumAtLeastK()
    
    test_cases = [
        ([1], 1, 1, "Single element equal to K"),
        ([1, 2], 4, -1, "No valid subarray"),
        ([2, -1, 2], 3, 3, "Example 1"),
        ([1, 2, 3], 3, 2, "Multiple valid subarrays"),
        ([1, 1, 1, 1], 3, 3, "All ones"),
        ([-1, 2, 3], 3, 2, "With negative numbers"),
        ([2, -1, -1, 2], 3, 4, "Negative in middle"),
        ([1, 2, 3, 4], 6, 2, "Consecutive sum"),
        ([5, -3, 5], 6, 2, "Negative affecting sum"),
        ([1, -1, 1, 1, 1], 2, 3, "Mixed positive/negative"),
    ]
    
    algorithms = [
        ("Deque Approach", solver.shortestSubarray_deque_approach),
        ("Brute Force", solver.shortestSubarray_brute_force),
        ("Prefix Sum Map", solver.shortestSubarray_prefix_sum_map),
        ("Sliding Window Modified", solver.shortestSubarray_sliding_window_modified),
        ("Heap Approach", solver.shortestSubarray_heap_approach),
        ("Two Pointers Optimized", solver.shortestSubarray_two_pointers_optimized),
    ]
    
    print("=== Testing Shortest Subarray with Sum at Least K ===")
    
    for nums, k, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Array: {nums}")
        print(f"K: {k}")
        print(f"Expected: {expected}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(nums, k)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:25} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:25} | ERROR: {str(e)[:40]}")


def demonstrate_deque_approach():
    """Demonstrate deque approach step by step"""
    print("\n=== Deque Approach Step-by-Step Demo ===")
    
    nums = [2, -1, 2]
    k = 3
    
    print(f"Array: {nums}")
    print(f"K: {k}")
    
    # Calculate prefix sums
    n = len(nums)
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + nums[i]
    
    print(f"Prefix sums: {prefix}")
    
    dq = deque()
    min_length = float('inf')
    
    print("\nProcessing each prefix sum:")
    
    for i in range(n + 1):
        print(f"\nStep {i+1}: Processing prefix[{i}] = {prefix[i]}")
        
        # Check for valid subarrays
        removed_valid = []
        while dq and prefix[i] - prefix[dq[0]] >= k:
            j = dq.popleft()
            length = i - j
            min_length = min(min_length, length)
            removed_valid.append(f"idx {j} (sum {prefix[j]}) -> length {length}")
        
        if removed_valid:
            print(f"  Found valid subarrays: {removed_valid}")
            print(f"  Current min length: {min_length}")
        
        # Maintain increasing order
        removed_larger = []
        while dq and prefix[dq[-1]] >= prefix[i]:
            j = dq.pop()
            removed_larger.append(f"idx {j} (sum {prefix[j]})")
        
        if removed_larger:
            print(f"  Removed larger/equal prefix sums: {removed_larger}")
        
        dq.append(i)
        print(f"  Added index {i}")
        print(f"  Deque indices: {list(dq)}")
        print(f"  Deque prefix sums: {[prefix[idx] for idx in dq]}")
    
    result = min_length if min_length != float('inf') else -1
    print(f"\nFinal result: {result}")


def demonstrate_prefix_sum_concept():
    """Demonstrate prefix sum concept"""
    print("\n=== Prefix Sum Concept Demonstration ===")
    
    nums = [1, -1, 2, 1]
    k = 2
    
    print(f"Array: {nums}")
    print(f"K: {k}")
    
    # Calculate prefix sums
    n = len(nums)
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + nums[i]
    
    print(f"Prefix sums: {prefix}")
    
    print("\nSubarray sums using prefix sums:")
    print("For subarray [i, j], sum = prefix[j+1] - prefix[i]")
    
    for i in range(n):
        for j in range(i, n):
            subarray_sum = prefix[j + 1] - prefix[i]
            subarray = nums[i:j+1]
            valid = "✓" if subarray_sum >= k else "✗"
            print(f"  Subarray [{i}:{j+1}] = {subarray} -> sum = {prefix[j+1]} - {prefix[i]} = {subarray_sum} {valid}")


def visualize_deque_operations():
    """Visualize deque operations"""
    print("\n=== Deque Operations Visualization ===")
    
    nums = [2, -1, 2]
    k = 3
    
    print(f"Finding shortest subarray with sum >= {k}")
    print(f"Array: {nums}")
    
    # Calculate prefix sums
    n = len(nums)
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + nums[i]
    
    print(f"Prefix: {prefix}")
    
    # Show the deque maintaining increasing prefix sums
    print("\nDeque maintains increasing prefix sums:")
    
    dq = deque()
    
    for i in range(n + 1):
        print(f"\nStep {i+1}: prefix[{i}] = {prefix[i]}")
        
        # Show current deque state
        if dq:
            deque_values = [f"idx{idx}({prefix[idx]})" for idx in dq]
            print(f"  Current deque: {deque_values}")
        else:
            print(f"  Current deque: empty")
        
        # Check for valid subarrays (prefix[i] - prefix[j] >= k)
        valid_found = False
        temp_dq = list(dq)
        
        for j in temp_dq:
            if prefix[i] - prefix[j] >= k:
                length = i - j
                print(f"  Valid subarray: prefix[{i}] - prefix[{j}] = {prefix[i]} - {prefix[j]} = {prefix[i] - prefix[j]} >= {k}")
                print(f"    Subarray length: {length}")
                valid_found = True
                break
        
        if not valid_found and dq:
            print(f"  No valid subarray ending at position {i-1}")
        
        # Simulate deque operations
        while dq and prefix[dq[-1]] >= prefix[i]:
            removed = dq.pop()
            print(f"  Removed idx {removed} (prefix {prefix[removed]}) - not increasing")
        
        dq.append(i)
        print(f"  Added idx {i}")


def benchmark_shortest_subarray():
    """Benchmark different approaches"""
    import time
    import random
    
    algorithms = [
        ("Deque Approach", ShortestSubarrayWithSumAtLeastK().shortestSubarray_deque_approach),
        ("Brute Force", ShortestSubarrayWithSumAtLeastK().shortestSubarray_brute_force),
        ("Prefix Sum Map", ShortestSubarrayWithSumAtLeastK().shortestSubarray_prefix_sum_map),
        ("Heap Approach", ShortestSubarrayWithSumAtLeastK().shortestSubarray_heap_approach),
    ]
    
    # Test with different array sizes
    test_sizes = [100, 1000, 5000]
    
    print("\n=== Shortest Subarray Performance Benchmark ===")
    
    for size in test_sizes:
        print(f"\n--- Array Size: {size} ---")
        
        # Generate random array with mix of positive and negative numbers
        nums = [random.randint(-10, 20) for _ in range(size)]
        k = size // 2  # Reasonable target sum
        
        for alg_name, alg_func in algorithms:
            start_time = time.time()
            
            try:
                result = alg_func(nums, k)
                end_time = time.time()
                print(f"{alg_name:25} | Time: {end_time - start_time:.4f}s | Result: {result}")
            except Exception as e:
                print(f"{alg_name:25} | ERROR: {str(e)[:30]}")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    solver = ShortestSubarrayWithSumAtLeastK()
    
    edge_cases = [
        ([], 1, -1, "Empty array"),
        ([1], 2, -1, "Single element < K"),
        ([5], 3, 1, "Single element > K"),
        ([1, 1], 3, -1, "Sum < K"),
        ([-1, -1], 1, -1, "All negative, positive K"),
        ([-1, -1], -1, 1, "All negative, negative K"),
        ([0, 0, 0], 0, 1, "All zeros, K=0"),
        ([1, 0, -1, 2], 2, 1, "Mixed with zero"),
        ([10], 5, 1, "Single large element"),
        ([-5, 10, -3], 2, 2, "Negative-Positive-Negative"),
    ]
    
    for nums, k, expected, description in edge_cases:
        try:
            result = solver.shortestSubarray_deque_approach(nums, k)
            status = "✓" if result == expected else "✗"
            print(f"{description:30} | {status} | nums: {nums}, k: {k} -> {result}")
        except Exception as e:
            print(f"{description:30} | ERROR: {str(e)[:30]}")


def compare_approaches():
    """Compare different approaches"""
    print("\n=== Approach Comparison ===")
    
    test_cases = [
        ([2, -1, 2], 3),
        ([1, 2, 3], 3),
        ([-1, 2, 3], 3),
        ([1, 1, 1, 1], 3),
    ]
    
    solver = ShortestSubarrayWithSumAtLeastK()
    
    approaches = [
        ("Deque", solver.shortestSubarray_deque_approach),
        ("Brute Force", solver.shortestSubarray_brute_force),
        ("Prefix Map", solver.shortestSubarray_prefix_sum_map),
        ("Heap", solver.shortestSubarray_heap_approach),
    ]
    
    for i, (nums, k) in enumerate(test_cases):
        print(f"\nTest case {i+1}: nums={nums}, k={k}")
        
        results = {}
        
        for name, func in approaches:
            try:
                result = func(nums, k)
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
        ("Deque Approach", "O(n)", "O(n)", "Each element added/removed once"),
        ("Brute Force", "O(n²)", "O(1)", "Check all possible subarrays"),
        ("Prefix Sum Map", "O(n²)", "O(n)", "Prefix sums with nested loops"),
        ("Sliding Window Modified", "O(n²)", "O(1)", "Modified sliding window"),
        ("Heap Approach", "O(n log n)", "O(n)", "Heap operations with prefix sums"),
        ("Segment Tree", "O(n log n)", "O(n)", "Range minimum queries"),
        ("Two Pointers Optimized", "O(n log n)", "O(n)", "Binary search for each position"),
    ]
    
    print(f"{'Approach':<25} | {'Time':<12} | {'Space':<8} | {'Notes'}")
    print("-" * 75)
    
    for approach, time_comp, space_comp, notes in approaches:
        print(f"{approach:<25} | {time_comp:<12} | {space_comp:<8} | {notes}")


def demonstrate_why_deque_works():
    """Demonstrate why deque approach works"""
    print("\n=== Why Deque Approach Works ===")
    
    print("Key insights:")
    print("1. We need to find shortest subarray with sum >= K")
    print("2. Use prefix sums: subarray[i,j] sum = prefix[j+1] - prefix[i]")
    print("3. We want: prefix[j+1] - prefix[i] >= K")
    print("4. Rearranged: prefix[j+1] >= prefix[i] + K")
    
    print("\nDeque properties:")
    print("1. Maintains indices in increasing order of prefix sums")
    print("2. For any position j, we want the rightmost i such that prefix[j] - prefix[i] >= K")
    print("3. This gives us the shortest subarray ending at position j-1")
    
    print("\nWhy increasing order?")
    print("- If prefix[a] >= prefix[b] and a < b, then a is never useful")
    print("- Any subarray ending after b that works with a also works with b")
    print("- But subarray with b is shorter, so a can be discarded")
    
    nums = [1, -1, 1, 1]
    k = 2
    
    print(f"\nExample: nums = {nums}, k = {k}")
    
    # Calculate prefix sums
    n = len(nums)
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + nums[i]
    
    print(f"Prefix sums: {prefix}")
    
    print("\nWhy we maintain increasing order:")
    print("- At prefix[2] = 0: we have prefix[0] = 0 and prefix[1] = 1")
    print("- Since prefix[1] >= prefix[0] and 1 > 0, we can remove prefix[0]")
    print("- Any future subarray that could use prefix[0] can use prefix[1] for shorter length")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    # Application 1: Minimum time to achieve target revenue
    print("1. Business Revenue - Minimum days to achieve target revenue:")
    daily_profits = [100, -50, 200, 150, -30, 180]  # Daily profit/loss
    target_revenue = 400
    
    solver = ShortestSubarrayWithSumAtLeastK()
    min_days = solver.shortestSubarray_deque_approach(daily_profits, target_revenue)
    
    print(f"  Daily profits: {daily_profits}")
    print(f"  Target revenue: ${target_revenue}")
    print(f"  Minimum days needed: {min_days if min_days != -1 else 'Not achievable'}")
    
    # Application 2: Battery charging optimization
    print("\n2. Battery Charging - Minimum time to reach charge level:")
    charge_rates = [20, -5, 15, 25, -10, 30]  # Charge rate per hour (+ charge, - discharge)
    target_charge = 50
    
    min_hours = solver.shortestSubarray_deque_approach(charge_rates, target_charge)
    
    print(f"  Hourly charge rates: {charge_rates}")
    print(f"  Target charge level: {target_charge}")
    print(f"  Minimum hours needed: {min_hours if min_hours != -1 else 'Not achievable'}")
    
    # Application 3: Stock trading - minimum holding period for profit
    print("\n3. Stock Trading - Minimum holding period for target profit:")
    daily_changes = [5, -2, 8, -3, 10, -1, 6]  # Daily price changes
    target_profit = 15
    
    min_period = solver.shortestSubarray_deque_approach(daily_changes, target_profit)
    
    print(f"  Daily price changes: {daily_changes}")
    print(f"  Target profit: ${target_profit}")
    print(f"  Minimum holding period: {min_period if min_period != -1 else 'Not achievable'} days")


if __name__ == "__main__":
    test_shortest_subarray_with_sum_at_least_k()
    demonstrate_deque_approach()
    demonstrate_prefix_sum_concept()
    visualize_deque_operations()
    demonstrate_why_deque_works()
    demonstrate_real_world_applications()
    test_edge_cases()
    compare_approaches()
    analyze_time_complexity()
    benchmark_shortest_subarray()

"""
Shortest Subarray with Sum at Least K demonstrates advanced deque applications
for optimization problems with negative numbers, including prefix sum techniques
and monotonic deque properties for efficient subarray finding.
"""
