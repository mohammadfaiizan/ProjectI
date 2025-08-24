"""
1438. Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit - Multiple Approaches
Difficulty: Hard

Given an array of integers nums and an integer limit, return the size of the longest non-empty subarray such that the absolute difference between any two elements in this subarray is at most limit.
"""

from typing import List, Deque
from collections import deque
import heapq

class LongestSubarrayWithLimit:
    """Multiple approaches to find longest subarray with absolute difference limit"""
    
    def longestSubarray_two_deques(self, nums: List[int], limit: int) -> int:
        """
        Approach 1: Two Deques (Optimal)
        
        Use two deques to track min and max in sliding window.
        
        Time: O(n), Space: O(n)
        """
        max_deque = deque()  # Decreasing order
        min_deque = deque()  # Increasing order
        
        left = 0
        max_length = 0
        
        for right in range(len(nums)):
            # Update max deque
            while max_deque and nums[max_deque[-1]] <= nums[right]:
                max_deque.pop()
            max_deque.append(right)
            
            # Update min deque
            while min_deque and nums[min_deque[-1]] >= nums[right]:
                min_deque.pop()
            min_deque.append(right)
            
            # Shrink window if difference exceeds limit
            while nums[max_deque[0]] - nums[min_deque[0]] > limit:
                if max_deque[0] == left:
                    max_deque.popleft()
                if min_deque[0] == left:
                    min_deque.popleft()
                left += 1
            
            max_length = max(max_length, right - left + 1)
        
        return max_length
    
    def longestSubarray_two_heaps(self, nums: List[int], limit: int) -> int:
        """
        Approach 2: Two Heaps with Lazy Deletion
        
        Use max heap and min heap to track extremes.
        
        Time: O(n log n), Space: O(n)
        """
        max_heap = []  # For maximum values
        min_heap = []  # For minimum values
        
        left = 0
        max_length = 0
        
        for right in range(len(nums)):
            heapq.heappush(max_heap, (-nums[right], right))
            heapq.heappush(min_heap, (nums[right], right))
            
            # Remove elements outside current window
            while max_heap and max_heap[0][1] < left:
                heapq.heappop(max_heap)
            while min_heap and min_heap[0][1] < left:
                heapq.heappop(min_heap)
            
            # Shrink window if difference exceeds limit
            while max_heap and min_heap and -max_heap[0][0] - min_heap[0][0] > limit:
                left += 1
                while max_heap and max_heap[0][1] < left:
                    heapq.heappop(max_heap)
                while min_heap and min_heap[0][1] < left:
                    heapq.heappop(min_heap)
            
            max_length = max(max_length, right - left + 1)
        
        return max_length
    
    def longestSubarray_brute_force(self, nums: List[int], limit: int) -> int:
        """
        Approach 3: Brute Force
        
        Check all possible subarrays.
        
        Time: O(n³), Space: O(1)
        """
        n = len(nums)
        max_length = 0
        
        for i in range(n):
            for j in range(i, n):
                # Check if subarray nums[i:j+1] satisfies condition
                subarray = nums[i:j+1]
                if max(subarray) - min(subarray) <= limit:
                    max_length = max(max_length, j - i + 1)
                else:
                    break  # No need to extend further
        
        return max_length
    
    def longestSubarray_optimized_brute_force(self, nums: List[int], limit: int) -> int:
        """
        Approach 4: Optimized Brute Force
        
        Track min/max while expanding subarray.
        
        Time: O(n²), Space: O(1)
        """
        n = len(nums)
        max_length = 0
        
        for i in range(n):
            min_val = nums[i]
            max_val = nums[i]
            
            for j in range(i, n):
                min_val = min(min_val, nums[j])
                max_val = max(max_val, nums[j])
                
                if max_val - min_val <= limit:
                    max_length = max(max_length, j - i + 1)
                else:
                    break  # No need to extend further
        
        return max_length
    
    def longestSubarray_segment_tree(self, nums: List[int], limit: int) -> int:
        """
        Approach 5: Segment Tree for Range Min/Max
        
        Use segment tree for efficient range queries.
        
        Time: O(n² log n), Space: O(n)
        """
        n = len(nums)
        
        # Build segment trees for min and max
        min_tree = [0] * (4 * n)
        max_tree = [0] * (4 * n)
        
        def build_min(node: int, start: int, end: int) -> None:
            if start == end:
                min_tree[node] = nums[start]
            else:
                mid = (start + end) // 2
                build_min(2 * node, start, mid)
                build_min(2 * node + 1, mid + 1, end)
                min_tree[node] = min(min_tree[2 * node], min_tree[2 * node + 1])
        
        def build_max(node: int, start: int, end: int) -> None:
            if start == end:
                max_tree[node] = nums[start]
            else:
                mid = (start + end) // 2
                build_max(2 * node, start, mid)
                build_max(2 * node + 1, mid + 1, end)
                max_tree[node] = max(max_tree[2 * node], max_tree[2 * node + 1])
        
        def query_min(node: int, start: int, end: int, l: int, r: int) -> int:
            if r < start or end < l:
                return float('inf')
            if l <= start and end <= r:
                return min_tree[node]
            
            mid = (start + end) // 2
            left_min = query_min(2 * node, start, mid, l, r)
            right_min = query_min(2 * node + 1, mid + 1, end, l, r)
            return min(left_min, right_min)
        
        def query_max(node: int, start: int, end: int, l: int, r: int) -> int:
            if r < start or end < l:
                return float('-inf')
            if l <= start and end <= r:
                return max_tree[node]
            
            mid = (start + end) // 2
            left_max = query_max(2 * node, start, mid, l, r)
            right_max = query_max(2 * node + 1, mid + 1, end, l, r)
            return max(left_max, right_max)
        
        build_min(1, 0, n - 1)
        build_max(1, 0, n - 1)
        
        max_length = 0
        
        for i in range(n):
            for j in range(i, n):
                min_val = query_min(1, 0, n - 1, i, j)
                max_val = query_max(1, 0, n - 1, i, j)
                
                if max_val - min_val <= limit:
                    max_length = max(max_length, j - i + 1)
                else:
                    break
        
        return max_length
    
    def longestSubarray_sliding_window_binary_search(self, nums: List[int], limit: int) -> int:
        """
        Approach 6: Sliding Window with Binary Search
        
        Use binary search on answer with sliding window validation.
        
        Time: O(n log n), Space: O(n)
        """
        def can_achieve_length(target_length: int) -> bool:
            """Check if we can achieve a subarray of target_length"""
            for i in range(len(nums) - target_length + 1):
                subarray = nums[i:i + target_length]
                if max(subarray) - min(subarray) <= limit:
                    return True
            return False
        
        left, right = 1, len(nums)
        result = 0
        
        while left <= right:
            mid = (left + right) // 2
            
            if can_achieve_length(mid):
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        
        return result
    
    def longestSubarray_multiset_simulation(self, nums: List[int], limit: int) -> int:
        """
        Approach 7: Multiset Simulation with Sorted List
        
        Simulate multiset behavior for tracking min/max.
        
        Time: O(n² log n), Space: O(n)
        """
        from bisect import bisect_left, insort
        
        left = 0
        max_length = 0
        
        for right in range(len(nums)):
            # For each right, find the leftmost valid left
            window = []
            
            for i in range(right, -1, -1):
                insort(window, nums[i])
                
                if window[-1] - window[0] <= limit:
                    max_length = max(max_length, right - i + 1)
                else:
                    break
        
        return max_length


def test_longest_subarray_with_limit():
    """Test longest subarray with limit algorithms"""
    solver = LongestSubarrayWithLimit()
    
    test_cases = [
        ([8,2,4,7], 4, 2, "Example 1"),
        ([10,1,2,4,7,2], 5, 4, "Example 2"),
        ([4,2,2,2,4,4,2,2], 0, 3, "Example 3"),
        ([1,5,6,7,8,10,6,5,6], 4, 5, "Complex case"),
        ([1], 0, 1, "Single element"),
        ([1,1,1,1], 0, 4, "All same elements"),
        ([1,2,3,4,5], 1, 2, "Increasing sequence"),
        ([5,4,3,2,1], 1, 2, "Decreasing sequence"),
        ([1,10,1,10], 9, 2, "Alternating values"),
    ]
    
    algorithms = [
        ("Two Deques", solver.longestSubarray_two_deques),
        ("Two Heaps", solver.longestSubarray_two_heaps),
        ("Brute Force", solver.longestSubarray_brute_force),
        ("Optimized Brute Force", solver.longestSubarray_optimized_brute_force),
        ("Binary Search", solver.longestSubarray_sliding_window_binary_search),
    ]
    
    print("=== Testing Longest Subarray With Limit ===")
    
    for nums, limit, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"nums: {nums}, limit: {limit}")
        print(f"Expected: {expected}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(nums, limit)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:20} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:40]}")


def demonstrate_two_deques_approach():
    """Demonstrate two deques approach step by step"""
    print("\n=== Two Deques Approach Step-by-Step Demo ===")
    
    nums = [8, 2, 4, 7]
    limit = 4
    
    print(f"Array: {nums}, Limit: {limit}")
    
    max_deque = deque()
    min_deque = deque()
    left = 0
    max_length = 0
    
    for right in range(len(nums)):
        print(f"\nStep {right+1}: Processing nums[{right}] = {nums[right]}")
        
        # Update max deque
        while max_deque and nums[max_deque[-1]] <= nums[right]:
            removed = max_deque.pop()
            print(f"  Removed {removed} from max_deque (nums[{removed}]={nums[removed]} <= {nums[right]})")
        max_deque.append(right)
        
        # Update min deque
        while min_deque and nums[min_deque[-1]] >= nums[right]:
            removed = min_deque.pop()
            print(f"  Removed {removed} from min_deque (nums[{removed}]={nums[removed]} >= {nums[right]})")
        min_deque.append(right)
        
        print(f"  Max deque: {list(max_deque)} -> values: {[nums[i] for i in max_deque]}")
        print(f"  Min deque: {list(min_deque)} -> values: {[nums[i] for i in min_deque]}")
        
        # Check if window is valid
        current_max = nums[max_deque[0]]
        current_min = nums[min_deque[0]]
        diff = current_max - current_min
        
        print(f"  Current window: [{left}, {right}] = {nums[left:right+1]}")
        print(f"  Max: {current_max}, Min: {current_min}, Diff: {diff}")
        
        # Shrink window if needed
        while diff > limit:
            print(f"  Diff {diff} > limit {limit}, shrinking window")
            if max_deque[0] == left:
                max_deque.popleft()
                print(f"    Removed {left} from max_deque")
            if min_deque[0] == left:
                min_deque.popleft()
                print(f"    Removed {left} from min_deque")
            left += 1
            
            if max_deque and min_deque:
                current_max = nums[max_deque[0]]
                current_min = nums[min_deque[0]]
                diff = current_max - current_min
                print(f"    New window: [{left}, {right}], Max: {current_max}, Min: {current_min}, Diff: {diff}")
        
        current_length = right - left + 1
        max_length = max(max_length, current_length)
        print(f"  Valid window length: {current_length}, Max so far: {max_length}")
    
    print(f"\nFinal result: {max_length}")


def benchmark_longest_subarray():
    """Benchmark different approaches"""
    import time
    import random
    
    algorithms = [
        ("Two Deques", LongestSubarrayWithLimit().longestSubarray_two_deques),
        ("Two Heaps", LongestSubarrayWithLimit().longestSubarray_two_heaps),
        ("Optimized Brute Force", LongestSubarrayWithLimit().longestSubarray_optimized_brute_force),
    ]
    
    # Test with different array sizes
    test_configs = [
        (1000, 10),
        (5000, 50),
        (10000, 100),
    ]
    
    print("\n=== Longest Subarray Performance Benchmark ===")
    
    for array_size, limit in test_configs:
        print(f"\n--- Array Size: {array_size}, Limit: {limit} ---")
        
        # Generate random array
        nums = [random.randint(1, 200) for _ in range(array_size)]
        
        for alg_name, alg_func in algorithms:
            start_time = time.time()
            
            try:
                result = alg_func(nums, limit)
                end_time = time.time()
                print(f"{alg_name:20} | Time: {end_time - start_time:.4f}s | Result: {result}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:30]}")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    solver = LongestSubarrayWithLimit()
    
    edge_cases = [
        ([1], 0, 1, "Single element"),
        ([1], 5, 1, "Single element, large limit"),
        ([1, 1], 0, 2, "Two same elements"),
        ([1, 2], 0, 1, "Two different, limit 0"),
        ([1, 2], 1, 2, "Two different, limit 1"),
        ([1, 100], 99, 2, "Large difference, exact limit"),
        ([1, 100], 98, 1, "Large difference, limit too small"),
        ([5, 5, 5, 5], 0, 4, "All same elements"),
        ([1, 2, 1, 2], 1, 4, "Alternating within limit"),
        ([1, 10, 1, 10], 9, 2, "Alternating at limit"),
    ]
    
    for nums, limit, expected, description in edge_cases:
        try:
            result = solver.longestSubarray_two_deques(nums, limit)
            status = "✓" if result == expected else "✗"
            print(f"{description:30} | {status} | nums: {nums}, limit: {limit} -> {result}")
        except Exception as e:
            print(f"{description:30} | ERROR: {str(e)[:30]}")


def visualize_sliding_window():
    """Visualize sliding window behavior"""
    print("\n=== Sliding Window Visualization ===")
    
    nums = [10, 1, 2, 4, 7, 2]
    limit = 5
    
    print(f"Array: {nums}")
    print(f"Limit: {limit}")
    print()
    
    solver = LongestSubarrayWithLimit()
    
    # Manual step-through for visualization
    max_deque = deque()
    min_deque = deque()
    left = 0
    max_length = 0
    
    for right in range(len(nums)):
        # Update deques (simplified for visualization)
        while max_deque and nums[max_deque[-1]] <= nums[right]:
            max_deque.pop()
        max_deque.append(right)
        
        while min_deque and nums[min_deque[-1]] >= nums[right]:
            min_deque.pop()
        min_deque.append(right)
        
        # Shrink window if needed
        while nums[max_deque[0]] - nums[min_deque[0]] > limit:
            if max_deque[0] == left:
                max_deque.popleft()
            if min_deque[0] == left:
                min_deque.popleft()
            left += 1
        
        # Visualize current window
        vis = [' '] * len(nums)
        for j in range(left, right + 1):
            vis[j] = '█'
        
        vis_str = ''.join(vis)
        window = nums[left:right + 1]
        window_max = nums[max_deque[0]] if max_deque else 0
        window_min = nums[min_deque[0]] if min_deque else 0
        diff = window_max - window_min
        
        max_length = max(max_length, right - left + 1)
        
        print(f"Step {right+1}: {vis_str} -> {window}")
        print(f"         Max: {window_max}, Min: {window_min}, Diff: {diff}, Length: {right - left + 1}")
        print()
    
    print(f"Maximum valid subarray length: {max_length}")


def test_correctness():
    """Test correctness by comparing approaches"""
    print("\n=== Correctness Test ===")
    
    solver = LongestSubarrayWithLimit()
    
    # Generate test cases
    import random
    
    test_cases = [
        ([random.randint(1, 50) for _ in range(10)], random.randint(1, 20))
        for _ in range(5)
    ]
    
    algorithms = [
        ("Two Deques", solver.longestSubarray_two_deques),
        ("Two Heaps", solver.longestSubarray_two_heaps),
        ("Optimized Brute Force", solver.longestSubarray_optimized_brute_force),
    ]
    
    for i, (nums, limit) in enumerate(test_cases):
        print(f"\nTest case {i+1}: nums={nums}, limit={limit}")
        
        results = {}
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(nums, limit)
                results[alg_name] = result
                print(f"{alg_name:20} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:40]}")
        
        # Check consistency
        if results:
            first_result = list(results.values())[0]
            all_same = all(result == first_result for result in results.values())
            print(f"All algorithms agree: {'✓' if all_same else '✗'}")


def analyze_time_complexity():
    """Analyze time complexity of different approaches"""
    print("\n=== Time Complexity Analysis ===")
    
    approaches = [
        ("Two Deques", "O(n)", "O(n)", "Optimal sliding window"),
        ("Two Heaps", "O(n log n)", "O(n)", "Heap operations"),
        ("Brute Force", "O(n³)", "O(1)", "Check all subarrays"),
        ("Optimized Brute Force", "O(n²)", "O(1)", "Track min/max while expanding"),
        ("Segment Tree", "O(n² log n)", "O(n)", "Range queries"),
        ("Binary Search", "O(n² log n)", "O(n)", "Binary search on answer"),
    ]
    
    print(f"{'Approach':<25} | {'Time':<12} | {'Space':<8} | {'Notes'}")
    print("-" * 70)
    
    for approach, time_comp, space_comp, notes in approaches:
        print(f"{approach:<25} | {time_comp:<12} | {space_comp:<8} | {notes}")


if __name__ == "__main__":
    test_longest_subarray_with_limit()
    demonstrate_two_deques_approach()
    visualize_sliding_window()
    test_edge_cases()
    test_correctness()
    analyze_time_complexity()
    benchmark_longest_subarray()

"""
Longest Continuous Subarray With Absolute Diff demonstrates advanced
sliding window techniques including two deques, heap-based solutions,
segment trees, and binary search approaches with comprehensive analysis.
"""
