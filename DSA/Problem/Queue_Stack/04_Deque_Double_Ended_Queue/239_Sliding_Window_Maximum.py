"""
239. Sliding Window Maximum - Multiple Approaches
Difficulty: Hard (but categorized as Medium in deque context)

You are given an array of integers nums, there is a sliding window of size k which is moving from the very left of the array to the very right. You can only see the k numbers in the window. Each time the sliding window moves right by one position.

Return the max sliding window.
"""

from typing import List, Deque
from collections import deque
import heapq

class SlidingWindowMaximum:
    """Multiple approaches to find sliding window maximum"""
    
    def maxSlidingWindow_deque_approach(self, nums: List[int], k: int) -> List[int]:
        """
        Approach 1: Monotonic Deque (Optimal)
        
        Use deque to maintain indices in decreasing order of values.
        
        Time: O(n), Space: O(k)
        """
        if not nums or k == 0:
            return []
        
        dq: Deque[int] = deque()  # Store indices
        result = []
        
        for i in range(len(nums)):
            # Remove indices outside current window
            while dq and dq[0] <= i - k:
                dq.popleft()
            
            # Remove indices with smaller values than current
            while dq and nums[dq[-1]] < nums[i]:
                dq.pop()
            
            dq.append(i)
            
            # Add maximum to result if window is complete
            if i >= k - 1:
                result.append(nums[dq[0]])
        
        return result
    
    def maxSlidingWindow_heap_approach(self, nums: List[int], k: int) -> List[int]:
        """
        Approach 2: Max Heap with Lazy Deletion
        
        Use max heap to track maximum values with lazy deletion.
        
        Time: O(n log n), Space: O(k)
        """
        if not nums or k == 0:
            return []
        
        # Max heap (negate values for max heap using min heap)
        heap = []
        result = []
        
        # Initialize heap with first k elements
        for i in range(k):
            heapq.heappush(heap, (-nums[i], i))
        
        result.append(-heap[0][0])  # First maximum
        
        # Process remaining elements
        for i in range(k, len(nums)):
            heapq.heappush(heap, (-nums[i], i))
            
            # Remove elements outside current window (lazy deletion)
            while heap and heap[0][1] <= i - k:
                heapq.heappop(heap)
            
            result.append(-heap[0][0])
        
        return result
    
    def maxSlidingWindow_brute_force(self, nums: List[int], k: int) -> List[int]:
        """
        Approach 3: Brute Force
        
        For each window, find maximum by scanning all elements.
        
        Time: O(n * k), Space: O(1)
        """
        if not nums or k == 0:
            return []
        
        result = []
        
        for i in range(len(nums) - k + 1):
            window_max = max(nums[i:i + k])
            result.append(window_max)
        
        return result
    
    def maxSlidingWindow_segment_tree(self, nums: List[int], k: int) -> List[int]:
        """
        Approach 4: Segment Tree for Range Maximum Query
        
        Build segment tree and query maximum for each window.
        
        Time: O(n log n), Space: O(n)
        """
        if not nums or k == 0:
            return []
        
        n = len(nums)
        
        # Build segment tree
        tree = [0] * (4 * n)
        
        def build(node: int, start: int, end: int) -> None:
            if start == end:
                tree[node] = nums[start]
            else:
                mid = (start + end) // 2
                build(2 * node, start, mid)
                build(2 * node + 1, mid + 1, end)
                tree[node] = max(tree[2 * node], tree[2 * node + 1])
        
        def query_max(node: int, start: int, end: int, l: int, r: int) -> int:
            if r < start or end < l:
                return float('-inf')
            if l <= start and end <= r:
                return tree[node]
            
            mid = (start + end) // 2
            left_max = query_max(2 * node, start, mid, l, r)
            right_max = query_max(2 * node + 1, mid + 1, end, l, r)
            return max(left_max, right_max)
        
        build(1, 0, n - 1)
        
        result = []
        for i in range(n - k + 1):
            window_max = query_max(1, 0, n - 1, i, i + k - 1)
            result.append(window_max)
        
        return result
    
    def maxSlidingWindow_sparse_table(self, nums: List[int], k: int) -> List[int]:
        """
        Approach 5: Sparse Table for Range Maximum Query
        
        Precompute sparse table for O(1) range maximum queries.
        
        Time: O(n log n) preprocessing + O(n), Space: O(n log n)
        """
        if not nums or k == 0:
            return []
        
        n = len(nums)
        
        # Build sparse table
        log_n = n.bit_length()
        sparse_table = [[0] * log_n for _ in range(n)]
        
        # Initialize for length 1
        for i in range(n):
            sparse_table[i][0] = nums[i]
        
        # Fill sparse table
        j = 1
        while (1 << j) <= n:
            i = 0
            while (i + (1 << j) - 1) < n:
                sparse_table[i][j] = max(sparse_table[i][j-1], 
                                        sparse_table[i + (1 << (j-1))][j-1])
                i += 1
            j += 1
        
        def query_max_sparse(l: int, r: int) -> int:
            length = r - l + 1
            k = length.bit_length() - 1
            return max(sparse_table[l][k], sparse_table[r - (1 << k) + 1][k])
        
        result = []
        for i in range(n - k + 1):
            window_max = query_max_sparse(i, i + k - 1)
            result.append(window_max)
        
        return result
    
    def maxSlidingWindow_divide_conquer(self, nums: List[int], k: int) -> List[int]:
        """
        Approach 6: Divide and Conquer
        
        Use divide and conquer to find maximum in each window.
        
        Time: O(n log n), Space: O(log n)
        """
        if not nums or k == 0:
            return []
        
        def find_max(left: int, right: int) -> int:
            if left == right:
                return nums[left]
            
            mid = (left + right) // 2
            left_max = find_max(left, mid)
            right_max = find_max(mid + 1, right)
            return max(left_max, right_max)
        
        result = []
        for i in range(len(nums) - k + 1):
            window_max = find_max(i, i + k - 1)
            result.append(window_max)
        
        return result
    
    def maxSlidingWindow_multiset_simulation(self, nums: List[int], k: int) -> List[int]:
        """
        Approach 7: Multiset Simulation using Sorted List
        
        Simulate multiset to maintain sorted window elements.
        
        Time: O(n * k), Space: O(k)
        """
        if not nums or k == 0:
            return []
        
        from bisect import bisect_left, insort
        
        window = []
        result = []
        
        # Initialize first window
        for i in range(k):
            insort(window, nums[i])
        
        result.append(window[-1])  # Maximum is last element
        
        # Slide window
        for i in range(k, len(nums)):
            # Remove leftmost element
            old_val = nums[i - k]
            idx = bisect_left(window, old_val)
            window.pop(idx)
            
            # Add new element
            insort(window, nums[i])
            
            result.append(window[-1])
        
        return result


def test_sliding_window_maximum():
    """Test sliding window maximum algorithms"""
    solver = SlidingWindowMaximum()
    
    test_cases = [
        ([1,3,-1,-3,5,3,6,7], 3, [3,3,5,5,6,7], "Example 1"),
        ([1], 1, [1], "Single element"),
        ([1,-1], 1, [1,-1], "Two elements, k=1"),
        ([9,11], 2, [11], "Two elements, k=2"),
        ([4,-2], 2, [4], "Two elements with negative"),
        ([1,3,1,2,0,5], 3, [3,3,2,5], "Mixed values"),
        ([7,2,4], 2, [7,4], "Small window"),
        ([1,2,3,4,5], 3, [3,4,5], "Increasing sequence"),
        ([5,4,3,2,1], 3, [5,4,3], "Decreasing sequence"),
        ([1,1,1,1,1], 3, [1,1,1], "All same values"),
    ]
    
    algorithms = [
        ("Deque Approach", solver.maxSlidingWindow_deque_approach),
        ("Heap Approach", solver.maxSlidingWindow_heap_approach),
        ("Brute Force", solver.maxSlidingWindow_brute_force),
        ("Segment Tree", solver.maxSlidingWindow_segment_tree),
        ("Sparse Table", solver.maxSlidingWindow_sparse_table),
        ("Divide Conquer", solver.maxSlidingWindow_divide_conquer),
        ("Multiset Simulation", solver.maxSlidingWindow_multiset_simulation),
    ]
    
    print("=== Testing Sliding Window Maximum ===")
    
    for nums, k, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Array: {nums}")
        print(f"Window size: {k}")
        print(f"Expected: {expected}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(nums, k)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:20} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:40]}")


def demonstrate_deque_approach():
    """Demonstrate deque approach step by step"""
    print("\n=== Deque Approach Step-by-Step Demo ===")
    
    nums = [1, 3, -1, -3, 5, 3, 6, 7]
    k = 3
    
    print(f"Array: {nums}")
    print(f"Window size: {k}")
    
    dq = deque()
    result = []
    
    print("\nProcessing each element:")
    
    for i in range(len(nums)):
        print(f"\nStep {i+1}: Processing nums[{i}] = {nums[i]}")
        
        # Remove indices outside current window
        while dq and dq[0] <= i - k:
            removed_idx = dq.popleft()
            print(f"  Removed index {removed_idx} (outside window)")
        
        # Remove indices with smaller values
        removed_smaller = []
        while dq and nums[dq[-1]] < nums[i]:
            removed_idx = dq.pop()
            removed_smaller.append(f"{removed_idx}({nums[removed_idx]})")
        
        if removed_smaller:
            print(f"  Removed smaller values: {removed_smaller}")
        
        dq.append(i)
        print(f"  Added index {i}")
        print(f"  Deque indices: {list(dq)}")
        print(f"  Deque values: {[nums[idx] for idx in dq]}")
        
        # Add to result if window is complete
        if i >= k - 1:
            max_val = nums[dq[0]]
            result.append(max_val)
            print(f"  Window [{i-k+1}:{i+1}] maximum: {max_val}")
    
    print(f"\nFinal result: {result}")


def visualize_sliding_window():
    """Visualize sliding window process"""
    print("\n=== Sliding Window Visualization ===")
    
    nums = [1, 3, -1, -3, 5, 3, 6, 7]
    k = 3
    
    print(f"Array: {nums}")
    print(f"Window size: {k}")
    
    print("\nSliding windows:")
    
    for i in range(len(nums) - k + 1):
        window = nums[i:i + k]
        window_max = max(window)
        
        # Create visual representation
        visual = []
        for j, val in enumerate(nums):
            if i <= j < i + k:
                if val == window_max:
                    visual.append(f"[{val}]")  # Highlight maximum
                else:
                    visual.append(f" {val} ")
            else:
                visual.append(f" {val} ")
        
        print(f"Window {i+1}: {' '.join(visual)} -> max = {window_max}")


def demonstrate_monotonic_property():
    """Demonstrate monotonic deque property"""
    print("\n=== Monotonic Deque Property Demonstration ===")
    
    nums = [4, 3, 5, 1, 2, 6]
    k = 3
    
    print(f"Array: {nums}")
    print("Deque maintains indices in decreasing order of values")
    
    dq = deque()
    
    for i, val in enumerate(nums):
        print(f"\nProcessing nums[{i}] = {val}")
        
        # Show what gets removed and why
        removed = []
        while dq and nums[dq[-1]] < val:
            removed_idx = dq.pop()
            removed.append(f"idx {removed_idx} (val {nums[removed_idx]})")
        
        if removed:
            print(f"  Removed: {removed} (smaller than {val})")
        
        dq.append(i)
        
        # Show current deque state
        deque_info = [(idx, nums[idx]) for idx in dq]
        print(f"  Deque: {deque_info}")
        
        # Verify monotonic property
        values = [nums[idx] for idx in dq]
        is_decreasing = all(values[j] >= values[j+1] for j in range(len(values)-1))
        print(f"  Monotonic (decreasing): {'✓' if is_decreasing else '✗'}")


def benchmark_sliding_window_maximum():
    """Benchmark different approaches"""
    import time
    import random
    
    algorithms = [
        ("Deque Approach", SlidingWindowMaximum().maxSlidingWindow_deque_approach),
        ("Heap Approach", SlidingWindowMaximum().maxSlidingWindow_heap_approach),
        ("Brute Force", SlidingWindowMaximum().maxSlidingWindow_brute_force),
        ("Segment Tree", SlidingWindowMaximum().maxSlidingWindow_segment_tree),
    ]
    
    # Test with different array sizes
    test_sizes = [(1000, 10), (5000, 50), (10000, 100)]
    
    print("\n=== Sliding Window Maximum Performance Benchmark ===")
    
    for size, k in test_sizes:
        print(f"\n--- Array Size: {size}, Window Size: {k} ---")
        
        # Generate random array
        nums = [random.randint(1, 1000) for _ in range(size)]
        
        for alg_name, alg_func in algorithms:
            start_time = time.time()
            
            try:
                result = alg_func(nums, k)
                end_time = time.time()
                print(f"{alg_name:20} | Time: {end_time - start_time:.4f}s | Result length: {len(result)}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:30]}")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    solver = SlidingWindowMaximum()
    
    edge_cases = [
        ([], 1, [], "Empty array"),
        ([1], 1, [1], "Single element, k=1"),
        ([1, 2], 1, [1, 2], "Two elements, k=1"),
        ([1, 2], 2, [2], "Two elements, k=2"),
        ([5], 2, [], "Single element, k > length"),
        ([-1, -2, -3], 2, [-1, -2], "All negative numbers"),
        ([0, 0, 0], 2, [0, 0], "All zeros"),
        ([1, 1, 1, 1], 3, [1, 1], "All same positive"),
        ([-5, -5, -5], 2, [-5, -5], "All same negative"),
    ]
    
    for nums, k, expected, description in edge_cases:
        try:
            result = solver.maxSlidingWindow_deque_approach(nums, k)
            status = "✓" if result == expected else "✗"
            print(f"{description:25} | {status} | nums: {nums}, k: {k} -> {result}")
        except Exception as e:
            print(f"{description:25} | ERROR: {str(e)[:30]}")


def compare_approaches():
    """Compare different approaches"""
    print("\n=== Approach Comparison ===")
    
    test_cases = [
        ([1, 3, -1, -3, 5, 3, 6, 7], 3),
        ([1, 2, 3, 4, 5], 3),
        ([5, 4, 3, 2, 1], 3),
        ([1, 1, 1, 1], 2),
    ]
    
    solver = SlidingWindowMaximum()
    
    approaches = [
        ("Deque", solver.maxSlidingWindow_deque_approach),
        ("Heap", solver.maxSlidingWindow_heap_approach),
        ("Brute Force", solver.maxSlidingWindow_brute_force),
        ("Segment Tree", solver.maxSlidingWindow_segment_tree),
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
        ("Deque Approach", "O(n)", "O(k)", "Each element added/removed once"),
        ("Heap Approach", "O(n log n)", "O(k)", "Heap operations with lazy deletion"),
        ("Brute Force", "O(n * k)", "O(1)", "Scan each window completely"),
        ("Segment Tree", "O(n log n)", "O(n)", "Build tree + range queries"),
        ("Sparse Table", "O(n log n)", "O(n log n)", "Precompute + O(1) queries"),
        ("Divide Conquer", "O(n * k log k)", "O(log k)", "Recursive maximum finding"),
        ("Multiset Simulation", "O(n * k)", "O(k)", "Sorted list operations"),
    ]
    
    print(f"{'Approach':<20} | {'Time':<15} | {'Space':<15} | {'Notes'}")
    print("-" * 80)
    
    for approach, time_comp, space_comp, notes in approaches:
        print(f"{approach:<20} | {time_comp:<15} | {space_comp:<15} | {notes}")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    # Application 1: Stock price analysis
    print("1. Stock Price Analysis - Maximum price in sliding window:")
    stock_prices = [100, 102, 98, 105, 110, 95, 108, 112, 99]
    window_days = 3
    
    solver = SlidingWindowMaximum()
    max_prices = solver.maxSlidingWindow_deque_approach(stock_prices, window_days)
    
    print(f"  Stock prices: {stock_prices}")
    print(f"  {window_days}-day window maximums: {max_prices}")
    
    for i, max_price in enumerate(max_prices):
        window_start = i
        window_end = i + window_days - 1
        print(f"    Days {window_start}-{window_end}: max price = ${max_price}")
    
    # Application 2: Network traffic monitoring
    print("\n2. Network Traffic Monitoring - Peak traffic in time windows:")
    traffic_data = [50, 75, 60, 90, 120, 80, 95, 110, 70]  # MB/s
    time_window = 4  # 4-second windows
    
    peak_traffic = solver.maxSlidingWindow_deque_approach(traffic_data, time_window)
    
    print(f"  Traffic data (MB/s): {traffic_data}")
    print(f"  {time_window}s window peaks: {peak_traffic}")
    
    # Application 3: Temperature monitoring
    print("\n3. Temperature Monitoring - Maximum temperature in hourly windows:")
    temperatures = [22, 25, 23, 28, 30, 26, 24, 29, 27]  # Celsius
    hour_window = 3
    
    max_temps = solver.maxSlidingWindow_deque_approach(temperatures, hour_window)
    
    print(f"  Temperatures (°C): {temperatures}")
    print(f"  {hour_window}-hour maximums: {max_temps}")


if __name__ == "__main__":
    test_sliding_window_maximum()
    demonstrate_deque_approach()
    visualize_sliding_window()
    demonstrate_monotonic_property()
    demonstrate_real_world_applications()
    test_edge_cases()
    compare_approaches()
    analyze_time_complexity()
    benchmark_sliding_window_maximum()

"""
Sliding Window Maximum demonstrates the power of monotonic deque for
efficient window-based optimization problems, including multiple approaches
and real-world applications in data analysis and monitoring systems.
"""
