"""
239. Sliding Window Maximum - Multiple Approaches
Difficulty: Hard

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
        
        Use deque to maintain decreasing order of elements.
        
        Time: O(n), Space: O(k)
        """
        if not nums or k == 0:
            return []
        
        dq = deque()  # Store indices
        result = []
        
        for i in range(len(nums)):
            # Remove indices that are out of current window
            while dq and dq[0] <= i - k:
                dq.popleft()
            
            # Remove indices whose corresponding values are smaller than current
            while dq and nums[dq[-1]] <= nums[i]:
                dq.pop()
            
            dq.append(i)
            
            # Add to result when window is complete
            if i >= k - 1:
                result.append(nums[dq[0]])
        
        return result
    
    def maxSlidingWindow_heap_approach(self, nums: List[int], k: int) -> List[int]:
        """
        Approach 2: Max Heap with Lazy Deletion
        
        Use max heap to track maximum elements.
        
        Time: O(n log n), Space: O(n)
        """
        if not nums or k == 0:
            return []
        
        # Use negative values for max heap
        heap = []
        result = []
        
        # Initialize heap with first k elements
        for i in range(k):
            heapq.heappush(heap, (-nums[i], i))
        
        result.append(-heap[0][0])
        
        # Process remaining elements
        for i in range(k, len(nums)):
            heapq.heappush(heap, (-nums[i], i))
            
            # Remove elements outside current window
            while heap and heap[0][1] <= i - k:
                heapq.heappop(heap)
            
            result.append(-heap[0][0])
        
        return result
    
    def maxSlidingWindow_brute_force(self, nums: List[int], k: int) -> List[int]:
        """
        Approach 3: Brute Force
        
        Check maximum in each window directly.
        
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
        Approach 4: Segment Tree
        
        Use segment tree for range maximum queries.
        
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
        
        def query(node: int, start: int, end: int, l: int, r: int) -> int:
            if r < start or end < l:
                return float('-inf')
            if l <= start and end <= r:
                return tree[node]
            
            mid = (start + end) // 2
            left_max = query(2 * node, start, mid, l, r)
            right_max = query(2 * node + 1, mid + 1, end, l, r)
            return max(left_max, right_max)
        
        build(1, 0, n - 1)
        
        result = []
        for i in range(n - k + 1):
            window_max = query(1, 0, n - 1, i, i + k - 1)
            result.append(window_max)
        
        return result
    
    def maxSlidingWindow_sparse_table(self, nums: List[int], k: int) -> List[int]:
        """
        Approach 5: Sparse Table
        
        Use sparse table for O(1) range maximum queries.
        
        Time: O(n log n) preprocessing + O(n), Space: O(n log n)
        """
        if not nums or k == 0:
            return []
        
        n = len(nums)
        
        # Build sparse table
        log_n = n.bit_length()
        st = [[0] * log_n for _ in range(n)]
        
        # Initialize for intervals of length 1
        for i in range(n):
            st[i][0] = nums[i]
        
        # Build sparse table
        j = 1
        while (1 << j) <= n:
            i = 0
            while (i + (1 << j) - 1) < n:
                st[i][j] = max(st[i][j - 1], st[i + (1 << (j - 1))][j - 1])
                i += 1
            j += 1
        
        def query_max(l: int, r: int) -> int:
            length = r - l + 1
            k_log = (length).bit_length() - 1
            return max(st[l][k_log], st[r - (1 << k_log) + 1][k_log])
        
        result = []
        for i in range(n - k + 1):
            window_max = query_max(i, i + k - 1)
            result.append(window_max)
        
        return result
    
    def maxSlidingWindow_divide_conquer(self, nums: List[int], k: int) -> List[int]:
        """
        Approach 6: Divide and Conquer with Preprocessing
        
        Preprocess left and right maximums for efficient queries.
        
        Time: O(n), Space: O(n)
        """
        if not nums or k == 0:
            return []
        
        n = len(nums)
        
        # Precompute left maximums
        left_max = [0] * n
        left_max[0] = nums[0]
        for i in range(1, n):
            if i % k == 0:
                left_max[i] = nums[i]
            else:
                left_max[i] = max(left_max[i - 1], nums[i])
        
        # Precompute right maximums
        right_max = [0] * n
        right_max[n - 1] = nums[n - 1]
        for i in range(n - 2, -1, -1):
            if (i + 1) % k == 0:
                right_max[i] = nums[i]
            else:
                right_max[i] = max(right_max[i + 1], nums[i])
        
        result = []
        for i in range(n - k + 1):
            # Maximum in window [i, i + k - 1]
            window_max = max(right_max[i], left_max[i + k - 1])
            result.append(window_max)
        
        return result
    
    def maxSlidingWindow_stack_approach(self, nums: List[int], k: int) -> List[int]:
        """
        Approach 7: Stack-based Approach
        
        Use stack to maintain potential maximums.
        
        Time: O(n), Space: O(k)
        """
        if not nums or k == 0:
            return []
        
        stack = []  # Store (value, count) pairs
        result = []
        
        # Process first window
        for i in range(k):
            count = 1
            while stack and stack[-1][0] <= nums[i]:
                count += stack.pop()[1]
            stack.append((nums[i], count))
        
        result.append(stack[0][0])
        
        # Process remaining elements
        for i in range(k, len(nums)):
            # Remove element going out of window
            if stack and stack[0][1] == k:
                stack[0] = (stack[0][0], stack[0][1] - 1)
                if stack[0][1] == 0:
                    stack.pop(0)
            else:
                for j in range(len(stack)):
                    stack[j] = (stack[j][0], stack[j][1] - 1)
                stack = [(val, cnt) for val, cnt in stack if cnt > 0]
            
            # Add new element
            count = 1
            while stack and stack[-1][0] <= nums[i]:
                count += stack.pop()[1]
            stack.append((nums[i], count))
            
            result.append(stack[0][0])
        
        return result


def test_sliding_window_maximum():
    """Test sliding window maximum algorithms"""
    solver = SlidingWindowMaximum()
    
    test_cases = [
        ([1,3,-1,-3,5,3,6,7], 3, [3,3,5,5,6,7], "Example 1"),
        ([1], 1, [1], "Single element"),
        ([1,-1], 1, [1,-1], "Window size 1"),
        ([9,11], 2, [11], "Two elements"),
        ([4,-2], 2, [4], "Negative numbers"),
        ([1,3,1,2,0,5], 3, [3,3,2,5], "Mixed values"),
        ([7,2,4], 2, [7,4], "Small window"),
        ([1,2,3,4,5], 3, [3,4,5], "Increasing sequence"),
        ([5,4,3,2,1], 3, [5,4,3], "Decreasing sequence"),
    ]
    
    algorithms = [
        ("Deque Approach", solver.maxSlidingWindow_deque_approach),
        ("Heap Approach", solver.maxSlidingWindow_heap_approach),
        ("Brute Force", solver.maxSlidingWindow_brute_force),
        ("Segment Tree", solver.maxSlidingWindow_segment_tree),
        ("Sparse Table", solver.maxSlidingWindow_sparse_table),
        ("Divide Conquer", solver.maxSlidingWindow_divide_conquer),
    ]
    
    print("=== Testing Sliding Window Maximum ===")
    
    for nums, k, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"nums: {nums}, k: {k}")
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
    
    print(f"Array: {nums}, Window size: {k}")
    
    dq = deque()
    result = []
    
    for i in range(len(nums)):
        print(f"\nStep {i+1}: Processing nums[{i}] = {nums[i]}")
        
        # Remove indices out of window
        while dq and dq[0] <= i - k:
            removed = dq.popleft()
            print(f"  Removed index {removed} (out of window)")
        
        # Remove smaller elements
        while dq and nums[dq[-1]] <= nums[i]:
            removed = dq.pop()
            print(f"  Removed index {removed} (nums[{removed}]={nums[removed]} <= {nums[i]})")
        
        dq.append(i)
        print(f"  Added index {i}")
        print(f"  Deque indices: {list(dq)}")
        print(f"  Deque values: {[nums[idx] for idx in dq]}")
        
        if i >= k - 1:
            max_val = nums[dq[0]]
            result.append(max_val)
            window_start = i - k + 1
            window_end = i
            print(f"  Window [{window_start}:{window_end+1}] = {nums[window_start:window_end+1]}")
            print(f"  Maximum: {max_val}")
    
    print(f"\nFinal result: {result}")


def benchmark_sliding_window_maximum():
    """Benchmark different approaches"""
    import time
    import random
    
    algorithms = [
        ("Deque Approach", SlidingWindowMaximum().maxSlidingWindow_deque_approach),
        ("Heap Approach", SlidingWindowMaximum().maxSlidingWindow_heap_approach),
        ("Divide Conquer", SlidingWindowMaximum().maxSlidingWindow_divide_conquer),
    ]
    
    # Test with different array sizes and window sizes
    test_configs = [
        (1000, 10),
        (5000, 50),
        (10000, 100),
    ]
    
    print("\n=== Sliding Window Maximum Performance Benchmark ===")
    
    for array_size, window_size in test_configs:
        print(f"\n--- Array Size: {array_size}, Window Size: {window_size} ---")
        
        # Generate random array
        nums = [random.randint(1, 1000) for _ in range(array_size)]
        
        for alg_name, alg_func in algorithms:
            start_time = time.time()
            
            try:
                result = alg_func(nums, window_size)
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
        ([1, 2], 2, [2], "Two elements, k=2"),
        ([5, 5, 5], 2, [5, 5], "All same elements"),
        ([-1, -2, -3], 2, [-1, -2], "All negative"),
        ([1, 2, 3, 4, 5], 5, [5], "k equals array length"),
        ([10, 9, 8, 7, 6], 3, [10, 9, 8], "Strictly decreasing"),
        ([1, 2, 3, 4, 5], 3, [3, 4, 5], "Strictly increasing"),
    ]
    
    for nums, k, expected, description in edge_cases:
        try:
            result = solver.maxSlidingWindow_deque_approach(nums, k)
            status = "✓" if result == expected else "✗"
            print(f"{description:25} | {status} | nums: {nums}, k: {k} -> {result}")
        except Exception as e:
            print(f"{description:25} | ERROR: {str(e)[:30]}")


def visualize_window_movement():
    """Visualize how the sliding window moves"""
    print("\n=== Sliding Window Movement Visualization ===")
    
    nums = [4, 3, 8, 2, 5, 1, 9, 6]
    k = 3
    
    print(f"Array: {nums}")
    print(f"Window size: {k}")
    print()
    
    solver = SlidingWindowMaximum()
    result = solver.maxSlidingWindow_deque_approach(nums, k)
    
    for i in range(len(nums) - k + 1):
        window = nums[i:i + k]
        window_max = result[i]
        
        # Create visualization
        vis = [' '] * len(nums)
        for j in range(i, i + k):
            vis[j] = '█'
        
        vis_str = ''.join(vis)
        print(f"Window {i+1}: {vis_str} -> {window} -> max: {window_max}")


def compare_time_complexity():
    """Compare time complexity of different approaches"""
    print("\n=== Time Complexity Comparison ===")
    
    approaches = [
        ("Deque (Monotonic)", "O(n)", "O(k)", "Optimal for most cases"),
        ("Max Heap", "O(n log n)", "O(n)", "Good for small arrays"),
        ("Brute Force", "O(n * k)", "O(1)", "Simple but slow"),
        ("Segment Tree", "O(n log n)", "O(n)", "Good for multiple queries"),
        ("Sparse Table", "O(n log n)", "O(n log n)", "O(1) queries after preprocessing"),
        ("Divide & Conquer", "O(n)", "O(n)", "Optimal with preprocessing"),
    ]
    
    print(f"{'Approach':<20} | {'Time':<12} | {'Space':<12} | {'Notes'}")
    print("-" * 70)
    
    for approach, time_comp, space_comp, notes in approaches:
        print(f"{approach:<20} | {time_comp:<12} | {space_comp:<12} | {notes}")


def stress_test():
    """Stress test with large input"""
    print("\n=== Stress Test ===")
    
    import random
    
    # Large array
    n = 50000
    k = 1000
    
    print(f"Generating array of size {n} with window size {k}...")
    
    nums = [random.randint(1, 10000) for _ in range(n)]
    
    solver = SlidingWindowMaximum()
    
    # Test deque approach (most efficient)
    start_time = time.time()
    result = solver.maxSlidingWindow_deque_approach(nums, k)
    end_time = time.time()
    
    print(f"Deque approach completed in {end_time - start_time:.4f}s")
    print(f"Result length: {len(result)}")
    print(f"First 10 results: {result[:10]}")
    print(f"Last 10 results: {result[-10:]}")


def test_correctness():
    """Test correctness by comparing all approaches"""
    print("\n=== Correctness Test ===")
    
    solver = SlidingWindowMaximum()
    
    # Generate test case
    nums = [random.randint(1, 100) for _ in range(20)]
    k = 5
    
    print(f"Test array: {nums}")
    print(f"Window size: {k}")
    
    algorithms = [
        ("Deque", solver.maxSlidingWindow_deque_approach),
        ("Heap", solver.maxSlidingWindow_heap_approach),
        ("Brute Force", solver.maxSlidingWindow_brute_force),
        ("Divide Conquer", solver.maxSlidingWindow_divide_conquer),
    ]
    
    results = {}
    
    for alg_name, alg_func in algorithms:
        try:
            result = alg_func(nums, k)
            results[alg_name] = result
            print(f"{alg_name:15} | Result: {result}")
        except Exception as e:
            print(f"{alg_name:15} | ERROR: {str(e)[:40]}")
    
    # Check if all results are the same
    if results:
        first_result = list(results.values())[0]
        all_same = all(result == first_result for result in results.values())
        print(f"\nAll algorithms agree: {'✓' if all_same else '✗'}")


if __name__ == "__main__":
    test_sliding_window_maximum()
    demonstrate_deque_approach()
    visualize_window_movement()
    test_edge_cases()
    compare_time_complexity()
    test_correctness()
    benchmark_sliding_window_maximum()
    stress_test()

"""
Sliding Window Maximum demonstrates multiple approaches including
monotonic deque, heap-based solutions, segment trees, sparse tables,
and divide-and-conquer techniques with comprehensive analysis.
"""
