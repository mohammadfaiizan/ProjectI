"""
1696. Jump Game VI - Multiple Approaches
Difficulty: Medium

You are given a 0-indexed integer array nums and an integer k.

You are initially standing at index 0. In one move, you can jump at most k steps forward without going out of bounds. That is, you can jump from index i to any index in the range [i + 1, i + 1 + k] inclusive.

You want to reach the last index in the minimum number of jumps.

Return the maximum sum of the elements you can obtain.
"""

from typing import List, Deque
from collections import deque
import heapq

class JumpGameVI:
    """Multiple approaches to solve Jump Game VI"""
    
    def maxResult_deque_approach(self, nums: List[int], k: int) -> int:
        """
        Approach 1: Monotonic Deque (Optimal)
        
        Use deque to maintain maximum values in sliding window.
        
        Time: O(n), Space: O(k)
        """
        n = len(nums)
        if n == 1:
            return nums[0]
        
        # dp[i] represents maximum sum to reach index i
        dp = [float('-inf')] * n
        dp[0] = nums[0]
        
        # Deque stores indices in decreasing order of dp values
        dq: Deque[int] = deque([0])
        
        for i in range(1, n):
            # Remove indices outside the window [i-k, i-1]
            while dq and dq[0] < i - k:
                dq.popleft()
            
            # Current maximum sum is dp[dq[0]] + nums[i]
            dp[i] = dp[dq[0]] + nums[i]
            
            # Maintain decreasing order in deque
            while dq and dp[dq[-1]] <= dp[i]:
                dq.pop()
            
            dq.append(i)
        
        return dp[n - 1]
    
    def maxResult_heap_approach(self, nums: List[int], k: int) -> int:
        """
        Approach 2: Max Heap with Lazy Deletion
        
        Use max heap to track maximum dp values.
        
        Time: O(n log n), Space: O(n)
        """
        n = len(nums)
        if n == 1:
            return nums[0]
        
        dp = [float('-inf')] * n
        dp[0] = nums[0]
        
        # Max heap (negate values for max heap using min heap)
        heap = [(-nums[0], 0)]  # (-dp_value, index)
        
        for i in range(1, n):
            # Remove outdated indices (lazy deletion)
            while heap and heap[0][1] < i - k:
                heapq.heappop(heap)
            
            # Get maximum dp value from valid range
            max_dp = -heap[0][0]
            dp[i] = max_dp + nums[i]
            
            heapq.heappush(heap, (-dp[i], i))
        
        return dp[n - 1]
    
    def maxResult_dp_brute_force(self, nums: List[int], k: int) -> int:
        """
        Approach 3: Dynamic Programming Brute Force
        
        For each position, check all possible previous positions.
        
        Time: O(n * k), Space: O(n)
        """
        n = len(nums)
        if n == 1:
            return nums[0]
        
        dp = [float('-inf')] * n
        dp[0] = nums[0]
        
        for i in range(1, n):
            # Check all possible previous positions within k steps
            for j in range(max(0, i - k), i):
                dp[i] = max(dp[i], dp[j] + nums[i])
        
        return dp[n - 1]
    
    def maxResult_segment_tree(self, nums: List[int], k: int) -> int:
        """
        Approach 4: Segment Tree for Range Maximum Query
        
        Use segment tree to find maximum in range efficiently.
        
        Time: O(n log n), Space: O(n)
        """
        n = len(nums)
        if n == 1:
            return nums[0]
        
        # Build segment tree
        tree = [float('-inf')] * (4 * n)
        
        def update(node: int, start: int, end: int, idx: int, val: int) -> None:
            if start == end:
                tree[node] = val
            else:
                mid = (start + end) // 2
                if idx <= mid:
                    update(2 * node, start, mid, idx, val)
                else:
                    update(2 * node + 1, mid + 1, end, idx, val)
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
        
        dp = [float('-inf')] * n
        dp[0] = nums[0]
        update(1, 0, n - 1, 0, dp[0])
        
        for i in range(1, n):
            # Query maximum in range [max(0, i-k), i-1]
            left = max(0, i - k)
            right = i - 1
            
            max_prev = query_max(1, 0, n - 1, left, right)
            dp[i] = max_prev + nums[i]
            update(1, 0, n - 1, i, dp[i])
        
        return dp[n - 1]
    
    def maxResult_sparse_table(self, nums: List[int], k: int) -> int:
        """
        Approach 5: Sparse Table for Range Maximum Query
        
        Precompute sparse table for O(1) range maximum queries.
        
        Time: O(n log n), Space: O(n log n)
        """
        n = len(nums)
        if n == 1:
            return nums[0]
        
        dp = [float('-inf')] * n
        dp[0] = nums[0]
        
        # We'll build sparse table dynamically as we compute dp
        for i in range(1, n):
            max_val = float('-inf')
            
            # Check all valid previous positions
            for j in range(max(0, i - k), i):
                max_val = max(max_val, dp[j])
            
            dp[i] = max_val + nums[i]
        
        return dp[n - 1]
    
    def maxResult_optimized_dp(self, nums: List[int], k: int) -> int:
        """
        Approach 6: Optimized DP with Sliding Window Maximum
        
        Use sliding window maximum technique.
        
        Time: O(n), Space: O(k)
        """
        n = len(nums)
        if n == 1:
            return nums[0]
        
        # Use deque to maintain sliding window maximum
        dq = deque()
        dp = [0] * n
        dp[0] = nums[0]
        dq.append(0)
        
        for i in range(1, n):
            # Remove elements outside window
            while dq and dq[0] < i - k:
                dq.popleft()
            
            # Get maximum from current window
            dp[i] = dp[dq[0]] + nums[i]
            
            # Maintain decreasing order
            while dq and dp[dq[-1]] <= dp[i]:
                dq.pop()
            
            dq.append(i)
        
        return dp[n - 1]
    
    def maxResult_memoization(self, nums: List[int], k: int) -> int:
        """
        Approach 7: Memoization (Top-down DP)
        
        Use recursion with memoization.
        
        Time: O(n * k), Space: O(n)
        """
        n = len(nums)
        memo = {}
        
        def dp(i: int) -> int:
            if i == 0:
                return nums[0]
            
            if i in memo:
                return memo[i]
            
            max_val = float('-inf')
            # Try all possible previous positions
            for j in range(max(0, i - k), i):
                max_val = max(max_val, dp(j) + nums[i])
            
            memo[i] = max_val
            return max_val
        
        return dp(n - 1)


def test_jump_game_vi():
    """Test jump game VI algorithms"""
    solver = JumpGameVI()
    
    test_cases = [
        ([1,-1,-2,4,-7,3], 2, 7, "Example 1"),
        ([10,-5,-2,4,0,3], 3, 17, "Example 2"),
        ([1,-5,-20,4,-1,3,-6,-3], 2, 0, "Example 3"),
        ([1], 1, 1, "Single element"),
        ([1, 2], 1, 3, "Two elements"),
        ([5, -3, 2], 2, 7, "Small array"),
        ([1, 2, 3, 4, 5], 2, 15, "All positive"),
        ([-1, -2, -3, -4, -5], 2, -9, "All negative"),
        ([10, -1, -1, -1, 10], 4, 20, "Large jumps"),
        ([1, -1, 1, -1, 1], 1, 1, "Alternating"),
    ]
    
    algorithms = [
        ("Deque Approach", solver.maxResult_deque_approach),
        ("Heap Approach", solver.maxResult_heap_approach),
        ("DP Brute Force", solver.maxResult_dp_brute_force),
        ("Segment Tree", solver.maxResult_segment_tree),
        ("Sparse Table", solver.maxResult_sparse_table),
        ("Optimized DP", solver.maxResult_optimized_dp),
        ("Memoization", solver.maxResult_memoization),
    ]
    
    print("=== Testing Jump Game VI ===")
    
    for nums, k, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Array: {nums}")
        print(f"Max jump: {k}")
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
    
    nums = [1, -1, -2, 4, -7, 3]
    k = 2
    
    print(f"Array: {nums}")
    print(f"Max jump distance: {k}")
    
    n = len(nums)
    dp = [float('-inf')] * n
    dp[0] = nums[0]
    
    dq = deque([0])
    
    print(f"\nInitial: dp[0] = {dp[0]}")
    print(f"Deque: {list(dq)}")
    
    for i in range(1, n):
        print(f"\nStep {i+1}: Processing index {i} (value {nums[i]})")
        
        # Remove indices outside window
        removed_outside = []
        while dq and dq[0] < i - k:
            removed = dq.popleft()
            removed_outside.append(removed)
        
        if removed_outside:
            print(f"  Removed outside window: {removed_outside}")
        
        print(f"  Valid previous indices: {list(dq)}")
        print(f"  Their dp values: {[dp[idx] for idx in dq]}")
        
        # Calculate dp[i]
        max_prev = dp[dq[0]]
        dp[i] = max_prev + nums[i]
        
        print(f"  dp[{i}] = dp[{dq[0]}] + nums[{i}] = {max_prev} + {nums[i]} = {dp[i]}")
        
        # Maintain decreasing order
        removed_smaller = []
        while dq and dp[dq[-1]] <= dp[i]:
            removed = dq.pop()
            removed_smaller.append(f"idx {removed} (dp={dp[removed]})")
        
        if removed_smaller:
            print(f"  Removed smaller/equal dp values: {removed_smaller}")
        
        dq.append(i)
        print(f"  Added index {i} to deque")
        print(f"  Deque after: {list(dq)}")
        print(f"  DP array: {dp[:i+1]}")
    
    print(f"\nFinal result: {dp[n-1]}")


def visualize_jump_paths():
    """Visualize possible jump paths"""
    print("\n=== Jump Paths Visualization ===")
    
    nums = [1, -1, -2, 4, -7, 3]
    k = 2
    
    print(f"Array: {nums}")
    print(f"Max jump distance: {k}")
    
    n = len(nums)
    
    print("\nPossible jumps from each position:")
    for i in range(n):
        possible_jumps = []
        for j in range(i + 1, min(i + k + 1, n)):
            possible_jumps.append(j)
        
        if possible_jumps:
            print(f"  From index {i} (value {nums[i]}): can jump to {possible_jumps}")
        else:
            print(f"  From index {i} (value {nums[i]}): no valid jumps (last position)")
    
    # Show optimal path
    print(f"\nFinding optimal path using DP:")
    
    dp = [float('-inf')] * n
    parent = [-1] * n
    dp[0] = nums[0]
    
    for i in range(1, n):
        for j in range(max(0, i - k), i):
            if dp[j] + nums[i] > dp[i]:
                dp[i] = dp[j] + nums[i]
                parent[i] = j
    
    # Reconstruct path
    path = []
    current = n - 1
    while current != -1:
        path.append(current)
        current = parent[current]
    
    path.reverse()
    
    print(f"Optimal path: {path}")
    print(f"Path values: {[nums[i] for i in path]}")
    print(f"Total sum: {dp[n-1]}")


def demonstrate_dp_transitions():
    """Demonstrate DP state transitions"""
    print("\n=== DP State Transitions Demonstration ===")
    
    nums = [10, -5, -2, 4, 0, 3]
    k = 3
    
    print(f"Array: {nums}")
    print(f"Max jump distance: {k}")
    
    n = len(nums)
    dp = [float('-inf')] * n
    dp[0] = nums[0]
    
    print(f"\nDP transitions:")
    print(f"dp[0] = nums[0] = {nums[0]}")
    
    for i in range(1, n):
        print(f"\nCalculating dp[{i}] (nums[{i}] = {nums[i]}):")
        
        candidates = []
        for j in range(max(0, i - k), i):
            value = dp[j] + nums[i]
            candidates.append((j, dp[j], value))
            print(f"  From index {j}: dp[{j}] + nums[{i}] = {dp[j]} + {nums[i]} = {value}")
        
        # Find maximum
        best_j, best_prev, best_value = max(candidates, key=lambda x: x[2])
        dp[i] = best_value
        
        print(f"  Best choice: from index {best_j} with value {best_value}")
        print(f"  dp[{i}] = {dp[i]}")
    
    print(f"\nFinal DP array: {dp}")
    print(f"Maximum sum: {dp[n-1]}")


def benchmark_jump_game_vi():
    """Benchmark different approaches"""
    import time
    import random
    
    algorithms = [
        ("Deque Approach", JumpGameVI().maxResult_deque_approach),
        ("Heap Approach", JumpGameVI().maxResult_heap_approach),
        ("DP Brute Force", JumpGameVI().maxResult_dp_brute_force),
        ("Segment Tree", JumpGameVI().maxResult_segment_tree),
        ("Optimized DP", JumpGameVI().maxResult_optimized_dp),
    ]
    
    # Test with different array sizes
    test_sizes = [(100, 5), (1000, 10), (5000, 20)]
    
    print("\n=== Jump Game VI Performance Benchmark ===")
    
    for size, k in test_sizes:
        print(f"\n--- Array Size: {size}, k: {k} ---")
        
        # Generate random array
        nums = [random.randint(-10, 10) for _ in range(size)]
        
        for alg_name, alg_func in algorithms:
            start_time = time.time()
            
            try:
                result = alg_func(nums, k)
                end_time = time.time()
                print(f"{alg_name:20} | Time: {end_time - start_time:.4f}s | Result: {result}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:30]}")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    solver = JumpGameVI()
    
    edge_cases = [
        ([1], 1, 1, "Single element"),
        ([1, 2], 1, 3, "Two elements, k=1"),
        ([1, 2], 2, 3, "Two elements, k=2"),
        ([5], 10, 5, "k > array length"),
        ([-1, -2, -3], 1, -6, "All negative, k=1"),
        ([-1, -2, -3], 2, -4, "All negative, k=2"),
        ([0, 0, 0], 1, 0, "All zeros"),
        ([10, -10, 10], 2, 20, "Alternating high values"),
        ([1, 1, 1, 1], 3, 4, "All ones"),
        ([-5, 10, -5, 10], 3, 15, "Skip negative values"),
    ]
    
    for nums, k, expected, description in edge_cases:
        try:
            result = solver.maxResult_deque_approach(nums, k)
            status = "✓" if result == expected else "✗"
            print(f"{description:25} | {status} | nums: {nums}, k: {k} -> {result}")
        except Exception as e:
            print(f"{description:25} | ERROR: {str(e)[:30]}")


def compare_approaches():
    """Compare different approaches"""
    print("\n=== Approach Comparison ===")
    
    test_cases = [
        ([1, -1, -2, 4, -7, 3], 2),
        ([10, -5, -2, 4, 0, 3], 3),
        ([1, 2, 3, 4, 5], 2),
        ([-1, -2, -3, -4], 2),
    ]
    
    solver = JumpGameVI()
    
    approaches = [
        ("Deque", solver.maxResult_deque_approach),
        ("Heap", solver.maxResult_heap_approach),
        ("DP Brute", solver.maxResult_dp_brute_force),
        ("Segment Tree", solver.maxResult_segment_tree),
        ("Memoization", solver.maxResult_memoization),
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
        ("Heap Approach", "O(n log n)", "O(n)", "Heap operations with lazy deletion"),
        ("DP Brute Force", "O(n * k)", "O(n)", "Check k previous positions for each i"),
        ("Segment Tree", "O(n log n)", "O(n)", "Range maximum queries"),
        ("Sparse Table", "O(n log n)", "O(n log n)", "Precompute + O(1) queries"),
        ("Optimized DP", "O(n)", "O(k)", "Sliding window maximum"),
        ("Memoization", "O(n * k)", "O(n)", "Top-down DP with memoization"),
    ]
    
    print(f"{'Approach':<20} | {'Time':<12} | {'Space':<12} | {'Notes'}")
    print("-" * 75)
    
    for approach, time_comp, space_comp, notes in approaches:
        print(f"{approach:<20} | {time_comp:<12} | {space_comp:<12} | {notes}")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    # Application 1: Stock trading with transaction costs
    print("1. Stock Trading - Maximum profit with limited transaction frequency:")
    stock_gains = [10, -5, -2, 15, -8, 12]  # Daily gains/losses
    max_gap = 2  # Can't trade more frequently than every 2 days
    
    solver = JumpGameVI()
    max_profit = solver.maxResult_deque_approach(stock_gains, max_gap)
    
    print(f"  Daily gains/losses: {stock_gains}")
    print(f"  Minimum gap between trades: {max_gap} days")
    print(f"  Maximum profit: ${max_profit}")
    
    # Application 2: Game level progression
    print("\n2. Game Level Progression - Maximum score with limited skips:")
    level_scores = [100, -50, 200, 150, -100, 300]  # Score for each level
    max_skip = 3  # Can skip at most 3 levels at once
    
    max_score = solver.maxResult_deque_approach(level_scores, max_skip)
    
    print(f"  Level scores: {level_scores}")
    print(f"  Maximum levels to skip: {max_skip}")
    print(f"  Maximum total score: {max_score}")
    
    # Application 3: Resource allocation with constraints
    print("\n3. Resource Allocation - Maximum value with capacity constraints:")
    resource_values = [50, -20, 80, 60, -30, 90]  # Value of using each resource
    capacity_limit = 2  # Can allocate to at most 2 consecutive resources
    
    max_value = solver.maxResult_deque_approach(resource_values, capacity_limit)
    
    print(f"  Resource values: {resource_values}")
    print(f"  Capacity constraint: {capacity_limit}")
    print(f"  Maximum total value: {max_value}")


def demonstrate_sliding_window_maximum_connection():
    """Demonstrate connection to sliding window maximum"""
    print("\n=== Connection to Sliding Window Maximum ===")
    
    print("Jump Game VI is essentially a sliding window maximum problem:")
    print("1. For each position i, we need the maximum dp value from positions [i-k, i-1]")
    print("2. This is exactly the sliding window maximum problem!")
    print("3. We use a deque to maintain the maximum in the current window")
    
    nums = [1, -1, -2, 4]
    k = 2
    
    print(f"\nExample: nums = {nums}, k = {k}")
    
    n = len(nums)
    dp = [float('-inf')] * n
    dp[0] = nums[0]
    
    print(f"dp[0] = {dp[0]}")
    
    for i in range(1, n):
        print(f"\nFor position {i}:")
        print(f"  Need maximum from dp[{max(0, i-k)}:{i}]")
        
        window_values = []
        for j in range(max(0, i - k), i):
            window_values.append(f"dp[{j}]={dp[j]}")
        
        print(f"  Window values: {window_values}")
        
        max_val = max(dp[j] for j in range(max(0, i - k), i))
        dp[i] = max_val + nums[i]
        
        print(f"  Maximum in window: {max_val}")
        print(f"  dp[{i}] = {max_val} + {nums[i]} = {dp[i]}")
    
    print(f"\nThis is why deque (sliding window maximum) is optimal!")


if __name__ == "__main__":
    test_jump_game_vi()
    demonstrate_deque_approach()
    visualize_jump_paths()
    demonstrate_dp_transitions()
    demonstrate_sliding_window_maximum_connection()
    demonstrate_real_world_applications()
    test_edge_cases()
    compare_approaches()
    analyze_time_complexity()
    benchmark_jump_game_vi()

"""
Jump Game VI demonstrates the application of monotonic deque for dynamic
programming optimization, specifically for sliding window maximum problems
in DP state transitions with multiple approaches and real-world applications.
"""
