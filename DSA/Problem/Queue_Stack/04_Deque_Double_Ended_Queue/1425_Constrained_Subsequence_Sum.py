"""
1425. Constrained Subsequence Sum - Multiple Approaches
Difficulty: Hard

Given an integer array nums and an integer k, return the maximum sum of a non-empty subsequence of nums such that for every two consecutive integers in the subsequence, nums[i] and nums[j], where i < j, the condition j - i <= k holds.

A subsequence of an array is a new array that is formed from the original array by deleting some (can be zero) of the elements without disturbing the relative order of the remaining elements.
"""

from typing import List, Deque
from collections import deque
import heapq

class ConstrainedSubsequenceSum:
    """Multiple approaches to find constrained subsequence sum"""
    
    def constrainedSubsetSum_deque_approach(self, nums: List[int], k: int) -> int:
        """
        Approach 1: Monotonic Deque (Optimal)
        
        Use deque to maintain maximum dp values in sliding window.
        
        Time: O(n), Space: O(k)
        """
        n = len(nums)
        # dp[i] represents maximum sum of subsequence ending at index i
        dp = [0] * n
        dp[0] = nums[0]
        
        # Deque stores indices in decreasing order of dp values
        dq: Deque[int] = deque([0])
        
        for i in range(1, n):
            # Remove indices outside the window [i-k, i-1]
            while dq and dq[0] < i - k:
                dq.popleft()
            
            # Current maximum sum: max(0, dp[best_prev]) + nums[i]
            # We can always start a new subsequence, so compare with 0
            dp[i] = max(0, dp[dq[0]]) + nums[i]
            
            # Maintain decreasing order in deque
            while dq and dp[dq[-1]] <= dp[i]:
                dq.pop()
            
            dq.append(i)
        
        return max(dp)
    
    def constrainedSubsetSum_heap_approach(self, nums: List[int], k: int) -> int:
        """
        Approach 2: Max Heap with Lazy Deletion
        
        Use max heap to track maximum dp values.
        
        Time: O(n log n), Space: O(n)
        """
        n = len(nums)
        dp = [0] * n
        dp[0] = nums[0]
        
        # Max heap (negate values for max heap using min heap)
        heap = [(-nums[0], 0)]  # (-dp_value, index)
        
        for i in range(1, n):
            # Remove outdated indices (lazy deletion)
            while heap and heap[0][1] < i - k:
                heapq.heappop(heap)
            
            # Get maximum dp value from valid range
            max_dp = max(0, -heap[0][0]) if heap else 0
            dp[i] = max_dp + nums[i]
            
            heapq.heappush(heap, (-dp[i], i))
        
        return max(dp)
    
    def constrainedSubsetSum_dp_brute_force(self, nums: List[int], k: int) -> int:
        """
        Approach 3: Dynamic Programming Brute Force
        
        For each position, check all valid previous positions.
        
        Time: O(n * k), Space: O(n)
        """
        n = len(nums)
        dp = [float('-inf')] * n
        dp[0] = nums[0]
        
        for i in range(1, n):
            # Option 1: Start new subsequence at i
            dp[i] = nums[i]
            
            # Option 2: Extend from previous positions within k distance
            for j in range(max(0, i - k), i):
                if dp[j] > 0:  # Only extend if previous sum is positive
                    dp[i] = max(dp[i], dp[j] + nums[i])
        
        return max(dp)
    
    def constrainedSubsetSum_segment_tree(self, nums: List[int], k: int) -> int:
        """
        Approach 4: Segment Tree for Range Maximum Query
        
        Use segment tree to find maximum in range efficiently.
        
        Time: O(n log n), Space: O(n)
        """
        n = len(nums)
        
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
        
        dp = [0] * n
        dp[0] = nums[0]
        update(1, 0, n - 1, 0, dp[0])
        
        for i in range(1, n):
            # Query maximum in range [max(0, i-k), i-1]
            left = max(0, i - k)
            right = i - 1
            
            max_prev = query_max(1, 0, n - 1, left, right)
            dp[i] = max(nums[i], max_prev + nums[i]) if max_prev != float('-inf') else nums[i]
            
            update(1, 0, n - 1, i, dp[i])
        
        return max(dp)
    
    def constrainedSubsetSum_memoization(self, nums: List[int], k: int) -> int:
        """
        Approach 5: Memoization (Top-down DP)
        
        Use recursion with memoization.
        
        Time: O(n * k), Space: O(n)
        """
        n = len(nums)
        memo = {}
        
        def dp(i: int) -> int:
            if i >= n:
                return 0
            
            if i in memo:
                return memo[i]
            
            # Option 1: Take current element and find best from next k positions
            take = nums[i]
            max_next = 0
            
            for j in range(i + 1, min(i + k + 1, n)):
                max_next = max(max_next, dp(j))
            
            take += max_next
            
            # Option 2: Skip current element
            skip = dp(i + 1) if i + 1 < n else 0
            
            memo[i] = max(take, skip)
            return memo[i]
        
        return dp(0)
    
    def constrainedSubsetSum_optimized_dp(self, nums: List[int], k: int) -> int:
        """
        Approach 6: Optimized DP with Sliding Window Maximum
        
        Use sliding window maximum technique with deque.
        
        Time: O(n), Space: O(k)
        """
        n = len(nums)
        dq = deque()
        dp = nums[0]
        max_sum = dp
        
        # Add first element to deque
        dq.append((dp, 0))
        
        for i in range(1, n):
            # Remove elements outside window
            while dq and dq[0][1] < i - k:
                dq.popleft()
            
            # Calculate dp[i]
            prev_max = max(0, dq[0][0]) if dq else 0
            dp = prev_max + nums[i]
            max_sum = max(max_sum, dp)
            
            # Maintain decreasing order
            while dq and dq[-1][0] <= dp:
                dq.pop()
            
            dq.append((dp, i))
        
        return max_sum
    
    def constrainedSubsetSum_greedy_approach(self, nums: List[int], k: int) -> int:
        """
        Approach 7: Greedy with DP
        
        Greedy selection with DP optimization.
        
        Time: O(n * k), Space: O(n)
        """
        n = len(nums)
        dp = [float('-inf')] * n
        
        # Base case
        dp[0] = nums[0]
        result = dp[0]
        
        for i in range(1, n):
            # Start new subsequence
            dp[i] = nums[i]
            
            # Try extending from previous positions
            for j in range(max(0, i - k), i):
                if dp[j] > 0:
                    dp[i] = max(dp[i], dp[j] + nums[i])
            
            result = max(result, dp[i])
        
        return result


def test_constrained_subsequence_sum():
    """Test constrained subsequence sum algorithms"""
    solver = ConstrainedSubsequenceSum()
    
    test_cases = [
        ([10,2,-10,5,20], 2, 37, "Example 1"),
        ([-1,-2,-3], 1, -1, "Example 2"),
        ([10,-2,-10,-5,20], 2, 23, "Example 3"),
        ([1], 1, 1, "Single element"),
        ([1, 2], 1, 3, "Two elements"),
        ([5, -3, 2], 2, 7, "Small array"),
        ([1, 2, 3, 4, 5], 2, 15, "All positive"),
        ([-1, -2, -3, -4, -5], 2, -1, "All negative"),
        ([10, -1, -1, -1, 10], 4, 20, "Large gaps"),
        ([1, -1, 1, -1, 1], 1, 3, "Alternating"),
    ]
    
    algorithms = [
        ("Deque Approach", solver.constrainedSubsetSum_deque_approach),
        ("Heap Approach", solver.constrainedSubsetSum_heap_approach),
        ("DP Brute Force", solver.constrainedSubsetSum_dp_brute_force),
        ("Segment Tree", solver.constrainedSubsetSum_segment_tree),
        ("Memoization", solver.constrainedSubsetSum_memoization),
        ("Optimized DP", solver.constrainedSubsetSum_optimized_dp),
        ("Greedy Approach", solver.constrainedSubsetSum_greedy_approach),
    ]
    
    print("=== Testing Constrained Subsequence Sum ===")
    
    for nums, k, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Array: {nums}")
        print(f"Constraint k: {k}")
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
    
    nums = [10, 2, -10, 5, 20]
    k = 2
    
    print(f"Array: {nums}")
    print(f"Constraint k: {k}")
    print("Constraint: consecutive elements in subsequence must be at most k positions apart")
    
    n = len(nums)
    dp = [0] * n
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
        if dq:
            max_prev = max(0, dp[dq[0]])  # Can start new subsequence (compare with 0)
            dp[i] = max_prev + nums[i]
            print(f"  dp[{i}] = max(0, dp[{dq[0]}]) + nums[{i}] = max(0, {dp[dq[0]]}) + {nums[i]} = {dp[i]}")
        else:
            dp[i] = nums[i]
            print(f"  dp[{i}] = nums[{i}] = {dp[i]} (no previous elements)")
        
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
    
    result = max(dp)
    print(f"\nFinal DP array: {dp}")
    print(f"Maximum subsequence sum: {result}")


def visualize_constraint():
    """Visualize the constraint"""
    print("\n=== Constraint Visualization ===")
    
    nums = [10, 2, -10, 5, 20]
    k = 2
    
    print(f"Array: {nums}")
    print(f"Constraint k: {k}")
    print("Constraint: If we pick nums[i] and nums[j] where i < j, then j - i <= k")
    
    n = len(nums)
    
    print(f"\nValid transitions from each position:")
    for i in range(n):
        valid_next = []
        for j in range(i + 1, min(i + k + 1, n)):
            valid_next.append(j)
        
        if valid_next:
            print(f"  From index {i} (value {nums[i]}): can go to indices {valid_next}")
            print(f"    Values: {[nums[j] for j in valid_next]}")
        else:
            print(f"  From index {i} (value {nums[i]}): no valid next positions")
    
    print(f"\nExample valid subsequences:")
    
    # Show some valid subsequences
    valid_subsequences = [
        [0, 2, 4],  # indices 0, 2, 4
        [0, 1, 3],  # indices 0, 1, 3
        [1, 3, 4],  # indices 1, 3, 4
        [4],        # just index 4
    ]
    
    for subseq_indices in valid_subsequences:
        values = [nums[i] for i in subseq_indices]
        total = sum(values)
        
        # Check if valid
        valid = True
        for i in range(len(subseq_indices) - 1):
            if subseq_indices[i + 1] - subseq_indices[i] > k:
                valid = False
                break
        
        status = "✓" if valid else "✗"
        print(f"  Indices {subseq_indices}: values {values} -> sum = {total} {status}")


def demonstrate_dp_transitions():
    """Demonstrate DP state transitions"""
    print("\n=== DP State Transitions Demonstration ===")
    
    nums = [10, 2, -10, 5, 20]
    k = 2
    
    print(f"Array: {nums}")
    print(f"Constraint k: {k}")
    
    n = len(nums)
    dp = [0] * n
    dp[0] = nums[0]
    
    print(f"\nDP transitions:")
    print(f"dp[0] = nums[0] = {nums[0]}")
    
    for i in range(1, n):
        print(f"\nCalculating dp[{i}] (nums[{i}] = {nums[i]}):")
        
        # Option 1: Start new subsequence
        option1 = nums[i]
        print(f"  Option 1 (start new): {nums[i]}")
        
        # Option 2: Extend from previous positions
        candidates = []
        for j in range(max(0, i - k), i):
            if dp[j] > 0:  # Only extend if beneficial
                value = dp[j] + nums[i]
                candidates.append((j, dp[j], value))
                print(f"  Option from index {j}: dp[{j}] + nums[{i}] = {dp[j]} + {nums[i]} = {value}")
        
        # Find best option
        if candidates:
            best_j, best_prev, best_value = max(candidates, key=lambda x: x[2])
            dp[i] = max(option1, best_value)
            
            if dp[i] == best_value:
                print(f"  Best choice: extend from index {best_j} with value {best_value}")
            else:
                print(f"  Best choice: start new subsequence with value {option1}")
        else:
            dp[i] = option1
            print(f"  Only option: start new subsequence")
        
        print(f"  dp[{i}] = {dp[i]}")
    
    print(f"\nFinal DP array: {dp}")
    print(f"Maximum sum: {max(dp)}")


def demonstrate_subsequence_vs_subarray():
    """Demonstrate difference between subsequence and subarray"""
    print("\n=== Subsequence vs Subarray ===")
    
    nums = [1, -2, 3, -1, 2]
    
    print(f"Array: {nums}")
    print("\nSubarray: contiguous elements")
    print("Subsequence: elements in original order, but can skip elements")
    
    print(f"\nAll subarrays:")
    n = len(nums)
    for i in range(n):
        for j in range(i, n):
            subarray = nums[i:j+1]
            print(f"  {subarray} -> sum = {sum(subarray)}")
    
    print(f"\nSome subsequences:")
    subsequences = [
        [0],           # [1]
        [2],           # [3]
        [0, 2],        # [1, 3]
        [0, 2, 4],     # [1, 3, 2]
        [2, 4],        # [3, 2]
        [0, 4],        # [1, 2]
    ]
    
    for indices in subsequences:
        subseq = [nums[i] for i in indices]
        print(f"  Indices {indices}: {subseq} -> sum = {sum(subseq)}")
    
    print(f"\nIn constrained subsequence sum:")
    print("- We can skip elements (subsequence property)")
    print("- But consecutive chosen elements must be within k distance")


def benchmark_constrained_subsequence():
    """Benchmark different approaches"""
    import time
    import random
    
    algorithms = [
        ("Deque Approach", ConstrainedSubsequenceSum().constrainedSubsetSum_deque_approach),
        ("Heap Approach", ConstrainedSubsequenceSum().constrainedSubsetSum_heap_approach),
        ("DP Brute Force", ConstrainedSubsequenceSum().constrainedSubsetSum_dp_brute_force),
        ("Segment Tree", ConstrainedSubsequenceSum().constrainedSubsetSum_segment_tree),
        ("Optimized DP", ConstrainedSubsequenceSum().constrainedSubsetSum_optimized_dp),
    ]
    
    # Test with different array sizes
    test_sizes = [(100, 5), (1000, 10), (5000, 20)]
    
    print("\n=== Constrained Subsequence Sum Performance Benchmark ===")
    
    for size, k in test_sizes:
        print(f"\n--- Array Size: {size}, k: {k} ---")
        
        # Generate random array
        nums = [random.randint(-10, 20) for _ in range(size)]
        
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
    
    solver = ConstrainedSubsequenceSum()
    
    edge_cases = [
        ([1], 1, 1, "Single element"),
        ([1, 2], 1, 3, "Two elements, k=1"),
        ([1, 2], 2, 3, "Two elements, k=2"),
        ([5], 10, 5, "k > array length"),
        ([-1, -2, -3], 1, -1, "All negative, k=1"),
        ([-1, -2, -3], 2, -1, "All negative, k=2"),
        ([0, 0, 0], 1, 0, "All zeros"),
        ([10, -10, 10], 2, 20, "Skip negative"),
        ([1, 1, 1, 1], 3, 4, "All ones"),
        ([-5, 10, -5, 10], 3, 20, "Optimal skipping"),
    ]
    
    for nums, k, expected, description in edge_cases:
        try:
            result = solver.constrainedSubsetSum_deque_approach(nums, k)
            status = "✓" if result == expected else "✗"
            print(f"{description:25} | {status} | nums: {nums}, k: {k} -> {result}")
        except Exception as e:
            print(f"{description:25} | ERROR: {str(e)[:30]}")


def compare_approaches():
    """Compare different approaches"""
    print("\n=== Approach Comparison ===")
    
    test_cases = [
        ([10, 2, -10, 5, 20], 2),
        ([-1, -2, -3], 1),
        ([10, -2, -10, -5, 20], 2),
        ([1, 2, 3, 4, 5], 2),
    ]
    
    solver = ConstrainedSubsequenceSum()
    
    approaches = [
        ("Deque", solver.constrainedSubsetSum_deque_approach),
        ("Heap", solver.constrainedSubsetSum_heap_approach),
        ("DP Brute", solver.constrainedSubsetSum_dp_brute_force),
        ("Segment Tree", solver.constrainedSubsetSum_segment_tree),
        ("Greedy", solver.constrainedSubsetSum_greedy_approach),
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
        ("Memoization", "O(n * k)", "O(n)", "Top-down DP with memoization"),
        ("Optimized DP", "O(n)", "O(k)", "Sliding window maximum with deque"),
        ("Greedy Approach", "O(n * k)", "O(n)", "Greedy selection with DP"),
    ]
    
    print(f"{'Approach':<20} | {'Time':<12} | {'Space':<8} | {'Notes'}")
    print("-" * 70)
    
    for approach, time_comp, space_comp, notes in approaches:
        print(f"{approach:<20} | {time_comp:<12} | {space_comp:<8} | {notes}")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    # Application 1: Investment portfolio with timing constraints
    print("1. Investment Portfolio - Maximum return with timing constraints:")
    investment_returns = [100, -50, 200, -30, 150, -20, 180]  # Monthly returns
    max_gap = 3  # Can't have investments more than 3 months apart
    
    solver = ConstrainedSubsequenceSum()
    max_return = solver.constrainedSubsetSum_deque_approach(investment_returns, max_gap)
    
    print(f"  Monthly returns: {investment_returns}")
    print(f"  Max gap between investments: {max_gap} months")
    print(f"  Maximum total return: ${max_return}")
    
    # Application 2: Task scheduling with dependencies
    print("\n2. Task Scheduling - Maximum value with dependency constraints:")
    task_values = [50, -20, 80, -10, 120, -30, 90]  # Value of completing each task
    dependency_window = 2  # Tasks must be completed within 2 time units
    
    max_value = solver.constrainedSubsetSum_deque_approach(task_values, dependency_window)
    
    print(f"  Task values: {task_values}")
    print(f"  Dependency window: {dependency_window} time units")
    print(f"  Maximum total value: {max_value}")
    
    # Application 3: Resource allocation with capacity constraints
    print("\n3. Resource Allocation - Maximum benefit with capacity limits:")
    resource_benefits = [80, -40, 120, -20, 100, -15, 140]  # Benefit from each resource
    capacity_constraint = 3  # Resources must be allocated within 3 units
    
    max_benefit = solver.constrainedSubsetSum_deque_approach(resource_benefits, capacity_constraint)
    
    print(f"  Resource benefits: {resource_benefits}")
    print(f"  Capacity constraint: {capacity_constraint} units")
    print(f"  Maximum total benefit: {max_benefit}")


def demonstrate_connection_to_jump_game():
    """Demonstrate connection to Jump Game VI"""
    print("\n=== Connection to Jump Game VI ===")
    
    print("Constrained Subsequence Sum is similar to Jump Game VI:")
    print("1. Both use DP with sliding window maximum")
    print("2. Both have constraint on distance between consecutive elements")
    print("3. Both can be solved optimally with monotonic deque")
    
    print("\nKey differences:")
    print("- Jump Game VI: Must reach the end")
    print("- Constrained Subsequence: Can end anywhere, find global maximum")
    print("- Jump Game VI: dp[i] = max(dp[j]) + nums[i] for j in [i-k, i-1]")
    print("- Constrained Subsequence: dp[i] = max(0, max(dp[j])) + nums[i] for j in [i-k, i-1]")
    
    nums = [10, -5, 20]
    k = 1
    
    print(f"\nExample: nums = {nums}, k = {k}")
    
    # Jump Game VI approach
    print("\nJump Game VI (must reach end):")
    n = len(nums)
    dp_jump = [float('-inf')] * n
    dp_jump[0] = nums[0]
    
    for i in range(1, n):
        for j in range(max(0, i - k), i):
            dp_jump[i] = max(dp_jump[i], dp_jump[j] + nums[i])
    
    print(f"  DP array: {dp_jump}")
    print(f"  Result (value at end): {dp_jump[-1]}")
    
    # Constrained Subsequence approach
    print("\nConstrained Subsequence (global maximum):")
    dp_subseq = [0] * n
    dp_subseq[0] = nums[0]
    
    for i in range(1, n):
        max_prev = 0
        for j in range(max(0, i - k), i):
            max_prev = max(max_prev, dp_subseq[j])
        dp_subseq[i] = max(0, max_prev) + nums[i]
    
    print(f"  DP array: {dp_subseq}")
    print(f"  Result (global maximum): {max(dp_subseq)}")


if __name__ == "__main__":
    test_constrained_subsequence_sum()
    demonstrate_deque_approach()
    visualize_constraint()
    demonstrate_dp_transitions()
    demonstrate_subsequence_vs_subarray()
    demonstrate_connection_to_jump_game()
    demonstrate_real_world_applications()
    test_edge_cases()
    compare_approaches()
    analyze_time_complexity()
    benchmark_constrained_subsequence()

"""
Constrained Subsequence Sum demonstrates advanced DP optimization with deque
for subsequence problems with distance constraints, including multiple approaches
and connections to similar sliding window maximum problems.
"""
