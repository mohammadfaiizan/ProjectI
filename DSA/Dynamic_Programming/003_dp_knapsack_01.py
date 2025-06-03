"""
Dynamic Programming - 0/1 Knapsack Pattern
This module implements the classic 0/1 Knapsack problem and its numerous variants including
subset sum, equal partition, target sum, and counting problems with detailed optimizations.
"""

from typing import List, Dict, Tuple, Optional, Set
import time
from collections import defaultdict

# ==================== CLASSIC 0/1 KNAPSACK ====================

class Knapsack01:
    """
    Classic 0/1 Knapsack Problem Implementation
    
    Given items with weights and values, and a knapsack capacity,
    find the maximum value that can be obtained by selecting items
    such that total weight doesn't exceed capacity.
    
    Key constraint: Each item can be taken at most once (0/1 choice).
    """
    
    def knapsack_recursive(self, weights: List[int], values: List[int], 
                          capacity: int, n: int) -> int:
        """
        Recursive solution (exponential time - for demonstration)
        
        Time Complexity: O(2^n)
        Space Complexity: O(n) - recursion stack
        
        Args:
            weights: List of item weights
            values: List of item values  
            capacity: Knapsack capacity
            n: Number of items to consider
        
        Returns:
            Maximum value achievable
        """
        # Base case
        if n == 0 or capacity == 0:
            return 0
        
        # If weight of current item exceeds capacity, skip it
        if weights[n - 1] > capacity:
            return self.knapsack_recursive(weights, values, capacity, n - 1)
        
        # Choose maximum of including or excluding current item
        include = values[n - 1] + self.knapsack_recursive(
            weights, values, capacity - weights[n - 1], n - 1
        )
        exclude = self.knapsack_recursive(weights, values, capacity, n - 1)
        
        return max(include, exclude)
    
    def knapsack_memoization(self, weights: List[int], values: List[int], 
                            capacity: int) -> int:
        """
        Memoization approach (top-down DP)
        
        Time Complexity: O(n * capacity)
        Space Complexity: O(n * capacity)
        
        Args:
            weights: List of item weights
            values: List of item values
            capacity: Knapsack capacity
        
        Returns:
            Maximum value achievable
        """
        n = len(weights)
        memo = {}
        
        def dp(i: int, remaining_capacity: int) -> int:
            # Base case
            if i == n or remaining_capacity == 0:
                return 0
            
            # Check memo
            if (i, remaining_capacity) in memo:
                return memo[(i, remaining_capacity)]
            
            # Skip current item if it doesn't fit
            if weights[i] > remaining_capacity:
                result = dp(i + 1, remaining_capacity)
            else:
                # Choose max of including or excluding current item
                include = values[i] + dp(i + 1, remaining_capacity - weights[i])
                exclude = dp(i + 1, remaining_capacity)
                result = max(include, exclude)
            
            memo[(i, remaining_capacity)] = result
            return result
        
        return dp(0, capacity)
    
    def knapsack_tabulation(self, weights: List[int], values: List[int], 
                           capacity: int) -> int:
        """
        Tabulation approach (bottom-up DP)
        
        Time Complexity: O(n * capacity)
        Space Complexity: O(n * capacity)
        
        dp[i][w] = maximum value using first i items with capacity w
        
        Args:
            weights: List of item weights
            values: List of item values
            capacity: Knapsack capacity
        
        Returns:
            Maximum value achievable
        """
        n = len(weights)
        
        # Create DP table
        dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
        
        # Fill DP table
        for i in range(1, n + 1):
            for w in range(1, capacity + 1):
                # Don't include current item
                dp[i][w] = dp[i - 1][w]
                
                # Include current item if it fits
                if weights[i - 1] <= w:
                    include_value = values[i - 1] + dp[i - 1][w - weights[i - 1]]
                    dp[i][w] = max(dp[i][w], include_value)
        
        return dp[n][capacity]
    
    def knapsack_space_optimized(self, weights: List[int], values: List[int], 
                                capacity: int) -> int:
        """
        Space-optimized version using 1D array
        
        Time Complexity: O(n * capacity)
        Space Complexity: O(capacity)
        
        Key insight: We only need the previous row to compute current row
        """
        n = len(weights)
        dp = [0 for _ in range(capacity + 1)]
        
        for i in range(n):
            # Traverse in reverse to avoid using updated values
            for w in range(capacity, weights[i] - 1, -1):
                dp[w] = max(dp[w], values[i] + dp[w - weights[i]])
        
        return dp[capacity]
    
    def knapsack_with_items(self, weights: List[int], values: List[int], 
                           capacity: int) -> Tuple[int, List[int]]:
        """
        Return both maximum value and selected items
        
        Args:
            weights: List of item weights
            values: List of item values
            capacity: Knapsack capacity
        
        Returns:
            Tuple of (max_value, selected_items_indices)
        """
        n = len(weights)
        dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
        
        # Fill DP table
        for i in range(1, n + 1):
            for w in range(1, capacity + 1):
                dp[i][w] = dp[i - 1][w]
                if weights[i - 1] <= w:
                    include_value = values[i - 1] + dp[i - 1][w - weights[i - 1]]
                    dp[i][w] = max(dp[i][w], include_value)
        
        # Backtrack to find selected items
        selected_items = []
        w = capacity
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i - 1][w]:
                selected_items.append(i - 1)  # Item index
                w -= weights[i - 1]
        
        selected_items.reverse()
        return dp[n][capacity], selected_items

# ==================== SUBSET SUM PROBLEM ====================

class SubsetSum:
    """
    Subset Sum Problem and Variants
    
    Given a set of positive integers, determine if there exists
    a subset with sum equal to given target.
    """
    
    def subset_sum_exists(self, nums: List[int], target: int) -> bool:
        """
        Check if subset with given sum exists
        
        Time Complexity: O(n * target)
        Space Complexity: O(target)
        
        Args:
            nums: List of positive integers
            target: Target sum
        
        Returns:
            True if subset with target sum exists
        """
        dp = [False] * (target + 1)
        dp[0] = True  # Empty subset has sum 0
        
        for num in nums:
            # Traverse in reverse to avoid using updated values
            for j in range(target, num - 1, -1):
                dp[j] = dp[j] or dp[j - num]
        
        return dp[target]
    
    def subset_sum_2d(self, nums: List[int], target: int) -> bool:
        """
        2D DP version for better understanding
        
        dp[i][j] = True if subset of first i numbers has sum j
        """
        n = len(nums)
        dp = [[False for _ in range(target + 1)] for _ in range(n + 1)]
        
        # Base case: empty subset has sum 0
        for i in range(n + 1):
            dp[i][0] = True
        
        for i in range(1, n + 1):
            for j in range(1, target + 1):
                # Don't include current number
                dp[i][j] = dp[i - 1][j]
                
                # Include current number if possible
                if nums[i - 1] <= j:
                    dp[i][j] = dp[i][j] or dp[i - 1][j - nums[i - 1]]
        
        return dp[n][target]
    
    def find_subset_with_sum(self, nums: List[int], target: int) -> List[int]:
        """
        Find actual subset that sums to target (if exists)
        
        Args:
            nums: List of positive integers
            target: Target sum
        
        Returns:
            List representing subset, empty if no solution
        """
        n = len(nums)
        dp = [[False for _ in range(target + 1)] for _ in range(n + 1)]
        
        # Fill DP table
        for i in range(n + 1):
            dp[i][0] = True
        
        for i in range(1, n + 1):
            for j in range(1, target + 1):
                dp[i][j] = dp[i - 1][j]
                if nums[i - 1] <= j:
                    dp[i][j] = dp[i][j] or dp[i - 1][j - nums[i - 1]]
        
        # If no solution exists
        if not dp[n][target]:
            return []
        
        # Backtrack to find subset
        subset = []
        i, j = n, target
        
        while i > 0 and j > 0:
            # If current element is included
            if dp[i][j] and not dp[i - 1][j]:
                subset.append(nums[i - 1])
                j -= nums[i - 1]
            i -= 1
        
        return subset
    
    def count_subsets_with_sum(self, nums: List[int], target: int) -> int:
        """
        Count number of subsets with given sum
        
        Time Complexity: O(n * target)
        Space Complexity: O(target)
        
        Args:
            nums: List of positive integers
            target: Target sum
        
        Returns:
            Number of subsets with target sum
        """
        dp = [0] * (target + 1)
        dp[0] = 1  # One way to make sum 0 (empty subset)
        
        for num in nums:
            for j in range(target, num - 1, -1):
                dp[j] += dp[j - num]
        
        return dp[target]
    
    def subset_sum_with_duplicates(self, nums: List[int], target: int) -> bool:
        """
        Subset sum when array may contain duplicates
        Need to handle duplicates carefully to avoid counting same subset multiple times
        """
        nums.sort()  # Sort to group duplicates
        dp = [False] * (target + 1)
        dp[0] = True
        
        i = 0
        while i < len(nums):
            # Count occurrences of current number
            count = 1
            while i + count < len(nums) and nums[i + count] == nums[i]:
                count += 1
            
            # Process this number with its count
            prev_dp = dp[:]
            for k in range(1, count + 1):
                for j in range(target, nums[i] * k - 1, -1):
                    dp[j] = dp[j] or prev_dp[j - nums[i] * k]
            
            i += count
        
        return dp[target]

# ==================== EQUAL SUM PARTITION ====================

class EqualSumPartition:
    """
    Equal Sum Partition Problem
    
    Given a set of positive integers, determine if it can be partitioned
    into two subsets with equal sum.
    """
    
    def can_partition(self, nums: List[int]) -> bool:
        """
        Check if array can be partitioned into two equal sum subsets
        
        Time Complexity: O(n * sum)
        Space Complexity: O(sum)
        
        Key insight: If total sum is odd, impossible to partition equally
        If total sum is even, find subset with sum = total_sum / 2
        
        Args:
            nums: List of positive integers
        
        Returns:
            True if equal partition is possible
        """
        total_sum = sum(nums)
        
        # If total sum is odd, can't partition equally
        if total_sum % 2 == 1:
            return False
        
        target = total_sum // 2
        
        # Use subset sum to find if target sum is achievable
        return SubsetSum().subset_sum_exists(nums, target)
    
    def can_partition_k_subsets(self, nums: List[int], k: int) -> bool:
        """
        Partition array into k subsets with equal sum
        
        This is more complex and uses backtracking with memoization
        
        Args:
            nums: List of positive integers
            k: Number of subsets
        
        Returns:
            True if can be partitioned into k equal sum subsets
        """
        total_sum = sum(nums)
        
        if total_sum % k != 0:
            return False
        
        target = total_sum // k
        nums.sort(reverse=True)  # Sort in descending order for pruning
        
        if nums[0] > target:
            return False
        
        used = [False] * len(nums)
        
        def backtrack(groups: int, current_sum: int, start_index: int) -> bool:
            if groups == k:
                return True
            
            if current_sum == target:
                return backtrack(groups + 1, 0, 0)
            
            for i in range(start_index, len(nums)):
                if used[i] or current_sum + nums[i] > target:
                    continue
                
                used[i] = True
                if backtrack(groups, current_sum + nums[i], i + 1):
                    return True
                used[i] = False
                
                # Pruning: if current number didn't work and current_sum is 0,
                # then other numbers of same value won't work either
                if current_sum == 0:
                    break
            
            return False
        
        return backtrack(0, 0, 0)
    
    def minimum_subset_sum_difference(self, nums: List[int]) -> int:
        """
        Find minimum difference between two subset sums
        
        Time Complexity: O(n * sum)
        Space Complexity: O(sum)
        
        Args:
            nums: List of positive integers
        
        Returns:
            Minimum possible difference between two subset sums
        """
        total_sum = sum(nums)
        target = total_sum // 2
        
        # Find all possible sums up to target
        dp = [False] * (target + 1)
        dp[0] = True
        
        for num in nums:
            for j in range(target, num - 1, -1):
                dp[j] = dp[j] or dp[j - num]
        
        # Find the largest sum <= target that's achievable
        for i in range(target, -1, -1):
            if dp[i]:
                # One subset has sum i, other has sum (total_sum - i)
                return total_sum - 2 * i
        
        return total_sum

# ==================== TARGET SUM PROBLEMS ====================

class TargetSum:
    """
    Target Sum Problems - including LeetCode 494
    
    Given array of non-negative integers and a target,
    assign + or - sign to each number to reach target sum.
    """
    
    def find_target_sum_ways(self, nums: List[int], target: int) -> int:
        """
        Count ways to assign signs to reach target sum
        
        Mathematical insight:
        Let P = sum of positive numbers, N = sum of negative numbers
        P - N = target
        P + N = sum(nums)
        Solving: P = (target + sum(nums)) / 2
        
        Problem reduces to: count subsets with sum = P
        
        Time Complexity: O(n * sum)
        Space Complexity: O(sum)
        
        Args:
            nums: List of non-negative integers
            target: Target sum to achieve
        
        Returns:
            Number of ways to assign signs
        """
        total_sum = sum(nums)
        
        # Check if target is achievable
        if target > total_sum or target < -total_sum or (target + total_sum) % 2 == 1:
            return 0
        
        subset_sum = (target + total_sum) // 2
        
        # Count subsets with sum = subset_sum
        dp = [0] * (subset_sum + 1)
        dp[0] = 1
        
        for num in nums:
            for j in range(subset_sum, num - 1, -1):
                dp[j] += dp[j - num]
        
        return dp[subset_sum]
    
    def find_target_sum_ways_with_zeros(self, nums: List[int], target: int) -> int:
        """
        Handle case where nums contains zeros
        
        Zeros can be assigned either + or - sign without affecting sum,
        so each zero doubles the number of ways.
        """
        # Count zeros and remove them
        zeros = nums.count(0)
        nums = [x for x in nums if x != 0]
        
        # Calculate ways without zeros
        ways = self.find_target_sum_ways(nums, target)
        
        # Each zero doubles the number of ways
        return ways * (2 ** zeros)
    
    def can_reach_target(self, nums: List[int], target: int) -> bool:
        """
        Check if target sum is reachable (boolean version)
        
        Args:
            nums: List of non-negative integers
            target: Target sum
        
        Returns:
            True if target is reachable
        """
        return self.find_target_sum_ways(nums, target) > 0

# ==================== ADVANCED KNAPSACK VARIANTS ====================

class AdvancedKnapsack:
    """
    Advanced variations of knapsack problems
    """
    
    def knapsack_with_exactly_k_items(self, weights: List[int], values: List[int], 
                                     capacity: int, k: int) -> int:
        """
        0/1 Knapsack with constraint: select exactly k items
        
        dp[i][w][items] = max value using first i items, weight w, exactly 'items' items
        
        Time Complexity: O(n * capacity * k)
        Space Complexity: O(capacity * k)
        """
        n = len(weights)
        
        # dp[w][items] = max value with weight w and exactly 'items' items
        dp = [[-1 for _ in range(k + 1)] for _ in range(capacity + 1)]
        dp[0][0] = 0  # Base case: 0 weight, 0 items, 0 value
        
        for i in range(n):
            # Create new dp table for current item
            new_dp = [[-1 for _ in range(k + 1)] for _ in range(capacity + 1)]
            
            # Copy previous state (not including current item)
            for w in range(capacity + 1):
                for items in range(k + 1):
                    new_dp[w][items] = dp[w][items]
            
            # Include current item if possible
            for w in range(capacity + 1):
                for items in range(k):
                    if (dp[w][items] != -1 and 
                        w + weights[i] <= capacity):
                        new_value = dp[w][items] + values[i]
                        new_dp[w + weights[i]][items + 1] = max(
                            new_dp[w + weights[i]][items + 1], new_value
                        )
            
            dp = new_dp
        
        # Find maximum value with exactly k items
        max_value = -1
        for w in range(capacity + 1):
            if dp[w][k] != -1:
                max_value = max(max_value, dp[w][k])
        
        return max_value if max_value != -1 else 0
    
    def count_ways_to_make_sum(self, nums: List[int], target: int) -> int:
        """
        Count number of ways to select items (with repetition allowed)
        to make target sum
        
        This is actually unbounded knapsack, but included here for completeness
        """
        dp = [0] * (target + 1)
        dp[0] = 1
        
        for i in range(1, target + 1):
            for num in nums:
                if i >= num:
                    dp[i] += dp[i - num]
        
        return dp[target]
    
    def knapsack_with_conflicts(self, weights: List[int], values: List[int],
                               capacity: int, conflicts: List[Tuple[int, int]]) -> int:
        """
        0/1 Knapsack where some items cannot be selected together
        
        This requires more complex DP or constraint programming
        For simplicity, implementing with bitmask DP for small n
        """
        n = len(weights)
        if n > 20:  # Too large for bitmask DP
            return -1
        
        # Create conflict set for each item
        conflict_sets = [set() for _ in range(n)]
        for i, j in conflicts:
            conflict_sets[i].add(j)
            conflict_sets[j].add(i)
        
        max_value = 0
        
        # Try all possible combinations (2^n)
        for mask in range(1 << n):
            total_weight = 0
            total_value = 0
            valid = True
            
            selected_items = []
            for i in range(n):
                if mask & (1 << i):
                    selected_items.append(i)
                    total_weight += weights[i]
                    total_value += values[i]
            
            # Check weight constraint
            if total_weight > capacity:
                continue
            
            # Check conflict constraints
            for i in selected_items:
                for j in selected_items:
                    if i != j and j in conflict_sets[i]:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                max_value = max(max_value, total_value)
        
        return max_value

# ==================== PERFORMANCE ANALYSIS ====================

def performance_comparison():
    """Compare performance of different knapsack implementations"""
    print("=== Knapsack Performance Comparison ===\n")
    
    # Test data
    weights = [10, 20, 30]
    values = [60, 100, 120]
    capacity = 50
    
    knapsack = Knapsack01()
    
    print(f"Test case: weights={weights}, values={values}, capacity={capacity}")
    
    # Memoization
    start_time = time.time()
    memo_result = knapsack.knapsack_memoization(weights, values, capacity)
    memo_time = time.time() - start_time
    
    # Tabulation
    start_time = time.time()
    tab_result = knapsack.knapsack_tabulation(weights, values, capacity)
    tab_time = time.time() - start_time
    
    # Space optimized
    start_time = time.time()
    opt_result = knapsack.knapsack_space_optimized(weights, values, capacity)
    opt_time = time.time() - start_time
    
    print(f"Memoization: {memo_result} ({memo_time:.6f}s)")
    print(f"Tabulation:  {tab_result} ({tab_time:.6f}s)")
    print(f"Optimized:   {opt_result} ({opt_time:.6f}s)")
    print(f"All match:   {memo_result == tab_result == opt_result}")

# ==================== EXAMPLE USAGE AND TESTING ====================

if __name__ == "__main__":
    print("=== 0/1 Knapsack Pattern Demo ===\n")
    
    # Classic 0/1 Knapsack
    print("1. Classic 0/1 Knapsack:")
    knapsack = Knapsack01()
    
    weights = [10, 20, 30]
    values = [60, 100, 120]
    capacity = 50
    
    max_value = knapsack.knapsack_space_optimized(weights, values, capacity)
    max_value_with_items, selected_items = knapsack.knapsack_with_items(weights, values, capacity)
    
    print(f"  Items: weights={weights}, values={values}")
    print(f"  Capacity: {capacity}")
    print(f"  Maximum value: {max_value}")
    print(f"  Selected items (indices): {selected_items}")
    print(f"  Selected weights: {[weights[i] for i in selected_items]}")
    print(f"  Selected values: {[values[i] for i in selected_items]}")
    print()
    
    # Subset Sum
    print("2. Subset Sum Problems:")
    subset_sum = SubsetSum()
    
    nums = [3, 34, 4, 12, 5, 2]
    target = 9
    
    exists = subset_sum.subset_sum_exists(nums, target)
    actual_subset = subset_sum.find_subset_with_sum(nums, target)
    count = subset_sum.count_subsets_with_sum(nums, target)
    
    print(f"  Array: {nums}, Target: {target}")
    print(f"  Subset exists: {exists}")
    print(f"  Actual subset: {actual_subset}")
    print(f"  Count of subsets: {count}")
    
    # Test with different target
    target2 = 15
    exists2 = subset_sum.subset_sum_exists(nums, target2)
    subset2 = subset_sum.find_subset_with_sum(nums, target2)
    
    print(f"  Target {target2}: exists={exists2}, subset={subset2}")
    print()
    
    # Equal Sum Partition
    print("3. Equal Sum Partition:")
    partition = EqualSumPartition()
    
    test_arrays = [
        [1, 5, 11, 5],
        [1, 2, 3, 5],
        [1, 2, 5]
    ]
    
    for arr in test_arrays:
        can_partition = partition.can_partition(arr)
        min_diff = partition.minimum_subset_sum_difference(arr)
        print(f"  Array {arr}: can_partition={can_partition}, min_diff={min_diff}")
    
    # K-subset partition
    nums_k = [4, 3, 2, 3, 5, 2, 1]
    k = 4
    can_k_partition = partition.can_partition_k_subsets(nums_k, k)
    print(f"  Array {nums_k} into {k} subsets: {can_k_partition}")
    print()
    
    # Target Sum
    print("4. Target Sum Problems:")
    target_sum = TargetSum()
    
    nums_target = [1, 1, 1, 1, 1]
    target_val = 3
    
    ways = target_sum.find_target_sum_ways(nums_target, target_val)
    can_reach = target_sum.can_reach_target(nums_target, target_val)
    
    print(f"  Array: {nums_target}, Target: {target_val}")
    print(f"  Number of ways: {ways}")
    print(f"  Can reach target: {can_reach}")
    
    # Test with zeros
    nums_with_zeros = [1, 0, 1]
    target_with_zeros = 0
    ways_with_zeros = target_sum.find_target_sum_ways_with_zeros(nums_with_zeros, target_with_zeros)
    print(f"  Array with zeros {nums_with_zeros}, Target {target_with_zeros}: {ways_with_zeros} ways")
    print()
    
    # Advanced Knapsack
    print("5. Advanced Knapsack Variants:")
    advanced = AdvancedKnapsack()
    
    # Exactly k items
    weights_k = [1, 2, 3, 4]
    values_k = [1, 4, 7, 9]
    capacity_k = 5
    k_items = 2
    
    max_value_k = advanced.knapsack_with_exactly_k_items(weights_k, values_k, capacity_k, k_items)
    print(f"  Knapsack with exactly {k_items} items:")
    print(f"    Weights: {weights_k}, Values: {values_k}, Capacity: {capacity_k}")
    print(f"    Maximum value: {max_value_k}")
    
    # Knapsack with conflicts
    conflicts = [(0, 1), (2, 3)]  # Items 0,1 conflict and items 2,3 conflict
    max_value_conflicts = advanced.knapsack_with_conflicts(weights_k, values_k, capacity_k, conflicts)
    print(f"  Knapsack with conflicts {conflicts}: {max_value_conflicts}")
    print()
    
    # Performance comparison
    performance_comparison()
    print()
    
    # Pattern Recognition Guide
    print("=== 0/1 Knapsack Pattern Recognition ===")
    print("Identify 0/1 Knapsack patterns when:")
    print("  1. Given a set of items with properties (weight, value, etc.)")
    print("  2. Each item can be chosen at most once (0/1 decision)")
    print("  3. Goal is to optimize (maximize/minimize) some objective")
    print("  4. Subject to constraints (capacity, sum, etc.)")
    
    print("\nCommon variants:")
    print("  1. Classic Knapsack: maximize value subject to weight constraint")
    print("  2. Subset Sum: find subset with specific sum")
    print("  3. Equal Partition: divide into two equal sum subsets")
    print("  4. Target Sum: assign +/- signs to reach target")
    print("  5. Count problems: count number of ways/subsets")
    
    print("\nOptimization techniques:")
    print("  1. Space: 2D → 1D when only previous row needed")
    print("  2. Early termination: break when target found")
    print("  3. Sorting: may enable pruning")
    print("  4. Bitmask DP: for small n with complex constraints")
    
    print("\nReal-world applications:")
    print("  1. Resource allocation with budget constraints")
    print("  2. Portfolio optimization")
    print("  3. Feature selection in machine learning")
    print("  4. Bin packing and scheduling")
    print("  5. Cryptography and combinatorial optimization")
    
    print("\n=== Demo Complete ===") 