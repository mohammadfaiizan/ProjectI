"""
LeetCode 698: Partition to K Equal Sum Subsets
Difficulty: Medium
Category: Knapsack Problems / Backtracking with DP

PROBLEM DESCRIPTION:
===================
Given an integer array nums and an integer k, return true if it is possible to divide this 
array into k non-empty subsets whose sums are all equal.

Example 1:
Input: nums = [4,3,2,3,5,2,1], k = 2
Output: true
Explanation: It's possible to divide it into two subsets with equal sum: {1,5,2} and {4,3,2,3}.

Example 2:
Input: nums = [1,2,3,4], k = 3
Output: false

Constraints:
- 1 <= k <= len(nums) <= 16
- 1 <= nums[i] <= 10^4
- The frequency of each element is in the range [1, 4].
"""

def can_partition_k_subsets_bruteforce(nums, k):
    """
    BRUTE FORCE APPROACH - RECURSION:
    ================================
    Try all possible ways to assign numbers to k subsets.
    
    Time Complexity: O(k^n) - k choices for each number
    Space Complexity: O(n) - recursion stack depth
    """
    total_sum = sum(nums)
    
    # Early termination checks
    if total_sum % k != 0:
        return False
    
    target = total_sum // k
    
    # Check if any number is larger than target
    if any(num > target for num in nums):
        return False
    
    def backtrack(index, subsets):
        if index == len(nums):
            # Check if all subsets have target sum
            return all(subset_sum == target for subset_sum in subsets)
        
        current_num = nums[index]
        
        # Try placing current number in each subset
        for i in range(k):
            if subsets[i] + current_num <= target:
                subsets[i] += current_num
                if backtrack(index + 1, subsets):
                    return True
                subsets[i] -= current_num
        
        return False
    
    return backtrack(0, [0] * k)


def can_partition_k_subsets_optimized_backtrack(nums, k):
    """
    OPTIMIZED BACKTRACKING:
    ======================
    Add pruning and optimizations to reduce search space.
    
    Time Complexity: O(k^n) - worst case, much better with pruning
    Space Complexity: O(n) - recursion stack
    """
    total_sum = sum(nums)
    
    if total_sum % k != 0:
        return False
    
    target = total_sum // k
    
    if any(num > target for num in nums):
        return False
    
    # Sort in descending order for better pruning
    nums.sort(reverse=True)
    
    def backtrack(index, subsets):
        if index == len(nums):
            return True  # All numbers placed, sums must be equal
        
        current_num = nums[index]
        
        # Try placing in each subset
        for i in range(k):
            if subsets[i] + current_num <= target:
                subsets[i] += current_num
                if backtrack(index + 1, subsets):
                    return True
                subsets[i] -= current_num
                
                # Pruning: if current subset is empty, no need to try other empty subsets
                if subsets[i] == 0:
                    break
        
        return False
    
    return backtrack(0, [0] * k)


def can_partition_k_subsets_memoization(nums, k):
    """
    MEMOIZATION WITH BITMASK:
    ========================
    Use bitmask to represent used numbers and memoize states.
    
    Time Complexity: O(n * 2^n) - n positions, 2^n masks
    Space Complexity: O(2^n) - memoization table
    """
    total_sum = sum(nums)
    
    if total_sum % k != 0:
        return False
    
    target = total_sum // k
    
    if any(num > target for num in nums):
        return False
    
    n = len(nums)
    memo = {}
    
    def dp(mask, current_sum):
        if mask == (1 << n) - 1:  # All numbers used
            return True
        
        if current_sum == target:  # Current subset complete
            return dp(mask, 0)
        
        if (mask, current_sum) in memo:
            return memo[(mask, current_sum)]
        
        # Try adding each unused number
        for i in range(n):
            if not (mask & (1 << i)):  # Number i not used
                if current_sum + nums[i] <= target:
                    if dp(mask | (1 << i), current_sum + nums[i]):
                        memo[(mask, current_sum)] = True
                        return True
        
        memo[(mask, current_sum)] = False
        return False
    
    return dp(0, 0)


def can_partition_k_subsets_bucket_filling(nums, k):
    """
    BUCKET FILLING APPROACH:
    =======================
    Fill buckets one by one instead of placing numbers.
    
    Time Complexity: O(k * 2^n) - fill k buckets, 2^n combinations each
    Space Complexity: O(n) - recursion stack
    """
    total_sum = sum(nums)
    
    if total_sum % k != 0:
        return False
    
    target = total_sum // k
    
    if any(num > target for num in nums):
        return False
    
    # Sort in descending order for better pruning
    nums.sort(reverse=True)
    used = [False] * len(nums)
    
    def fill_bucket(bucket_id, current_sum, start_index):
        if bucket_id == k:
            return True  # All buckets filled
        
        if current_sum == target:
            return fill_bucket(bucket_id + 1, 0, 0)  # Start next bucket
        
        for i in range(start_index, len(nums)):
            if not used[i] and current_sum + nums[i] <= target:
                used[i] = True
                if fill_bucket(bucket_id, current_sum + nums[i], i + 1):
                    return True
                used[i] = False
        
        return False
    
    return fill_bucket(0, 0, 0)


def can_partition_k_subsets_dp_bitmask(nums, k):
    """
    BOTTOM-UP DP WITH BITMASK:
    =========================
    Build solution bottom-up using bitmask DP.
    
    Time Complexity: O(2^n * n) - for each mask, try each number
    Space Complexity: O(2^n) - DP array
    """
    total_sum = sum(nums)
    
    if total_sum % k != 0:
        return False
    
    target = total_sum // k
    
    if any(num > target for num in nums):
        return False
    
    n = len(nums)
    
    # dp[mask] = True if the numbers in mask can be partitioned
    dp = [False] * (1 << n)
    dp[0] = True
    
    # subset_sum[mask] = sum of numbers in current incomplete subset
    subset_sum = [0] * (1 << n)
    
    for mask in range(1 << n):
        if not dp[mask]:
            continue
        
        current_sum = subset_sum[mask]
        
        if current_sum == target:
            # Current subset is complete, start new one
            subset_sum[mask] = 0
            current_sum = 0
        
        for i in range(n):
            if not (mask & (1 << i)):  # Number i not used
                if current_sum + nums[i] <= target:
                    new_mask = mask | (1 << i)
                    if not dp[new_mask]:
                        dp[new_mask] = True
                        subset_sum[new_mask] = current_sum + nums[i]
    
    return dp[(1 << n) - 1]


def can_partition_k_subsets_optimized_pruning(nums, k):
    """
    HEAVILY OPTIMIZED BACKTRACKING:
    ==============================
    Multiple pruning techniques for maximum efficiency.
    
    Time Complexity: O(k^n) - worst case, much better with pruning
    Space Complexity: O(n) - recursion stack
    """
    total_sum = sum(nums)
    
    if total_sum % k != 0:
        return False
    
    target = total_sum // k
    
    if any(num > target for num in nums):
        return False
    
    # Sort in descending order
    nums.sort(reverse=True)
    
    # Early termination: if largest number equals target
    if nums[0] == target:
        return can_partition_k_subsets_optimized_pruning(nums[1:], k - 1)
    
    def backtrack(index, subsets):
        if index == len(nums):
            return True
        
        current_num = nums[index]
        
        # Try placing in each subset
        for i in range(k):
            if subsets[i] + current_num <= target:
                subsets[i] += current_num
                if backtrack(index + 1, subsets):
                    return True
                subsets[i] -= current_num
                
                # Pruning 1: Skip equivalent empty subsets
                if subsets[i] == 0:
                    break
        
        return False
    
    return backtrack(0, [0] * k)


def can_partition_k_subsets_with_subsets(nums, k):
    """
    FIND ACTUAL SUBSETS:
    ===================
    Return whether partition is possible and show actual subsets.
    
    Time Complexity: O(k^n) - backtracking with subset tracking
    Space Complexity: O(n * k) - store subsets
    """
    total_sum = sum(nums)
    
    if total_sum % k != 0:
        return False, []
    
    target = total_sum // k
    
    if any(num > target for num in nums):
        return False, []
    
    nums.sort(reverse=True)
    subsets = [[] for _ in range(k)]
    subset_sums = [0] * k
    
    def backtrack(index):
        if index == len(nums):
            return all(s == target for s in subset_sums)
        
        current_num = nums[index]
        
        for i in range(k):
            if subset_sums[i] + current_num <= target:
                subsets[i].append(current_num)
                subset_sums[i] += current_num
                
                if backtrack(index + 1):
                    return True
                
                subsets[i].pop()
                subset_sums[i] -= current_num
                
                # Skip equivalent empty subsets
                if subset_sums[i] == 0:
                    break
        
        return False
    
    if backtrack(0):
        return True, [subset[:] for subset in subsets]
    else:
        return False, []


# Test cases
def test_can_partition_k_subsets():
    """Test all implementations with various inputs"""
    test_cases = [
        ([4,3,2,3,5,2,1], 2, True),
        ([1,2,3,4], 3, False),
        ([1,1,1,1], 2, True),
        ([1,1,1,1], 4, True),
        ([2,2,2,2,3,3,3,3], 4, True),
        ([4,4,4,4,4,4], 3, True),
        ([1,2,3,4,5,6], 3, True),
        ([10,10,10,7,7,7,7,7,7,6,6,6], 3, True),
        ([1], 1, True),
        ([1,1], 1, False),
        ([5,2,5,5,5,5,5,5,5,5,5,5,5,5,5,3], 15, False)
    ]
    
    print("Testing Partition to K Equal Sum Subsets Solutions:")
    print("=" * 70)
    
    for i, (nums, k, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: nums = {nums}, k = {k}")
        print(f"Expected: {expected}")
        
        # Test approaches (skip expensive ones for large inputs)
        if len(nums) <= 8:
            brute = can_partition_k_subsets_bruteforce(nums.copy(), k)
            print(f"Brute Force:      {brute} {'✓' if brute == expected else '✗'}")
        
        optimized = can_partition_k_subsets_optimized_backtrack(nums.copy(), k)
        bucket = can_partition_k_subsets_bucket_filling(nums.copy(), k)
        heavy_opt = can_partition_k_subsets_optimized_pruning(nums.copy(), k)
        
        print(f"Optimized BT:     {optimized} {'✓' if optimized == expected else '✗'}")
        print(f"Bucket Filling:   {bucket} {'✓' if bucket == expected else '✗'}")
        print(f"Heavy Optimized:  {heavy_opt} {'✓' if heavy_opt == expected else '✗'}")
        
        if len(nums) <= 12:
            memo = can_partition_k_subsets_memoization(nums.copy(), k)
            print(f"Memoization:      {memo} {'✓' if memo == expected else '✗'}")
        
        if len(nums) <= 10:
            dp_mask = can_partition_k_subsets_dp_bitmask(nums.copy(), k)
            print(f"DP Bitmask:       {dp_mask} {'✓' if dp_mask == expected else '✗'}")
        
        # Show actual subsets for positive small cases
        if expected and len(nums) <= 8:
            possible, subsets = can_partition_k_subsets_with_subsets(nums.copy(), k)
            if possible:
                print(f"Subsets: {subsets}")
                print(f"Sums: {[sum(subset) for subset in subsets]}")
    
    print("\n" + "=" * 70)
    print("Algorithm Comparison:")
    print("Brute Force:      Try all k^n assignments")
    print("Optimized BT:     Add pruning (skip empty subsets)")
    print("Bucket Filling:   Fill buckets sequentially")
    print("Memoization:      Cache states with bitmask")
    print("DP Bitmask:       Bottom-up with bitmask")
    print("Heavy Optimized:  Multiple pruning techniques")
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Brute Force:      Time: O(k^n),      Space: O(n)")
    print("Optimized BT:     Time: O(k^n),      Space: O(n) - better with pruning")
    print("Bucket Filling:   Time: O(k*2^n),    Space: O(n)")
    print("Memoization:      Time: O(n*2^n),    Space: O(2^n)")
    print("DP Bitmask:       Time: O(2^n*n),    Space: O(2^n)")
    print("Heavy Optimized:  Time: O(k^n),      Space: O(n) - best pruning")


if __name__ == "__main__":
    test_can_partition_k_subsets()


"""
PATTERN RECOGNITION:
==================
This is a multi-way partition problem with equal sums:
- Generalization of 2-way partition to k-way partition
- Combines backtracking with dynamic programming
- Multiple algorithmic approaches with different trade-offs

KEY INSIGHTS:
============
1. **Constraint Check**: total_sum must be divisible by k
2. **Target Sum**: Each subset must have sum = total_sum / k
3. **Early Termination**: If any number > target, impossible
4. **Algorithm Choice**: Backtracking vs DP vs Hybrid approaches

MULTIPLE SOLUTION APPROACHES:
============================

1. **Brute Force Backtracking**: O(k^n)
   - Try assigning each number to each of k subsets
   - Simple but exponential

2. **Optimized Backtracking**: O(k^n) with pruning
   - Sort descending for better pruning
   - Skip equivalent empty subsets
   - Much better practical performance

3. **Bucket Filling**: O(k × 2^n)
   - Fill buckets one by one
   - Different search strategy

4. **Memoization + Bitmask**: O(n × 2^n)
   - Use bitmask to represent used numbers
   - Cache (mask, current_sum) states

5. **Bottom-up DP**: O(2^n × n)
   - Build solution using bitmask DP
   - No recursion overhead

OPTIMIZATION TECHNIQUES:
=======================

1. **Sorting**: Process large numbers first for better pruning
2. **Empty Subset Pruning**: Skip equivalent empty subsets
3. **Early Termination**: Check divisibility and max element
4. **State Compression**: Use bitmask for subset representation
5. **Memoization**: Cache computed states

ALGORITHM SELECTION:
===================
- **Small n, small k**: Optimized backtracking
- **Small n, large k**: DP with bitmask  
- **Medium n**: Bucket filling
- **Need actual subsets**: Backtracking with tracking

STATE REPRESENTATION:
====================
- **Backtracking**: Current subset sums array
- **Bitmask DP**: Used numbers + current subset sum
- **Bucket Filling**: Boolean used array

PRUNING STRATEGIES:
==================
1. **Symmetry Breaking**: Skip equivalent empty subsets
2. **Ordering**: Sort numbers descending
3. **Early Termination**: Check impossible cases upfront
4. **Bound Checking**: Don't exceed target sum

VARIANTS TO PRACTICE:
====================
- Partition Equal Subset Sum (416) - k=2 case
- Target Sum (494) - assign +/- signs
- Fair Distribution of Cookies (2305) - similar partitioning
- Subset Sum problems - foundation techniques

INTERVIEW TIPS:
==============
1. Start with constraint checking (divisibility, max element)
2. Show brute force backtracking approach first
3. Add pruning optimizations step by step
4. Explain bitmask DP as alternative for small n
5. Discuss time/space trade-offs between approaches
6. Handle edge cases (k=1, k=n, impossible cases)
7. Consider which approach fits the constraints best
"""
