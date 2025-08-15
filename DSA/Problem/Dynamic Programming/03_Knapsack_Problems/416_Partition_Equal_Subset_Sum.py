"""
LeetCode 416: Partition Equal Subset Sum
Difficulty: Medium
Category: Knapsack Problems (0/1 Knapsack variant)

PROBLEM DESCRIPTION:
===================
Given a non-empty array nums containing only positive integers, find if the array can be 
partitioned into two subsets such that the sum of elements in both subsets is equal.

Example 1:
Input: nums = [1,5,11,5]
Output: true
Explanation: The array can be partitioned as [1, 5, 5] and [11].

Example 2:
Input: nums = [1,2,3,5]
Output: false
Explanation: The array cannot be partitioned into equal sum subsets.

Constraints:
- 1 <= nums.length <= 200
- 1 <= nums[i] <= 100
"""

def can_partition_bruteforce(nums):
    """
    BRUTE FORCE APPROACH - RECURSION:
    ================================
    Try all possible subsets to see if any has sum equal to total_sum // 2.
    
    Time Complexity: O(2^n) - check all possible subsets
    Space Complexity: O(n) - recursion stack depth
    """
    total_sum = sum(nums)
    
    # If total sum is odd, cannot partition into two equal subsets
    if total_sum % 2 != 0:
        return False
    
    target = total_sum // 2
    
    def can_achieve_sum(index, current_sum):
        # Base cases
        if current_sum == target:
            return True
        if current_sum > target or index >= len(nums):
            return False
        
        # Two choices: include current number or exclude it
        include = can_achieve_sum(index + 1, current_sum + nums[index])
        exclude = can_achieve_sum(index + 1, current_sum)
        
        return include or exclude
    
    return can_achieve_sum(0, 0)


def can_partition_memoization(nums):
    """
    DYNAMIC PROGRAMMING - TOP DOWN (MEMOIZATION):
    ============================================
    Use memoization to avoid recalculating same subproblems.
    
    Time Complexity: O(n * sum) - each subproblem calculated once
    Space Complexity: O(n * sum) - memoization table + recursion stack
    """
    total_sum = sum(nums)
    
    if total_sum % 2 != 0:
        return False
    
    target = total_sum // 2
    memo = {}
    
    def can_achieve_sum(index, current_sum):
        if current_sum == target:
            return True
        if current_sum > target or index >= len(nums):
            return False
        
        if (index, current_sum) in memo:
            return memo[(index, current_sum)]
        
        include = can_achieve_sum(index + 1, current_sum + nums[index])
        exclude = can_achieve_sum(index + 1, current_sum)
        
        memo[(index, current_sum)] = include or exclude
        return memo[(index, current_sum)]
    
    return can_achieve_sum(0, 0)


def can_partition_tabulation(nums):
    """
    DYNAMIC PROGRAMMING - BOTTOM UP (TABULATION):
    =============================================
    Build solution from bottom up using 2D DP table.
    dp[i][j] = True if subset sum j can be achieved using first i numbers
    
    Time Complexity: O(n * sum) - fill entire DP table
    Space Complexity: O(n * sum) - 2D DP table
    """
    total_sum = sum(nums)
    
    if total_sum % 2 != 0:
        return False
    
    target = total_sum // 2
    n = len(nums)
    
    # dp[i][j] = True if sum j can be achieved using first i numbers
    dp = [[False] * (target + 1) for _ in range(n + 1)]
    
    # Base case: sum 0 can always be achieved (empty subset)
    for i in range(n + 1):
        dp[i][0] = True
    
    # Fill the DP table
    for i in range(1, n + 1):
        for j in range(1, target + 1):
            # Don't include current number
            dp[i][j] = dp[i - 1][j]
            
            # Include current number if possible
            if j >= nums[i - 1]:
                dp[i][j] = dp[i][j] or dp[i - 1][j - nums[i - 1]]
    
    return dp[n][target]


def can_partition_space_optimized(nums):
    """
    SPACE OPTIMIZED DYNAMIC PROGRAMMING:
    ===================================
    Since we only need previous row, use 1D array instead of 2D.
    
    Time Complexity: O(n * sum) - same number of operations
    Space Complexity: O(sum) - only store one row
    """
    total_sum = sum(nums)
    
    if total_sum % 2 != 0:
        return False
    
    target = total_sum // 2
    
    # dp[j] = True if sum j can be achieved
    dp = [False] * (target + 1)
    dp[0] = True  # sum 0 can always be achieved
    
    # Process each number
    for num in nums:
        # Traverse backwards to avoid using updated values
        for j in range(target, num - 1, -1):
            dp[j] = dp[j] or dp[j - num]
    
    return dp[target]


def can_partition_bitset(nums):
    """
    BITSET OPTIMIZATION:
    ===================
    Use bitset to track possible sums efficiently.
    
    Time Complexity: O(n * sum / word_size) - bitwise operations
    Space Complexity: O(sum / word_size) - bitset storage
    """
    total_sum = sum(nums)
    
    if total_sum % 2 != 0:
        return False
    
    target = total_sum // 2
    
    # Use integer as bitset to track possible sums
    # bit i is set if sum i is achievable
    possible_sums = 1  # sum 0 is achievable (bit 0 set)
    
    for num in nums:
        # For each existing sum, we can also achieve sum + num
        possible_sums |= (possible_sums << num)
    
    # Check if target sum is achievable
    return bool(possible_sums & (1 << target))


def can_partition_early_termination(nums):
    """
    OPTIMIZED DP WITH EARLY TERMINATION:
    ===================================
    Add optimizations for early termination and pruning.
    
    Time Complexity: O(n * sum) - worst case, but often better
    Space Complexity: O(sum) - 1D DP array
    """
    total_sum = sum(nums)
    
    if total_sum % 2 != 0:
        return False
    
    target = total_sum // 2
    
    # Early termination: if any number equals target
    if target in nums:
        return True
    
    # Early termination: if largest number > target
    max_num = max(nums)
    if max_num > target:
        return False
    
    # Sort in descending order for better pruning
    nums.sort(reverse=True)
    
    dp = [False] * (target + 1)
    dp[0] = True
    
    for num in nums:
        # Early termination: if target is already achievable
        if dp[target]:
            return True
        
        # Process in reverse order
        for j in range(target, num - 1, -1):
            if dp[j - num]:
                dp[j] = True
    
    return dp[target]


def can_partition_subset_sum_analysis(nums):
    """
    COMPLETE ANALYSIS WITH SUBSET RECONSTRUCTION:
    ============================================
    Find if partition exists and return the actual subsets.
    
    Time Complexity: O(n * sum) - DP + subset reconstruction
    Space Complexity: O(n * sum) - store parent information
    """
    total_sum = sum(nums)
    
    if total_sum % 2 != 0:
        return False, None, None
    
    target = total_sum // 2
    n = len(nums)
    
    # Enhanced DP with parent tracking
    dp = [[False] * (target + 1) for _ in range(n + 1)]
    parent = [[None] * (target + 1) for _ in range(n + 1)]
    
    # Base case
    for i in range(n + 1):
        dp[i][0] = True
    
    # Fill DP table with parent tracking
    for i in range(1, n + 1):
        for j in range(1, target + 1):
            # Don't include current number
            if dp[i - 1][j]:
                dp[i][j] = True
                parent[i][j] = (i - 1, j)
            
            # Include current number if possible
            if j >= nums[i - 1] and dp[i - 1][j - nums[i - 1]]:
                dp[i][j] = True
                if parent[i][j] is None:  # Prefer inclusion if both work
                    parent[i][j] = (i - 1, j - nums[i - 1])
    
    if not dp[n][target]:
        return False, None, None
    
    # Reconstruct subset
    subset1 = []
    i, j = n, target
    
    while i > 0 and j > 0:
        prev_i, prev_j = parent[i][j]
        if prev_j != j:  # Number was included
            subset1.append(nums[i - 1])
        i, j = prev_i, prev_j
    
    # Calculate subset2
    subset2 = [num for num in nums if num not in subset1 or subset1.count(num) < nums.count(num)]
    
    return True, subset1, subset2


# Test cases
def test_can_partition():
    """Test all implementations with various inputs"""
    test_cases = [
        ([1, 5, 11, 5], True),
        ([1, 2, 3, 5], False),
        ([1, 1], True),
        ([1, 2, 5], False),
        ([1, 3, 5, 7], True),
        ([2, 2, 3, 5], False),
        ([23, 13, 11, 7, 6, 5, 5], True),
        ([1, 1, 1, 1], True),
        ([100], False)
    ]
    
    print("Testing Partition Equal Subset Sum Solutions:")
    print("=" * 70)
    
    for i, (nums, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: {nums}")
        print(f"Expected: {expected}")
        
        # Test all approaches (skip brute force for large inputs)
        if len(nums) <= 8:
            brute = can_partition_bruteforce(nums.copy())
            print(f"Brute Force:      {brute} {'✓' if brute == expected else '✗'}")
        
        memo = can_partition_memoization(nums.copy())
        tab = can_partition_tabulation(nums.copy())
        space_opt = can_partition_space_optimized(nums.copy())
        bitset = can_partition_bitset(nums.copy())
        early_term = can_partition_early_termination(nums.copy())
        
        print(f"Memoization:      {memo} {'✓' if memo == expected else '✗'}")
        print(f"Tabulation:       {tab} {'✓' if tab == expected else '✗'}")
        print(f"Space Optimized:  {space_opt} {'✓' if space_opt == expected else '✗'}")
        print(f"Bitset:           {bitset} {'✓' if bitset == expected else '✗'}")
        print(f"Early Term:       {early_term} {'✓' if early_term == expected else '✗'}")
        
        # Show subset analysis for positive cases
        if expected and len(nums) <= 6:
            can_part, s1, s2 = can_partition_subset_sum_analysis(nums.copy())
            if can_part:
                print(f"Subset 1: {s1}, Sum: {sum(s1) if s1 else 0}")
                print(f"Subset 2: {s2}, Sum: {sum(s2) if s2 else 0}")
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Brute Force:      Time: O(2^n),        Space: O(n)")
    print("Memoization:      Time: O(n*sum),      Space: O(n*sum)")
    print("Tabulation:       Time: O(n*sum),      Space: O(n*sum)")
    print("Space Optimized:  Time: O(n*sum),      Space: O(sum)")
    print("Bitset:           Time: O(n*sum/w),    Space: O(sum/w)")
    print("Early Term:       Time: O(n*sum),      Space: O(sum) - often faster")


if __name__ == "__main__":
    test_can_partition()


"""
PATTERN RECOGNITION:
==================
This is a classic 0/1 Knapsack variant:
- Each number can be used at most once
- Find if subset with specific sum exists
- Transform: "partition into two equal subsets" → "find subset with sum = total/2"

KEY INSIGHTS:
============
1. If total sum is odd, partition is impossible
2. If we can find subset with sum = total/2, remaining elements have sum = total/2
3. This reduces to subset sum problem with target = total_sum/2
4. Classic 0/1 knapsack: include/exclude each element

STATE DEFINITION:
================
dp[i][j] = True if sum j can be achieved using first i numbers

RECURRENCE RELATION:
===================
dp[i][j] = dp[i-1][j] OR (j >= nums[i-1] AND dp[i-1][j-nums[i-1]])
Base case: dp[i][0] = True (empty subset has sum 0)

SPACE OPTIMIZATION:
==================
Since we only need previous row, use 1D array.
Process from right to left to avoid using updated values.

BITSET OPTIMIZATION:
===================
Use bitwise operations to track possible sums efficiently.
Each bit represents whether a sum is achievable.

VARIANTS TO PRACTICE:
====================
- Target Sum (494) - assign +/- to reach target
- Last Stone Weight II (1049) - minimize difference
- Ones and Zeroes (474) - 2D knapsack variant
- Coin Change (322) - unbounded knapsack

INTERVIEW TIPS:
==============
1. Transform problem: "equal partition" → "subset sum = total/2"
2. Identify as 0/1 knapsack variant
3. Handle edge case: odd total sum
4. Show space optimization from 2D to 1D
5. Mention bitset optimization for advanced discussion
6. Consider early termination optimizations
"""
