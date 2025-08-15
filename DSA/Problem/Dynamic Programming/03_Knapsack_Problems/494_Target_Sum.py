"""
LeetCode 494: Target Sum
Difficulty: Medium
Category: Knapsack Problems (0/1 Knapsack variant)

PROBLEM DESCRIPTION:
===================
You are given an integer array nums and an integer target.

You want to build an expression out of nums by adding one of the symbols '+' and '-' 
before each integer in nums and then concatenate all the integers.

For example, if nums = [2, 1], you can add a '+' before 2 and a '-' before 1 and 
concatenate them to build the expression "+2-1".

Return the number of different expressions that you can build, which evaluates to target.

Example 1:
Input: nums = [1,1,1,1,1], target = 3
Output: 5
Explanation: There are 5 ways to assign symbols to make the sum of nums be target 3.
-1 + 1 + 1 + 1 + 1 = 3
+1 - 1 + 1 + 1 + 1 = 3
+1 + 1 - 1 + 1 + 1 = 3
+1 + 1 + 1 - 1 + 1 = 3
+1 + 1 + 1 + 1 - 1 = 3

Example 2:
Input: nums = [1], target = 1
Output: 1

Constraints:
- 1 <= nums.length <= 20
- 0 <= nums[i] <= 1000
- 0 <= sum(nums[i]) <= 1000
- -1000 <= target <= 1000
"""

def find_target_sum_ways_bruteforce(nums, target):
    """
    BRUTE FORCE APPROACH - RECURSION:
    ================================
    Try all possible combinations of + and - signs.
    
    Time Complexity: O(2^n) - two choices for each number
    Space Complexity: O(n) - recursion stack depth
    """
    def count_ways(index, current_sum):
        if index == len(nums):
            return 1 if current_sum == target else 0
        
        # Two choices: add or subtract current number
        positive = count_ways(index + 1, current_sum + nums[index])
        negative = count_ways(index + 1, current_sum - nums[index])
        
        return positive + negative
    
    return count_ways(0, 0)


def find_target_sum_ways_memoization(nums, target):
    """
    DYNAMIC PROGRAMMING - TOP DOWN (MEMOIZATION):
    ============================================
    Use memoization to avoid recalculating same subproblems.
    
    Time Complexity: O(n * sum) - each subproblem calculated once
    Space Complexity: O(n * sum) - memoization table + recursion stack
    """
    memo = {}
    
    def count_ways(index, current_sum):
        if index == len(nums):
            return 1 if current_sum == target else 0
        
        if (index, current_sum) in memo:
            return memo[(index, current_sum)]
        
        positive = count_ways(index + 1, current_sum + nums[index])
        negative = count_ways(index + 1, current_sum - nums[index])
        
        memo[(index, current_sum)] = positive + negative
        return memo[(index, current_sum)]
    
    return count_ways(0, 0)


def find_target_sum_ways_tabulation_naive(nums, target):
    """
    DYNAMIC PROGRAMMING - BOTTOM UP (NAIVE TABULATION):
    ==================================================
    Use 2D DP table with offset for negative sums.
    dp[i][j] = number of ways to achieve sum j using first i numbers
    
    Time Complexity: O(n * range) - where range is sum of all numbers
    Space Complexity: O(n * range) - 2D DP table
    """
    total_sum = sum(nums)
    
    # Check if target is achievable
    if abs(target) > total_sum:
        return 0
    
    n = len(nums)
    # Offset to handle negative indices (sum can range from -total_sum to +total_sum)
    offset = total_sum
    dp_size = 2 * total_sum + 1
    
    # dp[i][j] = ways to achieve sum (j - offset) using first i numbers
    dp = [[0] * dp_size for _ in range(n + 1)]
    
    # Base case: one way to achieve sum 0 with 0 numbers
    dp[0][offset] = 1
    
    for i in range(1, n + 1):
        for j in range(dp_size):
            current_sum = j - offset
            
            # Add current number
            if j - nums[i - 1] >= 0:
                dp[i][j] += dp[i - 1][j - nums[i - 1]]
            
            # Subtract current number
            if j + nums[i - 1] < dp_size:
                dp[i][j] += dp[i - 1][j + nums[i - 1]]
    
    return dp[n][target + offset]


def find_target_sum_ways_subset_sum(nums, target):
    """
    SUBSET SUM TRANSFORMATION (OPTIMAL):
    ===================================
    Transform problem to subset sum problem.
    Let P = positive subset, N = negative subset
    P + N = sum(nums), P - N = target
    Solving: P = (sum + target) / 2
    
    Time Complexity: O(n * sum) - single DP computation
    Space Complexity: O(sum) - 1D DP array
    """
    total_sum = sum(nums)
    
    # Check if transformation is valid
    if target > total_sum or target < -total_sum:
        return 0
    if (total_sum + target) % 2 != 0:
        return 0
    
    # Find number of ways to select subset with sum = (total_sum + target) / 2
    subset_sum = (total_sum + target) // 2
    
    # Standard subset sum DP
    dp = [0] * (subset_sum + 1)
    dp[0] = 1  # One way to achieve sum 0 (empty subset)
    
    for num in nums:
        # Traverse backwards to avoid using updated values
        for j in range(subset_sum, num - 1, -1):
            dp[j] += dp[j - num]
    
    return dp[subset_sum]


def find_target_sum_ways_optimized_memo(nums, target):
    """
    OPTIMIZED MEMOIZATION WITH PRUNING:
    ==================================
    Enhanced memoization with early termination and pruning.
    
    Time Complexity: O(n * sum) - worst case, often better with pruning
    Space Complexity: O(n * sum) - memoization table
    """
    total_sum = sum(nums)
    
    # Early termination checks
    if abs(target) > total_sum:
        return 0
    
    memo = {}
    
    def count_ways(index, current_sum):
        # Early termination: impossible to reach target
        remaining_sum = sum(nums[index:])
        if abs(current_sum - target) > remaining_sum:
            return 0
        
        if index == len(nums):
            return 1 if current_sum == target else 0
        
        if (index, current_sum) in memo:
            return memo[(index, current_sum)]
        
        # Only try positive if it doesn't overshoot
        positive = 0
        if current_sum + nums[index] <= target + remaining_sum:
            positive = count_ways(index + 1, current_sum + nums[index])
        
        # Only try negative if it doesn't undershoot
        negative = 0
        if current_sum - nums[index] >= target - remaining_sum:
            negative = count_ways(index + 1, current_sum - nums[index])
        
        memo[(index, current_sum)] = positive + negative
        return memo[(index, current_sum)]
    
    return count_ways(0, 0)


def find_target_sum_ways_bitmask(nums, target):
    """
    BITMASK APPROACH:
    ================
    Use bitmask to represent all possible sign combinations.
    
    Time Complexity: O(2^n) - try all combinations
    Space Complexity: O(1) - constant space
    """
    n = len(nums)
    count = 0
    
    # Try all 2^n possible combinations
    for mask in range(1 << n):
        current_sum = 0
        
        for i in range(n):
            if mask & (1 << i):
                current_sum += nums[i]  # Use positive sign
            else:
                current_sum -= nums[i]  # Use negative sign
        
        if current_sum == target:
            count += 1
    
    return count


def find_target_sum_ways_with_expressions(nums, target):
    """
    FIND ALL VALID EXPRESSIONS:
    ===========================
    Return both count and all valid expressions.
    
    Time Complexity: O(2^n) - generate all expressions
    Space Complexity: O(2^n) - store all expressions
    """
    expressions = []
    
    def generate_expressions(index, current_sum, current_expr):
        if index == len(nums):
            if current_sum == target:
                expressions.append(current_expr)
            return
        
        # Try positive sign
        generate_expressions(index + 1, current_sum + nums[index], 
                           current_expr + f"+{nums[index]}")
        
        # Try negative sign
        generate_expressions(index + 1, current_sum - nums[index], 
                           current_expr + f"-{nums[index]}")
    
    generate_expressions(0, 0, "")
    
    # Clean up expressions (remove leading +)
    cleaned_expressions = []
    for expr in expressions:
        if expr.startswith('+'):
            cleaned_expressions.append(expr[1:])
        else:
            cleaned_expressions.append(expr)
    
    return len(expressions), cleaned_expressions


# Test cases
def test_find_target_sum_ways():
    """Test all implementations with various inputs"""
    test_cases = [
        ([1, 1, 1, 1, 1], 3, 5),
        ([1], 1, 1),
        ([1], 2, 0),
        ([1, 0], 1, 2),
        ([100], -200, 0),
        ([1, 1, 2], 0, 2),
        ([1, 2, 3], 0, 2),
        ([2, 2, 2], 2, 3)
    ]
    
    print("Testing Target Sum Solutions:")
    print("=" * 70)
    
    for i, (nums, target, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: nums = {nums}, target = {target}")
        print(f"Expected: {expected}")
        
        # Test approaches (skip brute force for large inputs)
        if len(nums) <= 10:
            brute = find_target_sum_ways_bruteforce(nums.copy(), target)
            print(f"Brute Force:      {brute:>3} {'✓' if brute == expected else '✗'}")
        
        memo = find_target_sum_ways_memoization(nums.copy(), target)
        tab = find_target_sum_ways_tabulation_naive(nums.copy(), target)
        subset = find_target_sum_ways_subset_sum(nums.copy(), target)
        opt_memo = find_target_sum_ways_optimized_memo(nums.copy(), target)
        
        print(f"Memoization:      {memo:>3} {'✓' if memo == expected else '✗'}")
        print(f"Tabulation:       {tab:>3} {'✓' if tab == expected else '✗'}")
        print(f"Subset Sum:       {subset:>3} {'✓' if subset == expected else '✗'}")
        print(f"Optimized Memo:   {opt_memo:>3} {'✓' if opt_memo == expected else '✗'}")
        
        if len(nums) <= 8:
            bitmask = find_target_sum_ways_bitmask(nums.copy(), target)
            print(f"Bitmask:          {bitmask:>3} {'✓' if bitmask == expected else '✗'}")
        
        # Show expressions for small inputs
        if expected > 0 and len(nums) <= 5:
            count, expressions = find_target_sum_ways_with_expressions(nums.copy(), target)
            print(f"Expressions: {expressions}")
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Brute Force:      Time: O(2^n),      Space: O(n)")
    print("Memoization:      Time: O(n*sum),    Space: O(n*sum)")
    print("Tabulation:       Time: O(n*sum),    Space: O(n*sum)")
    print("Subset Sum:       Time: O(n*sum),    Space: O(sum)")
    print("Optimized Memo:   Time: O(n*sum),    Space: O(n*sum) - with pruning")
    print("Bitmask:          Time: O(2^n),      Space: O(1)")


if __name__ == "__main__":
    test_find_target_sum_ways()


"""
PATTERN RECOGNITION:
==================
This is a counting DP problem that can be transformed to subset sum:
- Assign + or - to each number to reach target
- Count number of ways to achieve target sum
- Can be transformed to: find subsets with specific sum

KEY INSIGHT - SUBSET SUM TRANSFORMATION:
=======================================
Let P = sum of positive numbers, N = sum of negative numbers
P + N = total_sum (sum of all numbers)
P - N = target (our goal)

Solving the system:
P = (total_sum + target) / 2
N = (total_sum - target) / 2

So we need to count subsets with sum = (total_sum + target) / 2

STATE DEFINITION:
================
Original: dp[i][sum] = ways to achieve sum using first i numbers
Transformed: dp[i] = ways to select subset with sum i

RECURRENCE RELATION:
===================
For each number, we can either include it (+) or exclude it (-)
dp[i][sum] = dp[i-1][sum+nums[i]] + dp[i-1][sum-nums[i]]

After transformation to subset sum:
dp[sum] += dp[sum - nums[i]] for each number

OPTIMIZATION TECHNIQUES:
=======================
1. Transform to subset sum problem (most efficient)
2. Use offset for handling negative sums in tabulation
3. Memoization with pruning for impossible paths
4. Early termination when target is unreachable

VARIANTS TO PRACTICE:
====================
- Partition Equal Subset Sum (416) - similar transformation
- Last Stone Weight II (1049) - minimize difference
- Ones and Zeroes (474) - 2D knapsack
- Expression Add Operators (282) - similar expression building

INTERVIEW TIPS:
==============
1. Recognize as DP counting problem
2. Show brute force first (try all + and - combinations)
3. Add memoization to optimize overlapping subproblems
4. **Key insight**: Transform to subset sum problem
5. Derive the mathematical transformation step by step
6. Handle edge cases (impossible targets, empty array)
7. Discuss space optimization for subset sum approach
"""
