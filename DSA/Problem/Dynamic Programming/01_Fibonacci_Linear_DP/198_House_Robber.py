"""
LeetCode 198: House Robber
Difficulty: Medium
Category: Fibonacci & Linear DP

PROBLEM DESCRIPTION:
===================
You are a professional robber planning to rob houses along a street. Each house has a certain 
amount of money stashed, the only constraint stopping you from robbing each of them is that 
adjacent houses have security systems connected and it will automatically contact the police 
if two adjacent houses were broken into on the same night.

Given an integer array nums representing the amount of money of each house, return the maximum 
amount of money you can rob tonight without alerting the police.

Example 1:
Input: nums = [1,2,3,1]
Output: 4
Explanation: Rob house 1 (money = 1) and house 3 (money = 3).
Total amount you can rob = 1 + 3 = 4.

Example 2:
Input: nums = [2,7,9,3,1]
Output: 12
Explanation: Rob house 1 (money = 2), house 3 (money = 9) and house 5 (money = 1).
Total amount you can rob = 2 + 9 + 1 = 12.

Constraints:
- 1 <= nums.length <= 100
- 0 <= nums[i] <= 400
"""

def house_robber_bruteforce(nums):
    """
    BRUTE FORCE APPROACH:
    ====================
    Use recursion to try all possible combinations.
    At each house, we have two choices: rob it or skip it.
    
    Time Complexity: O(2^n) - exponential due to overlapping subproblems
    Space Complexity: O(n) - recursion stack depth
    """
    def rob_from(index):
        if index >= len(nums):
            return 0
        
        # Two choices: rob current house or skip it
        rob_current = nums[index] + rob_from(index + 2)  # Rob current, skip next
        skip_current = rob_from(index + 1)               # Skip current
        
        return max(rob_current, skip_current)
    
    return rob_from(0)


def house_robber_memoization(nums):
    """
    DYNAMIC PROGRAMMING - TOP DOWN (MEMOIZATION):
    ============================================
    Use memoization to avoid recalculating same subproblems.
    
    Time Complexity: O(n) - each subproblem calculated once
    Space Complexity: O(n) - memoization table + recursion stack
    """
    memo = {}
    
    def rob_from(index):
        if index >= len(nums):
            return 0
        
        if index in memo:
            return memo[index]
        
        rob_current = nums[index] + rob_from(index + 2)
        skip_current = rob_from(index + 1)
        
        memo[index] = max(rob_current, skip_current)
        return memo[index]
    
    return rob_from(0)


def house_robber_tabulation(nums):
    """
    DYNAMIC PROGRAMMING - BOTTOM UP (TABULATION):
    =============================================
    Build solution from bottom up using iteration.
    dp[i] = maximum money that can be robbed from houses 0 to i
    
    Time Complexity: O(n) - single pass through all houses
    Space Complexity: O(n) - DP table
    """
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    
    n = len(nums)
    dp = [0] * n
    dp[0] = nums[0]
    dp[1] = max(nums[0], nums[1])
    
    for i in range(2, n):
        # Either rob current house + max from houses before previous
        # Or don't rob current house and take max from previous house
        dp[i] = max(nums[i] + dp[i - 2], dp[i - 1])
    
    return dp[n - 1]


def house_robber_optimized(nums):
    """
    SPACE OPTIMIZED DYNAMIC PROGRAMMING:
    ===================================
    Since we only need previous two values, use variables instead of array.
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(1) - constant space
    """
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    
    prev2 = nums[0]              # max money up to house i-2
    prev1 = max(nums[0], nums[1])  # max money up to house i-1
    
    for i in range(2, len(nums)):
        current = max(nums[i] + prev2, prev1)
        prev2 = prev1
        prev1 = current
    
    return prev1


def house_robber_alternative(nums):
    """
    ALTERNATIVE APPROACH - STATE MACHINE:
    ====================================
    Track two states: robbed previous house or not robbed previous house.
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(1) - constant space
    """
    if not nums:
        return 0
    
    robbed = 0      # Max money if we robbed the previous house
    not_robbed = 0  # Max money if we didn't rob the previous house
    
    for money in nums:
        # If we rob current house, we couldn't have robbed previous
        new_robbed = not_robbed + money
        # If we don't rob current, we take max of previous states
        new_not_robbed = max(robbed, not_robbed)
        
        robbed = new_robbed
        not_robbed = new_not_robbed
    
    return max(robbed, not_robbed)


def house_robber_one_pass(nums):
    """
    ONE PASS ELEGANT SOLUTION:
    =========================
    Very clean implementation using two variables.
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(1) - constant space
    """
    rob = not_rob = 0
    
    for money in nums:
        rob, not_rob = not_rob + money, max(rob, not_rob)
    
    return max(rob, not_rob)


# Test cases
def test_house_robber():
    """Test all implementations with various inputs"""
    test_cases = [
        ([1, 2, 3, 1], 4),
        ([2, 7, 9, 3, 1], 12),
        ([2, 1, 1, 2], 4),
        ([5], 5),
        ([1, 2], 2),
        ([2, 3, 2], 4),
        ([1, 3, 1, 3, 100], 103),
        ([100, 1, 1, 100], 200)
    ]
    
    print("Testing House Robber Solutions:")
    print("=" * 60)
    
    for i, (nums, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: {nums}")
        print(f"Expected: {expected}")
        
        # Test all approaches (skip brute force for large inputs)
        if len(nums) <= 10:
            brute = house_robber_bruteforce(nums)
            print(f"Brute Force:    {brute} {'✓' if brute == expected else '✗'}")
        
        memo = house_robber_memoization(nums)
        tab = house_robber_tabulation(nums)
        opt = house_robber_optimized(nums)
        alt = house_robber_alternative(nums)
        one_pass = house_robber_one_pass(nums)
        
        print(f"Memoization:    {memo} {'✓' if memo == expected else '✗'}")
        print(f"Tabulation:     {tab} {'✓' if tab == expected else '✗'}")
        print(f"Optimized:      {opt} {'✓' if opt == expected else '✗'}")
        print(f"Alternative:    {alt} {'✓' if alt == expected else '✗'}")
        print(f"One Pass:       {one_pass} {'✓' if one_pass == expected else '✗'}")
    
    print("\n" + "=" * 60)
    print("Complexity Analysis:")
    print("Brute Force:     Time: O(2^n),   Space: O(n)")
    print("Memoization:     Time: O(n),     Space: O(n)")
    print("Tabulation:      Time: O(n),     Space: O(n)")
    print("Optimized:       Time: O(n),     Space: O(1)")
    print("Alternative:     Time: O(n),     Space: O(1)")
    print("One Pass:        Time: O(n),     Space: O(1)")


if __name__ == "__main__":
    test_house_robber()


"""
PATTERN RECOGNITION:
==================
This is a classic linear DP problem with constraints:
- dp[i] = max(dp[i-1], dp[i-2] + nums[i])
- Cannot select adjacent elements
- Maximize sum of selected elements

KEY INSIGHTS:
============
1. At each house, we have two choices: rob or skip
2. If we rob current house, we must skip previous house
3. If we skip current house, we take the best result up to previous house
4. This creates a recurrence relation similar to Fibonacci

STATE DEFINITION:
================
dp[i] = maximum money that can be robbed from houses 0 to i

RECURRENCE RELATION:
===================
dp[i] = max(dp[i-1], dp[i-2] + nums[i])
Base cases: dp[0] = nums[0], dp[1] = max(nums[0], nums[1])

SPACE OPTIMIZATION:
==================
Since we only need the last two values, we can use O(1) space.

VARIANTS TO PRACTICE:
====================
- House Robber II (213) - circular array
- House Robber III (337) - binary tree
- Delete and Earn (740) - similar pattern
- Paint House (256) - multiple choices at each step

INTERVIEW TIPS:
==============
1. Start with identifying the constraint (no adjacent houses)
2. Define state clearly (max money up to house i)
3. Derive recurrence relation step by step
4. Optimize space when possible
5. Handle edge cases (empty array, single house)
6. Consider the state machine approach for clarity
"""
