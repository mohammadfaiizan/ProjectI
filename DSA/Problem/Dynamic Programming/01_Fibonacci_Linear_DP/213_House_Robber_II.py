"""
LeetCode 213: House Robber II
Difficulty: Medium
Category: Fibonacci & Linear DP

PROBLEM DESCRIPTION:
===================
You are a professional robber planning to rob houses along a street. Each house has a certain 
amount of money stashed. All houses at this street are arranged in a circle. That means the 
first house is the neighbor of the last house. Meanwhile, adjacent houses have security systems 
connected and it will automatically contact the police if two adjacent houses were broken into 
on the same night.

Given an integer array nums representing the amount of money of each house, return the maximum 
amount of money you can rob tonight without alerting the police.

Example 1:
Input: nums = [2,3,2]
Output: 3
Explanation: You cannot rob house 1 (money = 2) and then rob house 3 (money = 2), because they are adjacent houses.

Example 2:
Input: nums = [1,2,3,1]
Output: 4
Explanation: Rob house 1 (money = 1) and then rob house 3 (money = 3).
Total amount you can rob = 1 + 3 = 4.

Example 3:
Input: nums = [1,2,3]
Output: 3

Constraints:
- 1 <= nums.length <= 100
- 0 <= nums[i] <= 1000
"""

def rob_circular_bruteforce(nums):
    """
    BRUTE FORCE APPROACH - RECURSION:
    ================================
    Try all possible combinations considering circular constraint.
    Split into two cases: include first house or exclude first house.
    
    Time Complexity: O(2^n) - exponential due to overlapping subproblems
    Space Complexity: O(n) - recursion stack depth
    """
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    if len(nums) == 2:
        return max(nums)
    
    def rob_linear(houses, start, end):
        """Rob houses in linear arrangement from start to end"""
        def rob_from(index):
            if index > end:
                return 0
            if index == end:
                return houses[index]
            
            # Two choices: rob current or skip
            rob_current = houses[index] + rob_from(index + 2)
            skip_current = rob_from(index + 1)
            
            return max(rob_current, skip_current)
        
        return rob_from(start)
    
    # Case 1: Rob first house (cannot rob last house)
    case1 = rob_linear(nums, 0, len(nums) - 2)
    
    # Case 2: Don't rob first house (can rob last house)
    case2 = rob_linear(nums, 1, len(nums) - 1)
    
    return max(case1, case2)


def rob_circular_memoization(nums):
    """
    DYNAMIC PROGRAMMING - TOP DOWN (MEMOIZATION):
    ============================================
    Use memoization for both linear cases.
    
    Time Complexity: O(n) - each subproblem calculated once
    Space Complexity: O(n) - memoization table + recursion stack
    """
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    if len(nums) == 2:
        return max(nums)
    
    def rob_linear_memo(houses, start, end):
        memo = {}
        
        def rob_from(index):
            if index > end:
                return 0
            if index == end:
                return houses[index]
            
            if index in memo:
                return memo[index]
            
            rob_current = houses[index] + rob_from(index + 2)
            skip_current = rob_from(index + 1)
            
            memo[index] = max(rob_current, skip_current)
            return memo[index]
        
        return rob_from(start)
    
    # Case 1: Include first house (exclude last)
    case1 = rob_linear_memo(nums, 0, len(nums) - 2)
    
    # Case 2: Exclude first house (can include last)
    case2 = rob_linear_memo(nums, 1, len(nums) - 1)
    
    return max(case1, case2)


def rob_circular_tabulation(nums):
    """
    DYNAMIC PROGRAMMING - BOTTOM UP (TABULATION):
    =============================================
    Solve two linear house robber problems and take maximum.
    
    Time Complexity: O(n) - two linear passes
    Space Complexity: O(n) - DP arrays for both cases
    """
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    if len(nums) == 2:
        return max(nums)
    
    def rob_linear(houses):
        """Standard house robber for linear arrangement"""
        n = len(houses)
        if n == 0:
            return 0
        if n == 1:
            return houses[0]
        
        dp = [0] * n
        dp[0] = houses[0]
        dp[1] = max(houses[0], houses[1])
        
        for i in range(2, n):
            dp[i] = max(dp[i - 1], dp[i - 2] + houses[i])
        
        return dp[n - 1]
    
    # Case 1: Rob houses 0 to n-2 (include first, exclude last)
    case1 = rob_linear(nums[:-1])
    
    # Case 2: Rob houses 1 to n-1 (exclude first, include last)
    case2 = rob_linear(nums[1:])
    
    return max(case1, case2)


def rob_circular_space_optimized(nums):
    """
    SPACE OPTIMIZED DYNAMIC PROGRAMMING:
    ===================================
    Use O(1) space for both linear cases.
    
    Time Complexity: O(n) - two linear passes
    Space Complexity: O(1) - constant space
    """
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    if len(nums) == 2:
        return max(nums)
    
    def rob_linear_optimized(houses):
        """Space-optimized linear house robber"""
        if not houses:
            return 0
        if len(houses) == 1:
            return houses[0]
        
        prev2 = houses[0]  # dp[i-2]
        prev1 = max(houses[0], houses[1])  # dp[i-1]
        
        for i in range(2, len(houses)):
            current = max(prev1, prev2 + houses[i])
            prev2 = prev1
            prev1 = current
        
        return prev1
    
    # Case 1: Houses 0 to n-2
    case1 = rob_linear_optimized(nums[:-1])
    
    # Case 2: Houses 1 to n-1
    case2 = rob_linear_optimized(nums[1:])
    
    return max(case1, case2)


def rob_circular_single_pass(nums):
    """
    SINGLE PASS APPROACH:
    ====================
    Calculate both cases in a single pass through the array.
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(1) - constant space
    """
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    if len(nums) == 2:
        return max(nums)
    
    n = len(nums)
    
    # Case 1: Include first house (exclude last)
    prev2_case1 = nums[0]
    prev1_case1 = max(nums[0], nums[1])
    
    # Case 2: Exclude first house (can include last)
    prev2_case2 = 0
    prev1_case2 = nums[1]
    
    for i in range(2, n):
        if i < n - 1:  # Not the last house
            # Case 1: can include this house
            current_case1 = max(prev1_case1, prev2_case1 + nums[i])
            prev2_case1 = prev1_case1
            prev1_case1 = current_case1
        
        # Case 2: can always include this house
        current_case2 = max(prev1_case2, prev2_case2 + nums[i])
        prev2_case2 = prev1_case2
        prev1_case2 = current_case2
    
    return max(prev1_case1, prev1_case2)


def rob_circular_with_houses(nums):
    """
    FIND MAXIMUM MONEY AND ROBBED HOUSES:
    ====================================
    Return both maximum money and which houses were robbed.
    
    Time Complexity: O(n) - DP + house reconstruction
    Space Complexity: O(n) - DP arrays and house tracking
    """
    if not nums:
        return 0, []
    if len(nums) == 1:
        return nums[0], [0]
    if len(nums) == 2:
        if nums[0] >= nums[1]:
            return nums[0], [0]
        else:
            return nums[1], [1]
    
    def rob_linear_with_houses(houses, offset=0):
        n = len(houses)
        if n == 0:
            return 0, []
        if n == 1:
            return houses[0], [offset]
        
        dp = [0] * n
        parent = [-1] * n
        
        dp[0] = houses[0]
        dp[1] = max(houses[0], houses[1])
        parent[1] = 0 if houses[0] >= houses[1] else -1
        
        for i in range(2, n):
            if dp[i - 1] >= dp[i - 2] + houses[i]:
                dp[i] = dp[i - 1]
                parent[i] = i - 1
            else:
                dp[i] = dp[i - 2] + houses[i]
                parent[i] = i - 2
        
        # Reconstruct robbed houses
        robbed = []
        i = n - 1
        while i >= 0:
            if i == 0 or (i >= 2 and parent[i] == i - 2):
                robbed.append(i + offset)
                i -= 2
            else:
                i = parent[i]
        
        robbed.reverse()
        return dp[n - 1], robbed
    
    # Case 1: Include first house (exclude last)
    money1, houses1 = rob_linear_with_houses(nums[:-1], 0)
    
    # Case 2: Exclude first house (include last possible)
    money2, houses2 = rob_linear_with_houses(nums[1:], 1)
    
    if money1 >= money2:
        return money1, houses1
    else:
        return money2, houses2


def rob_circular_dp_states(nums):
    """
    STATE MACHINE APPROACH:
    ======================
    Track different states: robbed first house or not.
    
    Time Complexity: O(n) - single pass with state tracking
    Space Complexity: O(1) - constant space
    """
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    if len(nums) == 2:
        return max(nums)
    
    n = len(nums)
    
    # State 1: Robbed first house
    rob_first_prev2 = nums[0]
    rob_first_prev1 = nums[0]  # Can't rob second house
    
    # State 2: Didn't rob first house
    not_rob_first_prev2 = 0
    not_rob_first_prev1 = nums[1]
    
    for i in range(2, n):
        if i == n - 1:  # Last house
            # If robbed first, can't rob last
            rob_first_current = rob_first_prev1
            
            # If didn't rob first, can rob last
            not_rob_first_current = max(not_rob_first_prev1, 
                                      not_rob_first_prev2 + nums[i])
        else:
            # Normal processing for middle houses
            rob_first_current = max(rob_first_prev1, 
                                  rob_first_prev2 + nums[i])
            not_rob_first_current = max(not_rob_first_prev1, 
                                      not_rob_first_prev2 + nums[i])
        
        # Update previous values
        rob_first_prev2 = rob_first_prev1
        rob_first_prev1 = rob_first_current
        
        not_rob_first_prev2 = not_rob_first_prev1
        not_rob_first_prev1 = not_rob_first_current
    
    return max(rob_first_prev1, not_rob_first_prev1)


# Test cases
def test_rob_circular():
    """Test all implementations with various inputs"""
    test_cases = [
        ([2, 3, 2], 3),
        ([1, 2, 3, 1], 4),
        ([1, 2, 3], 3),
        ([1], 1),
        ([1, 2], 2),
        ([5, 1, 3, 9], 10),
        ([2, 7, 9, 3, 1], 11),
        ([1, 3, 1, 3, 100], 103),
        ([94, 40, 49, 65, 21, 21, 106, 80, 92, 81, 679, 4, 61, 6, 237, 12, 72, 74, 29, 95, 265, 35, 47, 1, 61, 397, 52, 72, 37, 51, 1, 81, 45, 435, 7, 36, 57, 86, 81, 72], 2926)
    ]
    
    print("Testing House Robber II (Circular) Solutions:")
    print("=" * 70)
    
    for i, (nums, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: {nums[:10]}{'...' if len(nums) > 10 else ''}")
        print(f"Expected: {expected}")
        
        # Test approaches (skip brute force for large inputs)
        if len(nums) <= 10:
            brute = rob_circular_bruteforce(nums.copy())
            print(f"Brute Force:      {brute:>4} {'✓' if brute == expected else '✗'}")
        
        memo = rob_circular_memoization(nums.copy())
        tab = rob_circular_tabulation(nums.copy())
        space_opt = rob_circular_space_optimized(nums.copy())
        single_pass = rob_circular_single_pass(nums.copy())
        dp_states = rob_circular_dp_states(nums.copy())
        
        print(f"Memoization:      {memo:>4} {'✓' if memo == expected else '✗'}")
        print(f"Tabulation:       {tab:>4} {'✓' if tab == expected else '✗'}")
        print(f"Space Optimized:  {space_opt:>4} {'✓' if space_opt == expected else '✗'}")
        print(f"Single Pass:      {single_pass:>4} {'✓' if single_pass == expected else '✗'}")
        print(f"DP States:        {dp_states:>4} {'✓' if dp_states == expected else '✗'}")
        
        # Show robbed houses for small examples
        if len(nums) <= 8:
            money, houses = rob_circular_with_houses(nums.copy())
            print(f"Robbed houses: {houses} (money: {money})")
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Brute Force:      Time: O(2^n),   Space: O(n)")
    print("Memoization:      Time: O(n),     Space: O(n)")
    print("Tabulation:       Time: O(n),     Space: O(n)")
    print("Space Optimized:  Time: O(n),     Space: O(1)")
    print("Single Pass:      Time: O(n),     Space: O(1)")
    print("DP States:        Time: O(n),     Space: O(1)")


if __name__ == "__main__":
    test_rob_circular()


"""
PATTERN RECOGNITION:
==================
This is a variation of House Robber with circular constraint:
- Houses arranged in circle (first and last are adjacent)
- Cannot rob adjacent houses
- Maximize total money robbed
- Key insight: Split into two linear subproblems

KEY INSIGHT - PROBLEM DECOMPOSITION:
===================================
Circular constraint means first and last house are adjacent.
Two mutually exclusive cases:
1. Rob first house → Cannot rob last house → Linear problem on houses [0...n-2]
2. Don't rob first house → Can rob last house → Linear problem on houses [1...n-1]

Answer = max(case1, case2)

STATE DEFINITION:
================
Same as linear house robber, but applied to two different subarrays:
dp[i] = maximum money from houses 0 to i (without robbing adjacent)

RECURRENCE RELATION:
===================
For each linear subproblem:
dp[i] = max(dp[i-1], dp[i-2] + nums[i])

Base cases:
- dp[0] = nums[0]
- dp[1] = max(nums[0], nums[1])

OPTIMIZATION TECHNIQUES:
=======================
1. Two separate linear DP solutions
2. Space optimization: O(1) space for each subproblem
3. Single pass: calculate both cases simultaneously
4. State machine: track "robbed first" vs "didn't rob first"

VARIANTS TO PRACTICE:
====================
- House Robber (198) - linear version
- House Robber III (337) - binary tree version
- Delete and Earn (740) - similar constraint pattern
- Paint House (256) - circular constraint with multiple choices

INTERVIEW TIPS:
==============
1. Identify circular constraint as key challenge
2. Transform to two linear subproblems
3. Explain why cases are mutually exclusive
4. Show how to reuse linear house robber solution
5. Optimize space for each subproblem
6. Handle edge cases (1 house, 2 houses)
7. Discuss state machine approach for advanced solution
"""
