"""
LeetCode 746: Min Cost Climbing Stairs
Difficulty: Easy
Category: Fibonacci & Linear DP

PROBLEM DESCRIPTION:
===================
You are given an integer array cost where cost[i] is the cost of ith step on a staircase. 
Once you pay the cost, you can either climb one or two steps.

You can either start from the step with index 0, or the step with index 1.

Return the minimum cost to reach the top of the floor.

Example 1:
Input: cost = [10,15,20]
Output: 15
Explanation: You will start at index 1.
- Pay 15 and climb two steps to reach the top.
The total cost is 15.

Example 2:
Input: cost = [1,100,1,1,1,100,1,1,100,1]
Output: 6
Explanation: You will start at index 0.
- Pay 1 and climb two steps to reach index 2.
- Pay 1 and climb two steps to reach index 4.
- Pay 1 and climb two steps to reach index 6.
- Pay 1 and climb one step to reach index 7.
- Pay 1 and climb two steps to reach index 9.
- Pay 1 and climb one step to reach the top.
The total cost is 6.

Constraints:
- 2 <= cost.length <= 1000
- 0 <= cost[i] <= 999
"""

def min_cost_climbing_stairs_bruteforce(cost):
    """
    BRUTE FORCE APPROACH - RECURSION:
    ================================
    Try all possible paths from each starting position.
    
    Time Complexity: O(2^n) - exponential due to overlapping subproblems
    Space Complexity: O(n) - recursion stack depth
    """
    def min_cost_from(index):
        # Base case: reached or passed the top
        if index >= len(cost):
            return 0
        
        # Current cost + minimum of next 1 or 2 steps
        one_step = cost[index] + min_cost_from(index + 1)
        two_steps = cost[index] + min_cost_from(index + 2)
        
        return min(one_step, two_steps)
    
    # Can start from index 0 or 1
    return min(min_cost_from(0), min_cost_from(1))


def min_cost_climbing_stairs_memoization(cost):
    """
    DYNAMIC PROGRAMMING - TOP DOWN (MEMOIZATION):
    ============================================
    Use memoization to avoid recalculating same subproblems.
    
    Time Complexity: O(n) - each subproblem calculated once
    Space Complexity: O(n) - memoization table + recursion stack
    """
    memo = {}
    
    def min_cost_from(index):
        if index >= len(cost):
            return 0
        
        if index in memo:
            return memo[index]
        
        one_step = cost[index] + min_cost_from(index + 1)
        two_steps = cost[index] + min_cost_from(index + 2)
        
        memo[index] = min(one_step, two_steps)
        return memo[index]
    
    return min(min_cost_from(0), min_cost_from(1))


def min_cost_climbing_stairs_tabulation(cost):
    """
    DYNAMIC PROGRAMMING - BOTTOM UP (TABULATION):
    =============================================
    Build solution from bottom up using DP array.
    dp[i] = minimum cost to reach the top from step i
    
    Time Complexity: O(n) - single pass through array
    Space Complexity: O(n) - DP array
    """
    n = len(cost)
    dp = [0] * (n + 2)  # Extra space for steps beyond array
    
    # Fill DP array from right to left
    for i in range(n - 1, -1, -1):
        dp[i] = cost[i] + min(dp[i + 1], dp[i + 2])
    
    # Can start from step 0 or step 1
    return min(dp[0], dp[1])


def min_cost_climbing_stairs_space_optimized(cost):
    """
    SPACE OPTIMIZED DYNAMIC PROGRAMMING:
    ===================================
    Since we only need next two values, use variables instead of array.
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(1) - constant space
    """
    n = len(cost)
    
    # Variables for next two steps
    next1 = 0  # Cost to reach top from step n
    next2 = 0  # Cost to reach top from step n+1
    
    # Work backwards through the array
    for i in range(n - 1, -1, -1):
        current = cost[i] + min(next1, next2)
        next2 = next1
        next1 = current
    
    # Can start from step 0 or step 1
    return min(next1, next2)


def min_cost_climbing_stairs_forward_dp(cost):
    """
    FORWARD DP APPROACH:
    ===================
    Build solution forward: dp[i] = minimum cost to reach step i
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(n) - DP array
    """
    n = len(cost)
    if n <= 2:
        return min(cost)
    
    # dp[i] = minimum cost to reach step i
    dp = [0] * n
    dp[0] = cost[0]  # Cost to reach step 0
    dp[1] = cost[1]  # Cost to reach step 1
    
    # Fill array forward
    for i in range(2, n):
        # Can reach step i from step i-1 or i-2
        dp[i] = cost[i] + min(dp[i - 1], dp[i - 2])
    
    # To reach top, can come from last or second-to-last step
    return min(dp[n - 1], dp[n - 2])


def min_cost_climbing_stairs_forward_optimized(cost):
    """
    FORWARD DP SPACE OPTIMIZED:
    ==========================
    Forward approach with O(1) space.
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(1) - constant space
    """
    n = len(cost)
    if n <= 2:
        return min(cost)
    
    # Track cost to reach previous two steps
    prev2 = cost[0]  # Cost to reach step 0
    prev1 = cost[1]  # Cost to reach step 1
    
    for i in range(2, n):
        current = cost[i] + min(prev1, prev2)
        prev2 = prev1
        prev1 = current
    
    # Can reach top from last or second-to-last step
    return min(prev1, prev2)


def min_cost_climbing_stairs_with_path(cost):
    """
    MIN COST WITH PATH RECONSTRUCTION:
    =================================
    Find minimum cost and reconstruct the optimal path.
    
    Time Complexity: O(n) - DP + path reconstruction
    Space Complexity: O(n) - DP array and path storage
    """
    n = len(cost)
    
    # DP to find minimum costs
    dp = [float('inf')] * (n + 1)
    dp[0] = cost[0]
    dp[1] = cost[1] if n > 1 else 0
    
    # Parent tracking for path reconstruction
    parent = [-1] * (n + 1)
    
    for i in range(2, n):
        # From step i-1
        if dp[i - 1] + cost[i] < dp[i]:
            dp[i] = dp[i - 1] + cost[i]
            parent[i] = i - 1
        
        # From step i-2
        if dp[i - 2] + cost[i] < dp[i]:
            dp[i] = dp[i - 2] + cost[i]
            parent[i] = i - 2
    
    # Find optimal ending point
    if dp[n - 1] < dp[n - 2]:
        min_cost = dp[n - 1]
        end_step = n - 1
    else:
        min_cost = dp[n - 2]
        end_step = n - 2
    
    # Reconstruct path
    path = []
    current = end_step
    while current != -1:
        path.append(current)
        current = parent[current]
    path.reverse()
    
    return min_cost, path


def min_cost_climbing_stairs_alternative(cost):
    """
    ALTERNATIVE INTERPRETATION:
    ==========================
    Alternative approach treating the problem as reaching step n+1.
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(1) - constant space
    """
    n = len(cost)
    
    # Cost to reach position before first step (0) and first step
    prev2 = 0      # Cost to reach position -1 (before array)
    prev1 = 0      # Cost to reach position 0 without paying cost[0]
    
    for i in range(n):
        # To reach step i+1, either:
        # 1. Pay cost[i] and step from current position
        # 2. Skip current step (only if coming from 2 steps back)
        current = min(prev1 + cost[i], prev2 + cost[i])
        prev2 = prev1
        prev1 = current
    
    return prev1


# Test cases
def test_min_cost_climbing_stairs():
    """Test all implementations with various inputs"""
    test_cases = [
        ([10, 15, 20], 15),
        ([1, 100, 1, 1, 1, 100, 1, 1, 100, 1], 6),
        ([0, 0, 0, 1], 0),
        ([1, 2], 1),
        ([0, 1, 2, 2], 2),
        ([5, 10], 5),
        ([1, 0, 0, 0], 0),
        ([0, 0, 1, 1], 1)
    ]
    
    print("Testing Min Cost Climbing Stairs Solutions:")
    print("=" * 70)
    
    for i, (cost, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: cost = {cost}")
        print(f"Expected: {expected}")
        
        # Test all approaches
        brute = min_cost_climbing_stairs_bruteforce(cost.copy())
        memo = min_cost_climbing_stairs_memoization(cost.copy())
        tab = min_cost_climbing_stairs_tabulation(cost.copy())
        space_opt = min_cost_climbing_stairs_space_optimized(cost.copy())
        forward = min_cost_climbing_stairs_forward_dp(cost.copy())
        forward_opt = min_cost_climbing_stairs_forward_optimized(cost.copy())
        alternative = min_cost_climbing_stairs_alternative(cost.copy())
        
        print(f"Brute Force:      {brute:>3} {'✓' if brute == expected else '✗'}")
        print(f"Memoization:      {memo:>3} {'✓' if memo == expected else '✗'}")
        print(f"Tabulation:       {tab:>3} {'✓' if tab == expected else '✗'}")
        print(f"Space Optimized:  {space_opt:>3} {'✓' if space_opt == expected else '✗'}")
        print(f"Forward DP:       {forward:>3} {'✓' if forward == expected else '✗'}")
        print(f"Forward Opt:      {forward_opt:>3} {'✓' if forward_opt == expected else '✗'}")
        print(f"Alternative:      {alternative:>3} {'✓' if alternative == expected else '✗'}")
        
        # Show optimal path for small examples
        if len(cost) <= 6:
            min_cost, path = min_cost_climbing_stairs_with_path(cost.copy())
            print(f"Optimal path: {' -> '.join(map(str, path))} (cost: {min_cost})")
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Brute Force:      Time: O(2^n),   Space: O(n)")
    print("Memoization:      Time: O(n),     Space: O(n)")
    print("Tabulation:       Time: O(n),     Space: O(n)")
    print("Space Optimized:  Time: O(n),     Space: O(1)")
    print("Forward DP:       Time: O(n),     Space: O(n)")
    print("Forward Opt:      Time: O(n),     Space: O(1)")
    print("Alternative:      Time: O(n),     Space: O(1)")


if __name__ == "__main__":
    test_min_cost_climbing_stairs()


"""
PATTERN RECOGNITION:
==================
This is a variation of the climbing stairs problem with costs:
- Can start from step 0 or step 1
- At each step, pay the cost and move 1 or 2 steps
- Goal: minimize total cost to reach the top
- Similar to Fibonacci but with optimization (minimum instead of counting)

KEY INSIGHTS:
============
1. Two starting positions: step 0 or step 1
2. At each step: cost[i] + min(next 1 step, next 2 steps)
3. Can solve forward (build up) or backward (break down)
4. Base case: cost to reach beyond array is 0

STATE DEFINITION:
================
Backward DP: dp[i] = minimum cost to reach top from step i
Forward DP: dp[i] = minimum cost to reach step i

RECURRENCE RELATION:
===================
Backward: dp[i] = cost[i] + min(dp[i+1], dp[i+2])
Forward: dp[i] = cost[i] + min(dp[i-1], dp[i-2])

Base cases:
- Backward: dp[n] = dp[n+1] = 0 (beyond array)
- Forward: dp[0] = cost[0], dp[1] = cost[1]

SPACE OPTIMIZATION:
==================
Since we only need 2 previous/next values, use O(1) space.

VARIANTS TO PRACTICE:
====================
- Climbing Stairs (70) - count ways instead of minimize cost
- House Robber (198) - similar cost optimization with constraints
- Jump Game II (45) - minimum jumps to reach end
- Minimum Path Sum (64) - 2D version of path cost optimization

INTERVIEW TIPS:
==============
1. Clarify starting positions (can start from 0 or 1)
2. Understand that you pay cost when stepping ON a step
3. Show both forward and backward DP approaches
4. Optimize space from O(n) to O(1)
5. Handle edge cases (array length 2)
6. Mention path reconstruction if asked
7. Compare with standard climbing stairs problem
"""
