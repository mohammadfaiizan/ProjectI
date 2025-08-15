"""
LeetCode 377: Combination Sum IV
Difficulty: Medium
Category: Fibonacci & Linear DP (Unbounded Knapsack variant)

PROBLEM DESCRIPTION:
===================
Given an array of distinct integers nums and a target integer target, return the number of 
possible combinations that add up to target.

The test cases are generated so that the answer can fit in a 32-bit integer.

Example 1:
Input: nums = [1,2,3], target = 4
Output: 7
Explanation:
The possible combinations are:
(1, 1, 1, 1)
(1, 1, 2)
(1, 2, 1)
(1, 3)
(2, 1, 1)
(2, 2)
(3, 1)

Example 2:
Input: nums = [9], target = 3
Output: 0

Example 3:
Input: nums = [1,2], target = 3
Output: 3
Explanation:
The possible combinations are:
(1, 1, 1)
(1, 2)
(2, 1)

Constraints:
- 1 <= nums.length <= 200
- 1 <= nums[i] <= 1000
- All the elements of nums are unique.
- 1 <= target <= 1000
"""

def combination_sum4_bruteforce(nums, target):
    """
    BRUTE FORCE APPROACH - RECURSION:
    ================================
    Try all possible combinations using recursion.
    
    Time Complexity: O(target^len(nums)) - exponential
    Space Complexity: O(target) - recursion stack depth
    """
    def count_combinations(remaining):
        if remaining == 0:
            return 1
        if remaining < 0:
            return 0
        
        total = 0
        for num in nums:
            total += count_combinations(remaining - num)
        
        return total
    
    return count_combinations(target)


def combination_sum4_memoization(nums, target):
    """
    DYNAMIC PROGRAMMING - TOP DOWN (MEMOIZATION):
    ============================================
    Use memoization to avoid recalculating same subproblems.
    
    Time Complexity: O(target * len(nums)) - target states, len(nums) transitions each
    Space Complexity: O(target) - memoization table + recursion stack
    """
    memo = {}
    
    def count_combinations(remaining):
        if remaining == 0:
            return 1
        if remaining < 0:
            return 0
        
        if remaining in memo:
            return memo[remaining]
        
        total = 0
        for num in nums:
            total += count_combinations(remaining - num)
        
        memo[remaining] = total
        return total
    
    return count_combinations(target)


def combination_sum4_tabulation(nums, target):
    """
    DYNAMIC PROGRAMMING - BOTTOM UP (TABULATION):
    =============================================
    Build solution from bottom up using DP array.
    dp[i] = number of combinations that sum to i
    
    Time Complexity: O(target * len(nums)) - nested loops
    Space Complexity: O(target) - DP array
    """
    # dp[i] = number of combinations that sum to i
    dp = [0] * (target + 1)
    dp[0] = 1  # Base case: one way to make sum 0 (empty combination)
    
    # For each target sum from 1 to target
    for i in range(1, target + 1):
        # Try each number
        for num in nums:
            if num <= i:
                dp[i] += dp[i - num]
    
    return dp[target]


def combination_sum4_space_optimized(nums, target):
    """
    SPACE OPTIMIZED (ALREADY OPTIMAL):
    =================================
    The tabulation approach is already space optimal for this problem.
    But we can add some optimizations.
    
    Time Complexity: O(target * len(nums))
    Space Complexity: O(target) - DP array
    """
    # Sort nums for potential early termination
    nums.sort()
    
    dp = [0] * (target + 1)
    dp[0] = 1
    
    for i in range(1, target + 1):
        for num in nums:
            if num > i:
                break  # Early termination since nums is sorted
            dp[i] += dp[i - num]
    
    return dp[target]


def combination_sum4_optimized_order(nums, target):
    """
    OPTIMIZED WITH DIFFERENT LOOP ORDER:
    ===================================
    Show difference between permutations vs combinations.
    Current order counts permutations (order matters).
    
    Time Complexity: O(target * len(nums))
    Space Complexity: O(target) - DP array
    """
    dp = [0] * (target + 1)
    dp[0] = 1
    
    # This order counts permutations (current problem)
    for i in range(1, target + 1):
        for num in nums:
            if num <= i:
                dp[i] += dp[i - num]
    
    return dp[target]


def combination_sum4_combinations_only(nums, target):
    """
    COMBINATIONS ONLY (ORDER DOESN'T MATTER):
    =========================================
    Different loop order to count combinations only.
    This is NOT the solution to the current problem but shows the difference.
    
    Time Complexity: O(target * len(nums))
    Space Complexity: O(target) - DP array
    """
    dp = [0] * (target + 1)
    dp[0] = 1
    
    # This order counts combinations (order doesn't matter)
    for num in nums:
        for i in range(num, target + 1):
            dp[i] += dp[i - num]
    
    return dp[target]


def combination_sum4_with_actual_combinations(nums, target):
    """
    FIND ALL ACTUAL COMBINATIONS:
    =============================
    Return both count and all actual combinations (permutations).
    
    Time Complexity: O(target^len(nums)) - generate all combinations
    Space Complexity: O(target^len(nums)) - store all combinations
    """
    all_combinations = []
    
    def backtrack(remaining, current_combination):
        if remaining == 0:
            all_combinations.append(current_combination[:])
            return
        
        for num in nums:
            if num <= remaining:
                current_combination.append(num)
                backtrack(remaining - num, current_combination)
                current_combination.pop()
    
    backtrack(target, [])
    return len(all_combinations), all_combinations


def combination_sum4_iterative_dp(nums, target):
    """
    ITERATIVE DP WITH DETAILED TRACKING:
    ===================================
    Track how combinations are built step by step.
    
    Time Complexity: O(target * len(nums))
    Space Complexity: O(target) - DP array
    """
    dp = [0] * (target + 1)
    dp[0] = 1
    
    for i in range(1, target + 1):
        for num in nums:
            if num <= i:
                # Add all ways to make (i - num) to ways to make i
                dp[i] += dp[i - num]
                
        # Optional: print progress for small targets
        if target <= 10:
            print(f"dp[{i}] = {dp[i]} (ways to make sum {i})")
    
    return dp[target]


def combination_sum4_bfs(nums, target):
    """
    BFS APPROACH:
    ============
    Use BFS to count paths to target. Not efficient but shows alternative.
    
    Time Complexity: O(target^len(nums)) - exponential paths
    Space Complexity: O(target) - queue storage
    """
    from collections import deque, defaultdict
    
    # Count ways to reach each sum
    ways = defaultdict(int)
    ways[0] = 1
    
    queue = deque([0])
    
    while queue:
        current_sum = queue.popleft()
        
        if current_sum == target:
            continue
        
        for num in nums:
            next_sum = current_sum + num
            if next_sum <= target:
                if next_sum not in ways:
                    queue.append(next_sum)
                ways[next_sum] += ways[current_sum]
    
    return ways[target]


def combination_sum4_mathematical_analysis(nums, target):
    """
    MATHEMATICAL ANALYSIS:
    =====================
    Analyze the problem mathematically and provide insights.
    
    Time Complexity: O(target * len(nums))
    Space Complexity: O(target)
    """
    # This is essentially a linear recurrence relation
    # dp[i] = sum(dp[i - num] for num in nums if num <= i)
    
    dp = [0] * (target + 1)
    dp[0] = 1
    
    contributions = [0] * (target + 1)  # Track which numbers contribute most
    
    for i in range(1, target + 1):
        max_contribution = 0
        best_num = -1
        
        for num in nums:
            if num <= i:
                contribution = dp[i - num]
                dp[i] += contribution
                
                if contribution > max_contribution:
                    max_contribution = contribution
                    best_num = num
        
        contributions[i] = best_num
    
    return dp[target], contributions


# Test cases
def test_combination_sum4():
    """Test all implementations with various inputs"""
    test_cases = [
        ([1, 2, 3], 4, 7),
        ([9], 3, 0),
        ([1, 2], 3, 3),
        ([1], 1, 1),
        ([1], 2, 1),
        ([2, 3, 5], 8, 5),
        ([1, 2, 3], 32, 181997601),
        ([4, 2, 1], 32, 39882198)
    ]
    
    print("Testing Combination Sum IV Solutions:")
    print("=" * 70)
    
    for i, (nums, target, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: nums = {nums}, target = {target}")
        print(f"Expected: {expected}")
        
        # Test approaches (skip brute force for large targets)
        if target <= 10:
            brute = combination_sum4_bruteforce(nums.copy(), target)
            print(f"Brute Force:      {brute:>10} {'✓' if brute == expected else '✗'}")
        
        memo = combination_sum4_memoization(nums.copy(), target)
        tab = combination_sum4_tabulation(nums.copy(), target)
        space_opt = combination_sum4_space_optimized(nums.copy(), target)
        opt_order = combination_sum4_optimized_order(nums.copy(), target)
        
        print(f"Memoization:      {memo:>10} {'✓' if memo == expected else '✗'}")
        print(f"Tabulation:       {tab:>10} {'✓' if tab == expected else '✗'}")
        print(f"Space Optimized:  {space_opt:>10} {'✓' if space_opt == expected else '✗'}")
        print(f"Optimized Order:  {opt_order:>10} {'✓' if opt_order == expected else '✗'}")
        
        # Show combinations only (order doesn't matter) for comparison
        if target <= 10:
            comb_only = combination_sum4_combinations_only(nums.copy(), target)
            print(f"Combinations Only: {comb_only:>9} (order doesn't matter)")
        
        # Show actual combinations for small targets
        if target <= 6 and len(nums) <= 3:
            count, combinations = combination_sum4_with_actual_combinations(nums.copy(), target)
            print(f"All combinations: {combinations}")
    
    print("\n" + "=" * 70)
    print("Key Difference - Loop Order:")
    print("Permutations (order matters): for i in range(target): for num in nums")
    print("Combinations (order doesn't matter): for num in nums: for i in range(target)")
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Brute Force:      Time: O(target^len(nums)), Space: O(target)")
    print("Memoization:      Time: O(target*len(nums)), Space: O(target)")
    print("Tabulation:       Time: O(target*len(nums)), Space: O(target)")
    print("Space Optimized:  Time: O(target*len(nums)), Space: O(target)")
    print("BFS:              Time: O(target^len(nums)), Space: O(target)")


if __name__ == "__main__":
    test_combination_sum4()


"""
PATTERN RECOGNITION:
==================
This is an Unbounded Knapsack variant with order consideration:
- Unlimited use of each number
- Count ways to reach target sum
- Order matters: (1,2) and (2,1) are different combinations
- Key insight: This counts PERMUTATIONS, not combinations

KEY INSIGHT - PERMUTATIONS vs COMBINATIONS:
==========================================
Loop order determines whether we count permutations or combinations:

PERMUTATIONS (this problem):
for i in range(target):
    for num in nums:
        dp[i] += dp[i - num]

COMBINATIONS (order doesn't matter):
for num in nums:
    for i in range(target):
        dp[i] += dp[i - num]

STATE DEFINITION:
================
dp[i] = number of ways to make sum i using numbers from nums (with repetition allowed)

RECURRENCE RELATION:
===================
dp[i] = sum(dp[i - num] for num in nums if num <= i)
Base case: dp[0] = 1 (one way to make sum 0: empty sequence)

ALGORITHM EXPLANATION:
=====================
For each target sum i:
- Try adding each number num from nums
- If num <= i, then dp[i] += dp[i - num]
- This counts all sequences ending with num that sum to i

COMPARISON WITH COIN CHANGE:
===========================
- Coin Change (322): minimize coins used
- Coin Change 2 (518): count combinations (order doesn't matter)  
- Combination Sum IV (377): count permutations (order matters)

VARIANTS TO PRACTICE:
====================
- Coin Change 2 (518) - combinations only (different loop order)
- Coin Change (322) - minimize count instead of counting ways
- Perfect Squares (279) - special case with perfect square "coins"
- Climbing Stairs (70) - special case with coins [1, 2]

INTERVIEW TIPS:
==============
1. Clarify: do permutations count as different? (Yes for this problem)
2. Identify as unbounded knapsack counting problem
3. Show the difference between permutation and combination loop orders
4. Explain why dp[0] = 1 (base case)
5. Trace through small example to verify understanding
6. Compare with Coin Change variants
7. Handle edge cases (target = 0, empty nums, no solution)
"""
