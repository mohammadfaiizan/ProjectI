"""
LeetCode 956: Tallest Billboard
Difficulty: Hard
Category: Knapsack Problems (Advanced DP with difference tracking)

PROBLEM DESCRIPTION:
===================
You are installing a billboard and want it as tall as possible. You have steel rods of specified lengths.

You can use the rods to create two supports, LEFT and RIGHT. Each rod can be used at most once, and you 
can choose to:
1. Use the rod for the LEFT support
2. Use the rod for the RIGHT support  
3. Not use the rod at all

You want both supports to have the same height. Return the largest possible height of your billboard. 
If you cannot support the billboard, return 0.

Example 1:
Input: rods = [1,2,3,6]
Output: 6
Explanation: We have two disjoint subsets {1,2,3} and {6} with the same sum = 6.

Example 2:
Input: rods = [1,2,3,4,5,6]
Output: 10
Explanation: We have two disjoint subsets {2,3,5} and {4,6} with the same sum = 10.

Example 3:
Input: rods = [1,2]
Output: 0
Explanation: The billboard cannot be supported, so we return 0.

Constraints:
- 1 <= rods.length <= 20
- 1 <= rods[i] <= 1000
"""

def tallest_billboard_bruteforce(rods):
    """
    BRUTE FORCE APPROACH - TRY ALL COMBINATIONS:
    ==========================================
    Try all possible ways to assign rods to left, right, or unused.
    
    Time Complexity: O(3^n) - 3 choices for each rod
    Space Complexity: O(n) - recursion stack depth
    """
    def max_height(index, left_height, right_height):
        if index == len(rods):
            return left_height if left_height == right_height else 0
        
        rod_length = rods[index]
        
        # Three choices: left, right, or skip
        option1 = max_height(index + 1, left_height + rod_length, right_height)
        option2 = max_height(index + 1, left_height, right_height + rod_length)
        option3 = max_height(index + 1, left_height, right_height)
        
        return max(option1, option2, option3)
    
    return max_height(0, 0, 0)


def tallest_billboard_memoization(rods):
    """
    MEMOIZATION WITH DIFFERENCE TRACKING:
    ===================================
    Use memoization with (index, difference) as key.
    
    Time Complexity: O(n * sum) - states: index × possible differences
    Space Complexity: O(n * sum) - memoization table
    """
    memo = {}
    
    def max_height(index, diff):
        # diff = left_height - right_height
        if index == len(rods):
            return 0 if diff == 0 else float('-inf')
        
        if (index, diff) in memo:
            return memo[(index, diff)]
        
        rod_length = rods[index]
        
        # Three choices:
        # 1. Add to left: diff increases by rod_length, left_height increases by rod_length
        option1 = rod_length + max_height(index + 1, diff + rod_length)
        
        # 2. Add to right: diff decreases by rod_length, right_height increases by rod_length
        # When we add to right, we want to track the shorter height
        # If diff becomes negative, we're tracking right - left now
        option2 = max_height(index + 1, diff - rod_length)
        if diff >= rod_length:  # left was taller, now we add rod_length to the shorter side
            option2 += rod_length
        
        # 3. Skip rod
        option3 = max_height(index + 1, diff)
        
        result = max(option1, option2, option3)
        memo[(index, diff)] = result
        return result
    
    return max_height(0, 0)


def tallest_billboard_memoization_corrected(rods):
    """
    CORRECTED MEMOIZATION:
    =====================
    Properly track the height of shorter support.
    
    Time Complexity: O(n * sum) - states: index × possible differences
    Space Complexity: O(n * sum) - memoization table
    """
    memo = {}
    
    def dp(index, diff):
        # diff = left_height - right_height
        # Returns height of shorter support when both supports are equal
        if index == len(rods):
            return 0 if diff == 0 else float('-inf')
        
        if (index, diff) in memo:
            return memo[(index, diff)]
        
        rod_length = rods[index]
        
        # Skip current rod
        result = dp(index + 1, diff)
        
        # Add to left support
        left_result = dp(index + 1, diff + rod_length)
        if left_result != float('-inf'):
            result = max(result, left_result + rod_length)
        
        # Add to right support  
        right_result = dp(index + 1, diff - rod_length)
        if right_result != float('-inf'):
            if diff >= rod_length:
                # Left was taller or equal, now they might be equal
                result = max(result, right_result + rod_length)
            else:
                # Right becomes taller
                result = max(result, right_result)
        
        memo[(index, diff)] = result
        return result
    
    return dp(0, 0)


def tallest_billboard_dp_difference(rods):
    """
    DP WITH DIFFERENCE TRACKING:
    ===========================
    Use DP table with difference between left and right heights.
    
    Time Complexity: O(n * sum) - process each rod for each difference
    Space Complexity: O(sum) - DP table
    """
    # dp[diff] = maximum height of shorter support when difference is diff
    # diff = left_height - right_height
    
    total_sum = sum(rods)
    dp = {}
    dp[0] = 0  # Base case: no difference, both heights are 0
    
    for rod in rods:
        new_dp = dp.copy()
        
        for diff, shorter_height in dp.items():
            # Option 1: Add rod to left support
            new_diff = diff + rod
            if abs(new_diff) <= total_sum:
                if new_diff >= 0:
                    # Left is still taller or equal
                    new_height = shorter_height + (rod if diff < 0 else 0)
                else:
                    # Right becomes taller
                    new_height = shorter_height + rod
                
                if new_diff not in new_dp or new_dp[new_diff] < new_height:
                    new_dp[new_diff] = new_height
            
            # Option 2: Add rod to right support
            new_diff = diff - rod
            if abs(new_diff) <= total_sum:
                if new_diff <= 0:
                    # Right is taller or equal
                    new_height = shorter_height + (rod if diff > 0 else 0)
                else:
                    # Left becomes taller
                    new_height = shorter_height + rod
                
                if new_diff not in new_dp or new_dp[new_diff] < new_height:
                    new_dp[new_diff] = new_height
        
        dp = new_dp
    
    return dp.get(0, 0)


def tallest_billboard_dp_simplified(rods):
    """
    SIMPLIFIED DP APPROACH:
    ======================
    Track sum of left support for each possible difference.
    
    Time Complexity: O(n * sum) - process rods and differences
    Space Complexity: O(sum) - DP dictionary
    """
    # dp[diff] = maximum sum of left support when left - right = diff
    dp = {0: 0}  # difference 0, left sum 0 (both sides have sum 0)
    
    for rod in rods:
        new_dp = dp.copy()
        
        for diff, left_sum in dp.items():
            # Add rod to left support
            new_diff = diff + rod
            new_left = left_sum + rod
            if new_diff not in new_dp or new_dp[new_diff] < new_left:
                new_dp[new_diff] = new_left
            
            # Add rod to right support
            new_diff = diff - rod
            if new_diff not in new_dp or new_dp[new_diff] < left_sum:
                new_dp[new_diff] = left_sum
        
        dp = new_dp
    
    return dp.get(0, 0)


def tallest_billboard_optimized(rods):
    """
    OPTIMIZED DP:
    ============
    Space and time optimizations.
    
    Time Complexity: O(n * sum) - process rods and differences
    Space Complexity: O(sum) - DP dictionary
    """
    # Sort rods in descending order for potential early pruning
    rods.sort(reverse=True)
    
    total_sum = sum(rods)
    max_diff = total_sum
    
    # dp[diff] = max left_sum when left - right = diff
    dp = {0: 0}
    
    for rod in rods:
        new_dp = {}
        
        # Copy existing states
        for diff, left_sum in dp.items():
            if diff not in new_dp or new_dp[diff] < left_sum:
                new_dp[diff] = left_sum
        
        # Add new states
        for diff, left_sum in dp.items():
            # Add to left
            new_diff = diff + rod
            if abs(new_diff) <= max_diff:
                new_left = left_sum + rod
                if new_diff not in new_dp or new_dp[new_diff] < new_left:
                    new_dp[new_diff] = new_left
            
            # Add to right
            new_diff = diff - rod
            if abs(new_diff) <= max_diff:
                if new_diff not in new_dp or new_dp[new_diff] < left_sum:
                    new_dp[new_diff] = left_sum
        
        dp = new_dp
    
    return dp.get(0, 0)


def tallest_billboard_with_construction(rods):
    """
    FIND ACTUAL CONSTRUCTION:
    ========================
    Return height and show which rods go to which support.
    
    Time Complexity: O(n * sum) - DP + reconstruction
    Space Complexity: O(n * sum) - store construction choices
    """
    # Track construction choices
    # dp[diff] = (max_left_sum, left_rods, right_rods)
    dp = {0: (0, [], [])}
    
    for i, rod in enumerate(rods):
        new_dp = {}
        
        # Keep existing states
        for diff, (left_sum, left_rods, right_rods) in dp.items():
            if diff not in new_dp or new_dp[diff][0] < left_sum:
                new_dp[diff] = (left_sum, left_rods[:], right_rods[:])
        
        # Add rod to left
        for diff, (left_sum, left_rods, right_rods) in dp.items():
            new_diff = diff + rod
            new_left = left_sum + rod
            new_left_rods = left_rods + [rod]
            
            if new_diff not in new_dp or new_dp[new_diff][0] < new_left:
                new_dp[new_diff] = (new_left, new_left_rods, right_rods[:])
        
        # Add rod to right
        for diff, (left_sum, left_rods, right_rods) in dp.items():
            new_diff = diff - rod
            new_right_rods = right_rods + [rod]
            
            if new_diff not in new_dp or new_dp[new_diff][0] < left_sum:
                new_dp[new_diff] = (left_sum, left_rods[:], new_right_rods)
        
        dp = new_dp
    
    if 0 in dp:
        height, left_rods, right_rods = dp[0]
        return height, left_rods, right_rods
    else:
        return 0, [], []


# Test cases
def test_tallest_billboard():
    """Test all implementations with various inputs"""
    test_cases = [
        ([1,2,3,6], 6),
        ([1,2,3,4,5,6], 10),
        ([1,2], 0),
        ([1,2,4,8], 4),
        ([1,1,1,1], 2),
        ([2,1,2], 2),
        ([1,2,3,4,5], 6),
        ([1], 0),
        ([3,4,3,3,2], 6),
        ([61,45,43,54,40,53,55,47,51,59,40], 275)
    ]
    
    print("Testing Tallest Billboard Solutions:")
    print("=" * 70)
    
    for i, (rods, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: rods = {rods}")
        print(f"Expected: {expected}")
        
        # Test approaches (skip expensive ones for large inputs)
        if len(rods) <= 8:
            brute = tallest_billboard_bruteforce(rods.copy())
            print(f"Brute Force:      {brute:>3} {'✓' if brute == expected else '✗'}")
        
        if len(rods) <= 15:
            memo = tallest_billboard_memoization_corrected(rods.copy())
            print(f"Memoization:      {memo:>3} {'✓' if memo == expected else '✗'}")
        
        dp_diff = tallest_billboard_dp_difference(rods.copy())
        simplified = tallest_billboard_dp_simplified(rods.copy())
        optimized = tallest_billboard_optimized(rods.copy())
        
        print(f"DP Difference:    {dp_diff:>3} {'✓' if dp_diff == expected else '✗'}")
        print(f"DP Simplified:    {simplified:>3} {'✓' if simplified == expected else '✗'}")
        print(f"Optimized:        {optimized:>3} {'✓' if optimized == expected else '✗'}")
        
        # Show actual construction for positive cases
        if expected > 0 and len(rods) <= 10:
            height, left_rods, right_rods = tallest_billboard_with_construction(rods.copy())
            if height > 0:
                print(f"Construction: Left={left_rods} (sum={sum(left_rods)}), Right={right_rods} (sum={sum(right_rods)})")
    
    print("\n" + "=" * 70)
    print("Key Insight:")
    print("Track difference = left_height - right_height")
    print("dp[diff] = maximum left_sum when left - right = diff")
    print("Goal: find dp[0] (equal heights)")
    print("State transitions: add rod to left (+rod to diff) or right (-rod from diff)")
    
    print("\n" + "=" * 70)
    print("Algorithm Comparison:")
    print("Brute Force:      Try all 3^n assignments")
    print("Memoization:      Cache (index, difference) states")
    print("DP Difference:    Track difference between supports")
    print("DP Simplified:    Track left sum for each difference")
    print("Optimized:        Add bounds and optimizations")
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Brute Force:      Time: O(3^n),    Space: O(n)")
    print("Memoization:      Time: O(n*sum),  Space: O(n*sum)")
    print("DP Difference:    Time: O(n*sum),  Space: O(sum)")
    print("DP Simplified:    Time: O(n*sum),  Space: O(sum)")
    print("Optimized:        Time: O(n*sum),  Space: O(sum)")


if __name__ == "__main__":
    test_tallest_billboard()


"""
PATTERN RECOGNITION:
==================
This is an advanced subset partition problem:
- Partition rods into two disjoint subsets with equal sums
- Maximize the common sum (height of billboard)
- Key insight: track difference between left and right heights
- Similar to "Split Array Same Average" but maximizing instead of just checking

KEY INSIGHT - DIFFERENCE TRACKING:
=================================
Instead of tracking (left_height, right_height), track:
- diff = left_height - right_height  
- left_sum = sum of rods assigned to left support

This reduces 2D state to 1D state while preserving all information.

STATE DEFINITION:
================
dp[diff] = maximum left_sum when left_height - right_height = diff

The goal is to find dp[0] (when both supports have equal height).

RECURRENCE RELATION:
===================
For each rod of length L:
1. **Add to left**: dp[diff + L] = max(dp[diff + L], dp[diff] + L)
2. **Add to right**: dp[diff - L] = max(dp[diff - L], dp[diff])
3. **Skip rod**: keep existing dp[diff]

When diff = 0, left_sum = right_sum = height of each support.

ALGORITHM PROGRESSION:
=====================
1. **Brute Force**: O(3^n) - try all assignments (left/right/skip)
2. **Memoization**: O(n×sum) - cache (index, difference) states
3. **DP Table**: O(n×sum) - build solution bottom-up
4. **Space Optimized**: O(sum) - use rolling DP dictionary

MATHEMATICAL PROPERTIES:
=======================
1. **Difference Range**: |diff| ≤ sum(rods)
2. **Symmetry**: Equal partitions have diff = 0
3. **Monotonicity**: Larger left_sum is always better for same diff
4. **Bounds**: Valid differences are limited by remaining rods

OPTIMIZATION TECHNIQUES:
=======================
1. **Difference Bounding**: Only track |diff| ≤ remaining_sum
2. **State Pruning**: Remove dominated states
3. **Early Termination**: Stop when optimal solution found
4. **Sorting**: Process larger rods first for better pruning

COMPARISON WITH RELATED PROBLEMS:
================================
- **Partition Equal Subset Sum (416)**: Check if equal partition exists
- **Split Array Same Average (805)**: Equal averages (different constraint)
- **Last Stone Weight II (1049)**: Minimize difference (similar state tracking)
- **This Problem**: Maximize equal partition sum

STATE SPACE ANALYSIS:
====================
- **Naive**: O(sum^2) states for (left_sum, right_sum)
- **Optimized**: O(sum) states for difference tracking
- **Insight**: diff = left - right uniquely determines the relationship

EDGE CASES:
==========
1. **Single rod**: Cannot create equal supports → return 0
2. **Two rods**: Equal only if they're identical
3. **All rods equal**: Can always create equal supports
4. **Impossible partitions**: No way to balance → return 0

IMPLEMENTATION VARIANTS:
=======================
1. **Dictionary DP**: Use dict for sparse state space
2. **Array DP**: Use array with offset for negative indices
3. **Set-based**: Track reachable states efficiently
4. **Backtracking**: For reconstruction of actual solution

VARIANTS TO PRACTICE:
====================
- Partition problems (equal sum, equal average)
- Subset sum with constraints
- Knapsack variants with multiple constraints
- Difference minimization problems

INTERVIEW TIPS:
==============
1. **Start with brute force**: Show 3^n approach first
2. **Key insight**: Explain difference tracking optimization
3. **State definition**: Clearly define dp[diff] meaning
4. **Recurrence**: Show how rod assignments update states
5. **Space optimization**: Explain rolling DP technique
6. **Edge cases**: Handle impossible partitions
7. **Reconstruction**: Show how to find actual rod assignment
8. **Complexity**: Explain why O(n×sum) is much better than O(3^n)
9. **Related problems**: Connect to other partition problems
10. **Mathematical insight**: Explain why difference tracking works
"""
