"""
LeetCode 1049: Last Stone Weight II
Difficulty: Medium
Category: Knapsack Problems (Subset Partition variant)

PROBLEM DESCRIPTION:
===================
You are given an array of integers stones where stones[i] is the weight of the ith stone.

We are playing a game with the stones. On each turn, we choose any two stones and smash them 
together. Suppose the stones have weights x and y with x <= y. The result of this smash is:

- If x == y, both stones are destroyed.
- If x != y, the stone of weight x is destroyed, and the stone of weight y has new weight y - x.

At the end of the game, there is at most 1 stone left.

Return the minimum possible weight of the last stone. If there are no stones left, return 0.

Example 1:
Input: stones = [2,7,4,1,8,1]
Output: 1
Explanation: 
We can combine 2 and 4 to get 2, so the array converts to [2,7,1,8,1] then,
we can combine 7 and 8 to get 1, so the array converts to [2,1,1,1] then,
we can combine 2 and 1 to get 1, so the array converts to [1,1,1] then,
we can combine 1 and 1 to get 0, so the array converts to [1] then that's the optimal value.

Example 2:
Input: stones = [31,26,33,21,40]
Output: 5

Example 3:
Input: stones = [1,2]
Output: 1

Constraints:
- 1 <= stones.length <= 30
- 1 <= stones[i] <= 100
"""

def last_stone_weight_ii_bruteforce(stones):
    """
    BRUTE FORCE APPROACH - RECURSION:
    ================================
    Try all possible ways to smash stones.
    
    Time Complexity: O(2^n) - exponential stone combinations
    Space Complexity: O(n) - recursion stack depth
    """
    def min_weight(stone_list):
        if len(stone_list) <= 1:
            return stone_list[0] if stone_list else 0
        
        min_result = float('inf')
        
        # Try all pairs of stones
        for i in range(len(stone_list)):
            for j in range(i + 1, len(stone_list)):
                x, y = stone_list[i], stone_list[j]
                
                # Create new list after smashing
                new_stones = []
                for k in range(len(stone_list)):
                    if k != i and k != j:
                        new_stones.append(stone_list[k])
                
                # Add result of smashing (if any stone remains)
                if x != y:
                    new_stones.append(abs(x - y))
                
                result = min_weight(new_stones)
                min_result = min(min_result, result)
        
        return min_result
    
    return min_weight(stones)


def last_stone_weight_ii_insight(stones):
    """
    KEY INSIGHT - SUBSET PARTITION:
    ==============================
    Transform to subset partition problem.
    
    Time Complexity: O(n * sum) - subset sum DP
    Space Complexity: O(sum) - DP array
    """
    # Key insight: The problem is equivalent to partitioning stones into two groups
    # such that the difference between their sums is minimized
    # Final result = |sum(group1) - sum(group2)|
    # We want to minimize this difference
    
    total_sum = sum(stones)
    target = total_sum // 2
    
    # Find if we can achieve each sum using subset of stones
    dp = [False] * (target + 1)
    dp[0] = True  # Can always achieve sum 0 with empty subset
    
    for stone in stones:
        # Process in reverse to avoid using updated values
        for j in range(target, stone - 1, -1):
            dp[j] = dp[j] or dp[j - stone]
    
    # Find the largest sum <= target that we can achieve
    closest_sum = 0
    for i in range(target, -1, -1):
        if dp[i]:
            closest_sum = i
            break
    
    # The two groups have sums: closest_sum and (total_sum - closest_sum)
    return total_sum - 2 * closest_sum


def last_stone_weight_ii_memoization(stones):
    """
    MEMOIZATION APPROACH:
    ====================
    Use memoization on the game simulation.
    
    Time Complexity: O(2^n) - still exponential but with pruning
    Space Complexity: O(2^n) - memoization table
    """
    memo = {}
    
    def min_weight(stone_tuple):
        if len(stone_tuple) <= 1:
            return stone_tuple[0] if stone_tuple else 0
        
        if stone_tuple in memo:
            return memo[stone_tuple]
        
        stone_list = list(stone_tuple)
        min_result = float('inf')
        
        for i in range(len(stone_list)):
            for j in range(i + 1, len(stone_list)):
                x, y = stone_list[i], stone_list[j]
                
                new_stones = []
                for k in range(len(stone_list)):
                    if k != i and k != j:
                        new_stones.append(stone_list[k])
                
                if x != y:
                    new_stones.append(abs(x - y))
                
                new_stones.sort()  # Sort for consistent memoization
                result = min_weight(tuple(new_stones))
                min_result = min(min_result, result)
        
        memo[stone_tuple] = min_result
        return min_result
    
    stones.sort()
    return min_weight(tuple(stones))


def last_stone_weight_ii_dp_detailed(stones):
    """
    DETAILED DP APPROACH:
    ====================
    Detailed subset sum DP with explanation.
    
    Time Complexity: O(n * sum) - DP computation
    Space Complexity: O(sum) - DP array
    """
    total_sum = sum(stones)
    
    # We want to find two subsets with sums as close as possible
    # Let S1 and S2 be the sums of two subsets
    # S1 + S2 = total_sum
    # We want to minimize |S1 - S2|
    # Since S2 = total_sum - S1, we minimize |S1 - (total_sum - S1)| = |2*S1 - total_sum|
    # To minimize this, we want S1 as close to total_sum/2 as possible
    
    target = total_sum // 2
    
    # dp[i] = True if sum i is achievable using subset of stones
    dp = [False] * (target + 1)
    dp[0] = True
    
    for stone in stones:
        # Update in reverse order to avoid using updated values
        for j in range(target, stone - 1, -1):
            if dp[j - stone]:
                dp[j] = True
    
    # Find the largest achievable sum <= target
    s1 = 0
    for i in range(target, -1, -1):
        if dp[i]:
            s1 = i
            break
    
    s2 = total_sum - s1
    return abs(s1 - s2)


def last_stone_weight_ii_optimized(stones):
    """
    OPTIMIZED DP WITH EARLY TERMINATION:
    ===================================
    Add optimizations for better performance.
    
    Time Complexity: O(n * sum) - worst case, often better
    Space Complexity: O(sum) - DP array
    """
    total_sum = sum(stones)
    
    # Early termination: if all stones are equal, result is 0 or 1
    if len(set(stones)) == 1:
        return stones[0] if len(stones) % 2 == 1 else 0
    
    target = total_sum // 2
    
    # Use bitset for space efficiency (if target is not too large)
    if target <= 10000:
        # Use integer as bitset
        possible = 1  # Bit 0 set (sum 0 is possible)
        
        for stone in stones:
            possible |= (possible << stone)
        
        # Find largest possible sum <= target
        for i in range(target, -1, -1):
            if possible & (1 << i):
                return total_sum - 2 * i
    
    # Fallback to regular DP for large targets
    return last_stone_weight_ii_dp_detailed(stones)


def last_stone_weight_ii_with_partition(stones):
    """
    SHOW ACTUAL PARTITION:
    =====================
    Return minimum weight and the actual partition.
    
    Time Complexity: O(n * sum) - DP + reconstruction
    Space Complexity: O(n * sum) - track choices
    """
    total_sum = sum(stones)
    target = total_sum // 2
    n = len(stones)
    
    # dp[i][j] = True if sum j is achievable using first i stones
    dp = [[False] * (target + 1) for _ in range(n + 1)]
    
    # Base case: sum 0 is always achievable
    for i in range(n + 1):
        dp[i][0] = True
    
    # Fill DP table
    for i in range(1, n + 1):
        for j in range(1, target + 1):
            # Don't include current stone
            dp[i][j] = dp[i - 1][j]
            
            # Include current stone if possible
            if j >= stones[i - 1]:
                dp[i][j] = dp[i][j] or dp[i - 1][j - stones[i - 1]]
    
    # Find best achievable sum
    best_sum = 0
    for j in range(target, -1, -1):
        if dp[n][j]:
            best_sum = j
            break
    
    # Reconstruct partition
    group1 = []
    group2 = []
    
    i, j = n, best_sum
    while i > 0 and j > 0:
        # If current stone was included
        if j >= stones[i - 1] and dp[i - 1][j - stones[i - 1]]:
            group1.append(stones[i - 1])
            j -= stones[i - 1]
        else:
            group2.append(stones[i - 1])
        i -= 1
    
    # Add remaining stones to group2
    while i > 0:
        group2.append(stones[i - 1])
        i -= 1
    
    result = abs(sum(group1) - sum(group2))
    return result, group1, group2


def last_stone_weight_ii_mathematical(stones):
    """
    MATHEMATICAL ANALYSIS:
    =====================
    Analyze the problem mathematically.
    
    Time Complexity: O(n * sum) - DP computation
    Space Complexity: O(sum) - DP array
    """
    # Mathematical insight:
    # 1. Each stone can be assigned a sign: +1 or -1
    # 2. Final result = |sum of all signed stones|
    # 3. We want to minimize this absolute value
    # 4. This is equivalent to partitioning into two groups with minimal difference
    
    total_sum = sum(stones)
    
    print(f"Mathematical Analysis:")
    print(f"Total sum: {total_sum}")
    print(f"Target for each group: {total_sum / 2}")
    
    # Find achievable sums
    dp = [False] * (total_sum + 1)
    dp[0] = True
    
    for stone in stones:
        for j in range(total_sum, stone - 1, -1):
            dp[j] = dp[j] or dp[j - stone]
    
    # Find closest sum to total_sum/2
    target = total_sum // 2
    closest_sum = 0
    
    for i in range(target, -1, -1):
        if dp[i]:
            closest_sum = i
            break
    
    result = total_sum - 2 * closest_sum
    
    print(f"Best partition: {closest_sum} and {total_sum - closest_sum}")
    print(f"Difference: {result}")
    
    return result


# Test cases
def test_last_stone_weight_ii():
    """Test all implementations with various inputs"""
    test_cases = [
        ([2,7,4,1,8,1], 1),
        ([31,26,33,21,40], 5),
        ([1,2], 1),
        ([1], 1),
        ([1,1], 0),
        ([2,2], 0),
        ([3,4,5], 2),
        ([1,4,3,2], 0),
        ([1,1,2,3,5,8], 1),
        ([100], 100),
        ([1,2,3,4,5,6,7,8,9,10], 1)
    ]
    
    print("Testing Last Stone Weight II Solutions:")
    print("=" * 70)
    
    for i, (stones, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: stones = {stones}")
        print(f"Expected: {expected}")
        
        # Test approaches (skip brute force for large inputs)
        if len(stones) <= 6:
            brute = last_stone_weight_ii_bruteforce(stones.copy())
            print(f"Brute Force:      {brute:>3} {'✓' if brute == expected else '✗'}")
        
        insight = last_stone_weight_ii_insight(stones.copy())
        dp_detailed = last_stone_weight_ii_dp_detailed(stones.copy())
        optimized = last_stone_weight_ii_optimized(stones.copy())
        
        print(f"Insight (Subset): {insight:>3} {'✓' if insight == expected else '✗'}")
        print(f"DP Detailed:      {dp_detailed:>3} {'✓' if dp_detailed == expected else '✗'}")
        print(f"Optimized:        {optimized:>3} {'✓' if optimized == expected else '✗'}")
        
        if len(stones) <= 8:
            memo = last_stone_weight_ii_memoization(stones.copy())
            print(f"Memoization:      {memo:>3} {'✓' if memo == expected else '✗'}")
        
        # Show actual partition for small cases
        if len(stones) <= 8:
            result, group1, group2 = last_stone_weight_ii_with_partition(stones.copy())
            print(f"Partition: {group1} (sum={sum(group1)}) vs {group2} (sum={sum(group2)})")
    
    # Mathematical analysis example
    print(f"\n" + "=" * 70)
    print("MATHEMATICAL ANALYSIS EXAMPLE:")
    print("-" * 40)
    last_stone_weight_ii_mathematical([2,7,4,1,8,1])
    
    print("\n" + "=" * 70)
    print("Key Insight:")
    print("Last Stone Weight II ≡ Minimize |sum(group1) - sum(group2)|")
    print("This reduces to: Find subset with sum closest to total_sum/2")
    print("Classic subset sum DP problem!")
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Brute Force:      Time: O(2^n),     Space: O(n)")
    print("Insight (Subset): Time: O(n*sum),   Space: O(sum)")
    print("Memoization:      Time: O(2^n),     Space: O(2^n)")
    print("DP Detailed:      Time: O(n*sum),   Space: O(sum)")
    print("Optimized:        Time: O(n*sum),   Space: O(sum)")


if __name__ == "__main__":
    test_last_stone_weight_ii()


"""
PATTERN RECOGNITION:
==================
This is a disguised Subset Partition problem:
- Game simulation looks complex, but has elegant mathematical transformation
- Key insight: Minimize |sum(group1) - sum(group2)| where group1 ∪ group2 = all stones
- Reduces to classic subset sum DP problem

KEY INSIGHT - MATHEMATICAL TRANSFORMATION:
=========================================
Complex game → Simple partition problem:

1. **Game Process**: Stones smash each other, leaving |x - y|
2. **Mathematical Reality**: Each stone effectively gets a +1 or -1 sign
3. **Final Result**: |sum of all signed stones|
4. **Optimization Goal**: Minimize this absolute value
5. **Equivalent Problem**: Partition into two groups with minimal difference

TRANSFORMATION PROOF:
====================
- Let S1, S2 be sums of two groups
- S1 + S2 = total_sum (all stones)  
- We minimize |S1 - S2|
- Since S2 = total_sum - S1, minimize |S1 - (total_sum - S1)| = |2×S1 - total_sum|
- To minimize this, make S1 as close to total_sum/2 as possible
- This is the classic "subset sum closest to target" problem!

STATE DEFINITION:
================
dp[i] = True if sum i is achievable using subset of stones

RECURRENCE RELATION:
===================
dp[j] = dp[j] OR dp[j - stone] for each stone
Process stones one by one, update DP array in reverse order.

ALGORITHM PROGRESSION:
=====================
1. **Brute Force**: O(2^n) - simulate all possible games
2. **Insight**: O(n×sum) - transform to subset sum
3. **Memoization**: O(2^n) - cache game states (still exponential)
4. **Optimized**: O(n×sum) - bitset optimization for small sums

OPTIMIZATION TECHNIQUES:
=======================
1. **Subset Sum DP**: Transform complex game to simple partition
2. **Space Optimization**: Use 1D array instead of 2D
3. **Bitset Optimization**: Use integer bits for small target sums
4. **Early Termination**: Handle special cases (all equal stones)

WHY THIS TRANSFORMATION WORKS:
=============================
- Stone smashing preserves total "signed weight"
- Each stone contributes +weight or -weight to final result
- Optimal strategy partitions stones to minimize difference
- Game rules guarantee we reach this optimal partition

VARIANTS TO PRACTICE:
====================
- Partition Equal Subset Sum (416) - check if perfect partition exists
- Target Sum (494) - assign +/- signs to reach target
- Minimize Subset Sum Difference - direct version of this problem
- 0/1 Knapsack variants

INTERVIEW TIPS:
==============
1. **Don't simulate the game!** Look for mathematical insight
2. Recognize as partition minimization problem
3. Transform to subset sum DP
4. Explain why transformation is valid
5. Show subset sum solution with O(n×sum) complexity
6. Mention bitset optimization for small sums
7. Handle edge cases (single stone, all equal stones)
8. Discuss how game rules guarantee optimal partitioning
"""
