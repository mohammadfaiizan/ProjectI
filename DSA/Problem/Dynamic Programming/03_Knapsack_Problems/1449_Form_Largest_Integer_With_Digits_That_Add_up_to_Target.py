"""
LeetCode 1449: Form Largest Integer With Digits That Add up to Target
Difficulty: Hard
Category: Knapsack Problems (Unbounded Knapsack with Lexicographic Optimization)

PROBLEM DESCRIPTION:
===================
Given an array of integers cost and an integer target, return the maximum number of digits you can 
write down to form an integer such that the total cost of all the digits you write down equals target.

Since the answer may be very large, return it as a string. If there is no way to form an integer, return "0".

The cost of writing digit i is given by cost[i-1] (0-indexed).

Example 1:
Input: cost = [4,3,2,5,6,7,2,5,5], target = 5
Output: "22"
Explanation: For a target of 5, the cheapest way to form a number is "22" (cost 2+2+1 = 5).

Example 2:
Input: cost = [7,6,5,5,5,6,8,7,8], target = 12
Output: "85"
Explanation: For a target of 12, the digits 8 and 5 have the minimum cost.
The number formed is "85".

Example 3:
Input: cost = [2,4,6,2,4,6,4,4,4], target = 5
Output: "0"
Explanation: It's impossible to form an integer with a cost of exactly 5.

Constraints:
- cost.length == 9
- 1 <= cost[i] <= 5000
- 1 <= target <= 5000
"""

def largest_number_bruteforce(cost, target):
    """
    BRUTE FORCE APPROACH - RECURSION:
    ================================
    Try all possible combinations of digits.
    
    Time Complexity: O(9^target) - up to 9 choices per position
    Space Complexity: O(target) - recursion stack depth
    """
    def max_digits(remaining_cost, current_number):
        if remaining_cost == 0:
            return current_number
        
        if remaining_cost < 0:
            return ""  # Invalid
        
        best = ""
        
        # Try each digit (9 down to 1 for lexicographic order)
        for digit in range(9, 0, -1):
            digit_cost = cost[digit - 1]
            
            if digit_cost <= remaining_cost:
                result = max_digits(remaining_cost - digit_cost, current_number + str(digit))
                
                # Choose better result (more digits, then lexicographically larger)
                if result and (not best or len(result) > len(best) or 
                             (len(result) == len(best) and result > best)):
                    best = result
        
        return best
    
    result = max_digits(target, "")
    return result if result else "0"


def largest_number_dp_max_digits(cost, target):
    """
    DP TO FIND MAXIMUM DIGITS:
    =========================
    First find maximum possible number of digits, then construct lexicographically largest.
    
    Time Complexity: O(target * 9) - DP computation
    Space Complexity: O(target) - DP array
    """
    # Step 1: Find maximum number of digits possible
    dp = [-1] * (target + 1)  # dp[i] = max digits achievable with cost i
    dp[0] = 0  # 0 cost, 0 digits
    
    for i in range(1, target + 1):
        for digit in range(1, 10):
            digit_cost = cost[digit - 1]
            
            if digit_cost <= i and dp[i - digit_cost] != -1:
                dp[i] = max(dp[i], dp[i - digit_cost] + 1)
    
    if dp[target] == -1:
        return "0"
    
    # Step 2: Construct lexicographically largest number
    max_digits = dp[target]
    result = []
    remaining_cost = target
    
    for pos in range(max_digits):
        # Try digits from 9 to 1 to get lexicographically largest
        for digit in range(9, 0, -1):
            digit_cost = cost[digit - 1]
            
            if (digit_cost <= remaining_cost and 
                dp[remaining_cost - digit_cost] == max_digits - pos - 1):
                result.append(str(digit))
                remaining_cost -= digit_cost
                break
    
    return "".join(result)


def largest_number_memoization(cost, target):
    """
    MEMOIZATION APPROACH:
    ====================
    Use memoization to cache subproblem results.
    
    Time Complexity: O(target * 9) - states × transitions
    Space Complexity: O(target) - memoization table
    """
    memo = {}
    
    def dp(remaining_cost):
        if remaining_cost == 0:
            return ""
        
        if remaining_cost < 0:
            return None  # Impossible
        
        if remaining_cost in memo:
            return memo[remaining_cost]
        
        best = None
        
        # Try each digit
        for digit in range(1, 10):
            digit_cost = cost[digit - 1]
            
            if digit_cost <= remaining_cost:
                sub_result = dp(remaining_cost - digit_cost)
                
                if sub_result is not None:
                    current = str(digit) + sub_result
                    
                    if (best is None or len(current) > len(best) or 
                        (len(current) == len(best) and current > best)):
                        best = current
        
        memo[remaining_cost] = best
        return best
    
    result = dp(target)
    return result if result else "0"


def largest_number_optimized_dp(cost, target):
    """
    OPTIMIZED DP APPROACH:
    =====================
    Optimize by finding cheapest digit first.
    
    Time Complexity: O(target) - optimized iteration
    Space Complexity: O(target) - DP array
    """
    # Find minimum cost to form any digit
    min_cost = min(cost)
    
    # dp[i] = maximum number of digits achievable with cost i
    dp = [-1] * (target + 1)
    dp[0] = 0
    
    for i in range(min_cost, target + 1):
        for digit in range(1, 10):
            digit_cost = cost[digit - 1]
            
            if digit_cost <= i and dp[i - digit_cost] != -1:
                dp[i] = max(dp[i], dp[i - digit_cost] + 1)
    
    if dp[target] == -1:
        return "0"
    
    # Construct result greedily
    result = []
    remaining_cost = target
    target_digits = dp[target]
    
    for pos in range(target_digits):
        # Find largest digit that maintains optimal solution
        for digit in range(9, 0, -1):
            digit_cost = cost[digit - 1]
            
            if (digit_cost <= remaining_cost and 
                dp[remaining_cost - digit_cost] == target_digits - pos - 1):
                result.append(str(digit))
                remaining_cost -= digit_cost
                break
    
    return "".join(result)


def largest_number_greedy_insight(cost, target):
    """
    GREEDY INSIGHT APPROACH:
    =======================
    Use mathematical insights for optimization.
    
    Time Complexity: O(target) - linear construction
    Space Complexity: O(target) - DP array
    """
    # Key insight: Always maximize number of digits first,
    # then make lexicographically largest
    
    # Find minimum cost among all digits
    min_cost = min(cost)
    
    # Check if target is achievable
    if target % min_cost != 0:
        # Use DP to check all possibilities
        dp = [False] * (target + 1)
        dp[0] = True
        
        for i in range(1, target + 1):
            for digit_cost in cost:
                if digit_cost <= i and dp[i - digit_cost]:
                    dp[i] = True
                    break
        
        if not dp[target]:
            return "0"
    
    # Find maximum number of digits
    max_digits = 0
    dp = [-1] * (target + 1)
    dp[0] = 0
    
    for i in range(1, target + 1):
        for j, digit_cost in enumerate(cost):
            if digit_cost <= i and dp[i - digit_cost] != -1:
                dp[i] = max(dp[i], dp[i - digit_cost] + 1)
    
    max_digits = dp[target]
    if max_digits == -1:
        return "0"
    
    # Construct lexicographically largest number
    result = []
    remaining_cost = target
    
    for _ in range(max_digits):
        for digit in range(9, 0, -1):
            digit_cost = cost[digit - 1]
            
            if (digit_cost <= remaining_cost and 
                dp[remaining_cost - digit_cost] == max_digits - len(result) - 1):
                result.append(str(digit))
                remaining_cost -= digit_cost
                break
    
    return "".join(result)


def largest_number_alternative_construction(cost, target):
    """
    ALTERNATIVE CONSTRUCTION METHOD:
    ==============================
    Different approach to construct the result.
    
    Time Complexity: O(target * 9) - DP + construction
    Space Complexity: O(target) - DP array
    """
    # Find if target is achievable and max digits
    dp = [-1] * (target + 1)
    dp[0] = 0
    
    for i in range(1, target + 1):
        for digit in range(1, 10):
            digit_cost = cost[digit - 1]
            if digit_cost <= i and dp[i - digit_cost] >= 0:
                dp[i] = max(dp[i], dp[i - digit_cost] + 1)
    
    if dp[target] <= 0:
        return "0"
    
    # Alternative construction: build from left to right
    result = []
    remaining_target = target
    remaining_digits = dp[target]
    
    while remaining_digits > 0:
        placed = False
        
        # Try to place the largest possible digit
        for digit in range(9, 0, -1):
            digit_cost = cost[digit - 1]
            
            if digit_cost <= remaining_target:
                # Check if we can still achieve remaining digits with remaining cost
                new_target = remaining_target - digit_cost
                new_digits = remaining_digits - 1
                
                if new_digits == 0:
                    if new_target == 0:
                        result.append(str(digit))
                        placed = True
                        break
                elif dp[new_target] >= new_digits:
                    result.append(str(digit))
                    remaining_target = new_target
                    remaining_digits = new_digits
                    placed = True
                    break
        
        if not placed:
            return "0"  # Should not happen with correct DP
    
    return "".join(result)


def largest_number_with_analysis(cost, target):
    """
    DETAILED ANALYSIS APPROACH:
    ==========================
    Provide detailed analysis of the problem structure.
    
    Time Complexity: O(target * 9) - DP computation
    Space Complexity: O(target) - DP array
    """
    print(f"Problem Analysis:")
    print(f"Cost array: {cost}")
    print(f"Target: {target}")
    
    # Analyze cost structure
    min_cost = min(cost)
    max_cost = max(cost)
    min_cost_digit = cost.index(min_cost) + 1
    
    print(f"Min cost: {min_cost} (digit {min_cost_digit})")
    print(f"Max cost: {max_cost}")
    
    # Find digits with same minimum cost
    min_cost_digits = [i + 1 for i, c in enumerate(cost) if c == min_cost]
    print(f"Digits with min cost: {min_cost_digits}")
    
    # Upper bound on number of digits
    max_possible_digits = target // min_cost
    print(f"Upper bound on digits: {max_possible_digits}")
    
    # Run optimized solution
    result = largest_number_optimized_dp(cost, target)
    print(f"Result: {result}")
    
    if result != "0":
        actual_cost = sum(cost[int(d) - 1] for d in result)
        print(f"Verification: cost = {actual_cost}, length = {len(result)}")
    
    return result


# Test cases
def test_largest_number():
    """Test all implementations with various inputs"""
    test_cases = [
        ([4,3,2,5,6,7,2,5,5], 5, "22"),
        ([7,6,5,5,5,6,8,7,8], 12, "85"),
        ([2,4,6,2,4,6,4,4,4], 5, "0"),
        ([6,10,15,40,40,40,40,40,40], 47, "32"),
        ([1,1,1,1,1,1,1,1,1], 9, "999999999"),
        ([5,4,4,5,5,5,5,5,5], 10, "33"),
        ([1,2,3,4,5,6,7,8,9], 10, "1111111111"),
        ([9,8,7,6,5,4,3,2,1], 10, "91"),
        ([2,2,2,2,2,2,2,2,2], 6, "333"),
        ([3,3,3,3,3,3,3,3,1], 9, "999999999")
    ]
    
    print("Testing Form Largest Integer Solutions:")
    print("=" * 70)
    
    for i, (cost, target, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: cost={cost}, target={target}")
        print(f"Expected: {expected}")
        
        # Test approaches (skip brute force for large inputs)
        if target <= 15:
            brute = largest_number_bruteforce(cost.copy(), target)
            print(f"Brute Force:      {brute:>12} {'✓' if brute == expected else '✗'}")
        
        dp_max = largest_number_dp_max_digits(cost.copy(), target)
        memo = largest_number_memoization(cost.copy(), target)
        optimized = largest_number_optimized_dp(cost.copy(), target)
        greedy = largest_number_greedy_insight(cost.copy(), target)
        alternative = largest_number_alternative_construction(cost.copy(), target)
        
        print(f"DP Max Digits:    {dp_max:>12} {'✓' if dp_max == expected else '✗'}")
        print(f"Memoization:      {memo:>12} {'✓' if memo == expected else '✗'}")
        print(f"Optimized:        {optimized:>12} {'✓' if optimized == expected else '✗'}")
        print(f"Greedy Insight:   {greedy:>12} {'✓' if greedy == expected else '✗'}")
        print(f"Alternative:      {alternative:>12} {'✓' if alternative == expected else '✗'}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    largest_number_with_analysis([4,3,2,5,6,7,2,5,5], 5)
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. Maximize number of digits first (greedy on length)")
    print("2. Then maximize lexicographic value (greedy on digits)")
    print("3. Two-phase approach: DP for max digits + greedy construction")
    print("4. Similar to unbounded knapsack but optimizing for length then value")
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Brute Force:      Time: O(9^target),    Space: O(target)")
    print("DP Max Digits:    Time: O(target*9),    Space: O(target)")
    print("Memoization:      Time: O(target*9),    Space: O(target)")
    print("Optimized:        Time: O(target),      Space: O(target)")
    print("Greedy Insight:   Time: O(target),      Space: O(target)")
    print("Alternative:      Time: O(target*9),    Space: O(target)")


if __name__ == "__main__":
    test_largest_number()


"""
PATTERN RECOGNITION:
==================
This is an Unbounded Knapsack problem with lexicographic optimization:
- Items: digits 1-9 with respective costs
- Constraint: Total cost must equal target exactly
- Goal: Maximize number of digits, then lexicographic value
- Two-phase optimization: length first, then lexicographic order

KEY INSIGHT - TWO-PHASE OPTIMIZATION:
====================================
1. **Phase 1**: Find maximum number of digits achievable with target cost
2. **Phase 2**: Construct lexicographically largest number with that many digits

This is critical because:
- "999" > "1000000" lexicographically, but "1000000" has more digits
- We want more digits first, then among equal-length numbers, larger value

MATHEMATICAL APPROACH:
=====================
**Phase 1 - Maximum Digits**: Classic unbounded knapsack
- dp[i] = maximum digits achievable with cost i
- dp[i] = max(dp[i], dp[i - cost[j]] + 1) for all digits j

**Phase 2 - Lexicographic Construction**: Greedy
- For each position, choose largest digit that maintains optimality
- Check: dp[remaining_cost - digit_cost] == remaining_positions - 1

STATE DEFINITION:
================
dp[cost] = maximum number of digits achievable with exactly this cost

RECURRENCE RELATION:
===================
dp[i] = max(dp[i], dp[i - cost[digit]] + 1) for all digits

Base case: dp[0] = 0 (no cost, no digits)

CONSTRUCTION ALGORITHM:
======================
```python
for position in range(max_digits):
    for digit in range(9, 1, -1):  # Try largest digits first
        if can_complete_remaining_positions_with_remaining_cost(digit):
            result.append(digit)
            update_remaining_cost_and_positions()
            break
```

ALGORITHM PROGRESSION:
=====================
1. **Brute Force**: O(9^target) - try all digit combinations
2. **Memoization**: O(target×9) - cache subproblems
3. **DP + Greedy**: O(target×9) - two-phase approach
4. **Optimized**: O(target) - use min-cost insight for bounds

OPTIMIZATION TECHNIQUES:
=======================
1. **Min Cost Insight**: Maximum possible digits = target / min_cost
2. **Early Termination**: Stop DP early when target reached
3. **Greedy Construction**: Build result left-to-right with largest digits
4. **State Pruning**: Only track achievable states

MATHEMATICAL PROPERTIES:
=======================
1. **Unbounded Nature**: Each digit can be used multiple times
2. **Exact Target**: Must use exactly target cost (not ≤ target)
3. **Lexicographic Order**: Among equal-length strings, larger is better
4. **Length Priority**: More digits always better than fewer digits

COMPARISON WITH RELATED PROBLEMS:
================================
- **Coin Change (322)**: Minimize coins (minimize count)
- **Coin Change 2 (518)**: Count ways (count solutions)
- **This Problem**: Maximize digits, then value (lexicographic optimization)

EDGE CASES:
==========
1. **Impossible Target**: No combination of costs sums to target
2. **Single Digit**: Only one digit affordable
3. **All Same Cost**: Choose largest digits greedily
4. **Minimum Cost = 1**: Can always achieve target with digit 1

CONSTRUCTION CORRECTNESS:
========================
The greedy construction works because:
1. We know the optimal number of digits from DP
2. At each position, choosing the largest valid digit is always optimal
3. Validity check ensures we can complete remaining positions

VARIANTS TO PRACTICE:
====================
- Unbounded knapsack variants
- Coin change problems (322, 518)
- Lexicographic optimization problems
- String construction with constraints

INTERVIEW TIPS:
==============
1. **Recognize as two-phase optimization** (length then value)
2. Start with unbounded knapsack for maximum digits
3. Show greedy construction for lexicographic optimization
4. **Critical insight**: Length priority over lexicographic value
5. Explain why greedy construction works
6. Handle edge case: impossible targets
7. Discuss time complexity optimization opportunities
8. Compare with standard knapsack problems
9. Show example of construction process
10. Mention alternative approaches (memoization vs DP)
"""
