"""
LeetCode 879: Profitable Schemes
Difficulty: Hard
Category: Knapsack Problems (3D DP variant)

PROBLEM DESCRIPTION:
===================
There is a group of n members, and a list of various crimes. The ith crime generates a profit[i] 
and requires group[i] members to participate in it.

If a member participates in one crime, that member can't participate in another crime.

Let's call a profitable scheme any subset of these crimes that generates at least minProfit profit, 
and the total number of members participating in that subset is at most n.

Return the number of schemes that can be chosen. Since the answer may be very large, return it modulo 10^9 + 7.

Example 1:
Input: n = 5, minProfit = 3, group = [2,2], profit = [2,3]
Output: 2
Explanation: To make a profit of at least 3, the group could either commit crimes 0 and 1, or just crime 1.
In total, there are 2 schemes.

Example 2:
Input: n = 10, minProfit = 5, group = [2,3,5], profit = [6,7,8]
Output: 7
Explanation: To make a profit of at least 5, the group could commit any crimes, as long as they commit one.
There are 7 possible schemes: (0), (1), (2), (0,1), (0,2), (1,2), and (0,1,2).

Constraints:
- 1 <= n <= 100
- 0 <= minProfit <= 100
- 1 <= group.length <= 100
- 1 <= group[i] <= 100
- profit.length == group.length
- 0 <= profit[i] <= 100
"""

def profitable_schemes_bruteforce(n, minProfit, group, profit):
    """
    BRUTE FORCE APPROACH - RECURSION:
    ================================
    Try all possible subsets of crimes.
    
    Time Complexity: O(2^len(group)) - exponential subsets
    Space Complexity: O(len(group)) - recursion stack depth
    """
    MOD = 10**9 + 7
    
    def count_schemes(index, people_used, profit_gained):
        if people_used > n:
            return 0
        
        if index == len(group):
            return 1 if profit_gained >= minProfit else 0
        
        # Two choices: include current crime or skip it
        skip = count_schemes(index + 1, people_used, profit_gained)
        include = count_schemes(index + 1, people_used + group[index], 
                              profit_gained + profit[index])
        
        return (skip + include) % MOD
    
    return count_schemes(0, 0, 0)


def profitable_schemes_memoization(n, minProfit, group, profit):
    """
    DYNAMIC PROGRAMMING - TOP DOWN (MEMOIZATION):
    ============================================
    Use memoization to cache subproblem results.
    
    Time Complexity: O(len(group) * n * minProfit) - states
    Space Complexity: O(len(group) * n * minProfit) - memoization table
    """
    MOD = 10**9 + 7
    memo = {}
    
    def count_schemes(index, people_used, profit_gained):
        if people_used > n:
            return 0
        
        if index == len(group):
            return 1 if profit_gained >= minProfit else 0
        
        # Cap profit_gained at minProfit for memoization efficiency
        profit_key = min(profit_gained, minProfit)
        
        if (index, people_used, profit_key) in memo:
            return memo[(index, people_used, profit_key)]
        
        # Skip current crime
        skip = count_schemes(index + 1, people_used, profit_gained)
        
        # Include current crime
        include = count_schemes(index + 1, people_used + group[index], 
                              profit_gained + profit[index])
        
        result = (skip + include) % MOD
        memo[(index, people_used, profit_key)] = result
        return result
    
    return count_schemes(0, 0, 0)


def profitable_schemes_dp_3d(n, minProfit, group, profit):
    """
    3D DYNAMIC PROGRAMMING:
    =======================
    Use 3D DP table for crimes, people, and profit.
    
    Time Complexity: O(len(group) * n * minProfit) - fill DP table
    Space Complexity: O(len(group) * n * minProfit) - 3D DP table
    """
    MOD = 10**9 + 7
    crimes = len(group)
    
    # dp[i][j][k] = ways to choose from first i crimes, using j people, with profit k
    dp = [[[0] * (minProfit + 1) for _ in range(n + 1)] for _ in range(crimes + 1)]
    
    # Base case: 0 crimes, 0 people, 0 profit = 1 way (choose nothing)
    dp[0][0][0] = 1
    
    for i in range(1, crimes + 1):
        crime_group = group[i - 1]
        crime_profit = profit[i - 1]
        
        for j in range(n + 1):
            for k in range(minProfit + 1):
                # Don't include current crime
                dp[i][j][k] = dp[i - 1][j][k]
                
                # Include current crime if possible
                if j >= crime_group:
                    prev_profit = max(0, k - crime_profit)
                    dp[i][j][k] = (dp[i][j][k] + dp[i - 1][j - crime_group][prev_profit]) % MOD
    
    # Sum all ways with profit >= minProfit
    result = 0
    for j in range(n + 1):
        result = (result + dp[crimes][j][minProfit]) % MOD
    
    return result


def profitable_schemes_dp_2d(n, minProfit, group, profit):
    """
    2D DYNAMIC PROGRAMMING (SPACE OPTIMIZED):
    =========================================
    Use 2D DP table, process crimes one by one.
    
    Time Complexity: O(len(group) * n * minProfit) - same iterations
    Space Complexity: O(n * minProfit) - 2D DP table
    """
    MOD = 10**9 + 7
    
    # dp[j][k] = ways using j people with profit k
    dp = [[0] * (minProfit + 1) for _ in range(n + 1)]
    dp[0][0] = 1  # Base case: 0 people, 0 profit = 1 way
    
    for i in range(len(group)):
        crime_group = group[i]
        crime_profit = profit[i]
        
        # Process in reverse order to avoid using updated values
        for j in range(n, crime_group - 1, -1):
            for k in range(minProfit, -1, -1):
                new_profit = min(minProfit, k + crime_profit)
                dp[j][new_profit] = (dp[j][new_profit] + dp[j - crime_group][k]) % MOD
    
    # Sum all ways with profit >= minProfit
    result = 0
    for j in range(n + 1):
        result = (result + dp[j][minProfit]) % MOD
    
    return result


def profitable_schemes_optimized(n, minProfit, group, profit):
    """
    OPTIMIZED DP WITH PROFIT CAPPING:
    ================================
    Optimize by capping profit at minProfit.
    
    Time Complexity: O(len(group) * n * minProfit) - same complexity
    Space Complexity: O(n * minProfit) - 2D DP table
    """
    MOD = 10**9 + 7
    
    # Key insight: We only care about profit >= minProfit
    # So we can cap profit at minProfit in our DP state
    
    dp = [[0] * (minProfit + 1) for _ in range(n + 1)]
    dp[0][0] = 1
    
    for crime_group, crime_profit in zip(group, profit):
        # Process in reverse order
        for j in range(n, crime_group - 1, -1):
            for k in range(minProfit + 1):
                # Add current crime
                new_profit = min(minProfit, k + crime_profit)
                dp[j][new_profit] = (dp[j][new_profit] + dp[j - crime_group][k]) % MOD
    
    return dp[n][minProfit] if n >= 0 else 0


def profitable_schemes_alternative_state(n, minProfit, group, profit):
    """
    ALTERNATIVE STATE DEFINITION:
    ============================
    Different way to define DP state.
    
    Time Complexity: O(len(group) * n * minProfit) - same complexity
    Space Complexity: O(n * minProfit) - 2D DP table
    """
    MOD = 10**9 + 7
    
    # dp[j][k] = ways to use exactly j people and get exactly k profit
    # Then sum over all valid combinations
    
    prev_dp = [[0] * (minProfit + 1) for _ in range(n + 1)]
    prev_dp[0][0] = 1
    
    for crime_group, crime_profit in zip(group, profit):
        curr_dp = [[0] * (minProfit + 1) for _ in range(n + 1)]
        
        # Copy previous state (don't include current crime)
        for j in range(n + 1):
            for k in range(minProfit + 1):
                curr_dp[j][k] = prev_dp[j][k]
        
        # Include current crime
        for j in range(crime_group, n + 1):
            for k in range(minProfit + 1):
                new_profit = min(minProfit, k + crime_profit)
                curr_dp[j][new_profit] = (curr_dp[j][new_profit] + 
                                        prev_dp[j - crime_group][k]) % MOD
        
        prev_dp = curr_dp
    
    return prev_dp[n][minProfit]


def profitable_schemes_with_details(n, minProfit, group, profit):
    """
    DETAILED SOLUTION WITH SCHEME TRACKING:
    ======================================
    Return count and show some example schemes.
    
    Time Complexity: O(len(group) * n * minProfit) - DP + scheme generation
    Space Complexity: O(len(group) * n * minProfit) - store schemes
    """
    MOD = 10**9 + 7
    
    # For small inputs, generate actual schemes
    if len(group) <= 10 and n <= 20 and minProfit <= 20:
        def generate_schemes(index, people_used, profit_gained, current_scheme):
            if people_used > n:
                return []
            
            if index == len(group):
                if profit_gained >= minProfit:
                    return [current_scheme[:]]
                else:
                    return []
            
            schemes = []
            
            # Skip current crime
            schemes.extend(generate_schemes(index + 1, people_used, 
                                          profit_gained, current_scheme))
            
            # Include current crime
            current_scheme.append(index)
            schemes.extend(generate_schemes(index + 1, people_used + group[index], 
                                          profit_gained + profit[index], current_scheme))
            current_scheme.pop()
            
            return schemes
        
        all_schemes = generate_schemes(0, 0, 0, [])
        return len(all_schemes), all_schemes
    
    # For larger inputs, just return count
    count = profitable_schemes_optimized(n, minProfit, group, profit)
    return count, []


def profitable_schemes_mathematical_analysis(n, minProfit, group, profit):
    """
    MATHEMATICAL ANALYSIS:
    =====================
    Analyze the problem structure mathematically.
    
    Time Complexity: O(len(group) * n * minProfit) - DP computation
    Space Complexity: O(n * minProfit) - DP table
    """
    MOD = 10**9 + 7
    
    print(f"Problem Analysis:")
    print(f"n (max people): {n}")
    print(f"minProfit: {minProfit}")
    print(f"Crimes: {len(group)}")
    print(f"Group sizes: {group}")
    print(f"Profits: {profit}")
    
    # Calculate some statistics
    total_people = sum(group)
    total_profit = sum(profit)
    avg_efficiency = total_profit / total_people if total_people > 0 else 0
    
    print(f"Total people needed for all crimes: {total_people}")
    print(f"Total profit from all crimes: {total_profit}")
    print(f"Average profit per person: {avg_efficiency:.2f}")
    
    # Check if it's possible to achieve minProfit
    if total_profit < minProfit:
        print("Impossible: Total profit < minProfit")
        return 0
    
    if total_people > n and total_profit == minProfit:
        print("Impossible: Need more people than available for minimum profit")
        return 0
    
    # Run the optimized solution
    result = profitable_schemes_optimized(n, minProfit, group, profit)
    print(f"Number of profitable schemes: {result}")
    
    return result


# Test cases
def test_profitable_schemes():
    """Test all implementations with various inputs"""
    test_cases = [
        (5, 3, [2,2], [2,3], 2),
        (10, 5, [2,3,5], [6,7,8], 7),
        (1, 1, [1], [1], 1),
        (1, 1, [2], [1], 0),
        (5, 3, [2,2,2], [2,2,2], 0),
        (10, 5, [5,5,5], [5,5,5], 1),
        (100, 100, [1,1,1], [1,1,1], 0),
        (1, 0, [1], [0], 2),  # Include or skip, both valid
        (1, 0, [1], [1], 2),  # Include (profit 1) or skip (profit 0), both >= 0
        (10, 10, [1,2,3,4,5], [1,2,3,4,5], 16)
    ]
    
    print("Testing Profitable Schemes Solutions:")
    print("=" * 70)
    
    for i, (n, minProfit, group, profit, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: n={n}, minProfit={minProfit}")
        print(f"Group: {group}, Profit: {profit}")
        print(f"Expected: {expected}")
        
        # Test approaches (skip brute force for large inputs)
        if len(group) <= 8:
            brute = profitable_schemes_bruteforce(n, minProfit, group.copy(), profit.copy())
            print(f"Brute Force:      {brute:>6} {'✓' if brute == expected else '✗'}")
        
        memo = profitable_schemes_memoization(n, minProfit, group.copy(), profit.copy())
        dp_3d = profitable_schemes_dp_3d(n, minProfit, group.copy(), profit.copy())
        dp_2d = profitable_schemes_dp_2d(n, minProfit, group.copy(), profit.copy())
        optimized = profitable_schemes_optimized(n, minProfit, group.copy(), profit.copy())
        alternative = profitable_schemes_alternative_state(n, minProfit, group.copy(), profit.copy())
        
        print(f"Memoization:      {memo:>6} {'✓' if memo == expected else '✗'}")
        print(f"3D DP:            {dp_3d:>6} {'✓' if dp_3d == expected else '✗'}")
        print(f"2D DP:            {dp_2d:>6} {'✓' if dp_2d == expected else '✗'}")
        print(f"Optimized:        {optimized:>6} {'✓' if optimized == expected else '✗'}")
        print(f"Alternative:      {alternative:>6} {'✓' if alternative == expected else '✗'}")
        
        # Show actual schemes for small cases
        if len(group) <= 6 and expected <= 20:
            count, schemes = profitable_schemes_with_details(n, minProfit, group.copy(), profit.copy())
            if schemes:
                print(f"Example schemes: {schemes[:5]}...")  # Show first 5
    
    # Mathematical analysis example
    print(f"\n" + "=" * 70)
    print("MATHEMATICAL ANALYSIS EXAMPLE:")
    print("-" * 40)
    profitable_schemes_mathematical_analysis(5, 3, [2,2], [2,3])
    
    print("\n" + "=" * 70)
    print("Key Insight:")
    print("3D Knapsack: crimes × people × profit")
    print("Profit capping: Only care about profit >= minProfit")
    print("State: (people_used, profit_gained) → count of ways")
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Brute Force:      Time: O(2^crimes),           Space: O(crimes)")
    print("Memoization:      Time: O(crimes*n*minProfit), Space: O(crimes*n*minProfit)")
    print("3D DP:            Time: O(crimes*n*minProfit), Space: O(crimes*n*minProfit)")
    print("2D DP:            Time: O(crimes*n*minProfit), Space: O(n*minProfit)")
    print("Optimized:        Time: O(crimes*n*minProfit), Space: O(n*minProfit)")
    print("Alternative:      Time: O(crimes*n*minProfit), Space: O(n*minProfit)")


if __name__ == "__main__":
    test_profitable_schemes()


"""
PATTERN RECOGNITION:
==================
This is a 3D Knapsack problem:
- Items: crimes with (group_size, profit) pairs
- Constraints: Total people ≤ n, Total profit ≥ minProfit
- Goal: Count number of valid combinations (not optimize value)
- 3D state: (crime_index, people_used, profit_gained)

KEY INSIGHT - PROFIT CAPPING:
============================
Critical optimization: Cap profit at minProfit in DP state
- We only care if profit ≥ minProfit, not exact amount
- This reduces state space and improves efficiency
- dp[people][min(profit, minProfit)] instead of dp[people][profit]

STATE DEFINITION:
================
dp[j][k] = number of ways to use j people and gain exactly k profit
(where k is capped at minProfit)

RECURRENCE RELATION:
===================
For each crime (group[i], profit[i]):
dp[j][k] += dp[j - group[i]][k - profit[i]]

Process in reverse order to avoid using updated values.

ALGORITHM PROGRESSION:
=====================
1. **Brute Force**: O(2^crimes) - try all subsets
2. **Memoization**: O(crimes×n×minProfit) - cache states
3. **3D DP**: O(crimes×n×minProfit) - tabulation
4. **2D DP**: O(crimes×n×minProfit) time, O(n×minProfit) space
5. **Optimized**: Same complexity with profit capping optimization

OPTIMIZATION TECHNIQUES:
=======================
1. **Profit Capping**: Cap profit at minProfit in DP state
2. **Space Optimization**: Reduce from 3D to 2D DP
3. **Reverse Processing**: Avoid using updated values in same iteration
4. **Early Termination**: Check impossible cases upfront

MATHEMATICAL PROPERTIES:
=======================
1. **Counting Problem**: Count valid combinations, not optimization
2. **Multiple Constraints**: Both people and profit constraints
3. **At Least Constraint**: Profit ≥ minProfit (different from exact)
4. **Modular Arithmetic**: Results modulo 10^9 + 7

COMPARISON WITH OTHER KNAPSACK VARIANTS:
======================================
- **0/1 Knapsack**: Single constraint (weight), maximize value
- **Unbounded Knapsack**: Unlimited use, single constraint
- **2D Knapsack**: Two constraints (like Ones and Zeros)
- **This Problem**: Two constraints + counting + "at least" condition

EDGE CASES:
==========
1. **minProfit = 0**: Always include empty set (1 way)
2. **Impossible Cases**: Total profit < minProfit
3. **Single Crime**: Either include or exclude
4. **All Crimes**: Check if total people ≤ n

VARIANTS TO PRACTICE:
====================
- Ones and Zeros (474) - 2D knapsack counting
- Coin Change 2 (518) - unbounded knapsack counting  
- Target Sum (494) - assign signs counting
- Partition Equal Subset Sum (416) - subset selection

INTERVIEW TIPS:
==============
1. Recognize as 3D knapsack counting problem
2. Identify the two constraints (people, profit)
3. Show brute force subset enumeration first
4. Add memoization with 3D state
5. **Critical**: Explain profit capping optimization
6. Space optimize from 3D to 2D
7. Handle modular arithmetic correctly
8. Discuss edge cases (minProfit=0, impossible cases)
9. Compare with other knapsack variants
"""
