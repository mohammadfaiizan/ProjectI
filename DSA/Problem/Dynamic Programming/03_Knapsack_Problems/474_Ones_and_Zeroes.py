"""
LeetCode 474: Ones and Zeroes
Difficulty: Medium
Category: Knapsack Problems (2D Knapsack variant)

PROBLEM DESCRIPTION:
===================
You are given an array of binary strings strs and two integers m and n.

Return the size of the largest subset of strs such that there are at most m 0's and n 1's in the subset.

A set x is a subset of a set y if all elements of x are also elements of y.

Example 1:
Input: strs = ["10","0001","111001","1","0"], m = 5, n = 3
Output: 4
Explanation: The largest subset with at most 5 0's and 3 1's is {"10", "0001", "1", "0"}, so the answer is 4.

Example 2:
Input: strs = ["10","0","1"], m = 1, n = 1
Output: 2
Explanation: The largest subset is {"0", "1"}, so the answer is 2.

Constraints:
- 1 <= strs.length <= 600
- 1 <= strs[i].length <= 100
- strs[i] consists only of digits '0' and '1'.
- 1 <= m, n <= 100
"""

def find_max_form_bruteforce(strs, m, n):
    """
    BRUTE FORCE APPROACH - RECURSION:
    ================================
    Try all possible subsets and find the largest valid one.
    
    Time Complexity: O(2^len(strs)) - exponential subsets
    Space Complexity: O(len(strs)) - recursion stack depth
    """
    def count_zeros_ones(s):
        zeros = s.count('0')
        ones = s.count('1')
        return zeros, ones
    
    def max_subset(index, zeros_left, ones_left):
        if index >= len(strs):
            return 0
        
        # Option 1: Skip current string
        skip = max_subset(index + 1, zeros_left, ones_left)
        
        # Option 2: Include current string if possible
        zeros, ones = count_zeros_ones(strs[index])
        include = 0
        if zeros <= zeros_left and ones <= ones_left:
            include = 1 + max_subset(index + 1, zeros_left - zeros, ones_left - ones)
        
        return max(skip, include)
    
    return max_subset(0, m, n)


def find_max_form_memoization(strs, m, n):
    """
    DYNAMIC PROGRAMMING - TOP DOWN (MEMOIZATION):
    ============================================
    Use memoization to cache subproblems.
    
    Time Complexity: O(len(strs) * m * n) - states: index, zeros, ones
    Space Complexity: O(len(strs) * m * n) - memoization table
    """
    def count_zeros_ones(s):
        zeros = s.count('0')
        ones = s.count('1')
        return zeros, ones
    
    memo = {}
    
    def max_subset(index, zeros_left, ones_left):
        if index >= len(strs):
            return 0
        
        if (index, zeros_left, ones_left) in memo:
            return memo[(index, zeros_left, ones_left)]
        
        # Skip current string
        skip = max_subset(index + 1, zeros_left, ones_left)
        
        # Include current string if possible
        zeros, ones = count_zeros_ones(strs[index])
        include = 0
        if zeros <= zeros_left and ones <= ones_left:
            include = 1 + max_subset(index + 1, zeros_left - zeros, ones_left - ones)
        
        result = max(skip, include)
        memo[(index, zeros_left, ones_left)] = result
        return result
    
    return max_subset(0, m, n)


def find_max_form_dp_3d(strs, m, n):
    """
    3D DYNAMIC PROGRAMMING:
    =======================
    Use 3D DP table for strings, zeros, and ones.
    
    Time Complexity: O(len(strs) * m * n) - fill DP table
    Space Complexity: O(len(strs) * m * n) - 3D DP table
    """
    def count_zeros_ones(s):
        zeros = s.count('0')
        ones = s.count('1')
        return zeros, ones
    
    L = len(strs)
    
    # dp[i][j][k] = max strings using first i strings with j zeros and k ones
    dp = [[[0] * (n + 1) for _ in range(m + 1)] for _ in range(L + 1)]
    
    for i in range(1, L + 1):
        zeros, ones = count_zeros_ones(strs[i - 1])
        
        for j in range(m + 1):
            for k in range(n + 1):
                # Don't include current string
                dp[i][j][k] = dp[i - 1][j][k]
                
                # Include current string if possible
                if j >= zeros and k >= ones:
                    dp[i][j][k] = max(dp[i][j][k], 
                                    dp[i - 1][j - zeros][k - ones] + 1)
    
    return dp[L][m][n]


def find_max_form_dp_2d(strs, m, n):
    """
    2D DYNAMIC PROGRAMMING (SPACE OPTIMIZED):
    =========================================
    Use 2D DP table, process strings one by one.
    
    Time Complexity: O(len(strs) * m * n) - same iterations
    Space Complexity: O(m * n) - 2D DP table
    """
    def count_zeros_ones(s):
        zeros = s.count('0')
        ones = s.count('1')
        return zeros, ones
    
    # dp[i][j] = max strings with i zeros and j ones
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for s in strs:
        zeros, ones = count_zeros_ones(s)
        
        # Process in reverse order to avoid using updated values
        for i in range(m, zeros - 1, -1):
            for j in range(n, ones - 1, -1):
                dp[i][j] = max(dp[i][j], dp[i - zeros][j - ones] + 1)
    
    return dp[m][n]


def find_max_form_optimized(strs, m, n):
    """
    OPTIMIZED 2D DP:
    ===============
    Optimized version with early termination and preprocessing.
    
    Time Complexity: O(len(strs) * m * n) - worst case, often better
    Space Complexity: O(m * n) - 2D DP table
    """
    # Preprocess strings to count zeros and ones
    string_counts = []
    for s in strs:
        zeros = s.count('0')
        ones = s.count('1')
        string_counts.append((zeros, ones))
    
    # Sort by total length for potential optimization
    string_counts.sort(key=lambda x: x[0] + x[1])
    
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for zeros, ones in string_counts:
        # Early termination: if string too large, skip
        if zeros > m or ones > n:
            continue
        
        # Update DP table in reverse order
        for i in range(m, zeros - 1, -1):
            for j in range(n, ones - 1, -1):
                dp[i][j] = max(dp[i][j], dp[i - zeros][j - ones] + 1)
    
    return dp[m][n]


def find_max_form_with_subset(strs, m, n):
    """
    FIND MAX FORM AND ACTUAL SUBSET:
    ===============================
    Return both maximum size and the actual subset.
    
    Time Complexity: O(len(strs) * m * n) - DP + reconstruction
    Space Complexity: O(len(strs) * m * n) - store choices
    """
    def count_zeros_ones(s):
        zeros = s.count('0')
        ones = s.count('1')
        return zeros, ones
    
    L = len(strs)
    
    # DP table and choice tracking
    dp = [[[0] * (n + 1) for _ in range(m + 1)] for _ in range(L + 1)]
    choice = [[[False] * (n + 1) for _ in range(m + 1)] for _ in range(L + 1)]
    
    for i in range(1, L + 1):
        zeros, ones = count_zeros_ones(strs[i - 1])
        
        for j in range(m + 1):
            for k in range(n + 1):
                # Don't include current string
                dp[i][j][k] = dp[i - 1][j][k]
                
                # Include current string if possible
                if j >= zeros and k >= ones:
                    include_value = dp[i - 1][j - zeros][k - ones] + 1
                    if include_value > dp[i][j][k]:
                        dp[i][j][k] = include_value
                        choice[i][j][k] = True
    
    # Reconstruct subset
    subset = []
    i, j, k = L, m, n
    
    while i > 0:
        if choice[i][j][k]:
            subset.append(strs[i - 1])
            zeros, ones = count_zeros_ones(strs[i - 1])
            j -= zeros
            k -= ones
        i -= 1
    
    subset.reverse()
    return dp[L][m][n], subset


def find_max_form_alternative_dp(strs, m, n):
    """
    ALTERNATIVE DP FORMULATION:
    ==========================
    Different way to think about the problem.
    
    Time Complexity: O(len(strs) * m * n) - same complexity
    Space Complexity: O(m * n) - 2D DP table
    """
    # dp[i][j] = maximum number of strings with exactly i zeros and j ones
    # We'll use <= instead of exactly
    
    dp = [[-1] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = 0
    
    for s in strs:
        zeros = s.count('0')
        ones = s.count('1')
        
        # Create new DP table for this iteration
        new_dp = [row[:] for row in dp]
        
        for i in range(m + 1):
            for j in range(n + 1):
                if dp[i][j] != -1:  # Valid state
                    # Try adding current string
                    new_i = i + zeros
                    new_j = j + ones
                    
                    if new_i <= m and new_j <= n:
                        new_dp[new_i][new_j] = max(new_dp[new_i][new_j], 
                                                  dp[i][j] + 1)
        
        dp = new_dp
    
    # Find maximum value in DP table
    result = 0
    for i in range(m + 1):
        for j in range(n + 1):
            if dp[i][j] != -1:
                result = max(result, dp[i][j])
    
    return result


def find_max_form_rolling_array(strs, m, n):
    """
    ROLLING ARRAY OPTIMIZATION:
    ===========================
    Further space optimization using rolling arrays.
    
    Time Complexity: O(len(strs) * m * n) - same iterations
    Space Complexity: O(m * n) - single 2D array
    """
    def count_zeros_ones(s):
        return s.count('0'), s.count('1')
    
    # Use single DP array that we update in place
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for s in strs:
        zeros, ones = count_zeros_ones(s)
        
        # Update in reverse order to avoid using updated values
        for i in range(m, zeros - 1, -1):
            for j in range(n, ones - 1, -1):
                if dp[i - zeros][j - ones] + 1 > dp[i][j]:
                    dp[i][j] = dp[i - zeros][j - ones] + 1
    
    return dp[m][n]


# Test cases
def test_find_max_form():
    """Test all implementations with various inputs"""
    test_cases = [
        (["10","0001","111001","1","0"], 5, 3, 4),
        (["10","0","1"], 1, 1, 2),
        (["10", "1", "0"], 1, 1, 2),
        (["0", "1"], 1, 1, 2),
        (["0"], 1, 0, 1),
        (["1"], 0, 1, 1),
        (["111", "000"], 3, 0, 1),
        (["00", "11"], 2, 2, 2),
        (["0", "0", "1", "1"], 2, 2, 4),
        (["0011", "1100"], 2, 2, 1)
    ]
    
    print("Testing Ones and Zeroes Solutions:")
    print("=" * 70)
    
    for i, (strs, m, n, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: strs = {strs}, m = {m}, n = {n}")
        print(f"Expected: {expected}")
        
        # Test approaches (skip brute force for large inputs)
        if len(strs) <= 8:
            brute = find_max_form_bruteforce(strs.copy(), m, n)
            print(f"Brute Force:      {brute:>3} {'✓' if brute == expected else '✗'}")
        
        memo = find_max_form_memoization(strs.copy(), m, n)
        dp_3d = find_max_form_dp_3d(strs.copy(), m, n)
        dp_2d = find_max_form_dp_2d(strs.copy(), m, n)
        optimized = find_max_form_optimized(strs.copy(), m, n)
        alternative = find_max_form_alternative_dp(strs.copy(), m, n)
        rolling = find_max_form_rolling_array(strs.copy(), m, n)
        
        print(f"Memoization:      {memo:>3} {'✓' if memo == expected else '✗'}")
        print(f"3D DP:            {dp_3d:>3} {'✓' if dp_3d == expected else '✗'}")
        print(f"2D DP:            {dp_2d:>3} {'✓' if dp_2d == expected else '✗'}")
        print(f"Optimized:        {optimized:>3} {'✓' if optimized == expected else '✗'}")
        print(f"Alternative:      {alternative:>3} {'✓' if alternative == expected else '✗'}")
        print(f"Rolling Array:    {rolling:>3} {'✓' if rolling == expected else '✗'}")
        
        # Show actual subset for small cases
        if expected > 0 and len(strs) <= 6:
            max_size, subset = find_max_form_with_subset(strs.copy(), m, n)
            zeros_used = sum(s.count('0') for s in subset)
            ones_used = sum(s.count('1') for s in subset)
            print(f"Subset: {subset} (0s: {zeros_used}, 1s: {ones_used})")
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Brute Force:      Time: O(2^len(strs)), Space: O(len(strs))")
    print("Memoization:      Time: O(L*m*n),       Space: O(L*m*n)")
    print("3D DP:            Time: O(L*m*n),       Space: O(L*m*n)")
    print("2D DP:            Time: O(L*m*n),       Space: O(m*n)")
    print("Optimized:        Time: O(L*m*n),       Space: O(m*n)")
    print("Alternative:      Time: O(L*m*n),       Space: O(m*n)")
    print("Rolling Array:    Time: O(L*m*n),       Space: O(m*n)")


if __name__ == "__main__":
    test_find_max_form()


"""
PATTERN RECOGNITION:
==================
This is a 2D Knapsack problem:
- Items: binary strings
- Two constraints: maximum 0s (m) and maximum 1s (n)
- Goal: maximize number of items (strings) selected
- Each string has "weight" in both dimensions

KEY INSIGHT - 2D KNAPSACK:
==========================
Extension of classic 0/1 knapsack to 2 dimensions:
- Weight 1: number of zeros in string
- Weight 2: number of ones in string
- Value: 1 (each string has same value)
- Constraints: total zeros ≤ m, total ones ≤ n

STATE DEFINITION:
================
dp[i][j] = maximum number of strings using at most i zeros and j ones

RECURRENCE RELATION:
===================
For each string s with zeros and ones:
dp[i][j] = max(dp[i][j], dp[i-zeros][j-ones] + 1)

Update in reverse order to avoid using updated values in same iteration.

SPACE OPTIMIZATION:
==================
Can reduce from 3D (strings × zeros × ones) to 2D (zeros × ones)
Process strings one by one, updating DP table in place.

ALGORITHM PROGRESSION:
=====================
1. **Brute Force**: O(2^L) - try all subsets
2. **Memoization**: O(L×m×n) - cache subproblems
3. **3D DP**: O(L×m×n) - tabulation approach
4. **2D DP**: O(L×m×n) time, O(m×n) space - space optimized
5. **Rolling Array**: Same complexity, optimized implementation

KEY IMPLEMENTATION DETAILS:
===========================
1. **Reverse Order Updates**: Prevent using updated values in same iteration
2. **Preprocessing**: Count zeros/ones for each string once
3. **Early Termination**: Skip strings that exceed constraints
4. **Subset Reconstruction**: Track choices for actual subset

VARIANTS TO PRACTICE:
====================
- 0/1 Knapsack (classic) - single constraint
- Coin Change (322) - unbounded version
- Partition Equal Subset Sum (416) - subset selection
- Target Sum (494) - assign +/- signs

INTERVIEW TIPS:
==============
1. Recognize as 2D knapsack extension
2. Start with recursive solution
3. Add memoization for optimization
4. Show space optimization to 2D
5. Explain reverse order updates
6. Handle edge cases (empty strings, zero constraints)
7. Discuss time/space trade-offs
8. Mention subset reconstruction if needed
"""
