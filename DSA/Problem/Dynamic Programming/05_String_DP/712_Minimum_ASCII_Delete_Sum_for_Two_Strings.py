"""
LeetCode 712: Minimum ASCII Delete Sum for Two Strings
Difficulty: Medium
Category: String DP

PROBLEM DESCRIPTION:
===================
Given two strings s1 and s2, return the lowest ASCII sum of deleted characters to make two strings equal.

Example 1:
Input: s1 = "sea", s2 = "eat"
Output: 231
Explanation: Deleting "s" from "sea" adds the ASCII value of "s" (115) to the sum.
Deleting "t" from "eat" adds 116 to the sum.
At the end, both strings are equal, and 115 + 116 = 231 is the minimum sum possible to achieve this.

Example 2:
Input: s1 = "delete", s2 = "leet"
Output: 403
Explanation: Deleting "de" from "delete" to turn the string into "lete".
Deleting "e" from "leet" adds 101 to the sum.
Next, deleting "e" from "lete" adds 101 to the sum.
At the end, both strings are equal to "lt", and the answer is 100+101+101+101 = 403.
If instead we turned both strings into "lee" or "eet", we would get answers of 433 or 417, which are higher.

Constraints:
- 1 <= s1.length, s2.length <= 1000
- s1 and s2 consist of lowercase English letters.
"""

def minimum_delete_sum_recursive(s1, s2):
    """
    RECURSIVE APPROACH:
    ==================
    Try all possible deletion sequences with ASCII costs.
    
    Time Complexity: O(2^(m+n)) - exponential possibilities
    Space Complexity: O(m+n) - recursion depth
    """
    def dfs(i, j):
        # Base cases
        if i >= len(s1):
            # Delete all remaining characters from s2
            return sum(ord(c) for c in s2[j:])
        if j >= len(s2):
            # Delete all remaining characters from s1
            return sum(ord(c) for c in s1[i:])
        
        # If characters match, no deletion needed
        if s1[i] == s2[j]:
            return dfs(i + 1, j + 1)
        
        # Try deleting from either string
        delete_from_s1 = ord(s1[i]) + dfs(i + 1, j)
        delete_from_s2 = ord(s2[j]) + dfs(i, j + 1)
        
        return min(delete_from_s1, delete_from_s2)
    
    return dfs(0, 0)


def minimum_delete_sum_memoization(s1, s2):
    """
    MEMOIZATION APPROACH:
    ====================
    Use memoization to avoid recomputing same subproblems.
    
    Time Complexity: O(m*n) - each state computed once
    Space Complexity: O(m*n) - memoization table
    """
    memo = {}
    
    def dfs(i, j):
        if (i, j) in memo:
            return memo[(i, j)]
        
        # Base cases
        if i >= len(s1):
            result = sum(ord(c) for c in s2[j:])
        elif j >= len(s2):
            result = sum(ord(c) for c in s1[i:])
        elif s1[i] == s2[j]:
            result = dfs(i + 1, j + 1)
        else:
            delete_from_s1 = ord(s1[i]) + dfs(i + 1, j)
            delete_from_s2 = ord(s2[j]) + dfs(i, j + 1)
            result = min(delete_from_s1, delete_from_s2)
        
        memo[(i, j)] = result
        return result
    
    return dfs(0, 0)


def minimum_delete_sum_dp(s1, s2):
    """
    DYNAMIC PROGRAMMING APPROACH:
    ============================
    Build solution bottom-up using 2D DP table.
    
    Time Complexity: O(m*n) - process each cell once
    Space Complexity: O(m*n) - DP table
    """
    m, n = len(s1), len(s2)
    
    # dp[i][j] = min ASCII sum to make s1[0:i] and s2[0:j] equal
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Base cases: one string is empty
    for i in range(1, m + 1):
        dp[i][0] = dp[i-1][0] + ord(s1[i-1])  # Delete all from s1
    
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j-1] + ord(s2[j-1])  # Delete all from s2
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                # Characters match, no deletion needed
                dp[i][j] = dp[i-1][j-1]
            else:
                # Try deleting from either string
                delete_from_s1 = dp[i-1][j] + ord(s1[i-1])
                delete_from_s2 = dp[i][j-1] + ord(s2[j-1])
                dp[i][j] = min(delete_from_s1, delete_from_s2)
    
    return dp[m][n]


def minimum_delete_sum_space_optimized(s1, s2):
    """
    SPACE OPTIMIZED APPROACH:
    =========================
    Use only O(min(m,n)) space by processing row by row.
    
    Time Complexity: O(m*n) - process each cell once
    Space Complexity: O(min(m,n)) - single row array
    """
    # Ensure s1 is the shorter string for space optimization
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    
    m, n = len(s1), len(s2)
    
    # Use single array to represent previous row
    prev = [0] * (m + 1)
    
    # Initialize base case for s1
    for i in range(1, m + 1):
        prev[i] = prev[i-1] + ord(s1[i-1])
    
    for j in range(1, n + 1):
        curr = [0] * (m + 1)
        curr[0] = prev[0] + ord(s2[j-1])  # Base case for s2
        
        for i in range(1, m + 1):
            if s1[i-1] == s2[j-1]:
                curr[i] = prev[i-1]  # Characters match
            else:
                delete_from_s1 = prev[i] + ord(s1[i-1])
                delete_from_s2 = curr[i-1] + ord(s2[j-1])
                curr[i] = min(delete_from_s1, delete_from_s2)
        
        prev = curr
    
    return prev[m]


def minimum_delete_sum_lcs_approach(s1, s2):
    """
    LCS APPROACH WITH ASCII OPTIMIZATION:
    ====================================
    Find LCS with maximum ASCII sum, then compute deletions.
    
    Time Complexity: O(m*n) - LCS computation with weights
    Space Complexity: O(m*n) - LCS DP table
    """
    m, n = len(s1), len(s2)
    
    # Find LCS with maximum ASCII sum
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + ord(s1[i-1])
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    max_common_ascii = dp[m][n]
    
    # Total ASCII sum of both strings
    total_s1 = sum(ord(c) for c in s1)
    total_s2 = sum(ord(c) for c in s2)
    
    # Deletions = total - 2 * common (keep common, delete rest)
    return total_s1 + total_s2 - 2 * max_common_ascii


def minimum_delete_sum_with_operations(s1, s2):
    """
    FIND ACTUAL DELETION OPERATIONS:
    ================================
    Return minimum sum and the actual sequence of operations.
    
    Time Complexity: O(m*n) - DP + operation reconstruction
    Space Complexity: O(m*n) - DP table + operation tracking
    """
    m, n = len(s1), len(s2)
    
    # DP table and operation tracking
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    operations = [[None] * (n + 1) for _ in range(m + 1)]
    
    # Base cases
    for i in range(1, m + 1):
        dp[i][0] = dp[i-1][0] + ord(s1[i-1])
        operations[i][0] = f"Delete '{s1[i-1]}' (ASCII {ord(s1[i-1])}) from s1"
    
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j-1] + ord(s2[j-1])
        operations[0][j] = f"Delete '{s2[j-1]}' (ASCII {ord(s2[j-1])}) from s2"
    
    # Fill DP table with operation tracking
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
                operations[i][j] = "Match"
            else:
                delete_from_s1 = dp[i-1][j] + ord(s1[i-1])
                delete_from_s2 = dp[i][j-1] + ord(s2[j-1])
                
                if delete_from_s1 <= delete_from_s2:
                    dp[i][j] = delete_from_s1
                    operations[i][j] = f"Delete '{s1[i-1]}' (ASCII {ord(s1[i-1])}) from s1"
                else:
                    dp[i][j] = delete_from_s2
                    operations[i][j] = f"Delete '{s2[j-1]}' (ASCII {ord(s2[j-1])}) from s2"
    
    # Reconstruct operation sequence
    operation_sequence = []
    i, j = m, n
    
    while i > 0 or j > 0:
        op = operations[i][j]
        if op and op != "Match":
            operation_sequence.append(op)
        
        if i > 0 and j > 0 and s1[i-1] == s2[j-1]:
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or dp[i][j] == dp[i-1][j] + ord(s1[i-1])):
            i -= 1
        else:
            j -= 1
    
    operation_sequence.reverse()
    
    return dp[m][n], operation_sequence


def minimum_delete_sum_analysis(s1, s2):
    """
    DETAILED ANALYSIS:
    =================
    Show step-by-step computation and ASCII deletion analysis.
    """
    m, n = len(s1), len(s2)
    
    print(f"Finding minimum ASCII delete sum:")
    print(f"  s1 = '{s1}' (length {m})")
    print(f"  s2 = '{s2}' (length {n})")
    
    # Show ASCII values
    print(f"\nASCII values:")
    print(f"  s1: {[f'{c}({ord(c)})' for c in s1]}")
    print(f"  s2: {[f'{c}({ord(c)})' for c in s2]}")
    
    # Build DP table with visualization
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    print(f"\nBuilding DP table:")
    print(f"  dp[i][j] = min ASCII sum to make s1[0:i] and s2[0:j] equal")
    
    # Base cases
    print(f"\nBase cases:")
    for i in range(1, m + 1):
        dp[i][0] = dp[i-1][0] + ord(s1[i-1])
        print(f"  dp[{i}][0] = {dp[i-1][0]} + {ord(s1[i-1])} = {dp[i][0]} (delete '{s1[i-1]}')")
    
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j-1] + ord(s2[j-1])
        print(f"  dp[0][{j}] = {dp[0][j-1]} + {ord(s2[j-1])} = {dp[0][j]} (delete '{s2[j-1]}')")
    
    # Fill DP table
    print(f"\nFilling DP table:")
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
                print(f"  dp[{i}][{j}]: '{s1[i-1]}' == '{s2[j-1]}' → dp[{i-1}][{j-1}] = {dp[i][j]}")
            else:
                delete_from_s1 = dp[i-1][j] + ord(s1[i-1])
                delete_from_s2 = dp[i][j-1] + ord(s2[j-1])
                dp[i][j] = min(delete_from_s1, delete_from_s2)
                
                action = f"delete '{s1[i-1]}'" if delete_from_s1 <= delete_from_s2 else f"delete '{s2[j-1]}'"
                print(f"  dp[{i}][{j}]: '{s1[i-1]}' != '{s2[j-1]}' → min({delete_from_s1}, {delete_from_s2}) = {dp[i][j]} ({action})")
    
    print(f"\nFinal DP table:")
    # Print column headers
    print("      ", end="")
    print("    ε", end="")
    for c in s2:
        print(f"    {c}", end="")
    print()
    
    # Print rows
    for i in range(m + 1):
        if i == 0:
            print("   ε: ", end="")
        else:
            print(f"   {s1[i-1]}: ", end="")
        
        for j in range(n + 1):
            print(f"{dp[i][j]:5}", end="")
        print()
    
    result = dp[m][n]
    print(f"\nMinimum ASCII delete sum: {result}")
    
    # Show LCS approach
    lcs_result = minimum_delete_sum_lcs_approach(s1, s2)
    print(f"LCS approach result: {lcs_result}")
    
    # Show actual operations
    min_sum, operations = minimum_delete_sum_with_operations(s1, s2)
    if operations:
        print(f"\nDeletion sequence (total cost: {min_sum}):")
        total_cost = 0
        for i, op in enumerate(operations):
            print(f"  {i+1}. {op}")
            # Extract cost from operation string
            if "ASCII" in op:
                cost = int(op.split("ASCII ")[1].split(")")[0])
                total_cost += cost
        print(f"Total cost verification: {total_cost}")
    
    return result


def minimum_delete_sum_variants():
    """
    ASCII DELETE SUM VARIANTS:
    ==========================
    Test different scenarios and related problems.
    """
    
    def minimum_delete_sum_custom_weights(s1, s2, weight_func):
        """Minimum delete sum with custom character weights"""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Base cases with custom weights
        for i in range(1, m + 1):
            dp[i][0] = dp[i-1][0] + weight_func(s1[i-1])
        
        for j in range(1, n + 1):
            dp[0][j] = dp[0][j-1] + weight_func(s2[j-1])
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    delete_from_s1 = dp[i-1][j] + weight_func(s1[i-1])
                    delete_from_s2 = dp[i][j-1] + weight_func(s2[j-1])
                    dp[i][j] = min(delete_from_s1, delete_from_s2)
        
        return dp[m][n]
    
    def make_palindrome_min_ascii_cost(s):
        """Minimum ASCII cost to make string palindrome by deletions"""
        # Find LCS of string with its reverse
        s_rev = s[::-1]
        
        # Modified LCS to maximize ASCII sum
        m = len(s)
        dp = [[0] * (m + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, m + 1):
                if s[i-1] == s_rev[j-1]:
                    dp[i][j] = dp[i-1][j-1] + ord(s[i-1])
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        max_palindrome_ascii = dp[m][m]
        total_ascii = sum(ord(c) for c in s)
        
        return total_ascii - max_palindrome_ascii
    
    def min_distance_vs_min_ascii_sum(s1, s2):
        """Compare minimum operations vs minimum ASCII sum"""
        # Regular delete operations (count)
        m, n = len(s1), len(s2)
        dp_count = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp_count[i][0] = i
        for j in range(n + 1):
            dp_count[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp_count[i][j] = dp_count[i-1][j-1]
                else:
                    dp_count[i][j] = min(dp_count[i-1][j] + 1, dp_count[i][j-1] + 1)
        
        min_operations = dp_count[m][n]
        min_ascii_sum = minimum_delete_sum_dp(s1, s2)
        
        return min_operations, min_ascii_sum
    
    # Test variants
    test_cases = [
        ("sea", "eat"),
        ("delete", "leet"),
        ("abc", "def"),
        ("hello", "world"),
        ("kitten", "sitting")
    ]
    
    print("ASCII Delete Sum Variants Analysis:")
    print("=" * 60)
    
    for s1, s2 in test_cases:
        print(f"\nStrings: '{s1}' and '{s2}'")
        
        # Standard ASCII sum
        ascii_sum = minimum_delete_sum_dp(s1, s2)
        print(f"  Min ASCII delete sum: {ascii_sum}")
        
        # Compare with operation count
        ops_count, ascii_sum_verify = min_distance_vs_min_ascii_sum(s1, s2)
        print(f"  Min operations count: {ops_count}")
        print(f"  Operations vs ASCII: {ops_count} ops, {ascii_sum_verify} ASCII sum")
        
        # Custom weight function (e.g., vowels cost more)
        def vowel_weight(c):
            return ord(c) * 2 if c in 'aeiou' else ord(c)
        
        custom_sum = minimum_delete_sum_custom_weights(s1, s2, vowel_weight)
        print(f"  Custom weights (vowels 2x): {custom_sum}")
        
        # Uniform weights (equivalent to operation count)
        uniform_sum = minimum_delete_sum_custom_weights(s1, s2, lambda c: 1)
        print(f"  Uniform weights: {uniform_sum}")
    
    # Palindrome example
    print(f"\nPalindrome analysis:")
    test_strings = ["racecar", "abcdef", "aabbcc"]
    for s in test_strings:
        cost = make_palindrome_min_ascii_cost(s)
        print(f"  Make '{s}' palindrome: {cost} ASCII cost")


# Test cases
def test_minimum_delete_sum():
    """Test all implementations with various inputs"""
    test_cases = [
        ("sea", "eat", 231),
        ("delete", "leet", 403),
        ("abc", "def", 594),  # ASCII: a=97, b=98, c=99, d=100, e=101, f=102
        ("abc", "abc", 0),
        ("", "", 0),
        ("", "abc", 294),  # a+b+c = 97+98+99 = 294
        ("abc", "", 294),
        ("a", "b", 195),   # a=97, b=98
        ("aa", "aa", 0),
        ("ab", "ba", 195)  # delete a+b = 97+98 = 195
    ]
    
    print("Testing Minimum ASCII Delete Sum Solutions:")
    print("=" * 70)
    
    for i, (s1, s2, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"s1='{s1}', s2='{s2}'")
        print(f"Expected: {expected}")
        
        # Test approaches (skip expensive ones for long strings)
        if len(s1) + len(s2) <= 10:
            try:
                recursive = minimum_delete_sum_recursive(s1, s2)
                print(f"Recursive:        {recursive:>5} {'✓' if recursive == expected else '✗'}")
            except:
                print(f"Recursive:        Timeout")
        
        memo = minimum_delete_sum_memoization(s1, s2)
        dp_result = minimum_delete_sum_dp(s1, s2)
        space_opt = minimum_delete_sum_space_optimized(s1, s2)
        lcs_approach = minimum_delete_sum_lcs_approach(s1, s2)
        
        print(f"Memoization:      {memo:>5} {'✓' if memo == expected else '✗'}")
        print(f"DP (2D):          {dp_result:>5} {'✓' if dp_result == expected else '✗'}")
        print(f"Space Optimized:  {space_opt:>5} {'✓' if space_opt == expected else '✗'}")
        print(f"LCS Approach:     {lcs_approach:>5} {'✓' if lcs_approach == expected else '✗'}")
        
        # Show operations for small cases
        if len(s1) <= 5 and len(s2) <= 5:
            min_sum, operations = minimum_delete_sum_with_operations(s1, s2)
            print(f"Operations: {min_sum}")
            if operations and len(operations) <= 5:
                for op in operations:
                    print(f"  • {op}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    minimum_delete_sum_analysis("sea", "eat")
    
    # Variants demonstration
    print(f"\n" + "=" * 70)
    minimum_delete_sum_variants()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. WEIGHTED DELETIONS: ASCII values create weighted cost function")
    print("2. GREEDY CHOICE: Always delete character with lower ASCII value")
    print("3. LCS CONNECTION: Maximize ASCII sum of common subsequence")
    print("4. COST OPTIMIZATION: Minimize total deletion cost, not count")
    print("5. CHARACTER VALUES: Different characters have different costs")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Text Processing: Weighted string similarity")
    print("• Data Compression: Cost-aware character removal")
    print("• Natural Language: Character importance weighting")
    print("• Bioinformatics: Sequence alignment with mutation costs")
    print("• Algorithm Design: Weighted optimization problems")


if __name__ == "__main__":
    test_minimum_delete_sum()


"""
MINIMUM ASCII DELETE SUM - WEIGHTED EDIT DISTANCE:
==================================================

This problem extends the delete operation concept with weighted costs:
- Each character has a different deletion cost (ASCII value)
- Must minimize total cost rather than operation count
- Demonstrates how weights change optimal decisions
- Foundation for understanding weighted optimization in DP

KEY INSIGHTS:
============
1. **WEIGHTED OPERATIONS**: Each deletion has different cost (ASCII value)
2. **COST MINIMIZATION**: Optimize total cost, not operation count
3. **GREEDY CHOICES**: When characters differ, delete the cheaper one
4. **LCS WITH WEIGHTS**: Maximize ASCII sum of preserved characters

RECURRENCE RELATION:
===================
```
if s1[i-1] == s2[j-1]:
    dp[i][j] = dp[i-1][j-1]  // No cost for matching characters
else:
    dp[i][j] = min(
        dp[i-1][j] + ord(s1[i-1]),   // Delete from s1
        dp[i][j-1] + ord(s2[j-1])    // Delete from s2
    )
```

WEIGHTED LCS APPROACH:
=====================
Alternative formulation using weighted LCS:
```
min_cost = total_ascii(s1) + total_ascii(s2) - 2 * max_common_ascii(s1, s2)
```

Where max_common_ascii finds LCS with maximum ASCII sum:
```
if s1[i-1] == s2[j-1]:
    lcs_dp[i][j] = lcs_dp[i-1][j-1] + ord(s1[i-1])
else:
    lcs_dp[i][j] = max(lcs_dp[i-1][j], lcs_dp[i][j-1])
```

COMPARISON WITH UNWEIGHTED VERSION:
==================================
| Aspect              | Unweighted (583) | Weighted (712)     |
|---------------------|------------------|--------------------|
| Cost Function       | Count operations | Sum ASCII values   |
| Optimal Decisions   | Either string    | Cheaper character  |
| Greedy Component    | None            | Delete cheaper     |
| LCS Connection      | Count formula    | ASCII sum formula  |

ALGORITHM APPROACHES:
====================

1. **Direct DP**: O(m×n) time, O(m×n) space
   - Extend basic delete DP with ASCII costs
   - Most straightforward weighted approach

2. **Space Optimized**: O(m×n) time, O(min(m,n)) space
   - Same optimization as unweighted version
   - Process shorter string as columns

3. **Weighted LCS**: O(m×n) time, O(m×n) space
   - Find LCS with maximum ASCII sum
   - Apply weighted formula for total cost

4. **Memoization**: O(m×n) time, O(m×n) space
   - Top-down approach with ASCII cost tracking
   - Natural recursive decomposition

GREEDY DECISION MAKING:
======================
Key insight: When characters don't match, always delete the one with lower ASCII value:

```python
if s1[i-1] != s2[j-1]:
    cost1 = dp[i-1][j] + ord(s1[i-1])  # Delete from s1
    cost2 = dp[i][j-1] + ord(s2[j-1])  # Delete from s2
    dp[i][j] = min(cost1, cost2)       # Choose cheaper option
```

This greedy choice is optimal due to the problem structure.

SPACE OPTIMIZATION:
==================
Same technique as unweighted version:
```python
# Ensure shorter string processed as columns
if len(s1) > len(s2):
    s1, s2 = s2, s1

prev = [sum(ord(c) for c in s1[:i]) for i in range(len(s1) + 1)]

for j in range(1, len(s2) + 1):
    curr = [prev[0] + ord(s2[j-1])]  # Base case
    
    for i in range(1, len(s1) + 1):
        if s1[i-1] == s2[j-1]:
            curr.append(prev[i-1])
        else:
            curr.append(min(
                prev[i] + ord(s1[i-1]),      # Delete from s1
                curr[i-1] + ord(s2[j-1])     # Delete from s2
            ))
    
    prev = curr
```

APPLICATIONS:
============
- **Text Processing**: Weighted string similarity metrics
- **Data Compression**: Cost-aware character removal
- **Natural Language Processing**: Character importance weighting
- **Bioinformatics**: Sequence alignment with mutation costs
- **Database Systems**: Fuzzy matching with character priorities

VARIANTS:
========
- **Custom weights**: Use different cost functions
- **Insertion costs**: Add weighted insertion operations
- **Position-dependent costs**: Costs vary by position
- **Multiple strings**: Extend to more than two strings

RELATED PROBLEMS:
================
- **Delete Operation (583)**: Unweighted version
- **Edit Distance (72)**: Full operations with unit costs
- **LCS (1143)**: Core algorithm for weighted approach
- **Weighted Edit Distance**: General weighted operations

COMPLEXITY:
==========
- **Time**: O(m×n) - same as unweighted version
- **Space**: O(m×n) → O(min(m,n)) with optimization
- **Cost calculation**: O(1) per character (ASCII lookup)

The weighted version demonstrates how adding costs to operations
changes the optimal strategy while maintaining the same complexity.
"""
