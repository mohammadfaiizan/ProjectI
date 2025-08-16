"""
LeetCode 583: Delete Operation for Two Strings
Difficulty: Medium
Category: String DP

PROBLEM DESCRIPTION:
===================
Given two strings word1 and word2, return the minimum number of steps required to make word1 and word2 the same.

In one step, you can delete exactly one character in either string.

Example 1:
Input: word1 = "sea", word2 = "eat"
Output: 2
Explanation: You need one step to make "sea" to "ea" and another step to make "eat" to "ea".

Example 2:
Input: word1 = "leetcode", word2 = "etco"
Output: 4

Constraints:
- 1 <= word1.length, word2.length <= 500
- word1 and word2 consist of only lowercase English letters.
"""

def min_distance_brute_force(word1, word2):
    """
    BRUTE FORCE APPROACH:
    ====================
    Try all possible deletion sequences.
    
    Time Complexity: O(2^(m+n)) - exponential possibilities
    Space Complexity: O(m+n) - recursion depth
    """
    def dfs(i, j):
        # Base cases
        if i >= len(word1):
            return len(word2) - j  # Delete remaining characters from word2
        if j >= len(word2):
            return len(word1) - i  # Delete remaining characters from word1
        
        # If characters match, no deletion needed
        if word1[i] == word2[j]:
            return dfs(i + 1, j + 1)
        
        # Try deleting from either string
        delete_from_word1 = 1 + dfs(i + 1, j)
        delete_from_word2 = 1 + dfs(i, j + 1)
        
        return min(delete_from_word1, delete_from_word2)
    
    return dfs(0, 0)


def min_distance_memoization(word1, word2):
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
        if i >= len(word1):
            result = len(word2) - j
        elif j >= len(word2):
            result = len(word1) - i
        elif word1[i] == word2[j]:
            result = dfs(i + 1, j + 1)
        else:
            delete_from_word1 = 1 + dfs(i + 1, j)
            delete_from_word2 = 1 + dfs(i, j + 1)
            result = min(delete_from_word1, delete_from_word2)
        
        memo[(i, j)] = result
        return result
    
    return dfs(0, 0)


def min_distance_dp(word1, word2):
    """
    DYNAMIC PROGRAMMING APPROACH:
    ============================
    Build solution bottom-up using 2D DP table.
    
    Time Complexity: O(m*n) - process each cell once
    Space Complexity: O(m*n) - DP table
    """
    m, n = len(word1), len(word2)
    
    # dp[i][j] = min deletions to make word1[0:i] and word2[0:j] the same
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Base cases: one string is empty
    for i in range(m + 1):
        dp[i][0] = i  # Delete all characters from word1
    
    for j in range(n + 1):
        dp[0][j] = j  # Delete all characters from word2
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                # Characters match, no deletion needed
                dp[i][j] = dp[i-1][j-1]
            else:
                # Try deleting from either string
                delete_from_word1 = dp[i-1][j] + 1
                delete_from_word2 = dp[i][j-1] + 1
                dp[i][j] = min(delete_from_word1, delete_from_word2)
    
    return dp[m][n]


def min_distance_space_optimized(word1, word2):
    """
    SPACE OPTIMIZED APPROACH:
    =========================
    Use only O(min(m,n)) space by processing row by row.
    
    Time Complexity: O(m*n) - process each cell once
    Space Complexity: O(min(m,n)) - single row array
    """
    # Ensure word1 is the shorter string for space optimization
    if len(word1) > len(word2):
        word1, word2 = word2, word1
    
    m, n = len(word1), len(word2)
    
    # Use single array to represent previous row
    prev = list(range(m + 1))  # Base case: delete all from word1
    
    for j in range(1, n + 1):
        curr = [j]  # Base case: delete all from word2
        
        for i in range(1, m + 1):
            if word1[i-1] == word2[j-1]:
                curr.append(prev[i-1])  # Characters match
            else:
                delete_from_word1 = prev[i] + 1
                delete_from_word2 = curr[i-1] + 1
                curr.append(min(delete_from_word1, delete_from_word2))
        
        prev = curr
    
    return prev[m]


def min_distance_lcs_approach(word1, word2):
    """
    LCS APPROACH:
    ============
    Use Longest Common Subsequence to find minimum deletions.
    
    Time Complexity: O(m*n) - LCS computation
    Space Complexity: O(m*n) - LCS DP table
    """
    def longest_common_subsequence(s1, s2):
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    lcs_length = longest_common_subsequence(word1, word2)
    
    # Total deletions = (len(word1) - lcs) + (len(word2) - lcs)
    return len(word1) + len(word2) - 2 * lcs_length


def min_distance_with_operations(word1, word2):
    """
    FIND ACTUAL DELETION OPERATIONS:
    ================================
    Return minimum deletions and the actual sequence of operations.
    
    Time Complexity: O(m*n) - DP + operation reconstruction
    Space Complexity: O(m*n) - DP table + operation tracking
    """
    m, n = len(word1), len(word2)
    
    # DP table and operation tracking
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    operations = [[None] * (n + 1) for _ in range(m + 1)]
    
    # Base cases
    for i in range(m + 1):
        dp[i][0] = i
        if i > 0:
            operations[i][0] = f"Delete '{word1[i-1]}' from word1"
    
    for j in range(n + 1):
        dp[0][j] = j
        if j > 0:
            operations[0][j] = f"Delete '{word2[j-1]}' from word2"
    
    # Fill DP table with operation tracking
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
                operations[i][j] = "Match"
            else:
                delete_from_word1 = dp[i-1][j] + 1
                delete_from_word2 = dp[i][j-1] + 1
                
                if delete_from_word1 <= delete_from_word2:
                    dp[i][j] = delete_from_word1
                    operations[i][j] = f"Delete '{word1[i-1]}' from word1"
                else:
                    dp[i][j] = delete_from_word2
                    operations[i][j] = f"Delete '{word2[j-1]}' from word2"
    
    # Reconstruct operation sequence
    operation_sequence = []
    i, j = m, n
    
    while i > 0 or j > 0:
        op = operations[i][j]
        if op and op != "Match":
            operation_sequence.append(op)
        
        if i > 0 and j > 0 and word1[i-1] == word2[j-1]:
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or dp[i][j] == dp[i-1][j] + 1):
            i -= 1
        else:
            j -= 1
    
    operation_sequence.reverse()
    
    return dp[m][n], operation_sequence


def min_distance_analysis(word1, word2):
    """
    DETAILED ANALYSIS:
    =================
    Show step-by-step computation and deletion analysis.
    """
    m, n = len(word1), len(word2)
    
    print(f"Finding minimum deletions to make strings equal:")
    print(f"  word1 = '{word1}' (length {m})")
    print(f"  word2 = '{word2}' (length {n})")
    
    # Build DP table with visualization
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    print(f"\nBuilding DP table:")
    print(f"  dp[i][j] = min deletions to make word1[0:i] and word2[0:j] equal")
    
    # Base cases
    print(f"\nBase cases:")
    for i in range(m + 1):
        dp[i][0] = i
        if i <= 3:  # Show first few
            print(f"  dp[{i}][0] = {i} (delete all from word1[0:{i}])")
    
    for j in range(n + 1):
        dp[0][j] = j
        if j <= 3:  # Show first few
            print(f"  dp[0][{j}] = {j} (delete all from word2[0:{j}])")
    
    # Fill DP table
    print(f"\nFilling DP table:")
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
                print(f"  dp[{i}][{j}]: '{word1[i-1]}' == '{word2[j-1]}' → dp[{i-1}][{j-1}] = {dp[i][j]}")
            else:
                delete_from_word1 = dp[i-1][j] + 1
                delete_from_word2 = dp[i][j-1] + 1
                dp[i][j] = min(delete_from_word1, delete_from_word2)
                
                action = "delete from word1" if delete_from_word1 <= delete_from_word2 else "delete from word2"
                print(f"  dp[{i}][{j}]: '{word1[i-1]}' != '{word2[j-1]}' → min({delete_from_word1}, {delete_from_word2}) = {dp[i][j]} ({action})")
    
    print(f"\nFinal DP table:")
    # Print column headers
    print("     ", end="")
    print("   ε", end="")
    for c in word2:
        print(f"   {c}", end="")
    print()
    
    # Print rows
    for i in range(m + 1):
        if i == 0:
            print("  ε: ", end="")
        else:
            print(f"  {word1[i-1]}: ", end="")
        
        for j in range(n + 1):
            print(f"{dp[i][j]:4}", end="")
        print()
    
    result = dp[m][n]
    print(f"\nMinimum deletions needed: {result}")
    
    # Show LCS approach
    lcs_result = min_distance_lcs_approach(word1, word2)
    print(f"LCS approach result: {lcs_result}")
    
    # Show actual operations
    min_ops, operations = min_distance_with_operations(word1, word2)
    if operations:
        print(f"\nDeletion sequence:")
        for i, op in enumerate(operations):
            print(f"  {i+1}. {op}")
    
    return result


def min_distance_variants():
    """
    DELETE OPERATION VARIANTS:
    =========================
    Test different scenarios and related problems.
    """
    
    def min_distance_only_one_string(word1, word2):
        """Minimum deletions from only one string to make them equal"""
        from collections import Counter
        
        count1 = Counter(word1)
        count2 = Counter(word2)
        
        # Find common characters
        common_chars = count1 & count2
        common_length = sum(common_chars.values())
        
        # Option 1: Delete from word1 to match word2
        delete_from_word1 = len(word1) - common_length
        
        # Option 2: Delete from word2 to match word1
        delete_from_word2 = len(word2) - common_length
        
        return min(delete_from_word1, delete_from_word2)
    
    def is_one_edit_distance(word1, word2):
        """Check if strings are one edit distance apart (insert/delete/replace)"""
        m, n = len(word1), len(word2)
        
        if abs(m - n) > 1:
            return False
        
        if m > n:
            word1, word2 = word2, word1
            m, n = n, m
        
        for i in range(m):
            if word1[i] != word2[i]:
                if m == n:
                    # Replace operation
                    return word1[i+1:] == word2[i+1:]
                else:
                    # Delete operation
                    return word1[i:] == word2[i+1:]
        
        # All characters match, check if lengths differ by 1
        return n - m == 1
    
    def make_strings_equal_with_min_operations(word1, word2):
        """Make strings equal with insert/delete operations"""
        # This is equivalent to finding LCS and then calculating operations
        lcs_len = min_distance_lcs_approach(word1, word2)
        total_ops = len(word1) + len(word2) - 2 * lcs_len
        
        # Breaking down operations
        deletions_word1 = len(word1) - lcs_len
        deletions_word2 = len(word2) - lcs_len
        
        return {
            'total_operations': total_ops,
            'deletions_from_word1': deletions_word1,
            'deletions_from_word2': deletions_word2,
            'lcs_length': lcs_len
        }
    
    # Test variants
    test_cases = [
        ("sea", "eat"),
        ("leetcode", "etco"),
        ("abc", "def"),
        ("abc", "abc"),
        ("", "abc"),
        ("abc", ""),
        ("kitten", "sitting"),
        ("intention", "execution")
    ]
    
    print("Delete Operation Variants Analysis:")
    print("=" * 60)
    
    for word1, word2 in test_cases:
        print(f"\nStrings: '{word1}' and '{word2}'")
        
        min_deletions = min_distance_dp(word1, word2)
        print(f"  Min deletions (both strings): {min_deletions}")
        
        min_deletions_one = min_distance_only_one_string(word1, word2)
        print(f"  Min deletions (one string only): {min_deletions_one}")
        
        one_edit = is_one_edit_distance(word1, word2)
        print(f"  One edit distance apart: {one_edit}")
        
        breakdown = make_strings_equal_with_min_operations(word1, word2)
        print(f"  Operation breakdown: {breakdown}")


# Test cases
def test_min_distance():
    """Test all implementations with various inputs"""
    test_cases = [
        ("sea", "eat", 2),
        ("leetcode", "etco", 4),
        ("abc", "def", 6),
        ("abc", "abc", 0),
        ("", "", 0),
        ("", "abc", 3),
        ("abc", "", 3),
        ("a", "b", 2),
        ("a", "a", 0),
        ("ab", "ba", 2),
        ("kitten", "sitting", 5),
        ("sunday", "saturday", 4)
    ]
    
    print("Testing Delete Operation for Two Strings Solutions:")
    print("=" * 70)
    
    for i, (word1, word2, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"word1='{word1}', word2='{word2}'")
        print(f"Expected: {expected}")
        
        # Test approaches (skip expensive ones for long strings)
        if len(word1) + len(word2) <= 12:
            try:
                brute_force = min_distance_brute_force(word1, word2)
                print(f"Brute Force:      {brute_force:>3} {'✓' if brute_force == expected else '✗'}")
            except:
                print(f"Brute Force:      Timeout")
        
        memo = min_distance_memoization(word1, word2)
        dp_result = min_distance_dp(word1, word2)
        space_opt = min_distance_space_optimized(word1, word2)
        lcs_approach = min_distance_lcs_approach(word1, word2)
        
        print(f"Memoization:      {memo:>3} {'✓' if memo == expected else '✗'}")
        print(f"DP (2D):          {dp_result:>3} {'✓' if dp_result == expected else '✗'}")
        print(f"Space Optimized:  {space_opt:>3} {'✓' if space_opt == expected else '✗'}")
        print(f"LCS Approach:     {lcs_approach:>3} {'✓' if lcs_approach == expected else '✗'}")
        
        # Show operations for small cases
        if len(word1) <= 6 and len(word2) <= 6:
            min_ops, operations = min_distance_with_operations(word1, word2)
            print(f"Operations: {min_ops}")
            if operations:
                for op in operations[:3]:  # Show first 3 operations
                    print(f"  • {op}")
                if len(operations) > 3:
                    print(f"  ... and {len(operations) - 3} more")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    min_distance_analysis("sea", "eat")
    
    # Variants demonstration
    print(f"\n" + "=" * 70)
    min_distance_variants()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. DELETE-ONLY OPERATIONS: Only deletion allowed, no insert/replace")
    print("2. TWO DELETION CHOICES: Can delete from either string at each step")
    print("3. LCS CONNECTION: Answer = len(word1) + len(word2) - 2*LCS")
    print("4. OPTIMAL SUBSTRUCTURE: Optimal solution contains optimal subsolutions")
    print("5. SPACE OPTIMIZATION: Can reduce to O(min(m,n)) space")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Text Processing: Making documents similar by removal")
    print("• Data Cleaning: Removing differences between datasets")
    print("• Version Control: Computing minimal deletions for diffs")
    print("• String Algorithms: Foundation for edit distance variants")
    print("• Bioinformatics: Sequence alignment with deletions only")


if __name__ == "__main__":
    test_min_distance()


"""
DELETE OPERATION FOR TWO STRINGS - CONSTRAINED EDIT DISTANCE:
=============================================================

This problem is a specialized version of edit distance with only deletion operations:
- Simpler than full edit distance (no insert/replace)
- Elegant connection to Longest Common Subsequence
- Foundation for understanding constrained string transformation
- Demonstrates how restrictions can simplify problems

KEY INSIGHTS:
============
1. **DELETE-ONLY CONSTRAINT**: Only deletion operations allowed
2. **TWO DELETION SOURCES**: Can delete from either string
3. **LCS CONNECTION**: Optimal result preserves LCS characters
4. **SYMMETRIC PROBLEM**: Deleting from word1 vs word2 are equivalent choices

RECURRENCE RELATION:
===================
```
if word1[i-1] == word2[j-1]:
    dp[i][j] = dp[i-1][j-1]  // No deletion needed
else:
    dp[i][j] = min(
        dp[i-1][j] + 1,      // Delete from word1
        dp[i][j-1] + 1       // Delete from word2
    )
```

BASE CASES:
==========
- dp[i][0] = i (delete all characters from word1[0:i])
- dp[0][j] = j (delete all characters from word2[0:j])

LCS APPROACH:
============
Elegant mathematical insight:
```
min_deletions = len(word1) + len(word2) - 2 * LCS(word1, word2)
```

**Reasoning**:
- Keep all LCS characters (they match in both strings)
- Delete remaining characters from both strings
- Total deletions = (len1 - LCS) + (len2 - LCS)

ALGORITHM APPROACHES:
====================

1. **Direct DP**: O(m×n) time, O(m×n) space
   - Build table considering deletion from either string
   - Most straightforward approach

2. **Space Optimized**: O(m×n) time, O(min(m,n)) space
   - Use rolling arrays to reduce space complexity
   - Same time complexity, better space usage

3. **LCS Method**: O(m×n) time, O(m×n) space
   - First compute LCS length
   - Apply formula: len1 + len2 - 2*LCS
   - Mathematically elegant approach

4. **Memoization**: O(m×n) time, O(m×n) space
   - Top-down recursive approach with caching
   - Natural problem decomposition

SPACE OPTIMIZATION:
==================
Can reduce from O(m×n) to O(min(m,n)):
```python
# Ensure shorter string is processed as columns
if len(word1) > len(word2):
    word1, word2 = word2, word1

prev = list(range(len(word1) + 1))  # Base case

for j in range(1, len(word2) + 1):
    curr = [j]  # Base case for current row
    for i in range(1, len(word1) + 1):
        if word1[i-1] == word2[j-1]:
            curr.append(prev[i-1])
        else:
            curr.append(min(prev[i] + 1, curr[i-1] + 1))
    prev = curr
```

COMPARISON WITH EDIT DISTANCE:
=============================
| Aspect           | Delete Only      | Full Edit Distance |
|------------------|------------------|--------------------|
| Operations       | Delete only      | Insert/Delete/Replace |
| Complexity       | O(m×n)          | O(m×n)             |
| Space           | O(min(m,n))      | O(min(m,n))        |
| LCS Connection  | Direct formula   | No direct formula  |
| Implementation  | Simpler         | More complex       |

APPLICATIONS:
============
- **Document Comparison**: Find minimal changes by removal only
- **Data Cleaning**: Remove inconsistencies between datasets
- **Text Processing**: Make texts similar by character removal
- **Version Control**: Compute deletions for diff algorithms
- **Bioinformatics**: Sequence alignment with gap penalties

RELATED PROBLEMS:
================
- **Edit Distance (72)**: Full version with all operations
- **One Edit Distance (161)**: Check if one operation away
- **LCS (1143)**: Core algorithm for this problem
- **Minimum ASCII Delete Sum (712)**: Weighted version

VARIANTS:
========
- **Delete from one string only**: Minimize deletions from single string
- **Weighted deletions**: Different costs for different characters  
- **Multiple strings**: Extend to more than two strings
- **With constraints**: Additional rules on what can be deleted

COMPLEXITY:
==========
- **Time**: O(m×n) - optimal for this problem type
- **Space**: O(m×n) → O(min(m,n)) with optimization
- **LCS Method**: Same complexity but different constant factors

This problem beautifully demonstrates how adding constraints (delete-only)
can lead to more elegant solutions while maintaining the same complexity.
"""
