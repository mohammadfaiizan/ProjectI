"""
LeetCode 1143: Longest Common Subsequence
Difficulty: Medium
Category: Longest Subsequence Problems (LCS - Two Sequence DP)

PROBLEM DESCRIPTION:
===================
Given two strings text1 and text2, return the length of their longest common subsequence. 
If there is no common subsequence, return 0.

A subsequence of a string is a new string generated from the original string with some characters 
(can be none) deleted without changing the relative order of the remaining characters.

For example, "ace" is a subsequence of "abcde".

A common subsequence of two strings is a subsequence that is common to both strings.

Example 1:
Input: text1 = "abcde", text2 = "ace"
Output: 3
Explanation: The longest common subsequence is "ace" and its length is 3.

Example 2:
Input: text1 = "abc", text2 = "abc"
Output: 3
Explanation: The longest common subsequence is "abc" and its length is 3.

Example 3:
Input: text1 = "abc", text2 = "def"
Output: 0
Explanation: There is no such common subsequence, so the result is 0.

Constraints:
- 1 <= text1.length, text2.length <= 1000
- text1 and text2 consist of only lowercase English characters.
"""

def longest_common_subsequence_brute_force(text1, text2):
    """
    BRUTE FORCE APPROACH - RECURSION:
    ================================
    Try all possible subsequences of both strings.
    
    Time Complexity: O(2^(m+n)) - exponential subsequences
    Space Complexity: O(min(m,n)) - recursion stack depth
    """
    def lcs_recursive(i, j):
        # Base case: reached end of either string
        if i >= len(text1) or j >= len(text2):
            return 0
        
        # If characters match, include in LCS
        if text1[i] == text2[j]:
            return 1 + lcs_recursive(i + 1, j + 1)
        else:
            # Try both options: advance in text1 or text2
            option1 = lcs_recursive(i + 1, j)
            option2 = lcs_recursive(i, j + 1)
            return max(option1, option2)
    
    return lcs_recursive(0, 0)


def longest_common_subsequence_memoization(text1, text2):
    """
    MEMOIZATION APPROACH:
    ====================
    Cache recursive results to avoid recomputation.
    
    Time Complexity: O(m * n) - each state computed once
    Space Complexity: O(m * n) - memoization table + recursion stack
    """
    memo = {}
    
    def lcs_memo(i, j):
        if i >= len(text1) or j >= len(text2):
            return 0
        
        if (i, j) in memo:
            return memo[(i, j)]
        
        if text1[i] == text2[j]:
            result = 1 + lcs_memo(i + 1, j + 1)
        else:
            result = max(lcs_memo(i + 1, j), lcs_memo(i, j + 1))
        
        memo[(i, j)] = result
        return result
    
    return lcs_memo(0, 0)


def longest_common_subsequence_dp_2d(text1, text2):
    """
    2D DYNAMIC PROGRAMMING:
    =======================
    Build solution bottom-up using 2D table.
    
    Time Complexity: O(m * n) - fill DP table
    Space Complexity: O(m * n) - 2D DP table
    """
    m, n = len(text1), len(text2)
    
    # dp[i][j] = LCS length of text1[0:i] and text2[0:j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n]


def longest_common_subsequence_dp_1d(text1, text2):
    """
    1D DYNAMIC PROGRAMMING (SPACE OPTIMIZED):
    =========================================
    Use only two rows instead of full 2D table.
    
    Time Complexity: O(m * n) - same iterations
    Space Complexity: O(min(m, n)) - two 1D arrays
    """
    # Make text1 the shorter string for space optimization
    if len(text1) > len(text2):
        text1, text2 = text2, text1
    
    m, n = len(text1), len(text2)
    
    # Use two arrays: previous and current row
    prev = [0] * (m + 1)
    curr = [0] * (m + 1)
    
    for j in range(1, n + 1):
        for i in range(1, m + 1):
            if text1[i - 1] == text2[j - 1]:
                curr[i] = prev[i - 1] + 1
            else:
                curr[i] = max(prev[i], curr[i - 1])
        
        # Swap arrays
        prev, curr = curr, prev
    
    return prev[m]


def longest_common_subsequence_dp_optimized(text1, text2):
    """
    FURTHER OPTIMIZED DP:
    ====================
    Use single array with careful index management.
    
    Time Complexity: O(m * n) - same iterations
    Space Complexity: O(min(m, n)) - single array
    """
    # Ensure text2 is the shorter string
    if len(text1) < len(text2):
        text1, text2 = text2, text1
    
    m, n = len(text1), len(text2)
    dp = [0] * (n + 1)
    
    for i in range(1, m + 1):
        prev_diag = 0
        for j in range(1, n + 1):
            temp = dp[j]
            if text1[i - 1] == text2[j - 1]:
                dp[j] = prev_diag + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev_diag = temp
    
    return dp[n]


def longest_common_subsequence_with_string(text1, text2):
    """
    FIND ACTUAL LCS STRING:
    ======================
    Return both length and the actual LCS string.
    
    Time Complexity: O(m * n) - DP + reconstruction
    Space Complexity: O(m * n) - DP table for reconstruction
    """
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Build DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    # Reconstruct LCS string
    lcs = []
    i, j = m, n
    
    while i > 0 and j > 0:
        if text1[i - 1] == text2[j - 1]:
            lcs.append(text1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    
    lcs.reverse()
    return dp[m][n], ''.join(lcs)


def longest_common_subsequence_all_lcs(text1, text2):
    """
    FIND ALL LCS STRINGS:
    ====================
    Return all possible LCS strings of maximum length.
    
    Time Complexity: O(m * n + k) where k is number of LCS
    Space Complexity: O(m * n + k) - DP table + all LCS strings
    """
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Build DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    # Reconstruct all LCS strings
    def get_all_lcs(i, j):
        if i == 0 or j == 0:
            return [""]
        
        if text1[i - 1] == text2[j - 1]:
            # Characters match, must include this character
            prev_lcs = get_all_lcs(i - 1, j - 1)
            return [lcs + text1[i - 1] for lcs in prev_lcs]
        else:
            # Characters don't match, try both directions
            result = []
            if dp[i - 1][j] == dp[i][j]:
                result.extend(get_all_lcs(i - 1, j))
            if dp[i][j - 1] == dp[i][j]:
                result.extend(get_all_lcs(i, j - 1))
            
            # Remove duplicates while preserving order
            seen = set()
            unique_result = []
            for lcs in result:
                if lcs not in seen:
                    seen.add(lcs)
                    unique_result.append(lcs)
            
            return unique_result
    
    all_lcs = get_all_lcs(m, n)
    return dp[m][n], all_lcs


def longest_common_subsequence_iterative_reconstruction(text1, text2):
    """
    ITERATIVE LCS RECONSTRUCTION:
    ============================
    Reconstruct LCS iteratively instead of recursively.
    
    Time Complexity: O(m * n) - DP + iterative reconstruction
    Space Complexity: O(m * n) - DP table
    """
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Build DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    # Iterative reconstruction using stack
    lcs = []
    stack = [(m, n)]
    
    while stack:
        i, j = stack.pop()
        
        if i == 0 or j == 0:
            continue
        
        if text1[i - 1] == text2[j - 1]:
            lcs.append(text1[i - 1])
            stack.append((i - 1, j - 1))
        elif dp[i - 1][j] > dp[i][j - 1]:
            stack.append((i - 1, j))
        else:
            stack.append((i, j - 1))
    
    lcs.reverse()
    return dp[m][n], ''.join(lcs)


def longest_common_subsequence_rolling_hash(text1, text2):
    """
    ROLLING HASH OPTIMIZATION:
    ===========================
    Use rolling hash for string comparison optimization.
    
    Time Complexity: O(m * n) - same as standard DP
    Space Complexity: O(min(m, n)) - optimized space
    """
    # For this problem, character comparison is O(1) anyway,
    # so rolling hash doesn't provide benefit.
    # But showing the technique for educational purposes.
    
    if len(text1) > len(text2):
        text1, text2 = text2, text1
    
    m, n = len(text1), len(text2)
    prev = [0] * (m + 1)
    curr = [0] * (m + 1)
    
    for j in range(1, n + 1):
        for i in range(1, m + 1):
            if text1[i - 1] == text2[j - 1]:  # O(1) comparison
                curr[i] = prev[i - 1] + 1
            else:
                curr[i] = max(prev[i], curr[i - 1])
        
        prev, curr = curr, prev
    
    return prev[m]


# Test cases
def test_longest_common_subsequence():
    """Test all implementations with various inputs"""
    test_cases = [
        ("abcde", "ace", 3),
        ("abc", "abc", 3),
        ("abc", "def", 0),
        ("ABCDGH", "AEDFHR", 3),
        ("AGGTAB", "GXTXAYB", 4),
        ("", "abc", 0),
        ("abc", "", 0),
        ("", "", 0),
        ("a", "a", 1),
        ("a", "b", 0),
        ("abcdef", "fbdeca", 3),
        ("MZJAWXU", "XMJYAUZ", 4)
    ]
    
    print("Testing Longest Common Subsequence Solutions:")
    print("=" * 70)
    
    for i, (text1, text2, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: text1 = '{text1}', text2 = '{text2}'")
        print(f"Expected: {expected}")
        
        # Test approaches (skip expensive ones for large inputs)
        if len(text1) <= 10 and len(text2) <= 10:
            brute = longest_common_subsequence_brute_force(text1, text2)
            print(f"Brute Force:      {brute:>3} {'✓' if brute == expected else '✗'}")
        
        memo = longest_common_subsequence_memoization(text1, text2)
        dp_2d = longest_common_subsequence_dp_2d(text1, text2)
        dp_1d = longest_common_subsequence_dp_1d(text1, text2)
        optimized = longest_common_subsequence_dp_optimized(text1, text2)
        rolling = longest_common_subsequence_rolling_hash(text1, text2)
        
        print(f"Memoization:      {memo:>3} {'✓' if memo == expected else '✗'}")
        print(f"2D DP:            {dp_2d:>3} {'✓' if dp_2d == expected else '✗'}")
        print(f"1D DP:            {dp_1d:>3} {'✓' if dp_1d == expected else '✗'}")
        print(f"Optimized:        {optimized:>3} {'✓' if optimized == expected else '✗'}")
        print(f"Rolling Hash:     {rolling:>3} {'✓' if rolling == expected else '✗'}")
        
        # Show actual LCS for interesting cases
        if expected > 0 and len(text1) <= 10 and len(text2) <= 10:
            length, lcs_str = longest_common_subsequence_with_string(text1, text2)
            print(f"LCS String: '{lcs_str}'")
            
            if expected <= 5:  # Don't show too many for cases with many LCS
                length, all_lcs = longest_common_subsequence_all_lcs(text1, text2)
                if len(all_lcs) <= 10:
                    print(f"All LCS: {all_lcs}")
    
    print("\n" + "=" * 70)
    print("LCS Applications:")
    print("• Diff utilities (git diff, file comparison)")
    print("• Bioinformatics (DNA sequence alignment)")
    print("• Version control systems")
    print("• Plagiarism detection")
    print("• Data synchronization")
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. TWO-SEQUENCE DP: State depends on positions in both strings")
    print("2. RECURRENCE: If chars match, add 1; else take max of two options")
    print("3. SPACE OPTIMIZATION: Can reduce from O(mn) to O(min(m,n))")
    print("4. RECONSTRUCTION: Trace back through DP table to find actual LCS")
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Brute Force:      Time: O(2^(m+n)), Space: O(min(m,n))")
    print("Memoization:      Time: O(m*n),     Space: O(m*n)")
    print("2D DP:            Time: O(m*n),     Space: O(m*n)")
    print("1D DP:            Time: O(m*n),     Space: O(min(m,n))")
    print("Optimized:        Time: O(m*n),     Space: O(min(m,n))")
    print("Rolling Hash:     Time: O(m*n),     Space: O(min(m,n))")


if __name__ == "__main__":
    test_longest_common_subsequence()


"""
PATTERN RECOGNITION:
==================
This is THE classic two-sequence DP problem:
- Foundation for many string matching algorithms
- Used in bioinformatics, version control, and text comparison
- Demonstrates optimal substructure and overlapping subproblems

KEY INSIGHT - PROBLEM STRUCTURE:
===============================
LCS has optimal substructure:
- If chars match: LCS(i,j) = 1 + LCS(i-1,j-1)
- If chars don't match: LCS(i,j) = max(LCS(i-1,j), LCS(i,j-1))

This leads to natural DP formulation.

STATE DEFINITION:
================
dp[i][j] = length of LCS of text1[0...i-1] and text2[0...j-1]

RECURRENCE RELATION:
===================
```
if text1[i-1] == text2[j-1]:
    dp[i][j] = dp[i-1][j-1] + 1
else:
    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
```

Base case: dp[0][j] = dp[i][0] = 0 (empty string)

SPACE OPTIMIZATION:
==================
Key observation: dp[i][j] only depends on:
- dp[i-1][j-1] (diagonal)
- dp[i-1][j] (above)  
- dp[i][j-1] (left)

This allows reduction from O(m×n) to O(min(m,n)) space.

RECONSTRUCTION ALGORITHM:
========================
To find actual LCS string:
1. Build full DP table
2. Trace back from dp[m][n]:
   - If chars match: include char, go diagonal
   - Else: go direction with larger value

ALGORITHM PROGRESSION:
=====================
1. **Brute Force**: O(2^(m+n)) - try all subsequences
2. **Memoization**: O(m×n) - cache recursive results
3. **2D DP**: O(m×n) space - standard tabulation
4. **1D DP**: O(min(m,n)) space - space optimized
5. **Advanced**: Various optimizations for special cases

APPLICATIONS:
============
1. **Version Control**: Git diff, file comparison
2. **Bioinformatics**: DNA/protein sequence alignment
3. **Plagiarism Detection**: Find common text segments
4. **Data Synchronization**: Minimize data transfer
5. **Edit Distance**: Related to LCS via transformations

VARIANTS TO PRACTICE:
====================
- Shortest Common Supersequence (1092) - closely related
- Edit Distance (72) - uses similar DP structure
- Distinct Subsequences (115) - counting variant
- Longest Palindromic Subsequence (516) - single string LCS

INTERVIEW TIPS:
==============
1. **Start with recursive formulation**: Show the recurrence
2. **Draw DP table**: Visualize for small examples
3. **Space optimization**: Show how to reduce space complexity
4. **Reconstruction**: Explain how to find actual LCS
5. **Applications**: Mention real-world uses
6. **Edge cases**: Empty strings, no common characters
7. **Follow-ups**: Multiple LCS, space optimization, variants
8. **Time complexity**: Explain why O(m×n) is necessary
9. **Related problems**: Connect to edit distance, SCS
10. **Optimization**: Discuss early termination possibilities

MATHEMATICAL INSIGHT:
====================
LCS length is related to edit distance:
edit_distance(s1, s2) = len(s1) + len(s2) - 2 × LCS(s1, s2)

This relationship connects LCS to many other string algorithms.
"""
