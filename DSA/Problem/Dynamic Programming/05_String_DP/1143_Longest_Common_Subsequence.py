"""
LeetCode 1143: Longest Common Subsequence
Difficulty: Medium
Category: String DP

PROBLEM DESCRIPTION:
===================
Given two strings text1 and text2, return the length of their longest common subsequence. 
If there is no common subsequence, return 0.

A subsequence of a string is a new string generated from the original string with some 
characters (can be none) deleted without changing the relative order of the remaining characters.

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

def longest_common_subsequence_recursive(text1, text2):
    """
    RECURSIVE APPROACH:
    ==================
    Try all possible subsequences recursively.
    
    Time Complexity: O(2^(m+n)) - exponential branching
    Space Complexity: O(m+n) - recursion depth
    """
    def dfs(i, j):
        # Base cases
        if i >= len(text1) or j >= len(text2):
            return 0
        
        # If characters match, include in LCS
        if text1[i] == text2[j]:
            return 1 + dfs(i + 1, j + 1)
        
        # Try skipping character from either string
        skip_text1 = dfs(i + 1, j)
        skip_text2 = dfs(i, j + 1)
        
        return max(skip_text1, skip_text2)
    
    return dfs(0, 0)


def longest_common_subsequence_memoization(text1, text2):
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
        if i >= len(text1) or j >= len(text2):
            result = 0
        elif text1[i] == text2[j]:
            result = 1 + dfs(i + 1, j + 1)
        else:
            skip_text1 = dfs(i + 1, j)
            skip_text2 = dfs(i, j + 1)
            result = max(skip_text1, skip_text2)
        
        memo[(i, j)] = result
        return result
    
    return dfs(0, 0)


def longest_common_subsequence_dp(text1, text2):
    """
    DYNAMIC PROGRAMMING APPROACH:
    ============================
    Build solution bottom-up using 2D DP table.
    
    Time Complexity: O(m*n) - process each cell once
    Space Complexity: O(m*n) - DP table
    """
    m, n = len(text1), len(text2)
    
    # dp[i][j] = length of LCS of text1[0:i] and text2[0:j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                # Characters match, extend LCS
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                # Take maximum from either skipping character
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]


def longest_common_subsequence_space_optimized(text1, text2):
    """
    SPACE OPTIMIZED APPROACH:
    =========================
    Use only O(min(m,n)) space by processing row by row.
    
    Time Complexity: O(m*n) - process each cell once
    Space Complexity: O(min(m,n)) - single row array
    """
    # Ensure text1 is the shorter string for space optimization
    if len(text1) > len(text2):
        text1, text2 = text2, text1
    
    m, n = len(text1), len(text2)
    
    # Use single array to represent previous row
    prev = [0] * (m + 1)
    
    for j in range(1, n + 1):
        curr = [0] * (m + 1)
        
        for i in range(1, m + 1):
            if text1[i-1] == text2[j-1]:
                curr[i] = prev[i-1] + 1
            else:
                curr[i] = max(prev[i], curr[i-1])
        
        prev = curr
    
    return prev[m]


def longest_common_subsequence_with_sequence(text1, text2):
    """
    FIND ACTUAL LCS SEQUENCE:
    =========================
    Return LCS length and the actual longest common subsequence.
    
    Time Complexity: O(m*n) - DP + sequence reconstruction
    Space Complexity: O(m*n) - DP table
    """
    m, n = len(text1), len(text2)
    
    # Build DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    # Reconstruct LCS sequence
    lcs = []
    i, j = m, n
    
    while i > 0 and j > 0:
        if text1[i-1] == text2[j-1]:
            # Character is part of LCS
            lcs.append(text1[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            # Move up (skip character from text1)
            i -= 1
        else:
            # Move left (skip character from text2)
            j -= 1
    
    lcs.reverse()
    return dp[m][n], ''.join(lcs)


def longest_common_subsequence_all_sequences(text1, text2):
    """
    FIND ALL LCS SEQUENCES:
    ======================
    Find all possible longest common subsequences.
    
    Time Complexity: O(m*n + k*L) where k is number of LCS and L is LCS length
    Space Complexity: O(m*n + k*L) - DP table + all sequences
    """
    m, n = len(text1), len(text2)
    
    # Build DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    # Find all LCS sequences using backtracking
    all_lcs = []
    
    def backtrack(i, j, current_lcs):
        if i == 0 or j == 0:
            all_lcs.append(current_lcs[::-1])  # Reverse to get correct order
            return
        
        if text1[i-1] == text2[j-1]:
            # Character is part of LCS
            current_lcs.append(text1[i-1])
            backtrack(i-1, j-1, current_lcs)
            current_lcs.pop()
        else:
            # Explore both directions if they lead to optimal solutions
            if dp[i-1][j] == dp[i][j]:
                backtrack(i-1, j, current_lcs)
            if dp[i][j-1] == dp[i][j]:
                backtrack(i, j-1, current_lcs)
    
    backtrack(m, n, [])
    
    # Remove duplicates and convert to strings
    unique_lcs = list(set(''.join(lcs) for lcs in all_lcs))
    
    return dp[m][n], unique_lcs


def longest_common_subsequence_with_analysis(text1, text2):
    """
    DETAILED ANALYSIS:
    =================
    Show step-by-step computation and LCS analysis.
    """
    m, n = len(text1), len(text2)
    
    print(f"Finding LCS of:")
    print(f"  text1 = '{text1}' (length {m})")
    print(f"  text2 = '{text2}' (length {n})")
    
    # Build DP table with visualization
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    print(f"\nBuilding DP table:")
    print(f"  dp[i][j] = LCS length of text1[0:i] and text2[0:j]")
    
    # Fill and show DP table construction
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
                print(f"  dp[{i}][{j}]: '{text1[i-1]}' == '{text2[j-1]}' → {dp[i-1][j-1]} + 1 = {dp[i][j]}")
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                print(f"  dp[{i}][{j}]: '{text1[i-1]}' != '{text2[j-1]}' → max({dp[i-1][j]}, {dp[i][j-1]}) = {dp[i][j]}")
    
    print(f"\nFinal DP table:")
    # Print column headers
    print("     ", end="")
    print("   ε", end="")
    for c in text2:
        print(f"   {c}", end="")
    print()
    
    # Print rows
    for i in range(m + 1):
        if i == 0:
            print("  ε: ", end="")
        else:
            print(f"  {text1[i-1]}: ", end="")
        
        for j in range(n + 1):
            print(f"{dp[i][j]:4}", end="")
        print()
    
    lcs_length = dp[m][n]
    print(f"\nLCS length: {lcs_length}")
    
    # Show LCS sequence(s)
    if lcs_length > 0:
        length, lcs_seq = longest_common_subsequence_with_sequence(text1, text2)
        print(f"One LCS: '{lcs_seq}'")
        
        # Show all LCS if not too many
        length, all_lcs = longest_common_subsequence_all_sequences(text1, text2)
        if len(all_lcs) <= 10:
            print(f"All LCS ({len(all_lcs)} total): {all_lcs}")
        else:
            print(f"Total LCS count: {len(all_lcs)} (showing first 5)")
            print(f"Sample LCS: {all_lcs[:5]}")
    
    return lcs_length


def lcs_applications():
    """
    LCS APPLICATIONS AND VARIANTS:
    ==============================
    Demonstrate various applications of LCS.
    """
    
    def lcs_based_similarity(s1, s2):
        """Compute similarity based on LCS"""
        lcs_len = longest_common_subsequence_dp(s1, s2)
        max_len = max(len(s1), len(s2))
        return lcs_len / max_len if max_len > 0 else 1.0
    
    def shortest_common_supersequence_length(s1, s2):
        """Length of shortest string containing both as subsequences"""
        lcs_len = longest_common_subsequence_dp(s1, s2)
        return len(s1) + len(s2) - lcs_len
    
    def minimum_deletions_to_make_equal(s1, s2):
        """Minimum deletions to make strings equal"""
        lcs_len = longest_common_subsequence_dp(s1, s2)
        return len(s1) + len(s2) - 2 * lcs_len
    
    def diff_algorithm_preview(s1, s2):
        """Simple diff algorithm using LCS"""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Build DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        # Generate diff operations
        diff_ops = []
        i, j = m, n
        
        while i > 0 or j > 0:
            if i > 0 and j > 0 and s1[i-1] == s2[j-1]:
                # Characters match
                i -= 1
                j -= 1
            elif i > 0 and (j == 0 or dp[i-1][j] >= dp[i][j-1]):
                # Delete from s1
                diff_ops.append(f"- {s1[i-1]}")
                i -= 1
            else:
                # Insert into s1 (or delete from s2 perspective)
                diff_ops.append(f"+ {s2[j-1]}")
                j -= 1
        
        diff_ops.reverse()
        return diff_ops
    
    # Test applications
    test_pairs = [
        ("ABCDGH", "AEDFHR"),
        ("programming", "logarithm"),
        ("kitten", "sitting"),
        ("AGGTAB", "GXTXAYB")
    ]
    
    print("LCS Applications Analysis:")
    print("=" * 50)
    
    for s1, s2 in test_pairs:
        print(f"\nStrings: '{s1}' and '{s2}'")
        
        lcs_len = longest_common_subsequence_dp(s1, s2)
        _, lcs_seq = longest_common_subsequence_with_sequence(s1, s2)
        
        print(f"  LCS length: {lcs_len}")
        print(f"  LCS sequence: '{lcs_seq}'")
        print(f"  Similarity: {lcs_based_similarity(s1, s2):.2f}")
        print(f"  Shortest common supersequence length: {shortest_common_supersequence_length(s1, s2)}")
        print(f"  Min deletions to make equal: {minimum_deletions_to_make_equal(s1, s2)}")
        
        if len(s1) <= 10 and len(s2) <= 10:
            diff_ops = diff_algorithm_preview(s1, s2)
            print(f"  Diff operations: {diff_ops}")


# Test cases
def test_longest_common_subsequence():
    """Test all implementations with various inputs"""
    test_cases = [
        ("abcde", "ace", 3),
        ("abc", "abc", 3),
        ("abc", "def", 0),
        ("ABCDGH", "AEDFHR", 3),
        ("AGGTAB", "GXTXAYB", 4),
        ("", "", 0),
        ("", "abc", 0),
        ("abc", "", 0),
        ("a", "a", 1),
        ("a", "b", 0),
        ("abcd", "acbd", 3),
        ("XMJYAUZ", "MZJAWXU", 4)
    ]
    
    print("Testing Longest Common Subsequence Solutions:")
    print("=" * 70)
    
    for i, (text1, text2, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"text1='{text1}', text2='{text2}'")
        print(f"Expected: {expected}")
        
        # Test approaches (skip expensive ones for long strings)
        if len(text1) + len(text2) <= 12:
            try:
                recursive = longest_common_subsequence_recursive(text1, text2)
                print(f"Recursive:        {recursive:>3} {'✓' if recursive == expected else '✗'}")
            except:
                print(f"Recursive:        Timeout")
        
        memo = longest_common_subsequence_memoization(text1, text2)
        dp_result = longest_common_subsequence_dp(text1, text2)
        space_opt = longest_common_subsequence_space_optimized(text1, text2)
        
        print(f"Memoization:      {memo:>3} {'✓' if memo == expected else '✗'}")
        print(f"DP (2D):          {dp_result:>3} {'✓' if dp_result == expected else '✗'}")
        print(f"Space Optimized:  {space_opt:>3} {'✓' if space_opt == expected else '✗'}")
        
        # Show LCS for small cases
        if len(text1) <= 8 and len(text2) <= 8 and expected > 0:
            length, lcs_seq = longest_common_subsequence_with_sequence(text1, text2)
            print(f"LCS sequence: '{lcs_seq}'")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    longest_common_subsequence_with_analysis("ABCDGH", "AEDFHR")
    
    # Applications demonstration
    print(f"\n" + "=" * 70)
    lcs_applications()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. SUBSEQUENCE vs SUBSTRING: Order preserved, but not necessarily contiguous")
    print("2. CHARACTER MATCHING: Include character in LCS when strings match")
    print("3. OPTIMAL SUBSTRUCTURE: LCS contains optimal sub-LCS")
    print("4. MULTIPLE SOLUTIONS: May have multiple LCS of same length")
    print("5. SPACE OPTIMIZATION: Can reduce to O(min(m,n)) space")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Diff Tools: Computing file differences")
    print("• Bioinformatics: DNA sequence comparison")
    print("• Version Control: Merge conflict resolution")
    print("• Text Analysis: Document similarity")
    print("• Data Mining: Pattern discovery in sequences")


if __name__ == "__main__":
    test_longest_common_subsequence()


"""
LONGEST COMMON SUBSEQUENCE - FOUNDATION OF SEQUENCE COMPARISON:
===============================================================

LCS is one of the most fundamental problems in string dynamic programming:
- Foundation for diff algorithms and version control
- Core technique in bioinformatics for sequence alignment
- Building block for many string comparison algorithms
- Demonstrates optimal substructure in sequence problems

KEY INSIGHTS:
============
1. **SUBSEQUENCE DEFINITION**: Maintains relative order but allows gaps
2. **TWO CHOICES**: When characters don't match, skip from either string
3. **OPTIMAL SUBSTRUCTURE**: LCS of prefixes determines LCS of full strings
4. **MULTIPLE SOLUTIONS**: Can have many LCS of the same optimal length

RECURRENCE RELATION:
===================
```
if text1[i-1] == text2[j-1]:
    dp[i][j] = dp[i-1][j-1] + 1  # Include matching character
else:
    dp[i][j] = max(dp[i-1][j], dp[i][j-1])  # Skip from either string
```

SEQUENCE RECONSTRUCTION:
=======================
Backtrack through DP table to find actual LCS:
- If characters match: include in LCS, move diagonally
- If dp[i-1][j] > dp[i][j-1]: move up (skip from text1)
- Otherwise: move left (skip from text2)

APPLICATIONS:
============
- **Version Control**: Git diff algorithm foundation
- **Bioinformatics**: DNA/protein sequence alignment
- **Text Processing**: Document comparison and similarity
- **Data Mining**: Pattern discovery in sequences
- **Machine Learning**: Feature extraction from sequences

RELATED PROBLEMS:
================
- **Shortest Common Supersequence**: length = len1 + len2 - LCS
- **Minimum Deletions**: deletions = len1 + len2 - 2*LCS
- **Edit Distance**: Uses similar DP structure with operations
- **Palindromic Subsequences**: LCS with string and its reverse

COMPLEXITY:
==========
- **Time**: O(m×n) - optimal for this problem
- **Space**: O(m×n) standard, O(min(m,n)) optimized
- **Reconstruction**: O(m+n) to find actual LCS

This problem teaches the fundamental pattern for sequence comparison problems
and serves as the foundation for understanding more complex string DP algorithms.
"""
