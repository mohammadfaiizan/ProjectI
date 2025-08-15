"""
LeetCode 1143: Longest Common Subsequence
Difficulty: Medium  
Category: Grid/Matrix DP / String DP

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

def lcs_bruteforce(text1, text2):
    """
    BRUTE FORCE APPROACH - RECURSION:
    ================================
    Try all possible subsequences using recursion.
    At each position, either match characters or skip one.
    
    Time Complexity: O(2^(m+n)) - exponential due to overlapping subproblems
    Space Complexity: O(m+n) - recursion stack depth
    """
    def lcs_helper(i, j):
        # Base case: reached end of either string
        if i >= len(text1) or j >= len(text2):
            return 0
        
        # If characters match, include in LCS
        if text1[i] == text2[j]:
            return 1 + lcs_helper(i + 1, j + 1)
        
        # If characters don't match, try skipping either character
        skip_first = lcs_helper(i + 1, j)
        skip_second = lcs_helper(i, j + 1)
        
        return max(skip_first, skip_second)
    
    return lcs_helper(0, 0)


def lcs_memoization(text1, text2):
    """
    DYNAMIC PROGRAMMING - TOP DOWN (MEMOIZATION):
    ============================================
    Use memoization to avoid recalculating same subproblems.
    
    Time Complexity: O(m * n) - each subproblem calculated once
    Space Complexity: O(m * n) - memoization table + recursion stack
    """
    memo = {}
    
    def lcs_helper(i, j):
        if i >= len(text1) or j >= len(text2):
            return 0
        
        if (i, j) in memo:
            return memo[(i, j)]
        
        if text1[i] == text2[j]:
            result = 1 + lcs_helper(i + 1, j + 1)
        else:
            skip_first = lcs_helper(i + 1, j)
            skip_second = lcs_helper(i, j + 1)
            result = max(skip_first, skip_second)
        
        memo[(i, j)] = result
        return result
    
    return lcs_helper(0, 0)


def lcs_tabulation(text1, text2):
    """
    DYNAMIC PROGRAMMING - BOTTOM UP (TABULATION):
    =============================================
    Build solution from bottom up using 2D DP table.
    dp[i][j] = length of LCS between text1[0:i] and text2[0:j]
    
    Time Complexity: O(m * n) - fill entire DP table
    Space Complexity: O(m * n) - 2D DP table
    """
    m, n = len(text1), len(text2)
    
    # Create DP table with extra row and column for empty strings
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                # Characters match: include in LCS
                dp[i][j] = 1 + dp[i - 1][j - 1]
            else:
                # Characters don't match: take maximum from excluding either
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n]


def lcs_space_optimized(text1, text2):
    """
    SPACE OPTIMIZED DYNAMIC PROGRAMMING:
    ===================================
    Since we only need previous row, use 1D array instead of 2D.
    
    Time Complexity: O(m * n) - same number of operations
    Space Complexity: O(min(m, n)) - use smaller dimension for space
    """
    # Ensure text1 is the longer string for optimization
    if len(text1) < len(text2):
        text1, text2 = text2, text1
    
    m, n = len(text1), len(text2)
    
    # Use two arrays: previous and current row
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                curr[j] = 1 + prev[j - 1]
            else:
                curr[j] = max(prev[j], curr[j - 1])
        
        # Swap arrays
        prev, curr = curr, prev
    
    return prev[n]


def lcs_one_array(text1, text2):
    """
    FURTHER SPACE OPTIMIZED - SINGLE ARRAY:
    ======================================
    Use only one array and update in place with careful ordering.
    
    Time Complexity: O(m * n) - same operations
    Space Complexity: O(min(m, n)) - single array
    """
    if len(text1) < len(text2):
        text1, text2 = text2, text1
    
    m, n = len(text1), len(text2)
    dp = [0] * (n + 1)
    
    for i in range(1, m + 1):
        prev_diag = 0  # dp[i-1][j-1]
        
        for j in range(1, n + 1):
            temp = dp[j]  # Store dp[i-1][j] before updating
            
            if text1[i - 1] == text2[j - 1]:
                dp[j] = 1 + prev_diag
            else:
                dp[j] = max(dp[j], dp[j - 1])
            
            prev_diag = temp
    
    return dp[n]


def lcs_with_sequence(text1, text2):
    """
    LCS WITH SEQUENCE RECONSTRUCTION:
    ================================
    Find LCS length and reconstruct the actual LCS string.
    
    Time Complexity: O(m * n) - DP + sequence reconstruction
    Space Complexity: O(m * n) - DP table for backtracking
    """
    m, n = len(text1), len(text2)
    
    # Build DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = 1 + dp[i - 1][j - 1]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    # Reconstruct LCS sequence
    lcs_length = dp[m][n]
    lcs_sequence = []
    
    i, j = m, n
    while i > 0 and j > 0:
        if text1[i - 1] == text2[j - 1]:
            lcs_sequence.append(text1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    
    lcs_sequence.reverse()
    return lcs_length, ''.join(lcs_sequence)


def lcs_multiple_sequences(text1, text2):
    """
    FIND ALL POSSIBLE LCS SEQUENCES:
    ===============================
    Find all possible longest common subsequences.
    
    Time Complexity: O(m * n * 2^min(m,n)) - exponential in worst case
    Space Complexity: O(m * n * 2^min(m,n)) - store all sequences
    """
    m, n = len(text1), len(text2)
    
    # Build DP table first
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = 1 + dp[i - 1][j - 1]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    # Find all LCS sequences using backtracking
    all_lcs = []
    
    def backtrack(i, j, current_lcs):
        if i == 0 or j == 0:
            all_lcs.append(current_lcs[::-1])  # Reverse to get correct order
            return
        
        if text1[i - 1] == text2[j - 1]:
            current_lcs.append(text1[i - 1])
            backtrack(i - 1, j - 1, current_lcs)
            current_lcs.pop()
        else:
            if dp[i - 1][j] == dp[i][j]:
                backtrack(i - 1, j, current_lcs)
            if dp[i][j - 1] == dp[i][j]:
                backtrack(i, j - 1, current_lcs)
    
    backtrack(m, n, [])
    
    # Remove duplicates and return
    unique_lcs = list(set(''.join(seq) for seq in all_lcs))
    return dp[m][n], unique_lcs


def lcs_iterative_with_path(text1, text2):
    """
    ITERATIVE LCS WITH PATH COMPRESSION:
    ===================================
    Space-efficient version that still allows sequence reconstruction.
    
    Time Complexity: O(m * n) - DP computation
    Space Complexity: O(m * n) - compressed path storage
    """
    m, n = len(text1), len(text2)
    
    # Store only the direction taken at each cell
    # 0: diagonal (match), 1: up (delete from text1), 2: left (delete from text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    directions = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = 1 + dp[i - 1][j - 1]
                directions[i][j] = 0  # Diagonal
            else:
                if dp[i - 1][j] >= dp[i][j - 1]:
                    dp[i][j] = dp[i - 1][j]
                    directions[i][j] = 1  # Up
                else:
                    dp[i][j] = dp[i][j - 1]
                    directions[i][j] = 2  # Left
    
    # Reconstruct using directions
    lcs_chars = []
    i, j = m, n
    
    while i > 0 and j > 0:
        if directions[i][j] == 0:  # Diagonal - match
            lcs_chars.append(text1[i - 1])
            i -= 1
            j -= 1
        elif directions[i][j] == 1:  # Up
            i -= 1
        else:  # Left
            j -= 1
    
    lcs_chars.reverse()
    return dp[m][n], ''.join(lcs_chars)


# Test cases
def test_lcs():
    """Test all implementations with various inputs"""
    test_cases = [
        ("abcde", "ace", 3),
        ("abc", "abc", 3),
        ("abc", "def", 0),
        ("", "", 0),
        ("", "abc", 0),
        ("abc", "", 0),
        ("ABCDGH", "AEDFHR", 3),  # ADH
        ("AGGTAB", "GXTXAYB", 4),  # GTAB
        ("programming", "logarithm", 6),  # rgramm
        ("abcdxyz", "xyzabcd", 4)   # abcd or xyza
    ]
    
    print("Testing Longest Common Subsequence Solutions:")
    print("=" * 70)
    
    for i, (text1, text2, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: '{text1}' vs '{text2}'")
        print(f"Expected: {expected}")
        
        # Test approaches (skip brute force for long strings)
        if len(text1) + len(text2) <= 12:
            brute = lcs_bruteforce(text1, text2)
            print(f"Brute Force:      {brute:>3} {'✓' if brute == expected else '✗'}")
        
        memo = lcs_memoization(text1, text2)
        tab = lcs_tabulation(text1, text2)
        space_opt = lcs_space_optimized(text1, text2)
        one_array = lcs_one_array(text1, text2)
        
        print(f"Memoization:      {memo:>3} {'✓' if memo == expected else '✗'}")
        print(f"Tabulation:       {tab:>3} {'✓' if tab == expected else '✗'}")
        print(f"Space Optimized:  {space_opt:>3} {'✓' if space_opt == expected else '✗'}")
        print(f"One Array:        {one_array:>3} {'✓' if one_array == expected else '✗'}")
        
        # Show actual LCS for non-zero cases
        if expected > 0 and len(text1) <= 10 and len(text2) <= 10:
            length, sequence = lcs_with_sequence(text1, text2)
            print(f"LCS: '{sequence}' (length: {length})")
            
            # Show multiple sequences for small inputs
            if len(text1) <= 6 and len(text2) <= 6:
                length, all_sequences = lcs_multiple_sequences(text1, text2)
                if len(all_sequences) > 1:
                    print(f"All LCS: {all_sequences}")
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Brute Force:      Time: O(2^(m+n)),    Space: O(m+n)")
    print("Memoization:      Time: O(m*n),        Space: O(m*n)")
    print("Tabulation:       Time: O(m*n),        Space: O(m*n)")
    print("Space Optimized:  Time: O(m*n),        Space: O(min(m,n))")
    print("One Array:        Time: O(m*n),        Space: O(min(m,n))")
    print("With Sequence:    Time: O(m*n),        Space: O(m*n)")


if __name__ == "__main__":
    test_lcs()


"""
PATTERN RECOGNITION:
==================
This is a classic 2D string DP problem:
- Compare characters from two strings
- If match: include in LCS and move both pointers
- If no match: try excluding either character and take maximum
- dp[i][j] = LCS length for text1[0:i] and text2[0:j]

KEY INSIGHTS:
============
1. If characters match: dp[i][j] = 1 + dp[i-1][j-1]
2. If different: dp[i][j] = max(dp[i-1][j], dp[i][j-1])
3. Base case: empty string has LCS length 0 with any string
4. This is the foundation for many string DP problems

STATE DEFINITION:
================
dp[i][j] = length of LCS between text1[0:i] and text2[0:j]

RECURRENCE RELATION:
===================
If text1[i-1] == text2[j-1]:
    dp[i][j] = 1 + dp[i-1][j-1]
Else:
    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

Base case: dp[0][j] = dp[i][0] = 0 (empty string)

SPACE OPTIMIZATION:
==================
1. Two arrays: O(min(m,n)) space
2. One array: O(min(m,n)) space with careful updates
3. Always make shorter string the column dimension

SEQUENCE RECONSTRUCTION:
=======================
Backtrack through DP table:
- If characters match: include character, go diagonal
- Else: go in direction of larger value

VARIANTS TO PRACTICE:
====================
- Edit Distance (72) - allow insertions, deletions, replacements
- Shortest Common Supersequence (1092) - find shortest string containing both
- Delete Operation for Two Strings (583) - only delete operations
- Minimum ASCII Delete Sum (712) - weighted deletions

INTERVIEW TIPS:
==============
1. Identify as classic LCS problem
2. Draw small examples to understand recurrence
3. Show progression from recursion to DP to space optimization
4. Discuss sequence reconstruction technique
5. Mention relationship to edit distance
6. Handle edge cases (empty strings)
7. Discuss applications (diff tools, DNA sequence analysis)
"""
