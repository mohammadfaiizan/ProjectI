"""
LeetCode 516: Longest Palindromic Subsequence
Difficulty: Medium
Category: Longest Subsequence Problems (Palindrome DP)

PROBLEM DESCRIPTION:
===================
Given a string s, find the longest palindromic subsequence's length in s.

A subsequence is a sequence that can be derived from another sequence by deleting some or no elements 
without changing the order of the remaining elements.

Example 1:
Input: s = "bbbab"
Output: 4
Explanation: One possible longest palindromic subsequence is "bbbb".

Example 2:
Input: s = "cbbd"
Output: 2
Explanation: One possible longest palindromic subsequence is "bb".

Constraints:
- 1 <= s.length <= 1000
- s consists only of lowercase English letters.
"""

def longest_palindromic_subsequence_brute_force(s):
    """
    BRUTE FORCE APPROACH:
    ====================
    Generate all subsequences and find longest palindromic one.
    
    Time Complexity: O(2^n * n) - 2^n subsequences, O(n) to check palindrome
    Space Complexity: O(n) - recursion stack
    """
    def is_palindrome(string):
        return string == string[::-1]
    
    def generate_subsequences(index, current):
        if index >= len(s):
            if is_palindrome(current):
                return len(current)
            return 0
        
        # Include current character
        include = generate_subsequences(index + 1, current + s[index])
        
        # Skip current character
        skip = generate_subsequences(index + 1, current)
        
        return max(include, skip)
    
    return generate_subsequences(0, "")


def longest_palindromic_subsequence_lcs(s):
    """
    LCS-BASED APPROACH:
    ==================
    LPS(s) = LCS(s, reverse(s))
    
    Time Complexity: O(n^2) - LCS computation
    Space Complexity: O(n^2) - DP table
    """
    def lcs(text1, text2):
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        return dp[m][n]
    
    return lcs(s, s[::-1])


def longest_palindromic_subsequence_dp_2d(s):
    """
    2D DYNAMIC PROGRAMMING:
    =======================
    dp[i][j] = LPS length in substring s[i:j+1]
    
    Time Complexity: O(n^2) - fill DP table
    Space Complexity: O(n^2) - 2D DP table
    """
    n = len(s)
    if n == 0:
        return 0
    
    # dp[i][j] = length of LPS in s[i:j+1]
    dp = [[0] * n for _ in range(n)]
    
    # Base case: single characters are palindromes of length 1
    for i in range(n):
        dp[i][i] = 1
    
    # Fill for substrings of length 2 to n
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            
            if s[i] == s[j]:
                if length == 2:
                    dp[i][j] = 2
                else:
                    dp[i][j] = dp[i + 1][j - 1] + 2
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
    
    return dp[0][n - 1]


def longest_palindromic_subsequence_memoization(s):
    """
    MEMOIZATION APPROACH:
    ====================
    Top-down DP with memoization.
    
    Time Complexity: O(n^2) - each state computed once
    Space Complexity: O(n^2) - memoization table + recursion stack
    """
    memo = {}
    
    def lps(i, j):
        if i > j:
            return 0
        if i == j:
            return 1
        
        if (i, j) in memo:
            return memo[(i, j)]
        
        if s[i] == s[j]:
            result = lps(i + 1, j - 1) + 2
        else:
            result = max(lps(i + 1, j), lps(i, j - 1))
        
        memo[(i, j)] = result
        return result
    
    return lps(0, len(s) - 1)


def longest_palindromic_subsequence_space_optimized(s):
    """
    SPACE OPTIMIZED DP:
    ==================
    Reduce space complexity using optimized DP.
    
    Time Complexity: O(n^2) - same iterations
    Space Complexity: O(n) - single array
    """
    n = len(s)
    if n == 0:
        return 0
    
    # Use two arrays instead of 2D matrix
    prev = [0] * n
    curr = [0] * n
    
    # Single characters
    for i in range(n):
        prev[i] = 1
    
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            
            if s[i] == s[j]:
                if length == 2:
                    curr[i] = 2
                else:
                    curr[i] = prev[i + 1] + 2
            else:
                curr[i] = max(prev[i], curr[i + 1])
        
        # Swap arrays
        prev, curr = curr, prev
    
    return prev[0]


def longest_palindromic_subsequence_with_string(s):
    """
    FIND ACTUAL LPS STRING:
    =======================
    Return both length and one possible LPS.
    
    Time Complexity: O(n^2) - DP + reconstruction
    Space Complexity: O(n^2) - DP table for reconstruction
    """
    n = len(s)
    if n == 0:
        return 0, ""
    
    dp = [[0] * n for _ in range(n)]
    
    # Base case
    for i in range(n):
        dp[i][i] = 1
    
    # Fill DP table
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            
            if s[i] == s[j]:
                if length == 2:
                    dp[i][j] = 2
                else:
                    dp[i][j] = dp[i + 1][j - 1] + 2
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
    
    # Reconstruct LPS
    def reconstruct(i, j):
        if i > j:
            return ""
        if i == j:
            return s[i]
        
        if s[i] == s[j]:
            return s[i] + reconstruct(i + 1, j - 1) + s[j]
        else:
            if dp[i + 1][j] > dp[i][j - 1]:
                return reconstruct(i + 1, j)
            else:
                return reconstruct(i, j - 1)
    
    lps_string = reconstruct(0, n - 1)
    return dp[0][n - 1], lps_string


def longest_palindromic_subsequence_all_lps(s):
    """
    FIND ALL LPS OF MAXIMUM LENGTH:
    ==============================
    Return all possible LPS strings of maximum length.
    
    Time Complexity: O(n^2 * k) where k is number of LPS
    Space Complexity: O(n^2 * k) - store all LPS
    """
    n = len(s)
    if n == 0:
        return 0, []
    
    dp = [[0] * n for _ in range(n)]
    
    # Base case
    for i in range(n):
        dp[i][i] = 1
    
    # Fill DP table
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            
            if s[i] == s[j]:
                if length == 2:
                    dp[i][j] = 2
                else:
                    dp[i][j] = dp[i + 1][j - 1] + 2
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
    
    # Find all LPS
    def get_all_lps(i, j):
        if i > j:
            return [""]
        if i == j:
            return [s[i]]
        
        if s[i] == s[j]:
            inner_lps = get_all_lps(i + 1, j - 1)
            return [s[i] + inner + s[j] for inner in inner_lps]
        else:
            result = []
            if dp[i + 1][j] == dp[i][j]:
                result.extend(get_all_lps(i + 1, j))
            if dp[i][j - 1] == dp[i][j]:
                result.extend(get_all_lps(i, j - 1))
            
            # Remove duplicates
            seen = set()
            unique_result = []
            for lps in result:
                if lps not in seen:
                    seen.add(lps)
                    unique_result.append(lps)
            
            return unique_result
    
    all_lps = get_all_lps(0, n - 1)
    return dp[0][n - 1], all_lps


def longest_palindromic_subsequence_iterative(s):
    """
    ITERATIVE APPROACH:
    ==================
    Non-recursive implementation for better space usage.
    
    Time Complexity: O(n^2) - iterative loops
    Space Complexity: O(n^2) - DP table
    """
    n = len(s)
    if n == 0:
        return 0
    
    dp = [[0] * n for _ in range(n)]
    
    # Every single character is a palindrome of length 1
    for i in range(n):
        dp[i][i] = 1
    
    # Check for palindromes of length 2
    for i in range(n - 1):
        if s[i] == s[i + 1]:
            dp[i][i + 1] = 2
        else:
            dp[i][i + 1] = 1
    
    # Check for palindromes of length 3 to n
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            
            if s[i] == s[j]:
                dp[i][j] = dp[i + 1][j - 1] + 2
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
    
    return dp[0][n - 1]


def longest_palindromic_subsequence_optimized_lcs(s):
    """
    OPTIMIZED LCS APPROACH:
    ======================
    Space-optimized LCS between s and reverse(s).
    
    Time Complexity: O(n^2) - LCS computation
    Space Complexity: O(n) - space-optimized LCS
    """
    def lcs_space_optimized(text1, text2):
        m, n = len(text1), len(text2)
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(prev[j], curr[j - 1])
            prev, curr = curr, prev
        
        return prev[n]
    
    return lcs_space_optimized(s, s[::-1])


# Test cases
def test_longest_palindromic_subsequence():
    """Test all implementations with various inputs"""
    test_cases = [
        ("bbbab", 4),
        ("cbbd", 2),
        ("a", 1),
        ("aa", 2),
        ("abc", 1),
        ("abcba", 5),
        ("racecar", 7),
        ("abcdcba", 7),
        ("abcdef", 1),
        ("", 0),
        ("aaaa", 4),
        ("abcdefghijklmnopqrstuvwxyz", 1),
        ("raceacar", 7),
        ("malayalam", 9)
    ]
    
    print("Testing Longest Palindromic Subsequence Solutions:")
    print("=" * 70)
    
    for i, (s, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: s = '{s}'")
        print(f"Expected: {expected}")
        
        # Test approaches (skip expensive ones for large inputs)
        if len(s) <= 10:
            brute = longest_palindromic_subsequence_brute_force(s)
            print(f"Brute Force:      {brute:>3} {'✓' if brute == expected else '✗'}")
        
        lcs_based = longest_palindromic_subsequence_lcs(s)
        dp_2d = longest_palindromic_subsequence_dp_2d(s)
        memo = longest_palindromic_subsequence_memoization(s)
        space_opt = longest_palindromic_subsequence_space_optimized(s)
        iterative = longest_palindromic_subsequence_iterative(s)
        opt_lcs = longest_palindromic_subsequence_optimized_lcs(s)
        
        print(f"LCS-based:        {lcs_based:>3} {'✓' if lcs_based == expected else '✗'}")
        print(f"2D DP:            {dp_2d:>3} {'✓' if dp_2d == expected else '✗'}")
        print(f"Memoization:      {memo:>3} {'✓' if memo == expected else '✗'}")
        print(f"Space Optimized:  {space_opt:>3} {'✓' if space_opt == expected else '✗'}")
        print(f"Iterative:        {iterative:>3} {'✓' if iterative == expected else '✗'}")
        print(f"Optimized LCS:    {opt_lcs:>3} {'✓' if opt_lcs == expected else '✗'}")
        
        # Show actual LPS for interesting cases
        if expected > 1 and len(s) <= 12:
            length, lps_str = longest_palindromic_subsequence_with_string(s)
            print(f"LPS String: '{lps_str}'")
            
            if expected <= 5 and len(s) <= 8:
                length, all_lps = longest_palindromic_subsequence_all_lps(s)
                if len(all_lps) <= 10:
                    print(f"All LPS: {all_lps}")
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. LCS REDUCTION: LPS(s) = LCS(s, reverse(s))")
    print("2. INTERVAL DP: dp[i][j] = LPS in substring s[i:j+1]")
    print("3. RECURRENCE: If s[i]==s[j]: dp[i][j] = dp[i+1][j-1] + 2")
    print("               Else: dp[i][j] = max(dp[i+1][j], dp[i][j-1])")
    print("4. BASE CASE: Single characters have LPS length 1")
    
    print("\n" + "=" * 70)
    print("Algorithm Comparison:")
    print("Brute Force:      Generate all subsequences")
    print("LCS-based:        LCS(s, reverse(s))")
    print("2D DP:            Classic interval DP")
    print("Memoization:      Top-down recursive DP")
    print("Space Optimized:  Reduce space to O(n)")
    print("Iterative:        Bottom-up without recursion")
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Brute Force:      Time: O(2^n * n), Space: O(n)")
    print("LCS-based:        Time: O(n²),      Space: O(n²)")
    print("2D DP:            Time: O(n²),      Space: O(n²)")
    print("Memoization:      Time: O(n²),      Space: O(n²)")
    print("Space Optimized:  Time: O(n²),      Space: O(n)")
    print("Optimized LCS:    Time: O(n²),      Space: O(n)")


if __name__ == "__main__":
    test_longest_palindromic_subsequence()


"""
PATTERN RECOGNITION:
==================
This is a classic interval DP problem with palindrome structure:
- Can be solved using LCS reduction: LPS(s) = LCS(s, reverse(s))
- Can be solved directly using interval DP
- Foundation for many palindrome-related problems

KEY INSIGHT - TWO APPROACHES:
============================

**Approach 1: LCS Reduction**
- LPS(s) = LCS(s, reverse(s))
- Why? Common subsequence of s and reverse(s) must be palindromic
- Simple to implement, reuses LCS algorithm

**Approach 2: Interval DP**
- dp[i][j] = LPS length in substring s[i:j+1]
- More direct, shows palindrome structure clearly
- Better for variations and follow-up problems

STATE DEFINITION (Interval DP):
==============================
dp[i][j] = length of longest palindromic subsequence in s[i:j+1]

RECURRENCE RELATION:
===================
```
if s[i] == s[j]:
    dp[i][j] = dp[i+1][j-1] + 2     # Include both characters
else:
    dp[i][j] = max(dp[i+1][j], dp[i][j-1])  # Take best from either side
```

Base cases:
- dp[i][i] = 1 (single character)
- i > j: return 0 (empty substring)

FILLING ORDER:
=============
Fill DP table by increasing substring length:
1. Length 1: dp[i][i] = 1
2. Length 2: dp[i][i+1] = 2 if s[i]==s[i+1], else 1
3. Length 3 to n: use recurrence relation

SPACE OPTIMIZATION:
==================
Can reduce from O(n²) to O(n) space:
- Only need previous row to compute current row
- Use rolling arrays or optimized indexing

ALGORITHM PROGRESSION:
=====================
1. **Brute Force**: O(2^n × n) - generate all subsequences
2. **LCS Reduction**: O(n²) - reuse LCS algorithm
3. **Interval DP**: O(n²) - direct palindrome DP
4. **Memoization**: O(n²) - top-down with caching
5. **Space Optimized**: O(n²) time, O(n) space

RECONSTRUCTION:
==============
To find actual LPS string:
```python
if s[i] == s[j]:
    return s[i] + reconstruct(i+1, j-1) + s[j]
else:
    if dp[i+1][j] > dp[i][j-1]:
        return reconstruct(i+1, j)
    else:
        return reconstruct(i, j-1)
```

APPLICATIONS:
============
1. **Text Analysis**: Find palindromic patterns
2. **DNA Sequencing**: Identify palindromic sequences
3. **String Compression**: Exploit palindromic structure
4. **Cryptography**: Palindrome-based encoding
5. **Data Validation**: Check for symmetric patterns

VARIANTS TO PRACTICE:
====================
- Longest Palindromic Substring (5) - contiguous version
- Palindromic Substrings (647) - count all palindromes
- Valid Palindrome II (680) - with one deletion allowed
- Minimum Insertion Steps (1312) - make string palindromic

EDGE CASES:
==========
1. **Empty string**: LPS length = 0
2. **Single character**: LPS length = 1
3. **All same characters**: LPS length = n
4. **No repeated characters**: LPS length = 1
5. **Already palindrome**: LPS length = n

INTERVIEW TIPS:
==============
1. **Show both approaches**: LCS reduction and interval DP
2. **Explain recurrence**: Why the formula works
3. **Draw DP table**: Visualize for small examples
4. **Space optimization**: Show how to reduce space
5. **Reconstruction**: Explain how to find actual LPS
6. **Edge cases**: Handle empty strings, single characters
7. **Follow-ups**: Related palindrome problems
8. **Complexity**: Justify why O(n²) is necessary
9. **Applications**: Real-world uses of palindromes
10. **Variants**: Connect to other palindrome problems
"""
