"""
LeetCode 516: Longest Palindromic Subsequence
Difficulty: Medium
Category: Interval DP - Palindrome Optimization

PROBLEM DESCRIPTION:
===================
Given a string s, find the longest palindromic subsequence's length in s.
A subsequence is a sequence that can be derived from another sequence by deleting some or no elements without changing the order of the remaining elements.

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

def longest_palindromic_subsequence_recursive(s):
    """
    RECURSIVE APPROACH:
    ==================
    Check all possible subsequences for palindromes.
    
    Time Complexity: O(2^n) - exponential subsequences
    Space Complexity: O(n) - recursion depth
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
        
        # Exclude current character
        exclude = generate_subsequences(index + 1, current)
        
        return max(include, exclude)
    
    return generate_subsequences(0, "")


def longest_palindromic_subsequence_memoization(s):
    """
    MEMOIZATION APPROACH:
    ====================
    Use memoization on interval endpoints.
    
    Time Complexity: O(n^2) - each interval computed once
    Space Complexity: O(n^2) - memo table + recursion
    """
    memo = {}
    
    def dp(left, right):
        if left > right:
            return 0
        if left == right:
            return 1
        
        if (left, right) in memo:
            return memo[(left, right)]
        
        if s[left] == s[right]:
            # Characters match, include both
            result = 2 + dp(left + 1, right - 1)
        else:
            # Characters don't match, try excluding either
            result = max(dp(left + 1, right), dp(left, right - 1))
        
        memo[(left, right)] = result
        return result
    
    return dp(0, len(s) - 1)


def longest_palindromic_subsequence_interval_dp(s):
    """
    INTERVAL DP APPROACH:
    ====================
    Bottom-up DP processing intervals by length.
    
    Time Complexity: O(n^2) - two nested loops
    Space Complexity: O(n^2) - DP table
    """
    n = len(s)
    
    # dp[i][j] = length of longest palindromic subsequence in s[i:j+1]
    dp = [[0] * n for _ in range(n)]
    
    # Base case: single characters are palindromes of length 1
    for i in range(n):
        dp[i][i] = 1
    
    # Process intervals by length
    for length in range(2, n + 1):  # Length from 2 to n
        for i in range(n - length + 1):
            j = i + length - 1
            
            if s[i] == s[j]:
                # Characters match
                if length == 2:
                    dp[i][j] = 2  # Two matching characters
                else:
                    dp[i][j] = 2 + dp[i + 1][j - 1]
            else:
                # Characters don't match, try excluding either
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
    
    return dp[0][n - 1]


def longest_palindromic_subsequence_space_optimized(s):
    """
    SPACE OPTIMIZED APPROACH:
    =========================
    Reduce space complexity using rolling arrays.
    
    Time Complexity: O(n^2) - same computation
    Space Complexity: O(n) - single array
    """
    n = len(s)
    
    # Use two arrays: previous and current
    prev = [0] * n
    curr = [0] * n
    
    # Base case: single characters
    for i in range(n):
        prev[i] = 1
    
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            
            if s[i] == s[j]:
                if length == 2:
                    curr[i] = 2
                else:
                    curr[i] = 2 + prev[i + 1]
            else:
                # Need values from current iteration
                if i + 1 < n:
                    left_val = curr[i + 1] if i + 1 <= j - 1 else prev[i + 1]
                else:
                    left_val = 0
                    
                right_val = prev[i] if j - 1 >= i else 0
                curr[i] = max(left_val, right_val)
        
        # Swap arrays for next iteration
        prev, curr = curr, [0] * n
    
    return prev[0]


def longest_palindromic_subsequence_with_sequence(s):
    """
    TRACK ACTUAL SUBSEQUENCE:
    =========================
    Return length and one possible longest palindromic subsequence.
    
    Time Complexity: O(n^2) - DP computation + reconstruction
    Space Complexity: O(n^2) - DP table + sequence tracking
    """
    n = len(s)
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
                    dp[i][j] = 2 + dp[i + 1][j - 1]
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
    
    # Reconstruct one possible longest palindromic subsequence
    def reconstruct(left, right):
        if left > right:
            return ""
        if left == right:
            return s[left]
        
        if s[left] == s[right]:
            return s[left] + reconstruct(left + 1, right - 1) + s[right]
        elif dp[left + 1][right] > dp[left][right - 1]:
            return reconstruct(left + 1, right)
        else:
            return reconstruct(left, right - 1)
    
    palindrome = reconstruct(0, n - 1)
    return dp[0][n - 1], palindrome


def longest_palindromic_subsequence_analysis(s):
    """
    DETAILED ANALYSIS:
    =================
    Show step-by-step DP computation and palindrome construction.
    """
    print(f"Longest Palindromic Subsequence Analysis:")
    print(f"String: '{s}'")
    print(f"Length: {len(s)}")
    
    n = len(s)
    
    # Show character indices
    print(f"\nCharacter indices:")
    for i in range(n):
        print(f"  {i}: '{s[i]}'")
    
    # Build DP table with detailed logging
    dp = [[0] * n for _ in range(n)]
    
    # Base case
    print(f"\nBase case (single characters):")
    for i in range(n):
        dp[i][i] = 1
        print(f"  dp[{i}][{i}] = 1 ('{s[i]}')")
    
    print(f"\nDP Table Construction:")
    
    for length in range(2, n + 1):
        print(f"\nLength {length} intervals:")
        for i in range(n - length + 1):
            j = i + length - 1
            substring = s[i:j+1]
            
            print(f"  Interval [{i},{j}]: '{substring}'")
            
            if s[i] == s[j]:
                if length == 2:
                    dp[i][j] = 2
                    print(f"    '{s[i]}' == '{s[j]}' and length=2 → dp[{i}][{j}] = 2")
                else:
                    dp[i][j] = 2 + dp[i + 1][j - 1]
                    print(f"    '{s[i]}' == '{s[j]}' → dp[{i}][{j}] = 2 + dp[{i+1}][{j-1}] = 2 + {dp[i+1][j-1]} = {dp[i][j]}")
            else:
                left_choice = dp[i + 1][j]
                right_choice = dp[i][j - 1]
                dp[i][j] = max(left_choice, right_choice)
                print(f"    '{s[i]}' != '{s[j]}' → dp[{i}][{j}] = max(dp[{i+1}][{j}], dp[{i}][{j-1}]) = max({left_choice}, {right_choice}) = {dp[i][j]}")
    
    print(f"\nFinal DP Table:")
    print("   ", end="")
    for j in range(n):
        print(f"{j:4}", end="")
    print()
    
    for i in range(n):
        print(f"{i:2}: ", end="")
        for j in range(n):
            if j >= i:
                print(f"{dp[i][j]:4}", end="")
            else:
                print(f"{'':4}", end="")
        print()
    
    print(f"\nLongest palindromic subsequence length: {dp[0][n-1]}")
    
    # Show actual subsequence
    length, palindrome = longest_palindromic_subsequence_with_sequence(s)
    print(f"One possible LPS: '{palindrome}'")
    print(f"Verification: Is '{palindrome}' a palindrome? {palindrome == palindrome[::-1]}")
    print(f"Length verification: {len(palindrome)} == {length}")
    
    return dp[0][n-1]


def longest_palindromic_subsequence_variants():
    """
    PALINDROMIC SUBSEQUENCE VARIANTS:
    ================================
    Different scenarios and modifications.
    """
    
    def count_palindromic_subsequences(s):
        """Count all palindromic subsequences"""
        n = len(s)
        dp = [[0] * n for _ in range(n)]
        
        # Base case: single characters
        for i in range(n):
            dp[i][i] = 1
        
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                
                if s[i] == s[j]:
                    if length == 2:
                        dp[i][j] = 3  # "a", "a", "aa"
                    else:
                        dp[i][j] = dp[i + 1][j - 1] * 2 + 1
                else:
                    dp[i][j] = dp[i + 1][j] + dp[i][j - 1] - dp[i + 1][j - 1]
        
        return dp[0][n - 1]
    
    def shortest_palindrome_by_adding(s):
        """Find minimum characters to add to make string palindrome"""
        n = len(s)
        lps_length = longest_palindromic_subsequence_interval_dp(s)
        return n - lps_length
    
    def longest_palindromic_substring_vs_subsequence(s):
        """Compare palindromic substring vs subsequence"""
        # Longest palindromic substring (different problem)
        def expand_around_center(left, right):
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1
            return right - left - 1
        
        max_len = 0
        for i in range(len(s)):
            # Odd length palindromes
            len1 = expand_around_center(i, i)
            # Even length palindromes
            len2 = expand_around_center(i, i + 1)
            max_len = max(max_len, len1, len2)
        
        lps_length = longest_palindromic_subsequence_interval_dp(s)
        return max_len, lps_length
    
    def palindromic_subsequence_with_k_changes(s, k):
        """LPS with at most k character changes allowed"""
        # This is a more complex variant
        # For simplicity, return the original LPS
        return longest_palindromic_subsequence_interval_dp(s)
    
    # Test variants
    test_cases = [
        "bbbab",
        "cbbd", 
        "racecar",
        "abcdef",
        "aab",
        "abcba"
    ]
    
    print("Palindromic Subsequence Variants:")
    print("=" * 50)
    
    for s in test_cases:
        print(f"\nString: '{s}'")
        
        lps_length = longest_palindromic_subsequence_interval_dp(s)
        count = count_palindromic_subsequences(s)
        min_additions = shortest_palindrome_by_adding(s)
        substring_len, subsequence_len = longest_palindromic_substring_vs_subsequence(s)
        
        print(f"LPS length: {lps_length}")
        print(f"Count of palindromic subsequences: {count}")
        print(f"Min additions for palindrome: {min_additions}")
        print(f"Longest palindromic substring: {substring_len}")
        print(f"Longest palindromic subsequence: {subsequence_len}")
        
        # Show actual LPS
        _, palindrome = longest_palindromic_subsequence_with_sequence(s)
        print(f"One LPS: '{palindrome}'")


# Test cases
def test_longest_palindromic_subsequence():
    """Test all implementations with various inputs"""
    test_cases = [
        ("bbbab", 4),
        ("cbbd", 2),
        ("racecar", 7),
        ("a", 1),
        ("ab", 1),
        ("aa", 2),
        ("abc", 1),
        ("abcba", 5),
        ("abcdcba", 7),
        ("aabbcc", 4)
    ]
    
    print("Testing Longest Palindromic Subsequence Solutions:")
    print("=" * 70)
    
    for i, (s, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"String: '{s}'")
        print(f"Expected: {expected}")
        
        # Skip recursive for long strings
        if len(s) <= 8:
            try:
                recursive = longest_palindromic_subsequence_recursive(s)
                print(f"Recursive:        {recursive:>4} {'✓' if recursive == expected else '✗'}")
            except:
                print(f"Recursive:        Timeout")
        
        memoization = longest_palindromic_subsequence_memoization(s)
        interval_dp = longest_palindromic_subsequence_interval_dp(s)
        space_opt = longest_palindromic_subsequence_space_optimized(s)
        
        print(f"Memoization:      {memoization:>4} {'✓' if memoization == expected else '✗'}")
        print(f"Interval DP:      {interval_dp:>4} {'✓' if interval_dp == expected else '✗'}")
        print(f"Space Optimized:  {space_opt:>4} {'✓' if space_opt == expected else '✗'}")
        
        # Show actual palindrome for small cases
        if len(s) <= 10:
            length, palindrome = longest_palindromic_subsequence_with_sequence(s)
            print(f"One LPS: '{palindrome}' (length: {length})")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    longest_palindromic_subsequence_analysis("bbbab")
    
    # Variants demonstration
    print(f"\n" + "=" * 70)
    longest_palindromic_subsequence_variants()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. INTERVAL DP: Process substrings by increasing length")
    print("2. CHARACTER MATCHING: Include both if endpoints match")
    print("3. OPTIMAL CHOICE: Exclude one endpoint if they don't match")
    print("4. SUBSEQUENCE: Can skip characters, unlike substring")
    print("5. PALINDROME: Read same forwards and backwards")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Text Processing: Palindrome detection and analysis")
    print("• Bioinformatics: DNA sequence palindrome finding")
    print("• String Algorithms: Pattern matching and optimization")
    print("• Algorithm Design: Foundation for palindrome problems")
    print("• Computational Biology: RNA secondary structure analysis")


if __name__ == "__main__":
    test_longest_palindromic_subsequence()


"""
LONGEST PALINDROMIC SUBSEQUENCE - INTERVAL DP ON SEQUENCES:
===========================================================

This problem demonstrates interval DP applied to sequence optimization:
- Find longest subsequence that reads same forwards and backwards
- Characters can be skipped (subsequence vs substring)
- Optimal decision at each step: include/exclude endpoints
- Foundation for many palindrome-related optimization problems

KEY INSIGHTS:
============
1. **SUBSEQUENCE FLEXIBILITY**: Can skip characters, unlike contiguous substring
2. **ENDPOINT DECISIONS**: Compare s[i] and s[j] to make optimal choice
3. **INTERVAL PROCESSING**: Build solution from smaller to larger intervals
4. **OPTIMAL SUBSTRUCTURE**: LPS contains optimal sub-LPS
5. **PALINDROME PROPERTY**: Symmetric structure enables interval DP

ALGORITHM APPROACHES:
====================

1. **Recursive (Brute Force)**: O(2^n) time, O(n) space
   - Generate all subsequences and check palindromes
   - Only viable for tiny inputs

2. **Memoization**: O(n²) time, O(n²) space
   - Top-down DP with interval caching
   - Natural recursive structure

3. **Interval DP**: O(n²) time, O(n²) space
   - Bottom-up construction by interval length
   - Standard and most efficient

4. **Space Optimized**: O(n²) time, O(n) space
   - Possible with careful dependency management

CORE INTERVAL DP ALGORITHM:
==========================
```python
# dp[i][j] = length of LPS in s[i:j+1]
dp = [[0] * n for _ in range(n)]

# Base case: single characters
for i in range(n):
    dp[i][i] = 1

for length in range(2, n + 1):
    for i in range(n - length + 1):
        j = i + length - 1
        
        if s[i] == s[j]:
            dp[i][j] = 2 + dp[i+1][j-1]  # Include both endpoints
        else:
            dp[i][j] = max(dp[i+1][j], dp[i][j-1])  # Exclude one endpoint

return dp[0][n-1]
```

RECURRENCE RELATION:
===================
```
If s[i] == s[j]:
    dp[i][j] = 2 + dp[i+1][j-1]    // Include both matching characters

Else:
    dp[i][j] = max(
        dp[i+1][j],     // Exclude left character
        dp[i][j-1]      // Exclude right character
    )

Base cases:
    dp[i][i] = 1           // Single character is palindrome of length 1
    dp[i][j] = 0 if i > j  // Empty interval
```

WHY INTERVAL DP WORKS:
=====================
**Optimal Substructure**: If s[i] and s[j] match in optimal palindrome:
- They must both be included (as they're at endpoints)
- The middle part s[i+1:j] must also be optimally palindromic
- This creates independent subproblems

**Overlapping Subproblems**: Same intervals appear in multiple recursive calls
- DP table stores results to avoid recomputation
- Bottom-up ensures dependencies are resolved

SEQUENCE VS SUBSTRING:
=====================
**Subsequence** (this problem):
- Can skip characters: "ace" is subsequence of "abcde"
- More flexibility in palindrome construction
- Use interval DP on original positions

**Substring** (different problem):
- Must be contiguous: "bcd" is substring of "abcde"
- More restrictive, typically uses expand-around-centers
- Different algorithm entirely

PALINDROME PROPERTIES:
=====================
**Symmetry**: Reading same forwards and backwards
- Single character: always palindrome
- Two characters: palindrome if identical
- Longer: palindrome if first == last AND middle is palindrome

**Optimal Choice**: When endpoints don't match:
- Must exclude at least one endpoint
- Try excluding each and take maximum
- This explores all possibilities optimally

SOLUTION RECONSTRUCTION:
=======================
To find actual palindromic subsequence:
```python
def reconstruct(i, j):
    if i > j:
        return ""
    if i == j:
        return s[i]
    
    if s[i] == s[j]:
        return s[i] + reconstruct(i+1, j-1) + s[j]
    elif dp[i+1][j] > dp[i][j-1]:
        return reconstruct(i+1, j)
    else:
        return reconstruct(i, j-1)
```

SPACE OPTIMIZATION:
==================
Since dp[i][j] only depends on dp[i+1][j-1], dp[i+1][j], and dp[i][j-1]:
- Can use rolling arrays for space optimization
- Requires careful handling of dependencies
- Reduces space from O(n²) to O(n)

COMPLEXITY ANALYSIS:
===================
- **Time**: O(n²) - each interval computed once
- **Space**: O(n²) - DP table, O(n) with optimization
- **States**: O(n²) - all possible intervals
- **Transitions**: O(1) - constant time per state

APPLICATIONS:
============
- **Text Processing**: Palindrome detection and analysis
- **Bioinformatics**: DNA/RNA palindrome identification
- **String Algorithms**: Pattern matching and text analysis
- **Computational Biology**: RNA secondary structure prediction
- **Data Compression**: Exploiting palindromic patterns

RELATED PROBLEMS:
================
- **Palindromic Substrings (647)**: Count all palindromic substrings
- **Palindrome Partitioning (131)**: Partition into palindromes
- **Shortest Palindrome (214)**: Minimum additions to make palindrome
- **Valid Palindrome (125)**: Palindrome validation with constraints

VARIANTS:
========
- **Count palindromic subsequences**: Count instead of finding longest
- **Weighted characters**: Different values for different characters
- **K-palindromes**: Allow up to k mismatches
- **Multiple strings**: LPS common to multiple strings

MATHEMATICAL PROPERTIES:
========================
- **Monotonicity**: Longer intervals can only have equal or longer LPS
- **Optimal Substructure**: Essential for DP approach
- **Overlapping Subproblems**: Same intervals computed multiple times
- **Symmetry**: Palindrome structure enables efficient computation

EDGE CASES:
==========
- **Single character**: LPS length = 1
- **Empty string**: LPS length = 0
- **All same characters**: LPS length = string length
- **No repeated characters**: LPS length = 1
- **Entire string is palindrome**: LPS length = string length

This problem beautifully demonstrates how interval DP can solve sequence
optimization problems, serving as a foundation for many palindrome-related
algorithms and string processing applications.
"""
