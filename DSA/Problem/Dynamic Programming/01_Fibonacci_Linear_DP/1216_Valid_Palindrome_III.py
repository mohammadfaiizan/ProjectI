"""
LeetCode 1216: Valid Palindrome III
Difficulty: Hard
Category: Fibonacci & Linear DP / String DP

PROBLEM DESCRIPTION:
===================
Given a string s and an integer k, return true if s is a k-palindrome.

A string is k-palindrome if it can be transformed into a palindrome by removing at most k characters from it.

Example 1:
Input: s = "abcdeca", k = 2
Output: true
Explanation: Remove 'b' and 'e' to get "acdca" which is a palindrome.

Example 2:
Input: s = "abbababa", k = 1
Output: true

Example 3:
Input: s = "babab", k = 0
Output: true

Constraints:
- 1 <= s.length <= 1000
- s consists of only lowercase English letters.
- 1 <= k <= s.length
"""

def is_k_palindrome_bruteforce(s, k):
    """
    BRUTE FORCE APPROACH:
    ====================
    Try all possible ways to remove k characters and check if result is palindrome.
    
    Time Complexity: O(C(n,k) * n) - combinations × palindrome check
    Space Complexity: O(n) - string manipulation
    """
    from itertools import combinations
    
    def is_palindrome(text):
        return text == text[::-1]
    
    n = len(s)
    
    # Try removing 0 to k characters
    for remove_count in range(k + 1):
        # Try all combinations of positions to remove
        for positions_to_remove in combinations(range(n), remove_count):
            # Build string after removal
            result = ""
            for i in range(n):
                if i not in positions_to_remove:
                    result += s[i]
            
            if is_palindrome(result):
                return True
    
    return False


def is_k_palindrome_dp_lcs(s, k):
    """
    DYNAMIC PROGRAMMING - LONGEST COMMON SUBSEQUENCE:
    ================================================
    Transform to LCS problem: find LCS between s and reverse(s).
    k-palindrome iff len(s) - LCS(s, reverse(s)) <= k
    
    Time Complexity: O(n^2) - LCS computation
    Space Complexity: O(n^2) - DP table
    """
    def longest_common_subsequence(text1, text2):
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = 1 + dp[i - 1][j - 1]
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        return dp[m][n]
    
    n = len(s)
    reverse_s = s[::-1]
    
    # Find LCS between s and its reverse
    lcs_length = longest_common_subsequence(s, reverse_s)
    
    # Characters to remove = total - palindromic subsequence length
    removals_needed = n - lcs_length
    
    return removals_needed <= k


def is_k_palindrome_dp_interval(s, k):
    """
    INTERVAL DP APPROACH:
    ====================
    Use interval DP to find minimum deletions needed to make palindrome.
    
    Time Complexity: O(n^2) - fill DP table
    Space Complexity: O(n^2) - 2D DP table
    """
    n = len(s)
    
    # dp[i][j] = minimum deletions to make s[i:j+1] a palindrome
    dp = [[0] * n for _ in range(n)]
    
    # Fill for all substring lengths
    for length in range(2, n + 1):  # length 1 substrings are already palindromes
        for i in range(n - length + 1):
            j = i + length - 1
            
            if s[i] == s[j]:
                dp[i][j] = dp[i + 1][j - 1]
            else:
                # Remove either s[i] or s[j]
                dp[i][j] = 1 + min(dp[i + 1][j], dp[i][j - 1])
    
    return dp[0][n - 1] <= k


def is_k_palindrome_memoization(s, k):
    """
    MEMOIZATION APPROACH:
    ====================
    Top-down DP with memoization for interval computation.
    
    Time Complexity: O(n^2) - each subproblem calculated once
    Space Complexity: O(n^2) - memoization table + recursion stack
    """
    memo = {}
    
    def min_deletions(i, j):
        if i >= j:
            return 0
        
        if (i, j) in memo:
            return memo[(i, j)]
        
        if s[i] == s[j]:
            result = min_deletions(i + 1, j - 1)
        else:
            # Remove either character at i or j
            result = 1 + min(min_deletions(i + 1, j), min_deletions(i, j - 1))
        
        memo[(i, j)] = result
        return result
    
    return min_deletions(0, len(s) - 1) <= k


def is_k_palindrome_space_optimized(s, k):
    """
    SPACE OPTIMIZED DP:
    ==================
    Optimize space using rolling arrays.
    
    Time Complexity: O(n^2) - same iterations
    Space Complexity: O(n) - only store current and previous rows
    """
    n = len(s)
    
    # Use LCS approach with space optimization
    reverse_s = s[::-1]
    
    # Space-optimized LCS
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if s[i - 1] == reverse_s[j - 1]:
                curr[j] = 1 + prev[j - 1]
            else:
                curr[j] = max(prev[j], curr[j - 1])
        
        prev, curr = curr, prev
    
    lcs_length = prev[n]
    removals_needed = n - lcs_length
    
    return removals_needed <= k


def is_k_palindrome_with_reconstruction(s, k):
    """
    FIND ACTUAL PALINDROME AFTER REMOVALS:
    =====================================
    Return whether k-palindrome is possible and show the result.
    
    Time Complexity: O(n^2) - DP + reconstruction
    Space Complexity: O(n^2) - DP table and reconstruction
    """
    n = len(s)
    
    # Use interval DP with parent tracking
    dp = [[0] * n for _ in range(n)]
    parent = [[None] * n for _ in range(n)]
    
    # Fill DP table with decision tracking
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            
            if s[i] == s[j]:
                dp[i][j] = dp[i + 1][j - 1]
                parent[i][j] = "match"
            else:
                left_cost = 1 + dp[i + 1][j]
                right_cost = 1 + dp[i][j - 1]
                
                if left_cost <= right_cost:
                    dp[i][j] = left_cost
                    parent[i][j] = "remove_left"
                else:
                    dp[i][j] = right_cost
                    parent[i][j] = "remove_right"
    
    min_deletions = dp[0][n - 1]
    is_possible = min_deletions <= k
    
    # Reconstruct palindrome if possible
    palindrome = ""
    if is_possible:
        def reconstruct(i, j):
            if i > j:
                return ""
            if i == j:
                return s[i]
            
            if parent[i][j] == "match":
                return s[i] + reconstruct(i + 1, j - 1) + s[j]
            elif parent[i][j] == "remove_left":
                return reconstruct(i + 1, j)
            else:  # remove_right
                return reconstruct(i, j - 1)
        
        palindrome = reconstruct(0, n - 1)
    
    return is_possible, min_deletions, palindrome


def is_k_palindrome_longest_palindromic_subsequence(s, k):
    """
    LONGEST PALINDROMIC SUBSEQUENCE APPROACH:
    ========================================
    Find longest palindromic subsequence and check if removals <= k.
    
    Time Complexity: O(n^2) - LPS computation
    Space Complexity: O(n^2) - DP table
    """
    def longest_palindromic_subsequence(text):
        n = len(text)
        dp = [[0] * n for _ in range(n)]
        
        # Single characters are palindromes of length 1
        for i in range(n):
            dp[i][i] = 1
        
        # Fill for all substring lengths
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                
                if text[i] == text[j]:
                    dp[i][j] = 2 + dp[i + 1][j - 1]
                else:
                    dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
        
        return dp[0][n - 1]
    
    n = len(s)
    lps_length = longest_palindromic_subsequence(s)
    
    # Characters to remove = total - longest palindromic subsequence
    removals_needed = n - lps_length
    
    return removals_needed <= k


def is_k_palindrome_optimized_check(s, k):
    """
    OPTIMIZED WITH EARLY TERMINATION:
    ================================
    Add optimizations for better average case performance.
    
    Time Complexity: O(n^2) - worst case, often better
    Space Complexity: O(n) - space optimized
    """
    n = len(s)
    
    # Quick check: if k >= n-1, always true (can remove all but 1 char)
    if k >= n - 1:
        return True
    
    # Quick check: if already palindrome
    if s == s[::-1]:
        return True
    
    # Use space-optimized LCS approach
    return is_k_palindrome_space_optimized(s, k)


# Test cases
def test_is_k_palindrome():
    """Test all implementations with various inputs"""
    test_cases = [
        ("abcdeca", 2, True),
        ("abbababa", 1, True),
        ("babab", 0, True),
        ("abc", 2, True),
        ("abcdef", 5, True),
        ("raceacar", 1, True),
        ("abcdefg", 3, False),
        ("a", 0, True),
        ("aa", 0, True),
        ("ab", 1, True),
        ("abcba", 0, True),
        ("abccba", 0, True),
        ("abcddcba", 0, True)
    ]
    
    print("Testing Valid Palindrome III Solutions:")
    print("=" * 70)
    
    for i, (s, k, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: s = '{s}', k = {k}")
        print(f"Expected: {expected}")
        
        # Test approaches (skip brute force for large inputs)
        if len(s) <= 8 and k <= 3:
            brute = is_k_palindrome_bruteforce(s, k)
            print(f"Brute Force:      {brute} {'✓' if brute == expected else '✗'}")
        
        dp_lcs = is_k_palindrome_dp_lcs(s, k)
        dp_interval = is_k_palindrome_dp_interval(s, k)
        memo = is_k_palindrome_memoization(s, k)
        space_opt = is_k_palindrome_space_optimized(s, k)
        lps = is_k_palindrome_longest_palindromic_subsequence(s, k)
        optimized = is_k_palindrome_optimized_check(s, k)
        
        print(f"DP LCS:           {dp_lcs} {'✓' if dp_lcs == expected else '✗'}")
        print(f"DP Interval:      {dp_interval} {'✓' if dp_interval == expected else '✗'}")
        print(f"Memoization:      {memo} {'✓' if memo == expected else '✗'}")
        print(f"Space Optimized:  {space_opt} {'✓' if space_opt == expected else '✗'}")
        print(f"LPS Approach:     {lps} {'✓' if lps == expected else '✗'}")
        print(f"Optimized:        {optimized} {'✓' if optimized == expected else '✗'}")
        
        # Show reconstruction for interesting cases
        if expected and len(s) <= 10:
            is_possible, deletions, palindrome = is_k_palindrome_with_reconstruction(s, k)
            print(f"Min deletions: {deletions}, Result: '{palindrome}'")
    
    print("\n" + "=" * 70)
    print("Algorithm Comparison:")
    print("LCS Approach:     Compare s with reverse(s)")
    print("Interval DP:      Direct palindrome deletion DP")
    print("LPS Approach:     Find longest palindromic subsequence")
    print("All approaches have O(n²) time complexity")
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Brute Force:      Time: O(C(n,k)*n),  Space: O(n)")
    print("DP LCS:           Time: O(n^2),       Space: O(n^2)")
    print("DP Interval:      Time: O(n^2),       Space: O(n^2)")
    print("Memoization:      Time: O(n^2),       Space: O(n^2)")
    print("Space Optimized:  Time: O(n^2),       Space: O(n)")
    print("LPS Approach:     Time: O(n^2),       Space: O(n^2)")


if __name__ == "__main__":
    test_is_k_palindrome()


"""
PATTERN RECOGNITION:
==================
This is a string DP problem with palindrome constraints:
- Check if string can become palindrome by removing ≤ k characters
- Multiple equivalent formulations using different DP approaches
- Core insight: minimum deletions = n - LCS(s, reverse(s))

KEY INSIGHT - MULTIPLE FORMULATIONS:
===================================
Problem can be solved using several equivalent approaches:

1. **LCS with Reverse**: LCS(s, reverse(s)) gives longest palindromic subsequence
2. **Interval DP**: Direct computation of minimum deletions
3. **LPS**: Find longest palindromic subsequence directly

All have same O(n²) complexity but different implementation styles.

TRANSFORMATION TO LCS:
=====================
Key insight: Longest palindromic subsequence = LCS(s, reverse(s))
- Characters that match in both directions form palindromic subsequence
- Minimum deletions = n - LCS_length
- Check if deletions ≤ k

INTERVAL DP FORMULATION:
=======================
dp[i][j] = minimum deletions to make s[i:j+1] palindromic

Recurrence:
- If s[i] == s[j]: dp[i][j] = dp[i+1][j-1]
- Else: dp[i][j] = 1 + min(dp[i+1][j], dp[i][j-1])

Base case: dp[i][i] = 0 (single chars are palindromic)

LONGEST PALINDROMIC SUBSEQUENCE:
===============================
Alternative formulation using LPS:
- Find longest subsequence that reads same forwards/backwards
- Minimum deletions = n - LPS_length

OPTIMIZATION TECHNIQUES:
=======================
1. Space optimization: O(n²) → O(n) for LCS approach
2. Early termination: check trivial cases first
3. Memoization: avoid recalculating subproblems

VARIANTS TO PRACTICE:
====================
- Valid Palindrome (125) - check if palindrome with simple rules
- Valid Palindrome II (680) - delete at most 1 character
- Longest Palindromic Subsequence (516) - find LPS length
- Palindromic Substrings (647) - count palindromic substrings

INTERVIEW TIPS:
==============
1. Identify multiple solution approaches
2. Start with LCS approach (most intuitive)
3. Show interval DP as alternative
4. Explain equivalence between approaches
5. Optimize space when possible
6. Handle edge cases (k ≥ n-1, already palindrome)
7. Discuss time/space trade-offs
8. Mention string reconstruction if needed
"""
