"""
LeetCode 516: Longest Palindromic Subsequence
Difficulty: Medium
Category: String DP

PROBLEM DESCRIPTION:
===================
Given a string s, find the longest palindromic subsequence's length in s.

A subsequence is a sequence that can be derived from another sequence by deleting some 
or no elements without changing the order of the remaining elements.

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
    Try all possible subsequences and check for palindromes.
    
    Time Complexity: O(2^n) - exponential branching
    Space Complexity: O(n) - recursion depth
    """
    def dfs(i, j):
        # Base cases
        if i > j:
            return 0
        if i == j:
            return 1
        
        # If characters match, include both in palindrome
        if s[i] == s[j]:
            return 2 + dfs(i + 1, j - 1)
        
        # Try skipping either character
        skip_left = dfs(i + 1, j)
        skip_right = dfs(i, j - 1)
        
        return max(skip_left, skip_right)
    
    return dfs(0, len(s) - 1)


def longest_palindromic_subsequence_memoization(s):
    """
    MEMOIZATION APPROACH:
    ====================
    Use memoization to avoid recomputing same subproblems.
    
    Time Complexity: O(n^2) - each state computed once
    Space Complexity: O(n^2) - memoization table
    """
    memo = {}
    
    def dfs(i, j):
        if (i, j) in memo:
            return memo[(i, j)]
        
        # Base cases
        if i > j:
            result = 0
        elif i == j:
            result = 1
        elif s[i] == s[j]:
            result = 2 + dfs(i + 1, j - 1)
        else:
            skip_left = dfs(i + 1, j)
            skip_right = dfs(i, j - 1)
            result = max(skip_left, skip_right)
        
        memo[(i, j)] = result
        return result
    
    return dfs(0, len(s) - 1)


def longest_palindromic_subsequence_dp(s):
    """
    DYNAMIC PROGRAMMING APPROACH:
    ============================
    Build solution bottom-up using 2D DP table.
    
    Time Complexity: O(n^2) - process each cell once
    Space Complexity: O(n^2) - DP table
    """
    n = len(s)
    
    # dp[i][j] = length of longest palindromic subsequence in s[i:j+1]
    dp = [[0] * n for _ in range(n)]
    
    # Base case: single characters are palindromes of length 1
    for i in range(n):
        dp[i][i] = 1
    
    # Fill DP table for increasing length
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            
            if s[i] == s[j]:
                # Characters match, include both in palindrome
                dp[i][j] = 2 + dp[i + 1][j - 1]
            else:
                # Take maximum from either skipping character
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
    
    return dp[0][n - 1]


def longest_palindromic_subsequence_space_optimized(s):
    """
    SPACE OPTIMIZED APPROACH:
    =========================
    Use only O(n) space by processing diagonally.
    
    Time Complexity: O(n^2) - process each cell once
    Space Complexity: O(n) - single row array
    """
    n = len(s)
    
    # Use two arrays: previous and current
    prev = [0] * n
    
    # Initialize for single characters
    for i in range(n):
        prev[i] = 1
    
    # Process for increasing lengths
    for length in range(2, n + 1):
        curr = [0] * n
        
        for i in range(n - length + 1):
            j = i + length - 1
            
            if s[i] == s[j]:
                if length == 2:
                    curr[i] = 2
                else:
                    curr[i] = 2 + prev[i + 1]
            else:
                curr[i] = max(prev[i], curr[i + 1] if i + 1 < n else 0)
        
        prev = curr
    
    return prev[0] if prev else 1


def longest_palindromic_subsequence_lcs_approach(s):
    """
    LCS APPROACH:
    ============
    Find LCS of string with its reverse.
    
    Time Complexity: O(n^2) - LCS computation
    Space Complexity: O(n^2) - LCS DP table
    """
    reversed_s = s[::-1]
    n = len(s)
    
    # Compute LCS of s and reversed_s
    dp = [[0] * (n + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if s[i-1] == reversed_s[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[n][n]


def longest_palindromic_subsequence_with_sequence(s):
    """
    FIND ACTUAL PALINDROMIC SUBSEQUENCE:
    ===================================
    Return length and one actual longest palindromic subsequence.
    
    Time Complexity: O(n^2) - DP + sequence reconstruction
    Space Complexity: O(n^2) - DP table
    """
    n = len(s)
    
    # Build DP table
    dp = [[0] * n for _ in range(n)]
    
    # Base case: single characters
    for i in range(n):
        dp[i][i] = 1
    
    # Fill DP table
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            
            if s[i] == s[j]:
                dp[i][j] = 2 + dp[i + 1][j - 1]
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
    
    # Reconstruct palindromic subsequence
    def reconstruct(i, j):
        if i > j:
            return ""
        if i == j:
            return s[i]
        
        if s[i] == s[j]:
            return s[i] + reconstruct(i + 1, j - 1) + s[j]
        elif dp[i + 1][j] > dp[i][j - 1]:
            return reconstruct(i + 1, j)
        else:
            return reconstruct(i, j - 1)
    
    palindrome = reconstruct(0, n - 1)
    return dp[0][n - 1], palindrome


def longest_palindromic_subsequence_all_sequences(s):
    """
    FIND ALL LONGEST PALINDROMIC SUBSEQUENCES:
    ==========================================
    Find all possible longest palindromic subsequences.
    
    Time Complexity: O(n^2 + k*L) where k is number of sequences and L is length
    Space Complexity: O(n^2 + k*L) - DP table + all sequences
    """
    n = len(s)
    
    # Build DP table
    dp = [[0] * n for _ in range(n)]
    
    for i in range(n):
        dp[i][i] = 1
    
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            
            if s[i] == s[j]:
                dp[i][j] = 2 + dp[i + 1][j - 1]
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
    
    # Find all longest palindromic subsequences
    all_palindromes = []
    
    def backtrack(i, j, current_palindrome):
        if i > j:
            all_palindromes.append(current_palindrome)
            return
        
        if i == j:
            # Insert single character in middle
            mid = len(current_palindrome) // 2
            result = current_palindrome[:mid] + s[i] + current_palindrome[mid:]
            all_palindromes.append(result)
            return
        
        if s[i] == s[j]:
            # Include both characters
            new_palindrome = s[i] + current_palindrome + s[j]
            backtrack(i + 1, j - 1, new_palindrome)
        else:
            # Try both directions if they lead to optimal solutions
            if dp[i + 1][j] == dp[i][j]:
                backtrack(i + 1, j, current_palindrome)
            if dp[i][j - 1] == dp[i][j]:
                backtrack(i, j - 1, current_palindrome)
    
    backtrack(0, n - 1, "")
    
    # Remove duplicates
    unique_palindromes = list(set(all_palindromes))
    
    return dp[0][n - 1], unique_palindromes


def longest_palindromic_subsequence_analysis(s):
    """
    DETAILED ANALYSIS:
    =================
    Show step-by-step computation and palindrome analysis.
    """
    n = len(s)
    
    print(f"Finding longest palindromic subsequence in:")
    print(f"  s = '{s}' (length {n})")
    
    # Build DP table with visualization
    dp = [[0] * n for _ in range(n)]
    
    print(f"\nBuilding DP table:")
    print(f"  dp[i][j] = longest palindromic subsequence length in s[i:j+1]")
    
    # Base case: single characters
    print(f"\nBase case (single characters):")
    for i in range(n):
        dp[i][i] = 1
        print(f"  dp[{i}][{i}] = 1 ('{s[i]}')")
    
    # Fill for increasing lengths
    print(f"\nFilling for increasing substring lengths:")
    for length in range(2, n + 1):
        print(f"\nLength {length}:")
        for i in range(n - length + 1):
            j = i + length - 1
            substring = s[i:j+1]
            
            if s[i] == s[j]:
                dp[i][j] = 2 + dp[i + 1][j - 1]
                print(f"  dp[{i}][{j}] ('{substring}'): '{s[i]}' == '{s[j]}' → 2 + dp[{i+1}][{j-1}] = 2 + {dp[i+1][j-1]} = {dp[i][j]}")
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
                print(f"  dp[{i}][{j}] ('{substring}'): '{s[i]}' != '{s[j]}' → max(dp[{i+1}][{j}], dp[{i}][{j-1}]) = max({dp[i+1][j]}, {dp[i][j-1]}) = {dp[i][j]}")
    
    print(f"\nFinal DP table:")
    # Print table
    print("   ", end="")
    for j in range(n):
        print(f"{j:4}", end="")
    print()
    
    for i in range(n):
        print(f"{i:2}: ", end="")
        for j in range(n):
            if i <= j:
                print(f"{dp[i][j]:4}", end="")
            else:
                print("   -", end="")
        print()
    
    result_length = dp[0][n - 1]
    print(f"\nLongest palindromic subsequence length: {result_length}")
    
    # Show actual palindrome(s)
    if result_length > 0:
        length, palindrome = longest_palindromic_subsequence_with_sequence(s)
        print(f"One longest palindromic subsequence: '{palindrome}'")
        
        # Show all if not too many
        length, all_palindromes = longest_palindromic_subsequence_all_sequences(s)
        if len(all_palindromes) <= 10:
            print(f"All longest palindromic subsequences ({len(all_palindromes)} total):")
            for i, p in enumerate(all_palindromes):
                print(f"  {i+1}: '{p}'")
        else:
            print(f"Total count: {len(all_palindromes)} (showing first 5)")
            for i, p in enumerate(all_palindromes[:5]):
                print(f"  {i+1}: '{p}'")
    
    return result_length


def palindromic_subsequence_variants():
    """
    PALINDROMIC SUBSEQUENCE VARIANTS:
    ================================
    Test different scenarios and related problems.
    """
    
    def min_insertions_to_make_palindrome(s):
        """Minimum insertions to make string palindrome"""
        lps_length = longest_palindromic_subsequence_dp(s)
        return len(s) - lps_length
    
    def min_deletions_to_make_palindrome(s):
        """Minimum deletions to make string palindrome"""
        lps_length = longest_palindromic_subsequence_dp(s)
        return len(s) - lps_length
    
    def palindromic_subsequence_count(s):
        """Count all palindromic subsequences (including non-longest)"""
        n = len(s)
        
        # dp[i][j] = count of palindromic subsequences in s[i:j+1]
        dp = [[0] * n for _ in range(n)]
        
        # Single characters
        for i in range(n):
            dp[i][i] = 1
        
        # Fill for increasing lengths
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                
                if s[i] == s[j]:
                    dp[i][j] = 1 + dp[i + 1][j] + dp[i][j - 1]
                else:
                    dp[i][j] = dp[i + 1][j] + dp[i][j - 1] - dp[i + 1][j - 1]
        
        return dp[0][n - 1]
    
    def longest_palindromic_substring_vs_subsequence(s):
        """Compare palindromic substring vs subsequence"""
        # Longest palindromic substring (different problem)
        def longest_palindromic_substring(s):
            n = len(s)
            max_len = 1
            start = 0
            
            # Check all possible centers
            for i in range(n):
                # Odd length palindromes
                left, right = i, i
                while left >= 0 and right < n and s[left] == s[right]:
                    current_len = right - left + 1
                    if current_len > max_len:
                        max_len = current_len
                        start = left
                    left -= 1
                    right += 1
                
                # Even length palindromes
                left, right = i, i + 1
                while left >= 0 and right < n and s[left] == s[right]:
                    current_len = right - left + 1
                    if current_len > max_len:
                        max_len = current_len
                        start = left
                    left -= 1
                    right += 1
            
            return max_len, s[start:start + max_len]
        
        subseq_len = longest_palindromic_subsequence_dp(s)
        substr_len, substr = longest_palindromic_substring(s)
        
        return subseq_len, substr_len, substr
    
    # Test variants
    test_strings = [
        "bbbab",
        "cbbd", 
        "racecar",
        "abcdef",
        "aabaa",
        "abcdcba"
    ]
    
    print("Palindromic Subsequence Variants Analysis:")
    print("=" * 60)
    
    for s in test_strings:
        print(f"\nString: '{s}'")
        
        lps_len = longest_palindromic_subsequence_dp(s)
        _, lps_seq = longest_palindromic_subsequence_with_sequence(s)
        
        print(f"  Longest palindromic subsequence: '{lps_seq}' (length: {lps_len})")
        print(f"  Min insertions to make palindrome: {min_insertions_to_make_palindrome(s)}")
        print(f"  Min deletions to make palindrome: {min_deletions_to_make_palindrome(s)}")
        
        if len(s) <= 8:  # Only for small strings due to exponential growth
            total_count = palindromic_subsequence_count(s)
            print(f"  Total palindromic subsequences: {total_count}")
        
        subseq_len, substr_len, substr = longest_palindromic_substring_vs_subsequence(s)
        print(f"  Longest palindromic substring: '{substr}' (length: {substr_len})")
        print(f"  Subsequence vs Substring: {subseq_len} vs {substr_len}")


# Test cases
def test_longest_palindromic_subsequence():
    """Test all implementations with various inputs"""
    test_cases = [
        ("bbbab", 4),
        ("cbbd", 2),
        ("a", 1),
        ("abc", 1),
        ("racecar", 7),
        ("abcdcba", 7),
        ("abcdef", 1),
        ("aabaa", 5),
        ("abcba", 5),
        ("abacabad", 5),
        ("agbdba", 5),
        ("cddpd", 3)
    ]
    
    print("Testing Longest Palindromic Subsequence Solutions:")
    print("=" * 70)
    
    for i, (s, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"s = '{s}'")
        print(f"Expected: {expected}")
        
        # Test approaches (skip expensive ones for long strings)
        if len(s) <= 10:
            try:
                recursive = longest_palindromic_subsequence_recursive(s)
                print(f"Recursive:        {recursive:>3} {'✓' if recursive == expected else '✗'}")
            except:
                print(f"Recursive:        Timeout")
        
        memo = longest_palindromic_subsequence_memoization(s)
        dp_result = longest_palindromic_subsequence_dp(s)
        space_opt = longest_palindromic_subsequence_space_optimized(s)
        lcs_approach = longest_palindromic_subsequence_lcs_approach(s)
        
        print(f"Memoization:      {memo:>3} {'✓' if memo == expected else '✗'}")
        print(f"DP (2D):          {dp_result:>3} {'✓' if dp_result == expected else '✗'}")
        print(f"Space Optimized:  {space_opt:>3} {'✓' if space_opt == expected else '✗'}")
        print(f"LCS Approach:     {lcs_approach:>3} {'✓' if lcs_approach == expected else '✗'}")
        
        # Show palindrome for small cases
        if len(s) <= 10:
            length, palindrome = longest_palindromic_subsequence_with_sequence(s)
            print(f"Palindrome: '{palindrome}'")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    longest_palindromic_subsequence_analysis("bbbab")
    
    # Variants demonstration
    print(f"\n" + "=" * 70)
    palindromic_subsequence_variants()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. PALINDROME STRUCTURE: Same reading forward and backward")
    print("2. CHARACTER MATCHING: Include both characters when endpoints match")
    print("3. OPTIMAL SUBSTRUCTURE: Longest palindrome contains optimal sub-palindromes")
    print("4. LCS CONNECTION: LPS(s) = LCS(s, reverse(s))")
    print("5. INTERVAL DP: Process by substring length, not sequential positions")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Text Processing: Finding palindromic patterns")
    print("• Bioinformatics: DNA palindrome detection")
    print("• String Reconstruction: Minimum edits to create palindromes")
    print("• Data Compression: Exploiting palindromic redundancy")
    print("• Cryptography: Palindromic key generation")


if __name__ == "__main__":
    test_longest_palindromic_subsequence()


"""
LONGEST PALINDROMIC SUBSEQUENCE - INTERVAL DP MASTERPIECE:
==========================================================

This problem demonstrates the power of interval dynamic programming:
- Process substrings by increasing length rather than sequential positions
- Natural structure for palindrome problems due to symmetry
- Foundation for many string reconstruction and optimization problems
- Elegant connection to LCS through string reversal

KEY INSIGHTS:
============
1. **INTERVAL DP STRUCTURE**: Process by substring length, not position
2. **PALINDROME SYMMETRY**: Matching endpoints can be included together
3. **OPTIMAL SUBSTRUCTURE**: Optimal palindrome contains optimal sub-palindromes
4. **LCS EQUIVALENCE**: LPS(s) = LCS(s, reverse(s))

RECURRENCE RELATION:
===================
```
if s[i] == s[j]:
    dp[i][j] = 2 + dp[i+1][j-1]  # Include both matching characters
else:
    dp[i][j] = max(dp[i+1][j], dp[i][j-1])  # Skip one character
```

PROCESSING ORDER:
================
Unlike sequential DP, process by substring length:
```
Length 1: All single characters (base case)
Length 2: All pairs of characters  
Length 3: All triplets
...
Length n: The entire string
```

ALTERNATIVE APPROACHES:
======================
1. **Direct Interval DP**: Process substrings by increasing length
2. **LCS Method**: LPS(s) = LCS(s, reverse(s))
3. **Recursive with Memoization**: Natural divide-and-conquer structure
4. **Space Optimized**: Reduce from O(n²) to O(n) space

APPLICATIONS:
============
- **String Reconstruction**: Minimum insertions/deletions for palindromes
- **Bioinformatics**: DNA palindrome detection and analysis
- **Text Analysis**: Finding symmetric patterns in documents
- **Data Compression**: Exploiting palindromic redundancy
- **Cryptography**: Palindromic sequence generation

RELATED PROBLEMS:
================
- **Palindromic Substrings**: Count all palindromic subsequences
- **Minimum Insertions**: Make string palindrome with min insertions
- **Palindrome Partitioning**: Partition string into palindromes
- **Longest Palindromic Substring**: Contiguous version of the problem

COMPLEXITY:
==========
- **Time**: O(n²) - optimal for this problem type
- **Space**: O(n²) standard, O(n) optimized
- **Reconstruction**: O(n) to find actual palindromic subsequence

This problem showcases interval DP at its finest and provides the foundation
for understanding palindrome-related optimization problems.
"""
