"""
LeetCode 115: Distinct Subsequences
Difficulty: Hard
Category: String DP

PROBLEM DESCRIPTION:
===================
Given two strings s and t, return the number of distinct subsequences of s which equals t.

A string's subsequence is a new string formed from the original string by deleting some 
(can be none) of the characters without disturbing the relative positions of the remaining 
characters. (i.e., "ACE" is a subsequence of "ABCDE" while "AEC" is not).

The test cases are generated so that the answer fits on a 32-bit signed integer.

Example 1:
Input: s = "rabbbit", t = "rabbit"
Output: 3
Explanation:
As shown below, there are 3 ways you can generate "rabbit" from S.
rabbbit
^^^^ ^^
rabbbit
^^ ^^^^
rabbbit
^^^ ^^^

Example 2:
Input: s = "babgbag", t = "bag"
Output: 5
Explanation:
As shown below, there are 5 ways you can generate "bag" from S.
babgbag
^^ ^
babgbag
^^    ^
babgbag
^    ^^
babgbag
  ^  ^^
babgbag
    ^^^

Constraints:
- 1 <= s.length, t.length <= 1000
- s and t consist of English letters.
"""

def num_distinct_recursive(s, t):
    """
    RECURSIVE APPROACH:
    ==================
    Try all possible ways to form subsequence.
    
    Time Complexity: O(2^n) - exponential branching
    Space Complexity: O(n) - recursion depth
    """
    def dfs(i, j):
        # Base cases
        if j >= len(t):
            return 1  # Successfully formed t
        if i >= len(s):
            return 0  # Ran out of characters in s
        
        # Always have option to skip current character in s
        result = dfs(i + 1, j)
        
        # If characters match, we can also use current character
        if s[i] == t[j]:
            result += dfs(i + 1, j + 1)
        
        return result
    
    return dfs(0, 0)


def num_distinct_memoization(s, t):
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
        if j >= len(t):
            result = 1  # Successfully formed t
        elif i >= len(s):
            result = 0  # Ran out of characters in s
        else:
            # Always have option to skip current character in s
            result = dfs(i + 1, j)
            
            # If characters match, we can also use current character
            if s[i] == t[j]:
                result += dfs(i + 1, j + 1)
        
        memo[(i, j)] = result
        return result
    
    return dfs(0, 0)


def num_distinct_dp(s, t):
    """
    DYNAMIC PROGRAMMING APPROACH:
    ============================
    Build solution bottom-up using 2D DP table.
    
    Time Complexity: O(m*n) - process each cell once
    Space Complexity: O(m*n) - DP table
    """
    m, n = len(s), len(t)
    
    # dp[i][j] = number of ways to form t[0:j] using s[0:i]
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Base case: empty target can be formed in 1 way (by selecting nothing)
    for i in range(m + 1):
        dp[i][0] = 1
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Always have option to skip current character in s
            dp[i][j] = dp[i - 1][j]
            
            # If characters match, add ways using current character
            if s[i - 1] == t[j - 1]:
                dp[i][j] += dp[i - 1][j - 1]
    
    return dp[m][n]


def num_distinct_space_optimized(s, t):
    """
    SPACE OPTIMIZED APPROACH:
    =========================
    Use only O(n) space by processing row by row.
    
    Time Complexity: O(m*n) - process each cell once
    Space Complexity: O(n) - single row array
    """
    m, n = len(s), len(t)
    
    # Use single array to represent previous row
    prev = [0] * (n + 1)
    prev[0] = 1  # Base case: empty target
    
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        curr[0] = 1  # Base case: empty target
        
        for j in range(1, n + 1):
            # Always have option to skip current character in s
            curr[j] = prev[j]
            
            # If characters match, add ways using current character
            if s[i - 1] == t[j - 1]:
                curr[j] += prev[j - 1]
        
        prev = curr
    
    return prev[n]


def num_distinct_space_optimized_1d(s, t):
    """
    ULTIMATE SPACE OPTIMIZATION:
    ============================
    Use only O(n) space with single array processed backwards.
    
    Time Complexity: O(m*n) - process each cell once
    Space Complexity: O(n) - single array
    """
    m, n = len(s), len(t)
    
    # dp[j] = number of ways to form t[0:j] using current prefix of s
    dp = [0] * (n + 1)
    dp[0] = 1  # Base case: empty target
    
    for i in range(1, m + 1):
        # Process backwards to avoid overwriting needed values
        for j in range(n, 0, -1):
            if s[i - 1] == t[j - 1]:
                dp[j] += dp[j - 1]
    
    return dp[n]


def num_distinct_with_paths(s, t):
    """
    FIND ACTUAL SUBSEQUENCE PATHS:
    ==============================
    Return count and show some actual subsequences.
    
    Time Complexity: O(m*n + k*L) where k is count and L is length
    Space Complexity: O(m*n + k*L) - DP table + paths
    """
    m, n = len(s), len(t)
    
    # Build DP table with path tracking
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Base case
    for i in range(m + 1):
        dp[i][0] = 1
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i - 1][j]
            if s[i - 1] == t[j - 1]:
                dp[i][j] += dp[i - 1][j - 1]
    
    # Find actual paths using backtracking
    all_paths = []
    
    def backtrack(i, j, current_indices):
        if j == 0:
            # Successfully formed target, record path
            all_paths.append(current_indices[::-1])  # Reverse to get correct order
            return
        
        if i == 0:
            return  # Can't form target with empty source
        
        # If we skip current character
        if dp[i - 1][j] > 0:
            backtrack(i - 1, j, current_indices)
        
        # If we use current character (and it matches)
        if s[i - 1] == t[j - 1] and dp[i - 1][j - 1] > 0:
            current_indices.append(i - 1)
            backtrack(i - 1, j - 1, current_indices)
            current_indices.pop()
    
    backtrack(m, n, [])
    
    return dp[m][n], all_paths


def num_distinct_analysis(s, t):
    """
    DETAILED ANALYSIS:
    =================
    Show step-by-step computation and subsequence analysis.
    """
    m, n = len(s), len(t)
    
    print(f"Finding distinct subsequences:")
    print(f"  s = '{s}' (length {m})")
    print(f"  t = '{t}' (length {n})")
    
    # Build DP table with visualization
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    print(f"\nBuilding DP table:")
    print(f"  dp[i][j] = number of ways to form t[0:j] using s[0:i]")
    
    # Base case
    print(f"\nBase case (empty target):")
    for i in range(m + 1):
        dp[i][0] = 1
        if i <= 5:  # Show first few for brevity
            print(f"  dp[{i}][0] = 1")
    
    # Fill DP table
    print(f"\nFilling DP table:")
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i - 1][j]
            
            if s[i - 1] == t[j - 1]:
                dp[i][j] += dp[i - 1][j - 1]
                print(f"  dp[{i}][{j}]: s[{i-1}]='{s[i-1]}' == t[{j-1}]='{t[j-1]}' → {dp[i-1][j]} + {dp[i-1][j-1]} = {dp[i][j]}")
            else:
                print(f"  dp[{i}][{j}]: s[{i-1}]='{s[i-1]}' != t[{j-1}]='{t[j-1]}' → {dp[i-1][j]} = {dp[i][j]}")
    
    print(f"\nFinal DP table:")
    # Print column headers
    print("     ", end="")
    print("   ε", end="")
    for c in t:
        print(f"   {c}", end="")
    print()
    
    # Print rows
    for i in range(m + 1):
        if i == 0:
            print("  ε: ", end="")
        else:
            print(f"  {s[i-1]}: ", end="")
        
        for j in range(n + 1):
            print(f"{dp[i][j]:4}", end="")
        print()
    
    result = dp[m][n]
    print(f"\nNumber of distinct subsequences: {result}")
    
    # Show some actual paths if reasonable number
    if result <= 20:
        count, paths = num_distinct_with_paths(s, t)
        print(f"\nAll distinct subsequences ({len(paths)} total):")
        for i, path in enumerate(paths):
            indices_str = ', '.join(map(str, path))
            chars = ''.join(s[idx] for idx in path)
            print(f"  {i+1}: indices [{indices_str}] → '{chars}'")
    elif result <= 100:
        count, paths = num_distinct_with_paths(s, t)
        print(f"\nSample distinct subsequences (showing first 10 of {len(paths)}):")
        for i, path in enumerate(paths[:10]):
            indices_str = ', '.join(map(str, path))
            chars = ''.join(s[idx] for idx in path)
            print(f"  {i+1}: indices [{indices_str}] → '{chars}'")
    
    return result


def distinct_subsequences_variants():
    """
    DISTINCT SUBSEQUENCES VARIANTS:
    ==============================
    Test different scenarios and related problems.
    """
    
    def count_distinct_subsequences_all(s):
        """Count all distinct subsequences of a string"""
        from collections import defaultdict
        
        n = len(s)
        dp = [0] * (n + 1)
        dp[0] = 1  # Empty subsequence
        
        last_occurrence = {}
        
        for i in range(1, n + 1):
            char = s[i - 1]
            dp[i] = 2 * dp[i - 1]  # Double previous count
            
            # Subtract duplicates if character appeared before
            if char in last_occurrence:
                dp[i] -= dp[last_occurrence[char] - 1]
            
            last_occurrence[char] = i
        
        return dp[n] - 1  # Exclude empty subsequence
    
    def is_subsequence_with_count(s, t):
        """Check if t is subsequence of s and count ways"""
        return num_distinct_dp(s, t)
    
    def longest_common_subsequence_count(s1, s2):
        """Count number of LCS sequences"""
        m, n = len(s1), len(s2)
        
        # First find LCS length
        dp_len = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp_len[i][j] = dp_len[i-1][j-1] + 1
                else:
                    dp_len[i][j] = max(dp_len[i-1][j], dp_len[i][j-1])
        
        lcs_length = dp_len[m][n]
        
        # Count number of LCS
        dp_count = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Base cases
        for i in range(m + 1):
            dp_count[i][0] = 1
        for j in range(n + 1):
            dp_count[0][j] = 1
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp_count[i][j] = dp_count[i-1][j-1]
                else:
                    count = 0
                    if dp_len[i-1][j] == dp_len[i][j]:
                        count += dp_count[i-1][j]
                    if dp_len[i][j-1] == dp_len[i][j]:
                        count += dp_count[i][j-1]
                    if dp_len[i-1][j] == dp_len[i][j] and dp_len[i][j-1] == dp_len[i][j]:
                        count -= dp_count[i-1][j-1]
                    dp_count[i][j] = count
        
        return lcs_length, dp_count[m][n]
    
    # Test variants
    test_cases = [
        ("rabbbit", "rabbit"),
        ("babgbag", "bag"),
        ("abc", ""),
        ("", "abc"),
        ("aaa", "a"),
        ("aabdba", "ab")
    ]
    
    print("Distinct Subsequences Variants Analysis:")
    print("=" * 60)
    
    for s, t in test_cases:
        print(f"\nStrings: s='{s}', t='{t}'")
        
        count = num_distinct_dp(s, t)
        print(f"  Distinct subsequences of '{t}' in '{s}': {count}")
        
        if t:
            # Check if t is subsequence at all
            is_subseq = count > 0
            print(f"  '{t}' is subsequence of '{s}': {is_subseq}")
        
        if len(s) <= 8:
            all_subseq_count = count_distinct_subsequences_all(s)
            print(f"  Total distinct subsequences of '{s}': {all_subseq_count}")
    
    # LCS counting example
    print(f"\nLCS Counting Example:")
    s1, s2 = "ABCDGH", "AEDFHR"
    lcs_len, lcs_count = longest_common_subsequence_count(s1, s2)
    print(f"  s1='{s1}', s2='{s2}'")
    print(f"  LCS length: {lcs_len}")
    print(f"  Number of LCS: {lcs_count}")


# Test cases
def test_num_distinct():
    """Test all implementations with various inputs"""
    test_cases = [
        ("rabbbit", "rabbit", 3),
        ("babgbag", "bag", 5),
        ("abc", "", 1),
        ("", "abc", 0),
        ("abc", "abc", 1),
        ("aaa", "a", 3),
        ("aaa", "aa", 3),
        ("aaa", "aaa", 1),
        ("abcdef", "ace", 1),
        ("adbdadeecadeadeccaeaabdabdbcdabddddabcaaadbabaaedabaaeaadcadcadadaeabaacadbdcdabaaccdbcdaddaaacddcaedadadadadaddccadcdaccbcccdcadadcdacaadadadcacacdeadadededaccbbbcdacceabbaaaddccccccdababdcacaaccdbcdaddbeccecbcddbddddccadceccaccdcdaacdadabaeedabadcdacaaabdccddabadcada", "bcddceeeebecbc", 700531452)
    ]
    
    print("Testing Distinct Subsequences Solutions:")
    print("=" * 70)
    
    for i, (s, t, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"s = '{s[:20]}{'...' if len(s) > 20 else ''}'")
        print(f"t = '{t}'")
        print(f"Expected: {expected}")
        
        # Test approaches (skip expensive ones for long strings)
        if len(s) + len(t) <= 20:
            try:
                recursive = num_distinct_recursive(s, t)
                print(f"Recursive:        {recursive:>10} {'✓' if recursive == expected else '✗'}")
            except:
                print(f"Recursive:        Timeout")
        
        memo = num_distinct_memoization(s, t)
        dp_result = num_distinct_dp(s, t)
        space_opt = num_distinct_space_optimized(s, t)
        space_opt_1d = num_distinct_space_optimized_1d(s, t)
        
        print(f"Memoization:      {memo:>10} {'✓' if memo == expected else '✗'}")
        print(f"DP (2D):          {dp_result:>10} {'✓' if dp_result == expected else '✗'}")
        print(f"Space Opt (2 arr): {space_opt:>10} {'✓' if space_opt == expected else '✗'}")
        print(f"Space Opt (1 arr): {space_opt_1d:>10} {'✓' if space_opt_1d == expected else '✗'}")
        
        # Show paths for small cases
        if len(s) <= 10 and len(t) <= 5 and expected <= 10:
            count, paths = num_distinct_with_paths(s, t)
            print(f"Sample paths: {len(paths)} total")
            for j, path in enumerate(paths[:3]):
                chars = ''.join(s[idx] for idx in path)
                print(f"  Path {j+1}: {path} → '{chars}'")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    num_distinct_analysis("rabbbit", "rabbit")
    
    # Variants demonstration
    print(f"\n" + "=" * 70)
    distinct_subsequences_variants()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. SUBSEQUENCE COUNTING: Count all ways to select characters")
    print("2. CHARACTER MATCHING: When match, add previous count + skip current")
    print("3. SKIP OPTION: Always can skip current character in source")
    print("4. OPTIMAL SUBSTRUCTURE: Counts of subproblems combine additively")
    print("5. SPACE OPTIMIZATION: Can reduce to O(n) space with careful processing")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Pattern Matching: Count occurrences of patterns in text")
    print("• Bioinformatics: Count ways to align sequences")
    print("• Combinatorics: Count selections satisfying constraints")
    print("• String Algorithms: Subsequence enumeration and analysis")
    print("• Dynamic Programming: Foundation for counting problems")


if __name__ == "__main__":
    test_num_distinct()


"""
DISTINCT SUBSEQUENCES - COUNTING WITH CONSTRAINTS:
==================================================

This problem demonstrates counting-based dynamic programming with constraints:
- Count all valid ways rather than finding optimal solution
- Additive recurrence relation instead of min/max
- Foundation for many combinatorial string problems
- Shows how DP can solve exponential counting problems efficiently

KEY INSIGHTS:
============
1. **COUNTING PATTERN**: Add all valid ways instead of taking best
2. **TWO CHOICES**: Always can skip character, use it if it matches
3. **ADDITIVE RECURRENCE**: dp[i][j] = dp[i-1][j] + (match ? dp[i-1][j-1] : 0)
4. **SUBSEQUENCE FLEXIBILITY**: Order matters but gaps are allowed

RECURRENCE RELATION:
===================
```
dp[i][j] = dp[i-1][j]  // Always can skip s[i-1]

if s[i-1] == t[j-1]:
    dp[i][j] += dp[i-1][j-1]  // Also can use matching character
```

BASE CASES:
==========
- dp[i][0] = 1 for all i (empty target has 1 way)
- dp[0][j] = 0 for j > 0 (empty source can't form non-empty target)

SPACE OPTIMIZATION:
==================
Two levels of optimization:
1. **Two arrays**: Current and previous row
2. **Single array**: Process backwards to avoid conflicts

```python
# Single array optimization
for i in range(1, m + 1):
    for j in range(n, 0, -1):  # Process backwards
        if s[i-1] == t[j-1]:
            dp[j] += dp[j-1]
```

APPLICATIONS:
============
- **Pattern Counting**: Count pattern occurrences in text
- **Bioinformatics**: Sequence alignment counting
- **Combinatorics**: Constrained selection counting
- **String Algorithms**: Subsequence analysis
- **Machine Learning**: Feature extraction from sequences

RELATED PROBLEMS:
================
- **Edit Distance**: Similar structure but with operations
- **LCS**: Find longest instead of counting all
- **Palindromic Subsequences**: Count palindromic patterns
- **Pattern Matching**: Various string matching problems

COMPLEXITY:
==========
- **Time**: O(m×n) - optimal for this problem
- **Space**: O(m×n) → O(n) → O(1) with optimizations
- **Counting**: Can be exponential in worst case, but DP makes it polynomial

This problem teaches the fundamental pattern for counting-based DP and
serves as foundation for many combinatorial optimization problems.
"""
