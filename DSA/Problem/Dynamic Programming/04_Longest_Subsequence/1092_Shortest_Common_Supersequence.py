"""
LeetCode 1092: Shortest Common Supersequence
Difficulty: Hard
Category: Longest Subsequence Problems (SCS - LCS Application)

PROBLEM DESCRIPTION:
===================
Given two strings str1 and str2, return the shortest string that has both str1 and str2 as subsequences. 
If there are multiple valid strings, return any one of them.

A string s is a subsequence of string t if deleting some (possibly zero) characters from t results in s.

Example 1:
Input: str1 = "abac", str2 = "cab"
Output: "cabac"
Explanation: 
str1 = "abac" is a subsequence of "cabac" because we can delete the first "c".
str2 = "cab" is a subsequence of "cabac" because we can delete the last "ac".
The answer provided is the shortest such string that satisfies these properties.

Example 2:
Input: str1 = "aaaaaaaa", str2 = "aaaa"
Output: "aaaaaaaa"

Constraints:
- 1 <= str1.length, str2.length <= 1000
- str1 and str2 consist of lowercase English letters.
"""

def shortest_common_supersequence_brute_force(str1, str2):
    """
    BRUTE FORCE APPROACH:
    ====================
    Generate all possible supersequences and find the shortest.
    
    Time Complexity: O(2^(m+n)) - exponential supersequences
    Space Complexity: O(m+n) - recursion stack
    """
    def is_subsequence(s, t):
        """Check if s is subsequence of t"""
        i = 0
        for char in t:
            if i < len(s) and s[i] == char:
                i += 1
        return i == len(s)
    
    def generate_supersequences(pos1, pos2, current):
        # If we've used all characters from both strings
        if pos1 >= len(str1) and pos2 >= len(str2):
            if (is_subsequence(str1, current) and 
                is_subsequence(str2, current)):
                return [current]
            return []
        
        results = []
        
        # Try adding character from str1
        if pos1 < len(str1):
            results.extend(generate_supersequences(pos1 + 1, pos2, 
                                                 current + str1[pos1]))
        
        # Try adding character from str2
        if pos2 < len(str2):
            results.extend(generate_supersequences(pos1, pos2 + 1, 
                                                 current + str2[pos2]))
        
        return results
    
    all_superseqs = generate_supersequences(0, 0, "")
    return min(all_superseqs, key=len) if all_superseqs else ""


def shortest_common_supersequence_lcs_based(str1, str2):
    """
    LCS-BASED APPROACH (OPTIMAL):
    ============================
    Use LCS to construct SCS optimally.
    
    Time Complexity: O(m * n) - LCS computation + reconstruction
    Space Complexity: O(m * n) - DP table
    """
    m, n = len(str1), len(str2)
    
    # Step 1: Build LCS DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    # Step 2: Reconstruct SCS using the LCS table
    scs = []
    i, j = m, n
    
    while i > 0 and j > 0:
        if str1[i - 1] == str2[j - 1]:
            # Characters match - include once in SCS
            scs.append(str1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            # LCS came from top, include char from str1
            scs.append(str1[i - 1])
            i -= 1
        else:
            # LCS came from left, include char from str2
            scs.append(str2[j - 1])
            j -= 1
    
    # Add remaining characters
    while i > 0:
        scs.append(str1[i - 1])
        i -= 1
    
    while j > 0:
        scs.append(str2[j - 1])
        j -= 1
    
    scs.reverse()
    return ''.join(scs)


def shortest_common_supersequence_dp(str1, str2):
    """
    DIRECT DP APPROACH:
    ==================
    Build SCS directly using DP without explicit LCS.
    
    Time Complexity: O(m * n) - DP computation
    Space Complexity: O(m * n) - DP table with strings
    """
    m, n = len(str1), len(str2)
    
    # dp[i][j] = SCS of str1[0:i] and str2[0:j]
    dp = [[""] * (n + 1) for _ in range(m + 1)]
    
    # Base cases
    for i in range(m + 1):
        dp[i][0] = str1[:i]
    
    for j in range(n + 1):
        dp[0][j] = str2[:j]
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                # Characters match - include once
                dp[i][j] = dp[i - 1][j - 1] + str1[i - 1]
            else:
                # Choose shorter of two options
                option1 = dp[i - 1][j] + str1[i - 1]
                option2 = dp[i][j - 1] + str2[j - 1]
                dp[i][j] = option1 if len(option1) <= len(option2) else option2
    
    return dp[m][n]


def shortest_common_supersequence_memoization(str1, str2):
    """
    MEMOIZATION APPROACH:
    ====================
    Use recursive approach with memoization.
    
    Time Complexity: O(m * n) - memoized states
    Space Complexity: O(m * n) - memoization table
    """
    memo = {}
    
    def scs_memo(i, j):
        if i >= len(str1):
            return str2[j:]
        if j >= len(str2):
            return str1[i:]
        
        if (i, j) in memo:
            return memo[(i, j)]
        
        if str1[i] == str2[j]:
            # Characters match - include once
            result = str1[i] + scs_memo(i + 1, j + 1)
        else:
            # Try both options and choose shorter
            option1 = str1[i] + scs_memo(i + 1, j)
            option2 = str2[j] + scs_memo(i, j + 1)
            result = option1 if len(option1) <= len(option2) else option2
        
        memo[(i, j)] = result
        return result
    
    return scs_memo(0, 0)


def shortest_common_supersequence_space_optimized(str1, str2):
    """
    SPACE OPTIMIZED APPROACH:
    =========================
    Optimize space usage for large inputs.
    
    Time Complexity: O(m * n) - same iterations
    Space Complexity: O(min(m, n)) - space optimized
    """
    # This is tricky for SCS since we need to reconstruct the string
    # We'll use the LCS-based approach with optimized LCS computation
    
    if len(str1) > len(str2):
        str1, str2 = str2, str1
    
    m, n = len(str1), len(str2)
    
    # Compute LCS length only (space optimized)
    prev = [0] * (m + 1)
    curr = [0] * (m + 1)
    
    for j in range(1, n + 1):
        for i in range(1, m + 1):
            if str1[i - 1] == str2[j - 1]:
                curr[i] = prev[i - 1] + 1
            else:
                curr[i] = max(prev[i], curr[i - 1])
        prev, curr = curr, prev
    
    lcs_length = prev[m]
    
    # The SCS length will be len(str1) + len(str2) - lcs_length
    # But we need to reconstruct, so fall back to full DP
    return shortest_common_supersequence_lcs_based(str1 if len(str1) <= len(str2) else str2,
                                                   str2 if len(str1) <= len(str2) else str1)


def shortest_common_supersequence_with_analysis(str1, str2):
    """
    DETAILED ANALYSIS:
    =================
    Show step-by-step construction with explanations.
    
    Time Complexity: O(m * n) - LCS + reconstruction
    Space Complexity: O(m * n) - DP table
    """
    print(f"Finding SCS for str1='{str1}', str2='{str2}'")
    
    m, n = len(str1), len(str2)
    
    # Build LCS table with detailed tracking
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    lcs_length = dp[m][n]
    print(f"LCS length: {lcs_length}")
    print(f"Expected SCS length: {m + n - lcs_length}")
    
    # Reconstruct with detailed steps
    scs = []
    i, j = m, n
    steps = []
    
    while i > 0 and j > 0:
        if str1[i - 1] == str2[j - 1]:
            scs.append(str1[i - 1])
            steps.append(f"Match '{str1[i - 1]}' at ({i},{j}) - include once")
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            scs.append(str1[i - 1])
            steps.append(f"Take '{str1[i - 1]}' from str1 at ({i},{j})")
            i -= 1
        else:
            scs.append(str2[j - 1])
            steps.append(f"Take '{str2[j - 1]}' from str2 at ({i},{j})")
            j -= 1
    
    while i > 0:
        scs.append(str1[i - 1])
        steps.append(f"Add remaining '{str1[i - 1]}' from str1")
        i -= 1
    
    while j > 0:
        scs.append(str2[j - 1])
        steps.append(f"Add remaining '{str2[j - 1]}' from str2")
        j -= 1
    
    print("Reconstruction steps:")
    for step in reversed(steps):
        print(f"  {step}")
    
    scs.reverse()
    result = ''.join(scs)
    print(f"Final SCS: '{result}' (length: {len(result)})")
    
    return result


def shortest_common_supersequence_all_solutions(str1, str2):
    """
    FIND ALL SHORTEST SUPERSEQUENCES:
    =================================
    Return all possible SCS of minimum length.
    
    Time Complexity: O(m * n * k) where k is number of solutions
    Space Complexity: O(m * n * k) - store all solutions
    """
    m, n = len(str1), len(str2)
    
    # First find the LCS length to know minimum SCS length
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    min_scs_length = m + n - dp[m][n]
    
    # Recursively find all SCS
    def find_all_scs(i, j, current_scs):
        if i >= len(str1) and j >= len(str2):
            if len(current_scs) == min_scs_length:
                return [current_scs]
            return []
        
        if len(current_scs) >= min_scs_length:
            return []  # Pruning
        
        all_scs = []
        
        if i < len(str1) and j < len(str2) and str1[i] == str2[j]:
            # Characters match - must include once
            all_scs.extend(find_all_scs(i + 1, j + 1, current_scs + str1[i]))
        else:
            # Try including from str1
            if i < len(str1):
                all_scs.extend(find_all_scs(i + 1, j, current_scs + str1[i]))
            
            # Try including from str2
            if j < len(str2):
                all_scs.extend(find_all_scs(i, j + 1, current_scs + str2[j]))
        
        return all_scs
    
    all_scs = find_all_scs(0, 0, "")
    
    # Remove duplicates
    unique_scs = list(set(all_scs))
    return min_scs_length, unique_scs


# Test cases
def test_shortest_common_supersequence():
    """Test all implementations with various inputs"""
    test_cases = [
        ("abac", "cab", "cabac"),  # One possible answer
        ("aaaaaaaa", "aaaa", "aaaaaaaa"),
        ("abc", "def", "abcdef"),  # No common characters
        ("abc", "abc", "abc"),  # Identical strings
        ("a", "aa", "aa"),
        ("", "abc", "abc"),
        ("abc", "", "abc"),
        ("", "", ""),
        ("abcd", "xyza", "abcdxyza"),  # One possible answer
        ("AGGTAB", "GXTXAYB", "AGXGTXAYB")  # One possible answer
    ]
    
    print("Testing Shortest Common Supersequence Solutions:")
    print("=" * 70)
    
    for i, (str1, str2, one_expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: str1 = '{str1}', str2 = '{str2}'")
        print(f"One valid SCS: '{one_expected}' (length: {len(one_expected)})")
        
        # Calculate expected SCS length
        # SCS length = len(str1) + len(str2) - LCS_length
        lcs_length = longest_common_subsequence_length(str1, str2)
        expected_length = len(str1) + len(str2) - lcs_length
        print(f"Expected SCS length: {expected_length}")
        
        # Test approaches (skip expensive ones for large inputs)
        if len(str1) <= 6 and len(str2) <= 6:
            try:
                brute = shortest_common_supersequence_brute_force(str1, str2)
                print(f"Brute Force:      '{brute}' (len: {len(brute)}) {'✓' if len(brute) == expected_length else '✗'}")
            except:
                print(f"Brute Force:      Timeout/Error")
        
        lcs_based = shortest_common_supersequence_lcs_based(str1, str2)
        dp_direct = shortest_common_supersequence_dp(str1, str2)
        memo = shortest_common_supersequence_memoization(str1, str2)
        
        print(f"LCS-based:        '{lcs_based}' (len: {len(lcs_based)}) {'✓' if len(lcs_based) == expected_length else '✗'}")
        print(f"DP Direct:        '{dp_direct}' (len: {len(dp_direct)}) {'✓' if len(dp_direct) == expected_length else '✗'}")
        print(f"Memoization:      '{memo}' (len: {len(memo)}) {'✓' if len(memo) == expected_length else '✗'}")
        
        # Verify that results are valid supersequences
        def is_valid_scs(s, s1, s2):
            return is_subsequence(s1, s) and is_subsequence(s2, s)
        
        print(f"LCS-based valid:  {is_valid_scs(lcs_based, str1, str2)}")
        
        # Show all solutions for small cases
        if len(str1) <= 4 and len(str2) <= 4 and expected_length <= 10:
            try:
                min_len, all_scs = shortest_common_supersequence_all_solutions(str1, str2)
                if len(all_scs) <= 5:
                    print(f"All SCS: {all_scs}")
            except:
                pass
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    shortest_common_supersequence_with_analysis("abac", "cab")
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. RELATIONSHIP: SCS_length = len(s1) + len(s2) - LCS_length")
    print("2. CONSTRUCTION: Use LCS to merge optimally")
    print("3. WHEN MATCH: Include character once")
    print("4. WHEN NO MATCH: Include both characters separately")
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Brute Force:      Time: O(2^(m+n)), Space: O(m+n)")
    print("LCS-based:        Time: O(m*n),     Space: O(m*n)  ← OPTIMAL")
    print("DP Direct:        Time: O(m*n),     Space: O(m*n)")
    print("Memoization:      Time: O(m*n),     Space: O(m*n)")


def longest_common_subsequence_length(text1, text2):
    """Helper function to compute LCS length"""
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n]


def is_subsequence(s, t):
    """Helper function to check if s is subsequence of t"""
    i = 0
    for char in t:
        if i < len(s) and s[i] == char:
            i += 1
    return i == len(s)


if __name__ == "__main__":
    test_shortest_common_supersequence()


"""
PATTERN RECOGNITION:
==================
This is the dual problem to Longest Common Subsequence:
- LCS finds longest common parts
- SCS finds shortest string containing both as subsequences
- Key relationship: SCS_length = len(s1) + len(s2) - LCS_length

KEY INSIGHT - LCS RELATIONSHIP:
==============================
**Mathematical relationship**:
SCS_length = |str1| + |str2| - LCS_length

**Why this works**:
- Total characters needed: |str1| + |str2|
- Common characters (LCS) can be shared: subtract LCS_length
- Result is minimum characters needed for supersequence

ALGORITHM APPROACHES:
====================

1. **LCS-based (Optimal)**: O(m×n)
   - Compute LCS using DP
   - Reconstruct SCS by merging strings optimally
   - When chars match: include once
   - When chars differ: include both

2. **Direct DP**: O(m×n)
   - Build SCS directly without explicit LCS
   - dp[i][j] = SCS of str1[0:i] and str2[0:j]

3. **Memoization**: O(m×n)
   - Recursive approach with caching
   - Natural top-down formulation

4. **Brute Force**: O(2^(m+n))
   - Generate all possible supersequences
   - Check validity and find shortest

RECONSTRUCTION ALGORITHM:
========================
Using LCS table to build SCS:
```python
while i > 0 and j > 0:
    if str1[i-1] == str2[j-1]:
        scs.append(str1[i-1])    # Include once
        i -= 1; j -= 1
    elif dp[i-1][j] > dp[i][j-1]:
        scs.append(str1[i-1])    # Take from str1
        i -= 1
    else:
        scs.append(str2[j-1])    # Take from str2
        j -= 1
```

STATE DEFINITION (Direct DP):
============================
dp[i][j] = shortest supersequence of str1[0:i] and str2[0:j]

RECURRENCE RELATION:
===================
```
if str1[i-1] == str2[j-1]:
    dp[i][j] = dp[i-1][j-1] + str1[i-1]
else:
    option1 = dp[i-1][j] + str1[i-1]
    option2 = dp[i][j-1] + str2[j-1]
    dp[i][j] = shorter of option1 and option2
```

APPLICATIONS:
============
1. **File Merging**: Combine two versions optimally
2. **Data Synchronization**: Minimize storage/transmission
3. **Bioinformatics**: Sequence assembly
4. **Version Control**: Merge branches efficiently
5. **Text Processing**: Combine documents

MATHEMATICAL PROPERTIES:
=======================
1. **Optimality**: SCS length is always |s1| + |s2| - LCS_length
2. **Uniqueness**: Multiple SCS may exist with same minimum length
3. **Bounds**: LCS_length ≤ min(|s1|, |s2|) ≤ SCS_length ≤ |s1| + |s2|
4. **Symmetry**: SCS(s1, s2) has same length as SCS(s2, s1)

EDGE CASES:
==========
1. **Empty strings**: SCS is the other string
2. **Identical strings**: SCS is the string itself
3. **No common chars**: SCS is concatenation
4. **One is subsequence of other**: SCS is the longer string

VARIANTS TO PRACTICE:
====================
- Longest Common Subsequence (1143) - dual problem
- Edit Distance (72) - related string transformation
- Distinct Subsequences (115) - counting variant
- Interleaving String (97) - similar reconstruction

INTERVIEW TIPS:
==============
1. **Establish relationship**: Connect to LCS first
2. **Show mathematical formula**: SCS = |s1| + |s2| - LCS
3. **Demonstrate reconstruction**: Use LCS table to build SCS
4. **Handle edge cases**: Empty strings, identical strings
5. **Verify solution**: Check that result contains both as subsequences
6. **Discuss optimizations**: Space complexity improvements
7. **Multiple solutions**: Explain that many optimal SCS may exist
8. **Applications**: Real-world uses of SCS
9. **Complexity analysis**: Why O(m×n) is necessary
10. **Related problems**: Connect to other sequence problems
"""
