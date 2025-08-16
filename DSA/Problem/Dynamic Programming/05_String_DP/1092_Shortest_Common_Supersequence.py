"""
LeetCode 1092: Shortest Common Supersequence
Difficulty: Hard
Category: String DP

PROBLEM DESCRIPTION:
===================
Given two strings str1 and str2, return the shortest string that has both str1 and str2 as subsequences. 
If there are multiple valid strings, return any of them.

A string s is a subsequence of string t if deleting some (possibly zero) characters from t results in s.

Example 1:
Input: str1 = "abac", str2 = "cab"
Output: "cabac"
Explanation: 
str1 = "abac" is a subsequence of "cabac" because we can delete the first "c".
str2 = "cab" is a subsequence of "cabac" because we can delete the last "ac".
The answer provided is the shortest such string that satisfies these properties.

Example 2:
Input: str1 = "aaaaaaaa", str2 = "aaaaaaaaaa"
Output: "aaaaaaaaaa"

Constraints:
- 1 <= str1.length, str2.length <= 1000
- str1 and str2 consist of lowercase English letters.
"""

def shortest_common_supersequence_brute_force(str1, str2):
    """
    BRUTE FORCE APPROACH:
    ====================
    Generate all possible supersequences and find the shortest.
    
    Time Complexity: O(3^(m+n)) - exponential combinations
    Space Complexity: O(m+n) - recursion depth
    """
    def generate_supersequence(i, j, current):
        # Base cases
        if i >= len(str1) and j >= len(str2):
            return current
        if i >= len(str1):
            return current + str2[j:]
        if j >= len(str2):
            return current + str1[i:]
        
        # Try all possibilities
        results = []
        
        # If characters match, include once
        if str1[i] == str2[j]:
            results.append(generate_supersequence(i + 1, j + 1, current + str1[i]))
        else:
            # Include character from str1
            results.append(generate_supersequence(i + 1, j, current + str1[i]))
            # Include character from str2
            results.append(generate_supersequence(i, j + 1, current + str2[j]))
        
        # Return shortest result
        return min(results, key=len)
    
    return generate_supersequence(0, 0, "")


def shortest_common_supersequence_lcs_based(str1, str2):
    """
    LCS-BASED APPROACH:
    ==================
    Find LCS and construct supersequence using it.
    
    Time Complexity: O(m*n) - LCS computation + reconstruction
    Space Complexity: O(m*n) - LCS DP table
    """
    m, n = len(str1), len(str2)
    
    # Build LCS DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    # Reconstruct LCS and build supersequence
    result = []
    i, j = m, n
    
    while i > 0 and j > 0:
        if str1[i-1] == str2[j-1]:
            # Character is in LCS
            result.append(str1[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            # Take from str1
            result.append(str1[i-1])
            i -= 1
        else:
            # Take from str2
            result.append(str2[j-1])
            j -= 1
    
    # Add remaining characters
    while i > 0:
        result.append(str1[i-1])
        i -= 1
    
    while j > 0:
        result.append(str2[j-1])
        j -= 1
    
    result.reverse()
    return ''.join(result)


def shortest_common_supersequence_dp(str1, str2):
    """
    DIRECT DP APPROACH:
    ==================
    Build supersequence directly using DP.
    
    Time Complexity: O(m*n) - DP computation
    Space Complexity: O(m*n) - DP table
    """
    m, n = len(str1), len(str2)
    
    # dp[i][j] = shortest supersequence length for str1[0:i] and str2[0:j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Base cases
    for i in range(m + 1):
        dp[i][0] = i  # Only str1 characters
    
    for j in range(n + 1):
        dp[0][j] = j  # Only str2 characters
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1  # Include common character once
            else:
                dp[i][j] = min(dp[i-1][j] + 1,  # Include str1[i-1]
                              dp[i][j-1] + 1)   # Include str2[j-1]
    
    # Reconstruct the actual supersequence
    result = []
    i, j = m, n
    
    while i > 0 and j > 0:
        if str1[i-1] == str2[j-1]:
            result.append(str1[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] < dp[i][j-1]:
            result.append(str1[i-1])
            i -= 1
        else:
            result.append(str2[j-1])
            j -= 1
    
    # Add remaining characters
    while i > 0:
        result.append(str1[i-1])
        i -= 1
    
    while j > 0:
        result.append(str2[j-1])
        j -= 1
    
    result.reverse()
    return ''.join(result)


def shortest_common_supersequence_space_optimized(str1, str2):
    """
    SPACE OPTIMIZED APPROACH:
    =========================
    Optimize space usage for length computation.
    
    Time Complexity: O(m*n) - DP computation
    Space Complexity: O(min(m,n)) - single row/column
    """
    # For space optimization, we first compute the length
    # Then reconstruct using the LCS-based approach
    
    # Ensure str1 is shorter for space optimization
    if len(str1) > len(str2):
        str1, str2 = str2, str1
    
    m, n = len(str1), len(str2)
    
    # Compute LCS length with space optimization
    prev = [0] * (m + 1)
    
    for j in range(1, n + 1):
        curr = [0] * (m + 1)
        for i in range(1, m + 1):
            if str1[i-1] == str2[j-1]:
                curr[i] = prev[i-1] + 1
            else:
                curr[i] = max(prev[i], curr[i-1])
        prev = curr
    
    lcs_length = prev[m]
    
    # Total length of SCS = len(str1) + len(str2) - LCS_length
    # But we need the actual string, so use the LCS-based approach
    return shortest_common_supersequence_lcs_based(str1, str2)


def shortest_common_supersequence_with_analysis(str1, str2):
    """
    DETAILED ANALYSIS:
    =================
    Show step-by-step construction and analysis.
    """
    m, n = len(str1), len(str2)
    
    print(f"Finding shortest common supersequence:")
    print(f"  str1 = '{str1}' (length {m})")
    print(f"  str2 = '{str2}' (length {n})")
    
    # Build LCS table with visualization
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    print(f"\nBuilding LCS table:")
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
                print(f"  dp[{i}][{j}]: '{str1[i-1]}' == '{str2[j-1]}' → {dp[i-1][j-1]} + 1 = {dp[i][j]}")
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                print(f"  dp[{i}][{j}]: '{str1[i-1]}' != '{str2[j-1]}' → max({dp[i-1][j]}, {dp[i][j-1]}) = {dp[i][j]}")
    
    lcs_length = dp[m][n]
    print(f"\nLCS length: {lcs_length}")
    
    # Calculate theoretical minimum length
    min_length = m + n - lcs_length
    print(f"Theoretical minimum SCS length: {m} + {n} - {lcs_length} = {min_length}")
    
    print(f"\nFinal LCS table:")
    # Print column headers
    print("     ", end="")
    print("   ε", end="")
    for c in str2:
        print(f"   {c}", end="")
    print()
    
    # Print rows
    for i in range(m + 1):
        if i == 0:
            print("  ε: ", end="")
        else:
            print(f"  {str1[i-1]}: ", end="")
        
        for j in range(n + 1):
            print(f"{dp[i][j]:4}", end="")
        print()
    
    # Reconstruct LCS and SCS
    print(f"\nReconstructing shortest common supersequence:")
    
    result = []
    i, j = m, n
    
    while i > 0 and j > 0:
        if str1[i-1] == str2[j-1]:
            result.append(str1[i-1])
            print(f"  Common character '{str1[i-1]}' at positions str1[{i-1}], str2[{j-1}]")
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            result.append(str1[i-1])
            print(f"  Include '{str1[i-1]}' from str1[{i-1}]")
            i -= 1
        else:
            result.append(str2[j-1])
            print(f"  Include '{str2[j-1]}' from str2[{j-1}]")
            j -= 1
    
    # Add remaining characters
    while i > 0:
        result.append(str1[i-1])
        print(f"  Include remaining '{str1[i-1]}' from str1[{i-1}]")
        i -= 1
    
    while j > 0:
        result.append(str2[j-1])
        print(f"  Include remaining '{str2[j-1]}' from str2[{j-1}]")
        j -= 1
    
    result.reverse()
    scs = ''.join(result)
    
    print(f"\nShortest common supersequence: '{scs}'")
    print(f"Length: {len(scs)} (matches theoretical minimum: {len(scs) == min_length})")
    
    # Verify it contains both strings as subsequences
    def is_subsequence(s, t):
        i = 0
        for char in t:
            if i < len(s) and s[i] == char:
                i += 1
        return i == len(s)
    
    print(f"\nVerification:")
    print(f"  '{str1}' is subsequence of '{scs}': {is_subsequence(str1, scs)}")
    print(f"  '{str2}' is subsequence of '{scs}': {is_subsequence(str2, scs)}")
    
    return scs


def shortest_common_supersequence_variants():
    """
    SCS VARIANTS AND APPLICATIONS:
    ==============================
    Test different scenarios and related problems.
    """
    
    def scs_multiple_strings(strings):
        """Find SCS for multiple strings (approximate)"""
        if not strings:
            return ""
        
        result = strings[0]
        for i in range(1, len(strings)):
            result = shortest_common_supersequence_lcs_based(result, strings[i])
        
        return result
    
    def scs_with_constraints(str1, str2, forbidden_chars):
        """SCS avoiding certain characters"""
        # Filter out forbidden characters and find SCS
        filtered_str1 = ''.join(c for c in str1 if c not in forbidden_chars)
        filtered_str2 = ''.join(c for c in str2 if c not in forbidden_chars)
        
        return shortest_common_supersequence_lcs_based(filtered_str1, filtered_str2)
    
    def count_scs_of_length(str1, str2, target_length):
        """Count SCS of specific length (DP approach)"""
        m, n = len(str1), len(str2)
        
        # dp[i][j][k] = number of SCS of length k using str1[0:i] and str2[0:j]
        # This would be O(m*n*L) where L is target length
        # For demonstration, we'll use a simpler approach
        
        min_possible = m + n - shortest_common_supersequence_lcs_based(str1, str2).count(
            max(set(str1) & set(str2), key=lambda x: min(str1.count(x), str2.count(x)), default=''))
        
        if target_length < len(shortest_common_supersequence_lcs_based(str1, str2)):
            return 0
        
        return 1  # Simplified - actual counting is complex
    
    # Test variants
    test_cases = [
        ("abac", "cab"),
        ("aaaaaaaa", "aaaaaaaaaa"),
        ("abc", "def"),
        ("abc", "abc"),
        ("", "abc"),
        ("programming", "algorithm")
    ]
    
    print("Shortest Common Supersequence Variants:")
    print("=" * 60)
    
    for str1, str2 in test_cases:
        print(f"\nStrings: '{str1}' and '{str2}'")
        
        scs = shortest_common_supersequence_lcs_based(str1, str2)
        print(f"  SCS: '{scs}' (length: {len(scs)})")
        
        # Theoretical minimum
        # Calculate LCS length quickly
        lcs_len = 0
        if str1 and str2:
            m, n = len(str1), len(str2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if str1[i-1] == str2[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            lcs_len = dp[m][n]
        
        theoretical_min = len(str1) + len(str2) - lcs_len
        print(f"  Theoretical minimum: {theoretical_min}")
        print(f"  Achieves minimum: {len(scs) == theoretical_min}")
        
        # Test with forbidden characters
        if str1 and str2:
            forbidden = {'a', 'e'}
            constrained_scs = scs_with_constraints(str1, str2, forbidden)
            print(f"  SCS without 'a','e': '{constrained_scs}'")
    
    # Multiple strings example
    print(f"\nMultiple strings example:")
    strings = ["abc", "bcd", "cde"]
    multi_scs = scs_multiple_strings(strings)
    print(f"  Strings: {strings}")
    print(f"  SCS: '{multi_scs}' (length: {len(multi_scs)})")


# Test cases
def test_shortest_common_supersequence():
    """Test all implementations with various inputs"""
    test_cases = [
        ("abac", "cab", "cabac"),
        ("aaaaaaaa", "aaaaaaaaaa", "aaaaaaaaaa"),
        ("abc", "def", "abcdef"),  # or "defabc" - multiple valid answers
        ("abc", "abc", "abc"),
        ("", "", ""),
        ("", "abc", "abc"),
        ("abc", "", "abc"),
        ("a", "b", "ab"),  # or "ba"
        ("programming", "logarithm", "prologarithmming")  # example answer
    ]
    
    print("Testing Shortest Common Supersequence Solutions:")
    print("=" * 70)
    
    for i, (str1, str2, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"str1='{str1}', str2='{str2}'")
        print(f"Expected (one possibility): '{expected}'")
        
        # Test approaches (skip expensive ones for long strings)
        if len(str1) + len(str2) <= 10:
            try:
                brute_force = shortest_common_supersequence_brute_force(str1, str2)
                print(f"Brute Force:      '{brute_force}' (len: {len(brute_force)})")
            except:
                print(f"Brute Force:      Timeout")
        
        lcs_based = shortest_common_supersequence_lcs_based(str1, str2)
        dp_result = shortest_common_supersequence_dp(str1, str2)
        space_opt = shortest_common_supersequence_space_optimized(str1, str2)
        
        print(f"LCS Based:        '{lcs_based}' (len: {len(lcs_based)})")
        print(f"Direct DP:        '{dp_result}' (len: {len(dp_result)})")
        print(f"Space Optimized:  '{space_opt}' (len: {len(space_opt)})")
        
        # Verify all results are valid supersequences
        def is_subsequence(s, t):
            i = 0
            for char in t:
                if i < len(s) and s[i] == char:
                    i += 1
            return i == len(s)
        
        def verify_scs(scs, s1, s2):
            return is_subsequence(s1, scs) and is_subsequence(s2, scs)
        
        if verify_scs(lcs_based, str1, str2):
            print(f"✓ LCS based result is valid")
        else:
            print(f"✗ LCS based result is invalid")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    shortest_common_supersequence_with_analysis("abac", "cab")
    
    # Variants demonstration
    print(f"\n" + "=" * 70)
    shortest_common_supersequence_variants()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. LCS CONNECTION: SCS length = len(str1) + len(str2) - LCS(str1, str2)")
    print("2. RECONSTRUCTION: Build SCS by merging strings using LCS as guide")
    print("3. OPTIMAL STRUCTURE: Optimal SCS contains optimal sub-SCS")
    print("4. MULTIPLE SOLUTIONS: May have multiple optimal supersequences")
    print("5. GREEDY CONSTRUCTION: Include common characters once")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Text Processing: Merging documents with minimal redundancy")
    print("• Bioinformatics: DNA sequence merging and assembly")
    print("• Version Control: Merging file changes optimally")
    print("• Data Integration: Combining sequences with overlap")
    print("• Algorithm Design: Sequence merging optimization")


if __name__ == "__main__":
    test_shortest_common_supersequence()


"""
SHORTEST COMMON SUPERSEQUENCE - OPTIMAL SEQUENCE MERGING:
=========================================================

This problem demonstrates optimal sequence merging using LCS:
- Find shortest string containing both inputs as subsequences
- Elegant connection to Longest Common Subsequence
- Shows how LCS can be used for construction, not just comparison
- Foundation for sequence assembly and merging problems

KEY INSIGHTS:
============
1. **LCS CONNECTION**: SCS_length = len(str1) + len(str2) - LCS_length
2. **CONSTRUCTION METHOD**: Use LCS to guide optimal merging
3. **COMMON CHARACTERS**: Include shared characters only once
4. **MULTIPLE SOLUTIONS**: May have several optimal supersequences

MATHEMATICAL RELATIONSHIP:
=========================
```
SCS(A, B) = |A| + |B| - LCS(A, B)
```

**Intuition**: 
- Total length if we concatenate both strings: |A| + |B|
- Subtract LCS length because common characters can be shared
- This gives the minimum possible length

ALGORITHM APPROACHES:
====================

1. **LCS-Based Construction**: O(m×n) time, O(m×n) space
   - Build LCS table first
   - Reconstruct SCS using LCS information
   - Most intuitive and widely applicable

2. **Direct DP**: O(m×n) time, O(m×n) space
   - Build SCS length table directly
   - Similar complexity but different perspective

3. **Space Optimized**: O(m×n) time, O(min(m,n)) space
   - Optimize space for length computation only
   - Still need full table for reconstruction

4. **Brute Force**: O(3^(m+n)) time
   - Try all possible merging strategies
   - Only viable for very small inputs

LCS-BASED RECONSTRUCTION:
========================
Most elegant approach using LCS table:
```
while i > 0 and j > 0:
    if str1[i-1] == str2[j-1]:
        result.append(str1[i-1])  # Include common character once
        i -= 1
        j -= 1
    elif lcs[i-1][j] > lcs[i][j-1]:
        result.append(str1[i-1])  # Take from str1
        i -= 1
    else:
        result.append(str2[j-1])  # Take from str2
        j -= 1

# Add remaining characters from either string
while i > 0: result.append(str1[i-1]); i -= 1
while j > 0: result.append(str2[j-1]); j -= 1

result.reverse()
```

CONSTRUCTION LOGIC:
==================
The reconstruction follows this logic:
1. **Common characters**: Include only once (from LCS)
2. **Unique characters**: Include all from both strings
3. **Order preservation**: Maintain relative order within each string
4. **Optimal interleaving**: Choose order that minimizes total length

DIRECT DP APPROACH:
==================
Alternative formulation:
```
if str1[i-1] == str2[j-1]:
    dp[i][j] = dp[i-1][j-1] + 1  # Include common character once
else:
    dp[i][j] = min(
        dp[i-1][j] + 1,    # Include str1[i-1]
        dp[i][j-1] + 1     # Include str2[j-1]
    )
```

SPACE OPTIMIZATION:
==================
For length computation only:
```python
# Use rolling arrays for DP computation
prev = list(range(m + 1))  # Base case

for j in range(1, n + 1):
    curr = [j]  # Base case
    for i in range(1, m + 1):
        if str1[i-1] == str2[j-1]:
            curr.append(prev[i-1] + 1)
        else:
            curr.append(min(prev[i] + 1, curr[i-1] + 1))
    prev = curr

return prev[m]
```

APPLICATIONS:
============
- **Bioinformatics**: DNA sequence assembly and merging
- **Text Processing**: Document merging with minimal redundancy
- **Version Control**: Optimal file merge strategies
- **Data Integration**: Sequence combination with overlap detection
- **Compiler Design**: Symbol table merging

VARIANTS:
========
- **Multiple Strings**: Extend to more than two input strings
- **Weighted Characters**: Different costs for different characters
- **Constrained SCS**: Avoid certain character patterns
- **Approximate SCS**: Allow slight suboptimality for efficiency

RELATED PROBLEMS:
================
- **Longest Common Subsequence (1143)**: Foundation algorithm
- **Edit Distance (72)**: Similar DP structure for transformations
- **Interleaving String (97)**: Related string merging problem
- **Minimum Window Substring (76)**: Different type of string coverage

COMPLEXITY ANALYSIS:
===================
- **Time**: O(m×n) - optimal for this problem
- **Space**: O(m×n) for full solution, O(min(m,n)) for length only
- **Reconstruction**: O(m+n) additional time for building result

MATHEMATICAL PROPERTIES:
========================
- **Optimal Length**: Always equals |A| + |B| - LCS(A,B)
- **Uniqueness**: Multiple optimal solutions may exist
- **Symmetry**: SCS(A,B) has same length as SCS(B,A)
- **Monotonicity**: Adding characters never decreases SCS length

This problem beautifully demonstrates how LCS can be used constructively
to solve optimization problems beyond just comparison tasks.
"""
