"""
LeetCode 97: Interleaving String
Difficulty: Medium
Category: Grid/Matrix DP

PROBLEM DESCRIPTION:
===================
Given strings s1, s2, and s3, find whether s3 is formed by an interleaving of s1 and s2.

An interleaving of two strings s and t is a configuration where s and t are divided into n and m 
substrings respectively, such that:

s = s1 + s2 + ... + sn
t = t1 + t2 + ... + tm
|n - m| <= 1
The interleaving is s1 + t1 + s2 + t2 + ... or t1 + s1 + t2 + s2 + ...

Note: a + b is the concatenation of strings a and b.

Example 1:
Input: s1 = "aabcc", s2 = "dbbca", s3 = "aadbbcbcac"
Output: true
Explanation: One possible way: s1 = "aa" + "bc" + "c", s2 = "dbbc" + "a", s3 = "aa" + "dbbc" + "bc" + "a" + "c"

Example 2:
Input: s1 = "aabcc", s2 = "dbbca", s3 = "aadbbbaccc"
Output: false

Example 3:
Input: s1 = "", s2 = "b", s3 = "b"
Output: true

Constraints:
- 0 <= s1.length, s2.length <= 100
- 0 <= s3.length <= 200
- s1, s2, and s3 consist of lowercase English letters.

Follow up: Could you solve it using only O(s2.length) extra space?
"""

def is_interleave_brute_force(s1, s2, s3):
    """
    BRUTE FORCE APPROACH:
    ====================
    Try all possible ways to interleave s1 and s2.
    
    Time Complexity: O(2^(m+n)) - exponential combinations
    Space Complexity: O(m+n) - recursion depth
    """
    if len(s1) + len(s2) != len(s3):
        return False
    
    def dfs(i1, i2, i3):
        # Base case: all strings consumed
        if i1 == len(s1) and i2 == len(s2) and i3 == len(s3):
            return True
        
        # If s3 is fully consumed but s1 or s2 is not
        if i3 == len(s3):
            return False
        
        result = False
        
        # Try taking character from s1
        if i1 < len(s1) and s1[i1] == s3[i3]:
            result |= dfs(i1 + 1, i2, i3 + 1)
        
        # Try taking character from s2
        if i2 < len(s2) and s2[i2] == s3[i3]:
            result |= dfs(i1, i2 + 1, i3 + 1)
        
        return result
    
    return dfs(0, 0, 0)


def is_interleave_memoization(s1, s2, s3):
    """
    MEMOIZATION APPROACH:
    ====================
    Use memoization to avoid recomputing same subproblems.
    
    Time Complexity: O(m*n) - each state computed once
    Space Complexity: O(m*n) - memoization table
    """
    if len(s1) + len(s2) != len(s3):
        return False
    
    memo = {}
    
    def dfs(i1, i2):
        # Base case: both strings consumed
        if i1 == len(s1) and i2 == len(s2):
            return True
        
        if (i1, i2) in memo:
            return memo[(i1, i2)]
        
        i3 = i1 + i2  # Current position in s3
        result = False
        
        # Try taking character from s1
        if i1 < len(s1) and s1[i1] == s3[i3]:
            result |= dfs(i1 + 1, i2)
        
        # Try taking character from s2
        if i2 < len(s2) and s2[i2] == s3[i3]:
            result |= dfs(i1, i2 + 1)
        
        memo[(i1, i2)] = result
        return result
    
    return dfs(0, 0)


def is_interleave_dp(s1, s2, s3):
    """
    DYNAMIC PROGRAMMING APPROACH:
    ============================
    Build solution bottom-up using 2D DP table.
    
    Time Complexity: O(m*n) - process each cell once
    Space Complexity: O(m*n) - DP table
    """
    if len(s1) + len(s2) != len(s3):
        return False
    
    m, n = len(s1), len(s2)
    
    # dp[i][j] = True if s3[0:i+j] can be formed by interleaving s1[0:i] and s2[0:j]
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    
    # Base case: empty strings
    dp[0][0] = True
    
    # Fill first row (only using s2)
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j-1] and s2[j-1] == s3[j-1]
    
    # Fill first column (only using s1)
    for i in range(1, m + 1):
        dp[i][0] = dp[i-1][0] and s1[i-1] == s3[i-1]
    
    # Fill rest of the table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            k = i + j - 1  # Current position in s3
            
            # Can we extend by taking from s1?
            take_from_s1 = dp[i-1][j] and s1[i-1] == s3[k]
            
            # Can we extend by taking from s2?
            take_from_s2 = dp[i][j-1] and s2[j-1] == s3[k]
            
            dp[i][j] = take_from_s1 or take_from_s2
    
    return dp[m][n]


def is_interleave_space_optimized(s1, s2, s3):
    """
    SPACE OPTIMIZED APPROACH:
    =========================
    Use only O(min(m,n)) space by processing row by row.
    
    Time Complexity: O(m*n) - process each cell once
    Space Complexity: O(min(m,n)) - single row array
    """
    if len(s1) + len(s2) != len(s3):
        return False
    
    # Ensure s1 is the shorter string for space optimization
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    
    m, n = len(s1), len(s2)
    
    # Use single array for previous row
    dp = [False] * (m + 1)
    
    # Initialize for empty s1
    dp[0] = True
    
    # Process each character of s2
    for j in range(1, n + 1):
        # Update for empty s1
        dp[0] = dp[0] and s2[j-1] == s3[j-1]
        
        # Update for each length of s1
        for i in range(1, m + 1):
            k = i + j - 1  # Current position in s3
            
            # Can we extend by taking from s1?
            take_from_s1 = dp[i-1] and s1[i-1] == s3[k]
            
            # Can we extend by taking from s2?
            take_from_s2 = dp[i] and s2[j-1] == s3[k]
            
            dp[i] = take_from_s1 or take_from_s2
    
    return dp[m]


def is_interleave_with_construction(s1, s2, s3):
    """
    FIND ACTUAL INTERLEAVING:
    =========================
    Return whether interleaving exists and construct one if possible.
    
    Time Complexity: O(m*n) - DP + reconstruction
    Space Complexity: O(m*n) - DP table + path tracking
    """
    if len(s1) + len(s2) != len(s3):
        return False, ""
    
    m, n = len(s1), len(s2)
    
    # DP table and choice tracking
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    choice = [[None] * (n + 1) for _ in range(m + 1)]  # 'S1' or 'S2'
    
    # Base case
    dp[0][0] = True
    
    # Fill first row
    for j in range(1, n + 1):
        if dp[0][j-1] and s2[j-1] == s3[j-1]:
            dp[0][j] = True
            choice[0][j] = 'S2'
    
    # Fill first column
    for i in range(1, m + 1):
        if dp[i-1][0] and s1[i-1] == s3[i-1]:
            dp[i][0] = True
            choice[i][0] = 'S1'
    
    # Fill rest of the table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            k = i + j - 1
            
            # Try taking from s1
            if dp[i-1][j] and s1[i-1] == s3[k]:
                dp[i][j] = True
                choice[i][j] = 'S1'
            
            # Try taking from s2 (if not already set)
            elif dp[i][j-1] and s2[j-1] == s3[k]:
                dp[i][j] = True
                choice[i][j] = 'S2'
    
    if not dp[m][n]:
        return False, ""
    
    # Reconstruct the interleaving pattern
    pattern = []
    i, j = m, n
    
    while i > 0 or j > 0:
        if choice[i][j] == 'S1':
            pattern.append(('S1', s1[i-1]))
            i -= 1
        else:  # choice[i][j] == 'S2'
            pattern.append(('S2', s2[j-1]))
            j -= 1
    
    pattern.reverse()
    
    # Build the interleaved string
    result = ''.join(char for _, char in pattern)
    
    return True, result


def is_interleave_analysis(s1, s2, s3):
    """
    DETAILED ANALYSIS:
    =================
    Show step-by-step DP computation and pattern analysis.
    
    Time Complexity: O(m*n) - analysis computation
    Space Complexity: O(m*n) - temporary tables
    """
    print(f"Input strings:")
    print(f"  s1 = '{s1}' (length {len(s1)})")
    print(f"  s2 = '{s2}' (length {len(s2)})")
    print(f"  s3 = '{s3}' (length {len(s3)})")
    
    if len(s1) + len(s2) != len(s3):
        print(f"\nLength mismatch: {len(s1)} + {len(s2)} ≠ {len(s3)}")
        print("Result: False")
        return False
    
    m, n = len(s1), len(s2)
    
    print(f"\nDP table construction:")
    print(f"  dp[i][j] represents: can s3[0:{i+j}] be formed by interleaving s1[0:{i}] and s2[0:{j}]")
    
    # Create DP table for visualization
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    
    # Base case
    dp[0][0] = True
    print(f"\nBase case: dp[0][0] = True (empty strings)")
    
    # Fill first row
    print(f"\nFilling first row (using only s2):")
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j-1] and s2[j-1] == s3[j-1]
        print(f"  dp[0][{j}] = {dp[0][j-1]} and s2[{j-1}]='{s2[j-1]}' == s3[{j-1}]='{s3[j-1]}' => {dp[0][j]}")
    
    # Fill first column
    print(f"\nFilling first column (using only s1):")
    for i in range(1, m + 1):
        dp[i][0] = dp[i-1][0] and s1[i-1] == s3[i-1]
        print(f"  dp[{i}][0] = {dp[i-1][0]} and s1[{i-1}]='{s1[i-1]}' == s3[{i-1}]='{s3[i-1]}' => {dp[i][0]}")
    
    # Fill rest of the table
    print(f"\nFilling rest of table:")
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            k = i + j - 1
            
            take_from_s1 = dp[i-1][j] and s1[i-1] == s3[k]
            take_from_s2 = dp[i][j-1] and s2[j-1] == s3[k]
            dp[i][j] = take_from_s1 or take_from_s2
            
            print(f"  dp[{i}][{j}] (s3[{k}]='{s3[k]}'):")
            print(f"    From s1: {dp[i-1][j]} and s1[{i-1}]='{s1[i-1]}' == s3[{k}] => {take_from_s1}")
            print(f"    From s2: {dp[i][j-1]} and s2[{j-1}]='{s2[j-1]}' == s3[{k}] => {take_from_s2}")
            print(f"    Result: {take_from_s1} or {take_from_s2} = {dp[i][j]}")
    
    print(f"\nFinal DP table:")
    print("     ", end="")
    for j in range(n + 1):
        print(f"{j:5}", end="")
    print()
    
    for i in range(m + 1):
        print(f"{i:3}: ", end="")
        for j in range(n + 1):
            print(f"{str(dp[i][j]):5}", end="")
        print()
    
    result = dp[m][n]
    print(f"\nResult: {result}")
    
    if result:
        exists, pattern = is_interleave_with_construction(s1, s2, s3)
        if exists:
            print(f"One possible interleaving: '{pattern}'")
    
    return result


def is_interleave_all_patterns(s1, s2, s3):
    """
    FIND ALL INTERLEAVING PATTERNS:
    ===============================
    Find all possible ways to interleave s1 and s2 to form s3.
    
    Time Complexity: O(m*n + k) - DP + k patterns
    Space Complexity: O(m*n + k*l) - DP table + patterns storage
    """
    if len(s1) + len(s2) != len(s3):
        return []
    
    m, n = len(s1), len(s2)
    
    # First check if interleaving is possible
    if not is_interleave_dp(s1, s2, s3):
        return []
    
    # Find all patterns using backtracking
    all_patterns = []
    
    def backtrack(i1, i2, current_pattern):
        # Base case: both strings consumed
        if i1 == len(s1) and i2 == len(s2):
            all_patterns.append(current_pattern[:])
            return
        
        i3 = i1 + i2  # Current position in s3
        
        # Try taking from s1
        if i1 < len(s1) and s1[i1] == s3[i3]:
            current_pattern.append(('S1', s1[i1]))
            backtrack(i1 + 1, i2, current_pattern)
            current_pattern.pop()
        
        # Try taking from s2
        if i2 < len(s2) and s2[i2] == s3[i3]:
            current_pattern.append(('S2', s2[i2]))
            backtrack(i1, i2 + 1, current_pattern)
            current_pattern.pop()
    
    backtrack(0, 0, [])
    
    # Convert patterns to readable format
    readable_patterns = []
    for pattern in all_patterns:
        segments = []
        current_source = None
        current_segment = ""
        
        for source, char in pattern:
            if source != current_source:
                if current_segment:
                    segments.append(f"{current_source}:'{current_segment}'")
                current_source = source
                current_segment = char
            else:
                current_segment += char
        
        if current_segment:
            segments.append(f"{current_source}:'{current_segment}'")
        
        readable_patterns.append(" + ".join(segments))
    
    return readable_patterns


# Test cases
def test_is_interleave():
    """Test all implementations with various inputs"""
    test_cases = [
        ("aabcc", "dbbca", "aadbbcbcac", True),
        ("aabcc", "dbbca", "aadbbbaccc", False),
        ("", "b", "b", True),
        ("", "", "", True),
        ("a", "", "a", True),
        ("", "a", "a", True),
        ("a", "b", "ab", True),
        ("a", "b", "ba", True),
        ("a", "b", "c", False),
        ("ab", "cd", "acbd", True),
        ("ab", "cd", "abcd", True),
        ("abc", "def", "adbecf", True),
        ("abc", "def", "adbfce", False)
    ]
    
    print("Testing Interleaving String Solutions:")
    print("=" * 70)
    
    for i, (s1, s2, s3, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"s1='{s1}', s2='{s2}', s3='{s3}'")
        print(f"Expected: {expected}")
        
        # Test approaches (skip expensive ones for long strings)
        if len(s1) + len(s2) <= 8:
            try:
                brute = is_interleave_brute_force(s1, s2, s3)
                print(f"Brute Force:      {brute} {'✓' if brute == expected else '✗'}")
            except:
                print(f"Brute Force:      Timeout")
        
        memo = is_interleave_memoization(s1, s2, s3)
        dp_result = is_interleave_dp(s1, s2, s3)
        space_opt = is_interleave_space_optimized(s1, s2, s3)
        
        print(f"Memoization:      {memo} {'✓' if memo == expected else '✗'}")
        print(f"DP (2D):          {dp_result} {'✓' if dp_result == expected else '✗'}")
        print(f"Space Optimized:  {space_opt} {'✓' if space_opt == expected else '✗'}")
        
        # Show construction for small positive cases
        if len(s1) + len(s2) <= 8 and expected:
            exists, pattern = is_interleave_with_construction(s1, s2, s3)
            if exists:
                print(f"One interleaving: '{pattern}'")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    is_interleave_analysis("aab", "abc", "aaabbc")
    
    # Show all patterns example
    print(f"\n" + "=" * 70)
    print("ALL PATTERNS EXAMPLE:")
    print("-" * 40)
    patterns = is_interleave_all_patterns("ab", "cd", "acbd")
    print(f"s1='ab', s2='cd', s3='acbd'")
    print(f"All interleaving patterns:")
    for i, pattern in enumerate(patterns):
        print(f"  {i+1}: {pattern}")
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. 2D DP PROBLEM: Track positions in both source strings")
    print("2. CHARACTER MATCHING: Must match current character in s3")
    print("3. CHOICE AT EACH STEP: Take from s1 or s2 if characters match")
    print("4. LENGTH CONSTRAINT: s1.length + s2.length must equal s3.length")
    print("5. SPACE OPTIMIZATION: Can reduce to O(min(m,n)) space")
    
    print("\n" + "=" * 70)
    print("Algorithm Comparison:")
    print("Brute Force:      Try all possible interleavings")
    print("Memoization:      Top-down with caching")
    print("DP (2D):          Bottom-up with 2D table")
    print("Space Optimized:  Row-by-row processing")
    print("With Construction: DP + pattern reconstruction")
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Brute Force:      Time: O(2^(m+n)), Space: O(m+n)")
    print("Memoization:      Time: O(m*n),     Space: O(m*n)")
    print("DP (2D):          Time: O(m*n),     Space: O(m*n)")
    print("Space Optimized:  Time: O(m*n),     Space: O(min(m,n))")
    print("With Construction: Time: O(m*n),     Space: O(m*n)")


if __name__ == "__main__":
    test_is_interleave()


"""
PATTERN RECOGNITION:
==================
Interleaving String is a classic 2D string DP problem:
- Combines two input strings to form a target string
- Each position in target must come from one of the source strings
- Maintains relative order within each source string
- Foundation for many string combination and validation problems

KEY INSIGHT - 2D COORDINATE SYSTEM:
==================================
**State Representation**: Use (i,j) to represent:
- Consumed i characters from s1
- Consumed j characters from s2  
- Currently at position i+j in s3

**Transition Logic**: At each step, can either:
- Take next character from s1 (if it matches s3[i+j])
- Take next character from s2 (if it matches s3[i+j])
- Both options possible → multiple valid interleavings

ALGORITHM APPROACHES:
====================

1. **2D DP (Standard)**: O(m×n) time, O(m×n) space
   - Build complete DP table
   - dp[i][j] = can form s3[0:i+j] using s1[0:i] and s2[0:j]
   - Most systematic approach

2. **Space Optimized**: O(m×n) time, O(min(m,n)) space
   - Process row by row using rolling array
   - Ensure shorter string is processed as columns
   - Optimal space complexity

3. **Memoization**: O(m×n) time, O(m×n) space
   - Top-down approach with caching
   - Natural recursive decomposition

4. **Brute Force**: O(2^(m+n)) time
   - Try all possible character selections
   - Exponential complexity

DP STATE DEFINITION:
===================
```
dp[i][j] = True if s3[0:i+j] can be formed by interleaving s1[0:i] and s2[0:j]
```

**Base Cases**:
- dp[0][0] = True (empty strings form empty result)
- dp[i][0] = dp[i-1][0] and s1[i-1] == s3[i-1] (only s1)
- dp[0][j] = dp[0][j-1] and s2[j-1] == s3[j-1] (only s2)

RECURRENCE RELATION:
===================
```
dp[i][j] = (dp[i-1][j] and s1[i-1] == s3[i+j-1]) or 
           (dp[i][j-1] and s2[j-1] == s3[i+j-1])
```

**Interpretation**:
- Can extend from (i-1,j) by taking s1[i-1] if it matches s3[i+j-1]
- Can extend from (i,j-1) by taking s2[j-1] if it matches s3[i+j-1]
- Either possibility makes current state achievable

SPACE OPTIMIZATION:
==================
**Key Observation**: Only need previous row to compute current row

```python
# Instead of dp[m+1][n+1], use dp[n+1]
dp = [False] * (n + 1)

for i in range(m + 1):
    for j in range(n + 1):
        if i == 0 and j == 0:
            dp[j] = True
        elif i == 0:
            dp[j] = dp[j-1] and s2[j-1] == s3[j-1]
        elif j == 0:
            dp[j] = dp[j] and s1[i-1] == s3[i-1]
        else:
            take_s1 = dp[j] and s1[i-1] == s3[i+j-1]
            take_s2 = dp[j-1] and s2[j-1] == s3[i+j-1]
            dp[j] = take_s1 or take_s2
```

**Further Optimization**: Ensure min(m,n) is used as the rolling dimension.

PATTERN RECONSTRUCTION:
======================
To find actual interleaving pattern:
1. **Track choices** during DP computation
2. **Backtrack** from dp[m][n] to dp[0][0]
3. **Record** which string each character came from

```python
choice[i][j] = 'S1' if took from s1, 'S2' if took from s2

# Reconstruction
pattern = []
i, j = m, n
while i > 0 or j > 0:
    if choice[i][j] == 'S1':
        pattern.append(('S1', s1[i-1]))
        i -= 1
    else:
        pattern.append(('S2', s2[j-1]))
        j -= 1
pattern.reverse()
```

EDGE CASES:
==========
1. **Length mismatch**: len(s1) + len(s2) ≠ len(s3) → False
2. **Empty strings**: Handle combinations of empty inputs
3. **Identical characters**: Multiple valid paths possible
4. **No valid interleaving**: All paths lead to False
5. **Single character strings**: Base case verification

MATHEMATICAL PROPERTIES:
=======================
- **Path counting**: Number of valid interleavings ≤ C(m+n, m)
- **Unique solution**: May have 0, 1, or multiple valid interleavings
- **Monotonicity**: Cannot "undo" character selections
- **Optimal substructure**: Valid interleaving contains valid sub-interleavings

APPLICATIONS:
============
1. **Bioinformatics**: DNA sequence alignment and merging
2. **Data Processing**: Merging sorted streams while preserving order
3. **Compiler Design**: Parsing interleaved language constructs
4. **Network Protocols**: Message sequence validation
5. **Version Control**: Merging file histories

VARIANTS TO PRACTICE:
====================
- Edit Distance (72) - similar 2D string DP
- Longest Common Subsequence (1143) - string comparison
- Distinct Subsequences (115) - counting string patterns
- Regular Expression Matching (10) - pattern validation

INTERVIEW TIPS:
==============
1. **Clarify interleaving rules**: Understand the definition clearly
2. **Check length constraint**: Early validation saves computation
3. **Explain 2D state space**: Why we need both string positions
4. **Show space optimization**: 2D → 1D reduction technique
5. **Handle edge cases**: Empty strings, length mismatches
6. **Trace small example**: Demonstrate DP table construction
7. **Pattern reconstruction**: How to find actual interleaving
8. **Multiple solutions**: Explain when multiple patterns exist
9. **Real applications**: Bioinformatics, data processing
10. **Complexity analysis**: Why O(mn) time is necessary

OPTIMIZATION OPPORTUNITIES:
==========================
1. **Early termination**: If remaining characters can't possibly match
2. **Character frequency**: Pre-check if s3 has same character counts
3. **Preprocessing**: Identify impossible positions quickly
4. **Rolling hash**: Fast character sequence comparison

MATHEMATICAL INSIGHT:
====================
This problem demonstrates **constrained string composition**:
- **Two source sequences** must be **merged** while **preserving internal order**
- **2D state space** naturally captures the **dual constraint system**
- **Boolean DP** effectively handles the **existence question**

The interleaving constraint creates a **lattice path problem** in 2D space,
where each path represents a valid way to combine the source strings.
"""
