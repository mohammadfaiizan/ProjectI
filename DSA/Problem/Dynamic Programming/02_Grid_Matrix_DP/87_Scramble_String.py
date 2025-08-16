"""
LeetCode 87: Scramble String
Difficulty: Hard
Category: Grid/Matrix DP / String DP

PROBLEM DESCRIPTION:
===================
We can scramble a string s to get a string t using the following algorithm:

1. If the length of the string is 1, stop.
2. If the length of the string is > 1, do the following:
   - Split the string into two non-empty substrings at a random index.
   - Randomly decide to swap the two substrings or to keep them in the same order.
   - Apply this algorithm recursively on each substring.

Given two strings s1 and s2 of the same length, return true if s2 is a scrambled string of s1, otherwise, return false.

Example 1:
Input: s1 = "great", s2 = "rgeat"
Output: true
Explanation: One possible way:
"great" --> "gr/eat" --> "gr/eat" --> "r/g/e/at" --> "r/g/e/a/t" --> "r/g/e/a/t"

Example 2:
Input: s1 = "abcdef", s2 = "fecabd"
Output: false

Example 3:
Input: s1 = "a", s2 = "a"
Output: true

Constraints:
- s1.length == s2.length
- 1 <= s1.length <= 30
- s1 and s2 consist of lowercase English letters.
"""

def is_scramble_brute_force(s1, s2):
    """
    BRUTE FORCE APPROACH:
    ====================
    Try all possible ways to split and scramble strings recursively.
    
    Time Complexity: O(4^n) - exponential in string length
    Space Complexity: O(n) - recursion depth
    """
    if len(s1) != len(s2):
        return False
    
    if s1 == s2:
        return True
    
    # Quick optimization: check if same characters
    if sorted(s1) != sorted(s2):
        return False
    
    n = len(s1)
    
    # Try all possible split points
    for i in range(1, n):
        # Case 1: No swap
        # s1 = s1[:i] + s1[i:]
        # s2 = s2[:i] + s2[i:]
        if (is_scramble_brute_force(s1[:i], s2[:i]) and 
            is_scramble_brute_force(s1[i:], s2[i:])):
            return True
        
        # Case 2: Swap
        # s1 = s1[:i] + s1[i:]
        # s2 = s2[n-i:] + s2[:n-i]
        if (is_scramble_brute_force(s1[:i], s2[n-i:]) and 
            is_scramble_brute_force(s1[i:], s2[:n-i])):
            return True
    
    return False


def is_scramble_memoization(s1, s2):
    """
    MEMOIZATION APPROACH:
    ====================
    Use memoization to avoid recomputing same subproblems.
    
    Time Complexity: O(n^4) - each substring pair computed once
    Space Complexity: O(n^3) - memoization table
    """
    memo = {}
    
    def helper(s1, s2):
        if (s1, s2) in memo:
            return memo[(s1, s2)]
        
        if len(s1) != len(s2):
            memo[(s1, s2)] = False
            return False
        
        if s1 == s2:
            memo[(s1, s2)] = True
            return True
        
        # Quick optimization: check character counts
        if sorted(s1) != sorted(s2):
            memo[(s1, s2)] = False
            return False
        
        n = len(s1)
        
        # Try all possible split points
        for i in range(1, n):
            # Case 1: No swap
            if (helper(s1[:i], s2[:i]) and helper(s1[i:], s2[i:])):
                memo[(s1, s2)] = True
                return True
            
            # Case 2: Swap
            if (helper(s1[:i], s2[n-i:]) and helper(s1[i:], s2[:n-i])):
                memo[(s1, s2)] = True
                return True
        
        memo[(s1, s2)] = False
        return False
    
    return helper(s1, s2)


def is_scramble_dp_3d(s1, s2):
    """
    3D DP APPROACH:
    ==============
    Use 3D DP table with indices instead of substrings.
    
    Time Complexity: O(n^4) - four nested loops
    Space Complexity: O(n^3) - 3D DP table
    """
    if len(s1) != len(s2):
        return False
    
    if s1 == s2:
        return True
    
    n = len(s1)
    
    # dp[i][j][length] = True if s1[i:i+length] can be scrambled to s2[j:j+length]
    dp = [[[False] * (n + 1) for _ in range(n)] for _ in range(n)]
    
    # Base case: single characters
    for i in range(n):
        for j in range(n):
            dp[i][j][1] = (s1[i] == s2[j])
    
    # Fill DP table for increasing lengths
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            for j in range(n - length + 1):
                # Try all possible split points
                for split in range(1, length):
                    # Case 1: No swap
                    if (dp[i][j][split] and 
                        dp[i + split][j + split][length - split]):
                        dp[i][j][length] = True
                        break
                    
                    # Case 2: Swap
                    if (dp[i][j + length - split][split] and 
                        dp[i + split][j][length - split]):
                        dp[i][j][length] = True
                        break
    
    return dp[0][0][n]


def is_scramble_optimized_dp(s1, s2):
    """
    OPTIMIZED DP APPROACH:
    =====================
    Use character count optimization and efficient DP.
    
    Time Complexity: O(n^4) - but with better pruning
    Space Complexity: O(n^3) - 3D DP table
    """
    if len(s1) != len(s2):
        return False
    
    if s1 == s2:
        return True
    
    # Quick character count check
    if sorted(s1) != sorted(s2):
        return False
    
    n = len(s1)
    
    # dp[i][j][length] = True if s1[i:i+length] can be scrambled to s2[j:j+length]
    dp = [[[False] * (n + 1) for _ in range(n)] for _ in range(n)]
    
    # Base case: single characters
    for i in range(n):
        for j in range(n):
            dp[i][j][1] = (s1[i] == s2[j])
    
    # Fill DP table for increasing lengths
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            for j in range(n - length + 1):
                # Quick character count check for this substring pair
                sub1 = s1[i:i+length]
                sub2 = s2[j:j+length]
                
                if sorted(sub1) != sorted(sub2):
                    continue
                
                # Try all possible split points
                for split in range(1, length):
                    # Case 1: No swap
                    if (dp[i][j][split] and 
                        dp[i + split][j + split][length - split]):
                        dp[i][j][length] = True
                        break
                    
                    # Case 2: Swap
                    if (dp[i][j + length - split][split] and 
                        dp[i + split][j][length - split]):
                        dp[i][j][length] = True
                        break
    
    return dp[0][0][n]


def is_scramble_with_path(s1, s2):
    """
    FIND SCRAMBLING PATH:
    ====================
    Return whether scramble is possible and show one possible scrambling sequence.
    
    Time Complexity: O(n^4) - DP computation
    Space Complexity: O(n^3) - DP table + path reconstruction
    """
    if len(s1) != len(s2):
        return False, []
    
    if s1 == s2:
        return True, [f"'{s1}' is already equal to '{s2}'"]
    
    n = len(s1)
    
    # dp[i][j][length] stores (is_possible, split_point, is_swapped)
    dp = [[[None] * (n + 1) for _ in range(n)] for _ in range(n)]
    
    # Base case: single characters
    for i in range(n):
        for j in range(n):
            if s1[i] == s2[j]:
                dp[i][j][1] = (True, -1, False)  # No split for single char
            else:
                dp[i][j][1] = (False, -1, False)
    
    # Fill DP table for increasing lengths
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            for j in range(n - length + 1):
                dp[i][j][length] = (False, -1, False)
                
                # Try all possible split points
                for split in range(1, length):
                    # Case 1: No swap
                    if (dp[i][j][split][0] and 
                        dp[i + split][j + split][length - split][0]):
                        dp[i][j][length] = (True, split, False)
                        break
                    
                    # Case 2: Swap
                    if (dp[i][j + length - split][split][0] and 
                        dp[i + split][j][length - split][0]):
                        dp[i][j][length] = (True, split, True)
                        break
    
    if not dp[0][0][n][0]:
        return False, []
    
    # Reconstruct scrambling path
    def reconstruct_path(i, j, length, prefix=""):
        if length == 1:
            return [f"{prefix}'{s1[i]}' matches '{s2[j]}'"]
        
        is_possible, split, is_swapped = dp[i][j][length]
        
        if not is_possible:
            return [f"{prefix}Cannot scramble '{s1[i:i+length]}' to '{s2[j:j+length]}'"]
        
        sub1 = s1[i:i+length]
        sub2 = s2[j:j+length]
        
        left_part = s1[i:i+split]
        right_part = s1[i+split:i+length]
        
        if is_swapped:
            result = [f"{prefix}Split '{sub1}' into '{left_part}' + '{right_part}', then SWAP to match '{sub2}'"]
            result.extend(reconstruct_path(i, j + length - split, split, prefix + "  "))
            result.extend(reconstruct_path(i + split, j, length - split, prefix + "  "))
        else:
            result = [f"{prefix}Split '{sub1}' into '{left_part}' + '{right_part}', keep order to match '{sub2}'"]
            result.extend(reconstruct_path(i, j, split, prefix + "  "))
            result.extend(reconstruct_path(i + split, j + split, length - split, prefix + "  "))
        
        return result
    
    path = reconstruct_path(0, 0, n)
    return True, path


def is_scramble_analysis(s1, s2):
    """
    DETAILED ANALYSIS:
    =================
    Show step-by-step computation and scrambling analysis.
    """
    print(f"Analyzing scramble possibility:")
    print(f"s1 = '{s1}' (length: {len(s1)})")
    print(f"s2 = '{s2}' (length: {len(s2)})")
    
    if len(s1) != len(s2):
        print("Different lengths - cannot be scrambled")
        return False
    
    if s1 == s2:
        print("Strings are identical - trivially scrambled")
        return True
    
    # Character count analysis
    from collections import Counter
    count1 = Counter(s1)
    count2 = Counter(s2)
    
    print(f"\nCharacter analysis:")
    print(f"s1 characters: {dict(count1)}")
    print(f"s2 characters: {dict(count2)}")
    
    if count1 != count2:
        print("Different character counts - cannot be scrambled")
        return False
    
    print("Character counts match - scrambling might be possible")
    
    # Build DP table with visualization for small strings
    n = len(s1)
    
    if n <= 6:  # Only visualize for small strings
        print(f"\nBuilding DP table:")
        
        dp = [[[False] * (n + 1) for _ in range(n)] for _ in range(n)]
        
        # Base case
        print(f"Base case (length 1):")
        for i in range(n):
            for j in range(n):
                dp[i][j][1] = (s1[i] == s2[j])
                if dp[i][j][1]:
                    print(f"  dp[{i}][{j}][1] = True ('{s1[i]}' == '{s2[j]}')")
        
        # Fill for increasing lengths
        for length in range(2, n + 1):
            print(f"\nLength {length}:")
            for i in range(n - length + 1):
                for j in range(n - length + 1):
                    sub1 = s1[i:i+length]
                    sub2 = s2[j:j+length]
                    
                    print(f"  Checking '{sub1}' vs '{sub2}':")
                    
                    found = False
                    for split in range(1, length):
                        # Case 1: No swap
                        if (dp[i][j][split] and 
                            dp[i + split][j + split][length - split]):
                            dp[i][j][length] = True
                            print(f"    Split at {split}, no swap: TRUE")
                            found = True
                            break
                        
                        # Case 2: Swap
                        if (dp[i][j + length - split][split] and 
                            dp[i + split][j][length - split]):
                            dp[i][j][length] = True
                            print(f"    Split at {split}, with swap: TRUE")
                            found = True
                            break
                    
                    if not found:
                        print(f"    No valid split found: FALSE")
        
        result = dp[0][0][n]
    else:
        result = is_scramble_dp_3d(s1, s2)
    
    print(f"\nFinal result: {result}")
    
    # Show scrambling path if possible
    if result:
        is_possible, path = is_scramble_with_path(s1, s2)
        if is_possible and path:
            print(f"\nOne possible scrambling sequence:")
            for step in path:
                print(f"  {step}")
    
    return result


# Test cases
def test_is_scramble():
    """Test all implementations with various inputs"""
    test_cases = [
        ("great", "rgeat", True),
        ("abcdef", "fecabd", False),
        ("a", "a", True),
        ("abc", "acb", True),
        ("abc", "bca", True),
        ("abc", "cab", True),
        ("abc", "def", False),
        ("abcd", "cdab", True),
        ("abcd", "cadb", False),
        ("hwareg", "grhwea", True),
        ("abcdbdacbdac", "bdacabcdbdac", True),
        ("", "", True),
        ("ab", "ba", True),
        ("abb", "bab", True)
    ]
    
    print("Testing Scramble String Solutions:")
    print("=" * 70)
    
    for i, (s1, s2, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"s1 = '{s1}', s2 = '{s2}'")
        print(f"Expected: {expected}")
        
        # Test approaches (skip expensive ones for long strings)
        if len(s1) <= 8:
            try:
                brute = is_scramble_brute_force(s1, s2)
                print(f"Brute Force:      {brute} {'✓' if brute == expected else '✗'}")
            except:
                print(f"Brute Force:      Error")
        
        memo_result = is_scramble_memoization(s1, s2)
        dp_result = is_scramble_dp_3d(s1, s2)
        optimized_result = is_scramble_optimized_dp(s1, s2)
        
        print(f"Memoization:      {memo_result} {'✓' if memo_result == expected else '✗'}")
        print(f"3D DP:            {dp_result} {'✓' if dp_result == expected else '✗'}")
        print(f"Optimized DP:     {optimized_result} {'✓' if optimized_result == expected else '✗'}")
        
        # Show path for small positive cases
        if len(s1) <= 6 and expected:
            is_possible, path = is_scramble_with_path(s1, s2)
            if is_possible and len(path) <= 10:
                print(f"Scrambling path: {len(path)} steps")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    is_scramble_analysis("great", "rgeat")
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. RECURSIVE STRUCTURE: Scrambling creates tree of substring comparisons")
    print("2. TWO CASES PER SPLIT: No swap vs swap after splitting")
    print("3. CHARACTER COUNT: Necessary but not sufficient condition")
    print("4. OVERLAPPING SUBPROBLEMS: Same substring pairs appear in multiple contexts")
    print("5. OPTIMAL SUBSTRUCTURE: Scrambling of parts determines whole")
    
    print("\n" + "=" * 70)
    print("Algorithm Comparison:")
    print("Brute Force:      Try all possible scrambling sequences")
    print("Memoization:      Cache results for substring pairs")
    print("3D DP:            Bottom-up with index-based states")
    print("Optimized DP:     Add character count pruning")
    print("With Path:        DP + reconstruction of scrambling steps")
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Brute Force:      Time: O(4^n),   Space: O(n)")
    print("Memoization:      Time: O(n^4),   Space: O(n^3)")
    print("3D DP:            Time: O(n^4),   Space: O(n^3)")
    print("Optimized DP:     Time: O(n^4),   Space: O(n^3)")
    print("With Path:        Time: O(n^4),   Space: O(n^3)")


if __name__ == "__main__":
    test_is_scramble()


"""
PATTERN RECOGNITION:
==================
Scramble String is a complex recursive string validation problem:
- Recursive structure with two branching cases at each split
- Requires checking all possible ways to partition and rearrange strings
- Demonstrates advanced memoization and 3D dynamic programming
- Foundation for problems involving string transformations and recursive validation

KEY INSIGHT - SCRAMBLING PROCESS:
=================================
**Scrambling Algorithm**:
1. Split string at any position into two non-empty parts
2. Choose to either keep order or swap the two parts
3. Recursively apply scrambling to each part

**Validation Logic**: s2 is a scramble of s1 if:
- They have same length and character counts
- There exists a split position where either:
  - **No swap**: left parts match AND right parts match
  - **Swap**: left part matches right part of s2 AND right part matches left part of s2

**Mathematical Representation**:
```
scramble(s1, s2) = ∃i ∈ [1, n-1] such that:
  (scramble(s1[0:i], s2[0:i]) ∧ scramble(s1[i:n], s2[i:n])) ∨
  (scramble(s1[0:i], s2[n-i:n]) ∧ scramble(s1[i:n], s2[0:n-i]))
```

ALGORITHM APPROACHES:
====================

1. **Brute Force Recursion**: O(4^n) time
   - Try all possible split points and both cases (swap/no swap)
   - Exponential time due to repeated subproblems
   - Only viable for very small strings

2. **Memoization**: O(n^4) time, O(n^3) space
   - Cache results for (s1_substring, s2_substring) pairs
   - Eliminates redundant computation
   - String-based memoization

3. **3D DP**: O(n^4) time, O(n^3) space
   - Use indices instead of substring objects
   - dp[i][j][len] = can s1[i:i+len] be scrambled to s2[j:j+len]
   - More memory efficient than string memoization

4. **Optimized DP**: O(n^4) time, O(n^3) space
   - Add character count pruning
   - Skip impossible cases early
   - Better practical performance

3D DP STATE DEFINITION:
======================
```
dp[i][j][length] = True if s1[i:i+length] can be scrambled to s2[j:j+length]
```

**Base Case**: 
```
dp[i][j][1] = (s1[i] == s2[j])  // Single characters must match
```

**Recurrence Relation**:
```
dp[i][j][length] = ∃split ∈ [1, length-1] such that:
  // Case 1: No swap
  (dp[i][j][split] ∧ dp[i+split][j+split][length-split]) ∨
  
  // Case 2: Swap
  (dp[i][j+length-split][split] ∧ dp[i+split][j][length-split])
```

OPTIMIZATION TECHNIQUES:
=======================

1. **Character Count Pruning**:
```python
if sorted(s1) != sorted(s2):
    return False  # Quick elimination
```

2. **Early Termination**:
```python
for split in range(1, length):
    if condition_satisfied:
        dp[i][j][length] = True
        break  // No need to check other splits
```

3. **Memory Optimization**:
```python
# Use tuple-based memoization instead of string concatenation
memo = {}  # (i1, j1, i2, j2) -> bool
```

SUBSTRING RELATIONSHIP ANALYSIS:
===============================
**Two Split Cases Explained**:

**Case 1 - No Swap**:
```
s1: [  A  |  B  ]
s2: [  A' |  B' ]

A must scramble to A', B must scramble to B'
```

**Case 2 - Swap**:
```
s1: [  A  |  B  ]
s2: [  B' |  A' ]

A must scramble to A', B must scramble to B'
(Note: A' and B' are swapped in s2)
```

**Index Relationships**:
- No swap: s1[i:i+split] ↔ s2[j:j+split], s1[i+split:i+length] ↔ s2[j+split:j+length]
- Swap: s1[i:i+split] ↔ s2[j+length-split:j+length], s1[i+split:i+length] ↔ s2[j:j+length-split]

PATH RECONSTRUCTION:
===================
To find actual scrambling sequence:
1. **Track split decisions** during DP computation
2. **Record swap/no-swap choice** for each valid state
3. **Recursively reconstruct** the scrambling tree

```python
# During DP
if valid_case_found:
    dp[i][j][length] = (True, split_point, is_swapped)

# Reconstruction
def reconstruct(i, j, length):
    if length == 1:
        return f"'{s1[i]}' matches '{s2[j]}'"
    
    _, split, swapped = dp[i][j][length]
    if swapped:
        return f"Split and swap: {reconstruct(left)} + {reconstruct(right)}"
    else:
        return f"Split no swap: {reconstruct(left)} + {reconstruct(right)}"
```

APPLICATIONS:
============
1. **Compiler Design**: Parsing tree transformations
2. **Bioinformatics**: DNA sequence rearrangement analysis
3. **Cryptography**: Permutation cipher analysis
4. **Data Structures**: Tree isomorphism checking
5. **Natural Language Processing**: Syntax tree comparison

VARIANTS TO PRACTICE:
====================
- Interleaving String (97) - different type of string combination
- Distinct Subsequences (115) - counting string patterns
- Regular Expression Matching (10) - pattern validation with rules
- Wildcard Matching (44) - another complex string validation

INTERVIEW TIPS:
==============
1. **Understand scrambling process**: Clearly explain the recursive splitting
2. **Identify two cases**: No swap vs swap after splitting
3. **Character count optimization**: Quick elimination technique
4. **State space design**: Why 3D DP with (i,j,length) is natural
5. **Base case handling**: Single character comparison
6. **Memoization vs DP**: Trade-offs between approaches
7. **Path reconstruction**: How to find actual scrambling sequence
8. **Edge cases**: Empty strings, single characters, identical strings
9. **Optimization opportunities**: Pruning and early termination
10. **Real applications**: Compiler design, bioinformatics

MATHEMATICAL INSIGHT:
====================
Scramble String demonstrates **recursive tree validation**:
- **Binary tree structure** emerges from the split-and-transform process
- **Two transformation types** (swap/no-swap) create complex state space
- **Dynamic programming** eliminates exponential redundancy through memoization

The problem showcases how **string transformation problems** can be solved
efficiently using **structured recursive decomposition** combined with
**memoization techniques** to achieve polynomial time complexity.

The **3D state space** (start1, start2, length) elegantly captures all
possible substring relationships while enabling systematic exploration
of the transformation space.
"""
