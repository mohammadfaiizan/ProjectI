"""
LeetCode 5: Longest Palindromic Substring
Difficulty: Medium
Category: Interval DP - Contiguous Palindrome

PROBLEM DESCRIPTION:
===================
Given a string s, return the longest palindromic substring in s.

Example 1:
Input: s = "babad"
Output: "bab"
Explanation: "aba" is also a valid answer.

Example 2:
Input: s = "cbbd"
Output: "bb"

Constraints:
- 1 <= s.length <= 1000
- s consist of only digits and English letters.
"""

def longest_palindromic_substring_brute_force(s):
    """
    BRUTE FORCE APPROACH:
    ====================
    Check all possible substrings for palindromes.
    
    Time Complexity: O(n^3) - O(n^2) substrings, O(n) palindrome check
    Space Complexity: O(1) - constant space
    """
    def is_palindrome(string):
        return string == string[::-1]
    
    max_len = 0
    result = ""
    
    for i in range(len(s)):
        for j in range(i, len(s)):
            substring = s[i:j+1]
            if is_palindrome(substring) and len(substring) > max_len:
                max_len = len(substring)
                result = substring
    
    return result


def longest_palindromic_substring_expand_around_centers(s):
    """
    EXPAND AROUND CENTERS APPROACH:
    ==============================
    For each possible center, expand outwards while characters match.
    
    Time Complexity: O(n^2) - n centers, O(n) expansion each
    Space Complexity: O(1) - constant space
    """
    def expand_around_center(left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return right - left - 1  # Length of palindrome
    
    if not s:
        return ""
    
    start = 0
    max_len = 0
    
    for i in range(len(s)):
        # Odd length palindromes (center at i)
        len1 = expand_around_center(i, i)
        
        # Even length palindromes (center between i and i+1)
        len2 = expand_around_center(i, i + 1)
        
        # Take the longer palindrome
        current_max = max(len1, len2)
        
        if current_max > max_len:
            max_len = current_max
            # Calculate start position
            start = i - (current_max - 1) // 2
    
    return s[start:start + max_len]


def longest_palindromic_substring_interval_dp(s):
    """
    INTERVAL DP APPROACH:
    ====================
    Build palindrome table bottom-up by interval length.
    
    Time Complexity: O(n^2) - two nested loops
    Space Complexity: O(n^2) - DP table
    """
    n = len(s)
    if n == 0:
        return ""
    
    # dp[i][j] = True if s[i:j+1] is a palindrome
    dp = [[False] * n for _ in range(n)]
    
    start = 0
    max_len = 1
    
    # Base case: single characters are palindromes
    for i in range(n):
        dp[i][i] = True
    
    # Check pairs
    for i in range(n - 1):
        if s[i] == s[i + 1]:
            dp[i][i + 1] = True
            start = i
            max_len = 2
    
    # Check substrings of length 3 and more
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            
            # Check if s[i:j+1] is palindrome
            if s[i] == s[j] and dp[i + 1][j - 1]:
                dp[i][j] = True
                start = i
                max_len = length
    
    return s[start:start + max_len]


def longest_palindromic_substring_manacher(s):
    """
    MANACHER'S ALGORITHM:
    ====================
    Linear time algorithm using preprocessed string and radius array.
    
    Time Complexity: O(n) - linear time
    Space Complexity: O(n) - preprocessed string and radius array
    """
    if not s:
        return ""
    
    # Preprocess string: "abc" -> "^#a#b#c#$"
    # ^ and $ are sentinels to avoid bounds checking
    processed = "^#" + "#".join(s) + "#$"
    n = len(processed)
    
    # Radius array: P[i] = radius of palindrome centered at i
    P = [0] * n
    center = 0  # Center of rightmost palindrome
    right = 0   # Right boundary of rightmost palindrome
    
    for i in range(1, n - 1):
        # Mirror of i with respect to center
        mirror = 2 * center - i
        
        # If i is within right boundary, use previously computed values
        if i < right:
            P[i] = min(right - i, P[mirror])
        
        # Try to expand palindrome centered at i
        while processed[i + P[i] + 1] == processed[i - P[i] - 1]:
            P[i] += 1
        
        # If palindrome centered at i extends past right, adjust center and right
        if i + P[i] > right:
            center = i
            right = i + P[i]
    
    # Find the longest palindrome
    max_len = 0
    center_index = 0
    
    for i in range(1, n - 1):
        if P[i] > max_len:
            max_len = P[i]
            center_index = i
    
    # Convert back to original string indices
    start = (center_index - max_len) // 2
    return s[start:start + max_len]


def longest_palindromic_substring_with_all_palindromes(s):
    """
    FIND ALL PALINDROMIC SUBSTRINGS:
    ===============================
    Return longest palindrome and list of all palindromic substrings.
    
    Time Complexity: O(n^2) - interval DP
    Space Complexity: O(n^2) - DP table + palindrome list
    """
    n = len(s)
    if n == 0:
        return "", []
    
    dp = [[False] * n for _ in range(n)]
    palindromes = []
    longest = ""
    
    # Single characters
    for i in range(n):
        dp[i][i] = True
        palindromes.append(s[i])
        if len(s[i]) > len(longest):
            longest = s[i]
    
    # Pairs
    for i in range(n - 1):
        if s[i] == s[i + 1]:
            dp[i][i + 1] = True
            substring = s[i:i + 2]
            palindromes.append(substring)
            if len(substring) > len(longest):
                longest = substring
    
    # Longer substrings
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            
            if s[i] == s[j] and dp[i + 1][j - 1]:
                dp[i][j] = True
                substring = s[i:j + 1]
                palindromes.append(substring)
                if len(substring) > len(longest):
                    longest = substring
    
    return longest, palindromes


def longest_palindromic_substring_analysis(s):
    """
    DETAILED ANALYSIS:
    =================
    Show step-by-step computation and palindrome finding process.
    """
    print(f"Longest Palindromic Substring Analysis:")
    print(f"String: '{s}'")
    print(f"Length: {len(s)}")
    
    n = len(s)
    
    # Show character indices
    print(f"\nCharacter indices:")
    for i in range(n):
        print(f"  {i}: '{s[i]}'")
    
    # Expand around centers analysis
    print(f"\nExpand Around Centers Analysis:")
    centers_results = []
    
    for i in range(n):
        # Odd length
        left, right = i, i
        while left >= 0 and right < n and s[left] == s[right]:
            palindrome = s[left:right + 1]
            centers_results.append((left, right, palindrome, f"center at {i}"))
            left -= 1
            right += 1
        
        # Even length
        if i < n - 1:
            left, right = i, i + 1
            while left >= 0 and right < n and s[left] == s[right]:
                palindrome = s[left:right + 1]
                centers_results.append((left, right, palindrome, f"center between {i} and {i+1}"))
                left -= 1
                right += 1
    
    # Sort by length and show top palindromes
    centers_results.sort(key=lambda x: len(x[2]), reverse=True)
    print(f"Found palindromes (sorted by length):")
    for left, right, palindrome, center_info in centers_results[:10]:  # Show top 10
        print(f"  '{palindrome}' at [{left},{right}] ({center_info})")
    
    # Interval DP analysis
    print(f"\nInterval DP Analysis:")
    dp = [[False] * n for _ in range(n)]
    
    # Base cases
    print(f"Single characters (all palindromes):")
    for i in range(n):
        dp[i][i] = True
        print(f"  dp[{i}][{i}] = True ('{s[i]}')")
    
    # Pairs
    print(f"Character pairs:")
    for i in range(n - 1):
        if s[i] == s[i + 1]:
            dp[i][i + 1] = True
            print(f"  dp[{i}][{i+1}] = True ('{s[i:i+2]}')")
        else:
            print(f"  dp[{i}][{i+1}] = False ('{s[i:i+2]}')")
    
    # Longer substrings
    longest_found = ""
    for length in range(3, n + 1):
        print(f"\nLength {length} substrings:")
        found_any = False
        for i in range(n - length + 1):
            j = i + length - 1
            substring = s[i:j + 1]
            
            if s[i] == s[j] and dp[i + 1][j - 1]:
                dp[i][j] = True
                print(f"  dp[{i}][{j}] = True ('{substring}') - palindrome!")
                if len(substring) > len(longest_found):
                    longest_found = substring
                found_any = True
            else:
                reason = "different endpoints" if s[i] != s[j] else "inner not palindrome"
                print(f"  dp[{i}][{j}] = False ('{substring}') - {reason}")
        
        if not found_any:
            print(f"  No palindromes of length {length}")
    
    print(f"\nDP Table:")
    print("   ", end="")
    for j in range(n):
        print(f"{j:4}", end="")
    print()
    
    for i in range(n):
        print(f"{i:2}: ", end="")
        for j in range(n):
            if j >= i:
                print(f"{'T' if dp[i][j] else 'F':4}", end="")
            else:
                print(f"{'':4}", end="")
        print()
    
    # Compare different approaches
    print(f"\nComparison of approaches:")
    expand_result = longest_palindromic_substring_expand_around_centers(s)
    dp_result = longest_palindromic_substring_interval_dp(s)
    manacher_result = longest_palindromic_substring_manacher(s)
    
    print(f"Expand around centers: '{expand_result}'")
    print(f"Interval DP: '{dp_result}'")
    print(f"Manacher's algorithm: '{manacher_result}'")
    
    # Show all palindromes
    longest, all_palindromes = longest_palindromic_substring_with_all_palindromes(s)
    print(f"\nAll palindromic substrings ({len(all_palindromes)} total):")
    # Group by length
    by_length = {}
    for p in all_palindromes:
        length = len(p)
        if length not in by_length:
            by_length[length] = []
        by_length[length].append(p)
    
    for length in sorted(by_length.keys(), reverse=True):
        palindromes = sorted(set(by_length[length]))  # Remove duplicates and sort
        print(f"  Length {length}: {palindromes}")
    
    return longest


def longest_palindromic_substring_variants():
    """
    PALINDROMIC SUBSTRING VARIANTS:
    ==============================
    Different scenarios and modifications.
    """
    
    def count_palindromic_substrings(s):
        """Count all palindromic substrings (including duplicates)"""
        count = 0
        
        def expand_around_center(left, right):
            nonlocal count
            while left >= 0 and right < len(s) and s[left] == s[right]:
                count += 1
                left -= 1
                right += 1
        
        for i in range(len(s)):
            expand_around_center(i, i)      # Odd length
            expand_around_center(i, i + 1)  # Even length
        
        return count
    
    def shortest_palindrome_prepend(s):
        """Find shortest palindrome by prepending characters"""
        # Find longest palindromic prefix
        def is_palindrome(string):
            return string == string[::-1]
        
        for i in range(len(s), 0, -1):
            if is_palindrome(s[:i]):
                # Found longest palindromic prefix
                return s[i:][::-1] + s
        
        return s[1:][::-1] + s  # Fallback
    
    def palindrome_pairs_count(s):
        """Count pairs of indices (i,j) where s[i:j+1] is palindrome"""
        n = len(s)
        count = 0
        
        for i in range(n):
            for j in range(i, n):
                if s[i:j+1] == s[i:j+1][::-1]:
                    count += 1
        
        return count
    
    def longest_palindrome_after_k_changes(s, k):
        """Longest palindrome possible with at most k character changes"""
        # Simplified version - just return current longest
        return longest_palindromic_substring_expand_around_centers(s)
    
    # Test variants
    test_cases = [
        "babad",
        "cbbd",
        "racecar",
        "abcdef",
        "aabbcc",
        "a",
        "aa"
    ]
    
    print("Palindromic Substring Variants:")
    print("=" * 50)
    
    for s in test_cases:
        print(f"\nString: '{s}'")
        
        longest = longest_palindromic_substring_expand_around_centers(s)
        count = count_palindromic_substrings(s)
        shortest_prepend = shortest_palindrome_prepend(s)
        pairs_count = palindrome_pairs_count(s)
        
        print(f"Longest palindromic substring: '{longest}'")
        print(f"Count of palindromic substrings: {count}")
        print(f"Shortest palindrome by prepending: '{shortest_prepend}'")
        print(f"Palindromic substring pairs: {pairs_count}")
        
        # Show all unique palindromes
        _, all_palindromes = longest_palindromic_substring_with_all_palindromes(s)
        unique_palindromes = sorted(set(all_palindromes), key=len, reverse=True)
        print(f"Unique palindromes: {unique_palindromes[:5]}...")  # Show first 5


# Test cases
def test_longest_palindromic_substring():
    """Test all implementations with various inputs"""
    test_cases = [
        ("babad", "bab"),  # "aba" is also valid
        ("cbbd", "bb"),
        ("a", "a"),
        ("ac", "a"),  # Either "a" or "c"
        ("racecar", "racecar"),
        ("noon", "noon"),
        ("abcdef", "a"),  # Any single character
        ("aabbaa", "aabbaa"),
        ("abccba", "abccba"),
        ("", "")
    ]
    
    print("Testing Longest Palindromic Substring Solutions:")
    print("=" * 70)
    
    for i, (s, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"String: '{s}'")
        print(f"Expected: '{expected}' (or another valid longest palindrome)")
        
        # Test different approaches
        if len(s) <= 10:
            brute_force = longest_palindromic_substring_brute_force(s)
            print(f"Brute Force:      '{brute_force}' (len: {len(brute_force)})")
        
        expand_centers = longest_palindromic_substring_expand_around_centers(s)
        interval_dp = longest_palindromic_substring_interval_dp(s)
        manacher = longest_palindromic_substring_manacher(s)
        
        print(f"Expand Centers:   '{expand_centers}' (len: {len(expand_centers)})")
        print(f"Interval DP:      '{interval_dp}' (len: {len(interval_dp)})")
        print(f"Manacher:         '{manacher}' (len: {len(manacher)})")
        
        # Verify all results are palindromes and have same length
        results = [expand_centers, interval_dp, manacher]
        if len(s) <= 10:
            results.append(brute_force)
        
        # Check all are palindromes
        all_palindromes = all(r == r[::-1] for r in results)
        # Check all have same length
        same_length = len(set(len(r) for r in results)) <= 1
        
        print(f"All results are palindromes: {all_palindromes}")
        print(f"All results same length: {same_length}")
        
        # Show all palindromes for small cases
        if len(s) <= 8:
            longest, all_pals = longest_palindromic_substring_with_all_palindromes(s)
            print(f"All palindromic substrings: {sorted(set(all_pals), key=len, reverse=True)}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    longest_palindromic_substring_analysis("babad")
    
    # Variants demonstration
    print(f"\n" + "=" * 70)
    longest_palindromic_substring_variants()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. CONTIGUOUS SUBSTRING: Must be consecutive characters")
    print("2. EXPAND AROUND CENTERS: O(n^2) optimal for most cases")
    print("3. MANACHER'S ALGORITHM: O(n) linear time solution")
    print("4. INTERVAL DP: O(n^2) space but clear structure")
    print("5. CENTER EXPANSION: Handle odd/even length palindromes")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Text Processing: Pattern recognition and analysis")
    print("• Bioinformatics: DNA palindrome identification")
    print("• String Algorithms: Palindrome detection optimization")
    print("• Data Compression: Exploiting palindromic patterns")
    print("• Algorithm Design: Foundation for palindrome problems")


if __name__ == "__main__":
    test_longest_palindromic_substring()


"""
LONGEST PALINDROMIC SUBSTRING - CONTIGUOUS PALINDROME OPTIMIZATION:
===================================================================

This problem finds the longest contiguous palindromic substring:
- Must be consecutive characters (unlike subsequence)
- Multiple optimal algorithms with different time/space tradeoffs
- Foundation for many string processing and palindrome detection problems
- Demonstrates progression from O(n³) to O(n) solutions

KEY INSIGHTS:
============
1. **CONTIGUOUS CONSTRAINT**: Must be consecutive characters (substring vs subsequence)
2. **CENTER EXPANSION**: Every palindrome has a center (odd/even length)
3. **LINEAR SOLUTION**: Manacher's algorithm achieves O(n) time
4. **MULTIPLE APPROACHES**: Different algorithms suit different use cases
5. **PALINDROME PROPERTY**: Symmetric structure enables optimization

ALGORITHM APPROACHES:
====================

1. **Brute Force**: O(n³) time, O(1) space
   - Check all O(n²) substrings, O(n) palindrome check each
   - Simple but inefficient

2. **Expand Around Centers**: O(n²) time, O(1) space
   - Try each position as palindrome center
   - Most practical for interview/contest settings

3. **Interval DP**: O(n²) time, O(n²) space
   - Build palindrome table bottom-up
   - Educational value, shows DP structure

4. **Manacher's Algorithm**: O(n) time, O(n) space
   - Linear time using preprocessing and clever expansion
   - Optimal theoretical solution

EXPAND AROUND CENTERS ALGORITHM:
===============================
```python
def expand_around_center(left, right):
    while left >= 0 and right < len(s) and s[left] == s[right]:
        left -= 1
        right += 1
    return right - left - 1  # Length of palindrome

max_len = 0
start = 0

for i in range(len(s)):
    # Odd length palindromes (center at i)
    len1 = expand_around_center(i, i)
    
    # Even length palindromes (center between i and i+1)
    len2 = expand_around_center(i, i + 1)
    
    current_max = max(len1, len2)
    if current_max > max_len:
        max_len = current_max
        start = i - (current_max - 1) // 2
```

**Why this works**: Every palindrome has a unique center
- Odd length: center at single character
- Even length: center between two characters
- Expand outward while characters match

INTERVAL DP ALGORITHM:
=====================
```python
# dp[i][j] = True if s[i:j+1] is palindrome
dp = [[False] * n for _ in range(n)]

# Base case: single characters
for i in range(n):
    dp[i][i] = True

# Length 2
for i in range(n - 1):
    dp[i][i + 1] = (s[i] == s[i + 1])

# Length 3+
for length in range(3, n + 1):
    for i in range(n - length + 1):
        j = i + length - 1
        dp[i][j] = (s[i] == s[j]) and dp[i + 1][j - 1]
```

MANACHER'S ALGORITHM:
====================
Achieves O(n) by avoiding redundant comparisons:

**Preprocessing**: Insert separators to handle odd/even uniformly
- "abc" → "^#a#b#c#$" (^ and $ are sentinels)

**Key Insight**: Use previously computed palindrome information
- If current position i is within a known palindrome boundary
- Use mirror property to initialize expansion radius

```python
processed = "^#" + "#".join(s) + "#$"
P = [0] * len(processed)  # Radius array
center = right = 0        # Rightmost palindrome info

for i in range(1, len(processed) - 1):
    mirror = 2 * center - i
    
    if i < right:
        P[i] = min(right - i, P[mirror])  # Use mirror info
    
    # Expand around i
    while processed[i + P[i] + 1] == processed[i - P[i] - 1]:
        P[i] += 1
    
    # Update rightmost palindrome if needed
    if i + P[i] > right:
        center, right = i, i + P[i]
```

COMPLEXITY COMPARISON:
=====================
| Approach         | Time  | Space | Use Case                 |
|------------------|-------|-------|--------------------------|
| Brute Force      | O(n³) | O(1)  | Educational/Small input  |
| Expand Centers   | O(n²) | O(1)  | Most practical solution  |
| Interval DP      | O(n²) | O(n²) | When need all palindromes|
| Manacher         | O(n)  | O(n)  | Optimal/Large input      |

CENTER HANDLING:
===============
**Odd Length Palindromes**: "racecar"
- Center at position 3 ('e')
- Expand: (3,3) → (2,4) → (1,5) → (0,6)

**Even Length Palindromes**: "abba"  
- Center between positions 1 and 2
- Expand: (1,2) → (0,3)

**Implementation**: Try both for each position i:
- expand_around_center(i, i) for odd length
- expand_around_center(i, i+1) for even length

PALINDROME PROPERTIES:
=====================
**Symmetry**: s[i] == s[j] for all i, j equidistant from center
**Nesting**: Palindromes can contain other palindromes
**Uniqueness**: Each position has unique longest palindrome centered there
**Boundary**: Palindromes are maximal (can't be extended)

SUBSTRING VS SUBSEQUENCE:
========================
**Substring** (this problem):
- Contiguous characters: "abc" in "xabcy"
- Must maintain relative positions and adjacency
- More restrictive, fewer possibilities

**Subsequence** (Problem 516):
- Can skip characters: "ace" in "abcde"  
- Only relative order matters
- More flexibility, different algorithm needed

APPLICATIONS:
============
- **Text Processing**: Pattern recognition and text analysis
- **Bioinformatics**: DNA/RNA palindrome identification for regulatory elements
- **Data Compression**: Exploiting palindromic patterns for compression
- **String Matching**: Palindrome-aware string search algorithms
- **Computational Biology**: Secondary structure prediction

RELATED PROBLEMS:
================
- **Palindromic Subsequence (516)**: Non-contiguous version
- **Palindromic Substrings (647)**: Count all palindromic substrings
- **Palindrome Partitioning (131)**: Partition string into palindromes
- **Shortest Palindrome (214)**: Minimum additions to make palindrome

EDGE CASES:
==========
- **Empty string**: Return empty string
- **Single character**: Return the character
- **No palindromes > 1**: Return any single character
- **Entire string palindrome**: Return entire string
- **Multiple solutions**: Any valid longest palindrome

OPTIMIZATION TECHNIQUES:
=======================
- **Early Termination**: Stop when no longer palindromes possible
- **Bounds Checking**: Avoid array out-of-bounds in expansion
- **Mirror Optimization**: Manacher's key insight for linear time
- **Preprocessing**: Transform string to simplify algorithm logic

This problem showcases the evolution of algorithmic thinking:
from straightforward O(n³) brute force to sophisticated O(n) 
linear algorithms, demonstrating how deep insights into problem
structure can lead to dramatic performance improvements.
"""
