"""
LeetCode 5: Longest Palindromic Substring
Difficulty: Medium
Category: String DP

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

def longest_palindrome_brute_force(s):
    """
    BRUTE FORCE APPROACH:
    ====================
    Check all possible substrings for palindrome property.
    
    Time Complexity: O(n^3) - O(n^2) substrings, O(n) to check each
    Space Complexity: O(1) - constant extra space
    """
    def is_palindrome(start, end):
        while start < end:
            if s[start] != s[end]:
                return False
            start += 1
            end -= 1
        return True
    
    n = len(s)
    max_len = 1
    result = s[0]
    
    # Check all possible substrings
    for i in range(n):
        for j in range(i + 1, n):
            if is_palindrome(i, j) and j - i + 1 > max_len:
                max_len = j - i + 1
                result = s[i:j + 1]
    
    return result


def longest_palindrome_expand_around_centers(s):
    """
    EXPAND AROUND CENTERS APPROACH:
    ==============================
    For each possible center, expand outwards to find longest palindrome.
    
    Time Complexity: O(n^2) - n centers, O(n) expansion each
    Space Complexity: O(1) - constant extra space
    """
    def expand_around_center(left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return right - left - 1
    
    if not s:
        return ""
    
    start = 0
    max_len = 1
    
    for i in range(len(s)):
        # Odd length palindromes (center at i)
        len1 = expand_around_center(i, i)
        
        # Even length palindromes (center between i and i+1)
        len2 = expand_around_center(i, i + 1)
        
        current_max = max(len1, len2)
        
        if current_max > max_len:
            max_len = current_max
            start = i - (current_max - 1) // 2
    
    return s[start:start + max_len]


def longest_palindrome_dp(s):
    """
    DYNAMIC PROGRAMMING APPROACH:
    ============================
    Build solution bottom-up using 2D DP table.
    
    Time Complexity: O(n^2) - process each cell once
    Space Complexity: O(n^2) - DP table
    """
    n = len(s)
    if n == 0:
        return ""
    
    # dp[i][j] = True if s[i:j+1] is a palindrome
    dp = [[False] * n for _ in range(n)]
    
    start = 0
    max_len = 1
    
    # Single characters are palindromes
    for i in range(n):
        dp[i][i] = True
    
    # Check for palindromes of length 2
    for i in range(n - 1):
        if s[i] == s[i + 1]:
            dp[i][i + 1] = True
            start = i
            max_len = 2
    
    # Check for palindromes of length 3 and more
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            
            # Check if s[i:j+1] is palindrome
            if s[i] == s[j] and dp[i + 1][j - 1]:
                dp[i][j] = True
                start = i
                max_len = length
    
    return s[start:start + max_len]


def longest_palindrome_manacher(s):
    """
    MANACHER'S ALGORITHM:
    ====================
    Linear time algorithm for finding longest palindromic substring.
    
    Time Complexity: O(n) - linear time
    Space Complexity: O(n) - processed string
    """
    if not s:
        return ""
    
    # Preprocess string: insert '#' between characters
    processed = '#'.join('^{}$'.format(s))
    n = len(processed)
    
    # Array to store palindrome radii
    P = [0] * n
    center = right = 0
    
    max_len = 0
    center_index = 0
    
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
            center, right = i, i + P[i]
        
        # Check if this is the longest palindrome found so far
        if P[i] > max_len:
            max_len = P[i]
            center_index = i
    
    # Extract the longest palindrome from original string
    start = (center_index - max_len) // 2
    return s[start:start + max_len]


def longest_palindrome_with_all_longest(s):
    """
    FIND ALL LONGEST PALINDROMES:
    =============================
    Return length and all palindromes of maximum length.
    
    Time Complexity: O(n^2) - expand around centers + collection
    Space Complexity: O(k) where k is number of longest palindromes
    """
    def expand_around_center(left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return right - left - 1, left + 1, right - 1
    
    if not s:
        return 0, []
    
    max_len = 1
    longest_palindromes = [s[0]]
    
    for i in range(len(s)):
        # Odd length palindromes
        len1, start1, end1 = expand_around_center(i, i)
        if len1 > max_len:
            max_len = len1
            longest_palindromes = [s[start1:end1 + 1]]
        elif len1 == max_len and len1 > 1:
            palindrome = s[start1:end1 + 1]
            if palindrome not in longest_palindromes:
                longest_palindromes.append(palindrome)
        
        # Even length palindromes
        len2, start2, end2 = expand_around_center(i, i + 1)
        if len2 > max_len:
            max_len = len2
            longest_palindromes = [s[start2:end2 + 1]]
        elif len2 == max_len and len2 > 1:
            palindrome = s[start2:end2 + 1]
            if palindrome not in longest_palindromes:
                longest_palindromes.append(palindrome)
    
    return max_len, longest_palindromes


def longest_palindrome_analysis(s):
    """
    DETAILED ANALYSIS:
    =================
    Show step-by-step computation and palindrome analysis.
    """
    n = len(s)
    
    print(f"Finding longest palindromic substring in:")
    print(f"  s = '{s}' (length {n})")
    
    if not s:
        print("Empty string!")
        return ""
    
    # Show expand around centers process
    print(f"\nExpand around centers analysis:")
    
    max_len = 1
    best_start = 0
    all_palindromes = []
    
    for i in range(n):
        # Odd length palindromes
        print(f"  Center at position {i} ('{s[i]}'):")
        
        left, right = i, i
        odd_palindromes = []
        
        while left >= 0 and right < n and s[left] == s[right]:
            palindrome = s[left:right + 1]
            odd_palindromes.append(palindrome)
            if len(palindrome) > max_len:
                max_len = len(palindrome)
                best_start = left
            left -= 1
            right += 1
        
        print(f"    Odd length palindromes: {odd_palindromes}")
        
        # Even length palindromes
        if i < n - 1:
            print(f"  Center between positions {i} and {i+1} ('{s[i]}|{s[i+1]}'):")
            
            left, right = i, i + 1
            even_palindromes = []
            
            while left >= 0 and right < n and s[left] == s[right]:
                palindrome = s[left:right + 1]
                even_palindromes.append(palindrome)
                if len(palindrome) > max_len:
                    max_len = len(palindrome)
                    best_start = left
                left -= 1
                right += 1
            
            print(f"    Even length palindromes: {even_palindromes}")
    
    result = s[best_start:best_start + max_len]
    print(f"\nLongest palindromic substring: '{result}' (length: {max_len})")
    
    # Show all longest palindromes
    length, all_longest = longest_palindrome_with_all_longest(s)
    if len(all_longest) > 1:
        print(f"All longest palindromes: {all_longest}")
    
    # Show DP table for small strings
    if n <= 8:
        print(f"\nDP table construction:")
        dp = [[False] * n for _ in range(n)]
        
        # Single characters
        for i in range(n):
            dp[i][i] = True
        
        # Length 2
        for i in range(n - 1):
            if s[i] == s[i + 1]:
                dp[i][i + 1] = True
        
        # Length 3+
        for length in range(3, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                if s[i] == s[j] and dp[i + 1][j - 1]:
                    dp[i][j] = True
        
        print("DP Table (T = palindrome, F = not palindrome):")
        print("   ", end="")
        for j in range(n):
            print(f"{j:4}", end="")
        print()
        
        for i in range(n):
            print(f"{i:2}: ", end="")
            for j in range(n):
                if i <= j:
                    print("   T" if dp[i][j] else "   F", end="")
                else:
                    print("   -", end="")
            print()
    
    return result


def longest_palindrome_variants():
    """
    LONGEST PALINDROMIC SUBSTRING VARIANTS:
    ======================================
    Test different approaches and related problems.
    """
    
    def count_palindromic_substrings(s):
        """Count all palindromic substrings"""
        def expand_around_center(left, right):
            count = 0
            while left >= 0 and right < len(s) and s[left] == s[right]:
                count += 1
                left -= 1
                right += 1
            return count
        
        total_count = 0
        for i in range(len(s)):
            # Odd length
            total_count += expand_around_center(i, i)
            # Even length
            total_count += expand_around_center(i, i + 1)
        
        return total_count
    
    def longest_palindrome_by_removing_one(s):
        """Longest palindrome after removing at most one character"""
        def is_palindrome(string):
            return string == string[::-1]
        
        # Check if already palindrome
        if is_palindrome(s):
            return s
        
        max_len = 0
        result = ""
        
        # Try removing each character
        for i in range(len(s)):
            modified = s[:i] + s[i+1:]
            if is_palindrome(modified) and len(modified) > max_len:
                max_len = len(modified)
                result = modified
        
        return result
    
    def shortest_palindrome_by_prepending(s):
        """Shortest palindrome by adding characters to the beginning"""
        def get_lps(pattern):
            """Get Longest Proper Prefix which is also Suffix array"""
            m = len(pattern)
            lps = [0] * m
            length = 0
            i = 1
            
            while i < m:
                if pattern[i] == pattern[length]:
                    length += 1
                    lps[i] = length
                    i += 1
                else:
                    if length != 0:
                        length = lps[length - 1]
                    else:
                        lps[i] = 0
                        i += 1
            return lps
        
        # Create pattern: s + "#" + reverse(s)
        pattern = s + "#" + s[::-1]
        lps = get_lps(pattern)
        
        # Characters to prepend = len(s) - lps[len(pattern)-1]
        chars_to_prepend = len(s) - lps[-1]
        return s[::-1][:chars_to_prepend] + s
    
    # Test variants
    test_strings = [
        "babad",
        "cbbd",
        "racecar",
        "abcdef",
        "aab",
        "abacabad",
        "aaaa",
        "abcba"
    ]
    
    print("Longest Palindromic Substring Variants Analysis:")
    print("=" * 60)
    
    for s in test_strings:
        print(f"\nString: '{s}'")
        
        # Longest palindromic substring
        longest = longest_palindrome_expand_around_centers(s)
        print(f"  Longest palindromic substring: '{longest}' (length: {len(longest)})")
        
        # Count all palindromic substrings
        count = count_palindromic_substrings(s)
        print(f"  Total palindromic substrings: {count}")
        
        # All longest palindromes
        max_len, all_longest = longest_palindrome_with_all_longest(s)
        if len(all_longest) > 1:
            print(f"  All longest palindromes: {all_longest}")
        
        # Longest palindrome by removing one character
        if len(s) <= 8:  # Only for small strings
            longest_after_removal = longest_palindrome_by_removing_one(s)
            print(f"  Longest after removing one char: '{longest_after_removal}'")
        
        # Shortest palindrome by prepending
        if len(s) <= 8:  # Only for small strings
            shortest_palindrome = shortest_palindrome_by_prepending(s)
            print(f"  Shortest palindrome by prepending: '{shortest_palindrome}'")


# Test cases
def test_longest_palindrome():
    """Test all implementations with various inputs"""
    test_cases = [
        ("babad", ["bab", "aba"]),  # Multiple valid answers
        ("cbbd", ["bb"]),
        ("a", ["a"]),
        ("ac", ["a", "c"]),  # Multiple single chars
        ("racecar", ["racecar"]),
        ("abcdef", ["a", "b", "c", "d", "e", "f"]),  # Multiple single chars
        ("noon", ["noon"]),
        ("abacabad", ["abacaba"]),
        ("forgeeksskeegfor", ["geeksskeeg"]),
        ("abcdcba", ["abcdcba"]),
        ("", [""]),
        ("aaaa", ["aaaa"])
    ]
    
    print("Testing Longest Palindromic Substring Solutions:")
    print("=" * 70)
    
    for i, (s, expected_list) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"s = '{s}'")
        print(f"Expected (any of): {expected_list}")
        
        if s:  # Skip empty string for some methods
            brute_force = longest_palindrome_brute_force(s)
            expand_centers = longest_palindrome_expand_around_centers(s)
            dp_result = longest_palindrome_dp(s)
            
            print(f"Brute Force:      '{brute_force}' {'✓' if brute_force in expected_list else '✗'}")
            print(f"Expand Centers:   '{expand_centers}' {'✓' if expand_centers in expected_list else '✗'}")
            print(f"DP (2D):          '{dp_result}' {'✓' if dp_result in expected_list else '✗'}")
            
            # Test Manacher's algorithm
            if len(s) <= 100:  # Skip for very long strings
                try:
                    manacher = longest_palindrome_manacher(s)
                    print(f"Manacher:         '{manacher}' {'✓' if manacher in expected_list else '✗'}")
                except:
                    print(f"Manacher:         Error")
        else:
            print(f"Empty string case")
        
        # Show all longest palindromes for interesting cases
        if s and len(s) <= 20:
            max_len, all_longest = longest_palindrome_with_all_longest(s)
            if len(all_longest) > 1:
                print(f"All longest: {all_longest}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    longest_palindrome_analysis("babad")
    
    # Variants demonstration
    print(f"\n" + "=" * 70)
    longest_palindrome_variants()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. CONTIGUOUS SUBSTRING: Must be continuous sequence of characters")
    print("2. EXPAND AROUND CENTERS: Most intuitive O(n²) approach")
    print("3. TWO CENTER TYPES: Odd length (single char) and even length (between chars)")
    print("4. MANACHER'S ALGORITHM: Achieves optimal O(n) time complexity")
    print("5. MULTIPLE SOLUTIONS: May have multiple palindromes of same max length")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Text Processing: Finding symmetric patterns")
    print("• Bioinformatics: DNA palindrome analysis") 
    print("• Data Compression: Exploiting palindromic redundancy")
    print("• String Algorithms: Foundation for palindrome problems")
    print("• Pattern Recognition: Symmetric sequence detection")


if __name__ == "__main__":
    test_longest_palindrome()


"""
LONGEST PALINDROMIC SUBSTRING - OPTIMAL PALINDROME DETECTION:
=============================================================

This is one of the most classic string algorithm problems, demonstrating
the evolution from brute force to optimal solutions:
- Showcases the power of the "expand around centers" technique
- Demonstrates Manacher's algorithm for linear time complexity
- Foundation for understanding palindromic pattern detection
- Critical for many string processing applications

KEY INSIGHTS:
============
1. **SUBSTRING vs SUBSEQUENCE**: Must be contiguous characters
2. **CENTER EXPANSION**: Each position can be center of palindromes
3. **TWO CENTER TYPES**: Character centers and between-character centers
4. **PALINDROME PROPERTY**: Symmetric around center point

ALGORITHM APPROACHES:
====================

1. **Brute Force**: O(n³) time, O(1) space
   - Check all O(n²) substrings in O(n) time each
   - Simple but inefficient for large inputs

2. **Expand Around Centers**: O(n²) time, O(1) space
   - Try each of 2n-1 possible centers
   - Expand while characters match
   - Most intuitive optimal approach

3. **Dynamic Programming**: O(n²) time, O(n²) space
   - Build palindrome table bottom-up
   - Good for multiple queries on same string

4. **Manacher's Algorithm**: O(n) time, O(n) space
   - Advanced linear time algorithm
   - Uses preprocessing and symmetry properties

EXPAND AROUND CENTERS TECHNIQUE:
===============================
Core optimal approach:
```
for each position i:
    # Odd length palindromes (center at i)
    expand(i, i)
    
    # Even length palindromes (center between i and i+1)
    expand(i, i+1)

def expand(left, right):
    while left >= 0 and right < n and s[left] == s[right]:
        # Found palindrome s[left:right+1]
        left -= 1
        right += 1
```

MANACHER'S ALGORITHM:
====================
Advanced linear time approach:
1. **Preprocess**: Insert delimiter between chars (handle odd/even uniformly)
2. **Use symmetry**: Reuse previously computed information
3. **Linear scan**: Each character processed at most twice

Key insight: If we know palindrome information for positions before i,
we can often determine information for position i without full expansion.

DP APPROACH:
===========
```
dp[i][j] = True if s[i:j+1] is palindrome

Base cases:
- dp[i][i] = True (single characters)
- dp[i][i+1] = (s[i] == s[i+1]) (length 2)

Recurrence:
- dp[i][j] = (s[i] == s[j]) && dp[i+1][j-1]
```

PRACTICAL CONSIDERATIONS:
========================
- **Expand Around Centers**: Best for single query, simple implementation
- **DP**: Good when building palindrome table for multiple uses
- **Manacher's**: Best asymptotic complexity but more complex
- **Multiple Solutions**: Problem may have multiple optimal answers

APPLICATIONS:
============
- **Text Editors**: Find/highlight palindromic patterns
- **Bioinformatics**: DNA palindrome detection for restriction sites
- **Data Compression**: Identify palindromic redundancy
- **String Processing**: Foundation for palindrome-based algorithms
- **Pattern Recognition**: Symmetric sequence detection

RELATED PROBLEMS:
================
- **Palindromic Substrings**: Count all palindromic substrings
- **Palindrome Partitioning**: Split string into palindromic parts
- **Shortest Palindrome**: Make palindrome with minimum additions
- **Valid Palindrome**: Check if string can become palindrome

COMPLEXITY COMPARISON:
=====================
| Algorithm           | Time | Space | Practical Use     |
|---------------------|------|-------|-------------------|
| Brute Force         | O(n³)| O(1)  | Very small inputs |
| Expand Centers      | O(n²)| O(1)  | General purpose   |
| Dynamic Programming | O(n²)| O(n²) | Multiple queries  |
| Manacher's          | O(n) | O(n)  | Large inputs      |

The expand around centers approach strikes the best balance of simplicity,
efficiency, and space usage for most practical applications.
"""
