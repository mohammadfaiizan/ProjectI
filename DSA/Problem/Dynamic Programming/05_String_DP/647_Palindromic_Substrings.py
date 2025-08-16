"""
LeetCode 647: Palindromic Substrings
Difficulty: Medium
Category: String DP

PROBLEM DESCRIPTION:
===================
Given a string s, return the number of palindromic substrings in it.

A string is a palindrome when it reads the same backward as forward.
A substring is a contiguous sequence of characters within the string.

Example 1:
Input: s = "abc"
Output: 3
Explanation: Three palindromic strings: "a", "b", "c".

Example 2:
Input: s = "aaa"
Output: 6
Explanation: Six palindromic strings: "a", "a", "a", "aa", "aa", "aaa".

Constraints:
- 1 <= s.length <= 1000
- s consists of lowercase English letters.
"""

def count_substrings_brute_force(s):
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
    count = 0
    
    # Check all possible substrings
    for i in range(n):
        for j in range(i, n):
            if is_palindrome(i, j):
                count += 1
    
    return count


def count_substrings_expand_around_centers(s):
    """
    EXPAND AROUND CENTERS APPROACH:
    ==============================
    For each possible center, expand outwards to find palindromes.
    
    Time Complexity: O(n^2) - n centers, O(n) expansion each
    Space Complexity: O(1) - constant extra space
    """
    def expand_around_center(left, right):
        count = 0
        while left >= 0 and right < len(s) and s[left] == s[right]:
            count += 1
            left -= 1
            right += 1
        return count
    
    n = len(s)
    total_count = 0
    
    for i in range(n):
        # Odd length palindromes (center at i)
        total_count += expand_around_center(i, i)
        
        # Even length palindromes (center between i and i+1)
        total_count += expand_around_center(i, i + 1)
    
    return total_count


def count_substrings_dp(s):
    """
    DYNAMIC PROGRAMMING APPROACH:
    ============================
    Build solution bottom-up using 2D DP table.
    
    Time Complexity: O(n^2) - process each cell once
    Space Complexity: O(n^2) - DP table
    """
    n = len(s)
    
    # dp[i][j] = True if s[i:j+1] is a palindrome
    dp = [[False] * n for _ in range(n)]
    count = 0
    
    # Single characters are palindromes
    for i in range(n):
        dp[i][i] = True
        count += 1
    
    # Check for palindromes of length 2
    for i in range(n - 1):
        if s[i] == s[i + 1]:
            dp[i][i + 1] = True
            count += 1
    
    # Check for palindromes of length 3 and more
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            
            # Check if s[i:j+1] is palindrome
            if s[i] == s[j] and dp[i + 1][j - 1]:
                dp[i][j] = True
                count += 1
    
    return count


def count_substrings_space_optimized(s):
    """
    SPACE OPTIMIZED APPROACH:
    =========================
    Use only O(n) space by processing diagonally.
    
    Time Complexity: O(n^2) - process each cell once
    Space Complexity: O(n) - single row array
    """
    n = len(s)
    count = 0
    
    # Use previous diagonal values
    prev_diag = [False] * n
    
    # Single characters
    for i in range(n):
        prev_diag[i] = True
        count += 1
    
    # Process for increasing lengths
    for length in range(2, n + 1):
        curr_diag = [False] * n
        
        for i in range(n - length + 1):
            j = i + length - 1
            
            if length == 2:
                curr_diag[i] = (s[i] == s[j])
            else:
                curr_diag[i] = (s[i] == s[j]) and prev_diag[i + 1]
            
            if curr_diag[i]:
                count += 1
        
        prev_diag = curr_diag
    
    return count


def count_substrings_manacher_variant(s):
    """
    MANACHER'S ALGORITHM VARIANT:
    ============================
    Modified Manacher's algorithm to count palindromes.
    
    Time Complexity: O(n) - linear time
    Space Complexity: O(n) - processed string
    """
    # Preprocess string: insert '#' between characters
    processed = '#'.join('^{}$'.format(s))
    n = len(processed)
    
    # Array to store palindrome radii
    P = [0] * n
    center = right = 0
    count = 0
    
    for i in range(1, n - 1):
        # Mirror of i with respect to center
        mirror = 2 * center - i
        
        # If i is within right boundary, use previously computed values
        if i < right:
            P[i] = min(right - i, P[mirror])
        
        # Try to expand palindrome centered at i
        try:
            while processed[i + P[i] + 1] == processed[i - P[i] - 1]:
                P[i] += 1
        except IndexError:
            pass
        
        # If palindrome centered at i extends past right, adjust center and right
        if i + P[i] > right:
            center, right = i, i + P[i]
        
        # Count palindromes: P[i] gives radius, so (P[i] + 1) // 2 palindromes
        count += (P[i] + 1) // 2
    
    return count


def count_substrings_with_details(s):
    """
    COUNT WITH PALINDROME DETAILS:
    ==============================
    Return count and all palindromic substrings.
    
    Time Complexity: O(n^2) - expand around centers
    Space Complexity: O(k) where k is number of palindromes
    """
    def expand_and_collect(left, right):
        palindromes = []
        while left >= 0 and right < len(s) and s[left] == s[right]:
            palindromes.append(s[left:right + 1])
            left -= 1
            right += 1
        return palindromes
    
    n = len(s)
    all_palindromes = []
    
    for i in range(n):
        # Odd length palindromes
        all_palindromes.extend(expand_and_collect(i, i))
        
        # Even length palindromes
        all_palindromes.extend(expand_and_collect(i, i + 1))
    
    return len(all_palindromes), all_palindromes


def count_substrings_by_length(s):
    """
    COUNT PALINDROMES BY LENGTH:
    ============================
    Return count breakdown by palindrome length.
    
    Time Complexity: O(n^2) - expand around centers
    Space Complexity: O(n) - count by length
    """
    from collections import defaultdict
    
    def expand_and_count_by_length(left, right, length_counts):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            length = right - left + 1
            length_counts[length] += 1
            left -= 1
            right += 1
    
    n = len(s)
    length_counts = defaultdict(int)
    
    for i in range(n):
        # Odd length palindromes
        expand_and_count_by_length(i, i, length_counts)
        
        # Even length palindromes
        expand_and_count_by_length(i, i + 1, length_counts)
    
    total_count = sum(length_counts.values())
    return total_count, dict(length_counts)


def count_substrings_analysis(s):
    """
    DETAILED ANALYSIS:
    =================
    Show step-by-step computation and palindrome analysis.
    """
    n = len(s)
    
    print(f"Counting palindromic substrings in:")
    print(f"  s = '{s}' (length {n})")
    
    # Show all palindromes with positions
    count, all_palindromes = count_substrings_with_details(s)
    
    print(f"\nAll palindromic substrings ({count} total):")
    
    # Group by length for better visualization
    from collections import defaultdict
    by_length = defaultdict(list)
    
    for palindrome in all_palindromes:
        by_length[len(palindrome)].append(palindrome)
    
    for length in sorted(by_length.keys()):
        palindromes = by_length[length]
        print(f"  Length {length}: {palindromes} (count: {len(palindromes)})")
    
    # Show DP table construction for small strings
    if n <= 8:
        print(f"\nDP table construction:")
        print(f"  dp[i][j] = True if s[i:j+1] is palindrome")
        
        dp = [[False] * n for _ in range(n)]
        
        # Single characters
        print(f"\nSingle characters:")
        for i in range(n):
            dp[i][i] = True
            print(f"  dp[{i}][{i}] = True ('{s[i]}')")
        
        # Length 2
        if n > 1:
            print(f"\nLength 2:")
            for i in range(n - 1):
                if s[i] == s[i + 1]:
                    dp[i][i + 1] = True
                    print(f"  dp[{i}][{i+1}] = True ('{s[i:i+2]}')")
                else:
                    print(f"  dp[{i}][{i+1}] = False ('{s[i:i+2]}')")
        
        # Length 3+
        for length in range(3, n + 1):
            print(f"\nLength {length}:")
            for i in range(n - length + 1):
                j = i + length - 1
                substring = s[i:j+1]
                
                if s[i] == s[j] and dp[i + 1][j - 1]:
                    dp[i][j] = True
                    print(f"  dp[{i}][{j}] = True ('{substring}': ends match and inner is palindrome)")
                else:
                    print(f"  dp[{i}][{j}] = False ('{substring}')")
        
        # Show final DP table
        print(f"\nFinal DP table:")
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
    
    # Show count by length breakdown
    total, length_breakdown = count_substrings_by_length(s)
    print(f"\nPalindrome count by length:")
    for length in sorted(length_breakdown.keys()):
        count = length_breakdown[length]
        print(f"  Length {length}: {count} palindromes")
    
    print(f"\nTotal palindromic substrings: {total}")
    
    return total


def palindromic_substrings_variants():
    """
    PALINDROMIC SUBSTRINGS VARIANTS:
    ===============================
    Test different approaches and related problems.
    """
    
    def longest_palindromic_substring(s):
        """Find longest palindromic substring"""
        def expand_around_center(left, right):
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1
            return right - left - 1
        
        start = 0
        max_len = 1
        
        for i in range(len(s)):
            # Odd length
            len1 = expand_around_center(i, i)
            # Even length
            len2 = expand_around_center(i, i + 1)
            
            current_max = max(len1, len2)
            if current_max > max_len:
                max_len = current_max
                start = i - (current_max - 1) // 2
        
        return s[start:start + max_len]
    
    def count_palindromic_subsequences(s):
        """Count palindromic subsequences (not substrings)"""
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
        
        return dp[0][n - 1] if n > 0 else 0
    
    def min_cuts_for_palindrome_partitioning(s):
        """Minimum cuts to partition string into palindromes"""
        n = len(s)
        
        # is_palindrome[i][j] = True if s[i:j+1] is palindrome
        is_palindrome = [[False] * n for _ in range(n)]
        
        # Build palindrome table
        for i in range(n):
            is_palindrome[i][i] = True
        
        for i in range(n - 1):
            is_palindrome[i][i + 1] = (s[i] == s[i + 1])
        
        for length in range(3, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                is_palindrome[i][j] = (s[i] == s[j]) and is_palindrome[i + 1][j - 1]
        
        # DP for minimum cuts
        cuts = [0] * n
        
        for i in range(n):
            if is_palindrome[0][i]:
                cuts[i] = 0
            else:
                cuts[i] = i  # Worst case: i cuts
                for j in range(i):
                    if is_palindrome[j + 1][i]:
                        cuts[i] = min(cuts[i], cuts[j] + 1)
        
        return cuts[n - 1]
    
    # Test variants
    test_strings = [
        "abc",
        "aaa", 
        "racecar",
        "abccba",
        "abcdef",
        "aba",
        "abab"
    ]
    
    print("Palindromic Substrings Variants Analysis:")
    print("=" * 60)
    
    for s in test_strings:
        print(f"\nString: '{s}'")
        
        # Count palindromic substrings
        substring_count = count_substrings_expand_around_centers(s)
        print(f"  Palindromic substrings: {substring_count}")
        
        # Longest palindromic substring
        longest = longest_palindromic_substring(s)
        print(f"  Longest palindromic substring: '{longest}' (length: {len(longest)})")
        
        # Count palindromic subsequences (for small strings)
        if len(s) <= 6:
            subsequence_count = count_palindromic_subsequences(s)
            print(f"  Palindromic subsequences: {subsequence_count}")
        
        # Minimum cuts for palindrome partitioning
        min_cuts = min_cuts_for_palindrome_partitioning(s)
        print(f"  Min cuts for palindrome partition: {min_cuts}")
        
        # Count breakdown by length
        total, breakdown = count_substrings_by_length(s)
        print(f"  Count by length: {dict(breakdown)}")


# Test cases
def test_count_substrings():
    """Test all implementations with various inputs"""
    test_cases = [
        ("abc", 3),
        ("aaa", 6),
        ("a", 1),
        ("aa", 3),
        ("aba", 4),
        ("abccba", 9),
        ("racecar", 10),
        ("abcdef", 6),
        ("abab", 6),
        ("abcba", 7),
        ("", 0),
        ("abcdeffedcba", 17)
    ]
    
    print("Testing Palindromic Substrings Solutions:")
    print("=" * 70)
    
    for i, (s, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"s = '{s}'")
        print(f"Expected: {expected}")
        
        if s:  # Skip empty string for some methods
            brute_force = count_substrings_brute_force(s)
            expand_centers = count_substrings_expand_around_centers(s)
            dp_result = count_substrings_dp(s)
            space_opt = count_substrings_space_optimized(s)
            
            print(f"Brute Force:      {brute_force:>3} {'✓' if brute_force == expected else '✗'}")
            print(f"Expand Centers:   {expand_centers:>3} {'✓' if expand_centers == expected else '✗'}")
            print(f"DP (2D):          {dp_result:>3} {'✓' if dp_result == expected else '✗'}")
            print(f"Space Optimized:  {space_opt:>3} {'✓' if space_opt == expected else '✗'}")
            
            # Test Manacher variant for non-empty strings
            if len(s) <= 20:  # Skip for very long strings
                try:
                    manacher = count_substrings_manacher_variant(s)
                    print(f"Manacher Variant: {manacher:>3} {'✓' if manacher == expected else '✗'}")
                except:
                    print(f"Manacher Variant: Error")
        else:
            print(f"Empty string case: {expected}")
        
        # Show palindromes for small cases
        if s and len(s) <= 8:
            count, palindromes = count_substrings_with_details(s)
            print(f"Palindromes: {palindromes}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    count_substrings_analysis("aaa")
    
    # Variants demonstration
    print(f"\n" + "=" * 70)
    palindromic_substrings_variants()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. CONTIGUOUS SUBSTRINGS: Must be continuous, unlike subsequences")
    print("2. EXPAND AROUND CENTERS: Efficient O(n²) approach")
    print("3. TWO CENTER TYPES: Odd length (single char) and even length (between chars)")
    print("4. DP RELATION: dp[i][j] = (s[i] == s[j]) && dp[i+1][j-1]")
    print("5. MANACHER'S ALGORITHM: Can achieve O(n) time complexity")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Text Processing: Finding palindromic patterns")
    print("• Bioinformatics: DNA palindrome analysis")
    print("• Data Compression: Exploiting palindromic redundancy")
    print("• String Algorithms: Pattern recognition")
    print("• Algorithm Design: Foundation for palindrome problems")


if __name__ == "__main__":
    test_count_substrings()


"""
PALINDROMIC SUBSTRINGS - EFFICIENT PALINDROME DETECTION:
========================================================

This problem showcases efficient algorithms for palindrome detection:
- Demonstrates the power of "expand around centers" technique
- Foundation for many palindrome-related algorithms
- Shows progression from O(n³) brute force to O(n²) optimal to O(n) advanced
- Critical for understanding palindrome pattern recognition

KEY INSIGHTS:
============
1. **CONTIGUOUS REQUIREMENT**: Substrings must be continuous (vs subsequences)
2. **CENTER EXPANSION**: Each position can be center of odd/even length palindromes
3. **TWO CENTER TYPES**: Single character centers and between-character centers
4. **OPTIMAL SUBSTRUCTURE**: Large palindromes contain smaller palindromes

ALGORITHM APPROACHES:
====================

1. **Brute Force**: O(n³) - Check all O(n²) substrings in O(n) time each
2. **Expand Around Centers**: O(n²) - Try each of 2n-1 centers, expand in O(n)
3. **Dynamic Programming**: O(n²) - Build palindrome table bottom-up
4. **Manacher's Algorithm**: O(n) - Advanced linear time algorithm

EXPAND AROUND CENTERS TECHNIQUE:
===============================
Most intuitive optimal approach:
```
for each possible center:
    expand left and right while characters match
    count all palindromes found during expansion
```

Two types of centers:
- **Odd length**: Center at character i
- **Even length**: Center between characters i and i+1

DP RECURRENCE:
=============
```
dp[i][j] = True if s[i:j+1] is palindrome

Base cases:
- dp[i][i] = True (single characters)
- dp[i][i+1] = (s[i] == s[i+1]) (length 2)

Recurrence:
- dp[i][j] = (s[i] == s[j]) && dp[i+1][j-1]
```

SPACE OPTIMIZATION:
==================
Can reduce from O(n²) to O(n) by processing diagonally:
- Only need previous diagonal values
- Process by increasing substring length

MANACHER'S ALGORITHM:
====================
Advanced O(n) approach:
- Preprocess string to handle even/odd lengths uniformly
- Use previously computed information to avoid redundant work
- Linear time complexity but more complex implementation

APPLICATIONS:
============
- **Text Analysis**: Finding palindromic patterns in documents
- **Bioinformatics**: DNA palindrome detection
- **Data Compression**: Exploiting palindromic redundancy
- **String Algorithms**: Foundation for palindrome partitioning
- **Pattern Recognition**: Symmetric pattern detection

RELATED PROBLEMS:
================
- **Longest Palindromic Substring**: Find single longest palindrome
- **Palindrome Partitioning**: Partition string into palindromes
- **Palindromic Subsequences**: Non-contiguous palindromes
- **Valid Palindrome**: Check if string can become palindrome

COMPLEXITY COMPARISON:
=====================
- **Brute Force**: O(n³) time, O(1) space
- **Expand Centers**: O(n²) time, O(1) space  
- **DP**: O(n²) time, O(n²) space
- **Space Optimized DP**: O(n²) time, O(n) space
- **Manacher's**: O(n) time, O(n) space

The expand around centers approach is often preferred for its simplicity
and optimal practical performance while maintaining O(1) space usage.
"""
