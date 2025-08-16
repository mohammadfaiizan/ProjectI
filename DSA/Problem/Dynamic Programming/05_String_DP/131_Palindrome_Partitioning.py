"""
LeetCode 131: Palindrome Partitioning
Difficulty: Medium
Category: String DP

PROBLEM DESCRIPTION:
===================
Given a string s, partition s such that every substring of the partition is a palindrome. 
Return all possible palindrome partitioning of s.

Example 1:
Input: s = "aab"
Output: [["a","a","b"],["aa","b"]]

Example 2:
Input: s = "racecar"
Output: [["racecar"]]

Example 3:
Input: s = "abcba"
Output: [["a","b","c","b","a"],["a","bcb","a"],["abcba"]]

Constraints:
- 1 <= s.length <= 16
- s contains only lowercase English letters.
"""

def partition_backtrack(s):
    """
    BACKTRACKING APPROACH:
    =====================
    Generate all possible partitions and filter palindromic ones.
    
    Time Complexity: O(2^n * n) - 2^n partitions, O(n) to check each palindrome
    Space Complexity: O(n) - recursion depth + current partition
    """
    def is_palindrome(string):
        return string == string[::-1]
    
    def backtrack(start, current_partition):
        # Base case: reached end of string
        if start >= len(s):
            result.append(current_partition[:])
            return
        
        # Try all possible end positions
        for end in range(start + 1, len(s) + 1):
            substring = s[start:end]
            if is_palindrome(substring):
                current_partition.append(substring)
                backtrack(end, current_partition)
                current_partition.pop()
    
    result = []
    backtrack(0, [])
    return result


def partition_dp_precompute(s):
    """
    DP WITH PRECOMPUTED PALINDROMES:
    ===============================
    Precompute palindrome table, then use backtracking.
    
    Time Complexity: O(n^2 + 2^n) - O(n^2) precompute + O(2^n) backtrack
    Space Complexity: O(n^2) - palindrome table
    """
    n = len(s)
    
    # Precompute palindrome table
    is_palindrome = [[False] * n for _ in range(n)]
    
    # Single characters are palindromes
    for i in range(n):
        is_palindrome[i][i] = True
    
    # Check pairs
    for i in range(n - 1):
        is_palindrome[i][i + 1] = (s[i] == s[i + 1])
    
    # Check longer substrings
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            is_palindrome[i][j] = (s[i] == s[j]) and is_palindrome[i + 1][j - 1]
    
    def backtrack(start, current_partition):
        if start >= n:
            result.append(current_partition[:])
            return
        
        for end in range(start, n):
            if is_palindrome[start][end]:
                current_partition.append(s[start:end + 1])
                backtrack(end + 1, current_partition)
                current_partition.pop()
    
    result = []
    backtrack(0, [])
    return result


def partition_dp_memoization(s):
    """
    MEMOIZATION APPROACH:
    ====================
    Use memoization to cache partitions for substrings.
    
    Time Complexity: O(2^n * n) - worst case, but with pruning
    Space Complexity: O(2^n * n) - memoization storage
    """
    def is_palindrome(string):
        return string == string[::-1]
    
    memo = {}
    
    def get_partitions(substring):
        if substring in memo:
            return memo[substring]
        
        if not substring:
            return [[]]
        
        result = []
        for i in range(1, len(substring) + 1):
            prefix = substring[:i]
            if is_palindrome(prefix):
                suffix_partitions = get_partitions(substring[i:])
                for suffix_partition in suffix_partitions:
                    result.append([prefix] + suffix_partition)
        
        memo[substring] = result
        return result
    
    return get_partitions(s)


def partition_optimized(s):
    """
    OPTIMIZED APPROACH:
    ==================
    Combine precomputed palindromes with efficient backtracking.
    
    Time Complexity: O(2^n) - optimal for this problem
    Space Complexity: O(n^2) - palindrome table
    """
    n = len(s)
    
    # Precompute palindrome table using expand around centers
    is_palindrome = [[False] * n for _ in range(n)]
    
    # Expand around centers for odd length palindromes
    for center in range(n):
        left = right = center
        while left >= 0 and right < n and s[left] == s[right]:
            is_palindrome[left][right] = True
            left -= 1
            right += 1
    
    # Expand around centers for even length palindromes
    for center in range(n - 1):
        left, right = center, center + 1
        while left >= 0 and right < n and s[left] == s[right]:
            is_palindrome[left][right] = True
            left -= 1
            right += 1
    
    def backtrack(start, current_partition):
        if start == n:
            result.append(current_partition[:])
            return
        
        for end in range(start, n):
            if is_palindrome[start][end]:
                current_partition.append(s[start:end + 1])
                backtrack(end + 1, current_partition)
                current_partition.pop()
    
    result = []
    backtrack(0, [])
    return result


def partition_with_analysis(s):
    """
    PARTITION WITH DETAILED ANALYSIS:
    ================================
    Show the partitioning process step by step.
    
    Time Complexity: O(2^n) - generate all partitions
    Space Complexity: O(n^2) - palindrome table + recursion
    """
    n = len(s)
    
    print(f"Analyzing palindrome partitioning for: '{s}'")
    
    # Build palindrome table with explanation
    is_palindrome = [[False] * n for _ in range(n)]
    
    print(f"\nBuilding palindrome table:")
    
    # Single characters
    for i in range(n):
        is_palindrome[i][i] = True
        print(f"  '{s[i]}' at [{i},{i}] is palindrome")
    
    # Length 2
    for i in range(n - 1):
        if s[i] == s[i + 1]:
            is_palindrome[i][i + 1] = True
            print(f"  '{s[i:i+2]}' at [{i},{i+1}] is palindrome")
    
    # Length 3+
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j] and is_palindrome[i + 1][j - 1]:
                is_palindrome[i][j] = True
                print(f"  '{s[i:j+1]}' at [{i},{j}] is palindrome")
    
    # Show palindrome table
    print(f"\nPalindrome table:")
    print("   ", end="")
    for j in range(n):
        print(f"{j:4}", end="")
    print()
    
    for i in range(n):
        print(f"{i:2}: ", end="")
        for j in range(n):
            if i <= j:
                print("   T" if is_palindrome[i][j] else "   F", end="")
            else:
                print("   -", end="")
        print()
    
    # Generate partitions with tracing
    partitions = []
    
    def backtrack(start, current_partition, depth=0):
        indent = "  " * depth
        print(f"{indent}Partitioning from position {start}: current = {current_partition}")
        
        if start == n:
            partitions.append(current_partition[:])
            print(f"{indent}Complete partition: {current_partition}")
            return
        
        for end in range(start, n):
            substring = s[start:end + 1]
            if is_palindrome[start][end]:
                print(f"{indent}Found palindrome: '{substring}' at [{start},{end}]")
                current_partition.append(substring)
                backtrack(end + 1, current_partition, depth + 1)
                current_partition.pop()
            else:
                print(f"{indent}Not palindrome: '{substring}' at [{start},{end}]")
    
    print(f"\nGenerating partitions:")
    backtrack(0, [])
    
    print(f"\nAll palindrome partitions:")
    for i, partition in enumerate(partitions):
        print(f"  {i + 1}: {partition}")
    
    return partitions


def partition_count_only(s):
    """
    COUNT PARTITIONS ONLY:
    ======================
    Count number of possible palindrome partitions without generating them.
    
    Time Complexity: O(n^2) - DP computation
    Space Complexity: O(n^2) - palindrome table + DP array
    """
    n = len(s)
    
    # Precompute palindrome table
    is_palindrome = [[False] * n for _ in range(n)]
    
    for i in range(n):
        is_palindrome[i][i] = True
    
    for i in range(n - 1):
        is_palindrome[i][i + 1] = (s[i] == s[i + 1])
    
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            is_palindrome[i][j] = (s[i] == s[j]) and is_palindrome[i + 1][j - 1]
    
    # DP to count partitions
    dp = [0] * (n + 1)
    dp[0] = 1  # Empty string has one way to partition
    
    for i in range(1, n + 1):
        for j in range(i):
            if is_palindrome[j][i - 1]:
                dp[i] += dp[j]
    
    return dp[n]


def partition_min_cuts(s):
    """
    MINIMUM CUTS FOR PALINDROME PARTITIONING:
    =========================================
    Find minimum number of cuts needed for palindrome partition.
    
    Time Complexity: O(n^2) - DP computation
    Space Complexity: O(n^2) - palindrome table + cuts array
    """
    n = len(s)
    
    # Precompute palindrome table
    is_palindrome = [[False] * n for _ in range(n)]
    
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
            cuts[i] = 0  # Entire prefix is palindrome
        else:
            cuts[i] = i  # Worst case: i cuts
            for j in range(i):
                if is_palindrome[j + 1][i]:
                    cuts[i] = min(cuts[i], cuts[j] + 1)
    
    return cuts[n - 1]


# Test cases
def test_palindrome_partitioning():
    """Test all implementations with various inputs"""
    test_cases = [
        ("aab", [["a","a","b"],["aa","b"]]),
        ("racecar", [["racecar"]]),
        ("abcba", [["a","b","c","b","a"],["a","bcb","a"],["abcba"]]),
        ("a", [["a"]]),
        ("aa", [["a","a"],["aa"]]),
        ("aba", [["a","b","a"],["aba"]]),
        ("abccba", [["a","b","c","c","b","a"],["a","b","cc","b","a"],["a","bccb","a"],["abccba"]]),
        ("abcde", [["a","b","c","d","e"]]),
        ("", [[]]),
        ("abab", [["a","b","a","b"]])
    ]
    
    print("Testing Palindrome Partitioning Solutions:")
    print("=" * 70)
    
    for i, (s, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"s = '{s}'")
        print(f"Expected partitions: {len(expected)}")
        
        if s:  # Skip empty string for some methods
            # Test different approaches
            backtrack_result = partition_backtrack(s)
            dp_precompute_result = partition_dp_precompute(s)
            optimized_result = partition_optimized(s)
            
            # Sort results for comparison (order may vary)
            backtrack_sorted = sorted([sorted(p) for p in backtrack_result])
            expected_sorted = sorted([sorted(p) for p in expected])
            
            print(f"Backtracking:     {len(backtrack_result):>3} {'✓' if len(backtrack_result) == len(expected) else '✗'}")
            print(f"DP Precompute:    {len(dp_precompute_result):>3} {'✓' if len(dp_precompute_result) == len(expected) else '✗'}")
            print(f"Optimized:        {len(optimized_result):>3} {'✓' if len(optimized_result) == len(expected) else '✗'}")
            
            # Show partitions for small cases
            if len(s) <= 6:
                print(f"Partitions: {backtrack_result}")
            
            # Test memoization for reasonable sizes
            if len(s) <= 8:
                memo_result = partition_dp_memoization(s)
                print(f"Memoization:      {len(memo_result):>3} {'✓' if len(memo_result) == len(expected) else '✗'}")
            
            # Count and min cuts
            if len(s) <= 10:
                count = partition_count_only(s)
                min_cuts = partition_min_cuts(s)
                print(f"Count only: {count}, Min cuts: {min_cuts}")
        else:
            print("Empty string case")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    partition_with_analysis("aab")
    
    print("\n" + "=" * 70)
    print("Performance Analysis:")
    
    # Test performance for different string lengths
    test_strings = ["a", "aa", "aba", "abba", "abcba", "abccba"]
    
    for s in test_strings:
        count = partition_count_only(s)
        min_cuts = partition_min_cuts(s)
        print(f"  '{s}': {count} partitions, {min_cuts} min cuts")
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. BACKTRACKING: Generate all possible partitions")
    print("2. PALINDROME PRECOMPUTATION: O(n²) preprocessing saves time")
    print("3. EXPONENTIAL PARTITIONS: 2^(n-1) possible partitions in worst case")
    print("4. COUNTING OPTIMIZATION: DP can count without generating")
    print("5. MIN CUTS VARIANT: Related optimization problem")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Text Processing: Splitting into palindromic segments")
    print("• Data Compression: Exploiting palindromic patterns")
    print("• Bioinformatics: DNA palindrome analysis")
    print("• Algorithm Design: Partition-based optimization")
    print("• String Algorithms: Foundation for palindrome problems")


if __name__ == "__main__":
    test_palindrome_partitioning()


"""
PALINDROME PARTITIONING - EXHAUSTIVE ENUMERATION WITH CONSTRAINTS:
==================================================================

This problem demonstrates exhaustive enumeration with palindrome constraints:
- Generate all possible string partitions
- Filter for palindromic substrings only
- Shows the power of precomputation in constraint checking
- Foundation for partition-based optimization problems

KEY INSIGHTS:
============
1. **EXPONENTIAL PARTITIONS**: Up to 2^(n-1) possible partitions
2. **CONSTRAINT CHECKING**: Palindrome validation for each substring
3. **PRECOMPUTATION OPTIMIZATION**: Build palindrome table once
4. **BACKTRACKING STRUCTURE**: Natural recursive partition generation

ALGORITHM APPROACHES:
====================

1. **Pure Backtracking**: O(2^n × n) time
   - Generate partitions and check palindromes on-the-fly
   - Simple but inefficient for palindrome checking

2. **DP Precomputed**: O(n² + 2^n) time, O(n²) space
   - Precompute palindrome table in O(n²)
   - Use table for O(1) palindrome checks
   - Optimal approach for this problem

3. **Memoization**: O(2^n × n) time, O(2^n × n) space
   - Cache partition results for substrings
   - Helps with overlapping subproblems

4. **Counting Only**: O(n²) time, O(n²) space
   - Count partitions without generating them
   - Much faster when only count is needed

PALINDROME TABLE CONSTRUCTION:
=============================
Critical optimization - build once, use many times:
```
# Single characters
for i in range(n):
    is_palindrome[i][i] = True

# Length 2
for i in range(n-1):
    is_palindrome[i][i+1] = (s[i] == s[i+1])

# Length 3+
for length in range(3, n+1):
    for i in range(n-length+1):
        j = i + length - 1
        is_palindrome[i][j] = (s[i] == s[j]) and is_palindrome[i+1][j-1]
```

BACKTRACKING STRUCTURE:
======================
Natural recursive enumeration:
```
def backtrack(start, current_partition):
    if start == len(s):
        result.append(current_partition[:])
        return
    
    for end in range(start, len(s)):
        if is_palindrome[start][end]:
            current_partition.append(s[start:end+1])
            backtrack(end + 1, current_partition)
            current_partition.pop()
```

COUNTING OPTIMIZATION:
=====================
When only count is needed, use DP:
```
dp[i] = number of ways to partition s[0:i]

dp[i] = sum(dp[j] for all j where s[j:i] is palindrome)
```

SPACE OPTIMIZATION:
==================
For palindrome table construction, can use expand-around-centers:
```
# More space-efficient palindrome detection
for center in range(n):
    # Odd length palindromes
    left = right = center
    while left >= 0 and right < n and s[left] == s[right]:
        is_palindrome[left][right] = True
        left -= 1
        right += 1
    
    # Even length palindromes  
    left, right = center, center + 1
    while left >= 0 and right < n and s[left] == s[right]:
        is_palindrome[left][right] = True
        left -= 1
        right += 1
```

COMPLEXITY ANALYSIS:
===================
- **Palindrome Table**: O(n²) time, O(n²) space
- **Partition Generation**: O(2^n) partitions in worst case
- **Total Time**: O(n² + 2^n) - dominated by partition generation
- **Total Space**: O(n²) for table + O(n) for recursion

APPLICATIONS:
============
- **Text Processing**: Splitting text into palindromic segments
- **Data Compression**: Exploiting palindromic redundancy patterns
- **Bioinformatics**: DNA palindrome sequence analysis
- **Algorithm Design**: Foundation for partition optimization
- **Pattern Recognition**: Symmetric structure identification

RELATED PROBLEMS:
================
- **Palindrome Partitioning II (132)**: Minimum cuts version
- **Palindromic Substrings (647)**: Count all palindromic substrings  
- **Longest Palindromic Subsequence (516)**: Optimization variant
- **Word Break (139)**: Similar partition structure with dictionary

VARIANTS:
========
- **Minimum Cuts**: Find minimum partitions needed
- **Maximum Palindrome Length**: Optimize for longest palindromes
- **Weighted Partitioning**: Different costs for different partitions
- **K-Partitions**: Exactly k palindromic parts

OPTIMIZATION TECHNIQUES:
=======================
1. **Precomputation**: Build palindrome table once
2. **Early Pruning**: Stop when no valid partitions possible
3. **Memoization**: Cache results for repeated subproblems
4. **Space-Time Tradeoffs**: Count vs generate tradeoffs

This problem excellently demonstrates how precomputation can transform
an O(2^n × n²) algorithm into O(n² + 2^n), making it practical for
reasonable input sizes while teaching fundamental partition techniques.
"""
