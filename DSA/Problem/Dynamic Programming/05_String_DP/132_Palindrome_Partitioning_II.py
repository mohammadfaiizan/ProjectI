"""
LeetCode 132: Palindrome Partitioning II
Difficulty: Hard
Category: String DP

PROBLEM DESCRIPTION:
===================
Given a string s, partition s such that every substring of the partition is a palindrome.
Return the minimum cuts needed for a palindrome partitioning of s.

Example 1:
Input: s = "aab"
Output: 1
Explanation: The palindrome partitioning ["aa","b"] could be produced using 1 cut.

Example 2:
Input: s = "racecar"
Output: 0
Explanation: s is already a palindrome, so no cuts are needed.

Example 3:
Input: s = "abcde"
Output: 4
Explanation: The palindrome partitioning ["a","b","c","d","e"] could be produced using 4 cuts.

Constraints:
- 1 <= s.length <= 2000
- s consists of lowercase English letters only.
"""

def min_cut_brute_force(s):
    """
    BRUTE FORCE APPROACH:
    ====================
    Try all possible partitions and find minimum cuts.
    
    Time Complexity: O(2^n * n) - exponential partitions, palindrome checks
    Space Complexity: O(n) - recursion depth
    """
    def is_palindrome(string):
        return string == string[::-1]
    
    def backtrack(start):
        if start >= len(s):
            return -1  # No cuts needed for empty string
        
        min_cuts = float('inf')
        
        for end in range(start + 1, len(s) + 1):
            substring = s[start:end]
            if is_palindrome(substring):
                remaining_cuts = backtrack(end)
                if remaining_cuts != float('inf'):
                    min_cuts = min(min_cuts, remaining_cuts + (1 if end < len(s) else 0))
        
        return min_cuts
    
    result = backtrack(0)
    return result if result != float('inf') else 0


def min_cut_memoization(s):
    """
    MEMOIZATION APPROACH:
    ====================
    Use memoization to avoid recomputing same subproblems.
    
    Time Complexity: O(n^3) - O(n^2) states, O(n) to check palindrome
    Space Complexity: O(n^2) - memoization table
    """
    def is_palindrome(string):
        return string == string[::-1]
    
    memo = {}
    
    def dp(start):
        if start >= len(s):
            return -1
        
        if start in memo:
            return memo[start]
        
        min_cuts = float('inf')
        
        for end in range(start + 1, len(s) + 1):
            substring = s[start:end]
            if is_palindrome(substring):
                remaining_cuts = dp(end)
                if remaining_cuts != float('inf'):
                    min_cuts = min(min_cuts, remaining_cuts + (1 if end < len(s) else 0))
        
        memo[start] = min_cuts
        return min_cuts
    
    result = dp(0)
    return result if result != float('inf') else 0


def min_cut_dp_basic(s):
    """
    BASIC DP APPROACH:
    =================
    Use DP with palindrome checking.
    
    Time Complexity: O(n^3) - O(n^2) DP, O(n) palindrome check
    Space Complexity: O(n) - DP array
    """
    def is_palindrome(start, end):
        while start < end:
            if s[start] != s[end]:
                return False
            start += 1
            end -= 1
        return True
    
    n = len(s)
    
    # dp[i] = minimum cuts needed for s[0:i]
    dp = [float('inf')] * (n + 1)
    dp[0] = -1  # No cuts needed for empty string
    
    for i in range(1, n + 1):
        for j in range(i):
            if is_palindrome(j, i - 1):
                dp[i] = min(dp[i], dp[j] + 1)
    
    return dp[n]


def min_cut_dp_optimized(s):
    """
    OPTIMIZED DP APPROACH:
    =====================
    Precompute palindrome table and use DP.
    
    Time Complexity: O(n^2) - palindrome table + DP
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
    
    # DP for minimum cuts
    dp = [float('inf')] * n
    
    for i in range(n):
        if is_palindrome[0][i]:
            dp[i] = 0  # Entire prefix is palindrome
        else:
            for j in range(i):
                if is_palindrome[j + 1][i]:
                    dp[i] = min(dp[i], dp[j] + 1)
    
    return dp[n - 1]


def min_cut_expand_around_centers(s):
    """
    EXPAND AROUND CENTERS APPROACH:
    ==============================
    Use expand around centers to find palindromes efficiently.
    
    Time Complexity: O(n^2) - expand around centers + DP
    Space Complexity: O(n) - DP array only
    """
    n = len(s)
    
    # Initialize cuts array
    cuts = list(range(-1, n))  # cuts[i] = i-1 initially (worst case)
    
    def expand_around_center(left, right):
        while left >= 0 and right < n and s[left] == s[right]:
            if left == 0:
                cuts[right] = 0  # Entire prefix is palindrome
            else:
                cuts[right] = min(cuts[right], cuts[left - 1] + 1)
            left -= 1
            right += 1
    
    for i in range(n):
        # Odd length palindromes
        expand_around_center(i, i)
        
        # Even length palindromes
        expand_around_center(i, i + 1)
    
    return cuts[n - 1]


def min_cut_manacher_variant(s):
    """
    MANACHER'S ALGORITHM VARIANT:
    ============================
    Use Manacher's algorithm concept for linear palindrome detection.
    
    Time Complexity: O(n^2) - practical optimization
    Space Complexity: O(n) - optimized space usage
    """
    n = len(s)
    
    # Preprocess string for Manacher's algorithm
    processed = '#'.join('^{}$'.format(s))
    m = len(processed)
    
    # Manacher's array
    P = [0] * m
    center = right = 0
    
    # Build Manacher's array
    for i in range(1, m - 1):
        mirror = 2 * center - i
        
        if i < right:
            P[i] = min(right - i, P[mirror])
        
        # Try to expand palindrome centered at i
        while processed[i + P[i] + 1] == processed[i - P[i] - 1]:
            P[i] += 1
        
        # If palindrome centered at i extends past right, adjust center and right
        if i + P[i] > right:
            center, right = i, i + P[i]
    
    # Convert Manacher's result to original string indices and compute cuts
    cuts = list(range(-1, n))
    
    for i in range(1, m - 1):
        if P[i] > 0:
            # Convert processed index to original indices
            start = (i - P[i]) // 2
            end = (i + P[i]) // 2 - 1
            
            if start == 0:
                cuts[end] = 0
            else:
                cuts[end] = min(cuts[end], cuts[start - 1] + 1)
    
    return cuts[n - 1]


def min_cut_with_partition(s):
    """
    FIND MINIMUM CUTS AND ACTUAL PARTITION:
    =======================================
    Return minimum cuts and one optimal partition.
    
    Time Complexity: O(n^2) - DP + partition reconstruction
    Space Complexity: O(n^2) - palindrome table + tracking
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
    
    # DP with parent tracking
    dp = [float('inf')] * n
    parent = [-1] * n
    
    for i in range(n):
        if is_palindrome[0][i]:
            dp[i] = 0
            parent[i] = -1  # No previous cut
        else:
            for j in range(i):
                if is_palindrome[j + 1][i] and dp[j] + 1 < dp[i]:
                    dp[i] = dp[j] + 1
                    parent[i] = j
    
    # Reconstruct partition
    partition = []
    i = n - 1
    
    while i >= 0:
        if parent[i] == -1:
            partition.append(s[0:i + 1])
            break
        else:
            partition.append(s[parent[i] + 1:i + 1])
            i = parent[i]
    
    partition.reverse()
    
    return dp[n - 1], partition


def min_cut_analysis(s):
    """
    DETAILED ANALYSIS:
    =================
    Show step-by-step computation and cut analysis.
    """
    n = len(s)
    
    print(f"Finding minimum cuts for palindrome partitioning:")
    print(f"  s = '{s}' (length {n})")
    
    # Build palindrome table with explanation
    is_palindrome = [[False] * n for _ in range(n)]
    
    print(f"\nBuilding palindrome table:")
    
    # Single characters
    for i in range(n):
        is_palindrome[i][i] = True
    
    # Pairs
    for i in range(n - 1):
        if s[i] == s[i + 1]:
            is_palindrome[i][i + 1] = True
            print(f"  Found palindrome: '{s[i:i+2]}' at [{i},{i+1}]")
    
    # Longer substrings
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j] and is_palindrome[i + 1][j - 1]:
                is_palindrome[i][j] = True
                print(f"  Found palindrome: '{s[i:j+1]}' at [{i},{j}]")
    
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
    
    # DP computation
    print(f"\nDP computation for minimum cuts:")
    dp = [float('inf')] * n
    
    for i in range(n):
        print(f"  Computing dp[{i}] for s[0:{i+1}] = '{s[:i+1]}':")
        
        if is_palindrome[0][i]:
            dp[i] = 0
            print(f"    Entire substring is palindrome → dp[{i}] = 0")
        else:
            for j in range(i):
                if is_palindrome[j + 1][i]:
                    new_cuts = dp[j] + 1
                    if new_cuts < dp[i]:
                        dp[i] = new_cuts
                        print(f"    Cut after position {j}: dp[{j}] + 1 = {dp[j]} + 1 = {new_cuts}")
            print(f"    Final: dp[{i}] = {dp[i]}")
    
    print(f"\nDP array: {dp}")
    print(f"Minimum cuts needed: {dp[n - 1]}")
    
    # Show optimal partition
    min_cuts, partition = min_cut_with_partition(s)
    print(f"One optimal partition: {partition}")
    print(f"Verification: {' | '.join(partition)} with {len(partition) - 1} cuts")
    
    return min_cuts


def min_cut_variants():
    """
    MINIMUM CUT VARIANTS:
    ====================
    Test different scenarios and related problems.
    """
    
    def min_cut_all_partitions(s):
        """Find all partitions with minimum cuts"""
        min_cuts = min_cut_dp_optimized(s)
        
        # Find all partitions with exactly min_cuts cuts
        n = len(s)
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
        
        # Find all optimal partitions
        all_partitions = []
        
        def backtrack(start, current_partition, cuts_used):
            if start == n:
                if cuts_used == min_cuts:
                    all_partitions.append(current_partition[:])
                return
            
            if cuts_used > min_cuts:
                return  # Pruning
            
            for end in range(start, n):
                if is_palindrome[start][end]:
                    current_partition.append(s[start:end + 1])
                    new_cuts = cuts_used + (1 if end < n - 1 else 0)
                    backtrack(end + 1, current_partition, new_cuts)
                    current_partition.pop()
        
        backtrack(0, [], 0)
        return all_partitions
    
    def max_palindrome_partition(s):
        """Find partition that maximizes total palindrome length"""
        # This is actually just the original string length
        # But we can find the partition with minimum number of parts
        n = len(s)
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
        
        # DP for minimum parts (maximum palindrome length per part)
        dp = [float('inf')] * n
        partition_info = [None] * n
        
        for i in range(n):
            for j in range(i + 1):
                if is_palindrome[j][i]:
                    parts = 1 if j == 0 else dp[j - 1] + 1
                    if parts < dp[i]:
                        dp[i] = parts
                        partition_info[i] = j
        
        # Reconstruct partition
        partition = []
        i = n - 1
        while i >= 0:
            start = partition_info[i]
            partition.append(s[start:i + 1])
            i = start - 1
        
        partition.reverse()
        return dp[n - 1], partition
    
    # Test variants
    test_strings = [
        "aab",
        "racecar", 
        "abcde",
        "abccba",
        "aaabaaaa",
        "abcba"
    ]
    
    print("Minimum Cut Variants Analysis:")
    print("=" * 60)
    
    for s in test_strings:
        print(f"\nString: '{s}'")
        
        min_cuts = min_cut_dp_optimized(s)
        print(f"  Minimum cuts: {min_cuts}")
        
        _, optimal_partition = min_cut_with_partition(s)
        print(f"  Optimal partition: {optimal_partition}")
        
        if len(s) <= 8:  # Only for small strings
            all_optimal = min_cut_all_partitions(s)
            print(f"  All optimal partitions: {len(all_optimal)}")
            if len(all_optimal) <= 3:
                for i, partition in enumerate(all_optimal):
                    print(f"    {i+1}: {partition}")
        
        min_parts, max_partition = max_palindrome_partition(s)
        print(f"  Min parts (max palindromes): {min_parts}, partition: {max_partition}")


# Test cases
def test_min_cut():
    """Test all implementations with various inputs"""
    test_cases = [
        ("aab", 1),
        ("racecar", 0),
        ("abcde", 4),
        ("a", 0),
        ("aa", 0),
        ("aba", 0),
        ("abcba", 0),
        ("abccba", 0),
        ("abcdef", 5),
        ("aabaa", 1),
        ("aaabaaaa", 2),
        ("abcbm", 3)
    ]
    
    print("Testing Palindrome Partitioning II Solutions:")
    print("=" * 70)
    
    for i, (s, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"s = '{s}'")
        print(f"Expected: {expected}")
        
        # Test approaches (skip expensive ones for long strings)
        if len(s) <= 8:
            try:
                brute_force = min_cut_brute_force(s)
                print(f"Brute Force:      {brute_force:>3} {'✓' if brute_force == expected else '✗'}")
            except:
                print(f"Brute Force:      Timeout")
        
        if len(s) <= 12:
            memo = min_cut_memoization(s)
            print(f"Memoization:      {memo:>3} {'✓' if memo == expected else '✗'}")
        
        dp_basic = min_cut_dp_basic(s)
        dp_optimized = min_cut_dp_optimized(s)
        expand_centers = min_cut_expand_around_centers(s)
        
        print(f"DP Basic:         {dp_basic:>3} {'✓' if dp_basic == expected else '✗'}")
        print(f"DP Optimized:     {dp_optimized:>3} {'✓' if dp_optimized == expected else '✗'}")
        print(f"Expand Centers:   {expand_centers:>3} {'✓' if expand_centers == expected else '✗'}")
        
        # Show partition for small cases
        if len(s) <= 10:
            min_cuts, partition = min_cut_with_partition(s)
            print(f"Partition: {partition}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    min_cut_analysis("aab")
    
    # Variants demonstration
    print(f"\n" + "=" * 70)
    min_cut_variants()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. OPTIMIZATION PROBLEM: Find minimum cuts, not all partitions")
    print("2. DP STRUCTURE: dp[i] = min cuts for prefix s[0:i+1]")
    print("3. PALINDROME PRECOMPUTATION: O(n²) table enables O(1) checks")
    print("4. OPTIMAL SUBSTRUCTURE: Optimal cuts contain optimal sub-cuts")
    print("5. EXPAND AROUND CENTERS: Can optimize palindrome detection")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Text Processing: Optimal palindromic segmentation")
    print("• Data Compression: Minimize palindromic partitions")
    print("• Algorithm Design: Cut optimization problems")
    print("• String Algorithms: Foundation for partition optimization")
    print("• Bioinformatics: DNA sequence palindrome optimization")


if __name__ == "__main__":
    test_min_cut()


"""
PALINDROME PARTITIONING II - OPTIMIZATION OVER EXPONENTIAL SPACE:
=================================================================

This problem transforms the exhaustive enumeration of Palindrome Partitioning I
into an elegant optimization problem:
- Instead of generating all partitions, find the minimum cuts
- Demonstrates how DP can optimize over exponential solution spaces
- Shows the power of palindrome precomputation techniques
- Foundation for many cut-minimization problems

KEY INSIGHTS:
============
1. **OPTIMIZATION vs ENUMERATION**: Find minimum cuts, not all partitions
2. **DP STATE**: dp[i] = minimum cuts needed for prefix s[0:i+1]
3. **PALINDROME PRECOMPUTATION**: O(n²) table saves repeated work
4. **OPTIMAL SUBSTRUCTURE**: Optimal partition contains optimal sub-partitions

RECURRENCE RELATION:
===================
```
dp[i] = min(dp[j] + 1) for all j where s[j+1:i+1] is palindrome

Special case: if s[0:i+1] is entirely palindrome, then dp[i] = 0
```

ALGORITHM APPROACHES:
====================

1. **Basic DP**: O(n³) time, O(n) space
   - Check palindromes on-the-fly during DP
   - Simple but not optimal

2. **Optimized DP**: O(n²) time, O(n²) space
   - Precompute palindrome table in O(n²)
   - Use table for O(1) palindrome checks
   - Optimal approach for this problem

3. **Expand Around Centers**: O(n²) time, O(n) space
   - Use expand-around-centers for palindrome detection
   - Update cuts array during expansion
   - Most space-efficient optimal solution

4. **Manacher Variant**: O(n²) time, O(n) space
   - Apply Manacher's algorithm concepts
   - Advanced optimization technique

PALINDROME TABLE OPTIMIZATION:
=============================
Critical for achieving O(n²) complexity:
```
# Build palindrome table bottom-up
for i in range(n):
    is_palindrome[i][i] = True

for i in range(n-1):
    is_palindrome[i][i+1] = (s[i] == s[i+1])

for length in range(3, n+1):
    for i in range(n-length+1):
        j = i + length - 1
        is_palindrome[i][j] = (s[i] == s[j]) and is_palindrome[i+1][j-1]
```

EXPAND AROUND CENTERS TECHNIQUE:
===============================
Space-efficient alternative:
```
def expand_around_center(left, right):
    while left >= 0 and right < n and s[left] == s[right]:
        if left == 0:
            cuts[right] = 0  # Entire prefix is palindrome
        else:
            cuts[right] = min(cuts[right], cuts[left-1] + 1)
        left -= 1
        right += 1

for i in range(n):
    expand_around_center(i, i)      # Odd length
    expand_around_center(i, i+1)    # Even length
```

DP STATE TRANSITION:
===================
For each position i, consider all possible last palindromes:
```
for i in range(n):
    # Check if entire prefix s[0:i+1] is palindrome
    if is_palindrome[0][i]:
        dp[i] = 0
    else:
        # Try all possible positions for last cut
        for j in range(i):
            if is_palindrome[j+1][i]:
                dp[i] = min(dp[i], dp[j] + 1)
```

SOLUTION RECONSTRUCTION:
=======================
To find actual optimal partition:
```
# Track parent information during DP
parent[i] = j if dp[i] was updated using dp[j] + 1

# Reconstruct partition
partition = []
i = n - 1
while i >= 0:
    if parent[i] == -1:
        partition.append(s[0:i+1])
        break
    else:
        partition.append(s[parent[i]+1:i+1])
        i = parent[i]
partition.reverse()
```

COMPLEXITY COMPARISON:
=====================
| Approach           | Time    | Space | Notes                    |
|--------------------|---------|-------|--------------------------|
| Brute Force        | O(2^n)  | O(n)  | Exponential enumeration  |
| Basic DP           | O(n³)   | O(n)  | Palindrome checks on-fly |
| Optimized DP       | O(n²)   | O(n²) | Precomputed palindromes  |
| Expand Centers     | O(n²)   | O(n)  | Space-efficient optimal  |

APPLICATIONS:
============
- **Text Processing**: Optimal palindromic segmentation
- **Data Compression**: Minimize redundant palindromic structures
- **Algorithm Design**: Cut minimization in partition problems
- **Bioinformatics**: DNA sequence palindrome optimization
- **String Algorithms**: Foundation for partition optimization

RELATED PROBLEMS:
================
- **Palindrome Partitioning I (131)**: Generate all partitions
- **Word Break II (140)**: Similar partition structure
- **Minimum Cost Tree From Leaf Values (1130)**: Cut optimization
- **Burst Balloons (312)**: Interval DP with cuts

OPTIMIZATION TECHNIQUES:
=======================
1. **Precomputation**: Build palindrome table once
2. **Space-Time Tradeoffs**: Expand-around-centers saves space
3. **Early Termination**: Stop when optimal solution found
4. **Solution Reconstruction**: Track parent pointers

This problem beautifully demonstrates how dynamic programming can solve
optimization problems over exponentially large solution spaces efficiently,
reducing O(2^n) enumeration to O(n²) optimization.
"""
