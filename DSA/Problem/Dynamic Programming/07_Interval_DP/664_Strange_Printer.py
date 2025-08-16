"""
LeetCode 664: Strange Printer
Difficulty: Hard
Category: Interval DP - Printing Optimization

PROBLEM DESCRIPTION:
===================
There is a strange printer with the following two special properties:
1. The printer can only print a sequence of the same character each time.
2. At each turn, the printer can print new characters starting from and ending at any place and will cover the original existing characters.

Given a string s, return the minimum number of turns the printer needs to print it.

Example 1:
Input: s = "aaabbb"
Output: 2
Explanation: Print "aaa" first and then print "bbb".

Example 2:
Input: s = "aba"
Output: 2
Explanation: Print "aaa" first and then print "b" at the second position.

Example 3:
Input: s = "abcabc"
Output: 5

Constraints:
- 1 <= s.length <= 100
- s consists of lowercase English letters.
"""

def strange_printer_recursive(s):
    """
    RECURSIVE APPROACH:
    ==================
    Try all possible ways to print the string.
    
    Time Complexity: O(exponential) - many overlapping subproblems
    Space Complexity: O(n) - recursion depth
    """
    def min_turns(start, end):
        if start > end:
            return 0
        if start == end:
            return 1
        
        # Option 1: Print s[start] for the entire range, then fix overlaps
        result = 1 + min_turns(start + 1, end)
        
        # Option 2: Find other positions with same character and merge
        for k in range(start + 1, end + 1):
            if s[k] == s[start]:
                # Split at position k and merge with start
                result = min(result, min_turns(start, k - 1) + min_turns(k + 1, end))
        
        return result
    
    return min_turns(0, len(s) - 1)


def strange_printer_memoization(s):
    """
    MEMOIZATION APPROACH:
    ====================
    Cache results for different substrings.
    
    Time Complexity: O(n^3) - n^2 states, O(n) transitions each
    Space Complexity: O(n^2) - memo table
    """
    n = len(s)
    memo = {}
    
    def min_turns(start, end):
        if start > end:
            return 0
        if start == end:
            return 1
        
        if (start, end) in memo:
            return memo[(start, end)]
        
        # Default: print s[start] across range, then fix rest
        result = 1 + min_turns(start + 1, end)
        
        # Try merging with same characters
        for k in range(start + 1, end + 1):
            if s[k] == s[start]:
                result = min(result, min_turns(start + 1, k) + min_turns(k + 1, end))
        
        memo[(start, end)] = result
        return result
    
    return min_turns(0, n - 1)


def strange_printer_interval_dp(s):
    """
    INTERVAL DP APPROACH:
    ====================
    Bottom-up DP processing intervals by length.
    
    Time Complexity: O(n^3) - three nested loops
    Space Complexity: O(n^2) - DP table
    """
    n = len(s)
    if n == 0:
        return 0
    
    # dp[i][j] = minimum turns to print s[i:j+1]
    dp = [[0] * n for _ in range(n)]
    
    # Base case: single characters
    for i in range(n):
        dp[i][i] = 1
    
    # Process intervals by length
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            
            # Default: print s[i] across entire range, then fix rest
            dp[i][j] = 1 + dp[i + 1][j]
            
            # Try merging s[i] with same characters at position k
            for k in range(i + 1, j + 1):
                if s[k] == s[i]:
                    # Merge s[i] with s[k], process middle separately
                    dp[i][j] = min(dp[i][j], dp[i + 1][k] + dp[k + 1][j])
    
    return dp[0][n - 1]


def strange_printer_optimized(s):
    """
    OPTIMIZED APPROACH:
    ==================
    Preprocess string to remove consecutive duplicates.
    
    Time Complexity: O(n^3) - same asymptotic, better constants
    Space Complexity: O(n^2) - DP table
    """
    if not s:
        return 0
    
    # Remove consecutive duplicates
    compressed = []
    for char in s:
        if not compressed or compressed[-1] != char:
            compressed.append(char)
    
    n = len(compressed)
    if n == 0:
        return 0
    
    dp = [[0] * n for _ in range(n)]
    
    # Base case
    for i in range(n):
        dp[i][i] = 1
    
    # Fill DP table
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            
            # Print compressed[i] across range, then fix overlaps
            dp[i][j] = 1 + dp[i + 1][j]
            
            # Try merging with same characters
            for k in range(i + 1, j + 1):
                if compressed[k] == compressed[i]:
                    dp[i][j] = min(dp[i][j], dp[i + 1][k] + dp[k + 1][j])
    
    return dp[0][n - 1]


def strange_printer_with_strategy(s):
    """
    TRACK PRINTING STRATEGY:
    =======================
    Return minimum turns and the actual printing strategy.
    
    Time Complexity: O(n^3) - DP computation + reconstruction
    Space Complexity: O(n^2) - DP table + strategy tracking
    """
    n = len(s)
    if n == 0:
        return 0, []
    
    dp = [[0] * n for _ in range(n)]
    choice = [[None] * n for _ in range(n)]  # Track optimal decisions
    
    # Base case
    for i in range(n):
        dp[i][i] = 1
        choice[i][i] = ('single', i)
    
    # Fill DP table
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            
            # Default strategy: print s[i] across range
            dp[i][j] = 1 + dp[i + 1][j]
            choice[i][j] = ('extend', i, j)
            
            # Try merging strategies
            for k in range(i + 1, j + 1):
                if s[k] == s[i]:
                    cost = dp[i + 1][k] + dp[k + 1][j]
                    if cost < dp[i][j]:
                        dp[i][j] = cost
                        choice[i][j] = ('merge', i, k)
    
    # Reconstruct strategy
    def build_strategy(start, end):
        if start > end:
            return []
        if start == end:
            return [('print', s[start], start, start)]
        
        decision = choice[start][end]
        strategy = []
        
        if decision[0] == 'extend':
            # Print s[start] across entire range
            char = decision[1]
            end_pos = decision[2]
            strategy.append(('print', s[char], start, end_pos))
            strategy.extend(build_strategy(start + 1, end))
        
        elif decision[0] == 'merge':
            # Merge s[start] with s[k]
            start_pos = decision[1]
            k = decision[2]
            strategy.extend(build_strategy(start + 1, k))
            strategy.extend(build_strategy(k + 1, end))
            # Print character across merged positions
            strategy.append(('print', s[start_pos], start_pos, k))
        
        return strategy
    
    strategy = build_strategy(0, n - 1)
    return dp[0][n - 1], strategy


def strange_printer_analysis(s):
    """
    DETAILED ANALYSIS:
    =================
    Show step-by-step DP computation and printing strategy.
    """
    print(f"Strange Printer Analysis:")
    print(f"String: '{s}'")
    print(f"Length: {len(s)}")
    
    n = len(s)
    if n == 0:
        print("Empty string, no printing needed.")
        return 0
    
    # Show character positions
    print(f"\nCharacter positions:")
    for i, char in enumerate(s):
        print(f"  {i}: '{char}'")
    
    # Show same character groupings
    from collections import defaultdict
    char_positions = defaultdict(list)
    for i, char in enumerate(s):
        char_positions[char].append(i)
    
    print(f"\nCharacter groupings:")
    for char, positions in sorted(char_positions.items()):
        print(f"  '{char}': positions {positions}")
    
    # Build DP table with detailed logging
    dp = [[0] * n for _ in range(n)]
    choice = [[None] * n for _ in range(n)]
    
    # Base case
    print(f"\nBase case (single characters):")
    for i in range(n):
        dp[i][i] = 1
        choice[i][i] = ('single', i)
        print(f"  dp[{i}][{i}] = 1 ('{s[i]}')")
    
    print(f"\nDP computation:")
    
    for length in range(2, min(n + 1, 7)):  # Show first few lengths
        print(f"\nLength {length} intervals:")
        for i in range(n - length + 1):
            j = i + length - 1
            substring = s[i:j+1]
            
            print(f"  Interval [{i},{j}]: '{substring}'")
            
            # Default: extend s[i] across range
            default_cost = 1 + dp[i + 1][j]
            dp[i][j] = default_cost
            choice[i][j] = ('extend', i, j)
            
            print(f"    Default (extend '{s[i]}'): 1 + {dp[i + 1][j]} = {default_cost}")
            
            # Try merging with same characters
            for k in range(i + 1, j + 1):
                if s[k] == s[i]:
                    merge_cost = dp[i + 1][k] + dp[k + 1][j]
                    print(f"    Merge with pos {k} ('{s[k]}'): {dp[i + 1][k]} + {dp[k + 1][j]} = {merge_cost}")
                    
                    if merge_cost < dp[i][j]:
                        dp[i][j] = merge_cost
                        choice[i][j] = ('merge', i, k)
                        print(f"      *** New minimum!")
            
            print(f"    Result: dp[{i}][{j}] = {dp[i][j]}")
    
    print(f"\nFinal DP Table:")
    print("   ", end="")
    for j in range(n):
        print(f"{j:4}", end="")
    print()
    
    for i in range(n):
        print(f"{i:2}: ", end="")
        for j in range(n):
            if j >= i:
                print(f"{dp[i][j]:4}", end="")
            else:
                print(f"{'':4}", end="")
        print()
    
    result = dp[0][n - 1]
    print(f"\nMinimum turns: {result}")
    
    # Show printing strategy
    min_turns, strategy = strange_printer_with_strategy(s)
    if strategy:
        print(f"\nPrinting strategy ({len(strategy)} operations):")
        for i, operation in enumerate(strategy):
            if operation[0] == 'print':
                char = operation[1]
                start = operation[2]
                end = operation[3]
                print(f"  {i + 1}. Print '{char}' from position {start} to {end}")
    
    return result


def strange_printer_variants():
    """
    STRANGE PRINTER VARIANTS:
    ========================
    Different scenarios and modifications.
    """
    
    def min_turns_with_limit(s, max_turns):
        """Check if string can be printed within max_turns"""
        min_needed = strange_printer_optimized(s)
        return min_needed <= max_turns
    
    def min_turns_different_costs(s, costs):
        """Different costs for printing different characters"""
        n = len(s)
        if n == 0:
            return 0
        
        dp = [[float('inf')] * n for _ in range(n)]
        
        # Base case with custom costs
        for i in range(n):
            char_idx = ord(s[i]) - ord('a')
            dp[i][i] = costs[char_idx] if char_idx < len(costs) else 1
        
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                
                # Extend s[i] across range
                char_idx = ord(s[i]) - ord('a')
                extend_cost = (costs[char_idx] if char_idx < len(costs) else 1) + dp[i + 1][j]
                dp[i][j] = min(dp[i][j], extend_cost)
                
                # Try merging
                for k in range(i + 1, j + 1):
                    if s[k] == s[i]:
                        merge_cost = dp[i + 1][k] + dp[k + 1][j]
                        dp[i][j] = min(dp[i][j], merge_cost)
        
        return dp[0][n - 1]
    
    def count_printing_ways(s):
        """Count number of ways to achieve minimum turns"""
        # This is complex to implement efficiently
        # For demonstration, return 1 if achievable
        return 1 if s else 0
    
    def min_turns_with_position_costs(s, position_costs):
        """Different costs based on starting position"""
        n = len(s)
        if n == 0:
            return 0
        
        # Simplified version - use position_costs[0] for all
        base_cost = position_costs[0] if position_costs else 1
        return strange_printer_optimized(s) * base_cost
    
    # Test variants
    test_cases = [
        "aaabbb",
        "aba",
        "abcabc",
        "aabbcc",
        "abab",
        "aaa"
    ]
    
    print("Strange Printer Variants:")
    print("=" * 50)
    
    for s in test_cases:
        print(f"\nString: '{s}'")
        
        min_turns = strange_printer_optimized(s)
        print(f"Min turns: {min_turns}")
        
        can_do_in_3 = min_turns_with_limit(s, 3)
        print(f"Can print in ≤3 turns: {can_do_in_3}")
        
        # With different character costs
        char_costs = [1, 2, 3, 4, 5]  # a=1, b=2, c=3, etc.
        different_costs = min_turns_different_costs(s, char_costs)
        print(f"With varying char costs: {different_costs}")
        
        ways = count_printing_ways(s)
        print(f"Number of optimal ways: {ways}")


# Test cases
def test_strange_printer():
    """Test all implementations with various inputs"""
    test_cases = [
        ("aaabbb", 2),
        ("aba", 2),
        ("abcabc", 5),
        ("a", 1),
        ("aa", 1),
        ("ab", 2),
        ("abc", 3),
        ("aab", 2),
        ("abab", 3),
        ("abcba", 3)
    ]
    
    print("Testing Strange Printer Solutions:")
    print("=" * 70)
    
    for i, (s, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"String: '{s}'")
        print(f"Expected: {expected}")
        
        # Skip recursive for long strings
        if len(s) <= 6:
            try:
                recursive = strange_printer_recursive(s)
                print(f"Recursive:        {recursive:>4} {'✓' if recursive == expected else '✗'}")
            except:
                print(f"Recursive:        Timeout")
        
        memoization = strange_printer_memoization(s)
        interval_dp = strange_printer_interval_dp(s)
        optimized = strange_printer_optimized(s)
        
        print(f"Memoization:      {memoization:>4} {'✓' if memoization == expected else '✗'}")
        print(f"Interval DP:      {interval_dp:>4} {'✓' if interval_dp == expected else '✗'}")
        print(f"Optimized:        {optimized:>4} {'✓' if optimized == expected else '✗'}")
        
        # Show strategy for small cases
        if len(s) <= 8:
            min_turns, strategy = strange_printer_with_strategy(s)
            print(f"Strategy steps: {len(strategy)}")
            if len(strategy) <= 4:
                for j, (op, char, start, end) in enumerate(strategy):
                    if op == 'print':
                        print(f"  {j+1}. Print '{char}' [{start}:{end}]")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    strange_printer_analysis("aba")
    
    # Variants demonstration
    print(f"\n" + "=" * 70)
    strange_printer_variants()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. COVERING STRATEGY: Print character across range, then overwrite")
    print("2. MERGING OPTIMIZATION: Combine same characters efficiently")
    print("3. INTERVAL DP: Optimal substructure for substring printing")
    print("4. OVERWRITING: Can print over existing characters")
    print("5. PREPROCESSING: Remove consecutive duplicates for efficiency")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Printing Optimization: Minimize ink cartridge changes")
    print("• Manufacturing: Minimize tool changes in production")
    print("• Algorithm Design: Resource allocation with overwriting")
    print("• Scheduling: Task batching with override capabilities")
    print("• Graphics: Optimal rendering with layer overwriting")


if __name__ == "__main__":
    test_strange_printer()


"""
STRANGE PRINTER - INTERVAL DP WITH OVERWRITING STRATEGY:
========================================================

This problem introduces overwriting mechanics to interval DP:
- Can print same character across any range
- New prints overwrite existing characters
- Goal: minimize number of printing operations
- Demonstrates interval DP with covering/overwriting strategies

KEY INSIGHTS:
============
1. **OVERWRITING STRATEGY**: Can print character across range and overwrite later
2. **MERGING OPTIMIZATION**: Combine distant same characters efficiently
3. **COVERING PRINCIPLE**: Print broad ranges first, refine with specifics
4. **INTERVAL DECOMPOSITION**: Optimal substructure for substring problems
5. **PREPROCESSING**: Remove consecutive duplicates for efficiency

ALGORITHM APPROACHES:
====================

1. **Recursive (Brute Force)**: O(exponential) time, O(n) space
   - Try all possible printing sequences
   - Exponential branching factor

2. **Memoization**: O(n³) time, O(n²) space
   - Top-down DP with substring caching
   - Natural recursive structure

3. **Interval DP**: O(n³) time, O(n²) space
   - Bottom-up construction by interval length
   - Standard approach for this problem

4. **Optimized DP**: O(n³) time, O(n²) space
   - Preprocess to remove consecutive duplicates
   - Better constants, same asymptotic complexity

CORE INTERVAL DP ALGORITHM:
==========================
```python
# dp[i][j] = minimum turns to print s[i:j+1]
dp = [[0] * n for _ in range(n)]

# Base case: single characters need 1 turn
for i in range(n):
    dp[i][i] = 1

for length in range(2, n + 1):
    for i in range(n - length + 1):
        j = i + length - 1
        
        # Strategy 1: Print s[i] across entire range, then fix rest
        dp[i][j] = 1 + dp[i + 1][j]
        
        # Strategy 2: Merge s[i] with same character at position k
        for k in range(i + 1, j + 1):
            if s[k] == s[i]:
                dp[i][j] = min(dp[i][j], dp[i + 1][k] + dp[k + 1][j])
```

RECURRENCE RELATION:
===================
```
dp[i][j] = min(
    1 + dp[i+1][j],                           // Print s[i] across range
    min(dp[i+1][k] + dp[k+1][j])             // Merge s[i] with s[k]
        for all k where s[k] == s[i]
)

Base case: dp[i][i] = 1 (single character needs 1 turn)
```

OVERWRITING STRATEGY:
====================
**Key Insight**: Can print character across entire range, then overwrite specific positions

**Example**: "aba"
1. Print 'a' across entire range: "aaa"
2. Print 'b' at position 1: "aba"
Total: 2 operations

**Why this works**:
- Broader prints create foundation
- Specific overwrites handle exceptions
- Often more efficient than character-by-character

MERGING OPTIMIZATION:
====================
**Distant Same Characters**: Can combine by removing middle parts first

**Example**: "abcab"
- Instead of printing each 'a' separately
- Remove middle "bcb" optimally
- Then print 'a' across both positions

**Algorithm**: For each character position, try merging with later same characters

PREPROCESSING OPTIMIZATION:
==========================
**Remove Consecutive Duplicates**:
- "aaabbb" → "ab"
- Consecutive same characters always printed together
- Reduces problem size significantly

```python
compressed = []
for char in s:
    if not compressed or compressed[-1] != char:
        compressed.append(char)
```

COMPLEXITY ANALYSIS:
===================
- **Time**: O(n³) - nested loops with merging attempts
- **Space**: O(n²) - DP table for all intervals
- **States**: O(n²) - all possible substrings
- **Transitions**: O(n) - try merging with each same character

STRATEGY RECONSTRUCTION:
=======================
To find actual printing sequence:
```python
def build_strategy(start, end, choice):
    if start > end:
        return []
    
    decision = choice[start][end]
    
    if decision[0] == 'extend':
        # Print character across range
        return [('print', s[start], start, end)] + build_strategy(start+1, end)
    elif decision[0] == 'merge':
        # Merge with same character
        k = decision[2]
        return (build_strategy(start+1, k) + 
                build_strategy(k+1, end) + 
                [('print', s[start], start, k)])
```

MATHEMATICAL PROPERTIES:
========================
- **Optimal Substructure**: Optimal printing contains optimal subprinting
- **Overlapping Subproblems**: Same substrings appear multiple times
- **Monotonicity**: Longer strings require at least as many operations
- **Covering Property**: Broader operations often more efficient

APPLICATIONS:
============
- **Printing Optimization**: Minimize ink cartridge/tool changes
- **Manufacturing**: Batch processing with equipment changes
- **Graphics Rendering**: Layer-based drawing optimization
- **Resource Allocation**: Minimize resource switching costs
- **Scheduling**: Task batching with override capabilities

RELATED PROBLEMS:
================
- **Remove Boxes (546)**: Similar merging strategies
- **Burst Balloons (312)**: Interval DP with optimal choices
- **Palindrome Partitioning**: String interval processing
- **Matrix Chain Multiplication**: Classic interval optimization

VARIANTS:
========
- **Different Character Costs**: Varying costs for different characters
- **Position-Based Costs**: Cost depends on printing position
- **Limited Operations**: Maximum number of printing operations
- **Multi-Color Printing**: Multiple colors with interaction rules

EDGE CASES:
==========
- **Single character**: Return 1
- **All same characters**: Return 1 (print once across range)
- **All different characters**: Return n (each needs separate operation)
- **Empty string**: Return 0

OPTIMIZATION TECHNIQUES:
=======================
- **Consecutive Merging**: Preprocess consecutive same characters
- **Pruning**: Skip obviously suboptimal strategies
- **Symmetry**: Use string symmetries to reduce computation
- **Memory Management**: Optimize cache usage for large inputs

This problem elegantly demonstrates how overwriting capabilities
can dramatically change optimal strategies in interval DP,
showing the importance of considering non-obvious approaches
like printing broad ranges before specific refinements.
"""
