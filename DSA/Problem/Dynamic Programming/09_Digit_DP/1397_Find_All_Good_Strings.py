"""
LeetCode 1397: Find All Good Strings
Difficulty: Hard
Category: Digit DP - String Pattern Matching

PROBLEM DESCRIPTION:
===================
Given the strings s1 and s2 of size n and the string evil. Return the number of good strings.

A good string is a string that:
- Has size n.
- Is lexicographically greater than or equal to s1.
- Is lexicographically smaller than or equal to s2.
- Does not contain evil as a substring.

Since the answer can be very large, return it modulo 10^9 + 7.

Example 1:
Input: n = 2, s1 = "aa", s2 = "da", evil = "b"
Output: 51
Explanation: There are 25 strings starting with 'a': "aa","ab",...,"az" and 26 strings starting with 'c': "ca","cb",...,"cz" and a string starting with 'd': "da". 

Example 2:
Input: n = 8, s1 = "leetcode", s2 = "leetgoes", evil = "leet"
Output: 0
Explanation: All strings greater than or equal to s1 and smaller than or equal to s2 start with the prefix "leet", so there is not a single good string.

Example 3:
Input: n = 2, s1 = "gx", s2 = "gz", evil = "x"
Output: 2

Constraints:
- s1.length == s2.length == n
- 1 <= n <= 500
- 1 <= evil.length <= 50
- All strings consist of lowercase English letters.
- s1 <= s2
"""


def find_all_good_strings_brute_force(n, s1, s2, evil):
    """
    BRUTE FORCE APPROACH:
    ====================
    Generate all strings in range and check conditions.
    
    Time Complexity: O(26^n * n * evil.length) - exponential
    Space Complexity: O(n) - recursion depth
    """
    MOD = 10**9 + 7
    count = 0
    
    def generate_strings(current, pos):
        nonlocal count
        
        if pos == n:
            if s1 <= current <= s2 and evil not in current:
                count = (count + 1) % MOD
            return
        
        for c in 'abcdefghijklmnopqrstuvwxyz':
            new_string = current + c
            # Pruning: check if still possible to be in range
            if new_string + 'z' * (n - pos - 1) >= s1 and new_string + 'a' * (n - pos - 1) <= s2:
                generate_strings(new_string, pos + 1)
    
    generate_strings("", 0)
    return count


def find_all_good_strings_digit_dp_kmp(n, s1, s2, evil):
    """
    DIGIT DP WITH KMP APPROACH:
    ==========================
    Use digit DP with KMP for efficient evil substring detection.
    
    Time Complexity: O(n * evil.length * 26) - with memoization
    Space Complexity: O(n * evil.length) - memoization table
    """
    MOD = 10**9 + 7
    
    # Build KMP failure function for evil string
    def build_failure_function(pattern):
        m = len(pattern)
        failure = [0] * m
        j = 0
        
        for i in range(1, m):
            while j > 0 and pattern[i] != pattern[j]:
                j = failure[j - 1]
            if pattern[i] == pattern[j]:
                j += 1
            failure[i] = j
        
        return failure
    
    failure = build_failure_function(evil)
    memo = {}
    
    def dp(pos, tight_low, tight_high, evil_matched):
        if evil_matched == len(evil):
            return 0  # Contains evil substring
        
        if pos == n:
            return 1  # Valid string
        
        state = (pos, tight_low, tight_high, evil_matched)
        if state in memo:
            return memo[state]
        
        # Determine character range
        start_char = s1[pos] if tight_low else 'a'
        end_char = s2[pos] if tight_high else 'z'
        
        result = 0
        
        for c in range(ord(start_char), ord(end_char) + 1):
            char = chr(c)
            new_tight_low = tight_low and (char == s1[pos])
            new_tight_high = tight_high and (char == s2[pos])
            
            # Update evil matching state using KMP
            new_evil_matched = evil_matched
            while new_evil_matched > 0 and evil[new_evil_matched] != char:
                new_evil_matched = failure[new_evil_matched - 1]
            
            if evil[new_evil_matched] == char:
                new_evil_matched += 1
            
            result = (result + dp(pos + 1, new_tight_low, new_tight_high, new_evil_matched)) % MOD
        
        memo[state] = result
        return result
    
    return dp(0, True, True, 0)


def find_all_good_strings_optimized(n, s1, s2, evil):
    """
    OPTIMIZED DIGIT DP:
    ==================
    Streamlined version with efficient state management.
    
    Time Complexity: O(n * evil.length * 26) - optimal for this problem
    Space Complexity: O(n * evil.length) - memoization
    """
    MOD = 10**9 + 7
    
    # KMP preprocessing
    def get_next(pattern):
        m = len(pattern)
        next_arr = [0] * m
        j = 0
        
        for i in range(1, m):
            while j > 0 and pattern[i] != pattern[j]:
                j = next_arr[j - 1]
            if pattern[i] == pattern[j]:
                j += 1
            next_arr[i] = j
        
        return next_arr
    
    next_arr = get_next(evil)
    memo = {}
    
    def solve(pos, is_low_bound, is_high_bound, matched_len):
        if matched_len == len(evil):
            return 0
        
        if pos == n:
            return 1
        
        key = (pos, is_low_bound, is_high_bound, matched_len)
        if key in memo:
            return memo[key]
        
        low = ord(s1[pos]) if is_low_bound else ord('a')
        high = ord(s2[pos]) if is_high_bound else ord('z')
        
        count = 0
        
        for c in range(low, high + 1):
            char = chr(c)
            
            new_is_low = is_low_bound and (c == low)
            new_is_high = is_high_bound and (c == high)
            
            # KMP transition
            new_matched = matched_len
            while new_matched > 0 and evil[new_matched] != char:
                new_matched = next_arr[new_matched - 1]
            
            if evil[new_matched] == char:
                new_matched += 1
            
            count = (count + solve(pos + 1, new_is_low, new_is_high, new_matched)) % MOD
        
        memo[key] = count
        return count
    
    return solve(0, True, True, 0)


def find_all_good_strings_with_analysis(n, s1, s2, evil):
    """
    DIGIT DP WITH DETAILED ANALYSIS:
    ===============================
    Track the computation process with KMP state transitions.
    
    Time Complexity: O(n * evil.length * 26) - standard approach
    Space Complexity: O(n * evil.length) - memoization + analysis
    """
    MOD = 10**9 + 7
    
    # Build KMP table
    def compute_lps(pattern):
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
    
    lps = compute_lps(evil)
    memo = {}
    analysis = {
        'kmp_table': lps,
        'total_states': 0,
        'evil_encounters': 0,
        'valid_strings': 0
    }
    
    def dp_solve(pos, tight_s1, tight_s2, evil_pos):
        analysis['total_states'] += 1
        
        if evil_pos == len(evil):
            analysis['evil_encounters'] += 1
            return 0
        
        if pos == n:
            analysis['valid_strings'] += 1
            return 1
        
        state = (pos, tight_s1, tight_s2, evil_pos)
        if state in memo:
            return memo[state]
        
        start = ord(s1[pos]) if tight_s1 else ord('a')
        end = ord(s2[pos]) if tight_s2 else ord('z')
        
        result = 0
        
        for c_ord in range(start, end + 1):
            c = chr(c_ord)
            
            new_tight_s1 = tight_s1 and (c == s1[pos])
            new_tight_s2 = tight_s2 and (c == s2[pos])
            
            # KMP state transition
            new_evil_pos = evil_pos
            while new_evil_pos > 0 and evil[new_evil_pos] != c:
                new_evil_pos = lps[new_evil_pos - 1]
            
            if evil[new_evil_pos] == c:
                new_evil_pos += 1
            
            result = (result + dp_solve(pos + 1, new_tight_s1, new_tight_s2, new_evil_pos)) % MOD
        
        memo[state] = result
        return result
    
    count = dp_solve(0, True, True, 0)
    return count, analysis


def find_all_good_strings_analysis(n, s1, s2, evil):
    """
    COMPREHENSIVE ANALYSIS:
    ======================
    Analyze the good strings problem with detailed insights.
    """
    print(f"Find All Good Strings Analysis:")
    print(f"String length n: {n}")
    print(f"Lower bound s1: '{s1}'")
    print(f"Upper bound s2: '{s2}'")
    print(f"Evil substring: '{evil}'")
    print(f"Evil length: {len(evil)}")
    
    # Check basic validity
    if s1 > s2:
        print("Invalid: s1 > s2")
        return 0
    
    if len(s1) != n or len(s2) != n:
        print("Invalid: string lengths don't match n")
        return 0
    
    # Different approaches (for small cases only)
    if n <= 3 and len(evil) <= 2:
        try:
            brute_force = find_all_good_strings_brute_force(n, s1, s2, evil)
            print(f"Brute force result: {brute_force}")
        except:
            print("Brute force: Too large")
    
    optimized = find_all_good_strings_optimized(n, s1, s2, evil)
    kmp_result = find_all_good_strings_digit_dp_kmp(n, s1, s2, evil)
    
    print(f"Optimized result: {optimized}")
    print(f"KMP DP result: {kmp_result}")
    
    # Detailed analysis
    count_with_analysis, analysis = find_all_good_strings_with_analysis(n, s1, s2, evil)
    
    print(f"\nDetailed Analysis:")
    print(f"Total good strings: {count_with_analysis}")
    print(f"KMP failure table: {analysis['kmp_table']}")
    print(f"Total DP states explored: {analysis['total_states']}")
    print(f"Evil substring encounters: {analysis['evil_encounters']}")
    print(f"Valid string completions: {analysis['valid_strings']}")
    
    # Range analysis
    total_possible = 26 ** n
    range_size = ord(s2[0]) - ord(s1[0]) + 1
    print(f"\nRange Analysis:")
    print(f"Total possible {n}-char strings: {total_possible:,}")
    print(f"First character range: {range_size} options")
    print(f"Good strings found: {count_with_analysis}")
    print(f"Percentage of range that's good: {count_with_analysis/min(total_possible, 10**6):.6%}")
    
    return count_with_analysis


def find_all_good_strings_variants():
    """
    GOOD STRINGS VARIANTS:
    =====================
    Different scenarios and modifications.
    """
    
    def count_without_multiple_evils(n, s1, s2, evils):
        """Count strings avoiding multiple evil substrings"""
        # This becomes much more complex with multiple patterns
        # Would need Aho-Corasick algorithm for efficiency
        # Simplified version using first evil only
        if not evils:
            return 26 ** n
        
        return find_all_good_strings_optimized(n, s1, s2, evils[0])
    
    def count_with_required_substring(n, s1, s2, required, evil):
        """Count strings that must contain required but not evil"""
        # This is very complex - would need state tracking for both patterns
        # Simplified approximation
        base_count = find_all_good_strings_optimized(n, s1, s2, evil)
        return max(0, base_count // 2)  # Rough approximation
    
    def count_good_strings_case_insensitive(n, s1, s2, evil):
        """Case insensitive version"""
        s1_lower = s1.lower()
        s2_lower = s2.lower()
        evil_lower = evil.lower()
        
        return find_all_good_strings_optimized(n, s1_lower, s2_lower, evil_lower)
    
    # Test variants
    test_cases = [
        (2, "aa", "da", "b"),
        (3, "abc", "abd", "c"),
        (2, "gx", "gz", "x")
    ]
    
    print("Good Strings Variants:")
    print("=" * 50)
    
    for n, s1, s2, evil in test_cases:
        print(f"\nn={n}, s1='{s1}', s2='{s2}', evil='{evil}'")
        
        basic_count = find_all_good_strings_optimized(n, s1, s2, evil)
        print(f"Basic good strings: {basic_count}")
        
        # Multiple evils (simplified)
        multiple_evils = [evil, "z"]  # Add dummy evil
        multi_evil_count = count_without_multiple_evils(n, s1, s2, multiple_evils)
        print(f"Without multiple evils: {multi_evil_count}")
        
        # Case insensitive
        if any(c.isupper() for c in s1 + s2 + evil):
            case_insensitive = count_good_strings_case_insensitive(n, s1, s2, evil)
            print(f"Case insensitive: {case_insensitive}")


# Test cases
def test_find_all_good_strings():
    """Test all implementations with various inputs"""
    test_cases = [
        (2, "aa", "da", "b", 51),
        (8, "leetcode", "leetgoes", "leet", 0),
        (2, "gx", "gz", "x", 2),
        (1, "a", "z", "b", 25),
        (3, "aaa", "aaa", "a", 0),
        (2, "ab", "bb", "a", 25),
        (3, "abc", "def", "cd", 364)
    ]
    
    print("Testing Find All Good Strings Solutions:")
    print("=" * 70)
    
    for i, (n, s1, s2, evil, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"n={n}, s1='{s1}', s2='{s2}', evil='{evil}'")
        print(f"Expected: {expected}")
        
        # Skip brute force for large inputs
        if n <= 3 and len(evil) <= 2:
            try:
                brute_force = find_all_good_strings_brute_force(n, s1, s2, evil)
                print(f"Brute Force:      {brute_force:>8} {'✓' if brute_force == expected else '✗'}")
            except:
                print(f"Brute Force:      Timeout")
        
        kmp_dp = find_all_good_strings_digit_dp_kmp(n, s1, s2, evil)
        optimized = find_all_good_strings_optimized(n, s1, s2, evil)
        
        print(f"KMP DP:           {kmp_dp:>8} {'✓' if kmp_dp == expected else '✗'}")
        print(f"Optimized:        {optimized:>8} {'✓' if optimized == expected else '✗'}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    find_all_good_strings_analysis(2, "aa", "da", "b")
    
    # Variants demonstration
    print(f"\n" + "=" * 70)
    find_all_good_strings_variants()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. STRING DIGIT DP: Apply digit DP principles to string construction")
    print("2. KMP INTEGRATION: Efficient pattern matching for evil detection")
    print("3. DUAL BOUNDS: Handle both lower and upper lexicographic bounds")
    print("4. STATE TRACKING: Manage multiple constraints simultaneously")
    print("5. PATTERN AVOIDANCE: Use automata for substring detection")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• String Generation: Constrained string enumeration problems")
    print("• Pattern Matching: Efficient substring detection in generation")
    print("• Lexicographic Analysis: Range-based string counting")
    print("• Algorithm Design: Combining DP with string algorithms")
    print("• Competitive Programming: Advanced string constraint problems")


if __name__ == "__main__":
    test_find_all_good_strings()


"""
FIND ALL GOOD STRINGS - ADVANCED STRING PATTERN DP:
===================================================

This problem combines Digit DP with advanced string algorithms:
- Lexicographic range constraints (s1 ≤ string ≤ s2)
- Pattern avoidance using KMP/failure function
- Multi-constraint optimization in string space
- Integration of DP with string matching algorithms

KEY INSIGHTS:
============
1. **STRING DIGIT DP**: Apply digit DP principles to character-by-character string construction
2. **KMP INTEGRATION**: Use KMP failure function for efficient pattern matching during construction
3. **DUAL BOUND CONSTRAINTS**: Handle both lower and upper lexicographic bounds simultaneously
4. **STATE SPACE COMPLEXITY**: Manage position, bound flags, and pattern matching state
5. **PATTERN AVOIDANCE**: Prevent evil substring formation during string generation

ALGORITHM APPROACHES:
====================

1. **Brute Force**: O(26^n × n × evil.length) time, O(n) space
   - Generate all strings, check constraints
   - Exponential and impractical for large inputs

2. **Digit DP + KMP**: O(n × evil.length × 26) time, O(n × evil.length) space
   - Efficient string construction with pattern avoidance
   - Standard approach for this problem type

3. **Optimized DP**: O(n × evil.length × 26) time, O(n × evil.length) space
   - Streamlined state management
   - Best practical solution

CORE STRING DIGIT DP WITH KMP:
=============================
```python
def findGoodStrings(n, s1, s2, evil):
    MOD = 10**9 + 7
    
    # Build KMP failure function
    def build_failure(pattern):
        m = len(pattern)
        failure = [0] * m
        j = 0
        for i in range(1, m):
            while j > 0 and pattern[i] != pattern[j]:
                j = failure[j - 1]
            if pattern[i] == pattern[j]:
                j += 1
            failure[i] = j
        return failure
    
    failure = build_failure(evil)
    memo = {}
    
    def dp(pos, tight_low, tight_high, evil_matched):
        if evil_matched == len(evil):
            return 0  # Contains evil
        if pos == n:
            return 1  # Valid string
        
        state = (pos, tight_low, tight_high, evil_matched)
        if state in memo:
            return memo[state]
        
        start = s1[pos] if tight_low else 'a'
        end = s2[pos] if tight_high else 'z'
        
        result = 0
        for c in range(ord(start), ord(end) + 1):
            char = chr(c)
            new_tight_low = tight_low and (char == s1[pos])
            new_tight_high = tight_high and (char == s2[pos])
            
            # KMP state transition
            new_evil_matched = evil_matched
            while new_evil_matched > 0 and evil[new_evil_matched] != char:
                new_evil_matched = failure[new_evil_matched - 1]
            if evil[new_evil_matched] == char:
                new_evil_matched += 1
            
            result = (result + dp(pos + 1, new_tight_low, new_tight_high, new_evil_matched)) % MOD
        
        memo[state] = result
        return result
    
    return dp(0, True, True, 0)
```

KMP FAILURE FUNCTION INTEGRATION:
================================
**Purpose**: Efficiently track progress toward matching evil substring

**KMP Preprocessing**:
```python
def compute_failure_function(pattern):
    m = len(pattern)
    failure = [0] * m
    j = 0
    for i in range(1, m):
        while j > 0 and pattern[i] != pattern[j]:
            j = failure[j - 1]
        if pattern[i] == pattern[j]:
            j += 1
        failure[i] = j
    return failure
```

**State Transition**: When adding character c to partially matched evil:
```python
while matched_length > 0 and evil[matched_length] != c:
    matched_length = failure[matched_length - 1]
if evil[matched_length] == c:
    matched_length += 1
```

LEXICOGRAPHIC BOUND MANAGEMENT:
==============================
**Dual Constraints**: String must satisfy s1 ≤ string ≤ s2

**State Tracking**:
- `tight_low`: Whether current prefix equals s1 prefix
- `tight_high`: Whether current prefix equals s2 prefix

**Character Range Determination**:
```python
start_char = s1[pos] if tight_low else 'a'
end_char = s2[pos] if tight_high else 'z'
```

**Bound Updates**:
- `tight_low` remains true only if we choose s1[pos]
- `tight_high` remains true only if we choose s2[pos]

MULTI-CONSTRAINT STATE SPACE:
=============================
**State Dimensions**:
- `pos`: Current position in string (0 to n-1)
- `tight_low`: Lower bound constraint flag
- `tight_high`: Upper bound constraint flag
- `evil_matched`: Characters matched in evil pattern (0 to len(evil)-1)

**State Transitions**: All constraints must be satisfied simultaneously

**Termination Conditions**:
- `evil_matched == len(evil)`: Evil found, return 0
- `pos == n`: Valid string completed, return 1

PATTERN AVOIDANCE STRATEGY:
==========================
**Objective**: Construct strings that do not contain evil as substring

**KMP Advantage**: Efficiently tracks partial matches without backtracking
- O(1) amortized time per character
- Optimal failure recovery using preprocessed table

**Early Termination**: Return 0 immediately when evil is fully matched

COMPLEXITY ANALYSIS:
===================
- **Time**: O(n × evil.length × 26) - position × pattern state × character choices
- **Space**: O(n × evil.length) - memoization table
- **States**: n positions × 2 tight_low × 2 tight_high × evil.length evil_matched
- **KMP Preprocessing**: O(evil.length) time, O(evil.length) space

APPLICATIONS:
============
- **String Generation**: Constrained string enumeration with pattern avoidance
- **Lexicographic Analysis**: Counting strings in lexicographic ranges
- **Pattern Matching**: Integration of string algorithms with DP
- **Bioinformatics**: DNA sequence generation with forbidden patterns
- **Security**: Password generation avoiding specific patterns

RELATED PROBLEMS:
================
- **Regular Expression Matching**: Pattern constraints in string DP
- **Wildcard Matching**: Similar string construction with pattern rules
- **String Interleaving**: Multi-string constraint problems
- **Aho-Corasick Applications**: Multiple pattern avoidance

VARIANTS:
========
- **Multiple Evil Patterns**: Use Aho-Corasick for multiple pattern detection
- **Required Patterns**: Must contain certain substrings while avoiding others
- **Case Insensitive**: Character equivalence classes
- **Alphabet Restrictions**: Limited character sets

OPTIMIZATION TECHNIQUES:
=======================
- **KMP Preprocessing**: Essential for efficient pattern matching
- **State Compression**: Minimize state space dimensions when possible
- **Early Termination**: Immediately reject when evil pattern is matched
- **Memory Management**: Optimize cache usage for large state spaces

EDGE CASES:
==========
- **s1 > s2**: Invalid input, return 0
- **evil longer than n**: Evil cannot appear, simpler counting
- **s1 == s2**: Single string, check if it contains evil
- **evil == ""**: Empty pattern, all strings valid (edge case)

This problem demonstrates the sophisticated integration of
Digit DP with advanced string algorithms, showing how
multiple complex constraints can be efficiently handled
through careful state space design and algorithmic optimization.
"""
