"""
LeetCode 233: Number of Digit One
Difficulty: Hard
Category: Digit DP - Digit Counting

PROBLEM DESCRIPTION:
===================
Given an integer n, count the total number of digit 1 appearing in all non-negative integers less than or equal to n.

Example 1:
Input: n = 13
Output: 6
Explanation: The digit 1 appeared in the following numbers: 1, 10, 11, 12, 13.

Example 2:
Input: n = 0
Output: 0

Constraints:
- 0 <= n <= 10^9
"""


def count_digit_one_brute_force(n):
    """
    BRUTE FORCE APPROACH:
    ====================
    Count digit 1 in each number from 0 to n.
    
    Time Complexity: O(n * log(n)) - for each number, count digits
    Space Complexity: O(1) - constant space
    """
    if n <= 0:
        return 0
    
    count = 0
    for i in range(1, n + 1):
        num = i
        while num > 0:
            if num % 10 == 1:
                count += 1
            num //= 10
    
    return count


def count_digit_one_digit_dp(n):
    """
    DIGIT DP APPROACH:
    =================
    Use digit DP to count occurrences of digit 1.
    
    Time Complexity: O(log(n)^2) - digits × states
    Space Complexity: O(log(n)^2) - memoization
    """
    if n <= 0:
        return 0
    
    s = str(n)
    memo = {}
    
    def dp(pos, count_ones, tight, started):
        """
        pos: current position in the number
        count_ones: count of 1s placed so far
        tight: whether we're bounded by the original number
        started: whether we've started placing non-zero digits
        """
        if pos == len(s):
            return count_ones
        
        if (pos, count_ones, tight, started) in memo:
            return memo[(pos, count_ones, tight, started)]
        
        limit = int(s[pos]) if tight else 9
        result = 0
        
        for digit in range(0, limit + 1):
            new_count = count_ones + (1 if digit == 1 and (started or digit > 0) else 0)
            new_tight = tight and (digit == limit)
            new_started = started or (digit > 0)
            
            result += dp(pos + 1, new_count, new_tight, new_started)
        
        memo[(pos, count_ones, tight, started)] = result
        return result
    
    return dp(0, 0, True, False)


def count_digit_one_mathematical(n):
    """
    MATHEMATICAL APPROACH:
    =====================
    Count digit 1 occurrences mathematically for each position.
    
    Time Complexity: O(log(n)) - process each digit position
    Space Complexity: O(1) - constant space
    """
    if n <= 0:
        return 0
    
    count = 0
    factor = 1
    
    while factor <= n:
        # Count 1s at current position
        lower = n - (n // factor) * factor
        current = (n // factor) % 10
        higher = n // (factor * 10)
        
        if current == 0:
            count += higher * factor
        elif current == 1:
            count += higher * factor + lower + 1
        else:
            count += (higher + 1) * factor
        
        factor *= 10
    
    return count


def count_digit_one_optimized_dp(n):
    """
    OPTIMIZED DIGIT DP:
    ==================
    Optimized version focusing on counting 1s efficiently.
    
    Time Complexity: O(log(n)) - linear in digits
    Space Complexity: O(log(n)) - memoization
    """
    if n <= 0:
        return 0
    
    s = str(n)
    memo = {}
    
    def dp(pos, tight):
        if pos == len(s):
            return 0
        
        if (pos, tight) in memo:
            return memo[(pos, tight)]
        
        limit = int(s[pos]) if tight else 9
        result = 0
        
        for digit in range(0, limit + 1):
            # Count of 1s from this digit choice
            ones_from_digit = 1 if digit == 1 else 0
            
            # Count of 1s from remaining positions
            new_tight = tight and (digit == limit)
            ones_from_rest = dp(pos + 1, new_tight)
            
            # Total ways to place remaining digits
            if tight and digit == limit:
                ways = 1  # Only one way when tight
            else:
                remaining_positions = len(s) - pos - 1
                ways = 10 ** remaining_positions if remaining_positions > 0 else 1
            
            if digit == 1:
                # This digit contributes 1 to all numbers formed
                result += ways + ones_from_rest
            else:
                result += ones_from_rest
        
        memo[(pos, tight)] = result
        return result
    
    return dp(0, True)


def count_digit_one_with_analysis(n):
    """
    DIGIT DP WITH DETAILED ANALYSIS:
    ===============================
    Track the counting process with detailed breakdown.
    
    Time Complexity: O(log(n)^2) - standard digit DP
    Space Complexity: O(log(n)^2) - memoization + analysis
    """
    if n <= 0:
        return 0
    
    s = str(n)
    memo = {}
    analysis = {
        'by_position': [0] * len(s),
        'total_calls': 0,
        'cache_hits': 0
    }
    
    def dp(pos, tight, started):
        analysis['total_calls'] += 1
        
        if pos == len(s):
            return 0
        
        if (pos, tight, started) in memo:
            analysis['cache_hits'] += 1
            return memo[(pos, tight, started)]
        
        limit = int(s[pos]) if tight else 9
        result = 0
        
        for digit in range(0, limit + 1):
            new_tight = tight and (digit == limit)
            new_started = started or (digit > 0)
            
            # Count 1s from current position
            if digit == 1 and new_started:
                # Calculate how many numbers this contributes to
                remaining_digits = len(s) - pos - 1
                if tight and digit == limit:
                    # Limited by original number
                    remaining_part = int(s[pos + 1:]) if pos + 1 < len(s) else 0
                    contribution = remaining_part + 1
                else:
                    # All possible combinations
                    contribution = 10 ** remaining_digits if remaining_digits > 0 else 1
                
                result += contribution
                analysis['by_position'][pos] += contribution
            
            # Add 1s from remaining positions
            result += dp(pos + 1, new_tight, new_started)
        
        memo[(pos, tight, started)] = result
        return result
    
    total = dp(0, True, False)
    return total, analysis


def count_digit_one_range(start, end):
    """
    COUNT DIGIT 1 IN RANGE:
    =======================
    Count digit 1 occurrences in range [start, end].
    
    Time Complexity: O(log(end)) - two digit DP calls
    Space Complexity: O(log(end)) - memoization
    """
    if start > end:
        return 0
    
    def count_up_to(n):
        if n < 0:
            return 0
        return count_digit_one_digit_dp(n)
    
    return count_up_to(end) - count_up_to(start - 1)


def count_digit_one_analysis(n):
    """
    COMPREHENSIVE ANALYSIS:
    ======================
    Analyze digit 1 counting patterns and provide insights.
    """
    if n <= 0:
        print(f"Number {n} has no positive digits to analyze.")
        return 0
    
    print(f"Number of Digit One Analysis for n = {n}:")
    print(f"Number as string: '{n}'")
    print(f"Number of digits: {len(str(n))}")
    
    # Different approaches
    if n <= 10000:  # Only for small numbers due to time complexity
        brute_force = count_digit_one_brute_force(n)
        print(f"Brute force result: {brute_force}")
    
    digit_dp = count_digit_one_digit_dp(n)
    mathematical = count_digit_one_mathematical(n)
    optimized_dp = count_digit_one_optimized_dp(n)
    
    print(f"Digit DP result: {digit_dp}")
    print(f"Mathematical result: {mathematical}")
    print(f"Optimized DP result: {optimized_dp}")
    
    # Detailed analysis
    total, analysis = count_digit_one_with_analysis(n)
    
    print(f"\nDetailed Analysis:")
    print(f"Total digit 1 count: {total}")
    print(f"DP function calls: {analysis['total_calls']}")
    print(f"Cache hits: {analysis['cache_hits']}")
    print(f"Cache hit rate: {analysis['cache_hits']/analysis['total_calls']:.2%}")
    
    print(f"\nContribution by position:")
    for i, count in enumerate(analysis['by_position']):
        print(f"  Position {i} (digit {str(n)[i]}): {count} ones")
    
    # Pattern analysis for powers of 10
    if str(n).count('0') == len(str(n)) - 1 and str(n)[0] == '1':
        print(f"\nPattern Recognition: {n} is a power of 10")
        theoretical = sum(10**i for i in range(len(str(n)) - 1))
        print(f"Theoretical count for 10^{len(str(n))-1}: {theoretical}")
    
    return total


def count_digit_one_variants():
    """
    DIGIT ONE COUNTING VARIANTS:
    ===========================
    Different scenarios and modifications.
    """
    
    def count_specific_digit(n, target_digit):
        """Count occurrences of any specific digit"""
        if n <= 0:
            return 0
        
        s = str(n)
        memo = {}
        
        def dp(pos, tight, started):
            if pos == len(s):
                return 0
            
            if (pos, tight, started) in memo:
                return memo[(pos, tight, started)]
            
            limit = int(s[pos]) if tight else 9
            result = 0
            
            for digit in range(0, limit + 1):
                new_tight = tight and (digit == limit)
                new_started = started or (digit > 0)
                
                count_from_here = 0
                if digit == target_digit and (new_started or target_digit == 0):
                    remaining = len(s) - pos - 1
                    if tight and digit == limit:
                        remaining_part = int(s[pos + 1:]) if pos + 1 < len(s) else 0
                        count_from_here = remaining_part + 1
                    else:
                        count_from_here = 10 ** remaining if remaining > 0 else 1
                
                result += count_from_here + dp(pos + 1, new_tight, new_started)
            
            memo[(pos, tight, started)] = result
            return result
        
        return dp(0, True, False)
    
    def count_digit_one_with_exclusions(n, excluded_positions):
        """Count digit 1 excluding certain positions"""
        if n <= 0:
            return 0
        
        s = str(n)
        memo = {}
        
        def dp(pos, tight, started):
            if pos == len(s):
                return 0
            
            if (pos, tight, started) in memo:
                return memo[(pos, tight, started)]
            
            limit = int(s[pos]) if tight else 9
            result = 0
            
            for digit in range(0, limit + 1):
                new_tight = tight and (digit == limit)
                new_started = started or (digit > 0)
                
                count_from_here = 0
                if digit == 1 and new_started and pos not in excluded_positions:
                    remaining = len(s) - pos - 1
                    if tight and digit == limit:
                        remaining_part = int(s[pos + 1:]) if pos + 1 < len(s) else 0
                        count_from_here = remaining_part + 1
                    else:
                        count_from_here = 10 ** remaining if remaining > 0 else 1
                
                result += count_from_here + dp(pos + 1, new_tight, new_started)
            
            memo[(pos, tight, started)] = result
            return result
        
        return dp(0, True, False)
    
    def count_digit_one_efficient_range(start, end):
        """Efficient range counting"""
        def count_up_to(n):
            return count_digit_one_mathematical(n) if n >= 0 else 0
        
        return count_up_to(end) - count_up_to(start - 1)
    
    # Test variants
    test_numbers = [13, 100, 1000, 1234, 9999]
    
    print("Digit One Counting Variants:")
    print("=" * 50)
    
    for n in test_numbers:
        print(f"\nn = {n}")
        
        ones = count_digit_one_digit_dp(n)
        twos = count_specific_digit(n, 2)
        zeros = count_specific_digit(n, 0)
        
        print(f"Digit 1 count: {ones}")
        print(f"Digit 2 count: {twos}")
        print(f"Digit 0 count: {zeros}")
        
        # Exclude first position
        if len(str(n)) > 1:
            ones_no_first = count_digit_one_with_exclusions(n, {0})
            print(f"Digit 1 count (excluding first position): {ones_no_first}")
    
    # Range testing
    print(f"\nRange Testing:")
    ranges = [(1, 13), (10, 99), (100, 200)]
    for start, end in ranges:
        count = count_digit_one_efficient_range(start, end)
        print(f"Range [{start}, {end}]: {count} ones")


# Test cases
def test_count_digit_one():
    """Test all implementations with various inputs"""
    test_cases = [
        (13, 6),
        (0, 0),
        (1, 1),
        (10, 2),
        (11, 4),
        (12, 5),
        (100, 21),
        (101, 23),
        (111, 26),
        (1000, 301),
        (1234, 689)
    ]
    
    print("Testing Number of Digit One Solutions:")
    print("=" * 70)
    
    for i, (n, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"n = {n}")
        print(f"Expected: {expected}")
        
        # Skip brute force for large numbers
        if n <= 10000:
            try:
                brute_force = count_digit_one_brute_force(n)
                print(f"Brute Force:      {brute_force:>4} {'✓' if brute_force == expected else '✗'}")
            except:
                print(f"Brute Force:      Timeout")
        
        digit_dp = count_digit_one_digit_dp(n)
        mathematical = count_digit_one_mathematical(n)
        optimized_dp = count_digit_one_optimized_dp(n)
        
        print(f"Digit DP:         {digit_dp:>4} {'✓' if digit_dp == expected else '✗'}")
        print(f"Mathematical:     {mathematical:>4} {'✓' if mathematical == expected else '✗'}")
        print(f"Optimized DP:     {optimized_dp:>4} {'✓' if optimized_dp == expected else '✗'}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    count_digit_one_analysis(1234)
    
    # Variants demonstration
    print(f"\n" + "=" * 70)
    count_digit_one_variants()
    
    # Performance comparison
    print(f"\n" + "=" * 70)
    print("PERFORMANCE COMPARISON:")
    large_numbers = [10**6, 10**7, 10**8]
    for n in large_numbers:
        print(f"\nn = {n:,}")
        
        import time
        
        start = time.time()
        mathematical_result = count_digit_one_mathematical(n)
        math_time = time.time() - start
        
        start = time.time()
        dp_result = count_digit_one_digit_dp(n)
        dp_time = time.time() - start
        
        print(f"Mathematical: {mathematical_result:,} (Time: {math_time:.6f}s)")
        print(f"Digit DP:     {dp_result:,} (Time: {dp_time:.6f}s)")
        print(f"Match: {'✓' if mathematical_result == dp_result else '✗'}")
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. DIGIT DP STATE: pos, tight, started flags for valid number construction")
    print("2. MATHEMATICAL PATTERN: Position-wise analysis for O(log n) solution")
    print("3. TIGHT CONSTRAINT: Upper bound enforcement during digit placement")
    print("4. LEADING ZEROS: Handle numbers that haven't started yet")
    print("5. MEMOIZATION: Cache states to avoid recomputation")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Number Theory: Digit pattern analysis and counting")
    print("• Competitive Programming: Classic digit DP template")
    print("• Statistics: Frequency analysis in number ranges")
    print("• Cryptography: Pattern analysis in numeric data")
    print("• Algorithm Design: Template for digit-based constraints")


if __name__ == "__main__":
    test_count_digit_one()


"""
NUMBER OF DIGIT ONE - FUNDAMENTAL DIGIT DP PATTERN:
===================================================

This problem establishes the core Digit DP framework:
- Position-wise number construction with constraints
- Tight bound management during digit placement
- Leading zero handling for valid number formation
- Efficient counting without explicit enumeration

KEY INSIGHTS:
============
1. **DIGIT DP STATE**: (position, tight, started) captures construction state
2. **TIGHT CONSTRAINT**: Maintains upper bound during digit-by-digit construction
3. **LEADING ZEROS**: Handle numbers that haven't started with non-zero digit
4. **COUNTING EFFICIENCY**: Count valid numbers without generating them
5. **MATHEMATICAL OPTIMIZATION**: Position-wise analysis for O(log n) solution

ALGORITHM APPROACHES:
====================

1. **Brute Force**: O(n log n) time, O(1) space
   - Check each number individually
   - Simple but inefficient for large n

2. **Digit DP**: O(log²n) time, O(log²n) space
   - State-based construction with memoization
   - Standard approach for digit constraint problems

3. **Mathematical**: O(log n) time, O(1) space
   - Position-wise mathematical analysis
   - Most efficient for this specific problem

4. **Optimized DP**: O(log n) time, O(log n) space
   - Streamlined digit DP focusing on counting

CORE DIGIT DP ALGORITHM:
=======================
```python
def countDigitOne(n):
    s = str(n)
    memo = {}
    
    def dp(pos, tight, started):
        if pos == len(s):
            return 0
        
        if (pos, tight, started) in memo:
            return memo[(pos, tight, started)]
        
        limit = int(s[pos]) if tight else 9
        result = 0
        
        for digit in range(0, limit + 1):
            new_tight = tight and (digit == limit)
            new_started = started or (digit > 0)
            
            # Count 1s from current position
            ones_here = 0
            if digit == 1 and new_started:
                remaining = len(s) - pos - 1
                if new_tight:
                    ones_here = int(s[pos+1:] or "0") + 1
                else:
                    ones_here = 10 ** remaining
            
            result += ones_here + dp(pos + 1, new_tight, new_started)
        
        memo[(pos, tight, started)] = result
        return result
    
    return dp(0, True, False)
```

DIGIT DP STATE DESIGN:
=====================
**State Parameters**:
- `pos`: Current position in number (left to right)
- `tight`: Whether we're bounded by original number at this position
- `started`: Whether we've placed a non-zero digit (handles leading zeros)

**State Transitions**:
- Choose digit ∈ [0, limit] where limit = original_digit if tight else 9
- Update tight: remains true only if we choose the limiting digit
- Update started: becomes true once we place any non-zero digit

**Counting Logic**:
- If current digit is 1 and number has started, count contributions
- Add recursive count from remaining positions

TIGHT CONSTRAINT MANAGEMENT:
===========================
**Purpose**: Ensure constructed numbers don't exceed original number

**Implementation**:
```python
limit = int(s[pos]) if tight else 9
new_tight = tight and (digit == limit)
```

**Behavior**:
- When tight=True: can only use digits up to original digit at this position
- When tight=False: can use any digit 0-9
- Tight becomes False once we choose a smaller digit

LEADING ZERO HANDLING:
=====================
**Problem**: Numbers like "0123" are invalid; should be "123"

**Solution**: Track `started` flag
- started=False: haven't placed non-zero digit yet
- started=True: valid number construction in progress

**Impact on Counting**:
- Only count digit 1 contributions after number has started
- Allows proper handling of numbers with different digit counts

MATHEMATICAL APPROACH:
=====================
**Position-wise Analysis**: For each digit position, calculate:
1. **Lower part**: Digits to the right
2. **Current digit**: Digit at current position  
3. **Higher part**: Digits to the left

**Counting Formula** for position with factor 10^i:
```python
if current == 0:
    count += higher * factor
elif current == 1:
    count += higher * factor + lower + 1
else:  # current > 1
    count += (higher + 1) * factor
```

**Intuition**:
- current=0: All 1s come from higher part cycling through
- current=1: Higher part cycles + current position contributes to lower+1 numbers
- current>1: Higher part cycles + current position contributes to all numbers

COMPLEXITY ANALYSIS:
===================
- **Digit DP**: O(log²n) time, O(log²n) space
- **Mathematical**: O(log n) time, O(1) space
- **States**: O(log n) positions × 2 tight values × 2 started values
- **Memoization**: Critical for avoiding exponential recomputation

APPLICATIONS:
============
- **Number Theory**: Digit frequency analysis in ranges
- **Competitive Programming**: Template for digit-based constraints
- **Statistics**: Pattern analysis in numeric datasets
- **Cryptography**: Frequency analysis for security applications
- **Algorithm Design**: Foundation for complex digit DP problems

RELATED PROBLEMS:
================
- **Count Numbers with Unique Digits**: Similar state management
- **Numbers At Most N Given Digit Set**: Restricted digit choices
- **Numbers With Repeated Digits**: Constraint on digit uniqueness
- **Find All Good Strings**: String version with additional constraints

VARIANTS:
========
- **Count Specific Digit**: Generalize to any digit, not just 1
- **Range Queries**: Count in range [L, R] = count(R) - count(L-1)
- **Multiple Digits**: Count numbers containing specific digit patterns
- **Position Constraints**: Exclude certain positions from counting

OPTIMIZATION TECHNIQUES:
=======================
- **State Compression**: Minimize state space dimensions
- **Mathematical Shortcuts**: Direct formulas for specific cases
- **Precomputation**: Cache results for repeated queries
- **Bottom-up DP**: Iterative implementation for space optimization

EDGE CASES:
==========
- **n = 0**: No positive numbers to analyze
- **Single Digit**: Simple base case
- **Powers of 10**: Special patterns in digit distribution
- **All Same Digits**: Uniform digit patterns

This problem represents the foundation of Digit DP:
demonstrating how to construct valid numbers digit-by-digit
while maintaining constraints and efficiently counting
patterns without explicit enumeration.
"""
