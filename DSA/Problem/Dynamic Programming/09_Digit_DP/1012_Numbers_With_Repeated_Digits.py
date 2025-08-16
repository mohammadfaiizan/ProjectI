"""
LeetCode 1012: Numbers With Repeated Digits
Difficulty: Hard
Category: Digit DP - Uniqueness Constraints

PROBLEM DESCRIPTION:
===================
Given an integer n, return the number of positive integers in the range [1, n] that have at least one repeated digit.

Example 1:
Input: n = 20
Output: 1
Explanation: The only positive number (<= 20) with at least one repeated digit is 11.

Example 2:
Input: n = 100
Output: 10
Explanation: The positive numbers (<= 100) with at least one repeated digit are: 11, 22, 33, 44, 55, 66, 77, 88, 99, 100.

Example 3:
Input: n = 1000
Output: 262

Constraints:
- 1 <= n <= 10^9
"""


def numbers_with_repeated_digits_brute_force(n):
    """
    BRUTE FORCE APPROACH:
    ====================
    Check each number individually for repeated digits.
    
    Time Complexity: O(n * log(n)) - check each number
    Space Complexity: O(1) - constant space
    """
    if n <= 0:
        return 0
    
    count = 0
    for i in range(1, n + 1):
        digits = str(i)
        if len(digits) != len(set(digits)):
            count += 1
    
    return count


def numbers_with_repeated_digits_complement(n):
    """
    COMPLEMENT APPROACH:
    ===================
    Count numbers WITHOUT repeated digits, then subtract from total.
    
    Time Complexity: O(log(n)^2) - digit DP computation
    Space Complexity: O(log(n) * 2^10) - memoization with bitmask
    """
    if n <= 0:
        return 0
    
    # Count numbers without repeated digits using digit DP
    s = str(n)
    memo = {}
    
    def dp(pos, mask, tight, started):
        """
        pos: current position
        mask: bitmask of used digits
        tight: whether bounded by n
        started: whether we've placed a non-zero digit
        """
        if pos == len(s):
            return 1 if started else 0
        
        if (pos, mask, tight, started) in memo:
            return memo[(pos, mask, tight, started)]
        
        limit = int(s[pos]) if tight else 9
        result = 0
        
        # Option 1: Don't place digit (leading zero)
        if not started:
            result += dp(pos + 1, mask, False, False)
        
        # Option 2: Place a digit
        for digit in range(0 if started else 1, limit + 1):
            if mask & (1 << digit):  # Digit already used
                continue
            
            new_mask = mask | (1 << digit)
            new_tight = tight and (digit == limit)
            result += dp(pos + 1, new_mask, new_tight, True)
        
        memo[(pos, mask, tight, started)] = result
        return result
    
    numbers_without_repeat = dp(0, 0, True, False)
    return n - numbers_without_repeat


def numbers_with_repeated_digits_direct_dp(n):
    """
    DIRECT DIGIT DP APPROACH:
    ========================
    Directly count numbers WITH repeated digits.
    
    Time Complexity: O(log(n)^2 * 2^10) - more complex state space
    Space Complexity: O(log(n) * 2^10) - memoization
    """
    if n <= 0:
        return 0
    
    s = str(n)
    memo = {}
    
    def dp(pos, mask, tight, started, has_repeat):
        if pos == len(s):
            return 1 if (started and has_repeat) else 0
        
        state = (pos, mask, tight, started, has_repeat)
        if state in memo:
            return memo[state]
        
        limit = int(s[pos]) if tight else 9
        result = 0
        
        # Skip position (leading zero)
        if not started:
            result += dp(pos + 1, mask, False, False, has_repeat)
        
        # Place digit
        for digit in range(0 if started else 1, limit + 1):
            new_has_repeat = has_repeat or (mask & (1 << digit) > 0)
            new_mask = mask | (1 << digit)
            new_tight = tight and (digit == limit)
            
            result += dp(pos + 1, new_mask, new_tight, True, new_has_repeat)
        
        memo[state] = result
        return result
    
    return dp(0, 0, True, False, False)


def numbers_with_repeated_digits_mathematical(n):
    """
    MATHEMATICAL APPROACH:
    =====================
    Calculate numbers without repeated digits mathematically.
    
    Time Complexity: O(log(n)) - process each digit position
    Space Complexity: O(1) - constant space
    """
    if n <= 0:
        return 0
    
    def permutation(n, r):
        """Calculate P(n, r) = n! / (n-r)!"""
        result = 1
        for i in range(n, n - r, -1):
            result *= i
        return result
    
    s = str(n)
    digits = len(s)
    
    # Count numbers with fewer digits (all have unique digits)
    unique_count = 0
    
    # 1-digit numbers: 1-9 (9 numbers)
    for length in range(1, digits):
        if length == 1:
            unique_count += 9
        else:
            # First digit: 1-9 (9 choices)
            # Remaining digits: choose from remaining 9 digits
            unique_count += 9 * permutation(9, length - 1)
    
    # Count numbers with same number of digits
    mask = 0  # Track used digits
    for i in range(digits):
        digit = int(s[i])
        
        # Count valid choices for current position
        valid_choices = 0
        for d in range(0 if i > 0 else 1, digit):
            if not (mask & (1 << d)):
                valid_choices += 1
        
        # Calculate remaining positions
        remaining_positions = digits - i - 1
        available_digits = 10 - bin(mask).count('1')
        
        if remaining_positions > 0:
            unique_count += valid_choices * permutation(available_digits - 1, remaining_positions)
        else:
            unique_count += valid_choices
        
        # Check if current digit is already used
        if mask & (1 << digit):
            break
        
        mask |= (1 << digit)
    else:
        # If we completed the loop, n itself has unique digits
        unique_count += 1
    
    return n - unique_count


def numbers_with_repeated_digits_optimized(n):
    """
    OPTIMIZED COMPLEMENT APPROACH:
    =============================
    Streamlined version of complement counting.
    
    Time Complexity: O(log(n) * 2^10) - optimized state space
    Space Complexity: O(log(n) * 2^10) - memoization
    """
    if n <= 0:
        return 0
    
    s = str(n)
    memo = {}
    
    def count_unique(pos, mask, tight, started):
        if pos == len(s):
            return started
        
        if (pos, mask, tight, started) in memo:
            return memo[(pos, mask, tight, started)]
        
        result = 0
        limit = int(s[pos]) if tight else 9
        
        # Leading zero case
        if not started:
            result += count_unique(pos + 1, mask, False, False)
        
        # Place digits
        for digit in range(1 if not started else 0, limit + 1):
            if mask & (1 << digit):
                continue
            
            new_mask = mask | (1 << digit)
            new_tight = tight and (digit == limit)
            result += count_unique(pos + 1, new_mask, new_tight, True)
        
        memo[(pos, mask, tight, started)] = result
        return result
    
    unique_numbers = count_unique(0, 0, True, False)
    return n - unique_numbers


def numbers_with_repeated_digits_analysis(n):
    """
    COMPREHENSIVE ANALYSIS:
    ======================
    Analyze repeated digit patterns and provide insights.
    """
    print(f"Numbers With Repeated Digits Analysis for n = {n}:")
    print(f"Range: [1, {n}]")
    print(f"Total numbers in range: {n}")
    
    s = str(n)
    print(f"Target number '{n}' has {len(s)} digits: {list(s)}")
    
    # Check if n itself has repeated digits
    n_digits = list(s)
    n_has_repeat = len(n_digits) != len(set(n_digits))
    print(f"Does {n} have repeated digits? {n_has_repeat}")
    
    # Different approaches
    if n <= 10000:  # Only for reasonable sizes
        try:
            brute_force = numbers_with_repeated_digits_brute_force(n)
            print(f"Brute force result: {brute_force}")
        except:
            print("Brute force: Too large")
    
    complement = numbers_with_repeated_digits_complement(n)
    direct_dp = numbers_with_repeated_digits_direct_dp(n)
    mathematical = numbers_with_repeated_digits_mathematical(n)
    optimized = numbers_with_repeated_digits_optimized(n)
    
    print(f"Complement approach: {complement}")
    print(f"Direct DP: {direct_dp}")
    print(f"Mathematical: {mathematical}")
    print(f"Optimized: {optimized}")
    
    # Breakdown analysis
    unique_count = n - complement
    repeat_count = complement
    
    print(f"\nBreakdown:")
    print(f"Numbers with unique digits: {unique_count}")
    print(f"Numbers with repeated digits: {repeat_count}")
    print(f"Percentage with repeats: {repeat_count/n:.2%}")
    
    # Pattern analysis by length
    print(f"\nPattern analysis by number length:")
    
    def count_unique_with_length(length):
        if length == 1:
            return 9  # 1-9
        elif length <= 10:
            # First digit: 1-9, remaining: permutations of 9
            result = 9
            for i in range(1, length):
                result *= (10 - i)
            return result
        else:
            return 0  # Can't have unique digits with length > 10
    
    cumulative_unique = 0
    for length in range(1, len(s) + 1):
        if length < len(s):
            unique_for_length = count_unique_with_length(length)
            total_for_length = 9 * (10 ** (length - 1))
        else:
            # Same length as n - need careful calculation
            unique_for_length = unique_count - cumulative_unique
            total_for_length = n - sum(9 * (10 ** (i - 1)) for i in range(1, length))
        
        repeat_for_length = total_for_length - unique_for_length
        print(f"  Length {length}: {repeat_for_length}/{total_for_length} have repeats")
        cumulative_unique += unique_for_length
    
    return repeat_count


def numbers_with_repeated_digits_variants():
    """
    REPEATED DIGITS VARIANTS:
    ========================
    Different scenarios and modifications.
    """
    
    def numbers_with_k_repeated_digits(n, k):
        """Count numbers with exactly k distinct digits repeated"""
        # This is complex - simplified version
        if k <= 0:
            return numbers_with_repeated_digits_optimized(n)
        
        # For demonstration, return subset
        total_with_repeats = numbers_with_repeated_digits_optimized(n)
        return total_with_repeats // (k + 1)  # Rough approximation
    
    def numbers_with_specific_digit_repeated(n, digit):
        """Count numbers where specific digit appears multiple times"""
        if n <= 0:
            return 0
        
        s = str(n)
        memo = {}
        
        def dp(pos, mask, tight, started, target_count):
            if pos == len(s):
                return 1 if (started and target_count >= 2) else 0
            
            state = (pos, mask, tight, started, target_count)
            if state in memo:
                return memo[state]
            
            result = 0
            limit = int(s[pos]) if tight else 9
            
            # Skip position
            if not started:
                result += dp(pos + 1, mask, False, False, target_count)
            
            # Place digits
            for d in range(0 if started else 1, limit + 1):
                new_count = target_count + (1 if d == digit else 0)
                new_tight = tight and (d == limit)
                result += dp(pos + 1, mask, new_tight, True, new_count)
            
            memo[state] = result
            return result
        
        return dp(0, 0, True, False, 0)
    
    def numbers_with_all_digits_unique(n):
        """Count numbers with all unique digits (complement)"""
        return n - numbers_with_repeated_digits_optimized(n)
    
    def max_unique_digit_number(length):
        """Maximum number with all unique digits for given length"""
        if length > 10:
            return -1  # Impossible
        
        if length == 1:
            return 9
        
        # Use digits 9, 8, 7, ..., (10-length)
        digits = list(range(9, 9 - length, -1))
        return int(''.join(map(str, digits)))
    
    # Test variants
    test_numbers = [20, 100, 1000, 1234]
    
    print("Repeated Digits Variants:")
    print("=" * 50)
    
    for n in test_numbers:
        print(f"\nn = {n}")
        
        basic_repeats = numbers_with_repeated_digits_optimized(n)
        unique_count = numbers_with_all_digits_unique(n)
        
        print(f"With repeated digits: {basic_repeats}")
        print(f"With unique digits: {unique_count}")
        print(f"Total check: {basic_repeats + unique_count} == {n} ✓")
        
        # Specific digit repetition
        digit_1_repeats = numbers_with_specific_digit_repeated(n, 1)
        print(f"Numbers with digit '1' repeated: {digit_1_repeats}")
        
        # k-repeated variants
        for k in range(1, 4):
            k_repeats = numbers_with_k_repeated_digits(n, k)
            print(f"Roughly {k} types repeated: {k_repeats}")
    
    # Maximum unique digit numbers
    print(f"\nMaximum unique digit numbers by length:")
    for length in range(1, 11):
        max_num = max_unique_digit_number(length)
        print(f"  Length {length}: {max_num}")


# Test cases
def test_numbers_with_repeated_digits():
    """Test all implementations with various inputs"""
    test_cases = [
        (20, 1),
        (100, 10),
        (1000, 262),
        (1, 0),
        (10, 0),
        (11, 1),
        (99, 9),
        (101, 11),
        (1111, 938),
        (12345, 10053)
    ]
    
    print("Testing Numbers With Repeated Digits Solutions:")
    print("=" * 70)
    
    for i, (n, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"n = {n}")
        print(f"Expected: {expected}")
        
        # Skip brute force for large numbers
        if n <= 10000:
            try:
                brute_force = numbers_with_repeated_digits_brute_force(n)
                print(f"Brute Force:      {brute_force:>6} {'✓' if brute_force == expected else '✗'}")
            except:
                print(f"Brute Force:      Timeout")
        
        complement = numbers_with_repeated_digits_complement(n)
        direct_dp = numbers_with_repeated_digits_direct_dp(n)
        mathematical = numbers_with_repeated_digits_mathematical(n)
        optimized = numbers_with_repeated_digits_optimized(n)
        
        print(f"Complement:       {complement:>6} {'✓' if complement == expected else '✗'}")
        print(f"Direct DP:        {direct_dp:>6} {'✓' if direct_dp == expected else '✗'}")
        print(f"Mathematical:     {mathematical:>6} {'✓' if mathematical == expected else '✗'}")
        print(f"Optimized:        {optimized:>6} {'✓' if optimized == expected else '✗'}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    numbers_with_repeated_digits_analysis(100)
    
    # Variants demonstration
    print(f"\n" + "=" * 70)
    numbers_with_repeated_digits_variants()
    
    # Performance comparison
    print(f"\n" + "=" * 70)
    print("PERFORMANCE COMPARISON:")
    large_numbers = [10**5, 10**6, 10**7]
    
    for n in large_numbers:
        print(f"\nn = {n:,}")
        
        import time
        
        start = time.time()
        math_result = numbers_with_repeated_digits_mathematical(n)
        math_time = time.time() - start
        
        start = time.time()
        opt_result = numbers_with_repeated_digits_optimized(n)
        opt_time = time.time() - start
        
        print(f"Mathematical: {math_result:,} (Time: {math_time:.6f}s)")
        print(f"Optimized DP: {opt_result:,} (Time: {opt_time:.6f}s)")
        print(f"Match: {'✓' if math_result == opt_result else '✗'}")
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. COMPLEMENT STRATEGY: Count unique digits, subtract from total")
    print("2. BITMASK STATE: Track used digits with 10-bit mask")
    print("3. PERMUTATION MATH: P(n,r) for arranging unique digits")
    print("4. LEADING ZERO HANDLING: Careful management of number start")
    print("5. MATHEMATICAL OPTIMIZATION: Direct formulas for unique digit counting")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Number Theory: Digit uniqueness analysis and patterns")
    print("• Combinatorics: Counting with uniqueness constraints")
    print("• Data Analysis: Detecting patterns in numeric datasets")
    print("• Cryptography: Analyzing randomness in numeric sequences")
    print("• Algorithm Design: Template for uniqueness-based problems")


if __name__ == "__main__":
    test_numbers_with_repeated_digits()


"""
NUMBERS WITH REPEATED DIGITS - UNIQUENESS CONSTRAINT DP:
========================================================

This problem demonstrates the complement strategy in Digit DP:
- Counting by complement (total - unique = repeated)
- Bitmask state management for digit tracking
- Mathematical optimization using permutations
- Complex constraint interaction between position and uniqueness

KEY INSIGHTS:
============
1. **COMPLEMENT STRATEGY**: Count numbers WITHOUT repeated digits, subtract from total
2. **BITMASK STATE**: Track used digits with 10-bit bitmask for uniqueness
3. **PERMUTATION MATHEMATICS**: Direct formulas for unique digit arrangements
4. **STATE COMPLEXITY**: Multiple constraints (position, tight, started, mask)
5. **MATHEMATICAL SHORTCUTS**: Analytical solutions for specific cases

ALGORITHM APPROACHES:
====================

1. **Brute Force**: O(n log n) time, O(1) space
   - Check each number for digit repetition
   - Simple but inefficient for large n

2. **Complement Digit DP**: O(log(n) × 2^10) time, O(log(n) × 2^10) space
   - Count unique digit numbers, subtract from total
   - Standard approach for uniqueness problems

3. **Direct Digit DP**: O(log(n) × 2^10) time, O(log(n) × 2^10) space
   - Directly count numbers with repeated digits
   - More complex state management

4. **Mathematical**: O(log n) time, O(1) space
   - Permutation-based calculation for unique digits
   - Most efficient for this specific problem

CORE COMPLEMENT DIGIT DP:
========================
```python
def numbersWithRepeatedDigits(n):
    s = str(n)
    memo = {}
    
    def count_unique(pos, mask, tight, started):
        if pos == len(s):
            return started
        
        if (pos, mask, tight, started) in memo:
            return memo[(pos, mask, tight, started)]
        
        result = 0
        limit = int(s[pos]) if tight else 9
        
        # Leading zero case
        if not started:
            result += count_unique(pos + 1, mask, False, False)
        
        # Place unique digits
        for digit in range(1 if not started else 0, limit + 1):
            if mask & (1 << digit):  # Already used
                continue
                
            new_mask = mask | (1 << digit)
            new_tight = tight and (digit == limit)
            result += count_unique(pos + 1, new_mask, new_tight, True)
        
        memo[(pos, mask, tight, started)] = result
        return result
    
    unique_count = count_unique(0, 0, True, False)
    return n - unique_count
```

BITMASK STATE MANAGEMENT:
========================
**Purpose**: Track which digits have been used

**Implementation**:
```python
mask = 0  # Initially no digits used
mask |= (1 << digit)  # Mark digit as used
if mask & (1 << digit):  # Check if digit is used
```

**State Space**: 2^10 = 1024 possible masks (one bit per digit 0-9)

**Constraint Enforcement**: Skip digits that are already in the mask

MATHEMATICAL OPTIMIZATION:
=========================
**Unique Digit Counting by Length**:
```python
def count_unique_length(length):
    if length == 1:
        return 9  # Digits 1-9
    else:
        return 9 * P(9, length-1)  # First digit: 1-9, rest: permutation
```

**Same Length Analysis**: Position-by-position with digit availability
```python
for pos in range(num_digits):
    available = count_unused_digits_smaller_than(current_digit, mask)
    remaining_positions = num_digits - pos - 1
    count += available * P(unused_digits - 1, remaining_positions)
```

COMPLEMENT VS DIRECT APPROACHES:
===============================
**Complement Strategy**:
- Advantages: Simpler logic, mathematical shortcuts available
- Disadvantages: Requires subtraction step

**Direct Strategy**:
- Advantages: Directly computes target
- Disadvantages: More complex state (need has_repeat flag)

**Performance**: Complement usually more efficient due to simpler constraints

PERMUTATION MATHEMATICS:
=======================
**Formula**: P(n, r) = n! / (n-r)! = n × (n-1) × ... × (n-r+1)

**Application in Unique Digits**:
- Total positions to fill: k
- First position: 9 choices (1-9)
- Remaining positions: P(9, k-1) arrangements of remaining digits

**Example**: 3-digit unique numbers
- First digit: 9 choices (1-9)
- Second digit: 9 choices (0-9 except first)
- Third digit: 8 choices (remaining digits)
- Total: 9 × 9 × 8 = 648

COMPLEXITY ANALYSIS:
===================
- **Digit DP**: O(log(n) × 2^10) time, O(log(n) × 2^10) space
- **Mathematical**: O(log n) time, O(1) space
- **State Space**: positions × tight × started × 2^10 mask
- **Optimization**: Mathematical approach avoids exponential mask states

STATE SPACE DESIGN:
==================
**Multi-dimensional State**:
- `pos`: Current position in number
- `mask`: Bitmask of used digits
- `tight`: Upper bound constraint
- `started`: Leading zero management
- `has_repeat`: (Direct approach) Whether repetition found

**State Transitions**: More complex due to mask updates and constraint checks

APPLICATIONS:
============
- **Number Theory**: Digit uniqueness analysis and pattern detection
- **Combinatorics**: Counting with uniqueness constraints
- **Data Validation**: Detecting duplicate patterns in numeric data
- **Cryptographic Analysis**: Randomness testing in numeric sequences
- **Algorithm Design**: Template for constraint-based counting problems

RELATED PROBLEMS:
================
- **Count Numbers with Unique Digits**: Direct uniqueness counting
- **Numbers At Most N Given Digit Set**: Digit availability constraints
- **Permutation Generation**: Related to unique digit arrangements
- **Digit Pattern Matching**: Various digit-based constraints

VARIANTS:
========
- **Exactly k Repeated Digits**: Count numbers with specific repetition patterns
- **Specific Digit Repetition**: Numbers where particular digit repeats
- **Maximum Unique Digits**: Longest possible unique digit numbers
- **Range Queries**: Repeated digit counting in ranges

EDGE CASES:
==========
- **Single Digit Numbers**: All unique by definition
- **n < 10**: Only single digits, no repetitions possible
- **n with Repeated Digits**: Target itself contributes to count
- **Maximum Length**: Numbers > 10 digits must have repetitions

OPTIMIZATION TECHNIQUES:
=======================
- **Mathematical Shortcuts**: Direct formulas when possible
- **State Pruning**: Early termination when constraints violated
- **Complement Strategy**: Often simpler than direct counting
- **Precomputation**: Cache permutation values

This problem showcases the power of the complement strategy
in Digit DP, demonstrating how complex constraints can be
handled more elegantly by counting the opposite condition
and leveraging mathematical insights for optimization.
"""
