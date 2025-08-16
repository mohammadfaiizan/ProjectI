"""
LeetCode 902: Numbers At Most N Given Digit Set
Difficulty: Hard
Category: Digit DP - Constrained Digit Selection

PROBLEM DESCRIPTION:
===================
Given an array of digits which is sorted in non-decreasing order. You can write numbers using each digits[i] as many times as we want. For example, if digits = ['1','3','5'], we can write numbers such as '13', '551', '1351315'.

Return the number of positive integers that can be written using the given digits that are less than or equal to n.

Example 1:
Input: digits = ["1","3","5"], n = 100
Output: 20
Explanation: 
The 20 numbers that can be written are:
1, 3, 5, 11, 13, 15, 31, 33, 35, 51, 53, 55, 111, 113, 115, 131, 133, 135, 151, 153, 155.

Example 2:
Input: digits = ["1","4","9"], n = 1000000000
Output: 29523

Example 3:
Input: digits = ["7"], n = 8
Output: 1

Constraints:
- 1 <= digits.length <= 9
- digits[i].length == 1
- digits[i] is a digit from '1' to '9'.
- All the values in digits are unique.
- digits is sorted in non-decreasing order.
- 1 <= n <= 10^9
"""


def at_most_n_given_digit_set_brute_force(digits, n):
    """
    BRUTE FORCE APPROACH:
    ====================
    Generate all possible numbers and count valid ones.
    
    Time Complexity: O(k^log(n)) where k = len(digits)
    Space Complexity: O(log(n)) - recursion depth
    """
    digit_set = set(int(d) for d in digits)
    count = 0
    
    def generate_numbers(current_num, target):
        nonlocal count
        
        if current_num > target:
            return
        
        if current_num > 0:
            count += 1
        
        for digit in digit_set:
            new_num = current_num * 10 + digit
            if new_num <= target:
                generate_numbers(new_num, target)
    
    generate_numbers(0, n)
    return count


def at_most_n_given_digit_set_digit_dp(digits, n):
    """
    DIGIT DP APPROACH:
    =================
    Use digit DP with constrained digit choices.
    
    Time Complexity: O(log(n) * k) where k = len(digits)
    Space Complexity: O(log(n)) - memoization
    """
    s = str(n)
    digit_set = sorted([int(d) for d in digits])
    memo = {}
    
    def dp(pos, tight, started):
        if pos == len(s):
            return 1 if started else 0
        
        if (pos, tight, started) in memo:
            return memo[(pos, tight, started)]
        
        result = 0
        
        # Option 1: Don't place any digit (leading zero)
        if not started:
            result += dp(pos + 1, False, False)
        
        # Option 2: Place a digit from the allowed set
        for digit in digit_set:
            if tight and digit > int(s[pos]):
                break
            
            new_tight = tight and (digit == int(s[pos]))
            result += dp(pos + 1, new_tight, True)
        
        memo[(pos, tight, started)] = result
        return result
    
    return dp(0, True, False)


def at_most_n_given_digit_set_mathematical(digits, n):
    """
    MATHEMATICAL APPROACH:
    =====================
    Count numbers by length and then handle the boundary case.
    
    Time Complexity: O(log(n) + k) where k = len(digits)
    Space Complexity: O(1) - constant space
    """
    if not digits:
        return 0
    
    digit_set = sorted([int(d) for d in digits])
    k = len(digit_set)
    s = str(n)
    n_digits = len(s)
    
    count = 0
    
    # Count numbers with fewer digits
    for length in range(1, n_digits):
        count += k ** length
    
    # Count numbers with same number of digits but <= n
    for pos in range(n_digits):
        # Count valid digits that are smaller than current digit of n
        smaller_count = 0
        for digit in digit_set:
            if digit < int(s[pos]):
                smaller_count += 1
            elif digit == int(s[pos]):
                break
            else:
                # All remaining digits are larger
                return count + smaller_count * (k ** (n_digits - pos - 1))
        
        count += smaller_count * (k ** (n_digits - pos - 1))
        
        # Check if current digit of n is in our digit set
        if int(s[pos]) not in digit_set:
            return count
    
    # If we reach here, n itself can be formed with given digits
    return count + 1


def at_most_n_given_digit_set_optimized(digits, n):
    """
    OPTIMIZED DIGIT DP:
    ==================
    Streamlined version focusing on essential states.
    
    Time Complexity: O(log(n) * k) - optimal for this problem
    Space Complexity: O(log(n)) - memoization
    """
    s = str(n)
    digit_ints = [int(d) for d in digits]
    memo = {}
    
    def dp(pos, tight, started):
        if pos == len(s):
            return started
        
        if (pos, tight, started) in memo:
            return memo[(pos, tight, started)]
        
        result = 0
        limit = int(s[pos]) if tight else 9
        
        # Skip this position (leading zero case)
        if not started:
            result += dp(pos + 1, False, False)
        
        # Use each available digit
        for digit in digit_ints:
            if digit > limit:
                break
            
            new_tight = tight and (digit == limit)
            result += dp(pos + 1, new_tight, True)
        
        memo[(pos, tight, started)] = result
        return result
    
    return dp(0, True, False)


def at_most_n_given_digit_set_with_path_tracking(digits, n):
    """
    DIGIT DP WITH PATH TRACKING:
    ============================
    Track some example valid numbers during construction.
    
    Time Complexity: O(log(n) * k) - standard digit DP
    Space Complexity: O(log(n)) - memoization + paths
    """
    s = str(n)
    digit_ints = [int(d) for d in digits]
    memo = {}
    example_numbers = []
    
    def dp(pos, tight, started, current_number=""):
        if pos == len(s):
            if started:
                if len(example_numbers) < 10:  # Limit examples
                    example_numbers.append(current_number)
                return 1
            return 0
        
        state = (pos, tight, started)
        if state in memo:
            return memo[state]
        
        result = 0
        
        # Skip position (leading zero)
        if not started:
            result += dp(pos + 1, False, False, current_number)
        
        # Use available digits
        for digit in digit_ints:
            if tight and digit > int(s[pos]):
                break
            
            new_tight = tight and (digit == int(s[pos]))
            new_number = current_number + str(digit)
            result += dp(pos + 1, new_tight, True, new_number)
        
        memo[state] = result
        return result
    
    count = dp(0, True, False)
    return count, example_numbers


def at_most_n_given_digit_set_analysis(digits, n):
    """
    COMPREHENSIVE ANALYSIS:
    ======================
    Analyze the digit constraint problem with detailed insights.
    """
    print(f"Numbers At Most N Given Digit Set Analysis:")
    print(f"Allowed digits: {digits}")
    print(f"Upper bound n: {n}")
    print(f"Number of available digits: {len(digits)}")
    
    digit_ints = sorted([int(d) for d in digits])
    print(f"Digits as integers: {digit_ints}")
    
    s = str(n)
    print(f"Target number '{n}' has {len(s)} digits: {list(s)}")
    
    # Different approaches
    if n <= 10000:  # Only for reasonable sizes
        try:
            brute_force = at_most_n_given_digit_set_brute_force(digits, n)
            print(f"Brute force result: {brute_force}")
        except:
            print("Brute force: Too large")
    
    digit_dp = at_most_n_given_digit_set_digit_dp(digits, n)
    mathematical = at_most_n_given_digit_set_mathematical(digits, n)
    optimized = at_most_n_given_digit_set_optimized(digits, n)
    
    print(f"Digit DP result: {digit_dp}")
    print(f"Mathematical result: {mathematical}")
    print(f"Optimized DP result: {optimized}")
    
    # Show examples
    count_with_examples, examples = at_most_n_given_digit_set_with_path_tracking(digits, n)
    print(f"\nExample valid numbers: {examples}")
    
    # Breakdown by number length
    k = len(digits)
    total_by_length = 0
    print(f"\nBreakdown by number length:")
    
    for length in range(1, len(s)):
        count_for_length = k ** length
        total_by_length += count_for_length
        print(f"  {length} digits: {count_for_length} numbers")
    
    same_length_count = digit_dp - total_by_length
    print(f"  {len(s)} digits: {same_length_count} numbers")
    print(f"  Total: {digit_dp} numbers")
    
    # Digit usage analysis
    print(f"\nDigit availability analysis:")
    for i, digit_char in enumerate(s):
        digit = int(digit_char)
        available = [d for d in digit_ints if d <= digit]
        print(f"  Position {i} (digit {digit}): {len(available)} choices <= {digit}")
    
    return digit_dp


def at_most_n_given_digit_set_variants():
    """
    DIGIT SET VARIANTS:
    ==================
    Different scenarios and modifications.
    """
    
    def exactly_k_digits(digits, k):
        """Count numbers with exactly k digits"""
        return len(digits) ** k if k > 0 else 0
    
    def at_least_n(digits, n):
        """Count numbers >= n (infinite upper bound)"""
        # This requires different approach - count by length
        if not digits or n <= 0:
            return 0
        
        # For infinite case, we can only count by considering
        # numbers with more digits than n
        s = str(n)
        result = 0
        
        # Numbers with more digits
        for length in range(len(s) + 1, len(s) + 5):  # Show first few
            result += len(digits) ** length
        
        return result  # This grows infinitely
    
    def digit_set_in_range(digits, start, end):
        """Count numbers in range [start, end]"""
        if start > end:
            return 0
        
        count_up_to_end = at_most_n_given_digit_set_digit_dp(digits, end)
        count_up_to_start = at_most_n_given_digit_set_digit_dp(digits, start - 1) if start > 1 else 0
        
        return count_up_to_end - count_up_to_start
    
    def min_max_numbers(digits, n):
        """Find minimum and maximum valid numbers <= n"""
        digit_ints = sorted([int(d) for d in digits])
        
        # Minimum: smallest digit repeated
        min_num = int(str(digit_ints[0]))
        
        # Maximum: construct greedily
        s = str(n)
        max_num = 0
        
        for pos in range(len(s)):
            best_digit = None
            for digit in reversed(digit_ints):
                if digit <= int(s[pos]):
                    best_digit = digit
                    break
            
            if best_digit is None:
                # Need to use smaller number of digits
                if pos > 0:
                    max_num = int(str(digit_ints[-1]) * (pos))
                break
            
            max_num = max_num * 10 + best_digit
            
            if digit < int(s[pos]):
                # Can use largest digit for remaining positions
                remaining = len(s) - pos - 1
                max_num = max_num * (10 ** remaining) + int(str(digit_ints[-1]) * remaining)
                break
        
        return min_num, min(max_num, n)
    
    # Test variants
    test_cases = [
        (["1", "3", "5"], 100),
        (["1", "4", "9"], 1000),
        (["7"], 8),
        (["2", "4", "6", "8"], 500)
    ]
    
    print("Digit Set Variants:")
    print("=" * 50)
    
    for digits, n in test_cases:
        print(f"\nDigits: {digits}, n = {n}")
        
        basic_count = at_most_n_given_digit_set_digit_dp(digits, n)
        print(f"Basic count <= {n}: {basic_count}")
        
        # Numbers with specific lengths
        for k in range(1, 5):
            k_digit_count = exactly_k_digits(digits, k)
            print(f"Exactly {k} digits: {k_digit_count}")
        
        # Range example
        if n >= 50:
            range_count = digit_set_in_range(digits, 50, n)
            print(f"In range [50, {n}]: {range_count}")
        
        # Min/max
        min_num, max_num = min_max_numbers(digits, n)
        print(f"Min valid: {min_num}, Max valid <= {n}: {max_num}")


# Test cases
def test_at_most_n_given_digit_set():
    """Test all implementations with various inputs"""
    test_cases = [
        (["1","3","5"], 100, 20),
        (["1","4","9"], 1000000000, 29523),
        (["7"], 8, 1),
        (["1"], 11, 2),
        (["1","2"], 12, 6),
        (["1","3","5"], 155, 22),
        (["2","4","8"], 200, 6),
        (["1","9"], 19, 4)
    ]
    
    print("Testing Numbers At Most N Given Digit Set Solutions:")
    print("=" * 70)
    
    for i, (digits, n, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"digits = {digits}, n = {n}")
        print(f"Expected: {expected}")
        
        # Skip brute force for large numbers
        if n <= 1000:
            try:
                brute_force = at_most_n_given_digit_set_brute_force(digits, n)
                print(f"Brute Force:      {brute_force:>6} {'✓' if brute_force == expected else '✗'}")
            except:
                print(f"Brute Force:      Timeout")
        
        digit_dp = at_most_n_given_digit_set_digit_dp(digits, n)
        mathematical = at_most_n_given_digit_set_mathematical(digits, n)
        optimized = at_most_n_given_digit_set_optimized(digits, n)
        
        print(f"Digit DP:         {digit_dp:>6} {'✓' if digit_dp == expected else '✗'}")
        print(f"Mathematical:     {mathematical:>6} {'✓' if mathematical == expected else '✗'}")
        print(f"Optimized:        {optimized:>6} {'✓' if optimized == expected else '✗'}")
        
        # Show examples for small cases
        if n <= 200:
            count, examples = at_most_n_given_digit_set_with_path_tracking(digits, n)
            print(f"Example numbers: {examples}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    at_most_n_given_digit_set_analysis(["1", "3", "5"], 100)
    
    # Variants demonstration
    print(f"\n" + "=" * 70)
    at_most_n_given_digit_set_variants()
    
    # Performance comparison
    print(f"\n" + "=" * 70)
    print("PERFORMANCE COMPARISON:")
    large_cases = [
        (["1", "2", "3"], 10**6),
        (["1", "4", "7"], 10**7),
        (["2", "5", "8"], 10**8)
    ]
    
    for digits, n in large_cases:
        print(f"\ndigits = {digits}, n = {n:,}")
        
        import time
        
        start = time.time()
        math_result = at_most_n_given_digit_set_mathematical(digits, n)
        math_time = time.time() - start
        
        start = time.time()
        dp_result = at_most_n_given_digit_set_digit_dp(digits, n)
        dp_time = time.time() - start
        
        print(f"Mathematical: {math_result:,} (Time: {math_time:.6f}s)")
        print(f"Digit DP:     {dp_result:,} (Time: {dp_time:.6f}s)")
        print(f"Match: {'✓' if math_result == dp_result else '✗'}")
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. DIGIT CONSTRAINTS: Only specific digits allowed for construction")
    print("2. LENGTH COUNTING: Numbers with fewer digits contribute k^length")
    print("3. BOUNDARY HANDLING: Careful analysis needed for same-length numbers")
    print("4. LEADING ZEROS: Handle valid number construction properly")
    print("5. MATHEMATICAL OPTIMIZATION: Direct formula for specific cases")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Number Theory: Constrained number generation and counting")
    print("• Combinatorics: Counting with digit restrictions")
    print("• Competitive Programming: Classic constrained counting problem")
    print("• Cryptography: Analyzing numeric patterns with constraints")
    print("• Computer Science: Template for constraint-based enumeration")


if __name__ == "__main__":
    test_at_most_n_given_digit_set()


"""
NUMBERS AT MOST N GIVEN DIGIT SET - CONSTRAINED DIGIT DP:
=========================================================

This problem demonstrates advanced Digit DP with digit restrictions:
- Limited digit choices during number construction
- Mathematical optimization for length-based counting
- Boundary case handling for numbers with same digit count
- Efficient counting under multiple constraints

KEY INSIGHTS:
============
1. **DIGIT CONSTRAINTS**: Only specific digits allowed for number construction
2. **LENGTH SEPARATION**: Count by number length for mathematical optimization
3. **BOUNDARY ANALYSIS**: Careful handling of numbers with same length as target
4. **TIGHT BOUND INTERACTION**: Digit constraints interact with upper bound
5. **MATHEMATICAL SHORTCUT**: Direct formula for complete length classes

ALGORITHM APPROACHES:
====================

1. **Brute Force**: O(k^log(n)) time, O(log(n)) space
   - Generate all possible numbers recursively
   - Exponential time complexity

2. **Digit DP**: O(log(n) × k) time, O(log(n)) space
   - Standard digit DP with constrained choices
   - Optimal for general digit constraint problems

3. **Mathematical**: O(log(n)) time, O(1) space
   - Length-based counting + boundary analysis
   - Most efficient for this specific pattern

4. **Optimized DP**: O(log(n) × k) time, O(log(n)) space
   - Streamlined digit DP implementation

CORE CONSTRAINED DIGIT DP:
==========================
```python
def atMostNGivenDigitSet(digits, n):
    s = str(n)
    digit_ints = [int(d) for d in digits]
    memo = {}
    
    def dp(pos, tight, started):
        if pos == len(s):
            return started  # Count only if we've built a valid number
        
        if (pos, tight, started) in memo:
            return memo[(pos, tight, started)]
        
        result = 0
        
        # Skip position (for leading zeros)
        if not started:
            result += dp(pos + 1, False, False)
        
        # Use each available digit
        for digit in digit_ints:
            if tight and digit > int(s[pos]):
                break
            
            new_tight = tight and (digit == int(s[pos]))
            result += dp(pos + 1, new_tight, True)
        
        memo[(pos, tight, started)] = result
        return result
    
    return dp(0, True, False)
```

MATHEMATICAL OPTIMIZATION:
=========================
**Length-based Counting**: For numbers with fewer digits than n:
```python
count = 0
k = len(digits)
n_digits = len(str(n))

# Count all numbers with length < n_digits
for length in range(1, n_digits):
    count += k ** length
```

**Same Length Analysis**: For numbers with same digit count:
- Position-by-position analysis
- Count digits smaller than current position
- Handle boundary where digit equals current position

**Combined Formula**:
```python
total = sum(k^i for i in range(1, n_digits)) + boundary_count
```

DIGIT CONSTRAINT HANDLING:
==========================
**Available Choices**: At each position, filter digits by:
1. **Set Membership**: Digit must be in allowed set
2. **Upper Bound**: If tight, digit ≤ current digit of n
3. **Ordering**: Process in sorted order for efficiency

**Implementation**:
```python
for digit in sorted_digit_set:
    if tight and digit > int(s[pos]):
        break  # All remaining digits are too large
    # Process this digit choice
```

LEADING ZERO MANAGEMENT:
=======================
**Problem**: Distinguish between actual zeros and leading zeros

**Solution**: Use `started` flag
- `started=False`: Can skip positions (leading zeros)
- `started=True`: Must count this as valid number

**Impact**: Ensures proper counting of numbers with different lengths

BOUNDARY CASE ANALYSIS:
======================
**Challenge**: Numbers with same digit count as target

**Approach**: Position-by-position analysis
1. **Smaller Digits**: Count all combinations with remaining positions
2. **Equal Digit**: Continue to next position if digit is available
3. **Larger Digits**: Stop processing (all invalid)

**Mathematical Version**:
```python
for pos in range(n_digits):
    smaller_count = sum(1 for d in digit_set if d < int(s[pos]))
    count += smaller_count * (k ** (n_digits - pos - 1))
    
    if int(s[pos]) not in digit_set:
        break  # Cannot continue building n
```

COMPLEXITY ANALYSIS:
===================
- **Digit DP**: O(log(n) × k) time, O(log(n)) space
- **Mathematical**: O(log(n)) time, O(1) space
- **State Space**: O(log(n)) positions × 2 tight × 2 started
- **Transitions**: Up to k digit choices per state

OPTIMIZATION TECHNIQUES:
=======================
**Early Termination**: In mathematical approach, stop when digit not available
**Precomputation**: Sort digits for efficient boundary checking
**State Minimization**: Combine related states when possible
**Mathematical Shortcuts**: Direct formulas for complete length classes

APPLICATIONS:
============
- **Number Theory**: Constrained number generation and analysis
- **Combinatorics**: Counting with digit restrictions
- **Competitive Programming**: Template for digit-based constraints
- **Cryptography**: Pattern analysis with restricted character sets
- **Computer Science**: Constraint satisfaction in numeric domains

RELATED PROBLEMS:
================
- **Count Numbers with Unique Digits**: Uniqueness constraint
- **Numbers With Repeated Digits**: Opposite constraint (repetition required)
- **Find All Good Strings**: String version with pattern constraints
- **Digit Sum Problems**: Additional arithmetic constraints

VARIANTS:
========
- **Exact Length**: Numbers with exactly k digits
- **Range Queries**: Count in range [L, R]
- **Multiple Constraints**: Combine digit and arithmetic constraints
- **Pattern Matching**: Specific digit patterns or sequences

EDGE CASES:
==========
- **Single Digit Set**: Simplifies to length-based counting
- **No Valid Digits**: Empty result set
- **Target Smaller than Min Digit**: Only shorter numbers valid
- **All Digits Available**: Reduces to standard range counting

MATHEMATICAL INSIGHTS:
=====================
**Exponential Growth**: k^length terms dominate for large lengths
**Boundary Dominance**: Same-length numbers often minority of total
**Digit Distribution**: Uniform vs. skewed digit availability affects patterns
**Combinatorial Structure**: Classic "words with restricted alphabet" problem

This problem showcases how digit constraints can be efficiently
handled through both algorithmic (Digit DP) and mathematical
approaches, demonstrating the power of constraint-aware
enumeration in combinatorial problems.
"""
