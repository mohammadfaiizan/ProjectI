"""
LeetCode 357: Count Numbers with Unique Digits
Difficulty: Medium
Category: Digit DP - Uniqueness Counting

PROBLEM DESCRIPTION:
===================
Given an integer n, return the count of all numbers with unique digits, x, where 0 <= x < 10^n.

Example 1:
Input: n = 2
Output: 91
Explanation: The answer should be the total numbers in the range of 0 ≤ x < 100, excluding 11,22,33,44,55,66,77,88,99

Example 2:
Input: n = 0
Output: 1

Constraints:
- 0 <= n <= 8
"""


def count_numbers_with_unique_digits_brute_force(n):
    """
    BRUTE FORCE APPROACH:
    ====================
    Check each number for unique digits.
    
    Time Complexity: O(10^n * n) - check each number
    Space Complexity: O(1) - constant space
    """
    if n == 0:
        return 1
    
    upper_limit = 10 ** n
    count = 0
    
    for i in range(upper_limit):
        digits = str(i)
        if len(digits) == len(set(digits)):
            count += 1
    
    return count


def count_numbers_with_unique_digits_mathematical(n):
    """
    MATHEMATICAL APPROACH:
    =====================
    Use combinatorial mathematics to count directly.
    
    Time Complexity: O(n) - calculate for each length
    Space Complexity: O(1) - constant space
    """
    if n == 0:
        return 1
    if n > 10:
        return 0  # Can't have more than 10 unique digits
    
    count = 1  # Count for 0
    
    # Count numbers with exactly k digits (k from 1 to n)
    for k in range(1, n + 1):
        if k == 1:
            count += 9  # 1-9
        else:
            # First digit: 9 choices (1-9)
            # Remaining digits: permutations of remaining 9 digits
            unique_for_k = 9
            for i in range(1, k):
                unique_for_k *= (10 - i)
            count += unique_for_k
    
    return count


def count_numbers_with_unique_digits_digit_dp(n):
    """
    DIGIT DP APPROACH:
    =================
    Use digit DP with bitmask for uniqueness tracking.
    
    Time Complexity: O(n * 2^10) - states with bitmask
    Space Complexity: O(n * 2^10) - memoization
    """
    if n == 0:
        return 1
    
    memo = {}
    
    def dp(pos, mask, started, max_pos):
        if pos > max_pos:
            return 1 if started else 0
        
        if (pos, mask, started, max_pos) in memo:
            return memo[(pos, mask, started, max_pos)]
        
        result = 0
        
        # Option 1: Don't place digit (leading zero)
        if not started:
            result += dp(pos + 1, mask, False, max_pos)
        
        # Option 2: Place a digit
        start_digit = 1 if not started else 0
        for digit in range(start_digit, 10):
            if mask & (1 << digit):  # Already used
                continue
            
            new_mask = mask | (1 << digit)
            result += dp(pos + 1, new_mask, True, max_pos)
        
        memo[(pos, mask, started, max_pos)] = result
        return result
    
    total = 0
    for length in range(1, n + 1):
        total += dp(1, 0, False, length)
    
    return total + 1  # +1 for number 0


def count_numbers_with_unique_digits_optimized(n):
    """
    OPTIMIZED MATHEMATICAL APPROACH:
    ===============================
    Direct calculation with permutation formulas.
    
    Time Complexity: O(n) - linear calculation
    Space Complexity: O(1) - constant space
    """
    if n == 0:
        return 1
    if n > 10:
        return 0
    
    # Start with single digit numbers (0-9), but 0 is handled separately
    count = 10  # 0, 1, 2, ..., 9
    
    # For k-digit numbers where k >= 2
    available_digits = 9  # First digit choices (1-9)
    current_count = 9     # Numbers with exactly 1 unique digit
    
    for k in range(2, n + 1):
        available_digits -= 1  # One less choice for next position
        current_count *= available_digits
        count += current_count
    
    return count


def count_numbers_with_unique_digits_analysis(n):
    """
    COMPREHENSIVE ANALYSIS:
    ======================
    Analyze unique digit counting patterns.
    """
    print(f"Count Numbers with Unique Digits Analysis for n = {n}:")
    print(f"Range: [0, 10^{n}) = [0, {10**n if n <= 8 else 'very large'})")
    
    if n > 10:
        print("n > 10: Impossible to have unique digits (only 10 digits available)")
        return 0
    
    # Different approaches
    if n <= 5:  # Only for reasonable sizes
        try:
            brute_force = count_numbers_with_unique_digits_brute_force(n)
            print(f"Brute force result: {brute_force}")
        except:
            print("Brute force: Too large")
    
    mathematical = count_numbers_with_unique_digits_mathematical(n)
    digit_dp = count_numbers_with_unique_digits_digit_dp(n)
    optimized = count_numbers_with_unique_digits_optimized(n)
    
    print(f"Mathematical result: {mathematical}")
    print(f"Digit DP result: {digit_dp}")
    print(f"Optimized result: {optimized}")
    
    # Breakdown by number length
    print(f"\nBreakdown by number length:")
    total_breakdown = 0
    
    for k in range(1, n + 1):
        if k == 1:
            count_k = 9
        else:
            count_k = 9
            for i in range(1, k):
                count_k *= (10 - i)
        
        total_breakdown += count_k
        print(f"  {k} digits: {count_k} numbers")
    
    print(f"  0 (special): 1 number")
    total_breakdown += 1
    print(f"  Total: {total_breakdown}")
    
    # Pattern analysis
    print(f"\nPattern Analysis:")
    print(f"Single digits (1-9): 9 numbers")
    print(f"Two digits: 9 × 9 = 81 numbers")
    print(f"Three digits: 9 × 9 × 8 = 648 numbers")
    if n >= 4:
        print(f"Four digits: 9 × 9 × 8 × 7 = 4536 numbers")
    
    # Maximum possible
    print(f"\nMaximum Analysis:")
    max_unique_length = min(n, 10)
    max_possible = count_numbers_with_unique_digits_mathematical(max_unique_length)
    print(f"Maximum unique digit numbers for length ≤ {max_unique_length}: {max_possible}")
    
    return mathematical


def count_numbers_with_unique_digits_variants():
    """
    UNIQUE DIGITS VARIANTS:
    ======================
    Different scenarios and modifications.
    """
    
    def count_with_exactly_k_digits(n, k):
        """Count n-digit numbers with exactly k unique digits"""
        if k > n or k > 10 or k <= 0:
            return 0
        
        if k == n:
            # All digits must be unique
            if n == 1:
                return 9  # 1-9
            else:
                result = 9
                for i in range(1, n):
                    result *= (10 - i)
                return result
        
        # This is more complex - simplified approximation
        return 0  # Would need inclusion-exclusion principle
    
    def count_with_at_least_k_unique(n, k):
        """Count numbers with at least k unique digits"""
        if k > n or k > 10:
            return 0
        
        total = count_numbers_with_unique_digits_mathematical(n)
        if k <= 1:
            return total
        
        # Subtract numbers with fewer than k unique digits
        # This requires more complex calculation
        return total  # Simplified
    
    def largest_unique_digit_number(n):
        """Find largest number with unique digits having at most n digits"""
        if n <= 0:
            return 0
        if n >= 10:
            return 9876543210
        
        # Use largest digits in descending order
        digits = list(range(9, 9 - n, -1))
        return int(''.join(map(str, digits)))
    
    def smallest_unique_digit_number(n):
        """Find smallest n-digit number with unique digits"""
        if n <= 0:
            return 0
        if n == 1:
            return 1
        if n > 10:
            return -1  # Impossible
        
        # Start with 1, then use smallest available digits
        digits = [1] + list(range(0, n - 1))
        return int(''.join(map(str, digits)))
    
    # Test variants
    test_values = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    
    print("Unique Digits Variants:")
    print("=" * 50)
    
    for n in test_values:
        print(f"\nn = {n}")
        
        basic_count = count_numbers_with_unique_digits_mathematical(n)
        print(f"Total unique digit numbers: {basic_count}")
        
        if n > 0:
            largest = largest_unique_digit_number(n)
            smallest = smallest_unique_digit_number(n)
            print(f"Largest {n}-digit unique: {largest}")
            print(f"Smallest {n}-digit unique: {smallest}")
        
        # Examples for small n
        if n <= 3:
            at_least_2 = count_with_at_least_k_unique(n, 2)
            print(f"At least 2 unique digits: {at_least_2}")


# Test cases
def test_count_numbers_with_unique_digits():
    """Test all implementations with various inputs"""
    test_cases = [
        (0, 1),
        (1, 10),
        (2, 91),
        (3, 739),
        (4, 5275),
        (5, 32491),
        (6, 168571),
        (7, 712891),
        (8, 2345851)
    ]
    
    print("Testing Count Numbers with Unique Digits Solutions:")
    print("=" * 70)
    
    for i, (n, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"n = {n}")
        print(f"Expected: {expected}")
        
        # Skip brute force for large numbers
        if n <= 4:
            try:
                brute_force = count_numbers_with_unique_digits_brute_force(n)
                print(f"Brute Force:      {brute_force:>8} {'✓' if brute_force == expected else '✗'}")
            except:
                print(f"Brute Force:      Timeout")
        
        mathematical = count_numbers_with_unique_digits_mathematical(n)
        digit_dp = count_numbers_with_unique_digits_digit_dp(n)
        optimized = count_numbers_with_unique_digits_optimized(n)
        
        print(f"Mathematical:     {mathematical:>8} {'✓' if mathematical == expected else '✗'}")
        print(f"Digit DP:         {digit_dp:>8} {'✓' if digit_dp == expected else '✗'}")
        print(f"Optimized:        {optimized:>8} {'✓' if optimized == expected else '✗'}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    count_numbers_with_unique_digits_analysis(3)
    
    # Variants demonstration
    print(f"\n" + "=" * 70)
    count_numbers_with_unique_digits_variants()
    
    # Pattern verification
    print(f"\n" + "=" * 70)
    print("PATTERN VERIFICATION:")
    print("Mathematical pattern: count(n) = count(n-1) + 9 × P(9, n-1)")
    
    for n in range(1, 6):
        if n == 1:
            count_n = 10  # 0-9
        else:
            count_n_minus_1 = count_numbers_with_unique_digits_mathematical(n - 1)
            
            # Calculate 9 × P(9, n-1)
            perm_9_n_minus_1 = 9
            for i in range(1, n):
                perm_9_n_minus_1 *= (10 - i)
            
            count_n = count_n_minus_1 + perm_9_n_minus_1
        
        actual = count_numbers_with_unique_digits_mathematical(n)
        print(f"n={n}: calculated={count_n}, actual={actual}, match={'✓' if count_n == actual else '✗'}")
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. PERMUTATION COUNTING: Use P(n,r) for arranging unique digits")
    print("2. LENGTH SEPARATION: Count by number of digits separately")
    print("3. FIRST DIGIT CONSTRAINT: Cannot be 0 for multi-digit numbers")
    print("4. COMBINATORIAL OPTIMIZATION: Direct formula avoids DP overhead")
    print("5. UPPER BOUND: Maximum 10 unique digits available")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Combinatorics: Counting arrangements with uniqueness constraints")
    print("• Number Theory: Digit pattern analysis and enumeration")
    print("• Mathematics: Permutation and combination applications")
    print("• Computer Science: Constraint-based counting problems")
    print("• Algorithm Design: Template for uniqueness-based enumeration")


if __name__ == "__main__":
    test_count_numbers_with_unique_digits()


"""
COUNT NUMBERS WITH UNIQUE DIGITS - COMBINATORIAL DIGIT COUNTING:
================================================================

This problem demonstrates pure combinatorial counting for digit uniqueness:
- Mathematical permutation-based solutions
- Length-based counting strategies
- Constraint handling for leading zeros
- Optimal mathematical formulation avoiding complex DP

KEY INSIGHTS:
============
1. **PERMUTATION COUNTING**: Use P(n,r) = n!/(n-r)! for unique digit arrangements
2. **LENGTH SEPARATION**: Count numbers by digit length for mathematical clarity
3. **LEADING ZERO CONSTRAINT**: First digit cannot be 0 for multi-digit numbers
4. **COMBINATORIAL OPTIMIZATION**: Direct mathematical formula more efficient than DP
5. **UPPER BOUND RECOGNITION**: Maximum 10 unique digits limits the problem scope

ALGORITHM APPROACHES:
====================

1. **Brute Force**: O(10^n × n) time, O(1) space
   - Check each number for digit uniqueness
   - Only viable for very small n

2. **Mathematical**: O(n) time, O(1) space
   - Direct combinatorial calculation
   - Most efficient and elegant approach

3. **Digit DP**: O(n × 2^10) time, O(n × 2^10) space
   - Standard digit DP with bitmask
   - Overkill for this specific problem

4. **Optimized Mathematical**: O(n) time, O(1) space
   - Streamlined permutation calculation

CORE MATHEMATICAL SOLUTION:
==========================
```python
def countNumbersWithUniqueDigits(n):
    if n == 0:
        return 1
    if n > 10:
        return 0  # Impossible with only 10 digits
    
    count = 10  # Single digit numbers: 0-9
    
    # For k-digit numbers (k >= 2)
    available = 9   # First digit choices: 1-9
    current = 9     # Current k-digit count
    
    for k in range(2, n + 1):
        available -= 1
        current *= available
        count += current
    
    return count
```

COMBINATORIAL ANALYSIS:
======================
**Single Digit (k=1)**: 10 numbers (0-9)

**Multi-digit (k≥2)**:
- **First digit**: 9 choices (1-9, cannot be 0)
- **Second digit**: 9 choices (0-9 except first digit)
- **Third digit**: 8 choices (remaining digits)
- **k-th digit**: (10-k+1) choices

**Formula for k digits**: 9 × P(9, k-1) = 9 × 9!/(9-(k-1))!

PERMUTATION MATHEMATICS:
=======================
**Permutation Formula**: P(n,r) = n!/(n-r)! = n×(n-1)×...×(n-r+1)

**Application**:
- Choose first digit: 9 ways (1-9)
- Arrange remaining k-1 digits from 9 available: P(9, k-1)

**Efficient Calculation**:
```python
def calculate_k_digit_unique(k):
    if k == 1:
        return 9
    
    result = 9  # First digit
    for i in range(1, k):
        result *= (10 - i)  # Remaining positions
    return result
```

LENGTH-BASED COUNTING STRATEGY:
==============================
**Separation by Length**: Count separately for each possible length
```
count(n) = count(1-digit) + count(2-digit) + ... + count(n-digit) + count(0)
```

**Cumulative Formula**:
```
Total = 1 + 9 + 9×9 + 9×9×8 + 9×9×8×7 + ... + 9×P(9,n-1)
```

LEADING ZERO CONSTRAINT:
=======================
**Problem**: Multi-digit numbers cannot start with 0

**Solution**: 
- Handle 0 as special case (count = 1)
- For k≥1: first digit has 9 choices (1-9)
- Remaining digits: permutation of available digits

**Mathematical Impact**: 
- Single digit: 9 valid (1-9) + 1 special (0) = 10
- Multi-digit: first digit constraint reduces choices

UPPER BOUND ANALYSIS:
====================
**Digit Limitation**: Only 10 unique digits (0-9) available

**Implications**:
- n > 10: Result = 0 (impossible)
- n = 10: Maximum possible unique digit numbers
- Practical limit makes brute force viable for verification

**Maximum Values**:
- Largest 10-digit unique: 9876543210
- Smallest k-digit unique: 1023...k-1

OPTIMIZATION TECHNIQUES:
=======================
**Avoid Factorial Calculation**: Use iterative multiplication
```python
result = 9
for i in range(1, k):
    result *= (10 - i)
```

**Early Termination**: Return 0 for n > 10

**Precomputation**: For multiple queries, precompute permutation values

MATHEMATICAL VERIFICATION:
=========================
**Recurrence Check**: count(n) = count(n-1) + 9×P(9,n-1)
**Direct Formula**: Sum of permutation terms
**Boundary Verification**: Special cases n=0,1,10

COMPLEXITY COMPARISON:
=====================
- **Mathematical**: O(n) time, O(1) space - optimal
- **Digit DP**: O(n×2^10) time - unnecessary complexity
- **Brute Force**: O(10^n×n) time - exponential

APPLICATIONS:
============
- **Combinatorics**: Pure permutation counting problems
- **Number Theory**: Digit arrangement analysis
- **Mathematics Education**: Permutation and combination examples
- **Algorithm Design**: Template for mathematical optimization over DP
- **Constraint Satisfaction**: Counting with uniqueness requirements

RELATED PROBLEMS:
================
- **Numbers With Repeated Digits**: Complement problem
- **Permutation Generation**: Related arrangement problems
- **Digit Pattern Counting**: Various digit-based constraints
- **Combinatorial Enumeration**: Counting with restrictions

VARIANTS:
========
- **Exactly k Unique Digits**: Count numbers with specific uniqueness level
- **Range Queries**: Unique digit counts in arbitrary ranges
- **Base Conversion**: Unique digits in different number bases
- **Pattern Constraints**: Additional restrictions on digit arrangements

EDGE CASES:
==========
- **n = 0**: Special case, return 1
- **n = 1**: Include single digits 0-9
- **n > 10**: Impossible, return 0
- **Large n**: Mathematical approach scales linearly

This problem beautifully demonstrates when mathematical insight
can completely eliminate the need for complex DP algorithms,
showing how combinatorial analysis can provide elegant and
efficient solutions for counting problems with clear structure.
"""
