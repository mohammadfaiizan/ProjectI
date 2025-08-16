"""
LeetCode 600: Non-negative Integers without Consecutive Ones
Difficulty: Hard
Category: Digit DP - Pattern Constraints

PROBLEM DESCRIPTION:
===================
Given a positive integer n, return the number of the integers in the range [0, n] whose binary representations do not contain consecutive 1s.

Example 1:
Input: n = 5
Output: 5
Explanation: Here are the non-negative integers <= 5 with their corresponding binary representations:
0 : 0
1 : 1
2 : 10
3 : 11 (contains consecutive 1s)
4 : 100
5 : 101
So the answer is 5.

Example 2:
Input: n = 1
Output: 2

Example 3:
Input: n = 2
Output: 3

Constraints:
- 1 <= n <= 10^9
"""


def find_integers_without_consecutive_ones_brute_force(n):
    """
    BRUTE FORCE APPROACH:
    ====================
    Check each number for consecutive 1s in binary representation.
    
    Time Complexity: O(n * log(n)) - check each number
    Space Complexity: O(1) - constant space
    """
    if n < 0:
        return 0
    
    count = 0
    for i in range(n + 1):
        binary = bin(i)[2:]  # Remove '0b' prefix
        has_consecutive = False
        
        for j in range(len(binary) - 1):
            if binary[j] == '1' and binary[j + 1] == '1':
                has_consecutive = True
                break
        
        if not has_consecutive:
            count += 1
    
    return count


def find_integers_without_consecutive_ones_digit_dp(n):
    """
    DIGIT DP APPROACH:
    =================
    Use digit DP on binary representation to avoid consecutive 1s.
    
    Time Complexity: O(log(n)^2) - binary digits with state
    Space Complexity: O(log(n)) - memoization
    """
    if n < 0:
        return 0
    
    binary = bin(n)[2:]  # Binary representation without '0b'
    memo = {}
    
    def dp(pos, tight, prev_bit, started):
        """
        pos: current position in binary string
        tight: whether bounded by n
        prev_bit: previous bit placed (0 or 1)
        started: whether we've started placing bits
        """
        if pos == len(binary):
            return 1 if started else 0  # Don't count empty number
        
        if (pos, tight, prev_bit, started) in memo:
            return memo[(pos, tight, prev_bit, started)]
        
        limit = int(binary[pos]) if tight else 1
        result = 0
        
        # Option 1: Place 0
        result += dp(pos + 1, tight and (0 == limit), 0, started)
        
        # Option 2: Place 1 (only if previous bit wasn't 1)
        if prev_bit != 1 and limit >= 1:
            result += dp(pos + 1, tight and (1 == limit), 1, True)
        
        memo[(pos, tight, prev_bit, started)] = result
        return result
    
    # Include the number 0 separately
    return dp(0, True, 0, False) + 1  # +1 for number 0


def find_integers_without_consecutive_ones_fibonacci(n):
    """
    FIBONACCI APPROACH:
    ==================
    Use Fibonacci-like pattern for counting valid binary strings.
    
    Time Complexity: O(log(n)) - process binary digits
    Space Complexity: O(log(n)) - Fibonacci array
    """
    if n < 0:
        return 0
    
    binary = bin(n)[2:]
    k = len(binary)
    
    # Fibonacci-like sequence for counting valid strings
    # f[i] = number of valid strings of length i
    f = [0] * (k + 2)
    f[0] = 1  # Empty string
    f[1] = 2  # "0", "1"
    
    for i in range(2, k + 2):
        f[i] = f[i - 1] + f[i - 2]
    
    count = 0
    prev_bit = 0
    
    for i in range(k):
        if binary[i] == '1':
            # Add count of valid strings starting with 0 at this position
            count += f[k - i]
            
            # Check for consecutive 1s
            if prev_bit == 1:
                # Can't continue - all remaining would have consecutive 1s
                return count
            
            prev_bit = 1
        else:
            prev_bit = 0
    
    # If we reach here, n itself is valid
    return count + 1


def find_integers_without_consecutive_ones_optimized_dp(n):
    """
    OPTIMIZED DIGIT DP:
    ==================
    Streamlined version focusing on binary constraints.
    
    Time Complexity: O(log(n)) - binary digits
    Space Complexity: O(log(n)) - memoization
    """
    if n < 0:
        return 0
    
    binary = bin(n)[2:]
    memo = {}
    
    def dp(pos, tight, last_bit):
        if pos == len(binary):
            return 1
        
        if (pos, tight, last_bit) in memo:
            return memo[(pos, tight, last_bit)]
        
        limit = int(binary[pos]) if tight else 1
        result = 0
        
        # Place 0
        result += dp(pos + 1, tight and (0 == limit), 0)
        
        # Place 1 (if not consecutive)
        if last_bit == 0 and limit >= 1:
            result += dp(pos + 1, tight and (1 == limit), 1)
        
        memo[(pos, tight, last_bit)] = result
        return result
    
    return dp(0, True, 0)


def find_integers_without_consecutive_ones_mathematical(n):
    """
    MATHEMATICAL APPROACH:
    =====================
    Direct mathematical calculation using Fibonacci patterns.
    
    Time Complexity: O(log(n)) - binary length
    Space Complexity: O(1) - constant space
    """
    if n <= 0:
        return 1 if n == 0 else 0
    
    # Convert to binary and process
    binary = bin(n)[2:]
    k = len(binary)
    
    # Calculate Fibonacci numbers for binary string counting
    # fib[i] represents valid binary strings of length i ending with 0 or 1
    fib = [1, 2]  # fib[0]=1 (length 1: "0"), fib[1]=2 (length 1: "0","1")
    
    for i in range(2, k + 1):
        fib.append(fib[i - 1] + fib[i - 2])
    
    result = 0
    consecutive_found = False
    
    for i in range(k):
        if binary[i] == '1':
            # Add all valid numbers with 0 at this position
            result += fib[k - i - 1]
            
            # Check for consecutive 1s
            if i > 0 and binary[i - 1] == '1':
                consecutive_found = True
                break
    
    # Add 1 if n itself doesn't have consecutive 1s
    if not consecutive_found:
        result += 1
    
    return result


def find_integers_without_consecutive_ones_with_analysis(n):
    """
    DIGIT DP WITH DETAILED ANALYSIS:
    ===============================
    Track the computation process with detailed breakdown.
    
    Time Complexity: O(log(n)^2) - standard digit DP
    Space Complexity: O(log(n)) - memoization + analysis
    """
    if n < 0:
        return 0
    
    binary = bin(n)[2:]
    memo = {}
    analysis = {
        'binary_rep': binary,
        'length': len(binary),
        'valid_patterns': [],
        'total_calls': 0,
        'cache_hits': 0
    }
    
    def dp(pos, tight, last_bit, current_pattern=""):
        analysis['total_calls'] += 1
        
        if pos == len(binary):
            if current_pattern and len(analysis['valid_patterns']) < 20:
                analysis['valid_patterns'].append(current_pattern)
            return 1
        
        state = (pos, tight, last_bit)
        if state in memo:
            analysis['cache_hits'] += 1
            return memo[state]
        
        limit = int(binary[pos]) if tight else 1
        result = 0
        
        # Place 0
        result += dp(pos + 1, tight and (0 == limit), 0, current_pattern + "0")
        
        # Place 1 (if not consecutive)
        if last_bit == 0 and limit >= 1:
            result += dp(pos + 1, tight and (1 == limit), 1, current_pattern + "1")
        
        memo[state] = result
        return result
    
    count = dp(0, True, 0)
    return count, analysis


def find_integers_without_consecutive_ones_analysis(n):
    """
    COMPREHENSIVE ANALYSIS:
    ======================
    Analyze the consecutive ones problem with detailed insights.
    """
    print(f"Non-negative Integers without Consecutive Ones Analysis for n = {n}:")
    print(f"Range: [0, {n}]")
    
    binary_n = bin(n)[2:]
    print(f"Binary representation of {n}: {binary_n}")
    print(f"Binary length: {len(binary_n)}")
    
    # Check if n itself has consecutive 1s
    has_consecutive = '11' in binary_n
    print(f"Does {n} have consecutive 1s? {has_consecutive}")
    
    # Different approaches
    if n <= 10000:  # Only for reasonable sizes
        try:
            brute_force = find_integers_without_consecutive_ones_brute_force(n)
            print(f"Brute force result: {brute_force}")
        except:
            print("Brute force: Too large")
    
    digit_dp = find_integers_without_consecutive_ones_digit_dp(n)
    fibonacci = find_integers_without_consecutive_ones_fibonacci(n)
    optimized = find_integers_without_consecutive_ones_optimized_dp(n)
    mathematical = find_integers_without_consecutive_ones_mathematical(n)
    
    print(f"Digit DP result: {digit_dp}")
    print(f"Fibonacci result: {fibonacci}")
    print(f"Optimized DP result: {optimized}")
    print(f"Mathematical result: {mathematical}")
    
    # Detailed analysis
    count_with_analysis, analysis = find_integers_without_consecutive_ones_with_analysis(n)
    
    print(f"\nDetailed Analysis:")
    print(f"Total count: {count_with_analysis}")
    print(f"DP function calls: {analysis['total_calls']}")
    print(f"Cache hits: {analysis['cache_hits']}")
    print(f"Cache hit rate: {analysis['cache_hits']/analysis['total_calls']:.2%}")
    
    print(f"\nValid binary patterns (first 20):")
    for i, pattern in enumerate(analysis['valid_patterns'][:20]):
        decimal_val = int(pattern, 2) if pattern else 0
        print(f"  {pattern} → {decimal_val}")
    
    # Fibonacci pattern analysis
    print(f"\nFibonacci pattern analysis:")
    fib = [1, 2]
    for i in range(2, len(binary_n) + 1):
        fib.append(fib[i-1] + fib[i-2])
    
    print(f"Fibonacci sequence: {fib}")
    print(f"f({len(binary_n)}) = {fib[len(binary_n)]}")
    
    return count_with_analysis


def find_integers_without_consecutive_ones_variants():
    """
    CONSECUTIVE ONES VARIANTS:
    =========================
    Different scenarios and modifications.
    """
    
    def integers_without_k_consecutive_ones(n, k):
        """Count integers without k consecutive 1s"""
        if n < 0 or k <= 0:
            return 0
        
        binary = bin(n)[2:]
        memo = {}
        
        def dp(pos, tight, ones_count):
            if pos == len(binary):
                return 1
            
            if (pos, tight, ones_count) in memo:
                return memo[(pos, tight, ones_count)]
            
            limit = int(binary[pos]) if tight else 1
            result = 0
            
            # Place 0
            result += dp(pos + 1, tight and (0 == limit), 0)
            
            # Place 1 (if not k consecutive)
            if ones_count < k - 1 and limit >= 1:
                result += dp(pos + 1, tight and (1 == limit), ones_count + 1)
            
            memo[(pos, tight, ones_count)] = result
            return result
        
        return dp(0, True, 0)
    
    def count_by_binary_length(max_length):
        """Count valid numbers by binary length using Fibonacci"""
        fib = [1, 2]  # f(1)=2: "0","1"
        for i in range(2, max_length + 1):
            fib.append(fib[i-1] + fib[i-2])
        
        return fib
    
    def integers_with_exactly_k_ones(n, k):
        """Count integers with exactly k ones and no consecutive 1s"""
        binary = bin(n)[2:]
        memo = {}
        
        def dp(pos, tight, ones_placed, last_bit):
            if pos == len(binary):
                return 1 if ones_placed == k else 0
            
            state = (pos, tight, ones_placed, last_bit)
            if state in memo:
                return memo[state]
            
            limit = int(binary[pos]) if tight else 1
            result = 0
            
            # Place 0
            result += dp(pos + 1, tight and (0 == limit), ones_placed, 0)
            
            # Place 1
            if last_bit == 0 and ones_placed < k and limit >= 1:
                result += dp(pos + 1, tight and (1 == limit), ones_placed + 1, 1)
            
            memo[state] = result
            return result
        
        return dp(0, True, 0, 0)
    
    def max_number_without_consecutive_ones(binary_length):
        """Find maximum number without consecutive 1s for given length"""
        if binary_length <= 0:
            return 0
        
        # Greedy approach: 101010... pattern
        result = 0
        for i in range(binary_length):
            if i % 2 == 0:  # Place 1 at even positions (0-indexed)
                result |= (1 << (binary_length - 1 - i))
        
        return result
    
    # Test variants
    test_numbers = [5, 15, 31, 63, 100]
    
    print("Consecutive Ones Variants:")
    print("=" * 50)
    
    for n in test_numbers:
        print(f"\nn = {n} (binary: {bin(n)[2:]})")
        
        basic_count = find_integers_without_consecutive_ones_optimized_dp(n)
        print(f"Without consecutive 1s: {basic_count}")
        
        # k-consecutive variants
        for k in range(2, 5):
            k_count = integers_without_k_consecutive_ones(n, k)
            print(f"Without {k} consecutive 1s: {k_count}")
        
        # Exactly k ones
        binary_len = len(bin(n)[2:])
        for k in range(1, min(binary_len + 1, 4)):
            k_ones = integers_with_exactly_k_ones(n, k)
            print(f"Exactly {k} ones (no consecutive): {k_ones}")
    
    # Fibonacci pattern by length
    print(f"\nFibonacci pattern by binary length:")
    fib_counts = count_by_binary_length(10)
    for i, count in enumerate(fib_counts):
        if i > 0:
            max_num = max_number_without_consecutive_ones(i)
            print(f"  Length {i}: {count} numbers, max = {max_num} ({bin(max_num)[2:]})")


# Test cases
def test_find_integers_without_consecutive_ones():
    """Test all implementations with various inputs"""
    test_cases = [
        (5, 5),
        (1, 2),
        (2, 3),
        (0, 1),
        (3, 3),
        (4, 4),
        (7, 5),
        (8, 6),
        (15, 8),
        (31, 13)
    ]
    
    print("Testing Non-negative Integers without Consecutive Ones Solutions:")
    print("=" * 70)
    
    for i, (n, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"n = {n} (binary: {bin(n)[2:]})")
        print(f"Expected: {expected}")
        
        # Skip brute force for large numbers
        if n <= 1000:
            try:
                brute_force = find_integers_without_consecutive_ones_brute_force(n)
                print(f"Brute Force:      {brute_force:>4} {'✓' if brute_force == expected else '✗'}")
            except:
                print(f"Brute Force:      Timeout")
        
        digit_dp = find_integers_without_consecutive_ones_digit_dp(n)
        fibonacci = find_integers_without_consecutive_ones_fibonacci(n)
        optimized = find_integers_without_consecutive_ones_optimized_dp(n)
        mathematical = find_integers_without_consecutive_ones_mathematical(n)
        
        print(f"Digit DP:         {digit_dp:>4} {'✓' if digit_dp == expected else '✗'}")
        print(f"Fibonacci:        {fibonacci:>4} {'✓' if fibonacci == expected else '✗'}")
        print(f"Optimized:        {optimized:>4} {'✓' if optimized == expected else '✗'}")
        print(f"Mathematical:     {mathematical:>4} {'✓' if mathematical == expected else '✗'}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    find_integers_without_consecutive_ones_analysis(15)
    
    # Variants demonstration
    print(f"\n" + "=" * 70)
    find_integers_without_consecutive_ones_variants()
    
    # Performance comparison
    print(f"\n" + "=" * 70)
    print("PERFORMANCE COMPARISON:")
    large_numbers = [10**6, 10**7, 10**8]
    
    for n in large_numbers:
        print(f"\nn = {n:,} (binary length: {len(bin(n)[2:])})")
        
        import time
        
        start = time.time()
        math_result = find_integers_without_consecutive_ones_mathematical(n)
        math_time = time.time() - start
        
        start = time.time()
        fib_result = find_integers_without_consecutive_ones_fibonacci(n)
        fib_time = time.time() - start
        
        print(f"Mathematical: {math_result:,} (Time: {math_time:.6f}s)")
        print(f"Fibonacci:    {fib_result:,} (Time: {fib_time:.6f}s)")
        print(f"Match: {'✓' if math_result == fib_result else '✗'}")
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. BINARY DIGIT DP: Apply digit DP to binary representation")
    print("2. CONSECUTIVE CONSTRAINT: Track previous bit to avoid 11 pattern")
    print("3. FIBONACCI PATTERN: Valid binary strings follow Fibonacci sequence")
    print("4. MATHEMATICAL OPTIMIZATION: Direct calculation using Fibonacci")
    print("5. TIGHT BOUND: Binary upper bound constraint management")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Digital Design: Avoiding problematic bit patterns in circuits")
    print("• Coding Theory: Error-correcting codes with pattern constraints")
    print("• Combinatorics: Counting binary sequences with restrictions")
    print("• Algorithm Design: Template for pattern-based constraints")
    print("• Computer Science: Binary representation analysis and optimization")


if __name__ == "__main__":
    test_find_integers_without_consecutive_ones()


"""
NON-NEGATIVE INTEGERS WITHOUT CONSECUTIVE ONES - BINARY PATTERN DP:
===================================================================

This problem demonstrates Digit DP applied to binary representations:
- Pattern constraint enforcement in binary strings
- Fibonacci-based mathematical optimization
- Previous state dependency for consecutive checking
- Binary digit-by-digit construction with constraints

KEY INSIGHTS:
============
1. **BINARY DIGIT DP**: Apply digit DP principles to binary representation
2. **CONSECUTIVE CONSTRAINT**: Track previous bit to prevent 11 pattern
3. **FIBONACCI CONNECTION**: Valid binary strings follow Fibonacci sequence
4. **MATHEMATICAL OPTIMIZATION**: Direct calculation using Fibonacci numbers
5. **PATTERN RECOGNITION**: Binary constraints create recognizable mathematical patterns

ALGORITHM APPROACHES:
====================

1. **Brute Force**: O(n log n) time, O(1) space
   - Check each number for consecutive 1s in binary
   - Simple but inefficient for large n

2. **Binary Digit DP**: O(log²n) time, O(log n) space
   - Standard digit DP on binary representation
   - Track previous bit for constraint enforcement

3. **Fibonacci Pattern**: O(log n) time, O(log n) space
   - Mathematical approach using Fibonacci sequence
   - Leverages pattern recognition in valid binary strings

4. **Mathematical**: O(log n) time, O(1) space
   - Direct calculation with Fibonacci numbers
   - Most efficient approach

CORE BINARY DIGIT DP:
====================
```python
def findIntegers(n):
    binary = bin(n)[2:]
    memo = {}
    
    def dp(pos, tight, last_bit):
        if pos == len(binary):
            return 1
        
        if (pos, tight, last_bit) in memo:
            return memo[(pos, tight, last_bit)]
        
        limit = int(binary[pos]) if tight else 1
        result = 0
        
        # Place 0
        result += dp(pos + 1, tight and (0 == limit), 0)
        
        # Place 1 (only if last bit wasn't 1)
        if last_bit == 0 and limit >= 1:
            result += dp(pos + 1, tight and (1 == limit), 1)
        
        memo[(pos, tight, last_bit)] = result
        return result
    
    return dp(0, True, 0)
```

FIBONACCI MATHEMATICAL APPROACH:
===============================
**Pattern Recognition**: Valid binary strings of length k follow Fibonacci sequence
- f(1) = 2: "0", "1"
- f(2) = 3: "00", "01", "10"
- f(3) = 5: "000", "001", "010", "100", "101"
- f(k) = f(k-1) + f(k-2)

**Intuition**: 
- Can append "0" to any valid string of length k-1
- Can append "1" only to valid strings of length k-2 ending with "0"

**Algorithm**:
```python
def findIntegers(n):
    binary = bin(n)[2:]
    k = len(binary)
    
    # Build Fibonacci sequence
    fib = [1, 2]
    for i in range(2, k + 1):
        fib.append(fib[i-1] + fib[i-2])
    
    result = 0
    for i in range(k):
        if binary[i] == '1':
            result += fib[k - i - 1]  # Count of valid strings with 0 at position i
            if i > 0 and binary[i-1] == '1':  # Consecutive 1s found
                break
    else:
        result += 1  # n itself is valid
    
    return result
```

CONSECUTIVE CONSTRAINT HANDLING:
===============================
**Problem**: Avoid "11" pattern in binary representation

**DP State Design**: Include `last_bit` in state
- `last_bit = 0`: Can place either 0 or 1
- `last_bit = 1`: Can only place 0

**Mathematical Verification**: Check for consecutive 1s during construction
```python
if i > 0 and binary[i-1] == '1' and binary[i] == '1':
    # Found consecutive 1s, stop processing
    break
```

FIBONACCI SEQUENCE CONNECTION:
=============================
**Recurrence Relation**: f(n) = f(n-1) + f(n-2)
- f(n-1): Valid strings of length n-1, append "0"
- f(n-2): Valid strings of length n-2 ending with "0", append "01"

**Base Cases**:
- f(0) = 1 (empty string)
- f(1) = 2 ("0", "1")

**Growth Pattern**: Exponential growth following golden ratio

BINARY UPPER BOUND MANAGEMENT:
==============================
**Tight Constraint in Binary**: Similar to decimal but simpler
- At each position, can place 0 or 1
- If tight and current bit of n is 0, can only place 0
- If tight and current bit of n is 1, can place 0 or 1

**Position-wise Analysis**:
- If placing 0 at position i: count all valid strings for remaining positions
- If placing 1 at position i: must check consecutive constraint

COMPLEXITY ANALYSIS:
===================
- **Binary Digit DP**: O(log²n) time, O(log n) space
- **Fibonacci**: O(log n) time, O(log n) space
- **Mathematical**: O(log n) time, O(1) space
- **State Space**: O(log n) positions × 2 tight × 2 last_bit

MATHEMATICAL OPTIMIZATION:
=========================
**Direct Fibonacci Calculation**: Avoid DP overhead
```python
def fibonacci_count(k):
    if k <= 1:
        return k + 1
    
    a, b = 1, 2
    for _ in range(2, k + 1):
        a, b = b, a + b
    return b
```

**Position-wise Contribution**: Sum Fibonacci values for each valid position choice

APPLICATIONS:
============
- **Digital Circuit Design**: Avoiding problematic bit patterns
- **Error-Correcting Codes**: Designing codes with pattern constraints
- **Data Transmission**: Avoiding synchronization problems in protocols
- **Combinatorics**: Counting restricted binary sequences
- **Algorithm Design**: Template for binary pattern constraints

RELATED PROBLEMS:
================
- **Climbing Stairs**: Same Fibonacci recurrence in different context
- **House Robber**: Similar consecutive constraint pattern
- **Binary String Generation**: Pattern-based string construction
- **Digit DP on Other Bases**: Extension to non-binary representations

VARIANTS:
========
- **No k Consecutive 1s**: Generalize to avoid k consecutive 1s
- **Exactly k Ones**: Count numbers with specific number of 1s
- **Pattern Matching**: Other forbidden/required patterns
- **Range Queries**: Count valid numbers in specific ranges

EDGE CASES:
==========
- **n = 0**: Single valid number (0 itself)
- **n = 1**: Two valid numbers (0, 1)
- **Powers of 2 minus 1**: All 1s, need careful handling
- **Large n**: Fibonacci numbers grow exponentially

OPTIMIZATION TECHNIQUES:
=======================
- **Fibonacci Precomputation**: Cache Fibonacci values
- **Early Termination**: Stop when consecutive 1s found in n
- **Mathematical Shortcuts**: Direct formulas when possible
- **Bit Manipulation**: Efficient binary operations

This problem beautifully demonstrates how binary representations
can be analyzed using Digit DP principles, while also showcasing
the elegant mathematical structure (Fibonacci sequence) that
emerges from pattern constraints in binary strings.
"""
