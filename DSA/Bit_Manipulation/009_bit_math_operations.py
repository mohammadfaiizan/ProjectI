"""
🧮 BITWISE MATH OPERATIONS
=========================

This module implements fundamental mathematical operations using only bitwise operators.
These techniques demonstrate the power of bit manipulation for arithmetic computations.

Topics Covered:
1. Add Two Numbers Without + (Using Bitwise)
2. Multiply Two Numbers using Bitwise
3. Divide Two Numbers using Bitwise
4. Modulo with Power of Two

Author: Interview Preparation Collection
LeetCode Problems: 371, 29, 190, 461
"""

class BitwiseArithmetic:
    """Basic arithmetic operations using bitwise operators."""
    
    @staticmethod
    def add_without_plus(a: int, b: int) -> int:
        """
        Add two numbers without using + operator.
        
        Key Insight: Sum = XOR + Carry
        XOR gives sum without carry, AND gives carry positions.
        
        Args:
            a: First number
            b: Second number
            
        Returns:
            Sum of a and b
            
        Time: O(log max(a,b)), Space: O(1)
        LeetCode: 371
        """
        # Handle negative numbers using 32-bit representation
        mask = 0xFFFFFFFF
        
        while b != 0:
            # Calculate sum without carry
            sum_without_carry = (a ^ b) & mask
            
            # Calculate carry
            carry = ((a & b) << 1) & mask
            
            a = sum_without_carry
            b = carry
        
        # Handle overflow for negative results
        if a > 0x7FFFFFFF:
            return ~(a ^ mask)
        
        return a
    
    @staticmethod
    def add_iterative_explanation(a: int, b: int) -> tuple:
        """
        Add numbers with step-by-step explanation.
        
        Args:
            a: First number
            b: Second number
            
        Returns:
            Tuple (result, steps)
        """
        steps = []
        original_a, original_b = a, b
        
        while b != 0:
            sum_without_carry = a ^ b
            carry = (a & b) << 1
            
            steps.append(f"  {a} ^ {b} = {sum_without_carry} (sum without carry)")
            steps.append(f"  ({a} & {b}) << 1 = {carry} (carry)")
            
            a = sum_without_carry
            b = carry
            
            steps.append(f"  New a={a}, b={b}")
        
        return a, steps
    
    @staticmethod
    def subtract_without_minus(a: int, b: int) -> int:
        """
        Subtract two numbers without using - operator.
        
        Subtraction: a - b = a + (-b) = a + (~b + 1)
        
        Args:
            a: Minuend
            b: Subtrahend
            
        Returns:
            Difference a - b
            
        Time: O(log max(a,b)), Space: O(1)
        """
        # Calculate -b using two's complement: -b = ~b + 1
        neg_b = BitwiseArithmetic.add_without_plus(~b, 1)
        
        # Add a + (-b)
        return BitwiseArithmetic.add_without_plus(a, neg_b)
    
    @staticmethod
    def increment(x: int) -> int:
        """
        Increment number by 1 without using +.
        
        Args:
            x: Number to increment
            
        Returns:
            x + 1
            
        Time: O(1), Space: O(1)
        """
        return BitwiseArithmetic.add_without_plus(x, 1)
    
    @staticmethod
    def decrement(x: int) -> int:
        """
        Decrement number by 1 without using -.
        
        Args:
            x: Number to decrement
            
        Returns:
            x - 1
            
        Time: O(1), Space: O(1)
        """
        return BitwiseArithmetic.add_without_plus(x, -1)


class BitwiseMultiplication:
    """Multiplication operations using bitwise techniques."""
    
    @staticmethod
    def multiply_simple(a: int, b: int) -> int:
        """
        Multiply two numbers using repeated addition.
        
        Args:
            a: First number
            b: Second number
            
        Returns:
            Product a * b
            
        Time: O(b), Space: O(1)
        """
        result = 0
        
        # Handle negative numbers
        negative = (a < 0) ^ (b < 0)
        a, b = abs(a), abs(b)
        
        # Repeated addition
        for _ in range(b):
            result = BitwiseArithmetic.add_without_plus(result, a)
        
        return -result if negative else result
    
    @staticmethod
    def multiply_optimized(a: int, b: int) -> int:
        """
        Multiply using bit shifting (optimized approach).
        
        Key Insight: Multiplication by powers of 2 can be done with left shift.
        For other numbers, decompose into powers of 2.
        
        Args:
            a: First number
            b: Second number
            
        Returns:
            Product a * b
            
        Time: O(log b), Space: O(1)
        """
        result = 0
        
        # Handle negative numbers
        negative = (a < 0) ^ (b < 0)
        a, b = abs(a), abs(b)
        
        while b > 0:
            # If current bit of b is set, add current a to result
            if b & 1:
                result = BitwiseArithmetic.add_without_plus(result, a)
            
            # Shift a left (multiply by 2)
            a <<= 1
            
            # Shift b right (divide by 2)
            b >>= 1
        
        return -result if negative else result
    
    @staticmethod
    def multiply_recursive(a: int, b: int) -> int:
        """
        Multiply using recursive approach.
        
        Args:
            a: First number
            b: Second number
            
        Returns:
            Product a * b
            
        Time: O(log b), Space: O(log b)
        """
        if b == 0:
            return 0
        if b == 1:
            return a
        
        # Handle negative numbers
        if b < 0:
            return -BitwiseMultiplication.multiply_recursive(a, -b)
        
        # Recursive multiplication: a * b = a * (b/2) * 2 + a * (b%2)
        half = BitwiseMultiplication.multiply_recursive(a, b >> 1)
        half_doubled = half << 1
        
        if b & 1:  # If b is odd
            return BitwiseArithmetic.add_without_plus(half_doubled, a)
        else:
            return half_doubled
    
    @staticmethod
    def square(x: int) -> int:
        """
        Calculate square of a number using bitwise operations.
        
        Args:
            x: Number to square
            
        Returns:
            x^2
            
        Time: O(log x), Space: O(1)
        """
        return BitwiseMultiplication.multiply_optimized(x, x)
    
    @staticmethod
    def power_of_two_multiply(x: int, power: int) -> int:
        """
        Multiply by 2^power using left shift.
        
        Args:
            x: Number to multiply
            power: Power of 2
            
        Returns:
            x * (2^power)
            
        Time: O(1), Space: O(1)
        """
        return x << power


class BitwiseDivision:
    """Division operations using bitwise techniques."""
    
    @staticmethod
    def divide_simple(dividend: int, divisor: int) -> int:
        """
        Divide two numbers using repeated subtraction.
        
        Args:
            dividend: Number to be divided
            divisor: Number to divide by
            
        Returns:
            Quotient (dividend / divisor)
            
        Time: O(dividend/divisor), Space: O(1)
        LeetCode: 29
        """
        if divisor == 0:
            raise ValueError("Division by zero")
        
        # Handle signs
        negative = (dividend < 0) ^ (divisor < 0)
        dividend, divisor = abs(dividend), abs(divisor)
        
        quotient = 0
        
        # Repeated subtraction
        while dividend >= divisor:
            dividend = BitwiseArithmetic.subtract_without_minus(dividend, divisor)
            quotient = BitwiseArithmetic.add_without_plus(quotient, 1)
        
        return -quotient if negative else quotient
    
    @staticmethod
    def divide_optimized(dividend: int, divisor: int) -> int:
        """
        Divide using bit shifting (optimized approach).
        
        Key Insight: Division by powers of 2 can be done with right shift.
        For other numbers, find largest multiple and subtract.
        
        Args:
            dividend: Number to be divided
            divisor: Number to divide by
            
        Returns:
            Quotient (dividend / divisor)
            
        Time: O(log dividend), Space: O(1)
        """
        if divisor == 0:
            raise ValueError("Division by zero")
        
        # Handle overflow
        if dividend == -(2**31) and divisor == -1:
            return 2**31 - 1
        
        # Handle signs
        negative = (dividend < 0) ^ (divisor < 0)
        dividend, divisor = abs(dividend), abs(divisor)
        
        quotient = 0
        
        while dividend >= divisor:
            # Find the largest multiple of divisor that fits in dividend
            temp_divisor = divisor
            multiple = 1
            
            while dividend >= (temp_divisor << 1):
                temp_divisor <<= 1
                multiple <<= 1
            
            # Subtract the largest multiple
            dividend = BitwiseArithmetic.subtract_without_minus(dividend, temp_divisor)
            quotient = BitwiseArithmetic.add_without_plus(quotient, multiple)
        
        return -quotient if negative else quotient
    
    @staticmethod
    def power_of_two_divide(x: int, power: int) -> int:
        """
        Divide by 2^power using right shift.
        
        Args:
            x: Number to divide
            power: Power of 2
            
        Returns:
            x / (2^power)
            
        Time: O(1), Space: O(1)
        """
        return x >> power
    
    @staticmethod
    def modulo_power_of_two(x: int, power: int) -> int:
        """
        Calculate x % (2^power) using bitwise AND.
        
        Key Insight: x % (2^n) = x & ((2^n) - 1)
        
        Args:
            x: Number
            power: Power of 2
            
        Returns:
            x % (2^power)
            
        Time: O(1), Space: O(1)
        """
        if power <= 0:
            raise ValueError("Power must be positive")
        
        mask = (1 << power) - 1
        return x & mask
    
    @staticmethod
    def is_divisible_by_power_of_two(x: int, power: int) -> bool:
        """
        Check if x is divisible by 2^power.
        
        Args:
            x: Number to check
            power: Power of 2
            
        Returns:
            True if x is divisible by 2^power
            
        Time: O(1), Space: O(1)
        """
        return BitwiseDivision.modulo_power_of_two(x, power) == 0


class AdvancedBitwiseMath:
    """Advanced mathematical operations using bitwise techniques."""
    
    @staticmethod
    def fast_exponentiation(base: int, exp: int) -> int:
        """
        Calculate base^exp using bitwise fast exponentiation.
        
        Args:
            base: Base number
            exp: Exponent
            
        Returns:
            base^exp
            
        Time: O(log exp), Space: O(1)
        """
        if exp == 0:
            return 1
        
        result = 1
        
        while exp > 0:
            # If current bit of exponent is set
            if exp & 1:
                result = BitwiseMultiplication.multiply_optimized(result, base)
            
            # Square the base for next bit
            base = BitwiseMultiplication.multiply_optimized(base, base)
            exp >>= 1
        
        return result
    
    @staticmethod
    def gcd_bitwise(a: int, b: int) -> int:
        """
        Calculate GCD using bitwise operations (Binary GCD algorithm).
        
        Args:
            a: First number
            b: Second number
            
        Returns:
            GCD of a and b
            
        Time: O(log min(a,b)), Space: O(1)
        """
        if a == 0:
            return b
        if b == 0:
            return a
        
        # Make both numbers positive
        a, b = abs(a), abs(b)
        
        # Count common factors 2
        shift = 0
        while ((a | b) & 1) == 0:
            a >>= 1
            b >>= 1
            shift += 1
        
        # Remove remaining factors of 2 from a
        while (a & 1) == 0:
            a >>= 1
        
        while b != 0:
            # Remove factors of 2 from b
            while (b & 1) == 0:
                b >>= 1
            
            # Ensure a >= b
            if a > b:
                a, b = b, a
            
            b = BitwiseArithmetic.subtract_without_minus(b, a)
        
        return a << shift
    
    @staticmethod
    def sqrt_bitwise(x: int) -> int:
        """
        Calculate integer square root using bitwise operations.
        
        Args:
            x: Number to find square root of
            
        Returns:
            Floor of square root of x
            
        Time: O(log x), Space: O(1)
        """
        if x < 2:
            return x
        
        # Binary search approach
        left, right = 1, x
        
        while left <= right:
            mid = (left + right) >> 1  # Divide by 2
            square = BitwiseMultiplication.multiply_optimized(mid, mid)
            
            if square == x:
                return mid
            elif square < x:
                left = BitwiseArithmetic.add_without_plus(mid, 1)
            else:
                right = BitwiseArithmetic.subtract_without_minus(mid, 1)
        
        return right
    
    @staticmethod
    def absolute_value(x: int) -> int:
        """
        Calculate absolute value using bitwise operations.
        
        Args:
            x: Input number
            
        Returns:
            |x|
            
        Time: O(1), Space: O(1)
        """
        # For negative numbers, sign bit is 1
        # Right shift by 31 gives all 1s for negative, all 0s for positive
        mask = x >> 31
        
        # XOR with mask and subtract mask
        return (x ^ mask) - mask
    
    @staticmethod
    def sign(x: int) -> int:
        """
        Get sign of number using bitwise operations.
        
        Args:
            x: Input number
            
        Returns:
            -1 if negative, 0 if zero, 1 if positive
            
        Time: O(1), Space: O(1)
        """
        if x == 0:
            return 0
        
        return 1 if (x >> 31) == 0 else -1
    
    @staticmethod
    def min_bitwise(a: int, b: int) -> int:
        """
        Find minimum of two numbers using bitwise operations.
        
        Args:
            a: First number
            b: Second number
            
        Returns:
            min(a, b)
            
        Time: O(1), Space: O(1)
        """
        diff = BitwiseArithmetic.subtract_without_minus(a, b)
        # If diff < 0, then a < b
        sign_bit = (diff >> 31) & 1
        return b if sign_bit else a
    
    @staticmethod
    def max_bitwise(a: int, b: int) -> int:
        """
        Find maximum of two numbers using bitwise operations.
        
        Args:
            a: First number
            b: Second number
            
        Returns:
            max(a, b)
            
        Time: O(1), Space: O(1)
        """
        diff = BitwiseArithmetic.subtract_without_minus(a, b)
        # If diff >= 0, then a >= b
        sign_bit = (diff >> 31) & 1
        return a if not sign_bit else b


class BitwiseMathDemo:
    """Demonstration of bitwise mathematical operations."""
    
    @staticmethod
    def demonstrate_addition():
        """Demonstrate bitwise addition."""
        print("=== BITWISE ADDITION ===")
        
        test_cases = [(5, 3), (15, 7), (-5, 3), (10, -8)]
        
        for a, b in test_cases:
            result = BitwiseArithmetic.add_without_plus(a, b)
            result_with_steps, steps = BitwiseArithmetic.add_iterative_explanation(a, b)
            
            print(f"\nAdding {a} + {b}:")
            print(f"Result: {result}")
            print("Steps:")
            for step in steps:
                print(step)
            
            # Verify with built-in
            print(f"Built-in verification: {a + b}")
            print(f"Match: {result == a + b}")
    
    @staticmethod
    def demonstrate_multiplication():
        """Demonstrate bitwise multiplication."""
        print("\n=== BITWISE MULTIPLICATION ===")
        
        test_cases = [(6, 4), (7, 8), (-3, 5), (0, 10)]
        
        for a, b in test_cases:
            simple = BitwiseMultiplication.multiply_simple(a, b)
            optimized = BitwiseMultiplication.multiply_optimized(a, b)
            recursive = BitwiseMultiplication.multiply_recursive(a, b)
            
            print(f"\nMultiplying {a} * {b}:")
            print(f"Simple method: {simple}")
            print(f"Optimized method: {optimized}")
            print(f"Recursive method: {recursive}")
            print(f"Built-in verification: {a * b}")
            print(f"All match: {simple == optimized == recursive == a * b}")
        
        # Power of 2 multiplication
        x = 5
        for power in range(1, 5):
            result = BitwiseMultiplication.power_of_two_multiply(x, power)
            expected = x * (2 ** power)
            print(f"{x} * 2^{power} = {result} (expected: {expected})")
    
    @staticmethod
    def demonstrate_division():
        """Demonstrate bitwise division."""
        print("\n=== BITWISE DIVISION ===")
        
        test_cases = [(20, 4), (17, 3), (-15, 4), (10, -2)]
        
        for dividend, divisor in test_cases:
            try:
                optimized = BitwiseDivision.divide_optimized(dividend, divisor)
                
                print(f"\nDividing {dividend} / {divisor}:")
                print(f"Optimized method: {optimized}")
                print(f"Built-in verification: {dividend // divisor}")
                print(f"Match: {optimized == dividend // divisor}")
                
                # Modulo with powers of 2
                if divisor > 0 and (divisor & (divisor - 1)) == 0:  # Power of 2
                    power = divisor.bit_length() - 1
                    mod_result = BitwiseDivision.modulo_power_of_two(dividend, power)
                    print(f"Modulo {dividend} % {divisor} = {mod_result}")
                    print(f"Verification: {dividend % divisor}")
                
            except ValueError as e:
                print(f"Error: {e}")
    
    @staticmethod
    def demonstrate_advanced_operations():
        """Demonstrate advanced bitwise math operations."""
        print("\n=== ADVANCED BITWISE MATH ===")
        
        # Fast exponentiation
        base, exp = 3, 4
        power_result = AdvancedBitwiseMath.fast_exponentiation(base, exp)
        print(f"{base}^{exp} = {power_result} (verification: {base**exp})")
        
        # GCD
        a, b = 48, 18
        gcd_result = AdvancedBitwiseMath.gcd_bitwise(a, b)
        import math
        print(f"GCD({a}, {b}) = {gcd_result} (verification: {math.gcd(a, b)})")
        
        # Square root
        x = 16
        sqrt_result = AdvancedBitwiseMath.sqrt_bitwise(x)
        print(f"sqrt({x}) = {sqrt_result} (verification: {int(x**0.5)})")
        
        # Absolute value
        numbers = [5, -7, 0, -15]
        for num in numbers:
            abs_result = AdvancedBitwiseMath.absolute_value(num)
            print(f"|{num}| = {abs_result} (verification: {abs(num)})")
        
        # Min/Max
        pairs = [(5, 8), (-3, 2), (0, -1)]
        for a, b in pairs:
            min_result = AdvancedBitwiseMath.min_bitwise(a, b)
            max_result = AdvancedBitwiseMath.max_bitwise(a, b)
            print(f"min({a}, {b}) = {min_result}, max({a}, {b}) = {max_result}")
            print(f"Verification: min={min(a, b)}, max={max(a, b)}")


def performance_comparison():
    """Compare performance of bitwise vs traditional operations."""
    import time
    
    print("\n=== PERFORMANCE COMPARISON ===")
    
    # Test data
    test_pairs = [(i, i+1) for i in range(1, 1001)]
    
    # Addition comparison
    start = time.time()
    for a, b in test_pairs:
        BitwiseArithmetic.add_without_plus(a, b)
    bitwise_add_time = time.time() - start
    
    start = time.time()
    for a, b in test_pairs:
        a + b
    builtin_add_time = time.time() - start
    
    print(f"Addition (1000 operations):")
    print(f"  Bitwise: {bitwise_add_time:.6f} seconds")
    print(f"  Built-in: {builtin_add_time:.6f} seconds")
    print(f"  Ratio: {bitwise_add_time/builtin_add_time:.2f}x slower")
    
    # Multiplication comparison
    start = time.time()
    for a, b in test_pairs[:100]:  # Smaller set for multiplication
        BitwiseMultiplication.multiply_optimized(a, b)
    bitwise_mult_time = time.time() - start
    
    start = time.time()
    for a, b in test_pairs[:100]:
        a * b
    builtin_mult_time = time.time() - start
    
    print(f"\nMultiplication (100 operations):")
    print(f"  Bitwise: {bitwise_mult_time:.6f} seconds")
    print(f"  Built-in: {builtin_mult_time:.6f} seconds")
    print(f"  Ratio: {bitwise_mult_time/builtin_mult_time:.2f}x slower")


def practical_applications():
    """Discuss practical applications of bitwise math."""
    print("\n=== PRACTICAL APPLICATIONS ===")
    
    print("Real-world uses of bitwise arithmetic:")
    print("1. Embedded systems: Resource-constrained environments")
    print("2. Cryptography: Custom arithmetic for security algorithms")
    print("3. Computer graphics: Fast mathematical computations")
    print("4. Game engines: Performance-critical calculations")
    print("5. Compiler optimization: Low-level code generation")
    print("6. Digital signal processing: Efficient mathematical operations")
    print("7. Interview questions: Demonstrating bit manipulation mastery")


if __name__ == "__main__":
    # Run all demonstrations
    demo = BitwiseMathDemo()
    
    demo.demonstrate_addition()
    demo.demonstrate_multiplication()
    demo.demonstrate_division()
    demo.demonstrate_advanced_operations()
    
    performance_comparison()
    practical_applications()
    
    print("\n🎯 Key Bitwise Math Patterns:")
    print("1. Addition: XOR for sum, AND+shift for carry")
    print("2. Multiplication: Shift and add based on binary representation")
    print("3. Division: Repeated subtraction with optimization")
    print("4. Powers of 2: Use shifts for multiplication/division")
    print("5. Modulo: Use AND mask for powers of 2")
    print("6. Optimization: Replace expensive operations with bitwise equivalents") 