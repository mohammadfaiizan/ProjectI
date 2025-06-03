"""
🔢 BIT MANIPULATION - NUMBER PROPERTIES & TRICKS
===============================================

This module covers essential number properties and bit manipulation tricks.
These techniques are frequently used in competitive programming and interviews.

Topics Covered:
1. Check if a Number is Power of 2, 4, or 8
2. Multiply/Divide by 2 using << / >>
3. Odd/Even using Bitmask
4. XOR Properties and Patterns
5. Find the ith Bit
6. Turn Off Rightmost Set Bit
7. Turn On Rightmost 0 Bit
8. Remove the Last Set Bit

Author: Interview Preparation Collection
"""

class PowerChecks:
    """Check if numbers are powers of 2, 4, 8, etc."""
    
    @staticmethod
    def is_power_of_2(n: int) -> bool:
        """
        Check if number is power of 2.
        
        Key Insight: Power of 2 has only one bit set.
        n & (n-1) removes the rightmost set bit.
        
        Args:
            n: Number to check
            
        Returns:
            True if n is power of 2
            
        Time: O(1), Space: O(1)
        
        Examples:
            8 (1000) & 7 (0111) = 0
            16 (10000) & 15 (01111) = 0
        """
        return n > 0 and (n & (n - 1)) == 0
    
    @staticmethod
    def is_power_of_4(n: int) -> bool:
        """
        Check if number is power of 4.
        
        Approach 1: Power of 2 + odd position bit
        Power of 4 has bit set at odd positions (1, 3, 5, ...)
        0x55555555 = 01010101... (odd positions)
        
        Args:
            n: Number to check
            
        Returns:
            True if n is power of 4
            
        Time: O(1), Space: O(1)
        """
        return n > 0 and (n & (n - 1)) == 0 and (n & 0x55555555) != 0
    
    @staticmethod
    def is_power_of_4_alternative(n: int) -> bool:
        """
        Alternative method: Power of 4 using modulo.
        Power of 4 when divided by 3 gives remainder 1.
        """
        return n > 0 and (n & (n - 1)) == 0 and n % 3 == 1
    
    @staticmethod
    def is_power_of_8(n: int) -> bool:
        """
        Check if number is power of 8.
        
        Power of 8 has bits set at positions 0, 3, 6, 9, ...
        0x49249249 represents these positions.
        
        Args:
            n: Number to check
            
        Returns:
            True if n is power of 8
            
        Time: O(1), Space: O(1)
        """
        return n > 0 and (n & (n - 1)) == 0 and (n & 0x49249249) != 0
    
    @staticmethod
    def is_power_of_k(n: int, k: int) -> bool:
        """
        Check if number is power of k.
        
        Args:
            n: Number to check
            k: Base
            
        Returns:
            True if n is power of k
            
        Time: O(log n), Space: O(1)
        """
        if n <= 0 or k <= 1:
            return False
        
        while n > 1:
            if n % k != 0:
                return False
            n //= k
        
        return True
    
    @staticmethod
    def next_power_of_2(n: int) -> int:
        """
        Find next power of 2 greater than or equal to n.
        
        Args:
            n: Input number
            
        Returns:
            Next power of 2
            
        Time: O(log n), Space: O(1)
        """
        if n <= 1:
            return 1
        
        # If already power of 2
        if (n & (n - 1)) == 0:
            return n
        
        # Find position of MSB
        power = 0
        while n > 0:
            n >>= 1
            power += 1
        
        return 1 << power


class ArithmeticTricks:
    """Fast arithmetic operations using bit manipulation."""
    
    @staticmethod
    def multiply_by_2(n: int) -> int:
        """
        Multiply by 2 using left shift.
        
        Args:
            n: Number to multiply
            
        Returns:
            n * 2
            
        Time: O(1), Space: O(1)
        """
        return n << 1
    
    @staticmethod
    def divide_by_2(n: int) -> int:
        """
        Divide by 2 using right shift.
        
        Args:
            n: Number to divide
            
        Returns:
            n // 2
            
        Time: O(1), Space: O(1)
        """
        return n >> 1
    
    @staticmethod
    def multiply_by_power_of_2(n: int, power: int) -> int:
        """
        Multiply by 2^power using left shift.
        
        Args:
            n: Number to multiply
            power: Power of 2
            
        Returns:
            n * (2^power)
            
        Time: O(1), Space: O(1)
        """
        return n << power
    
    @staticmethod
    def divide_by_power_of_2(n: int, power: int) -> int:
        """
        Divide by 2^power using right shift.
        
        Args:
            n: Number to divide
            power: Power of 2
            
        Returns:
            n // (2^power)
            
        Time: O(1), Space: O(1)
        """
        return n >> power
    
    @staticmethod
    def is_odd(n: int) -> bool:
        """
        Check if number is odd using bitmask.
        
        Args:
            n: Number to check
            
        Returns:
            True if odd
            
        Time: O(1), Space: O(1)
        """
        return bool(n & 1)
    
    @staticmethod
    def is_even(n: int) -> bool:
        """
        Check if number is even using bitmask.
        
        Args:
            n: Number to check
            
        Returns:
            True if even
            
        Time: O(1), Space: O(1)
        """
        return not bool(n & 1)
    
    @staticmethod
    def absolute_value(n: int) -> int:
        """
        Calculate absolute value using bit manipulation.
        
        Uses the fact that right shift of negative number
        fills with 1s (arithmetic shift).
        
        Args:
            n: Input number
            
        Returns:
            |n|
            
        Time: O(1), Space: O(1)
        """
        mask = n >> 31  # All 1s if negative, all 0s if positive
        return (n + mask) ^ mask
    
    @staticmethod
    def sign(n: int) -> int:
        """
        Get sign of number (-1, 0, 1).
        
        Args:
            n: Input number
            
        Returns:
            Sign of n
            
        Time: O(1), Space: O(1)
        """
        return (n > 0) - (n < 0)


class XORProperties:
    """XOR properties and patterns for problem solving."""
    
    @staticmethod
    def xor_properties_demo():
        """Demonstrate key XOR properties."""
        print("=== XOR PROPERTIES ===")
        
        # Property 1: a ^ a = 0
        a = 5
        print(f"{a} ^ {a} = {a ^ a}")
        
        # Property 2: a ^ 0 = a
        print(f"{a} ^ 0 = {a ^ 0}")
        
        # Property 3: XOR is commutative and associative
        b, c = 3, 7
        print(f"({a} ^ {b}) ^ {c} = {(a ^ b) ^ c}")
        print(f"{a} ^ ({b} ^ {c}) = {a ^ (b ^ c)}")
        
        # Property 4: Self-inverse
        encrypted = a ^ b
        decrypted = encrypted ^ b
        print(f"Encrypt {a} with key {b}: {encrypted}")
        print(f"Decrypt {encrypted} with key {b}: {decrypted}")
    
    @staticmethod
    def swap_without_temp(a: int, b: int) -> tuple:
        """
        Swap two numbers without temporary variable using XOR.
        
        Args:
            a, b: Numbers to swap
            
        Returns:
            Tuple (b, a)
            
        Time: O(1), Space: O(1)
        """
        if a != b:  # Avoid swapping same memory location
            a = a ^ b
            b = a ^ b  # b = (a ^ b) ^ b = a
            a = a ^ b  # a = (a ^ b) ^ a = b
        
        return a, b
    
    @staticmethod
    def find_missing_number(arr: list, n: int) -> int:
        """
        Find missing number in array [1, 2, ..., n].
        
        Uses XOR property: a ^ a = 0
        XOR all numbers 1 to n, then XOR with array elements.
        
        Args:
            arr: Array with one missing number
            n: Range [1, n]
            
        Returns:
            Missing number
            
        Time: O(n), Space: O(1)
        """
        xor_all = 0
        xor_arr = 0
        
        # XOR all numbers from 1 to n
        for i in range(1, n + 1):
            xor_all ^= i
        
        # XOR all array elements
        for num in arr:
            xor_arr ^= num
        
        # Missing number
        return xor_all ^ xor_arr
    
    @staticmethod
    def find_duplicate_number(arr: list) -> int:
        """
        Find duplicate number in array where all others appear once.
        
        Args:
            arr: Array with one duplicate
            
        Returns:
            Duplicate number
            
        Time: O(n), Space: O(1)
        """
        result = 0
        for num in arr:
            result ^= num
        return result


class BitPositionTricks:
    """Tricks for working with specific bit positions."""
    
    @staticmethod
    def get_ith_bit(n: int, i: int) -> int:
        """
        Get the ith bit (0-indexed from right).
        
        Args:
            n: Number
            i: Bit position
            
        Returns:
            Bit value (0 or 1)
            
        Time: O(1), Space: O(1)
        """
        return (n >> i) & 1
    
    @staticmethod
    def set_ith_bit(n: int, i: int) -> int:
        """
        Set the ith bit to 1.
        
        Args:
            n: Number
            i: Bit position
            
        Returns:
            Number with ith bit set
            
        Time: O(1), Space: O(1)
        """
        return n | (1 << i)
    
    @staticmethod
    def clear_ith_bit(n: int, i: int) -> int:
        """
        Clear the ith bit (set to 0).
        
        Args:
            n: Number
            i: Bit position
            
        Returns:
            Number with ith bit cleared
            
        Time: O(1), Space: O(1)
        """
        return n & ~(1 << i)
    
    @staticmethod
    def toggle_ith_bit(n: int, i: int) -> int:
        """
        Toggle the ith bit.
        
        Args:
            n: Number
            i: Bit position
            
        Returns:
            Number with ith bit toggled
            
        Time: O(1), Space: O(1)
        """
        return n ^ (1 << i)
    
    @staticmethod
    def turn_off_rightmost_set_bit(n: int) -> int:
        """
        Turn off the rightmost set bit.
        
        Key Insight: n & (n-1) removes rightmost set bit.
        
        Args:
            n: Input number
            
        Returns:
            Number with rightmost set bit turned off
            
        Time: O(1), Space: O(1)
        
        Example: 12 (1100) -> 8 (1000)
        """
        return n & (n - 1)
    
    @staticmethod
    def turn_on_rightmost_zero_bit(n: int) -> int:
        """
        Turn on the rightmost 0 bit.
        
        Key Insight: n | (n+1) sets rightmost 0 bit.
        
        Args:
            n: Input number
            
        Returns:
            Number with rightmost 0 bit turned on
            
        Time: O(1), Space: O(1)
        
        Example: 10 (1010) -> 11 (1011)
        """
        return n | (n + 1)
    
    @staticmethod
    def isolate_rightmost_set_bit(n: int) -> int:
        """
        Isolate the rightmost set bit.
        
        Key Insight: n & (-n) isolates rightmost set bit.
        
        Args:
            n: Input number
            
        Returns:
            Number with only rightmost set bit
            
        Time: O(1), Space: O(1)
        
        Example: 12 (1100) -> 4 (0100)
        """
        return n & (-n)
    
    @staticmethod
    def isolate_rightmost_zero_bit(n: int) -> int:
        """
        Isolate the rightmost 0 bit.
        
        Key Insight: ~n & (n+1) isolates rightmost 0 bit.
        
        Args:
            n: Input number
            
        Returns:
            Number with only rightmost 0 bit set
            
        Time: O(1), Space: O(1)
        """
        return ~n & (n + 1)
    
    @staticmethod
    def count_trailing_zeros(n: int) -> int:
        """
        Count trailing zeros (rightmost zeros).
        
        Args:
            n: Input number
            
        Returns:
            Number of trailing zeros
            
        Time: O(log n), Space: O(1)
        """
        if n == 0:
            return 32  # Assuming 32-bit integer
        
        count = 0
        while (n & 1) == 0:
            count += 1
            n >>= 1
        
        return count
    
    @staticmethod
    def count_leading_zeros(n: int, bits: int = 32) -> int:
        """
        Count leading zeros.
        
        Args:
            n: Input number
            bits: Total bits to consider
            
        Returns:
            Number of leading zeros
            
        Time: O(log n), Space: O(1)
        """
        if n == 0:
            return bits
        
        count = 0
        msb_position = bits - 1
        
        while msb_position >= 0 and not (n & (1 << msb_position)):
            count += 1
            msb_position -= 1
        
        return count


class AdvancedBitTricks:
    """Advanced bit manipulation tricks and patterns."""
    
    @staticmethod
    def reverse_bits(n: int, bits: int = 32) -> int:
        """
        Reverse bits of a number.
        
        Args:
            n: Input number
            bits: Number of bits to consider
            
        Returns:
            Number with reversed bits
            
        Time: O(bits), Space: O(1)
        """
        result = 0
        for i in range(bits):
            if n & (1 << i):
                result |= 1 << (bits - 1 - i)
        
        return result
    
    @staticmethod
    def count_different_bits(a: int, b: int) -> int:
        """
        Count number of different bits between two numbers.
        
        Args:
            a, b: Input numbers
            
        Returns:
            Number of different bits
            
        Time: O(log max(a,b)), Space: O(1)
        """
        xor_result = a ^ b
        count = 0
        
        while xor_result:
            count += xor_result & 1
            xor_result >>= 1
        
        return count
    
    @staticmethod
    def gray_code(n: int) -> int:
        """
        Convert binary to Gray code.
        
        Gray code: Adjacent values differ by exactly one bit.
        
        Args:
            n: Binary number
            
        Returns:
            Gray code equivalent
            
        Time: O(1), Space: O(1)
        """
        return n ^ (n >> 1)
    
    @staticmethod
    def binary_from_gray(gray: int) -> int:
        """
        Convert Gray code to binary.
        
        Args:
            gray: Gray code number
            
        Returns:
            Binary equivalent
            
        Time: O(log gray), Space: O(1)
        """
        binary = gray
        while gray:
            gray >>= 1
            binary ^= gray
        
        return binary
    
    @staticmethod
    def find_position_of_set_bit(n: int) -> int:
        """
        Find position of the only set bit.
        Assumes exactly one bit is set.
        
        Args:
            n: Number with exactly one set bit
            
        Returns:
            Position of set bit (0-indexed)
            
        Time: O(log n), Space: O(1)
        """
        if n == 0 or (n & (n - 1)) != 0:
            return -1  # Not exactly one bit set
        
        position = 0
        while n > 1:
            n >>= 1
            position += 1
        
        return position


class NumberTricksDemo:
    """Demonstration of all number tricks and properties."""
    
    @staticmethod
    def demonstrate_power_checks():
        """Demonstrate power checking functions."""
        print("=== POWER CHECKS ===")
        test_numbers = [1, 2, 4, 8, 16, 32, 64, 15, 17]
        
        for num in test_numbers:
            print(f"Number: {num}")
            print(f"  Power of 2: {PowerChecks.is_power_of_2(num)}")
            print(f"  Power of 4: {PowerChecks.is_power_of_4(num)}")
            print(f"  Power of 8: {PowerChecks.is_power_of_8(num)}")
            print(f"  Next power of 2: {PowerChecks.next_power_of_2(num)}")
            print()
    
    @staticmethod
    def demonstrate_arithmetic_tricks():
        """Demonstrate arithmetic bit tricks."""
        print("=== ARITHMETIC TRICKS ===")
        numbers = [5, 12, -7, 0]
        
        for num in numbers:
            print(f"Number: {num}")
            print(f"  Multiply by 2: {ArithmeticTricks.multiply_by_2(num)}")
            print(f"  Divide by 2: {ArithmeticTricks.divide_by_2(num)}")
            print(f"  Is odd: {ArithmeticTricks.is_odd(num)}")
            print(f"  Is even: {ArithmeticTricks.is_even(num)}")
            print(f"  Absolute value: {ArithmeticTricks.absolute_value(num)}")
            print(f"  Sign: {ArithmeticTricks.sign(num)}")
            print()
    
    @staticmethod
    def demonstrate_xor_properties():
        """Demonstrate XOR properties and applications."""
        print("=== XOR PROPERTIES ===")
        XORProperties.xor_properties_demo()
        
        # Swap demonstration
        a, b = 5, 7
        print(f"\nSwapping {a} and {b}:")
        swapped_a, swapped_b = XORProperties.swap_without_temp(a, b)
        print(f"After swap: {swapped_a}, {swapped_b}")
        
        # Missing number
        arr = [1, 2, 4, 5, 6]
        missing = XORProperties.find_missing_number(arr, 6)
        print(f"\nMissing number in {arr}: {missing}")
    
    @staticmethod
    def demonstrate_bit_position_tricks():
        """Demonstrate bit position manipulation."""
        print("=== BIT POSITION TRICKS ===")
        n = 12  # 1100 in binary
        print(f"Original number: {n} ({bin(n)})")
        
        # Various bit operations
        print(f"Get 2nd bit: {BitPositionTricks.get_ith_bit(n, 2)}")
        print(f"Set 1st bit: {BitPositionTricks.set_ith_bit(n, 1)} ({bin(BitPositionTricks.set_ith_bit(n, 1))})")
        print(f"Clear 3rd bit: {BitPositionTricks.clear_ith_bit(n, 3)} ({bin(BitPositionTricks.clear_ith_bit(n, 3))})")
        print(f"Toggle 0th bit: {BitPositionTricks.toggle_ith_bit(n, 0)} ({bin(BitPositionTricks.toggle_ith_bit(n, 0))})")
        
        # Advanced tricks
        print(f"Turn off rightmost set bit: {BitPositionTricks.turn_off_rightmost_set_bit(n)} ({bin(BitPositionTricks.turn_off_rightmost_set_bit(n))})")
        print(f"Isolate rightmost set bit: {BitPositionTricks.isolate_rightmost_set_bit(n)} ({bin(BitPositionTricks.isolate_rightmost_set_bit(n))})")
        print(f"Count trailing zeros: {BitPositionTricks.count_trailing_zeros(n)}")
    
    @staticmethod
    def demonstrate_advanced_tricks():
        """Demonstrate advanced bit manipulation tricks."""
        print("=== ADVANCED BIT TRICKS ===")
        
        # Reverse bits
        n = 12
        reversed_bits = AdvancedBitTricks.reverse_bits(n, 8)
        print(f"Reverse bits of {n}: {reversed_bits} ({bin(reversed_bits)})")
        
        # Count different bits
        a, b = 10, 15
        diff_bits = AdvancedBitTricks.count_different_bits(a, b)
        print(f"Different bits between {a} and {b}: {diff_bits}")
        
        # Gray code
        binary = 5
        gray = AdvancedBitTricks.gray_code(binary)
        back_to_binary = AdvancedBitTricks.binary_from_gray(gray)
        print(f"Binary {binary} -> Gray {gray} -> Binary {back_to_binary}")


def performance_analysis():
    """Analyze performance of bit manipulation vs traditional methods."""
    import time
    
    print("=== PERFORMANCE ANALYSIS ===")
    
    # Test multiplication by 2
    numbers = list(range(1, 100000))
    
    # Bit manipulation
    start = time.time()
    for num in numbers:
        ArithmeticTricks.multiply_by_2(num)
    bit_time = time.time() - start
    
    # Traditional multiplication
    start = time.time()
    for num in numbers:
        num * 2
    mult_time = time.time() - start
    
    print(f"Bit manipulation (<<): {bit_time:.4f} seconds")
    print(f"Traditional (*): {mult_time:.4f} seconds")
    print(f"Speedup: {mult_time/bit_time:.2f}x")
    
    # Test power of 2 check
    start = time.time()
    for num in numbers:
        PowerChecks.is_power_of_2(num)
    bit_power_time = time.time() - start
    
    start = time.time()
    for num in numbers:
        num > 0 and (num & (num - 1)) == 0
    traditional_time = time.time() - start
    
    print(f"\nPower of 2 check (bit): {bit_power_time:.4f} seconds")
    print(f"Direct bit operation: {traditional_time:.4f} seconds")


if __name__ == "__main__":
    # Run all demonstrations
    demo = NumberTricksDemo()
    
    demo.demonstrate_power_checks()
    demo.demonstrate_arithmetic_tricks()
    demo.demonstrate_xor_properties()
    demo.demonstrate_bit_position_tricks()
    demo.demonstrate_advanced_tricks()
    
    performance_analysis()
    
    print("\n🎯 Key Takeaways:")
    print("1. n & (n-1) removes rightmost set bit - very useful!")
    print("2. XOR has unique properties: self-inverse, commutative")
    print("3. Bit shifts are faster than multiplication/division by powers of 2")
    print("4. Power checks using bits are O(1) operations")
    print("5. Many problems can be solved elegantly with bit manipulation") 