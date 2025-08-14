"""
🧱 BIT MANIPULATION BASICS
==========================

This module covers fundamental bit manipulation concepts and operations.
Essential for competitive programming and technical interviews.

Topics Covered:
1. Binary Representation of Numbers
2. Bitwise Operators: &, |, ^, ~, <<, >>
3. Set, Clear, Toggle, Check a Bit
4. Counting Set Bits (Brian Kernighan's Algorithm)
5. Fast Exponentiation using Bits

Author: Interview Preparation Collection
"""

class BinaryRepresentation:
    """Binary representation and conversion utilities."""
    
    @staticmethod
    def decimal_to_binary(n: int, width: int = 8) -> str:
        """
        Convert decimal to binary representation.
        
        Args:
            n: Decimal number
            width: Minimum width of binary string
            
        Returns:
            Binary string representation
            
        Time: O(log n), Space: O(log n)
        """
        if n == 0:
            return '0' * width
        
        binary = ""
        temp = abs(n)
        
        while temp > 0:
            binary = str(temp & 1) + binary
            temp >>= 1
        
        # Handle negative numbers (two's complement)
        if n < 0:
            # Flip bits and add 1
            flipped = ""
            for bit in binary:
                flipped += '0' if bit == '1' else '1'
            
            # Add 1 to flipped representation
            carry = 1
            result = ""
            for i in range(len(flipped) - 1, -1, -1):
                sum_bit = int(flipped[i]) + carry
                result = str(sum_bit % 2) + result
                carry = sum_bit // 2
            
            binary = result
        
        return binary.zfill(width)
    
    @staticmethod
    def binary_to_decimal(binary: str) -> int:
        """
        Convert binary string to decimal.
        
        Args:
            binary: Binary string
            
        Returns:
            Decimal number
            
        Time: O(n), Space: O(1)
        """
        decimal = 0
        power = 0
        
        for i in range(len(binary) - 1, -1, -1):
            if binary[i] == '1':
                decimal += 2 ** power
            power += 1
        
        return decimal
    
    @staticmethod
    def print_binary_analysis(n: int):
        """Print detailed binary analysis of a number."""
        print(f"Number: {n}")
        print(f"Binary: {bin(n)}")
        print(f"8-bit:  {BinaryRepresentation.decimal_to_binary(n, 8)}")
        print(f"16-bit: {BinaryRepresentation.decimal_to_binary(n, 16)}")
        print(f"Hex:    {hex(n)}")
        print(f"Octal:  {oct(n)}")
        print("-" * 30)


class BitwiseOperators:
    """Comprehensive bitwise operators implementation and examples."""
    
    @staticmethod
    def and_operation(a: int, b: int) -> dict:
        """
        Bitwise AND operation analysis.
        
        Args:
            a, b: Input numbers
            
        Returns:
            Dictionary with operation details
        """
        result = a & b
        return {
            'operation': f"{a} & {b}",
            'binary_a': bin(a),
            'binary_b': bin(b),
            'result': result,
            'binary_result': bin(result),
            'explanation': "AND: 1 only when both bits are 1"
        }
    
    @staticmethod
    def or_operation(a: int, b: int) -> dict:
        """Bitwise OR operation analysis."""
        result = a | b
        return {
            'operation': f"{a} | {b}",
            'binary_a': bin(a),
            'binary_b': bin(b),
            'result': result,
            'binary_result': bin(result),
            'explanation': "OR: 1 when at least one bit is 1"
        }
    
    @staticmethod
    def xor_operation(a: int, b: int) -> dict:
        """Bitwise XOR operation analysis."""
        result = a ^ b
        return {
            'operation': f"{a} ^ {b}",
            'binary_a': bin(a),
            'binary_b': bin(b),
            'result': result,
            'binary_result': bin(result),
            'explanation': "XOR: 1 when bits are different"
        }
    
    @staticmethod
    def not_operation(a: int, bits: int = 8) -> dict:
        """Bitwise NOT operation analysis."""
        # Python's ~ operator gives -(n+1), so we mask for specific bit width
        mask = (1 << bits) - 1
        result = (~a) & mask
        return {
            'operation': f"~{a}",
            'binary_a': bin(a),
            'result': result,
            'binary_result': bin(result),
            'explanation': f"NOT: Flip all bits (masked to {bits} bits)"
        }
    
    @staticmethod
    def left_shift(a: int, positions: int) -> dict:
        """Left shift operation analysis."""
        result = a << positions
        return {
            'operation': f"{a} << {positions}",
            'binary_a': bin(a),
            'result': result,
            'binary_result': bin(result),
            'explanation': f"Left shift: Multiply by 2^{positions} = {2**positions}"
        }
    
    @staticmethod
    def right_shift(a: int, positions: int) -> dict:
        """Right shift operation analysis."""
        result = a >> positions
        return {
            'operation': f"{a} >> {positions}",
            'binary_a': bin(a),
            'result': result,
            'binary_result': bin(result),
            'explanation': f"Right shift: Divide by 2^{positions} = {2**positions}"
        }


class BitOperations:
    """Individual bit manipulation operations."""
    
    @staticmethod
    def set_bit(n: int, position: int) -> int:
        """
        Set bit at given position to 1.
        
        Args:
            n: Number
            position: Bit position (0-indexed from right)
            
        Returns:
            Number with bit set
            
        Time: O(1), Space: O(1)
        """
        return n | (1 << position)
    
    @staticmethod
    def clear_bit(n: int, position: int) -> int:
        """
        Clear bit at given position (set to 0).
        
        Args:
            n: Number
            position: Bit position (0-indexed from right)
            
        Returns:
            Number with bit cleared
            
        Time: O(1), Space: O(1)
        """
        return n & ~(1 << position)
    
    @staticmethod
    def toggle_bit(n: int, position: int) -> int:
        """
        Toggle bit at given position.
        
        Args:
            n: Number
            position: Bit position (0-indexed from right)
            
        Returns:
            Number with bit toggled
            
        Time: O(1), Space: O(1)
        """
        return n ^ (1 << position)
    
    @staticmethod
    def check_bit(n: int, position: int) -> bool:
        """
        Check if bit at given position is set.
        
        Args:
            n: Number
            position: Bit position (0-indexed from right)
            
        Returns:
            True if bit is set, False otherwise
            
        Time: O(1), Space: O(1)
        """
        return bool(n & (1 << position))
    
    @staticmethod
    def get_bit(n: int, position: int) -> int:
        """Get the bit value at given position."""
        return (n >> position) & 1
    
    @staticmethod
    def update_bit(n: int, position: int, value: int) -> int:
        """
        Update bit at given position to specific value.
        
        Args:
            n: Number
            position: Bit position
            value: New bit value (0 or 1)
            
        Returns:
            Updated number
        """
        # Clear the bit first, then set if value is 1
        cleared = n & ~(1 << position)
        return cleared | (value << position)


class CountingSetBits:
    """Various algorithms for counting set bits (1s) in numbers."""
    
    @staticmethod
    def count_set_bits_naive(n: int) -> int:
        """
        Count set bits using naive approach.
        
        Time: O(log n), Space: O(1)
        """
        count = 0
        while n:
            count += n & 1
            n >>= 1
        return count
    
    @staticmethod
    def count_set_bits_brian_kernighan(n: int) -> int:
        """
        Count set bits using Brian Kernighan's algorithm.
        Efficient: Only iterates for each set bit.
        
        Time: O(number of set bits), Space: O(1)
        """
        count = 0
        while n:
            n &= n - 1  # Remove rightmost set bit
            count += 1
        return count
    
    @staticmethod
    def count_set_bits_builtin(n: int) -> int:
        """Count set bits using built-in function."""
        return bin(n).count('1')
    
    @staticmethod
    def count_set_bits_lookup_table(n: int) -> int:
        """
        Count set bits using lookup table for optimization.
        Precompute for 8-bit values.
        """
        # Precomputed table for 0-255
        table = [0] * 256
        for i in range(256):
            table[i] = (i & 1) + table[i // 2]
        
        count = 0
        while n:
            count += table[n & 0xFF]  # Process 8 bits at a time
            n >>= 8
        
        return count
    
    @staticmethod
    def count_set_bits_range(start: int, end: int) -> int:
        """Count total set bits in range [start, end]."""
        total = 0
        for i in range(start, end + 1):
            total += CountingSetBits.count_set_bits_brian_kernighan(i)
        return total


class FastExponentiation:
    """Fast exponentiation using bit manipulation."""
    
    @staticmethod
    def power_iterative(base: int, exp: int) -> int:
        """
        Calculate base^exp using iterative bit manipulation.
        
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
        current_power = base
        
        while exp > 0:
            # If current bit is set, multiply result by current power
            if exp & 1:
                result *= current_power
            
            # Square the current power for next bit
            current_power *= current_power
            exp >>= 1  # Move to next bit
        
        return result
    
    @staticmethod
    def power_recursive(base: int, exp: int) -> int:
        """
        Calculate base^exp using recursive approach.
        
        Time: O(log exp), Space: O(log exp)
        """
        if exp == 0:
            return 1
        if exp == 1:
            return base
        
        # If exponent is even: base^exp = (base^(exp/2))^2
        if exp % 2 == 0:
            half_power = FastExponentiation.power_recursive(base, exp // 2)
            return half_power * half_power
        else:
            # If exponent is odd: base^exp = base * base^(exp-1)
            return base * FastExponentiation.power_recursive(base, exp - 1)
    
    @staticmethod
    def power_modular(base: int, exp: int, mod: int) -> int:
        """
        Calculate (base^exp) % mod efficiently.
        
        Args:
            base: Base number
            exp: Exponent
            mod: Modulus
            
        Returns:
            (base^exp) % mod
            
        Time: O(log exp), Space: O(1)
        """
        if mod == 1:
            return 0
        
        result = 1
        base = base % mod
        
        while exp > 0:
            if exp & 1:
                result = (result * base) % mod
            
            exp >>= 1
            base = (base * base) % mod
        
        return result


class BitBasicsDemo:
    """Demonstration and testing of bit manipulation basics."""
    
    @staticmethod
    def demonstrate_binary_representation():
        """Demonstrate binary representation concepts."""
        print("=== BINARY REPRESENTATION ===")
        numbers = [5, 13, 255, -5, 0]
        
        for num in numbers:
            BinaryRepresentation.print_binary_analysis(num)
    
    @staticmethod
    def demonstrate_bitwise_operators():
        """Demonstrate all bitwise operators."""
        print("=== BITWISE OPERATORS ===")
        a, b = 12, 10  # 1100 and 1010 in binary
        
        operations = [
            BitwiseOperators.and_operation(a, b),
            BitwiseOperators.or_operation(a, b),
            BitwiseOperators.xor_operation(a, b),
            BitwiseOperators.not_operation(a),
            BitwiseOperators.left_shift(a, 2),
            BitwiseOperators.right_shift(a, 2)
        ]
        
        for op in operations:
            print(f"{op['operation']}: {op['result']} ({op['binary_result']})")
            print(f"  {op['explanation']}")
            print()
    
    @staticmethod
    def demonstrate_bit_operations():
        """Demonstrate individual bit operations."""
        print("=== BIT OPERATIONS ===")
        n = 12  # 1100 in binary
        print(f"Original number: {n} ({bin(n)})")
        
        # Set bit at position 1
        set_result = BitOperations.set_bit(n, 1)
        print(f"Set bit 1: {set_result} ({bin(set_result)})")
        
        # Clear bit at position 2
        clear_result = BitOperations.clear_bit(n, 2)
        print(f"Clear bit 2: {clear_result} ({bin(clear_result)})")
        
        # Toggle bit at position 0
        toggle_result = BitOperations.toggle_bit(n, 0)
        print(f"Toggle bit 0: {toggle_result} ({bin(toggle_result)})")
        
        # Check various bits
        for i in range(4):
            is_set = BitOperations.check_bit(n, i)
            print(f"Bit {i} is {'set' if is_set else 'not set'}")
    
    @staticmethod
    def demonstrate_counting_set_bits():
        """Demonstrate set bit counting algorithms."""
        print("=== COUNTING SET BITS ===")
        numbers = [7, 15, 255, 1023]
        
        for num in numbers:
            print(f"Number: {num} ({bin(num)})")
            
            # Compare different methods
            naive = CountingSetBits.count_set_bits_naive(num)
            brian = CountingSetBits.count_set_bits_brian_kernighan(num)
            builtin = CountingSetBits.count_set_bits_builtin(num)
            
            print(f"  Naive: {naive}")
            print(f"  Brian Kernighan: {brian}")
            print(f"  Built-in: {builtin}")
            print()
    
    @staticmethod
    def demonstrate_fast_exponentiation():
        """Demonstrate fast exponentiation algorithms."""
        print("=== FAST EXPONENTIATION ===")
        test_cases = [(2, 10), (3, 5), (5, 0), (7, 8)]
        
        for base, exp in test_cases:
            iterative = FastExponentiation.power_iterative(base, exp)
            recursive = FastExponentiation.power_recursive(base, exp)
            builtin = base ** exp
            
            print(f"{base}^{exp}:")
            print(f"  Iterative: {iterative}")
            print(f"  Recursive: {recursive}")
            print(f"  Built-in: {builtin}")
            print(f"  All match: {iterative == recursive == builtin}")
            print()
        
        # Demonstrate modular exponentiation
        print("Modular Exponentiation:")
        result = FastExponentiation.power_modular(2, 10, 1000)
        print(f"(2^10) % 1000 = {result}")


def performance_comparison():
    """Compare performance of different algorithms."""
    import time
    
    print("=== PERFORMANCE COMPARISON ===")
    
    # Test counting set bits
    test_numbers = list(range(1, 10000))
    
    # Brian Kernighan's algorithm
    start = time.time()
    for num in test_numbers:
        CountingSetBits.count_set_bits_brian_kernighan(num)
    brian_time = time.time() - start
    
    # Built-in method
    start = time.time()
    for num in test_numbers:
        CountingSetBits.count_set_bits_builtin(num)
    builtin_time = time.time() - start
    
    print(f"Brian Kernighan's Algorithm: {brian_time:.4f} seconds")
    print(f"Built-in bin().count(): {builtin_time:.4f} seconds")
    print(f"Brian Kernighan is {builtin_time/brian_time:.2f}x {'faster' if brian_time < builtin_time else 'slower'}")


if __name__ == "__main__":
    # Run all demonstrations
    demo = BitBasicsDemo()
    
    demo.demonstrate_binary_representation()
    demo.demonstrate_bitwise_operators()
    demo.demonstrate_bit_operations()
    demo.demonstrate_counting_set_bits()
    demo.demonstrate_fast_exponentiation()
    
    performance_comparison()
    
    print("\n🎯 Key Takeaways:")
    print("1. Bitwise operations are O(1) and very fast")
    print("2. Brian Kernighan's algorithm is optimal for counting set bits")
    print("3. Fast exponentiation reduces O(n) to O(log n)")
    print("4. Bit manipulation is essential for optimization")
    print("5. Understanding binary representation is fundamental") 