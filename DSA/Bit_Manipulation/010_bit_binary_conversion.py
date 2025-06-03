"""
🔢 BINARY REPRESENTATIONS & CONVERSIONS
======================================

This module covers binary representation systems and conversion algorithms.
These techniques are fundamental for understanding computer number systems.

Topics Covered:
1. Convert Decimal to Binary
2. Convert Binary to Decimal
3. Convert Integer to IEEE 754 Representation
4. Float to Binary Approximation

Author: Interview Preparation Collection
LeetCode Problems: 190, 191, 762, 1009, 1432
"""

class DecimalToBinary:
    """Converting decimal numbers to binary representation."""
    
    @staticmethod
    def decimal_to_binary_simple(n: int) -> str:
        """
        Convert decimal to binary using division by 2 method.
        
        Args:
            n: Decimal number
            
        Returns:
            Binary string representation
            
        Time: O(log n), Space: O(log n)
        """
        if n == 0:
            return "0"
        
        result = []
        negative = n < 0
        n = abs(n)
        
        while n > 0:
            result.append(str(n % 2))
            n //= 2
        
        binary = ''.join(reversed(result))
        return f"-{binary}" if negative else binary
    
    @staticmethod
    def decimal_to_binary_bitwise(n: int) -> str:
        """
        Convert decimal to binary using bitwise operations.
        
        Args:
            n: Decimal number
            
        Returns:
            Binary string representation
            
        Time: O(log n), Space: O(log n)
        """
        if n == 0:
            return "0"
        
        result = []
        negative = n < 0
        n = abs(n)
        
        while n > 0:
            result.append(str(n & 1))  # Get least significant bit
            n >>= 1  # Right shift by 1
        
        binary = ''.join(reversed(result))
        return f"-{binary}" if negative else binary
    
    @staticmethod
    def decimal_to_binary_recursive(n: int) -> str:
        """
        Convert decimal to binary using recursion.
        
        Args:
            n: Decimal number
            
        Returns:
            Binary string representation
            
        Time: O(log n), Space: O(log n)
        """
        def convert_positive(num):
            if num == 0:
                return ""
            return convert_positive(num // 2) + str(num % 2)
        
        if n == 0:
            return "0"
        
        negative = n < 0
        result = convert_positive(abs(n))
        return f"-{result}" if negative else result
    
    @staticmethod
    def decimal_to_binary_fixed_width(n: int, width: int = 32) -> str:
        """
        Convert decimal to binary with fixed width (useful for representing negative numbers).
        
        Args:
            n: Decimal number
            width: Width of binary representation
            
        Returns:
            Fixed-width binary string
            
        Time: O(width), Space: O(width)
        """
        if n >= 0:
            binary = bin(n)[2:]  # Remove '0b' prefix
            return binary.zfill(width)
        else:
            # Two's complement representation
            positive = (1 << width) + n  # Add 2^width to negative number
            return bin(positive)[2:].zfill(width)
    
    @staticmethod
    def decimal_to_binary_with_steps(n: int) -> tuple:
        """
        Convert decimal to binary with step-by-step explanation.
        
        Args:
            n: Decimal number
            
        Returns:
            Tuple (binary_result, conversion_steps)
        """
        steps = []
        original_n = n
        
        if n == 0:
            return "0", ["0 in binary is 0"]
        
        negative = n < 0
        n = abs(n)
        remainders = []
        
        while n > 0:
            remainder = n % 2
            remainders.append(remainder)
            steps.append(f"{n} ÷ 2 = {n // 2} remainder {remainder}")
            n //= 2
        
        binary = ''.join(str(r) for r in reversed(remainders))
        steps.append(f"Reading remainders from bottom to top: {binary}")
        
        if negative:
            binary = f"-{binary}"
            steps.append(f"Adding negative sign: {binary}")
        
        return binary, steps


class BinaryToDecimal:
    """Converting binary numbers to decimal representation."""
    
    @staticmethod
    def binary_to_decimal_simple(binary_str: str) -> int:
        """
        Convert binary string to decimal using positional notation.
        
        Args:
            binary_str: Binary string (e.g., "1010")
            
        Returns:
            Decimal number
            
        Time: O(n), Space: O(1)
        """
        negative = binary_str.startswith('-')
        if negative:
            binary_str = binary_str[1:]
        
        decimal = 0
        power = 0
        
        # Process from right to left
        for i in range(len(binary_str) - 1, -1, -1):
            if binary_str[i] == '1':
                decimal += 2 ** power
            power += 1
        
        return -decimal if negative else decimal
    
    @staticmethod
    def binary_to_decimal_bitwise(binary_str: str) -> int:
        """
        Convert binary string to decimal using bitwise operations.
        
        Args:
            binary_str: Binary string
            
        Returns:
            Decimal number
            
        Time: O(n), Space: O(1)
        """
        negative = binary_str.startswith('-')
        if negative:
            binary_str = binary_str[1:]
        
        decimal = 0
        
        for bit in binary_str:
            decimal = (decimal << 1) + int(bit)
        
        return -decimal if negative else decimal
    
    @staticmethod
    def binary_to_decimal_horner(binary_str: str) -> int:
        """
        Convert binary to decimal using Horner's method.
        
        Args:
            binary_str: Binary string
            
        Returns:
            Decimal number
            
        Time: O(n), Space: O(1)
        """
        negative = binary_str.startswith('-')
        if negative:
            binary_str = binary_str[1:]
        
        decimal = 0
        
        for bit in binary_str:
            decimal = decimal * 2 + int(bit)
        
        return -decimal if negative else decimal
    
    @staticmethod
    def binary_to_decimal_with_steps(binary_str: str) -> tuple:
        """
        Convert binary to decimal with step-by-step explanation.
        
        Args:
            binary_str: Binary string
            
        Returns:
            Tuple (decimal_result, conversion_steps)
        """
        steps = []
        negative = binary_str.startswith('-')
        if negative:
            binary_str = binary_str[1:]
        
        decimal = 0
        terms = []
        
        # Process from right to left for explanation
        for i, bit in enumerate(reversed(binary_str)):
            position = i
            if bit == '1':
                value = 2 ** position
                terms.append(f"{bit} × 2^{position} = {value}")
                decimal += value
            else:
                terms.append(f"{bit} × 2^{position} = 0")
        
        steps.extend(reversed(terms))
        steps.append(f"Sum: {' + '.join(str(2**i) for i, bit in enumerate(reversed(binary_str)) if bit == '1')} = {decimal}")
        
        if negative:
            decimal = -decimal
            steps.append(f"With negative sign: {decimal}")
        
        return decimal, steps


class IEEE754Converter:
    """IEEE 754 floating-point representation converter."""
    
    @staticmethod
    def float_to_ieee754_32bit(num: float) -> str:
        """
        Convert float to IEEE 754 32-bit representation.
        
        IEEE 754 format: [Sign][Exponent (8 bits)][Mantissa (23 bits)]
        
        Args:
            num: Floating-point number
            
        Returns:
            32-bit IEEE 754 binary representation
            
        Time: O(1), Space: O(1)
        """
        import struct
        
        # Pack float as 32-bit IEEE 754 format
        packed = struct.pack('>f', num)
        
        # Unpack as 32-bit unsigned integer
        int_repr = struct.unpack('>I', packed)[0]
        
        # Convert to 32-bit binary string
        binary = format(int_repr, '032b')
        
        return binary
    
    @staticmethod
    def ieee754_32bit_to_float(binary_str: str) -> float:
        """
        Convert IEEE 754 32-bit binary to float.
        
        Args:
            binary_str: 32-bit binary string
            
        Returns:
            Floating-point number
            
        Time: O(1), Space: O(1)
        """
        import struct
        
        # Convert binary string to integer
        int_repr = int(binary_str, 2)
        
        # Pack as 32-bit unsigned integer
        packed = struct.pack('>I', int_repr)
        
        # Unpack as IEEE 754 float
        return struct.unpack('>f', packed)[0]
    
    @staticmethod
    def analyze_ieee754_components(binary_str: str) -> dict:
        """
        Analyze IEEE 754 binary representation components.
        
        Args:
            binary_str: 32-bit IEEE 754 binary string
            
        Returns:
            Dictionary with sign, exponent, mantissa details
            
        Time: O(1), Space: O(1)
        """
        if len(binary_str) != 32:
            raise ValueError("IEEE 754 32-bit representation must be 32 bits")
        
        # Extract components
        sign_bit = binary_str[0]
        exponent_bits = binary_str[1:9]
        mantissa_bits = binary_str[9:32]
        
        # Convert to decimal
        sign = -1 if sign_bit == '1' else 1
        exponent_biased = int(exponent_bits, 2)
        mantissa_fractional = int(mantissa_bits, 2) / (2 ** 23)
        
        # Calculate actual exponent (subtract bias of 127)
        exponent_actual = exponent_biased - 127 if exponent_biased != 0 else -126
        
        # Calculate mantissa (add implicit leading 1 for normalized numbers)
        mantissa = 1 + mantissa_fractional if exponent_biased != 0 else mantissa_fractional
        
        # Special cases
        if exponent_biased == 255:
            if mantissa_fractional == 0:
                value_type = "Infinity"
            else:
                value_type = "NaN"
        elif exponent_biased == 0:
            if mantissa_fractional == 0:
                value_type = "Zero"
            else:
                value_type = "Denormalized"
        else:
            value_type = "Normalized"
        
        return {
            'sign_bit': sign_bit,
            'sign': sign,
            'exponent_bits': exponent_bits,
            'exponent_biased': exponent_biased,
            'exponent_actual': exponent_actual,
            'mantissa_bits': mantissa_bits,
            'mantissa_fractional': mantissa_fractional,
            'mantissa': mantissa,
            'value_type': value_type
        }
    
    @staticmethod
    def manual_float_to_ieee754(num: float) -> str:
        """
        Manually convert float to IEEE 754 (educational purposes).
        
        Args:
            num: Floating-point number
            
        Returns:
            IEEE 754 binary representation with explanation
            
        Time: O(1), Space: O(1)
        """
        if num == 0.0:
            return "00000000000000000000000000000000"
        
        # Handle sign
        sign = '0' if num >= 0 else '1'
        num = abs(num)
        
        # Find integer and fractional parts
        integer_part = int(num)
        fractional_part = num - integer_part
        
        # Convert integer part to binary
        if integer_part == 0:
            integer_binary = "0"
        else:
            integer_binary = bin(integer_part)[2:]
        
        # Convert fractional part to binary
        fractional_binary = ""
        max_precision = 50  # Limit precision to avoid infinite loops
        
        while fractional_part > 0 and len(fractional_binary) < max_precision:
            fractional_part *= 2
            if fractional_part >= 1:
                fractional_binary += "1"
                fractional_part -= 1
            else:
                fractional_binary += "0"
        
        # Combine and normalize
        combined = integer_binary + fractional_binary
        
        # Find the first '1' to normalize
        first_one = combined.find('1')
        if first_one == -1:
            return "00000000000000000000000000000000"  # Zero
        
        # Calculate exponent
        if first_one < len(integer_binary):
            # Number >= 1, exponent is positive
            exponent = len(integer_binary) - 1 - first_one
        else:
            # Number < 1, exponent is negative
            exponent = len(integer_binary) - first_one
        
        # Bias the exponent (add 127)
        biased_exponent = exponent + 127
        
        # Handle overflow/underflow
        if biased_exponent >= 255:
            # Infinity
            return sign + "11111111" + "0" * 23
        elif biased_exponent <= 0:
            # Denormalized number (simplified)
            return sign + "00000000" + "0" * 23
        
        # Convert biased exponent to 8-bit binary
        exponent_binary = format(biased_exponent, '08b')
        
        # Get mantissa (23 bits after the implicit leading 1)
        mantissa_start = first_one + 1
        mantissa = combined[mantissa_start:mantissa_start + 23]
        mantissa = mantissa.ljust(23, '0')  # Pad with zeros if needed
        
        return sign + exponent_binary + mantissa


class BinaryFractionConverter:
    """Convert fractional numbers between decimal and binary."""
    
    @staticmethod
    def decimal_fraction_to_binary(decimal_frac: float, precision: int = 20) -> str:
        """
        Convert decimal fraction to binary.
        
        Args:
            decimal_frac: Decimal fraction (0 < decimal_frac < 1)
            precision: Maximum number of binary digits
            
        Returns:
            Binary fraction string
            
        Time: O(precision), Space: O(precision)
        """
        if decimal_frac <= 0 or decimal_frac >= 1:
            raise ValueError("Input must be a fraction between 0 and 1")
        
        result = "0."
        
        for _ in range(precision):
            decimal_frac *= 2
            if decimal_frac >= 1:
                result += "1"
                decimal_frac -= 1
            else:
                result += "0"
            
            if decimal_frac == 0:
                break
        
        return result
    
    @staticmethod
    def binary_fraction_to_decimal(binary_frac: str) -> float:
        """
        Convert binary fraction to decimal.
        
        Args:
            binary_frac: Binary fraction string (e.g., "0.101")
            
        Returns:
            Decimal fraction
            
        Time: O(n), Space: O(1)
        """
        if not binary_frac.startswith("0."):
            raise ValueError("Binary fraction must start with '0.'")
        
        fractional_part = binary_frac[2:]  # Remove "0."
        decimal = 0.0
        
        for i, bit in enumerate(fractional_part):
            if bit == '1':
                decimal += 2 ** -(i + 1)
        
        return decimal
    
    @staticmethod
    def decimal_to_binary_with_fraction(num: float, precision: int = 20) -> str:
        """
        Convert decimal number (including fraction) to binary.
        
        Args:
            num: Decimal number
            precision: Precision for fractional part
            
        Returns:
            Binary representation
            
        Time: O(log(integer_part) + precision), Space: O(log(integer_part) + precision)
        """
        negative = num < 0
        num = abs(num)
        
        # Separate integer and fractional parts
        integer_part = int(num)
        fractional_part = num - integer_part
        
        # Convert integer part
        if integer_part == 0:
            integer_binary = "0"
        else:
            integer_binary = DecimalToBinary.decimal_to_binary_simple(integer_part)
        
        # Convert fractional part
        if fractional_part == 0:
            result = integer_binary
        else:
            fractional_binary = BinaryFractionConverter.decimal_fraction_to_binary(fractional_part, precision)
            result = integer_binary + fractional_binary[1:]  # Remove "0." from fraction
        
        return f"-{result}" if negative else result


class BinaryConversionDemo:
    """Demonstration of binary conversion techniques."""
    
    @staticmethod
    def demonstrate_decimal_to_binary():
        """Demonstrate decimal to binary conversion."""
        print("=== DECIMAL TO BINARY CONVERSION ===")
        
        test_numbers = [0, 5, 10, 255, -8, 1024]
        
        for num in test_numbers:
            simple = DecimalToBinary.decimal_to_binary_simple(num)
            bitwise = DecimalToBinary.decimal_to_binary_bitwise(num)
            recursive = DecimalToBinary.decimal_to_binary_recursive(num)
            builtin = bin(num)
            fixed_width = DecimalToBinary.decimal_to_binary_fixed_width(num, 8)
            
            print(f"\nNumber: {num}")
            print(f"  Simple method: {simple}")
            print(f"  Bitwise method: {bitwise}")
            print(f"  Recursive method: {recursive}")
            print(f"  Built-in bin(): {builtin}")
            print(f"  Fixed width (8-bit): {fixed_width}")
            
            # Show steps for one example
            if num == 10:
                result, steps = DecimalToBinary.decimal_to_binary_with_steps(num)
                print(f"  Conversion steps for {num}:")
                for step in steps:
                    print(f"    {step}")
    
    @staticmethod
    def demonstrate_binary_to_decimal():
        """Demonstrate binary to decimal conversion."""
        print("\n=== BINARY TO DECIMAL CONVERSION ===")
        
        test_binaries = ["0", "101", "1010", "11111111", "-1000"]
        
        for binary in test_binaries:
            simple = BinaryToDecimal.binary_to_decimal_simple(binary)
            bitwise = BinaryToDecimal.binary_to_decimal_bitwise(binary)
            horner = BinaryToDecimal.binary_to_decimal_horner(binary)
            builtin = int(binary, 2) if not binary.startswith('-') else -int(binary[1:], 2)
            
            print(f"\nBinary: {binary}")
            print(f"  Simple method: {simple}")
            print(f"  Bitwise method: {bitwise}")
            print(f"  Horner method: {horner}")
            print(f"  Built-in int(): {builtin}")
            
            # Show steps for one example
            if binary == "1010":
                result, steps = BinaryToDecimal.binary_to_decimal_with_steps(binary)
                print(f"  Conversion steps for {binary}:")
                for step in steps:
                    print(f"    {step}")
    
    @staticmethod
    def demonstrate_ieee754():
        """Demonstrate IEEE 754 conversion."""
        print("\n=== IEEE 754 CONVERSION ===")
        
        test_floats = [0.0, 1.0, -1.0, 3.14159, 0.1, float('inf'), float('-inf')]
        
        for num in test_floats:
            try:
                ieee754 = IEEE754Converter.float_to_ieee754_32bit(num)
                back_to_float = IEEE754Converter.ieee754_32bit_to_float(ieee754)
                components = IEEE754Converter.analyze_ieee754_components(ieee754)
                
                print(f"\nFloat: {num}")
                print(f"  IEEE 754: {ieee754}")
                print(f"  Back to float: {back_to_float}")
                print(f"  Components:")
                print(f"    Sign: {components['sign']} (bit: {components['sign_bit']})")
                print(f"    Exponent: {components['exponent_actual']} (biased: {components['exponent_biased']}, bits: {components['exponent_bits']})")
                print(f"    Mantissa: {components['mantissa']:.6f} (bits: {components['mantissa_bits']})")
                print(f"    Type: {components['value_type']}")
                
            except (OverflowError, ValueError) as e:
                print(f"  Error: {e}")
    
    @staticmethod
    def demonstrate_fraction_conversion():
        """Demonstrate fractional number conversion."""
        print("\n=== FRACTIONAL NUMBER CONVERSION ===")
        
        # Decimal fractions to binary
        decimal_fractions = [0.5, 0.25, 0.1, 0.125, 0.375]
        
        for frac in decimal_fractions:
            binary_frac = BinaryFractionConverter.decimal_fraction_to_binary(frac, 10)
            back_to_decimal = BinaryFractionConverter.binary_fraction_to_decimal(binary_frac)
            
            print(f"\nDecimal fraction: {frac}")
            print(f"  Binary: {binary_frac}")
            print(f"  Back to decimal: {back_to_decimal}")
            print(f"  Exact match: {abs(frac - back_to_decimal) < 1e-10}")
        
        # Mixed numbers (integer + fraction)
        mixed_numbers = [5.5, 10.25, -3.125, 0.1]
        
        print("\n--- Mixed Numbers ---")
        for num in mixed_numbers:
            binary_mixed = BinaryFractionConverter.decimal_to_binary_with_fraction(num, 15)
            print(f"Decimal: {num} → Binary: {binary_mixed}")
    
    @staticmethod
    def demonstrate_special_cases():
        """Demonstrate special cases and edge conditions."""
        print("\n=== SPECIAL CASES ===")
        
        # Powers of 2
        print("Powers of 2:")
        for i in range(5):
            power = 2 ** i
            binary = DecimalToBinary.decimal_to_binary_simple(power)
            print(f"  2^{i} = {power} → {binary}")
        
        # Numbers close to powers of 2
        print("\nNumbers close to powers of 2:")
        for base in [8, 16, 32]:
            for offset in [-1, 1]:
                num = base + offset
                binary = DecimalToBinary.decimal_to_binary_simple(num)
                print(f"  {num} → {binary}")
        
        # Very small fractions
        print("\nVery small fractions:")
        small_fractions = [0.001, 0.0001, 1/3, 1/7]
        for frac in small_fractions:
            try:
                binary_frac = BinaryFractionConverter.decimal_fraction_to_binary(frac, 20)
                print(f"  {frac} → {binary_frac}")
            except ValueError:
                print(f"  {frac} → Cannot convert (not a proper fraction)")


def conversion_accuracy_analysis():
    """Analyze accuracy of different conversion methods."""
    print("\n=== CONVERSION ACCURACY ANALYSIS ===")
    
    import random
    
    # Test conversion accuracy
    test_numbers = [random.randint(-1000, 1000) for _ in range(10)]
    
    print("Testing decimal ↔ binary conversion accuracy:")
    for num in test_numbers[:5]:
        # Convert to binary and back
        binary = DecimalToBinary.decimal_to_binary_simple(num)
        back_to_decimal = BinaryToDecimal.binary_to_decimal_simple(binary)
        
        print(f"  {num} → {binary} → {back_to_decimal} (Match: {num == back_to_decimal})")
    
    print("\nTesting float → IEEE 754 → float accuracy:")
    test_floats = [3.14159, 0.1, 0.2, 0.3, 2.718281828]
    
    for num in test_floats:
        ieee754 = IEEE754Converter.float_to_ieee754_32bit(num)
        back_to_float = IEEE754Converter.ieee754_32bit_to_float(ieee754)
        error = abs(num - back_to_float)
        
        print(f"  {num} → {back_to_float} (Error: {error:.2e})")


def practical_applications():
    """Discuss practical applications of binary conversions."""
    print("\n=== PRACTICAL APPLICATIONS ===")
    
    print("Real-world uses of binary conversion:")
    print("1. Computer graphics: Color representation (RGB to hex)")
    print("2. Network protocols: IP address conversion")
    print("3. File formats: Binary data encoding")
    print("4. Cryptography: Data representation and manipulation")
    print("5. Embedded systems: Hardware register programming")
    print("6. Computer architecture: Instruction encoding")
    print("7. Data compression: Bit-level data manipulation")
    print("8. Scientific computing: Floating-point precision analysis")


if __name__ == "__main__":
    # Run all demonstrations
    demo = BinaryConversionDemo()
    
    demo.demonstrate_decimal_to_binary()
    demo.demonstrate_binary_to_decimal()
    demo.demonstrate_ieee754()
    demo.demonstrate_fraction_conversion()
    demo.demonstrate_special_cases()
    
    conversion_accuracy_analysis()
    practical_applications()
    
    print("\n🎯 Key Binary Conversion Patterns:")
    print("1. Decimal to binary: Repeated division by 2")
    print("2. Binary to decimal: Positional notation with powers of 2")
    print("3. IEEE 754: Sign + biased exponent + normalized mantissa")
    print("4. Fractions: Repeated multiplication by 2 for binary digits")
    print("5. Fixed-width: Two's complement for negative numbers")
    print("6. Precision: Floating-point has inherent limitations") 