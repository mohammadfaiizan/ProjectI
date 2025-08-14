class BitManipulation:
    @staticmethod
    def decimal_to_binary(num):
        """
        Convert a decimal number to its binary representation as a string using bit shifting.
        """
        if num == 0:
            return "0"
        binary = ""
        while num > 0:
            binary = str(num & 1) + binary
            num >>= 1
        return binary

    @staticmethod
    def set_bit(num, pos):
        """
        Set the bit at the specified position (0-indexed) to 1.
        """
        mask = 1 << pos
        result = num | mask
        return BitManipulation.decimal_to_binary(result)

    @staticmethod
    def reset_bit(num, pos):
        """
        Reset the bit at the specified position (0-indexed) to 0.
        """
        mask = ~(1 << pos)
        result = num & mask
        return BitManipulation.decimal_to_binary(result)

    @staticmethod
    def toggle_bit(num, pos):
        """
        Toggle the bit at the specified position (0-indexed).
        """
        mask = 1 << pos
        result = num ^ mask
        return BitManipulation.decimal_to_binary(result)

    @staticmethod
    def check_bit(num, pos):
        """
        Check if the bit at the specified position (0-indexed) is set (1).
        Returns True if the bit is 1, otherwise False.
        """
        mask = 1 << pos
        result = num & mask
        return result != 0

    @staticmethod
    def count_set_bits(num):
        """
        Count the number of set bits (1s) in the binary representation of the number.
        """
        count = 0
        while num:
            count += num & 1
            num >>= 1
        return count

    @staticmethod
    def clear_bits_msb_to_pos(num, pos):
        """
        Clear all bits from the most significant bit (MSB) to the specified position (inclusive).
        """
        mask = (1 << pos) - 1
        result = num & mask
        return BitManipulation.decimal_to_binary(result)

    @staticmethod
    def clear_bits_lsb_to_pos(num, pos):
        """
        Clear all bits from the least significant bit (LSB) to the specified position (inclusive).
        """
        mask = ~((1 << (pos + 1)) - 1)
        result = num & mask
        return BitManipulation.decimal_to_binary(result)

    @staticmethod
    def isolate_rightmost_set_bit(num):
        """
        Isolate the rightmost set bit of the number.
        """
        result = num & -num
        return BitManipulation.decimal_to_binary(result)

    @staticmethod
    def remove_rightmost_set_bit(num):
        """
        Remove the rightmost set bit of the number.
        """
        result = num & (num - 1)
        return BitManipulation.decimal_to_binary(result)

    @staticmethod
    def is_power_of_two(num):
        """
        Check if a number is a power of two.
        Returns True if it is, otherwise False.
        """
        return num > 0 and (num & (num - 1)) == 0

# Example usage
if __name__ == "__main__":
    bm = BitManipulation()

    num = 20  # Example number
    pos = 2   # Example position

    print("Decimal to Binary:", bm.decimal_to_binary(num))
    print("Set Bit:", bm.set_bit(num, pos))
    print("Reset Bit:", bm.reset_bit(num, pos))
    print("Toggle Bit:", bm.toggle_bit(num, pos))
    print("Check Bit:", bm.check_bit(num, pos))
    print("Count Set Bits:", bm.count_set_bits(num))
    print("Clear Bits MSB to Position:", bm.clear_bits_msb_to_pos(num, pos))
    print("Clear Bits LSB to Position:", bm.clear_bits_lsb_to_pos(num, pos))
    print("Isolate Rightmost Set Bit:", bm.isolate_rightmost_set_bit(num))
    print("Remove Rightmost Set Bit:", bm.remove_rightmost_set_bit(num))
    print("Is Power of Two:", bm.is_power_of_two(num))
