"""
Problem: Divide Two Integers Without Using Multiplication, Division, or Mod
URL: https://leetcode.com/problems/divide-two-integers/

Problem Statement:
Given dividend and divisor, compute quotient without *, /, %.

Sample Input/Output:
Input: 43, 8
Output: 5

Input: 10, 3
Output: 3

Input: -7, 2
Output: -3

Input: INT_MIN, -1
Output: INT_MAX
"""

import sys


class Solution:
    def Divide_Bit_Shift(self, dividend, divisor):
        """
        Double divisor using left shift until > dividend, subtract and accumulate
        Time Complexity: O(log^2 n)
        Space Complexity: O(1)
        """
        if divisor == 0:
            return sys.maxsize
        if dividend == -sys.maxsize - 1 and divisor == -1:
            return sys.maxsize

        negative = (dividend < 0) ^ (divisor < 0)
        dvd = abs(dividend)
        dvs = abs(divisor)

        result = 0
        while dvd >= dvs:
            temp = dvs
            multiple = 1
            while dvd >= (temp << 1):
                temp <<= 1
                multiple <<= 1
            dvd -= temp
            result += multiple

        return -result if negative else result

    def Divide_Subtract(self, dividend, divisor):
        """
        Repeated subtraction
        Time Complexity: O(dividend/divisor)
        Space Complexity: O(1)
        """
        if divisor == 0:
            return sys.maxsize
        if dividend == -sys.maxsize - 1 and divisor == -1:
            return sys.maxsize

        negative = (dividend < 0) ^ (divisor < 0)
        dvd = abs(dividend)
        dvs = abs(divisor)

        result = 0
        while dvd >= dvs:
            dvd -= dvs
            result += 1

        return -result if negative else result


def Test_Divide_Without_Operators():
    solution = Solution()

    print("Testing Divide_Bit_Shift:")
    print("43 / 8 ->", solution.Divide_Bit_Shift(43, 8), "(expected: 5)")
    print("10 / 3 ->", solution.Divide_Bit_Shift(10, 3), "(expected: 3)")
    print("-7 / 2 ->", solution.Divide_Bit_Shift(-7, 2), "(expected: -3)")
    print("INT_MIN / -1 ->", solution.Divide_Bit_Shift(-sys.maxsize - 1, -1), "(expected: INT_MAX)")

    print("\nTesting Divide_Subtract:")
    print("43 / 8 ->", solution.Divide_Subtract(43, 8), "(expected: 5)")
    print("10 / 3 ->", solution.Divide_Subtract(10, 3), "(expected: 3)")
    print("-7 / 2 ->", solution.Divide_Subtract(-7, 2), "(expected: -3)")
    print("INT_MIN / -1 ->", solution.Divide_Subtract(-sys.maxsize - 1, -1), "(expected: INT_MAX)")


if __name__ == "__main__":
    Test_Divide_Without_Operators()
