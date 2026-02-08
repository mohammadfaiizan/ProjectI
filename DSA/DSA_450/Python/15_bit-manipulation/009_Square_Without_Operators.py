"""
Problem: Calculate Square of a Number Without *, /, pow
URL: https://www.geeksforgeeks.org/calculate-square-of-a-number-without-using-and-pow/

Problem Statement:
Calculate n^2 without using multiplication, division, or pow.

Sample Input/Output:
Input: 5
Output: 25

Input: -7
Output: 49

Input: 0
Output: 0

Input: 12
Output: 144
"""


class Solution:
    def Square_Bit_Shift(self, n):
        """
        For each set bit i in |n|, add n << i; works because n*n = n * sum(2^i for set bits)
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        if n == 0:
            return 0
        num = abs(n)
        result = 0
        temp = num
        i = 0
        while num:
            if num & 1:
                result += (temp << i)
            num >>= 1
            i += 1
        return result

    def Square_Odd_Sum(self, n):
        """
        n^2 = sum of first n odd numbers
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if n == 0:
            return 0
        num = abs(n)
        result = 0
        odd = 1
        for i in range(num):
            result += odd
            odd += 2
        return result


def Test_Square_Without_Operators():
    solution = Solution()

    print("Testing Square_Bit_Shift:")
    print("5 ->", solution.Square_Bit_Shift(5), "(expected: 25)")
    print("-7 ->", solution.Square_Bit_Shift(-7), "(expected: 49)")
    print("0 ->", solution.Square_Bit_Shift(0), "(expected: 0)")
    print("12 ->", solution.Square_Bit_Shift(12), "(expected: 144)")

    print("\nTesting Square_Odd_Sum:")
    print("5 ->", solution.Square_Odd_Sum(5), "(expected: 25)")
    print("-7 ->", solution.Square_Odd_Sum(-7), "(expected: 49)")
    print("0 ->", solution.Square_Odd_Sum(0), "(expected: 0)")
    print("12 ->", solution.Square_Odd_Sum(12), "(expected: 144)")


if __name__ == "__main__":
    Test_Square_Without_Operators()
