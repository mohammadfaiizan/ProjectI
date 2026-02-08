"""
Problem: Check if Number is Power of Two
URL: https://practice.geeksforgeeks.org/problems/power-of-2-1587115620/1

Problem Statement:
Check if a given positive number is a power of 2.

Sample Input/Output:
Input: 1
Output: true

Input: 16
Output: true

Input: 18
Output: false

Input: 0
Output: false
"""

import math


class Solution:
    def Power_Two_Bit_Trick(self, n):
        """
        Bit trick: n > 0 && (n & (n-1)) == 0
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return n > 0 and (n & (n - 1)) == 0

    def Power_Two_Count_Bits(self, n):
        """
        Count set bits, must be exactly 1
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        if n <= 0:
            return False
        count = 0
        temp = n
        while temp:
            temp &= (temp - 1)
            count += 1
        return count == 1

    def Power_Two_Log(self, n):
        """
        Use log2
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if n <= 0:
            return False
        log_val = math.log2(n)
        return log_val == math.floor(log_val)


def Test_Power_Of_Two():
    solution = Solution()

    print("Testing Power_Two_Bit_Trick:")
    print("1 ->", solution.Power_Two_Bit_Trick(1), "(expected: True)")
    print("16 ->", solution.Power_Two_Bit_Trick(16), "(expected: True)")
    print("18 ->", solution.Power_Two_Bit_Trick(18), "(expected: False)")
    print("0 ->", solution.Power_Two_Bit_Trick(0), "(expected: False)")

    print("\nTesting Power_Two_Count_Bits:")
    print("1 ->", solution.Power_Two_Count_Bits(1), "(expected: True)")
    print("16 ->", solution.Power_Two_Count_Bits(16), "(expected: True)")
    print("18 ->", solution.Power_Two_Count_Bits(18), "(expected: False)")
    print("0 ->", solution.Power_Two_Count_Bits(0), "(expected: False)")

    print("\nTesting Power_Two_Log:")
    print("1 ->", solution.Power_Two_Log(1), "(expected: True)")
    print("16 ->", solution.Power_Two_Log(16), "(expected: True)")
    print("18 ->", solution.Power_Two_Log(18), "(expected: False)")
    print("0 ->", solution.Power_Two_Log(0), "(expected: False)")


if __name__ == "__main__":
    Test_Power_Of_Two()
