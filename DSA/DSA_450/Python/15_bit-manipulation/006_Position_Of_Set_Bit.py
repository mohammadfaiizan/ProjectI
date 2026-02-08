"""
Problem: Find Position of the Only Set Bit
URL: https://practice.geeksforgeeks.org/problems/find-position-of-set-bit3706/1

Problem Statement:
If a number has exactly one set bit, return its position (1-indexed). Otherwise return -1.

Sample Input/Output:
Input: 2
Output: 2

Input: 5
Output: -1

Input: 32
Output: 6

Input: 0
Output: -1
"""

import math


class Solution:
    def Position_Bit_Log(self, n):
        """
        Use log2 to find position
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if n == 0 or (n & (n - 1)) != 0:
            return -1
        return int(math.log2(n)) + 1

    def Position_Bit_Loop(self, n):
        """
        Shift right and count position
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        if n == 0:
            return -1
        if (n & (n - 1)) != 0:
            return -1
        pos = 0
        temp = n
        while temp:
            pos += 1
            temp >>= 1
        return pos

    def Position_Bit_Power_Check(self, n):
        """
        First check power of 2, then find position
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        if n == 0:
            return -1
        if (n & (n - 1)) != 0:
            return -1
        pos = 1
        temp = n
        while temp != 1:
            temp >>= 1
            pos += 1
        return pos


def Test_Position_Of_Set_Bit():
    solution = Solution()

    print("Testing Position_Bit_Log:")
    print("2 ->", solution.Position_Bit_Log(2), "(expected: 2)")
    print("5 ->", solution.Position_Bit_Log(5), "(expected: -1)")
    print("32 ->", solution.Position_Bit_Log(32), "(expected: 6)")
    print("0 ->", solution.Position_Bit_Log(0), "(expected: -1)")

    print("\nTesting Position_Bit_Loop:")
    print("2 ->", solution.Position_Bit_Loop(2), "(expected: 2)")
    print("5 ->", solution.Position_Bit_Loop(5), "(expected: -1)")
    print("32 ->", solution.Position_Bit_Loop(32), "(expected: 6)")
    print("0 ->", solution.Position_Bit_Loop(0), "(expected: -1)")

    print("\nTesting Position_Bit_Power_Check:")
    print("2 ->", solution.Position_Bit_Power_Check(2), "(expected: 2)")
    print("5 ->", solution.Position_Bit_Power_Check(5), "(expected: -1)")
    print("32 ->", solution.Position_Bit_Power_Check(32), "(expected: 6)")
    print("0 ->", solution.Position_Bit_Power_Check(0), "(expected: -1)")


if __name__ == "__main__":
    Test_Position_Of_Set_Bit()
