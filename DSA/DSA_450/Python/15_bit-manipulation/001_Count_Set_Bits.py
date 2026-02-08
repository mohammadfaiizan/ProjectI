"""
Problem: Count Set Bits in a Number
URL: https://practice.geeksforgeeks.org/problems/set-bits0143/1

Problem Statement:
Count number of 1s in binary representation of a given number.

Sample Input/Output:
Input: 6
Output: 2

Input: 13
Output: 3

Input: 0
Output: 0

Input: 255
Output: 8
"""


class Solution:
    def Count_Set_Bits_Builtin(self, n):
        """
        Built-in function
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return bin(n).count('1')

    def Count_Set_Bits_Brian_Kernighan(self, n):
        """
        Brian Kernighan's algorithm
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        count = 0
        while n:
            n &= (n - 1)
            count += 1
        return count

    def Count_Set_Bits_Loop(self, n):
        """
        Check each bit
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        count = 0
        while n:
            if n & 1:
                count += 1
            n >>= 1
        return count


def Test_Count_Set_Bits():
    solution = Solution()

    print("Testing Count_Set_Bits_Builtin:")
    print("6 ->", solution.Count_Set_Bits_Builtin(6), "(expected: 2)")
    print("13 ->", solution.Count_Set_Bits_Builtin(13), "(expected: 3)")
    print("0 ->", solution.Count_Set_Bits_Builtin(0), "(expected: 0)")
    print("255 ->", solution.Count_Set_Bits_Builtin(255), "(expected: 8)")

    print("\nTesting Count_Set_Bits_Brian_Kernighan:")
    print("6 ->", solution.Count_Set_Bits_Brian_Kernighan(6), "(expected: 2)")
    print("13 ->", solution.Count_Set_Bits_Brian_Kernighan(13), "(expected: 3)")
    print("0 ->", solution.Count_Set_Bits_Brian_Kernighan(0), "(expected: 0)")
    print("255 ->", solution.Count_Set_Bits_Brian_Kernighan(255), "(expected: 8)")

    print("\nTesting Count_Set_Bits_Loop:")
    print("6 ->", solution.Count_Set_Bits_Loop(6), "(expected: 2)")
    print("13 ->", solution.Count_Set_Bits_Loop(13), "(expected: 3)")
    print("0 ->", solution.Count_Set_Bits_Loop(0), "(expected: 0)")
    print("255 ->", solution.Count_Set_Bits_Loop(255), "(expected: 8)")


if __name__ == "__main__":
    Test_Count_Set_Bits()
