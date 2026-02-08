"""
Problem: Count Number of Bits to Flip to Convert A to B
URL: https://practice.geeksforgeeks.org/problems/bit-difference-1587115620/1

Problem Statement:
Count the number of bits that need to be flipped to convert number A to number B.

Sample Input/Output:
Input: A=10, B=20
Output: 4

Input: A=7, B=10
Output: 3
"""


class Solution:
    def Bits_Flipped_XOR_Count(self, A, B):
        """
        XOR then count set bits using Brian Kernighan
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        diff = A ^ B
        count = 0
        while diff:
            diff &= (diff - 1)
            count += 1
        return count

    def Bits_Flipped_Loop(self, A, B):
        """
        Check each bit position
        Time Complexity: O(32)
        Space Complexity: O(1)
        """
        count = 0
        for i in range(32):
            if ((A >> i) & 1) != ((B >> i) & 1):
                count += 1
        return count


def Test_Bits_Flipped():
    solution = Solution()

    print("Testing Bits_Flipped_XOR_Count:")
    print("A=10, B=20 ->", solution.Bits_Flipped_XOR_Count(10, 20), "(expected: 4)")
    print("A=7, B=10 ->", solution.Bits_Flipped_XOR_Count(7, 10), "(expected: 3)")

    print("\nTesting Bits_Flipped_Loop:")
    print("A=10, B=20 ->", solution.Bits_Flipped_Loop(10, 20), "(expected: 4)")
    print("A=7, B=10 ->", solution.Bits_Flipped_Loop(7, 10), "(expected: 3)")


if __name__ == "__main__":
    Test_Bits_Flipped()
