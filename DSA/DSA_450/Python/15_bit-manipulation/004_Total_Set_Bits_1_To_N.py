"""
Problem: Count Total Set Bits from 1 to N
URL: https://practice.geeksforgeeks.org/problems/count-total-set-bits-1587115620/1

Problem Statement:
Count the total number of set bits in all numbers from 1 to N.

Sample Input/Output:
Input: N=4
Output: 5

Input: N=17
Output: 35
"""


class Solution:
    def Total_Bits_Recursive(self, N):
        """
        Recursive using power-of-2 pattern
        Time Complexity: O(log N)
        Space Complexity: O(log N)
        """
        if N <= 0:
            return 0
        if N == 1:
            return 1

        x = 0
        while (1 << x) <= N:
            x += 1
        x -= 1

        bits_upto_2x = x * (1 << (x - 1))
        msb_from_2x_to_N = N - (1 << x) + 1
        rest = N - (1 << x)

        return bits_upto_2x + msb_from_2x_to_N + self.Total_Bits_Recursive(rest)

    def Total_Bits_Brute(self, N):
        """
        Count each number
        Time Complexity: O(N log N)
        Space Complexity: O(1)
        """
        total = 0
        for i in range(1, N + 1):
            num = i
            while num:
                num &= (num - 1)
                total += 1
        return total


def Test_Total_Set_Bits_1_To_N():
    solution = Solution()

    print("Testing Total_Bits_Recursive:")
    print("N=4 ->", solution.Total_Bits_Recursive(4), "(expected: 5)")
    print("N=17 ->", solution.Total_Bits_Recursive(17), "(expected: 35)")

    print("\nTesting Total_Bits_Brute:")
    print("N=4 ->", solution.Total_Bits_Brute(4), "(expected: 5)")
    print("N=17 ->", solution.Total_Bits_Brute(17), "(expected: 35)")


if __name__ == "__main__":
    Test_Total_Set_Bits_1_To_N()
