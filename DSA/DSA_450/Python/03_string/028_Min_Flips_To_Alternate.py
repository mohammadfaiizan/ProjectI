"""
Problem: Minimum Number of Flips to Make Binary String Alternating
URL: https://practice.geeksforgeeks.org/problems/min-number-of-flips3210/1

Problem Statement:
Given a binary string, find the minimum number of flips required to make it alternating.

Sample Input/Output:
Input: "001"
Output: 1

Input: "0001010111"
Output: 2
"""


class Solution:
    def Min_Flips_Two_Patterns(self, s):
        """
        Compare with both possible alternating patterns (010... and 101...)
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        ans1 = ans2 = 0
        for i in range(len(s)):
            if (i % 2 == 0 and s[i] != '1') or (i % 2 and s[i] != '0'):
                ans1 += 1
            if (i % 2 == 0 and s[i] != '0') or (i % 2 and s[i] != '1'):
                ans2 += 1
        return min(ans1, ans2)

    def Min_Flips_Expected_Char(self, s):
        """
        Build expected char and count mismatches
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        n = len(s)
        flips1 = flips2 = 0
        for i in range(n):
            expected1 = '0' if i % 2 == 0 else '1'
            expected2 = '1' if i % 2 == 0 else '0'
            if s[i] != expected1:
                flips1 += 1
            if s[i] != expected2:
                flips2 += 1
        return min(flips1, flips2)


def Test_Min_Flips_To_Alternate():
    sol = Solution()
    tests = ["001", "0001010111", "01", "10", "1111", "0000", "0101"]

    for s in tests:
        print(f"Input: {s}")
        print(f"Two Patterns: {sol.Min_Flips_Two_Patterns(s)}")
        print(f"Expected Char: {sol.Min_Flips_Expected_Char(s)}")
        print('-' * 50)


if __name__ == "__main__":
    Test_Min_Flips_To_Alternate()
