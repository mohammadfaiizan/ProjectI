"""
Problem: Split Binary String into Substrings with Equal 0s and 1s
URL: https://www.geeksforgeeks.org/split-the-binary-string-into-substrings-with-equal-number-of-0s-and-1s/

Problem Statement:
Given a binary string, split it into maximum number of substrings such that
each substring contains equal number of 0s and 1s. Return -1 if not possible.

Sample Input/Output:
Input: "0100110101"
Output: 4

Input: "0111100010"
Output: 3

Input: "0"
Output: -1
"""

import math


class Solution:
    def Split_Binary_Counter(self, s):
        """
        Count 0s and 1s, increment result when counts are equal
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        count0 = count1 = cnt = 0
        n = len(s)
        for i in range(n):
            if s[i] == '0':
                count0 += 1
            else:
                count1 += 1
            if count0 == count1:
                cnt += 1
        if count0 != count1:
            return -1
        return cnt

    def Split_Binary_Prefix_Sum(self, s):
        """
        Using prefix sum: treat 0 as -1 and 1 as +1
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        n = len(s)
        sum_val = count = 0
        for i in range(n):
            sum_val += 1 if s[i] == '1' else -1
            if sum_val == 0:
                count += 1
        if sum_val != 0:
            return -1
        return count


def Test_Split_Binary_String():
    sol = Solution()
    tests = ["0100110101", "0111100010", "0", "01", "0011", "000111"]

    for s in tests:
        print(f"Input: {s}")
        print(f"Counter: {sol.Split_Binary_Counter(s)}")
        print(f"Prefix Sum: {sol.Split_Binary_Prefix_Sum(s)}")
        print('-' * 50)


if __name__ == "__main__":
    Test_Split_Binary_String()
