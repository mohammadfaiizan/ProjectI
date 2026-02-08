"""
Problem: Smallest Window Containing All Distinct Characters of String
URL: https://practice.geeksforgeeks.org/problems/smallest-distant-window3132/1

Problem Statement:
Given a string s, find the smallest window (substring) that contains all
distinct characters of the string.

Sample Input/Output:
Input: "aabcbcdbca"
Output: "dbca" (length 4)

Input: "aaab"
Output: "ab" (length 2)
"""

import sys


class Solution:
    def Smallest_Window_Sliding(self, str_val):
        """
        Sliding window approach
        Time Complexity: O(n)
        Space Complexity: O(1) - fixed 256 chars
        """
        n = len(str_val)
        dist_count = len(set(str_val))

        start = 0
        start_index = -1
        min_len = sys.maxsize
        count = 0
        curr_count = [0] * 256

        for j in range(n):
            curr_count[ord(str_val[j])] += 1
            if curr_count[ord(str_val[j])] == 1:
                count += 1

            if count == dist_count:
                while curr_count[ord(str_val[start])] > 1:
                    curr_count[ord(str_val[start])] -= 1
                    start += 1

                len_window = j - start + 1
                if min_len > len_window:
                    min_len = len_window
                    start_index = start

        return str_val[start_index:start_index + min_len]

    def Smallest_Window_Brute(self, str_val):
        """
        Brute force - check all substrings
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        """
        n = len(str_val)
        dist_count = len(set(str_val))
        min_len = sys.maxsize
        res = ""

        for i in range(n):
            count = 0
            visited = [0] * 256
            sub = ""
            for j in range(i, n):
                if visited[ord(str_val[j])] == 0:
                    count += 1
                    visited[ord(str_val[j])] = 1
                sub += str_val[j]
                if count == dist_count:
                    break
            if len(sub) < min_len and count == dist_count:
                res = sub
                min_len = len(res)

        return res


def Test_Smallest_Window_All_Distinct():
    sol = Solution()
    tests = ["aabcbcdbca", "aaab", "abcdef", "aabcbcdbcaabc"]

    for s in tests:
        print(f"Input: {s}")
        print(f"Sliding: {sol.Smallest_Window_Sliding(s)}")
        print(f"Brute: {sol.Smallest_Window_Brute(s)}")
        print('-' * 50)


if __name__ == "__main__":
    Test_Smallest_Window_All_Distinct()
