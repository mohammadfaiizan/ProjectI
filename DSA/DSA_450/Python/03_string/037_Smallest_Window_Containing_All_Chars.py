"""
Problem: Smallest Window in a String Containing All Characters of Another String
URL: https://www.geeksforgeeks.org/find-the-smallest-window-in-a-string-containing-all-characters-of-another-string/

Problem Statement:
Given two strings s and t, find the smallest window in s which contains all
characters of t (including duplicates).

Sample Input/Output:
Input: s = "ADOBECODEBANC", t = "ABC"
Output: "BANC"

Input: s = "this is a test string", t = "tist"
Output: "t stri"
"""

import sys


class Solution:
    def Min_Window_Sliding(self, s, t):
        """
        Sliding window with frequency counting
        Time Complexity: O(n)
        Space Complexity: O(1) - fixed 256
        """
        if len(s) < len(t):
            return ""

        hash_pat = [0] * 256
        hash_str = [0] * 256
        for c in t:
            hash_pat[ord(c)] += 1

        start = 0
        start_index = -1
        min_len = sys.maxsize
        count = 0
        len2 = len(t)

        for j in range(len(s)):
            hash_str[ord(s[j])] += 1
            if hash_str[ord(s[j])] <= hash_pat[ord(s[j])]:
                count += 1

            if count == len2:
                while (hash_str[ord(s[start])] > hash_pat[ord(s[start])] or
                       hash_pat[ord(s[start])] == 0):
                    if hash_str[ord(s[start])] > hash_pat[ord(s[start])]:
                        hash_str[ord(s[start])] -= 1
                    start += 1

                len_window = j - start + 1
                if min_len > len_window:
                    min_len = len_window
                    start_index = start

        if start_index == -1:
            return ""
        return s[start_index:start_index + min_len]

    def Min_Window_Optimized(self, s, t):
        """
        Optimized sliding window with distinct char count
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        m = [0] * 256
        count = 0
        for c in t:
            if m[ord(c)] == 0:
                count += 1
            m[ord(c)] += 1

        ans = sys.maxsize
        start = 0
        i = j = 0

        while j < len(s):
            m[ord(s[j])] -= 1
            if m[ord(s[j])] == 0:
                count -= 1

            if count == 0:
                while count == 0:
                    if ans > j - i + 1:
                        ans = j - i + 1
                        start = i
                    m[ord(s[i])] += 1
                    if m[ord(s[i])] > 0:
                        count += 1
                    i += 1
            j += 1

        return "" if ans == sys.maxsize else s[start:start + ans]


def Test_Smallest_Window_Containing_All():
    sol = Solution()
    tests = [
        ("ADOBECODEBANC", "ABC"),
        ("this is a test string", "tist"),
        ("aa", "aa"),
        ("a", "aa"),
        ("ab", "b")
    ]

    for s, t in tests:
        print(f's: "{s}", t: "{t}"')
        print(f'Sliding: "{sol.Min_Window_Sliding(s, t)}"')
        print(f'Optimized: "{sol.Min_Window_Optimized(s, t)}"')
        print('-' * 50)


if __name__ == "__main__":
    Test_Smallest_Window_Containing_All()
