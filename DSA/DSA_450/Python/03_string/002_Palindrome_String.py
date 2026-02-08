"""
Problem: Palindrome String
URL: https://practice.geeksforgeeks.org/problems/palindrome-string0817/1

Problem Statement:
Given a string S, check if it is palindrome or not.

Sample Input/Output:
Input: S = "abba"
Output: 1

Input: S = "abc"
Output: 0
"""


class Solution:
    def Is_Palindrome_Two_Pointer(self, s):
        """
        Two Pointer - compare from both ends
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        n = len(s)
        for i in range(n // 2):
            if s[i] != s[n - i - 1]:
                return 0
        return 1

    def Is_Palindrome_Reverse(self, s):
        """
        Reverse and compare
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        return 1 if s == s[::-1] else 0

    def Is_Palindrome_Recursive(self, s, left, right):
        """
        Recursive check
        Time Complexity: O(n)
        Space Complexity: O(n) recursion stack
        """
        if left >= right:
            return 1
        if s[left] != s[right]:
            return 0
        return self.Is_Palindrome_Recursive(s, left + 1, right - 1)


def Test_Palindrome_String():
    sol = Solution()
    tests = ["abba", "abc", "a", "aa", "racecar", "abcba", "abcd"]

    for s in tests:
        print(f"Input: {s}")
        print(f"Two Pointer: {sol.Is_Palindrome_Two_Pointer(s)}")
        print(f"Reverse: {sol.Is_Palindrome_Reverse(s)}")
        print(f"Recursive: {sol.Is_Palindrome_Recursive(s, 0, len(s) - 1)}")
        print('-' * 50)


if __name__ == "__main__":
    Test_Palindrome_String()
