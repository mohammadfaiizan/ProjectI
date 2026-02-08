"""
Problem: Minimum Characters to Add at Front to Make String Palindrome
URL: https://www.geeksforgeeks.org/minimum-characters-added-front-make-string-palindrome/

Problem Statement:
Given a string str, find the minimum number of characters to be added at the
front to make the string a palindrome.

Sample Input/Output:
Input: "AACECAAAA"
Output: 2

Input: "ABC"
Output: 2
"""


class Solution:
    def Min_Chars_LPS(self, str_val):
        """
        Using KMP LPS array on str + "$" + reverse(str)
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        rev = str_val[::-1]
        concat = str_val + "$" + rev
        n = len(concat)

        lps = [0] * n
        length = 0
        i = 1
        while i < n:
            if concat[i] == concat[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1

        return len(str_val) - lps[n - 1]

    def Min_Chars_Brute(self, str_val):
        """
        Keep removing last char until palindrome
        Time Complexity: O(n^2)
        Space Complexity: O(n)
        """
        cnt = 0
        s = str_val
        while s:
            rev = s[::-1]
            if s == rev:
                break
            s = s[:-1]
            cnt += 1
        return cnt

    def Min_Chars_Two_Pointer(self, str_val):
        """
        Two pointer - find longest palindromic prefix
        Time Complexity: O(n^2) worst case
        Space Complexity: O(n)
        """
        n = len(str_val)
        i = 0
        j = n - 1
        suffixEnd = n - 1

        while i < j:
            if str_val[i] == str_val[j]:
                i += 1
                j -= 1
            else:
                i = 0
                suffixEnd -= 1
                j = suffixEnd

        return n - suffixEnd - 1


def Test_Min_Chars_Front_Palindrome():
    sol = Solution()
    tests = ["AACECAAAA", "ABC", "BABABAA", "a", "ab", "aaa"]

    for s in tests:
        print(f"Input: {s}")
        print(f"LPS: {sol.Min_Chars_LPS(s)}")
        print(f"Brute: {sol.Min_Chars_Brute(s)}")
        print(f"Two Pointer: {sol.Min_Chars_Two_Pointer(s)}")
        print('-' * 50)


if __name__ == "__main__":
    Test_Min_Chars_Front_Palindrome()
