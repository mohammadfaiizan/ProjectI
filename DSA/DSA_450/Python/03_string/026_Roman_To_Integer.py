"""
Problem: Roman Numeral to Integer
URL: https://practice.geeksforgeeks.org/problems/roman-number-to-integer3201/1

Problem Statement:
Given a string in Roman numeral format, convert it to an integer.

Sample Input/Output:
Input: "III"
Output: 3

Input: "MCMXCIV"
Output: 1994

Input: "IX"
Output: 9
"""


class Solution:
    def Roman_To_Int_Right_To_Left(self, s):
        """
        Iterate right to left, subtract if current < next
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        def val(c):
            if c == 'M':
                return 1000
            if c == 'D':
                return 500
            if c == 'C':
                return 100
            if c == 'L':
                return 50
            if c == 'X':
                return 10
            if c == 'V':
                return 5
            if c == 'I':
                return 1
            return 0

        m = len(s)
        ans = 0
        i = m - 1
        while i >= 0:
            if i > 0 and val(s[i - 1]) < val(s[i]):
                ans += val(s[i]) - val(s[i - 1])
                i -= 2
            else:
                ans += val(s[i])
                i -= 1
        return ans

    def Roman_To_Int_Left_To_Right(self, s):
        """
        Iterate left to right, add or subtract based on next value
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        mp = {
            'I': 1, 'V': 5, 'X': 10, 'L': 50,
            'C': 100, 'D': 500, 'M': 1000
        }

        result = 0
        i = 0
        while i < len(s):
            if i + 1 < len(s) and mp[s[i]] < mp[s[i + 1]]:
                result += mp[s[i + 1]] - mp[s[i]]
                i += 2
            else:
                result += mp[s[i]]
                i += 1
        return result

    def Roman_To_Int_Prev_Track(self, s):
        """
        Track previous value, subtract if prev < current
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        mp = {
            'I': 1, 'V': 5, 'X': 10, 'L': 50,
            'C': 100, 'D': 500, 'M': 1000
        }

        m = len(s)
        ans = mp[s[m - 1]]
        prev = mp[s[m - 1]]
        for i in range(m - 2, -1, -1):
            curr = mp[s[i]]
            if curr >= prev:
                ans += curr
            else:
                ans -= curr
            prev = curr
        return ans


def Test_Roman_To_Integer():
    sol = Solution()
    tests = ["III", "IV", "IX", "LVIII", "MCMXCIV", "MMMDCCXLIX", "CDXLIV"]

    for s in tests:
        print(f"Input: {s}")
        print(f"Right to Left: {sol.Roman_To_Int_Right_To_Left(s)}")
        print(f"Left to Right: {sol.Roman_To_Int_Left_To_Right(s)}")
        print(f"Prev Track: {sol.Roman_To_Int_Prev_Track(s)}")
        print('-' * 50)


if __name__ == "__main__":
    Test_Roman_To_Integer()
