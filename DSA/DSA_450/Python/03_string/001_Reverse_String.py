"""
Problem: Reverse String
URL: https://leetcode.com/problems/reverse-string/

Problem Statement:
Write a function that reverses a string. The input string is given as an array of characters.
You must do this by modifying the input array in-place with O(1) extra memory.

Sample Input/Output:
Input: s = ["h","e","l","l","o"]
Output: ["o","l","l","e","h"]

Input: s = ["H","a","n","n","a","h"]
Output: ["h","a","n","n","a","H"]
"""


class Solution:
    def Reverse_String_Two_Pointer(self, s):
        """
        Two Pointer - swap from both ends towards center
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        left, right = 0, len(s) - 1
        while left < right:
            s[left], s[right] = s[right], s[left]
            left += 1
            right -= 1

    def Reverse_String_STL(self, s):
        """
        Using built-in reverse
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        s.reverse()

    def Reverse_String_Recursive(self, s, left, right):
        """
        Recursive approach
        Time Complexity: O(n)
        Space Complexity: O(n) recursion stack
        """
        if left >= right:
            return
        s[left], s[right] = s[right], s[left]
        self.Reverse_String_Recursive(s, left + 1, right - 1)


def Test_Reverse_String():
    sol = Solution()
    tests = [
        ['h', 'e', 'l', 'l', 'o'],
        ['H', 'a', 'n', 'n', 'a', 'h'],
        ['a'],
        ['a', 'b']
    ]

    for s in tests:
        s1, s2, s3 = s[:], s[:], s[:]
        print(f"Input: {''.join(s)}")

        sol.Reverse_String_Two_Pointer(s1)
        print(f"Two Pointer: {''.join(s1)}")

        sol.Reverse_String_STL(s2)
        print(f"STL: {''.join(s2)}")

        sol.Reverse_String_Recursive(s3, 0, len(s3) - 1)
        print(f"Recursive: {''.join(s3)}")

        print('-' * 50)


if __name__ == "__main__":
    Test_Reverse_String()
