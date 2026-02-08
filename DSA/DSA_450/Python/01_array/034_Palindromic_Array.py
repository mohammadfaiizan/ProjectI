"""
Problem: Palindromic Array
URL: https://practice.geeksforgeeks.org/problems/palindromic-array-1587115620/1

Problem Statement:
Given a positive integer array arr of size N, check if every element of the array
is a palindrome or not. Return 1 if all elements are palindromes, otherwise return 0.

Sample Input/Output:
Input: arr = [111, 222, 333, 444, 555]
Output: 1
Explanation: All elements are palindromes.

Input: arr = [121, 131, 20]
Output: 0
Explanation: 20 is not a palindrome.
"""


class Solution:
    def Palindromic_Array_String_Optimal(self, arr):
        """
        String Conversion - Convert each number to string and check
        Time Complexity: O(n * d) where d is max digits
        Space Complexity: O(d)
        """
        for x in arr:
            s = str(x)
            i = 0
            j = len(s) - 1
            while i < j:
                if s[i] != s[j]:
                    return 0
                i += 1
                j -= 1
        return 1

    def Palindromic_Array_Digit_Reversal(self, arr):
        """
        Digit Reversal - Reverse digits mathematically and compare
        Time Complexity: O(n * d)
        Space Complexity: O(1)
        """
        for x in arr:
            if not self.Is_Palindrome_Number(x):
                return 0
        return 1

    def Is_Palindrome_Number(self, n):
        original = n
        reversed_num = 0
        while n > 0:
            reversed_num = reversed_num * 10 + n % 10
            n //= 10
        return original == reversed_num


def Test_Palindromic_Array():
    solution = Solution()

    class TestCase:
        def __init__(self, arr, expected):
            self.arr = arr
            self.expected = expected

    test_cases = [
        TestCase([111, 222, 333, 444, 555], 1),
        TestCase([121, 131, 20], 0),
        TestCase([1, 2, 3, 4, 5], 1),
        TestCase([12321, 45654, 78987], 1)
    ]

    for tc in test_cases:
        print(f"Array: {tc.arr}, Expected: {tc.expected}")

        print("String:", solution.Palindromic_Array_String_Optimal(tc.arr))
        print("Digit Reversal:", solution.Palindromic_Array_Digit_Reversal(tc.arr))

        print("-" * 50)


if __name__ == "__main__":
    Test_Palindromic_Array()
