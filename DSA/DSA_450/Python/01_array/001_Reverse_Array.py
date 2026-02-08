"""
Problem: Reverse an Array
URL: https://www.geeksforgeeks.org/write-a-program-to-reverse-an-array-or-string/

Problem Statement:
Given an array (or string), the task is to reverse the array/string.

Sample Input/Output:
Input: arr = [1, 2, 3, 4, 5]
Output: [5, 4, 3, 2, 1]
Explanation: Array elements are reversed in-place.

Input: arr = [4, 5, 1, 2]
Output: [2, 1, 5, 4]
"""


class Solution:
    def Reverse_Array_Two_Pointer_Optimal(self, arr):
        """
        Two Pointer Iterative - Swap elements from both ends
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        start, end = 0, len(arr) - 1
        while start < end:
            arr[start], arr[end] = arr[end], arr[start]
            start += 1
            end -= 1

    def Reverse_Array_Recursive(self, arr, start, end):
        """
        Recursive Approach - Recursively swap endpoints
        Time Complexity: O(n)
        Space Complexity: O(n) - recursion stack
        """
        if start >= end:
            return
        arr[start], arr[end] = arr[end], arr[start]
        self.Reverse_Array_Recursive(arr, start + 1, end - 1)

    def Reverse_Array_STL(self, arr):
        """
        STL Reverse - Using built-in reverse function
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        arr.reverse()


def Test_Reverse_Array():
    solution = Solution()

    test_cases = [
        [1, 2, 3, 4, 5],
        [4, 5, 1, 2],
        [1],
        [1, 2]
    ]

    for arr in test_cases:
        print("Original:", arr)

        arr1 = arr.copy()
        arr2 = arr.copy()
        arr3 = arr.copy()

        solution.Reverse_Array_Two_Pointer_Optimal(arr1)
        print("Two Pointer:", arr1)

        solution.Reverse_Array_Recursive(arr2, 0, len(arr2) - 1)
        print("Recursive:", arr2)

        solution.Reverse_Array_STL(arr3)
        print("STL:", arr3)

        print("-" * 50)


if __name__ == "__main__":
    Test_Reverse_Array()
