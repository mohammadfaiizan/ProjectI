"""
Problem: Cyclically Rotate an Array by One
URL: https://practice.geeksforgeeks.org/problems/cyclically-rotate-an-array-by-one2614/1

Problem Statement:
Given an array, rotate the array by one position in clock-wise direction.
The last element becomes the first element.

Sample Input/Output:
Input: arr = [1, 2, 3, 4, 5]
Output: [5, 1, 2, 3, 4]

Input: arr = [9, 8, 7, 6, 4, 2, 1, 3]
Output: [3, 9, 8, 7, 6, 4, 2, 1]
"""


class Solution:
    def Rotate_By_One_Shift_Optimal(self, arr):
        """
        Shift Based - Store last element and shift all right
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        n = len(arr)
        temp = arr[n - 1]
        for i in range(n - 1, 0, -1):
            arr[i] = arr[i - 1]
        arr[0] = temp

    def Rotate_By_One_STL(self, arr):
        """
        STL Rotate - Using rotate with reverse iterators
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        arr.insert(0, arr.pop())


def Test_Cyclically_Rotate():
    solution = Solution()

    test_cases = [
        [1, 2, 3, 4, 5],
        [9, 8, 7, 6, 4, 2, 1, 3],
        [1],
        [1, 2]
    ]

    for arr in test_cases:
        print("Original:", arr)

        arr1 = arr.copy()
        arr2 = arr.copy()

        solution.Rotate_By_One_Shift_Optimal(arr1)
        print("Shift:", arr1)

        solution.Rotate_By_One_STL(arr2)
        print("STL:", arr2)

        print("-" * 50)


if __name__ == "__main__":
    Test_Cyclically_Rotate()
