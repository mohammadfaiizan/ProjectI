"""
Problem: Move All Negative Elements to One Side
URL: https://www.geeksforgeeks.org/move-negative-numbers-beginning-positive-end-constant-extra-space/

Problem Statement:
Given an unsorted array of both negative and positive integers, move all negative numbers
to the beginning and all positive numbers to the end. Order of elements is not important.

Sample Input/Output:
Input: arr = [-12, 11, -13, -5, 6, -7, 5, -3, -6]
Output: [-12, -13, -5, -7, -3, -6, 11, 6, 5] (one possible output)

Input: arr = [1, -1, 3, 2, -7, -5, 11, 6]
Output: [-1, -7, -5, 3, 2, 1, 11, 6] (one possible output)
"""


class Solution:
    def Negative_One_Side_Two_Pointer_Optimal(self, arr):
        """
        Two Pointer Approach - Pointers from both ends
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        left, right = 0, len(arr) - 1
        while left <= right:
            if arr[left] < 0 and arr[right] < 0:
                left += 1
            elif arr[left] > 0 and arr[right] < 0:
                arr[left], arr[right] = arr[right], arr[left]
                left += 1
                right -= 1
            elif arr[left] > 0 and arr[right] > 0:
                right -= 1
            else:
                left += 1
                right -= 1

    def Negative_One_Side_Partition(self, arr):
        """
        Partition Based - Similar to quicksort partition around 0
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        j = 0
        for i in range(len(arr)):
            if arr[i] < 0:
                if i != j:
                    arr[i], arr[j] = arr[j], arr[i]
                j += 1


def Test_Negative_Elements_One_Side():
    solution = Solution()

    test_cases = [
        [-12, 11, -13, -5, 6, -7, 5, -3, -6],
        [1, -1, 3, 2, -7, -5, 11, 6],
        [-1, -2, -3],
        [1, 2, 3]
    ]

    for arr in test_cases:
        print("Original:", arr)

        arr1 = arr.copy()
        arr2 = arr.copy()

        solution.Negative_One_Side_Two_Pointer_Optimal(arr1)
        print("Two Pointer:", arr1)

        solution.Negative_One_Side_Partition(arr2)
        print("Partition:", arr2)

        print("-" * 50)


if __name__ == "__main__":
    Test_Negative_Elements_One_Side()
