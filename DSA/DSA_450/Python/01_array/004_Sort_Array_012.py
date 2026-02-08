"""
Problem: Sort an Array of 0s, 1s and 2s
URL: https://practice.geeksforgeeks.org/problems/sort-an-array-of-0s-1s-and-2s4231/1

Problem Statement:
Given an array of size N containing only 0s, 1s, and 2s, sort the array in ascending order
without using any sorting algorithm (Dutch National Flag Problem).

Sample Input/Output:
Input: arr = [0, 2, 1, 2, 0]
Output: [0, 0, 1, 2, 2]

Input: arr = [0, 1, 0, 1, 2, 1, 2, 0]
Output: [0, 0, 0, 1, 1, 1, 2, 2]
"""


class Solution:
    def Sort_012_Dutch_National_Flag_Optimal(self, arr):
        """
        Dutch National Flag - Three pointer approach (low, mid, high)
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        low, mid, high = 0, 0, len(arr) - 1
        while mid <= high:
            if arr[mid] == 0:
                arr[mid], arr[low] = arr[low], arr[mid]
                mid += 1
                low += 1
            elif arr[mid] == 1:
                mid += 1
            else:
                arr[mid], arr[high] = arr[high], arr[mid]
                high -= 1

    def Sort_012_Counting(self, arr):
        """
        Counting Approach - Count 0s, 1s, 2s and overwrite
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        c0 = c1 = c2 = 0
        for x in arr:
            if x == 0:
                c0 += 1
            elif x == 1:
                c1 += 1
            else:
                c2 += 1
        i = 0
        while c0 > 0:
            arr[i] = 0
            i += 1
            c0 -= 1
        while c1 > 0:
            arr[i] = 1
            i += 1
            c1 -= 1
        while c2 > 0:
            arr[i] = 2
            i += 1
            c2 -= 1


def Test_Sort_Array_012():
    solution = Solution()

    test_cases = [
        [0, 2, 1, 2, 0],
        [0, 1, 0, 1, 2, 1, 2, 0],
        [2, 2, 2, 0, 0, 0, 1, 1],
        [0],
        [1, 0]
    ]

    for arr in test_cases:
        print("Original:", arr)

        arr1 = arr.copy()
        arr2 = arr.copy()

        solution.Sort_012_Dutch_National_Flag_Optimal(arr1)
        print("Dutch National Flag:", arr1)

        solution.Sort_012_Counting(arr2)
        print("Counting:", arr2)

        print("-" * 50)


if __name__ == "__main__":
    Test_Sort_Array_012()
