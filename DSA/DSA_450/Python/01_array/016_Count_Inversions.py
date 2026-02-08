"""
Problem: Count Inversions
URL: https://practice.geeksforgeeks.org/problems/inversion-of-array-1587115620/1

Problem Statement:
Given an array of N integers, count the number of inversions in the array.
An inversion occurs when arr[i] > arr[j] and i < j.

Sample Input/Output:
Input: arr = [2, 4, 1, 3, 5]
Output: 3
Explanation: Inversions are (2,1), (4,1), (4,3).

Input: arr = [5, 4, 3, 2, 1]
Output: 10
Explanation: Every pair is an inversion in a reverse-sorted array.
"""


class Solution:
    def Count_Inversions_Merge_Sort_Optimal(self, arr):
        """
        Merge Sort Based - Count inversions during merge step
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        temp = [0] * len(arr)
        return self.Merge_Sort_Count(arr, temp, 0, len(arr) - 1)

    def Count_Inversions_Brute_Force(self, arr):
        """
        Brute Force - Check all pairs
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        """
        count = 0
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                if arr[i] > arr[j]:
                    count += 1
        return count

    def Merge_Sort_Count(self, arr, temp, left, right):
        inv_count = 0
        if left < right:
            mid = (left + right) // 2
            inv_count += self.Merge_Sort_Count(arr, temp, left, mid)
            inv_count += self.Merge_Sort_Count(arr, temp, mid + 1, right)
            inv_count += self.Merge_Count(arr, temp, left, mid, right)
        return inv_count

    def Merge_Count(self, arr, temp, left, mid, right):
        i = left
        j = mid + 1
        k = left
        inv_count = 0
        while i <= mid and j <= right:
            if arr[i] <= arr[j]:
                temp[k] = arr[i]
                i += 1
            else:
                temp[k] = arr[j]
                j += 1
                inv_count += (mid - i + 1)
            k += 1
        while i <= mid:
            temp[k] = arr[i]
            i += 1
            k += 1
        while j <= right:
            temp[k] = arr[j]
            j += 1
            k += 1
        for i in range(left, right + 1):
            arr[i] = temp[i]
        return inv_count


def Test_Count_Inversions():
    solution = Solution()

    class TestCase:
        def __init__(self, arr, expected):
            self.arr = arr
            self.expected = expected

    test_cases = [
        TestCase([2, 4, 1, 3, 5], 3),
        TestCase([5, 4, 3, 2, 1], 10),
        TestCase([1, 20, 6, 4, 5], 5),
        TestCase([1, 2, 3, 4, 5], 0)
    ]

    for tc in test_cases:
        print(f"Array: {tc.arr}, Expected: {tc.expected}")

        arr1 = tc.arr.copy()
        print("Merge Sort:", solution.Count_Inversions_Merge_Sort_Optimal(arr1))
        print("Brute Force:", solution.Count_Inversions_Brute_Force(tc.arr))

        print("-" * 50)


if __name__ == "__main__":
    Test_Count_Inversions()
