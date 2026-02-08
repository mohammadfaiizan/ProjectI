"""
Problem: Merge Two Sorted Arrays Without Extra Space
URL: https://practice.geeksforgeeks.org/problems/merge-two-sorted-arrays5135/1

Problem Statement:
Given two sorted arrays arr1[] of size N and arr2[] of size M, merge both arrays without
using extra space. Modify arr1 to contain first N smallest and arr2 to contain remaining
M elements in sorted order.

Sample Input/Output:
Input: arr1 = [1, 3, 5, 7], arr2 = [0, 2, 6, 8, 9]
Output: arr1 = [0, 1, 2, 3], arr2 = [5, 6, 7, 8, 9]

Input: arr1 = [10, 12], arr2 = [5, 18, 20]
Output: arr1 = [5, 10], arr2 = [12, 18, 20]
"""


class Solution:
    def Merge_Gap_Method_Optimal(self, arr1, arr2):
        """
        Gap Method (Shell Sort Variant) - Compare elements at gap distance
        Time Complexity: O((n+m) * log(n+m))
        Space Complexity: O(1)
        """
        n = len(arr1)
        m = len(arr2)
        gap = (n + m + 1) // 2
        while gap > 0:
            i = 0
            j = gap
            while j < n + m:
                val_i = arr1[i] if i < n else arr2[i - n]
                val_j = arr1[j] if j < n else arr2[j - n]
                if val_i > val_j:
                    if i < n:
                        if j < n:
                            arr1[i], arr1[j] = arr1[j], arr1[i]
                        else:
                            arr1[i], arr2[j - n] = arr2[j - n], arr1[i]
                    else:
                        if j < n:
                            arr2[i - n], arr1[j] = arr1[j], arr2[i - n]
                        else:
                            arr2[i - n], arr2[j - n] = arr2[j - n], arr2[i - n]
                i += 1
                j += 1
            if gap == 1:
                break
            gap = (gap + 1) // 2

    def Merge_Compare_And_Sort(self, arr1, arr2):
        """
        Compare and Sort - Swap larger of arr1 with smaller of arr2
        Time Complexity: O(n * m)
        Space Complexity: O(1)
        """
        n = len(arr1)
        m = len(arr2)
        for i in range(n - 1, -1, -1):
            if arr1[i] > arr2[0]:
                arr1[i], arr2[0] = arr2[0], arr1[i]
                first = arr2[0]
                k = 1
                while k < m and arr2[k] < first:
                    arr2[k - 1] = arr2[k]
                    k += 1
                arr2[k - 1] = first

    def Merge_Extra_Space(self, arr1, arr2):
        """
        Extra Space Merge - Standard merge into new array
        Time Complexity: O(n + m)
        Space Complexity: O(n + m)
        """
        result = []
        i = 0
        j = 0
        while i < len(arr1) and j < len(arr2):
            if arr1[i] <= arr2[j]:
                result.append(arr1[i])
                i += 1
            else:
                result.append(arr2[j])
                j += 1
        while i < len(arr1):
            result.append(arr1[i])
            i += 1
        while j < len(arr2):
            result.append(arr2[j])
            j += 1
        return result


def Test_Merge_Two_Sorted_Arrays():
    solution = Solution()

    test_cases = [
        ([1, 3, 5, 7], [0, 2, 6, 8, 9]),
        ([10, 12], [5, 18, 20]),
        ([1, 2, 3], [4, 5, 6]),
        ([2, 4, 6], [1, 3, 5])
    ]

    for arr1, arr2 in test_cases:
        print(f"arr1: {arr1}, arr2: {arr2}")

        a1 = arr1[:]
        a2 = arr2[:]
        solution.Merge_Gap_Method_Optimal(a1, a2)
        print(f"Gap Method - arr1: {a1}, arr2: {a2}")

        a1 = arr1[:]
        a2 = arr2[:]
        solution.Merge_Compare_And_Sort(a1, a2)
        print(f"Compare&Sort - arr1: {a1}, arr2: {a2}")

        a1 = arr1[:]
        a2 = arr2[:]
        merged = solution.Merge_Extra_Space(a1, a2)
        print(f"Extra Space: {merged}")
        print("-" * 50)


if __name__ == "__main__":
    Test_Merge_Two_Sorted_Arrays()
