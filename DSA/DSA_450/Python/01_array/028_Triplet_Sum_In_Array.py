"""
Problem: Triplet Sum in Array
URL: https://practice.geeksforgeeks.org/problems/triplet-sum-in-array-1587115621/1

Problem Statement:
Given an array arr[] of distinct integers of size N and a value X, find if there is a
triplet in the array whose sum is equal to X.

Sample Input/Output:
Input: arr = [1, 4, 45, 6, 10, 8], X = 22
Output: true
Explanation: Triplet (4, 10, 8) has sum 22.

Input: arr = [1, 2, 4, 3, 6], X = 10
Output: true
Explanation: Triplet (1, 3, 6) has sum 10.
"""


class Solution:
    def Triplet_Sum_Two_Pointer_Optimal(self, arr, x):
        """
        Sorting + Two Pointer - Fix one element, use two pointers for rest
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        """
        arr = sorted(arr)
        n = len(arr)
        for i in range(n - 2):
            left = i + 1
            right = n - 1
            while left < right:
                sum_val = arr[i] + arr[left] + arr[right]
                if sum_val == x:
                    return True
                elif sum_val < x:
                    left += 1
                else:
                    right -= 1
        return False

    def Triplet_Sum_Hashing(self, arr, x):
        """
        Hashing Approach - Fix one element, use set for pair sum
        Time Complexity: O(n^2)
        Space Complexity: O(n)
        """
        n = len(arr)
        for i in range(n - 1):
            s = set()
            target = x - arr[i]
            for j in range(i + 1, n):
                if target - arr[j] in s:
                    return True
                s.add(arr[j])
        return False

    def Triplet_Sum_Brute_Force(self, arr, x):
        """
        Brute Force - Check all triplets
        Time Complexity: O(n^3)
        Space Complexity: O(1)
        """
        n = len(arr)
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                for k in range(j + 1, n):
                    if arr[i] + arr[j] + arr[k] == x:
                        return True
        return False


def Test_Triplet_Sum():
    solution = Solution()

    test_cases = [
        ([1, 4, 45, 6, 10, 8], 22, True),
        ([1, 2, 4, 3, 6], 10, True),
        ([1, 2, 4, 3, 6], 20, False),
        ([1, 2, 3], 6, True)
    ]

    for arr, x, expected in test_cases:
        print(f"Array: {arr}, X={x}, Expected: {expected}")
        result_two_pointer = solution.Triplet_Sum_Two_Pointer_Optimal(arr, x)
        result_hashing = solution.Triplet_Sum_Hashing(arr, x)
        result_brute = solution.Triplet_Sum_Brute_Force(arr, x)
        print(f"Two Pointer: {result_two_pointer}")
        print(f"Hashing: {result_hashing}")
        print(f"Brute Force: {result_brute}")
        print("-" * 50)


if __name__ == "__main__":
    Test_Triplet_Sum()
