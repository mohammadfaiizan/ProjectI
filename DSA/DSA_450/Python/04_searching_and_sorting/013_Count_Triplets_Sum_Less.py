"""
Problem: Count Triplets with Sum Smaller than X
URL: https://practice.geeksforgeeks.org/problems/count-triplets-with-sum-smaller-than-x5549/1

Problem Statement:
Given an array arr[] of distinct integers of size n and a value X, find the count of triplets whose sum is smaller than X.

Sample Input/Output:
Input: n = 4, X = 2, arr[] = {-2, 0, 1, 3}
Output: 2

Input: n = 5, X = 12, arr[] = {5, 1, 3, 4, 7}
Output: 4
"""


class Solution:
    def Count_Triplets_Sorting_Two_Pointer(self, arr, n, X):
        """
        Sort array and use two pointers to count triplets with sum less than X
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        """
        arr_sorted = sorted(arr)
        count = 0
        
        for i in range(n - 2):
            left = i + 1
            right = n - 1
            while left < right:
                sum_val = arr_sorted[i] + arr_sorted[left] + arr_sorted[right]
                if sum_val < X:
                    count += (right - left)
                    left += 1
                else:
                    right -= 1
        
        return count

    def Count_Triplets_Brute_Force(self, arr, n, X):
        """
        Check all possible triplets using three nested loops
        Time Complexity: O(n^3)
        Space Complexity: O(1)
        """
        count = 0
        
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                for k in range(j + 1, n):
                    if arr[i] + arr[j] + arr[k] < X:
                        count += 1
        
        return count


def Test_Count_Triplets_Sum_Less():
    sol = Solution()
    tests = [
        ([-2, 0, 1, 3], 2),
        ([5, 1, 3, 4, 7], 12),
        ([-1, 0, 1, 2], 2),
        ([1, 2, 3, 4, 5], 10)
    ]

    for test in tests:
        arr = test[0]
        X = test[1]
        n = len(arr)
        
        print("Array:", end=" ")
        for num in arr:
            print(num, end=" ")
        print(f", X = {X}")
        
        arr1 = arr[:]
        arr2 = arr[:]
        res1 = sol.Count_Triplets_Sorting_Two_Pointer(arr1, n, X)
        res2 = sol.Count_Triplets_Brute_Force(arr2, n, X)
        
        print(f"Sorting + Two Pointer: {res1}")
        print(f"Brute Force: {res2}")
        
        print("-" * 50)


if __name__ == "__main__":
    Test_Count_Triplets_Sum_Less()
