"""
Problem: Value Equal to Index Value
URL: https://practice.geeksforgeeks.org/problems/value-equal-to-index-value1330/1

Problem Statement:
Given an array Arr of N positive integers. Your task is to find the elements whose value is equal to that of its index value (Consider 1-based indexing).

Sample Input/Output:
Input: N = 5, Arr[] = {15, 2, 45, 12, 7}
Output: 2

Input: N = 1, Arr[] = {1}
Output: 1
"""


class Solution:
    def Find_Fixed_Point_Linear(self, arr, n):
        """
        Linear search - check each element if arr[i] == i+1
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        result = []
        for i in range(n):
            if arr[i] == i + 1:
                result.append(i + 1)
        return result

    def Find_Fixed_Point_Binary_Search(self, arr, n):
        """
        Binary search for single fixed point in sorted array
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        left = 0
        right = n - 1
        while left <= right:
            mid = left + (right - left) // 2
            if arr[mid] == mid + 1:
                return mid + 1
            elif arr[mid] < mid + 1:
                left = mid + 1
            else:
                right = mid - 1
        return -1


def Test_Find_Fixed_Point():
    sol = Solution()
    tests = [
        [15, 2, 45, 12, 7],
        [1],
        [1, 2, 3, 4, 5],
        [10, 20, 30, 40, 50],
        [-10, -5, 0, 3, 7]
    ]

    for arr in tests:
        n = len(arr)
        print("Array:", end=" ")
        for num in arr:
            print(num, end=" ")
        print()
        
        res1 = sol.Find_Fixed_Point_Linear(arr, n)
        print("Linear:", end=" ")
        if not res1:
            print("No fixed point found")
        else:
            for val in res1:
                print(val, end=" ")
            print()
        
        res2 = sol.Find_Fixed_Point_Binary_Search(arr, n)
        print("Binary Search:", end=" ")
        if res2 == -1:
            print("No fixed point found")
        else:
            print(res2)
        
        print("-" * 50)


if __name__ == "__main__":
    Test_Find_Fixed_Point()
