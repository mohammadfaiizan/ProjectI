"""
Problem: First and Last Occurrences of X
URL: https://practice.geeksforgeeks.org/problems/first-and-last-occurrences-of-x3116/1

Problem Statement:
Given a sorted array arr containing n elements, possibly with duplicates, find the first and last occurrences of an element x in the given array.

Sample Input/Output:
Input: n = 9, x = 5, arr[] = {1, 3, 5, 5, 5, 5, 67, 123, 125}
Output: 2 5

Input: n = 9, x = 7, arr[] = {1, 3, 5, 5, 5, 5, 7, 123, 125}
Output: 6 6
"""


class Solution:
    def First_Last_Linear(self, arr, n, x):
        """
        Linear search - find first and last occurrence by scanning array
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        first = -1
        last = -1
        for i in range(n):
            if arr[i] == x:
                if first == -1:
                    first = i
                last = i
        return [first, last]

    def First_Last_Binary_Two_Passes(self, arr, n, x):
        """
        Binary search with two passes - find first occurrence, then last occurrence
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        first = -1
        last = -1
        
        left = 0
        right = n - 1
        while left <= right:
            mid = left + (right - left) // 2
            if arr[mid] == x:
                first = mid
                right = mid - 1
            elif arr[mid] < x:
                left = mid + 1
            else:
                right = mid - 1
        
        left = 0
        right = n - 1
        while left <= right:
            mid = left + (right - left) // 2
            if arr[mid] == x:
                last = mid
                left = mid + 1
            elif arr[mid] < x:
                left = mid + 1
            else:
                right = mid - 1
        
        return [first, last]

    def First_Last_Binary_Boundary(self, arr, n, x):
        """
        Binary search checking boundary conditions
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        first = -1
        last = -1
        
        left = 0
        right = n - 1
        while left <= right:
            mid = left + (right - left) // 2
            if arr[mid] == x:
                if mid == 0 or arr[mid - 1] != x:
                    first = mid
                    break
                right = mid - 1
            elif arr[mid] < x:
                left = mid + 1
            else:
                right = mid - 1
        
        left = 0
        right = n - 1
        while left <= right:
            mid = left + (right - left) // 2
            if arr[mid] == x:
                if mid == n - 1 or arr[mid + 1] != x:
                    last = mid
                    break
                left = mid + 1
            elif arr[mid] < x:
                left = mid + 1
            else:
                right = mid - 1
        
        return [first, last]


def Test_First_Last_Position():
    sol = Solution()
    tests = [
        ([1, 3, 5, 5, 5, 5, 67, 123, 125], 5),
        ([1, 3, 5, 5, 5, 5, 7, 123, 125], 7),
        ([1, 2, 3, 4, 5], 3),
        ([1, 2, 3, 4, 5], 6),
        ([5, 5, 5, 5, 5], 5)
    ]

    for arr, x in tests:
        n = len(arr)
        
        print("Array:", end=" ")
        for num in arr:
            print(num, end=" ")
        print(f", x = {x}")
        
        res1 = sol.First_Last_Linear(arr, n, x)
        print(f"Linear: First = {res1[0]}, Last = {res1[1]}")
        
        res2 = sol.First_Last_Binary_Two_Passes(arr, n, x)
        print(f"Binary Two Passes: First = {res2[0]}, Last = {res2[1]}")
        
        res3 = sol.First_Last_Binary_Boundary(arr, n, x)
        print(f"Binary Boundary: First = {res3[0]}, Last = {res3[1]}")
        
        print("-" * 50)


if __name__ == "__main__":
    Test_First_Last_Position()
