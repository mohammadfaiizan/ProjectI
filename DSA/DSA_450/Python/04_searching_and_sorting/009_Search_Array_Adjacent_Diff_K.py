"""
Problem: Searching in Array with Adjacent Differ by at Most K
URL: https://www.geeksforgeeks.org/searching-array-adjacent-differ-k/

Problem Statement:
Given an array where each element is at most k positions away from its target position, search for an element x in the array.

Sample Input/Output:
Input: arr[] = {20, 40, 50, 70, 70, 60}, k = 20, x = 60
Output: 5

Input: arr[] = {20, 40, 50, 70, 70, 60}, k = 20, x = 10
Output: -1
"""


class Solution:
    def Search_Adjacent_Diff_K_Jump_Search(self, arr, n, x, k):
        """
        Jump search based on difference - jump by max(1, abs(arr[i]-x)/k)
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        i = 0
        while i < n:
            if arr[i] == x:
                return i
            i = i + max(1, abs(arr[i] - x) // k)
        return -1

    def Search_Adjacent_Diff_K_Linear(self, arr, n, x, k):
        """
        Linear search through array
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        for i in range(n):
            if arr[i] == x:
                return i
        return -1


def Test_Search_Array_Adjacent_Diff_K():
    sol = Solution()
    tests = [
        ([20, 40, 50, 70, 70, 60], 20, 60),
        ([20, 40, 50, 70, 70, 60], 20, 10),
        ([2, 4, 5, 7, 7, 6], 2, 5),
        ([10, 20, 30, 40, 50], 10, 30)
    ]

    for test in tests:
        arr = test[0]
        k = test[1]
        x = test[2]
        n = len(arr)
        
        print("Array:", end=" ")
        for num in arr:
            print(num, end=" ")
        print(f", k = {k}, x = {x}")
        
        res1 = sol.Search_Adjacent_Diff_K_Jump_Search(arr, n, x, k)
        print(f"Jump Search: {res1}")
        
        res2 = sol.Search_Adjacent_Diff_K_Linear(arr, n, x, k)
        print(f"Linear: {res2}")
        
        print("-" * 50)


if __name__ == "__main__":
    Test_Search_Array_Adjacent_Diff_K()
