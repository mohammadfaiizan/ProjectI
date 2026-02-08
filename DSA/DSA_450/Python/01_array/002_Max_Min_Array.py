"""
Problem: Maximum and Minimum in Array
URL: https://www.geeksforgeeks.org/maximum-and-minimum-in-an-array/

Problem Statement:
Given an array of size N, find the maximum and minimum elements in the array.

Sample Input/Output:
Input: arr = [1000, 11, 445, 1, 330, 3000]
Output: Min = 1, Max = 3000

Input: arr = [3, 5, 4, 1, 9]
Output: Min = 1, Max = 9
"""


class Solution:
    def Max_Min_Linear_Scan_Optimal(self, arr):
        """
        Linear Scan - Single pass tracking min and max
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        mn = mx = arr[0]
        for i in range(1, len(arr)):
            if arr[i] > mx:
                mx = arr[i]
            if arr[i] < mn:
                mn = arr[i]
        return (mn, mx)

    def Max_Min_Pairs_Comparison(self, arr):
        """
        Pairs Comparison - Compare elements in pairs to reduce comparisons
        Time Complexity: O(n) - ~1.5n comparisons
        Space Complexity: O(1)
        """
        n = len(arr)
        if n % 2 == 0:
            mn = min(arr[0], arr[1])
            mx = max(arr[0], arr[1])
            i = 2
        else:
            mn = mx = arr[0]
            i = 1
        while i < n - 1:
            if arr[i] < arr[i + 1]:
                mn = min(mn, arr[i])
                mx = max(mx, arr[i + 1])
            else:
                mn = min(mn, arr[i + 1])
                mx = max(mx, arr[i])
            i += 2
        return (mn, mx)

    def Max_Min_STL(self, arr):
        """
        STL Approach - Using minmax_element
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        return (min(arr), max(arr))


def Test_Max_Min_Array():
    solution = Solution()

    test_cases = [
        [1000, 11, 445, 1, 330, 3000],
        [3, 5, 4, 1, 9],
        [7],
        [2, 8]
    ]

    for arr in test_cases:
        print("Array:", arr)

        mn1, mx1 = solution.Max_Min_Linear_Scan_Optimal(arr)
        print(f"Linear Scan: Min={mn1} Max={mx1}")

        mn2, mx2 = solution.Max_Min_Pairs_Comparison(arr)
        print(f"Pairs: Min={mn2} Max={mx2}")

        mn3, mx3 = solution.Max_Min_STL(arr)
        print(f"STL: Min={mn3} Max={mx3}")

        print("-" * 50)


if __name__ == "__main__":
    Test_Max_Min_Array()
