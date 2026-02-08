"""
Problem: Row with Maximum Number of 1s
URL: https://practice.geeksforgeeks.org/problems/row-with-max-1s0023/1

Problem Statement:
Given a boolean 2D array where each row is sorted in non-decreasing order,
find the index of the row with the maximum number of 1s.

Sample Input/Output:
Input: matrix = [[0, 1, 1, 1],
                 [0, 0, 1, 1],
                 [1, 1, 1, 1],
                 [0, 0, 0, 0]]
Output: 2
Explanation: Row 2 has maximum 1s (4 ones).

Input: matrix = [[0, 0], [1, 1]]
Output: 1
"""

import bisect


class Solution:
    def Row_Max_Ones_Staircase_Optimal(self, arr):
        """
        Staircase Approach - Start from top-right, move left on 1 and down on 0
        Time Complexity: O(m + n)
        Space Complexity: O(1)
        """
        n, m = len(arr), len(arr[0])
        row, col = 0, m - 1
        max_row_index = -1
        while row < n and col >= 0:
            if arr[row][col] == 1:
                max_row_index = row
                col -= 1
            else:
                row += 1
        return max_row_index

    def Row_Max_Ones_Binary_Search(self, arr):
        """
        Binary Search - Find first 1 in each row using binary search
        Time Complexity: O(m * log n)
        Space Complexity: O(1)
        """
        n, m = len(arr), len(arr[0])
        max_ones = 0
        max_row = -1
        for i in range(n):
            first_one = self.First_One_Index(arr[i], m)
            if first_one != -1:
                ones = m - first_one
                if ones > max_ones:
                    max_ones = ones
                    max_row = i
        return max_row

    def First_One_Index(self, row, m):
        lo, hi, result = 0, m - 1, -1
        while lo <= hi:
            mid = lo + (hi - lo) // 2
            if row[mid] == 1:
                result = mid
                hi = mid - 1
            else:
                lo = mid + 1
        return result


def Test_Row_With_Maximum_Ones():
    solution = Solution()

    class TestCase:
        def __init__(self, arr, expected):
            self.arr = arr
            self.expected = expected

    test_cases = [
        TestCase([[0, 1, 1, 1], [0, 0, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]], 2),
        TestCase([[0, 0], [1, 1]], 1),
        TestCase([[0, 0, 0], [0, 0, 1], [0, 1, 1]], 2),
        TestCase([[1, 1, 1], [1, 1, 1], [0, 0, 0]], 0)
    ]

    for tc in test_cases:
        print("Matrix:")
        for row in tc.arr:
            print(" ".join(str(x) for x in row))
        print(f"Expected: {tc.expected}")

        print("Staircase:", solution.Row_Max_Ones_Staircase_Optimal(tc.arr))
        print("Binary Search:", solution.Row_Max_Ones_Binary_Search(tc.arr))

        print("-" * 50)


if __name__ == "__main__":
    Test_Row_With_Maximum_Ones()
