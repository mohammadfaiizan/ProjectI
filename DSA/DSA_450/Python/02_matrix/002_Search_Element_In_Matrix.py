"""
Problem: Search an Element in a Matrix
URL: https://leetcode.com/problems/search-a-2d-matrix/

Problem Statement:
Given a row-wise and column-wise sorted matrix where the first integer of each row
is greater than the last integer of the previous row, search for a target value.

Sample Input/Output:
Input: matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3
Output: true

Input: matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 13
Output: false
"""


class Solution:
    def Search_Matrix_Binary_Search_Optimal(self, matrix, target):
        """
        Binary Search on Flattened Matrix - Treat matrix as sorted 1D array
        Time Complexity: O(log(m * n))
        Space Complexity: O(1)
        """
        if not matrix or not matrix[0]:
            return False
        m, n = len(matrix), len(matrix[0])
        start, end = 0, m * n - 1
        while start <= end:
            mid = start + (end - start) // 2
            val = matrix[mid // n][mid % n]
            if val == target:
                return True
            elif val < target:
                start = mid + 1
            else:
                end = mid - 1
        return False

    def Search_Matrix_Staircase(self, matrix, target):
        """
        Staircase Search - Start from bottom-left or top-right corner
        Time Complexity: O(m + n)
        Space Complexity: O(1)
        """
        m, n = len(matrix), len(matrix[0])
        i, j = m - 1, 0
        while i >= 0 and j < n:
            if matrix[i][j] == target:
                return True
            elif matrix[i][j] > target:
                i -= 1
            else:
                j += 1
        return False

    def Search_Matrix_Row_Then_Col(self, matrix, target):
        """
        Row Binary Search + Col Binary Search - Find row then search column
        Time Complexity: O(log m + log n)
        Space Complexity: O(1)
        """
        m, n = len(matrix), len(matrix[0])
        lo, hi = 0, m - 1
        while lo <= hi:
            mid = lo + (hi - lo) // 2
            if matrix[mid][0] <= target and (mid == m - 1 or matrix[mid + 1][0] > target):
                clo, chi = 0, n - 1
                while clo <= chi:
                    cmid = clo + (chi - clo) // 2
                    if matrix[mid][cmid] == target:
                        return True
                    elif matrix[mid][cmid] < target:
                        clo = cmid + 1
                    else:
                        chi = cmid - 1
                return False
            elif matrix[mid][0] > target:
                hi = mid - 1
            else:
                lo = mid + 1
        return False


def Test_Search_Element_In_Matrix():
    solution = Solution()

    class TestCase:
        def __init__(self, matrix, target, expected):
            self.matrix = matrix
            self.target = target
            self.expected = expected

    test_cases = [
        TestCase([[1, 3, 5, 7], [10, 11, 16, 20], [23, 30, 34, 60]], 3, True),
        TestCase([[1, 3, 5, 7], [10, 11, 16, 20], [23, 30, 34, 60]], 13, False),
        TestCase([[1]], 1, True),
        TestCase([[1, 3], [5, 7]], 5, True)
    ]

    for tc in test_cases:
        print(f"Target={tc.target}, Expected: {tc.expected}")

        print("Binary Search:", solution.Search_Matrix_Binary_Search_Optimal(tc.matrix, tc.target))
        print("Staircase:", solution.Search_Matrix_Staircase(tc.matrix, tc.target))
        print("Row+Col:", solution.Search_Matrix_Row_Then_Col(tc.matrix, tc.target))

        print("-" * 50)


if __name__ == "__main__":
    Test_Search_Element_In_Matrix()
