"""
Problem: Median in a Row-wise Sorted Matrix
URL: https://practice.geeksforgeeks.org/problems/median-in-a-row-wise-sorted-matrix1527/1

Problem Statement:
Given a row-wise sorted matrix of size R x C where R and C are always odd,
find the median of the matrix.

Sample Input/Output:
Input: matrix = [[1, 3, 5],
                 [2, 6, 9],
                 [3, 6, 9]]
Output: 5
Explanation: Sorted array is [1, 2, 3, 3, 5, 6, 6, 9, 9]. Median is 5.

Input: matrix = [[1], [2], [3]]
Output: 2
"""

import bisect


class Solution:
    def Median_Binary_Search_Optimal(self, matrix):
        """
        Binary Search on Value Range - Count elements less than mid
        Time Complexity: O(R * log(C) * log(max - min))
        Space Complexity: O(1)
        """
        r, c = len(matrix), len(matrix[0])
        mn = min(matrix[i][0] for i in range(r))
        mx = max(matrix[i][c - 1] for i in range(r))
        desired = (r * c + 1) // 2
        while mn < mx:
            mid = mn + (mx - mn) // 2
            count = 0
            for i in range(r):
                count += bisect.bisect_right(matrix[i], mid)
            if count < desired:
                mn = mid + 1
            else:
                mx = mid
        return mn

    def Median_Flatten_And_Sort(self, matrix):
        """
        Flatten and Sort - Merge all elements and find median
        Time Complexity: O(R * C * log(R * C))
        Space Complexity: O(R * C)
        """
        all_elements = []
        for row in matrix:
            all_elements.extend(row)
        all_elements.sort()
        return all_elements[len(all_elements) // 2]


def Test_Median_Row_Wise_Sorted():
    solution = Solution()

    class TestCase:
        def __init__(self, matrix, expected):
            self.matrix = matrix
            self.expected = expected

    test_cases = [
        TestCase([[1, 3, 5], [2, 6, 9], [3, 6, 9]], 5),
        TestCase([[1], [2], [3]], 2),
        TestCase([[1, 3, 4], [2, 5, 6], [7, 8, 9]], 5)
    ]

    for tc in test_cases:
        print("Matrix:")
        for row in tc.matrix:
            print(" ".join(str(x) for x in row))
        print(f"Expected: {tc.expected}")

        print("Binary Search:", solution.Median_Binary_Search_Optimal(tc.matrix))
        print("Flatten & Sort:", solution.Median_Flatten_And_Sort(tc.matrix))

        print("-" * 50)


if __name__ == "__main__":
    Test_Median_Row_Wise_Sorted()
