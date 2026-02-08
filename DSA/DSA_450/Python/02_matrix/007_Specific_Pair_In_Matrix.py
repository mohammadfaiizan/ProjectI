"""
Problem: Find a Specific Pair in Matrix
URL: https://www.geeksforgeeks.org/find-a-specific-pair-in-matrix/

Problem Statement:
Given an N x N matrix, find the maximum value of mat[c][d] - mat[a][b] over all choices
of indices such that both c > a and d > b.

Sample Input/Output:
Input: mat = [[1,  2,  -1, -4, -20],
              [-8, -3,  4,  2,   1],
              [3,   8,  6,  1,   3],
              [-4, -1,  1,  7,  -6],
              [0,  -4, 10, -5,   1]]
Output: 18
Explanation: mat[4][2] - mat[0][0] = 10 - (-8) is not valid. Max is mat[2][1] - mat[0][2] etc.
             Actually max is mat[4][2] - mat[1][0] = 10 - (-8) = 18.

Input: mat = [[1, 2], [3, 4]]
Output: 2
Explanation: mat[1][1] - mat[0][0] = 4 - 1 = 3 but we need c>a and d>b. Max = 4-2=2 or 3-1=2 or 4-1=3.
"""

import sys


class Solution:
    def Specific_Pair_Preprocess_Optimal(self, mat):
        """
        Bottom-Right Max Preprocessing - Build max matrix from bottom-right
        Time Complexity: O(n^2)
        Space Complexity: O(n^2)
        """
        n = len(mat)
        maxArr = [[0] * n for _ in range(n)]
        maxArr[n - 1][n - 1] = mat[n - 1][n - 1]
        maxv = mat[n - 1][n - 1]
        for j in range(n - 2, -1, -1):
            maxv = max(maxv, mat[n - 1][j])
            maxArr[n - 1][j] = maxv
        maxv = mat[n - 1][n - 1]
        for i in range(n - 2, -1, -1):
            maxv = max(maxv, mat[i][n - 1])
            maxArr[i][n - 1] = maxv
        result = -sys.maxsize
        for i in range(n - 2, -1, -1):
            for j in range(n - 2, -1, -1):
                result = max(result, maxArr[i + 1][j + 1] - mat[i][j])
                maxArr[i][j] = max(mat[i][j], max(maxArr[i + 1][j], maxArr[i][j + 1]))
        return result

    def Specific_Pair_Brute_Force(self, mat):
        """
        Brute Force - Check all valid pairs (a,b) and (c,d)
        Time Complexity: O(n^4)
        Space Complexity: O(1)
        """
        n = len(mat)
        result = -sys.maxsize
        for a in range(n):
            for b in range(n):
                for c in range(a + 1, n):
                    for d in range(b + 1, n):
                        result = max(result, mat[c][d] - mat[a][b])
        return result


def Test_Specific_Pair():
    solution = Solution()

    class TestCase:
        def __init__(self, mat, expected):
            self.mat = mat
            self.expected = expected

    test_cases = [
        TestCase([[1, 2, -1, -4, -20], [-8, -3, 4, 2, 1], [3, 8, 6, 1, 3], [-4, -1, 1, 7, -6], [0, -4, 10, -5, 1]], 18),
        TestCase([[1, 2], [3, 4]], 3)
    ]

    for tc in test_cases:
        print("Matrix:")
        for row in tc.mat:
            print("\t".join(str(x) for x in row))
        print(f"Expected: {tc.expected}")

        print("Preprocess:", solution.Specific_Pair_Preprocess_Optimal(tc.mat))
        print("Brute Force:", solution.Specific_Pair_Brute_Force(tc.mat))

        print("-" * 50)


if __name__ == "__main__":
    Test_Specific_Pair()
