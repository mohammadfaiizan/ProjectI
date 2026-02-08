"""
Problem: Maximum Size Rectangle of All 1s
URL: https://practice.geeksforgeeks.org/problems/max-rectangle/1

Problem Statement:
Given a binary matrix M of size N x M, find the maximum area of a rectangle
formed only of 1s in the given matrix.

Sample Input/Output:
Input: M = [[0, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 0, 0]]
Output: 8
Explanation: Rectangle from (1,0) to (2,3) has area 2*4 = 8.

Input: M = [[0, 1, 1],
            [1, 1, 1],
            [0, 1, 1]]
Output: 6
"""


class Solution:
    def Max_Rectangle_Histogram_Optimal(self, M):
        """
        Histogram Based - Build histogram row by row, find largest rectangle
        Time Complexity: O(n * m)
        Space Complexity: O(m)
        """
        n, m = len(M), len(M[0])
        heights = [0] * m
        max_area = 0
        for i in range(n):
            for j in range(m):
                heights[j] = heights[j] + 1 if M[i][j] else 0
            max_area = max(max_area, self.Largest_Rectangle_Histogram(heights))
        return max_area

    def Max_Rectangle_Brute_Force(self, M):
        """
        Brute Force - Check all possible rectangles
        Time Complexity: O(n^2 * m^2)
        Space Complexity: O(1)
        """
        n, m = len(M), len(M[0])
        max_area = 0
        for i in range(n):
            col_sum = [0] * m
            for j in range(i, n):
                for k in range(m):
                    col_sum[k] += M[j][k]
                height = j - i + 1
                width = 0
                for k in range(m):
                    if col_sum[k] == height:
                        width += 1
                        max_area = max(max_area, height * width)
                    else:
                        width = 0
        return max_area

    def Largest_Rectangle_Histogram(self, heights):
        st = []
        max_area = 0
        n = len(heights)
        for i in range(n + 1):
            h = 0 if i == n else heights[i]
            while st and h < heights[st[-1]]:
                top = heights[st.pop()]
                width = i if not st else i - st[-1] - 1
                max_area = max(max_area, top * width)
            st.append(i)
        return max_area


def Test_Max_Size_Rectangle():
    solution = Solution()

    class TestCase:
        def __init__(self, M, expected):
            self.M = M
            self.expected = expected

    test_cases = [
        TestCase([[0, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 0, 0]], 8),
        TestCase([[0, 1, 1], [1, 1, 1], [0, 1, 1]], 6),
        TestCase([[1]], 1),
        TestCase([[0, 0], [0, 0]], 0)
    ]

    for tc in test_cases:
        print("Matrix:")
        for row in tc.M:
            print(" ".join(str(x) for x in row))
        print(f"Expected: {tc.expected}")

        print("Histogram:", solution.Max_Rectangle_Histogram_Optimal(tc.M))
        print("Brute Force:", solution.Max_Rectangle_Brute_Force(tc.M))

        print("-" * 50)


if __name__ == "__main__":
    Test_Max_Size_Rectangle()
