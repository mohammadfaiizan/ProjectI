"""
Problem: Spiral Traversal of a Matrix
URL: https://practice.geeksforgeeks.org/problems/spirally-traversing-a-matrix-1587115621/1

Problem Statement:
Given a matrix of size R x C, print the elements in spiral order traversal.

Sample Input/Output:
Input: matrix = [[1, 2, 3, 4],
                 [5, 6, 7, 8],
                 [9, 10, 11, 12],
                 [13, 14, 15, 16]]
Output: [1, 2, 3, 4, 8, 12, 16, 15, 14, 13, 9, 5, 6, 7, 11, 10]

Input: matrix = [[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]]
Output: [1, 2, 3, 6, 9, 8, 7, 4, 5]
"""


class Solution:
    def Spiral_Traversal_Boundary_Optimal(self, matrix):
        """
        Boundary Shrinking - Track top, bottom, left, right boundaries
        Time Complexity: O(m * n)
        Space Complexity: O(1) excluding result
        """
        ans = []
        if not matrix:
            return ans
        top, bottom, right, left = 0, len(matrix), len(matrix[0]), 0
        while top < bottom and left < right:
            for i in range(left, right):
                ans.append(matrix[top][i])
            top += 1
            for i in range(top, bottom):
                ans.append(matrix[i][right - 1])
            right -= 1
            if top < bottom:
                for i in range(right - 1, left - 1, -1):
                    ans.append(matrix[bottom - 1][i])
                bottom -= 1
            if left < right:
                for i in range(bottom - 1, top - 1, -1):
                    ans.append(matrix[i][left])
                left += 1
        return ans

    def Spiral_Traversal_Direction_Array(self, matrix):
        """
        Direction Array - Use direction vectors and visited tracking
        Time Complexity: O(m * n)
        Space Complexity: O(m * n)
        """
        ans = []
        if not matrix:
            return ans
        R, C = len(matrix), len(matrix[0])
        seen = [[False] * C for _ in range(R)]
        dr = [0, 1, 0, -1]
        dc = [1, 0, -1, 0]
        r, c, di = 0, 0, 0
        for _ in range(R * C):
            ans.append(matrix[r][c])
            seen[r][c] = True
            cr, cc = r + dr[di], c + dc[di]
            if 0 <= cr < R and 0 <= cc < C and not seen[cr][cc]:
                r, c = cr, cc
            else:
                di = (di + 1) % 4
                r += dr[di]
                c += dc[di]
        return ans


def Test_Spiral_Traversal():
    solution = Solution()

    test_cases = [
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[1, 2, 3, 4], [5, 6, 7, 8]],
        [[1], [2], [3]]
    ]

    for matrix in test_cases:
        print("Matrix:")
        for row in matrix:
            print("\t".join(str(x) for x in row))

        r1 = solution.Spiral_Traversal_Boundary_Optimal(matrix)
        print("Boundary:", " ".join(str(x) for x in r1))

        r2 = solution.Spiral_Traversal_Direction_Array(matrix)
        print("Direction:", " ".join(str(x) for x in r2))

        print("-" * 50)


if __name__ == "__main__":
    Test_Spiral_Traversal()
