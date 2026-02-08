"""
Problem: Rotate Matrix by 90 Degrees Clockwise
URL: https://www.geeksforgeeks.org/rotate-a-matrix-by-90-degree-in-clockwise-direction-without-using-any-extra-space/

Problem Statement:
Given an N x N square matrix, rotate it by 90 degrees in clockwise direction
without using any extra space.

Sample Input/Output:
Input: matrix = [[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]]
Output: [[7, 4, 1],
         [8, 5, 2],
         [9, 6, 3]]

Input: matrix = [[1, 2],
                 [3, 4]]
Output: [[3, 1],
         [4, 2]]
"""


class Solution:
    def Rotate_90_Transpose_Reverse_Optimal(self, mat):
        """
        Transpose + Reverse Rows - Transpose matrix then reverse each row
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        """
        n = len(mat)
        for i in range(n):
            for j in range(i + 1, n):
                mat[i][j], mat[j][i] = mat[j][i], mat[i][j]
        for i in range(n):
            mat[i].reverse()

    def Rotate_90_Cycle_Swap(self, mat):
        """
        Cycle Swap - Swap elements of each cycle in clockwise direction
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        """
        n = len(mat)
        for i in range(n // 2):
            for j in range(i, n - i - 1):
                temp = mat[i][j]
                mat[i][j] = mat[n - 1 - j][i]
                mat[n - 1 - j][i] = mat[n - 1 - i][n - 1 - j]
                mat[n - 1 - i][n - 1 - j] = mat[j][n - 1 - i]
                mat[j][n - 1 - i] = temp


def Test_Rotate_Matrix():
    solution = Solution()

    test_cases = [
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[1, 2], [3, 4]],
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    ]

    for mat in test_cases:
        print("Original:")
        for row in mat:
            print("\t".join(str(x) for x in row))

        import copy
        mat1 = copy.deepcopy(mat)
        solution.Rotate_90_Transpose_Reverse_Optimal(mat1)
        print("Transpose+Reverse:")
        for row in mat1:
            print("\t".join(str(x) for x in row))

        mat2 = copy.deepcopy(mat)
        solution.Rotate_90_Cycle_Swap(mat2)
        print("Cycle Swap:")
        for row in mat2:
            print("\t".join(str(x) for x in row))

        print("-" * 50)


if __name__ == "__main__":
    Test_Rotate_Matrix()
