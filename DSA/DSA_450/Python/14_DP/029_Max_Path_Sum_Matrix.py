"""
Problem: Maximum Path Sum in Matrix
URL: https://www.geeksforgeeks.org/maximum-path-sum-matrix/

Problem Statement:
Given a matrix of N * M. Find the maximum path sum in matrix. The maximum path is sum of all elements from first row to last row where you are allowed to move only down or diagonally down left or diagonally down right from the current cell.

Sample Input/Output:
Input: Matrix with values
Output: Maximum path sum
"""

class Solution:
    def Max_Path_DP(self, matrix, m, n):
        """
        DP approach
        Time Complexity: O(m*n)
        Space Complexity: O(m*n)
        """
        dp = [[0] * n for _ in range(m)]
        for j in range(n):
            dp[0][j] = matrix[0][j]
        for i in range(1, m):
            for j in range(n):
                max_val = dp[i-1][j]
                if j > 0:
                    max_val = max(max_val, dp[i-1][j-1])
                if j < n-1:
                    max_val = max(max_val, dp[i-1][j+1])
                dp[i][j] = matrix[i][j] + max_val
        return max(dp[m-1])

def Test_Max_Path():
    solution = Solution()
    matrix = [
        [10, 10, 2, 0, 20, 4],
        [1, 0, 0, 30, 2, 5],
        [0, 10, 4, 0, 2, 0],
        [1, 0, 2, 20, 0, 4]
    ]
    print("Max Path Sum:", solution.Max_Path_DP(matrix, len(matrix), len(matrix[0])))

if __name__ == "__main__":
    Test_Max_Path()
