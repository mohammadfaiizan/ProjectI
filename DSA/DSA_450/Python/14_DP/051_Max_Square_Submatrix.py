"""
Problem: Maximum Square Submatrix
URL: https://practice.geeksforgeeks.org/problems/largest-square-formed-in-a-matrix0806/1

Problem Statement:
Given a binary matrix, find the maximum size square submatrix with all 1s.

Sample Input/Output:
Input: matrix = [[0,1,1,0,1],[1,1,0,1,0],[0,1,1,1,0],[1,1,1,1,0],[1,1,1,1,1],[0,0,0,0,0]]
Output: 3
"""


class Solution:
    def Max_Square_DP(self, matrix: list[list[int]]) -> int:
        """
        DP approach
        Time Complexity: O(mn)
        Space Complexity: O(mn)
        """
        m = len(matrix)
        if m == 0:
            return 0
        n = len(matrix[0])
        if n == 0:
            return 0
        
        dp = [[0] * n for _ in range(m)]
        max_size = 0
        
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == 1:
                    if i == 0 or j == 0:
                        dp[i][j] = 1
                    else:
                        dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
                    max_size = max(max_size, dp[i][j])
        
        return max_size
    
    def Max_Square_Space(self, matrix: list[list[int]]) -> int:
        """
        Space optimized approach
        Time Complexity: O(mn)
        Space Complexity: O(n)
        """
        m = len(matrix)
        if m == 0:
            return 0
        n = len(matrix[0])
        if n == 0:
            return 0
        
        prev = [0] * n
        curr = [0] * n
        max_size = 0
        
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == 1:
                    if i == 0 or j == 0:
                        curr[j] = 1
                    else:
                        curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
                    max_size = max(max_size, curr[j])
                else:
                    curr[j] = 0
            prev = curr[:]
        
        return max_size


def Test_MaxSquareSubmatrix():
    solution = Solution()
    
    matrix = [
        [0, 1, 1, 0, 1],
        [1, 1, 0, 1, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0]
    ]
    
    assert solution.Max_Square_DP(matrix) == 3
    assert solution.Max_Square_Space(matrix) == 3


if __name__ == "__main__":
    Test_MaxSquareSubmatrix()
