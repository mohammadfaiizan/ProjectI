"""
Problem: Gold Mine Problem
URL: https://practice.geeksforgeeks.org/problems/gold-mine-problem2608/1

Problem Statement:
Given a gold mine called M of (n x m) dimensions. Each field in this mine contains a positive integer which is the amount of gold in tons. Initially the miner can start from any row in the first column. From a given cell, the miner can move to the cell diagonally up towards the right, right, or diagonally down towards the right. Find out maximum amount of gold which he can collect.

Sample Input/Output:
Input: M = [[1, 3, 3], [2, 1, 4], [0, 6, 4]]
Output: 12
"""

class Solution:
    def Gold_Mine_Gold_Mine_DP(self, M, n, m):
        """
        DP approach
        Time Complexity: O(m*n)
        Space Complexity: O(m*n)
        """
        dp = [[0] * m for _ in range(n)]
        for j in range(m-1, -1, -1):
            for i in range(n):
                if j == m-1:
                    dp[i][j] = M[i][j]
                else:
                    right = dp[i][j+1]
                    right_up = dp[i-1][j+1] if i > 0 else 0
                    right_down = dp[i+1][j+1] if i < n-1 else 0
                    dp[i][j] = M[i][j] + max(right, right_up, right_down)
        result = 0
        for i in range(n):
            result = max(result, dp[i][0])
        return result

    def Gold_Mine_Gold_Mine_Recursive_Memo(self, M, n, m):
        """
        Recursive Memoization approach
        Time Complexity: O(m*n)
        Space Complexity: O(m*n)
        """
        memo = [[-1] * m for _ in range(n)]
        result = 0
        for i in range(n):
            result = max(result, self.Gold_Mine_Memo_Helper(M, i, 0, n, m, memo))
        return result

    def Gold_Mine_Memo_Helper(self, M, i, j, n, m, memo):
        if j == m:
            return 0
        if memo[i][j] != -1:
            return memo[i][j]
        right = self.Gold_Mine_Memo_Helper(M, i, j+1, n, m, memo)
        right_up = self.Gold_Mine_Memo_Helper(M, i-1, j+1, n, m, memo) if i > 0 else 0
        right_down = self.Gold_Mine_Memo_Helper(M, i+1, j+1, n, m, memo) if i < n-1 else 0
        memo[i][j] = M[i][j] + max(right, right_up, right_down)
        return memo[i][j]

def Test_Gold_Mine():
    solution = Solution()
    M = [[1, 3, 3], [2, 1, 4], [0, 6, 4]]
    n = len(M)
    m = len(M[0])
    
    print("DP:", solution.Gold_Mine_Gold_Mine_DP(M, n, m))
    print("Recursive Memo:", solution.Gold_Mine_Gold_Mine_Recursive_Memo(M, n, m))

if __name__ == "__main__":
    Test_Gold_Mine()
