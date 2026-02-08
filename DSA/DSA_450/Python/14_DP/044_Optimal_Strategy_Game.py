"""
Problem: Optimal Strategy for a Game
URL: https://practice.geeksforgeeks.org/problems/optimal-strategy-for-a-game-1587115620/1

Problem Statement:
You are given an array A of size N. The array contains integers. You need to find the maximum value you can get by picking coins optimally. Two players play a game where they can pick coins from either end of the array. You play first.

Sample Input/Output:
Input: [8,15,3,7]
Output: 22
"""


class Solution:
    def Optimal_Game_DP(self, arr: list[int]) -> int:
        """
        Dynamic Programming
        Time Complexity: O(n^2)
        Space Complexity: O(n^2)
        """
        n = len(arr)
        dp = [[0] * n for _ in range(n)]
        
        for i in range(n):
            dp[i][i] = arr[i]
        
        for i in range(n - 1):
            dp[i][i + 1] = max(arr[i], arr[i + 1])
        
        for length in range(3, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                
                pick_left = arr[i] + min(dp[i + 2][j] if i + 2 <= j else 0, 
                                         dp[i + 1][j - 1] if i + 1 <= j - 1 else 0)
                pick_right = arr[j] + min(dp[i + 1][j - 1] if i + 1 <= j - 1 else 0,
                                         dp[i][j - 2] if i <= j - 2 else 0)
                
                dp[i][j] = max(pick_left, pick_right)
        
        return dp[0][n - 1]


def Test_OptimalStrategyGame():
    solution = Solution()
    arr = [8, 15, 3, 7]
    assert solution.Optimal_Game_DP(arr) == 22


if __name__ == "__main__":
    Test_OptimalStrategyGame()
