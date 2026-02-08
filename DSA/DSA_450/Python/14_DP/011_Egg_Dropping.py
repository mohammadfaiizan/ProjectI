"""
Problem: Egg Dropping Puzzle
URL: https://practice.geeksforgeeks.org/problems/egg-dropping-puzzle-1587115620/1

Problem Statement:
You are given N identical eggs and you have access to a K-floored building from 1 to K. There exists a floor f where 0 <= f <= K such that any egg dropped at a floor higher than f will break, and any egg dropped at or below floor f will not break. There are few rules given below. An egg that survives a fall can be used again. A broken egg must be discarded. The effect of a fall is the same for all eggs. If the egg doesn't break at a certain floor, it will not break at any floor below. If the eggs breaks at a certain floor, it will break at any floor above. Find the minimum number of moves that you need to determine with certainty what the value of f is.

Sample Input/Output:
Input: eggs = 2, floors = 10
Output: 4
"""

import sys

class Solution:
    def Egg_Dropping_Egg_Drop_Recursive(self, n, k):
        """
        Recursive approach
        Time Complexity: O(n*k^2)
        Space Complexity: O(n*k)
        """
        if k == 0 or k == 1:
            return k
        if n == 1:
            return k
        min_attempts = sys.maxsize
        for x in range(1, k+1):
            res = max(self.Egg_Dropping_Egg_Drop_Recursive(n-1, x-1),
                     self.Egg_Dropping_Egg_Drop_Recursive(n, k-x))
            min_attempts = min(min_attempts, res)
        return min_attempts + 1

    def Egg_Dropping_Egg_Drop_Memo(self, n, k):
        """
        Memoization approach
        Time Complexity: O(n*k^2)
        Space Complexity: O(n*k)
        """
        memo = [[-1] * (k+1) for _ in range(n+1)]
        return self.Egg_Drop_Memo_Helper(n, k, memo)

    def Egg_Drop_Memo_Helper(self, n, k, memo):
        if k == 0 or k == 1:
            return k
        if n == 1:
            return k
        if memo[n][k] != -1:
            return memo[n][k]
        min_attempts = sys.maxsize
        for x in range(1, k+1):
            res = max(self.Egg_Drop_Memo_Helper(n-1, x-1, memo),
                     self.Egg_Drop_Memo_Helper(n, k-x, memo))
            min_attempts = min(min_attempts, res)
        memo[n][k] = min_attempts + 1
        return memo[n][k]

    def Egg_Dropping_Egg_Drop_DP(self, n, k):
        """
        DP approach
        Time Complexity: O(n*k^2)
        Space Complexity: O(n*k)
        """
        dp = [[0] * (k+1) for _ in range(n+1)]
        for i in range(1, n+1):
            dp[i][0] = 0
            dp[i][1] = 1
        for j in range(1, k+1):
            dp[1][j] = j
        for i in range(2, n+1):
            for j in range(2, k+1):
                dp[i][j] = sys.maxsize
                for x in range(1, j+1):
                    res = 1 + max(dp[i-1][x-1], dp[i][j-x])
                    dp[i][j] = min(dp[i][j], res)
        return dp[n][k]

    def Egg_Dropping_Egg_Drop_Binary_Search(self, n, k):
        """
        Binary search optimization
        Time Complexity: O(n*k*log k)
        Space Complexity: O(n*k)
        """
        dp = [[0] * (k+1) for _ in range(n+1)]
        for i in range(1, n+1):
            dp[i][0] = 0
            dp[i][1] = 1
        for j in range(1, k+1):
            dp[1][j] = j
        for i in range(2, n+1):
            for j in range(2, k+1):
                dp[i][j] = sys.maxsize
                left, right = 1, j
                while left <= right:
                    mid = left + (right - left) // 2
                    broken = dp[i-1][mid-1]
                    not_broken = dp[i][j-mid]
                    res = 1 + max(broken, not_broken)
                    if broken < not_broken:
                        left = mid + 1
                    else:
                        right = mid - 1
                    dp[i][j] = min(dp[i][j], res)
        return dp[n][k]

def Test_Egg_Dropping():
    solution = Solution()
    eggs, floors = 2, 10
    
    print("Recursive:", solution.Egg_Dropping_Egg_Drop_Recursive(eggs, floors))
    print("Memoization:", solution.Egg_Dropping_Egg_Drop_Memo(eggs, floors))
    print("DP:", solution.Egg_Dropping_Egg_Drop_DP(eggs, floors))
    print("Binary Search:", solution.Egg_Dropping_Egg_Drop_Binary_Search(eggs, floors))

if __name__ == "__main__":
    Test_Egg_Dropping()
