"""
Problem: Optimal Binary Search Tree
URL: https://www.geeksforgeeks.org/optimal-binary-search-tree-dp-24/

Problem Statement:
Given a sorted array of keys and an array of search frequencies, construct a binary search tree that minimizes the total search cost.

Sample Input/Output:
Input: keys = [10,12,20], freq = [34,8,50]
Output: 142
"""


class Solution:
    def Optimal_BST_Recursive(self, keys: list[int], freq: list[int], i: int, j: int, level: int) -> int:
        """
        Recursive approach
        Time Complexity: O(n^3)
        Space Complexity: O(n)
        """
        if i > j:
            return 0
        if i == j:
            return freq[i] * level
        
        min_cost = float('inf')
        for r in range(i, j + 1):
            cost = (self.Optimal_BST_Recursive(keys, freq, i, r - 1, level + 1) +
                   self.Optimal_BST_Recursive(keys, freq, r + 1, j, level + 1) +
                   freq[r] * level)
            min_cost = min(min_cost, cost)
        
        return min_cost
    
    def Optimal_BST_DP(self, keys: list[int], freq: list[int]) -> int:
        """
        DP approach
        Time Complexity: O(n^3)
        Space Complexity: O(n^2)
        """
        n = len(keys)
        dp = [[0] * n for _ in range(n)]
        prefix_sum = [0] * (n + 1)
        
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + freq[i]
        
        for i in range(n):
            dp[i][i] = freq[i]
        
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                dp[i][j] = float('inf')
                
                sum_val = prefix_sum[j + 1] - prefix_sum[i]
                
                for r in range(i, j + 1):
                    cost = sum_val
                    if r > i:
                        cost += dp[i][r - 1]
                    if r < j:
                        cost += dp[r + 1][j]
                    dp[i][j] = min(dp[i][j], cost)
        
        return dp[0][n - 1]


def Test_OptimalBST():
    solution = Solution()
    
    keys = [10, 12, 20]
    freq = [34, 8, 50]
    
    n = len(keys)
    result1 = solution.Optimal_BST_Recursive(keys, freq, 0, n - 1, 1)
    assert result1 == 142
    
    result2 = solution.Optimal_BST_DP(keys, freq)
    assert result2 == 142


if __name__ == "__main__":
    Test_OptimalBST()
