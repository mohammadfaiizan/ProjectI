"""
Problem: Count Balanced Binary Trees of Height h
URL: https://www.geeksforgeeks.org/count-balanced-binary-trees-height-h/

Problem Statement:
Count the number of balanced binary trees of height h. A balanced binary tree is one where the difference between heights of left and right subtrees is at most 1.

Sample Input/Output:
Input: h = 3
Output: 15
"""


class Solution:
    def Count_BBT_DP(self, h: int) -> int:
        """
        DP approach
        Time Complexity: O(h)
        Space Complexity: O(h)
        """
        if h == 0 or h == 1:
            return 1
        
        dp = [0] * (h + 1)
        dp[0] = 1
        dp[1] = 1
        
        for i in range(2, h + 1):
            dp[i] = dp[i - 1] * dp[i - 1] + 2 * dp[i - 1] * dp[i - 2]
        
        return dp[h]


def Test_CountBalancedBinaryTrees():
    solution = Solution()
    h = 3
    assert solution.Count_BBT_DP(h) == 15


if __name__ == "__main__":
    Test_CountBalancedBinaryTrees()
