"""
Problem: Count Ways to Reach Score
URL: https://practice.geeksforgeeks.org/problems/number-of-ways/1

Problem Statement:
Given a score n, find the number of ways to reach the score using 3, 5, and 10 points.

Sample Input/Output:
Input: n=20
Output: 4
"""


class Solution:
    def Count_Score_DP(self, n: int) -> int:
        """
        Dynamic Programming
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        dp = [0] * (n + 1)
        dp[0] = 1
        
        scores = [3, 5, 10]
        
        for score in scores:
            for i in range(score, n + 1):
                dp[i] += dp[i - score]
        
        return dp[n]


def Test_CountWaysScore():
    solution = Solution()
    assert solution.Count_Score_DP(20) == 4


if __name__ == "__main__":
    Test_CountWaysScore()
