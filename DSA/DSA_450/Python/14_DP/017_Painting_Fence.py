"""
Problem: Painting Fence
URL: https://practice.geeksforgeeks.org/problems/painting-the-fence3727/1

Problem Statement:
Given a fence with n posts and k colors, find out the number of ways of painting the fence such that at most 2 adjacent posts have the same color.

Sample Input/Output:
Input: n = 3, k = 2
Output: 6
"""

class Solution:
    def Paint_Fence_DP(self, n, k):
        """
        DP approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if n == 0:
            return 0
        if n == 1:
            return k
        dp = [0] * (n+1)
        dp[1] = k
        dp[2] = k * k
        for i in range(3, n+1):
            dp[i] = (k-1) * (dp[i-1] + dp[i-2])
        return dp[n]

    def Paint_Fence_Space(self, n, k):
        """
        Space optimized approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if n == 0:
            return 0
        if n == 1:
            return k
        prev2 = k
        prev1 = k * k
        for i in range(3, n+1):
            curr = (k-1) * (prev1 + prev2)
            prev2 = prev1
            prev1 = curr
        return prev1

def Test_Painting_Fence():
    solution = Solution()
    n, k = 3, 2
    print("DP:", solution.Paint_Fence_DP(n, k))
    print("Space Optimized:", solution.Paint_Fence_Space(n, k))

if __name__ == "__main__":
    Test_Painting_Fence()
