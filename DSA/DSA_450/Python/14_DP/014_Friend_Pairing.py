"""
Problem: Friends Pairing Problem
URL: https://practice.geeksforgeeks.org/problems/friends-pairing-problem5425/1

Problem Statement:
Given n friends, each one can remain single or can be paired up with some other friend. Each friend can be paired only once. Find out the total number of ways in which friends can remain single or can be paired up.

Sample Input/Output:
Input: n = 4
Output: 10
Explanation: {1}, {2}, {3}, {4}, {1,2}, {3,4}, {1,3}, {2,4}, {1,4}, {2,3}, {1,2}, {3}, {4}, {1,3}, {2}, {4}, {1,4}, {2}, {3}, {2,3}, {1}, {4}, {2,4}, {1}, {3}, {3,4}, {1}, {2}
"""

class Solution:
    def Friend_Pairing_Friend_Pair_DP(self, n):
        """
        DP approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if n <= 2:
            return n
        dp = [0] * (n+1)
        dp[0] = 0
        dp[1] = 1
        dp[2] = 2
        for i in range(3, n+1):
            dp[i] = dp[i-1] + (i-1) * dp[i-2]
        return dp[n]

    def Friend_Pairing_Friend_Pair_Space(self, n):
        """
        Space optimized approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if n <= 2:
            return n
        prev2 = 1
        prev1 = 2
        for i in range(3, n+1):
            curr = prev1 + (i-1) * prev2
            prev2 = prev1
            prev1 = curr
        return prev1

def Test_Friend_Pairing():
    solution = Solution()
    n = 4
    
    print("DP:", solution.Friend_Pairing_Friend_Pair_DP(n))
    print("Space Optimized:", solution.Friend_Pairing_Friend_Pair_Space(n))

if __name__ == "__main__":
    Test_Friend_Pairing()
