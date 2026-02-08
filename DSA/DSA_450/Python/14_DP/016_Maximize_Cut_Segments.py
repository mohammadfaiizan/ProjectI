"""
Problem: Maximize Cut Segments
URL: https://practice.geeksforgeeks.org/problems/cutted-segments1642/1

Problem Statement:
Given an integer N denoting the Length of a line segment. You need to cut the line segment in such a way that the cut length of a line segment each time is either x, y or z. Here x, y, and z are integers. After performing all the cut operations, your total number of cut segments must be maximum.

Sample Input/Output:
Input: N = 4, x = 2, y = 1, z = 1
Output: 4
"""

import sys

class Solution:
    def Maximize_Cuts_Memo(self, n, x, y, z):
        """
        Memoization approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        dp = [-2] * (n+1)
        return self.Maximize_Cuts_Memo_Helper(n, x, y, z, dp)

    def Maximize_Cuts_Memo_Helper(self, n, x, y, z, dp):
        if n == 0:
            return 0
        if n < 0:
            return -sys.maxsize
        if dp[n] != -2:
            return dp[n]
        cut_x = self.Maximize_Cuts_Memo_Helper(n-x, x, y, z, dp)
        cut_y = self.Maximize_Cuts_Memo_Helper(n-y, x, y, z, dp)
        cut_z = self.Maximize_Cuts_Memo_Helper(n-z, x, y, z, dp)
        result = max(cut_x, cut_y, cut_z)
        dp[n] = (-sys.maxsize if result == -sys.maxsize else result + 1)
        return dp[n]

    def Maximize_Cuts_Tab(self, n, x, y, z):
        """
        Tabulation approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        dp = [-sys.maxsize] * (n+1)
        dp[0] = 0
        for i in range(1, n+1):
            if i >= x and dp[i-x] != -sys.maxsize:
                dp[i] = max(dp[i], dp[i-x] + 1)
            if i >= y and dp[i-y] != -sys.maxsize:
                dp[i] = max(dp[i], dp[i-y] + 1)
            if i >= z and dp[i-z] != -sys.maxsize:
                dp[i] = max(dp[i], dp[i-z] + 1)
        return 0 if dp[n] == -sys.maxsize else dp[n]

def Test_Maximize_Cuts():
    solution = Solution()
    n, x, y, z = 4, 2, 1, 1
    print("Memoization:", solution.Maximize_Cuts_Memo(n, x, y, z))
    print("Tabulation:", solution.Maximize_Cuts_Tab(n, x, y, z))

if __name__ == "__main__":
    Test_Maximize_Cuts()
