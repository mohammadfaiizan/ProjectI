"""
Problem: Minimum Jumps to Reach End
URL: https://practice.geeksforgeeks.org/problems/minimum-number-of-jumps-1587115620/1

Problem Statement:
Given an array of integers where each element represents the max number of steps that can be made forward from that element. Find the minimum number of jumps to reach the end of the array (starting from the first element). If an element is 0, then you cannot move through that element.

Sample Input/Output:
Input: [2, 3, 1, 1, 4]
Output: 2
"""

import sys

class Solution:
    def Min_Jumps_Greedy(self, arr, n):
        """
        Greedy approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if n <= 1:
            return 0
        if arr[0] == 0:
            return -1
        maxReach = arr[0]
        steps = arr[0]
        jumps = 1
        for i in range(1, n):
            if i == n-1:
                return jumps
            maxReach = max(maxReach, i + arr[i])
            steps -= 1
            if steps == 0:
                jumps += 1
                if i >= maxReach:
                    return -1
                steps = maxReach - i
        return -1

    def Min_Jumps_DP(self, arr, n):
        """
        DP approach
        Time Complexity: O(n^2)
        Space Complexity: O(n)
        """
        if n <= 1:
            return 0
        if arr[0] == 0:
            return -1
        dp = [sys.maxsize] * n
        dp[0] = 0
        for i in range(n):
            if dp[i] == sys.maxsize:
                continue
            for j in range(i+1, min(i+arr[i]+1, n)):
                dp[j] = min(dp[j], dp[i] + 1)
        return -1 if dp[n-1] == sys.maxsize else dp[n-1]

def Test_Min_Jumps():
    solution = Solution()
    arr = [2, 3, 1, 1, 4]
    print("Greedy:", solution.Min_Jumps_Greedy(arr, len(arr)))
    print("DP:", solution.Min_Jumps_DP(arr, len(arr)))

if __name__ == "__main__":
    Test_Min_Jumps()
