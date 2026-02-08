"""
Problem: Maximum Sum with No Two Adjacent Elements
URL: https://practice.geeksforgeeks.org/problems/stickler-theif-1587115621/1

Problem Statement:
Find the maximum sum such that no two elements are adjacent.

Sample Input/Output:
Input: [5, 5, 10, 100, 10, 5]
Output: 110
"""

class Solution:
    def Max_Sum_DP(self, arr, n):
        """
        DP approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if n == 0:
            return 0
        if n == 1:
            return arr[0]
        dp = [0] * n
        dp[0] = arr[0]
        dp[1] = max(arr[0], arr[1])
        for i in range(2, n):
            dp[i] = max(dp[i-1], dp[i-2] + arr[i])
        return dp[n-1]

    def Max_Sum_Space(self, arr, n):
        """
        Space optimized approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if n == 0:
            return 0
        if n == 1:
            return arr[0]
        prev2 = arr[0]
        prev1 = max(arr[0], arr[1])
        for i in range(2, n):
            curr = max(prev1, prev2 + arr[i])
            prev2 = prev1
            prev1 = curr
        return prev1

def Test_Max_Sum():
    solution = Solution()
    arr = [5, 5, 10, 100, 10, 5]
    print("DP:", solution.Max_Sum_DP(arr, len(arr)))
    print("Space Optimized:", solution.Max_Sum_Space(arr, len(arr)))

if __name__ == "__main__":
    Test_Max_Sum()
