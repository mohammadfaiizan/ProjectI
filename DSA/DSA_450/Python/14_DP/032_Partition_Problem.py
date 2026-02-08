"""
Problem: Partition Problem
URL: https://practice.geeksforgeeks.org/problems/subset-sum-problem2014/1

Problem Statement:
Given an array arr[] of size N, check if it can be partitioned into two parts such that the sum of elements in both parts is the same.

Sample Input/Output:
Input: [1,5,11,5]
Output: true
"""


class Solution:
    def Partition_DP(self, arr: list[int]) -> bool:
        """
        Dynamic Programming
        Time Complexity: O(n*sum)
        Space Complexity: O(n*sum)
        """
        n = len(arr)
        total_sum = sum(arr)
        
        if total_sum % 2 != 0:
            return False
        
        target = total_sum // 2
        dp = [[False] * (target + 1) for _ in range(n + 1)]
        
        for i in range(n + 1):
            dp[i][0] = True
        
        for i in range(1, n + 1):
            for j in range(1, target + 1):
                if arr[i - 1] > j:
                    dp[i][j] = dp[i - 1][j]
                else:
                    dp[i][j] = dp[i - 1][j] or dp[i - 1][j - arr[i - 1]]
        
        return dp[n][target]
    
    def Partition_Space(self, arr: list[int]) -> bool:
        """
        Space Optimized
        Time Complexity: O(n*sum)
        Space Complexity: O(sum)
        """
        n = len(arr)
        total_sum = sum(arr)
        
        if total_sum % 2 != 0:
            return False
        
        target = total_sum // 2
        dp = [False] * (target + 1)
        dp[0] = True
        
        for i in range(n):
            for j in range(target, arr[i] - 1, -1):
                dp[j] = dp[j] or dp[j - arr[i]]
        
        return dp[target]


def Test_PartitionProblem():
    solution = Solution()
    arr1 = [1, 5, 11, 5]
    assert solution.Partition_DP(arr1) == True
    assert solution.Partition_Space(arr1) == True


if __name__ == "__main__":
    Test_PartitionProblem()
