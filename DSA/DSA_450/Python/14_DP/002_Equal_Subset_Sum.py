"""
Problem: Equal Subset Sum Partition
URL: https://practice.geeksforgeeks.org/problems/subset-sum-problem2014/1

Problem Statement:
Given an array arr[] of size N, check if it can be partitioned into two parts such that the sum of elements in both parts is the same.

Sample Input/Output:
Input: [1, 5, 11, 5]
Output: true
Input: [1, 2, 3, 5]
Output: false
"""

class Solution:
    def Equal_Subset_Sum_DP_Tabulation(self, arr, n):
        """
        DP Tabulation approach
        Time Complexity: O(n*sum)
        Space Complexity: O(n*sum)
        """
        total_sum = sum(arr)
        if total_sum % 2 != 0:
            return False
        target = total_sum // 2
        dp = [[False] * (target+1) for _ in range(n+1)]
        for i in range(n+1):
            dp[i][0] = True
        for i in range(1, n+1):
            for j in range(1, target+1):
                if arr[i-1] > j:
                    dp[i][j] = dp[i-1][j]
                else:
                    dp[i][j] = dp[i-1][j] or dp[i-1][j-arr[i-1]]
        return dp[n][target]

    def Equal_Subset_Sum_Space_Optimized(self, arr, n):
        """
        Space optimized approach
        Time Complexity: O(n*sum)
        Space Complexity: O(sum)
        """
        total_sum = sum(arr)
        if total_sum % 2 != 0:
            return False
        target = total_sum // 2
        dp = [False] * (target+1)
        dp[0] = True
        for i in range(n):
            for j in range(target, arr[i]-1, -1):
                dp[j] = dp[j] or dp[j-arr[i]]
        return dp[target]

def Test_Equal_Subset_Sum():
    solution = Solution()
    arr1 = [1, 5, 11, 5]
    arr2 = [1, 2, 3, 5]
    
    print("Test 1 [1,5,11,5]:", solution.Equal_Subset_Sum_DP_Tabulation(arr1, len(arr1)))
    print("Test 2 [1,2,3,5]:", solution.Equal_Subset_Sum_DP_Tabulation(arr2, len(arr2)))
    print("Test 1 Space Optimized:", solution.Equal_Subset_Sum_Space_Optimized(arr1, len(arr1)))
    print("Test 2 Space Optimized:", solution.Equal_Subset_Sum_Space_Optimized(arr2, len(arr2)))

if __name__ == "__main__":
    Test_Equal_Subset_Sum()
