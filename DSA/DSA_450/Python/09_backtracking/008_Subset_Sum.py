"""
Problem: Subset Sum
URL: https://practice.geeksforgeeks.org/problems/subset-sum-problem2014/1

Problem Statement:
Given array and target sum, check if array can be partitioned into two subsets with equal sum (uses subset sum backtracking).

Sample Input/Output:
Input: arr[]={1,5,11,5}
Output: true
Explanation: Partition {1,5,5} and {11} have equal sum
"""


class Solution:
    def Partition_Equal_Subset_Sum_Backtracking(self, arr):
        """
        Backtracking include/exclude
        Time Complexity: O(2^n)
        Space Complexity: O(n)
        """
        total_sum = sum(arr)
        if total_sum % 2 != 0:
            return False
        
        target = total_sum // 2
        
        def backtrack(index, current_sum):
            if current_sum == target:
                return True
            if index >= len(arr) or current_sum > target:
                return False
            
            return (backtrack(index + 1, current_sum + arr[index]) or
                    backtrack(index + 1, current_sum))
        
        return backtrack(0, 0)
    
    def Partition_Equal_Subset_Sum_DP(self, arr):
        """
        DP
        Time Complexity: O(n*sum)
        Space Complexity: O(n*sum)
        """
        total_sum = sum(arr)
        if total_sum % 2 != 0:
            return False
        
        target = total_sum // 2
        n = len(arr)
        dp = [[False] * (target + 1) for _ in range(n + 1)]
        
        for i in range(n + 1):
            dp[i][0] = True
        
        for i in range(1, n + 1):
            for j in range(1, target + 1):
                if arr[i-1] > j:
                    dp[i][j] = dp[i-1][j]
                else:
                    dp[i][j] = dp[i-1][j] or dp[i-1][j - arr[i-1]]
        
        return dp[n][target]


def Test_Subset_Sum():
    solution = Solution()
    
    arr = [1, 5, 11, 5]
    
    result1 = solution.Partition_Equal_Subset_Sum_Backtracking(arr)
    print("Backtracking result:", result1)
    
    result2 = solution.Partition_Equal_Subset_Sum_DP(arr)
    print("DP result:", result2)


if __name__ == "__main__":
    Test_Subset_Sum()
