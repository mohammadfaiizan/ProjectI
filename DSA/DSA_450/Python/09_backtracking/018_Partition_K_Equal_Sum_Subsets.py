"""
Problem: Partition K Equal Sum Subsets
URL: https://practice.geeksforgeeks.org/problems/partition-array-to-k-subsets/1

Problem Statement:
Check if an array can be partitioned into K subsets with equal sum.

Sample Input/Output:
Input: arr = [4, 3, 2, 3, 5, 2, 1], K = 4
Output: true
Explanation: Can be partitioned into 4 subsets: [5], [1,4], [2,3], [2,3]
"""


class Solution:
    def Partition_K_Equal_Sum_Backtracking(self, arr, k):
        """
        Backtracking with bucket sums
        Time Complexity: O(K^N)
        Space Complexity: O(N)
        """
        total_sum = sum(arr)
        if total_sum % k != 0:
            return False
        
        target = total_sum // k
        subset_sums = [0] * k
        arr.sort(reverse=True)
        
        def backtrack(idx):
            if idx == len(arr):
                for s in subset_sums:
                    if s != target:
                        return False
                return True
            
            for i in range(k):
                if subset_sums[i] + arr[idx] <= target:
                    subset_sums[i] += arr[idx]
                    if backtrack(idx + 1):
                        return True
                    subset_sums[i] -= arr[idx]
                
                if subset_sums[i] == 0:
                    break
            
            return False
        
        return backtrack(0)
    
    def Partition_K_Equal_Sum_Bitmask_DP(self, arr, k):
        """
        Bitmask DP
        Time Complexity: O(N * 2^N)
        Space Complexity: O(2^N)
        """
        n = len(arr)
        total_sum = sum(arr)
        if total_sum % k != 0:
            return False
        
        target = total_sum // k
        dp = [False] * (1 << n)
        sum_arr = [0] * (1 << n)
        dp[0] = True
        
        for mask in range(1 << n):
            if not dp[mask]:
                continue
            
            for i in range(n):
                if mask & (1 << i):
                    continue
                
                new_mask = mask | (1 << i)
                new_sum = sum_arr[mask] + arr[i]
                
                if new_sum <= target:
                    sum_arr[new_mask] = new_sum % target
                    dp[new_mask] = True
        
        return dp[(1 << n) - 1]


def Test_Partition_K_Equal_Sum_Subsets():
    solution = Solution()
    arr = [4, 3, 2, 3, 5, 2, 1]
    k = 4
    result1 = solution.Partition_K_Equal_Sum_Backtracking(arr, k)
    result2 = solution.Partition_K_Equal_Sum_Bitmask_DP(arr, k)
    print("Backtracking Approach:", result1)
    print("Bitmask DP Approach:", result2)


if __name__ == "__main__":
    Test_Partition_K_Equal_Sum_Subsets()
