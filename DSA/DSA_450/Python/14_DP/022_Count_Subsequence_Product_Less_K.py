"""
Problem: Count Subsequences with Product Less Than K
URL: https://www.geeksforgeeks.org/count-subsequences-product-less-k/

Problem Statement:
Given a non-negative array, find the number of subsequences having product smaller than K.

Sample Input/Output:
Input: [1, 2, 3, 4], k = 10
Output: 11
"""

class Solution:
    def Count_Subseq_DP(self, arr, n, k):
        """
        DP approach
        Time Complexity: O(n*k)
        Space Complexity: O(n*k)
        """
        dp = [[0] * (k+1) for _ in range(n+1)]
        for i in range(1, n+1):
            for j in range(1, k+1):
                dp[i][j] = dp[i-1][j]
                if arr[i-1] <= j and arr[i-1] > 0:
                    dp[i][j] += dp[i-1][j//arr[i-1]] + 1
        return dp[n][k]

def Test_Count_Subseq():
    solution = Solution()
    arr = [1, 2, 3, 4]
    k = 10
    print("Count:", solution.Count_Subseq_DP(arr, len(arr), k))

if __name__ == "__main__":
    Test_Count_Subseq()
