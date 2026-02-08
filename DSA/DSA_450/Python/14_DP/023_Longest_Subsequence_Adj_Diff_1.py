"""
Problem: Longest Subsequence with Adjacent Difference 1
URL: https://www.geeksforgeeks.org/longest-subsequence-such-that-difference-between-adjacents-is-one/

Problem Statement:
Given an array of n integers, find the length of the longest subsequence such that adjacent elements of the subsequence have a difference of 1.

Sample Input/Output:
Input: [1, 2, 3, 4, 5, 3, 2]
Output: 6
"""

class Solution:
    def Longest_Subseq_DP(self, arr, n):
        """
        DP approach
        Time Complexity: O(n^2)
        Space Complexity: O(n)
        """
        dp = [1] * n
        for i in range(1, n):
            for j in range(i):
                if abs(arr[i] - arr[j]) == 1:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)

def Test_Longest_Subseq():
    solution = Solution()
    arr = [1, 2, 3, 4, 5, 3, 2]
    print("Longest Length:", solution.Longest_Subseq_DP(arr, len(arr)))

if __name__ == "__main__":
    Test_Longest_Subseq()
