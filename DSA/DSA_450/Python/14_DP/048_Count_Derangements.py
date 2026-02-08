"""
Problem: Count Derangements
URL: https://www.geeksforgeeks.org/count-derangements-permutation-such-that-no-element-appears-in-its-original-position/

Problem Statement:
Count the number of derangements of n elements. A derangement is a permutation where no element appears in its original position. D(n) = (n-1) * (D(n-1) + D(n-2))

Sample Input/Output:
Input: n = 4
Output: 9
"""


class Solution:
    def Derange_DP(self, n: int) -> int:
        """
        DP approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if n == 0 or n == 1:
            return 0
        if n == 2:
            return 1
        
        dp = [0] * (n + 1)
        dp[0] = 0
        dp[1] = 0
        dp[2] = 1
        
        for i in range(3, n + 1):
            dp[i] = (i - 1) * (dp[i - 1] + dp[i - 2])
        
        return dp[n]
    
    def Derange_Space(self, n: int) -> int:
        """
        Space optimized approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if n == 0 or n == 1:
            return 0
        if n == 2:
            return 1
        
        prev2 = 0
        prev1 = 1
        
        for i in range(3, n + 1):
            current = (i - 1) * (prev1 + prev2)
            prev2 = prev1
            prev1 = current
        
        return prev1


def Test_CountDerangements():
    solution = Solution()
    n = 4
    assert solution.Derange_DP(n) == 9
    assert solution.Derange_Space(n) == 9


if __name__ == "__main__":
    Test_CountDerangements()
