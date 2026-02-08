"""
Problem: Permutation Coefficient
URL: https://www.geeksforgeeks.org/permutation-coefficient/

Problem Statement:
Permutation refers to the process of arranging all the members of a given set to form a sequence. The number of permutations on a set of n elements is given by n! (n factorial). The Permutation Coefficient represented by P(n, r) is used to represent the number of ways to obtain an ordered subset having r elements from a set of n elements.

Sample Input/Output:
Input: n = 10, r = 2
Output: 90
"""

class Solution:
    def Permutation_Coefficient_Permutation_Recursive(self, n, r):
        """
        Recursive approach
        Time Complexity: O(n-r)
        Space Complexity: O(n-r)
        """
        if r == 0:
            return 1
        if r > n:
            return 0
        return n * self.Permutation_Coefficient_Permutation_Recursive(n-1, r-1)

    def Permutation_Coefficient_Permutation_DP(self, n, r):
        """
        DP approach
        Time Complexity: O(n*r)
        Space Complexity: O(n*r)
        """
        if r > n:
            return 0
        dp = [[0] * (r+1) for _ in range(n+1)]
        for i in range(n+1):
            for j in range(min(i, r) + 1):
                if j == 0:
                    dp[i][j] = 1
                else:
                    dp[i][j] = dp[i-1][j] + j * dp[i-1][j-1]
        return dp[n][r]

    def Permutation_Coefficient_Permutation_Optimized(self, n, r):
        """
        Space optimized approach
        Time Complexity: O(n*r)
        Space Complexity: O(r)
        """
        if r > n:
            return 0
        dp = [0] * (r+1)
        dp[0] = 1
        for i in range(1, n+1):
            for j in range(min(i, r), 0, -1):
                dp[j] = dp[j] + j * dp[j-1]
        return dp[r]

def Test_Permutation_Coefficient():
    solution = Solution()
    n, r = 10, 2
    
    print("Recursive:", solution.Permutation_Coefficient_Permutation_Recursive(n, r))
    print("DP:", solution.Permutation_Coefficient_Permutation_DP(n, r))
    print("Optimized:", solution.Permutation_Coefficient_Permutation_Optimized(n, r))

if __name__ == "__main__":
    Test_Permutation_Coefficient()
