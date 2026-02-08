"""
Problem: Nth Catalan Number
URL: https://practice.geeksforgeeks.org/problems/nth-catalan-number0817/1

Problem Statement:
Catalan numbers are a sequence of natural numbers that occurs in many interesting counting problems. The first few Catalan numbers for n = 0, 1, 2, 3, … are 1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862, …

Sample Input/Output:
Input: n = 5
Output: 42
Input: n = 10
Output: 16796
"""

class Solution:
    def Catalan_Number_Catalan_Recursive(self, n):
        """
        Recursive approach
        Time Complexity: O(4^n/sqrt(n))
        Space Complexity: O(n)
        """
        if n <= 1:
            return 1
        res = 0
        for i in range(n):
            res += self.Catalan_Number_Catalan_Recursive(i) * self.Catalan_Number_Catalan_Recursive(n-1-i)
        return res

    def Catalan_Number_Catalan_DP(self, n):
        """
        DP approach
        Time Complexity: O(n^2)
        Space Complexity: O(n)
        """
        dp = [0] * (n+1)
        dp[0] = dp[1] = 1
        for i in range(2, n+1):
            for j in range(i):
                dp[i] += dp[j] * dp[i-1-j]
        return dp[n]

    def Catalan_Number_Catalan_Binomial(self, n):
        """
        Binomial coefficient approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        res = 1
        for i in range(n):
            res = res * (2 * n - i)
            res = res // (i + 1)
        return res // (n + 1)

def Test_Catalan_Number():
    solution = Solution()
    
    print("n=5 Recursive:", solution.Catalan_Number_Catalan_Recursive(5))
    print("n=5 DP:", solution.Catalan_Number_Catalan_DP(5))
    print("n=5 Binomial:", solution.Catalan_Number_Catalan_Binomial(5))
    print("n=10 DP:", solution.Catalan_Number_Catalan_DP(10))
    print("n=10 Binomial:", solution.Catalan_Number_Catalan_Binomial(10))

if __name__ == "__main__":
    Test_Catalan_Number()
