"""
Problem: Interleaved String
URL: https://practice.geeksforgeeks.org/problems/interleaved-strings/1

Problem Statement:
Given strings A, B, and C, find whether C is formed by an interleaving of A and B. An interleaving of two strings S and T is a configuration such that it creates a new string Y from the concatenation of the two strings, maintaining the right order of characters.

Sample Input/Output:
Input: A="YX", B="X", C="XXY"
Output: false
Input: A="XY", B="X", C="XXY"
Output: true
"""


class Solution:
    def Interleave_DP(self, A: str, B: str, C: str) -> bool:
        """
        Dynamic Programming
        Time Complexity: O(m*n)
        Space Complexity: O(m*n)
        """
        m = len(A)
        n = len(B)
        
        if m + n != len(C):
            return False
        
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = True
        
        for i in range(1, m + 1):
            if A[i - 1] == C[i - 1]:
                dp[i][0] = dp[i - 1][0]
        
        for j in range(1, n + 1):
            if B[j - 1] == C[j - 1]:
                dp[0][j] = dp[0][j - 1]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if A[i - 1] == C[i + j - 1] and dp[i - 1][j]:
                    dp[i][j] = True
                if B[j - 1] == C[i + j - 1] and dp[i][j - 1]:
                    dp[i][j] = True
        
        return dp[m][n]
    
    def Interleave_Recursive_Memo(self, A: str, B: str, C: str) -> bool:
        """
        Recursive with Memoization
        Time Complexity: O(m*n)
        Space Complexity: O(m*n)
        """
        m = len(A)
        n = len(B)
        
        if m + n != len(C):
            return False
        
        memo = [[-1] * (n + 1) for _ in range(m + 1)]
        return self._solve(A, B, C, 0, 0, 0, memo)
    
    def _solve(self, A: str, B: str, C: str, i: int, j: int, k: int, memo: list[list[int]]) -> bool:
        if k == len(C):
            return True
        
        if memo[i][j] != -1:
            return bool(memo[i][j])
        
        result = False
        
        if i < len(A) and A[i] == C[k]:
            result = result or self._solve(A, B, C, i + 1, j, k + 1, memo)
        
        if j < len(B) and B[j] == C[k]:
            result = result or self._solve(A, B, C, i, j + 1, k + 1, memo)
        
        memo[i][j] = 1 if result else 0
        return result


def Test_InterleavedString():
    solution = Solution()
    assert solution.Interleave_DP("YX", "X", "XXY") == False
    assert solution.Interleave_DP("XY", "X", "XXY") == True
    assert solution.Interleave_Recursive_Memo("YX", "X", "XXY") == False
    assert solution.Interleave_Recursive_Memo("XY", "X", "XXY") == True


if __name__ == "__main__":
    Test_InterleavedString()
