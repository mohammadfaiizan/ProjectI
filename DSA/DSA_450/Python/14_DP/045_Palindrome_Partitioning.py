"""
Problem: Palindrome Partitioning
URL: https://practice.geeksforgeeks.org/problems/palindromic-patitioning4845/1

Problem Statement:
Given a string str, a partitioning of the string is a palindrome partitioning if every sub-string of the partition is a palindrome. Determine the fewest cuts needed for palindrome partitioning of given string.

Sample Input/Output:
Input: "ababbbabbababa"
Output: 3
"""


class Solution:
    def Pal_Partition_DP(self, s: str) -> int:
        """
        Dynamic Programming
        Time Complexity: O(n^2)
        Space Complexity: O(n^2)
        """
        n = len(s)
        is_pal = [[False] * n for _ in range(n)]
        cuts = [0] * n
        
        for i in range(n):
            is_pal[i][i] = True
        
        for i in range(n - 1):
            if s[i] == s[i + 1]:
                is_pal[i][i + 1] = True
        
        for length in range(3, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                if s[i] == s[j] and is_pal[i + 1][j - 1]:
                    is_pal[i][j] = True
        
        for i in range(n):
            if is_pal[0][i]:
                cuts[i] = 0
            else:
                cuts[i] = float('inf')
                for j in range(i):
                    if is_pal[j + 1][i] and cuts[j] + 1 < cuts[i]:
                        cuts[i] = cuts[j] + 1
        
        return cuts[n - 1]
    
    def Pal_Partition_Memo(self, s: str) -> int:
        """
        Memoization
        Time Complexity: O(n^3)
        Space Complexity: O(n^2)
        """
        n = len(s)
        memo = [[-1] * n for _ in range(n)]
        return self._solve(s, 0, n - 1, memo)
    
    def _is_palindrome(self, s: str, i: int, j: int) -> bool:
        while i < j:
            if s[i] != s[j]:
                return False
            i += 1
            j -= 1
        return True
    
    def _solve(self, s: str, i: int, j: int, memo: list[list[int]]) -> int:
        if i >= j or self._is_palindrome(s, i, j):
            return 0
        
        if memo[i][j] != -1:
            return memo[i][j]
        
        min_cuts = float('inf')
        
        for k in range(i, j):
            cuts = self._solve(s, i, k, memo) + self._solve(s, k + 1, j, memo) + 1
            min_cuts = min(min_cuts, cuts)
        
        memo[i][j] = min_cuts
        return min_cuts


def Test_PalindromePartitioning():
    solution = Solution()
    assert solution.Pal_Partition_DP("ababbbabbababa") == 3
    assert solution.Pal_Partition_Memo("ababbbabbababa") == 3


if __name__ == "__main__":
    Test_PalindromePartitioning()
