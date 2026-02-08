"""
Problem: Boolean Parenthesization
URL: https://practice.geeksforgeeks.org/problems/boolean-parenthesization5610/1

Problem Statement:
Given a boolean expression S of length N with following symbols: Symbols 'T' represents true and 'F' represents false and following operators: &, |, ^ (AND, OR, XOR). Count the number of ways we can parenthesize the expression so that the value of expression evaluates to true.

Sample Input/Output:
Input: "T|T&F^T"
Output: 4
"""


class Solution:
    def Bool_Paren_Memo(self, s: str) -> int:
        """
        Memoization
        Time Complexity: O(n^3)
        Space Complexity: O(n^2)
        """
        n = len(s)
        memo = [[[-1] * 2 for _ in range(n)] for _ in range(n)]
        return self._solve(s, 0, n - 1, True, memo)
    
    def Bool_Paren_Tab(self, s: str) -> int:
        """
        Tabulation
        Time Complexity: O(n^3)
        Space Complexity: O(n^2)
        """
        n = len(s)
        true_dp = [[0] * n for _ in range(n)]
        false_dp = [[0] * n for _ in range(n)]
        
        for i in range(0, n, 2):
            if s[i] == 'T':
                true_dp[i][i] = 1
                false_dp[i][i] = 0
            else:
                true_dp[i][i] = 0
                false_dp[i][i] = 1
        
        for length in range(3, n + 1, 2):
            for i in range(0, n - length + 1, 2):
                j = i + length - 1
                
                for k in range(i + 1, j, 2):
                    left_true = true_dp[i][k - 1]
                    left_false = false_dp[i][k - 1]
                    right_true = true_dp[k + 1][j]
                    right_false = false_dp[k + 1][j]
                    
                    if s[k] == '&':
                        true_dp[i][j] += left_true * right_true
                        false_dp[i][j] += left_true * right_false + left_false * right_true + left_false * right_false
                    elif s[k] == '|':
                        true_dp[i][j] += left_true * right_true + left_true * right_false + left_false * right_true
                        false_dp[i][j] += left_false * right_false
                    elif s[k] == '^':
                        true_dp[i][j] += left_true * right_false + left_false * right_true
                        false_dp[i][j] += left_true * right_true + left_false * right_false
        
        return true_dp[0][n - 1]
    
    def _solve(self, s: str, i: int, j: int, is_true: bool, memo: list[list[list[int]]]) -> int:
        if i > j:
            return 0
        
        if i == j:
            if is_true:
                return 1 if s[i] == 'T' else 0
            else:
                return 1 if s[i] == 'F' else 0
        
        idx = 1 if is_true else 0
        if memo[i][j][idx] != -1:
            return memo[i][j][idx]
        
        ways = 0
        
        for k in range(i + 1, j, 2):
            left_true = self._solve(s, i, k - 1, True, memo)
            left_false = self._solve(s, i, k - 1, False, memo)
            right_true = self._solve(s, k + 1, j, True, memo)
            right_false = self._solve(s, k + 1, j, False, memo)
            
            if s[k] == '&':
                if is_true:
                    ways += left_true * right_true
                else:
                    ways += left_true * right_false + left_false * right_true + left_false * right_false
            elif s[k] == '|':
                if is_true:
                    ways += left_true * right_true + left_true * right_false + left_false * right_true
                else:
                    ways += left_false * right_false
            elif s[k] == '^':
                if is_true:
                    ways += left_true * right_false + left_false * right_true
                else:
                    ways += left_true * right_true + left_false * right_false
        
        memo[i][j][idx] = ways
        return ways


def Test_BooleanParenthesization():
    solution = Solution()
    assert solution.Bool_Paren_Memo("T|T&F^T") == 4
    assert solution.Bool_Paren_Tab("T|T&F^T") == 4


if __name__ == "__main__":
    Test_BooleanParenthesization()
