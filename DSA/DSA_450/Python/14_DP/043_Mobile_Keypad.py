"""
Problem: Mobile Numeric Keypad
URL: https://practice.geeksforgeeks.org/problems/mobile-numeric-keypad5

Problem Statement:
Given the mobile numeric keypad. You can only press buttons that are up, left, right, or down to the current button. You are not allowed to press bottom row corner buttons (i.e. * and #). Given a number N, find out the number of possible numbers of given length.

Sample Input/Output:
Input: n=1
Output: 10
Input: n=2
Output: 36
"""


class Solution:
    def Keypad_DP(self, n: int) -> int:
        """
        Dynamic Programming
        Time Complexity: O(n*10)
        Space Complexity: O(n*10)
        """
        if n == 0:
            return 0
        if n == 1:
            return 10
        
        moves = [
            [0, 8],
            [1, 2, 4],
            [2, 1, 3, 5],
            [3, 2, 6],
            [4, 1, 5, 7],
            [5, 2, 4, 6, 8],
            [6, 3, 5, 9],
            [7, 4, 8],
            [8, 0, 5, 7, 9],
            [9, 6, 8]
        ]
        
        dp = [[0] * 10 for _ in range(n + 1)]
        
        for i in range(10):
            dp[1][i] = 1
        
        for length in range(2, n + 1):
            for digit in range(10):
                for next_digit in moves[digit]:
                    dp[length][digit] += dp[length - 1][next_digit]
        
        result = 0
        for i in range(10):
            result += dp[n][i]
        
        return result
    
    def Keypad_Space(self, n: int) -> int:
        """
        Space Optimized
        Time Complexity: O(n*10)
        Space Complexity: O(10)
        """
        if n == 0:
            return 0
        if n == 1:
            return 10
        
        moves = [
            [0, 8],
            [1, 2, 4],
            [2, 1, 3, 5],
            [3, 2, 6],
            [4, 1, 5, 7],
            [5, 2, 4, 6, 8],
            [6, 3, 5, 9],
            [7, 4, 8],
            [8, 0, 5, 7, 9],
            [9, 6, 8]
        ]
        
        prev = [1] * 10
        curr = [0] * 10
        
        for length in range(2, n + 1):
            curr = [0] * 10
            for digit in range(10):
                for next_digit in moves[digit]:
                    curr[digit] += prev[next_digit]
            prev = curr
        
        result = 0
        for i in range(10):
            result += prev[i]
        
        return result


def Test_MobileKeypad():
    solution = Solution()
    assert solution.Keypad_DP(1) == 10
    assert solution.Keypad_DP(2) == 36
    assert solution.Keypad_Space(1) == 10
    assert solution.Keypad_Space(2) == 36


if __name__ == "__main__":
    Test_MobileKeypad()
