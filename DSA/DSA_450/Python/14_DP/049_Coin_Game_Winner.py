"""
Problem: Coin Game Winner
URL: https://www.geeksforgeeks.org/coin-game-winner-every-player-three-choices/

Problem Statement:
Two players are playing a coin game. In each turn, a player can pick 1, x, or y coins. The player who picks the last coin wins. Determine if the first player can win given n coins and choices x, y.

Sample Input/Output:
Input: n = 5, x = 2, y = 3
Output: true (First player wins)
"""


class Solution:
    def Coin_Game_DP(self, n: int, x: int, y: int) -> bool:
        """
        DP approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        dp = [False] * (n + 1)
        
        dp[0] = False
        dp[1] = True
        
        for i in range(2, n + 1):
            if i >= 1 and not dp[i - 1]:
                dp[i] = True
            elif i >= x and not dp[i - x]:
                dp[i] = True
            elif i >= y and not dp[i - y]:
                dp[i] = True
            else:
                dp[i] = False
        
        return dp[n]


def Test_CoinGameWinner():
    solution = Solution()
    n = 5
    x = 2
    y = 3
    result = solution.Coin_Game_DP(n, x, y)
    assert result == True


if __name__ == "__main__":
    Test_CoinGameWinner()
