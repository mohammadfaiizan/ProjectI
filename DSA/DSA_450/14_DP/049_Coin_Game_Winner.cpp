/*
Problem: Coin Game Winner
URL: https://www.geeksforgeeks.org/coin-game-winner-every-player-three-choices/

Problem Statement:
Two players are playing a coin game. In each turn, a player can pick 1, x, or y coins. The player who picks the last coin wins. Determine if the first player can win given n coins and choices x, y.

Sample Input/Output:
Input: n = 5, x = 2, y = 3
Output: true (First player wins)
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    bool Coin_Game_DP(int n, int x, int y) {
        /*
        DP approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        vector<bool> dp(n + 1, false);
        
        dp[0] = false;
        dp[1] = true;
        
        for (int i = 2; i <= n; i++) {
            if (i >= 1 && !dp[i - 1]) {
                dp[i] = true;
            } else if (i >= x && !dp[i - x]) {
                dp[i] = true;
            } else if (i >= y && !dp[i - y]) {
                dp[i] = true;
            } else {
                dp[i] = false;
            }
        }
        
        return dp[n];
    }
};

void Test_Coin_Game() {
    Solution solution;
    
    int n = 5, x = 2, y = 3;
    bool result = solution.Coin_Game_DP(n, x, y);
    cout << "n=" << n << ", x=" << x << ", y=" << y << " -> " 
         << (result ? "First wins" : "Second wins") << endl;
}

int main() {
    Test_Coin_Game();
    return 0;
}
