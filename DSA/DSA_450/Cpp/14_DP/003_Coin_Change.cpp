/*
Problem: Coin Change
URL: https://practice.geeksforgeeks.org/problems/coin-change2448/1

Problem Statement:
Given a value N, find the number of ways to make change for N cents, if we have infinite supply of each of S = {S1, S2, .. , SM} valued coins.

Sample Input/Output:
Input: coins = [1, 2, 3], amount = 4
Output: 4
Explanation: {1,1,1,1}, {1,1,2}, {2,2}, {1,3}
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    long long Coin_Change_DP_Tabulation(vector<int>& coins, int n, int amount) {
        /*
        DP Tabulation approach
        Time Complexity: O(n*amount)
        Space Complexity: O(amount)
        */
        vector<long long> dp(amount+1, 0);
        dp[0] = 1;
        for (int i = 0; i < n; i++) {
            for (int j = coins[i]; j <= amount; j++) {
                dp[j] += dp[j-coins[i]];
            }
        }
        return dp[amount];
    }

    long long Coin_Change_Recursive_Memo(vector<int>& coins, int n, int amount) {
        /*
        Recursive Memoization approach
        Time Complexity: O(n*amount)
        Space Complexity: O(n*amount)
        */
        vector<vector<long long>> memo(n+1, vector<long long>(amount+1, -1));
        return Coin_Change_Memo_Helper(coins, n, amount, memo);
    }

    long long Coin_Change_Memo_Helper(vector<int>& coins, int n, int amount, vector<vector<long long>>& memo) {
        if (amount == 0) return 1;
        if (n == 0) return 0;
        if (memo[n][amount] != -1) return memo[n][amount];
        if (coins[n-1] > amount) {
            memo[n][amount] = Coin_Change_Memo_Helper(coins, n-1, amount, memo);
        } else {
            memo[n][amount] = Coin_Change_Memo_Helper(coins, n, amount-coins[n-1], memo) +
                              Coin_Change_Memo_Helper(coins, n-1, amount, memo);
        }
        return memo[n][amount];
    }
};

void Test_Coin_Change() {
    Solution solution;
    vector<int> coins = {1, 2, 3};
    int amount = 4;
    
    cout << "DP Tabulation: " << solution.Coin_Change_DP_Tabulation(coins, coins.size(), amount) << endl;
    cout << "Recursive Memo: " << solution.Coin_Change_Recursive_Memo(coins, coins.size(), amount) << endl;
}

int main() {
    Test_Coin_Change();
    return 0;
}
