/*
Problem: Optimal Strategy for a Game
URL: https://practice.geeksforgeeks.org/problems/optimal-strategy-for-a-game-1587115620/1

Problem Statement:
You are given an array A of size N. The array contains integers. You need to find the maximum value you can get by picking coins optimally. Two players play a game where they can pick coins from either end of the array. You play first.

Sample Input/Output:
Input: [8,15,3,7]
Output: 22
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Optimal_Game_DP(vector<int>& arr) {
        /*
        Dynamic Programming
        Time Complexity: O(n^2)
        Space Complexity: O(n^2)
        */
        int n = arr.size();
        vector<vector<int>> dp(n, vector<int>(n, 0));
        
        for (int i = 0; i < n; i++) {
            dp[i][i] = arr[i];
        }
        
        for (int i = 0; i < n - 1; i++) {
            dp[i][i + 1] = max(arr[i], arr[i + 1]);
        }
        
        for (int len = 3; len <= n; len++) {
            for (int i = 0; i <= n - len; i++) {
                int j = i + len - 1;
                
                int pickLeft = arr[i] + min(dp[i + 2][j], dp[i + 1][j - 1]);
                int pickRight = arr[j] + min(dp[i + 1][j - 1], dp[i][j - 2]);
                
                dp[i][j] = max(pickLeft, pickRight);
            }
        }
        
        return dp[0][n - 1];
    }
};

void Test_Optimal_Game_DP() {
    Solution solution;
    vector<int> arr = {8, 15, 3, 7};
    assert(solution.Optimal_Game_DP(arr) == 22);
}

int main() {
    Test_Optimal_Game_DP();
    return 0;
}
