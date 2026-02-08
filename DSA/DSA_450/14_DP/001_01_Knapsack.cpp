/*
Problem: 0/1 Knapsack
URL: https://practice.geeksforgeeks.org/problems/0-1-knapsack-problem0945/1

Problem Statement:
Given weights and values of n items, put these items in a knapsack of capacity W to get the maximum total value in the knapsack. In other words, given two integer arrays val[0..n-1] and wt[0..n-1] which represent values and weights associated with n items respectively. Also given an integer W which represents knapsack capacity, find out the maximum value subset of val[] such that sum of the weights of this subset is smaller than or equal to W. You cannot break an item, either pick the complete item or don't pick it (0-1 property).

Sample Input/Output:
Input: val = [60, 100, 120], wt = [10, 20, 30], W = 50
Output: 220
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Knapsack_Recursive(vector<int>& val, vector<int>& wt, int n, int W) {
        /*
        Recursive approach
        Time Complexity: O(2^n)
        Space Complexity: O(n)
        */
        if (n == 0 || W == 0) return 0;
        if (wt[n-1] > W) return Knapsack_Recursive(val, wt, n-1, W);
        return max(val[n-1] + Knapsack_Recursive(val, wt, n-1, W-wt[n-1]),
                   Knapsack_Recursive(val, wt, n-1, W));
    }

    int Knapsack_Memoization(vector<int>& val, vector<int>& wt, int n, int W) {
        /*
        Memoization approach
        Time Complexity: O(n*W)
        Space Complexity: O(n*W)
        */
        vector<vector<int>> dp(n+1, vector<int>(W+1, -1));
        return Knapsack_Memo_Helper(val, wt, n, W, dp);
    }

    int Knapsack_Memo_Helper(vector<int>& val, vector<int>& wt, int n, int W, vector<vector<int>>& dp) {
        if (n == 0 || W == 0) return 0;
        if (dp[n][W] != -1) return dp[n][W];
        if (wt[n-1] > W) {
            dp[n][W] = Knapsack_Memo_Helper(val, wt, n-1, W, dp);
        } else {
            dp[n][W] = max(val[n-1] + Knapsack_Memo_Helper(val, wt, n-1, W-wt[n-1], dp),
                          Knapsack_Memo_Helper(val, wt, n-1, W, dp));
        }
        return dp[n][W];
    }

    int Knapsack_Tabulation(vector<int>& val, vector<int>& wt, int n, int W) {
        /*
        Tabulation approach
        Time Complexity: O(n*W)
        Space Complexity: O(n*W)
        */
        vector<vector<int>> dp(n+1, vector<int>(W+1, 0));
        for (int i = 1; i <= n; i++) {
            for (int w = 1; w <= W; w++) {
                if (wt[i-1] <= w) {
                    dp[i][w] = max(val[i-1] + dp[i-1][w-wt[i-1]], dp[i-1][w]);
                } else {
                    dp[i][w] = dp[i-1][w];
                }
            }
        }
        return dp[n][W];
    }

    int Knapsack_Space_Optimized(vector<int>& val, vector<int>& wt, int n, int W) {
        /*
        Space optimized approach
        Time Complexity: O(n*W)
        Space Complexity: O(W)
        */
        vector<int> dp(W+1, 0);
        for (int i = 0; i < n; i++) {
            for (int w = W; w >= wt[i]; w--) {
                dp[w] = max(dp[w], val[i] + dp[w-wt[i]]);
            }
        }
        return dp[W];
    }
};

void Test_01_Knapsack() {
    Solution solution;
    vector<int> val = {60, 100, 120};
    vector<int> wt = {10, 20, 30};
    int W = 50;
    int n = val.size();
    
    cout << "Recursive: " << solution.Knapsack_Recursive(val, wt, n, W) << endl;
    cout << "Memoization: " << solution.Knapsack_Memoization(val, wt, n, W) << endl;
    cout << "Tabulation: " << solution.Knapsack_Tabulation(val, wt, n, W) << endl;
    cout << "Space Optimized: " << solution.Knapsack_Space_Optimized(val, wt, n, W) << endl;
}

int main() {
    Test_01_Knapsack();
    return 0;
}
