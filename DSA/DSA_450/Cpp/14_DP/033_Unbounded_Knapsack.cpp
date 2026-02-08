/*
Problem: Unbounded Knapsack
URL: https://practice.geeksforgeeks.org/problems/knapsack-with-duplicate-items4201/1

Problem Statement:
Given a knapsack weight W and a set of items with certain value and weight, we need to calculate the maximum amount that could make up this quantity exactly. This is different from classical knapsack problem, here we are allowed to use unlimited number of instances of an item.

Sample Input/Output:
Input: val=[1,30], wt=[1,50], W=100
Output: 100
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Unbounded_KS_Tab(vector<int>& val, vector<int>& wt, int W) {
        /*
        Tabulation
        Time Complexity: O(n*W)
        Space Complexity: O(n*W)
        */
        int n = val.size();
        vector<vector<int>> dp(n + 1, vector<int>(W + 1, 0));
        
        for (int i = 1; i <= n; i++) {
            for (int w = 1; w <= W; w++) {
                if (wt[i - 1] <= w) {
                    dp[i][w] = max(dp[i - 1][w], val[i - 1] + dp[i][w - wt[i - 1]]);
                } else {
                    dp[i][w] = dp[i - 1][w];
                }
            }
        }
        
        return dp[n][W];
    }
    
    int Unbounded_KS_Space(vector<int>& val, vector<int>& wt, int W) {
        /*
        Space Optimized
        Time Complexity: O(n*W)
        Space Complexity: O(W)
        */
        int n = val.size();
        vector<int> dp(W + 1, 0);
        
        for (int i = 0; i < n; i++) {
            for (int w = wt[i]; w <= W; w++) {
                dp[w] = max(dp[w], val[i] + dp[w - wt[i]]);
            }
        }
        
        return dp[W];
    }
};

void Test_Unbounded_KS_Tab() {
    Solution solution;
    vector<int> val = {1, 30};
    vector<int> wt = {1, 50};
    assert(solution.Unbounded_KS_Tab(val, wt, 100) == 100);
}

void Test_Unbounded_KS_Space() {
    Solution solution;
    vector<int> val = {1, 30};
    vector<int> wt = {1, 50};
    assert(solution.Unbounded_KS_Space(val, wt, 100) == 100);
}

int main() {
    Test_Unbounded_KS_Tab();
    Test_Unbounded_KS_Space();
    return 0;
}
