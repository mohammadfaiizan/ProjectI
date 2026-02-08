/*
Problem: Matrix Chain Multiplication
URL: https://practice.geeksforgeeks.org/problems/matrix-chain-multiplication0303/1

Problem Statement:
Given an array p[] which represents the chain of matrices such that the ith matrix Ai is of dimension p[i-1] x p[i]. Find the minimum number of multiplications needed to multiply the chain.

Sample Input/Output:
Input: p = [40,20,30,10,30]
Output: 26000
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int MCM_Recursive(vector<int>& p, int i, int j) {
        /*
        Recursive approach
        Time Complexity: O(2^n)
        Space Complexity: O(n)
        */
        if (i >= j) return 0;
        
        int minCost = INT_MAX;
        for (int k = i; k < j; k++) {
            int cost = MCM_Recursive(p, i, k) + 
                      MCM_Recursive(p, k + 1, j) + 
                      p[i - 1] * p[k] * p[j];
            minCost = min(minCost, cost);
        }
        
        return minCost;
    }
    
    int MCM_Memo(vector<int>& p, int i, int j, vector<vector<int>>& dp) {
        /*
        Memoization approach
        Time Complexity: O(n^3)
        Space Complexity: O(n^2)
        */
        if (i >= j) return 0;
        if (dp[i][j] != -1) return dp[i][j];
        
        int minCost = INT_MAX;
        for (int k = i; k < j; k++) {
            int cost = MCM_Memo(p, i, k, dp) + 
                      MCM_Memo(p, k + 1, j, dp) + 
                      p[i - 1] * p[k] * p[j];
            minCost = min(minCost, cost);
        }
        
        dp[i][j] = minCost;
        return dp[i][j];
    }
    
    int MCM_Tab(vector<int>& p) {
        /*
        Tabulation approach
        Time Complexity: O(n^3)
        Space Complexity: O(n^2)
        */
        int n = p.size();
        vector<vector<int>> dp(n, vector<int>(n, 0));
        
        for (int length = 2; length < n; length++) {
            for (int i = 1; i < n - length + 1; i++) {
                int j = i + length - 1;
                dp[i][j] = INT_MAX;
                
                for (int k = i; k < j; k++) {
                    int cost = dp[i][k] + dp[k + 1][j] + p[i - 1] * p[k] * p[j];
                    dp[i][j] = min(dp[i][j], cost);
                }
            }
        }
        
        return dp[1][n - 1];
    }
};

void Test_MCM() {
    Solution solution;
    
    vector<int> p = {40, 20, 30, 10, 30};
    
    int result1 = solution.MCM_Recursive(p, 1, p.size() - 1);
    cout << "Recursive: " << result1 << endl;
    
    vector<vector<int>> dp(p.size(), vector<int>(p.size(), -1));
    int result2 = solution.MCM_Memo(p, 1, p.size() - 1, dp);
    cout << "Memo: " << result2 << endl;
    
    cout << "Tab: " << solution.MCM_Tab(p) << endl;
}

int main() {
    Test_MCM();
    return 0;
}
