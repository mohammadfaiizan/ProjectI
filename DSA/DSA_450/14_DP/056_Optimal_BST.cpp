/*
Problem: Optimal Binary Search Tree
URL: https://www.geeksforgeeks.org/optimal-binary-search-tree-dp-24/

Problem Statement:
Given a sorted array of keys and an array of search frequencies, construct a binary search tree that minimizes the total search cost.

Sample Input/Output:
Input: keys = [10,12,20], freq = [34,8,50]
Output: 142
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Optimal_BST_Recursive(vector<int>& keys, vector<int>& freq, int i, int j, int level) {
        /*
        Recursive approach
        Time Complexity: O(n^3)
        Space Complexity: O(n)
        */
        if (i > j) return 0;
        if (i == j) return freq[i] * level;
        
        int minCost = INT_MAX;
        for (int r = i; r <= j; r++) {
            int cost = Optimal_BST_Recursive(keys, freq, i, r - 1, level + 1) +
                      Optimal_BST_Recursive(keys, freq, r + 1, j, level + 1) +
                      freq[r] * level;
            minCost = min(minCost, cost);
        }
        
        return minCost;
    }
    
    int Optimal_BST_DP(vector<int>& keys, vector<int>& freq) {
        /*
        DP approach
        Time Complexity: O(n^3)
        Space Complexity: O(n^2)
        */
        int n = keys.size();
        vector<vector<int>> dp(n, vector<int>(n, 0));
        vector<int> prefixSum(n + 1, 0);
        
        for (int i = 0; i < n; i++) {
            prefixSum[i + 1] = prefixSum[i] + freq[i];
        }
        
        for (int i = 0; i < n; i++) {
            dp[i][i] = freq[i];
        }
        
        for (int length = 2; length <= n; length++) {
            for (int i = 0; i <= n - length; i++) {
                int j = i + length - 1;
                dp[i][j] = INT_MAX;
                
                int sum = prefixSum[j + 1] - prefixSum[i];
                
                for (int r = i; r <= j; r++) {
                    int cost = sum;
                    if (r > i) cost += dp[i][r - 1];
                    if (r < j) cost += dp[r + 1][j];
                    dp[i][j] = min(dp[i][j], cost);
                }
            }
        }
        
        return dp[0][n - 1];
    }
};

void Test_Optimal_BST() {
    Solution solution;
    
    vector<int> keys = {10, 12, 20};
    vector<int> freq = {34, 8, 50};
    
    int n = keys.size();
    int result = solution.Optimal_BST_Recursive(keys, freq, 0, n - 1, 1);
    cout << "Recursive: " << result << endl;
    cout << "DP: " << solution.Optimal_BST_DP(keys, freq) << endl;
}

int main() {
    Test_Optimal_BST();
    return 0;
}
