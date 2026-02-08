/*
Problem: Maximum Profit Stock
URL: https://practice.geeksforgeeks.org/problems/maximum-profit4657/1

Problem Statement:
In the stock market, a person buys a stock and sells it on some future date. Given the stock prices of N days in an array A[] and a positive integer K, find out the maximum profit a person can make in at most K transactions. A transaction is equivalent to (buying + selling) of a stock and new transaction can start only when the previous transaction has been completed.

Sample Input/Output:
Input: prices=[2,4,7,5,4,3,5], k=2
Output: 7
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Stock_K_Trans_DP(vector<int>& prices, int k) {
        /*
        Dynamic Programming
        Time Complexity: O(n*k)
        Space Complexity: O(n*k)
        */
        int n = prices.size();
        if (n <= 1 || k == 0) return 0;
        
        if (k >= n / 2) {
            int profit = 0;
            for (int i = 1; i < n; i++) {
                if (prices[i] > prices[i - 1]) {
                    profit += prices[i] - prices[i - 1];
                }
            }
            return profit;
        }
        
        vector<vector<int>> dp(k + 1, vector<int>(n, 0));
        
        for (int t = 1; t <= k; t++) {
            int maxDiff = -prices[0];
            for (int i = 1; i < n; i++) {
                dp[t][i] = max(dp[t][i - 1], prices[i] + maxDiff);
                maxDiff = max(maxDiff, dp[t - 1][i] - prices[i]);
            }
        }
        
        return dp[k][n - 1];
    }
    
    int Stock_Two_Trans(vector<int>& prices) {
        /*
        Two Transactions
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        int n = prices.size();
        if (n <= 1) return 0;
        
        vector<int> profit(n, 0);
        
        int maxPrice = prices[n - 1];
        for (int i = n - 2; i >= 0; i--) {
            maxPrice = max(maxPrice, prices[i]);
            profit[i] = max(profit[i + 1], maxPrice - prices[i]);
        }
        
        int minPrice = prices[0];
        for (int i = 1; i < n; i++) {
            minPrice = min(minPrice, prices[i]);
            profit[i] = max(profit[i - 1], profit[i] + (prices[i] - minPrice));
        }
        
        return profit[n - 1];
    }
};

void Test_Stock_K_Trans_DP() {
    Solution solution;
    vector<int> prices = {2, 4, 7, 5, 4, 3, 5};
    assert(solution.Stock_K_Trans_DP(prices, 2) == 7);
}

void Test_Stock_Two_Trans() {
    Solution solution;
    vector<int> prices = {2, 4, 7, 5, 4, 3, 5};
    assert(solution.Stock_Two_Trans(prices) >= 7);
}

int main() {
    Test_Stock_K_Trans_DP();
    Test_Stock_Two_Trans();
    return 0;
}
