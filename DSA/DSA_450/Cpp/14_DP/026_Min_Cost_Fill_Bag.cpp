/*
Problem: Minimum Cost to Fill Bag
URL: https://practice.geeksforgeeks.org/problems/minimum-cost-to-fill-given-weight-in-a-bag1956/1

Problem Statement:
Given an array cost[] of positive integers of size n where cost[i] represents the cost of i kg packet of oranges, the task is to find the minimum cost to buy W kgs of oranges. If it is impossible to buy exactly W kg oranges then the output will be -1.

Sample Input/Output:
Input: cost = [20, 10, 4, 50, 100], W = 5
Output: 14
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Min_Cost_DP(vector<int>& cost, int n, int W) {
        /*
        DP approach
        Time Complexity: O(n*W)
        Space Complexity: O(W)
        */
        vector<int> dp(W+1, INT_MAX);
        dp[0] = 0;
        for (int i = 1; i <= W; i++) {
            for (int j = 0; j < n && j < i; j++) {
                if (cost[j] != -1 && dp[i-j-1] != INT_MAX) {
                    dp[i] = min(dp[i], cost[j] + dp[i-j-1]);
                }
            }
        }
        return (dp[W] == INT_MAX) ? -1 : dp[W];
    }
};

void Test_Min_Cost() {
    Solution solution;
    vector<int> cost = {20, 10, 4, 50, 100};
    int W = 5;
    cout << "Min Cost: " << solution.Min_Cost_DP(cost, cost.size(), W) << endl;
}

int main() {
    Test_Min_Cost();
    return 0;
}
