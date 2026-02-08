/*
Problem: Binomial Coefficient
URL: https://practice.geeksforgeeks.org/problems/ncr1019/1

Problem Statement:
Given two integers n and r, find nCr. Since the answer may be very large, calculate the answer modulo 10^9+7.

Sample Input/Output:
Input: n = 5, r = 2
Output: 10
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Binomial_Coefficient_Binomial_Recursive(int n, int r) {
        /*
        Recursive approach
        Time Complexity: O(2^n)
        Space Complexity: O(n)
        */
        if (r == 0 || r == n) return 1;
        return Binomial_Coefficient_Binomial_Recursive(n-1, r-1) + 
               Binomial_Coefficient_Binomial_Recursive(n-1, r);
    }

    int Binomial_Coefficient_Binomial_DP(int n, int r) {
        /*
        DP approach
        Time Complexity: O(n*r)
        Space Complexity: O(n*r)
        */
        if (r > n) return 0;
        vector<vector<int>> dp(n+1, vector<int>(r+1, 0));
        for (int i = 0; i <= n; i++) {
            for (int j = 0; j <= min(i, r); j++) {
                if (j == 0 || j == i) {
                    dp[i][j] = 1;
                } else {
                    dp[i][j] = dp[i-1][j-1] + dp[i-1][j];
                }
            }
        }
        return dp[n][r];
    }

    int Binomial_Coefficient_Binomial_Optimized(int n, int r) {
        /*
        Space optimized approach
        Time Complexity: O(n*r)
        Space Complexity: O(r)
        */
        if (r > n) return 0;
        if (r > n - r) r = n - r;
        vector<int> dp(r+1, 0);
        dp[0] = 1;
        for (int i = 1; i <= n; i++) {
            for (int j = min(i, r); j > 0; j--) {
                dp[j] = dp[j] + dp[j-1];
            }
        }
        return dp[r];
    }
};

void Test_Binomial_Coefficient() {
    Solution solution;
    int n = 5, r = 2;
    
    cout << "Recursive: " << solution.Binomial_Coefficient_Binomial_Recursive(n, r) << endl;
    cout << "DP: " << solution.Binomial_Coefficient_Binomial_DP(n, r) << endl;
    cout << "Optimized: " << solution.Binomial_Coefficient_Binomial_Optimized(n, r) << endl;
}

int main() {
    Test_Binomial_Coefficient();
    return 0;
}
