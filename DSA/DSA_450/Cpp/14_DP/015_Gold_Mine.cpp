/*
Problem: Gold Mine Problem
URL: https://practice.geeksforgeeks.org/problems/gold-mine-problem2608/1

Problem Statement:
Given a gold mine called M of (n x m) dimensions. Each field in this mine contains a positive integer which is the amount of gold in tons. Initially the miner can start from any row in the first column. From a given cell, the miner can move to the cell diagonally up towards the right, right, or diagonally down towards the right. Find out maximum amount of gold which he can collect.

Sample Input/Output:
Input: M = {{1, 3, 3}, {2, 1, 4}, {0, 6, 4}}
Output: 12
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Gold_Mine_Gold_Mine_DP(vector<vector<int>>& M, int n, int m) {
        /*
        DP approach
        Time Complexity: O(m*n)
        Space Complexity: O(m*n)
        */
        vector<vector<int>> dp(n, vector<int>(m, 0));
        for (int j = m-1; j >= 0; j--) {
            for (int i = 0; i < n; i++) {
                if (j == m-1) {
                    dp[i][j] = M[i][j];
                } else {
                    int right = dp[i][j+1];
                    int right_up = (i > 0) ? dp[i-1][j+1] : 0;
                    int right_down = (i < n-1) ? dp[i+1][j+1] : 0;
                    dp[i][j] = M[i][j] + max({right, right_up, right_down});
                }
            }
        }
        int result = 0;
        for (int i = 0; i < n; i++) {
            result = max(result, dp[i][0]);
        }
        return result;
    }

    int Gold_Mine_Gold_Mine_Recursive_Memo(vector<vector<int>>& M, int n, int m) {
        /*
        Recursive Memoization approach
        Time Complexity: O(m*n)
        Space Complexity: O(m*n)
        */
        vector<vector<int>> memo(n, vector<int>(m, -1));
        int result = 0;
        for (int i = 0; i < n; i++) {
            result = max(result, Gold_Mine_Memo_Helper(M, i, 0, n, m, memo));
        }
        return result;
    }

    int Gold_Mine_Memo_Helper(vector<vector<int>>& M, int i, int j, int n, int m, vector<vector<int>>& memo) {
        if (j == m) return 0;
        if (memo[i][j] != -1) return memo[i][j];
        int right = Gold_Mine_Memo_Helper(M, i, j+1, n, m, memo);
        int right_up = (i > 0) ? Gold_Mine_Memo_Helper(M, i-1, j+1, n, m, memo) : 0;
        int right_down = (i < n-1) ? Gold_Mine_Memo_Helper(M, i+1, j+1, n, m, memo) : 0;
        memo[i][j] = M[i][j] + max({right, right_up, right_down});
        return memo[i][j];
    }
};

void Test_Gold_Mine() {
    Solution solution;
    vector<vector<int>> M = {{1, 3, 3}, {2, 1, 4}, {0, 6, 4}};
    int n = M.size();
    int m = M[0].size();
    
    cout << "DP: " << solution.Gold_Mine_Gold_Mine_DP(M, n, m) << endl;
    cout << "Recursive Memo: " << solution.Gold_Mine_Gold_Mine_Recursive_Memo(M, n, m) << endl;
}

int main() {
    Test_Gold_Mine();
    return 0;
}
