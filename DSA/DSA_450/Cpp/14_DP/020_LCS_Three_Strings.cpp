/*
Problem: LCS of Three Strings
URL: https://practice.geeksforgeeks.org/problems/lcs-of-three-strings0028/1

Problem Statement:
Given 3 strings A, B and C, the task is to find the length of the longest sub-sequence that is common in all the three given strings.

Sample Input/Output:
Input: A = "geeks", B = "geeksfor", C = "geeksforgeeks"
Output: 5
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int LCS_Three_Tab(string& A, string& B, string& C) {
        /*
        Tabulation approach
        Time Complexity: O(l*m*n)
        Space Complexity: O(l*m*n)
        */
        int l = A.length(), m = B.length(), n = C.length();
        vector<vector<vector<int>>> dp(l+1, vector<vector<int>>(m+1, vector<int>(n+1, 0)));
        for (int i = 1; i <= l; i++) {
            for (int j = 1; j <= m; j++) {
                for (int k = 1; k <= n; k++) {
                    if (A[i-1] == B[j-1] && B[j-1] == C[k-1]) {
                        dp[i][j][k] = 1 + dp[i-1][j-1][k-1];
                    } else {
                        dp[i][j][k] = max({dp[i-1][j][k], dp[i][j-1][k], dp[i][j][k-1]});
                    }
                }
            }
        }
        return dp[l][m][n];
    }

    int LCS_Three_Memo(string& A, string& B, string& C) {
        /*
        Memoization approach
        Time Complexity: O(l*m*n)
        Space Complexity: O(l*m*n)
        */
        int l = A.length(), m = B.length(), n = C.length();
        vector<vector<vector<int>>> dp(l+1, vector<vector<int>>(m+1, vector<int>(n+1, -1)));
        return LCS_Three_Memo_Helper(A, B, C, l, m, n, dp);
    }

    int LCS_Three_Memo_Helper(string& A, string& B, string& C, int i, int j, int k, vector<vector<vector<int>>>& dp) {
        if (i == 0 || j == 0 || k == 0) return 0;
        if (dp[i][j][k] != -1) return dp[i][j][k];
        if (A[i-1] == B[j-1] && B[j-1] == C[k-1]) {
            dp[i][j][k] = 1 + LCS_Three_Memo_Helper(A, B, C, i-1, j-1, k-1, dp);
        } else {
            dp[i][j][k] = max({LCS_Three_Memo_Helper(A, B, C, i-1, j, k, dp),
                               LCS_Three_Memo_Helper(A, B, C, i, j-1, k, dp),
                               LCS_Three_Memo_Helper(A, B, C, i, j, k-1, dp)});
        }
        return dp[i][j][k];
    }
};

void Test_LCS_Three() {
    Solution solution;
    string A = "geeks", B = "geeksfor", C = "geeksforgeeks";
    cout << "Tabulation: " << solution.LCS_Three_Tab(A, B, C) << endl;
    cout << "Memoization: " << solution.LCS_Three_Memo(A, B, C) << endl;
}

int main() {
    Test_LCS_Three();
    return 0;
}
