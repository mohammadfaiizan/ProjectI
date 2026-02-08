/*
Problem: Longest Palindromic Subsequence
URL: https://leetcode.com/problems/longest-palindromic-subsequence/

Problem Statement:
Given a string s, find the longest palindromic subsequence's length in s. A subsequence is a sequence that can be derived from another sequence by deleting some or no elements without changing the order of the remaining elements.

Sample Input/Output:
Input: "bbbab"
Output: 4
Input: "cbbd"
Output: 2
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Longest_Palindromic_Subsequence_LPS_DP(string& s, int n) {
        /*
        Direct DP approach
        Time Complexity: O(n^2)
        Space Complexity: O(n^2)
        */
        vector<vector<int>> dp(n, vector<int>(n, 0));
        for (int i = 0; i < n; i++) {
            dp[i][i] = 1;
        }
        for (int len = 2; len <= n; len++) {
            for (int i = 0; i <= n - len; i++) {
                int j = i + len - 1;
                if (s[i] == s[j]) {
                    dp[i][j] = 2 + (len == 2 ? 0 : dp[i+1][j-1]);
                } else {
                    dp[i][j] = max(dp[i+1][j], dp[i][j-1]);
                }
            }
        }
        return dp[0][n-1];
    }

    int Longest_Palindromic_Subsequence_LPS_Via_LCS(string& s, int n) {
        /*
        LPS via LCS approach
        Time Complexity: O(n^2)
        Space Complexity: O(n^2)
        */
        string rev = s;
        reverse(rev.begin(), rev.end());
        return LCS_Helper(s, rev, n, n);
    }

    int LCS_Helper(string& s1, string& s2, int m, int n) {
        vector<vector<int>> dp(m+1, vector<int>(n+1, 0));
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (s1[i-1] == s2[j-1]) {
                    dp[i][j] = 1 + dp[i-1][j-1];
                } else {
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
                }
            }
        }
        return dp[m][n];
    }
};

void Test_Longest_Palindromic_Subsequence() {
    Solution solution;
    string s1 = "bbbab";
    string s2 = "cbbd";
    
    cout << "Test 1 (bbbab) - LPS DP: " << solution.Longest_Palindromic_Subsequence_LPS_DP(s1, s1.length()) << endl;
    cout << "Test 1 (bbbab) - Via LCS: " << solution.Longest_Palindromic_Subsequence_LPS_Via_LCS(s1, s1.length()) << endl;
    cout << "Test 2 (cbbd) - LPS DP: " << solution.Longest_Palindromic_Subsequence_LPS_DP(s2, s2.length()) << endl;
    cout << "Test 2 (cbbd) - Via LCS: " << solution.Longest_Palindromic_Subsequence_LPS_Via_LCS(s2, s2.length()) << endl;
}

int main() {
    Test_Longest_Palindromic_Subsequence();
    return 0;
}
