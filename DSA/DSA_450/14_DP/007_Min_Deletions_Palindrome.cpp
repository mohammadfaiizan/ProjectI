/*
Problem: Minimum Deletions to Make Palindrome
URL: https://practice.geeksforgeeks.org/problems/minimum-number-of-deletions4610/1

Problem Statement:
Given a string of S as input. Your task is to write a program to remove or delete minimum number of characters from the string so that the resultant string is palindrome.

Sample Input/Output:
Input: "aebcbda"
Output: 2
Explanation: Remove characters 'e' and 'd', result: "abcba"
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Min_Deletions_Min_Del_Via_LPS(string& s, int n) {
        /*
        Min deletions via LPS approach
        Time Complexity: O(n^2)
        Space Complexity: O(n^2)
        */
        int lps = Longest_Palindromic_Subsequence(s, n);
        return n - lps;
    }

    int Longest_Palindromic_Subsequence(string& s, int n) {
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

    int Min_Deletions_Min_Del_Direct_DP(string& s, int n) {
        /*
        Direct DP approach
        Time Complexity: O(n^2)
        Space Complexity: O(n^2)
        */
        vector<vector<int>> dp(n, vector<int>(n, 0));
        for (int i = 0; i < n; i++) {
            dp[i][i] = 0;
        }
        for (int len = 2; len <= n; len++) {
            for (int i = 0; i <= n - len; i++) {
                int j = i + len - 1;
                if (s[i] == s[j]) {
                    dp[i][j] = dp[i+1][j-1];
                } else {
                    dp[i][j] = 1 + min(dp[i+1][j], dp[i][j-1]);
                }
            }
        }
        return dp[0][n-1];
    }
};

void Test_Min_Deletions_Palindrome() {
    Solution solution;
    string s = "aebcbda";
    
    cout << "Via LPS: " << solution.Min_Deletions_Min_Del_Via_LPS(s, s.length()) << endl;
    cout << "Direct DP: " << solution.Min_Deletions_Min_Del_Direct_DP(s, s.length()) << endl;
}

int main() {
    Test_Min_Deletions_Palindrome();
    return 0;
}
