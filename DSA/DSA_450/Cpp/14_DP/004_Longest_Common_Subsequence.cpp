/*
Problem: Longest Common Subsequence
URL: https://practice.geeksforgeeks.org/problems/longest-common-subsequence-1587115620/1

Problem Statement:
Given two strings, find the length of longest subsequence present in both of them. A subsequence is a sequence that appears in the same relative order, but not necessarily contiguous.

Sample Input/Output:
Input: "ABCBDAB", "BDCAB"
Output: 4
Explanation: LCS is "BCAB" or "BDAB"
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int LCS_Recursive(string& s1, string& s2, int m, int n) {
        /*
        Recursive approach
        Time Complexity: O(2^(m+n))
        Space Complexity: O(m+n)
        */
        if (m == 0 || n == 0) return 0;
        if (s1[m-1] == s2[n-1]) {
            return 1 + LCS_Recursive(s1, s2, m-1, n-1);
        }
        return max(LCS_Recursive(s1, s2, m-1, n), LCS_Recursive(s1, s2, m, n-1));
    }

    int LCS_Memoization(string& s1, string& s2, int m, int n) {
        /*
        Memoization approach
        Time Complexity: O(m*n)
        Space Complexity: O(m*n)
        */
        vector<vector<int>> memo(m+1, vector<int>(n+1, -1));
        return LCS_Memo_Helper(s1, s2, m, n, memo);
    }

    int LCS_Memo_Helper(string& s1, string& s2, int m, int n, vector<vector<int>>& memo) {
        if (m == 0 || n == 0) return 0;
        if (memo[m][n] != -1) return memo[m][n];
        if (s1[m-1] == s2[n-1]) {
            memo[m][n] = 1 + LCS_Memo_Helper(s1, s2, m-1, n-1, memo);
        } else {
            memo[m][n] = max(LCS_Memo_Helper(s1, s2, m-1, n, memo),
                            LCS_Memo_Helper(s1, s2, m, n-1, memo));
        }
        return memo[m][n];
    }

    int LCS_Tabulation(string& s1, string& s2, int m, int n) {
        /*
        Tabulation approach
        Time Complexity: O(m*n)
        Space Complexity: O(m*n)
        */
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

    int LCS_Space_Optimized(string& s1, string& s2, int m, int n) {
        /*
        Space optimized approach
        Time Complexity: O(m*n)
        Space Complexity: O(min(m,n))
        */
        if (m < n) {
            swap(s1, s2);
            swap(m, n);
        }
        vector<int> prev(n+1, 0), curr(n+1, 0);
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (s1[i-1] == s2[j-1]) {
                    curr[j] = 1 + prev[j-1];
                } else {
                    curr[j] = max(prev[j], curr[j-1]);
                }
            }
            prev = curr;
        }
        return curr[n];
    }
};

void Test_Longest_Common_Subsequence() {
    Solution solution;
    string s1 = "ABCBDAB";
    string s2 = "BDCAB";
    
    cout << "Recursive: " << solution.LCS_Recursive(s1, s2, s1.length(), s2.length()) << endl;
    cout << "Memoization: " << solution.LCS_Memoization(s1, s2, s1.length(), s2.length()) << endl;
    cout << "Tabulation: " << solution.LCS_Tabulation(s1, s2, s1.length(), s2.length()) << endl;
    cout << "Space Optimized: " << solution.LCS_Space_Optimized(s1, s2, s1.length(), s2.length()) << endl;
}

int main() {
    Test_Longest_Common_Subsequence();
    return 0;
}
