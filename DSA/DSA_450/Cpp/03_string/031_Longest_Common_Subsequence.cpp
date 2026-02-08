/*
Problem: Longest Common Subsequence
URL: https://practice.geeksforgeeks.org/problems/longest-common-subsequence-1587115620/1

Problem Statement:
Given two strings s1 and s2, find the length of the longest common subsequence.
A subsequence is a sequence that appears in the same relative order but not
necessarily contiguous.

Sample Input/Output:
Input: s1 = "ABCDGH", s2 = "AEDFHR"
Output: 3 (ADH)

Input: s1 = "ABC", s2 = "AC"
Output: 2 (AC)
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int LCS_Tabulation(string s1, string s2) {
        /*
        Bottom-up DP
        Time Complexity: O(m * n)
        Space Complexity: O(m * n)
        */
        int m = s1.size(), n = s2.size();
        vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (s1[i - 1] == s2[j - 1])
                    dp[i][j] = 1 + dp[i - 1][j - 1];
                else
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
            }
        }
        return dp[m][n];
    }

    int LCS_Space_Optimized(string s1, string s2) {
        /*
        Space optimized using two rows
        Time Complexity: O(m * n)
        Space Complexity: O(n)
        */
        int m = s1.size(), n = s2.size();
        vector<int> prev(n + 1, 0), curr(n + 1, 0);

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (s1[i - 1] == s2[j - 1])
                    curr[j] = 1 + prev[j - 1];
                else
                    curr[j] = max(prev[j], curr[j - 1]);
            }
            prev = curr;
            fill(curr.begin(), curr.end(), 0);
        }
        return prev[n];
    }

    int LCS_Memoization(string& s1, string& s2, int i, int j, vector<vector<int>>& memo) {
        /*
        Top-down memoization
        Time Complexity: O(m * n)
        Space Complexity: O(m * n)
        */
        if (i == 0 || j == 0) return 0;
        if (memo[i][j] != -1) return memo[i][j];

        if (s1[i - 1] == s2[j - 1])
            return memo[i][j] = 1 + LCS_Memoization(s1, s2, i - 1, j - 1, memo);
        return memo[i][j] = max(LCS_Memoization(s1, s2, i - 1, j, memo),
                                LCS_Memoization(s1, s2, i, j - 1, memo));
    }

    string Print_LCS(string s1, string s2) {
        /*
        Print the actual LCS string
        Time Complexity: O(m * n)
        Space Complexity: O(m * n)
        */
        int m = s1.size(), n = s2.size();
        vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));

        for (int i = 1; i <= m; i++)
            for (int j = 1; j <= n; j++)
                dp[i][j] = (s1[i-1] == s2[j-1]) ? 1 + dp[i-1][j-1] : max(dp[i-1][j], dp[i][j-1]);

        string lcs = "";
        int i = m, j = n;
        while (i > 0 && j > 0) {
            if (s1[i - 1] == s2[j - 1]) {
                lcs = s1[i - 1] + lcs;
                i--; j--;
            } else if (dp[i - 1][j] > dp[i][j - 1]) {
                i--;
            } else {
                j--;
            }
        }
        return lcs;
    }
};

void Test_Longest_Common_Subsequence() {
    Solution sol;
    struct TestCase { string s1, s2; };
    vector<TestCase> tests = {
        {"ABCDGH", "AEDFHR"},
        {"ABC", "AC"},
        {"AGGTAB", "GXTXAYB"},
        {"abc", "def"}
    };

    for (auto& t : tests) {
        cout << "s1: " << t.s1 << ", s2: " << t.s2 << endl;
        cout << "Tabulation: " << sol.LCS_Tabulation(t.s1, t.s2) << endl;
        cout << "Space Optimized: " << sol.LCS_Space_Optimized(t.s1, t.s2) << endl;
        int m = t.s1.size(), n = t.s2.size();
        vector<vector<int>> memo(m + 1, vector<int>(n + 1, -1));
        cout << "Memoization: " << sol.LCS_Memoization(t.s1, t.s2, m, n, memo) << endl;
        cout << "LCS String: " << sol.Print_LCS(t.s1, t.s2) << endl;
        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Longest_Common_Subsequence();
    return 0;
}
