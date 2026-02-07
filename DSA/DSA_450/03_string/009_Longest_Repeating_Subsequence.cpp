/*
Problem: Longest Repeating Subsequence
URL: https://practice.geeksforgeeks.org/problems/longest-repeating-subsequence2004/1

Problem Statement:
Given string str, find the length of the longest repeating subsequence such that
the two subsequences don't use same element at same position.

Sample Input/Output:
Input: str = "axxxy"
Output: 2 (xx)

Input: str = "aab"
Output: 1 (a)
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int LRS_Tabulation(string s) {
        /*
        LCS variant - LCS of string with itself where i != j
        Time Complexity: O(n^2)
        Space Complexity: O(n^2)
        */
        int n = s.size();
        vector<vector<int>> dp(n + 1, vector<int>(n + 1, 0));

        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= n; j++) {
                if (s[i - 1] == s[j - 1] && i != j)
                    dp[i][j] = 1 + dp[i - 1][j - 1];
                else
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
            }
        }
        return dp[n][n];
    }

    int LRS_Space_Optimized(string s) {
        /*
        Space optimized using two rows
        Time Complexity: O(n^2)
        Space Complexity: O(n)
        */
        int n = s.size();
        vector<int> prev(n + 1, 0), curr(n + 1, 0);

        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= n; j++) {
                if (s[i - 1] == s[j - 1] && i != j)
                    curr[j] = 1 + prev[j - 1];
                else
                    curr[j] = max(prev[j], curr[j - 1]);
            }
            prev = curr;
        }
        return prev[n];
    }

    int LRS_Memoization(string& s, int i, int j, vector<vector<int>>& memo) {
        /*
        Top-down memoization
        Time Complexity: O(n^2)
        Space Complexity: O(n^2)
        */
        if (i == 0 || j == 0) return 0;
        if (memo[i][j] != -1) return memo[i][j];

        if (s[i - 1] == s[j - 1] && i != j)
            return memo[i][j] = 1 + LRS_Memoization(s, i - 1, j - 1, memo);
        return memo[i][j] = max(LRS_Memoization(s, i - 1, j, memo), LRS_Memoization(s, i, j - 1, memo));
    }
};

void Test_Longest_Repeating_Subsequence() {
    Solution sol;
    vector<string> tests = {"axxxy", "aab", "aabb", "abc", "aabebcdd"};

    for (auto& s : tests) {
        cout << "Input: " << s << endl;
        cout << "Tabulation: " << sol.LRS_Tabulation(s) << endl;
        cout << "Space Optimized: " << sol.LRS_Space_Optimized(s) << endl;
        int n = s.size();
        vector<vector<int>> memo(n + 1, vector<int>(n + 1, -1));
        cout << "Memoization: " << sol.LRS_Memoization(s, n, n, memo) << endl;
        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Longest_Repeating_Subsequence();
    return 0;
}
