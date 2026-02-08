/*
Problem: Wildcard Pattern Matching
URL: https://practice.geeksforgeeks.org/problems/wildcard-string-matching1126/1

Problem Statement:
Given a text string and a wildcard pattern, implement wildcard pattern matching
with support for '?' (matches single character) and '*' (matches any sequence
of characters including empty).

Sample Input/Output:
Input: str = "baaabab", pattern = "*****ba*****ab"
Output: true

Input: str = "aa", pattern = "a"
Output: false
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    bool Wildcard_DP(string s, string p) {
        /*
        Bottom-up DP
        Time Complexity: O(m * n)
        Space Complexity: O(m * n)
        */
        int m = s.size(), n = p.size();
        vector<vector<bool>> dp(m + 1, vector<bool>(n + 1, false));
        dp[0][0] = true;

        for (int j = 1; j <= n; j++)
            if (p[j - 1] == '*') dp[0][j] = dp[0][j - 1];

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (p[j - 1] == '*')
                    dp[i][j] = dp[i - 1][j] || dp[i][j - 1];
                else if (p[j - 1] == '?' || s[i - 1] == p[j - 1])
                    dp[i][j] = dp[i - 1][j - 1];
            }
        }
        return dp[m][n];
    }

    bool Wildcard_Memoization(string& s, string& p, int i, int j, vector<vector<int>>& memo) {
        /*
        Top-down memoization
        Time Complexity: O(m * n)
        Space Complexity: O(m * n)
        */
        if (i < 0 && j < 0) return true;
        if (j < 0) return false;
        if (i < 0) {
            while (j >= 0) {
                if (p[j] != '*') return false;
                j--;
            }
            return true;
        }

        if (memo[i][j] != -1) return memo[i][j];

        if (p[j] == '*')
            return memo[i][j] = Wildcard_Memoization(s, p, i - 1, j, memo) ||
                                Wildcard_Memoization(s, p, i, j - 1, memo);

        if (p[j] == '?' || s[i] == p[j])
            return memo[i][j] = Wildcard_Memoization(s, p, i - 1, j - 1, memo);

        return memo[i][j] = 0;
    }

    bool Wildcard_Two_Pointer(string s, string p) {
        /*
        Two pointer / greedy approach
        Time Complexity: O(m * n) worst case, O(m + n) average
        Space Complexity: O(1)
        */
        int si = 0, pi = 0;
        int starIdx = -1, matchIdx = -1;
        int m = s.size(), n = p.size();

        while (si < m) {
            if (pi < n && (p[pi] == '?' || p[pi] == s[si])) {
                si++;
                pi++;
            } else if (pi < n && p[pi] == '*') {
                starIdx = pi;
                matchIdx = si;
                pi++;
            } else if (starIdx != -1) {
                pi = starIdx + 1;
                matchIdx++;
                si = matchIdx;
            } else {
                return false;
            }
        }

        while (pi < n && p[pi] == '*') pi++;
        return pi == n;
    }
};

void Test_Wildcard_Pattern_Matching() {
    Solution sol;
    struct TestCase { string s, p; };
    vector<TestCase> tests = {
        {"baaabab", "*****ba*****ab"},
        {"aa", "a"},
        {"aa", "*"},
        {"cb", "?a"},
        {"adceb", "*a*b"},
        {"acdcb", "a*c?b"},
        {"", "*"}
    };

    for (auto& t : tests) {
        cout << "str: \"" << t.s << "\", pattern: \"" << t.p << "\"" << endl;
        cout << "DP: " << sol.Wildcard_DP(t.s, t.p) << endl;
        int m = t.s.size(), n = t.p.size();
        vector<vector<int>> memo(m, vector<int>(n, -1));
        cout << "Memoization: " << (m > 0 && n > 0 ? sol.Wildcard_Memoization(t.s, t.p, m - 1, n - 1, memo) : sol.Wildcard_DP(t.s, t.p)) << endl;
        cout << "Two Pointer: " << sol.Wildcard_Two_Pointer(t.s, t.p) << endl;
        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Wildcard_Pattern_Matching();
    return 0;
}
