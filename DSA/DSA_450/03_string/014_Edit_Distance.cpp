/*
Problem: Edit Distance
URL: https://practice.geeksforgeeks.org/problems/edit-distance3702/1

Problem Statement:
Given two strings s and t of lengths m and n respectively, find the edit distance
between them. Edit Distance is defined as the minimum number of operations required
to convert string s to string t. Operations: Insert, Remove, Replace.

Sample Input/Output:
Input: s = "geek", t = "gesek"
Output: 1 (insert 's')

Input: s = "horse", t = "ros"
Output: 3
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Edit_Distance_Tabulation(string s, string t) {
        /*
        Bottom-up DP
        Time Complexity: O(m * n)
        Space Complexity: O(m * n)
        */
        int m = s.size(), n = t.size();
        vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));

        for (int i = 0; i <= m; i++) dp[i][0] = i;
        for (int j = 0; j <= n; j++) dp[0][j] = j;

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (s[i - 1] == t[j - 1])
                    dp[i][j] = dp[i - 1][j - 1];
                else
                    dp[i][j] = 1 + min({dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]});
            }
        }
        return dp[m][n];
    }

    int Edit_Distance_Space_Optimized(string s, string t) {
        /*
        Space optimized using two rows
        Time Complexity: O(m * n)
        Space Complexity: O(n)
        */
        int m = s.size(), n = t.size();
        vector<int> prev(n + 1), curr(n + 1);
        for (int j = 0; j <= n; j++) prev[j] = j;

        for (int i = 1; i <= m; i++) {
            curr[0] = i;
            for (int j = 1; j <= n; j++) {
                if (s[i - 1] == t[j - 1])
                    curr[j] = prev[j - 1];
                else
                    curr[j] = 1 + min({prev[j], curr[j - 1], prev[j - 1]});
            }
            prev = curr;
        }
        return prev[n];
    }

    int Edit_Distance_Recursive(string& s, string& t, int i, int j, vector<vector<int>>& memo) {
        /*
        Top-down memoization
        Time Complexity: O(m * n)
        Space Complexity: O(m * n)
        */
        if (i == 0) return j;
        if (j == 0) return i;
        if (memo[i][j] != -1) return memo[i][j];

        if (s[i - 1] == t[j - 1])
            return memo[i][j] = Edit_Distance_Recursive(s, t, i - 1, j - 1, memo);

        return memo[i][j] = 1 + min({
            Edit_Distance_Recursive(s, t, i - 1, j, memo),
            Edit_Distance_Recursive(s, t, i, j - 1, memo),
            Edit_Distance_Recursive(s, t, i - 1, j - 1, memo)
        });
    }
};

void Test_Edit_Distance() {
    Solution sol;
    struct TestCase { string s, t; };
    vector<TestCase> tests = {
        {"geek", "gesek"},
        {"horse", "ros"},
        {"intention", "execution"},
        {"abc", "abc"},
        {"", "abc"}
    };

    for (auto& tc : tests) {
        cout << "s: \"" << tc.s << "\", t: \"" << tc.t << "\"" << endl;
        cout << "Tabulation: " << sol.Edit_Distance_Tabulation(tc.s, tc.t) << endl;
        cout << "Space Optimized: " << sol.Edit_Distance_Space_Optimized(tc.s, tc.t) << endl;
        int m = tc.s.size(), n = tc.t.size();
        vector<vector<int>> memo(m + 1, vector<int>(n + 1, -1));
        cout << "Memoization: " << sol.Edit_Distance_Recursive(tc.s, tc.t, m, n, memo) << endl;
        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Edit_Distance();
    return 0;
}
