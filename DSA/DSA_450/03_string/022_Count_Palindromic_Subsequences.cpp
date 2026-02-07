/*
Problem: Count All Palindromic Subsequences
URL: https://practice.geeksforgeeks.org/problems/count-palindromic-subsequences/1

Problem Statement:
Given a string str of length N, find the number of palindromic subsequences
of length greater than or equal to 1.

Sample Input/Output:
Input: str = "abcd"
Output: 4 (a, b, c, d)

Input: str = "aab"
Output: 4 (a, a, b, aa)
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    long long Count_Palindromic_Subseq_DP(string s) {
        /*
        DP approach - dp[i][j] = count of palindromic subsequences in s[i..j]
        Time Complexity: O(n^2)
        Space Complexity: O(n^2)
        */
        int n = s.size();
        vector<vector<long long>> dp(n, vector<long long>(n, 0));

        for (int i = 0; i < n; i++) dp[i][i] = 1;

        for (int len = 2; len <= n; len++) {
            for (int i = 0; i <= n - len; i++) {
                int j = i + len - 1;
                if (s[i] == s[j])
                    dp[i][j] = dp[i + 1][j] + dp[i][j - 1] + 1;
                else
                    dp[i][j] = dp[i + 1][j] + dp[i][j - 1] - dp[i + 1][j - 1];
            }
        }
        return dp[0][n - 1];
    }

    long long Count_Palindromic_Subseq_Recursive(string& s, int i, int j, vector<vector<long long>>& memo) {
        /*
        Top-down memoization
        Time Complexity: O(n^2)
        Space Complexity: O(n^2)
        */
        if (i > j) return 0;
        if (i == j) return 1;
        if (memo[i][j] != -1) return memo[i][j];

        if (s[i] == s[j])
            memo[i][j] = Count_Palindromic_Subseq_Recursive(s, i + 1, j, memo)
                       + Count_Palindromic_Subseq_Recursive(s, i, j - 1, memo) + 1;
        else
            memo[i][j] = Count_Palindromic_Subseq_Recursive(s, i + 1, j, memo)
                       + Count_Palindromic_Subseq_Recursive(s, i, j - 1, memo)
                       - Count_Palindromic_Subseq_Recursive(s, i + 1, j - 1, memo);
        return memo[i][j];
    }

    int Count_Palindromic_Substrings(string s) {
        /*
        Count palindromic substrings (not subsequences) using expand around center
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        */
        int n = s.size(), count = 0;
        for (int center = 0; center < n; center++) {
            int lo = center, hi = center;
            while (lo >= 0 && hi < n && s[lo] == s[hi]) { count++; lo--; hi++; }

            lo = center;
            hi = center + 1;
            while (lo >= 0 && hi < n && s[lo] == s[hi]) { count++; lo--; hi++; }
        }
        return count;
    }
};

void Test_Count_Palindromic_Subsequences() {
    Solution sol;
    vector<string> tests = {"abcd", "aab", "aaaa", "abcb", "a"};

    for (auto& s : tests) {
        cout << "Input: " << s << endl;
        cout << "DP: " << sol.Count_Palindromic_Subseq_DP(s) << endl;
        int n = s.size();
        vector<vector<long long>> memo(n, vector<long long>(n, -1));
        cout << "Recursive: " << sol.Count_Palindromic_Subseq_Recursive(s, 0, n - 1, memo) << endl;
        cout << "Substrings: " << sol.Count_Palindromic_Substrings(s) << endl;
        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Count_Palindromic_Subsequences();
    return 0;
}
