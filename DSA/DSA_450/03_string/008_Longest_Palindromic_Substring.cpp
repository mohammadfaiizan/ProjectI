/*
Problem: Longest Palindromic Substring
URL: https://practice.geeksforgeeks.org/problems/longest-palindrome-in-a-string3411/1

Problem Statement:
Given a string S, find the longest palindromic substring in S.

Sample Input/Output:
Input: S = "aaaabbaa"
Output: "aabbaa"

Input: S = "abc"
Output: "a" (or "b" or "c")
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    string Longest_Palindrome_Expand_Center(string s) {
        /*
        Expand around center for both odd and even length palindromes
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        */
        int n = s.size();
        if (n == 0) return "";
        int start = 0, maxLen = 1;

        for (int i = 1; i < n; i++) {
            int low = i - 1, high = i;
            while (low >= 0 && high < n && s[low] == s[high]) {
                if (high - low + 1 > maxLen) {
                    start = low;
                    maxLen = high - low + 1;
                }
                low--;
                high++;
            }

            low = i - 1;
            high = i + 1;
            while (low >= 0 && high < n && s[low] == s[high]) {
                if (high - low + 1 > maxLen) {
                    start = low;
                    maxLen = high - low + 1;
                }
                low--;
                high++;
            }
        }
        return s.substr(start, maxLen);
    }

    string Longest_Palindrome_DP(string s) {
        /*
        DP - dp[i][j] = true if s[i..j] is palindrome
        Time Complexity: O(n^2)
        Space Complexity: O(n^2)
        */
        int n = s.size();
        if (n == 0) return "";
        vector<vector<bool>> dp(n, vector<bool>(n, false));
        int start = 0, maxLen = 1;

        for (int i = 0; i < n; i++) dp[i][i] = true;

        for (int i = 0; i < n - 1; i++) {
            if (s[i] == s[i + 1]) {
                dp[i][i + 1] = true;
                start = i;
                maxLen = 2;
            }
        }

        for (int len = 3; len <= n; len++) {
            for (int i = 0; i <= n - len; i++) {
                int j = i + len - 1;
                if (s[i] == s[j] && dp[i + 1][j - 1]) {
                    dp[i][j] = true;
                    if (len > maxLen) {
                        start = i;
                        maxLen = len;
                    }
                }
            }
        }
        return s.substr(start, maxLen);
    }
};

void Test_Longest_Palindromic_Substring() {
    Solution sol;
    vector<string> tests = {"aaaabbaa", "abc", "babad", "cbbd", "a", "forgeeksskeegfor"};

    for (auto& s : tests) {
        cout << "Input: " << s << endl;
        cout << "Expand Center: " << sol.Longest_Palindrome_Expand_Center(s) << endl;
        cout << "DP: " << sol.Longest_Palindrome_DP(s) << endl;
        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Longest_Palindromic_Substring();
    return 0;
}
