/*
Problem: Longest Palindromic Substring
URL: https://leetcode.com/problems/longest-palindromic-substring/

Problem Statement:
Given a string s, return the longest palindromic substring in s.

Sample Input/Output:
Input: "babad"
Output: "bab"
Input: "cbbd"
Output: "bb"
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    string LPS_Expand(string s) {
        /*
        Expand Around Centers
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        */
        int n = s.length();
        if (n == 0) return "";
        
        int start = 0, maxLen = 1;
        
        for (int i = 0; i < n; i++) {
            int len1 = expandAroundCenter(s, i, i);
            int len2 = expandAroundCenter(s, i, i + 1);
            int len = max(len1, len2);
            
            if (len > maxLen) {
                maxLen = len;
                start = i - (len - 1) / 2;
            }
        }
        
        return s.substr(start, maxLen);
    }
    
    string LPS_DP(string s) {
        /*
        Dynamic Programming
        Time Complexity: O(n^2)
        Space Complexity: O(n^2)
        */
        int n = s.length();
        if (n == 0) return "";
        
        vector<vector<bool>> dp(n, vector<bool>(n, false));
        int start = 0, maxLen = 1;
        
        for (int i = 0; i < n; i++) {
            dp[i][i] = true;
        }
        
        for (int i = 0; i < n - 1; i++) {
            if (s[i] == s[i + 1]) {
                dp[i][i + 1] = true;
                start = i;
                maxLen = 2;
            }
        }
        
        for (int len = 3; len <= n; len++) {
            for (int i = 0; i < n - len + 1; i++) {
                int j = i + len - 1;
                if (s[i] == s[j] && dp[i + 1][j - 1]) {
                    dp[i][j] = true;
                    start = i;
                    maxLen = len;
                }
            }
        }
        
        return s.substr(start, maxLen);
    }
    
private:
    int expandAroundCenter(string& s, int left, int right) {
        while (left >= 0 && right < s.length() && s[left] == s[right]) {
            left--;
            right++;
        }
        return right - left - 1;
    }
};

void Test_LPS_Expand() {
    Solution solution;
    assert(solution.LPS_Expand("babad") == "bab" || solution.LPS_Expand("babad") == "aba");
    assert(solution.LPS_Expand("cbbd") == "bb");
}

void Test_LPS_DP() {
    Solution solution;
    assert(solution.LPS_DP("babad") == "bab" || solution.LPS_DP("babad") == "aba");
    assert(solution.LPS_DP("cbbd") == "bb");
}

int main() {
    Test_LPS_Expand();
    Test_LPS_DP();
    return 0;
}
