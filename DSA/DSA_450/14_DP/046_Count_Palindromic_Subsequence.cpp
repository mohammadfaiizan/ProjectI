/*
Problem: Count Palindromic Subsequence
URL: https://practice.geeksforgeeks.org/problems/count-palindromic-subsequences/1

Problem Statement:
Given a string str of length N, count the number of palindromic subsequences (not necessarily contiguous) in the string.

Sample Input/Output:
Input: "abcd"
Output: 4
Input: "aab"
Output: 4
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    long long Count_Pal_Sub_Memo(string& s, int i, int j, vector<vector<long long>>& dp) {
        /*
        Memoization approach
        Time Complexity: O(n^2)
        Space Complexity: O(n^2)
        */
        if (i > j) return 0;
        if (i == j) return 1;
        if (dp[i][j] != -1) return dp[i][j];
        
        if (s[i] == s[j]) {
            dp[i][j] = Count_Pal_Sub_Memo(s, i + 1, j, dp) + 
                       Count_Pal_Sub_Memo(s, i, j - 1, dp) + 1;
        } else {
            dp[i][j] = Count_Pal_Sub_Memo(s, i + 1, j, dp) + 
                       Count_Pal_Sub_Memo(s, i, j - 1, dp) - 
                       Count_Pal_Sub_Memo(s, i + 1, j - 1, dp);
        }
        return dp[i][j];
    }
    
    long long Count_Pal_Sub_Tab(string& s) {
        /*
        Tabulation approach
        Time Complexity: O(n^2)
        Space Complexity: O(n^2)
        */
        int n = s.length();
        vector<vector<long long>> dp(n, vector<long long>(n, 0));
        
        for (int i = n - 1; i >= 0; i--) {
            for (int j = i; j < n; j++) {
                if (i == j) {
                    dp[i][j] = 1;
                } else if (s[i] == s[j]) {
                    dp[i][j] = dp[i + 1][j] + dp[i][j - 1] + 1;
                } else {
                    dp[i][j] = dp[i + 1][j] + dp[i][j - 1] - dp[i + 1][j - 1];
                }
            }
        }
        return dp[0][n - 1];
    }
};

void Test_Count_Pal_Sub() {
    Solution solution;
    
    string s1 = "abcd";
    vector<vector<long long>> dp1(s1.length(), vector<long long>(s1.length(), -1));
    long long result1 = solution.Count_Pal_Sub_Memo(s1, 0, s1.length() - 1, dp1);
    cout << "Memo: " << s1 << " -> " << result1 << endl;
    cout << "Tab: " << s1 << " -> " << solution.Count_Pal_Sub_Tab(s1) << endl;
    
    string s2 = "aab";
    vector<vector<long long>> dp2(s2.length(), vector<long long>(s2.length(), -1));
    long long result2 = solution.Count_Pal_Sub_Memo(s2, 0, s2.length() - 1, dp2);
    cout << "Memo: " << s2 << " -> " << result2 << endl;
    cout << "Tab: " << s2 << " -> " << solution.Count_Pal_Sub_Tab(s2) << endl;
}

int main() {
    Test_Count_Pal_Sub();
    return 0;
}
