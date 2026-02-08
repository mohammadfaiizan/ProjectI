/*
Problem: Longest Common Substring
URL: https://practice.geeksforgeeks.org/problems/longest-common-substring1452/1

Problem Statement:
Given two strings X and Y. The task is to find the length of the longest common substring.

Sample Input/Output:
Input: "ABCDGH", "ACDGHR"
Output: 4
Explanation: Longest common substring is "CDGH"
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Longest_Common_Substring_DP_Tabulation(string& s1, string& s2, int m, int n) {
        /*
        DP Tabulation approach
        Time Complexity: O(m*n)
        Space Complexity: O(m*n)
        */
        vector<vector<int>> dp(m+1, vector<int>(n+1, 0));
        int result = 0;
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (s1[i-1] == s2[j-1]) {
                    dp[i][j] = 1 + dp[i-1][j-1];
                    result = max(result, dp[i][j]);
                } else {
                    dp[i][j] = 0;
                }
            }
        }
        return result;
    }

    int Longest_Common_Substring_Space_Optimized(string& s1, string& s2, int m, int n) {
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
        int result = 0;
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (s1[i-1] == s2[j-1]) {
                    curr[j] = 1 + prev[j-1];
                    result = max(result, curr[j]);
                } else {
                    curr[j] = 0;
                }
            }
            prev = curr;
        }
        return result;
    }
};

void Test_Longest_Common_Substring() {
    Solution solution;
    string s1 = "ABCDGH";
    string s2 = "ACDGHR";
    
    cout << "DP Tabulation: " << solution.Longest_Common_Substring_DP_Tabulation(s1, s2, s1.length(), s2.length()) << endl;
    cout << "Space Optimized: " << solution.Longest_Common_Substring_Space_Optimized(s1, s2, s1.length(), s2.length()) << endl;
}

int main() {
    Test_Longest_Common_Substring();
    return 0;
}
