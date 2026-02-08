/*
Problem: Longest Repeated Subsequence
URL: https://practice.geeksforgeeks.org/problems/longest-repeating-subsequence2004/1

Problem Statement:
Given string str, find the length of the longest repeating subsequence such that it can be found twice in the given string. The two identified subsequences A and B can use the same characters from the string but the positions of the characters in A and B must be different.

Sample Input/Output:
Input: "axxzxy"
Output: 2
Explanation: The longest repeating subsequence is "xx" or "xy"
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Longest_Repeated_Subsequence_LRS_Tabulation(string& str, int n) {
        /*
        LRS Tabulation approach
        Time Complexity: O(n^2)
        Space Complexity: O(n^2)
        */
        vector<vector<int>> dp(n+1, vector<int>(n+1, 0));
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= n; j++) {
                if (str[i-1] == str[j-1] && i != j) {
                    dp[i][j] = 1 + dp[i-1][j-1];
                } else {
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
                }
            }
        }
        return dp[n][n];
    }
};

void Test_Longest_Repeated_Subsequence() {
    Solution solution;
    string str = "axxzxy";
    
    cout << "LRS Tabulation: " << solution.Longest_Repeated_Subsequence_LRS_Tabulation(str, str.length()) << endl;
}

int main() {
    Test_Longest_Repeated_Subsequence();
    return 0;
}
