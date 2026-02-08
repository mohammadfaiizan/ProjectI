/*
Problem: LCS Space Optimized
URL: https://www.geeksforgeeks.org/space-optimized-solution-lcs/

Problem Statement:
Find the length of longest common subsequence of two strings using O(min(m,n)) space complexity.

Sample Input/Output:
Input: "AGGTAB", "GXTXAYB"
Output: 4
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int LCS_Space_Two_Row(string& s1, string& s2) {
        /*
        Space optimized using two rows
        Time Complexity: O(m*n)
        Space Complexity: O(min(m,n))
        */
        int m = s1.length(), n = s2.length();
        string *str1 = &s1, *str2 = &s2;
        if (m < n) {
            swap(str1, str2);
            swap(m, n);
        }
        vector<vector<int>> dp(2, vector<int>(n+1, 0));
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if ((*str1)[i-1] == (*str2)[j-1]) {
                    dp[i%2][j] = 1 + dp[(i-1)%2][j-1];
                } else {
                    dp[i%2][j] = max(dp[(i-1)%2][j], dp[i%2][j-1]);
                }
            }
        }
        return dp[m%2][n];
    }
};

void Test_LCS_Space() {
    Solution solution;
    string s1 = "AGGTAB", s2 = "GXTXAYB";
    cout << "LCS Length: " << solution.LCS_Space_Two_Row(s1, s2) << endl;
}

int main() {
    Test_LCS_Space();
    return 0;
}
