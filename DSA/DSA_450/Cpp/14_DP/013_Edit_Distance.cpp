/*
Problem: Edit Distance
URL: https://practice.geeksforgeeks.org/problems/edit-distance3702/1

Problem Statement:
Given two strings s and t. Find the minimum number of operations that need to be performed on str1 to convert it to str2. The possible operations are: Insert, Remove, Replace.

Sample Input/Output:
Input: "horse", "ros"
Output: 3
Explanation: horse -> rorse -> rose -> ros
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Edit_Distance_Edit_Dist_Recursive(string& s1, string& s2, int m, int n) {
        /*
        Recursive approach
        Time Complexity: O(3^(m+n))
        Space Complexity: O(m+n)
        */
        if (m == 0) return n;
        if (n == 0) return m;
        if (s1[m-1] == s2[n-1]) {
            return Edit_Distance_Edit_Dist_Recursive(s1, s2, m-1, n-1);
        }
        return 1 + min({Edit_Distance_Edit_Dist_Recursive(s1, s2, m, n-1),
                        Edit_Distance_Edit_Dist_Recursive(s1, s2, m-1, n),
                        Edit_Distance_Edit_Dist_Recursive(s1, s2, m-1, n-1)});
    }

    int Edit_Distance_Edit_Dist_Memo(string& s1, string& s2, int m, int n) {
        /*
        Memoization approach
        Time Complexity: O(m*n)
        Space Complexity: O(m*n)
        */
        vector<vector<int>> memo(m+1, vector<int>(n+1, -1));
        return Edit_Dist_Memo_Helper(s1, s2, m, n, memo);
    }

    int Edit_Dist_Memo_Helper(string& s1, string& s2, int m, int n, vector<vector<int>>& memo) {
        if (m == 0) return n;
        if (n == 0) return m;
        if (memo[m][n] != -1) return memo[m][n];
        if (s1[m-1] == s2[n-1]) {
            memo[m][n] = Edit_Dist_Memo_Helper(s1, s2, m-1, n-1, memo);
        } else {
            memo[m][n] = 1 + min({Edit_Dist_Memo_Helper(s1, s2, m, n-1, memo),
                                 Edit_Dist_Memo_Helper(s1, s2, m-1, n, memo),
                                 Edit_Dist_Memo_Helper(s1, s2, m-1, n-1, memo)});
        }
        return memo[m][n];
    }

    int Edit_Distance_Edit_Dist_Tab(string& s1, string& s2, int m, int n) {
        /*
        Tabulation approach
        Time Complexity: O(m*n)
        Space Complexity: O(m*n)
        */
        vector<vector<int>> dp(m+1, vector<int>(n+1, 0));
        for (int i = 0; i <= m; i++) dp[i][0] = i;
        for (int j = 0; j <= n; j++) dp[0][j] = j;
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (s1[i-1] == s2[j-1]) {
                    dp[i][j] = dp[i-1][j-1];
                } else {
                    dp[i][j] = 1 + min({dp[i][j-1], dp[i-1][j], dp[i-1][j-1]});
                }
            }
        }
        return dp[m][n];
    }

    int Edit_Distance_Edit_Dist_Space(string& s1, string& s2, int m, int n) {
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
        for (int j = 0; j <= n; j++) prev[j] = j;
        for (int i = 1; i <= m; i++) {
            curr[0] = i;
            for (int j = 1; j <= n; j++) {
                if (s1[i-1] == s2[j-1]) {
                    curr[j] = prev[j-1];
                } else {
                    curr[j] = 1 + min({curr[j-1], prev[j], prev[j-1]});
                }
            }
            prev = curr;
        }
        return curr[n];
    }
};

void Test_Edit_Distance() {
    Solution solution;
    string s1 = "horse";
    string s2 = "ros";
    
    cout << "Recursive: " << solution.Edit_Distance_Edit_Dist_Recursive(s1, s2, s1.length(), s2.length()) << endl;
    cout << "Memoization: " << solution.Edit_Distance_Edit_Dist_Memo(s1, s2, s1.length(), s2.length()) << endl;
    cout << "Tabulation: " << solution.Edit_Distance_Edit_Dist_Tab(s1, s2, s1.length(), s2.length()) << endl;
    cout << "Space Optimized: " << solution.Edit_Distance_Edit_Dist_Space(s1, s2, s1.length(), s2.length()) << endl;
}

int main() {
    Test_Edit_Distance();
    return 0;
}
