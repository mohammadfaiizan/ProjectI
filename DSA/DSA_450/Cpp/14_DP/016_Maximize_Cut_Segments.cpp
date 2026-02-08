/*
Problem: Maximize Cut Segments
URL: https://practice.geeksforgeeks.org/problems/cutted-segments1642/1

Problem Statement:
Given an integer N denoting the Length of a line segment. You need to cut the line segment in such a way that the cut length of a line segment each time is either x, y or z. Here x, y, and z are integers. After performing all the cut operations, your total number of cut segments must be maximum.

Sample Input/Output:
Input: N = 4, x = 2, y = 1, z = 1
Output: 4
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Maximize_Cuts_Memo(int n, int x, int y, int z) {
        /*
        Memoization approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        vector<int> dp(n+1, -2);
        return Maximize_Cuts_Memo_Helper(n, x, y, z, dp);
    }

    int Maximize_Cuts_Memo_Helper(int n, int x, int y, int z, vector<int>& dp) {
        if (n == 0) return 0;
        if (n < 0) return INT_MIN;
        if (dp[n] != -2) return dp[n];
        int cut_x = Maximize_Cuts_Memo_Helper(n-x, x, y, z, dp);
        int cut_y = Maximize_Cuts_Memo_Helper(n-y, x, y, z, dp);
        int cut_z = Maximize_Cuts_Memo_Helper(n-z, x, y, z, dp);
        int result = max({cut_x, cut_y, cut_z});
        dp[n] = (result == INT_MIN) ? INT_MIN : result + 1;
        return dp[n];
    }

    int Maximize_Cuts_Tab(int n, int x, int y, int z) {
        /*
        Tabulation approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        vector<int> dp(n+1, INT_MIN);
        dp[0] = 0;
        for (int i = 1; i <= n; i++) {
            if (i >= x && dp[i-x] != INT_MIN) dp[i] = max(dp[i], dp[i-x] + 1);
            if (i >= y && dp[i-y] != INT_MIN) dp[i] = max(dp[i], dp[i-y] + 1);
            if (i >= z && dp[i-z] != INT_MIN) dp[i] = max(dp[i], dp[i-z] + 1);
        }
        return (dp[n] == INT_MIN) ? 0 : dp[n];
    }
};

void Test_Maximize_Cuts() {
    Solution solution;
    int n = 4, x = 2, y = 1, z = 1;
    cout << "Memoization: " << solution.Maximize_Cuts_Memo(n, x, y, z) << endl;
    cout << "Tabulation: " << solution.Maximize_Cuts_Tab(n, x, y, z) << endl;
}

int main() {
    Test_Maximize_Cuts();
    return 0;
}
