/*
Problem: Maximum Path Sum in Matrix
URL: https://www.geeksforgeeks.org/maximum-path-sum-matrix/

Problem Statement:
Given a matrix of N * M. Find the maximum path sum in matrix. The maximum path is sum of all elements from first row to last row where you are allowed to move only down or diagonally down left or diagonally down right from the current cell.

Sample Input/Output:
Input: Matrix with values
Output: Maximum path sum
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Max_Path_DP(vector<vector<int>>& matrix, int m, int n) {
        /*
        DP approach
        Time Complexity: O(m*n)
        Space Complexity: O(m*n)
        */
        vector<vector<int>> dp(m, vector<int>(n));
        for (int j = 0; j < n; j++) {
            dp[0][j] = matrix[0][j];
        }
        for (int i = 1; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int max_val = dp[i-1][j];
                if (j > 0) max_val = max(max_val, dp[i-1][j-1]);
                if (j < n-1) max_val = max(max_val, dp[i-1][j+1]);
                dp[i][j] = matrix[i][j] + max_val;
            }
        }
        return *max_element(dp[m-1].begin(), dp[m-1].end());
    }
};

void Test_Max_Path() {
    Solution solution;
    vector<vector<int>> matrix = {
        {10, 10, 2, 0, 20, 4},
        {1, 0, 0, 30, 2, 5},
        {0, 10, 4, 0, 2, 0},
        {1, 0, 2, 20, 0, 4}
    };
    cout << "Max Path Sum: " << solution.Max_Path_DP(matrix, matrix.size(), matrix[0].size()) << endl;
}

int main() {
    Test_Max_Path();
    return 0;
}
