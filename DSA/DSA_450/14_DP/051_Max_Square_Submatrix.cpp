/*
Problem: Maximum Square Submatrix
URL: https://practice.geeksforgeeks.org/problems/largest-square-formed-in-a-matrix0806/1

Problem Statement:
Given a binary matrix, find the maximum size square submatrix with all 1s.

Sample Input/Output:
Input: matrix = [[0,1,1,0,1],[1,1,0,1,0],[0,1,1,1,0],[1,1,1,1,0],[1,1,1,1,1],[0,0,0,0,0]]
Output: 3
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Max_Square_DP(vector<vector<int>>& matrix) {
        /*
        DP approach
        Time Complexity: O(mn)
        Space Complexity: O(mn)
        */
        int m = matrix.size();
        if (m == 0) return 0;
        int n = matrix[0].size();
        if (n == 0) return 0;
        
        vector<vector<int>> dp(m, vector<int>(n, 0));
        int maxSize = 0;
        
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == 1) {
                    if (i == 0 || j == 0) {
                        dp[i][j] = 1;
                    } else {
                        dp[i][j] = 1 + min({dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]});
                    }
                    maxSize = max(maxSize, dp[i][j]);
                }
            }
        }
        
        return maxSize;
    }
    
    int Max_Square_Space(vector<vector<int>>& matrix) {
        /*
        Space optimized approach
        Time Complexity: O(mn)
        Space Complexity: O(n)
        */
        int m = matrix.size();
        if (m == 0) return 0;
        int n = matrix[0].size();
        if (n == 0) return 0;
        
        vector<int> prev(n, 0);
        vector<int> curr(n, 0);
        int maxSize = 0;
        
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == 1) {
                    if (i == 0 || j == 0) {
                        curr[j] = 1;
                    } else {
                        curr[j] = 1 + min({prev[j], curr[j - 1], prev[j - 1]});
                    }
                    maxSize = max(maxSize, curr[j]);
                } else {
                    curr[j] = 0;
                }
            }
            prev = curr;
        }
        
        return maxSize;
    }
};

void Test_Max_Square() {
    Solution solution;
    
    vector<vector<int>> matrix = {
        {0,1,1,0,1},
        {1,1,0,1,0},
        {0,1,1,1,0},
        {1,1,1,1,0},
        {1,1,1,1,1},
        {0,0,0,0,0}
    };
    
    cout << "DP: " << solution.Max_Square_DP(matrix) << endl;
    cout << "Space Optimized: " << solution.Max_Square_Space(matrix) << endl;
}

int main() {
    Test_Max_Square();
    return 0;
}
