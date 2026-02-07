/*
Problem: Find a Specific Pair in Matrix
URL: https://www.geeksforgeeks.org/find-a-specific-pair-in-matrix/

Problem Statement:
Given an N x N matrix, find the maximum value of mat[c][d] - mat[a][b] over all choices
of indices such that both c > a and d > b.

Sample Input/Output:
Input: mat = [[1,  2,  -1, -4, -20],
              [-8, -3,  4,  2,   1],
              [3,   8,  6,  1,   3],
              [-4, -1,  1,  7,  -6],
              [0,  -4, 10, -5,   1]]
Output: 18
Explanation: mat[4][2] - mat[0][0] = 10 - (-8) is not valid. Max is mat[2][1] - mat[0][2] etc.
             Actually max is mat[4][2] - mat[1][0] = 10 - (-8) = 18.

Input: mat = [[1, 2], [3, 4]]
Output: 2
Explanation: mat[1][1] - mat[0][0] = 4 - 1 = 3 but we need c>a and d>b. Max = 4-2=2 or 3-1=2 or 4-1=3.
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Specific_Pair_Preprocess_Optimal(vector<vector<int>>& mat) {
        /*
        Bottom-Right Max Preprocessing - Build max matrix from bottom-right
        Time Complexity: O(n^2)
        Space Complexity: O(n^2)
        */
        int n = mat.size();
        vector<vector<int>> maxArr(n, vector<int>(n));
        maxArr[n - 1][n - 1] = mat[n - 1][n - 1];
        int maxv = mat[n - 1][n - 1];
        for (int j = n - 2; j >= 0; j--) {
            maxv = max(maxv, mat[n - 1][j]);
            maxArr[n - 1][j] = maxv;
        }
        maxv = mat[n - 1][n - 1];
        for (int i = n - 2; i >= 0; i--) {
            maxv = max(maxv, mat[i][n - 1]);
            maxArr[i][n - 1] = maxv;
        }
        int result = INT_MIN;
        for (int i = n - 2; i >= 0; i--) {
            for (int j = n - 2; j >= 0; j--) {
                result = max(result, maxArr[i + 1][j + 1] - mat[i][j]);
                maxArr[i][j] = max(mat[i][j], max(maxArr[i + 1][j], maxArr[i][j + 1]));
            }
        }
        return result;
    }

    int Specific_Pair_Brute_Force(vector<vector<int>>& mat) {
        /*
        Brute Force - Check all valid pairs (a,b) and (c,d)
        Time Complexity: O(n^4)
        Space Complexity: O(1)
        */
        int n = mat.size();
        int result = INT_MIN;
        for (int a = 0; a < n; a++)
            for (int b = 0; b < n; b++)
                for (int c = a + 1; c < n; c++)
                    for (int d = b + 1; d < n; d++)
                        result = max(result, mat[c][d] - mat[a][b]);
        return result;
    }
};

void Test_Specific_Pair() {
    Solution solution;

    struct TestCase {
        vector<vector<int>> mat;
        int expected;
    };

    vector<TestCase> test_cases = {
        {{{1,2,-1,-4,-20},{-8,-3,4,2,1},{3,8,6,1,3},{-4,-1,1,7,-6},{0,-4,10,-5,1}}, 18},
        {{{1,2},{3,4}}, 3}
    };

    for (auto& tc : test_cases) {
        cout << "Matrix:" << endl;
        for (auto& row : tc.mat) {
            for (int x : row) cout << x << "\t";
            cout << endl;
        }
        cout << "Expected: " << tc.expected << endl;

        cout << "Preprocess: " << solution.Specific_Pair_Preprocess_Optimal(tc.mat) << endl;
        cout << "Brute Force: " << solution.Specific_Pair_Brute_Force(tc.mat) << endl;

        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Specific_Pair();
    return 0;
}
