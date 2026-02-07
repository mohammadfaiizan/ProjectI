/*
Problem: Rotate Matrix by 90 Degrees Clockwise
URL: https://www.geeksforgeeks.org/rotate-a-matrix-by-90-degree-in-clockwise-direction-without-using-any-extra-space/

Problem Statement:
Given an N x N square matrix, rotate it by 90 degrees in clockwise direction
without using any extra space.

Sample Input/Output:
Input: matrix = [[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]]
Output: [[7, 4, 1],
         [8, 5, 2],
         [9, 6, 3]]

Input: matrix = [[1, 2],
                 [3, 4]]
Output: [[3, 1],
         [4, 2]]
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    void Rotate_90_Transpose_Reverse_Optimal(vector<vector<int>>& mat) {
        /*
        Transpose + Reverse Rows - Transpose matrix then reverse each row
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        */
        int n = mat.size();
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                swap(mat[i][j], mat[j][i]);
        for (int i = 0; i < n; i++)
            reverse(mat[i].begin(), mat[i].end());
    }

    void Rotate_90_Cycle_Swap(vector<vector<int>>& mat) {
        /*
        Cycle Swap - Swap elements of each cycle in clockwise direction
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        */
        int n = mat.size();
        for (int i = 0; i < n / 2; i++) {
            for (int j = i; j < n - i - 1; j++) {
                int temp = mat[i][j];
                mat[i][j] = mat[n - 1 - j][i];
                mat[n - 1 - j][i] = mat[n - 1 - i][n - 1 - j];
                mat[n - 1 - i][n - 1 - j] = mat[j][n - 1 - i];
                mat[j][n - 1 - i] = temp;
            }
        }
    }
};

void Test_Rotate_Matrix() {
    Solution solution;

    vector<vector<vector<int>>> test_cases = {
        {{1,2,3},{4,5,6},{7,8,9}},
        {{1,2},{3,4}},
        {{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}}
    };

    for (auto& mat : test_cases) {
        cout << "Original:" << endl;
        for (auto& row : mat) {
            for (int x : row) cout << x << "\t";
            cout << endl;
        }

        auto mat1 = mat, mat2 = mat;

        solution.Rotate_90_Transpose_Reverse_Optimal(mat1);
        cout << "Transpose+Reverse:" << endl;
        for (auto& row : mat1) {
            for (int x : row) cout << x << "\t";
            cout << endl;
        }

        solution.Rotate_90_Cycle_Swap(mat2);
        cout << "Cycle Swap:" << endl;
        for (auto& row : mat2) {
            for (int x : row) cout << x << "\t";
            cout << endl;
        }

        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Rotate_Matrix();
    return 0;
}
