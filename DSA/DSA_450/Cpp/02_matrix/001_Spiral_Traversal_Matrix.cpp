/*
Problem: Spiral Traversal of a Matrix
URL: https://practice.geeksforgeeks.org/problems/spirally-traversing-a-matrix-1587115621/1

Problem Statement:
Given a matrix of size R x C, print the elements in spiral order traversal.

Sample Input/Output:
Input: matrix = [[1, 2, 3, 4],
                 [5, 6, 7, 8],
                 [9, 10, 11, 12],
                 [13, 14, 15, 16]]
Output: [1, 2, 3, 4, 8, 12, 16, 15, 14, 13, 9, 5, 6, 7, 11, 10]

Input: matrix = [[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]]
Output: [1, 2, 3, 6, 9, 8, 7, 4, 5]
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<int> Spiral_Traversal_Boundary_Optimal(vector<vector<int>>& matrix) {
        /*
        Boundary Shrinking - Track top, bottom, left, right boundaries
        Time Complexity: O(m * n)
        Space Complexity: O(1) excluding result
        */
        vector<int> ans;
        if (matrix.empty()) return ans;
        int top = 0, bottom = matrix.size(), right = matrix[0].size(), left = 0;
        while (top < bottom && left < right) {
            for (int i = left; i < right; i++) ans.push_back(matrix[top][i]);
            top++;
            for (int i = top; i < bottom; i++) ans.push_back(matrix[i][right - 1]);
            right--;
            if (top < bottom) {
                for (int i = right - 1; i >= left; i--) ans.push_back(matrix[bottom - 1][i]);
                bottom--;
            }
            if (left < right) {
                for (int i = bottom - 1; i >= top; i--) ans.push_back(matrix[i][left]);
                left++;
            }
        }
        return ans;
    }

    vector<int> Spiral_Traversal_Direction_Array(vector<vector<int>>& matrix) {
        /*
        Direction Array - Use direction vectors and visited tracking
        Time Complexity: O(m * n)
        Space Complexity: O(m * n)
        */
        vector<int> ans;
        if (matrix.empty()) return ans;
        int R = matrix.size(), C = matrix[0].size();
        vector<vector<bool>> seen(R, vector<bool>(C, false));
        int dr[] = {0, 1, 0, -1};
        int dc[] = {1, 0, -1, 0};
        int r = 0, c = 0, di = 0;
        for (int i = 0; i < R * C; i++) {
            ans.push_back(matrix[r][c]);
            seen[r][c] = true;
            int cr = r + dr[di], cc = c + dc[di];
            if (cr >= 0 && cr < R && cc >= 0 && cc < C && !seen[cr][cc]) {
                r = cr;
                c = cc;
            } else {
                di = (di + 1) % 4;
                r += dr[di];
                c += dc[di];
            }
        }
        return ans;
    }
};

void Test_Spiral_Traversal() {
    Solution solution;

    vector<vector<vector<int>>> test_cases = {
        {{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}},
        {{1,2,3},{4,5,6},{7,8,9}},
        {{1,2,3,4},{5,6,7,8}},
        {{1},{2},{3}}
    };

    for (auto& matrix : test_cases) {
        cout << "Matrix:" << endl;
        for (auto& row : matrix) {
            for (int x : row) cout << x << "\t";
            cout << endl;
        }

        auto r1 = solution.Spiral_Traversal_Boundary_Optimal(matrix);
        cout << "Boundary: ";
        for (int x : r1) cout << x << " ";
        cout << endl;

        auto r2 = solution.Spiral_Traversal_Direction_Array(matrix);
        cout << "Direction: ";
        for (int x : r2) cout << x << " ";
        cout << endl;

        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Spiral_Traversal();
    return 0;
}
