/*
Problem: Maximum Size Rectangle of All 1s
URL: https://practice.geeksforgeeks.org/problems/max-rectangle/1

Problem Statement:
Given a binary matrix M of size N x M, find the maximum area of a rectangle
formed only of 1s in the given matrix.

Sample Input/Output:
Input: M = [[0, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 0, 0]]
Output: 8
Explanation: Rectangle from (1,0) to (2,3) has area 2*4 = 8.

Input: M = [[0, 1, 1],
            [1, 1, 1],
            [0, 1, 1]]
Output: 6
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Max_Rectangle_Histogram_Optimal(vector<vector<int>>& M) {
        /*
        Histogram Based - Build histogram row by row, find largest rectangle
        Time Complexity: O(n * m)
        Space Complexity: O(m)
        */
        int n = M.size(), m = M[0].size();
        vector<int> heights(m, 0);
        int max_area = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                heights[j] = M[i][j] ? heights[j] + 1 : 0;
            }
            max_area = max(max_area, Largest_Rectangle_Histogram(heights));
        }
        return max_area;
    }

    int Max_Rectangle_Brute_Force(vector<vector<int>>& M) {
        /*
        Brute Force - Check all possible rectangles
        Time Complexity: O(n^2 * m^2)
        Space Complexity: O(1)
        */
        int n = M.size(), m = M[0].size();
        int max_area = 0;
        for (int i = 0; i < n; i++) {
            vector<int> col_sum(m, 0);
            for (int j = i; j < n; j++) {
                for (int k = 0; k < m; k++) {
                    col_sum[k] += M[j][k];
                }
                int height = j - i + 1;
                int width = 0;
                for (int k = 0; k < m; k++) {
                    if (col_sum[k] == height) {
                        width++;
                        max_area = max(max_area, height * width);
                    } else {
                        width = 0;
                    }
                }
            }
        }
        return max_area;
    }

private:
    int Largest_Rectangle_Histogram(vector<int>& heights) {
        stack<int> st;
        int max_area = 0, n = heights.size();
        for (int i = 0; i <= n; i++) {
            int h = (i == n) ? 0 : heights[i];
            while (!st.empty() && h < heights[st.top()]) {
                int top = heights[st.top()];
                st.pop();
                int width = st.empty() ? i : i - st.top() - 1;
                max_area = max(max_area, top * width);
            }
            st.push(i);
        }
        return max_area;
    }
};

void Test_Max_Size_Rectangle() {
    Solution solution;

    struct TestCase {
        vector<vector<int>> M;
        int expected;
    };

    vector<TestCase> test_cases = {
        {{{0,1,1,0},{1,1,1,1},{1,1,1,1},{1,1,0,0}}, 8},
        {{{0,1,1},{1,1,1},{0,1,1}}, 6},
        {{{1}}, 1},
        {{{0,0},{0,0}}, 0}
    };

    for (auto& tc : test_cases) {
        cout << "Matrix:" << endl;
        for (auto& row : tc.M) {
            for (int x : row) cout << x << " ";
            cout << endl;
        }
        cout << "Expected: " << tc.expected << endl;

        cout << "Histogram: " << solution.Max_Rectangle_Histogram_Optimal(tc.M) << endl;
        cout << "Brute Force: " << solution.Max_Rectangle_Brute_Force(tc.M) << endl;

        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Max_Size_Rectangle();
    return 0;
}
