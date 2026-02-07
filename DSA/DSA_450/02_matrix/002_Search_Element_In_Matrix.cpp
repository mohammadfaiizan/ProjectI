/*
Problem: Search an Element in a Matrix
URL: https://leetcode.com/problems/search-a-2d-matrix/

Problem Statement:
Given a row-wise and column-wise sorted matrix where the first integer of each row
is greater than the last integer of the previous row, search for a target value.

Sample Input/Output:
Input: matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3
Output: true

Input: matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 13
Output: false
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    bool Search_Matrix_Binary_Search_Optimal(vector<vector<int>>& matrix, int target) {
        /*
        Binary Search on Flattened Matrix - Treat matrix as sorted 1D array
        Time Complexity: O(log(m * n))
        Space Complexity: O(1)
        */
        if (matrix.empty() || matrix[0].empty()) return false;
        int m = matrix.size(), n = matrix[0].size();
        int start = 0, end = m * n - 1;
        while (start <= end) {
            int mid = start + (end - start) / 2;
            int val = matrix[mid / n][mid % n];
            if (val == target) return true;
            else if (val < target) start = mid + 1;
            else end = mid - 1;
        }
        return false;
    }

    bool Search_Matrix_Staircase(vector<vector<int>>& matrix, int target) {
        /*
        Staircase Search - Start from bottom-left or top-right corner
        Time Complexity: O(m + n)
        Space Complexity: O(1)
        */
        int m = matrix.size(), n = matrix[0].size();
        int i = m - 1, j = 0;
        while (i >= 0 && j < n) {
            if (matrix[i][j] == target) return true;
            else if (matrix[i][j] > target) i--;
            else j++;
        }
        return false;
    }

    bool Search_Matrix_Row_Then_Col(vector<vector<int>>& matrix, int target) {
        /*
        Row Binary Search + Col Binary Search - Find row then search column
        Time Complexity: O(log m + log n)
        Space Complexity: O(1)
        */
        int m = matrix.size(), n = matrix[0].size();
        int lo = 0, hi = m - 1;
        while (lo <= hi) {
            int mid = lo + (hi - lo) / 2;
            if (matrix[mid][0] <= target && (mid == m - 1 || matrix[mid + 1][0] > target)) {
                int clo = 0, chi = n - 1;
                while (clo <= chi) {
                    int cmid = clo + (chi - clo) / 2;
                    if (matrix[mid][cmid] == target) return true;
                    else if (matrix[mid][cmid] < target) clo = cmid + 1;
                    else chi = cmid - 1;
                }
                return false;
            } else if (matrix[mid][0] > target) hi = mid - 1;
            else lo = mid + 1;
        }
        return false;
    }
};

void Test_Search_Element_In_Matrix() {
    Solution solution;

    struct TestCase {
        vector<vector<int>> matrix;
        int target;
        bool expected;
    };

    vector<TestCase> test_cases = {
        {{{1,3,5,7},{10,11,16,20},{23,30,34,60}}, 3, true},
        {{{1,3,5,7},{10,11,16,20},{23,30,34,60}}, 13, false},
        {{{1}}, 1, true},
        {{{1,3},{5,7}}, 5, true}
    };

    for (auto& tc : test_cases) {
        cout << "Target=" << tc.target << ", Expected: " << (tc.expected ? "true" : "false") << endl;

        cout << "Binary Search: " << (solution.Search_Matrix_Binary_Search_Optimal(tc.matrix, tc.target) ? "true" : "false") << endl;
        cout << "Staircase: " << (solution.Search_Matrix_Staircase(tc.matrix, tc.target) ? "true" : "false") << endl;
        cout << "Row+Col: " << (solution.Search_Matrix_Row_Then_Col(tc.matrix, tc.target) ? "true" : "false") << endl;

        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Search_Element_In_Matrix();
    return 0;
}
