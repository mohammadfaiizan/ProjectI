/*
Problem: Row with Maximum Number of 1s
URL: https://practice.geeksforgeeks.org/problems/row-with-max-1s0023/1

Problem Statement:
Given a boolean 2D array where each row is sorted in non-decreasing order,
find the index of the row with the maximum number of 1s.

Sample Input/Output:
Input: matrix = [[0, 1, 1, 1],
                 [0, 0, 1, 1],
                 [1, 1, 1, 1],
                 [0, 0, 0, 0]]
Output: 2
Explanation: Row 2 has maximum 1s (4 ones).

Input: matrix = [[0, 0], [1, 1]]
Output: 1
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Row_Max_Ones_Staircase_Optimal(vector<vector<int>>& arr) {
        /*
        Staircase Approach - Start from top-right, move left on 1 and down on 0
        Time Complexity: O(m + n)
        Space Complexity: O(1)
        */
        int n = arr.size(), m = arr[0].size();
        int row = 0, col = m - 1;
        int max_row_index = -1;
        while (row < n && col >= 0) {
            if (arr[row][col] == 1) {
                max_row_index = row;
                col--;
            } else {
                row++;
            }
        }
        return max_row_index;
    }

    int Row_Max_Ones_Binary_Search(vector<vector<int>>& arr) {
        /*
        Binary Search - Find first 1 in each row using binary search
        Time Complexity: O(m * log n)
        Space Complexity: O(1)
        */
        int n = arr.size(), m = arr[0].size();
        int max_ones = 0, max_row = -1;
        for (int i = 0; i < n; i++) {
            int first_one = First_One_Index(arr[i], m);
            if (first_one != -1) {
                int ones = m - first_one;
                if (ones > max_ones) {
                    max_ones = ones;
                    max_row = i;
                }
            }
        }
        return max_row;
    }

private:
    int First_One_Index(vector<int>& row, int m) {
        int lo = 0, hi = m - 1, result = -1;
        while (lo <= hi) {
            int mid = lo + (hi - lo) / 2;
            if (row[mid] == 1) {
                result = mid;
                hi = mid - 1;
            } else {
                lo = mid + 1;
            }
        }
        return result;
    }
};

void Test_Row_With_Maximum_Ones() {
    Solution solution;

    struct TestCase {
        vector<vector<int>> arr;
        int expected;
    };

    vector<TestCase> test_cases = {
        {{{0,1,1,1},{0,0,1,1},{1,1,1,1},{0,0,0,0}}, 2},
        {{{0,0},{1,1}}, 1},
        {{{0,0,0},{0,0,1},{0,1,1}}, 2},
        {{{1,1,1},{1,1,1},{0,0,0}}, 0}
    };

    for (auto& tc : test_cases) {
        cout << "Matrix:" << endl;
        for (auto& row : tc.arr) {
            for (int x : row) cout << x << " ";
            cout << endl;
        }
        cout << "Expected: " << tc.expected << endl;

        cout << "Staircase: " << solution.Row_Max_Ones_Staircase_Optimal(tc.arr) << endl;
        cout << "Binary Search: " << solution.Row_Max_Ones_Binary_Search(tc.arr) << endl;

        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Row_With_Maximum_Ones();
    return 0;
}
