/*
Problem: Median in a Row-wise Sorted Matrix
URL: https://practice.geeksforgeeks.org/problems/median-in-a-row-wise-sorted-matrix1527/1

Problem Statement:
Given a row-wise sorted matrix of size R x C where R and C are always odd,
find the median of the matrix.

Sample Input/Output:
Input: matrix = [[1, 3, 5],
                 [2, 6, 9],
                 [3, 6, 9]]
Output: 5
Explanation: Sorted array is [1, 2, 3, 3, 5, 6, 6, 9, 9]. Median is 5.

Input: matrix = [[1], [2], [3]]
Output: 2
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Median_Binary_Search_Optimal(vector<vector<int>>& matrix) {
        /*
        Binary Search on Value Range - Count elements less than mid
        Time Complexity: O(R * log(C) * log(max - min))
        Space Complexity: O(1)
        */
        int r = matrix.size(), c = matrix[0].size();
        int mn = INT_MAX, mx = INT_MIN;
        for (int i = 0; i < r; i++) {
            mn = min(mn, matrix[i][0]);
            mx = max(mx, matrix[i][c - 1]);
        }
        int desired = (r * c + 1) / 2;
        while (mn < mx) {
            int mid = mn + (mx - mn) / 2;
            int count = 0;
            for (int i = 0; i < r; i++) {
                count += upper_bound(matrix[i].begin(), matrix[i].end(), mid) - matrix[i].begin();
            }
            if (count < desired) mn = mid + 1;
            else mx = mid;
        }
        return mn;
    }

    int Median_Flatten_And_Sort(vector<vector<int>>& matrix) {
        /*
        Flatten and Sort - Merge all elements and find median
        Time Complexity: O(R * C * log(R * C))
        Space Complexity: O(R * C)
        */
        vector<int> all;
        for (auto& row : matrix)
            for (int x : row) all.push_back(x);
        sort(all.begin(), all.end());
        return all[all.size() / 2];
    }
};

void Test_Median_Row_Wise_Sorted() {
    Solution solution;

    struct TestCase {
        vector<vector<int>> matrix;
        int expected;
    };

    vector<TestCase> test_cases = {
        {{{1, 3, 5}, {2, 6, 9}, {3, 6, 9}}, 5},
        {{{1}, {2}, {3}}, 2},
        {{{1, 3, 4}, {2, 5, 6}, {7, 8, 9}}, 5}
    };

    for (auto& tc : test_cases) {
        cout << "Matrix:" << endl;
        for (auto& row : tc.matrix) {
            for (int x : row) cout << x << " ";
            cout << endl;
        }
        cout << "Expected: " << tc.expected << endl;

        cout << "Binary Search: " << solution.Median_Binary_Search_Optimal(tc.matrix) << endl;
        cout << "Flatten & Sort: " << solution.Median_Flatten_And_Sort(tc.matrix) << endl;

        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Median_Row_Wise_Sorted();
    return 0;
}
