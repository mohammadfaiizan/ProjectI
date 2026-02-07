/*
Problem: Common Elements in All Rows of a Matrix
URL: https://www.geeksforgeeks.org/common-elements-in-all-rows-of-a-given-matrix/

Problem Statement:
Given an M x N matrix, find all common elements present in all rows.

Sample Input/Output:
Input: mat = [[1, 2, 1, 4, 8],
              [3, 7, 8, 5, 1],
              [8, 7, 7, 3, 1],
              [8, 1, 2, 7, 9]]
Output: [1, 8]
Explanation: 1 and 8 appear in all rows.

Input: mat = [[1, 2, 3],
              [4, 5, 6]]
Output: []
Explanation: No common element in all rows.
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<int> Common_Elements_Map_Optimal(vector<vector<int>>& mat) {
        /*
        Row-wise Map Counting - Track elements appearing in each row
        Time Complexity: O(m * n)
        Space Complexity: O(n)
        */
        int rows = mat.size(), cols = mat[0].size();
        map<int, int> mp;
        for (int j = 0; j < cols; j++) mp[mat[0][j]] = 1;
        for (int i = 1; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (mp[mat[i][j]] == i) {
                    mp[mat[i][j]] = i + 1;
                }
            }
        }
        vector<int> result;
        for (auto& [val, count] : mp) {
            if (count == rows) result.push_back(val);
        }
        return result;
    }

    vector<int> Common_Elements_Set_Intersection(vector<vector<int>>& mat) {
        /*
        Set Intersection - Intersect sets of each row
        Time Complexity: O(m * n * log n)
        Space Complexity: O(n)
        */
        set<int> common(mat[0].begin(), mat[0].end());
        for (int i = 1; i < (int)mat.size(); i++) {
            set<int> row_set(mat[i].begin(), mat[i].end());
            set<int> intersection;
            set_intersection(common.begin(), common.end(),
                             row_set.begin(), row_set.end(),
                             inserter(intersection, intersection.begin()));
            common = intersection;
        }
        return vector<int>(common.begin(), common.end());
    }
};

void Test_Common_Elements_All_Rows() {
    Solution solution;

    vector<vector<vector<int>>> test_cases = {
        {{1,2,1,4,8},{3,7,8,5,1},{8,7,7,3,1},{8,1,2,7,9}},
        {{1,2,3},{4,5,6}},
        {{5,3,7},{5,7,3},{5,3,7}}
    };

    for (auto& mat : test_cases) {
        cout << "Matrix:" << endl;
        for (auto& row : mat) {
            for (int x : row) cout << x << " ";
            cout << endl;
        }

        auto r1 = solution.Common_Elements_Map_Optimal(mat);
        cout << "Map: ";
        for (int x : r1) cout << x << " ";
        cout << (r1.empty() ? "(none)" : "") << endl;

        auto r2 = solution.Common_Elements_Set_Intersection(mat);
        cout << "Set Intersection: ";
        for (int x : r2) cout << x << " ";
        cout << (r2.empty() ? "(none)" : "") << endl;

        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Common_Elements_All_Rows();
    return 0;
}
