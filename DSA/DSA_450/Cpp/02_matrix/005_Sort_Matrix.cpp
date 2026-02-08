/*
Problem: Sort the Given Matrix
URL: https://practice.geeksforgeeks.org/problems/sorted-matrix2333/1

Problem Statement:
Given an N x N matrix, sort all elements of the matrix in increasing order
and put them back into the matrix in row-wise fashion.

Sample Input/Output:
Input: matrix = [[10, 20, 30, 40],
                 [15, 25, 35, 45],
                 [27, 29, 37, 48],
                 [32, 33, 39, 50]]
Output: [[10, 15, 20, 25],
         [27, 29, 30, 32],
         [33, 35, 37, 39],
         [40, 45, 48, 50]]

Input: matrix = [[5, 4], [3, 1]]
Output: [[1, 3], [4, 5]]
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<vector<int>> Sort_Matrix_Flatten_Optimal(vector<vector<int>> mat) {
        /*
        Flatten, Sort, Refill - Extract all elements, sort, put back
        Time Complexity: O(n^2 * log(n^2))
        Space Complexity: O(n^2)
        */
        int n = mat.size();
        vector<int> temp;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                temp.push_back(mat[i][j]);
        sort(temp.begin(), temp.end());
        int k = 0;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                mat[i][j] = temp[k++];
        return mat;
    }

    vector<vector<int>> Sort_Matrix_Min_Heap(vector<vector<int>> mat) {
        /*
        Min Heap - Use priority queue for sorted extraction
        Time Complexity: O(n^2 * log(n^2))
        Space Complexity: O(n^2)
        */
        int n = mat.size();
        priority_queue<int, vector<int>, greater<int>> pq;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                pq.push(mat[i][j]);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                mat[i][j] = pq.top();
                pq.pop();
            }
        }
        return mat;
    }
};

void Test_Sort_Matrix() {
    Solution solution;

    vector<vector<vector<int>>> test_cases = {
        {{10,20,30,40},{15,25,35,45},{27,29,37,48},{32,33,39,50}},
        {{5,4},{3,1}},
        {{9,8,7},{6,5,4},{3,2,1}}
    };

    for (auto& mat : test_cases) {
        cout << "Original:" << endl;
        for (auto& row : mat) {
            for (int x : row) cout << x << "\t";
            cout << endl;
        }

        auto r1 = solution.Sort_Matrix_Flatten_Optimal(mat);
        cout << "Flatten Sort:" << endl;
        for (auto& row : r1) {
            for (int x : row) cout << x << "\t";
            cout << endl;
        }

        auto r2 = solution.Sort_Matrix_Min_Heap(mat);
        cout << "Min Heap:" << endl;
        for (auto& row : r2) {
            for (int x : row) cout << x << "\t";
            cout << endl;
        }

        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Sort_Matrix();
    return 0;
}
