/*
Problem: Kth Smallest Element in a Row-Column Sorted Matrix
URL: https://practice.geeksforgeeks.org/problems/kth-element-in-matrix/1

Problem Statement:
Given an N x N matrix where every row and column is sorted in non-decreasing order,
find the kth smallest element in the matrix.

Sample Input/Output:
Input: matrix = [[10, 20, 30, 40],
                 [15, 25, 35, 45],
                 [25, 29, 37, 48],
                 [32, 33, 39, 50]], K = 7
Output: 30
Explanation: Sorted elements: 10,15,20,25,25,29,30,... 7th is 30.

Input: matrix = [[16, 28, 60, 64],
                 [22, 41, 63, 91],
                 [27, 50, 87, 93],
                 [36, 78, 87, 94]], K = 3
Output: 27
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Kth_Smallest_Min_Heap_Optimal(vector<vector<int>>& mat, int k) {
        /*
        Min Heap - Push first row/col, extract min k times
        Time Complexity: O(n + k * log n)
        Space Complexity: O(n)
        */
        int n = mat.size();
        auto cmp = [&](pair<int, int>& a, pair<int, int>& b) {
            return mat[a.first][a.second] > mat[b.first][b.second];
        };
        priority_queue<pair<int, int>, vector<pair<int, int>>, decltype(cmp)> pq(cmp);
        for (int i = 0; i < n; i++) pq.push({i, 0});
        for (int i = 1; i < k; i++) {
            auto [r, c] = pq.top();
            pq.pop();
            if (c + 1 < n) pq.push({r, c + 1});
        }
        return mat[pq.top().first][pq.top().second];
    }

    int Kth_Smallest_Binary_Search(vector<vector<int>>& mat, int k) {
        /*
        Binary Search on Value - Count elements less than or equal to mid
        Time Complexity: O(n * log(max - min))
        Space Complexity: O(1)
        */
        int n = mat.size();
        int lo = mat[0][0], hi = mat[n - 1][n - 1];
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            int count = Count_Less_Equal(mat, mid, n);
            if (count < k) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }

    int Kth_Smallest_Flatten(vector<vector<int>>& mat, int k) {
        /*
        Flatten and Sort - Put all elements in array and sort
        Time Complexity: O(n^2 * log(n^2))
        Space Complexity: O(n^2)
        */
        vector<int> all;
        for (auto& row : mat)
            for (int x : row) all.push_back(x);
        sort(all.begin(), all.end());
        return all[k - 1];
    }

private:
    int Count_Less_Equal(vector<vector<int>>& mat, int mid, int n) {
        int count = 0;
        int j = n - 1;
        for (int i = 0; i < n; i++) {
            while (j >= 0 && mat[i][j] > mid) j--;
            count += j + 1;
        }
        return count;
    }
};

void Test_Kth_Smallest_Matrix() {
    Solution solution;

    struct TestCase {
        vector<vector<int>> mat;
        int k;
        int expected;
    };

    vector<TestCase> test_cases = {
        {{{10,20,30,40},{15,25,35,45},{25,29,37,48},{32,33,39,50}}, 7, 30},
        {{{16,28,60,64},{22,41,63,91},{27,50,87,93},{36,78,87,94}}, 3, 27},
        {{{1,5,9},{10,11,13},{12,13,15}}, 8, 13}
    };

    for (auto& tc : test_cases) {
        cout << "K=" << tc.k << ", Expected=" << tc.expected << endl;

        cout << "Min Heap: " << solution.Kth_Smallest_Min_Heap_Optimal(tc.mat, tc.k) << endl;
        cout << "Binary Search: " << solution.Kth_Smallest_Binary_Search(tc.mat, tc.k) << endl;
        cout << "Flatten: " << solution.Kth_Smallest_Flatten(tc.mat, tc.k) << endl;

        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Kth_Smallest_Matrix();
    return 0;
}
