/*
Problem: Minimize the Heights
URL: https://practice.geeksforgeeks.org/problems/minimize-the-heights3351/1

Problem Statement:
Given an array arr[] denoting heights of N towers and a positive integer K, for each tower
you must perform exactly one of: increase height by K or decrease height by K.
Find the minimum possible difference between the tallest and shortest towers.
Negative heights are not allowed.

Sample Input/Output:
Input: K = 2, arr = [1, 5, 8, 10]
Output: 5
Explanation: Modified array [3, 3, 6, 8]. Diff = 8 - 3 = 5.

Input: K = 3, arr = [3, 9, 12, 16, 20]
Output: 11
Explanation: Modified array [6, 12, 9, 13, 17]. Diff = 17 - 6 = 11.
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Minimize_Heights_Sorting_Optimal(vector<int> arr, int k) {
        /*
        Sorting + Greedy - Sort and try all split points
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        */
        int n = arr.size();
        if (n == 1) return 0;
        sort(arr.begin(), arr.end());
        int ans = arr[n - 1] - arr[0];
        for (int i = 1; i < n; i++) {
            if (arr[i] - k < 0) continue;
            int curr_min = min(arr[0] + k, arr[i] - k);
            int curr_max = max(arr[n - 1] - k, arr[i - 1] + k);
            ans = min(ans, curr_max - curr_min);
        }
        return ans;
    }
};

void Test_Minimize_The_Heights() {
    Solution solution;

    struct TestCase {
        vector<int> arr;
        int k;
        int expected;
    };

    vector<TestCase> test_cases = {
        {{1, 5, 8, 10}, 2, 5},
        {{3, 9, 12, 16, 20}, 3, 11},
        {{1}, 10, 0},
        {{1, 10, 14, 14, 14, 15}, 6, 5}
    };

    for (auto& tc : test_cases) {
        cout << "Array: ";
        for (int x : tc.arr) cout << x << " ";
        cout << ", K=" << tc.k << ", Expected=" << tc.expected << endl;

        cout << "Sorting+Greedy: " << solution.Minimize_Heights_Sorting_Optimal(tc.arr, tc.k) << endl;

        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Minimize_The_Heights();
    return 0;
}
