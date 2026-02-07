/*
Problem: Maximum and Minimum in Array
URL: https://www.geeksforgeeks.org/maximum-and-minimum-in-an-array/

Problem Statement:
Given an array of size N, find the maximum and minimum elements in the array.

Sample Input/Output:
Input: arr = [1000, 11, 445, 1, 330, 3000]
Output: Min = 1, Max = 3000

Input: arr = [3, 5, 4, 1, 9]
Output: Min = 1, Max = 9
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    pair<int, int> Max_Min_Linear_Scan_Optimal(vector<int>& arr) {
        /*
        Linear Scan - Single pass tracking min and max
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        int mn = arr[0], mx = arr[0];
        for (int i = 1; i < (int)arr.size(); i++) {
            if (arr[i] > mx) mx = arr[i];
            if (arr[i] < mn) mn = arr[i];
        }
        return {mn, mx};
    }

    pair<int, int> Max_Min_Pairs_Comparison(vector<int>& arr) {
        /*
        Pairs Comparison - Compare elements in pairs to reduce comparisons
        Time Complexity: O(n) - ~1.5n comparisons
        Space Complexity: O(1)
        */
        int n = arr.size();
        int mn, mx, i;
        if (n % 2 == 0) {
            mn = min(arr[0], arr[1]);
            mx = max(arr[0], arr[1]);
            i = 2;
        } else {
            mn = mx = arr[0];
            i = 1;
        }
        while (i < n - 1) {
            if (arr[i] < arr[i + 1]) {
                mn = min(mn, arr[i]);
                mx = max(mx, arr[i + 1]);
            } else {
                mn = min(mn, arr[i + 1]);
                mx = max(mx, arr[i]);
            }
            i += 2;
        }
        return {mn, mx};
    }

    pair<int, int> Max_Min_STL(vector<int>& arr) {
        /*
        STL Approach - Using minmax_element
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        auto result = minmax_element(arr.begin(), arr.end());
        return {*result.first, *result.second};
    }
};

void Test_Max_Min_Array() {
    Solution solution;

    vector<vector<int>> test_cases = {
        {1000, 11, 445, 1, 330, 3000},
        {3, 5, 4, 1, 9},
        {7},
        {2, 8}
    };

    for (auto& arr : test_cases) {
        cout << "Array: ";
        for (int x : arr) cout << x << " ";
        cout << endl;

        auto [mn1, mx1] = solution.Max_Min_Linear_Scan_Optimal(arr);
        cout << "Linear Scan: Min=" << mn1 << " Max=" << mx1 << endl;

        auto [mn2, mx2] = solution.Max_Min_Pairs_Comparison(arr);
        cout << "Pairs: Min=" << mn2 << " Max=" << mx2 << endl;

        auto [mn3, mx3] = solution.Max_Min_STL(arr);
        cout << "STL: Min=" << mn3 << " Max=" << mx3 << endl;

        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Max_Min_Array();
    return 0;
}
