/*
Problem: Searching in Array with Adjacent Differ by at Most K
URL: https://www.geeksforgeeks.org/searching-array-adjacent-differ-k/

Problem Statement:
Given an array where each element is at most k positions away from its target position, search for an element x in the array.

Sample Input/Output:
Input: arr[] = {20, 40, 50, 70, 70, 60}, k = 20, x = 60
Output: 5

Input: arr[] = {20, 40, 50, 70, 70, 60}, k = 20, x = 10
Output: -1
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Search_Adjacent_Diff_K_Jump_Search(vector<int>& arr, int n, int x, int k) {
        /*
        Jump search based on difference - jump by max(1, abs(arr[i]-x)/k)
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        int i = 0;
        while (i < n) {
            if (arr[i] == x) {
                return i;
            }
            i = i + max(1, abs(arr[i] - x) / k);
        }
        return -1;
    }

    int Search_Adjacent_Diff_K_Linear(vector<int>& arr, int n, int x, int k) {
        /*
        Linear search through array
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        for (int i = 0; i < n; i++) {
            if (arr[i] == x) {
                return i;
            }
        }
        return -1;
    }
};

void Test_Search_Array_Adjacent_Diff_K() {
    Solution sol;
    vector<pair<vector<int>, pair<int, int>>> tests = {
        {{20, 40, 50, 70, 70, 60}, {20, 60}},
        {{20, 40, 50, 70, 70, 60}, {20, 10}},
        {{2, 4, 5, 7, 7, 6}, {2, 5}},
        {{10, 20, 30, 40, 50}, {10, 30}}
    };

    for (auto& test : tests) {
        vector<int> arr = test.first;
        int k = test.second.first;
        int x = test.second.second;
        int n = arr.size();
        
        cout << "Array: ";
        for (int num : arr) cout << num << " ";
        cout << ", k = " << k << ", x = " << x << endl;
        
        int res1 = sol.Search_Adjacent_Diff_K_Jump_Search(arr, n, x, k);
        cout << "Jump Search: " << res1 << endl;
        
        int res2 = sol.Search_Adjacent_Diff_K_Linear(arr, n, x, k);
        cout << "Linear: " << res2 << endl;
        
        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Search_Array_Adjacent_Diff_K();
    return 0;
}
