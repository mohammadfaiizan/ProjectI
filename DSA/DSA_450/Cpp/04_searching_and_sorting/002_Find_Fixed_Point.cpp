/*
Problem: Value Equal to Index Value
URL: https://practice.geeksforgeeks.org/problems/value-equal-to-index-value1330/1

Problem Statement:
Given an array Arr of N positive integers. Your task is to find the elements whose value is equal to that of its index value (Consider 1-based indexing).

Sample Input/Output:
Input: N = 5, Arr[] = {15, 2, 45, 12, 7}
Output: 2

Input: N = 1, Arr[] = {1}
Output: 1
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<int> Find_Fixed_Point_Linear(vector<int>& arr, int n) {
        /*
        Linear search - check each element if arr[i] == i+1
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        vector<int> result;
        for (int i = 0; i < n; i++) {
            if (arr[i] == i + 1) {
                result.push_back(i + 1);
            }
        }
        return result;
    }

    int Find_Fixed_Point_Binary_Search(vector<int>& arr, int n) {
        /*
        Binary search for single fixed point in sorted array
        Time Complexity: O(log n)
        Space Complexity: O(1)
        */
        int left = 0, right = n - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (arr[mid] == mid + 1) {
                return mid + 1;
            } else if (arr[mid] < mid + 1) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return -1;
    }
};

void Test_Find_Fixed_Point() {
    Solution sol;
    vector<vector<int>> tests = {
        {15, 2, 45, 12, 7},
        {1},
        {1, 2, 3, 4, 5},
        {10, 20, 30, 40, 50},
        {-10, -5, 0, 3, 7}
    };

    for (auto& arr : tests) {
        int n = arr.size();
        cout << "Array: ";
        for (int num : arr) cout << num << " ";
        cout << endl;
        
        vector<int> res1 = sol.Find_Fixed_Point_Linear(arr, n);
        cout << "Linear: ";
        if (res1.empty()) {
            cout << "No fixed point found" << endl;
        } else {
            for (int val : res1) cout << val << " ";
            cout << endl;
        }
        
        int res2 = sol.Find_Fixed_Point_Binary_Search(arr, n);
        cout << "Binary Search: ";
        if (res2 == -1) {
            cout << "No fixed point found" << endl;
        } else {
            cout << res2 << endl;
        }
        
        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Find_Fixed_Point();
    return 0;
}
