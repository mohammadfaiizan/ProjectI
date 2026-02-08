/*
Problem: First and Last Occurrences of X
URL: https://practice.geeksforgeeks.org/problems/first-and-last-occurrences-of-x3116/1

Problem Statement:
Given a sorted array arr containing n elements, possibly with duplicates, find the first and last occurrences of an element x in the given array.

Sample Input/Output:
Input: n = 9, x = 5, arr[] = {1, 3, 5, 5, 5, 5, 67, 123, 125}
Output: 2 5

Input: n = 9, x = 7, arr[] = {1, 3, 5, 5, 5, 5, 7, 123, 125}
Output: 6 6
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<int> First_Last_Linear(vector<int>& arr, int n, int x) {
        /*
        Linear search - find first and last occurrence by scanning array
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        int first = -1, last = -1;
        for (int i = 0; i < n; i++) {
            if (arr[i] == x) {
                if (first == -1) first = i;
                last = i;
            }
        }
        return {first, last};
    }

    vector<int> First_Last_Binary_Two_Passes(vector<int>& arr, int n, int x) {
        /*
        Binary search with two passes - find first occurrence, then last occurrence
        Time Complexity: O(log n)
        Space Complexity: O(1)
        */
        int first = -1, last = -1;
        
        int left = 0, right = n - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (arr[mid] == x) {
                first = mid;
                right = mid - 1;
            } else if (arr[mid] < x) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        
        left = 0, right = n - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (arr[mid] == x) {
                last = mid;
                left = mid + 1;
            } else if (arr[mid] < x) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        
        return {first, last};
    }

    vector<int> First_Last_Binary_Boundary(vector<int>& arr, int n, int x) {
        /*
        Binary search checking boundary conditions
        Time Complexity: O(log n)
        Space Complexity: O(1)
        */
        int first = -1, last = -1;
        
        int left = 0, right = n - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (arr[mid] == x) {
                if (mid == 0 || arr[mid - 1] != x) {
                    first = mid;
                    break;
                }
                right = mid - 1;
            } else if (arr[mid] < x) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        
        left = 0, right = n - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (arr[mid] == x) {
                if (mid == n - 1 || arr[mid + 1] != x) {
                    last = mid;
                    break;
                }
                left = mid + 1;
            } else if (arr[mid] < x) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        
        return {first, last};
    }
};

void Test_First_Last_Position() {
    Solution sol;
    vector<pair<vector<int>, int>> tests = {
        {{1, 3, 5, 5, 5, 5, 67, 123, 125}, 5},
        {{1, 3, 5, 5, 5, 5, 7, 123, 125}, 7},
        {{1, 2, 3, 4, 5}, 3},
        {{1, 2, 3, 4, 5}, 6},
        {{5, 5, 5, 5, 5}, 5}
    };

    for (auto& test : tests) {
        vector<int> arr = test.first;
        int x = test.second;
        int n = arr.size();
        
        cout << "Array: ";
        for (int num : arr) cout << num << " ";
        cout << ", x = " << x << endl;
        
        vector<int> res1 = sol.First_Last_Linear(arr, n, x);
        cout << "Linear: First = " << res1[0] << ", Last = " << res1[1] << endl;
        
        vector<int> res2 = sol.First_Last_Binary_Two_Passes(arr, n, x);
        cout << "Binary Two Passes: First = " << res2[0] << ", Last = " << res2[1] << endl;
        
        vector<int> res3 = sol.First_Last_Binary_Boundary(arr, n, x);
        cout << "Binary Boundary: First = " << res3[0] << ", Last = " << res3[1] << endl;
        
        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_First_Last_Position();
    return 0;
}
