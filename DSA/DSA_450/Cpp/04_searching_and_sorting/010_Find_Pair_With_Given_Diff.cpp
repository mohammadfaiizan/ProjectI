/*
Problem: Find Pair Given Difference
URL: https://practice.geeksforgeeks.org/problems/find-pair-given-difference1559/1

Problem Statement:
Given an unsorted array arr[] of size n and an integer diff, find if there exists a pair of elements in the array whose difference is diff.

Sample Input/Output:
Input: n = 6, diff = 78, arr[] = {5, 20, 3, 2, 5, 80}
Output: 1

Input: n = 5, diff = 45, arr[] = {90, 70, 20, 80, 50}
Output: -1
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    bool Find_Pair_Sorting_Two_Pointer(vector<int>& arr, int n, int diff) {
        /*
        Sort array and use two pointers to find pair with given difference
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        */
        sort(arr.begin(), arr.end());
        int i = 0, j = 1;
        while (i < n && j < n) {
            if (i != j && arr[j] - arr[i] == diff) {
                return true;
            } else if (arr[j] - arr[i] < diff) {
                j++;
            } else {
                i++;
            }
        }
        return false;
    }

    bool Find_Pair_HashSet(vector<int>& arr, int n, int diff) {
        /*
        Use hash set to store elements and check if arr[i] + diff or arr[i] - diff exists
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        unordered_set<int> seen;
        for (int i = 0; i < n; i++) {
            if (seen.find(arr[i] + diff) != seen.end() || seen.find(arr[i] - diff) != seen.end()) {
                return true;
            }
            seen.insert(arr[i]);
        }
        return false;
    }
};

void Test_Find_Pair_With_Given_Diff() {
    Solution sol;
    vector<pair<vector<int>, int>> tests = {
        {{5, 20, 3, 2, 5, 80}, 78},
        {{90, 70, 20, 80, 50}, 45},
        {{1, 8, 30, 40, 100}, 60},
        {{10, 20, 30}, 10},
        {{1, 2, 3, 4, 5}, 0}
    };

    for (auto& test : tests) {
        vector<int> arr = test.first;
        int diff = test.second;
        int n = arr.size();
        
        cout << "Array: ";
        for (int num : arr) cout << num << " ";
        cout << ", diff = " << diff << endl;
        
        vector<int> arr1 = arr, arr2 = arr;
        bool res1 = sol.Find_Pair_Sorting_Two_Pointer(arr1, n, diff);
        bool res2 = sol.Find_Pair_HashSet(arr2, n, diff);
        
        cout << "Sorting + Two Pointer: " << (res1 ? "Found" : "Not Found") << endl;
        cout << "HashSet: " << (res2 ? "Found" : "Not Found") << endl;
        
        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Find_Pair_With_Given_Diff();
    return 0;
}
