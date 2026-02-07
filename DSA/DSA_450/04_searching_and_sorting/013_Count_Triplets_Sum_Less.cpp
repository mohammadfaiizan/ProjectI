/*
Problem: Count Triplets with Sum Smaller than X
URL: https://practice.geeksforgeeks.org/problems/count-triplets-with-sum-smaller-than-x5549/1

Problem Statement:
Given an array arr[] of distinct integers of size n and a value X, find the count of triplets whose sum is smaller than X.

Sample Input/Output:
Input: n = 4, X = 2, arr[] = {-2, 0, 1, 3}
Output: 2

Input: n = 5, X = 12, arr[] = {5, 1, 3, 4, 7}
Output: 4
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    long long Count_Triplets_Sorting_Two_Pointer(vector<int>& arr, int n, int X) {
        /*
        Sort array and use two pointers to count triplets with sum less than X
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        */
        sort(arr.begin(), arr.end());
        long long count = 0;
        
        for (int i = 0; i < n - 2; i++) {
            int left = i + 1, right = n - 1;
            while (left < right) {
                int sum = arr[i] + arr[left] + arr[right];
                if (sum < X) {
                    count += (right - left);
                    left++;
                } else {
                    right--;
                }
            }
        }
        
        return count;
    }

    long long Count_Triplets_Brute_Force(vector<int>& arr, int n, int X) {
        /*
        Check all possible triplets using three nested loops
        Time Complexity: O(n^3)
        Space Complexity: O(1)
        */
        long long count = 0;
        
        for (int i = 0; i < n - 2; i++) {
            for (int j = i + 1; j < n - 1; j++) {
                for (int k = j + 1; k < n; k++) {
                    if (arr[i] + arr[j] + arr[k] < X) {
                        count++;
                    }
                }
            }
        }
        
        return count;
    }
};

void Test_Count_Triplets_Sum_Less() {
    Solution sol;
    vector<pair<vector<int>, int>> tests = {
        {{-2, 0, 1, 3}, 2},
        {{5, 1, 3, 4, 7}, 12},
        {{-1, 0, 1, 2}, 2},
        {{1, 2, 3, 4, 5}, 10}
    };

    for (auto& test : tests) {
        vector<int> arr = test.first;
        int X = test.second;
        int n = arr.size();
        
        cout << "Array: ";
        for (int num : arr) cout << num << " ";
        cout << ", X = " << X << endl;
        
        vector<int> arr1 = arr, arr2 = arr;
        long long res1 = sol.Count_Triplets_Sorting_Two_Pointer(arr1, n, X);
        long long res2 = sol.Count_Triplets_Brute_Force(arr2, n, X);
        
        cout << "Sorting + Two Pointer: " << res1 << endl;
        cout << "Brute Force: " << res2 << endl;
        
        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Count_Triplets_Sum_Less();
    return 0;
}
