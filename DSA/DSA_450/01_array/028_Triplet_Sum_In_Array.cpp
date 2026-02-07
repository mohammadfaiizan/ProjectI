/*
Problem: Triplet Sum in Array
URL: https://practice.geeksforgeeks.org/problems/triplet-sum-in-array-1587115621/1

Problem Statement:
Given an array arr[] of distinct integers of size N and a value X, find if there is a
triplet in the array whose sum is equal to X.

Sample Input/Output:
Input: arr = [1, 4, 45, 6, 10, 8], X = 22
Output: true
Explanation: Triplet (4, 10, 8) has sum 22.

Input: arr = [1, 2, 4, 3, 6], X = 10
Output: true
Explanation: Triplet (1, 3, 6) has sum 10.
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    bool Triplet_Sum_Two_Pointer_Optimal(vector<int> arr, int x) {
        /*
        Sorting + Two Pointer - Fix one element, use two pointers for rest
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        */
        sort(arr.begin(), arr.end());
        int n = arr.size();
        for (int i = 0; i < n - 2; i++) {
            int left = i + 1, right = n - 1;
            while (left < right) {
                int sum = arr[i] + arr[left] + arr[right];
                if (sum == x) return true;
                else if (sum < x) left++;
                else right--;
            }
        }
        return false;
    }

    bool Triplet_Sum_Hashing(vector<int>& arr, int x) {
        /*
        Hashing Approach - Fix one element, use set for pair sum
        Time Complexity: O(n^2)
        Space Complexity: O(n)
        */
        int n = arr.size();
        for (int i = 0; i < n - 1; i++) {
            unordered_set<int> s;
            int target = x - arr[i];
            for (int j = i + 1; j < n; j++) {
                if (s.count(target - arr[j])) return true;
                s.insert(arr[j]);
            }
        }
        return false;
    }

    bool Triplet_Sum_Brute_Force(vector<int>& arr, int x) {
        /*
        Brute Force - Check all triplets
        Time Complexity: O(n^3)
        Space Complexity: O(1)
        */
        int n = arr.size();
        for (int i = 0; i < n - 2; i++)
            for (int j = i + 1; j < n - 1; j++)
                for (int k = j + 1; k < n; k++)
                    if (arr[i] + arr[j] + arr[k] == x) return true;
        return false;
    }
};

void Test_Triplet_Sum() {
    Solution solution;

    struct TestCase {
        vector<int> arr;
        int x;
        bool expected;
    };

    vector<TestCase> test_cases = {
        {{1, 4, 45, 6, 10, 8}, 22, true},
        {{1, 2, 4, 3, 6}, 10, true},
        {{1, 2, 4, 3, 6}, 20, false},
        {{1, 2, 3}, 6, true}
    };

    for (auto& tc : test_cases) {
        cout << "Array: ";
        for (int x : tc.arr) cout << x << " ";
        cout << ", X=" << tc.x << ", Expected: " << (tc.expected ? "true" : "false") << endl;

        cout << "Two Pointer: " << (solution.Triplet_Sum_Two_Pointer_Optimal(tc.arr, tc.x) ? "true" : "false") << endl;
        cout << "Hashing: " << (solution.Triplet_Sum_Hashing(tc.arr, tc.x) ? "true" : "false") << endl;
        cout << "Brute Force: " << (solution.Triplet_Sum_Brute_Force(tc.arr, tc.x) ? "true" : "false") << endl;

        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Triplet_Sum();
    return 0;
}
