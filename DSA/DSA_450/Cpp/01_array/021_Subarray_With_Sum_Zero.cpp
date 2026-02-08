/*
Problem: Subarray with Sum 0
URL: https://practice.geeksforgeeks.org/problems/subarray-with-0-sum-1587115621/1

Problem Statement:
Given an array of positive and negative numbers, find if there is a subarray
(of size at least one) with 0 sum.

Sample Input/Output:
Input: arr = [4, 2, -3, 1, 6]
Output: true
Explanation: Subarray [2, -3, 1] has sum 0.

Input: arr = [4, 2, 0, 1, 6]
Output: true
Explanation: Subarray [0] has sum 0.
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    bool Subarray_Sum_Zero_Hashing_Optimal(vector<int>& arr) {
        /*
        Prefix Sum + HashSet - Check for repeated prefix sums
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        unordered_set<int> prefix_sums;
        int sum = 0;
        for (int x : arr) {
            sum += x;
            if (sum == 0 || prefix_sums.count(sum)) return true;
            prefix_sums.insert(sum);
        }
        return false;
    }

    bool Subarray_Sum_Zero_Map(vector<int>& arr) {
        /*
        Prefix Sum + HashMap - Use map for prefix sum tracking
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        unordered_map<int, bool> sum_map;
        int sum = 0;
        for (int i = 0; i < (int)arr.size(); i++) {
            sum += arr[i];
            if (sum == 0 || sum_map[sum]) return true;
            sum_map[sum] = true;
        }
        return false;
    }

    bool Subarray_Sum_Zero_Brute_Force(vector<int>& arr) {
        /*
        Brute Force - Check all subarrays
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        */
        int n = arr.size();
        for (int i = 0; i < n; i++) {
            int sum = 0;
            for (int j = i; j < n; j++) {
                sum += arr[j];
                if (sum == 0) return true;
            }
        }
        return false;
    }
};

void Test_Subarray_With_Sum_Zero() {
    Solution solution;

    struct TestCase {
        vector<int> arr;
        bool expected;
    };

    vector<TestCase> test_cases = {
        {{4, 2, -3, 1, 6}, true},
        {{4, 2, 0, 1, 6}, true},
        {{-3, 2, 3, 1, 6}, false},
        {{1, -1}, true}
    };

    for (auto& tc : test_cases) {
        cout << "Array: ";
        for (int x : tc.arr) cout << x << " ";
        cout << ", Expected: " << (tc.expected ? "true" : "false") << endl;

        cout << "Hashing: " << (solution.Subarray_Sum_Zero_Hashing_Optimal(tc.arr) ? "true" : "false") << endl;
        cout << "Map: " << (solution.Subarray_Sum_Zero_Map(tc.arr) ? "true" : "false") << endl;
        cout << "Brute Force: " << (solution.Subarray_Sum_Zero_Brute_Force(tc.arr) ? "true" : "false") << endl;

        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Subarray_With_Sum_Zero();
    return 0;
}
