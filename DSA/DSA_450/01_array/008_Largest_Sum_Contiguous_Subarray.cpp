/*
Problem: Largest Sum Contiguous Subarray
URL: https://practice.geeksforgeeks.org/problems/kadanes-algorithm-1587115620/1

Problem Statement:
Given an array arr[] of N integers, find the contiguous sub-array (containing at least
one number) which has the maximum sum and return its sum.

Sample Input/Output:
Input: arr = [1, 2, 3, -2, 5]
Output: 9
Explanation: Max subarray sum is 1 + 2 + 3 + (-2) + 5 = 9.

Input: arr = [-1, -2, -3, -4]
Output: -1
Explanation: Max subarray sum is -1 (single element).
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Max_Subarray_Kadane_Optimal(vector<int>& arr) {
        /*
        Kadane's Algorithm - Track current sum and global max
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        int current_sum = arr[0], max_sum = arr[0];
        for (int i = 1; i < (int)arr.size(); i++) {
            current_sum = max(arr[i], current_sum + arr[i]);
            max_sum = max(max_sum, current_sum);
        }
        return max_sum;
    }

    int Max_Subarray_DP(vector<int>& arr) {
        /*
        DP Array - Store max subarray sum ending at each index
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        int n = arr.size();
        vector<int> dp(n);
        dp[0] = arr[0];
        int max_sum = dp[0];
        for (int i = 1; i < n; i++) {
            dp[i] = max(arr[i], dp[i - 1] + arr[i]);
            max_sum = max(max_sum, dp[i]);
        }
        return max_sum;
    }

    int Max_Subarray_Brute_Force(vector<int>& arr) {
        /*
        Brute Force - Check all subarrays
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        */
        int n = arr.size();
        int max_sum = INT_MIN;
        for (int i = 0; i < n; i++) {
            int current_sum = 0;
            for (int j = i; j < n; j++) {
                current_sum += arr[j];
                max_sum = max(max_sum, current_sum);
            }
        }
        return max_sum;
    }
};

void Test_Largest_Sum_Contiguous_Subarray() {
    Solution solution;

    struct TestCase {
        vector<int> arr;
        int expected;
    };

    vector<TestCase> test_cases = {
        {{1, 2, 3, -2, 5}, 9},
        {{-1, -2, -3, -4}, -1},
        {{-2, -3, 4, -1, -2, 1, 5, -3}, 7},
        {{5, 4, -1, 7, 8}, 23}
    };

    for (auto& tc : test_cases) {
        cout << "Array: ";
        for (int x : tc.arr) cout << x << " ";
        cout << ", Expected: " << tc.expected << endl;

        cout << "Kadane's: " << solution.Max_Subarray_Kadane_Optimal(tc.arr) << endl;
        cout << "DP: " << solution.Max_Subarray_DP(tc.arr) << endl;
        cout << "Brute Force: " << solution.Max_Subarray_Brute_Force(tc.arr) << endl;

        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Largest_Sum_Contiguous_Subarray();
    return 0;
}
