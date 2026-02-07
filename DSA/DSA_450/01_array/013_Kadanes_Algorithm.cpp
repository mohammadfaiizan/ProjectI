/*
Problem: Kadane's Algorithm
URL: https://practice.geeksforgeeks.org/problems/kadanes-algorithm-1587115620/1

Problem Statement:
Given an integer array arr of size N, find the maximum sum subarray using Kadane's
Algorithm. The subarray must contain at least one element. Also track the subarray indices.

Sample Input/Output:
Input: arr = [-2, -3, 4, -1, -2, 1, 5, -3]
Output: 7
Explanation: Subarray [4, -1, -2, 1, 5] has maximum sum 7.

Input: arr = [1, 2, 3, -2, 5]
Output: 9
Explanation: Subarray [1, 2, 3, -2, 5] has maximum sum 9.
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Kadane_Standard_Optimal(vector<int>& arr) {
        /*
        Standard Kadane's - Track max ending here and global max
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        int max_ending_here = arr[0], max_so_far = arr[0];
        for (int i = 1; i < (int)arr.size(); i++) {
            max_ending_here = max(arr[i], max_ending_here + arr[i]);
            max_so_far = max(max_so_far, max_ending_here);
        }
        return max_so_far;
    }

    pair<int, pair<int, int>> Kadane_With_Indices(vector<int>& arr) {
        /*
        Kadane's with Subarray Indices - Track start and end of max subarray
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        int max_ending_here = arr[0], max_so_far = arr[0];
        int start = 0, end = 0, temp_start = 0;
        for (int i = 1; i < (int)arr.size(); i++) {
            if (arr[i] > max_ending_here + arr[i]) {
                max_ending_here = arr[i];
                temp_start = i;
            } else {
                max_ending_here += arr[i];
            }
            if (max_ending_here > max_so_far) {
                max_so_far = max_ending_here;
                start = temp_start;
                end = i;
            }
        }
        return {max_so_far, {start, end}};
    }

    int Kadane_DP_Array(vector<int>& arr) {
        /*
        DP Array Variant - Explicit DP array for max sum ending at each index
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
};

void Test_Kadanes_Algorithm() {
    Solution solution;

    struct TestCase {
        vector<int> arr;
        int expected;
    };

    vector<TestCase> test_cases = {
        {{-2, -3, 4, -1, -2, 1, 5, -3}, 7},
        {{1, 2, 3, -2, 5}, 9},
        {{-1, -2, -3, -4}, -1},
        {{5, 4, -1, 7, 8}, 23}
    };

    for (auto& tc : test_cases) {
        cout << "Array: ";
        for (int x : tc.arr) cout << x << " ";
        cout << ", Expected: " << tc.expected << endl;

        cout << "Standard: " << solution.Kadane_Standard_Optimal(tc.arr) << endl;

        auto [sum, indices] = solution.Kadane_With_Indices(tc.arr);
        cout << "With Indices: Sum=" << sum
             << ", Range=[" << indices.first << ", " << indices.second << "]" << endl;

        cout << "DP Array: " << solution.Kadane_DP_Array(tc.arr) << endl;

        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Kadanes_Algorithm();
    return 0;
}
