/*
Problem: Maximum Sum Increasing Subsequence
URL: https://practice.geeksforgeeks.org/problems/maximum-sum-increasing-subsequence4749/1

Problem Statement:
Given an array of n positive integers. Find the sum of maximum sum increasing subsequence of the given array.

Sample Input/Output:
Input: [1, 101, 2, 3, 100, 4, 5]
Output: 106
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int MSIS_DP(vector<int>& arr, int n) {
        /*
        DP approach
        Time Complexity: O(n^2)
        Space Complexity: O(n)
        */
        vector<int> dp(n);
        for (int i = 0; i < n; i++) {
            dp[i] = arr[i];
        }
        for (int i = 1; i < n; i++) {
            for (int j = 0; j < i; j++) {
                if (arr[j] < arr[i]) {
                    dp[i] = max(dp[i], dp[j] + arr[i]);
                }
            }
        }
        return *max_element(dp.begin(), dp.end());
    }
};

void Test_MSIS() {
    Solution solution;
    vector<int> arr = {1, 101, 2, 3, 100, 4, 5};
    cout << "Max Sum: " << solution.MSIS_DP(arr, arr.size()) << endl;
}

int main() {
    Test_MSIS();
    return 0;
}
