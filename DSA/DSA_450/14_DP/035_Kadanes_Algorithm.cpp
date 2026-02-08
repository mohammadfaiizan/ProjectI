/*
Problem: Kadane's Algorithm
URL: https://practice.geeksforgeeks.org/problems/kadanes-algorithm-1587115620/1

Problem Statement:
Given an array Arr[] of N integers. Find the contiguous sub-array(containing at least one number) which has the maximum sum and return its sum.

Sample Input/Output:
Input: [-2,-3,4,-1,-2,1,5,-3]
Output: 7
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Kadane_Standard(vector<int>& arr) {
        /*
        Standard Kadane's Algorithm
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        int n = arr.size();
        int maxSum = arr[0];
        int currentSum = arr[0];
        
        for (int i = 1; i < n; i++) {
            currentSum = max(arr[i], currentSum + arr[i]);
            maxSum = max(maxSum, currentSum);
        }
        
        return maxSum;
    }
    
    int Kadane_DP(vector<int>& arr) {
        /*
        Dynamic Programming Approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        int n = arr.size();
        vector<int> dp(n);
        dp[0] = arr[0];
        int maxSum = dp[0];
        
        for (int i = 1; i < n; i++) {
            dp[i] = max(arr[i], dp[i - 1] + arr[i]);
            maxSum = max(maxSum, dp[i]);
        }
        
        return maxSum;
    }
};

void Test_Kadane_Standard() {
    Solution solution;
    vector<int> arr = {-2, -3, 4, -1, -2, 1, 5, -3};
    assert(solution.Kadane_Standard(arr) == 7);
}

void Test_Kadane_DP() {
    Solution solution;
    vector<int> arr = {-2, -3, 4, -1, -2, 1, 5, -3};
    assert(solution.Kadane_DP(arr) == 7);
}

int main() {
    Test_Kadane_Standard();
    Test_Kadane_DP();
    return 0;
}
