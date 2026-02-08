/*
Problem: Longest Alternating Subsequence
URL: https://practice.geeksforgeeks.org/problems/longest-alternating-subsequence5951/1

Problem Statement:
Given an array of integers, find the longest alternating subsequence. A sequence is alternating if the elements alternate between increasing and decreasing.

Sample Input/Output:
Input: [1,5,4]
Output: 3
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int LAS_DP(vector<int>& arr) {
        /*
        Dynamic Programming
        Time Complexity: O(n^2)
        Space Complexity: O(n)
        */
        int n = arr.size();
        if (n <= 1) return n;
        
        vector<vector<int>> dp(n, vector<int>(2, 1));
        int result = 1;
        
        for (int i = 1; i < n; i++) {
            for (int j = 0; j < i; j++) {
                if (arr[j] < arr[i]) {
                    dp[i][0] = max(dp[i][0], dp[j][1] + 1);
                } else if (arr[j] > arr[i]) {
                    dp[i][1] = max(dp[i][1], dp[j][0] + 1);
                }
            }
            result = max(result, max(dp[i][0], dp[i][1]));
        }
        
        return result;
    }
    
    int LAS_Optimized(vector<int>& arr) {
        /*
        Optimized Approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        int n = arr.size();
        if (n <= 1) return n;
        
        int inc = 1, dec = 1;
        
        for (int i = 1; i < n; i++) {
            if (arr[i] > arr[i - 1]) {
                inc = dec + 1;
            } else if (arr[i] < arr[i - 1]) {
                dec = inc + 1;
            }
        }
        
        return max(inc, dec);
    }
};

void Test_LAS_DP() {
    Solution solution;
    vector<int> arr = {1, 5, 4};
    assert(solution.LAS_DP(arr) == 3);
}

void Test_LAS_Optimized() {
    Solution solution;
    vector<int> arr = {1, 5, 4};
    assert(solution.LAS_Optimized(arr) == 3);
}

int main() {
    Test_LAS_DP();
    Test_LAS_Optimized();
    return 0;
}
