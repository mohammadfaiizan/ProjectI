/*
Problem: Partition Problem
URL: https://practice.geeksforgeeks.org/problems/subset-sum-problem2014/1

Problem Statement:
Given an array arr[] of size N, check if it can be partitioned into two parts such that the sum of elements in both parts is the same.

Sample Input/Output:
Input: [1,5,11,5]
Output: true
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    bool Partition_DP(vector<int>& arr) {
        /*
        Dynamic Programming
        Time Complexity: O(n*sum)
        Space Complexity: O(n*sum)
        */
        int n = arr.size();
        int sum = accumulate(arr.begin(), arr.end(), 0);
        
        if (sum % 2 != 0) return false;
        
        int target = sum / 2;
        vector<vector<bool>> dp(n + 1, vector<bool>(target + 1, false));
        
        for (int i = 0; i <= n; i++) {
            dp[i][0] = true;
        }
        
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= target; j++) {
                if (arr[i - 1] > j) {
                    dp[i][j] = dp[i - 1][j];
                } else {
                    dp[i][j] = dp[i - 1][j] || dp[i - 1][j - arr[i - 1]];
                }
            }
        }
        
        return dp[n][target];
    }
    
    bool Partition_Space(vector<int>& arr) {
        /*
        Space Optimized
        Time Complexity: O(n*sum)
        Space Complexity: O(sum)
        */
        int n = arr.size();
        int sum = accumulate(arr.begin(), arr.end(), 0);
        
        if (sum % 2 != 0) return false;
        
        int target = sum / 2;
        vector<bool> dp(target + 1, false);
        dp[0] = true;
        
        for (int i = 0; i < n; i++) {
            for (int j = target; j >= arr[i]; j--) {
                dp[j] = dp[j] || dp[j - arr[i]];
            }
        }
        
        return dp[target];
    }
};

void Test_Partition_DP() {
    Solution solution;
    vector<int> arr1 = {1, 5, 11, 5};
    assert(solution.Partition_DP(arr1) == true);
}

void Test_Partition_Space() {
    Solution solution;
    vector<int> arr1 = {1, 5, 11, 5};
    assert(solution.Partition_Space(arr1) == true);
}

int main() {
    Test_Partition_DP();
    Test_Partition_Space();
    return 0;
}
