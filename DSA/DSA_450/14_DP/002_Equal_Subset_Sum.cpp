/*
Problem: Equal Subset Sum Partition
URL: https://practice.geeksforgeeks.org/problems/subset-sum-problem2014/1

Problem Statement:
Given an array arr[] of size N, check if it can be partitioned into two parts such that the sum of elements in both parts is the same.

Sample Input/Output:
Input: [1, 5, 11, 5]
Output: true
Input: [1, 2, 3, 5]
Output: false
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    bool Equal_Subset_Sum_DP_Tabulation(vector<int>& arr, int n) {
        /*
        DP Tabulation approach
        Time Complexity: O(n*sum)
        Space Complexity: O(n*sum)
        */
        int sum = accumulate(arr.begin(), arr.end(), 0);
        if (sum % 2 != 0) return false;
        int target = sum / 2;
        vector<vector<bool>> dp(n+1, vector<bool>(target+1, false));
        for (int i = 0; i <= n; i++) dp[i][0] = true;
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= target; j++) {
                if (arr[i-1] > j) {
                    dp[i][j] = dp[i-1][j];
                } else {
                    dp[i][j] = dp[i-1][j] || dp[i-1][j-arr[i-1]];
                }
            }
        }
        return dp[n][target];
    }

    bool Equal_Subset_Sum_Space_Optimized(vector<int>& arr, int n) {
        /*
        Space optimized approach
        Time Complexity: O(n*sum)
        Space Complexity: O(sum)
        */
        int sum = accumulate(arr.begin(), arr.end(), 0);
        if (sum % 2 != 0) return false;
        int target = sum / 2;
        vector<bool> dp(target+1, false);
        dp[0] = true;
        for (int i = 0; i < n; i++) {
            for (int j = target; j >= arr[i]; j--) {
                dp[j] = dp[j] || dp[j-arr[i]];
            }
        }
        return dp[target];
    }
};

void Test_Equal_Subset_Sum() {
    Solution solution;
    vector<int> arr1 = {1, 5, 11, 5};
    vector<int> arr2 = {1, 2, 3, 5};
    
    cout << "Test 1 [1,5,11,5]: " << solution.Equal_Subset_Sum_DP_Tabulation(arr1, arr1.size()) << endl;
    cout << "Test 2 [1,2,3,5]: " << solution.Equal_Subset_Sum_DP_Tabulation(arr2, arr2.size()) << endl;
    cout << "Test 1 Space Optimized: " << solution.Equal_Subset_Sum_Space_Optimized(arr1, arr1.size()) << endl;
    cout << "Test 2 Space Optimized: " << solution.Equal_Subset_Sum_Space_Optimized(arr2, arr2.size()) << endl;
}

int main() {
    Test_Equal_Subset_Sum();
    return 0;
}
