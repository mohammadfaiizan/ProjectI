/*
Problem: Subset Sum
URL: https://practice.geeksforgeeks.org/problems/subset-sum-problem2014/1

Problem Statement:
Given array and target sum, check if array can be partitioned into two subsets with equal sum (uses subset sum backtracking).

Sample Input/Output:
Input: arr[]={1,5,11,5}
Output: true
Explanation: Partition {1,5,5} and {11} have equal sum
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    bool Partition_Equal_Subset_Sum_Backtracking(vector<int> &arr) {
        /*
        Backtracking include/exclude
        Time Complexity: O(2^n)
        Space Complexity: O(n)
        */
        int total_sum = accumulate(arr.begin(), arr.end(), 0);
        if (total_sum % 2 != 0) return false;
        
        int target = total_sum / 2;
        
        function<bool(int, int)> backtrack = [&](int index, int current_sum) {
            if (current_sum == target) {
                return true;
            }
            if (index >= arr.size() || current_sum > target) {
                return false;
            }
            
            return backtrack(index + 1, current_sum + arr[index]) ||
                   backtrack(index + 1, current_sum);
        };
        
        return backtrack(0, 0);
    }
    
    bool Partition_Equal_Subset_Sum_DP(vector<int> &arr) {
        /*
        DP
        Time Complexity: O(n*sum)
        Space Complexity: O(n*sum)
        */
        int total_sum = accumulate(arr.begin(), arr.end(), 0);
        if (total_sum % 2 != 0) return false;
        
        int target = total_sum / 2;
        int n = arr.size();
        vector<vector<bool>> dp(n + 1, vector<bool>(target + 1, false));
        
        for (int i = 0; i <= n; i++) {
            dp[i][0] = true;
        }
        
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= target; j++) {
                if (arr[i-1] > j) {
                    dp[i][j] = dp[i-1][j];
                } else {
                    dp[i][j] = dp[i-1][j] || dp[i-1][j - arr[i-1]];
                }
            }
        }
        
        return dp[n][target];
    }
};

void Test_Subset_Sum() {
    Solution solution;
    
    vector<int> arr = {1, 5, 11, 5};
    
    bool result1 = solution.Partition_Equal_Subset_Sum_Backtracking(arr);
    cout << "Backtracking result: " << (result1 ? "true" : "false") << endl;
    
    bool result2 = solution.Partition_Equal_Subset_Sum_DP(arr);
    cout << "DP result: " << (result2 ? "true" : "false") << endl;
}

int main() {
    Test_Subset_Sum();
    return 0;
}
