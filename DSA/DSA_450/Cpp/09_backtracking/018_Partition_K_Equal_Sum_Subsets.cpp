/*
Problem: Partition K Equal Sum Subsets
URL: https://practice.geeksforgeeks.org/problems/partition-array-to-k-subsets/1

Problem Statement:
Check if an array can be partitioned into K subsets with equal sum.

Sample Input/Output:
Input: arr = [4, 3, 2, 3, 5, 2, 1], K = 4
Output: true
Explanation: Can be partitioned into 4 subsets: [5], [1,4], [2,3], [2,3]
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    bool Partition_K_Equal_Sum_Backtracking(vector<int>& arr, int k) {
        /*
        Backtracking with bucket sums
        Time Complexity: O(K^N)
        Space Complexity: O(N)
        */
        int total_sum = accumulate(arr.begin(), arr.end(), 0);
        if (total_sum % k != 0) return false;
        
        int target = total_sum / k;
        vector<int> subset_sums(k, 0);
        sort(arr.rbegin(), arr.rend());
        
        function<bool(int)> backtrack = [&](int idx) -> bool {
            if (idx == arr.size()) {
                for (int sum : subset_sums) {
                    if (sum != target) return false;
                }
                return true;
            }
            
            for (int i = 0; i < k; i++) {
                if (subset_sums[i] + arr[idx] <= target) {
                    subset_sums[i] += arr[idx];
                    if (backtrack(idx + 1)) return true;
                    subset_sums[i] -= arr[idx];
                }
                
                if (subset_sums[i] == 0) break;
            }
            
            return false;
        };
        
        return backtrack(0);
    }
    
    bool Partition_K_Equal_Sum_Bitmask_DP(vector<int>& arr, int k) {
        /*
        Bitmask DP
        Time Complexity: O(N * 2^N)
        Space Complexity: O(2^N)
        */
        int n = arr.size();
        int total_sum = accumulate(arr.begin(), arr.end(), 0);
        if (total_sum % k != 0) return false;
        
        int target = total_sum / k;
        vector<bool> dp(1 << n, false);
        vector<int> sum(1 << n, 0);
        dp[0] = true;
        
        for (int mask = 0; mask < (1 << n); mask++) {
            if (!dp[mask]) continue;
            
            for (int i = 0; i < n; i++) {
                if (mask & (1 << i)) continue;
                
                int new_mask = mask | (1 << i);
                int new_sum = sum[mask] + arr[i];
                
                if (new_sum <= target) {
                    sum[new_mask] = new_sum % target;
                    dp[new_mask] = true;
                }
            }
        }
        
        return dp[(1 << n) - 1];
    }
};

void Test_Partition_K_Equal_Sum_Subsets() {
    Solution solution;
    vector<int> arr = {4, 3, 2, 3, 5, 2, 1};
    int k = 4;
    bool result1 = solution.Partition_K_Equal_Sum_Backtracking(arr, k);
    bool result2 = solution.Partition_K_Equal_Sum_Bitmask_DP(arr, k);
    cout << "Backtracking Approach: " << (result1 ? "true" : "false") << endl;
    cout << "Bitmask DP Approach: " << (result2 ? "true" : "false") << endl;
}

int main() {
    Test_Partition_K_Equal_Sum_Subsets();
    return 0;
}
