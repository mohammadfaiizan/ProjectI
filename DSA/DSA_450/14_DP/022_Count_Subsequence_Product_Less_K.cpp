/*
Problem: Count Subsequences with Product Less Than K
URL: https://www.geeksforgeeks.org/count-subsequences-product-less-k/

Problem Statement:
Given a non-negative array, find the number of subsequences having product smaller than K.

Sample Input/Output:
Input: [1, 2, 3, 4], k = 10
Output: 11
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Count_Subseq_DP(vector<int>& arr, int n, int k) {
        /*
        DP approach
        Time Complexity: O(n*k)
        Space Complexity: O(n*k)
        */
        vector<vector<int>> dp(n+1, vector<int>(k+1, 0));
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= k; j++) {
                dp[i][j] = dp[i-1][j];
                if (arr[i-1] <= j && arr[i-1] > 0) {
                    dp[i][j] += dp[i-1][j/arr[i-1]] + 1;
                }
            }
        }
        return dp[n][k];
    }
};

void Test_Count_Subseq() {
    Solution solution;
    vector<int> arr = {1, 2, 3, 4};
    int k = 10;
    cout << "Count: " << solution.Count_Subseq_DP(arr, arr.size(), k) << endl;
}

int main() {
    Test_Count_Subseq();
    return 0;
}
