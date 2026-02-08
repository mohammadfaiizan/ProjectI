/*
Problem: Maximum Sum with No Two Adjacent Elements
URL: https://practice.geeksforgeeks.org/problems/stickler-theif-1587115621/1

Problem Statement:
Find the maximum sum such that no two elements are adjacent.

Sample Input/Output:
Input: [5, 5, 10, 100, 10, 5]
Output: 110
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Max_Sum_DP(vector<int>& arr, int n) {
        /*
        DP approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        if (n == 0) return 0;
        if (n == 1) return arr[0];
        vector<int> dp(n);
        dp[0] = arr[0];
        dp[1] = max(arr[0], arr[1]);
        for (int i = 2; i < n; i++) {
            dp[i] = max(dp[i-1], dp[i-2] + arr[i]);
        }
        return dp[n-1];
    }

    int Max_Sum_Space(vector<int>& arr, int n) {
        /*
        Space optimized approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        if (n == 0) return 0;
        if (n == 1) return arr[0];
        int prev2 = arr[0];
        int prev1 = max(arr[0], arr[1]);
        for (int i = 2; i < n; i++) {
            int curr = max(prev1, prev2 + arr[i]);
            prev2 = prev1;
            prev1 = curr;
        }
        return prev1;
    }
};

void Test_Max_Sum() {
    Solution solution;
    vector<int> arr = {5, 5, 10, 100, 10, 5};
    cout << "DP: " << solution.Max_Sum_DP(arr, arr.size()) << endl;
    cout << "Space Optimized: " << solution.Max_Sum_Space(arr, arr.size()) << endl;
}

int main() {
    Test_Max_Sum();
    return 0;
}
