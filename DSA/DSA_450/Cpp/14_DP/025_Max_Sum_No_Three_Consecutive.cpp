/*
Problem: Maximum Sum with No Three Consecutive Elements
URL: https://www.geeksforgeeks.org/maximum-subsequence-sum-such-that-no-three-are-consecutive/

Problem Statement:
Given a sequence of positive numbers, find the maximum sum that can be formed which has no three consecutive elements present.

Sample Input/Output:
Input: [100, 1000, 100, 1000, 1]
Output: 2101
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Max_Sum_Three_DP(vector<int>& arr, int n) {
        /*
        DP approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        if (n == 0) return 0;
        if (n == 1) return arr[0];
        if (n == 2) return arr[0] + arr[1];
        vector<int> dp(n);
        dp[0] = arr[0];
        dp[1] = arr[0] + arr[1];
        dp[2] = max({arr[0] + arr[1], arr[0] + arr[2], arr[1] + arr[2]});
        for (int i = 3; i < n; i++) {
            dp[i] = max({dp[i-1], dp[i-2] + arr[i], dp[i-3] + arr[i-1] + arr[i]});
        }
        return dp[n-1];
    }
};

void Test_Max_Sum_Three() {
    Solution solution;
    vector<int> arr = {100, 1000, 100, 1000, 1};
    cout << "Max Sum: " << solution.Max_Sum_Three_DP(arr, arr.size()) << endl;
}

int main() {
    Test_Max_Sum_Three();
    return 0;
}
