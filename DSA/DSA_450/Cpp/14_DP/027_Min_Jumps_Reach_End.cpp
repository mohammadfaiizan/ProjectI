/*
Problem: Minimum Jumps to Reach End
URL: https://practice.geeksforgeeks.org/problems/minimum-number-of-jumps-1587115620/1

Problem Statement:
Given an array of integers where each element represents the max number of steps that can be made forward from that element. Find the minimum number of jumps to reach the end of the array (starting from the first element). If an element is 0, then you cannot move through that element.

Sample Input/Output:
Input: [2, 3, 1, 1, 4]
Output: 2
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Min_Jumps_Greedy(vector<int>& arr, int n) {
        /*
        Greedy approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        if (n <= 1) return 0;
        if (arr[0] == 0) return -1;
        int maxReach = arr[0];
        int steps = arr[0];
        int jumps = 1;
        for (int i = 1; i < n; i++) {
            if (i == n-1) return jumps;
            maxReach = max(maxReach, i + arr[i]);
            steps--;
            if (steps == 0) {
                jumps++;
                if (i >= maxReach) return -1;
                steps = maxReach - i;
            }
        }
        return -1;
    }

    int Min_Jumps_DP(vector<int>& arr, int n) {
        /*
        DP approach
        Time Complexity: O(n^2)
        Space Complexity: O(n)
        */
        if (n <= 1) return 0;
        if (arr[0] == 0) return -1;
        vector<int> dp(n, INT_MAX);
        dp[0] = 0;
        for (int i = 0; i < n; i++) {
            if (dp[i] == INT_MAX) continue;
            for (int j = i+1; j <= i+arr[i] && j < n; j++) {
                dp[j] = min(dp[j], dp[i] + 1);
            }
        }
        return (dp[n-1] == INT_MAX) ? -1 : dp[n-1];
    }
};

void Test_Min_Jumps() {
    Solution solution;
    vector<int> arr = {2, 3, 1, 1, 4};
    cout << "Greedy: " << solution.Min_Jumps_Greedy(arr, arr.size()) << endl;
    cout << "DP: " << solution.Min_Jumps_DP(arr, arr.size()) << endl;
}

int main() {
    Test_Min_Jumps();
    return 0;
}
