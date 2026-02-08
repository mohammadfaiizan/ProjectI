/*
Problem: Friends Pairing Problem
URL: https://practice.geeksforgeeks.org/problems/friends-pairing-problem5425/1

Problem Statement:
Given n friends, each one can remain single or can be paired up with some other friend. Each friend can be paired only once. Find out the total number of ways in which friends can remain single or can be paired up.

Sample Input/Output:
Input: n = 4
Output: 10
Explanation: {1}, {2}, {3}, {4}, {1,2}, {3,4}, {1,3}, {2,4}, {1,4}, {2,3}, {1,2}, {3}, {4}, {1,3}, {2}, {4}, {1,4}, {2}, {3}, {2,3}, {1}, {4}, {2,4}, {1}, {3}, {3,4}, {1}, {2}
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    long long Friend_Pairing_Friend_Pair_DP(int n) {
        /*
        DP approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        if (n <= 2) return n;
        vector<long long> dp(n+1, 0);
        dp[0] = 0;
        dp[1] = 1;
        dp[2] = 2;
        for (int i = 3; i <= n; i++) {
            dp[i] = dp[i-1] + (i-1) * dp[i-2];
        }
        return dp[n];
    }

    long long Friend_Pairing_Friend_Pair_Space(int n) {
        /*
        Space optimized approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        if (n <= 2) return n;
        long long prev2 = 1;
        long long prev1 = 2;
        for (int i = 3; i <= n; i++) {
            long long curr = prev1 + (i-1) * prev2;
            prev2 = prev1;
            prev1 = curr;
        }
        return prev1;
    }
};

void Test_Friend_Pairing() {
    Solution solution;
    int n = 4;
    
    cout << "DP: " << solution.Friend_Pairing_Friend_Pair_DP(n) << endl;
    cout << "Space Optimized: " << solution.Friend_Pairing_Friend_Pair_Space(n) << endl;
}

int main() {
    Test_Friend_Pairing();
    return 0;
}
