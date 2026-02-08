/*
Problem: Painting Fence
URL: https://practice.geeksforgeeks.org/problems/painting-the-fence3727/1

Problem Statement:
Given a fence with n posts and k colors, find out the number of ways of painting the fence such that at most 2 adjacent posts have the same color.

Sample Input/Output:
Input: n = 3, k = 2
Output: 6
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    long long Paint_Fence_DP(int n, int k) {
        /*
        DP approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        if (n == 0) return 0;
        if (n == 1) return k;
        vector<long long> dp(n+1);
        dp[1] = k;
        dp[2] = k * k;
        for (int i = 3; i <= n; i++) {
            dp[i] = (k-1) * (dp[i-1] + dp[i-2]);
        }
        return dp[n];
    }

    long long Paint_Fence_Space(int n, int k) {
        /*
        Space optimized approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        if (n == 0) return 0;
        if (n == 1) return k;
        long long prev2 = k;
        long long prev1 = k * k;
        for (int i = 3; i <= n; i++) {
            long long curr = (k-1) * (prev1 + prev2);
            prev2 = prev1;
            prev1 = curr;
        }
        return prev1;
    }
};

void Test_Painting_Fence() {
    Solution solution;
    int n = 3, k = 2;
    cout << "DP: " << solution.Paint_Fence_DP(n, k) << endl;
    cout << "Space Optimized: " << solution.Paint_Fence_Space(n, k) << endl;
}

int main() {
    Test_Painting_Fence();
    return 0;
}
