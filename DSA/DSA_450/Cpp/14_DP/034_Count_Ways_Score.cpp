/*
Problem: Count Ways to Reach Score
URL: https://practice.geeksforgeeks.org/problems/number-of-ways/1

Problem Statement:
Given a score n, find the number of ways to reach the score using 3, 5, and 10 points.

Sample Input/Output:
Input: n=20
Output: 4
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Count_Score_DP(int n) {
        /*
        Dynamic Programming
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        vector<int> dp(n + 1, 0);
        dp[0] = 1;
        
        vector<int> scores = {3, 5, 10};
        
        for (int score : scores) {
            for (int i = score; i <= n; i++) {
                dp[i] += dp[i - score];
            }
        }
        
        return dp[n];
    }
};

void Test_Count_Score_DP() {
    Solution solution;
    assert(solution.Count_Score_DP(20) == 4);
}

int main() {
    Test_Count_Score_DP();
    return 0;
}
