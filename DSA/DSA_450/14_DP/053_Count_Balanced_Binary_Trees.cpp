/*
Problem: Count Balanced Binary Trees of Height h
URL: https://www.geeksforgeeks.org/count-balanced-binary-trees-height-h/

Problem Statement:
Count the number of balanced binary trees of height h. A balanced binary tree is one where the difference between heights of left and right subtrees is at most 1.

Sample Input/Output:
Input: h = 3
Output: 15
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    long long Count_BBT_DP(int h) {
        /*
        DP approach
        Time Complexity: O(h)
        Space Complexity: O(h)
        */
        if (h == 0 || h == 1) return 1;
        
        vector<long long> dp(h + 1);
        dp[0] = 1;
        dp[1] = 1;
        
        for (int i = 2; i <= h; i++) {
            dp[i] = dp[i - 1] * dp[i - 1] + 2 * dp[i - 1] * dp[i - 2];
        }
        
        return dp[h];
    }
};

void Test_Count_BBT() {
    Solution solution;
    
    int h = 3;
    cout << "h=" << h << " -> " << solution.Count_BBT_DP(h) << endl;
}

int main() {
    Test_Count_BBT();
    return 0;
}
