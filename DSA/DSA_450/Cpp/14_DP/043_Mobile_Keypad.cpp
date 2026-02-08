/*
Problem: Mobile Numeric Keypad
URL: https://practice.geeksforgeeks.org/problems/mobile-numeric-keypad5

Problem Statement:
Given the mobile numeric keypad. You can only press buttons that are up, left, right, or down to the current button. You are not allowed to press bottom row corner buttons (i.e. * and #). Given a number N, find out the number of possible numbers of given length.

Sample Input/Output:
Input: n=1
Output: 10
Input: n=2
Output: 36
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    long long Keypad_DP(int n) {
        /*
        Dynamic Programming
        Time Complexity: O(n*10)
        Space Complexity: O(n*10)
        */
        if (n == 0) return 0;
        if (n == 1) return 10;
        
        vector<vector<int>> moves = {
            {0, 8},
            {1, 2, 4},
            {2, 1, 3, 5},
            {3, 2, 6},
            {4, 1, 5, 7},
            {5, 2, 4, 6, 8},
            {6, 3, 5, 9},
            {7, 4, 8},
            {8, 0, 5, 7, 9},
            {9, 6, 8}
        };
        
        vector<vector<long long>> dp(n + 1, vector<long long>(10, 0));
        
        for (int i = 0; i < 10; i++) {
            dp[1][i] = 1;
        }
        
        for (int len = 2; len <= n; len++) {
            for (int digit = 0; digit < 10; digit++) {
                for (int next : moves[digit]) {
                    dp[len][digit] += dp[len - 1][next];
                }
            }
        }
        
        long long result = 0;
        for (int i = 0; i < 10; i++) {
            result += dp[n][i];
        }
        
        return result;
    }
    
    long long Keypad_Space(int n) {
        /*
        Space Optimized
        Time Complexity: O(n*10)
        Space Complexity: O(10)
        */
        if (n == 0) return 0;
        if (n == 1) return 10;
        
        vector<vector<int>> moves = {
            {0, 8},
            {1, 2, 4},
            {2, 1, 3, 5},
            {3, 2, 6},
            {4, 1, 5, 7},
            {5, 2, 4, 6, 8},
            {6, 3, 5, 9},
            {7, 4, 8},
            {8, 0, 5, 7, 9},
            {9, 6, 8}
        };
        
        vector<long long> prev(10, 1);
        vector<long long> curr(10, 0);
        
        for (int len = 2; len <= n; len++) {
            fill(curr.begin(), curr.end(), 0);
            for (int digit = 0; digit < 10; digit++) {
                for (int next : moves[digit]) {
                    curr[digit] += prev[next];
                }
            }
            prev = curr;
        }
        
        long long result = 0;
        for (int i = 0; i < 10; i++) {
            result += prev[i];
        }
        
        return result;
    }
};

void Test_Keypad_DP() {
    Solution solution;
    assert(solution.Keypad_DP(1) == 10);
    assert(solution.Keypad_DP(2) == 36);
}

void Test_Keypad_Space() {
    Solution solution;
    assert(solution.Keypad_Space(1) == 10);
    assert(solution.Keypad_Space(2) == 36);
}

int main() {
    Test_Keypad_DP();
    Test_Keypad_Space();
    return 0;
}
