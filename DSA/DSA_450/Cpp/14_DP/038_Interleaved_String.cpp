/*
Problem: Interleaved String
URL: https://practice.geeksforgeeks.org/problems/interleaved-strings/1

Problem Statement:
Given strings A, B, and C, find whether C is formed by an interleaving of A and B. An interleaving of two strings S and T is a configuration such that it creates a new string Y from the concatenation of the two strings, maintaining the right order of characters.

Sample Input/Output:
Input: A="YX", B="X", C="XXY"
Output: false
Input: A="XY", B="X", C="XXY"
Output: true
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    bool Interleave_DP(string A, string B, string C) {
        /*
        Dynamic Programming
        Time Complexity: O(m*n)
        Space Complexity: O(m*n)
        */
        int m = A.length();
        int n = B.length();
        
        if (m + n != C.length()) return false;
        
        vector<vector<bool>> dp(m + 1, vector<bool>(n + 1, false));
        dp[0][0] = true;
        
        for (int i = 1; i <= m; i++) {
            if (A[i - 1] == C[i - 1]) {
                dp[i][0] = dp[i - 1][0];
            }
        }
        
        for (int j = 1; j <= n; j++) {
            if (B[j - 1] == C[j - 1]) {
                dp[0][j] = dp[0][j - 1];
            }
        }
        
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (A[i - 1] == C[i + j - 1] && dp[i - 1][j]) {
                    dp[i][j] = true;
                }
                if (B[j - 1] == C[i + j - 1] && dp[i][j - 1]) {
                    dp[i][j] = true;
                }
            }
        }
        
        return dp[m][n];
    }
    
    bool Interleave_Recursive_Memo(string A, string B, string C) {
        /*
        Recursive with Memoization
        Time Complexity: O(m*n)
        Space Complexity: O(m*n)
        */
        int m = A.length();
        int n = B.length();
        
        if (m + n != C.length()) return false;
        
        vector<vector<int>> memo(m + 1, vector<int>(n + 1, -1));
        return solve(A, B, C, 0, 0, 0, memo);
    }
    
private:
    bool solve(string& A, string& B, string& C, int i, int j, int k, vector<vector<int>>& memo) {
        if (k == C.length()) return true;
        
        if (memo[i][j] != -1) return memo[i][j];
        
        bool result = false;
        
        if (i < A.length() && A[i] == C[k]) {
            result = result || solve(A, B, C, i + 1, j, k + 1, memo);
        }
        
        if (j < B.length() && B[j] == C[k]) {
            result = result || solve(A, B, C, i, j + 1, k + 1, memo);
        }
        
        return memo[i][j] = result;
    }
};

void Test_Interleave_DP() {
    Solution solution;
    assert(solution.Interleave_DP("YX", "X", "XXY") == false);
    assert(solution.Interleave_DP("XY", "X", "XXY") == true);
}

void Test_Interleave_Recursive_Memo() {
    Solution solution;
    assert(solution.Interleave_Recursive_Memo("YX", "X", "XXY") == false);
    assert(solution.Interleave_Recursive_Memo("XY", "X", "XXY") == true);
}

int main() {
    Test_Interleave_DP();
    Test_Interleave_Recursive_Memo();
    return 0;
}
