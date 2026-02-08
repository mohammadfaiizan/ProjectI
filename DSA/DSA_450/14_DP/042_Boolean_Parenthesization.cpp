/*
Problem: Boolean Parenthesization
URL: https://practice.geeksforgeeks.org/problems/boolean-parenthesization5610/1

Problem Statement:
Given a boolean expression S of length N with following symbols: Symbols 'T' represents true and 'F' represents false and following operators: &, |, ^ (AND, OR, XOR). Count the number of ways we can parenthesize the expression so that the value of expression evaluates to true.

Sample Input/Output:
Input: "T|T&F^T"
Output: 4
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Bool_Paren_Memo(string s) {
        /*
        Memoization
        Time Complexity: O(n^3)
        Space Complexity: O(n^2)
        */
        int n = s.length();
        vector<vector<vector<int>>> memo(n, vector<vector<int>>(n, vector<int>(2, -1)));
        return solve(s, 0, n - 1, true, memo);
    }
    
    int Bool_Paren_Tab(string s) {
        /*
        Tabulation
        Time Complexity: O(n^3)
        Space Complexity: O(n^2)
        */
        int n = s.length();
        vector<vector<int>> trueDP(n, vector<int>(n, 0));
        vector<vector<int>> falseDP(n, vector<int>(n, 0));
        
        for (int i = 0; i < n; i += 2) {
            if (s[i] == 'T') {
                trueDP[i][i] = 1;
                falseDP[i][i] = 0;
            } else {
                trueDP[i][i] = 0;
                falseDP[i][i] = 1;
            }
        }
        
        for (int len = 3; len <= n; len += 2) {
            for (int i = 0; i <= n - len; i += 2) {
                int j = i + len - 1;
                
                for (int k = i + 1; k < j; k += 2) {
                    int leftTrue = trueDP[i][k - 1];
                    int leftFalse = falseDP[i][k - 1];
                    int rightTrue = trueDP[k + 1][j];
                    int rightFalse = falseDP[k + 1][j];
                    
                    if (s[k] == '&') {
                        trueDP[i][j] += leftTrue * rightTrue;
                        falseDP[i][j] += leftTrue * rightFalse + leftFalse * rightTrue + leftFalse * rightFalse;
                    } else if (s[k] == '|') {
                        trueDP[i][j] += leftTrue * rightTrue + leftTrue * rightFalse + leftFalse * rightTrue;
                        falseDP[i][j] += leftFalse * rightFalse;
                    } else if (s[k] == '^') {
                        trueDP[i][j] += leftTrue * rightFalse + leftFalse * rightTrue;
                        falseDP[i][j] += leftTrue * rightTrue + leftFalse * rightFalse;
                    }
                }
            }
        }
        
        return trueDP[0][n - 1];
    }
    
private:
    int solve(string& s, int i, int j, bool isTrue, vector<vector<vector<int>>>& memo) {
        if (i > j) return 0;
        
        if (i == j) {
            if (isTrue) {
                return s[i] == 'T' ? 1 : 0;
            } else {
                return s[i] == 'F' ? 1 : 0;
            }
        }
        
        if (memo[i][j][isTrue] != -1) {
            return memo[i][j][isTrue];
        }
        
        int ways = 0;
        
        for (int k = i + 1; k < j; k += 2) {
            int leftTrue = solve(s, i, k - 1, true, memo);
            int leftFalse = solve(s, i, k - 1, false, memo);
            int rightTrue = solve(s, k + 1, j, true, memo);
            int rightFalse = solve(s, k + 1, j, false, memo);
            
            if (s[k] == '&') {
                if (isTrue) {
                    ways += leftTrue * rightTrue;
                } else {
                    ways += leftTrue * rightFalse + leftFalse * rightTrue + leftFalse * rightFalse;
                }
            } else if (s[k] == '|') {
                if (isTrue) {
                    ways += leftTrue * rightTrue + leftTrue * rightFalse + leftFalse * rightTrue;
                } else {
                    ways += leftFalse * rightFalse;
                }
            } else if (s[k] == '^') {
                if (isTrue) {
                    ways += leftTrue * rightFalse + leftFalse * rightTrue;
                } else {
                    ways += leftTrue * rightTrue + leftFalse * rightFalse;
                }
            }
        }
        
        return memo[i][j][isTrue] = ways;
    }
};

void Test_Bool_Paren_Memo() {
    Solution solution;
    assert(solution.Bool_Paren_Memo("T|T&F^T") == 4);
}

void Test_Bool_Paren_Tab() {
    Solution solution;
    assert(solution.Bool_Paren_Tab("T|T&F^T") == 4);
}

int main() {
    Test_Bool_Paren_Memo();
    Test_Bool_Paren_Tab();
    return 0;
}
