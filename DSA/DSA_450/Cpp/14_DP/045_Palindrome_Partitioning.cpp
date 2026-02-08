/*
Problem: Palindrome Partitioning
URL: https://practice.geeksforgeeks.org/problems/palindromic-patitioning4845/1

Problem Statement:
Given a string str, a partitioning of the string is a palindrome partitioning if every sub-string of the partition is a palindrome. Determine the fewest cuts needed for palindrome partitioning of given string.

Sample Input/Output:
Input: "ababbbabbababa"
Output: 3
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Pal_Partition_DP(string str) {
        /*
        Dynamic Programming
        Time Complexity: O(n^2)
        Space Complexity: O(n^2)
        */
        int n = str.length();
        vector<vector<bool>> isPal(n, vector<bool>(n, false));
        vector<int> cuts(n, 0);
        
        for (int i = 0; i < n; i++) {
            isPal[i][i] = true;
        }
        
        for (int i = 0; i < n - 1; i++) {
            if (str[i] == str[i + 1]) {
                isPal[i][i + 1] = true;
            }
        }
        
        for (int len = 3; len <= n; len++) {
            for (int i = 0; i <= n - len; i++) {
                int j = i + len - 1;
                if (str[i] == str[j] && isPal[i + 1][j - 1]) {
                    isPal[i][j] = true;
                }
            }
        }
        
        for (int i = 0; i < n; i++) {
            if (isPal[0][i]) {
                cuts[i] = 0;
            } else {
                cuts[i] = INT_MAX;
                for (int j = 0; j < i; j++) {
                    if (isPal[j + 1][i] && cuts[j] + 1 < cuts[i]) {
                        cuts[i] = cuts[j] + 1;
                    }
                }
            }
        }
        
        return cuts[n - 1];
    }
    
    int Pal_Partition_Memo(string str) {
        /*
        Memoization
        Time Complexity: O(n^3)
        Space Complexity: O(n^2)
        */
        int n = str.length();
        vector<vector<int>> memo(n, vector<int>(n, -1));
        return solve(str, 0, n - 1, memo);
    }
    
private:
    bool isPalindrome(string& s, int i, int j) {
        while (i < j) {
            if (s[i] != s[j]) return false;
            i++;
            j--;
        }
        return true;
    }
    
    int solve(string& str, int i, int j, vector<vector<int>>& memo) {
        if (i >= j || isPalindrome(str, i, j)) {
            return 0;
        }
        
        if (memo[i][j] != -1) {
            return memo[i][j];
        }
        
        int minCuts = INT_MAX;
        
        for (int k = i; k < j; k++) {
            int cuts = solve(str, i, k, memo) + solve(str, k + 1, j, memo) + 1;
            minCuts = min(minCuts, cuts);
        }
        
        return memo[i][j] = minCuts;
    }
};

void Test_Pal_Partition_DP() {
    Solution solution;
    assert(solution.Pal_Partition_DP("ababbbabbababa") == 3);
}

void Test_Pal_Partition_Memo() {
    Solution solution;
    assert(solution.Pal_Partition_Memo("ababbbabbababa") == 3);
}

int main() {
    Test_Pal_Partition_DP();
    Test_Pal_Partition_Memo();
    return 0;
}
