/*
Problem: Permutation Coefficient
URL: https://www.geeksforgeeks.org/permutation-coefficient/

Problem Statement:
Permutation refers to the process of arranging all the members of a given set to form a sequence. The number of permutations on a set of n elements is given by n! (n factorial). The Permutation Coefficient represented by P(n, r) is used to represent the number of ways to obtain an ordered subset having r elements from a set of n elements.

Sample Input/Output:
Input: n = 10, r = 2
Output: 90
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Permutation_Coefficient_Permutation_Recursive(int n, int r) {
        /*
        Recursive approach
        Time Complexity: O(n-r)
        Space Complexity: O(n-r)
        */
        if (r == 0) return 1;
        if (r > n) return 0;
        return n * Permutation_Coefficient_Permutation_Recursive(n-1, r-1);
    }

    int Permutation_Coefficient_Permutation_DP(int n, int r) {
        /*
        DP approach
        Time Complexity: O(n*r)
        Space Complexity: O(n*r)
        */
        if (r > n) return 0;
        vector<vector<int>> dp(n+1, vector<int>(r+1, 0));
        for (int i = 0; i <= n; i++) {
            for (int j = 0; j <= min(i, r); j++) {
                if (j == 0) {
                    dp[i][j] = 1;
                } else {
                    dp[i][j] = dp[i-1][j] + j * dp[i-1][j-1];
                }
            }
        }
        return dp[n][r];
    }

    int Permutation_Coefficient_Permutation_Optimized(int n, int r) {
        /*
        Space optimized approach
        Time Complexity: O(n*r)
        Space Complexity: O(r)
        */
        if (r > n) return 0;
        vector<int> dp(r+1, 0);
        dp[0] = 1;
        for (int i = 1; i <= n; i++) {
            for (int j = min(i, r); j > 0; j--) {
                dp[j] = dp[j] + j * dp[j-1];
            }
        }
        return dp[r];
    }
};

void Test_Permutation_Coefficient() {
    Solution solution;
    int n = 10, r = 2;
    
    cout << "Recursive: " << solution.Permutation_Coefficient_Permutation_Recursive(n, r) << endl;
    cout << "DP: " << solution.Permutation_Coefficient_Permutation_DP(n, r) << endl;
    cout << "Optimized: " << solution.Permutation_Coefficient_Permutation_Optimized(n, r) << endl;
}

int main() {
    Test_Permutation_Coefficient();
    return 0;
}
