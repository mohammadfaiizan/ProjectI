/*
Problem: Egg Dropping Puzzle
URL: https://practice.geeksforgeeks.org/problems/egg-dropping-puzzle-1587115620/1

Problem Statement:
You are given N identical eggs and you have access to a K-floored building from 1 to K. There exists a floor f where 0 <= f <= K such that any egg dropped at a floor higher than f will break, and any egg dropped at or below floor f will not break. There are few rules given below. An egg that survives a fall can be used again. A broken egg must be discarded. The effect of a fall is the same for all eggs. If the egg doesn't break at a certain floor, it will not break at any floor below. If the eggs breaks at a certain floor, it will break at any floor above. Find the minimum number of moves that you need to determine with certainty what the value of f is.

Sample Input/Output:
Input: eggs = 2, floors = 10
Output: 4
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Egg_Dropping_Egg_Drop_Recursive(int n, int k) {
        /*
        Recursive approach
        Time Complexity: O(n*k^2)
        Space Complexity: O(n*k)
        */
        if (k == 0 || k == 1) return k;
        if (n == 1) return k;
        int min_attempts = INT_MAX;
        for (int x = 1; x <= k; x++) {
            int res = max(Egg_Dropping_Egg_Drop_Recursive(n-1, x-1),
                         Egg_Dropping_Egg_Drop_Recursive(n, k-x));
            min_attempts = min(min_attempts, res);
        }
        return min_attempts + 1;
    }

    int Egg_Dropping_Egg_Drop_Memo(int n, int k) {
        /*
        Memoization approach
        Time Complexity: O(n*k^2)
        Space Complexity: O(n*k)
        */
        vector<vector<int>> memo(n+1, vector<int>(k+1, -1));
        return Egg_Drop_Memo_Helper(n, k, memo);
    }

    int Egg_Drop_Memo_Helper(int n, int k, vector<vector<int>>& memo) {
        if (k == 0 || k == 1) return k;
        if (n == 1) return k;
        if (memo[n][k] != -1) return memo[n][k];
        int min_attempts = INT_MAX;
        for (int x = 1; x <= k; x++) {
            int res = max(Egg_Drop_Memo_Helper(n-1, x-1, memo),
                         Egg_Drop_Memo_Helper(n, k-x, memo));
            min_attempts = min(min_attempts, res);
        }
        memo[n][k] = min_attempts + 1;
        return memo[n][k];
    }

    int Egg_Dropping_Egg_Drop_DP(int n, int k) {
        /*
        DP approach
        Time Complexity: O(n*k^2)
        Space Complexity: O(n*k)
        */
        vector<vector<int>> dp(n+1, vector<int>(k+1, 0));
        for (int i = 1; i <= n; i++) {
            dp[i][0] = 0;
            dp[i][1] = 1;
        }
        for (int j = 1; j <= k; j++) {
            dp[1][j] = j;
        }
        for (int i = 2; i <= n; i++) {
            for (int j = 2; j <= k; j++) {
                dp[i][j] = INT_MAX;
                for (int x = 1; x <= j; x++) {
                    int res = 1 + max(dp[i-1][x-1], dp[i][j-x]);
                    dp[i][j] = min(dp[i][j], res);
                }
            }
        }
        return dp[n][k];
    }

    int Egg_Dropping_Egg_Drop_Binary_Search(int n, int k) {
        /*
        Binary search optimization
        Time Complexity: O(n*k*log k)
        Space Complexity: O(n*k)
        */
        vector<vector<int>> dp(n+1, vector<int>(k+1, 0));
        for (int i = 1; i <= n; i++) {
            dp[i][0] = 0;
            dp[i][1] = 1;
        }
        for (int j = 1; j <= k; j++) {
            dp[1][j] = j;
        }
        for (int i = 2; i <= n; i++) {
            for (int j = 2; j <= k; j++) {
                dp[i][j] = INT_MAX;
                int left = 1, right = j;
                while (left <= right) {
                    int mid = left + (right - left) / 2;
                    int broken = dp[i-1][mid-1];
                    int not_broken = dp[i][j-mid];
                    int res = 1 + max(broken, not_broken);
                    if (broken < not_broken) {
                        left = mid + 1;
                    } else {
                        right = mid - 1;
                    }
                    dp[i][j] = min(dp[i][j], res);
                }
            }
        }
        return dp[n][k];
    }
};

void Test_Egg_Dropping() {
    Solution solution;
    int eggs = 2, floors = 10;
    
    cout << "Recursive: " << solution.Egg_Dropping_Egg_Drop_Recursive(eggs, floors) << endl;
    cout << "Memoization: " << solution.Egg_Dropping_Egg_Drop_Memo(eggs, floors) << endl;
    cout << "DP: " << solution.Egg_Dropping_Egg_Drop_DP(eggs, floors) << endl;
    cout << "Binary Search: " << solution.Egg_Dropping_Egg_Drop_Binary_Search(eggs, floors) << endl;
}

int main() {
    Test_Egg_Dropping();
    return 0;
}
