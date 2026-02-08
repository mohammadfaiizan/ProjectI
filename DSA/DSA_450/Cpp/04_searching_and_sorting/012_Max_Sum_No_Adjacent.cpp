/*
Problem: Stickler Thief / Max Sum No Two Adjacent
URL: https://practice.geeksforgeeks.org/problems/stickler-theif-1587115621/1

Problem Statement:
Find the maximum sum of a subsequence such that no two elements are adjacent.

Sample Input/Output:
Input: arr[] = {5, 5, 10, 100, 10, 5}
Output: 110

Input: arr[] = {1, 2, 3}
Output: 4
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Max_Sum_DP_Array(vector<int>& arr, int n) {
        /*
        Dynamic programming with array to store maximum sum up to each index
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        if (n == 0) return 0;
        if (n == 1) return arr[0];
        
        vector<int> dp(n);
        dp[0] = arr[0];
        dp[1] = max(arr[0], arr[1]);
        
        for (int i = 2; i < n; i++) {
            dp[i] = max(dp[i - 1], dp[i - 2] + arr[i]);
        }
        
        return dp[n - 1];
    }

    int Max_Sum_DP_Two_Variables(vector<int>& arr, int n) {
        /*
        Dynamic programming using only two variables to track previous two states
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        if (n == 0) return 0;
        if (n == 1) return arr[0];
        
        int prev2 = arr[0];
        int prev1 = max(arr[0], arr[1]);
        
        for (int i = 2; i < n; i++) {
            int current = max(prev1, prev2 + arr[i]);
            prev2 = prev1;
            prev1 = current;
        }
        
        return prev1;
    }

    int Max_Sum_Recursive_Memo(vector<int>& arr, int n) {
        /*
        Recursive solution with memoization
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        vector<int> memo(n, -1);
        
        function<int(int)> solve = [&](int idx) -> int {
            if (idx >= n) return 0;
            if (memo[idx] != -1) return memo[idx];
            
            int take = arr[idx] + solve(idx + 2);
            int skip = solve(idx + 1);
            
            return memo[idx] = max(take, skip);
        };
        
        return solve(0);
    }
};

void Test_Max_Sum_No_Adjacent() {
    Solution sol;
    vector<vector<int>> tests = {
        {5, 5, 10, 100, 10, 5},
        {1, 2, 3},
        {3, 2, 5, 10, 7},
        {1},
        {2, 1, 4, 9}
    };

    for (auto& arr : tests) {
        int n = arr.size();
        cout << "Array: ";
        for (int num : arr) cout << num << " ";
        cout << endl;
        
        int res1 = sol.Max_Sum_DP_Array(arr, n);
        int res2 = sol.Max_Sum_DP_Two_Variables(arr, n);
        int res3 = sol.Max_Sum_Recursive_Memo(arr, n);
        
        cout << "DP Array: " << res1 << endl;
        cout << "DP Two Variables: " << res2 << endl;
        cout << "Recursive + Memo: " << res3 << endl;
        
        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Max_Sum_No_Adjacent();
    return 0;
}
