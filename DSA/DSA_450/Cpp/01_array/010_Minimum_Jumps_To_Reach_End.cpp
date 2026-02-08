/*
Problem: Minimum Number of Jumps to Reach End
URL: https://practice.geeksforgeeks.org/problems/minimum-number-of-jumps-1587115620/1

Problem Statement:
Given an array of N integers where each element represents the max length of the jump
that can be made forward from that element. Find the minimum number of jumps to reach
the end of the array. Return -1 if end is not reachable.

Sample Input/Output:
Input: arr = [1, 3, 5, 8, 9, 2, 6, 7, 6, 8, 9]
Output: 3
Explanation: Jump 1->3->9->last.

Input: arr = [1, 4, 3, 2, 6, 7]
Output: 2
Explanation: Jump 1->4->last.
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Min_Jumps_Greedy_Optimal(vector<int>& arr) {
        /*
        Greedy Approach - Track max reachable position
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        int n = arr.size();
        if (n <= 1) return 0;
        if (arr[0] == 0) return -1;
        int max_reach = arr[0], steps = arr[0], jumps = 1;
        for (int i = 1; i < n; i++) {
            if (i == n - 1) return jumps;
            max_reach = max(max_reach, i + arr[i]);
            steps--;
            if (steps == 0) {
                jumps++;
                if (i >= max_reach) return -1;
                steps = max_reach - i;
            }
        }
        return -1;
    }

    int Min_Jumps_DP(vector<int>& arr) {
        /*
        Dynamic Programming - Build dp array for min jumps to each index
        Time Complexity: O(n^2)
        Space Complexity: O(n)
        */
        int n = arr.size();
        if (n <= 1) return 0;
        if (arr[0] == 0) return -1;
        vector<int> dp(n, INT_MAX);
        dp[0] = 0;
        for (int i = 1; i < n; i++) {
            for (int j = 0; j < i; j++) {
                if (dp[j] != INT_MAX && j + arr[j] >= i) {
                    dp[i] = min(dp[i], dp[j] + 1);
                }
            }
        }
        return dp[n - 1] == INT_MAX ? -1 : dp[n - 1];
    }
};

void Test_Minimum_Jumps() {
    Solution solution;

    struct TestCase {
        vector<int> arr;
        int expected;
    };

    vector<TestCase> test_cases = {
        {{1, 3, 5, 8, 9, 2, 6, 7, 6, 8, 9}, 3},
        {{1, 4, 3, 2, 6, 7}, 2},
        {{0, 1, 2}, -1},
        {{1}, 0},
        {{2, 3, 1, 1, 4}, 2}
    };

    for (auto& tc : test_cases) {
        cout << "Array: ";
        for (int x : tc.arr) cout << x << " ";
        cout << ", Expected: " << tc.expected << endl;

        cout << "Greedy: " << solution.Min_Jumps_Greedy_Optimal(tc.arr) << endl;
        cout << "DP: " << solution.Min_Jumps_DP(tc.arr) << endl;

        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Minimum_Jumps();
    return 0;
}
