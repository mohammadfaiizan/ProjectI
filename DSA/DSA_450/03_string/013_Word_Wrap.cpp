/*
Problem: Word Wrap
URL: https://practice.geeksforgeeks.org/problems/word-wrap1646/1

Problem Statement:
Given an array nums[] of size n, where nums[i] represents the number of characters
in the ith word. Let K be the limit on the number of characters that can be put in
one line (including spaces). Put line breaks in the given sequence such that the
lines are printed neatly. The cost of a line = (Number of extra spaces)^2.
Find the minimum total cost.

Sample Input/Output:
Input: nums = [3, 2, 2, 5], K = 6
Output: 10

Input: nums = [3, 2, 2], K = 4
Output: 5
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Word_Wrap_DP(vector<int>& nums, int k) {
        /*
        DP approach - minimize total cost of extra spaces squared
        Time Complexity: O(n^2)
        Space Complexity: O(n)
        */
        int n = nums.size();
        vector<int> dp(n + 1, INT_MAX);
        dp[n] = 0;

        for (int i = n - 1; i >= 0; i--) {
            int len = 0;
            for (int j = i; j < n; j++) {
                len += nums[j];
                if (len > k) break;
                int extra = k - len;
                int cost = (j == n - 1) ? 0 : extra * extra;
                if (dp[j + 1] != INT_MAX)
                    dp[i] = min(dp[i], cost + dp[j + 1]);
                len += 1;
            }
        }
        return dp[0];
    }

    int Word_Wrap_Memoization(vector<int>& nums, int k, int i, vector<int>& memo) {
        /*
        Top-down memoization
        Time Complexity: O(n^2)
        Space Complexity: O(n)
        */
        int n = nums.size();
        if (i >= n) return 0;
        if (memo[i] != -1) return memo[i];

        int len = 0;
        memo[i] = INT_MAX;
        for (int j = i; j < n; j++) {
            len += nums[j];
            if (len > k) break;
            int extra = k - len;
            int cost = (j == n - 1) ? 0 : extra * extra;
            int sub = Word_Wrap_Memoization(nums, k, j + 1, memo);
            if (sub != INT_MAX)
                memo[i] = min(memo[i], cost + sub);
            len += 1;
        }
        return memo[i];
    }

    int Word_Wrap_Greedy(vector<int>& nums, int k) {
        /*
        Greedy - fill as many words per line as possible (not optimal but simple)
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        int n = nums.size();
        int totalCost = 0;
        int i = 0;
        while (i < n) {
            int len = nums[i];
            int j = i + 1;
            while (j < n && len + 1 + nums[j] <= k) {
                len += 1 + nums[j];
                j++;
            }
            if (j < n) {
                int extra = k - len;
                totalCost += extra * extra;
            }
            i = j;
        }
        return totalCost;
    }
};

void Test_Word_Wrap() {
    Solution sol;
    struct TestCase { vector<int> nums; int k; };
    vector<TestCase> tests = {
        {{3, 2, 2, 5}, 6},
        {{3, 2, 2}, 4},
        {{1, 1, 1, 1, 1}, 5}
    };

    for (auto& t : tests) {
        cout << "Words: ";
        for (int x : t.nums) cout << x << " ";
        cout << " K=" << t.k << endl;

        cout << "DP: " << sol.Word_Wrap_DP(t.nums, t.k) << endl;
        vector<int> memo(t.nums.size(), -1);
        cout << "Memoization: " << sol.Word_Wrap_Memoization(t.nums, t.k, 0, memo) << endl;
        cout << "Greedy: " << sol.Word_Wrap_Greedy(t.nums, t.k) << endl;

        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Word_Wrap();
    return 0;
}
