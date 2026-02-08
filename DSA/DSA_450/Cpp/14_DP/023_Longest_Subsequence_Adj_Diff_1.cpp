/*
Problem: Longest Subsequence with Adjacent Difference 1
URL: https://www.geeksforgeeks.org/longest-subsequence-such-that-difference-between-adjacents-is-one/

Problem Statement:
Given an array of n integers, find the length of the longest subsequence such that adjacent elements of the subsequence have a difference of 1.

Sample Input/Output:
Input: [1, 2, 3, 4, 5, 3, 2]
Output: 6
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Longest_Subseq_DP(vector<int>& arr, int n) {
        /*
        DP approach
        Time Complexity: O(n^2)
        Space Complexity: O(n)
        */
        vector<int> dp(n, 1);
        for (int i = 1; i < n; i++) {
            for (int j = 0; j < i; j++) {
                if (abs(arr[i] - arr[j]) == 1) {
                    dp[i] = max(dp[i], dp[j] + 1);
                }
            }
        }
        return *max_element(dp.begin(), dp.end());
    }
};

void Test_Longest_Subseq() {
    Solution solution;
    vector<int> arr = {1, 2, 3, 4, 5, 3, 2};
    cout << "Longest Length: " << solution.Longest_Subseq_DP(arr, arr.size()) << endl;
}

int main() {
    Test_Longest_Subseq();
    return 0;
}
