/*
Problem: Maximum Length Chain of Pairs
URL: https://practice.geeksforgeeks.org/problems/max-length-chain/1

Problem Statement:
You are given N pairs of numbers. In every pair, the first number is always smaller than the second number. A pair (c, d) can follow another pair (a, b) if b < c. Chain of pairs can be formed in this fashion. Find the longest chain which can be formed from a given set of pairs.

Sample Input/Output:
Input: [[5,24], [39,60], [15,28], [27,40], [50,90]]
Output: 3
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Chain_DP(vector<pair<int, int>>& pairs, int n) {
        /*
        DP approach
        Time Complexity: O(n^2)
        Space Complexity: O(n)
        */
        sort(pairs.begin(), pairs.end());
        vector<int> dp(n, 1);
        for (int i = 1; i < n; i++) {
            for (int j = 0; j < i; j++) {
                if (pairs[j].second < pairs[i].first) {
                    dp[i] = max(dp[i], dp[j] + 1);
                }
            }
        }
        return *max_element(dp.begin(), dp.end());
    }

    int Chain_Greedy(vector<pair<int, int>>& pairs, int n) {
        /*
        Greedy approach
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        */
        sort(pairs.begin(), pairs.end(), [](pair<int,int>& a, pair<int,int>& b) {
            return a.second < b.second;
        });
        int count = 1;
        int last = pairs[0].second;
        for (int i = 1; i < n; i++) {
            if (pairs[i].first > last) {
                count++;
                last = pairs[i].second;
            }
        }
        return count;
    }
};

void Test_Chain() {
    Solution solution;
    vector<pair<int, int>> pairs = {{5,24}, {39,60}, {15,28}, {27,40}, {50,90}};
    cout << "DP: " << solution.Chain_DP(pairs, pairs.size()) << endl;
    cout << "Greedy: " << solution.Chain_Greedy(pairs, pairs.size()) << endl;
}

int main() {
    Test_Chain();
    return 0;
}
