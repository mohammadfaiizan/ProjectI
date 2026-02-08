/*
Problem: Minimum Swaps Required to Group Together
URL: https://practice.geeksforgeeks.org/problems/minimum-swaps-required-to-bring-all-elements-less-than-or-equal-to-k-together4847/1

Problem Statement:
Given an array of n positive integers and a number k, find the minimum number of swaps
required to bring all the numbers less than or equal to k together in a contiguous subarray.

Sample Input/Output:
Input: arr = [2, 1, 5, 6, 3], K = 3
Output: 1
Explanation: Swap 5 with 3, resulting [2, 1, 3, 6, 5].

Input: arr = [2, 7, 9, 5, 8, 7, 4], K = 6
Output: 2
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Min_Swaps_Sliding_Window_Optimal(vector<int>& arr, int k) {
        /*
        Sliding Window - Count bad elements in windows of size = count of elements <= k
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        int n = arr.size();
        int count = 0;
        for (int x : arr) {
            if (x <= k) count++;
        }
        int bad = 0;
        for (int i = 0; i < count; i++) {
            if (arr[i] > k) bad++;
        }
        int ans = bad;
        for (int i = 0, j = count; j < n; i++, j++) {
            if (arr[i] > k) bad--;
            if (arr[j] > k) bad++;
            ans = min(ans, bad);
        }
        return ans;
    }
};

void Test_Minimum_Swaps_Group() {
    Solution solution;

    struct TestCase {
        vector<int> arr;
        int k;
        int expected;
    };

    vector<TestCase> test_cases = {
        {{2, 1, 5, 6, 3}, 3, 1},
        {{2, 7, 9, 5, 8, 7, 4}, 6, 2},
        {{1, 2, 3}, 3, 0},
        {{5, 4, 3, 2, 1}, 3, 0}
    };

    for (auto& tc : test_cases) {
        cout << "Array: ";
        for (int x : tc.arr) cout << x << " ";
        cout << ", K=" << tc.k << ", Expected=" << tc.expected << endl;

        cout << "Sliding Window: " << solution.Min_Swaps_Sliding_Window_Optimal(tc.arr, tc.k) << endl;

        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Minimum_Swaps_Group();
    return 0;
}
