/*
Problem: Chocolate Distribution Problem
URL: https://practice.geeksforgeeks.org/problems/chocolate-distribution-problem3825/1

Problem Statement:
Given an array of N integers where each value represents the number of chocolates in a packet.
There are M students, distribute chocolate packets such that each student gets one packet,
and the difference between max and min chocolates given is minimized.

Sample Input/Output:
Input: arr = [3, 4, 1, 9, 56, 7, 9, 12], M = 5
Output: 6
Explanation: Selected packets: [3, 4, 7, 9, 9]. Max-Min = 9-3 = 6.

Input: arr = [7, 3, 2, 4, 9, 12, 56], M = 3
Output: 2
Explanation: Selected packets: [2, 3, 4]. Max-Min = 4-2 = 2.
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    long long Chocolate_Distribution_Sliding_Window_Optimal(vector<long long> arr, long long m) {
        /*
        Sorting + Sliding Window - Sort and check all windows of size m
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        */
        sort(arr.begin(), arr.end());
        long long n = arr.size();
        long long min_diff = arr[m - 1] - arr[0];
        for (long long i = 1; i <= n - m; i++) {
            min_diff = min(min_diff, arr[i + m - 1] - arr[i]);
        }
        return min_diff;
    }
};

void Test_Chocolate_Distribution() {
    Solution solution;

    struct TestCase {
        vector<long long> arr;
        long long m;
        long long expected;
    };

    vector<TestCase> test_cases = {
        {{3, 4, 1, 9, 56, 7, 9, 12}, 5, 6},
        {{7, 3, 2, 4, 9, 12, 56}, 3, 2},
        {{12, 4, 7, 9, 2, 23, 25, 41, 30, 40, 28, 42, 30, 44, 48, 43, 50}, 7, 10}
    };

    for (auto& tc : test_cases) {
        cout << "Chocolates: ";
        for (long long x : tc.arr) cout << x << " ";
        cout << ", M=" << tc.m << ", Expected=" << tc.expected << endl;

        cout << "Sliding Window: " << solution.Chocolate_Distribution_Sliding_Window_Optimal(tc.arr, tc.m) << endl;

        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Chocolate_Distribution();
    return 0;
}
