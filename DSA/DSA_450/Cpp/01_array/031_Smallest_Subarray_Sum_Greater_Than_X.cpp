/*
Problem: Smallest Subarray with Sum Greater Than X
URL: https://practice.geeksforgeeks.org/problems/smallest-subarray-with-sum-greater-than-x5651/1

Problem Statement:
Given an array of integers and a number x, find the smallest subarray with sum greater
than the given value x.

Sample Input/Output:
Input: arr = [1, 4, 45, 6, 0, 19], X = 51
Output: 3
Explanation: Subarray [4, 45, 6] has sum 55 > 51 with length 3.

Input: arr = [1, 10, 5, 2, 7], X = 9
Output: 1
Explanation: Element [10] has sum 10 > 9 with length 1.
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Smallest_Subarray_Sliding_Window_Optimal(vector<int>& arr, int x) {
        /*
        Sliding Window - Expand right, shrink left when sum > x
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        int n = arr.size();
        int curr_sum = 0, min_len = n + 1;
        int start = 0;
        for (int end = 0; end < n; end++) {
            curr_sum += arr[end];
            while (curr_sum > x) {
                min_len = min(min_len, end - start + 1);
                curr_sum -= arr[start++];
            }
        }
        return min_len;
    }

    int Smallest_Subarray_Brute_Force(vector<int>& arr, int x) {
        /*
        Brute Force - Check all subarrays
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        */
        int n = arr.size();
        int min_len = n + 1;
        for (int i = 0; i < n; i++) {
            int sum = 0;
            for (int j = i; j < n; j++) {
                sum += arr[j];
                if (sum > x) {
                    min_len = min(min_len, j - i + 1);
                    break;
                }
            }
        }
        return min_len;
    }
};

void Test_Smallest_Subarray() {
    Solution solution;

    struct TestCase {
        vector<int> arr;
        int x;
        int expected;
    };

    vector<TestCase> test_cases = {
        {{1, 4, 45, 6, 0, 19}, 51, 3},
        {{1, 10, 5, 2, 7}, 9, 1},
        {{1, 11, 100, 1, 0, 200, 3, 2, 1, 250}, 280, 4},
        {{1, 2, 4}, 8, 4}
    };

    for (auto& tc : test_cases) {
        cout << "Array: ";
        for (int x : tc.arr) cout << x << " ";
        cout << ", X=" << tc.x << ", Expected=" << tc.expected << endl;

        cout << "Sliding Window: " << solution.Smallest_Subarray_Sliding_Window_Optimal(tc.arr, tc.x) << endl;
        cout << "Brute Force: " << solution.Smallest_Subarray_Brute_Force(tc.arr, tc.x) << endl;

        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Smallest_Subarray();
    return 0;
}
