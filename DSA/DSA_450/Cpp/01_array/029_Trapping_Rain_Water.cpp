/*
Problem: Trapping Rain Water
URL: https://practice.geeksforgeeks.org/problems/trapping-rain-water-1587115621/1

Problem Statement:
Given an array arr[] of N non-negative integers representing an elevation map where
the width of each bar is 1, compute how much water it can trap after raining.

Sample Input/Output:
Input: arr = [3, 0, 2, 0, 4]
Output: 7

Input: arr = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]
Output: 6
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Trap_Water_Two_Pointer_Optimal(vector<int>& arr) {
        /*
        Two Pointer Approach - Left and right pointers with max tracking
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        int left = 0, right = arr.size() - 1;
        int left_max = 0, right_max = 0, water = 0;
        while (left <= right) {
            if (arr[left] <= arr[right]) {
                if (arr[left] >= left_max) left_max = arr[left];
                else water += left_max - arr[left];
                left++;
            } else {
                if (arr[right] >= right_max) right_max = arr[right];
                else water += right_max - arr[right];
                right--;
            }
        }
        return water;
    }

    int Trap_Water_Prefix_Suffix(vector<int>& arr) {
        /*
        Prefix-Suffix Max Arrays - Precompute left and right max heights
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        int n = arr.size();
        vector<int> left_max(n), right_max(n);
        left_max[0] = arr[0];
        right_max[n - 1] = arr[n - 1];
        for (int i = 1; i < n; i++)
            left_max[i] = max(arr[i], left_max[i - 1]);
        for (int i = n - 2; i >= 0; i--)
            right_max[i] = max(arr[i], right_max[i + 1]);
        int water = 0;
        for (int i = 1; i < n - 1; i++)
            water += max(0, min(left_max[i], right_max[i]) - arr[i]);
        return water;
    }
};

void Test_Trapping_Rain_Water() {
    Solution solution;

    struct TestCase {
        vector<int> arr;
        int expected;
    };

    vector<TestCase> test_cases = {
        {{3, 0, 2, 0, 4}, 7},
        {{0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1}, 6},
        {{4, 2, 0, 3, 2, 5}, 9},
        {{1, 2, 3, 4, 5}, 0}
    };

    for (auto& tc : test_cases) {
        cout << "Heights: ";
        for (int x : tc.arr) cout << x << " ";
        cout << ", Expected: " << tc.expected << endl;

        cout << "Two Pointer: " << solution.Trap_Water_Two_Pointer_Optimal(tc.arr) << endl;
        cout << "Prefix-Suffix: " << solution.Trap_Water_Prefix_Suffix(tc.arr) << endl;

        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Trapping_Rain_Water();
    return 0;
}
