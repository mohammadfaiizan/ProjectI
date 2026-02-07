/*
Problem: Longest Consecutive Subsequence
URL: https://practice.geeksforgeeks.org/problems/longest-consecutive-subsequence2449/1

Problem Statement:
Given an array of positive integers, find the length of the longest sub-sequence such that
elements are consecutive integers (can be in any order).

Sample Input/Output:
Input: arr = [2, 6, 1, 9, 4, 5, 3]
Output: 6
Explanation: The consecutive subsequence is [1, 2, 3, 4, 5, 6].

Input: arr = [1, 9, 3, 10, 4, 20, 2]
Output: 4
Explanation: The consecutive subsequence is [1, 2, 3, 4].
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Longest_Consecutive_Hashing_Optimal(vector<int>& arr) {
        /*
        HashSet Approach - Check sequence start and count forward
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        unordered_set<int> s(arr.begin(), arr.end());
        int longest = 0;
        for (int x : s) {
            if (s.find(x - 1) == s.end()) {
                int current = x, count = 1;
                while (s.find(current + 1) != s.end()) {
                    current++;
                    count++;
                }
                longest = max(longest, count);
            }
        }
        return longest;
    }

    int Longest_Consecutive_Sorting(vector<int> arr) {
        /*
        Sorting Approach - Sort and find longest consecutive run
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        */
        if (arr.empty()) return 0;
        sort(arr.begin(), arr.end());
        arr.erase(unique(arr.begin(), arr.end()), arr.end());
        int longest = 1, current = 1;
        for (int i = 1; i < (int)arr.size(); i++) {
            if (arr[i] == arr[i - 1] + 1) {
                current++;
                longest = max(longest, current);
            } else {
                current = 1;
            }
        }
        return longest;
    }
};

void Test_Longest_Consecutive_Subsequence() {
    Solution solution;

    struct TestCase {
        vector<int> arr;
        int expected;
    };

    vector<TestCase> test_cases = {
        {{2, 6, 1, 9, 4, 5, 3}, 6},
        {{1, 9, 3, 10, 4, 20, 2}, 4},
        {{100, 4, 200, 1, 3, 2}, 4},
        {{0, 3, 7, 2, 5, 8, 4, 6, 0, 1}, 9}
    };

    for (auto& tc : test_cases) {
        cout << "Array: ";
        for (int x : tc.arr) cout << x << " ";
        cout << ", Expected: " << tc.expected << endl;

        cout << "Hashing: " << solution.Longest_Consecutive_Hashing_Optimal(tc.arr) << endl;
        cout << "Sorting: " << solution.Longest_Consecutive_Sorting(tc.arr) << endl;

        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Longest_Consecutive_Subsequence();
    return 0;
}
