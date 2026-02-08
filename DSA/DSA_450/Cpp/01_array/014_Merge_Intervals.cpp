/*
Problem: Merge Intervals
URL: https://leetcode.com/problems/merge-intervals/

Problem Statement:
Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping
intervals, and return an array of non-overlapping intervals that cover all intervals.

Sample Input/Output:
Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
Explanation: Intervals [1,3] and [2,6] overlap, merged into [1,6].

Input: intervals = [[1,4],[4,5]]
Output: [[1,5]]
Explanation: Intervals [1,4] and [4,5] overlap.
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<vector<int>> Merge_Intervals_Sort_And_Merge_Optimal(vector<vector<int>> intervals) {
        /*
        Sort and Merge - Sort by start, merge overlapping
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        */
        sort(intervals.begin(), intervals.end());
        vector<vector<int>> result;
        result.push_back(intervals[0]);
        for (int i = 1; i < (int)intervals.size(); i++) {
            if (intervals[i][0] <= result.back()[1]) {
                result.back()[1] = max(result.back()[1], intervals[i][1]);
            } else {
                result.push_back(intervals[i]);
            }
        }
        return result;
    }

    vector<vector<int>> Merge_Intervals_In_Place(vector<vector<int>> intervals) {
        /*
        In-Place Merge - Track merge index in sorted array
        Time Complexity: O(n log n)
        Space Complexity: O(1) excluding result
        */
        sort(intervals.begin(), intervals.end());
        int idx = 0;
        for (int i = 1; i < (int)intervals.size(); i++) {
            if (intervals[i][0] <= intervals[idx][1]) {
                intervals[idx][1] = max(intervals[idx][1], intervals[i][1]);
            } else {
                idx++;
                intervals[idx] = intervals[i];
            }
        }
        return vector<vector<int>>(intervals.begin(), intervals.begin() + idx + 1);
    }
};

void Test_Merge_Intervals() {
    Solution solution;

    vector<vector<vector<int>>> test_cases = {
        {{1,3},{2,6},{8,10},{15,18}},
        {{1,4},{4,5}},
        {{1,4},{0,4}},
        {{1,4},{2,3}}
    };

    for (auto& intervals : test_cases) {
        cout << "Intervals: ";
        for (auto& i : intervals) cout << "[" << i[0] << "," << i[1] << "] ";
        cout << endl;

        auto r1 = solution.Merge_Intervals_Sort_And_Merge_Optimal(intervals);
        cout << "Sort & Merge: ";
        for (auto& i : r1) cout << "[" << i[0] << "," << i[1] << "] ";
        cout << endl;

        auto r2 = solution.Merge_Intervals_In_Place(intervals);
        cout << "In-Place: ";
        for (auto& i : r2) cout << "[" << i[0] << "," << i[1] << "] ";
        cout << endl;

        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Merge_Intervals();
    return 0;
}
