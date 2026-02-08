/*
Problem: Merge Overlapping Intervals
URL: https://practice.geeksforgeeks.org/problems/overlapping-intervals--170633/1

Problem Statement:
Given intervals as pairs, merge all overlapping intervals.

Sample Input/Output:
Input: [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<vector<int>> Merge_Overlapping_Intervals_Sort(vector<vector<int>>& intervals) {
        if (intervals.empty()) return {};
        sort(intervals.begin(), intervals.end());
        vector<vector<int>> merged;
        merged.push_back(intervals[0]);
        for (int i = 1; i < intervals.size(); i++) {
            if (merged.back()[1] >= intervals[i][0]) {
                merged.back()[1] = max(merged.back()[1], intervals[i][1]);
            } else {
                merged.push_back(intervals[i]);
            }
        }
        return merged;
    }

    vector<vector<int>> Merge_Overlapping_Intervals_Stack(vector<vector<int>>& intervals) {
        if (intervals.empty()) return {};
        sort(intervals.begin(), intervals.end());
        stack<vector<int>> st;
        st.push(intervals[0]);
        for (int i = 1; i < intervals.size(); i++) {
            if (st.top()[1] >= intervals[i][0]) {
                vector<int> top = st.top();
                st.pop();
                top[1] = max(top[1], intervals[i][1]);
                st.push(top);
            } else {
                st.push(intervals[i]);
            }
        }
        vector<vector<int>> merged;
        while (!st.empty()) {
            merged.push_back(st.top());
            st.pop();
        }
        reverse(merged.begin(), merged.end());
        return merged;
    }
};

void Test_Merge_Overlapping_Intervals() {
    Solution solution;
    
    cout << "=== Sort Approach ===" << endl;
    vector<vector<int>> intervals1 = {{1,3}, {2,6}, {8,10}, {15,18}};
    cout << "Input: [1,3] [2,6] [8,10] [15,18]" << endl;
    vector<vector<int>> result1 = solution.Merge_Overlapping_Intervals_Sort(intervals1);
    cout << "Output: ";
    for (auto& interval : result1) {
        cout << "[" << interval[0] << "," << interval[1] << "] ";
    }
    cout << endl;
    
    vector<vector<int>> intervals2 = {{1,4}, {4,5}};
    cout << "\nInput: [1,4] [4,5]" << endl;
    vector<vector<int>> result2 = solution.Merge_Overlapping_Intervals_Sort(intervals2);
    cout << "Output: ";
    for (auto& interval : result2) {
        cout << "[" << interval[0] << "," << interval[1] << "] ";
    }
    cout << endl;
    
    vector<vector<int>> intervals3 = {{1,9}, {2,4}, {4,7}, {6,8}};
    cout << "\nInput: [1,9] [2,4] [4,7] [6,8]" << endl;
    vector<vector<int>> result3 = solution.Merge_Overlapping_Intervals_Sort(intervals3);
    cout << "Output: ";
    for (auto& interval : result3) {
        cout << "[" << interval[0] << "," << interval[1] << "] ";
    }
    cout << endl;
    
    cout << "\n=== Stack Approach ===" << endl;
    vector<vector<int>> intervals4 = {{1,3}, {2,6}, {8,10}, {15,18}};
    cout << "Input: [1,3] [2,6] [8,10] [15,18]" << endl;
    vector<vector<int>> result4 = solution.Merge_Overlapping_Intervals_Stack(intervals4);
    cout << "Output: ";
    for (auto& interval : result4) {
        cout << "[" << interval[0] << "," << interval[1] << "] ";
    }
    cout << endl;
}

int main() {
    Test_Merge_Overlapping_Intervals();
    return 0;
}
