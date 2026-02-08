/*
Problem: Activity Selection
URL: https://practice.geeksforgeeks.org/problems/n-meetings-in-one-room-1587115620/1

Problem Statement:
Given N activities with start and end times, find the maximum number of activities that can be performed by a single person, assuming that a person can only work on a single activity at a time.

Sample Input/Output:
Input: start[] = {1, 3, 0, 5, 8, 5}, end[] = {2, 4, 6, 7, 9, 9}
Output: 4
Explanation: Activities that can be performed are: (1,2), (3,4), (5,7), (8,9)
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Max_Meetings_Greedy(vector<int>& start, vector<int>& end, int n) {
        /*
        Sort activities by finish time, then greedily select non-overlapping activities
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        */
        vector<pair<int, int>> activities;
        for (int i = 0; i < n; i++) {
            activities.push_back({end[i], start[i]});
        }
        sort(activities.begin(), activities.end());
        
        int count = 1;
        int last_end = activities[0].first;
        
        for (int i = 1; i < n; i++) {
            if (activities[i].second > last_end) {
                count++;
                last_end = activities[i].first;
            }
        }
        
        return count;
    }
};

void Test_Activity_Selection() {
    Solution solution;
    vector<int> start = {1, 3, 0, 5, 8, 5};
    vector<int> end = {2, 4, 6, 7, 9, 9};
    int n = start.size();
    cout << "Max meetings: " << solution.Max_Meetings_Greedy(start, end, n) << endl;
}

int main() {
    Test_Activity_Selection();
    return 0;
}
