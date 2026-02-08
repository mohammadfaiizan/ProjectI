/*
Problem: Maximum Meetings
URL: https://www.geeksforgeeks.org/find-maximum-meetings-in-one-room/

Problem Statement:
Given N meetings with start and end times, find max meetings in one room. Print which meetings are selected.

Sample Input/Output:
Input: start[] = {1, 3, 0, 5, 8, 5}, end[] = {2, 4, 6, 7, 9, 9}
Output: 0 1 3 4
Explanation: Meetings at indices 0, 1, 3, 4 can be scheduled (4 meetings total).
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<int> Maximum_Meetings_Sort_Finish_Time(vector<int>& start, vector<int>& end) {
        /*
        Sort by finish time greedy approach: Always pick meeting that ends earliest
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        */
        int n = start.size();
        vector<pair<int, pair<int, int>>> meetings;
        
        for (int i = 0; i < n; i++) {
            meetings.push_back({end[i], {start[i], i}});
        }
        
        sort(meetings.begin(), meetings.end());
        
        vector<int> result;
        int last_end_time = -1;
        
        for (auto& meeting : meetings) {
            int end_time = meeting.first;
            int start_time = meeting.second.first;
            int index = meeting.second.second;
            
            if (start_time > last_end_time) {
                result.push_back(index);
                last_end_time = end_time;
            }
        }
        
        return result;
    }
};

void Test_Maximum_Meetings() {
    Solution solution;
    
    vector<int> start1 = {1, 3, 0, 5, 8, 5};
    vector<int> end1 = {2, 4, 6, 7, 9, 9};
    vector<int> result1 = solution.Maximum_Meetings_Sort_Finish_Time(start1, end1);
    cout << "Test 1 - Selected meetings: ";
    for (int idx : result1) {
        cout << idx << " ";
    }
    cout << endl;
    
    vector<int> start2 = {1, 2};
    vector<int> end2 = {2, 3};
    vector<int> result2 = solution.Maximum_Meetings_Sort_Finish_Time(start2, end2);
    cout << "Test 2 - Selected meetings: ";
    for (int idx : result2) {
        cout << idx << " ";
    }
    cout << endl;
}

int main() {
    Test_Maximum_Meetings();
    return 0;
}
