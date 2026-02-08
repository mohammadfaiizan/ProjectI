/*
Problem: Shortest Job First
URL: https://www.geeksforgeeks.org/program-for-shortest-job-first-or-sjf-cpu-scheduling-set-1-non-preemptive/

Problem Statement:
Implement non-preemptive SJF CPU scheduling. Given processes with arrival and burst times, find order and waiting times.

Sample Input/Output:
Input: processes = {{1, 6}, {2, 8}, {3, 7}, {4, 3}}
Output: Process order: 1 4 3 2, Average waiting time: 7
Explanation: Process 1 arrives first, then 4 (shortest), then 3, then 2.
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    pair<vector<int>, double> Shortest_Job_First_Min_Heap(vector<vector<int>>& processes) {
        /*
        Min-heap by burst time greedy approach
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        */
        int n = processes.size();
        vector<pair<int, int>> process_list;
        
        for (int i = 0; i < n; i++) {
            process_list.push_back({processes[i][0], processes[i][1]});
        }
        
        sort(process_list.begin(), process_list.end());
        
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
        vector<int> order;
        vector<int> waiting_time(n, 0);
        
        int current_time = 0;
        int idx = 0;
        
        while (idx < n || !pq.empty()) {
            while (idx < n && process_list[idx].first <= current_time) {
                pq.push({process_list[idx].second, idx});
                idx++;
            }
            
            if (pq.empty()) {
                current_time = process_list[idx].first;
                continue;
            }
            
            auto current = pq.top();
            pq.pop();
            
            int process_idx = current.second;
            int burst_time = current.first;
            int arrival_time = process_list[process_idx].first;
            
            waiting_time[process_idx] = current_time - arrival_time;
            order.push_back(process_idx + 1);
            current_time += burst_time;
        }
        
        double avg_waiting = accumulate(waiting_time.begin(), waiting_time.end(), 0.0) / n;
        
        return {order, avg_waiting};
    }
};

void Test_Shortest_Job_First() {
    Solution solution;
    
    vector<vector<int>> processes1 = {{1, 6}, {2, 8}, {3, 7}, {4, 3}};
    auto result1 = solution.Shortest_Job_First_Min_Heap(processes1);
    cout << "Test 1 - Process order: ";
    for (int p : result1.first) {
        cout << p << " ";
    }
    cout << ", Average waiting time: " << result1.second << endl;
    
    vector<vector<int>> processes2 = {{0, 3}, {1, 6}, {2, 4}};
    auto result2 = solution.Shortest_Job_First_Min_Heap(processes2);
    cout << "Test 2 - Process order: ";
    for (int p : result2.first) {
        cout << p << " ";
    }
    cout << ", Average waiting time: " << result2.second << endl;
}

int main() {
    Test_Shortest_Job_First();
    return 0;
}
