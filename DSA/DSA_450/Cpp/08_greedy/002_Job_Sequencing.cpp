/*
Problem: Job Sequencing
URL: https://practice.geeksforgeeks.org/problems/job-sequencing-problem-1587115620/1

Problem Statement:
Given a set of N jobs where each job has a deadline and profit associated with it. Each job takes 1 unit of time to complete and only one job can be scheduled at a time. Find the maximum profit and the number of jobs done.

Sample Input/Output:
Input: N = 4, Jobs = {(1,4,20),(2,1,10),(3,1,40),(4,1,30)}
Output: 2 60
Explanation: Job1 and Job3 can be done with maximum profit of 60 (20+40).
*/

#include <bits/stdc++.h>
using namespace std;

struct Job {
    int id;
    int dead;
    int profit;
};

class Solution {
public:
    vector<int> Job_Scheduling_Greedy(Job arr[], int n) {
        /*
        Sort by profit descending, greedily assign to latest available slot
        Time Complexity: O(n^2)
        Space Complexity: O(n)
        */
        sort(arr, arr + n, [](Job a, Job b) {
            return a.profit > b.profit;
        });
        
        int max_deadline = 0;
        for (int i = 0; i < n; i++) {
            max_deadline = max(max_deadline, arr[i].dead);
        }
        
        vector<int> slot(max_deadline + 1, -1);
        int count = 0;
        int profit = 0;
        
        for (int i = 0; i < n; i++) {
            for (int j = arr[i].dead; j > 0; j--) {
                if (slot[j] == -1) {
                    slot[j] = arr[i].id;
                    count++;
                    profit += arr[i].profit;
                    break;
                }
            }
        }
        
        return {count, profit};
    }
};

void Test_Job_Sequencing() {
    Solution solution;
    Job arr[] = {{1, 4, 20}, {2, 1, 10}, {3, 1, 40}, {4, 1, 30}};
    int n = 4;
    vector<int> result = solution.Job_Scheduling_Greedy(arr, n);
    cout << "Jobs done: " << result[0] << ", Profit: " << result[1] << endl;
}

int main() {
    Test_Job_Sequencing();
    return 0;
}
