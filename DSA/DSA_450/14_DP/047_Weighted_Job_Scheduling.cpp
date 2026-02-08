/*
Problem: Weighted Job Scheduling
URL: https://www.geeksforgeeks.org/weighted-job-scheduling/

Problem Statement:
Given N jobs where every job is represented by start time, finish time and profit. Find the maximum profit subset of jobs such that no two jobs in the subset overlap.

Sample Input/Output:
Input: jobs = [(1,2,50), (3,5,20), (6,19,100), (2,100,200)]
Output: 250
*/

#include <bits/stdc++.h>
using namespace std;

struct Job {
    int start, finish, profit;
};

class Solution {
public:
    int Weighted_Job_DP(vector<Job>& jobs) {
        /*
        DP with linear search
        Time Complexity: O(n^2)
        Space Complexity: O(n)
        */
        int n = jobs.size();
        sort(jobs.begin(), jobs.end(), [](const Job& a, const Job& b) {
            return a.finish < b.finish;
        });
        
        vector<int> dp(n);
        dp[0] = jobs[0].profit;
        
        for (int i = 1; i < n; i++) {
            int include = jobs[i].profit;
            int lastNonConflicting = -1;
            
            for (int j = i - 1; j >= 0; j--) {
                if (jobs[j].finish <= jobs[i].start) {
                    lastNonConflicting = j;
                    break;
                }
            }
            
            if (lastNonConflicting != -1) {
                include += dp[lastNonConflicting];
            }
            
            dp[i] = max(dp[i - 1], include);
        }
        
        return dp[n - 1];
    }
    
    int Weighted_Job_Binary_Search(vector<Job>& jobs) {
        /*
        DP with binary search
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        */
        int n = jobs.size();
        sort(jobs.begin(), jobs.end(), [](const Job& a, const Job& b) {
            return a.finish < b.finish;
        });
        
        vector<int> dp(n);
        dp[0] = jobs[0].profit;
        
        for (int i = 1; i < n; i++) {
            int include = jobs[i].profit;
            
            int left = 0, right = i - 1, lastNonConflicting = -1;
            while (left <= right) {
                int mid = left + (right - left) / 2;
                if (jobs[mid].finish <= jobs[i].start) {
                    lastNonConflicting = mid;
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
            
            if (lastNonConflicting != -1) {
                include += dp[lastNonConflicting];
            }
            
            dp[i] = max(dp[i - 1], include);
        }
        
        return dp[n - 1];
    }
};

void Test_Weighted_Job() {
    Solution solution;
    
    vector<Job> jobs = {{1, 2, 50}, {3, 5, 20}, {6, 19, 100}, {2, 100, 200}};
    cout << "DP: " << solution.Weighted_Job_DP(jobs) << endl;
    
    vector<Job> jobs2 = {{1, 2, 50}, {3, 5, 20}, {6, 19, 100}, {2, 100, 200}};
    cout << "Binary Search: " << solution.Weighted_Job_Binary_Search(jobs2) << endl;
}

int main() {
    Test_Weighted_Job();
    return 0;
}
