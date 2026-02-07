/*
 * Problem: Weighted Job Scheduling
 * URL: https://www.geeksforgeeks.org/weighted-job-scheduling-log-n-time/
 * Problem Statement:
 * Find maximum profit from non-overlapping jobs.
 * Each job has start time, finish time, and profit.
 * Sort by finish time, use DP with binary search.
 * 
 * Sample Input:
 * jobs = [(1, 3, 5), (2, 5, 6), (4, 6, 5), (6, 7, 4)]
 * 
 * Sample Output:
 * 10
 */

#include <bits/stdc++.h>
using namespace std;

struct Job {
    int start, finish, profit;
};

class Solution {
public:
    int Solve_DP_Binary_Search(vector<Job>& jobs) {
        /*
         * Approach: Sort jobs by finish time. For each job, use binary search
         * to find last non-overlapping job, then use DP to maximize profit.
         * Time Complexity: O(n log n)
         * Space Complexity: O(n)
         */
        sort(jobs.begin(), jobs.end(), [](const Job& a, const Job& b) {
            return a.finish < b.finish;
        });
        
        int n = jobs.size();
        vector<int> dp(n, 0);
        dp[0] = jobs[0].profit;
        
        for (int i = 1; i < n; i++) {
            int profit_including_current = jobs[i].profit;
            int last_non_overlapping = Find_Last_Non_Overlapping(jobs, i);
            
            if (last_non_overlapping != -1) {
                profit_including_current += dp[last_non_overlapping];
            }
            
            dp[i] = max(dp[i - 1], profit_including_current);
        }
        
        return dp[n - 1];
    }
    
private:
    int Find_Last_Non_Overlapping(vector<Job>& jobs, int index) {
        int left = 0, right = index - 1;
        int result = -1;
        
        while (left <= right) {
            int mid = left + (right - left) / 2;
            
            if (jobs[mid].finish <= jobs[index].start) {
                result = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        
        return result;
    }
};

void Test_Weighted_Job_Scheduling() {
    Solution sol;
    
    vector<Job> jobs1 = {
        {1, 3, 5},
        {2, 5, 6},
        {4, 6, 5},
        {6, 7, 4}
    };
    assert(sol.Solve_DP_Binary_Search(jobs1) == 10);
    
    vector<Job> jobs2 = {
        {1, 2, 50},
        {3, 5, 20},
        {6, 19, 100},
        {2, 100, 200}
    };
    int result2 = sol.Solve_DP_Binary_Search(jobs2);
    assert(result2 == 250);
    
    vector<Job> jobs3 = {
        {1, 4, 3},
        {2, 6, 5},
        {4, 7, 2},
        {6, 8, 6}
    };
    assert(sol.Solve_DP_Binary_Search(jobs3) == 9);
    
    vector<Job> jobs4 = {
        {1, 3, 10}
    };
    assert(sol.Solve_DP_Binary_Search(jobs4) == 10);
    
    cout << "All test cases passed!" << endl;
}

int main() {
    Test_Weighted_Job_Scheduling();
    return 0;
}
