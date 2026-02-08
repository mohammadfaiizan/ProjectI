"""
Problem: Weighted Job Scheduling
URL: https://www.geeksforgeeks.org/weighted-job-scheduling-log-n-time/

Problem Statement:
Find maximum profit from non-overlapping jobs.
Each job has start time, finish time, and profit.
Sort by finish time, use DP with binary search.

Sample Input:
jobs = [(1, 3, 5), (2, 5, 6), (4, 6, 5), (6, 7, 4)]

Sample Output:
10
"""


class Solution:
    def Solve_DP_Binary_Search(self, jobs):
        """
        Approach: Sort jobs by finish time. For each job, use binary search
        to find last non-overlapping job, then use DP to maximize profit.
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        jobs_sorted = sorted(jobs, key=lambda x: x[1])
        
        n = len(jobs_sorted)
        dp = [0] * n
        dp[0] = jobs_sorted[0][2]
        
        for i in range(1, n):
            profit_including_current = jobs_sorted[i][2]
            last_non_overlapping = self.Find_Last_Non_Overlapping(jobs_sorted, i)
            
            if last_non_overlapping != -1:
                profit_including_current += dp[last_non_overlapping]
            
            dp[i] = max(dp[i - 1], profit_including_current)
        
        return dp[n - 1]
    
    def Find_Last_Non_Overlapping(self, jobs, index):
        left = 0
        right = index - 1
        result = -1
        
        while left <= right:
            mid = left + (right - left) // 2
            
            if jobs[mid][1] <= jobs[index][0]:
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        
        return result


def Test_Weighted_Job_Scheduling():
    sol = Solution()
    
    jobs1 = [
        (1, 3, 5),
        (2, 5, 6),
        (4, 6, 5),
        (6, 7, 4)
    ]
    assert sol.Solve_DP_Binary_Search(jobs1) == 10
    
    jobs2 = [
        (1, 2, 50),
        (3, 5, 20),
        (6, 19, 100),
        (2, 100, 200)
    ]
    result2 = sol.Solve_DP_Binary_Search(jobs2)
    assert result2 == 250
    
    jobs3 = [
        (1, 4, 3),
        (2, 6, 5),
        (4, 7, 2),
        (6, 8, 6)
    ]
    assert sol.Solve_DP_Binary_Search(jobs3) == 9
    
    jobs4 = [
        (1, 3, 10)
    ]
    assert sol.Solve_DP_Binary_Search(jobs4) == 10
    
    print("All test cases passed!")


if __name__ == "__main__":
    Test_Weighted_Job_Scheduling()
