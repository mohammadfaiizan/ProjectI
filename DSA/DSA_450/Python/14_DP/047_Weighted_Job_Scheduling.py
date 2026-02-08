"""
Problem: Weighted Job Scheduling
URL: https://www.geeksforgeeks.org/weighted-job-scheduling/

Problem Statement:
Given N jobs where every job is represented by start time, finish time and profit. Find the maximum profit subset of jobs such that no two jobs in the subset overlap.

Sample Input/Output:
Input: jobs = [(1,2,50), (3,5,20), (6,19,100), (2,100,200)]
Output: 250
"""


class Job:
    def __init__(self, start: int, finish: int, profit: int):
        self.start = start
        self.finish = finish
        self.profit = profit


class Solution:
    def Weighted_Job_DP(self, jobs: list[Job]) -> int:
        """
        DP with linear search
        Time Complexity: O(n^2)
        Space Complexity: O(n)
        """
        n = len(jobs)
        jobs.sort(key=lambda x: x.finish)
        
        dp = [0] * n
        dp[0] = jobs[0].profit
        
        for i in range(1, n):
            include = jobs[i].profit
            last_non_conflicting = -1
            
            for j in range(i - 1, -1, -1):
                if jobs[j].finish <= jobs[i].start:
                    last_non_conflicting = j
                    break
            
            if last_non_conflicting != -1:
                include += dp[last_non_conflicting]
            
            dp[i] = max(dp[i - 1], include)
        
        return dp[n - 1]
    
    def Weighted_Job_Binary_Search(self, jobs: list[Job]) -> int:
        """
        DP with binary search
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        n = len(jobs)
        jobs.sort(key=lambda x: x.finish)
        
        dp = [0] * n
        dp[0] = jobs[0].profit
        
        for i in range(1, n):
            include = jobs[i].profit
            
            left = 0
            right = i - 1
            last_non_conflicting = -1
            
            while left <= right:
                mid = left + (right - left) // 2
                if jobs[mid].finish <= jobs[i].start:
                    last_non_conflicting = mid
                    left = mid + 1
                else:
                    right = mid - 1
            
            if last_non_conflicting != -1:
                include += dp[last_non_conflicting]
            
            dp[i] = max(dp[i - 1], include)
        
        return dp[n - 1]


def Test_WeightedJobScheduling():
    solution = Solution()
    
    jobs = [Job(1, 2, 50), Job(3, 5, 20), Job(6, 19, 100), Job(2, 100, 200)]
    result1 = solution.Weighted_Job_DP(jobs)
    assert result1 == 250
    
    jobs2 = [Job(1, 2, 50), Job(3, 5, 20), Job(6, 19, 100), Job(2, 100, 200)]
    result2 = solution.Weighted_Job_Binary_Search(jobs2)
    assert result2 == 250


if __name__ == "__main__":
    Test_WeightedJobScheduling()
