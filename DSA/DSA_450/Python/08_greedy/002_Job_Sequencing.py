"""
Problem: Job Sequencing
URL: https://practice.geeksforgeeks.org/problems/job-sequencing-problem-1587115620/1

Problem Statement:
Given a set of N jobs where each job has a deadline and profit associated with it. Each job takes 1 unit of time to complete and only one job can be scheduled at a time. Find the maximum profit and the number of jobs done.

Sample Input/Output:
Input: N = 4, Jobs = {(1,4,20),(2,1,10),(3,1,40),(4,1,30)}
Output: 2 60
Explanation: Job1 and Job3 can be done with maximum profit of 60 (20+40).
"""


class Solution:
    def Job_Scheduling_Greedy(self, arr, n):
        """
        Sort by profit descending, greedily assign to latest available slot
        Time Complexity: O(n^2)
        Space Complexity: O(n)
        """
        arr.sort(key=lambda x: x.profit, reverse=True)
        
        max_deadline = max(job.dead for job in arr)
        
        slot = [-1] * (max_deadline + 1)
        count = 0
        profit = 0
        
        for i in range(n):
            for j in range(arr[i].dead, 0, -1):
                if slot[j] == -1:
                    slot[j] = arr[i].id
                    count += 1
                    profit += arr[i].profit
                    break
        
        return [count, profit]


class Job:
    def __init__(self, id, dead, profit):
        self.id = id
        self.dead = dead
        self.profit = profit


def Test_Job_Sequencing():
    solution = Solution()
    arr = [Job(1, 4, 20), Job(2, 1, 10), Job(3, 1, 40), Job(4, 1, 30)]
    n = 4
    result = solution.Job_Scheduling_Greedy(arr, n)
    print(f"Jobs done: {result[0]}, Profit: {result[1]}")


if __name__ == "__main__":
    Test_Job_Sequencing()
