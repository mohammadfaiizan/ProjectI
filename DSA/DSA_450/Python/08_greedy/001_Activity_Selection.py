"""
Problem: Activity Selection
URL: https://practice.geeksforgeeks.org/problems/n-meetings-in-one-room-1587115620/1

Problem Statement:
Given N activities with start and end times, find the maximum number of activities that can be performed by a single person, assuming that a person can only work on a single activity at a time.

Sample Input/Output:
Input: start[] = {1, 3, 0, 5, 8, 5}, end[] = {2, 4, 6, 7, 9, 9}
Output: 4
Explanation: Activities that can be performed are: (1,2), (3,4), (5,7), (8,9)
"""


class Solution:
    def Max_Meetings_Greedy(self, start, end, n):
        """
        Sort activities by finish time, then greedily select non-overlapping activities
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        activities = [(end[i], start[i]) for i in range(n)]
        activities.sort()
        
        count = 1
        last_end = activities[0][0]
        
        for i in range(1, n):
            if activities[i][1] > last_end:
                count += 1
                last_end = activities[i][0]
        
        return count


def Test_Activity_Selection():
    solution = Solution()
    start = [1, 3, 0, 5, 8, 5]
    end = [2, 4, 6, 7, 9, 9]
    n = len(start)
    print(f"Max meetings: {solution.Max_Meetings_Greedy(start, end, n)}")


if __name__ == "__main__":
    Test_Activity_Selection()
