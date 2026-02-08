"""
Problem: Maximum Meetings
URL: https://www.geeksforgeeks.org/find-maximum-meetings-in-one-room/

Problem Statement:
Given N meetings with start and end times, find max meetings in one room. Print which meetings are selected.

Sample Input/Output:
Input: start[] = {1, 3, 0, 5, 8, 5}, end[] = {2, 4, 6, 7, 9, 9}
Output: 0 1 3 4
Explanation: Meetings at indices 0, 1, 3, 4 can be scheduled (4 meetings total).
"""


class Solution:
    def Maximum_Meetings_Sort_Finish_Time(self, start, end):
        """
        Sort by finish time greedy approach: Always pick meeting that ends earliest
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        n = len(start)
        meetings = [(end[i], start[i], i) for i in range(n)]
        
        meetings.sort()
        
        result = []
        last_end_time = -1
        
        for meeting in meetings:
            end_time, start_time, index = meeting
            
            if start_time > last_end_time:
                result.append(index)
                last_end_time = end_time
        
        return result


def Test_Maximum_Meetings():
    solution = Solution()
    
    start1 = [1, 3, 0, 5, 8, 5]
    end1 = [2, 4, 6, 7, 9, 9]
    result1 = solution.Maximum_Meetings_Sort_Finish_Time(start1, end1)
    print("Test 1 - Selected meetings:", end=" ")
    for idx in result1:
        print(idx, end=" ")
    print()
    
    start2 = [1, 2]
    end2 = [2, 3]
    result2 = solution.Maximum_Meetings_Sort_Finish_Time(start2, end2)
    print("Test 2 - Selected meetings:", end=" ")
    for idx in result2:
        print(idx, end=" ")
    print()


if __name__ == "__main__":
    Test_Maximum_Meetings()
