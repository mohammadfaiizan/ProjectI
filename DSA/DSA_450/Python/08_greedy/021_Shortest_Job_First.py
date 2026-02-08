"""
Problem: Shortest Job First
URL: https://www.geeksforgeeks.org/program-for-shortest-job-first-or-sjf-cpu-scheduling-set-1-non-preemptive/

Problem Statement:
Implement non-preemptive SJF CPU scheduling. Given processes with arrival and burst times, find order and waiting times.

Sample Input/Output:
Input: processes = {{1, 6}, {2, 8}, {3, 7}, {4, 3}}
Output: Process order: 1 4 3 2, Average waiting time: 7
Explanation: Process 1 arrives first, then 4 (shortest), then 3, then 2.
"""

import heapq


class Solution:
    def Shortest_Job_First_Min_Heap(self, processes):
        """
        Min-heap by burst time greedy approach
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        n = len(processes)
        process_list = [(processes[i][0], processes[i][1]) for i in range(n)]
        
        process_list.sort()
        
        pq = []
        order = []
        waiting_time = [0] * n
        
        current_time = 0
        idx = 0
        
        while idx < n or pq:
            while idx < n and process_list[idx][0] <= current_time:
                heapq.heappush(pq, (process_list[idx][1], idx))
                idx += 1
            
            if not pq:
                current_time = process_list[idx][0]
                continue
            
            current = heapq.heappop(pq)
            
            process_idx = current[1]
            burst_time = current[0]
            arrival_time = process_list[process_idx][0]
            
            waiting_time[process_idx] = current_time - arrival_time
            order.append(process_idx + 1)
            current_time += burst_time
        
        avg_waiting = sum(waiting_time) / n
        
        return (order, avg_waiting)


def Test_Shortest_Job_First():
    solution = Solution()
    
    processes1 = [[1, 6], [2, 8], [3, 7], [4, 3]]
    result1 = solution.Shortest_Job_First_Min_Heap(processes1)
    print("Test 1 - Process order:", end=" ")
    for p in result1[0]:
        print(p, end=" ")
    print(f", Average waiting time: {result1[1]}")
    
    processes2 = [[0, 3], [1, 6], [2, 4]]
    result2 = solution.Shortest_Job_First_Min_Heap(processes2)
    print("Test 2 - Process order:", end=" ")
    for p in result2[0]:
        print(p, end=" ")
    print(f", Average waiting time: {result2[1]}")


if __name__ == "__main__":
    Test_Shortest_Job_First()
