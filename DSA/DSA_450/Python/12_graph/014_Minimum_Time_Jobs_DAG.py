"""
Problem: Minimum Time Taken by Each Job in a DAG
URL: https://www.geeksforgeeks.org/find-the-ordering-of-tasks-from-given-dependencies/

Problem Statement:
Given a DAG of jobs with dependencies, find the minimum time taken by each job. Each job takes 1 unit of time. A job can only start after all its dependencies are completed.

Sample Input/Output:
Input: 10 jobs, dependencies: 1->2, 1->3, 2->4, 3->4, 4->5
Output: Job 1: 1, Job 2: 2, Job 3: 2, Job 4: 3, Job 5: 4
"""

from collections import deque


class Solution:
    def Min_Time_Jobs_Topological(self, V, adj):
        """
        Topological sort + BFS level computation
        Time Complexity: O(V+E)
        Space Complexity: O(V)
        """
        inDegree = [0] * V
        for u in range(V):
            for v in adj[u]:
                inDegree[v] += 1
        
        q = deque()
        time = [0] * V
        
        for i in range(V):
            if inDegree[i] == 0:
                q.append(i)
                time[i] = 1
        
        while q:
            u = q.popleft()
            
            for v in adj[u]:
                inDegree[v] -= 1
                if inDegree[v] == 0:
                    time[v] = time[u] + 1
                    q.append(v)
        
        return time


def Test_Min_Time_Jobs():
    solution = Solution()
    
    print("Test Case 1: 10 jobs with dependencies")
    V = 10
    adj = [[] for _ in range(V)]
    adj[0].append(1)
    adj[0].append(2)
    adj[1].append(3)
    adj[2].append(3)
    adj[3].append(4)
    
    time = solution.Min_Time_Jobs_Topological(V, adj)
    for i in range(V):
        print(f"Job {i}: {time[i]} unit(s)")
    
    print("\nTest Case 2: Linear chain")
    V2 = 5
    adj2 = [[] for _ in range(V2)]
    adj2[0].append(1)
    adj2[1].append(2)
    adj2[2].append(3)
    adj2[3].append(4)
    
    time2 = solution.Min_Time_Jobs_Topological(V2, adj2)
    for i in range(V2):
        print(f"Job {i}: {time2[i]} unit(s)")


if __name__ == "__main__":
    Test_Min_Time_Jobs()
