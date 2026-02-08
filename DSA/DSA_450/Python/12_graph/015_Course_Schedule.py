"""
Problem: Find Whether It Is Possible to Finish All Tasks (Course Schedule)
URL: https://leetcode.com/problems/course-schedule/

Problem Statement:
Given numCourses and prerequisites array, determine if all courses can be finished. This is equivalent to checking if the dependency graph has a cycle.

Sample Input/Output:
Input: numCourses=4, prerequisites=[[1,0],[2,1],[3,2]]
Output: true
Input: numCourses=2, prerequisites=[[1,0],[0,1]]
Output: false
"""


class Solution:
    def Course_Schedule_DFS(self, numCourses, prerequisites):
        """
        Cycle detection with coloring
        Time Complexity: O(V+E)
        Space Complexity: O(V)
        """
        adj = [[] for _ in range(numCourses)]
        for edge in prerequisites:
            adj[edge[1]].append(edge[0])
        
        color = [0] * numCourses
        
        def has_cycle(u):
            if color[u] == 1:
                return True
            if color[u] == 2:
                return False
            
            color[u] = 1
            for v in adj[u]:
                if has_cycle(v):
                    return True
            color[u] = 2
            return False
        
        for i in range(numCourses):
            if color[i] == 0 and has_cycle(i):
                return False
        return True
    
    def Course_Schedule_BFS_Kahn(self, numCourses, prerequisites):
        """
        Kahn's topological sort
        Time Complexity: O(V+E)
        Space Complexity: O(V)
        """
        adj = [[] for _ in range(numCourses)]
        in_degree = [0] * numCourses
        
        for edge in prerequisites:
            adj[edge[1]].append(edge[0])
            in_degree[edge[0]] += 1
        
        from collections import deque
        q = deque()
        for i in range(numCourses):
            if in_degree[i] == 0:
                q.append(i)
        
        count = 0
        while q:
            u = q.popleft()
            count += 1
            
            for v in adj[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    q.append(v)
        
        return count == numCourses


def Test_Course_Schedule():
    solution = Solution()
    
    print("Test Case 1: Valid schedule (no cycle)")
    numCourses1 = 4
    prerequisites1 = [[1, 0], [2, 1], [3, 2]]
    print("DFS Result:", solution.Course_Schedule_DFS(numCourses1, prerequisites1))
    print("BFS Result:", solution.Course_Schedule_BFS_Kahn(numCourses1, prerequisites1))
    
    print("\nTest Case 2: Invalid schedule (cycle)")
    numCourses2 = 2
    prerequisites2 = [[1, 0], [0, 1]]
    print("DFS Result:", solution.Course_Schedule_DFS(numCourses2, prerequisites2))
    print("BFS Result:", solution.Course_Schedule_BFS_Kahn(numCourses2, prerequisites2))
    
    print("\nTest Case 3: No prerequisites")
    numCourses3 = 3
    prerequisites3 = []
    print("DFS Result:", solution.Course_Schedule_DFS(numCourses3, prerequisites3))
    print("BFS Result:", solution.Course_Schedule_BFS_Kahn(numCourses3, prerequisites3))


if __name__ == "__main__":
    Test_Course_Schedule()
