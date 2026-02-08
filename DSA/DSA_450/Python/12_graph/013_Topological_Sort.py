"""
Problem: Topological Sort
URL: https://practice.geeksforgeeks.org/problems/topological-sort/1

Problem Statement:
Given a Directed Acyclic Graph (DAG), perform topological sort to find a linear ordering of vertices such that for every directed edge (u, v), vertex u comes before v in the ordering.

Sample Input/Output:
Input: Graph with 6 vertices, edges: 5->0, 5->2, 4->0, 4->1, 2->3, 3->1
Output: 5 4 2 3 1 0 (or other valid topological order)
"""

from collections import deque


class Solution:
    def Topological_Sort_DFS(self, V, adj):
        """
        DFS with stack
        Time Complexity: O(V+E)
        Space Complexity: O(V)
        """
        visited = [False] * V
        st = []
        
        def dfs(u):
            visited[u] = True
            for v in adj[u]:
                if not visited[v]:
                    dfs(v)
            st.append(u)
        
        for i in range(V):
            if not visited[i]:
                dfs(i)
        
        return st[::-1]
    
    def Topological_Sort_BFS_Kahn(self, V, adj):
        """
        Kahn's algorithm with in-degree
        Time Complexity: O(V+E)
        Space Complexity: O(V)
        """
        inDegree = [0] * V
        for u in range(V):
            for v in adj[u]:
                inDegree[v] += 1
        
        q = deque()
        for i in range(V):
            if inDegree[i] == 0:
                q.append(i)
        
        result = []
        while q:
            u = q.popleft()
            result.append(u)
            
            for v in adj[u]:
                inDegree[v] -= 1
                if inDegree[v] == 0:
                    q.append(v)
        
        return result


def Test_Topological_Sort():
    solution = Solution()
    
    V = 6
    adj = [[] for _ in range(V)]
    adj[5].append(0)
    adj[5].append(2)
    adj[4].append(0)
    adj[4].append(1)
    adj[2].append(3)
    adj[3].append(1)
    
    print("Test Case 1: DAG with 6 vertices")
    
    result1 = solution.Topological_Sort_DFS(V, adj)
    print("DFS Topological Sort:", end=" ")
    for x in result1:
        print(x, end=" ")
    print()
    
    result2 = solution.Topological_Sort_BFS_Kahn(V, adj)
    print("BFS Kahn Topological Sort:", end=" ")
    for x in result2:
        print(x, end=" ")
    print()
    
    print("Test Case 2: Simple linear DAG")
    V2 = 4
    adj2 = [[] for _ in range(V2)]
    adj2[0].append(1)
    adj2[1].append(2)
    adj2[2].append(3)
    
    result3 = solution.Topological_Sort_DFS(V2, adj2)
    print("DFS Result:", end=" ")
    for x in result3:
        print(x, end=" ")
    print()


if __name__ == "__main__":
    Test_Topological_Sort()
