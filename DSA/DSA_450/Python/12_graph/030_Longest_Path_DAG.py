"""
Problem: Longest Path in a Directed Acyclic Graph
URL: https://www.geeksforgeeks.org/find-longest-path-directed-acyclic-graph/

Problem Statement:
Find longest path from a given source vertex in a Directed Acyclic Graph (DAG). The graph has weighted edges.

Sample Input/Output:
Input: V=6, edges = [[0,1,5],[0,2,3],[1,3,6],[1,2,2],[2,4,4],[2,5,2],[2,3,7],[3,5,1],[3,4,-1],[4,5,-2]], src=1
Output: [0, 5, 3, 11, 7, 9]
"""

from collections import deque


class Solution:
    def Longest_Path_Topological(self, V, edges, src):
        """
        Topological sort + DP relaxation with negated weights
        Time Complexity: O(V+E)
        Space Complexity: O(V+E)
        """
        adj = [[] for _ in range(V)]
        for e in edges:
            adj[e[0]].append((e[1], e[2]))
        
        indegree = [0] * V
        for e in edges:
            indegree[e[1]] += 1
        
        q = deque()
        for i in range(V):
            if indegree[i] == 0:
                q.append(i)
        
        topo = []
        while q:
            u = q.popleft()
            topo.append(u)
            
            for v, w in adj[u]:
                indegree[v] -= 1
                if indegree[v] == 0:
                    q.append(v)
        
        dist = [float('-inf')] * V
        dist[src] = 0
        
        for u in topo:
            if dist[u] != float('-inf'):
                for v, w in adj[u]:
                    if dist[u] + w > dist[v]:
                        dist[v] = dist[u] + w
        
        return dist


def Test_Longest_Path_Topological():
    solution = Solution()
    
    V1 = 6
    edges1 = [[0, 1, 5], [0, 2, 3], [1, 3, 6], [1, 2, 2], [2, 4, 4], [2, 5, 2], [2, 3, 7], [3, 5, 1], [3, 4, -1], [4, 5, -2]]
    result1 = solution.Longest_Path_Topological(V1, edges1, 1)
    print("Test 1 Longest paths from src=1:", end=" ")
    for d in result1:
        if d == float('-inf'):
            print("-INF", end=" ")
        else:
            print(int(d), end=" ")
    print()
    
    V2 = 4
    edges2 = [[0, 1, 1], [0, 2, 4], [1, 2, 2], [1, 3, 5], [2, 3, 1]]
    result2 = solution.Longest_Path_Topological(V2, edges2, 0)
    print("Test 2 Longest paths from src=0:", end=" ")
    for d in result2:
        if d == float('-inf'):
            print("-INF", end=" ")
        else:
            print(int(d), end=" ")
    print()
    
    V3 = 3
    edges3 = [[0, 1, 10], [1, 2, 20]]
    result3 = solution.Longest_Path_Topological(V3, edges3, 0)
    print("Test 3 Longest paths from src=0:", end=" ")
    for d in result3:
        if d == float('-inf'):
            print("-INF", end=" ")
        else:
            print(int(d), end=" ")
    print()


if __name__ == "__main__":
    Test_Longest_Path_Topological()
