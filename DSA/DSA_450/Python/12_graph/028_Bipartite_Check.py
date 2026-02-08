"""
Problem: Check if Graph is Bipartite
URL: https://leetcode.com/problems/is-graph-bipartite/

Problem Statement:
Check if a graph can be 2-colored (bipartite). A graph is bipartite if we can split its set of nodes into two independent subsets A and B such that every edge in the graph has one node in A and another node in B.

Sample Input/Output:
Input: graph = [[1,2,3],[0,2],[0,1,3],[0,2]]
Output: false
"""

from collections import deque


class Solution:
    def Bipartite_BFS(self, graph):
        """
        BFS coloring
        Time Complexity: O(V+E)
        Space Complexity: O(V)
        """
        n = len(graph)
        color = [-1] * n
        
        for i in range(n):
            if color[i] == -1:
                q = deque()
                q.append(i)
                color[i] = 0
                
                while q:
                    u = q.popleft()
                    
                    for v in graph[u]:
                        if color[v] == -1:
                            color[v] = 1 - color[u]
                            q.append(v)
                        elif color[v] == color[u]:
                            return False
        
        return True
    
    def Bipartite_DFS(self, graph):
        """
        DFS coloring
        Time Complexity: O(V+E)
        Space Complexity: O(V)
        """
        n = len(graph)
        color = [-1] * n
        
        def dfs(u, c):
            color[u] = c
            for v in graph[u]:
                if color[v] == -1:
                    if not dfs(v, 1 - c):
                        return False
                elif color[v] == c:
                    return False
            return True
        
        for i in range(n):
            if color[i] == -1:
                if not dfs(i, 0):
                    return False
        
        return True


def Test_Bipartite_BFS():
    solution = Solution()
    
    graph1 = [[1, 2, 3], [0, 2], [0, 1, 3], [0, 2]]
    print(f"Test 1 BFS: {solution.Bipartite_BFS(graph1)}")
    
    graph2 = [[1, 3], [0, 2], [1, 3], [0, 2]]
    print(f"Test 2 BFS: {solution.Bipartite_BFS(graph2)}")
    
    graph3 = [[1], [0, 2], [1]]
    print(f"Test 3 DFS: {solution.Bipartite_DFS(graph3)}")
    
    graph4 = [[1, 2], [0, 2], [0, 1]]
    print(f"Test 4 DFS: {solution.Bipartite_DFS(graph4)}")


if __name__ == "__main__":
    Test_Bipartite_BFS()
