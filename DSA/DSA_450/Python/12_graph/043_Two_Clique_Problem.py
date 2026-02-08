"""
Problem: Two Clique Problem
URL: https://www.geeksforgeeks.org/two-clique-problem-check-graph-can-divided-two-cliques/

Problem Statement:
Check if vertices of a graph can be divided into two cliques.

Sample Input/Output:
Input: Graph with edges
Output: True if can be divided, False otherwise
"""

from collections import deque


class Solution:
    def Is_Bipartite(self, V, adj):
        color = [-1] * V
        q = deque()
        
        for start in range(V):
            if color[start] != -1:
                continue
            
            color[start] = 0
            q.append(start)
            
            while q:
                u = q.popleft()
                
                for v in adj[u]:
                    if color[v] == -1:
                        color[v] = 1 - color[u]
                        q.append(v)
                    elif color[v] == color[u]:
                        return False
        
        return True

    def Two_Clique_Complement_Bipartite(self, V, edges):
        """
        Complement + BFS bipartite check
        Time Complexity: O(V^2)
        Space Complexity: O(V^2)
        """
        original = [[False] * V for _ in range(V)]
        for edge in edges:
            u, v = edge[0], edge[1]
            original[u][v] = True
            original[v][u] = True
        
        complement = [[] for _ in range(V)]
        for i in range(V):
            for j in range(V):
                if i != j and not original[i][j]:
                    complement[i].append(j)
        
        return self.Is_Bipartite(V, complement)


def Test_Two_Clique():
    solution = Solution()
    
    print("Test Case 1: Can be divided into two cliques")
    V1 = 4
    edges1 = [(0, 1), (0, 2), (1, 2), (3, 0), (3, 1), (3, 2)]
    result1 = solution.Two_Clique_Complement_Bipartite(V1, edges1)
    print(f"Result: {result1}")
    print()
    
    print("Test Case 2: Cannot be divided")
    V2 = 5
    edges2 = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2)]
    result2 = solution.Two_Clique_Complement_Bipartite(V2, edges2)
    print(f"Result: {result2}")
    print()
    
    print("Test Case 3: Complete bipartite graph")
    V3 = 4
    edges3 = [(0, 2), (0, 3), (1, 2), (1, 3)]
    result3 = solution.Two_Clique_Complement_Bipartite(V3, edges3)
    print(f"Result: {result3}")
    print()
    
    print("Test Case 4: Two separate cliques")
    V4 = 6
    edges4 = [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]
    result4 = solution.Two_Clique_Complement_Bipartite(V4, edges4)
    print(f"Result: {result4}")


if __name__ == "__main__":
    Test_Two_Clique()
