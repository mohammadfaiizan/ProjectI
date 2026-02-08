"""
Problem: Euler Path and Circuit (Seven Bridges of Konigsberg)
URL: https://www.geeksforgeeks.org/eulerian-path-and-circuit/

Problem Statement:
Determine if a graph has an Eulerian Circuit (all vertices have even degree), Eulerian Path (exactly 2 vertices have odd degree), or neither.

Sample Input/Output:
Input: Graph with edges
Output: Eulerian Circuit, Eulerian Path, or Neither
"""


class Solution:
    def DFS(self, u, adj, visited):
        visited[u] = True
        for v in adj[u]:
            if not visited[v]:
                self.DFS(v, adj, visited)

    def Is_Connected(self, V, adj):
        visited = [False] * V
        start = -1
        for i in range(V):
            if len(adj[i]) > 0:
                start = i
                break
        if start == -1:
            return True
        
        self.DFS(start, adj, visited)
        
        for i in range(V):
            if len(adj[i]) > 0 and not visited[i]:
                return False
        return True

    def Euler_Check(self, V, edges):
        """
        Check connectivity + count odd-degree vertices
        Time Complexity: O(V+E)
        Space Complexity: O(V+E)
        """
        adj = [[] for _ in range(V)]
        degree = [0] * V
        
        for edge in edges:
            u, v = edge[0], edge[1]
            adj[u].append(v)
            adj[v].append(u)
            degree[u] += 1
            degree[v] += 1
        
        if not self.Is_Connected(V, adj):
            return "Neither"
        
        oddCount = 0
        for i in range(V):
            if degree[i] % 2 != 0:
                oddCount += 1
        
        if oddCount == 0:
            return "Eulerian Circuit"
        elif oddCount == 2:
            return "Eulerian Path"
        else:
            return "Neither"


def Test_Euler_Check():
    solution = Solution()
    
    print("Test Case 1: Eulerian Circuit")
    V1 = 3
    edges1 = [(0, 1), (1, 2), (2, 0)]
    result1 = solution.Euler_Check(V1, edges1)
    print(f"Result: {result1}")
    print()
    
    print("Test Case 2: Eulerian Path")
    V2 = 4
    edges2 = [(0, 1), (1, 2), (2, 3)]
    result2 = solution.Euler_Check(V2, edges2)
    print(f"Result: {result2}")
    print()
    
    print("Test Case 3: Neither")
    V3 = 4
    edges3 = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
    result3 = solution.Euler_Check(V3, edges3)
    print(f"Result: {result3}")
    print()
    
    print("Test Case 4: Eulerian Circuit (Complex)")
    V4 = 5
    edges4 = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2), (1, 3)]
    result4 = solution.Euler_Check(V4, edges4)
    print(f"Result: {result4}")


if __name__ == "__main__":
    Test_Euler_Check()
