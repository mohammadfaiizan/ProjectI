"""
Problem: Number of Operations to Make Network Connected
URL: https://leetcode.com/problems/number-of-operations-to-make-network-connected/

Problem Statement:
Given n computers and connections, find minimum connections to move to connect all. If not enough cables, return -1.

Sample Input/Output:
Input: n=4, connections=[[0,1],[0,2],[1,2]]
Output: 1
"""


class Solution:
    def Wired_Connections_DFS_Helper(self, node, adj, visited):
        visited[node] = True
        for neighbor in adj[node]:
            if not visited[neighbor]:
                self.Wired_Connections_DFS_Helper(neighbor, adj, visited)

    def Wired_Connections_DFS(self, n, connections):
        """
        Count Components - Need components-1 extra cables
        Time Complexity: O(V+E)
        Space Complexity: O(V+E)
        """
        if len(connections) < n - 1:
            return -1
        
        adj = [[] for _ in range(n)]
        for u, v in connections:
            adj[u].append(v)
            adj[v].append(u)
        
        visited = [False] * n
        components = 0
        
        for i in range(n):
            if not visited[i]:
                components += 1
                self.Wired_Connections_DFS_Helper(i, adj, visited)
        
        return components - 1

    def Find_Parent(self, parent, x):
        if parent[x] != x:
            parent[x] = self.Find_Parent(parent, parent[x])
        return parent[x]

    def Union_Set(self, parent, rank, x, y):
        px = self.Find_Parent(parent, x)
        py = self.Find_Parent(parent, y)
        
        if px == py:
            return
        
        if rank[px] < rank[py]:
            parent[px] = py
        elif rank[px] > rank[py]:
            parent[py] = px
        else:
            parent[py] = px
            rank[px] += 1

    def Wired_Connections_Union_Find(self, n, connections):
        """
        DSU-based
        Time Complexity: O(V+E)
        Space Complexity: O(V)
        """
        if len(connections) < n - 1:
            return -1
        
        parent = list(range(n))
        rank = [0] * n
        
        for u, v in connections:
            self.Union_Set(parent, rank, u, v)
        
        components = 0
        for i in range(n):
            if parent[i] == i:
                components += 1
        
        return components - 1


def Test_Making_Wired_Connections():
    solution = Solution()
    
    print("Test 1: n=4, connections=[[0,1],[0,2],[1,2]]")
    n1 = 4
    connections1 = [[0, 1], [0, 2], [1, 2]]
    result1 = solution.Wired_Connections_DFS(n1, connections1)
    print(f"Minimum operations (DFS): {result1}")
    
    result1_uf = solution.Wired_Connections_Union_Find(n1, connections1)
    print(f"Minimum operations (Union-Find): {result1_uf}")
    
    print("\nTest 2: n=6, connections=[[0,1],[0,2],[0,3],[1,2]]")
    n2 = 6
    connections2 = [[0, 1], [0, 2], [0, 3], [1, 2]]
    result2 = solution.Wired_Connections_DFS(n2, connections2)
    print(f"Minimum operations (DFS): {result2}")
    
    result2_uf = solution.Wired_Connections_Union_Find(n2, connections2)
    print(f"Minimum operations (Union-Find): {result2_uf}")
    
    print("\nTest 3: Not enough cables")
    n3 = 5
    connections3 = [[0, 1], [0, 2]]
    result3 = solution.Wired_Connections_DFS(n3, connections3)
    print(f"Minimum operations: {result3}")


if __name__ == "__main__":
    Test_Making_Wired_Connections()
