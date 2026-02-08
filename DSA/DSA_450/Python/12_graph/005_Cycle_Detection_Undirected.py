"""
Problem: Detect Cycle in an Undirected Graph
URL: https://practice.geeksforgeeks.org/problems/detect-cycle-in-an-undirected-graph/1

Problem Statement:
Detect if an undirected graph contains a cycle.

Sample Input/Output:
Input: Graph with cycle: 0-1-2-0
Output: Cycle detected: true
"""

from collections import deque


class Solution:
    def Cycle_Undirected_DFS_Helper(self, node, parent, adj, visited):
        visited[node] = True
        
        for neighbor in adj[node]:
            if not visited[neighbor]:
                if self.Cycle_Undirected_DFS_Helper(neighbor, node, adj, visited):
                    return True
            elif neighbor != parent:
                return True
        
        return False

    def Cycle_Undirected_DFS(self, V, adj):
        """
        DFS with Parent Tracking
        Time Complexity: O(V+E)
        Space Complexity: O(V)
        """
        visited = [False] * V
        
        for i in range(V):
            if not visited[i]:
                if self.Cycle_Undirected_DFS_Helper(i, -1, adj, visited):
                    return True
        
        return False

    def Cycle_Undirected_BFS(self, V, adj):
        """
        BFS with Parent Tracking
        Time Complexity: O(V+E)
        Space Complexity: O(V)
        """
        visited = [False] * V
        
        for i in range(V):
            if not visited[i]:
                q = deque()
                visited[i] = True
                q.append((i, -1))
                
                while q:
                    node, parent = q.popleft()
                    
                    for neighbor in adj[node]:
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            q.append((neighbor, node))
                        elif neighbor != parent:
                            return True
        
        return False

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

    def Cycle_Undirected_Union_Find(self, V, adj):
        """
        Union-Find / DSU
        Time Complexity: O(V+E)
        Space Complexity: O(V)
        """
        parent = list(range(V))
        rank = [0] * V
        
        for u in range(V):
            for v in adj[u]:
                if u < v:
                    pu = self.Find_Parent(parent, u)
                    pv = self.Find_Parent(parent, v)
                    
                    if pu == pv:
                        return True
                    
                    self.Union_Set(parent, rank, u, v)
        
        return False


def Test_Cycle_Detection_Undirected():
    solution = Solution()
    
    print("Test 1: Graph with Cycle")
    V1 = 4
    adj1 = [[] for _ in range(V1)]
    adj1[0] = [1, 2]
    adj1[1] = [0, 2]
    adj1[2] = [0, 1, 3]
    adj1[3] = [2]
    
    hasCycle1 = solution.Cycle_Undirected_DFS(V1, adj1)
    print(f"Cycle detected (DFS): {'Yes' if hasCycle1 else 'No'}")
    
    hasCycle1_bfs = solution.Cycle_Undirected_BFS(V1, adj1)
    print(f"Cycle detected (BFS): {'Yes' if hasCycle1_bfs else 'No'}")
    
    hasCycle1_uf = solution.Cycle_Undirected_Union_Find(V1, adj1)
    print(f"Cycle detected (Union-Find): {'Yes' if hasCycle1_uf else 'No'}")
    
    print("\nTest 2: Graph without Cycle")
    V2 = 4
    adj2 = [[] for _ in range(V2)]
    adj2[0] = [1]
    adj2[1] = [0, 2]
    adj2[2] = [1, 3]
    adj2[3] = [2]
    
    hasCycle2 = solution.Cycle_Undirected_DFS(V2, adj2)
    print(f"Cycle detected (DFS): {'Yes' if hasCycle2 else 'No'}")
    
    hasCycle2_bfs = solution.Cycle_Undirected_BFS(V2, adj2)
    print(f"Cycle detected (BFS): {'Yes' if hasCycle2_bfs else 'No'}")
    
    hasCycle2_uf = solution.Cycle_Undirected_Union_Find(V2, adj2)
    print(f"Cycle detected (Union-Find): {'Yes' if hasCycle2_uf else 'No'}")


if __name__ == "__main__":
    Test_Cycle_Detection_Undirected()
