"""
Problem: Find Bridges in a Graph (Cut Edges)
URL: https://www.geeksforgeeks.org/bridge-in-a-graph/

Problem Statement:
Find all bridges (edges whose removal disconnects the graph) in an undirected graph using Tarjan's algorithm.

Sample Input/Output:
Input: V=4, edges = [[0,1],[1,2],[2,3]]
Output: [[0,1],[1,2],[2,3]]
"""


class Solution:
    def Bridges_Tarjan(self, V, edges):
        """
        DFS with disc[] and low[]
        Time Complexity: O(V+E)
        Space Complexity: O(V+E)
        """
        adj = [[] for _ in range(V)]
        for e in edges:
            adj[e[0]].append(e[1])
            adj[e[1]].append(e[0])
        
        disc = [-1] * V
        low = [-1] * V
        parent = [-1] * V
        bridges = []
        time = [0]
        
        def dfs(u):
            disc[u] = low[u] = time[0]
            time[0] += 1
            
            for v in adj[u]:
                if disc[v] == -1:
                    parent[v] = u
                    dfs(v)
                    low[u] = min(low[u], low[v])
                    
                    if low[v] > disc[u]:
                        bridges.append([u, v])
                elif v != parent[u]:
                    low[u] = min(low[u], disc[v])
        
        for i in range(V):
            if disc[i] == -1:
                dfs(i)
        
        return bridges


def Test_Bridges_Tarjan():
    solution = Solution()
    
    V1 = 4
    edges1 = [[0, 1], [1, 2], [2, 3]]
    result1 = solution.Bridges_Tarjan(V1, edges1)
    print("Test 1 Bridges:", end=" ")
    for b in result1:
        print(f"[{b[0]},{b[1]}]", end=" ")
    print()
    
    V2 = 5
    edges2 = [[0, 1], [1, 2], [2, 0], [1, 3], [3, 4]]
    result2 = solution.Bridges_Tarjan(V2, edges2)
    print("Test 2 Bridges:", end=" ")
    for b in result2:
        print(f"[{b[0]},{b[1]}]", end=" ")
    print()
    
    V3 = 3
    edges3 = [[0, 1], [1, 2], [2, 0]]
    result3 = solution.Bridges_Tarjan(V3, edges3)
    print("Test 3 Bridges:", end=" ")
    for b in result3:
        print(f"[{b[0]},{b[1]}]", end=" ")
    print()


if __name__ == "__main__":
    Test_Bridges_Tarjan()
