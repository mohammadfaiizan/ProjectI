"""
Problem: Articulation Points (Cut Vertices)
URL: https://www.geeksforgeeks.org/articulation-points-or-cut-vertices-in-a-graph/

Problem Statement:
Find all articulation points (vertices whose removal disconnects the graph).

Sample Input/Output:
Input: Graph with edges
Output: List of articulation points
"""


class Solution:
    def DFS_Articulation(self, u, parent, adj, disc, low, visited, isArticulation, time):
        visited[u] = True
        disc[u] = low[u] = time[0]
        time[0] += 1
        children = 0
        
        for v in adj[u]:
            if not visited[v]:
                children += 1
                self.DFS_Articulation(v, u, adj, disc, low, visited, isArticulation, time)
                
                low[u] = min(low[u], low[v])
                
                if parent == -1 and children > 1:
                    isArticulation[u] = True
                
                if parent != -1 and low[v] >= disc[u]:
                    isArticulation[u] = True
            elif v != parent:
                low[u] = min(low[u], disc[v])

    def Articulation_Points_Tarjan(self, V, edges):
        """
        DFS with disc[] and low[]
        Time Complexity: O(V+E)
        Space Complexity: O(V+E)
        """
        adj = [[] for _ in range(V)]
        for edge in edges:
            u, v = edge[0], edge[1]
            adj[u].append(v)
            adj[v].append(u)
        
        disc = [-1] * V
        low = [-1] * V
        visited = [False] * V
        isArticulation = [False] * V
        time = [0]
        
        for i in range(V):
            if not visited[i]:
                self.DFS_Articulation(i, -1, adj, disc, low, visited, isArticulation, time)
        
        articulationPoints = []
        for i in range(V):
            if isArticulation[i]:
                articulationPoints.append(i)
        
        return articulationPoints


def Test_Articulation_Points():
    solution = Solution()
    
    print("Test Case 1:")
    V1 = 5
    edges1 = [(0, 1), (1, 2), (2, 0), (1, 3), (3, 4)]
    result1 = solution.Articulation_Points_Tarjan(V1, edges1)
    print("Articulation Points:", end=" ")
    for v in result1:
        print(v, end=" ")
    print()
    print()
    
    print("Test Case 2:")
    V2 = 7
    edges2 = [(0, 1), (1, 2), (2, 0), (1, 3), (1, 4), (1, 6), (3, 5), (4, 5)]
    result2 = solution.Articulation_Points_Tarjan(V2, edges2)
    print("Articulation Points:", end=" ")
    for v in result2:
        print(v, end=" ")
    print()
    print()
    
    print("Test Case 3:")
    V3 = 4
    edges3 = [(0, 1), (1, 2), (2, 3)]
    result3 = solution.Articulation_Points_Tarjan(V3, edges3)
    print("Articulation Points:", end=" ")
    for v in result3:
        print(v, end=" ")
    print()
    print()
    
    print("Test Case 4: No articulation points")
    V4 = 4
    edges4 = [(0, 1), (1, 2), (2, 3), (3, 0)]
    result4 = solution.Articulation_Points_Tarjan(V4, edges4)
    print("Articulation Points:", end=" ")
    if not result4:
        print("None")
    else:
        for v in result4:
            print(v, end=" ")
    print()


if __name__ == "__main__":
    Test_Articulation_Points()
