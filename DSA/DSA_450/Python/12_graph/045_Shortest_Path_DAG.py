"""
Problem: Shortest Path in a DAG with Weights
URL: https://www.geeksforgeeks.org/shortest-path-for-directed-acyclic-graph/

Problem Statement:
Find shortest paths from source in a weighted DAG.

Sample Input/Output:
Input: Weighted DAG with source vertex
Output: Shortest distances from source to all vertices
"""


class Solution:
    def Topological_Sort(self, u, adj, visited, st):
        visited[u] = True
        for neighbor in adj[u]:
            v = neighbor[0]
            if not visited[v]:
                self.Topological_Sort(v, adj, visited, st)
        st.append(u)

    def Shortest_Path_DAG_Topological(self, V, weightedEdges, src):
        """
        Topological sort + relax in order
        Time Complexity: O(V+E)
        Space Complexity: O(V+E)
        """
        adj = [[] for _ in range(V)]
        for edge in weightedEdges:
            u, v, w = edge[0][0], edge[0][1], edge[1]
            adj[u].append((v, w))
        
        visited = [False] * V
        st = []
        
        for i in range(V):
            if not visited[i]:
                self.Topological_Sort(i, adj, visited, st)
        
        dist = [float('inf')] * V
        dist[src] = 0
        
        while st:
            u = st.pop()
            
            if dist[u] != float('inf'):
                for neighbor in adj[u]:
                    v, w = neighbor[0], neighbor[1]
                    if dist[v] > dist[u] + w:
                        dist[v] = dist[u] + w
        
        return dist


def Test_Shortest_Path_DAG():
    solution = Solution()
    
    print("Test Case 1:")
    V1 = 6
    edges1 = [
        ((0, 1), 5), ((0, 2), 3), ((1, 3), 6), ((1, 2), 2),
        ((2, 4), 4), ((2, 5), 2), ((2, 3), 7), ((3, 4), -1),
        ((4, 5), -2)
    ]
    src1 = 1
    dist1 = solution.Shortest_Path_DAG_Topological(V1, edges1, src1)
    print(f"Source: {src1}")
    print("Distances:", end=" ")
    for i in range(V1):
        if dist1[i] == float('inf'):
            print(f"[{i}:INF]", end=" ")
        else:
            print(f"[{i}:{int(dist1[i])}]", end=" ")
    print()
    print()
    
    print("Test Case 2:")
    V2 = 4
    edges2 = [
        ((0, 1), 1), ((0, 2), 4), ((1, 2), 2), ((1, 3), 5), ((2, 3), 1)
    ]
    src2 = 0
    dist2 = solution.Shortest_Path_DAG_Topological(V2, edges2, src2)
    print(f"Source: {src2}")
    print("Distances:", end=" ")
    for i in range(V2):
        if dist2[i] == float('inf'):
            print(f"[{i}:INF]", end=" ")
        else:
            print(f"[{i}:{int(dist2[i])}]", end=" ")
    print()
    print()
    
    print("Test Case 3:")
    V3 = 5
    edges3 = [
        ((0, 1), 2), ((0, 2), 3), ((1, 3), 1), ((2, 3), 4), ((3, 4), 2)
    ]
    src3 = 0
    dist3 = solution.Shortest_Path_DAG_Topological(V3, edges3, src3)
    print(f"Source: {src3}")
    print("Distances:", end=" ")
    for i in range(V3):
        if dist3[i] == float('inf'):
            print(f"[{i}:INF]", end=" ")
        else:
            print(f"[{i}:{int(dist3[i])}]", end=" ")
    print()


if __name__ == "__main__":
    Test_Shortest_Path_DAG()
