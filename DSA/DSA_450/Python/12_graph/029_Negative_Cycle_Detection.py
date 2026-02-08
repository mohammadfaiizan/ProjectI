"""
Problem: Detect Negative Weight Cycle in a Graph
URL: https://www.geeksforgeeks.org/detect-negative-cycle-graph-bellman-ford/

Problem Statement:
Detect if a graph contains a negative weight cycle. A negative weight cycle is a cycle whose edges sum to a negative value.

Sample Input/Output:
Input: V=4, edges = [[0,1,1],[1,2,-1],[2,3,-1],[3,0,-1]]
Output: true (negative cycle exists)
"""


class Solution:
    def Negative_Cycle_Bellman_Ford(self, V, edges, src):
        """
        Run Bellman-Ford, check if Vth relaxation reduces any distance
        Time Complexity: O(V*E)
        Space Complexity: O(V)
        """
        dist = [float('inf')] * V
        dist[src] = 0
        
        for _ in range(V - 1):
            for e in edges:
                u, v, w = e[0], e[1], e[2]
                if dist[u] != float('inf') and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
        
        for e in edges:
            u, v, w = e[0], e[1], e[2]
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                return True
        
        return False
    
    def Negative_Cycle_Floyd_Warshall(self, V, graph):
        """
        Check diagonal of all-pairs matrix for negative
        Time Complexity: O(V^3)
        Space Complexity: O(V^2)
        """
        dist = [row[:] for row in graph]
        
        for k in range(V):
            for i in range(V):
                for j in range(V):
                    if dist[i][k] != float('inf') and dist[k][j] != float('inf'):
                        dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
        
        for i in range(V):
            if dist[i][i] < 0:
                return True
        
        return False


def Test_Negative_Cycle_Bellman_Ford():
    solution = Solution()
    
    V1 = 4
    edges1 = [[0, 1, 1], [1, 2, -1], [2, 3, -1], [3, 0, -1]]
    print(f"Test 1 Bellman-Ford: {solution.Negative_Cycle_Bellman_Ford(V1, edges1, 0)}")
    
    V2 = 3
    edges2 = [[0, 1, 1], [1, 2, 2], [2, 0, 3]]
    print(f"Test 2 Bellman-Ford: {solution.Negative_Cycle_Bellman_Ford(V2, edges2, 0)}")
    
    V3 = 4
    graph3 = [[float('inf')] * V3 for _ in range(V3)]
    graph3[0][1] = 1
    graph3[1][2] = -1
    graph3[2][3] = -1
    graph3[3][0] = -1
    for i in range(V3):
        graph3[i][i] = 0
    print(f"Test 3 Floyd-Warshall: {solution.Negative_Cycle_Floyd_Warshall(V3, graph3)}")


if __name__ == "__main__":
    Test_Negative_Cycle_Bellman_Ford()
