"""
Problem: Bellman-Ford Algorithm
URL: https://practice.geeksforgeeks.org/problems/distance-from-the-source-bellman-ford-algorithm/1

Problem Statement:
Find single-source shortest paths from a source vertex to all other vertices in a weighted directed graph. The algorithm can handle negative weight edges and detect negative cycles.

Sample Input/Output:
Input: Graph with edges (0,1,5), (0,2,3), (1,2,2), (1,3,6), (2,3,7), source=0
Output: Distances: [0, 5, 3, 9]
"""


class Solution:
    def Bellman_Ford_Standard(self, V, edges, src):
        """
        Relax all edges V-1 times
        Time Complexity: O(V*E)
        Space Complexity: O(V)
        """
        dist = [float('inf')] * V
        dist[src] = 0
        
        for _ in range(V - 1):
            for edge in edges:
                u, v, w = edge[0], edge[1], edge[2]
                
                if dist[u] != float('inf') and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
        
        for edge in edges:
            u, v, w = edge[0], edge[1], edge[2]
            
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                return [-1]
        
        return dist
    
    def Has_Negative_Cycle(self, V, edges):
        """
        Check for negative cycle
        Time Complexity: O(V*E)
        Space Complexity: O(V)
        """
        dist = [0] * V
        
        for _ in range(V - 1):
            for edge in edges:
                u, v, w = edge[0], edge[1], edge[2]
                
                if dist[u] != float('inf') and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
        
        for edge in edges:
            u, v, w = edge[0], edge[1], edge[2]
            
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                return True
        
        return False


def Test_Bellman_Ford():
    solution = Solution()
    
    print("Test Case 1: Graph with negative edges (no cycle)")
    V1 = 4
    edges1 = [
        [0, 1, 5],
        [0, 2, 3],
        [1, 2, -2],
        [1, 3, 6],
        [2, 3, 7]
    ]
    dist1 = solution.Bellman_Ford_Standard(V1, edges1, 0)
    if dist1[0] == -1:
        print("Negative cycle detected!")
    else:
        print("Distances from source 0:", end=" ")
        for d in dist1:
            print(int(d), end=" ")
        print()
    
    print("\nTest Case 2: Graph with negative cycle")
    V2 = 3
    edges2 = [
        [0, 1, 1],
        [1, 2, -3],
        [2, 0, 2]
    ]
    hasCycle = solution.Has_Negative_Cycle(V2, edges2)
    print(f"Has negative cycle: {'Yes' if hasCycle else 'No'}")
    
    print("\nTest Case 3: Simple path graph")
    V3 = 5
    edges3 = [
        [0, 1, 1],
        [1, 2, 2],
        [2, 3, 3],
        [3, 4, 4]
    ]
    dist3 = solution.Bellman_Ford_Standard(V3, edges3, 0)
    print("Distances from source 0:", end=" ")
    for d in dist3:
        print(int(d), end=" ")
    print()


if __name__ == "__main__":
    Test_Bellman_Ford()
