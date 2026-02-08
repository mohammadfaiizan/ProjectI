"""
Problem: Dijkstra's Shortest Path Algorithm
URL: https://practice.geeksforgeeks.org/problems/implementing-dijkstra-set-1-adjacency-matrix/1

Problem Statement:
Find shortest distance from source to all vertices in a weighted graph with non-negative weights.

Sample Input/Output:
Input: Weighted graph with 5+ nodes
Output: Shortest distances from source to all vertices
"""

import heapq


class Solution:
    def Dijkstra_Priority_Queue(self, V, adj, src):
        """
        Min-Heap Based
        Time Complexity: O((V+E) log V)
        Space Complexity: O(V)
        """
        dist = [float('inf')] * V
        pq = []
        
        dist[src] = 0
        heapq.heappush(pq, (0, src))
        
        while pq:
            d, u = heapq.heappop(pq)
            
            if d > dist[u]:
                continue
            
            for v, weight in adj[u]:
                if dist[u] + weight < dist[v]:
                    dist[v] = dist[u] + weight
                    heapq.heappush(pq, (dist[v], v))
        
        return dist

    def Dijkstra_Set(self, V, adj, src):
        """
        Set-Based
        Time Complexity: O((V+E) log V)
        Space Complexity: O(V)
        """
        dist = [float('inf')] * V
        s = set()
        
        dist[src] = 0
        s.add((0, src))
        
        while s:
            d, u = min(s)
            s.remove((d, u))
            
            for v, weight in adj[u]:
                if dist[u] + weight < dist[v]:
                    if dist[v] != float('inf'):
                        s.discard((dist[v], v))
                    dist[v] = dist[u] + weight
                    s.add((dist[v], v))
        
        return dist


def Test_Dijkstra_Algorithm():
    solution = Solution()
    
    print("Test: Dijkstra's Algorithm")
    V = 5
    adj = [[] for _ in range(V)]
    
    adj[0].append((1, 4))
    adj[0].append((2, 1))
    adj[1].append((3, 1))
    adj[2].append((1, 2))
    adj[2].append((3, 5))
    adj[3].append((4, 3))
    
    src = 0
    
    dist1 = solution.Dijkstra_Priority_Queue(V, adj, src)
    print(f"Shortest distances from source {src} (Priority Queue):")
    for i in range(V):
        print(f"Distance to {i}: {-1 if dist1[i] == float('inf') else int(dist1[i])}")
    
    dist2 = solution.Dijkstra_Set(V, adj, src)
    print(f"\nShortest distances from source {src} (Set):")
    for i in range(V):
        print(f"Distance to {i}: {-1 if dist2[i] == float('inf') else int(dist2[i])}")
    
    print("\nTest 2: Larger Graph")
    V2 = 6
    adj2 = [[] for _ in range(V2)]
    
    adj2[0].append((1, 5))
    adj2[0].append((2, 3))
    adj2[1].append((3, 6))
    adj2[1].append((2, 2))
    adj2[2].append((4, 4))
    adj2[2].append((5, 2))
    adj2[3].append((4, 1))
    adj2[4].append((5, 3))
    
    src2 = 0
    dist3 = solution.Dijkstra_Priority_Queue(V2, adj2, src2)
    print(f"Shortest distances from source {src2}:")
    for i in range(V2):
        print(f"Distance to {i}: {-1 if dist3[i] == float('inf') else int(dist3[i])}")


if __name__ == "__main__":
    Test_Dijkstra_Algorithm()
