"""
Problem: Minimum Edges to Reverse to Make Path from Source to Destination
URL: https://www.geeksforgeeks.org/minimum-edges-reverse-make-path-source-destination/

Problem Statement:
Given a directed graph and a source vertex and destination vertex, find the minimum number of edges that need to be reversed to make a path from source to destination.

Sample Input/Output:
Input: Directed graph with edges, src=0, dst=6
Output: Minimum edges to reverse
"""

from collections import deque
import heapq


class Solution:
    def Min_Edges_Reverse_BFS_01(self, V, edges, src, dst):
        """
        0-1 BFS using deque
        Time Complexity: O(V+E)
        Space Complexity: O(V+E)
        """
        adj = [[] for _ in range(V)]
        for edge in edges:
            u, v = edge[0], edge[1]
            adj[u].append((v, 0))
            adj[v].append((u, 1))
        
        dq = deque()
        dist = [float('inf')] * V
        dist[src] = 0
        dq.append((src, 0))
        
        while dq:
            u, cost = dq.popleft()
            
            if u == dst:
                return cost
            
            for v, weight in adj[u]:
                if dist[v] > dist[u] + weight:
                    dist[v] = dist[u] + weight
                    if weight == 0:
                        dq.appendleft((v, dist[v]))
                    else:
                        dq.append((v, dist[v]))
        
        return -1

    def Min_Edges_Reverse_Dijkstra(self, V, edges, src, dst):
        """
        Dijkstra on modified graph
        Time Complexity: O((V+E) log V)
        Space Complexity: O(V+E)
        """
        adj = [[] for _ in range(V)]
        for edge in edges:
            u, v = edge[0], edge[1]
            adj[u].append((v, 0))
            adj[v].append((u, 1))
        
        pq = [(0, src)]
        dist = [float('inf')] * V
        dist[src] = 0
        
        while pq:
            cost, u = heapq.heappop(pq)
            
            if u == dst:
                return cost
            if cost > dist[u]:
                continue
            
            for v, weight in adj[u]:
                if dist[v] > dist[u] + weight:
                    dist[v] = dist[u] + weight
                    heapq.heappush(pq, (dist[v], v))
        
        return -1


def Test_Min_Edges_Reverse():
    solution = Solution()
    V = 7
    edges = [(0, 1), (2, 1), (2, 3), (5, 1), (4, 5), (6, 4), (6, 3)]
    src, dst = 0, 6
    
    print("Test Case 1:")
    print(f"Source: {src}, Destination: {dst}")
    result1 = solution.Min_Edges_Reverse_BFS_01(V, edges, src, dst)
    print(f"0-1 BFS Result: {result1} edges to reverse")
    
    result2 = solution.Min_Edges_Reverse_Dijkstra(V, edges, src, dst)
    print(f"Dijkstra Result: {result2} edges to reverse")
    print()
    
    V = 4
    edges = [(0, 1), (2, 0), (2, 3), (3, 1)]
    src, dst = 0, 3
    print("Test Case 2:")
    print(f"Source: {src}, Destination: {dst}")
    result1 = solution.Min_Edges_Reverse_BFS_01(V, edges, src, dst)
    print(f"0-1 BFS Result: {result1} edges to reverse")
    result2 = solution.Min_Edges_Reverse_Dijkstra(V, edges, src, dst)
    print(f"Dijkstra Result: {result2} edges to reverse")


if __name__ == "__main__":
    Test_Min_Edges_Reverse()
