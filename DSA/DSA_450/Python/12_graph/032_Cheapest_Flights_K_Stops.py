"""
Problem: Cheapest Flights Within K Stops
URL: https://leetcode.com/problems/cheapest-flights-within-k-stops/

Problem Statement:
Find the cheapest flight from src to dst with at most K stops. Given flights array where flights[i] = [fromi, toi, pricei].

Sample Input/Output:
Input: n=4, flights = [[0,1,100],[1,2,100],[2,0,100],[1,3,600],[2,3,200]], src=0, dst=3, k=1
Output: 700
"""

from collections import deque
import heapq


class Solution:
    def Cheapest_Flights_Bellman_Ford(self, n, flights, src, dst, k):
        """
        Modified Bellman-Ford with K+1 iterations
        Time Complexity: O(K*E)
        Space Complexity: O(V)
        """
        dist = [float('inf')] * n
        dist[src] = 0
        
        for _ in range(k + 1):
            temp = dist[:]
            for flight in flights:
                u, v, w = flight[0], flight[1], flight[2]
                if dist[u] != float('inf'):
                    temp[v] = min(temp[v], dist[u] + w)
            dist = temp
        
        return dist[dst] if dist[dst] != float('inf') else -1
    
    def Cheapest_Flights_BFS(self, n, flights, src, dst, k):
        """
        BFS with cost tracking
        Time Complexity: O(V*E)
        Space Complexity: O(V)
        """
        adj = [[] for _ in range(n)]
        for flight in flights:
            adj[flight[0]].append((flight[1], flight[2]))
        
        q = deque()
        cost = [float('inf')] * n
        q.append((src, 0))
        cost[src] = 0
        stops = 0
        
        while q and stops <= k:
            sz = len(q)
            while sz > 0:
                u, c = q.popleft()
                
                for v, w in adj[u]:
                    if c + w < cost[v]:
                        cost[v] = c + w
                        q.append((v, cost[v]))
                sz -= 1
            stops += 1
        
        return cost[dst] if cost[dst] != float('inf') else -1
    
    def Cheapest_Flights_Dijkstra(self, n, flights, src, dst, k):
        """
        Priority queue with stops constraint
        Time Complexity: O(E log V)
        Space Complexity: O(V)
        """
        adj = [[] for _ in range(n)]
        for flight in flights:
            adj[flight[0]].append((flight[1], flight[2]))
        
        pq = [(0, src, 0)]
        dist = [[float('inf')] * (k + 2) for _ in range(n)]
        dist[src][0] = 0
        
        while pq:
            cost, u, stops = heapq.heappop(pq)
            
            if u == dst:
                return cost
            
            if stops > k:
                continue
            
            for v, w in adj[u]:
                newCost = cost + w
                if newCost < dist[v][stops + 1]:
                    dist[v][stops + 1] = newCost
                    heapq.heappush(pq, (newCost, v, stops + 1))
        
        return -1


def Test_Cheapest_Flights_Bellman_Ford():
    solution = Solution()
    
    n1 = 4
    flights1 = [[0, 1, 100], [1, 2, 100], [2, 0, 100], [1, 3, 600], [2, 3, 200]]
    print(f"Test 1 Bellman-Ford: {solution.Cheapest_Flights_Bellman_Ford(n1, flights1, 0, 3, 1)}")
    
    n2 = 3
    flights2 = [[0, 1, 100], [1, 2, 100], [0, 2, 500]]
    print(f"Test 2 BFS: {solution.Cheapest_Flights_BFS(n2, flights2, 0, 2, 1)}")
    
    n3 = 3
    flights3 = [[0, 1, 100], [1, 2, 100], [0, 2, 500]]
    print(f"Test 3 Dijkstra: {solution.Cheapest_Flights_Dijkstra(n3, flights3, 0, 2, 0)}")


if __name__ == "__main__":
    Test_Cheapest_Flights_Bellman_Ford()
