"""
Problem: Chinese Postman Problem (Route Inspection)
URL: https://www.geeksforgeeks.org/chinese-postman-route-inspection-algorithm/

Problem Statement:
Find the shortest closed path that visits every edge at least once in a weighted undirected graph.

Sample Input/Output:
Input: Weighted undirected graph
Output: Minimum cost to traverse all edges
"""


class Solution:
    def Is_Eulerian(self, V, adj):
        for i in range(V):
            if len(adj[i]) % 2 != 0:
                return False
        return True

    def Get_Odd_Degree_Vertices(self, V, adj):
        oddVertices = []
        for i in range(V):
            if len(adj[i]) % 2 != 0:
                oddVertices.append(i)
        return oddVertices

    def Floyd_Warshall(self, V, adj):
        dist = [[float('inf')] * V for _ in range(V)]
        
        for i in range(V):
            dist[i][i] = 0
            for neighbor in adj[i]:
                v, w = neighbor[0], neighbor[1]
                dist[i][v] = min(dist[i][v], w)
        
        for k in range(V):
            for i in range(V):
                for j in range(V):
                    if dist[i][k] != float('inf') and dist[k][j] != float('inf'):
                        dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
        
        return dist

    def Min_Weight_Perfect_Matching(self, oddVertices, dist):
        n = len(oddVertices)
        if n == 0:
            return 0
        
        dp = [float('inf')] * (1 << n)
        dp[0] = 0
        
        for mask in range(1 << n):
            if dp[mask] == float('inf'):
                continue
            
            first = -1
            for i in range(n):
                if not (mask & (1 << i)):
                    first = i
                    break
            
            if first == -1:
                continue
            
            for j in range(first + 1, n):
                if mask & (1 << j):
                    continue
                
                newMask = mask | (1 << first) | (1 << j)
                u = oddVertices[first]
                v = oddVertices[j]
                cost = dist[u][v]
                
                if cost != float('inf'):
                    dp[newMask] = min(dp[newMask], dp[mask] + cost)
        
        return dp[(1 << n) - 1]

    def Chinese_Postman_Solve(self, V, weightedEdges):
        """
        Check Eulerian, if not find odd-degree vertices and add shortest paths between pairs
        Time Complexity: O(V^3)
        Space Complexity: O(V^2)
        """
        adj = [[] for _ in range(V)]
        totalWeight = 0
        
        for edge in weightedEdges:
            u, v, w = edge[0][0], edge[0][1], edge[1]
            adj[u].append((v, w))
            adj[v].append((u, w))
            totalWeight += w
        
        if self.Is_Eulerian(V, adj):
            return totalWeight
        
        oddVertices = self.Get_Odd_Degree_Vertices(V, adj)
        dist = self.Floyd_Warshall(V, adj)
        
        matchingCost = self.Min_Weight_Perfect_Matching(oddVertices, dist)
        
        return totalWeight + matchingCost


def Test_Chinese_Postman():
    solution = Solution()
    
    print("Test Case 1: Eulerian Graph")
    V1 = 4
    edges1 = [
        ((0, 1), 1), ((1, 2), 2), ((2, 3), 3), ((3, 0), 4), ((0, 2), 5), ((1, 3), 6)
    ]
    result1 = solution.Chinese_Postman_Solve(V1, edges1)
    print(f"Minimum Cost: {int(result1)}")
    print()
    
    print("Test Case 2: Non-Eulerian Graph")
    V2 = 4
    edges2 = [
        ((0, 1), 1), ((1, 2), 2), ((2, 3), 3), ((3, 0), 4)
    ]
    result2 = solution.Chinese_Postman_Solve(V2, edges2)
    print(f"Minimum Cost: {int(result2)}")
    print()
    
    print("Test Case 3: Complex Graph")
    V3 = 5
    edges3 = [
        ((0, 1), 2), ((1, 2), 3), ((2, 3), 1), ((3, 4), 4), ((4, 0), 5)
    ]
    result3 = solution.Chinese_Postman_Solve(V3, edges3)
    print(f"Minimum Cost: {int(result3)}")


if __name__ == "__main__":
    Test_Chinese_Postman()
