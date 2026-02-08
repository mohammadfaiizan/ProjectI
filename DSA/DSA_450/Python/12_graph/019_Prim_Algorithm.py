"""
Problem: Prim's Minimum Spanning Tree Algorithm
URL: https://practice.geeksforgeeks.org/problems/minimum-spanning-tree/1

Problem Statement:
Find the Minimum Spanning Tree (MST) of a weighted undirected graph using Prim's algorithm. The algorithm starts from any vertex and greedily adds the minimum weight edge connecting a vertex in MST to a vertex outside MST.

Sample Input/Output:
Input: Graph with edges (0,1,10), (0,2,6), (0,3,5), (1,3,15), (2,3,4)
Output: MST weight = 19
"""

import heapq


class Solution:
    def Prim_MST_Priority_Queue(self, V, adj):
        """
        Min-heap based
        Time Complexity: O((V+E) log V)
        Space Complexity: O(V)
        """
        inMST = [False] * V
        pq = []
        
        heapq.heappush(pq, (0, 0))
        mstWeight = 0
        
        while pq:
            weight, u = heapq.heappop(pq)
            
            if inMST[u]:
                continue
            
            inMST[u] = True
            mstWeight += weight
            
            for v, w in adj[u]:
                if not inMST[v]:
                    heapq.heappush(pq, (w, v))
        
        return mstWeight
    
    def Prim_MST_Adjacency_Matrix(self, V, graph):
        """
        Brute force with adjacency matrix
        Time Complexity: O(V^2)
        Space Complexity: O(V)
        """
        inMST = [False] * V
        key = [float('inf')] * V
        key[0] = 0
        mstWeight = 0
        
        for _ in range(V):
            u = -1
            for i in range(V):
                if not inMST[i] and (u == -1 or key[i] < key[u]):
                    u = i
            
            inMST[u] = True
            mstWeight += key[u]
            
            for v in range(V):
                if graph[u][v] != 0 and not inMST[v] and graph[u][v] < key[v]:
                    key[v] = graph[u][v]
        
        return mstWeight


def Test_Prim():
    solution = Solution()
    
    print("Test Case 1: Weighted graph with adjacency list")
    V1 = 4
    adj1 = [[] for _ in range(4)]
    adj1[0].append((1, 10))
    adj1[0].append((2, 6))
    adj1[0].append((3, 5))
    adj1[1].append((0, 10))
    adj1[1].append((3, 15))
    adj1[2].append((0, 6))
    adj1[2].append((3, 4))
    adj1[3].append((0, 5))
    adj1[3].append((1, 15))
    adj1[3].append((2, 4))
    print(f"Priority Queue MST Weight: {solution.Prim_MST_Priority_Queue(V1, adj1)}")
    
    print("\nTest Case 2: Weighted graph with adjacency matrix")
    V2 = 4
    graph2 = [
        [0, 10, 6, 5],
        [10, 0, 0, 15],
        [6, 0, 0, 4],
        [5, 15, 4, 0]
    ]
    print(f"Adjacency Matrix MST Weight: {solution.Prim_MST_Adjacency_Matrix(V2, graph2)}")
    
    print("\nTest Case 3: Complete graph")
    V3 = 5
    adj3 = [[] for _ in range(5)]
    adj3[0].append((1, 2))
    adj3[0].append((2, 3))
    adj3[0].append((3, 6))
    adj3[0].append((4, 5))
    adj3[1].append((0, 2))
    adj3[1].append((2, 5))
    adj3[1].append((3, 3))
    adj3[1].append((4, 4))
    adj3[2].append((0, 3))
    adj3[2].append((1, 5))
    adj3[2].append((3, 1))
    adj3[2].append((4, 2))
    adj3[3].append((0, 6))
    adj3[3].append((1, 3))
    adj3[3].append((2, 1))
    adj3[3].append((4, 3))
    adj3[4].append((0, 5))
    adj3[4].append((1, 4))
    adj3[4].append((2, 2))
    adj3[4].append((3, 3))
    print(f"Priority Queue MST Weight: {solution.Prim_MST_Priority_Queue(V3, adj3)}")


if __name__ == "__main__":
    Test_Prim()
