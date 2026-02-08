"""
Problem: Implement BFS Algorithm
URL: https://practice.geeksforgeeks.org/problems/bfs-traversal-of-graph/1

Problem Statement:
Implement Breadth-First Search traversal for both connected and disconnected graphs.

Sample Input/Output:
Input: Connected graph with 5 vertices
Output: BFS traversal: 0 1 2 3 4
"""

from collections import deque


class Solution:
    def BFS_Connected(self, V, adj, start):
        """
        Single Source BFS
        Time Complexity: O(V+E)
        Space Complexity: O(V)
        """
        result = []
        visited = [False] * V
        q = deque()
        
        visited[start] = True
        q.append(start)
        
        while q:
            node = q.popleft()
            result.append(node)
            
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    q.append(neighbor)
        
        return result

    def BFS_Disconnected(self, V, adj):
        """
        BFS for Disconnected Graph
        Time Complexity: O(V+E)
        Space Complexity: O(V)
        """
        result = []
        visited = [False] * V
        
        for i in range(V):
            if not visited[i]:
                q = deque()
                visited[i] = True
                q.append(i)
                
                while q:
                    node = q.popleft()
                    result.append(node)
                    
                    for neighbor in adj[node]:
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            q.append(neighbor)
        
        return result


def Test_BFS_Algorithm():
    solution = Solution()
    
    print("Test 1: Connected Graph")
    V1 = 5
    adj1 = [[] for _ in range(V1)]
    adj1[0] = [1, 2]
    adj1[1] = [0, 3, 4]
    adj1[2] = [0]
    adj1[3] = [1]
    adj1[4] = [1]
    
    bfs1 = solution.BFS_Connected(V1, adj1, 0)
    print("BFS Traversal:", end=" ")
    for node in bfs1:
        print(node, end=" ")
    print()
    
    print("\nTest 2: Disconnected Graph")
    V2 = 6
    adj2 = [[] for _ in range(V2)]
    adj2[0] = [1]
    adj2[1] = [0]
    adj2[2] = [3]
    adj2[3] = [2]
    adj2[4] = [5]
    adj2[5] = [4]
    
    bfs2 = solution.BFS_Disconnected(V2, adj2)
    print("BFS Traversal:", end=" ")
    for node in bfs2:
        print(node, end=" ")
    print()


if __name__ == "__main__":
    Test_BFS_Algorithm()
