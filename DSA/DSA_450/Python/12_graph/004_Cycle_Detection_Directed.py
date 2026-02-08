"""
Problem: Detect Cycle in a Directed Graph
URL: https://practice.geeksforgeeks.org/problems/detect-cycle-in-a-directed-graph/1

Problem Statement:
Detect if a directed graph contains a cycle.

Sample Input/Output:
Input: Graph with cycle: 0->1->2->0
Output: Cycle detected: true
"""

from collections import deque


class Solution:
    def Cycle_Directed_DFS_Helper(self, node, adj, visited, recStack):
        visited[node] = True
        recStack[node] = True
        
        for neighbor in adj[node]:
            if not visited[neighbor]:
                if self.Cycle_Directed_DFS_Helper(neighbor, adj, visited, recStack):
                    return True
            elif recStack[neighbor]:
                return True
        
        recStack[node] = False
        return False

    def Cycle_Directed_DFS(self, V, adj):
        """
        DFS with Recursion Stack Tracking
        Time Complexity: O(V+E)
        Space Complexity: O(V)
        """
        visited = [False] * V
        recStack = [False] * V
        
        for i in range(V):
            if not visited[i]:
                if self.Cycle_Directed_DFS_Helper(i, adj, visited, recStack):
                    return True
        
        return False

    def Cycle_Directed_BFS_Kahn(self, V, adj):
        """
        Kahn's Algorithm - Topological Sort
        Time Complexity: O(V+E)
        Space Complexity: O(V)
        """
        inDegree = [0] * V
        
        for i in range(V):
            for neighbor in adj[i]:
                inDegree[neighbor] += 1
        
        q = deque()
        for i in range(V):
            if inDegree[i] == 0:
                q.append(i)
        
        count = 0
        while q:
            node = q.popleft()
            count += 1
            
            for neighbor in adj[node]:
                inDegree[neighbor] -= 1
                if inDegree[neighbor] == 0:
                    q.append(neighbor)
        
        return count != V


def Test_Cycle_Detection_Directed():
    solution = Solution()
    
    print("Test 1: Graph with Cycle")
    V1 = 4
    adj1 = [[] for _ in range(V1)]
    adj1[0] = [1]
    adj1[1] = [2]
    adj1[2] = [3]
    adj1[3] = [1]
    
    hasCycle1 = solution.Cycle_Directed_DFS(V1, adj1)
    print(f"Cycle detected (DFS): {'Yes' if hasCycle1 else 'No'}")
    
    hasCycle1_bfs = solution.Cycle_Directed_BFS_Kahn(V1, adj1)
    print(f"Cycle detected (BFS/Kahn): {'Yes' if hasCycle1_bfs else 'No'}")
    
    print("\nTest 2: Graph without Cycle")
    V2 = 4
    adj2 = [[] for _ in range(V2)]
    adj2[0] = [1]
    adj2[1] = [2]
    adj2[2] = [3]
    
    hasCycle2 = solution.Cycle_Directed_DFS(V2, adj2)
    print(f"Cycle detected (DFS): {'Yes' if hasCycle2 else 'No'}")
    
    hasCycle2_bfs = solution.Cycle_Directed_BFS_Kahn(V2, adj2)
    print(f"Cycle detected (BFS/Kahn): {'Yes' if hasCycle2_bfs else 'No'}")


if __name__ == "__main__":
    Test_Cycle_Detection_Directed()
