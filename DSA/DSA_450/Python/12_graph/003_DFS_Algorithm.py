"""
Problem: Implement DFS Algorithm
URL: https://practice.geeksforgeeks.org/problems/depth-first-traversal-for-a-graph/1

Problem Statement:
Implement Depth-First Search traversal for both connected and disconnected graphs.

Sample Input/Output:
Input: Connected graph with 5 vertices
Output: DFS traversal: 0 1 3 4 2
"""


class Solution:
    def DFS_Recursive_Helper(self, node, adj, visited, result):
        visited[node] = True
        result.append(node)
        
        for neighbor in adj[node]:
            if not visited[neighbor]:
                self.DFS_Recursive_Helper(neighbor, adj, visited, result)

    def DFS_Recursive(self, V, adj, start):
        """
        Recursive DFS
        Time Complexity: O(V+E)
        Space Complexity: O(V)
        """
        result = []
        visited = [False] * V
        self.DFS_Recursive_Helper(start, adj, visited, result)
        return result

    def DFS_Iterative(self, V, adj, start):
        """
        Iterative DFS using Stack
        Time Complexity: O(V+E)
        Space Complexity: O(V)
        """
        result = []
        visited = [False] * V
        st = [start]
        
        while st:
            node = st.pop()
            
            if not visited[node]:
                visited[node] = True
                result.append(node)
                
                for neighbor in reversed(adj[node]):
                    if not visited[neighbor]:
                        st.append(neighbor)
        
        return result


def Test_DFS_Algorithm():
    solution = Solution()
    
    print("Test 1: Connected Graph (Recursive)")
    V1 = 5
    adj1 = [[] for _ in range(V1)]
    adj1[0] = [1, 2]
    adj1[1] = [0, 3, 4]
    adj1[2] = [0]
    adj1[3] = [1]
    adj1[4] = [1]
    
    dfs1 = solution.DFS_Recursive(V1, adj1, 0)
    print("DFS Traversal:", end=" ")
    for node in dfs1:
        print(node, end=" ")
    print()
    
    print("\nTest 2: Connected Graph (Iterative)")
    dfs2 = solution.DFS_Iterative(V1, adj1, 0)
    print("DFS Traversal:", end=" ")
    for node in dfs2:
        print(node, end=" ")
    print()
    
    print("\nTest 3: Disconnected Graph")
    V2 = 6
    adj2 = [[] for _ in range(V2)]
    adj2[0] = [1]
    adj2[1] = [0]
    adj2[2] = [3]
    adj2[3] = [2]
    adj2[4] = [5]
    adj2[5] = [4]
    
    visited = [False] * V2
    result = []
    for i in range(V2):
        if not visited[i]:
            solution.DFS_Recursive_Helper(i, adj2, visited, result)
    print("DFS Traversal:", end=" ")
    for node in result:
        print(node, end=" ")
    print()


if __name__ == "__main__":
    Test_DFS_Algorithm()
