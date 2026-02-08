"""
Problem: Vertex Cover Problem
URL: https://www.geeksforgeeks.org/vertex-cover-problem-set-1-introduction-approximate-algorithm-2/

Problem Statement:
Find an approximate minimum vertex cover (set of vertices that covers all edges).

Sample Input/Output:
Input: Graph with edges
Output: Vertex cover set
"""


class Solution:
    def Vertex_Cover_Approximate(self, V, edges):
        """
        Greedy: pick edge, add both endpoints, remove covered edges
        Time Complexity: O(V+E)
        Space Complexity: O(V+E)
        """
        adj = [[] for _ in range(V)]
        inCover = [False] * V
        edgeCovered = [False] * len(edges)
        
        for i in range(len(edges)):
            u, v = edges[i][0], edges[i][1]
            adj[u].append(i)
            adj[v].append(i)
        
        for i in range(len(edges)):
            if edgeCovered[i]:
                continue
            
            u, v = edges[i][0], edges[i][1]
            
            inCover[u] = True
            inCover[v] = True
            
            for e in adj[u]:
                edgeCovered[e] = True
            for e in adj[v]:
                edgeCovered[e] = True
        
        cover = []
        for i in range(V):
            if inCover[i]:
                cover.append(i)
        return cover

    def Vertex_Cover_Tree_DP(self, V, edges, root):
        """
        DP on tree (for trees only)
        Time Complexity: O(V)
        Space Complexity: O(V)
        """
        adj = [[] for _ in range(V)]
        for edge in edges:
            u, v = edge[0], edge[1]
            adj[u].append(v)
            adj[v].append(u)
        
        dp = [[-1] * 2 for _ in range(V)]
        
        def dfs(u, parent, include):
            if dp[u][include] != -1:
                return dp[u][include]
            
            result = 1 if include else 0
            
            for v in adj[u]:
                if v == parent:
                    continue
                
                if include:
                    result += min(dfs(v, u, True), dfs(v, u, False))
                else:
                    result += dfs(v, u, True)
            
            dp[u][include] = result
            return result
        
        return min(dfs(root, -1, True), dfs(root, -1, False))


def Test_Vertex_Cover():
    solution = Solution()
    
    print("Test Case 1: General Graph")
    V1 = 7
    edges1 = [(0, 1), (0, 2), (1, 3), (2, 4), (2, 5), (4, 6)]
    cover1 = solution.Vertex_Cover_Approximate(V1, edges1)
    print(f"Vertex Cover: {cover1}")
    print(f"Size: {len(cover1)}")
    print()
    
    print("Test Case 2: Tree Graph")
    V2 = 5
    edges2 = [(0, 1), (0, 2), (1, 3), (1, 4)]
    cover2 = solution.Vertex_Cover_Approximate(V2, edges2)
    print(f"Approximate Vertex Cover: {cover2}")
    print(f"Size: {len(cover2)}")
    optimal2 = solution.Vertex_Cover_Tree_DP(V2, edges2, 0)
    print(f"Optimal Tree DP Size: {optimal2}")
    print()
    
    print("Test Case 3: Complete Graph K3")
    V3 = 3
    edges3 = [(0, 1), (1, 2), (2, 0)]
    cover3 = solution.Vertex_Cover_Approximate(V3, edges3)
    print(f"Vertex Cover: {cover3}")
    print(f"Size: {len(cover3)}")


if __name__ == "__main__":
    Test_Vertex_Cover()
