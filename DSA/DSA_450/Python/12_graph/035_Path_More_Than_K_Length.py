"""
Problem: Find if There is a Path of More Than K Length From a Source
URL: https://www.geeksforgeeks.org/find-if-there-is-a-path-of-more-than-k-length-from-a-source/

Problem Statement:
Check if there exists a simple path (no repeated vertices) of total weight > K from a source vertex in a weighted graph.

Sample Input/Output:
Input: V=9, edges = [[0,1,4],[0,7,8],[1,2,8],[1,7,11],[2,3,7],[2,8,2],[2,5,4],[3,4,9],[3,5,14],[4,5,10],[5,6,2],[6,7,1],[6,8,6],[7,8,7]], src=0, k=58
Output: true
"""


class Solution:
    def Path_K_Backtracking(self, V, edges, src, k):
        """
        DFS backtracking, avoid revisiting
        Time Complexity: O(V!)
        Space Complexity: O(V+E)
        """
        adj = [[] for _ in range(V)]
        for e in edges:
            adj[e[0]].append((e[1], e[2]))
            adj[e[1]].append((e[0], e[2]))
        
        visited = [False] * V
        
        def dfs(u, pathLen):
            if pathLen > k:
                return True
            
            visited[u] = True
            for v, w in adj[u]:
                if not visited[v]:
                    if dfs(v, pathLen + w):
                        return True
            visited[u] = False
            
            return False
        
        return dfs(src, 0)


def Test_Path_K_Backtracking():
    solution = Solution()
    
    V1 = 9
    edges1 = [[0, 1, 4], [0, 7, 8], [1, 2, 8], [1, 7, 11], [2, 3, 7], [2, 8, 2], [2, 5, 4], [3, 4, 9], [3, 5, 14], [4, 5, 10], [5, 6, 2], [6, 7, 1], [6, 8, 6], [7, 8, 7]]
    print(f"Test 1 (k=58): {solution.Path_K_Backtracking(V1, edges1, 0, 58)}")
    
    V2 = 4
    edges2 = [[0, 1, 10], [1, 2, 20], [2, 3, 30], [0, 3, 40]]
    print(f"Test 2 (k=50): {solution.Path_K_Backtracking(V2, edges2, 0, 50)}")
    
    V3 = 3
    edges3 = [[0, 1, 5], [1, 2, 5], [0, 2, 5]]
    print(f"Test 3 (k=15): {solution.Path_K_Backtracking(V3, edges3, 0, 15)}")


if __name__ == "__main__":
    Test_Path_K_Backtracking()
