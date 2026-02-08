"""
Problem: M-Coloring Problem
URL: https://practice.geeksforgeeks.org/problems/m-coloring-problem-1587115620/1

Problem Statement:
Check if the graph can be colored with at most M colors such that no two adjacent vertices have the same color. If yes, print the coloring.

Sample Input/Output:
Input: V=4, edges = [[0,1],[1,2],[2,3],[0,3]], m=3
Output: true, coloring = [0,1,0,1]
"""


class Solution:
    def M_Coloring_Backtracking(self, V, edges, m, coloring):
        """
        Try each color, backtrack if conflict
        Time Complexity: O(M^V)
        Space Complexity: O(V+E)
        """
        adj = [[] for _ in range(V)]
        for e in edges:
            adj[e[0]].append(e[1])
            adj[e[1]].append(e[0])
        
        coloring[:] = [-1] * V
        
        def canColor(u):
            if u == V:
                return True
            
            for c in range(m):
                valid = True
                for v in adj[u]:
                    if coloring[v] == c:
                        valid = False
                        break
                
                if valid:
                    coloring[u] = c
                    if canColor(u + 1):
                        return True
                    coloring[u] = -1
            
            return False
        
        return canColor(0)
    
    def M_Coloring_Greedy(self, V, edges, m, coloring):
        """
        Greedy assignment
        Time Complexity: O(V+E)
        Space Complexity: O(V+E)
        """
        adj = [[] for _ in range(V)]
        for e in edges:
            adj[e[0]].append(e[1])
            adj[e[1]].append(e[0])
        
        coloring[:] = [-1] * V
        used = [False] * m
        
        for u in range(V):
            used = [False] * m
            
            for v in adj[u]:
                if coloring[v] != -1:
                    used[coloring[v]] = True
            
            c = 0
            for c in range(m):
                if not used[c]:
                    break
            
            if c == m:
                return False
            
            coloring[u] = c
        
        return True


def Test_M_Coloring_Backtracking():
    solution = Solution()
    
    V1 = 4
    edges1 = [[0, 1], [1, 2], [2, 3], [0, 3]]
    coloring1 = []
    result1 = solution.M_Coloring_Backtracking(V1, edges1, 3, coloring1)
    print(f"Test 1 Backtracking: {result1}", end="")
    if result1:
        print(f", Coloring: {coloring1}")
    else:
        print()
    
    V2 = 3
    edges2 = [[0, 1], [1, 2], [0, 2]]
    coloring2 = []
    result2 = solution.M_Coloring_Backtracking(V2, edges2, 2, coloring2)
    print(f"Test 2 Backtracking: {result2}", end="")
    if result2:
        print(f", Coloring: {coloring2}")
    else:
        print()
    
    V3 = 4
    edges3 = [[0, 1], [1, 2], [2, 3], [3, 0]]
    coloring3 = []
    result3 = solution.M_Coloring_Greedy(V3, edges3, 2, coloring3)
    print(f"Test 3 Greedy: {result3}", end="")
    if result3:
        print(f", Coloring: {coloring3}")
    else:
        print()


if __name__ == "__main__":
    Test_M_Coloring_Backtracking()
