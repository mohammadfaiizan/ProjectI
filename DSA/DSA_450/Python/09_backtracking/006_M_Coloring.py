"""
Problem: M Coloring
URL: https://practice.geeksforgeeks.org/problems/m-coloring-problem-1587115620/1

Problem Statement:
Given undirected graph and M colors, determine if the graph can be colored with at most M colors such that no two adjacent vertices have same color.

Sample Input/Output:
Input: N=4, M=3, E=5, edges={{0,1},{1,2},{2,3},{3,0},{0,2}}
Output: true
Explanation: Graph can be colored with 3 colors
"""


class Solution:
    def Graph_Coloring_Backtracking(self, n, m, edges):
        """
        Backtracking
        Time Complexity: O(m^V)
        Space Complexity: O(V)
        """
        graph = [[] for _ in range(n)]
        for edge in edges:
            graph[edge[0]].append(edge[1])
            graph[edge[1]].append(edge[0])
        
        color = [0] * n
        
        def Is_Safe(vertex, c):
            for neighbor in graph[vertex]:
                if color[neighbor] == c:
                    return False
            return True
        
        def backtrack(vertex):
            if vertex == n:
                return True
            
            for c in range(1, m + 1):
                if Is_Safe(vertex, c):
                    color[vertex] = c
                    if backtrack(vertex + 1):
                        return True
                    color[vertex] = 0
            
            return False
        
        return backtrack(0)


def Test_M_Coloring():
    solution = Solution()
    
    n = 4
    m = 3
    edges = [[0,1],[1,2],[2,3],[3,0],[0,2]]
    
    can_color = solution.Graph_Coloring_Backtracking(n, m, edges)
    print("Can color graph with", m, "colors:", "Yes" if can_color else "No")


if __name__ == "__main__":
    Test_M_Coloring()
