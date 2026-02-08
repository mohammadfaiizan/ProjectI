"""
Problem: Graph Coloring Problem
URL: https://www.geeksforgeeks.org/graph-coloring-applications/

Problem Statement:
Assign colors to vertices such that no two adjacent vertices share the same color. Find the minimum number of colors needed (chromatic number approximation) or check if a graph can be colored with m colors.

Sample Input/Output:
Input: Graph with edges, number of colors
Output: Color assignment or whether coloring is possible
"""


class Solution:
    def Graph_Coloring_Greedy(self, V, adj):
        """
        Greedy coloring
        Time Complexity: O(V^2)
        Space Complexity: O(V)
        """
        color = [-1] * V
        color[0] = 0
        
        available = [False] * V
        
        for u in range(1, V):
            for v in adj[u]:
                if color[v] != -1:
                    available[color[v]] = True
            
            cr = 0
            for cr in range(V):
                if not available[cr]:
                    break
            
            color[u] = cr
            
            for v in adj[u]:
                if color[v] != -1:
                    available[color[v]] = False
        
        maxColor = max(color)
        return maxColor + 1
    
    def Graph_Coloring_Backtracking(self, V, adj, m):
        """
        Try m colors with backtracking
        Time Complexity: O(m^V)
        Space Complexity: O(V)
        """
        color = [-1] * V
        
        def isSafe(u):
            for v in adj[u]:
                if color[v] != -1 and color[v] == color[u]:
                    return False
            return True
        
        def solve(u):
            if u == V:
                return True
            
            for c in range(m):
                color[u] = c
                if isSafe(u) and solve(u + 1):
                    return True
                color[u] = -1
            
            return False
        
        return solve(0)
    
    def Get_Coloring_Result(self, V, adj, m):
        color = [-1] * V
        
        def isSafe(u):
            for v in adj[u]:
                if color[v] != -1 and color[v] == color[u]:
                    return False
            return True
        
        def solve(u):
            if u == V:
                return True
            
            for c in range(m):
                color[u] = c
                if isSafe(u) and solve(u + 1):
                    return True
                color[u] = -1
            
            return False
        
        solve(0)
        return color


def Test_Graph_Coloring():
    solution = Solution()
    
    print("Test Case 1: Simple graph")
    V1 = 4
    adj1 = [[] for _ in range(4)]
    adj1[0].append(1)
    adj1[0].append(2)
    adj1[1].append(0)
    adj1[1].append(3)
    adj1[2].append(0)
    adj1[2].append(3)
    adj1[3].append(1)
    adj1[3].append(2)
    
    minColors1 = solution.Graph_Coloring_Greedy(V1, adj1)
    print(f"Greedy minimum colors: {minColors1}")
    
    canColor1 = solution.Graph_Coloring_Backtracking(V1, adj1, 2)
    print(f"Can color with 2 colors: {'Yes' if canColor1 else 'No'}")
    
    print("\nTest Case 2: Complete graph K4")
    V2 = 4
    adj2 = [[] for _ in range(4)]
    for i in range(4):
        for j in range(4):
            if i != j:
                adj2[i].append(j)
    
    minColors2 = solution.Graph_Coloring_Greedy(V2, adj2)
    print(f"Greedy minimum colors: {minColors2}")
    
    canColor2 = solution.Graph_Coloring_Backtracking(V2, adj2, 4)
    print(f"Can color with 4 colors: {'Yes' if canColor2 else 'No'}")
    
    print("\nTest Case 3: Bipartite graph")
    V3 = 6
    adj3 = [[] for _ in range(6)]
    adj3[0].append(1)
    adj3[0].append(3)
    adj3[1].append(0)
    adj3[1].append(2)
    adj3[2].append(1)
    adj3[2].append(4)
    adj3[3].append(0)
    adj3[3].append(4)
    adj3[4].append(2)
    adj3[4].append(3)
    adj3[4].append(5)
    adj3[5].append(4)
    
    minColors3 = solution.Graph_Coloring_Greedy(V3, adj3)
    print(f"Greedy minimum colors: {minColors3}")
    
    canColor3 = solution.Graph_Coloring_Backtracking(V3, adj3, 2)
    print(f"Can color with 2 colors: {'Yes' if canColor3 else 'No'}")
    
    if canColor3:
        coloring = solution.Get_Coloring_Result(V3, adj3, 2)
        print("Coloring:", end=" ")
        for c in coloring:
            print(c, end=" ")
        print()


if __name__ == "__main__":
    Test_Graph_Coloring()
