"""
Problem: Strongly Connected Components (Kosaraju's Algorithm)
URL: https://practice.geeksforgeeks.org/problems/strongly-connected-components-kosarajus-algo/1

Problem Statement:
Find all strongly connected components (SCCs) in a directed graph. A strongly connected component is a maximal set of vertices such that every vertex can reach every other vertex in the set.

Sample Input/Output:
Input: V=5, edges = [[1,0],[0,2],[2,1],[0,3],[3,4]]
Output: 3 SCCs: [[0,1,2],[3],[4]]
"""


class Solution:
    def SCC_Kosaraju(self, V, edges):
        """
        Finish order DFS + transpose + DFS
        Time Complexity: O(V+E)
        Space Complexity: O(V+E)
        """
        adj = [[] for _ in range(V)]
        adjT = [[] for _ in range(V)]
        
        for e in edges:
            adj[e[0]].append(e[1])
            adjT[e[1]].append(e[0])
        
        visited = [False] * V
        st = []
        
        def dfs1(u):
            visited[u] = True
            for v in adj[u]:
                if not visited[v]:
                    dfs1(v)
            st.append(u)
        
        for i in range(V):
            if not visited[i]:
                dfs1(i)
        
        visited = [False] * V
        sccs = []
        
        def dfs2(u, comp):
            visited[u] = True
            comp.append(u)
            for v in adjT[u]:
                if not visited[v]:
                    dfs2(v, comp)
        
        while st:
            u = st.pop()
            if not visited[u]:
                comp = []
                dfs2(u, comp)
                sccs.append(comp)
        
        return sccs
    
    def SCC_Tarjan(self, V, edges):
        """
        Tarjan's algorithm with low-link
        Time Complexity: O(V+E)
        Space Complexity: O(V+E)
        """
        adj = [[] for _ in range(V)]
        for e in edges:
            adj[e[0]].append(e[1])
        
        disc = [-1] * V
        low = [-1] * V
        inStack = [False] * V
        st = []
        sccs = []
        time = [0]
        
        def dfs(u):
            disc[u] = low[u] = time[0]
            time[0] += 1
            st.append(u)
            inStack[u] = True
            
            for v in adj[u]:
                if disc[v] == -1:
                    dfs(v)
                    low[u] = min(low[u], low[v])
                elif inStack[v]:
                    low[u] = min(low[u], disc[v])
            
            if low[u] == disc[u]:
                comp = []
                while st[-1] != u:
                    v = st.pop()
                    inStack[v] = False
                    comp.append(v)
                st.pop()
                inStack[u] = False
                comp.append(u)
                sccs.append(comp)
        
        for i in range(V):
            if disc[i] == -1:
                dfs(i)
        
        return sccs


def Test_SCC_Kosaraju():
    solution = Solution()
    
    V1 = 5
    edges1 = [[1, 0], [0, 2], [2, 1], [0, 3], [3, 4]]
    result1 = solution.SCC_Kosaraju(V1, edges1)
    print(f"Test 1 Kosaraju SCCs: {len(result1)}")
    for scc in result1:
        print("[", end="")
        for v in scc:
            print(v, end=" ")
        print("]", end=" ")
    print()
    
    V2 = 4
    edges2 = [[0, 1], [1, 2], [2, 3], [3, 0]]
    result2 = solution.SCC_Kosaraju(V2, edges2)
    print(f"Test 2 Kosaraju SCCs: {len(result2)}")
    
    V3 = 3
    edges3 = [[0, 1], [1, 2]]
    result3 = solution.SCC_Tarjan(V3, edges3)
    print(f"Test 3 Tarjan SCCs: {len(result3)}")


if __name__ == "__main__":
    Test_SCC_Kosaraju()
