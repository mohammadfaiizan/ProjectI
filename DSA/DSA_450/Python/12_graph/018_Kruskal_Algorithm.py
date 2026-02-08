"""
Problem: Kruskal's Minimum Spanning Tree Algorithm
URL: https://practice.geeksforgeeks.org/problems/minimum-spanning-tree/1

Problem Statement:
Find the Minimum Spanning Tree (MST) of a weighted undirected graph using Kruskal's algorithm. The algorithm sorts all edges by weight and uses Union-Find to avoid cycles.

Sample Input/Output:
Input: Graph with edges (0,1,10), (0,2,6), (0,3,5), (1,3,15), (2,3,4)
Output: MST weight = 19
"""


class Solution:
    def Kruskal_MST(self, V, edges):
        """
        Sort edges by weight, union-find with path compression and rank
        Time Complexity: O(E log E)
        Space Complexity: O(V)
        """
        edges.sort(key=lambda x: x[2])
        
        parent = list(range(V))
        rank = [0] * V
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def unite(x, y):
            x = find(x)
            y = find(y)
            if x == y:
                return False
            if rank[x] < rank[y]:
                x, y = y, x
            parent[y] = x
            if rank[x] == rank[y]:
                rank[x] += 1
            return True
        
        mstWeight = 0
        edgesAdded = 0
        
        for edge in edges:
            u, v, w = edge[0], edge[1], edge[2]
            
            if unite(u, v):
                mstWeight += w
                edgesAdded += 1
                if edgesAdded == V - 1:
                    break
        
        return mstWeight


def Test_Kruskal():
    solution = Solution()
    
    print("Test Case 1: Weighted graph with 5 vertices")
    V1 = 4
    edges1 = [
        [0, 1, 10],
        [0, 2, 6],
        [0, 3, 5],
        [1, 3, 15],
        [2, 3, 4]
    ]
    print(f"MST Weight: {solution.Kruskal_MST(V1, edges1)}")
    
    print("\nTest Case 2: Complete graph")
    V2 = 5
    edges2 = [
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 6],
        [0, 4, 5],
        [1, 2, 5],
        [1, 3, 3],
        [1, 4, 4],
        [2, 3, 1],
        [2, 4, 2],
        [3, 4, 3]
    ]
    print(f"MST Weight: {solution.Kruskal_MST(V2, edges2)}")
    
    print("\nTest Case 3: Simple triangle")
    V3 = 3
    edges3 = [
        [0, 1, 1],
        [1, 2, 2],
        [0, 2, 3]
    ]
    print(f"MST Weight: {solution.Kruskal_MST(V3, edges3)}")


if __name__ == "__main__":
    Test_Kruskal()
