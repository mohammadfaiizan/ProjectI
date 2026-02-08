"""
Problem: Disjoint Set Union (Union-Find)
URL: https://www.geeksforgeeks.org/disjoint-set-data-structures/

Problem Statement:
Implement DSU with union by rank and path compression.

Sample Input/Output:
Input: Operations to unite elements and check connectivity
Output: Results of connectivity checks
"""


class Solution:
    class DSU:
        def __init__(self, n):
            self.parent = list(range(n))
            self.rank = [0] * n
        
        def Find(self, x):
            if self.parent[x] != x:
                self.parent[x] = self.Find(self.parent[x])
            return self.parent[x]
        
        def Unite(self, x, y):
            px = self.Find(x)
            py = self.Find(y)
            
            if px == py:
                return
            
            if self.rank[px] < self.rank[py]:
                self.parent[px] = py
            elif self.rank[px] > self.rank[py]:
                self.parent[py] = px
            else:
                self.parent[py] = px
                self.rank[px] += 1
        
        def Connected(self, x, y):
            return self.Find(x) == self.Find(y)

    def DSU_Rank_Path_Compression(self, n):
        """
        Union by rank + path compression
        Time Complexity: Near O(1) amortized per operation
        Space Complexity: O(n)
        """
        return self.DSU(n)


def Test_DSU():
    solution = Solution()
    
    print("Test Case 1:")
    dsu1 = solution.DSU_Rank_Path_Compression(5)
    dsu1.Unite(0, 1)
    dsu1.Unite(2, 3)
    dsu1.Unite(1, 2)
    
    print(f"0 and 3 connected: {'Yes' if dsu1.Connected(0, 3) else 'No'}")
    print(f"0 and 4 connected: {'Yes' if dsu1.Connected(0, 4) else 'No'}")
    print()
    
    print("Test Case 2:")
    dsu2 = solution.DSU_Rank_Path_Compression(7)
    dsu2.Unite(0, 1)
    dsu2.Unite(1, 2)
    dsu2.Unite(3, 4)
    dsu2.Unite(5, 6)
    dsu2.Unite(2, 3)
    
    print(f"0 and 4 connected: {'Yes' if dsu2.Connected(0, 4) else 'No'}")
    print(f"0 and 5 connected: {'Yes' if dsu2.Connected(0, 5) else 'No'}")
    print(f"5 and 6 connected: {'Yes' if dsu2.Connected(5, 6) else 'No'}")
    print()
    
    print("Test Case 3:")
    dsu3 = solution.DSU_Rank_Path_Compression(4)
    dsu3.Unite(0, 1)
    dsu3.Unite(2, 3)
    
    print(f"0 and 1 connected: {'Yes' if dsu3.Connected(0, 1) else 'No'}")
    print(f"2 and 3 connected: {'Yes' if dsu3.Connected(2, 3) else 'No'}")
    print(f"0 and 2 connected: {'Yes' if dsu3.Connected(0, 2) else 'No'}")
    dsu3.Unite(1, 2)
    print("After uniting 1 and 2:")
    print(f"0 and 2 connected: {'Yes' if dsu3.Connected(0, 2) else 'No'}")
    print(f"0 and 3 connected: {'Yes' if dsu3.Connected(0, 3) else 'No'}")


if __name__ == "__main__":
    Test_DSU()
