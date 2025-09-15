"""
Problem: Number of Islands
URL: https://leetcode.com/problems/number-of-islands/description/

Problem Statement:
Given an m x n 2D binary grid grid which represents a map of '1's (land) and '0's (water), 
return the number of islands.
An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. 
You may assume all four edges of the grid are all surrounded by water.

Sample Input/Output:
Input: grid = [["1","1","1","1","0"],
               ["1","1","0","1","0"],
               ["1","1","0","0","0"],
               ["0","0","0","0","0"]]
Output: 1

Input: grid = [["1","1","0","0","0"],
               ["1","1","0","0","0"],
               ["0","0","1","0","0"],
               ["0","0","0","1","1"]]
Output: 3
"""

from typing import List
from collections import deque

class Solution:
    def Num_Islands_DFS_Recursive(self, grid: List[List[str]]) -> int:
        """
        DFS Recursive - Mark visited lands recursively
        Time Complexity: O(m * n)
        Space Complexity: O(m * n)
        """
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        count = 0
        
        def DFS(i: int, j: int) -> None:
            if i < 0 or i >= m or j < 0 or j >= n or grid[i][j] != '1':
                return
            
            grid[i][j] = '0'
            
            DFS(i + 1, j)
            DFS(i - 1, j)
            DFS(i, j + 1)
            DFS(i, j - 1)
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    count += 1
                    DFS(i, j)
        
        return count
    
    def Num_Islands_BFS_Queue(self, grid: List[List[str]]) -> int:
        """
        BFS Queue - Use queue for level-wise exploration
        Time Complexity: O(m * n)
        Space Complexity: O(min(m, n))
        """
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        count = 0
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    count += 1
                    queue = deque([(i, j)])
                    grid[i][j] = '0'
                    
                    while queue:
                        x, y = queue.popleft()
                        for dx, dy in directions:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < m and 0 <= ny < n and grid[nx][ny] == '1':
                                grid[nx][ny] = '0'
                                queue.append((nx, ny))
        
        return count
    
    def Num_Islands_Union_Find(self, grid: List[List[str]]) -> int:
        """
        Union Find - Use disjoint set union
        Time Complexity: O(m * n * α(m * n))
        Space Complexity: O(m * n)
        """
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        
        class UnionFind:
            def __init__(self, size: int):
                self.parent = list(range(size))
                self.rank = [0] * size
                self.count = 0
            
            def Find(self, x: int) -> int:
                if self.parent[x] != x:
                    self.parent[x] = self.Find(self.parent[x])
                return self.parent[x]
            
            def Union(self, x: int, y: int) -> None:
                root_x, root_y = self.Find(x), self.Find(y)
                if root_x != root_y:
                    if self.rank[root_x] < self.rank[root_y]:
                        self.parent[root_x] = root_y
                    elif self.rank[root_x] > self.rank[root_y]:
                        self.parent[root_y] = root_x
                    else:
                        self.parent[root_y] = root_x
                        self.rank[root_x] += 1
                    self.count -= 1
        
        uf = UnionFind(m * n)
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    uf.count += 1
        
        directions = [(0, 1), (1, 0)]
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    for dx, dy in directions:
                        ni, nj = i + dx, j + dy
                        if 0 <= ni < m and 0 <= nj < n and grid[ni][nj] == '1':
                            uf.Union(i * n + j, ni * n + nj)
        
        return uf.count
    
    def Num_Islands_DFS_Iterative(self, grid: List[List[str]]) -> int:
        """
        DFS Iterative - Use stack for DFS
        Time Complexity: O(m * n)
        Space Complexity: O(m * n)
        """
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        count = 0
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    count += 1
                    stack = [(i, j)]
                    
                    while stack:
                        x, y = stack.pop()
                        if 0 <= x < m and 0 <= y < n and grid[x][y] == '1':
                            grid[x][y] = '0'
                            for dx, dy in directions:
                                stack.append((x + dx, y + dy))
        
        return count
    
    def Num_Islands_Visited_Array(self, grid: List[List[str]]) -> int:
        """
        Visited Array - Use separate visited array
        Time Complexity: O(m * n)
        Space Complexity: O(m * n)
        """
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        visited = [[False] * n for _ in range(m)]
        count = 0
        
        def DFS(i: int, j: int) -> None:
            if (i < 0 or i >= m or j < 0 or j >= n or 
                visited[i][j] or grid[i][j] != '1'):
                return
            
            visited[i][j] = True
            
            DFS(i + 1, j)
            DFS(i - 1, j)
            DFS(i, j + 1)
            DFS(i, j - 1)
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1' and not visited[i][j]:
                    count += 1
                    DFS(i, j)
        
        return count

def Test_Num_Islands():
    solution = Solution()
    
    test_cases = [
        ([["1","1","1","1","0"],
          ["1","1","0","1","0"],
          ["1","1","0","0","0"],
          ["0","0","0","0","0"]], 1),
        ([["1","1","0","0","0"],
          ["1","1","0","0","0"],
          ["0","0","1","0","0"],
          ["0","0","0","1","1"]], 3),
        ([["1","0","1"],
          ["0","1","0"],
          ["1","0","1"]], 5),
        ([["1"]], 1),
        ([["0"]], 0)
    ]
    
    methods = [
        ("DFS Recursive", solution.Num_Islands_DFS_Recursive),
        ("BFS Queue", solution.Num_Islands_BFS_Queue),
        ("Union Find", solution.Num_Islands_Union_Find),
        ("DFS Iterative", solution.Num_Islands_DFS_Iterative),
        ("Visited Array", solution.Num_Islands_Visited_Array)
    ]
    
    for grid, expected in test_cases:
        print(f"Grid:")
        for row in grid:
            print(f"  {row}")
        print(f"Expected: {expected}")
        
        for method_name, method in methods:
            grid_copy = [row.copy() for row in grid]
            result = method(grid_copy)
            print(f"{method_name}: {result}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Num_Islands()
