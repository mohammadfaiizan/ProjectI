"""
Problem: Number of Islands
URL: https://leetcode.com/problems/number-of-islands/

Problem Statement:
Given a 2D grid of '1' (land) and '0' (water), count the number of islands. An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically.

Sample Input/Output:
Input: grid = [["1","1","1","1","0"],["1","1","0","1","0"],["1","1","0","0","0"],["0","0","0","0","0"]]
Output: 1
"""


class Solution:
    def Islands_DFS(self, grid):
        """
        DFS flood fill
        Time Complexity: O(M*N)
        Space Complexity: O(M*N)
        """
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        count = 0
        
        def dfs(i, j):
            if i < 0 or i >= m or j < 0 or j >= n or grid[i][j] != '1':
                return
            grid[i][j] = '0'
            dfs(i + 1, j)
            dfs(i - 1, j)
            dfs(i, j + 1)
            dfs(i, j - 1)
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    count += 1
                    dfs(i, j)
        
        return count
    
    def Islands_BFS(self, grid):
        """
        BFS flood fill
        Time Complexity: O(M*N)
        Space Complexity: O(M*N)
        """
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        count = 0
        dx = [1, -1, 0, 0]
        dy = [0, 0, 1, -1]
        
        from collections import deque
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    count += 1
                    q = deque([(i, j)])
                    grid[i][j] = '0'
                    
                    while q:
                        x, y = q.popleft()
                        
                        for k in range(4):
                            nx = x + dx[k]
                            ny = y + dy[k]
                            if 0 <= nx < m and 0 <= ny < n and grid[nx][ny] == '1':
                                grid[nx][ny] = '0'
                                q.append((nx, ny))
        
        return count
    
    def Islands_Union_Find(self, grid):
        """
        Disjoint Set Union
        Time Complexity: O(M*N)
        Space Complexity: O(M*N)
        """
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        parent = list(range(m * n))
        rank = [0] * (m * n)
        count = 0
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def unite(x, y):
            nonlocal count
            x = find(x)
            y = find(y)
            if x != y:
                if rank[x] < rank[y]:
                    x, y = y, x
                parent[y] = x
                if rank[x] == rank[y]:
                    rank[x] += 1
                count -= 1
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    count += 1
                    idx = i * n + j
                    if i > 0 and grid[i - 1][j] == '1':
                        unite(idx, (i - 1) * n + j)
                    if j > 0 and grid[i][j - 1] == '1':
                        unite(idx, i * n + (j - 1))
        
        return count


def Test_Islands():
    solution = Solution()
    
    print("Test Case 1: Single island")
    grid1 = [
        ['1', '1', '1', '1', '0'],
        ['1', '1', '0', '1', '0'],
        ['1', '1', '0', '0', '0'],
        ['0', '0', '0', '0', '0']
    ]
    grid1_copy1 = [row[:] for row in grid1]
    grid1_copy2 = [row[:] for row in grid1]
    print("DFS Result:", solution.Islands_DFS(grid1_copy1))
    print("BFS Result:", solution.Islands_BFS(grid1_copy2))
    print("Union-Find Result:", solution.Islands_Union_Find([row[:] for row in grid1]))
    
    print("\nTest Case 2: Multiple islands")
    grid2 = [
        ['1', '1', '0', '0', '0'],
        ['1', '1', '0', '0', '0'],
        ['0', '0', '1', '0', '0'],
        ['0', '0', '0', '1', '1']
    ]
    grid2_copy1 = [row[:] for row in grid2]
    grid2_copy2 = [row[:] for row in grid2]
    print("DFS Result:", solution.Islands_DFS(grid2_copy1))
    print("BFS Result:", solution.Islands_BFS(grid2_copy2))
    print("Union-Find Result:", solution.Islands_Union_Find([row[:] for row in grid2]))


if __name__ == "__main__":
    Test_Islands()
