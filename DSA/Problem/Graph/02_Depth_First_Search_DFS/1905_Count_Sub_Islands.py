"""
1905. Count Sub Islands - Multiple Approaches
Difficulty: Medium

You are given two m x n binary matrices grid1 and grid2 containing only 0's (representing water) and 1's (representing land). An island is a group of 1's connected 4-directionally (horizontal or vertical). Any cells outside of the grid are considered water cells.

An island in grid2 is considered a sub-island if there is an island in grid1 that contains all the cells that make up this island in grid2.

Return the number of islands in grid2 that are considered sub-islands of grid1.
"""

from typing import List, Set, Tuple
from collections import deque

class CountSubIslands:
    """Multiple approaches to count sub-islands"""
    
    def countSubIslands_dfs_validation(self, grid1: List[List[int]], grid2: List[List[int]]) -> int:
        """
        Approach 1: DFS with Island Validation
        
        For each island in grid2, check if it's completely contained in grid1.
        
        Time: O(mn), Space: O(mn)
        """
        m, n = len(grid1), len(grid1[0])
        visited = [[False] * n for _ in range(m)]
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        def dfs(i: int, j: int, island_cells: List[Tuple[int, int]]) -> None:
            """DFS to collect all cells of current island"""
            if (i < 0 or i >= m or j < 0 or j >= n or 
                visited[i][j] or grid2[i][j] == 0):
                return
            
            visited[i][j] = True
            island_cells.append((i, j))
            
            for di, dj in directions:
                dfs(i + di, j + dj, island_cells)
        
        def is_sub_island(island_cells: List[Tuple[int, int]]) -> bool:
            """Check if all cells of island exist in grid1"""
            return all(grid1[i][j] == 1 for i, j in island_cells)
        
        sub_islands = 0
        
        # Find all islands in grid2
        for i in range(m):
            for j in range(n):
                if grid2[i][j] == 1 and not visited[i][j]:
                    island_cells = []
                    dfs(i, j, island_cells)
                    
                    if is_sub_island(island_cells):
                        sub_islands += 1
        
        return sub_islands
    
    def countSubIslands_dfs_early_termination(self, grid1: List[List[int]], grid2: List[List[int]]) -> int:
        """
        Approach 2: DFS with Early Termination
        
        Check validity during DFS traversal for early termination.
        
        Time: O(mn), Space: O(mn)
        """
        m, n = len(grid1), len(grid1[0])
        visited = [[False] * n for _ in range(m)]
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        def dfs(i: int, j: int) -> bool:
            """DFS that returns False if island is not a sub-island"""
            if i < 0 or i >= m or j < 0 or j >= n or visited[i][j] or grid2[i][j] == 0:
                return True
            
            visited[i][j] = True
            
            # If current cell is not land in grid1, this is not a sub-island
            is_valid = grid1[i][j] == 1
            
            # Continue DFS and combine results
            for di, dj in directions:
                is_valid &= dfs(i + di, j + dj)
            
            return is_valid
        
        sub_islands = 0
        
        for i in range(m):
            for j in range(n):
                if grid2[i][j] == 1 and not visited[i][j]:
                    if dfs(i, j):
                        sub_islands += 1
        
        return sub_islands
    
    def countSubIslands_bfs(self, grid1: List[List[int]], grid2: List[List[int]]) -> int:
        """
        Approach 3: BFS Island Detection
        
        Use BFS to find islands and validate them.
        
        Time: O(mn), Space: O(mn)
        """
        m, n = len(grid1), len(grid1[0])
        visited = [[False] * n for _ in range(m)]
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        def bfs(start_i: int, start_j: int) -> bool:
            """BFS to check if island is a sub-island"""
            queue = deque([(start_i, start_j)])
            visited[start_i][start_j] = True
            is_sub_island = True
            
            while queue:
                i, j = queue.popleft()
                
                # Check if current cell exists in grid1
                if grid1[i][j] == 0:
                    is_sub_island = False
                
                # Add neighbors to queue
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    
                    if (0 <= ni < m and 0 <= nj < n and 
                        not visited[ni][nj] and grid2[ni][nj] == 1):
                        visited[ni][nj] = True
                        queue.append((ni, nj))
            
            return is_sub_island
        
        sub_islands = 0
        
        for i in range(m):
            for j in range(n):
                if grid2[i][j] == 1 and not visited[i][j]:
                    if bfs(i, j):
                        sub_islands += 1
        
        return sub_islands
    
    def countSubIslands_mark_invalid(self, grid1: List[List[int]], grid2: List[List[int]]) -> int:
        """
        Approach 4: Mark Invalid Islands First
        
        First mark all invalid islands in grid2, then count remaining islands.
        
        Time: O(mn), Space: O(mn)
        """
        m, n = len(grid1), len(grid1[0])
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        def dfs_mark_invalid(i: int, j: int):
            """DFS to mark invalid island cells as water"""
            if i < 0 or i >= m or j < 0 or j >= n or grid2[i][j] == 0:
                return
            
            grid2[i][j] = 0  # Mark as water
            
            for di, dj in directions:
                dfs_mark_invalid(i + di, j + dj)
        
        def dfs_count_island(i: int, j: int):
            """DFS to mark island as visited"""
            if i < 0 or i >= m or j < 0 or j >= n or grid2[i][j] == 0:
                return
            
            grid2[i][j] = 0  # Mark as visited
            
            for di, dj in directions:
                dfs_count_island(i + di, j + dj)
        
        # First pass: mark invalid islands
        for i in range(m):
            for j in range(n):
                if grid2[i][j] == 1 and grid1[i][j] == 0:
                    dfs_mark_invalid(i, j)
        
        # Second pass: count remaining valid islands
        sub_islands = 0
        for i in range(m):
            for j in range(n):
                if grid2[i][j] == 1:
                    dfs_count_island(i, j)
                    sub_islands += 1
        
        return sub_islands
    
    def countSubIslands_union_find(self, grid1: List[List[int]], grid2: List[List[int]]) -> int:
        """
        Approach 5: Union-Find for Island Grouping
        
        Use Union-Find to group island cells and validate each group.
        
        Time: O(mn α(mn)), Space: O(mn)
        """
        m, n = len(grid1), len(grid1[0])
        
        # Union-Find implementation
        parent = {}
        
        def find(x):
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Connect adjacent land cells in grid2
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        for i in range(m):
            for j in range(n):
                if grid2[i][j] == 1:
                    for di, dj in directions:
                        ni, nj = i + di, j + dj
                        if (0 <= ni < m and 0 <= nj < n and 
                            grid2[ni][nj] == 1):
                            union((i, j), (ni, nj))
        
        # Group cells by island
        islands = {}
        for i in range(m):
            for j in range(n):
                if grid2[i][j] == 1:
                    root = find((i, j))
                    if root not in islands:
                        islands[root] = []
                    islands[root].append((i, j))
        
        # Count valid sub-islands
        sub_islands = 0
        for island_cells in islands.values():
            if all(grid1[i][j] == 1 for i, j in island_cells):
                sub_islands += 1
        
        return sub_islands
    
    def countSubIslands_optimized_single_pass(self, grid1: List[List[int]], grid2: List[List[int]]) -> int:
        """
        Approach 6: Optimized Single Pass
        
        Combine marking and counting in single DFS pass.
        
        Time: O(mn), Space: O(mn)
        """
        m, n = len(grid1), len(grid1[0])
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        def dfs(i: int, j: int) -> bool:
            """DFS that marks visited and returns validity"""
            if i < 0 or i >= m or j < 0 or j >= n or grid2[i][j] == 0:
                return True
            
            grid2[i][j] = 0  # Mark as visited
            
            # Check if current cell is valid
            is_valid = grid1[i][j] == 1
            
            # Continue DFS and combine results
            for di, dj in directions:
                is_valid &= dfs(i + di, j + dj)
            
            return is_valid
        
        sub_islands = 0
        
        for i in range(m):
            for j in range(n):
                if grid2[i][j] == 1:
                    if dfs(i, j):
                        sub_islands += 1
        
        return sub_islands

def test_count_sub_islands():
    """Test count sub-islands algorithms"""
    solver = CountSubIslands()
    
    test_cases = [
        (
            [[1,1,1,0,0],[0,1,1,1,1],[0,0,0,0,0],[1,0,0,0,0],[1,1,0,1,1]],
            [[1,1,1,0,0],[0,0,1,1,1],[0,1,0,0,0],[1,0,1,1,0],[0,1,0,1,0]],
            3, "Example 1"
        ),
        (
            [[1,0,1,0,1],[1,1,1,1,1],[0,0,0,0,0],[1,1,1,1,1],[1,0,1,0,1]],
            [[0,0,0,0,0],[1,1,1,1,1],[0,1,0,1,0],[0,1,0,1,0],[1,0,0,0,1]],
            2, "Example 2"
        ),
        (
            [[1,1],[1,1]],
            [[1,0],[0,1]],
            0, "No sub-islands"
        ),
        (
            [[1,1],[1,1]],
            [[1,1],[1,1]],
            1, "Complete overlap"
        ),
    ]
    
    algorithms = [
        ("DFS Validation", solver.countSubIslands_dfs_validation),
        ("DFS Early Term", solver.countSubIslands_dfs_early_termination),
        ("BFS", solver.countSubIslands_bfs),
        ("Union-Find", solver.countSubIslands_union_find),
    ]
    
    print("=== Testing Count Sub Islands ===")
    
    for grid1, grid2, expected, description in test_cases:
        print(f"\n--- {description} (Expected: {expected}) ---")
        print(f"Grid1: {grid1}")
        print(f"Grid2: {grid2}")
        
        for alg_name, alg_func in algorithms:
            try:
                # Create copies since some algorithms modify input
                g1_copy = [row[:] for row in grid1]
                g2_copy = [row[:] for row in grid2]
                result = alg_func(g1_copy, g2_copy)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:15} | {status} | Sub-islands: {result}")
            except Exception as e:
                print(f"{alg_name:15} | ERROR: {str(e)[:30]}")

if __name__ == "__main__":
    test_count_sub_islands()

"""
Count Sub Islands demonstrates advanced island detection
with containment validation using DFS, BFS, and Union-Find
for complex grid-based relationship analysis.
"""
