"""
200. Number of Islands
Difficulty: Easy

Problem:
Given an m x n 2D binary grid which represents a map of '1's (land) and '0's (water), 
return the number of islands.

An island is surrounded by water and is formed by connecting adjacent lands horizontally 
or vertically. You may assume all four edges of the grid are all surrounded by water.

Examples:
Input: grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
Output: 1

Input: grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
Output: 3

Constraints:
- m == grid.length
- n == grid[i].length
- 1 <= m, n <= 300
- grid[i][j] is '0' or '1'
"""

from typing import List
from collections import deque

class Solution:
    def numIslands_approach1_dfs_recursive(self, grid: List[List[str]]) -> int:
        """
        Approach 1: DFS with Recursion (Classic Island Problem)
        
        For each unvisited land cell, start DFS to mark entire island.
        Count how many times we start a new DFS.
        
        Time: O(M*N) - visit each cell once
        Space: O(M*N) - recursion stack in worst case
        """
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        island_count = 0
        
        def dfs(i, j):
            """Mark all connected land cells as visited"""
            # Boundary check and water check
            if (i < 0 or i >= m or j < 0 or j >= n or grid[i][j] == '0'):
                return
            
            # Mark current cell as visited (sink it)
            grid[i][j] = '0'
            
            # Explore all 4 directions
            dfs(i + 1, j)  # Down
            dfs(i - 1, j)  # Up
            dfs(i, j + 1)  # Right
            dfs(i, j - 1)  # Left
        
        # Scan entire grid
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':  # Found unvisited land
                    island_count += 1
                    dfs(i, j)  # Sink entire island
        
        return island_count
    
    def numIslands_approach2_dfs_iterative(self, grid: List[List[str]]) -> int:
        """
        Approach 2: DFS with Iteration (Stack-based)
        
        Use explicit stack to avoid recursion depth issues.
        
        Time: O(M*N)
        Space: O(M*N) - stack size
        """
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        island_count = 0
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    island_count += 1
                    
                    # DFS using stack
                    stack = [(i, j)]
                    while stack:
                        x, y = stack.pop()
                        
                        # Skip if out of bounds or water
                        if (x < 0 or x >= m or y < 0 or y >= n or grid[x][y] == '0'):
                            continue
                        
                        # Mark as visited
                        grid[x][y] = '0'
                        
                        # Add neighbors to stack
                        for dx, dy in directions:
                            stack.append((x + dx, y + dy))
        
        return island_count
    
    def numIslands_approach3_bfs(self, grid: List[List[str]]) -> int:
        """
        Approach 3: BFS (Level-order traversal)
        
        Use BFS to explore each island level by level.
        
        Time: O(M*N)
        Space: O(min(M,N)) - queue size
        """
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        island_count = 0
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    island_count += 1
                    
                    # BFS using queue
                    queue = deque([(i, j)])
                    grid[i][j] = '0'  # Mark as visited immediately
                    
                    while queue:
                        x, y = queue.popleft()
                        
                        # Explore neighbors
                        for dx, dy in directions:
                            nx, ny = x + dx, y + dy
                            
                            if (0 <= nx < m and 0 <= ny < n and grid[nx][ny] == '1'):
                                grid[nx][ny] = '0'  # Mark as visited
                                queue.append((nx, ny))
        
        return island_count
    
    def numIslands_approach4_union_find(self, grid: List[List[str]]) -> int:
        """
        Approach 4: Union-Find (Disjoint Set Union)
        
        Connect adjacent land cells and count connected components.
        
        Time: O(M*N*α(M*N)) where α is inverse Ackermann function
        Space: O(M*N)
        """
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        
        class UnionFind:
            def __init__(self, grid):
                self.count = 0
                self.parent = []
                self.rank = []
                
                for i in range(m):
                    for j in range(n):
                        if grid[i][j] == '1':
                            self.parent.append(i * n + j)
                            self.count += 1
                        else:
                            self.parent.append(-1)
                        self.rank.append(0)
            
            def find(self, x):
                if self.parent[x] != x:
                    self.parent[x] = self.find(self.parent[x])
                return self.parent[x]
            
            def union(self, x, y):
                px, py = self.find(x), self.find(y)
                if px == py:
                    return
                
                if self.rank[px] < self.rank[py]:
                    px, py = py, px
                
                self.parent[py] = px
                if self.rank[px] == self.rank[py]:
                    self.rank[px] += 1
                
                self.count -= 1
        
        uf = UnionFind(grid)
        directions = [(1, 0), (0, 1)]  # Only check down and right to avoid duplicates
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    for dx, dy in directions:
                        ni, nj = i + dx, j + dy
                        if (0 <= ni < m and 0 <= nj < n and grid[ni][nj] == '1'):
                            uf.union(i * n + j, ni * n + nj)
        
        return uf.count
    
    def numIslands_approach5_non_destructive(self, grid: List[List[str]]) -> int:
        """
        Approach 5: Non-destructive DFS (preserves original grid)
        
        Use visited set instead of modifying the grid.
        
        Time: O(M*N)
        Space: O(M*N) - visited set
        """
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        visited = set()
        island_count = 0
        
        def dfs(i, j):
            if ((i, j) in visited or i < 0 or i >= m or 
                j < 0 or j >= n or grid[i][j] == '0'):
                return
            
            visited.add((i, j))
            
            # Explore 4 directions
            dfs(i + 1, j)
            dfs(i - 1, j)
            dfs(i, j + 1)
            dfs(i, j - 1)
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1' and (i, j) not in visited:
                    island_count += 1
                    dfs(i, j)
        
        return island_count

def test_number_of_islands():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (grid, expected)
        ([
            ["1","1","1","1","0"],
            ["1","1","0","1","0"],
            ["1","1","0","0","0"],
            ["0","0","0","0","0"]
        ], 1),
        ([
            ["1","1","0","0","0"],
            ["1","1","0","0","0"],
            ["0","0","1","0","0"],
            ["0","0","0","1","1"]
        ], 3),
        ([["1"]], 1),  # Single island
        ([["0"]], 0),  # No islands
        ([
            ["1","0","1"],
            ["0","1","0"],
            ["1","0","1"]
        ], 5),  # Diagonal pattern
        ([], 0),  # Empty grid
    ]
    
    approaches = [
        ("DFS Recursive", solution.numIslands_approach1_dfs_recursive),
        ("DFS Iterative", solution.numIslands_approach2_dfs_iterative),
        ("BFS", solution.numIslands_approach3_bfs),
        ("Union-Find", solution.numIslands_approach4_union_find),
        ("Non-destructive DFS", solution.numIslands_approach5_non_destructive),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (grid, expected) in enumerate(test_cases):
            # Create deep copy for testing
            test_grid = [row[:] for row in grid]
            result = func(test_grid)
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} Expected: {expected}, Got: {result}")

def demonstrate_island_exploration():
    """Demonstrate how DFS explores islands"""
    print("\n=== Island Exploration Demo ===")
    
    grid = [
        ["1","1","0","0","0"],
        ["1","1","0","0","0"],
        ["0","0","1","0","0"],
        ["0","0","0","1","1"]
    ]
    
    print("Original Grid:")
    for i, row in enumerate(grid):
        print(f"  Row {i}: {row}")
    
    # Manual DFS exploration
    m, n = len(grid), len(grid[0])
    visited = set()
    islands = []
    
    def dfs_trace(i, j, island_cells):
        if ((i, j) in visited or i < 0 or i >= m or 
            j < 0 or j >= n or grid[i][j] == '0'):
            return
        
        visited.add((i, j))
        island_cells.append((i, j))
        
        # Explore 4 directions
        dfs_trace(i + 1, j, island_cells)
        dfs_trace(i - 1, j, island_cells)
        dfs_trace(i, j + 1, island_cells)
        dfs_trace(i, j - 1, island_cells)
    
    for i in range(m):
        for j in range(n):
            if grid[i][j] == '1' and (i, j) not in visited:
                island_cells = []
                dfs_trace(i, j, island_cells)
                if island_cells:
                    islands.append(island_cells)
    
    print(f"\nFound {len(islands)} islands:")
    for idx, island in enumerate(islands):
        print(f"  Island {idx + 1}: {island}")

def analyze_algorithm_performance():
    """Analyze performance characteristics of different approaches"""
    print("\n=== Algorithm Performance Analysis ===")
    
    approaches = [
        ("DFS Recursive", "O(M*N)", "O(M*N)", "Simple, may hit recursion limit"),
        ("DFS Iterative", "O(M*N)", "O(M*N)", "No recursion limit, explicit stack"),
        ("BFS", "O(M*N)", "O(min(M,N))", "Level-order, better space in some cases"),
        ("Union-Find", "O(M*N*α(M*N))", "O(M*N)", "Good for dynamic connectivity"),
        ("Non-destructive", "O(M*N)", "O(M*N)", "Preserves input, extra space"),
    ]
    
    print(f"{'Approach':<18} {'Time':<12} {'Space':<12} {'Notes'}")
    print("-" * 70)
    
    for approach, time_comp, space_comp, notes in approaches:
        print(f"{approach:<18} {time_comp:<12} {space_comp:<12} {notes}")
    
    print(f"\nKey Insights:")
    print(f"- All approaches have same time complexity O(M*N)")
    print(f"- Space complexity varies based on data structure choice")
    print(f"- DFS is most natural for connected component problems")
    print(f"- Choice depends on constraints and requirements")

if __name__ == "__main__":
    test_number_of_islands()
    demonstrate_island_exploration()
    analyze_algorithm_performance()

"""
Graph Theory Concepts:
1. Connected Components in Grid Graphs
2. Depth-First Search (DFS) traversal
3. Graph representation in 2D grids
4. Island detection and counting

Key DFS Concepts:
- Recursive vs Iterative implementation
- Marking visited nodes (destructive vs non-destructive)
- 4-directional movement in grids
- Connected component enumeration

Algorithm Variants:
- DFS: Natural recursive approach
- BFS: Level-order exploration
- Union-Find: Dynamic connectivity approach
- Stack-based: Explicit stack management

Real-world Applications:
- Geographic information systems (land mass detection)
- Image processing (connected region detection)
- Network analysis (cluster identification)
- Game development (pathfinding, territory detection)
- Computer graphics (flood fill algorithms)

This is the foundational island problem that appears in many variations!
"""
