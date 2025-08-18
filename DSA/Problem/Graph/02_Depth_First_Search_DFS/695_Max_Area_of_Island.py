"""
695. Max Area of Island
Difficulty: Easy

Problem:
You are given an m x n binary matrix grid. An island is a group of 1's (representing land) 
connected 4-directionally (horizontal or vertical.) You may assume all four edges of the 
grid are surrounded by water.

The area of an island is the number of cells with a value 1 in the island.

Return the maximum area of an island in grid. If there is no island, return 0.

Examples:
Input: grid = [[0,0,1,0,0,0,0,1,0,0,0,0,0],
               [0,0,0,0,0,0,0,1,1,1,0,0,0],
               [0,1,1,0,1,0,0,0,0,0,0,0,0],
               [0,1,0,0,1,1,0,0,1,0,1,0,0],
               [0,1,0,0,1,1,0,0,1,1,1,0,0],
               [0,0,0,0,0,0,0,0,0,0,1,0,0],
               [0,0,0,0,0,0,0,1,1,1,0,0,0],
               [0,0,0,0,0,0,0,1,1,0,0,0,0]]
Output: 6

Input: grid = [[0,0,0,0,0,0,0,0]]
Output: 0

Constraints:
- m == grid.length
- n == grid[i].length
- 1 <= m, n <= 50
- grid[i][j] is 0 or 1
"""

from typing import List
from collections import deque

class Solution:
    def maxAreaOfIsland_approach1_dfs_recursive(self, grid: List[List[int]]) -> int:
        """
        Approach 1: DFS with Recursion (Return area)
        
        For each unvisited island cell, calculate area using DFS.
        Track maximum area seen so far.
        
        Time: O(M*N) - visit each cell once
        Space: O(M*N) - recursion stack depth
        """
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        max_area = 0
        
        def dfs(i, j):
            """Return area of island starting from (i,j)"""
            # Base case: out of bounds or water
            if (i < 0 or i >= m or j < 0 or j >= n or grid[i][j] == 0):
                return 0
            
            # Mark current cell as visited
            grid[i][j] = 0
            
            # Count current cell + area from all 4 directions
            area = 1
            area += dfs(i + 1, j)  # Down
            area += dfs(i - 1, j)  # Up  
            area += dfs(i, j + 1)  # Right
            area += dfs(i, j - 1)  # Left
            
            return area
        
        # Scan entire grid
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:  # Found unvisited land
                    area = dfs(i, j)
                    max_area = max(max_area, area)
        
        return max_area
    
    def maxAreaOfIsland_approach2_dfs_iterative(self, grid: List[List[int]]) -> int:
        """
        Approach 2: DFS with Iteration (Stack-based)
        
        Use explicit stack to avoid recursion depth issues.
        
        Time: O(M*N)
        Space: O(M*N) - stack size
        """
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        max_area = 0
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    # Calculate area using DFS with stack
                    area = 0
                    stack = [(i, j)]
                    
                    while stack:
                        x, y = stack.pop()
                        
                        # Skip if out of bounds or water
                        if (x < 0 or x >= m or y < 0 or y >= n or grid[x][y] == 0):
                            continue
                        
                        # Mark as visited and count
                        grid[x][y] = 0
                        area += 1
                        
                        # Add neighbors to stack
                        for dx, dy in directions:
                            stack.append((x + dx, y + dy))
                    
                    max_area = max(max_area, area)
        
        return max_area
    
    def maxAreaOfIsland_approach3_bfs(self, grid: List[List[int]]) -> int:
        """
        Approach 3: BFS (Level-order traversal)
        
        Use BFS to explore each island and calculate its area.
        
        Time: O(M*N)
        Space: O(min(M,N)) - queue size
        """
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        max_area = 0
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    # Calculate area using BFS
                    area = 0
                    queue = deque([(i, j)])
                    grid[i][j] = 0  # Mark as visited immediately
                    
                    while queue:
                        x, y = queue.popleft()
                        area += 1
                        
                        # Explore neighbors
                        for dx, dy in directions:
                            nx, ny = x + dx, y + dy
                            
                            if (0 <= nx < m and 0 <= ny < n and grid[nx][ny] == 1):
                                grid[nx][ny] = 0  # Mark as visited
                                queue.append((nx, ny))
                    
                    max_area = max(max_area, area)
        
        return max_area
    
    def maxAreaOfIsland_approach4_non_destructive(self, grid: List[List[int]]) -> int:
        """
        Approach 4: Non-destructive DFS (preserves original grid)
        
        Use visited set instead of modifying the grid.
        
        Time: O(M*N)
        Space: O(M*N) - visited set + recursion stack
        """
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        visited = set()
        max_area = 0
        
        def dfs(i, j):
            if ((i, j) in visited or i < 0 or i >= m or 
                j < 0 or j >= n or grid[i][j] == 0):
                return 0
            
            visited.add((i, j))
            
            # Count current cell + area from 4 directions
            area = 1
            area += dfs(i + 1, j)
            area += dfs(i - 1, j)
            area += dfs(i, j + 1)
            area += dfs(i, j - 1)
            
            return area
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1 and (i, j) not in visited:
                    area = dfs(i, j)
                    max_area = max(max_area, area)
        
        return max_area
    
    def maxAreaOfIsland_approach5_union_find(self, grid: List[List[int]]) -> int:
        """
        Approach 5: Union-Find with component size tracking
        
        Connect adjacent land cells and track component sizes.
        
        Time: O(M*N*α(M*N))
        Space: O(M*N)
        """
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        
        class UnionFind:
            def __init__(self, grid):
                self.parent = {}
                self.size = {}
                self.max_size = 0
                
                for i in range(m):
                    for j in range(n):
                        if grid[i][j] == 1:
                            idx = i * n + j
                            self.parent[idx] = idx
                            self.size[idx] = 1
                            self.max_size = max(self.max_size, 1)
            
            def find(self, x):
                if self.parent[x] != x:
                    self.parent[x] = self.find(self.parent[x])
                return self.parent[x]
            
            def union(self, x, y):
                px, py = self.find(x), self.find(y)
                if px == py:
                    return
                
                # Union by size
                if self.size[px] < self.size[py]:
                    px, py = py, px
                
                self.parent[py] = px
                self.size[px] += self.size[py]
                self.max_size = max(self.max_size, self.size[px])
        
        uf = UnionFind(grid)
        directions = [(1, 0), (0, 1)]  # Only check down and right
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    for dx, dy in directions:
                        ni, nj = i + dx, j + dy
                        if (0 <= ni < m and 0 <= nj < n and grid[ni][nj] == 1):
                            uf.union(i * n + j, ni * n + nj)
        
        return uf.max_size

def test_max_area_of_island():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (grid, expected)
        ([[0,0,1,0,0,0,0,1,0,0,0,0,0],
          [0,0,0,0,0,0,0,1,1,1,0,0,0],
          [0,1,1,0,1,0,0,0,0,0,0,0,0],
          [0,1,0,0,1,1,0,0,1,0,1,0,0],
          [0,1,0,0,1,1,0,0,1,1,1,0,0],
          [0,0,0,0,0,0,0,0,0,0,1,0,0],
          [0,0,0,0,0,0,0,1,1,1,0,0,0],
          [0,0,0,0,0,0,0,1,1,0,0,0,0]], 6),
        ([[0,0,0,0,0,0,0,0]], 0),
        ([[1]], 1),
        ([[1,1,0],[0,1,0],[0,0,1]], 3),
        ([[1,1,1,1,1]], 5),
        ([[1],[1],[1],[1],[1]], 5),
    ]
    
    approaches = [
        ("DFS Recursive", solution.maxAreaOfIsland_approach1_dfs_recursive),
        ("DFS Iterative", solution.maxAreaOfIsland_approach2_dfs_iterative),
        ("BFS", solution.maxAreaOfIsland_approach3_bfs),
        ("Non-destructive DFS", solution.maxAreaOfIsland_approach4_non_destructive),
        ("Union-Find", solution.maxAreaOfIsland_approach5_union_find),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (grid, expected) in enumerate(test_cases):
            # Create deep copy for testing
            test_grid = [row[:] for row in grid]
            result = func(test_grid)
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} Expected: {expected}, Got: {result}")

def demonstrate_area_calculation():
    """Demonstrate how DFS calculates island areas"""
    print("\n=== Area Calculation Demo ===")
    
    grid = [
        [1,1,0,0,0],
        [1,1,0,0,0],
        [0,0,1,0,0],
        [0,0,0,1,1]
    ]
    
    print("Original Grid:")
    for i, row in enumerate(grid):
        print(f"  Row {i}: {row}")
    
    # Manual area calculation
    m, n = len(grid), len(grid[0])
    visited = set()
    islands = []
    
    def dfs_area(i, j):
        if ((i, j) in visited or i < 0 or i >= m or 
            j < 0 or j >= n or grid[i][j] == 0):
            return 0, []
        
        visited.add((i, j))
        cells = [(i, j)]
        area = 1
        
        # Explore 4 directions
        for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            sub_area, sub_cells = dfs_area(i + di, j + dj)
            area += sub_area
            cells.extend(sub_cells)
        
        return area, cells
    
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1 and (i, j) not in visited:
                area, cells = dfs_area(i, j)
                islands.append((area, cells))
    
    print(f"\nIsland Analysis:")
    max_area = 0
    for idx, (area, cells) in enumerate(islands):
        print(f"  Island {idx + 1}: Area = {area}, Cells = {cells}")
        max_area = max(max_area, area)
    
    print(f"\nMaximum Area: {max_area}")

def visualize_island_areas():
    """Create visual representation of islands with areas"""
    print("\n=== Island Area Visualization ===")
    
    grid = [
        [1,1,0,0,0],
        [1,0,0,1,1],
        [0,0,0,1,0],
        [0,1,1,0,0]
    ]
    
    print("Grid with islands:")
    for i, row in enumerate(grid):
        display_row = []
        for j, cell in enumerate(row):
            if cell == 1:
                display_row.append("🏝️")
            else:
                display_row.append("🌊")
        print(f"  Row {i}: {' '.join(display_row)}")
    
    # Calculate and label islands
    m, n = len(grid), len(grid[0])
    visited = set()
    island_map = {}
    island_num = 1
    
    def dfs_label(i, j, label):
        if ((i, j) in visited or i < 0 or i >= m or 
            j < 0 or j >= n or grid[i][j] == 0):
            return 0
        
        visited.add((i, j))
        island_map[(i, j)] = label
        area = 1
        
        for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            area += dfs_label(i + di, j + dj, label)
        
        return area
    
    island_areas = {}
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1 and (i, j) not in visited:
                area = dfs_label(i, j, island_num)
                island_areas[island_num] = area
                island_num += 1
    
    print(f"\nLabeled islands with areas:")
    for i in range(m):
        display_row = []
        for j in range(n):
            if (i, j) in island_map:
                label = island_map[(i, j)]
                area = island_areas[label]
                display_row.append(f"{label}({area})")
            else:
                display_row.append(" 🌊 ")
        print(f"  Row {i}: {' '.join(display_row)}")

def compare_algorithms():
    """Compare different algorithm approaches"""
    print("\n=== Algorithm Comparison ===")
    
    algorithms = [
        ("DFS Recursive", "Natural, clean code", "May hit recursion limit"),
        ("DFS Iterative", "No recursion limit", "More complex code"),
        ("BFS", "Level-order exploration", "Queue overhead"),
        ("Non-destructive", "Preserves input", "Extra space for visited"),
        ("Union-Find", "Good for multiple queries", "Overkill for single query"),
    ]
    
    print(f"{'Algorithm':<18} {'Advantages':<25} {'Disadvantages'}")
    print("-" * 70)
    
    for alg, advantages, disadvantages in algorithms:
        print(f"{alg:<18} {advantages:<25} {disadvantages}")
    
    print(f"\nRecommendation:")
    print(f"- Use DFS Recursive for clean, readable code")
    print(f"- Use DFS Iterative for very large grids")
    print(f"- Use Non-destructive if input must be preserved")

if __name__ == "__main__":
    test_max_area_of_island()
    demonstrate_area_calculation()
    visualize_island_areas()
    compare_algorithms()

"""
Graph Theory Concepts:
1. Connected Component Size Calculation
2. DFS with Return Values
3. Area/Weight Aggregation during Traversal
4. Maximum Finding in Connected Components

Key DFS Enhancements:
- Returning values from DFS (area calculation)
- Tracking maximum across multiple components
- Accumulating metrics during traversal
- Component size tracking

Algorithm Patterns:
- DFS with aggregation (sum, max, count)
- Component analysis and comparison
- Metric calculation during graph traversal
- State accumulation in recursive calls

Real-world Applications:
- Land area calculation in GIS systems
- Connected region analysis in image processing
- Network cluster size analysis
- Resource allocation problems
- Territory calculation in games

This extends the basic island counting to area calculation - a common pattern
in graph problems where we need to measure component properties.
"""
