"""
827. Making A Large Island
Difficulty: Hard

Problem:
You are given an n x n binary matrix grid. You are allowed to change at most one 0 to a 1.

Return the size of the largest island after applying this operation. An island is a 
4-directionally connected group of 1s.

Examples:
Input: grid = [[1,0],[0,1]]
Output: 3

Input: grid = [[1,1],[1,0]]
Output: 4

Input: grid = [[1,1],[1,1]]
Output: 4

Constraints:
- n == grid.length
- n == grid[i].length
- 1 <= n <= 500
- grid[i][j] is either 0 or 1
"""

from typing import List, Dict, Set
from collections import defaultdict

class Solution:
    def largestIsland_approach1_brute_force(self, grid: List[List[int]]) -> int:
        """
        Approach 1: Brute Force - Try Every 0
        
        For each 0, temporarily change it to 1 and calculate resulting island size.
        
        Time: O(N^4) - for each of N^2 zeros, do O(N^2) DFS
        Space: O(N^2) - recursion stack
        """
        n = len(grid)
        
        def dfs_island_size(i, j, visited):
            """Calculate island size starting from (i,j)"""
            if (i < 0 or i >= n or j < 0 or j >= n or 
                grid[i][j] == 0 or (i, j) in visited):
                return 0
            
            visited.add((i, j))
            size = 1
            
            # Explore 4 directions
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                size += dfs_island_size(i + di, j + dj, visited)
            
            return size
        
        def get_largest_island():
            """Get largest island size in current grid"""
            visited = set()
            max_size = 0
            
            for i in range(n):
                for j in range(n):
                    if grid[i][j] == 1 and (i, j) not in visited:
                        size = dfs_island_size(i, j, visited)
                        max_size = max(max_size, size)
            
            return max_size
        
        max_island = get_largest_island()  # Current largest island
        
        # Try changing each 0 to 1
        for i in range(n):
            for j in range(n):
                if grid[i][j] == 0:
                    grid[i][j] = 1  # Change 0 to 1
                    max_island = max(max_island, get_largest_island())
                    grid[i][j] = 0  # Restore
        
        return max_island
    
    def largestIsland_approach2_island_id_mapping(self, grid: List[List[int]]) -> int:
        """
        Approach 2: Island ID Mapping (Optimal)
        
        1. Assign unique ID to each island and record their sizes
        2. For each 0, check adjacent islands and sum their sizes
        
        Time: O(N^2) - two passes through grid
        Space: O(N^2) - island mapping
        """
        n = len(grid)
        
        def dfs_mark_island(i, j, island_id):
            """Mark all cells of an island with unique ID and return size"""
            if (i < 0 or i >= n or j < 0 or j >= n or grid[i][j] != 1):
                return 0
            
            grid[i][j] = island_id
            size = 1
            
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                size += dfs_mark_island(i + di, j + dj, island_id)
            
            return size
        
        # Phase 1: Identify and size all existing islands
        island_sizes = {}
        island_id = 2  # Start from 2 (since 0 and 1 are used)
        
        for i in range(n):
            for j in range(n):
                if grid[i][j] == 1:
                    size = dfs_mark_island(i, j, island_id)
                    island_sizes[island_id] = size
                    island_id += 1
        
        # Phase 2: For each 0, calculate potential island size
        max_island = max(island_sizes.values()) if island_sizes else 0
        
        for i in range(n):
            for j in range(n):
                if grid[i][j] == 0:
                    # Find adjacent unique islands
                    adjacent_islands = set()
                    
                    for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        ni, nj = i + di, j + dj
                        if (0 <= ni < n and 0 <= nj < n and grid[ni][nj] > 1):
                            adjacent_islands.add(grid[ni][nj])
                    
                    # Calculate total size if we change this 0 to 1
                    total_size = 1  # The cell itself
                    for island_id in adjacent_islands:
                        total_size += island_sizes[island_id]
                    
                    max_island = max(max_island, total_size)
        
        return max_island
    
    def largestIsland_approach3_union_find(self, grid: List[List[int]]) -> int:
        """
        Approach 3: Union-Find with Virtual Connections
        
        Use Union-Find to track connected components and sizes.
        
        Time: O(N^2 * α(N^2))
        Space: O(N^2)
        """
        n = len(grid)
        
        class UnionFind:
            def __init__(self, size):
                self.parent = list(range(size))
                self.component_size = [1] * size
                self.max_size = 1
            
            def find(self, x):
                if self.parent[x] != x:
                    self.parent[x] = self.find(self.parent[x])
                return self.parent[x]
            
            def union(self, x, y):
                px, py = self.find(x), self.find(y)
                if px == py:
                    return
                
                # Union by size
                if self.component_size[px] < self.component_size[py]:
                    px, py = py, px
                
                self.parent[py] = px
                self.component_size[px] += self.component_size[py]
                self.max_size = max(self.max_size, self.component_size[px])
            
            def get_size(self, x):
                return self.component_size[self.find(x)]
        
        def get_id(i, j):
            return i * n + j
        
        uf = UnionFind(n * n)
        
        # Connect all existing islands
        for i in range(n):
            for j in range(n):
                if grid[i][j] == 1:
                    current_id = get_id(i, j)
                    
                    # Connect to right and down neighbors (avoid duplicates)
                    for di, dj in [(0, 1), (1, 0)]:
                        ni, nj = i + di, j + dj
                        if (ni < n and nj < n and grid[ni][nj] == 1):
                            uf.union(current_id, get_id(ni, nj))
        
        max_island = uf.max_size if any(grid[i][j] == 1 for i in range(n) for j in range(n)) else 0
        
        # Try changing each 0 to 1
        for i in range(n):
            for j in range(n):
                if grid[i][j] == 0:
                    # Find adjacent islands
                    adjacent_components = set()
                    
                    for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        ni, nj = i + di, j + dj
                        if (0 <= ni < n and 0 <= nj < n and grid[ni][nj] == 1):
                            adjacent_components.add(uf.find(get_id(ni, nj)))
                    
                    # Calculate total size
                    total_size = 1  # The new cell
                    for component_root in adjacent_components:
                        total_size += uf.get_size(component_root)
                    
                    max_island = max(max_island, total_size)
        
        return max_island
    
    def largestIsland_approach4_optimized_neighbor_checking(self, grid: List[List[int]]) -> int:
        """
        Approach 4: Optimized Neighbor Checking
        
        More efficient implementation of island ID approach with optimizations.
        
        Time: O(N^2)
        Space: O(N^2)
        """
        n = len(grid)
        
        def dfs_size_and_mark(i, j, island_id):
            """DFS to calculate size and mark island with ID"""
            if (i < 0 or i >= n or j < 0 or j >= n or grid[i][j] != 1):
                return 0
            
            grid[i][j] = island_id
            size = 1
            
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                size += dfs_size_and_mark(i + di, j + dj, island_id)
            
            return size
        
        # Map each island to its size
        island_sizes = {0: 0}  # For water cells
        island_id = 2
        
        # Phase 1: Mark all islands and record sizes
        for i in range(n):
            for j in range(n):
                if grid[i][j] == 1:
                    size = dfs_size_and_mark(i, j, island_id)
                    island_sizes[island_id] = size
                    island_id += 1
        
        # Handle edge case: all cells are already 1
        if not any(grid[i][j] == 0 for i in range(n) for j in range(n)):
            return n * n
        
        max_island = max(island_sizes.values()) if len(island_sizes) > 1 else 0
        
        # Phase 2: Try each 0 position
        for i in range(n):
            for j in range(n):
                if grid[i][j] == 0:
                    # Collect unique adjacent island IDs
                    neighbors = set()
                    
                    for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < n and 0 <= nj < n:
                            neighbors.add(grid[ni][nj])
                    
                    # Calculate total size
                    total_size = 1  # The converted cell
                    for neighbor_id in neighbors:
                        if neighbor_id in island_sizes:
                            total_size += island_sizes[neighbor_id]
                    
                    max_island = max(max_island, total_size)
        
        return max_island

def test_largest_island():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (grid, expected)
        ([[1,0],[0,1]], 3),
        ([[1,1],[1,0]], 4),
        ([[1,1],[1,1]], 4),
        ([[0,0],[0,0]], 1),
        ([[1]], 1),
        ([[0]], 1),
        ([[1,0,1],[0,0,0],[1,0,1]], 5),
    ]
    
    approaches = [
        ("Brute Force", solution.largestIsland_approach1_brute_force),
        ("Island ID Mapping", solution.largestIsland_approach2_island_id_mapping),
        ("Union-Find", solution.largestIsland_approach3_union_find),
        ("Optimized Neighbor", solution.largestIsland_approach4_optimized_neighbor_checking),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (grid, expected) in enumerate(test_cases):
            # Create deep copy for testing
            test_grid = [row[:] for row in grid]
            result = func(test_grid)
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} Grid: {grid}, Expected: {expected}, Got: {result}")

def demonstrate_island_optimization():
    """Demonstrate the island optimization strategy"""
    print("\n=== Island Optimization Strategy Demo ===")
    
    grid = [[1,0,1],
            [0,0,0],
            [1,0,1]]
    
    print("Original grid:")
    for i, row in enumerate(grid):
        print(f"  Row {i}: {row}")
    
    # Phase 1: Identify and size existing islands
    n = len(grid)
    work_grid = [row[:] for row in grid]
    
    def dfs_mark(i, j, island_id):
        if (i < 0 or i >= n or j < 0 or j >= n or work_grid[i][j] != 1):
            return 0
        
        work_grid[i][j] = island_id
        size = 1
        
        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            size += dfs_mark(i + di, j + dj, island_id)
        
        return size
    
    island_sizes = {}
    island_id = 2
    
    print(f"\nPhase 1: Identifying existing islands")
    for i in range(n):
        for j in range(n):
            if work_grid[i][j] == 1:
                size = dfs_mark(i, j, island_id)
                island_sizes[island_id] = size
                print(f"  Island {island_id} starting at ({i},{j}): size {size}")
                island_id += 1
    
    print(f"\nGrid after island marking:")
    for i, row in enumerate(work_grid):
        print(f"  Row {i}: {row}")
    
    print(f"\nIsland sizes: {island_sizes}")
    
    # Phase 2: Try each 0 position
    print(f"\nPhase 2: Testing each 0 position")
    max_size = max(island_sizes.values()) if island_sizes else 0
    
    for i in range(n):
        for j in range(n):
            if work_grid[i][j] == 0:
                adjacent_islands = set()
                
                for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    ni, nj = i + di, j + dj
                    if (0 <= ni < n and 0 <= nj < n and work_grid[ni][nj] > 1):
                        adjacent_islands.add(work_grid[ni][nj])
                
                total_size = 1  # The cell itself
                for adj_id in adjacent_islands:
                    total_size += island_sizes[adj_id]
                
                print(f"  Position ({i},{j}): adjacent islands {adjacent_islands}, total size {total_size}")
                max_size = max(max_size, total_size)
    
    print(f"\nMaximum possible island size: {max_size}")

if __name__ == "__main__":
    test_largest_island()
    demonstrate_island_optimization()

"""
Graph Theory Concepts:
1. Connected Component Optimization
2. Component Merging Analysis
3. Strategic Single Point Modification
4. Efficient Neighbor Identification

Key Optimization Insights:
- Pre-compute all island sizes to avoid repeated DFS
- Use unique island IDs to efficiently identify neighbors
- Single modification can merge multiple existing islands
- O(N^2) solution vs naive O(N^4) approach

Algorithm Strategy:
1. Phase 1: Identify and size all existing islands
2. Phase 2: For each 0, calculate merged island size
3. Track maximum possible size across all positions

Advanced Techniques:
- Island ID mapping for O(1) neighbor identification
- Union-Find for dynamic connectivity tracking
- Efficient DFS with component marking
- Optimal neighbor checking without redundant computation

Real-world Applications:
- Land development planning (optimal land reclamation)
- Network design (strategic connection placement)
- Image processing (optimal pixel modification)
- Game AI (territorial expansion optimization)
- Circuit design (optimal connection placement)

This problem demonstrates strategic optimization in graph modification,
showing how preprocessing can dramatically improve algorithmic efficiency.
"""

