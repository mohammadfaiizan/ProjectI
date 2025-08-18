"""
694. Number of Distinct Islands
Difficulty: Medium

Problem:
You are given an m x n binary matrix grid. An island is a group of 1's (representing land) 
connected 4-directionally (horizontal or vertical.) You may assume all four edges of the 
grid are surrounded by water.

An island is considered to be the same as another if and only if one island can be translated 
(and not rotated or reflected) to equal the other.

Return the number of distinct islands.

Examples:
Input: grid = [[1,1,0,0,0],
               [1,1,0,0,0],
               [0,0,0,1,1],
               [0,0,0,1,1]]
Output: 1

Input: grid = [[1,1,0,1,1],
               [1,0,0,0,0],
               [0,0,0,0,1],
               [1,1,0,1,1]]
Output: 3

Constraints:
- m == grid.length
- n == grid[i].length
- 1 <= m, n <= 50
- grid[i][j] is either 0 or 1
"""

from typing import List, Set, Tuple

class Solution:
    def numDistinctIslands_approach1_path_signature(self, grid: List[List[int]]) -> int:
        """
        Approach 1: Path Signature (DFS Direction Encoding)
        
        Record the sequence of moves during DFS to create a unique signature.
        Islands with same shape will have same signature.
        
        Time: O(M*N) - visit each cell once
        Space: O(M*N) - recursion stack + signature storage
        """
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        distinct_shapes = set()
        
        def dfs(i, j, direction):
            """DFS with direction tracking to build signature"""
            if (i < 0 or i >= m or j < 0 or j >= n or grid[i][j] != 1):
                return ""
            
            grid[i][j] = 0  # Mark as visited
            
            # Build signature with current direction
            signature = direction
            
            # Explore 4 directions with specific order for consistency
            signature += dfs(i + 1, j, "D")  # Down
            signature += dfs(i - 1, j, "U")  # Up
            signature += dfs(i, j + 1, "R")  # Right
            signature += dfs(i, j - 1, "L")  # Left
            
            # Add return marker to distinguish different branch endings
            signature += "B"  # Back
            
            return signature
        
        # Find all islands and their signatures
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    signature = dfs(i, j, "S")  # Start
                    distinct_shapes.add(signature)
        
        return len(distinct_shapes)
    
    def numDistinctIslands_approach2_coordinate_normalization(self, grid: List[List[int]]) -> int:
        """
        Approach 2: Coordinate Normalization
        
        Collect all coordinates of each island, then normalize by translating
        to origin (subtract min coordinates).
        
        Time: O(M*N)
        Space: O(M*N)
        """
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        distinct_shapes = set()
        
        def dfs(i, j, island_coords):
            """Collect all coordinates of the island"""
            if (i < 0 or i >= m or j < 0 or j >= n or grid[i][j] != 1):
                return
            
            grid[i][j] = 0  # Mark as visited
            island_coords.append((i, j))
            
            # Explore 4 directions
            dfs(i + 1, j, island_coords)
            dfs(i - 1, j, island_coords)
            dfs(i, j + 1, island_coords)
            dfs(i, j - 1, island_coords)
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    island_coords = []
                    dfs(i, j, island_coords)
                    
                    if island_coords:
                        # Normalize coordinates (translate to origin)
                        min_r = min(r for r, c in island_coords)
                        min_c = min(c for r, c in island_coords)
                        
                        normalized = tuple(sorted([(r - min_r, c - min_c) 
                                                 for r, c in island_coords]))
                        distinct_shapes.add(normalized)
        
        return len(distinct_shapes)
    
    def numDistinctIslands_approach3_relative_coordinates(self, grid: List[List[int]]) -> int:
        """
        Approach 3: Relative Coordinates from Start Point
        
        Record coordinates relative to the starting point of DFS.
        
        Time: O(M*N)
        Space: O(M*N)
        """
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        distinct_shapes = set()
        
        def dfs(i, j, start_i, start_j, shape):
            """Record coordinates relative to start point"""
            if (i < 0 or i >= m or j < 0 or j >= n or grid[i][j] != 1):
                return
            
            grid[i][j] = 0  # Mark as visited
            
            # Add relative coordinate
            shape.append((i - start_i, j - start_j))
            
            # Explore 4 directions
            dfs(i + 1, j, start_i, start_j, shape)
            dfs(i - 1, j, start_i, start_j, shape)
            dfs(i, j + 1, start_i, start_j, shape)
            dfs(i, j - 1, start_i, start_j, shape)
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    shape = []
                    dfs(i, j, i, j, shape)
                    
                    if shape:
                        # Sort for canonical representation
                        canonical_shape = tuple(sorted(shape))
                        distinct_shapes.add(canonical_shape)
        
        return len(distinct_shapes)
    
    def numDistinctIslands_approach4_bfs_signature(self, grid: List[List[int]]) -> int:
        """
        Approach 4: BFS with Coordinate Collection
        
        Use BFS to explore islands and collect normalized coordinates.
        
        Time: O(M*N)
        Space: O(M*N)
        """
        if not grid or not grid[0]:
            return 0
        
        from collections import deque
        
        m, n = len(grid), len(grid[0])
        distinct_shapes = set()
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    # BFS to find all cells of this island
                    island_coords = []
                    queue = deque([(i, j)])
                    grid[i][j] = 0
                    
                    while queue:
                        r, c = queue.popleft()
                        island_coords.append((r, c))
                        
                        for dr, dc in directions:
                            nr, nc = r + dr, c + dc
                            
                            if (0 <= nr < m and 0 <= nc < n and grid[nr][nc] == 1):
                                grid[nr][nc] = 0
                                queue.append((nr, nc))
                    
                    # Normalize coordinates
                    if island_coords:
                        min_r = min(r for r, c in island_coords)
                        min_c = min(c for r, c in island_coords)
                        
                        normalized = tuple(sorted([(r - min_r, c - min_c) 
                                                 for r, c in island_coords]))
                        distinct_shapes.add(normalized)
        
        return len(distinct_shapes)
    
    def numDistinctIslands_approach5_hash_based(self, grid: List[List[int]]) -> int:
        """
        Approach 5: Hash-based Shape Recognition
        
        Create hash from island structure for efficient comparison.
        
        Time: O(M*N)
        Space: O(M*N)
        """
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        distinct_shapes = set()
        
        def get_island_hash(start_i, start_j):
            """Get hash representation of island shape"""
            coords = []
            stack = [(start_i, start_j)]
            
            while stack:
                i, j = stack.pop()
                
                if (i < 0 or i >= m or j < 0 or j >= n or grid[i][j] != 1):
                    continue
                
                grid[i][j] = 0  # Mark as visited
                coords.append((i, j))
                
                # Add neighbors
                for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    stack.append((i + di, j + dj))
            
            if not coords:
                return None
            
            # Normalize coordinates
            min_r, min_c = min(coords)
            normalized = tuple(sorted([(r - min_r, c - min_c) for r, c in coords]))
            
            return hash(normalized)
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    island_hash = get_island_hash(i, j)
                    if island_hash is not None:
                        distinct_shapes.add(island_hash)
        
        return len(distinct_shapes)

def test_distinct_islands():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (grid, expected)
        ([[1,1,0,0,0],
          [1,1,0,0,0],
          [0,0,0,1,1],
          [0,0,0,1,1]], 1),
        ([[1,1,0,1,1],
          [1,0,0,0,0],
          [0,0,0,0,1],
          [1,1,0,1,1]], 3),
        ([[1,1,1],
          [0,1,0],
          [0,0,0]], 1),
        ([[1,0,1],
          [0,0,0],
          [1,0,1]], 2),  # Two single-cell islands
        ([[0]], 0),  # No islands
        ([[1]], 1),  # Single island
    ]
    
    approaches = [
        ("Path Signature", solution.numDistinctIslands_approach1_path_signature),
        ("Coordinate Normalization", solution.numDistinctIslands_approach2_coordinate_normalization),
        ("Relative Coordinates", solution.numDistinctIslands_approach3_relative_coordinates),
        ("BFS Signature", solution.numDistinctIslands_approach4_bfs_signature),
        ("Hash-based", solution.numDistinctIslands_approach5_hash_based),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (grid, expected) in enumerate(test_cases):
            # Create deep copy for testing
            test_grid = [row[:] for row in grid]
            result = func(test_grid)
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} Expected: {expected}, Got: {result}")

def demonstrate_island_shapes():
    """Demonstrate island shape analysis"""
    print("\n=== Island Shape Analysis Demo ===")
    
    grid = [
        [1,1,0,1,1],
        [1,0,0,0,0],
        [0,0,0,0,1],
        [1,1,0,1,1]
    ]
    
    print("Original grid:")
    for i, row in enumerate(grid):
        print(f"  Row {i}: {row}")
    
    # Find and analyze each island
    m, n = len(grid), len(grid[0])
    visited = [[False] * n for _ in range(m)]
    islands = []
    
    def dfs_collect(i, j, coords):
        if (i < 0 or i >= m or j < 0 or j >= n or 
            visited[i][j] or grid[i][j] != 1):
            return
        
        visited[i][j] = True
        coords.append((i, j))
        
        dfs_collect(i + 1, j, coords)
        dfs_collect(i - 1, j, coords)
        dfs_collect(i, j + 1, coords)
        dfs_collect(i, j - 1, coords)
    
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1 and not visited[i][j]:
                coords = []
                dfs_collect(i, j, coords)
                if coords:
                    islands.append(coords)
    
    print(f"\nFound {len(islands)} islands:")
    
    distinct_shapes = set()
    
    for idx, coords in enumerate(islands):
        print(f"\nIsland {idx + 1}:")
        print(f"  Coordinates: {coords}")
        
        # Normalize coordinates
        min_r = min(r for r, c in coords)
        min_c = min(c for r, c in coords)
        normalized = tuple(sorted([(r - min_r, c - min_c) for r, c in coords]))
        
        print(f"  Normalized: {normalized}")
        
        # Check if shape is unique
        if normalized in distinct_shapes:
            print(f"  Shape: DUPLICATE")
        else:
            print(f"  Shape: UNIQUE")
            distinct_shapes.add(normalized)
        
        # Visualize island shape
        if coords:
            max_r = max(r - min_r for r, c in coords)
            max_c = max(c - min_c for r, c in coords)
            
            shape_grid = [['.' for _ in range(max_c + 1)] for _ in range(max_r + 1)]
            for r, c in coords:
                shape_grid[r - min_r][c - min_c] = '█'
            
            print(f"  Visual:")
            for row in shape_grid:
                print(f"    {''.join(row)}")
    
    print(f"\nTotal distinct island shapes: {len(distinct_shapes)}")

if __name__ == "__main__":
    test_distinct_islands()
    demonstrate_island_shapes()

"""
Graph Theory Concepts:
1. Shape Recognition in Connected Components
2. Coordinate Normalization and Translation
3. Canonical Representation of Graphs
4. Path Signature Generation

Key Shape Recognition Techniques:
- Path signature: Record DFS traversal sequence
- Coordinate normalization: Translate to standard position
- Relative coordinates: Express relative to starting point
- Hash-based comparison: Efficient shape matching

Algorithm Insights:
- Translation invariance: Same shape at different positions
- Canonical form: Unique representation for each distinct shape
- DFS/BFS order consistency: Ensure reproducible signatures
- Normalization: Remove position dependency

Real-world Applications:
- Computer vision (shape recognition)
- Geographic analysis (landform classification)
- Pattern matching in images
- Game development (tile pattern recognition)
- Bioinformatics (protein structure comparison)
"""

