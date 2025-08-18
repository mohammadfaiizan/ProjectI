"""
1020. Number of Enclaves
Difficulty: Medium

Problem:
You are given an m x n binary matrix grid, where 0 represents a sea cell and 1 represents 
a land cell.

A move consists of walking from one land cell to another adjacent (4-directionally) land cell 
or walking off the boundary of the grid.

Return the number of land cells in grid for which we cannot walk off the boundary of the grid 
in any number of moves.

Examples:
Input: grid = [[0,0,0,0],[1,0,1,0],[0,1,1,0],[0,0,0,0]]
Output: 3

Input: grid = [[0,1,1,0],[0,0,1,0],[0,0,1,0],[0,0,0,0]]
Output: 0

Constraints:
- m == grid.length
- n == grid[i].length
- 1 <= m, n <= 500
- grid[i][j] is either 0 or 1
"""

from typing import List

class Solution:
    def numEnclaves_approach1_boundary_elimination(self, grid: List[List[int]]) -> int:
        """
        Approach 1: Boundary Elimination Strategy
        
        Eliminate all land cells connected to boundary, then count remaining.
        Same strategy as "Surrounded Regions" problem.
        
        Time: O(M*N) - visit each cell at most once
        Space: O(M*N) - recursion stack depth
        """
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        
        def dfs_eliminate(i, j):
            """Remove all connected land cells (mark as water)"""
            if (i < 0 or i >= m or j < 0 or j >= n or grid[i][j] != 1):
                return
            
            grid[i][j] = 0  # Mark as water/visited
            
            # Explore 4 directions
            dfs_eliminate(i + 1, j)
            dfs_eliminate(i - 1, j)
            dfs_eliminate(i, j + 1)
            dfs_eliminate(i, j - 1)
        
        # Eliminate boundary-connected land cells
        # Top and bottom rows
        for j in range(n):
            if grid[0][j] == 1:
                dfs_eliminate(0, j)
            if grid[m-1][j] == 1:
                dfs_eliminate(m-1, j)
        
        # Left and right columns
        for i in range(m):
            if grid[i][0] == 1:
                dfs_eliminate(i, 0)
            if grid[i][n-1] == 1:
                dfs_eliminate(i, n-1)
        
        # Count remaining land cells (these are enclaves)
        return sum(grid[i][j] for i in range(m) for j in range(n))
    
    def numEnclaves_approach2_visited_tracking(self, grid: List[List[int]]) -> int:
        """
        Approach 2: Non-destructive with Visited Tracking
        
        Mark boundary-connected cells without modifying original grid.
        
        Time: O(M*N)
        Space: O(M*N) - visited set + recursion stack
        """
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        boundary_connected = set()
        
        def dfs_mark(i, j):
            """Mark all connected land cells as boundary-connected"""
            if ((i, j) in boundary_connected or i < 0 or i >= m or 
                j < 0 or j >= n or grid[i][j] != 1):
                return
            
            boundary_connected.add((i, j))
            
            # Explore 4 directions
            dfs_mark(i + 1, j)
            dfs_mark(i - 1, j)
            dfs_mark(i, j + 1)
            dfs_mark(i, j - 1)
        
        # Mark all boundary-connected land cells
        for j in range(n):
            dfs_mark(0, j)      # Top row
            dfs_mark(m-1, j)    # Bottom row
        
        for i in range(m):
            dfs_mark(i, 0)      # Left column
            dfs_mark(i, n-1)    # Right column
        
        # Count land cells not connected to boundary
        enclaves = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1 and (i, j) not in boundary_connected:
                    enclaves += 1
        
        return enclaves
    
    def numEnclaves_approach3_bfs_elimination(self, grid: List[List[int]]) -> int:
        """
        Approach 3: BFS Boundary Elimination
        
        Use BFS instead of DFS to eliminate boundary-connected cells.
        
        Time: O(M*N)
        Space: O(M*N) - queue size
        """
        if not grid or not grid[0]:
            return 0
        
        from collections import deque
        
        m, n = len(grid), len(grid[0])
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        # Collect all boundary land cells
        queue = deque()
        
        for j in range(n):
            if grid[0][j] == 1:
                queue.append((0, j))
                grid[0][j] = 0
            if grid[m-1][j] == 1:
                queue.append((m-1, j))
                grid[m-1][j] = 0
        
        for i in range(1, m-1):  # Avoid corners already processed
            if grid[i][0] == 1:
                queue.append((i, 0))
                grid[i][0] = 0
            if grid[i][n-1] == 1:
                queue.append((i, n-1))
                grid[i][n-1] = 0
        
        # BFS to eliminate all connected land cells
        while queue:
            i, j = queue.popleft()
            
            for di, dj in directions:
                ni, nj = i + di, j + dj
                
                if (0 <= ni < m and 0 <= nj < n and grid[ni][nj] == 1):
                    grid[ni][nj] = 0
                    queue.append((ni, nj))
        
        # Count remaining land cells
        return sum(grid[i][j] for i in range(m) for j in range(n))
    
    def numEnclaves_approach4_iterative_dfs(self, grid: List[List[int]]) -> int:
        """
        Approach 4: Iterative DFS to avoid recursion limits
        
        Use explicit stack for DFS implementation.
        
        Time: O(M*N)
        Space: O(M*N) - stack size
        """
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        def iterative_dfs_eliminate(start_i, start_j):
            """Eliminate connected land cells using iterative DFS"""
            if grid[start_i][start_j] != 1:
                return
            
            stack = [(start_i, start_j)]
            
            while stack:
                i, j = stack.pop()
                
                if (i < 0 or i >= m or j < 0 or j >= n or grid[i][j] != 1):
                    continue
                
                grid[i][j] = 0  # Mark as visited/eliminated
                
                # Add neighbors to stack
                for di, dj in directions:
                    stack.append((i + di, j + dj))
        
        # Eliminate boundary-connected land cells
        for j in range(n):
            iterative_dfs_eliminate(0, j)       # Top row
            iterative_dfs_eliminate(m-1, j)     # Bottom row
        
        for i in range(m):
            iterative_dfs_eliminate(i, 0)       # Left column
            iterative_dfs_eliminate(i, n-1)     # Right column
        
        # Count remaining land cells
        return sum(grid[i][j] for i in range(m) for j in range(n))

def test_number_of_enclaves():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (grid, expected)
        ([[0,0,0,0],[1,0,1,0],[0,1,1,0],[0,0,0,0]], 3),
        ([[0,1,1,0],[0,0,1,0],[0,0,1,0],[0,0,0,0]], 0),
        ([[0,0,0,0],[0,1,1,0],[0,1,1,0],[0,0,0,0]], 4),  # Completely enclosed
        ([[1,1,1],[1,1,1],[1,1,1]], 0),  # All boundary connected
        ([[0,0,0],[0,1,0],[0,0,0]], 1),  # Single enclosed cell
        ([[1]], 0),  # Single boundary cell
        ([[0]], 0),  # No land
    ]
    
    approaches = [
        ("Boundary Elimination", solution.numEnclaves_approach1_boundary_elimination),
        ("Visited Tracking", solution.numEnclaves_approach2_visited_tracking),
        ("BFS Elimination", solution.numEnclaves_approach3_bfs_elimination),
        ("Iterative DFS", solution.numEnclaves_approach4_iterative_dfs),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (grid, expected) in enumerate(test_cases):
            # Create deep copy for testing
            test_grid = [row[:] for row in grid]
            result = func(test_grid)
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} Expected: {expected}, Got: {result}")

def demonstrate_enclave_detection():
    """Demonstrate enclave detection process"""
    print("\n=== Enclave Detection Demo ===")
    
    grid = [
        [0,0,0,0],
        [1,0,1,0],
        [0,1,1,0],
        [0,0,0,0]
    ]
    
    print("Original grid (0=water, 1=land):")
    for i, row in enumerate(grid):
        print(f"  Row {i}: {row}")
    
    m, n = len(grid), len(grid[0])
    
    # Identify boundary cells
    boundary_land = []
    for j in range(n):
        if grid[0][j] == 1:
            boundary_land.append((0, j, "top"))
        if grid[m-1][j] == 1:
            boundary_land.append((m-1, j, "bottom"))
    
    for i in range(m):
        if grid[i][0] == 1:
            boundary_land.append((i, 0, "left"))
        if grid[i][n-1] == 1:
            boundary_land.append((i, n-1, "right"))
    
    print(f"\nBoundary land cells: {boundary_land}")
    
    # Find all land cells
    all_land = []
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1:
                all_land.append((i, j))
    
    print(f"All land cells: {all_land}")
    
    # Simulate boundary elimination
    work_grid = [row[:] for row in grid]
    boundary_connected = set()
    
    def dfs_trace(i, j, path):
        if ((i, j) in boundary_connected or i < 0 or i >= m or 
            j < 0 or j >= n or work_grid[i][j] != 1):
            return
        
        boundary_connected.add((i, j))
        path.append((i, j))
        print(f"  Marking ({i},{j}) as boundary-connected")
        
        for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            dfs_trace(i + di, j + dj, path[:])
    
    print(f"\nTracing boundary-connected regions:")
    for i, j, position in boundary_land:
        if (i, j) not in boundary_connected:
            print(f"Starting from boundary cell ({i},{j}) at {position}")
            dfs_trace(i, j, [])
    
    print(f"\nBoundary-connected cells: {sorted(boundary_connected)}")
    
    # Find enclaves
    enclaves = []
    for i, j in all_land:
        if (i, j) not in boundary_connected:
            enclaves.append((i, j))
    
    print(f"Enclave cells: {enclaves}")
    print(f"Number of enclaves: {len(enclaves)}")
    
    # Visual representation
    print(f"\nVisual representation:")
    for i in range(m):
        row_display = []
        for j in range(n):
            if grid[i][j] == 0:
                row_display.append("🌊")  # Water
            elif (i, j) in enclaves:
                row_display.append("🏝️")  # Enclave
            else:
                row_display.append("🔗")  # Boundary-connected land
        print(f"  Row {i}: {''.join(row_display)}")

if __name__ == "__main__":
    test_number_of_enclaves()
    demonstrate_enclave_detection()

"""
Graph Theory Concepts:
1. Enclosed Connected Components
2. Boundary-based Component Classification
3. Complement Set Analysis
4. Reachability from Boundary

Key Enclave Concepts:
- Enclave: Land region completely surrounded by water or grid boundary
- Cannot reach boundary: No path from enclave to any boundary cell
- Boundary elimination: Remove all boundary-reachable land first
- Remaining land cells are enclaves by definition

Algorithm Strategy:
- Identify boundary land cells as starting points
- Use DFS/BFS to find all land reachable from boundary
- Count remaining land cells (these cannot reach boundary)
- Equivalent to finding "closed islands" in complementary view

Real-world Applications:
- Geographic analysis (landlocked regions)
- Island territory classification
- Network isolation detection
- Game development (unreachable areas)
- Circuit design (isolated components)
"""

