"""
1254. Number of Closed Islands
Difficulty: Easy

Problem:
Given a 2D grid consists of 0s (land) and 1s (water). An island is a maximal 
4-directionally connected group of 0s and a closed island is an island totally 
surrounded by 1s (i.e., not on the boundary of the grid).

Return the number of closed islands.

Examples:
Input: grid = [[1,1,1,1,1,1,1,0],
               [1,0,0,0,0,1,1,0],
               [1,0,1,0,1,1,1,0],
               [1,0,0,0,0,1,0,1],
               [1,1,1,1,1,1,1,0]]
Output: 2

Input: grid = [[0,0,1,0,0],
               [0,1,0,1,0],
               [0,1,1,1,0],
               [0,0,0,0,0]]
Output: 1

Input: grid = [[1,1,1,1,1,1,1],
               [1,0,0,0,0,0,1],
               [1,0,1,1,1,0,1],
               [1,0,1,0,1,0,1],
               [1,0,1,1,1,0,1],
               [1,0,0,0,0,0,1],
               [1,1,1,1,1,1,1]]
Output: 2

Constraints:
- 1 <= grid.length, grid[i].length <= 100
- grid[i][j] ∈ {0, 1}
"""

from typing import List
from collections import deque

class Solution:
    def closedIsland_approach1_eliminate_boundary(self, grid: List[List[int]]) -> int:
        """
        Approach 1: Eliminate Boundary Islands First
        
        First, eliminate (mark as water) all islands connected to boundary.
        Then count remaining islands - these are guaranteed to be closed.
        
        Time: O(M*N) - visit each cell at most twice
        Space: O(M*N) - recursion stack depth
        """
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        
        def dfs_eliminate(i, j):
            """Mark all connected land (0s) as water (1s)"""
            if (i < 0 or i >= m or j < 0 or j >= n or grid[i][j] == 1):
                return
            
            grid[i][j] = 1  # Mark as water
            
            # Explore 4 directions
            dfs_eliminate(i + 1, j)
            dfs_eliminate(i - 1, j)
            dfs_eliminate(i, j + 1)
            dfs_eliminate(i, j - 1)
        
        # Eliminate all boundary-connected islands
        # Top and bottom rows
        for j in range(n):
            if grid[0][j] == 0:
                dfs_eliminate(0, j)
            if grid[m-1][j] == 0:
                dfs_eliminate(m-1, j)
        
        # Left and right columns
        for i in range(m):
            if grid[i][0] == 0:
                dfs_eliminate(i, 0)
            if grid[i][n-1] == 0:
                dfs_eliminate(i, n-1)
        
        # Count remaining islands (these are closed)
        def dfs_count(i, j):
            """Mark visited island cells"""
            if (i < 0 or i >= m or j < 0 or j >= n or grid[i][j] == 1):
                return
            
            grid[i][j] = 1  # Mark as visited
            
            dfs_count(i + 1, j)
            dfs_count(i - 1, j)
            dfs_count(i, j + 1)
            dfs_count(i, j - 1)
        
        closed_islands = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 0:  # Found unvisited land
                    closed_islands += 1
                    dfs_count(i, j)
        
        return closed_islands
    
    def closedIsland_approach2_boundary_check_during_dfs(self, grid: List[List[int]]) -> int:
        """
        Approach 2: Check boundary connection during DFS
        
        For each island, use DFS to check if it touches boundary.
        Only count islands that don't touch boundary.
        
        Time: O(M*N)
        Space: O(M*N) - recursion stack + visited set
        """
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        visited = set()
        
        def dfs_check_closed(i, j):
            """Return True if island is closed, False if touches boundary"""
            if (i < 0 or i >= m or j < 0 or j >= n):
                return False  # Reached boundary - not closed
            
            if grid[i][j] == 1 or (i, j) in visited:
                return True  # Water or already visited
            
            visited.add((i, j))
            
            # Check all 4 directions - all must be closed
            up = dfs_check_closed(i - 1, j)
            down = dfs_check_closed(i + 1, j)
            left = dfs_check_closed(i, j - 1)
            right = dfs_check_closed(i, j + 1)
            
            return up and down and left and right
        
        closed_islands = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 0 and (i, j) not in visited:
                    if dfs_check_closed(i, j):
                        closed_islands += 1
        
        return closed_islands
    
    def closedIsland_approach3_bfs_boundary_check(self, grid: List[List[int]]) -> int:
        """
        Approach 3: BFS with boundary checking
        
        Use BFS to explore each island and check if it touches boundary.
        
        Time: O(M*N)
        Space: O(M*N) - queue + visited set
        """
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        visited = set()
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        def bfs_is_closed(start_i, start_j):
            """Check if island starting at (start_i, start_j) is closed"""
            if (start_i, start_j) in visited:
                return False
            
            queue = deque([(start_i, start_j)])
            island_cells = []
            touches_boundary = False
            
            while queue:
                i, j = queue.popleft()
                
                if (i, j) in visited:
                    continue
                
                visited.add((i, j))
                island_cells.append((i, j))
                
                # Check if touches boundary
                if i == 0 or i == m-1 or j == 0 or j == n-1:
                    touches_boundary = True
                
                # Explore neighbors
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    
                    if (0 <= ni < m and 0 <= nj < n and 
                        grid[ni][nj] == 0 and (ni, nj) not in visited):
                        queue.append((ni, nj))
            
            return not touches_boundary
        
        closed_islands = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 0 and (i, j) not in visited:
                    if bfs_is_closed(i, j):
                        closed_islands += 1
        
        return closed_islands
    
    def closedIsland_approach4_non_destructive(self, grid: List[List[int]]) -> int:
        """
        Approach 4: Non-destructive with explicit visited tracking
        
        Preserve original grid by using separate visited tracking.
        
        Time: O(M*N)
        Space: O(M*N) - visited set
        """
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        visited = set()
        
        def dfs_explore_island(i, j, island_cells):
            """Explore island and collect all its cells"""
            if ((i, j) in visited or i < 0 or i >= m or 
                j < 0 or j >= n or grid[i][j] == 1):
                return
            
            visited.add((i, j))
            island_cells.append((i, j))
            
            # Explore 4 directions
            dfs_explore_island(i + 1, j, island_cells)
            dfs_explore_island(i - 1, j, island_cells)
            dfs_explore_island(i, j + 1, island_cells)
            dfs_explore_island(i, j - 1, island_cells)
        
        def is_closed_island(island_cells):
            """Check if island is closed (no cell touches boundary)"""
            for i, j in island_cells:
                if i == 0 or i == m-1 or j == 0 or j == n-1:
                    return False
            return True
        
        closed_islands = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 0 and (i, j) not in visited:
                    island_cells = []
                    dfs_explore_island(i, j, island_cells)
                    
                    if island_cells and is_closed_island(island_cells):
                        closed_islands += 1
        
        return closed_islands
    
    def closedIsland_approach5_iterative_dfs(self, grid: List[List[int]]) -> int:
        """
        Approach 5: Iterative DFS to avoid recursion limits
        
        Use explicit stack for DFS to handle large grids.
        
        Time: O(M*N)
        Space: O(M*N) - stack size
        """
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        visited = set()
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        def iterative_dfs_is_closed(start_i, start_j):
            """Check if island is closed using iterative DFS"""
            if (start_i, start_j) in visited:
                return False
            
            stack = [(start_i, start_j)]
            island_cells = []
            touches_boundary = False
            
            while stack:
                i, j = stack.pop()
                
                if (i, j) in visited:
                    continue
                
                visited.add((i, j))
                island_cells.append((i, j))
                
                # Check boundary
                if i == 0 or i == m-1 or j == 0 or j == n-1:
                    touches_boundary = True
                
                # Add unvisited land neighbors
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    
                    if (0 <= ni < m and 0 <= nj < n and 
                        grid[ni][nj] == 0 and (ni, nj) not in visited):
                        stack.append((ni, nj))
            
            return not touches_boundary
        
        closed_islands = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 0 and (i, j) not in visited:
                    if iterative_dfs_is_closed(i, j):
                        closed_islands += 1
        
        return closed_islands

def test_closed_islands():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (grid, expected)
        ([[1,1,1,1,1,1,1,0],
          [1,0,0,0,0,1,1,0],
          [1,0,1,0,1,1,1,0],
          [1,0,0,0,0,1,0,1],
          [1,1,1,1,1,1,1,0]], 2),
        ([[0,0,1,0,0],
          [0,1,0,1,0],
          [0,1,1,1,0],
          [0,0,0,0,0]], 1),
        ([[1,1,1,1,1,1,1],
          [1,0,0,0,0,0,1],
          [1,0,1,1,1,0,1],
          [1,0,1,0,1,0,1],
          [1,0,1,1,1,0,1],
          [1,0,0,0,0,0,1],
          [1,1,1,1,1,1,1]], 2),
        ([[0]], 0),  # Single boundary cell
        ([[1]], 0),  # Single water cell
        ([[1,0,1],
          [0,0,0],
          [1,0,1]], 0),  # Island touches boundary
        ([[1,1,1],
          [1,0,1],
          [1,1,1]], 1),  # Single closed island
    ]
    
    approaches = [
        ("Eliminate Boundary", solution.closedIsland_approach1_eliminate_boundary),
        ("Boundary Check DFS", solution.closedIsland_approach2_boundary_check_during_dfs),
        ("BFS Boundary Check", solution.closedIsland_approach3_bfs_boundary_check),
        ("Non-destructive", solution.closedIsland_approach4_non_destructive),
        ("Iterative DFS", solution.closedIsland_approach5_iterative_dfs),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (grid, expected) in enumerate(test_cases):
            # Create deep copy for testing
            test_grid = [row[:] for row in grid]
            result = func(test_grid)
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} Expected: {expected}, Got: {result}")

def demonstrate_closed_island_detection():
    """Demonstrate the closed island detection process"""
    print("\n=== Closed Island Detection Demo ===")
    
    grid = [
        [1,1,1,1,1,1,1,0],
        [1,0,0,0,0,1,1,0],
        [1,0,1,0,1,1,1,0],
        [1,0,0,0,0,1,0,1],
        [1,1,1,1,1,1,1,0]
    ]
    
    print("Original Grid (0=land, 1=water):")
    for i, row in enumerate(grid):
        print(f"  Row {i}: {row}")
    
    # Analyze each island
    m, n = len(grid), len(grid[0])
    visited = set()
    islands = []
    
    def dfs_collect(i, j, island_cells):
        if ((i, j) in visited or i < 0 or i >= m or 
            j < 0 or j >= n or grid[i][j] == 1):
            return
        
        visited.add((i, j))
        island_cells.append((i, j))
        
        dfs_collect(i + 1, j, island_cells)
        dfs_collect(i - 1, j, island_cells)
        dfs_collect(i, j + 1, island_cells)
        dfs_collect(i, j - 1, island_cells)
    
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 0 and (i, j) not in visited:
                island_cells = []
                dfs_collect(i, j, island_cells)
                if island_cells:
                    islands.append(island_cells)
    
    print(f"\nFound {len(islands)} islands:")
    closed_count = 0
    
    for idx, island in enumerate(islands):
        # Check if touches boundary
        touches_boundary = any(i == 0 or i == m-1 or j == 0 or j == n-1 
                             for i, j in island)
        
        status = "Open (touches boundary)" if touches_boundary else "Closed"
        if not touches_boundary:
            closed_count += 1
        
        print(f"  Island {idx + 1}: {len(island)} cells, {status}")
        print(f"    Cells: {island}")
    
    print(f"\nClosed islands: {closed_count}")

def visualize_closed_islands():
    """Create visual representation of closed vs open islands"""
    print("\n=== Closed Island Visualization ===")
    
    grid = [
        [1,1,1,1,1],
        [1,0,1,0,1],
        [1,1,1,1,1],
        [0,0,1,0,0],
        [0,1,1,1,0]
    ]
    
    print("Grid visualization:")
    emoji_map = {0: "🏝️", 1: "🌊"}
    
    for i, row in enumerate(grid):
        display_row = [emoji_map[cell] for cell in row]
        print(f"  Row {i}: {''.join(display_row)} {row}")
    
    # Analyze islands
    m, n = len(grid), len(grid[0])
    visited = set()
    
    def analyze_island(start_i, start_j):
        if (start_i, start_j) in visited:
            return [], False
        
        stack = [(start_i, start_j)]
        island_cells = []
        touches_boundary = False
        
        while stack:
            i, j = stack.pop()
            
            if (i, j) in visited:
                continue
            
            visited.add((i, j))
            island_cells.append((i, j))
            
            if i == 0 or i == m-1 or j == 0 or j == n-1:
                touches_boundary = True
            
            for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                ni, nj = i + di, j + dj
                if (0 <= ni < m and 0 <= nj < n and 
                    grid[ni][nj] == 0 and (ni, nj) not in visited):
                    stack.append((ni, nj))
        
        return island_cells, touches_boundary
    
    print(f"\nIsland Analysis:")
    island_num = 1
    closed_count = 0
    
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 0 and (i, j) not in visited:
                cells, touches_boundary = analyze_island(i, j)
                if cells:
                    status = "🔒 CLOSED" if not touches_boundary else "🔓 OPEN"
                    if not touches_boundary:
                        closed_count += 1
                    
                    print(f"  Island {island_num}: {status}")
                    print(f"    Cells: {cells}")
                    print(f"    Touches boundary: {touches_boundary}")
                    island_num += 1
    
    print(f"\nTotal closed islands: {closed_count}")

def compare_approaches():
    """Compare different approaches for closed island detection"""
    print("\n=== Approach Comparison ===")
    
    approaches = [
        ("Eliminate Boundary", "Modify grid, clear strategy", "Destructive to input"),
        ("Boundary Check DFS", "Clean recursive logic", "Complex boundary logic"),
        ("BFS Boundary Check", "Level-order exploration", "Queue management overhead"),
        ("Non-destructive", "Preserves input grid", "Extra space for visited"),
        ("Iterative DFS", "No recursion limits", "More complex implementation"),
    ]
    
    print(f"{'Approach':<20} {'Advantages':<25} {'Disadvantages'}")
    print("-" * 70)
    
    for approach, advantages, disadvantages in approaches:
        print(f"{approach:<20} {advantages:<25} {disadvantages}")
    
    print(f"\nKey Insights:")
    print(f"- Closed island = island that doesn't touch grid boundary")
    print(f"- Two main strategies: eliminate boundary islands first, or check during traversal")
    print(f"- All approaches have O(M*N) time complexity")
    print(f"- Choice depends on whether input modification is acceptable")

if __name__ == "__main__":
    test_closed_islands()
    demonstrate_closed_island_detection()
    visualize_closed_islands()
    compare_approaches()

"""
Graph Theory Concepts:
1. Boundary-Connected vs Isolated Components
2. Island Classification based on Position
3. Component Analysis with Constraints
4. Boundary Detection in Grid Graphs

Key Closed Island Concepts:
- Definition: Island completely surrounded by water (not touching boundary)
- Boundary detection: Check if any cell in island is on grid edge
- Two-phase approach: Eliminate boundary islands, then count remaining
- Single-phase approach: Check boundary connection during traversal

Algorithm Strategies:
1. Eliminate then count: Remove boundary-connected islands first
2. Check during DFS: Verify boundary connection while exploring
3. Collect then analyze: Gather island cells, then check boundary
4. BFS variant: Level-order with boundary checking

Optimization Considerations:
- Destructive vs non-destructive approaches
- Early termination when boundary is detected
- Memory usage (recursion vs iteration)
- Input preservation requirements

Real-world Applications:
- Geographic analysis (landlocked regions)
- Image processing (completely enclosed regions)
- Game development (safe zones, enclosed territories)
- Network analysis (isolated sub-networks)
- Urban planning (enclosed districts)

This problem extends basic island detection with additional constraints,
demonstrating how graph problems can have spatial/positional requirements.
"""
