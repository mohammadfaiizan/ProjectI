"""
463. Island Perimeter
Difficulty: Easy

Problem:
You are given row x col grid representing a map where grid[i][j] = 1 represents 
land and grid[i][j] = 0 represents water.

Grid cells are connected horizontally/vertically (not diagonally). The grid is 
completely surrounded by water, and there is exactly one island (i.e., one or 
more connected land cells).

The island doesn't have "lakes", meaning the water inside isn't connected to the 
water around the island. One cell is a square with side length 1. The grid is 
rectangular, width and height don't exceed 100. Determine the perimeter of the island.

Examples:
Input: grid = [[0,1,0,0],
               [1,1,1,0],
               [0,1,0,0],
               [1,1,0,0]]
Output: 16

Input: grid = [[1]]
Output: 4

Input: grid = [[1,0]]
Output: 4

Constraints:
- row, col <= 100
- grid[i][j] is 0 or 1
- There is exactly one island in grid
"""

from typing import List

class Solution:
    def islandPerimeter_approach1_edge_counting(self, grid: List[List[int]]) -> int:
        """
        Approach 1: Direct Edge Counting
        
        For each land cell, count how many of its 4 sides are exposed
        (either at boundary or adjacent to water).
        
        Time: O(M*N) - visit each cell once
        Space: O(1) - constant extra space
        """
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        perimeter = 0
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:  # Land cell
                    # Check all 4 directions
                    # Each direction contributes 1 to perimeter if it's water or boundary
                    
                    # Up
                    if i == 0 or grid[i-1][j] == 0:
                        perimeter += 1
                    
                    # Down
                    if i == m-1 or grid[i+1][j] == 0:
                        perimeter += 1
                    
                    # Left
                    if j == 0 or grid[i][j-1] == 0:
                        perimeter += 1
                    
                    # Right
                    if j == n-1 or grid[i][j+1] == 0:
                        perimeter += 1
        
        return perimeter
    
    def islandPerimeter_approach2_neighbor_subtraction(self, grid: List[List[int]]) -> int:
        """
        Approach 2: Neighbor Subtraction Method
        
        Start with 4 edges per land cell, then subtract 2 for each
        adjacent land cell (shared edge).
        
        Time: O(M*N)
        Space: O(1)
        """
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        land_cells = 0
        adjacent_pairs = 0
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    land_cells += 1
                    
                    # Count adjacent land cells to the right and down
                    # (to avoid double counting)
                    if i + 1 < m and grid[i+1][j] == 1:
                        adjacent_pairs += 1
                    
                    if j + 1 < n and grid[i][j+1] == 1:
                        adjacent_pairs += 1
        
        # Each land cell has 4 edges, subtract 2 for each shared edge
        return land_cells * 4 - adjacent_pairs * 2
    
    def islandPerimeter_approach3_dfs_traversal(self, grid: List[List[int]]) -> int:
        """
        Approach 3: DFS with Perimeter Calculation
        
        Use DFS to traverse the island and calculate perimeter during traversal.
        
        Time: O(M*N)
        Space: O(M*N) - recursion stack + visited set
        """
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        visited = set()
        
        def dfs(i, j):
            if (i, j) in visited:
                return 0
            
            visited.add((i, j))
            perimeter = 0
            
            # Check all 4 directions
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            
            for di, dj in directions:
                ni, nj = i + di, j + dj
                
                # If neighbor is water or out of bounds, add to perimeter
                if (ni < 0 or ni >= m or nj < 0 or nj >= n or grid[ni][nj] == 0):
                    perimeter += 1
                # If neighbor is unvisited land, recurse
                elif grid[ni][nj] == 1 and (ni, nj) not in visited:
                    perimeter += dfs(ni, nj)
            
            return perimeter
        
        # Find first land cell and start DFS
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    return dfs(i, j)
        
        return 0
    
    def islandPerimeter_approach4_boundary_detection(self, grid: List[List[int]]) -> int:
        """
        Approach 4: Boundary Detection Method
        
        Detect boundaries by checking transitions between land and water.
        
        Time: O(M*N)
        Space: O(1)
        """
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        perimeter = 0
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    # Count boundaries for current land cell
                    boundaries = 0
                    
                    # Check each direction
                    for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        ni, nj = i + di, j + dj
                        
                        # Boundary condition: out of bounds or water
                        if (ni < 0 or ni >= m or nj < 0 or nj >= n or grid[ni][nj] == 0):
                            boundaries += 1
                    
                    perimeter += boundaries
        
        return perimeter
    
    def islandPerimeter_approach5_mathematical_formula(self, grid: List[List[int]]) -> int:
        """
        Approach 5: Mathematical Formula Approach
        
        Use mathematical relationship: 
        Perimeter = 4 * (land cells) - 2 * (internal connections)
        
        Time: O(M*N)
        Space: O(1)
        """
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        land_count = 0
        horizontal_connections = 0
        vertical_connections = 0
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    land_count += 1
                    
                    # Count horizontal connections (to the right)
                    if j + 1 < n and grid[i][j+1] == 1:
                        horizontal_connections += 1
                    
                    # Count vertical connections (downward)
                    if i + 1 < m and grid[i+1][j] == 1:
                        vertical_connections += 1
        
        total_connections = horizontal_connections + vertical_connections
        return 4 * land_count - 2 * total_connections

def test_island_perimeter():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (grid, expected)
        ([[0,1,0,0],
          [1,1,1,0],
          [0,1,0,0],
          [1,1,0,0]], 16),
        ([[1]], 4),
        ([[1,0]], 4),
        ([[1,1],
          [1,1]], 8),
        ([[1,1,1],
          [1,0,1],
          [1,1,1]], 12),  # Island with hole
        ([[0,0,0],
          [0,1,0],
          [0,0,0]], 4),  # Single isolated cell
    ]
    
    approaches = [
        ("Edge Counting", solution.islandPerimeter_approach1_edge_counting),
        ("Neighbor Subtraction", solution.islandPerimeter_approach2_neighbor_subtraction),
        ("DFS Traversal", solution.islandPerimeter_approach3_dfs_traversal),
        ("Boundary Detection", solution.islandPerimeter_approach4_boundary_detection),
        ("Mathematical Formula", solution.islandPerimeter_approach5_mathematical_formula),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (grid, expected) in enumerate(test_cases):
            result = func(grid)
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} Expected: {expected}, Got: {result}")

def demonstrate_perimeter_calculation():
    """Demonstrate step-by-step perimeter calculation"""
    print("\n=== Perimeter Calculation Demo ===")
    
    grid = [
        [0, 1, 0, 0],
        [1, 1, 1, 0],
        [0, 1, 0, 0],
        [1, 1, 0, 0]
    ]
    
    print("Grid:")
    for i, row in enumerate(grid):
        print(f"  Row {i}: {row}")
    
    m, n = len(grid), len(grid[0])
    print(f"\nPerimeter calculation for each land cell:")
    
    total_perimeter = 0
    
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1:
                # Count exposed edges
                exposed_edges = 0
                edges_detail = []
                
                # Check up
                if i == 0 or grid[i-1][j] == 0:
                    exposed_edges += 1
                    edges_detail.append("up")
                
                # Check down
                if i == m-1 or grid[i+1][j] == 0:
                    exposed_edges += 1
                    edges_detail.append("down")
                
                # Check left
                if j == 0 or grid[i][j-1] == 0:
                    exposed_edges += 1
                    edges_detail.append("left")
                
                # Check right
                if j == n-1 or grid[i][j+1] == 0:
                    exposed_edges += 1
                    edges_detail.append("right")
                
                print(f"  Cell ({i},{j}): {exposed_edges} exposed edges {edges_detail}")
                total_perimeter += exposed_edges
    
    print(f"\nTotal perimeter: {total_perimeter}")

def visualize_perimeter():
    """Create visual representation of island perimeter"""
    print("\n=== Island Perimeter Visualization ===")
    
    grid = [
        [0, 1, 1, 0],
        [1, 1, 1, 0],
        [0, 1, 0, 0]
    ]
    
    print("Island with perimeter markers:")
    
    m, n = len(grid), len(grid[0])
    
    # Create visualization grid
    vis_grid = []
    for i in range(m * 2 + 1):
        vis_grid.append([' '] * (n * 2 + 1))
    
    # Place land cells
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1:
                vis_grid[i * 2 + 1][j * 2 + 1] = '🏝️'
            else:
                vis_grid[i * 2 + 1][j * 2 + 1] = '🌊'
    
    # Add perimeter markers
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1:
                # Check each direction and add boundary markers
                
                # Up boundary
                if i == 0 or grid[i-1][j] == 0:
                    vis_grid[i * 2][j * 2 + 1] = '━'
                
                # Down boundary  
                if i == m-1 or grid[i+1][j] == 0:
                    vis_grid[i * 2 + 2][j * 2 + 1] = '━'
                
                # Left boundary
                if j == 0 or grid[i][j-1] == 0:
                    vis_grid[i * 2 + 1][j * 2] = '┃'
                
                # Right boundary
                if j == n-1 or grid[i][j+1] == 0:
                    vis_grid[i * 2 + 1][j * 2 + 2] = '┃'
    
    # Print visualization
    for row in vis_grid:
        print(''.join(row))

def analyze_perimeter_patterns():
    """Analyze different island patterns and their perimeters"""
    print("\n=== Perimeter Pattern Analysis ===")
    
    patterns = [
        ("Single Cell", [[1]], 4),
        ("Line (Horizontal)", [[1, 1, 1]], 8),
        ("Line (Vertical)", [[1], [1], [1]], 8),
        ("Square 2x2", [[1, 1], [1, 1]], 8),
        ("Square 3x3", [[1, 1, 1], [1, 1, 1], [1, 1, 1]], 12),
        ("L-Shape", [[1, 0], [1, 1]], 8),
        ("Plus Shape", [[0, 1, 0], [1, 1, 1], [0, 1, 0]], 12),
        ("Ring", [[1, 1, 1], [1, 0, 1], [1, 1, 1]], 16),
    ]
    
    solution = Solution()
    
    print(f"{'Pattern':<15} {'Cells':<6} {'Expected':<8} {'Calculated':<10} {'Efficiency'}")
    print("-" * 65)
    
    for name, grid, expected in patterns:
        calculated = solution.islandPerimeter_approach1_edge_counting(grid)
        cell_count = sum(sum(row) for row in grid)
        efficiency = calculated / (4 * cell_count) if cell_count > 0 else 0
        
        status = "✓" if calculated == expected else "✗"
        print(f"{name:<15} {cell_count:<6} {expected:<8} {calculated:<10} {efficiency:.2f}")
    
    print(f"\nEfficiency = Actual Perimeter / (4 × Cell Count)")
    print(f"- Efficiency = 1.0: Maximum perimeter (no internal connections)")
    print(f"- Efficiency < 1.0: More compact shape (more internal connections)")

def mathematical_analysis():
    """Mathematical analysis of perimeter calculation"""
    print("\n=== Mathematical Analysis ===")
    
    print("Perimeter Formula:")
    print("P = 4 × (number of land cells) - 2 × (number of adjacent pairs)")
    print()
    
    print("Where:")
    print("- Each land cell contributes 4 potential edges")
    print("- Each adjacent pair shares 1 edge (reduces perimeter by 2)")
    print("- Adjacent pairs are counted horizontally and vertically")
    print()
    
    # Example calculation
    grid = [[1, 1], [1, 0]]
    
    print("Example:")
    print(f"Grid: {grid}")
    
    land_cells = sum(sum(row) for row in grid)
    
    # Count adjacent pairs
    m, n = len(grid), len(grid[0])
    adjacent_pairs = 0
    
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1:
                # Count right neighbor
                if j + 1 < n and grid[i][j+1] == 1:
                    adjacent_pairs += 1
                # Count down neighbor
                if i + 1 < m and grid[i+1][j] == 1:
                    adjacent_pairs += 1
    
    perimeter = 4 * land_cells - 2 * adjacent_pairs
    
    print(f"Land cells: {land_cells}")
    print(f"Adjacent pairs: {adjacent_pairs}")
    print(f"Perimeter = 4 × {land_cells} - 2 × {adjacent_pairs} = {perimeter}")

if __name__ == "__main__":
    test_island_perimeter()
    demonstrate_perimeter_calculation()
    visualize_perimeter()
    analyze_perimeter_patterns()
    mathematical_analysis()

"""
Graph Theory Concepts:
1. Boundary Detection in Connected Components
2. Edge Counting in Grid Graphs
3. Perimeter Calculation for 2D Shapes
4. Mathematical Relationships in Graph Metrics

Key Perimeter Concepts:
- External vs Internal edges
- Boundary conditions and edge cases
- Mathematical formulation of perimeter
- Efficiency of different counting approaches

Algorithm Variants:
- Direct counting: Check each cell's exposed edges
- Subtraction method: Start with max, subtract internal connections
- DFS approach: Traverse and count during traversal
- Mathematical formula: Use relationship between cells and connections

Optimization Insights:
- All approaches are O(M*N) time
- Space complexity varies: O(1) for direct methods, O(M*N) for DFS
- Direct edge counting is most intuitive
- Mathematical formula provides insight into structure

Real-world Applications:
- Geographic perimeter calculation
- Image processing (shape analysis)
- Computer graphics (mesh boundary detection)
- Game development (territory boundaries)
- Manufacturing (material usage calculation)

Mathematical Insight:
Perimeter = 4 × (land cells) - 2 × (internal connections)
This formula reveals the trade-off between shape size and compactness.
"""
