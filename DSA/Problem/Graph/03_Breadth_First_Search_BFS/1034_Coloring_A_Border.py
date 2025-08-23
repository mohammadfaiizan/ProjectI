"""
1034. Coloring A Border - Multiple Approaches
Difficulty: Easy

You are given an m x n integer matrix grid, and three integers row, col, and color. Each value in the grid represents the color of the grid square at that location.

Two squares belong to the same connected component if they have the same color and are next to each other in any of the 4 directions.

The border of a connected component is all the squares in the connected component that are either 4-directionally adjacent to a square not in the connected component, or on the boundary of the grid.

You should color the border of the connected component that contains the square grid[row][col] with color.

Return the final grid.
"""

from typing import List, Set, Tuple
from collections import deque

class ColoringBorder:
    """Multiple approaches to color connected component borders"""
    
    def colorBorder_bfs_two_pass(self, grid: List[List[int]], row: int, col: int, color: int) -> List[List[int]]:
        """
        Approach 1: BFS Two-Pass Algorithm
        
        First pass finds connected component, second pass colors border.
        
        Time: O(mn), Space: O(mn)
        """
        m, n = len(grid), len(grid[0])
        original_color = grid[row][col]
        
        if original_color == color:
            return grid
        
        # First pass: Find all cells in connected component
        visited = set()
        component = set()
        queue = deque([(row, col)])
        visited.add((row, col))
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        while queue:
            r, c = queue.popleft()
            component.add((r, c))
            
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                
                if (0 <= nr < m and 0 <= nc < n and 
                    (nr, nc) not in visited and 
                    grid[nr][nc] == original_color):
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        
        # Second pass: Identify and color border cells
        border_cells = set()
        
        for r, c in component:
            is_border = False
            
            # Check if on grid boundary
            if r == 0 or r == m - 1 or c == 0 or c == n - 1:
                is_border = True
            else:
                # Check if adjacent to different color
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) not in component:
                        is_border = True
                        break
            
            if is_border:
                border_cells.add((r, c))
        
        # Color the border
        result = [row[:] for row in grid]
        for r, c in border_cells:
            result[r][c] = color
        
        return result
    
    def colorBorder_bfs_single_pass(self, grid: List[List[int]], row: int, col: int, color: int) -> List[List[int]]:
        """
        Approach 2: BFS Single-Pass with Border Detection
        
        Detect border cells during BFS traversal.
        
        Time: O(mn), Space: O(mn)
        """
        m, n = len(grid), len(grid[0])
        original_color = grid[row][col]
        
        if original_color == color:
            return grid
        
        visited = set()
        border_cells = set()
        queue = deque([(row, col)])
        visited.add((row, col))
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        while queue:
            r, c = queue.popleft()
            
            # Check if current cell is on border
            is_border = (r == 0 or r == m - 1 or c == 0 or c == n - 1)
            
            same_color_neighbors = 0
            
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                
                if 0 <= nr < m and 0 <= nc < n:
                    if grid[nr][nc] == original_color:
                        same_color_neighbors += 1
                        if (nr, nc) not in visited:
                            visited.add((nr, nc))
                            queue.append((nr, nc))
                else:
                    # Out of bounds counts as different color
                    pass
            
            # Cell is on border if it has fewer than 4 same-color neighbors
            if is_border or same_color_neighbors < 4:
                border_cells.add((r, c))
        
        # Color the border
        result = [row[:] for row in grid]
        for r, c in border_cells:
            result[r][c] = color
        
        return result
    
    def colorBorder_dfs_recursive(self, grid: List[List[int]], row: int, col: int, color: int) -> List[List[int]]:
        """
        Approach 3: Recursive DFS with Border Detection
        
        Use DFS to find component and detect border simultaneously.
        
        Time: O(mn), Space: O(mn)
        """
        m, n = len(grid), len(grid[0])
        original_color = grid[row][col]
        
        if original_color == color:
            return grid
        
        visited = set()
        border_cells = set()
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        def dfs(r: int, c: int) -> int:
            """DFS that returns count of same-color neighbors"""
            if (r, c) in visited:
                return 0
            
            visited.add((r, c))
            same_color_neighbors = 0
            
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                
                if 0 <= nr < m and 0 <= nc < n:
                    if grid[nr][nc] == original_color:
                        same_color_neighbors += 1
                        if (nr, nc) not in visited:
                            dfs(nr, nc)
            
            # Check if on border
            if (r == 0 or r == m - 1 or c == 0 or c == n - 1 or 
                same_color_neighbors < 4):
                border_cells.add((r, c))
            
            return same_color_neighbors
        
        dfs(row, col)
        
        # Color the border
        result = [row[:] for row in grid]
        for r, c in border_cells:
            result[r][c] = color
        
        return result
    
    def colorBorder_iterative_dfs(self, grid: List[List[int]], row: int, col: int, color: int) -> List[List[int]]:
        """
        Approach 4: Iterative DFS using Stack
        
        Use explicit stack for DFS to avoid recursion depth issues.
        
        Time: O(mn), Space: O(mn)
        """
        m, n = len(grid), len(grid[0])
        original_color = grid[row][col]
        
        if original_color == color:
            return grid
        
        visited = set()
        component = []
        stack = [(row, col)]
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        # Find all cells in connected component
        while stack:
            r, c = stack.pop()
            
            if (r, c) in visited:
                continue
            
            visited.add((r, c))
            component.append((r, c))
            
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                
                if (0 <= nr < m and 0 <= nc < n and 
                    (nr, nc) not in visited and 
                    grid[nr][nc] == original_color):
                    stack.append((nr, nc))
        
        # Find border cells
        border_cells = []
        
        for r, c in component:
            is_border = False
            
            # Check boundary
            if r == 0 or r == m - 1 or c == 0 or c == n - 1:
                is_border = True
            else:
                # Check neighbors
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if grid[nr][nc] != original_color:
                        is_border = True
                        break
            
            if is_border:
                border_cells.append((r, c))
        
        # Color the border
        result = [row[:] for row in grid]
        for r, c in border_cells:
            result[r][c] = color
        
        return result
    
    def colorBorder_optimized_marking(self, grid: List[List[int]], row: int, col: int, color: int) -> List[List[int]]:
        """
        Approach 5: Optimized with In-place Marking
        
        Use negative values to mark visited cells temporarily.
        
        Time: O(mn), Space: O(mn)
        """
        m, n = len(grid), len(grid[0])
        original_color = grid[row][col]
        
        if original_color == color:
            return grid
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        def dfs(r: int, c: int) -> bool:
            """DFS that returns True if cell is on border"""
            if (r < 0 or r >= m or c < 0 or c >= n or 
                grid[r][c] != original_color):
                return False
            
            # Mark as visited with negative value
            grid[r][c] = -original_color
            
            is_border = False
            
            # Check if on grid boundary
            if r == 0 or r == m - 1 or c == 0 or c == n - 1:
                is_border = True
            
            # Explore neighbors
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                
                if 0 <= nr < m and 0 <= nc < n:
                    if grid[nr][nc] == original_color:
                        if dfs(nr, nc):
                            is_border = True
                    elif grid[nr][nc] != -original_color:
                        # Different color neighbor
                        is_border = True
                else:
                    # Out of bounds
                    is_border = True
            
            return is_border
        
        # Find component and detect borders
        border_cells = []
        queue = deque([(row, col)])
        
        while queue:
            r, c = queue.popleft()
            
            if grid[r][c] == original_color:
                if dfs(r, c):
                    border_cells.append((r, c))
        
        # Restore original colors and color borders
        for i in range(m):
            for j in range(n):
                if grid[i][j] == -original_color:
                    grid[i][j] = original_color
        
        for r, c in border_cells:
            grid[r][c] = color
        
        return grid

def test_coloring_border():
    """Test coloring border algorithms"""
    solver = ColoringBorder()
    
    test_cases = [
        ([[1,1],[1,2]], 0, 0, 3, "Simple 2x2 grid"),
        ([[1,2,2],[2,3,2]], 0, 1, 3, "Mixed colors"),
        ([[1,1,1],[1,1,1],[1,1,1]], 1, 1, 2, "All same color"),
        ([[1,2,1,2,1,2],[2,2,2,2,1,2],[1,2,2,2,1,2]], 1, 3, 1, "Complex pattern"),
    ]
    
    algorithms = [
        ("BFS Two-Pass", solver.colorBorder_bfs_two_pass),
        ("BFS Single-Pass", solver.colorBorder_bfs_single_pass),
        ("DFS Recursive", solver.colorBorder_dfs_recursive),
        ("DFS Iterative", solver.colorBorder_iterative_dfs),
    ]
    
    print("=== Testing Coloring Border ===")
    
    for grid, row, col, color, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Original: {grid}")
        print(f"Color border at ({row},{col}) with color {color}")
        
        for alg_name, alg_func in algorithms:
            try:
                # Create copy since algorithms may modify input
                grid_copy = [row[:] for row in grid]
                result = alg_func(grid_copy, row, col, color)
                print(f"{alg_name:15} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:15} | ERROR: {str(e)[:30]}")

if __name__ == "__main__":
    test_coloring_border()

"""
Coloring Border demonstrates BFS and DFS techniques for
connected component analysis with boundary detection
and selective modification in grid-based problems.
"""
