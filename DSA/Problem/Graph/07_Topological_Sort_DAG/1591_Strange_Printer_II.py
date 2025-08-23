"""
1591. Strange Printer II - Multiple Approaches
Difficulty: Medium

There is a strange printer with the following two special requirements:
1. On each turn, the printer will print a solid rectangular pattern of a single color on the grid.
2. Once the printer has used a color for the above operation, the same color cannot be used again.

You are given a m x n matrix targetGrid, where targetGrid[i][j] is the color in the position (i, j) of the grid.

Return true if it is possible to print the target grid, otherwise, return false.
"""

from typing import List, Set, Dict, Tuple
from collections import defaultdict, deque

class StrangePrinterII:
    """Multiple approaches to solve strange printer II problem"""
    
    def isPrintable_topological_sort(self, targetGrid: List[List[int]]) -> bool:
        """
        Approach 1: Topological Sort with Color Dependencies
        
        Build dependency graph between colors and check for cycles.
        
        Time: O(m*n + C²), Space: O(C²) where C is number of colors
        """
        if not targetGrid or not targetGrid[0]:
            return True
        
        m, n = len(targetGrid), len(targetGrid[0])
        
        # Find bounding box for each color
        color_bounds = {}
        for i in range(m):
            for j in range(n):
                color = targetGrid[i][j]
                if color not in color_bounds:
                    color_bounds[color] = [i, i, j, j]  # top, bottom, left, right
                else:
                    bounds = color_bounds[color]
                    bounds[0] = min(bounds[0], i)  # top
                    bounds[1] = max(bounds[1], i)  # bottom
                    bounds[2] = min(bounds[2], j)  # left
                    bounds[3] = max(bounds[3], j)  # right
        
        # Build dependency graph
        graph = defaultdict(set)
        in_degree = defaultdict(int)
        colors = set(color_bounds.keys())
        
        for color in colors:
            in_degree[color] = 0
        
        # For each color, check what other colors are in its bounding box
        for color, (top, bottom, left, right) in color_bounds.items():
            overlapping_colors = set()
            
            for i in range(top, bottom + 1):
                for j in range(left, right + 1):
                    if targetGrid[i][j] != color:
                        overlapping_colors.add(targetGrid[i][j])
            
            # Color must be printed before all overlapping colors
            for other_color in overlapping_colors:
                if other_color not in graph[color]:
                    graph[color].add(other_color)
                    in_degree[other_color] += 1
        
        # Topological sort using Kahn's algorithm
        queue = deque()
        for color in colors:
            if in_degree[color] == 0:
                queue.append(color)
        
        processed = 0
        while queue:
            color = queue.popleft()
            processed += 1
            
            for neighbor in graph[color]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return processed == len(colors)
    
    def isPrintable_dfs_cycle_detection(self, targetGrid: List[List[int]]) -> bool:
        """
        Approach 2: DFS with Cycle Detection
        
        Use DFS to detect cycles in color dependency graph.
        
        Time: O(m*n + C²), Space: O(C²)
        """
        if not targetGrid or not targetGrid[0]:
            return True
        
        m, n = len(targetGrid), len(targetGrid[0])
        
        # Find color rectangles
        color_rect = {}
        for i in range(m):
            for j in range(n):
                color = targetGrid[i][j]
                if color not in color_rect:
                    color_rect[color] = [i, i, j, j]
                else:
                    rect = color_rect[color]
                    rect[0] = min(rect[0], i)
                    rect[1] = max(rect[1], i)
                    rect[2] = min(rect[2], j)
                    rect[3] = max(rect[3], j)
        
        # Build dependency graph
        dependencies = defaultdict(set)
        
        for color, (r1, r2, c1, c2) in color_rect.items():
            for i in range(r1, r2 + 1):
                for j in range(c1, c2 + 1):
                    if targetGrid[i][j] != color:
                        dependencies[targetGrid[i][j]].add(color)
        
        # DFS cycle detection
        WHITE, GRAY, BLACK = 0, 1, 2
        color_state = {color: WHITE for color in color_rect}
        
        def has_cycle(color: int) -> bool:
            if color_state[color] == GRAY:  # Back edge found
                return True
            if color_state[color] == BLACK:  # Already processed
                return False
            
            color_state[color] = GRAY
            
            for dependent in dependencies[color]:
                if has_cycle(dependent):
                    return True
            
            color_state[color] = BLACK
            return False
        
        # Check for cycles starting from each unvisited color
        for color in color_rect:
            if color_state[color] == WHITE:
                if has_cycle(color):
                    return False
        
        return True
    
    def isPrintable_iterative_removal(self, targetGrid: List[List[int]]) -> bool:
        """
        Approach 3: Iterative Color Removal
        
        Iteratively remove colors that can be printed without conflicts.
        
        Time: O(m*n*C), Space: O(C)
        """
        if not targetGrid or not targetGrid[0]:
            return True
        
        m, n = len(targetGrid), len(targetGrid[0])
        
        # Get all unique colors
        colors = set()
        for row in targetGrid:
            colors.update(row)
        
        def get_bounding_box(color: int) -> Tuple[int, int, int, int]:
            """Get bounding box for a color"""
            min_r, max_r = m, -1
            min_c, max_c = n, -1
            
            for i in range(m):
                for j in range(n):
                    if targetGrid[i][j] == color:
                        min_r = min(min_r, i)
                        max_r = max(max_r, i)
                        min_c = min(min_c, j)
                        max_c = max(max_c, j)
            
            return min_r, max_r, min_c, max_c
        
        def can_print_color(color: int) -> bool:
            """Check if color can be printed (no other colors in its bounding box)"""
            min_r, max_r, min_c, max_c = get_bounding_box(color)
            
            if min_r > max_r:  # Color not found
                return True
            
            for i in range(min_r, max_r + 1):
                for j in range(min_c, max_c + 1):
                    if targetGrid[i][j] != color and targetGrid[i][j] != 0:
                        return False
            
            return True
        
        def remove_color(color: int):
            """Remove color from grid (set to 0)"""
            for i in range(m):
                for j in range(n):
                    if targetGrid[i][j] == color:
                        targetGrid[i][j] = 0
        
        # Make a copy to avoid modifying original
        grid_copy = [row[:] for row in targetGrid]
        targetGrid = grid_copy
        
        remaining_colors = colors.copy()
        
        while remaining_colors:
            removed_any = False
            
            for color in list(remaining_colors):
                if can_print_color(color):
                    remove_color(color)
                    remaining_colors.remove(color)
                    removed_any = True
            
            if not removed_any:
                return False  # Cycle detected
        
        return True
    
    def isPrintable_constraint_propagation(self, targetGrid: List[List[int]]) -> bool:
        """
        Approach 4: Constraint Propagation
        
        Use constraint propagation to determine printing order.
        
        Time: O(m*n + C²), Space: O(C²)
        """
        if not targetGrid or not targetGrid[0]:
            return True
        
        m, n = len(targetGrid), len(targetGrid[0])
        
        # Find all colors and their positions
        color_positions = defaultdict(list)
        for i in range(m):
            for j in range(n):
                color_positions[targetGrid[i][j]].append((i, j))
        
        colors = list(color_positions.keys())
        
        # Build constraint graph
        must_print_before = defaultdict(set)
        
        for color in colors:
            positions = color_positions[color]
            if not positions:
                continue
            
            # Find bounding rectangle
            min_r = min(pos[0] for pos in positions)
            max_r = max(pos[0] for pos in positions)
            min_c = min(pos[1] for pos in positions)
            max_c = max(pos[1] for pos in positions)
            
            # Check what colors are in the bounding rectangle
            colors_in_rect = set()
            for i in range(min_r, max_r + 1):
                for j in range(min_c, max_c + 1):
                    colors_in_rect.add(targetGrid[i][j])
            
            # Current color must be printed before all other colors in its rectangle
            for other_color in colors_in_rect:
                if other_color != color:
                    must_print_before[other_color].add(color)
        
        # Check for cycles using topological sort
        in_degree = {color: 0 for color in colors}
        for color in colors:
            for dependent in must_print_before[color]:
                in_degree[dependent] += 1
        
        queue = deque()
        for color in colors:
            if in_degree[color] == 0:
                queue.append(color)
        
        processed = 0
        while queue:
            color = queue.popleft()
            processed += 1
            
            for dependent in must_print_before[color]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        return processed == len(colors)
    
    def isPrintable_optimized_bounds_check(self, targetGrid: List[List[int]]) -> bool:
        """
        Approach 5: Optimized Bounds Checking
        
        Optimized approach with efficient bounds checking.
        
        Time: O(m*n + C²), Space: O(C)
        """
        if not targetGrid or not targetGrid[0]:
            return True
        
        m, n = len(targetGrid), len(targetGrid[0])
        
        # Collect color information
        color_info = {}
        for i in range(m):
            for j in range(n):
                color = targetGrid[i][j]
                if color not in color_info:
                    color_info[color] = {
                        'min_r': i, 'max_r': i,
                        'min_c': j, 'max_c': j,
                        'positions': set()
                    }
                
                info = color_info[color]
                info['min_r'] = min(info['min_r'], i)
                info['max_r'] = max(info['max_r'], i)
                info['min_c'] = min(info['min_c'], j)
                info['max_c'] = max(info['max_c'], j)
                info['positions'].add((i, j))
        
        # Build dependency graph
        graph = defaultdict(set)
        in_degree = defaultdict(int)
        
        for color in color_info:
            in_degree[color] = 0
        
        for color, info in color_info.items():
            # Check all cells in bounding rectangle
            for i in range(info['min_r'], info['max_r'] + 1):
                for j in range(info['min_c'], info['max_c'] + 1):
                    cell_color = targetGrid[i][j]
                    if cell_color != color:
                        # cell_color must be printed after color
                        if cell_color not in graph[color]:
                            graph[color].add(cell_color)
                            in_degree[cell_color] += 1
        
        # Topological sort
        queue = deque()
        for color in color_info:
            if in_degree[color] == 0:
                queue.append(color)
        
        count = 0
        while queue:
            color = queue.popleft()
            count += 1
            
            for neighbor in graph[color]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return count == len(color_info)

def test_strange_printer_ii():
    """Test strange printer II algorithms"""
    solver = StrangePrinterII()
    
    test_cases = [
        ([[1,1,1,1],[1,2,2,1],[1,2,2,1],[1,1,1,1]], True, "Example 1"),
        ([[1,1,1,1],[1,1,3,3],[1,1,3,4],[5,5,1,4]], True, "Example 2"),
        ([[1,2,1],[2,1,2],[1,2,1]], False, "Impossible pattern"),
        ([[1]], True, "Single cell"),
        ([[1,1],[2,2]], True, "Simple 2x2"),
    ]
    
    algorithms = [
        ("Topological Sort", solver.isPrintable_topological_sort),
        ("DFS Cycle Detection", solver.isPrintable_dfs_cycle_detection),
        ("Iterative Removal", solver.isPrintable_iterative_removal),
        ("Constraint Propagation", solver.isPrintable_constraint_propagation),
        ("Optimized Bounds Check", solver.isPrintable_optimized_bounds_check),
    ]
    
    print("=== Testing Strange Printer II ===")
    
    for targetGrid, expected, description in test_cases:
        print(f"\n--- {description} (Expected: {expected}) ---")
        print(f"Grid: {targetGrid}")
        
        for alg_name, alg_func in algorithms:
            try:
                # Make a copy since some algorithms modify the grid
                grid_copy = [row[:] for row in targetGrid]
                result = alg_func(grid_copy)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:22} | {status} | Printable: {result}")
            except Exception as e:
                print(f"{alg_name:22} | ERROR: {str(e)[:30]}")

if __name__ == "__main__":
    test_strange_printer_ii()

"""
Strange Printer II demonstrates topological sorting applications
for constraint satisfaction problems with geometric constraints
and dependency analysis in 2D grids.
"""
