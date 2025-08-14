"""
Grid-Based Graph Algorithms
This module implements various graph algorithms specifically designed for grid/matrix problems.
"""

from collections import deque
import heapq
from typing import List, Tuple, Set, Dict

class GridBasedGraphs:
    
    def __init__(self):
        """Initialize grid-based graph algorithms"""
        # Common directions for 4-directional movement
        self.directions_4 = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        # Common directions for 8-directional movement
        self.directions_8 = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    
    def is_valid(self, grid: List[List[int]], row: int, col: int) -> bool:
        """Check if coordinates are valid within the grid"""
        return 0 <= row < len(grid) and 0 <= col < len(grid[0])
    
    def is_valid_with_value(self, grid: List[List[int]], row: int, col: int, target_value: int) -> bool:
        """Check if coordinates are valid and have the target value"""
        return (self.is_valid(grid, row, col) and grid[row][col] == target_value)
    
    # ==================== GRAPH FROM GRID ====================
    
    def grid_to_graph(self, grid: List[List[int]], connectivity: str = "4-way") -> Dict:
        """
        Convert a 2D grid to graph representation
        
        Args:
            grid: 2D grid/matrix
            connectivity: "4-way" or "8-way" connectivity
        
        Returns:
            dict: Graph representation with adjacency list
        """
        if not grid or not grid[0]:
            return {}
        
        rows, cols = len(grid), len(grid[0])
        directions = self.directions_4 if connectivity == "4-way" else self.directions_8
        
        # Convert 2D coordinates to 1D node ID
        def get_node_id(r, c):
            return r * cols + c
        
        # Convert 1D node ID back to 2D coordinates
        def get_coordinates(node_id):
            return node_id // cols, node_id % cols
        
        graph = {}
        
        for r in range(rows):
            for c in range(cols):
                node_id = get_node_id(r, c)
                graph[node_id] = []
                
                # Add neighbors based on connectivity
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if self.is_valid(grid, nr, nc):
                        neighbor_id = get_node_id(nr, nc)
                        # You can add conditions here (e.g., only connect cells with same value)
                        graph[node_id].append(neighbor_id)
        
        return {
            'graph': graph,
            'get_node_id': get_node_id,
            'get_coordinates': get_coordinates,
            'rows': rows,
            'cols': cols
        }
    
    # ==================== SHORTEST PATH IN GRID ====================
    
    def shortest_path_bfs(self, grid: List[List[int]], start: Tuple[int, int], 
                         end: Tuple[int, int], obstacle_value: int = 1) -> int:
        """
        Find shortest path in grid using BFS (unweighted)
        
        Time Complexity: O(rows * cols)
        Space Complexity: O(rows * cols)
        
        Args:
            grid: 2D grid where 0 is walkable, obstacle_value is blocked
            start: Starting coordinates (row, col)
            end: Ending coordinates (row, col)
            obstacle_value: Value representing obstacles
        
        Returns:
            int: Shortest path length, -1 if no path exists
        """
        if not grid or not grid[0]:
            return -1
        
        rows, cols = len(grid), len(grid[0])
        start_r, start_c = start
        end_r, end_c = end
        
        # Check if start or end is blocked
        if (grid[start_r][start_c] == obstacle_value or 
            grid[end_r][end_c] == obstacle_value):
            return -1
        
        if start == end:
            return 0
        
        queue = deque([(start_r, start_c, 0)])  # (row, col, distance)
        visited = set()
        visited.add((start_r, start_c))
        
        while queue:
            row, col, dist = queue.popleft()
            
            # Check all 4 directions
            for dr, dc in self.directions_4:
                new_row, new_col = row + dr, col + dc
                
                if (self.is_valid(grid, new_row, new_col) and 
                    (new_row, new_col) not in visited and 
                    grid[new_row][new_col] != obstacle_value):
                    
                    if (new_row, new_col) == (end_r, end_c):
                        return dist + 1
                    
                    visited.add((new_row, new_col))
                    queue.append((new_row, new_col, dist + 1))
        
        return -1
    
    def shortest_path_dijkstra(self, grid: List[List[int]], start: Tuple[int, int], 
                              end: Tuple[int, int]) -> int:
        """
        Find shortest path in weighted grid using Dijkstra's algorithm
        
        Time Complexity: O(rows * cols * log(rows * cols))
        Space Complexity: O(rows * cols)
        
        Args:
            grid: 2D grid with weights (negative values can represent obstacles)
            start: Starting coordinates
            end: Ending coordinates
        
        Returns:
            int: Shortest path weight, float('inf') if no path
        """
        if not grid or not grid[0]:
            return float('inf')
        
        rows, cols = len(grid), len(grid[0])
        start_r, start_c = start
        end_r, end_c = end
        
        # Priority queue: (distance, row, col)
        pq = [(grid[start_r][start_c], start_r, start_c)]
        distances = {}
        distances[(start_r, start_c)] = grid[start_r][start_c]
        
        while pq:
            current_dist, row, col = heapq.heappop(pq)
            
            if (row, col) == (end_r, end_c):
                return current_dist
            
            if current_dist > distances.get((row, col), float('inf')):
                continue
            
            # Check all 4 directions
            for dr, dc in self.directions_4:
                new_row, new_col = row + dr, col + dc
                
                if self.is_valid(grid, new_row, new_col) and grid[new_row][new_col] >= 0:
                    new_dist = current_dist + grid[new_row][new_col]
                    
                    if new_dist < distances.get((new_row, new_col), float('inf')):
                        distances[(new_row, new_col)] = new_dist
                        heapq.heappush(pq, (new_dist, new_row, new_col))
        
        return distances.get((end_r, end_c), float('inf'))
    
    def shortest_path_with_obstacles(self, grid: List[List[int]], start: Tuple[int, int], 
                                   end: Tuple[int, int], k: int) -> int:
        """
        Find shortest path with ability to eliminate k obstacles
        
        Args:
            grid: 2D grid where 0 is walkable, 1 is obstacle
            start: Starting coordinates
            end: Ending coordinates
            k: Number of obstacles that can be eliminated
        
        Returns:
            int: Shortest path length, -1 if impossible
        """
        if not grid or not grid[0]:
            return -1
        
        rows, cols = len(grid), len(grid[0])
        start_r, start_c = start
        end_r, end_c = end
        
        if start == end:
            return 0
        
        # State: (row, col, obstacles_eliminated)
        queue = deque([(start_r, start_c, 0, 0)])  # (row, col, obstacles_used, steps)
        visited = set()
        visited.add((start_r, start_c, 0))
        
        while queue:
            row, col, obstacles_used, steps = queue.popleft()
            
            for dr, dc in self.directions_4:
                new_row, new_col = row + dr, col + dc
                
                if self.is_valid(grid, new_row, new_col):
                    new_obstacles_used = obstacles_used
                    
                    # If it's an obstacle, we need to eliminate it
                    if grid[new_row][new_col] == 1:
                        new_obstacles_used += 1
                    
                    # Check if we can proceed with current obstacle elimination count
                    if (new_obstacles_used <= k and 
                        (new_row, new_col, new_obstacles_used) not in visited):
                        
                        if (new_row, new_col) == (end_r, end_c):
                            return steps + 1
                        
                        visited.add((new_row, new_col, new_obstacles_used))
                        queue.append((new_row, new_col, new_obstacles_used, steps + 1))
        
        return -1
    
    # ==================== BFS/DFS IN MATRIX ====================
    
    def bfs_matrix(self, grid: List[List[int]], start: Tuple[int, int], 
                   target_value: int = 0) -> List[Tuple[int, int]]:
        """
        Perform BFS traversal in matrix starting from given position
        
        Args:
            grid: 2D matrix
            start: Starting position
            target_value: Value to consider as valid for traversal
        
        Returns:
            list: Visited cells in BFS order
        """
        if not grid or not grid[0]:
            return []
        
        rows, cols = len(grid), len(grid[0])
        start_r, start_c = start
        
        if not self.is_valid_with_value(grid, start_r, start_c, target_value):
            return []
        
        queue = deque([(start_r, start_c)])
        visited = set()
        visited.add((start_r, start_c))
        result = []
        
        while queue:
            row, col = queue.popleft()
            result.append((row, col))
            
            for dr, dc in self.directions_4:
                new_row, new_col = row + dr, col + dc
                
                if (self.is_valid_with_value(grid, new_row, new_col, target_value) and 
                    (new_row, new_col) not in visited):
                    
                    visited.add((new_row, new_col))
                    queue.append((new_row, new_col))
        
        return result
    
    def dfs_matrix(self, grid: List[List[int]], start: Tuple[int, int], 
                   target_value: int = 0) -> List[Tuple[int, int]]:
        """
        Perform DFS traversal in matrix starting from given position
        
        Args:
            grid: 2D matrix
            start: Starting position
            target_value: Value to consider as valid for traversal
        
        Returns:
            list: Visited cells in DFS order
        """
        if not grid or not grid[0]:
            return []
        
        rows, cols = len(grid), len(grid[0])
        start_r, start_c = start
        
        if not self.is_valid_with_value(grid, start_r, start_c, target_value):
            return []
        
        visited = set()
        result = []
        
        def dfs(row, col):
            visited.add((row, col))
            result.append((row, col))
            
            for dr, dc in self.directions_4:
                new_row, new_col = row + dr, col + dc
                
                if (self.is_valid_with_value(grid, new_row, new_col, target_value) and 
                    (new_row, new_col) not in visited):
                    
                    dfs(new_row, new_col)
        
        dfs(start_r, start_c)
        return result
    
    # ==================== FLOOD FILL ALGORITHM ====================
    
    def flood_fill(self, image: List[List[int]], sr: int, sc: int, new_color: int) -> List[List[int]]:
        """
        Flood fill algorithm - changes connected pixels of same color to new color
        
        Time Complexity: O(rows * cols)
        Space Complexity: O(rows * cols) for recursion stack
        
        Args:
            image: 2D image matrix
            sr, sc: Starting pixel coordinates
            new_color: New color to fill with
        
        Returns:
            Modified image matrix
        """
        if not image or not image[0]:
            return image
        
        rows, cols = len(image), len(image[0])
        original_color = image[sr][sc]
        
        # If new color is same as original, no change needed
        if original_color == new_color:
            return image
        
        def fill(row, col):
            # Base case: out of bounds or different color
            if (not self.is_valid(image, row, col) or 
                image[row][col] != original_color):
                return
            
            # Fill current pixel
            image[row][col] = new_color
            
            # Recursively fill in 4 directions
            for dr, dc in self.directions_4:
                fill(row + dr, col + dc)
        
        fill(sr, sc)
        return image
    
    def flood_fill_iterative(self, image: List[List[int]], sr: int, sc: int, 
                           new_color: int) -> List[List[int]]:
        """
        Iterative version of flood fill using stack
        """
        if not image or not image[0]:
            return image
        
        original_color = image[sr][sc]
        if original_color == new_color:
            return image
        
        stack = [(sr, sc)]
        
        while stack:
            row, col = stack.pop()
            
            if (not self.is_valid(image, row, col) or 
                image[row][col] != original_color):
                continue
            
            image[row][col] = new_color
            
            # Add neighbors to stack
            for dr, dc in self.directions_4:
                stack.append((row + dr, col + dc))
        
        return image
    
    # ==================== ISLANDS COUNT (CONNECTED COMPONENTS) ====================
    
    def count_islands(self, grid: List[List[str]]) -> int:
        """
        Count number of islands (connected components of '1's)
        
        Time Complexity: O(rows * cols)
        Space Complexity: O(rows * cols)
        
        Args:
            grid: 2D grid where '1' is land and '0' is water
        
        Returns:
            int: Number of islands
        """
        if not grid or not grid[0]:
            return 0
        
        rows, cols = len(grid), len(grid[0])
        visited = set()
        island_count = 0
        
        def dfs(row, col):
            if ((row, col) in visited or 
                not self.is_valid(grid, row, col) or 
                grid[row][col] == '0'):
                return
            
            visited.add((row, col))
            
            # Visit all connected land cells
            for dr, dc in self.directions_4:
                dfs(row + dr, col + dc)
        
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == '1' and (r, c) not in visited:
                    dfs(r, c)
                    island_count += 1
        
        return island_count
    
    def max_area_of_island(self, grid: List[List[int]]) -> int:
        """
        Find the maximum area of an island
        
        Args:
            grid: 2D grid where 1 is land and 0 is water
        
        Returns:
            int: Maximum island area
        """
        if not grid or not grid[0]:
            return 0
        
        rows, cols = len(grid), len(grid[0])
        visited = set()
        max_area = 0
        
        def dfs(row, col):
            if ((row, col) in visited or 
                not self.is_valid(grid, row, col) or 
                grid[row][col] == 0):
                return 0
            
            visited.add((row, col))
            area = 1
            
            # Count area of connected land cells
            for dr, dc in self.directions_4:
                area += dfs(row + dr, col + dc)
            
            return area
        
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 1 and (r, c) not in visited:
                    area = dfs(r, c)
                    max_area = max(max_area, area)
        
        return max_area
    
    # ==================== WALLS AND GATES ====================
    
    def walls_and_gates(self, rooms: List[List[int]]) -> None:
        """
        Fill each empty room with distance to nearest gate
        
        Args:
            rooms: 2D grid where -1 is wall, 0 is gate, INF is empty room
        """
        if not rooms or not rooms[0]:
            return
        
        rows, cols = len(rooms), len(rooms[0])
        INF = 2147483647
        
        # Multi-source BFS from all gates
        queue = deque()
        
        # Find all gates and add to queue
        for r in range(rows):
            for c in range(cols):
                if rooms[r][c] == 0:
                    queue.append((r, c))
        
        # BFS to fill distances
        while queue:
            row, col = queue.popleft()
            
            for dr, dc in self.directions_4:
                new_row, new_col = row + dr, col + dc
                
                # If it's a valid empty room and we found a shorter path
                if (self.is_valid(rooms, new_row, new_col) and 
                    rooms[new_row][new_col] == INF):
                    
                    rooms[new_row][new_col] = rooms[row][col] + 1
                    queue.append((new_row, new_col))
    
    # ==================== ROTTING ORANGES ====================
    
    def oranges_rotting(self, grid: List[List[int]]) -> int:
        """
        Find minimum time for all oranges to rot
        
        Time Complexity: O(rows * cols)
        Space Complexity: O(rows * cols)
        
        Args:
            grid: 2D grid where 0=empty, 1=fresh orange, 2=rotten orange
        
        Returns:
            int: Minimum minutes for all oranges to rot, -1 if impossible
        """
        if not grid or not grid[0]:
            return 0
        
        rows, cols = len(grid), len(grid[0])
        queue = deque()
        fresh_count = 0
        
        # Find all initially rotten oranges and count fresh ones
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 2:
                    queue.append((r, c, 0))  # (row, col, time)
                elif grid[r][c] == 1:
                    fresh_count += 1
        
        # If no fresh oranges, no time needed
        if fresh_count == 0:
            return 0
        
        max_time = 0
        
        # Multi-source BFS from all rotten oranges
        while queue:
            row, col, time = queue.popleft()
            max_time = max(max_time, time)
            
            for dr, dc in self.directions_4:
                new_row, new_col = row + dr, col + dc
                
                # If it's a fresh orange, make it rotten
                if (self.is_valid(grid, new_row, new_col) and 
                    grid[new_row][new_col] == 1):
                    
                    grid[new_row][new_col] = 2
                    fresh_count -= 1
                    queue.append((new_row, new_col, time + 1))
        
        # Return time if all oranges are rotten, otherwise -1
        return max_time if fresh_count == 0 else -1
    
    # ==================== UTILITY METHODS ====================
    
    def display_grid(self, grid: List[List[int]]) -> None:
        """Display grid in a formatted way"""
        if not grid:
            print("Empty grid")
            return
        
        for row in grid:
            print(' '.join(f'{cell:2}' for cell in row))
        print()
    
    def create_test_grid(self, rows: int, cols: int, pattern: str = "empty") -> List[List[int]]:
        """Create test grids for demonstration"""
        if pattern == "empty":
            return [[0 for _ in range(cols)] for _ in range(rows)]
        elif pattern == "obstacles":
            # Create a grid with some obstacles
            grid = [[0 for _ in range(cols)] for _ in range(rows)]
            for r in range(rows):
                for c in range(cols):
                    if (r + c) % 3 == 0:
                        grid[r][c] = 1
            return grid
        elif pattern == "islands":
            # Create a grid with islands pattern
            grid = [
                [1, 1, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 1]
            ]
            return grid
        return [[0 for _ in range(cols)] for _ in range(rows)]


# ==================== EXAMPLE USAGE AND TESTING ====================

if __name__ == "__main__":
    print("=== Grid-Based Graph Algorithms Demo ===\n")
    
    grid_algo = GridBasedGraphs()
    
    # Example 1: Graph from Grid
    print("1. Graph from Grid Conversion:")
    test_grid = [
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ]
    graph_info = grid_algo.grid_to_graph(test_grid)
    print("Original Grid:")
    grid_algo.display_grid(test_grid)
    print(f"Graph representation (first few nodes): {dict(list(graph_info['graph'].items())[:4])}")
    print()
    
    # Example 2: Shortest Path in Grid
    print("2. Shortest Path in Grid:")
    path_grid = [
        [0, 0, 0, 0],
        [1, 1, 0, 1],
        [0, 0, 0, 0],
        [0, 1, 1, 0]
    ]
    print("Grid (0=walkable, 1=obstacle):")
    grid_algo.display_grid(path_grid)
    
    start = (0, 0)
    end = (3, 3)
    path_length = grid_algo.shortest_path_bfs(path_grid, start, end)
    print(f"Shortest path from {start} to {end}: {path_length}")
    
    # Test with obstacles elimination
    path_with_k = grid_algo.shortest_path_with_obstacles(path_grid, start, end, k=1)
    print(f"Shortest path with 1 obstacle elimination: {path_with_k}")
    print()
    
    # Example 3: BFS and DFS in Matrix
    print("3. BFS and DFS Traversal:")
    traversal_grid = [
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 0]
    ]
    print("Grid for traversal (traversing 0s):")
    grid_algo.display_grid(traversal_grid)
    
    bfs_result = grid_algo.bfs_matrix(traversal_grid, (0, 0), target_value=0)
    dfs_result = grid_algo.dfs_matrix(traversal_grid, (0, 0), target_value=0)
    
    print(f"BFS traversal from (0,0): {bfs_result[:8]}...")
    print(f"DFS traversal from (0,0): {dfs_result[:8]}...")
    print()
    
    # Example 4: Flood Fill
    print("4. Flood Fill Algorithm:")
    image = [
        [1, 1, 1],
        [1, 1, 0],
        [1, 0, 1]
    ]
    print("Original image:")
    grid_algo.display_grid(image)
    
    filled_image = grid_algo.flood_fill([row[:] for row in image], 1, 1, 2)
    print("After flood fill from (1,1) with color 2:")
    grid_algo.display_grid(filled_image)
    
    # Example 5: Count Islands
    print("5. Islands Count:")
    islands_grid = [
        ['1', '1', '0', '0', '0'],
        ['1', '1', '0', '0', '0'],
        ['0', '0', '1', '0', '0'],
        ['0', '0', '0', '1', '1']
    ]
    print("Islands grid ('1'=land, '0'=water):")
    for row in islands_grid:
        print(' '.join(row))
    
    island_count = grid_algo.count_islands(islands_grid)
    print(f"Number of islands: {island_count}")
    
    # Max area of island
    area_grid = [
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 1]
    ]
    max_area = grid_algo.max_area_of_island(area_grid)
    print(f"Maximum island area: {max_area}")
    print()
    
    # Example 6: Walls and Gates
    print("6. Walls and Gates:")
    INF = 2147483647
    rooms = [
        [INF, -1, 0, INF],
        [INF, INF, INF, -1],
        [INF, -1, INF, -1],
        [0, -1, INF, INF]
    ]
    print("Rooms (-1=wall, 0=gate, INF=empty room):")
    for row in rooms:
        formatted_row = []
        for cell in row:
            if cell == INF:
                formatted_row.append("INF")
            else:
                formatted_row.append(str(cell))
        print(' '.join(f'{cell:>3}' for cell in formatted_row))
    
    grid_algo.walls_and_gates(rooms)
    print("\nAfter filling with distances to nearest gate:")
    for row in rooms:
        formatted_row = []
        for cell in row:
            if cell == INF:
                formatted_row.append("INF")
            else:
                formatted_row.append(str(cell))
        print(' '.join(f'{cell:>3}' for cell in formatted_row))
    print()
    
    # Example 7: Rotting Oranges
    print("7. Rotting Oranges:")
    oranges = [
        [2, 1, 1],
        [1, 1, 0],
        [0, 1, 1]
    ]
    print("Oranges grid (0=empty, 1=fresh, 2=rotten):")
    grid_algo.display_grid(oranges)
    
    time_to_rot = grid_algo.oranges_rotting([row[:] for row in oranges])
    print(f"Time for all oranges to rot: {time_to_rot} minutes")
    
    # Another example
    oranges2 = [
        [2, 1, 1],
        [0, 1, 1],
        [1, 0, 1]
    ]
    time_to_rot2 = grid_algo.oranges_rotting(oranges2)
    print(f"Time for second grid: {time_to_rot2} minutes")
    
    print("\n=== Demo Complete ===") 