"""
1391. Check if There is a Valid Path in a Grid - Multiple Approaches
Difficulty: Medium

You are given an m x n grid. Each cell of the grid represents a street. The street of grid[i][j] can be:

1 which means a street connecting the left and right.
2 which means a street connecting the upper and lower.
3 which means a street connecting the left and lower.
4 which means a street connecting the right and lower.
5 which means a street connecting the left and upper.
6 which means a street connecting the right and upper.

You will initially start at the street at the upper-left corner (0,0) and end at the lower-right corner (m-1,n-1).

Return true if there is a valid path from the start to the end, false otherwise.
"""

from typing import List, Tuple, Set
from collections import deque

class ValidPathInGrid:
    """Multiple approaches to check valid path in directional grid"""
    
    def __init__(self):
        # Define connections for each street type
        # Each street type maps to (can_go_directions, can_come_from_directions)
        self.connections = {
            1: ([(0, -1), (0, 1)], [(0, -1), (0, 1)]),      # left-right
            2: ([(-1, 0), (1, 0)], [(-1, 0), (1, 0)]),      # up-down
            3: ([(0, -1), (1, 0)], [(0, 1), (-1, 0)]),      # left-down
            4: ([(0, 1), (1, 0)], [(0, -1), (-1, 0)]),      # right-down
            5: ([(0, -1), (-1, 0)], [(0, 1), (1, 0)]),      # left-up
            6: ([(0, 1), (-1, 0)], [(0, -1), (1, 0)])       # right-up
        }
    
    def hasValidPath_dfs_recursive(self, grid: List[List[int]]) -> bool:
        """
        Approach 1: Recursive DFS with Connection Validation
        
        Use DFS to explore valid connections between streets.
        
        Time: O(mn), Space: O(mn)
        """
        m, n = len(grid), len(grid[0])
        visited = set()
        
        def can_connect(from_row: int, from_col: int, to_row: int, to_col: int) -> bool:
            """Check if two adjacent cells can connect"""
            if not (0 <= to_row < m and 0 <= to_col < n):
                return False
            
            from_street = grid[from_row][from_col]
            to_street = grid[to_row][to_col]
            
            # Direction from 'from' to 'to'
            direction = (to_row - from_row, to_col - from_col)
            
            # Check if 'from' street can go in this direction
            from_directions, _ = self.connections[from_street]
            if direction not in from_directions:
                return False
            
            # Check if 'to' street can accept connection from this direction
            _, to_directions = self.connections[to_street]
            reverse_direction = (-direction[0], -direction[1])
            if reverse_direction not in to_directions:
                return False
            
            return True
        
        def dfs(row: int, col: int) -> bool:
            if row == m - 1 and col == n - 1:
                return True
            
            if (row, col) in visited:
                return False
            
            visited.add((row, col))
            
            # Try all possible directions from current street
            street_type = grid[row][col]
            directions, _ = self.connections[street_type]
            
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                
                if can_connect(row, col, new_row, new_col):
                    if dfs(new_row, new_col):
                        return True
            
            return False
        
        return dfs(0, 0)
    
    def hasValidPath_dfs_iterative(self, grid: List[List[int]]) -> bool:
        """
        Approach 2: Iterative DFS using Stack
        
        Use explicit stack for DFS traversal.
        
        Time: O(mn), Space: O(mn)
        """
        m, n = len(grid), len(grid[0])
        visited = set()
        stack = [(0, 0)]
        
        def can_connect(from_row: int, from_col: int, to_row: int, to_col: int) -> bool:
            if not (0 <= to_row < m and 0 <= to_col < n):
                return False
            
            from_street = grid[from_row][from_col]
            to_street = grid[to_row][to_col]
            
            direction = (to_row - from_row, to_col - from_col)
            
            from_directions, _ = self.connections[from_street]
            if direction not in from_directions:
                return False
            
            _, to_directions = self.connections[to_street]
            reverse_direction = (-direction[0], -direction[1])
            if reverse_direction not in to_directions:
                return False
            
            return True
        
        while stack:
            row, col = stack.pop()
            
            if row == m - 1 and col == n - 1:
                return True
            
            if (row, col) in visited:
                continue
            
            visited.add((row, col))
            
            # Explore all valid connections
            street_type = grid[row][col]
            directions, _ = self.connections[street_type]
            
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                
                if (can_connect(row, col, new_row, new_col) and 
                    (new_row, new_col) not in visited):
                    stack.append((new_row, new_col))
        
        return False
    
    def hasValidPath_bfs(self, grid: List[List[int]]) -> bool:
        """
        Approach 3: BFS using Queue
        
        Use BFS to find shortest valid path.
        
        Time: O(mn), Space: O(mn)
        """
        m, n = len(grid), len(grid[0])
        visited = set()
        queue = deque([(0, 0)])
        visited.add((0, 0))
        
        def can_connect(from_row: int, from_col: int, to_row: int, to_col: int) -> bool:
            if not (0 <= to_row < m and 0 <= to_col < n):
                return False
            
            from_street = grid[from_row][from_col]
            to_street = grid[to_row][to_col]
            
            direction = (to_row - from_row, to_col - from_col)
            
            from_directions, _ = self.connections[from_street]
            if direction not in from_directions:
                return False
            
            _, to_directions = self.connections[to_street]
            reverse_direction = (-direction[0], -direction[1])
            if reverse_direction not in to_directions:
                return False
            
            return True
        
        while queue:
            row, col = queue.popleft()
            
            if row == m - 1 and col == n - 1:
                return True
            
            # Explore all valid connections
            street_type = grid[row][col]
            directions, _ = self.connections[street_type]
            
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                
                if ((new_row, new_col) not in visited and 
                    can_connect(row, col, new_row, new_col)):
                    visited.add((new_row, new_col))
                    queue.append((new_row, new_col))
        
        return False
    
    def hasValidPath_union_find(self, grid: List[List[int]]) -> bool:
        """
        Approach 4: Union-Find for Connectivity
        
        Use Union-Find to connect valid adjacent cells.
        
        Time: O(mn α(mn)), Space: O(mn)
        """
        m, n = len(grid), len(grid[0])
        
        # Union-Find implementation
        parent = {}
        
        def find(x):
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        def can_connect(from_row: int, from_col: int, to_row: int, to_col: int) -> bool:
            if not (0 <= to_row < m and 0 <= to_col < n):
                return False
            
            from_street = grid[from_row][from_col]
            to_street = grid[to_row][to_col]
            
            direction = (to_row - from_row, to_col - from_col)
            
            from_directions, _ = self.connections[from_street]
            if direction not in from_directions:
                return False
            
            _, to_directions = self.connections[to_street]
            reverse_direction = (-direction[0], -direction[1])
            if reverse_direction not in to_directions:
                return False
            
            return True
        
        # Connect all valid adjacent cells
        for i in range(m):
            for j in range(n):
                street_type = grid[i][j]
                directions, _ = self.connections[street_type]
                
                for dr, dc in directions:
                    ni, nj = i + dr, j + dc
                    
                    if can_connect(i, j, ni, nj):
                        union((i, j), (ni, nj))
        
        # Check if start and end are connected
        return find((0, 0)) == find((m - 1, n - 1))
    
    def hasValidPath_optimized_lookup(self, grid: List[List[int]]) -> bool:
        """
        Approach 5: Optimized with Precomputed Connection Table
        
        Use precomputed table for faster connection checking.
        
        Time: O(mn), Space: O(mn)
        """
        m, n = len(grid), len(grid[0])
        
        # Precomputed connection table: (street1, street2, direction) -> can_connect
        connection_table = {}
        
        def precompute_connections():
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
            
            for street1 in range(1, 7):
                for street2 in range(1, 7):
                    for direction in directions:
                        from_dirs, _ = self.connections[street1]
                        _, to_dirs = self.connections[street2]
                        reverse_dir = (-direction[0], -direction[1])
                        
                        can_connect = (direction in from_dirs and 
                                     reverse_dir in to_dirs)
                        
                        connection_table[(street1, street2, direction)] = can_connect
        
        precompute_connections()
        
        # DFS with optimized connection checking
        visited = set()
        
        def dfs(row: int, col: int) -> bool:
            if row == m - 1 and col == n - 1:
                return True
            
            if (row, col) in visited:
                return False
            
            visited.add((row, col))
            
            street_type = grid[row][col]
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            
            for direction in directions:
                dr, dc = direction
                new_row, new_col = row + dr, col + dc
                
                if (0 <= new_row < m and 0 <= new_col < n):
                    new_street = grid[new_row][new_col]
                    
                    if connection_table.get((street_type, new_street, direction), False):
                        if dfs(new_row, new_col):
                            return True
            
            return False
        
        return dfs(0, 0)

def test_valid_path_in_grid():
    """Test valid path in grid algorithms"""
    solver = ValidPathInGrid()
    
    test_cases = [
        ([[2,4,3],[6,5,2]], True, "Simple valid path"),
        ([[1,2,1],[1,2,1]], False, "No valid path"),
        ([[1,1,2]], False, "Horizontal then vertical mismatch"),
        ([[1,1,1,1,1,1,3]], True, "Long horizontal path"),
        ([[6],[4]], False, "Vertical mismatch"),
        ([[4,1],[6,1]], True, "L-shaped path"),
    ]
    
    algorithms = [
        ("Recursive DFS", solver.hasValidPath_dfs_recursive),
        ("Iterative DFS", solver.hasValidPath_dfs_iterative),
        ("BFS", solver.hasValidPath_bfs),
        ("Union-Find", solver.hasValidPath_union_find),
        ("Optimized Lookup", solver.hasValidPath_optimized_lookup),
    ]
    
    print("=== Testing Valid Path in Grid ===")
    
    for grid, expected, description in test_cases:
        print(f"\n--- {description} (Expected: {expected}) ---")
        print(f"Grid: {grid}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(grid)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:18} | {status} | Valid path: {result}")
            except Exception as e:
                print(f"{alg_name:18} | ERROR: {str(e)[:30]}")

if __name__ == "__main__":
    test_valid_path_in_grid()

"""
Valid Path in Grid demonstrates advanced DFS/BFS techniques
with directional constraints and connection validation
for complex grid-based pathfinding problems.
"""
