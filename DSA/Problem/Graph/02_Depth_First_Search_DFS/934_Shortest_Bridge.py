"""
934. Shortest Bridge - Multiple Approaches
Difficulty: Medium

You are given an n x n binary matrix grid where 1 represents land and 0 represents water.

An island is a 4-directionally connected group of 1's not connected to any other group of 1's. 
There are exactly two islands in grid.

You may change 0's to 1's to connect the two islands to form one island.

Return the smallest number of 0's you must flip to connect the two islands.
"""

from typing import List, Tuple, Set
from collections import deque

class ShortestBridge:
    """Multiple approaches to find shortest bridge between two islands"""
    
    def shortestBridge_dfs_bfs(self, grid: List[List[int]]) -> int:
        """
        Approach 1: DFS to find first island + BFS to find shortest path
        
        Use DFS to mark first island, then BFS to expand until reaching second island.
        
        Time: O(n²), Space: O(n²)
        """
        n = len(grid)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        def dfs(i: int, j: int, island_cells: List[Tuple[int, int]]):
            """DFS to mark all cells of first island"""
            if (i < 0 or i >= n or j < 0 or j >= n or 
                grid[i][j] != 1):
                return
            
            grid[i][j] = 2  # Mark as visited (first island)
            island_cells.append((i, j))
            
            for di, dj in directions:
                dfs(i + di, j + dj, island_cells)
        
        # Find and mark first island using DFS
        first_island = []
        found = False
        
        for i in range(n):
            if found:
                break
            for j in range(n):
                if grid[i][j] == 1:
                    dfs(i, j, first_island)
                    found = True
                    break
        
        # BFS from first island to find shortest path to second island
        queue = deque([(i, j, 0) for i, j in first_island])
        
        while queue:
            x, y, dist = queue.popleft()
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                
                if 0 <= nx < n and 0 <= ny < n:
                    if grid[nx][ny] == 1:  # Found second island
                        return dist
                    elif grid[nx][ny] == 0:  # Water, can expand
                        grid[nx][ny] = 2  # Mark as visited
                        queue.append((nx, ny, dist + 1))
        
        return -1  # Should not reach here
    
    def shortestBridge_two_pass_bfs(self, grid: List[List[int]]) -> int:
        """
        Approach 2: Two-pass BFS
        
        First BFS to identify and mark first island, second BFS to find bridge.
        
        Time: O(n²), Space: O(n²)
        """
        n = len(grid)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        # First pass: Find and mark first island
        def mark_first_island():
            for i in range(n):
                for j in range(n):
                    if grid[i][j] == 1:
                        # BFS to mark entire first island
                        queue = deque([(i, j)])
                        grid[i][j] = 2
                        island_boundary = []
                        
                        while queue:
                            x, y = queue.popleft()
                            is_boundary = False
                            
                            for dx, dy in directions:
                                nx, ny = x + dx, y + dy
                                
                                if 0 <= nx < n and 0 <= ny < n:
                                    if grid[nx][ny] == 1:
                                        grid[nx][ny] = 2
                                        queue.append((nx, ny))
                                    elif grid[nx][ny] == 0:
                                        is_boundary = True
                                
                            if is_boundary:
                                island_boundary.append((x, y))
                        
                        return island_boundary
            return []
        
        # Mark first island and get boundary cells
        boundary = mark_first_island()
        
        # Second pass: BFS from boundary to find shortest bridge
        queue = deque([(x, y, 0) for x, y in boundary])
        visited = set(boundary)
        
        while queue:
            x, y, dist = queue.popleft()
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                
                if (0 <= nx < n and 0 <= ny < n and 
                    (nx, ny) not in visited):
                    
                    if grid[nx][ny] == 1:  # Found second island
                        return dist
                    elif grid[nx][ny] == 0:  # Water
                        visited.add((nx, ny))
                        queue.append((nx, ny, dist + 1))
        
        return -1
    
    def shortestBridge_optimized_dfs_bfs(self, grid: List[List[int]]) -> int:
        """
        Approach 3: Optimized DFS + BFS with boundary tracking
        
        Optimize by tracking boundary during DFS phase.
        
        Time: O(n²), Space: O(n²)
        """
        n = len(grid)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        def dfs_with_boundary(i: int, j: int, boundary: Set[Tuple[int, int]]):
            """DFS that also tracks boundary cells"""
            if (i < 0 or i >= n or j < 0 or j >= n or 
                grid[i][j] != 1):
                return
            
            grid[i][j] = 2  # Mark as visited
            
            # Check if this cell is on the boundary
            is_boundary = False
            for di, dj in directions:
                ni, nj = i + di, j + dj
                if (0 <= ni < n and 0 <= nj < n and 
                    grid[ni][nj] == 0):
                    is_boundary = True
                    break
            
            if is_boundary:
                boundary.add((i, j))
            
            # Continue DFS
            for di, dj in directions:
                dfs_with_boundary(i + di, j + dj, boundary)
        
        # Find first island and its boundary
        boundary = set()
        found = False
        
        for i in range(n):
            if found:
                break
            for j in range(n):
                if grid[i][j] == 1:
                    dfs_with_boundary(i, j, boundary)
                    found = True
                    break
        
        # BFS from boundary to find shortest bridge
        queue = deque([(x, y, 0) for x, y in boundary])
        visited = set(boundary)
        
        while queue:
            x, y, dist = queue.popleft()
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                
                if (0 <= nx < n and 0 <= ny < n and 
                    (nx, ny) not in visited):
                    
                    if grid[nx][ny] == 1:  # Found second island
                        return dist
                    elif grid[nx][ny] == 0:  # Water
                        visited.add((nx, ny))
                        queue.append((nx, ny, dist + 1))
        
        return -1
    
    def shortestBridge_bidirectional_bfs(self, grid: List[List[int]]) -> int:
        """
        Approach 4: Bidirectional BFS
        
        Expand from both islands simultaneously until they meet.
        
        Time: O(n²), Space: O(n²)
        """
        n = len(grid)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        def get_island_cells(start_i: int, start_j: int, mark_value: int) -> Set[Tuple[int, int]]:
            """Get all cells of an island and mark them"""
            island = set()
            queue = deque([(start_i, start_j)])
            grid[start_i][start_j] = mark_value
            
            while queue:
                i, j = queue.popleft()
                island.add((i, j))
                
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    
                    if (0 <= ni < n and 0 <= nj < n and 
                        grid[ni][nj] == 1):
                        grid[ni][nj] = mark_value
                        queue.append((ni, nj))
            
            return island
        
        # Find both islands
        islands = []
        for i in range(n):
            for j in range(n):
                if grid[i][j] == 1:
                    island = get_island_cells(i, j, len(islands) + 2)
                    islands.append(island)
                    if len(islands) == 2:
                        break
            if len(islands) == 2:
                break
        
        # Bidirectional BFS
        queue1 = deque([(x, y, 0) for x, y in islands[0]])
        queue2 = deque([(x, y, 0) for x, y in islands[1]])
        
        visited1 = set(islands[0])
        visited2 = set(islands[1])
        
        while queue1 or queue2:
            # Expand from first island
            if queue1:
                for _ in range(len(queue1)):
                    x, y, dist = queue1.popleft()
                    
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        
                        if (0 <= nx < n and 0 <= ny < n and 
                            (nx, ny) not in visited1):
                            
                            if (nx, ny) in visited2:
                                return dist
                            elif grid[nx][ny] == 0:
                                visited1.add((nx, ny))
                                queue1.append((nx, ny, dist + 1))
            
            # Expand from second island
            if queue2:
                for _ in range(len(queue2)):
                    x, y, dist = queue2.popleft()
                    
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        
                        if (0 <= nx < n and 0 <= ny < n and 
                            (nx, ny) not in visited2):
                            
                            if (nx, ny) in visited1:
                                return dist
                            elif grid[nx][ny] == 0:
                                visited2.add((nx, ny))
                                queue2.append((nx, ny, dist + 1))
        
        return -1
    
    def shortestBridge_union_find_optimization(self, grid: List[List[int]]) -> int:
        """
        Approach 5: Union-Find for Island Identification + BFS
        
        Use Union-Find to identify islands, then BFS for shortest bridge.
        
        Time: O(n² α(n²)), Space: O(n²)
        """
        n = len(grid)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
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
        
        # Build connected components using Union-Find
        land_cells = []
        for i in range(n):
            for j in range(n):
                if grid[i][j] == 1:
                    land_cells.append((i, j))
                    
                    # Union with adjacent land cells
                    for di, dj in directions:
                        ni, nj = i + di, j + dj
                        if (0 <= ni < n and 0 <= nj < n and 
                            grid[ni][nj] == 1):
                            union((i, j), (ni, nj))
        
        # Group cells by island
        islands = {}
        for cell in land_cells:
            root = find(cell)
            if root not in islands:
                islands[root] = []
            islands[root].append(cell)
        
        # Get the two islands
        island_list = list(islands.values())
        island1, island2 = island_list[0], island_list[1]
        
        # BFS from first island to second
        queue = deque([(x, y, 0) for x, y in island1])
        visited = set(island1)
        
        while queue:
            x, y, dist = queue.popleft()
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                
                if (0 <= nx < n and 0 <= ny < n and 
                    (nx, ny) not in visited):
                    
                    if (nx, ny) in island2:
                        return dist - 1  # Subtract 1 since we don't count the destination
                    elif grid[nx][ny] == 0:
                        visited.add((nx, ny))
                        queue.append((nx, ny, dist + 1))
        
        return -1

def test_shortest_bridge():
    """Test shortest bridge algorithms"""
    solver = ShortestBridge()
    
    test_cases = [
        ([[0,1],[1,0]], 1, "Simple 2x2 case"),
        ([[0,1,0],[0,0,0],[0,0,1]], 2, "Diagonal islands"),
        ([[1,1,1,1,1],[1,0,0,0,1],[1,0,1,0,1],[1,0,0,0,1],[1,1,1,1,1]], 1, "Islands with hole"),
        ([[1,1,0,0,0],[1,0,0,0,0],[0,0,0,1,1],[0,0,0,1,1],[0,0,0,0,0]], 3, "Separated islands"),
    ]
    
    algorithms = [
        ("DFS + BFS", solver.shortestBridge_dfs_bfs),
        ("Two-pass BFS", solver.shortestBridge_two_pass_bfs),
        ("Optimized DFS+BFS", solver.shortestBridge_optimized_dfs_bfs),
        ("Bidirectional BFS", solver.shortestBridge_bidirectional_bfs),
    ]
    
    print("=== Testing Shortest Bridge ===")
    
    for grid, expected, description in test_cases:
        print(f"\n--- {description} (Expected: {expected}) ---")
        print(f"Grid: {grid}")
        
        for alg_name, alg_func in algorithms:
            try:
                # Create a copy since algorithms modify the grid
                grid_copy = [row[:] for row in grid]
                result = alg_func(grid_copy)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:18} | {status} | Bridge length: {result}")
            except Exception as e:
                print(f"{alg_name:18} | ERROR: {str(e)[:30]}")

if __name__ == "__main__":
    test_shortest_bridge()

"""
Shortest Bridge demonstrates advanced DFS and BFS techniques
for finding optimal connections between disconnected components
in grid-based problems with multiple algorithmic approaches.
"""
