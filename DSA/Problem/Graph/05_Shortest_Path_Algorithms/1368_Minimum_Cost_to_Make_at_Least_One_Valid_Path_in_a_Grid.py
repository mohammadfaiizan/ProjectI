"""
1368. Minimum Cost to Make at Least One Valid Path in a Grid - Multiple Approaches
Difficulty: Medium

Given an m x n grid. Each cell of the grid has a sign pointing to the next cell you should visit if you are currently in this cell. The sign of grid[i][j] can be:

1 which means go to the cell to the right. (i.e go from grid[i][j] to grid[i][j + 1])
2 which means go to the cell to the left. (i.e go from grid[i][j] to grid[i][j - 1])
3 which means go to the cell below. (i.e go from grid[i][j] to grid[i + 1][j])
4 which means go to the cell above. (i.e go from grid[i][j] to grid[i - 1][j])

Notice that there could be some signs on the grid at the same position which points outside of the grid.

You will initially start at the upper left cell (0, 0). A valid path in the grid is a path that starts from the upper left cell (0, 0) and ends at the bottom-right cell (m - 1, n - 1) following the signs on the grid. The valid path does not have to be the shortest.

You can modify the sign on a grid cell, and your goal is to make the grid have at least one valid path from the upper left cell to the bottom-right cell.

Return the minimum number of times you need to change the sign on a grid cell to make the grid have at least one valid path.
"""

from typing import List, Tuple, Deque
from collections import deque
import heapq

class MinimumCostValidPath:
    """Multiple approaches to find minimum cost to make valid path"""
    
    def minCost_dijkstra(self, grid: List[List[int]]) -> int:
        """
        Approach 1: Dijkstra's Algorithm
        
        Use Dijkstra to find minimum cost path where following direction costs 0,
        changing direction costs 1.
        
        Time: O(mn log(mn)), Space: O(mn)
        """
        m, n = len(grid), len(grid[0])
        
        # Direction mappings: right, left, down, up
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        # Priority queue: (cost, row, col)
        pq = [(0, 0, 0)]
        visited = set()
        
        while pq:
            cost, row, col = heapq.heappop(pq)
            
            if (row, col) in visited:
                continue
            
            visited.add((row, col))
            
            if row == m - 1 and col == n - 1:
                return cost
            
            # Try all 4 directions
            for i, (dr, dc) in enumerate(directions):
                new_row, new_col = row + dr, col + dc
                
                if 0 <= new_row < m and 0 <= new_col < n:
                    if (new_row, new_col) not in visited:
                        # Cost is 0 if following current direction, 1 if changing
                        new_cost = cost + (0 if grid[row][col] == i + 1 else 1)
                        heapq.heappush(pq, (new_cost, new_row, new_col))
        
        return -1  # Should not reach here
    
    def minCost_0_1_bfs(self, grid: List[List[int]]) -> int:
        """
        Approach 2: 0-1 BFS
        
        Use 0-1 BFS where edges have weight 0 or 1.
        
        Time: O(mn), Space: O(mn)
        """
        m, n = len(grid), len(grid[0])
        
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        # Deque for 0-1 BFS
        dq = deque([(0, 0, 0)])  # (cost, row, col)
        visited = {}
        
        while dq:
            cost, row, col = dq.popleft()
            
            if (row, col) in visited:
                continue
            
            visited[(row, col)] = cost
            
            if row == m - 1 and col == n - 1:
                return cost
            
            # Try all 4 directions
            for i, (dr, dc) in enumerate(directions):
                new_row, new_col = row + dr, col + dc
                
                if 0 <= new_row < m and 0 <= new_col < n:
                    if (new_row, new_col) not in visited:
                        new_cost = cost + (0 if grid[row][col] == i + 1 else 1)
                        
                        # Add to front if cost is 0, back if cost is 1
                        if grid[row][col] == i + 1:
                            dq.appendleft((new_cost, new_row, new_col))
                        else:
                            dq.append((new_cost, new_row, new_col))
        
        return -1
    
    def minCost_bfs_layers(self, grid: List[List[int]]) -> int:
        """
        Approach 3: BFS with Cost Layers
        
        Process nodes layer by layer based on cost.
        
        Time: O(mn), Space: O(mn)
        """
        m, n = len(grid), len(grid[0])
        
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        # Start with cost 0 nodes
        current_layer = [(0, 0)]
        visited = set()
        cost = 0
        
        while current_layer:
            next_layer = []
            
            for row, col in current_layer:
                if (row, col) in visited:
                    continue
                
                visited.add((row, col))
                
                if row == m - 1 and col == n - 1:
                    return cost
                
                # Follow the current direction with cost 0
                direction = grid[row][col] - 1
                dr, dc = directions[direction]
                new_row, new_col = row + dr, col + dc
                
                if (0 <= new_row < m and 0 <= new_col < n and 
                    (new_row, new_col) not in visited):
                    current_layer.append((new_row, new_col))
                
                # Try other directions with cost 1
                for i, (dr, dc) in enumerate(directions):
                    if i != direction:
                        new_row, new_col = row + dr, col + dc
                        
                        if (0 <= new_row < m and 0 <= new_col < n and 
                            (new_row, new_col) not in visited):
                            next_layer.append((new_row, new_col))
            
            if not current_layer:
                current_layer = next_layer
                cost += 1
        
        return -1
    
    def minCost_dfs_memoization(self, grid: List[List[int]]) -> int:
        """
        Approach 4: DFS with Memoization
        
        Use DFS with memoization to explore paths.
        
        Time: O(mn), Space: O(mn)
        """
        m, n = len(grid), len(grid[0])
        
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        memo = {}
        
        def dfs(row: int, col: int) -> int:
            if row == m - 1 and col == n - 1:
                return 0
            
            if (row, col) in memo:
                return memo[(row, col)]
            
            min_cost = float('inf')
            
            # Try all 4 directions
            for i, (dr, dc) in enumerate(directions):
                new_row, new_col = row + dr, col + dc
                
                if 0 <= new_row < m and 0 <= new_col < n:
                    # Cost is 0 if following current direction, 1 if changing
                    cost = 0 if grid[row][col] == i + 1 else 1
                    min_cost = min(min_cost, cost + dfs(new_row, new_col))
            
            memo[(row, col)] = min_cost
            return min_cost
        
        return dfs(0, 0)
    
    def minCost_union_find_optimization(self, grid: List[List[int]]) -> int:
        """
        Approach 5: Union-Find with Path Following
        
        Use Union-Find to group cells reachable with cost 0.
        
        Time: O(mn α(mn)), Space: O(mn)
        """
        m, n = len(grid), len(grid[0])
        
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        # Union-Find
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
        
        # Connect cells reachable with cost 0
        for i in range(m):
            for j in range(n):
                direction = grid[i][j] - 1
                dr, dc = directions[direction]
                new_i, new_j = i + dr, j + dc
                
                if 0 <= new_i < m and 0 <= new_j < n:
                    union((i, j), (new_i, new_j))
        
        # If start and end are already connected, cost is 0
        if find((0, 0)) == find((m - 1, n - 1)):
            return 0
        
        # Otherwise, use BFS to find minimum changes needed
        visited = set()
        queue = deque([(0, 0, 0)])  # (row, col, cost)
        
        while queue:
            row, col, cost = queue.popleft()
            
            if (row, col) in visited:
                continue
            
            visited.add((row, col))
            
            if row == m - 1 and col == n - 1:
                return cost
            
            # Try changing direction (cost + 1)
            for i, (dr, dc) in enumerate(directions):
                if i != grid[row][col] - 1:  # Different from current direction
                    new_row, new_col = row + dr, col + dc
                    
                    if (0 <= new_row < m and 0 <= new_col < n and 
                        (new_row, new_col) not in visited):
                        queue.append((new_row, new_col, cost + 1))
        
        return -1

def test_minimum_cost_valid_path():
    """Test minimum cost valid path algorithms"""
    solver = MinimumCostValidPath()
    
    test_cases = [
        ([[1,1,1,1],[2,2,2,2],[1,1,1,1],[2,2,2,2]], 3, "Alternating pattern"),
        ([[1,1,3],[3,2,2],[1,1,1]], 0, "Already valid path"),
        ([[1,2],[4,3]], 1, "Small grid"),
        ([[2,2,2],[2,2,2]], 3, "All left directions"),
    ]
    
    algorithms = [
        ("Dijkstra", solver.minCost_dijkstra),
        ("0-1 BFS", solver.minCost_0_1_bfs),
        ("BFS Layers", solver.minCost_bfs_layers),
        ("DFS Memoization", solver.minCost_dfs_memoization),
    ]
    
    print("=== Testing Minimum Cost Valid Path ===")
    
    for grid, expected, description in test_cases:
        print(f"\n--- {description} (Expected: {expected}) ---")
        print(f"Grid: {grid}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(grid)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:15} | {status} | Cost: {result}")
            except Exception as e:
                print(f"{alg_name:15} | ERROR: {str(e)[:30]}")

if __name__ == "__main__":
    test_minimum_cost_valid_path()

"""
Minimum Cost Valid Path demonstrates advanced shortest path techniques
in grid-based problems with direction constraints and modification costs.
"""
