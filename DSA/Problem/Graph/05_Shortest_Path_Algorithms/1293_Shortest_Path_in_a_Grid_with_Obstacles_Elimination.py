"""
1293. Shortest Path in a Grid with Obstacles Elimination - Multiple Approaches
Difficulty: Medium

You are given an m x n integer matrix grid where each cell is either 0 (empty) or 1 (obstacle). You can move up, down, left, or right from and to an empty cell in one step.

Return the minimum number of steps to walk from the upper left corner (0, 0) to the lower right corner (m - 1, n - 1) given that you can eliminate at most k obstacles. If it is not possible to find such walk return -1.
"""

from typing import List, Tuple, Set
from collections import deque
import heapq

class ShortestPathObstacleElimination:
    """Multiple approaches for shortest path with obstacle elimination"""
    
    def shortestPath_bfs_3d_state(self, grid: List[List[int]], k: int) -> int:
        """
        Approach 1: BFS with 3D State Space
        
        Use BFS with state (row, col, obstacles_eliminated).
        
        Time: O(mnk), Space: O(mnk)
        """
        m, n = len(grid), len(grid[0])
        
        if k >= m + n - 2:  # Can eliminate all obstacles on shortest path
            return m + n - 2
        
        # BFS with state (row, col, obstacles_used, steps)
        queue = deque([(0, 0, 0, 0)])  # (row, col, obstacles_used, steps)
        visited = set([(0, 0, 0)])  # (row, col, obstacles_used)
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        while queue:
            row, col, obstacles_used, steps = queue.popleft()
            
            if row == m - 1 and col == n - 1:
                return steps
            
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                
                if 0 <= new_row < m and 0 <= new_col < n:
                    new_obstacles = obstacles_used + grid[new_row][new_col]
                    
                    if (new_obstacles <= k and 
                        (new_row, new_col, new_obstacles) not in visited):
                        visited.add((new_row, new_col, new_obstacles))
                        queue.append((new_row, new_col, new_obstacles, steps + 1))
        
        return -1
    
    def shortestPath_dijkstra(self, grid: List[List[int]], k: int) -> int:
        """
        Approach 2: Dijkstra's Algorithm
        
        Use Dijkstra with priority on steps, then obstacles.
        
        Time: O(mnk log(mnk)), Space: O(mnk)
        """
        m, n = len(grid), len(grid[0])
        
        if k >= m + n - 2:
            return m + n - 2
        
        # Priority queue: (steps, obstacles_used, row, col)
        pq = [(0, 0, 0, 0)]
        visited = {}  # (row, col) -> min_obstacles_used
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        while pq:
            steps, obstacles_used, row, col = heapq.heappop(pq)
            
            if row == m - 1 and col == n - 1:
                return steps
            
            # Skip if we've seen this position with fewer obstacles
            if (row, col) in visited and visited[(row, col)] <= obstacles_used:
                continue
            
            visited[(row, col)] = obstacles_used
            
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                
                if 0 <= new_row < m and 0 <= new_col < n:
                    new_obstacles = obstacles_used + grid[new_row][new_col]
                    
                    if new_obstacles <= k:
                        heapq.heappush(pq, (steps + 1, new_obstacles, new_row, new_col))
        
        return -1
    
    def shortestPath_a_star(self, grid: List[List[int]], k: int) -> int:
        """
        Approach 3: A* Algorithm
        
        Use A* with Manhattan distance heuristic.
        
        Time: O(mnk log(mnk)), Space: O(mnk)
        """
        m, n = len(grid), len(grid[0])
        
        if k >= m + n - 2:
            return m + n - 2
        
        def manhattan_distance(row: int, col: int) -> int:
            return (m - 1 - row) + (n - 1 - col)
        
        # Priority queue: (f_score, steps, obstacles_used, row, col)
        pq = [(manhattan_distance(0, 0), 0, 0, 0, 0)]
        visited = set()
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        while pq:
            f_score, steps, obstacles_used, row, col = heapq.heappop(pq)
            
            if row == m - 1 and col == n - 1:
                return steps
            
            state = (row, col, obstacles_used)
            if state in visited:
                continue
            
            visited.add(state)
            
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                
                if 0 <= new_row < m and 0 <= new_col < n:
                    new_obstacles = obstacles_used + grid[new_row][new_col]
                    new_steps = steps + 1
                    
                    if new_obstacles <= k:
                        new_state = (new_row, new_col, new_obstacles)
                        if new_state not in visited:
                            f_score = new_steps + manhattan_distance(new_row, new_col)
                            heapq.heappush(pq, (f_score, new_steps, new_obstacles, new_row, new_col))
        
        return -1
    
    def shortestPath_bidirectional_bfs(self, grid: List[List[int]], k: int) -> int:
        """
        Approach 4: Bidirectional BFS
        
        Search from both start and end simultaneously.
        
        Time: O(mnk), Space: O(mnk)
        """
        m, n = len(grid), len(grid[0])
        
        if k >= m + n - 2:
            return m + n - 2
        
        if m == 1 and n == 1:
            return 0
        
        # Forward and backward search
        forward_queue = deque([(0, 0, 0, 0)])  # (row, col, obstacles, steps)
        backward_queue = deque([(m-1, n-1, 0, 0)])
        
        forward_visited = {(0, 0, 0): 0}
        backward_visited = {(m-1, n-1, 0): 0}
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        while forward_queue or backward_queue:
            # Forward search
            if forward_queue:
                for _ in range(len(forward_queue)):
                    row, col, obstacles, steps = forward_queue.popleft()
                    
                    for dr, dc in directions:
                        new_row, new_col = row + dr, col + dc
                        
                        if 0 <= new_row < m and 0 <= new_col < n:
                            new_obstacles = obstacles + grid[new_row][new_col]
                            
                            if new_obstacles <= k:
                                state = (new_row, new_col, new_obstacles)
                                
                                # Check if met backward search
                                for back_obstacles in range(k + 1):
                                    back_state = (new_row, new_col, back_obstacles)
                                    if (back_state in backward_visited and 
                                        new_obstacles + back_obstacles <= k):
                                        return steps + 1 + backward_visited[back_state]
                                
                                if state not in forward_visited:
                                    forward_visited[state] = steps + 1
                                    forward_queue.append((new_row, new_col, new_obstacles, steps + 1))
            
            # Backward search
            if backward_queue:
                for _ in range(len(backward_queue)):
                    row, col, obstacles, steps = backward_queue.popleft()
                    
                    for dr, dc in directions:
                        new_row, new_col = row + dr, col + dc
                        
                        if 0 <= new_row < m and 0 <= new_col < n:
                            new_obstacles = obstacles + grid[new_row][new_col]
                            
                            if new_obstacles <= k:
                                state = (new_row, new_col, new_obstacles)
                                
                                # Check if met forward search
                                for forward_obstacles in range(k + 1):
                                    forward_state = (new_row, new_col, forward_obstacles)
                                    if (forward_state in forward_visited and 
                                        new_obstacles + forward_obstacles <= k):
                                        return steps + 1 + forward_visited[forward_state]
                                
                                if state not in backward_visited:
                                    backward_visited[state] = steps + 1
                                    backward_queue.append((new_row, new_col, new_obstacles, steps + 1))
        
        return -1
    
    def shortestPath_optimized_bfs(self, grid: List[List[int]], k: int) -> int:
        """
        Approach 5: Optimized BFS with State Pruning
        
        Use BFS with optimized state representation.
        
        Time: O(mnk), Space: O(mnk)
        """
        m, n = len(grid), len(grid[0])
        
        if k >= m + n - 2:
            return m + n - 2
        
        # Use array instead of set for better performance
        visited = [[[False] * (k + 1) for _ in range(n)] for _ in range(m)]
        visited[0][0][0] = True
        
        queue = deque([(0, 0, 0, 0)])  # (row, col, obstacles, steps)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        while queue:
            row, col, obstacles, steps = queue.popleft()
            
            if row == m - 1 and col == n - 1:
                return steps
            
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                
                if 0 <= new_row < m and 0 <= new_col < n:
                    new_obstacles = obstacles + grid[new_row][new_col]
                    
                    if (new_obstacles <= k and 
                        not visited[new_row][new_col][new_obstacles]):
                        visited[new_row][new_col][new_obstacles] = True
                        queue.append((new_row, new_col, new_obstacles, steps + 1))
        
        return -1

def test_shortest_path_obstacles():
    """Test shortest path with obstacle elimination"""
    solver = ShortestPathObstacleElimination()
    
    test_cases = [
        ([[0,0,0],[1,1,0],[0,0,0],[0,1,1],[0,0,0]], 1, 6, "Example 1"),
        ([[0,1,1],[1,1,1],[1,0,0]], 1, -1, "Impossible case"),
        ([[0,0,0],[1,1,0],[0,0,0]], 0, 4, "No elimination allowed"),
        ([[0]], 0, 0, "Single cell"),
    ]
    
    algorithms = [
        ("BFS 3D State", solver.shortestPath_bfs_3d_state),
        ("Dijkstra", solver.shortestPath_dijkstra),
        ("A* Algorithm", solver.shortestPath_a_star),
        ("Optimized BFS", solver.shortestPath_optimized_bfs),
    ]
    
    print("=== Testing Shortest Path with Obstacle Elimination ===")
    
    for grid, k, expected, description in test_cases:
        print(f"\n--- {description} (Expected: {expected}) ---")
        print(f"Grid: {grid}, k={k}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(grid, k)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:15} | {status} | Steps: {result}")
            except Exception as e:
                print(f"{alg_name:15} | ERROR: {str(e)[:30]}")

if __name__ == "__main__":
    test_shortest_path_obstacles()

"""
Shortest Path with Obstacle Elimination demonstrates advanced BFS
and shortest path techniques with state space expansion for
constraint-based pathfinding problems.
"""
