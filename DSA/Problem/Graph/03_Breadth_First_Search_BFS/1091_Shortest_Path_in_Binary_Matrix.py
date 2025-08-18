"""
1091. Shortest Path in Binary Matrix
Difficulty: Easy

Problem:
Given an n x n binary matrix grid, return the length of the shortest clear path from 
top-left to bottom-right. If there is no clear path, return -1.

A clear path in a binary matrix is a path from the top-left cell (i.e., (0, 0)) to 
the bottom-right cell (i.e., (n - 1, n - 1)) such that:
- All the visited cells of the path are 0.
- All the adjacent cells of the path are 8-directionally connected (i.e., they are 
  different and they share an edge or a corner).

The length of a clear path is the number of visited cells of this path.

Examples:
Input: grid = [[0,0,0],[1,1,0],[1,1,0]]
Output: 4

Input: grid = [[0,1],[1,0]]
Output: -1

Input: grid = [[1,0,0],[1,1,0],[1,1,0]]
Output: -1

Constraints:
- n == grid.length
- n == grid[i].length
- 1 <= n <= 100
- grid[i][j] is 0 or 1
"""

from typing import List
from collections import deque
import heapq

class Solution:
    def shortestPathBinaryMatrix_approach1_standard_bfs(self, grid: List[List[int]]) -> int:
        """
        Approach 1: Standard BFS (Optimal)
        
        Use BFS with 8-directional movement to find shortest path.
        
        Time: O(N^2) - visit each cell at most once
        Space: O(N^2) - queue and visited set
        """
        n = len(grid)
        
        # Check if start or end is blocked
        if grid[0][0] == 1 or grid[n-1][n-1] == 1:
            return -1
        
        # Special case: single cell
        if n == 1:
            return 1
        
        # 8 directions: up, down, left, right, and 4 diagonals
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        queue = deque([(0, 0, 1)])  # (row, col, path_length)
        visited = {(0, 0)}
        
        while queue:
            row, col, path_len = queue.popleft()
            
            # Check if we reached the destination
            if row == n-1 and col == n-1:
                return path_len
            
            # Explore all 8 directions
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                
                if (0 <= new_row < n and 0 <= new_col < n and 
                    grid[new_row][new_col] == 0 and 
                    (new_row, new_col) not in visited):
                    
                    visited.add((new_row, new_col))
                    queue.append((new_row, new_col, path_len + 1))
        
        return -1
    
    def shortestPathBinaryMatrix_approach2_a_star(self, grid: List[List[int]]) -> int:
        """
        Approach 2: A* Algorithm
        
        Use A* with Manhattan distance heuristic for optimization.
        
        Time: O(N^2 log N) - priority queue operations
        Space: O(N^2)
        """
        n = len(grid)
        
        if grid[0][0] == 1 or grid[n-1][n-1] == 1:
            return -1
        
        if n == 1:
            return 1
        
        def heuristic(row, col):
            """Manhattan distance to destination"""
            return max(abs(row - (n-1)), abs(col - (n-1)))
        
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        # Priority queue: (f_score, g_score, row, col)
        pq = [(1 + heuristic(0, 0), 1, 0, 0)]
        visited = {(0, 0): 1}  # (row, col) -> g_score
        
        while pq:
            f_score, g_score, row, col = heapq.heappop(pq)
            
            # Skip if we've found a better path
            if (row, col) in visited and visited[(row, col)] < g_score:
                continue
            
            if row == n-1 and col == n-1:
                return g_score
            
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                
                if (0 <= new_row < n and 0 <= new_col < n and 
                    grid[new_row][new_col] == 0):
                    
                    new_g_score = g_score + 1
                    
                    if ((new_row, new_col) not in visited or 
                        visited[(new_row, new_col)] > new_g_score):
                        
                        visited[(new_row, new_col)] = new_g_score
                        f_score = new_g_score + heuristic(new_row, new_col)
                        heapq.heappush(pq, (f_score, new_g_score, new_row, new_col))
        
        return -1
    
    def shortestPathBinaryMatrix_approach3_bidirectional_bfs(self, grid: List[List[int]]) -> int:
        """
        Approach 3: Bidirectional BFS
        
        Search from both start and end simultaneously.
        
        Time: O(N^2)
        Space: O(N^2)
        """
        n = len(grid)
        
        if grid[0][0] == 1 or grid[n-1][n-1] == 1:
            return -1
        
        if n == 1:
            return 1
        
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        # Two BFS queues
        start_queue = deque([(0, 0)])
        end_queue = deque([(n-1, n-1)])
        
        start_visited = {(0, 0): 1}
        end_visited = {(n-1, n-1): 1}
        
        while start_queue or end_queue:
            # Expand from start
            if start_queue:
                for _ in range(len(start_queue)):
                    row, col = start_queue.popleft()
                    
                    for dr, dc in directions:
                        new_row, new_col = row + dr, col + dc
                        
                        if (0 <= new_row < n and 0 <= new_col < n and 
                            grid[new_row][new_col] == 0):
                            
                            if (new_row, new_col) in end_visited:
                                return start_visited[(row, col)] + end_visited[(new_row, new_col)]
                            
                            if (new_row, new_col) not in start_visited:
                                start_visited[(new_row, new_col)] = start_visited[(row, col)] + 1
                                start_queue.append((new_row, new_col))
            
            # Expand from end
            if end_queue:
                for _ in range(len(end_queue)):
                    row, col = end_queue.popleft()
                    
                    for dr, dc in directions:
                        new_row, new_col = row + dr, col + dc
                        
                        if (0 <= new_row < n and 0 <= new_col < n and 
                            grid[new_row][new_col] == 0):
                            
                            if (new_row, new_col) in start_visited:
                                return end_visited[(row, col)] + start_visited[(new_row, new_col)]
                            
                            if (new_row, new_col) not in end_visited:
                                end_visited[(new_row, new_col)] = end_visited[(row, col)] + 1
                                end_queue.append((new_row, new_col))
        
        return -1
    
    def shortestPathBinaryMatrix_approach4_dijkstra(self, grid: List[List[int]]) -> int:
        """
        Approach 4: Dijkstra's Algorithm
        
        Use Dijkstra for comparison (overkill for unweighted graph).
        
        Time: O(N^2 log N)
        Space: O(N^2)
        """
        n = len(grid)
        
        if grid[0][0] == 1 or grid[n-1][n-1] == 1:
            return -1
        
        if n == 1:
            return 1
        
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        # Priority queue: (distance, row, col)
        pq = [(1, 0, 0)]
        distances = {}
        distances[(0, 0)] = 1
        
        while pq:
            dist, row, col = heapq.heappop(pq)
            
            if row == n-1 and col == n-1:
                return dist
            
            if dist > distances.get((row, col), float('inf')):
                continue
            
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                
                if (0 <= new_row < n and 0 <= new_col < n and 
                    grid[new_row][new_col] == 0):
                    
                    new_dist = dist + 1
                    
                    if new_dist < distances.get((new_row, new_col), float('inf')):
                        distances[(new_row, new_col)] = new_dist
                        heapq.heappush(pq, (new_dist, new_row, new_col))
        
        return -1

def test_shortest_path_binary_matrix():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (grid, expected)
        ([[0,0,0],[1,1,0],[1,1,0]], 4),
        ([[0,1],[1,0]], -1),
        ([[1,0,0],[1,1,0],[1,1,0]], -1),
        ([[0]], 1),
        ([[0,0],[1,0]], 2),
        ([[0,0,0],[0,1,0],[0,0,0]], 4),
        ([[0,1,1,0,0,0],[0,1,0,1,1,0],[0,1,1,0,1,0],[0,0,0,1,1,0],[1,1,1,1,1,0],[0,0,0,0,0,0]], 14),
    ]
    
    approaches = [
        ("Standard BFS", solution.shortestPathBinaryMatrix_approach1_standard_bfs),
        ("A* Algorithm", solution.shortestPathBinaryMatrix_approach2_a_star),
        ("Bidirectional BFS", solution.shortestPathBinaryMatrix_approach3_bidirectional_bfs),
        ("Dijkstra", solution.shortestPathBinaryMatrix_approach4_dijkstra),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (grid, expected) in enumerate(test_cases):
            # Create deep copy for testing
            test_grid = [row[:] for row in grid]
            result = func(test_grid)
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} Expected: {expected}, Got: {result}")

def demonstrate_8_directional_movement():
    """Demonstrate 8-directional movement pathfinding"""
    print("\n=== 8-Directional Movement Demo ===")
    
    grid = [[0,0,0],
            [1,1,0],
            [1,1,0]]
    
    print("Grid (0=clear, 1=blocked):")
    print_grid_with_path(grid)
    
    print("\nBFS exploration with 8-directional movement:")
    print("Directions: ↖ ↑ ↗ ← → ↙ ↓ ↘")
    
    n = len(grid)
    directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    direction_symbols = ["↖", "↑", "↗", "←", "→", "↙", "↓", "↘"]
    
    queue = deque([(0, 0, 1, [(0, 0)])])
    visited = {(0, 0)}
    
    while queue:
        row, col, path_len, path = queue.popleft()
        
        print(f"\nStep {path_len}: At ({row},{col}), Path: {path}")
        
        if row == n-1 and col == n-1:
            print(f"🎯 Destination reached! Path length: {path_len}")
            print(f"Complete path: {path}")
            break
        
        valid_moves = []
        for i, (dr, dc) in enumerate(directions):
            new_row, new_col = row + dr, col + dc
            
            if (0 <= new_row < n and 0 <= new_col < n and 
                grid[new_row][new_col] == 0 and 
                (new_row, new_col) not in visited):
                
                valid_moves.append((direction_symbols[i], new_row, new_col))
                visited.add((new_row, new_col))
                queue.append((new_row, new_col, path_len + 1, path + [(new_row, new_col)]))
        
        if valid_moves:
            print(f"  Valid moves: {valid_moves}")

def print_grid_with_path(grid):
    """Print grid with visual representation"""
    symbols = {0: "⬜", 1: "⬛"}
    n = len(grid)
    
    for i, row in enumerate(grid):
        line = ""
        for j, cell in enumerate(row):
            if i == 0 and j == 0:
                line += "🟢"  # Start
            elif i == n-1 and j == n-1:
                line += "🔴"  # End
            else:
                line += symbols[cell]
        print(f"  {line}")
    
    print("  Legend: 🟢 = Start, 🔴 = End, ⬜ = Clear, ⬛ = Blocked")

def compare_pathfinding_algorithms():
    """Compare different pathfinding algorithms"""
    print("\n=== Pathfinding Algorithm Comparison ===")
    
    print("1. Standard BFS:")
    print("   ✅ Optimal for unweighted graphs")
    print("   ✅ Simple implementation")
    print("   ✅ Guaranteed shortest path")
    print("   ❌ Explores uniformly in all directions")
    print("   • Time: O(N^2), Space: O(N^2)")
    
    print("\n2. A* Algorithm:")
    print("   ✅ Guided search with heuristic")
    print("   ✅ Often faster than BFS in practice")
    print("   ✅ Optimal with admissible heuristic")
    print("   ❌ More complex implementation")
    print("   • Time: O(N^2 log N), Space: O(N^2)")
    
    print("\n3. Bidirectional BFS:")
    print("   ✅ Reduces search space")
    print("   ✅ Especially good for long paths")
    print("   ✅ Optimal shortest path")
    print("   ❌ More complex bookkeeping")
    print("   • Time: O(N^2), Space: O(N^2)")
    
    print("\n4. Dijkstra's Algorithm:")
    print("   ✅ Handles weighted graphs")
    print("   ✅ Guaranteed optimal")
    print("   ❌ Overkill for unweighted graphs")
    print("   ❌ Slower due to priority queue")
    print("   • Time: O(N^2 log N), Space: O(N^2)")
    
    print("\nWhen to use each:")
    print("• BFS: Standard choice for unweighted grids")
    print("• A*: When you have good heuristic and need speed")
    print("• Bidirectional BFS: For long paths in large grids")
    print("• Dijkstra: When weights matter or graph structure unknown")

if __name__ == "__main__":
    test_shortest_path_binary_matrix()
    demonstrate_8_directional_movement()
    compare_pathfinding_algorithms()

"""
Graph Theory Concepts:
1. Shortest Path in Unweighted Grid
2. 8-Directional Movement
3. Multiple Pathfinding Algorithms
4. Algorithm Performance Comparison

Key Pathfinding Insights:
- 8-directional movement: Includes diagonal moves
- BFS optimal for unweighted graphs
- A* uses heuristic for guided search
- Bidirectional BFS reduces search space

Algorithm Variations:
- Standard BFS: Uniform exploration
- A*: Heuristic-guided search
- Bidirectional: Search from both ends
- Dijkstra: Handles weighted scenarios

Real-world Applications:
- Robot navigation in 2D spaces
- Game AI pathfinding
- GPS routing systems
- Autonomous vehicle navigation
- Drone flight path planning
- Grid-based game movement

This problem demonstrates the evolution from basic BFS
to sophisticated pathfinding algorithms.
"""
