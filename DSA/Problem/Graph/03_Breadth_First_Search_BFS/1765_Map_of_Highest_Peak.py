"""
1765. Map of Highest Peak - Multiple Approaches
Difficulty: Medium

You are given an integer matrix isWater of size m x n that represents a map of land and water cells.

If isWater[i][j] == 0, cell (i, j) is a land cell.
If isWater[i][j] == 1, cell (i, j) is a water cell.

You must assign each cell a height in a way that follows these rules:

The height of each cell must be non-negative.
If the cell is a water cell, its height must be 0.
For any two adjacent cells, the absolute difference of their heights must be at most 1.

Return an integer matrix height representing the map of heights after assigning each cell a height. If there are multiple solutions, return any of them.
"""

from typing import List
from collections import deque

class MapOfHighestPeak:
    """Multiple approaches to assign heights with constraints"""
    
    def highestPeak_multi_source_bfs(self, isWater: List[List[int]]) -> List[List[int]]:
        """
        Approach 1: Multi-source BFS
        
        Start BFS from all water cells simultaneously.
        
        Time: O(mn), Space: O(mn)
        """
        m, n = len(isWater), len(isWater[0])
        heights = [[-1] * n for _ in range(m)]
        queue = deque()
        
        # Initialize water cells and add to queue
        for i in range(m):
            for j in range(n):
                if isWater[i][j] == 1:
                    heights[i][j] = 0
                    queue.append((i, j))
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        while queue:
            row, col = queue.popleft()
            current_height = heights[row][col]
            
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                
                if (0 <= new_row < m and 0 <= new_col < n and 
                    heights[new_row][new_col] == -1):
                    heights[new_row][new_col] = current_height + 1
                    queue.append((new_row, new_col))
        
        return heights
    
    def highestPeak_level_by_level_bfs(self, isWater: List[List[int]]) -> List[List[int]]:
        """
        Approach 2: Level-by-level BFS
        
        Process each distance level separately.
        
        Time: O(mn), Space: O(mn)
        """
        m, n = len(isWater), len(isWater[0])
        heights = [[-1] * n for _ in range(m)]
        
        # Find all water cells
        current_level = []
        for i in range(m):
            for j in range(n):
                if isWater[i][j] == 1:
                    heights[i][j] = 0
                    current_level.append((i, j))
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        height = 0
        
        while current_level:
            next_level = []
            
            for row, col in current_level:
                for dr, dc in directions:
                    new_row, new_col = row + dr, col + dc
                    
                    if (0 <= new_row < m and 0 <= new_col < n and 
                        heights[new_row][new_col] == -1):
                        heights[new_row][new_col] = height + 1
                        next_level.append((new_row, new_col))
            
            current_level = next_level
            height += 1
        
        return heights
    
    def highestPeak_optimized_bfs(self, isWater: List[List[int]]) -> List[List[int]]:
        """
        Approach 3: Optimized BFS with Early Termination
        
        Optimize BFS with better memory usage and early termination.
        
        Time: O(mn), Space: O(mn)
        """
        m, n = len(isWater), len(isWater[0])
        heights = [[float('inf')] * n for _ in range(m)]
        queue = deque()
        
        # Initialize water cells
        for i in range(m):
            for j in range(n):
                if isWater[i][j] == 1:
                    heights[i][j] = 0
                    queue.append((i, j, 0))
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        while queue:
            row, col, height = queue.popleft()
            
            # Skip if we've found a better path
            if height > heights[row][col]:
                continue
            
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                new_height = height + 1
                
                if (0 <= new_row < m and 0 <= new_col < n and 
                    new_height < heights[new_row][new_col]):
                    heights[new_row][new_col] = new_height
                    queue.append((new_row, new_col, new_height))
        
        return heights
    
    def highestPeak_dp_approach(self, isWater: List[List[int]]) -> List[List[int]]:
        """
        Approach 4: Dynamic Programming Approach
        
        Use DP with multiple passes to compute minimum distances.
        
        Time: O(mn), Space: O(mn)
        """
        m, n = len(isWater), len(isWater[0])
        heights = [[float('inf')] * n for _ in range(m)]
        
        # Initialize water cells
        for i in range(m):
            for j in range(n):
                if isWater[i][j] == 1:
                    heights[i][j] = 0
        
        # Forward pass (top-left to bottom-right)
        for i in range(m):
            for j in range(n):
                if heights[i][j] != 0:  # Not a water cell
                    if i > 0:
                        heights[i][j] = min(heights[i][j], heights[i-1][j] + 1)
                    if j > 0:
                        heights[i][j] = min(heights[i][j], heights[i][j-1] + 1)
        
        # Backward pass (bottom-right to top-left)
        for i in range(m-1, -1, -1):
            for j in range(n-1, -1, -1):
                if heights[i][j] != 0:  # Not a water cell
                    if i < m-1:
                        heights[i][j] = min(heights[i][j], heights[i+1][j] + 1)
                    if j < n-1:
                        heights[i][j] = min(heights[i][j], heights[i][j+1] + 1)
        
        return heights
    
    def highestPeak_dijkstra_like(self, isWater: List[List[int]]) -> List[List[int]]:
        """
        Approach 5: Dijkstra-like Algorithm
        
        Use priority queue for optimal path finding.
        
        Time: O(mn log(mn)), Space: O(mn)
        """
        import heapq
        
        m, n = len(isWater), len(isWater[0])
        heights = [[float('inf')] * n for _ in range(m)]
        pq = []
        
        # Initialize water cells
        for i in range(m):
            for j in range(n):
                if isWater[i][j] == 1:
                    heights[i][j] = 0
                    heapq.heappush(pq, (0, i, j))
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        while pq:
            height, row, col = heapq.heappop(pq)
            
            # Skip if we've found a better path
            if height > heights[row][col]:
                continue
            
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                new_height = height + 1
                
                if (0 <= new_row < m and 0 <= new_col < n and 
                    new_height < heights[new_row][new_col]):
                    heights[new_row][new_col] = new_height
                    heapq.heappush(pq, (new_height, new_row, new_col))
        
        return heights

def test_map_of_highest_peak():
    """Test map of highest peak algorithms"""
    solver = MapOfHighestPeak()
    
    test_cases = [
        ([[0,1],[0,0]], "Simple 2x2 with one water cell"),
        ([[0,0,1],[1,0,0],[0,0,0]], "3x3 with two water cells"),
        ([[1,1],[1,1]], "All water cells"),
        ([[0,0,0],[0,1,0],[0,0,0]], "Water in center"),
    ]
    
    algorithms = [
        ("Multi-source BFS", solver.highestPeak_multi_source_bfs),
        ("Level-by-level BFS", solver.highestPeak_level_by_level_bfs),
        ("Optimized BFS", solver.highestPeak_optimized_bfs),
        ("DP Approach", solver.highestPeak_dp_approach),
    ]
    
    print("=== Testing Map of Highest Peak ===")
    
    for isWater, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Input: {isWater}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(isWater)
                print(f"{alg_name:18} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:18} | ERROR: {str(e)[:30]}")

if __name__ == "__main__":
    test_map_of_highest_peak()

"""
Map of Highest Peak demonstrates multi-source BFS
and distance propagation techniques for constraint
satisfaction problems in grid-based scenarios.
"""
