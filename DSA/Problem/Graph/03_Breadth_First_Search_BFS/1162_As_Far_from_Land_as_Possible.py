"""
1162. As Far from Land as Possible
Difficulty: Medium

Problem:
Given an n x n grid containing only values 0 and 1, where 0 represents water and 1 
represents land, find a water cell such that its distance to the nearest land cell 
is maximized, and return the distance. If no land or water exists in the grid, 
return -1.

The distance used in this problem is the Manhattan distance: |x1 - x2| + |y1 - y2|.

Examples:
Input: grid = [[1,0,1],[0,0,0],[1,0,1]]
Output: 2

Input: grid = [[1,0,0],[0,0,0],[0,0,0]]
Output: 4

Input: grid = [[0,0,0],[0,0,0],[0,0,0]]
Output: -1

Constraints:
- n == grid.length
- n == grid[i].length
- 1 <= n <= 100
- grid[i][j] is 0 or 1
"""

from typing import List
from collections import deque

class Solution:
    def maxDistance_approach1_multi_source_bfs(self, grid: List[List[int]]) -> int:
        """
        Approach 1: Multi-Source BFS (Optimal)
        
        Start BFS from all land cells, find maximum distance reached.
        
        Time: O(N^2) - visit each cell at most once
        Space: O(N^2) - queue + distance matrix
        """
        if not grid or not grid[0]:
            return -1
        
        n = len(grid)
        queue = deque()
        distances = [[-1] * n for _ in range(n)]
        
        # Initialize: add all land cells to queue
        land_count = 0
        for i in range(n):
            for j in range(n):
                if grid[i][j] == 1:
                    queue.append((i, j, 0))
                    distances[i][j] = 0
                    land_count += 1
        
        # Edge cases
        if land_count == 0 or land_count == n * n:
            return -1
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        max_distance = 0
        
        # Multi-source BFS
        while queue:
            i, j, dist = queue.popleft()
            
            for di, dj in directions:
                ni, nj = i + di, j + dj
                
                if (0 <= ni < n and 0 <= nj < n and distances[ni][nj] == -1):
                    distances[ni][nj] = dist + 1
                    max_distance = max(max_distance, dist + 1)
                    queue.append((ni, nj, dist + 1))
        
        return max_distance
    
    def maxDistance_approach2_level_by_level_bfs(self, grid: List[List[int]]) -> int:
        """
        Approach 2: Level-by-Level BFS
        
        Process all cells at current distance before next distance.
        
        Time: O(N^2)
        Space: O(N^2)
        """
        if not grid or not grid[0]:
            return -1
        
        n = len(grid)
        queue = deque()
        
        # Add all land cells to queue
        for i in range(n):
            for j in range(n):
                if grid[i][j] == 1:
                    queue.append((i, j))
        
        # Edge cases
        if len(queue) == 0 or len(queue) == n * n:
            return -1
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        distance = 0
        
        # Level-by-level BFS
        while queue:
            distance += 1
            size = len(queue)
            
            for _ in range(size):
                i, j = queue.popleft()
                
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    
                    if (0 <= ni < n and 0 <= nj < n and grid[ni][nj] == 0):
                        grid[ni][nj] = 1  # Mark as visited
                        queue.append((ni, nj))
        
        return distance - 1  # Last level was empty, so distance - 1
    
    def maxDistance_approach3_binary_search_bfs(self, grid: List[List[int]]) -> int:
        """
        Approach 3: Binary Search + BFS
        
        Binary search on answer, use BFS to check if distance is achievable.
        
        Time: O(N^2 * log(N))
        Space: O(N^2)
        """
        if not grid or not grid[0]:
            return -1
        
        n = len(grid)
        
        # Find all land cells
        land_cells = []
        water_count = 0
        for i in range(n):
            for j in range(n):
                if grid[i][j] == 1:
                    land_cells.append((i, j))
                else:
                    water_count += 1
        
        if len(land_cells) == 0 or water_count == 0:
            return -1
        
        def can_achieve_distance(target_dist):
            """Check if there exists water cell with distance >= target_dist"""
            distances = [[-1] * n for _ in range(n)]
            queue = deque()
            
            # Initialize with land cells
            for i, j in land_cells:
                distances[i][j] = 0
                queue.append((i, j))
            
            # BFS to calculate distances
            while queue:
                i, j = queue.popleft()
                
                if distances[i][j] >= target_dist:
                    continue  # No need to expand further
                
                for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    ni, nj = i + di, j + dj
                    
                    if (0 <= ni < n and 0 <= nj < n and distances[ni][nj] == -1):
                        distances[ni][nj] = distances[i][j] + 1
                        queue.append((ni, nj))
            
            # Check if any water cell has distance >= target_dist
            for i in range(n):
                for j in range(n):
                    if grid[i][j] == 0 and distances[i][j] >= target_dist:
                        return True
            return False
        
        # Binary search on answer
        left, right = 1, n + n
        result = -1
        
        while left <= right:
            mid = (left + right) // 2
            if can_achieve_distance(mid):
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        
        return result
    
    def maxDistance_approach4_dp_distance_transform(self, grid: List[List[int]]) -> int:
        """
        Approach 4: DP Distance Transform
        
        Use dynamic programming to compute distance transform.
        
        Time: O(N^2)
        Space: O(N^2)
        """
        if not grid or not grid[0]:
            return -1
        
        n = len(grid)
        distances = [[float('inf')] * n for _ in range(n)]
        
        # Initialize land cells
        land_count = 0
        for i in range(n):
            for j in range(n):
                if grid[i][j] == 1:
                    distances[i][j] = 0
                    land_count += 1
        
        if land_count == 0 or land_count == n * n:
            return -1
        
        # Forward pass: top-left to bottom-right
        for i in range(n):
            for j in range(n):
                if distances[i][j] != 0:
                    if i > 0:
                        distances[i][j] = min(distances[i][j], distances[i-1][j] + 1)
                    if j > 0:
                        distances[i][j] = min(distances[i][j], distances[i][j-1] + 1)
        
        # Backward pass: bottom-right to top-left
        for i in range(n-1, -1, -1):
            for j in range(n-1, -1, -1):
                if distances[i][j] != 0:
                    if i < n-1:
                        distances[i][j] = min(distances[i][j], distances[i+1][j] + 1)
                    if j < n-1:
                        distances[i][j] = min(distances[i][j], distances[i][j+1] + 1)
        
        # Find maximum distance for water cells
        max_dist = 0
        for i in range(n):
            for j in range(n):
                if grid[i][j] == 0:
                    max_dist = max(max_dist, distances[i][j])
        
        return max_dist if max_dist != float('inf') else -1

def test_max_distance():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (grid, expected)
        ([[1,0,1],[0,0,0],[1,0,1]], 2),
        ([[1,0,0],[0,0,0],[0,0,0]], 4),
        ([[0,0,0],[0,0,0],[0,0,0]], -1),
        ([[1,1,1],[1,1,1],[1,1,1]], -1),
        ([[1]], -1),
        ([[0]], -1),
        ([[1,0],[0,1]], 1),
        ([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]], 6),
    ]
    
    approaches = [
        ("Multi-Source BFS", solution.maxDistance_approach1_multi_source_bfs),
        ("Level-by-Level BFS", solution.maxDistance_approach2_level_by_level_bfs),
        ("Binary Search + BFS", solution.maxDistance_approach3_binary_search_bfs),
        ("DP Distance Transform", solution.maxDistance_approach4_dp_distance_transform),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (grid, expected) in enumerate(test_cases):
            # Create deep copy for testing
            test_grid = [row[:] for row in grid]
            result = func(test_grid)
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} Grid: {grid}, Expected: {expected}, Got: {result}")

def demonstrate_farthest_water():
    """Demonstrate finding the farthest water cell"""
    print("\n=== Farthest Water Cell Demo ===")
    
    grid = [[1,0,1],
            [0,0,0],
            [1,0,1]]
    
    print("Grid (1=land, 0=water):")
    print_grid_visual(grid)
    
    # Multi-source BFS simulation
    n = len(grid)
    distances = [[-1] * n for _ in range(n)]
    queue = deque()
    
    # Initialize with land cells
    print("\nInitialization - Land cells:")
    for i in range(n):
        for j in range(n):
            if grid[i][j] == 1:
                distances[i][j] = 0
                queue.append((i, j, 0))
                print(f"  Land at ({i},{j})")
    
    print_distance_grid(distances)
    
    # BFS expansion
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    step = 0
    max_distance = 0
    
    while queue:
        step += 1
        print(f"\nStep {step}:")
        size = len(queue)
        step_updates = []
        
        for _ in range(size):
            i, j, dist = queue.popleft()
            
            for di, dj in directions:
                ni, nj = i + di, j + dj
                
                if (0 <= ni < n and 0 <= nj < n and distances[ni][nj] == -1):
                    distances[ni][nj] = dist + 1
                    max_distance = max(max_distance, dist + 1)
                    queue.append((ni, nj, dist + 1))
                    step_updates.append(f"({ni},{nj})={dist+1}")
        
        if step_updates:
            print(f"  Updates: {step_updates}")
            print_distance_grid(distances)
        else:
            break
    
    print(f"\nMaximum distance from land: {max_distance}")
    
    # Find the farthest water cells
    farthest_cells = []
    for i in range(n):
        for j in range(n):
            if grid[i][j] == 0 and distances[i][j] == max_distance:
                farthest_cells.append((i, j))
    
    print(f"Farthest water cell(s): {farthest_cells}")

def print_grid_visual(grid):
    """Print grid with visual symbols"""
    symbols = {0: "🌊", 1: "🏝️"}
    for row in grid:
        print(f"  {''.join(symbols[cell] for cell in row)}")

def print_distance_grid(distances):
    """Print distance grid"""
    print("  Distance grid:")
    for row in distances:
        formatted_row = []
        for val in row:
            if val == -1:
                formatted_row.append(".")
            else:
                formatted_row.append(str(val))
        print(f"    {formatted_row}")

def analyze_problem_variants():
    """Analyze different variants of the distance problem"""
    print("\n=== Problem Variants Analysis ===")
    
    print("Related Distance Problems:")
    print("1. 🏝️ As Far from Land (this problem)")
    print("   • Find water cell farthest from any land")
    print("   • Multi-source BFS from all land cells")
    print("   • Return maximum distance to any water cell")
    
    print("\n2. 🚪 Walls and Gates")
    print("   • Fill rooms with distance to nearest gate")
    print("   • Multi-source BFS from all gates")
    print("   • Update all reachable cells with distances")
    
    print("\n3. 📊 01 Matrix")
    print("   • Distance to nearest 0 for each cell")
    print("   • Multi-source BFS from all 0s")
    print("   • Return distance matrix for all cells")
    
    print("\n4. 🍊 Rotting Oranges")
    print("   • Time for all oranges to rot")
    print("   • Multi-source BFS from all rotten oranges")
    print("   • Return time when all fresh oranges rot")
    
    print("\nCommon Pattern - Multi-Source BFS:")
    print("• Start from all source cells simultaneously")
    print("• Use BFS to propagate distances level by level")
    print("• Track maximum distance or specific conditions")
    print("• O(M*N) time complexity for grid problems")
    
    print("\nKey Insights:")
    print("• Multi-source BFS naturally handles multiple starting points")
    print("• Level-order processing ensures optimal distances")
    print("• First visit to cell guarantees shortest distance")
    print("• Edge cases: no sources, all sources, single cell")

if __name__ == "__main__":
    test_max_distance()
    demonstrate_farthest_water()
    analyze_problem_variants()

"""
Graph Theory Concepts:
1. Multi-Source BFS for Maximum Distance
2. Distance Maximization in Grids
3. Manhattan Distance Calculation
4. Farthest Point Queries

Key Algorithm Insights:
- Multi-source BFS from all land cells simultaneously
- Track maximum distance reached during BFS traversal
- Level-order processing ensures optimal distance calculation
- Handle edge cases: no land, no water, single cell

Optimization Techniques:
- Early termination when all cells visited
- Distance tracking during BFS expansion
- Binary search on answer for alternative approach
- DP distance transform for space-optimized solution

Real-world Applications:
- Urban planning (maximum distance from services)
- Emergency response (farthest point from rescue stations)
- Game AI (safe zones farthest from enemies)
- Network analysis (nodes farthest from critical infrastructure)
- Environmental science (pollution dispersion modeling)
- Logistics (worst-case delivery distances)

This problem showcases multi-source BFS for finding optimal
locations in spatial analysis and distance optimization.
"""
