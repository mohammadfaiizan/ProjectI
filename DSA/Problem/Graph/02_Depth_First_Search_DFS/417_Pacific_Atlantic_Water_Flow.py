"""
417. Pacific Atlantic Water Flow
Difficulty: Medium

Problem:
There is an m x n rectangular island that borders both the Pacific Ocean and Atlantic Ocean. 
The Pacific Ocean touches the island's left and top edges, and the Atlantic Ocean touches 
the island's right and bottom edges.

The island is partitioned into a grid of square cells. You are given an m x n integer 
matrix heights where heights[r][c] represents the height above sea level of the cell at (r, c).

The island receives a lot of rain, and the rain water can flow to neighboring cells directly 
north, south, east, or west if the neighboring cell's height is less than or equal to the 
current cell's height. Water can flow from any cell adjacent to an ocean into that ocean.

Return a 2D list of grid coordinates result where result[i] = [ri, ci] denotes that rain 
water can flow from cell (ri, ci) to both the Pacific and Atlantic oceans.

Examples:
Input: heights = [[1,2,2,3,5],[3,2,3,4,4],[2,4,5,3,1],[6,7,1,4,5],[5,1,1,2,4]]
Output: [[0,4],[1,3],[1,4],[2,2],[3,0],[3,1],[4,0]]

Input: heights = [[1]]
Output: [[0,0]]

Constraints:
- m == heights.length
- n == heights[i].length
- 1 <= m, n <= 200
- 0 <= heights[r][c] <= 10^5
"""

from typing import List
from collections import deque

class Solution:
    def pacificAtlantic_approach1_dual_dfs(self, heights: List[List[int]]) -> List[List[int]]:
        """
        Approach 1: Dual DFS from Ocean Boundaries
        
        Start DFS from Pacific and Atlantic boundaries separately.
        Find cells reachable by both oceans.
        
        Time: O(M*N) - each cell visited at most twice
        Space: O(M*N) - recursion stack + visited sets
        """
        if not heights or not heights[0]:
            return []
        
        m, n = len(heights), len(heights[0])
        
        def dfs(i, j, visited, prev_height):
            """DFS to mark cells reachable from ocean"""
            if (i < 0 or i >= m or j < 0 or j >= n or 
                (i, j) in visited or heights[i][j] < prev_height):
                return
            
            visited.add((i, j))
            
            # Explore 4 directions with current height as minimum
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                dfs(i + di, j + dj, visited, heights[i][j])
        
        pacific_reachable = set()
        atlantic_reachable = set()
        
        # DFS from Pacific boundaries (top and left edges)
        for i in range(m):
            dfs(i, 0, pacific_reachable, 0)  # Left edge
        for j in range(n):
            dfs(0, j, pacific_reachable, 0)  # Top edge
        
        # DFS from Atlantic boundaries (bottom and right edges)
        for i in range(m):
            dfs(i, n-1, atlantic_reachable, 0)  # Right edge
        for j in range(n):
            dfs(m-1, j, atlantic_reachable, 0)  # Bottom edge
        
        # Find intersection - cells reachable by both oceans
        return [[i, j] for i, j in pacific_reachable & atlantic_reachable]
    
    def pacificAtlantic_approach2_dual_bfs(self, heights: List[List[int]]) -> List[List[int]]:
        """
        Approach 2: Dual BFS from Ocean Boundaries
        
        Use BFS instead of DFS for breadth-first exploration.
        
        Time: O(M*N)
        Space: O(M*N)
        """
        if not heights or not heights[0]:
            return []
        
        m, n = len(heights), len(heights[0])
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        def bfs(start_cells):
            """BFS from multiple starting cells"""
            visited = set()
            queue = deque(start_cells)
            
            for i, j in start_cells:
                visited.add((i, j))
            
            while queue:
                i, j = queue.popleft()
                
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    
                    if (0 <= ni < m and 0 <= nj < n and 
                        (ni, nj) not in visited and 
                        heights[ni][nj] >= heights[i][j]):
                        visited.add((ni, nj))
                        queue.append((ni, nj))
            
            return visited
        
        # Starting cells for Pacific (top and left boundaries)
        pacific_starts = []
        for i in range(m):
            pacific_starts.append((i, 0))
        for j in range(1, n):  # Avoid corner duplicate
            pacific_starts.append((0, j))
        
        # Starting cells for Atlantic (bottom and right boundaries)
        atlantic_starts = []
        for i in range(m):
            atlantic_starts.append((i, n-1))
        for j in range(n-1):  # Avoid corner duplicate
            atlantic_starts.append((m-1, j))
        
        pacific_reachable = bfs(pacific_starts)
        atlantic_reachable = bfs(atlantic_starts)
        
        return [[i, j] for i, j in pacific_reachable & atlantic_reachable]
    
    def pacificAtlantic_approach3_single_traversal(self, heights: List[List[int]]) -> List[List[int]]:
        """
        Approach 3: Single Traversal with State Tracking
        
        Use single DFS with state tracking for both oceans.
        
        Time: O(M*N)
        Space: O(M*N)
        """
        if not heights or not heights[0]:
            return []
        
        m, n = len(heights), len(heights[0])
        
        # 0: not visited, 1: Pacific reachable, 2: Atlantic reachable, 3: both
        state = [[0] * n for _ in range(m)]
        
        def dfs(i, j, ocean_bit, prev_height):
            """DFS with ocean state tracking"""
            if (i < 0 or i >= m or j < 0 or j >= n or 
                heights[i][j] < prev_height or 
                state[i][j] & ocean_bit):
                return
            
            state[i][j] |= ocean_bit
            
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                dfs(i + di, j + dj, ocean_bit, heights[i][j])
        
        # Mark Pacific reachable cells (bit 1)
        for i in range(m):
            dfs(i, 0, 1, 0)
        for j in range(n):
            dfs(0, j, 1, 0)
        
        # Mark Atlantic reachable cells (bit 2)
        for i in range(m):
            dfs(i, n-1, 2, 0)
        for j in range(n):
            dfs(m-1, j, 2, 0)
        
        # Find cells reachable by both (state == 3)
        result = []
        for i in range(m):
            for j in range(n):
                if state[i][j] == 3:
                    result.append([i, j])
        
        return result
    
    def pacificAtlantic_approach4_iterative_dfs(self, heights: List[List[int]]) -> List[List[int]]:
        """
        Approach 4: Iterative DFS to avoid recursion limits
        
        Use explicit stack for DFS implementation.
        
        Time: O(M*N)
        Space: O(M*N)
        """
        if not heights or not heights[0]:
            return []
        
        m, n = len(heights), len(heights[0])
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        def iterative_dfs(start_cells):
            """Iterative DFS from multiple starting cells"""
            visited = set()
            stack = list(start_cells)
            
            for cell in start_cells:
                visited.add(cell)
            
            while stack:
                i, j = stack.pop()
                
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    
                    if (0 <= ni < m and 0 <= nj < n and 
                        (ni, nj) not in visited and 
                        heights[ni][nj] >= heights[i][j]):
                        visited.add((ni, nj))
                        stack.append((ni, nj))
            
            return visited
        
        # Pacific boundary cells
        pacific_starts = [(i, 0) for i in range(m)] + [(0, j) for j in range(1, n)]
        
        # Atlantic boundary cells
        atlantic_starts = [(i, n-1) for i in range(m)] + [(m-1, j) for j in range(n-1)]
        
        pacific_reachable = iterative_dfs(pacific_starts)
        atlantic_reachable = iterative_dfs(atlantic_starts)
        
        return [[i, j] for i, j in pacific_reachable & atlantic_reachable]

def test_pacific_atlantic():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (heights, expected_sorted)
        ([[1,2,2,3,5],[3,2,3,4,4],[2,4,5,3,1],[6,7,1,4,5],[5,1,1,2,4]], 
         [[0,4],[1,3],[1,4],[2,2],[3,0],[3,1],[4,0]]),
        ([[1]], [[0,0]]),
        ([[1,1],[1,1]], [[0,0],[0,1],[1,0],[1,1]]),
        ([[3,3,3],[3,1,3],[0,2,4]], [[0,0],[0,1],[0,2],[1,0],[1,2],[2,2]]),
    ]
    
    approaches = [
        ("Dual DFS", solution.pacificAtlantic_approach1_dual_dfs),
        ("Dual BFS", solution.pacificAtlantic_approach2_dual_bfs),
        ("Single Traversal", solution.pacificAtlantic_approach3_single_traversal),
        ("Iterative DFS", solution.pacificAtlantic_approach4_iterative_dfs),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (heights, expected) in enumerate(test_cases):
            result = func(heights)
            result_sorted = sorted(result)
            expected_sorted = sorted(expected)
            status = "✓" if result_sorted == expected_sorted else "✗"
            print(f"Test {i+1}: {status}")
            print(f"         Expected: {expected_sorted}")
            print(f"         Got: {result_sorted}")

def demonstrate_water_flow():
    """Demonstrate water flow analysis"""
    print("\n=== Water Flow Analysis Demo ===")
    
    heights = [
        [1,2,2,3,5],
        [3,2,3,4,4],
        [2,4,5,3,1],
        [6,7,1,4,5],
        [5,1,1,2,4]
    ]
    
    print("Height map:")
    for i, row in enumerate(heights):
        print(f"  Row {i}: {row}")
    
    print(f"\nOcean boundaries:")
    print(f"  Pacific: Top edge (row 0) and Left edge (col 0)")
    print(f"  Atlantic: Bottom edge (row {len(heights)-1}) and Right edge (col {len(heights[0])-1})")
    
    m, n = len(heights), len(heights[0])
    
    # Analyze reachability from each ocean
    def dfs_reachable(start_cells, ocean_name):
        visited = set()
        
        def dfs(i, j, prev_height):
            if (i < 0 or i >= m or j < 0 or j >= n or 
                (i, j) in visited or heights[i][j] < prev_height):
                return
            
            visited.add((i, j))
            
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                dfs(i + di, j + dj, heights[i][j])
        
        for i, j in start_cells:
            dfs(i, j, 0)
        
        print(f"\n{ocean_name} reachable cells:")
        reachable_grid = [['.' for _ in range(n)] for _ in range(m)]
        for i, j in visited:
            reachable_grid[i][j] = 'O'
        
        for i, row in enumerate(reachable_grid):
            print(f"  Row {i}: {' '.join(row)}")
        
        return visited
    
    # Pacific analysis
    pacific_starts = [(i, 0) for i in range(m)] + [(0, j) for j in range(1, n)]
    pacific_reachable = dfs_reachable(pacific_starts, "Pacific")
    
    # Atlantic analysis
    atlantic_starts = [(i, n-1) for i in range(m)] + [(m-1, j) for j in range(n-1)]
    atlantic_reachable = dfs_reachable(atlantic_starts, "Atlantic")
    
    # Both oceans
    both_reachable = pacific_reachable & atlantic_reachable
    print(f"\nCells reachable by BOTH oceans:")
    both_grid = [['.' for _ in range(n)] for _ in range(m)]
    for i, j in both_reachable:
        both_grid[i][j] = 'B'
    
    for i, row in enumerate(both_grid):
        print(f"  Row {i}: {' '.join(row)}")
    
    print(f"\nResult coordinates: {sorted(list(both_reachable))}")

if __name__ == "__main__":
    test_pacific_atlantic()
    demonstrate_water_flow()

"""
Graph Theory Concepts:
1. Multi-source Graph Traversal
2. Reverse Flow Analysis
3. Set Intersection for Reachability
4. Boundary-initiated Exploration

Key Water Flow Insights:
- Water flows from high to low (or equal) elevation
- Reverse problem: Start from oceans, flow upward
- Cell reachable by both oceans if in intersection of reachable sets
- Boundary cells are natural starting points for traversal

Algorithm Strategy:
- Start DFS/BFS from ocean boundaries
- Flow "upward" (to higher or equal elevations)
- Track reachability from each ocean separately
- Find intersection of both reachable sets

Real-world Applications:
- Watershed analysis in geography
- Drainage system design
- Flood modeling and simulation
- Topographic analysis
- Environmental impact assessment
"""

