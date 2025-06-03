"""
Dynamic Programming - Grid/Matrix Patterns
This module implements various DP problems on grids and matrices including path counting,
path optimization, and grid-based optimization problems with detailed analysis.
"""

from typing import List, Dict, Tuple, Optional
import time
from collections import deque

# ==================== UNIQUE PATHS PROBLEMS ====================

class UniquePaths:
    """
    Unique Paths Problems - counting paths in grids
    
    Classic problems involving counting the number of ways to traverse
    from one corner of a grid to another with movement constraints.
    """
    
    def unique_paths_basic(self, m: int, n: int) -> int:
        """
        Count unique paths from top-left to bottom-right
        Can only move right or down
        
        LeetCode 62 - Unique Paths
        
        Time Complexity: O(m * n)
        Space Complexity: O(n)
        
        Args:
            m: Number of rows
            n: Number of columns
        
        Returns:
            Number of unique paths
        """
        # dp[j] represents number of ways to reach column j in current row
        dp = [1] * n
        
        for i in range(1, m):
            for j in range(1, n):
                dp[j] = dp[j] + dp[j - 1]  # from top + from left
        
        return dp[n - 1]
    
    def unique_paths_2d(self, m: int, n: int) -> int:
        """
        2D DP version for better understanding
        
        dp[i][j] = number of ways to reach cell (i, j)
        
        Time Complexity: O(m * n)
        Space Complexity: O(m * n)
        """
        dp = [[1 for _ in range(n)] for _ in range(m)]
        
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        
        return dp[m - 1][n - 1]
    
    def unique_paths_mathematical(self, m: int, n: int) -> int:
        """
        Mathematical solution using combinations
        
        Total moves = (m-1) down + (n-1) right = m+n-2
        Choose (m-1) positions for down moves: C(m+n-2, m-1)
        
        Time Complexity: O(min(m, n))
        Space Complexity: O(1)
        """
        total_moves = m + n - 2
        down_moves = m - 1
        
        # Calculate C(total_moves, down_moves)
        result = 1
        for i in range(min(down_moves, total_moves - down_moves)):
            result = result * (total_moves - i) // (i + 1)
        
        return result
    
    def unique_paths_with_obstacles(self, obstacle_grid: List[List[int]]) -> int:
        """
        Count unique paths with obstacles
        
        LeetCode 63 - Unique Paths II
        
        Time Complexity: O(m * n)
        Space Complexity: O(n)
        
        Args:
            obstacle_grid: Grid where 1 represents obstacle, 0 is free
        
        Returns:
            Number of unique paths avoiding obstacles
        """
        if not obstacle_grid or not obstacle_grid[0] or obstacle_grid[0][0] == 1:
            return 0
        
        m, n = len(obstacle_grid), len(obstacle_grid[0])
        dp = [0] * n
        dp[0] = 1
        
        for i in range(m):
            for j in range(n):
                if obstacle_grid[i][j] == 1:
                    dp[j] = 0
                elif j > 0:
                    dp[j] += dp[j - 1]
        
        return dp[n - 1]
    
    def unique_paths_k_moves(self, m: int, n: int, k: int) -> int:
        """
        Count paths with exactly k moves (can move in 4 directions)
        
        Time Complexity: O(m * n * k)
        Space Complexity: O(m * n * k)
        
        Args:
            m, n: Grid dimensions
            k: Exact number of moves allowed
        
        Returns:
            Number of paths with exactly k moves
        """
        # dp[i][j][moves] = ways to reach (i,j) with 'moves' moves
        dp = [[[0 for _ in range(k + 1)] for _ in range(n)] for _ in range(m)]
        dp[0][0][0] = 1  # Start position with 0 moves
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        for moves in range(1, k + 1):
            for i in range(m):
                for j in range(n):
                    for di, dj in directions:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < m and 0 <= nj < n:
                            dp[i][j][moves] += dp[ni][nj][moves - 1]
        
        return sum(dp[i][j][k] for i in range(m) for j in range(n))
    
    def unique_paths_with_keys(self, grid: List[str]) -> int:
        """
        Count paths collecting all keys before reaching end
        
        More complex variant with state-based DP
        
        Args:
            grid: Grid with 'S' (start), 'E' (end), keys ('a'-'f'), locks ('A'-'F')
        
        Returns:
            Number of ways to collect all keys and reach end
        """
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        start = end = None
        all_keys = set()
        
        # Find start, end, and all keys
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 'S':
                    start = (i, j)
                elif grid[i][j] == 'E':
                    end = (i, j)
                elif 'a' <= grid[i][j] <= 'f':
                    all_keys.add(grid[i][j])
        
        if not start or not end:
            return 0
        
        total_keys = len(all_keys)
        target_mask = (1 << total_keys) - 1
        
        # BFS with state (i, j, key_mask)
        from collections import defaultdict
        
        # dp[i][j][mask] = number of ways to reach (i,j) with key_mask
        dp = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        dp[start[0]][start[1]][0] = 1
        
        queue = deque([(start[0], start[1], 0)])
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        while queue:
            i, j, key_mask = queue.popleft()
            
            for di, dj in directions:
                ni, nj = i + di, j + dj
                
                if 0 <= ni < m and 0 <= nj < n and grid[ni][nj] != '#':
                    new_mask = key_mask
                    cell = grid[ni][nj]
                    
                    # Check if we can pass through lock
                    if 'A' <= cell <= 'F':
                        key_needed = cell.lower()
                        if key_needed not in all_keys:
                            continue
                        key_index = ord(key_needed) - ord('a')
                        if not (key_mask & (1 << key_index)):
                            continue
                    
                    # Collect key
                    if 'a' <= cell <= 'f':
                        key_index = ord(cell) - ord('a')
                        new_mask |= (1 << key_index)
                    
                    if dp[ni][nj][new_mask] == 0 and (ni, nj, new_mask) not in queue:
                        queue.append((ni, nj, new_mask))
                    
                    dp[ni][nj][new_mask] += dp[i][j][key_mask]
        
        return dp[end[0]][end[1]][target_mask]

# ==================== PATH SUM PROBLEMS ====================

class PathSum:
    """
    Path Sum Problems - finding optimal paths in grids
    
    Problems involving finding paths that minimize or maximize
    the sum of values along the path.
    """
    
    def min_path_sum(self, grid: List[List[int]]) -> int:
        """
        Find minimum path sum from top-left to bottom-right
        
        LeetCode 64 - Minimum Path Sum
        
        Time Complexity: O(m * n)
        Space Complexity: O(n)
        
        Args:
            grid: 2D grid of positive integers
        
        Returns:
            Minimum path sum
        """
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        dp = [float('inf')] * n
        dp[0] = 0
        
        for i in range(m):
            dp[0] += grid[i][0]  # First column
            for j in range(1, n):
                if i == 0:
                    dp[j] = dp[j - 1] + grid[i][j]  # First row
                else:
                    dp[j] = min(dp[j], dp[j - 1]) + grid[i][j]
        
        return dp[n - 1]
    
    def min_path_sum_with_path(self, grid: List[List[int]]) -> Tuple[int, List[Tuple[int, int]]]:
        """
        Find minimum path sum and return the actual path
        
        Args:
            grid: 2D grid of positive integers
        
        Returns:
            Tuple of (min_sum, path_coordinates)
        """
        if not grid or not grid[0]:
            return 0, []
        
        m, n = len(grid), len(grid[0])
        dp = [[float('inf')] * n for _ in range(m)]
        parent = [[None] * n for _ in range(m)]
        
        dp[0][0] = grid[0][0]
        
        # Fill first row
        for j in range(1, n):
            dp[0][j] = dp[0][j - 1] + grid[0][j]
            parent[0][j] = (0, j - 1)
        
        # Fill first column
        for i in range(1, m):
            dp[i][0] = dp[i - 1][0] + grid[i][0]
            parent[i][0] = (i - 1, 0)
        
        # Fill rest of the grid
        for i in range(1, m):
            for j in range(1, n):
                if dp[i - 1][j] < dp[i][j - 1]:
                    dp[i][j] = dp[i - 1][j] + grid[i][j]
                    parent[i][j] = (i - 1, j)
                else:
                    dp[i][j] = dp[i][j - 1] + grid[i][j]
                    parent[i][j] = (i, j - 1)
        
        # Reconstruct path
        path = []
        i, j = m - 1, n - 1
        while i >= 0 and j >= 0:
            path.append((i, j))
            if parent[i][j] is None:
                break
            i, j = parent[i][j]
        
        path.reverse()
        return dp[m - 1][n - 1], path
    
    def max_path_sum_triangle(self, triangle: List[List[int]]) -> int:
        """
        Maximum path sum in triangle (top to bottom)
        
        LeetCode 120 - Triangle
        
        Time Complexity: O(n²)
        Space Complexity: O(n)
        
        Args:
            triangle: Triangle represented as list of lists
        
        Returns:
            Maximum path sum from top to bottom
        """
        if not triangle:
            return 0
        
        n = len(triangle)
        dp = triangle[-1][:]  # Start from bottom row
        
        # Work backwards from second-last row
        for i in range(n - 2, -1, -1):
            for j in range(len(triangle[i])):
                dp[j] = triangle[i][j] + max(dp[j], dp[j + 1])
        
        return dp[0]
    
    def max_path_sum_matrix(self, matrix: List[List[int]]) -> int:
        """
        Maximum path sum in matrix (any direction allowed)
        
        Time Complexity: O(m * n)
        Space Complexity: O(m * n)
        
        Args:
            matrix: 2D matrix of integers
        
        Returns:
            Maximum path sum
        """
        if not matrix or not matrix[0]:
            return 0
        
        m, n = len(matrix), len(matrix[0])
        dp = [[float('-inf')] * n for _ in range(m)]
        dp[0][0] = matrix[0][0]
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        # Use modified Dijkstra's algorithm for maximum path
        from heapq import heappush, heappop
        
        pq = [(-matrix[0][0], 0, 0)]  # Use negative for max heap
        visited = set()
        
        while pq:
            neg_dist, i, j = heappop(pq)
            dist = -neg_dist
            
            if (i, j) in visited:
                continue
            
            visited.add((i, j))
            dp[i][j] = max(dp[i][j], dist)
            
            for di, dj in directions:
                ni, nj = i + di, j + dj
                if (0 <= ni < m and 0 <= nj < n and 
                    (ni, nj) not in visited):
                    new_dist = dist + matrix[ni][nj]
                    if new_dist > dp[ni][nj]:
                        dp[ni][nj] = new_dist
                        heappush(pq, (-new_dist, ni, nj))
        
        return max(max(row) for row in dp)
    
    def falling_path_sum(self, matrix: List[List[int]]) -> int:
        """
        Minimum falling path sum (can move diagonally down)
        
        LeetCode 931 - Minimum Falling Path Sum
        
        Time Complexity: O(n²)
        Space Complexity: O(n)
        
        Args:
            matrix: n×n matrix of integers
        
        Returns:
            Minimum falling path sum
        """
        if not matrix or not matrix[0]:
            return 0
        
        n = len(matrix)
        dp = matrix[0][:]
        
        for i in range(1, n):
            new_dp = [float('inf')] * n
            for j in range(n):
                # From directly above
                new_dp[j] = min(new_dp[j], dp[j] + matrix[i][j])
                
                # From diagonally above-left
                if j > 0:
                    new_dp[j] = min(new_dp[j], dp[j - 1] + matrix[i][j])
                
                # From diagonally above-right
                if j < n - 1:
                    new_dp[j] = min(new_dp[j], dp[j + 1] + matrix[i][j])
            
            dp = new_dp
        
        return min(dp)

# ==================== GOLD MINE PROBLEM ====================

class GoldMine:
    """
    Gold Mine Problem - collect maximum gold from grid
    
    Problems involving collecting maximum value while traversing grid
    with various constraints and movement patterns.
    """
    
    def max_gold_collected(self, gold: List[List[int]]) -> int:
        """
        Collect maximum gold starting from leftmost column
        Can move right, diagonally up-right, or diagonally down-right
        
        Time Complexity: O(m * n)
        Space Complexity: O(n)
        
        Args:
            gold: 2D grid representing gold amounts
        
        Returns:
            Maximum gold that can be collected
        """
        if not gold or not gold[0]:
            return 0
        
        m, n = len(gold), len(gold[0])
        
        # dp[j] represents max gold collected ending at column j
        dp = [gold[i][0] for i in range(m)]
        
        for j in range(1, n):
            new_dp = [0] * m
            for i in range(m):
                # From left
                new_dp[i] = dp[i] + gold[i][j]
                
                # From diagonally up-left
                if i > 0:
                    new_dp[i] = max(new_dp[i], dp[i - 1] + gold[i][j])
                
                # From diagonally down-left
                if i < m - 1:
                    new_dp[i] = max(new_dp[i], dp[i + 1] + gold[i][j])
            
            dp = new_dp
        
        return max(dp)
    
    def max_gold_path_finding(self, grid: List[List[int]]) -> int:
        """
        LeetCode 1219 - Path with Maximum Gold
        
        Can start from any cell and move in 4 directions.
        Cannot revisit cells in same path.
        
        Time Complexity: O(4^(m*n)) worst case
        Space Complexity: O(m * n)
        
        Args:
            grid: Grid where 0 is obstacle, positive numbers are gold
        
        Returns:
            Maximum gold collected in any path
        """
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        max_gold = 0
        
        def dfs(i: int, j: int, visited: set) -> int:
            if (i < 0 or i >= m or j < 0 or j >= n or 
                grid[i][j] == 0 or (i, j) in visited):
                return 0
            
            visited.add((i, j))
            current_gold = grid[i][j]
            
            # Try all 4 directions
            max_path = 0
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                max_path = max(max_path, dfs(i + di, j + dj, visited))
            
            visited.remove((i, j))  # Backtrack
            return current_gold + max_path
        
        # Try starting from each cell
        for i in range(m):
            for j in range(n):
                if grid[i][j] > 0:
                    max_gold = max(max_gold, dfs(i, j, set()))
        
        return max_gold
    
    def gold_mine_with_time_limit(self, gold: List[List[int]], time_limit: int) -> int:
        """
        Collect maximum gold within time limit
        Each move takes 1 unit of time
        
        Args:
            gold: 2D gold grid
            time_limit: Maximum time allowed
        
        Returns:
            Maximum gold collected within time limit
        """
        if not gold or not gold[0] or time_limit <= 0:
            return 0
        
        m, n = len(gold), len(gold[0])
        
        # dp[i][j][t] = max gold at (i,j) with time t
        dp = [[[0 for _ in range(time_limit + 1)] 
               for _ in range(n)] for _ in range(m)]
        
        # Initialize: can start at any cell at time 0
        for i in range(m):
            for j in range(n):
                if gold[i][j] > 0:
                    dp[i][j][1] = gold[i][j]
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        for t in range(2, time_limit + 1):
            for i in range(m):
                for j in range(n):
                    if gold[i][j] == 0:
                        continue
                    
                    for di, dj in directions:
                        ni, nj = i + di, j + dj
                        if (0 <= ni < m and 0 <= nj < n and 
                            dp[ni][nj][t - 1] > 0):
                            dp[i][j][t] = max(dp[i][j][t], 
                                            dp[ni][nj][t - 1] + gold[i][j])
        
        max_gold = 0
        for i in range(m):
            for j in range(n):
                for t in range(time_limit + 1):
                    max_gold = max(max_gold, dp[i][j][t])
        
        return max_gold

# ==================== COLLECT MAXIMUM POINTS ====================

class MaximumPoints:
    """
    Maximum Points Collection Problems
    
    Various problems involving collecting maximum points from grids
    with different movement patterns and constraints.
    """
    
    def cherry_pickup(self, grid: List[List[int]]) -> int:
        """
        LeetCode 741 - Cherry Pickup
        
        Go from (0,0) to (n-1,n-1) and back, collect maximum cherries.
        Equivalent to two people going from (0,0) to (n-1,n-1) simultaneously.
        
        Time Complexity: O(n³)
        Space Complexity: O(n³)
        
        Args:
            grid: Grid where 1 is cherry, 0 is empty, -1 is thorn
        
        Returns:
            Maximum cherries collected
        """
        if not grid or not grid[0] or grid[0][0] == -1:
            return 0
        
        n = len(grid)
        if grid[n - 1][n - 1] == -1:
            return 0
        
        # dp[k][i1][i2] = max cherries when both people have taken k steps
        # Person 1 at (i1, k-i1), Person 2 at (i2, k-i2)
        dp = {}
        
        def solve(k: int, i1: int, i2: int) -> int:
            j1, j2 = k - i1, k - i2
            
            # Check bounds and obstacles
            if (k >= 2 * n - 1 or i1 < 0 or i1 >= n or i2 < 0 or i2 >= n or
                j1 < 0 or j1 >= n or j2 < 0 or j2 >= n or
                grid[i1][j1] == -1 or grid[i2][j2] == -1):
                return float('-inf')
            
            if k == 2 * n - 2:  # Reached end
                return grid[n - 1][n - 1]
            
            if (k, i1, i2) in dp:
                return dp[(k, i1, i2)]
            
            # Collect cherries at current positions
            cherries = grid[i1][j1]
            if i1 != i2:  # Different positions
                cherries += grid[i2][j2]
            
            # Try all possible moves for both people
            max_future = float('-inf')
            for di1 in [0, 1]:  # Person 1: right or down
                for di2 in [0, 1]:  # Person 2: right or down
                    max_future = max(max_future, 
                                   solve(k + 1, i1 + di1, i2 + di2))
            
            dp[(k, i1, i2)] = cherries + max_future
            return dp[(k, i1, i2)]
        
        result = solve(0, 0, 0)
        return max(0, result)
    
    def cherry_pickup_ii(self, grid: List[List[int]]) -> int:
        """
        LeetCode 1463 - Cherry Pickup II
        
        Two robots start at top corners, move down collecting cherries.
        
        Time Complexity: O(m * n²)
        Space Complexity: O(n²)
        
        Args:
            grid: Grid with cherry values
        
        Returns:
            Maximum cherries collected by both robots
        """
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        
        # dp[j1][j2] = max cherries with robot1 at col j1, robot2 at col j2
        dp = [[0] * n for _ in range(n)]
        dp[0][n - 1] = grid[0][0] + grid[0][n - 1] if n > 1 else grid[0][0]
        
        for i in range(1, m):
            new_dp = [[0] * n for _ in range(n)]
            
            for j1 in range(n):
                for j2 in range(n):
                    for dj1 in [-1, 0, 1]:  # Robot 1 moves
                        for dj2 in [-1, 0, 1]:  # Robot 2 moves
                            nj1, nj2 = j1 + dj1, j2 + dj2
                            
                            if 0 <= nj1 < n and 0 <= nj2 < n:
                                cherries = grid[i][j1]
                                if j1 != j2:
                                    cherries += grid[i][j2]
                                
                                new_dp[j1][j2] = max(new_dp[j1][j2],
                                                    dp[nj1][nj2] + cherries)
            
            dp = new_dp
        
        return max(max(row) for row in dp)
    
    def max_points_with_cooldown(self, points: List[List[int]], 
                                cooldown: int) -> int:
        """
        Collect maximum points with cooldown between collections
        
        Args:
            points: 2D grid of points
            cooldown: Number of steps to wait after collecting
        
        Returns:
            Maximum points collected
        """
        if not points or not points[0]:
            return 0
        
        m, n = len(points), len(points[0])
        
        # dp[i][j][last_collect] = max points at (i,j) with last collection time
        dp = {}
        
        def solve(i: int, j: int, last_collect: int, step: int) -> int:
            if i >= m or j >= n:
                return 0
            
            if (i, j, last_collect) in dp:
                return dp[(i, j, last_collect)]
            
            result = 0
            
            # Move right without collecting
            result = max(result, solve(i, j + 1, last_collect, step + 1))
            
            # Move down without collecting
            result = max(result, solve(i + 1, j, last_collect, step + 1))
            
            # Collect if cooldown period has passed
            if step - last_collect >= cooldown:
                # Collect and move right
                if j + 1 < n:
                    result = max(result, points[i][j] + 
                               solve(i, j + 1, step, step + 1))
                
                # Collect and move down
                if i + 1 < m:
                    result = max(result, points[i][j] + 
                               solve(i + 1, j, step, step + 1))
            
            dp[(i, j, last_collect)] = result
            return result
        
        return solve(0, 0, -cooldown, 0)
    
    def max_points_multi_path(self, grid: List[List[int]], k: int) -> int:
        """
        Collect maximum points using k different paths
        
        Args:
            grid: 2D grid of points
            k: Number of paths allowed
        
        Returns:
            Maximum points from k non-overlapping paths
        """
        if not grid or not grid[0] or k <= 0:
            return 0
        
        m, n = len(grid), len(grid[0])
        
        # For simplicity, implement for k=2 (can be extended)
        if k >= 2:
            return self.cherry_pickup_ii(grid)
        
        # Single path case
        return self.min_path_sum([row[:] for row in grid])

# ==================== PERFORMANCE ANALYSIS ====================

def performance_analysis():
    """Analyze performance of different grid DP approaches"""
    print("=== Grid DP Performance Analysis ===\n")
    
    # Test unique paths
    paths = UniquePaths()
    m, n = 10, 10
    
    print(f"Unique Paths for {m}×{n} grid:")
    
    start_time = time.time()
    result_2d = paths.unique_paths_2d(m, n)
    time_2d = time.time() - start_time
    
    start_time = time.time()
    result_1d = paths.unique_paths_basic(m, n)
    time_1d = time.time() - start_time
    
    start_time = time.time()
    result_math = paths.unique_paths_mathematical(m, n)
    time_math = time.time() - start_time
    
    print(f"  2D DP: {result_2d} ({time_2d:.6f}s)")
    print(f"  1D DP: {result_1d} ({time_1d:.6f}s)")
    print(f"  Mathematical: {result_math} ({time_math:.6f}s)")
    print(f"  All results match: {result_2d == result_1d == result_math}")

# ==================== EXAMPLE USAGE AND TESTING ====================

if __name__ == "__main__":
    print("=== Grid/Matrix DP Demo ===\n")
    
    # Unique Paths
    print("1. Unique Paths Problems:")
    paths = UniquePaths()
    
    m, n = 3, 7
    unique_count = paths.unique_paths_basic(m, n)
    print(f"  Unique paths in {m}×{n} grid: {unique_count}")
    
    # With obstacles
    obstacle_grid = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    paths_with_obstacles = paths.unique_paths_with_obstacles(obstacle_grid)
    print(f"  Paths with obstacles: {paths_with_obstacles}")
    print()
    
    # Path Sum Problems
    print("2. Path Sum Problems:")
    path_sum = PathSum()
    
    grid = [[1, 3, 1], [1, 5, 1], [4, 2, 1]]
    min_sum = path_sum.min_path_sum(grid)
    min_sum_with_path, path = path_sum.min_path_sum_with_path(grid)
    
    print(f"  Grid: {grid}")
    print(f"  Minimum path sum: {min_sum}")
    print(f"  Optimal path: {path}")
    
    # Triangle
    triangle = [[2], [3, 4], [6, 5, 7], [4, 1, 8, 3]]
    max_triangle = path_sum.max_path_sum_triangle(triangle)
    print(f"  Maximum path sum in triangle: {max_triangle}")
    
    # Falling path
    matrix = [[2, 1, 3], [6, 5, 4], [7, 8, 9]]
    falling_sum = path_sum.falling_path_sum(matrix)
    print(f"  Minimum falling path sum: {falling_sum}")
    print()
    
    # Gold Mine Problems
    print("3. Gold Mine Problems:")
    gold_mine = GoldMine()
    
    gold_grid = [[1, 3, 3], [2, 1, 4], [0, 6, 4]]
    max_gold_collected = gold_mine.max_gold_collected(gold_grid)
    print(f"  Gold grid: {gold_grid}")
    print(f"  Maximum gold collected: {max_gold_collected}")
    
    # Path finding version
    gold_path_grid = [[0, 6, 0], [5, 8, 7], [0, 9, 0]]
    max_gold_path = gold_mine.max_gold_path_finding(gold_path_grid)
    print(f"  Maximum gold with path finding: {max_gold_path}")
    print()
    
    # Maximum Points
    print("4. Maximum Points Collection:")
    max_points = MaximumPoints()
    
    # Cherry pickup
    cherry_grid = [[0, 1, -1], [1, 0, -1], [1, 1, 1]]
    cherries = max_points.cherry_pickup(cherry_grid)
    print(f"  Cherry grid: {cherry_grid}")
    print(f"  Maximum cherries collected: {cherries}")
    
    # Cherry pickup II
    cherry_grid_ii = [[3, 1, 1], [2, 5, 1], [1, 5, 5], [2, 1, 1]]
    cherries_ii = max_points.cherry_pickup_ii(cherry_grid_ii)
    print(f"  Cherry pickup II result: {cherries_ii}")
    print()
    
    # Performance analysis
    performance_analysis()
    print()
    
    # Pattern Recognition Guide
    print("=== Grid DP Pattern Recognition ===")
    print("Common Grid DP Patterns:")
    print("  1. Path Counting: Count ways to reach destination")
    print("  2. Path Optimization: Find optimal (min/max) path")
    print("  3. Collection Problems: Collect maximum items along path")
    print("  4. State-based: Track additional state (keys, time, etc.)")
    print("  5. Multi-path: Multiple agents/paths simultaneously")
    
    print("\nSpace Optimization Techniques:")
    print("  1. 2D → 1D: When only previous row/column needed")
    print("  2. Rolling arrays: For multi-dimensional state")
    print("  3. Mathematical solutions: For simple counting problems")
    print("  4. State compression: Use bitmasks for state")
    
    print("\nCommon Pitfalls:")
    print("  1. Index bounds checking")
    print("  2. Handling obstacles correctly")
    print("  3. State explosion in complex problems")
    print("  4. Off-by-one errors in grid traversal")
    
    print("\nReal-world Applications:")
    print("  1. Robot path planning")
    print("  2. Game level optimization")
    print("  3. Resource collection in games")
    print("  4. Route optimization with constraints")
    print("  5. Image processing and computer vision")
    
    print("\n=== Demo Complete ===") 