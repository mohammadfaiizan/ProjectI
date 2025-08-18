"""
994. Rotting Oranges
Difficulty: Easy

Problem:
You are given an m x n grid where each cell can have one of three values:
- 0 representing an empty cell,
- 1 representing a fresh orange, or
- 2 representing a rotten orange.

Every minute, any fresh orange that is 4-directionally adjacent to a rotten orange becomes rotten.

Return the minimum number of minutes that must elapse until no cell has a fresh orange. 
If this is impossible, return -1.

Examples:
Input: grid = [[2,1,1],[1,1,0],[0,1,1]]
Output: 4

Input: grid = [[2,1,1],[0,1,1],[1,0,1]]
Output: -1

Input: grid = [[0,2]]
Output: 0

Constraints:
- m == grid.length
- n == grid[i].length
- 1 <= m, n <= 10
- grid[i][j] is 0, 1, or 2
"""

from typing import List
from collections import deque

class Solution:
    def orangesRotting_approach1_multi_source_bfs(self, grid: List[List[int]]) -> int:
        """
        Approach 1: Multi-Source BFS (Optimal)
        
        Start BFS from all initially rotten oranges simultaneously.
        Each level represents one minute of time.
        
        Time: O(M*N) - visit each cell at most once
        Space: O(M*N) - queue size in worst case
        """
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        queue = deque()
        fresh_count = 0
        
        # Find all initially rotten oranges and count fresh ones
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 2:
                    queue.append((i, j, 0))  # (row, col, time)
                elif grid[i][j] == 1:
                    fresh_count += 1
        
        # If no fresh oranges, return 0
        if fresh_count == 0:
            return 0
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        max_time = 0
        
        # Multi-source BFS
        while queue:
            i, j, time = queue.popleft()
            max_time = max(max_time, time)
            
            # Check all 4 directions
            for di, dj in directions:
                ni, nj = i + di, j + dj
                
                if (0 <= ni < m and 0 <= nj < n and grid[ni][nj] == 1):
                    grid[ni][nj] = 2  # Mark as rotten
                    fresh_count -= 1
                    queue.append((ni, nj, time + 1))
        
        # Return -1 if some fresh oranges remain unreachable
        return max_time if fresh_count == 0 else -1
    
    def orangesRotting_approach2_level_by_level_bfs(self, grid: List[List[int]]) -> int:
        """
        Approach 2: Level-by-Level BFS
        
        Process all oranges at current level before moving to next level.
        
        Time: O(M*N)
        Space: O(M*N)
        """
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        queue = deque()
        fresh_count = 0
        
        # Initialize queue with all rotten oranges
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 2:
                    queue.append((i, j))
                elif grid[i][j] == 1:
                    fresh_count += 1
        
        if fresh_count == 0:
            return 0
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        minutes = 0
        
        # Level-by-level BFS
        while queue and fresh_count > 0:
            minutes += 1
            size = len(queue)
            
            # Process all oranges at current level
            for _ in range(size):
                i, j = queue.popleft()
                
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    
                    if (0 <= ni < m and 0 <= nj < n and grid[ni][nj] == 1):
                        grid[ni][nj] = 2
                        fresh_count -= 1
                        queue.append((ni, nj))
        
        return minutes if fresh_count == 0 else -1
    
    def orangesRotting_approach3_simulation_with_visited(self, grid: List[List[int]]) -> int:
        """
        Approach 3: Simulation with Visited Array
        
        Use separate visited array to avoid modifying original grid.
        
        Time: O(M*N)
        Space: O(M*N) - visited array + queue
        """
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        visited = [[False] * n for _ in range(m)]
        queue = deque()
        fresh_count = 0
        
        # Initialize
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 2:
                    queue.append((i, j, 0))
                    visited[i][j] = True
                elif grid[i][j] == 1:
                    fresh_count += 1
        
        if fresh_count == 0:
            return 0
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        max_time = 0
        
        while queue:
            i, j, time = queue.popleft()
            
            for di, dj in directions:
                ni, nj = i + di, j + dj
                
                if (0 <= ni < m and 0 <= nj < n and 
                    not visited[ni][nj] and grid[ni][nj] == 1):
                    visited[ni][nj] = True
                    fresh_count -= 1
                    max_time = time + 1
                    queue.append((ni, nj, time + 1))
        
        return max_time if fresh_count == 0 else -1
    
    def orangesRotting_approach4_iterative_simulation(self, grid: List[List[int]]) -> int:
        """
        Approach 4: Iterative Simulation
        
        Simulate minute by minute without using queue.
        
        Time: O(M*N*T) where T is total time
        Space: O(1) - in-place modification
        """
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        def count_fresh():
            return sum(1 for i in range(m) for j in range(n) if grid[i][j] == 1)
        
        def rot_adjacent():
            """Rot all fresh oranges adjacent to rotten ones"""
            newly_rotten = []
            
            for i in range(m):
                for j in range(n):
                    if grid[i][j] == 2:  # Rotten orange
                        for di, dj in directions:
                            ni, nj = i + di, j + dj
                            if (0 <= ni < m and 0 <= nj < n and grid[ni][nj] == 1):
                                newly_rotten.append((ni, nj))
            
            # Mark newly rotten oranges
            for i, j in newly_rotten:
                grid[i][j] = 2
            
            return len(newly_rotten) > 0
        
        initial_fresh = count_fresh()
        if initial_fresh == 0:
            return 0
        
        minutes = 0
        
        # Simulate minute by minute
        while True:
            if not rot_adjacent():
                break
            minutes += 1
        
        # Check if all oranges are rotten
        return minutes if count_fresh() == 0 else -1

def test_rotting_oranges():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (grid, expected)
        ([[2,1,1],[1,1,0],[0,1,1]], 4),
        ([[2,1,1],[0,1,1],[1,0,1]], -1),
        ([[0,2]], 0),
        ([[0]], 0),
        ([[1]], -1),
        ([[2]], 0),
        ([[2,1,1],[1,1,1],[0,1,2]], 2),
        ([[1,2]], 1),
    ]
    
    approaches = [
        ("Multi-Source BFS", solution.orangesRotting_approach1_multi_source_bfs),
        ("Level-by-Level BFS", solution.orangesRotting_approach2_level_by_level_bfs),
        ("Simulation with Visited", solution.orangesRotting_approach3_simulation_with_visited),
        ("Iterative Simulation", solution.orangesRotting_approach4_iterative_simulation),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (grid, expected) in enumerate(test_cases):
            # Create deep copy for testing
            test_grid = [row[:] for row in grid]
            result = func(test_grid)
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} Grid: {grid}, Expected: {expected}, Got: {result}")

def demonstrate_rotting_process():
    """Demonstrate the orange rotting process step by step"""
    print("\n=== Orange Rotting Process Demo ===")
    
    grid = [[2,1,1],
            [1,1,0],
            [0,1,1]]
    
    print("Initial state:")
    print_grid(grid, 0)
    
    # Simulate the process
    m, n = len(grid), len(grid[0])
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    minute = 0
    
    while True:
        newly_rotten = []
        
        # Find oranges that will rot this minute
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 2:  # Currently rotten
                    for di, dj in directions:
                        ni, nj = i + di, j + dj
                        if (0 <= ni < m and 0 <= nj < n and grid[ni][nj] == 1):
                            newly_rotten.append((ni, nj))
        
        if not newly_rotten:
            break
        
        # Apply rotting
        for i, j in newly_rotten:
            grid[i][j] = 2
        
        minute += 1
        print(f"\nAfter minute {minute}:")
        print_grid(grid, minute)
    
    # Check final state
    fresh_count = sum(1 for i in range(m) for j in range(n) if grid[i][j] == 1)
    if fresh_count == 0:
        print(f"\n✓ All oranges rotted in {minute} minutes!")
    else:
        print(f"\n✗ {fresh_count} fresh oranges remain unreachable")

def print_grid(grid, minute):
    """Helper function to print grid state"""
    symbols = {0: "⬜", 1: "🟩", 2: "🟥"}
    
    print(f"  Minute {minute}:")
    for row in grid:
        print(f"    {''.join(symbols[cell] for cell in row)}")
    
    fresh = sum(1 for row in grid for cell in row if cell == 1)
    rotten = sum(1 for row in grid for cell in row if cell == 2)
    print(f"    Fresh: {fresh}, Rotten: {rotten}")

def analyze_bfs_strategy():
    """Analyze why BFS is optimal for this problem"""
    print("\n=== BFS Strategy Analysis ===")
    
    print("Why BFS is perfect for Rotting Oranges:")
    print("1. 📊 Level-order processing: Each minute is a BFS level")
    print("2. 🎯 Multi-source: All rotten oranges spread simultaneously")
    print("3. ⏱️ Time tracking: BFS naturally tracks minimum time")
    print("4. 🔄 Simultaneous spread: All adjacent oranges rot at same time")
    
    print("\nBFS vs other approaches:")
    print("• DFS: ❌ Would not guarantee minimum time")
    print("• Greedy: ❌ No clear greedy strategy")
    print("• DP: ❌ Overkill for this problem")
    print("• BFS: ✅ Natural fit for level-order time simulation")
    
    print("\nKey BFS insights:")
    print("• Queue initially contains all rotten oranges")
    print("• Each queue level represents one minute")
    print("• Process all oranges at current level before next level")
    print("• Track maximum time reached during BFS")
    print("• Check if all fresh oranges were reached")

if __name__ == "__main__":
    test_rotting_oranges()
    demonstrate_rotting_process()
    analyze_bfs_strategy()

"""
Graph Theory Concepts:
1. Multi-Source BFS
2. Level-Order Traversal with Time
3. Simultaneous Propagation
4. Reachability Analysis

Key BFS Insights:
- Multi-source BFS: Start from all sources simultaneously
- Time simulation: Each BFS level represents one time unit
- Simultaneous spread: All adjacent cells affected at same time
- Optimality: BFS guarantees minimum time to reach all cells

Algorithm Strategy:
- Initialize queue with all initially rotten oranges
- Process level by level (minute by minute)
- Mark fresh oranges as rotten when reached
- Track time and count remaining fresh oranges

Real-world Applications:
- Epidemic/disease spread modeling
- Fire propagation simulation
- Network flooding algorithms
- Chemical reaction diffusion
- Information propagation in social networks
- Computer virus spread analysis

This problem perfectly demonstrates the power of multi-source BFS
for time-based propagation problems.
"""
