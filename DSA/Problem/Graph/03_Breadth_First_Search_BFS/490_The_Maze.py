"""
490. The Maze
Difficulty: Easy

Problem:
There is a ball in a maze with empty spaces (represented as 0) and walls (represented as 1). 
The ball can go through the empty spaces by rolling in one of the four directions (up, down, 
left or right), but it won't stop rolling until hitting a wall. When the ball stops, it 
could choose the next direction.

Given the m x n maze, the ball's start position and the destination, where start = [startrow, startcol] 
and destination = [destrow, destcol], return true if the ball can stop at the destination, 
otherwise return false.

You may assume that the borders of the maze are all walls (see examples).

Examples:
Input: maze = [[0,0,1,0,0],[0,0,0,0,0],[0,0,0,1,0],[1,1,0,1,1],[0,0,0,0,0]], 
       start = [0,4], destination = [4,4]
Output: true

Input: maze = [[0,0,1,0,0],[0,0,0,0,0],[0,0,0,1,0],[1,1,0,1,1],[0,0,0,0,0]], 
       start = [0,4], destination = [3,2]
Output: false

Input: maze = [[0,0,0,0,0],[1,1,0,0,1],[0,0,0,0,0],[0,1,0,0,1],[0,1,0,0,0]], 
       start = [4,3], destination = [0,1]
Output: false

Constraints:
- m == maze.length
- n == maze[i].length
- 1 <= m, n <= 100
- maze[i][j] is 0 or 1
- start.length == 2
- destination.length == 2
- 0 <= startrow, destrow < m
- 0 <= startcol, destcol < n
- Both the start and destination exist in empty spaces
"""

from typing import List
from collections import deque

class Solution:
    def hasPath_approach1_bfs_rolling(self, maze: List[List[int]], start: List[int], destination: List[int]) -> bool:
        """
        Approach 1: BFS with Rolling Simulation (Optimal)
        
        Use BFS to explore all possible stopping positions.
        Ball rolls until it hits a wall, then can choose new direction.
        
        Time: O(M*N) - each cell visited at most once as stopping point
        Space: O(M*N) - queue and visited set
        """
        if not maze or not maze[0]:
            return False
        
        m, n = len(maze), len(maze[0])
        start_r, start_c = start
        dest_r, dest_c = destination
        
        # If start is destination
        if start_r == dest_r and start_c == dest_c:
            return True
        
        visited = set()
        queue = deque([(start_r, start_c)])
        visited.add((start_r, start_c))
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        
        while queue:
            row, col = queue.popleft()
            
            # Try rolling in each direction
            for dr, dc in directions:
                # Roll until hitting a wall
                new_row, new_col = row, col
                
                while (0 <= new_row + dr < m and 0 <= new_col + dc < n and 
                       maze[new_row + dr][new_col + dc] == 0):
                    new_row += dr
                    new_col += dc
                
                # Check if we reached destination
                if new_row == dest_r and new_col == dest_c:
                    return True
                
                # Add stopping position to queue if not visited
                if (new_row, new_col) not in visited:
                    visited.add((new_row, new_col))
                    queue.append((new_row, new_col))
        
        return False
    
    def hasPath_approach2_dfs_rolling(self, maze: List[List[int]], start: List[int], destination: List[int]) -> bool:
        """
        Approach 2: DFS with Rolling Simulation
        
        Use DFS to explore paths with rolling mechanics.
        
        Time: O(M*N)
        Space: O(M*N) - recursion stack + visited set
        """
        if not maze or not maze[0]:
            return False
        
        m, n = len(maze), len(maze[0])
        visited = set()
        
        def dfs(row, col):
            """DFS to check if destination is reachable"""
            if row == destination[0] and col == destination[1]:
                return True
            
            if (row, col) in visited:
                return False
            
            visited.add((row, col))
            
            # Try rolling in each direction
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                # Roll until hitting a wall
                new_row, new_col = row, col
                
                while (0 <= new_row + dr < m and 0 <= new_col + dc < n and 
                       maze[new_row + dr][new_col + dc] == 0):
                    new_row += dr
                    new_col += dc
                
                if dfs(new_row, new_col):
                    return True
            
            return False
        
        return dfs(start[0], start[1])
    
    def hasPath_approach3_optimized_bfs(self, maze: List[List[int]], start: List[int], destination: List[int]) -> bool:
        """
        Approach 3: Optimized BFS with Early Termination
        
        Add optimizations like early termination and boundary checks.
        
        Time: O(M*N)
        Space: O(M*N)
        """
        if not maze or not maze[0]:
            return False
        
        m, n = len(maze), len(maze[0])
        start_r, start_c = start
        dest_r, dest_c = destination
        
        if start_r == dest_r and start_c == dest_c:
            return True
        
        visited = [[False] * n for _ in range(m)]
        queue = deque([(start_r, start_c)])
        visited[start_r][start_c] = True
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        while queue:
            row, col = queue.popleft()
            
            for dr, dc in directions:
                new_row, new_col = row, col
                
                # Roll until hitting boundary or wall
                while (0 <= new_row + dr < m and 0 <= new_col + dc < n and 
                       maze[new_row + dr][new_col + dc] == 0):
                    new_row += dr
                    new_col += dc
                
                # Early termination if destination reached
                if new_row == dest_r and new_col == dest_c:
                    return True
                
                # Continue BFS if not visited
                if not visited[new_row][new_col]:
                    visited[new_row][new_col] = True
                    queue.append((new_row, new_col))
        
        return False
    
    def hasPath_approach4_bidirectional_bfs(self, maze: List[List[int]], start: List[int], destination: List[int]) -> bool:
        """
        Approach 4: Bidirectional BFS
        
        Search from both start and destination simultaneously.
        
        Time: O(M*N)
        Space: O(M*N)
        """
        if not maze or not maze[0]:
            return False
        
        m, n = len(maze), len(maze[0])
        start_r, start_c = start
        dest_r, dest_c = destination
        
        if start_r == dest_r and start_c == dest_c:
            return True
        
        def get_stopping_positions(row, col):
            """Get all stopping positions from current position"""
            stops = []
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                new_row, new_col = row, col
                
                while (0 <= new_row + dr < m and 0 <= new_col + dc < n and 
                       maze[new_row + dr][new_col + dc] == 0):
                    new_row += dr
                    new_col += dc
                
                stops.append((new_row, new_col))
            return stops
        
        # Two BFS searches
        start_visited = {(start_r, start_c)}
        dest_visited = {(dest_r, dest_c)}
        
        start_queue = deque([(start_r, start_c)])
        dest_queue = deque([(dest_r, dest_c)])
        
        while start_queue or dest_queue:
            # Expand from start
            if start_queue:
                for _ in range(len(start_queue)):
                    row, col = start_queue.popleft()
                    
                    for stop_row, stop_col in get_stopping_positions(row, col):
                        if (stop_row, stop_col) in dest_visited:
                            return True
                        
                        if (stop_row, stop_col) not in start_visited:
                            start_visited.add((stop_row, stop_col))
                            start_queue.append((stop_row, stop_col))
            
            # Expand from destination
            if dest_queue:
                for _ in range(len(dest_queue)):
                    row, col = dest_queue.popleft()
                    
                    for stop_row, stop_col in get_stopping_positions(row, col):
                        if (stop_row, stop_col) in start_visited:
                            return True
                        
                        if (stop_row, stop_col) not in dest_visited:
                            dest_visited.add((stop_row, stop_col))
                            dest_queue.append((stop_row, stop_col))
        
        return False

def test_maze_path():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (maze, start, destination, expected)
        ([[0,0,1,0,0],[0,0,0,0,0],[0,0,0,1,0],[1,1,0,1,1],[0,0,0,0,0]], 
         [0,4], [4,4], True),
        ([[0,0,1,0,0],[0,0,0,0,0],[0,0,0,1,0],[1,1,0,1,1],[0,0,0,0,0]], 
         [0,4], [3,2], False),
        ([[0,0,0,0,0],[1,1,0,0,1],[0,0,0,0,0],[0,1,0,0,1],[0,1,0,0,0]], 
         [4,3], [0,1], False),
        ([[0]], [0,0], [0,0], True),
        ([[0,0],[0,0]], [0,0], [1,1], True),
        ([[0,1,0],[0,0,0],[0,0,0]], [0,0], [2,2], True),
    ]
    
    approaches = [
        ("BFS Rolling", solution.hasPath_approach1_bfs_rolling),
        ("DFS Rolling", solution.hasPath_approach2_dfs_rolling),
        ("Optimized BFS", solution.hasPath_approach3_optimized_bfs),
        ("Bidirectional BFS", solution.hasPath_approach4_bidirectional_bfs),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (maze, start, destination, expected) in enumerate(test_cases):
            result = func(maze, start, destination)
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} Start: {start}, Dest: {destination}, Expected: {expected}, Got: {result}")

def demonstrate_rolling_mechanics():
    """Demonstrate ball rolling mechanics"""
    print("\n=== Ball Rolling Mechanics Demo ===")
    
    maze = [[0,0,1,0,0],
            [0,0,0,0,0],
            [0,0,0,1,0],
            [1,1,0,1,1],
            [0,0,0,0,0]]
    
    start = [0, 4]
    destination = [4, 4]
    
    print("Maze (0=empty, 1=wall):")
    print_maze_visual(maze, start, destination)
    
    print(f"\nBall starts at {start}, trying to reach {destination}")
    print("Rolling mechanics: Ball rolls until it hits a wall")
    
    # Simulate rolling from start position
    m, n = len(maze), len(maze[0])
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    direction_names = ["right", "down", "left", "up"]
    
    print(f"\nFrom starting position {start}:")
    
    for i, (dr, dc) in enumerate(directions):
        row, col = start
        new_row, new_col = row, col
        
        # Simulate rolling
        while (0 <= new_row + dr < m and 0 <= new_col + dc < n and 
               maze[new_row + dr][new_col + dc] == 0):
            new_row += dr
            new_col += dc
        
        print(f"  Rolling {direction_names[i]}: {start} -> ({new_row}, {new_col})")

def print_maze_visual(maze, start, destination):
    """Print maze with visual representation"""
    m, n = len(maze), len(maze[0])
    
    for i in range(m):
        line = ""
        for j in range(n):
            if [i, j] == start:
                line += "🟢"  # Start
            elif [i, j] == destination:
                line += "🔴"  # Destination
            elif maze[i][j] == 0:
                line += "⬜"  # Empty
            else:
                line += "⬛"  # Wall
        print(f"  {line}")
    
    print("  Legend: 🟢 = Start, 🔴 = Destination, ⬜ = Empty, ⬛ = Wall")

def analyze_rolling_vs_normal_movement():
    """Analyze rolling mechanics vs normal movement"""
    print("\n=== Rolling vs Normal Movement Analysis ===")
    
    print("Rolling Mechanics (The Maze):")
    print("  🎾 Ball rolls until hitting obstacle")
    print("  🎯 Can only stop when blocked")
    print("  📍 Stopping positions are decision points")
    print("  🔄 State space = all possible stopping positions")
    
    print("\nNormal Movement (Regular Pathfinding):")
    print("  👤 Can stop at any cell")
    print("  🎯 Full control over movement")
    print("  📍 Every cell is a decision point")
    print("  🔄 State space = all reachable cells")
    
    print("\nKey Differences:")
    print("• Rolling reduces state space significantly")
    print("• Rolling mechanics add physics constraints")
    print("• Some positions reachable by rolling but not walking")
    print("• Some positions reachable by walking but not rolling")
    
    print("\nReal-world Applications:")
    print("• Physics-based games (ball/marble games)")
    print("• Robot navigation with momentum")
    print("• Sliding puzzle games")
    print("• Ice skating/sliding mechanics")
    print("• Billiards and pool game AI")
    
    print("\nAlgorithm Considerations:")
    print("• BFS still optimal for shortest 'number of stops'")
    print("• State space is stopping positions, not all positions")
    print("• Need to simulate rolling for each direction")
    print("• Visited tracking based on stopping positions")

def visualize_bfs_exploration():
    """Visualize BFS exploration in rolling maze"""
    print("\n=== BFS Exploration Visualization ===")
    
    maze = [[0,0,1],
            [0,0,0],
            [1,0,0]]
    
    start = [0, 0]
    destination = [2, 2]
    
    print("Small maze example:")
    print_maze_visual(maze, start, destination)
    
    # BFS simulation
    m, n = len(maze), len(maze[0])
    queue = deque([(start[0], start[1], 0, [start])])
    visited = {tuple(start)}
    
    print(f"\nBFS exploration:")
    
    while queue:
        row, col, steps, path = queue.popleft()
        
        print(f"\nStep {steps}: At ({row},{col})")
        print(f"  Path so far: {path}")
        
        if row == destination[0] and col == destination[1]:
            print(f"  🎯 Destination reached!")
            break
        
        # Try rolling in each direction
        next_positions = []
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            new_row, new_col = row, col
            
            # Roll until hitting wall
            while (0 <= new_row + dr < m and 0 <= new_col + dc < n and 
                   maze[new_row + dr][new_col + dc] == 0):
                new_row += dr
                new_col += dc
            
            if (new_row, new_col) not in visited:
                next_positions.append((new_row, new_col))
                visited.add((new_row, new_col))
                queue.append((new_row, new_col, steps + 1, path + [(new_row, new_col)]))
        
        print(f"  Next positions: {next_positions}")

if __name__ == "__main__":
    test_maze_path()
    demonstrate_rolling_mechanics()
    analyze_rolling_vs_normal_movement()
    visualize_bfs_exploration()

"""
Graph Theory Concepts:
1. Physics-Constrained Pathfinding
2. State Space with Movement Constraints
3. Rolling/Sliding Mechanics in Graphs
4. Modified BFS for Constrained Movement

Key Rolling Mechanics Insights:
- Ball rolls until hitting obstacle (wall or boundary)
- State space consists of stopping positions only
- BFS explores reachable stopping positions
- Movement simulation required for each direction

Algorithm Adaptations:
- Standard BFS structure maintained
- Movement step replaced with rolling simulation
- Visited tracking based on stopping positions
- Early termination when destination reached

Real-world Applications:
- Physics-based game AI
- Robot navigation with momentum
- Sliding puzzle solvers
- Ice skating path optimization
- Billiards/pool game analysis
- Marble run design

This problem demonstrates how physical constraints
modify traditional pathfinding algorithms while
maintaining core BFS optimality properties.
"""
