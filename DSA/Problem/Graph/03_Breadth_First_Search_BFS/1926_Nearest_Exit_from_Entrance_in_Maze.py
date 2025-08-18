"""
1926. Nearest Exit from Entrance in Maze
Difficulty: Easy

Problem:
You are given an m x n matrix maze (0-indexed) with empty cells (represented as '.') and 
walls (represented as '+'). You are also given the entrance of the maze, where 
entrance = [entrancerow, entrancecol] denotes the row and column of the cell you are 
initially standing at.

In one step, you can move one cell up, down, left, or right. You cannot step into a 
cell with a wall, and you cannot step outside of the maze. Your goal is to find the 
nearest exit from the maze. An exit is defined as an empty cell that is at the border 
of the maze. The entrance does not count as an exit.

Return the number of steps in the shortest path from the entrance to the nearest exit, 
or -1 if no such path exists.

Examples:
Input: maze = [["+","+",".","+"],[".",".",".","+"],["+","+","+","."]], entrance = [1,2]
Output: 1

Input: maze = [["+","+","+"],[".",".","."],["+","+","+"]], entrance = [1,1]
Output: 2

Input: maze = [[".","+","+"]], entrance = [0,0]
Output: -1

Constraints:
- maze.length == m
- maze[i].length == n
- 1 <= m, n <= 100
- maze[i][j] is either '.' or '+'
- entrance.length == 2
- 0 <= entrancerow < m and 0 <= entrancecol < n
- entrance will always be an empty cell
"""

from typing import List
from collections import deque

class Solution:
    def nearestExit_approach1_standard_bfs(self, maze: List[List[str]], entrance: List[int]) -> int:
        """
        Approach 1: Standard BFS (Optimal)
        
        Use BFS to find shortest path from entrance to any border cell.
        BFS guarantees minimum steps due to level-order exploration.
        
        Time: O(M*N) - visit each cell at most once
        Space: O(M*N) - queue and visited set
        """
        if not maze or not maze[0]:
            return -1
        
        m, n = len(maze), len(maze[0])
        start_r, start_c = entrance
        
        # BFS setup
        queue = deque([(start_r, start_c, 0)])  # (row, col, steps)
        visited = {(start_r, start_c)}
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        def is_exit(r, c):
            """Check if cell is an exit (border cell, not entrance)"""
            return ((r == 0 or r == m-1 or c == 0 or c == n-1) and 
                    (r, c) != (start_r, start_c))
        
        while queue:
            r, c, steps = queue.popleft()
            
            # Check if current cell is an exit
            if is_exit(r, c):
                return steps
            
            # Explore all 4 directions
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                
                if (0 <= nr < m and 0 <= nc < n and 
                    (nr, nc) not in visited and maze[nr][nc] == '.'):
                    visited.add((nr, nc))
                    queue.append((nr, nc, steps + 1))
        
        return -1  # No exit found
    
    def nearestExit_approach2_level_by_level_bfs(self, maze: List[List[str]], entrance: List[int]) -> int:
        """
        Approach 2: Level-by-Level BFS
        
        Process all cells at current distance before moving to next distance.
        
        Time: O(M*N)
        Space: O(M*N)
        """
        if not maze or not maze[0]:
            return -1
        
        m, n = len(maze), len(maze[0])
        start_r, start_c = entrance
        
        queue = deque([(start_r, start_c)])
        visited = {(start_r, start_c)}
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        steps = 0
        
        def is_exit(r, c):
            return ((r == 0 or r == m-1 or c == 0 or c == n-1) and 
                    (r, c) != (start_r, start_c))
        
        while queue:
            steps += 1
            size = len(queue)
            
            # Process all cells at current level
            for _ in range(size):
                r, c = queue.popleft()
                
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    
                    if (0 <= nr < m and 0 <= nc < n and 
                        (nr, nc) not in visited and maze[nr][nc] == '.'):
                        
                        if is_exit(nr, nc):
                            return steps
                        
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        
        return -1
    
    def nearestExit_approach3_bidirectional_bfs(self, maze: List[List[str]], entrance: List[int]) -> int:
        """
        Approach 3: Bidirectional BFS
        
        Start BFS from entrance and from all exits simultaneously.
        
        Time: O(M*N)
        Space: O(M*N)
        """
        if not maze or not maze[0]:
            return -1
        
        m, n = len(maze), len(maze[0])
        start_r, start_c = entrance
        
        # Find all exits
        exits = []
        for i in range(m):
            for j in range(n):
                if (maze[i][j] == '.' and 
                    (i == 0 or i == m-1 or j == 0 or j == n-1) and 
                    (i, j) != (start_r, start_c)):
                    exits.append((i, j))
        
        if not exits:
            return -1
        
        # BFS from entrance
        queue_start = deque([(start_r, start_c, 0)])
        visited_start = {(start_r, start_c)}
        
        # BFS from exits
        queue_exit = deque([(r, c, 0) for r, c in exits])
        visited_exit = set(exits)
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        while queue_start or queue_exit:
            # Expand from entrance side
            if queue_start:
                for _ in range(len(queue_start)):
                    r, c, steps = queue_start.popleft()
                    
                    if (r, c) in visited_exit:
                        return steps
                    
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        
                        if (0 <= nr < m and 0 <= nc < n and 
                            (nr, nc) not in visited_start and maze[nr][nc] == '.'):
                            visited_start.add((nr, nc))
                            queue_start.append((nr, nc, steps + 1))
            
            # Expand from exit side
            if queue_exit:
                for _ in range(len(queue_exit)):
                    r, c, steps = queue_exit.popleft()
                    
                    if (r, c) in visited_start:
                        return steps
                    
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        
                        if (0 <= nr < m and 0 <= nc < n and 
                            (nr, nc) not in visited_exit and maze[nr][nc] == '.'):
                            visited_exit.add((nr, nc))
                            queue_exit.append((nr, nc, steps + 1))
        
        return -1
    
    def nearestExit_approach4_optimized_border_check(self, maze: List[List[str]], entrance: List[int]) -> int:
        """
        Approach 4: Optimized with Early Border Detection
        
        Check if we reach border immediately during BFS expansion.
        
        Time: O(M*N)
        Space: O(M*N)
        """
        if not maze or not maze[0]:
            return -1
        
        m, n = len(maze), len(maze[0])
        start_r, start_c = entrance
        
        queue = deque([(start_r, start_c, 0)])
        visited = [[False] * n for _ in range(m)]
        visited[start_r][start_c] = True
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        while queue:
            r, c, steps = queue.popleft()
            
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                
                # Check bounds and wall
                if (nr < 0 or nr >= m or nc < 0 or nc >= n or 
                    maze[nr][nc] == '+' or visited[nr][nc]):
                    continue
                
                # Check if this is an exit (reached border)
                if (nr == 0 or nr == m-1 or nc == 0 or nc == n-1):
                    return steps + 1
                
                visited[nr][nc] = True
                queue.append((nr, nc, steps + 1))
        
        return -1

def test_nearest_exit():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (maze, entrance, expected)
        ([["+","+",".","+"],[".",".",".","+"],["+","+","+","."]], [1,2], 1),
        ([["+","+","+"],[".",".","."],["+"," +","+"]], [1,1], 2),
        ([[".","+","+"]], [0,0], -1),
        ([["+"]], [0,0], -1),  # Invalid case (entrance can't be wall)
        ([[".","."]], [0,0], 1),
        ([[".",".","."],[".",".","."],[".",".","."]], [1,1], 2),
        ([[".","+"],["+","."]], [0,0], 3),
    ]
    
    approaches = [
        ("Standard BFS", solution.nearestExit_approach1_standard_bfs),
        ("Level-by-Level BFS", solution.nearestExit_approach2_level_by_level_bfs),
        ("Bidirectional BFS", solution.nearestExit_approach3_bidirectional_bfs),
        ("Optimized Border Check", solution.nearestExit_approach4_optimized_border_check),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (maze, entrance, expected) in enumerate(test_cases):
            try:
                # Skip invalid test case for certain approaches
                if maze == [["+"]] and entrance == [0,0]:
                    print(f"Test {i+1}: SKIP (invalid entrance)")
                    continue
                
                result = func(maze, entrance)
                status = "✓" if result == expected else "✗"
                print(f"Test {i+1}: {status} Entrance: {entrance}, Expected: {expected}, Got: {result}")
            except Exception as e:
                print(f"Test {i+1}: ERROR - {e}")

def demonstrate_maze_navigation():
    """Demonstrate maze navigation process"""
    print("\n=== Maze Navigation Demo ===")
    
    maze = [["+","+",".","+"],[".",".",".","+"],["+","+","+","."]]
    entrance = [1, 2]
    
    print("Maze layout:")
    print_maze(maze, entrance)
    
    # BFS simulation
    m, n = len(maze), len(maze[0])
    start_r, start_c = entrance
    
    queue = deque([(start_r, start_c, 0)])
    visited = {(start_r, start_c)}
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    path = {}  # To track the path
    
    print(f"\nBFS exploration from entrance {entrance}:")
    
    while queue:
        r, c, steps = queue.popleft()
        
        print(f"  Step {steps}: Exploring ({r},{c})")
        
        # Check if this is an exit
        if ((r == 0 or r == m-1 or c == 0 or c == n-1) and 
            (r, c) != (start_r, start_c)):
            print(f"  🎯 EXIT FOUND at ({r},{c}) in {steps} steps!")
            break
        
        # Explore neighbors
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            
            if (0 <= nr < m and 0 <= nc < n and 
                (nr, nc) not in visited and maze[nr][nc] == '.'):
                visited.add((nr, nc))
                queue.append((nr, nc, steps + 1))
                path[(nr, nc)] = (r, c)
                print(f"    Adding ({nr},{nc}) to queue")

def print_maze(maze, entrance):
    """Helper function to print maze with entrance marked"""
    m, n = len(maze), len(maze[0])
    
    for i in range(m):
        row = ""
        for j in range(n):
            if [i, j] == entrance:
                row += "🚪"  # Entrance
            elif maze[i][j] == '.':
                if i == 0 or i == m-1 or j == 0 or j == n-1:
                    row += "🚪"  # Exit
                else:
                    row += "⬜"  # Empty
            else:
                row += "⬛"  # Wall
        print(f"  {row}")
    
    print("  Legend: 🚪 = Entrance/Exit, ⬜ = Empty, ⬛ = Wall")

def analyze_exit_finding_strategy():
    """Analyze the exit finding strategy"""
    print("\n=== Exit Finding Strategy Analysis ===")
    
    print("Why BFS is optimal for maze exit finding:")
    print("1. 🎯 Shortest path: BFS guarantees minimum steps")
    print("2. 📊 Level exploration: Explores by distance from start")
    print("3. 🔍 First exit found: First border cell reached is nearest")
    print("4. ⚡ Efficient: O(M*N) time, visits each cell once")
    
    print("\nBFS vs other approaches:")
    print("• DFS: ❌ May find longer path to exit")
    print("• Dijkstra: ❌ Overkill (unweighted graph)")
    print("• A*: ❌ No clear heuristic for 'nearest exit'")
    print("• BFS: ✅ Perfect for unweighted shortest path")
    
    print("\nKey implementation insights:")
    print("• Check for exit during neighbor exploration")
    print("• Border cells are potential exits (except entrance)")
    print("• Use visited set to avoid cycles")
    print("• Track steps/distance during BFS")
    print("• Return immediately when first exit found")

if __name__ == "__main__":
    test_nearest_exit()
    demonstrate_maze_navigation()
    analyze_exit_finding_strategy()

"""
Graph Theory Concepts:
1. Shortest Path in Unweighted Graph
2. BFS for Minimum Distance
3. Target Detection During Traversal
4. Maze Navigation Algorithms

Key BFS Insights:
- BFS guarantees shortest path in unweighted graphs
- Level-order exploration ensures minimum steps
- Exit detection: Check if reached border during exploration
- Early termination: Return immediately when exit found

Algorithm Strategy:
- Start BFS from entrance position
- Explore all reachable empty cells level by level
- Check if each cell is an exit (border cell ≠ entrance)
- Return steps when first exit is reached

Real-world Applications:
- Robot navigation and pathfinding
- Game AI for maze solving
- Emergency exit planning
- Network routing algorithms
- GPS navigation in restricted areas
- Escape route optimization in buildings

This problem demonstrates BFS for target-finding in constrained
environments, a fundamental pattern in pathfinding algorithms.
"""
