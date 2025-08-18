"""
1730. Shortest Path to Get Food
Difficulty: Easy

Problem:
You are starving and you want to eat food as quickly as possible. You want to find the 
shortest path to arrive at any food cell.

You are given an m x n character matrix, grid, of these different types of cells:
- '*' is your location. There is exactly one '*' cell.
- '#' is a food cell. There may be multiple food cells.
- 'O' is free space, and you can travel through these cells.
- 'X' is an obstacle, and you cannot travel through these cells.

You can travel to any adjacent cell north, east, south, or west of your current location 
if there is not an obstacle.

Return the length of the shortest path for you to reach any food cell. If there is no 
path for you to reach food, return -1.

Examples:
Input: grid = [["X","X","X","X","X","X"],["X","*","O","O","O","X"],["X","O","O","#","O","X"],["X","X","X","X","X","X"]]
Output: 3

Input: grid = [["X","X","X","X","X"],["X","*","X","O","X"],["X","O","X","#","X"],["X","X","X","X","X"]]
Output: -1

Input: grid = [["X","X","X","X","X","X","X","X"],["X","*","O","X","O","#","O","X"],["X","O","O","X","O","O","X","X"],["X","O","*","O","O","#","O","X"],["X","X","X","X","X","X","X","X"]]
Output: 6

Constraints:
- m == grid.length
- n == grid[i].length
- 1 <= m, n <= 200
- grid[i][j] is '*', 'X', 'O', or '#'
- The grid contains exactly one '*'
"""

from typing import List
from collections import deque
import heapq

class Solution:
    def getFood_approach1_standard_bfs(self, grid: List[List[str]]) -> int:
        """
        Approach 1: Standard BFS (Optimal)
        
        Use BFS to find shortest path from start to any food cell.
        
        Time: O(M*N) - visit each cell at most once
        Space: O(M*N) - queue and visited set
        """
        if not grid or not grid[0]:
            return -1
        
        m, n = len(grid), len(grid[0])
        
        # Find starting position
        start_r = start_c = -1
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '*':
                    start_r, start_c = i, j
                    break
            if start_r != -1:
                break
        
        if start_r == -1:
            return -1  # No starting position found
        
        queue = deque([(start_r, start_c, 0)])  # (row, col, steps)
        visited = {(start_r, start_c)}
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        while queue:
            row, col, steps = queue.popleft()
            
            # Check if we reached food
            if grid[row][col] == '#':
                return steps
            
            # Explore adjacent cells
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                
                if (0 <= new_row < m and 0 <= new_col < n and 
                    (new_row, new_col) not in visited and 
                    grid[new_row][new_col] != 'X'):
                    
                    visited.add((new_row, new_col))
                    queue.append((new_row, new_col, steps + 1))
        
        return -1
    
    def getFood_approach2_multi_source_bfs(self, grid: List[List[str]]) -> int:
        """
        Approach 2: Multi-Source BFS (Alternative Perspective)
        
        Start BFS from all food cells and find distance to player.
        
        Time: O(M*N)
        Space: O(M*N)
        """
        if not grid or not grid[0]:
            return -1
        
        m, n = len(grid), len(grid[0])
        
        # Find player position and food positions
        player_pos = None
        food_positions = []
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '*':
                    player_pos = (i, j)
                elif grid[i][j] == '#':
                    food_positions.append((i, j))
        
        if not player_pos or not food_positions:
            return -1
        
        # Multi-source BFS from all food cells
        queue = deque()
        visited = set()
        
        for food_r, food_c in food_positions:
            queue.append((food_r, food_c, 0))
            visited.add((food_r, food_c))
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        while queue:
            row, col, steps = queue.popleft()
            
            # Check if we reached player
            if (row, col) == player_pos:
                return steps
            
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                
                if (0 <= new_row < m and 0 <= new_col < n and 
                    (new_row, new_col) not in visited and 
                    grid[new_row][new_col] != 'X'):
                    
                    visited.add((new_row, new_col))
                    queue.append((new_row, new_col, steps + 1))
        
        return -1
    
    def getFood_approach3_a_star_to_closest_food(self, grid: List[List[str]]) -> int:
        """
        Approach 3: A* to Closest Food
        
        Use A* with heuristic to nearest food cell.
        
        Time: O(M*N*log(M*N))
        Space: O(M*N)
        """
        if not grid or not grid[0]:
            return -1
        
        m, n = len(grid), len(grid[0])
        
        # Find positions
        player_pos = None
        food_positions = []
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '*':
                    player_pos = (i, j)
                elif grid[i][j] == '#':
                    food_positions.append((i, j))
        
        if not player_pos or not food_positions:
            return -1
        
        def heuristic(row, col):
            """Manhattan distance to nearest food"""
            return min(abs(row - fr) + abs(col - fc) for fr, fc in food_positions)
        
        start_r, start_c = player_pos
        pq = [(heuristic(start_r, start_c), 0, start_r, start_c)]  # (f_score, g_score, row, col)
        visited = {(start_r, start_c): 0}
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        while pq:
            f_score, g_score, row, col = heapq.heappop(pq)
            
            # Skip if we've found better path
            if (row, col) in visited and visited[(row, col)] < g_score:
                continue
            
            # Check if reached food
            if grid[row][col] == '#':
                return g_score
            
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                
                if (0 <= new_row < m and 0 <= new_col < n and 
                    grid[new_row][new_col] != 'X'):
                    
                    new_g_score = g_score + 1
                    
                    if ((new_row, new_col) not in visited or 
                        visited[(new_row, new_col)] > new_g_score):
                        
                        visited[(new_row, new_col)] = new_g_score
                        f_score = new_g_score + heuristic(new_row, new_col)
                        heapq.heappush(pq, (f_score, new_g_score, new_row, new_col))
        
        return -1
    
    def getFood_approach4_bidirectional_bfs(self, grid: List[List[str]]) -> int:
        """
        Approach 4: Bidirectional BFS
        
        Search from player and from all food cells simultaneously.
        
        Time: O(M*N)
        Space: O(M*N)
        """
        if not grid or not grid[0]:
            return -1
        
        m, n = len(grid), len(grid[0])
        
        # Find positions
        player_pos = None
        food_positions = []
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '*':
                    player_pos = (i, j)
                elif grid[i][j] == '#':
                    food_positions.append((i, j))
        
        if not player_pos or not food_positions:
            return -1
        
        # Two BFS frontiers
        player_queue = deque([player_pos])
        food_queue = deque(food_positions)
        
        player_visited = {player_pos: 0}
        food_visited = {pos: 0 for pos in food_positions}
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        while player_queue or food_queue:
            # Expand from player side
            if player_queue:
                for _ in range(len(player_queue)):
                    row, col = player_queue.popleft()
                    
                    # Check intersection with food search
                    if (row, col) in food_visited:
                        return player_visited[(row, col)] + food_visited[(row, col)]
                    
                    for dr, dc in directions:
                        new_row, new_col = row + dr, col + dc
                        
                        if (0 <= new_row < m and 0 <= new_col < n and 
                            grid[new_row][new_col] != 'X' and 
                            (new_row, new_col) not in player_visited):
                            
                            player_visited[(new_row, new_col)] = player_visited[(row, col)] + 1
                            player_queue.append((new_row, new_col))
            
            # Expand from food side
            if food_queue:
                for _ in range(len(food_queue)):
                    row, col = food_queue.popleft()
                    
                    # Check intersection with player search
                    if (row, col) in player_visited:
                        return player_visited[(row, col)] + food_visited[(row, col)]
                    
                    for dr, dc in directions:
                        new_row, new_col = row + dr, col + dc
                        
                        if (0 <= new_row < m and 0 <= new_col < n and 
                            grid[new_row][new_col] != 'X' and 
                            (new_row, new_col) not in food_visited):
                            
                            food_visited[(new_row, new_col)] = food_visited[(row, col)] + 1
                            food_queue.append((new_row, new_col))
        
        return -1

def test_shortest_path_to_food():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (grid, expected)
        ([["X","X","X","X","X","X"],["X","*","O","O","O","X"],["X","O","O","#","O","X"],["X","X","X","X","X","X"]], 3),
        ([["X","X","X","X","X"],["X","*","X","O","X"],["X","O","X","#","X"],["X","X","X","X","X"]], -1),
        ([["X","X","X","X","X","X","X","X"],["X","*","O","X","O","#","O","X"],["X","O","O","X","O","O","X","X"],["X","O","*","O","O","#","O","X"],["X","X","X","X","X","X","X","X"]], 6),
        ([["*","#"]], 1),
        ([["*","X","#"]], -1),
        ([["*","O","#","O","#"]], 2),
        ([["#","*","#"]], 1),
    ]
    
    approaches = [
        ("Standard BFS", solution.getFood_approach1_standard_bfs),
        ("Multi-Source BFS", solution.getFood_approach2_multi_source_bfs),
        ("A* to Closest Food", solution.getFood_approach3_a_star_to_closest_food),
        ("Bidirectional BFS", solution.getFood_approach4_bidirectional_bfs),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (grid, expected) in enumerate(test_cases):
            result = func(grid)
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} Expected: {expected}, Got: {result}")

def demonstrate_food_search():
    """Demonstrate food search process"""
    print("\n=== Food Search Demo ===")
    
    grid = [["X","X","X","X","X","X"],
            ["X","*","O","O","O","X"],
            ["X","O","O","#","O","X"],
            ["X","X","X","X","X","X"]]
    
    print("Grid:")
    print_food_grid(grid)
    
    # Find positions
    m, n = len(grid), len(grid[0])
    player_pos = None
    food_positions = []
    
    for i in range(m):
        for j in range(n):
            if grid[i][j] == '*':
                player_pos = (i, j)
            elif grid[i][j] == '#':
                food_positions.append((i, j))
    
    print(f"\nPlayer at: {player_pos}")
    print(f"Food at: {food_positions}")
    
    # BFS simulation
    queue = deque([(player_pos[0], player_pos[1], 0, [player_pos])])
    visited = {player_pos}
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    print(f"\nBFS exploration:")
    
    while queue:
        row, col, steps, path = queue.popleft()
        
        print(f"  Step {steps}: At ({row},{col})")
        
        # Check if reached food
        if grid[row][col] == '#':
            print(f"  🎯 Food found! Path length: {steps}")
            print(f"  Complete path: {path}")
            break
        
        # Explore neighbors
        next_moves = []
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            if (0 <= new_row < m and 0 <= new_col < n and 
                (new_row, new_col) not in visited and 
                grid[new_row][new_col] != 'X'):
                
                visited.add((new_row, new_col))
                queue.append((new_row, new_col, steps + 1, path + [(new_row, new_col)]))
                next_moves.append((new_row, new_col))
        
        if next_moves:
            print(f"    Next moves: {next_moves}")

def print_food_grid(grid):
    """Print grid with visual symbols"""
    symbols = {'*': '🧑', '#': '🍎', 'O': '⬜', 'X': '⬛'}
    
    for row in grid:
        line = ""
        for cell in row:
            line += symbols.get(cell, cell)
        print(f"  {line}")
    
    print("  Legend: 🧑 = Player, 🍎 = Food, ⬜ = Free space, ⬛ = Obstacle")

def analyze_food_search_strategies():
    """Analyze different food search strategies"""
    print("\n=== Food Search Strategy Analysis ===")
    
    print("1. Standard BFS from Player:")
    print("   ✅ Natural approach - search from current position")
    print("   ✅ Guaranteed shortest path to ANY food")
    print("   ✅ Early termination when first food reached")
    print("   ❌ Explores uniformly in all directions")
    
    print("\n2. Multi-Source BFS from Food:")
    print("   ✅ Alternative perspective - which food is closest")
    print("   ✅ Same result as standard BFS")
    print("   ✅ Useful when multiple food sources")
    print("   ❌ Might be less intuitive")
    
    print("\n3. A* with Food Heuristic:")
    print("   ✅ Guided search toward nearest food")
    print("   ✅ Often faster in practice")
    print("   ✅ Optimal with Manhattan distance heuristic")
    print("   ❌ More complex implementation")
    
    print("\n4. Bidirectional BFS:")
    print("   ✅ Reduces search space")
    print("   ✅ Especially good for large grids")
    print("   ✅ Meets in the middle approach")
    print("   ❌ Complex bookkeeping for multiple food sources")
    
    print("\nKey Considerations:")
    print("• Multiple food sources: Any food is acceptable target")
    print("• Obstacles: Must navigate around walls")
    print("• Grid boundaries: Natural movement constraints")
    print("• Distance metric: Steps/moves rather than Euclidean")
    
    print("\nReal-world Applications:")
    print("• Game AI: NPC food/resource gathering")
    print("• Robot navigation: Shortest path to charging station")
    print("• Emergency response: Nearest hospital/shelter")
    print("• Logistics: Closest warehouse/supply point")
    print("• Wildlife simulation: Foraging behavior")

def compare_single_vs_multiple_targets():
    """Compare single target vs multiple target scenarios"""
    print("\n=== Single vs Multiple Target Analysis ===")
    
    print("Single Target Pathfinding:")
    print("  🎯 One specific destination")
    print("  📍 BFS explores until target found")
    print("  ⚡ Can optimize path to specific location")
    print("  📊 Simpler state space")
    
    print("\nMultiple Target Pathfinding (like Food Search):")
    print("  🎯 Any of several acceptable destinations")
    print("  📍 BFS explores until ANY target found")
    print("  ⚡ Finds closest target automatically")
    print("  📊 More complex but often more realistic")
    
    print("\nOptimization Opportunities:")
    print("• Early termination: Stop at first target reached")
    print("• Heuristic: Distance to nearest target")
    print("• Multi-source: Start from all targets simultaneously")
    print("• Preprocessing: Precompute distances from all targets")
    
    print("\nWhen Multiple Targets Occur:")
    print("• Resource gathering (food, fuel, materials)")
    print("• Service location (hospitals, gas stations)")
    print("• Evacuation planning (multiple exits)")
    print("• Network routing (multiple servers)")

if __name__ == "__main__":
    test_shortest_path_to_food()
    demonstrate_food_search()
    analyze_food_search_strategies()
    compare_single_vs_multiple_targets()

"""
Graph Theory Concepts:
1. Multi-Target Shortest Path
2. Early Termination on Target Achievement
3. Alternative BFS Perspectives
4. Heuristic-Guided Search

Key Multi-Target Insights:
- Find shortest path to ANY of several targets
- BFS naturally finds closest target first
- Early termination when any target reached
- Multiple starting approaches possible

Algorithm Variations:
- Standard BFS: Search from player position
- Multi-source BFS: Search from all food positions
- A*: Heuristic-guided toward nearest food
- Bidirectional: Meet-in-the-middle approach

Real-world Applications:
- Resource gathering in games
- Emergency service location
- Robot navigation to charging stations
- Logistics and supply chain optimization
- Wildlife foraging simulation
- Network routing to multiple servers

This problem demonstrates BFS adaptation for
multi-target scenarios with early termination.
"""
