"""
934. Shortest Bridge
Difficulty: Medium

Problem:
You are given an n x n binary matrix grid where 1 represents land and 0 represents water.

An island is a 4-directionally connected group of 1's not connected to any other group of 1's. 
There are exactly two islands in grid.

You may change 0's to 1's to connect the two islands to form one island.

Return the smallest number of 0's you must flip to connect the two islands.

Examples:
Input: grid = [[0,1],[1,0]]
Output: 1

Input: grid = [[0,1,0],[0,0,0],[0,0,1]]
Output: 2

Input: grid = [[1,1,1,1,1],[1,0,0,0,1],[1,0,1,0,1],[1,0,0,0,1],[1,1,1,1,1]]
Output: 1

Constraints:
- n == grid.length == grid[i].length
- 2 <= n <= 100
- grid[i][j] is either 0 or 1
- There are exactly two islands
"""

from typing import List
from collections import deque

class Solution:
    def shortestBridge_approach1_dfs_bfs_combination(self, grid: List[List[int]]) -> int:
        """
        Approach 1: DFS + BFS Combination (Optimal)
        
        1. Use DFS to find and mark first island
        2. Use BFS from first island to find shortest path to second island
        
        Time: O(N^2) - DFS and BFS each visit all cells once
        Space: O(N^2) - recursion stack + queue
        """
        n = len(grid)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        def dfs_mark_island(r, c, island_cells):
            """DFS to mark all cells of first island"""
            if (r < 0 or r >= n or c < 0 or c >= n or grid[r][c] != 1):
                return
            
            grid[r][c] = 2  # Mark as part of first island
            island_cells.append((r, c))
            
            for dr, dc in directions:
                dfs_mark_island(r + dr, c + dc, island_cells)
        
        # Find and mark first island
        first_island = []
        found = False
        
        for i in range(n):
            for j in range(n):
                if grid[i][j] == 1:
                    dfs_mark_island(i, j, first_island)
                    found = True
                    break
            if found:
                break
        
        # BFS from first island to find shortest bridge
        queue = deque()
        for r, c in first_island:
            queue.append((r, c, 0))
        
        while queue:
            r, c, dist = queue.popleft()
            
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                
                if 0 <= nr < n and 0 <= nc < n:
                    if grid[nr][nc] == 1:  # Found second island
                        return dist
                    elif grid[nr][nc] == 0:  # Water - continue BFS
                        grid[nr][nc] = 2  # Mark as visited
                        queue.append((nr, nc, dist + 1))
        
        return -1  # Should not reach here given problem constraints
    
    def shortestBridge_approach2_multi_source_bfs(self, grid: List[List[int]]) -> int:
        """
        Approach 2: Multi-Source BFS
        
        Find both islands, then use multi-source BFS from one island.
        
        Time: O(N^2)
        Space: O(N^2)
        """
        n = len(grid)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        def find_island_dfs(r, c, island_cells):
            """DFS to find all cells of an island"""
            if (r < 0 or r >= n or c < 0 or c >= n or 
                grid[r][c] != 1 or (r, c) in island_cells):
                return
            
            island_cells.add((r, c))
            
            for dr, dc in directions:
                find_island_dfs(r + dr, c + dc, island_cells)
        
        # Find both islands
        islands = []
        visited_global = set()
        
        for i in range(n):
            for j in range(n):
                if grid[i][j] == 1 and (i, j) not in visited_global:
                    island = set()
                    find_island_dfs(i, j, island)
                    islands.append(island)
                    visited_global.update(island)
        
        # Multi-source BFS from first island
        queue = deque()
        visited = set()
        
        for r, c in islands[0]:
            queue.append((r, c, 0))
            visited.add((r, c))
        
        while queue:
            r, c, dist = queue.popleft()
            
            # Check if reached second island
            if (r, c) in islands[1]:
                return dist - 1  # Subtract 1 since we don't count landing on island
            
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                
                if (0 <= nr < n and 0 <= nc < n and 
                    (nr, nc) not in visited):
                    
                    visited.add((nr, nc))
                    queue.append((nr, nc, dist + 1))
        
        return -1
    
    def shortestBridge_approach3_bidirectional_bfs(self, grid: List[List[int]]) -> int:
        """
        Approach 3: Bidirectional BFS
        
        BFS from both islands simultaneously until they meet.
        
        Time: O(N^2)
        Space: O(N^2)
        """
        n = len(grid)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        def find_island_cells(start_r, start_c):
            """Find all cells of island starting from given cell"""
            cells = []
            visited = set()
            stack = [(start_r, start_c)]
            
            while stack:
                r, c = stack.pop()
                
                if ((r, c) in visited or r < 0 or r >= n or 
                    c < 0 or c >= n or grid[r][c] != 1):
                    continue
                
                visited.add((r, c))
                cells.append((r, c))
                
                for dr, dc in directions:
                    stack.append((r + dr, c + dc))
            
            return cells
        
        # Find both islands
        islands = []
        processed = set()
        
        for i in range(n):
            for j in range(n):
                if grid[i][j] == 1 and (i, j) not in processed:
                    island_cells = find_island_cells(i, j)
                    islands.append(island_cells)
                    processed.update(island_cells)
        
        # Bidirectional BFS
        queue1 = deque([(r, c, 0) for r, c in islands[0]])
        queue2 = deque([(r, c, 0) for r, c in islands[1]])
        
        visited1 = set(islands[0])
        visited2 = set(islands[1])
        
        while queue1 or queue2:
            # Expand from island 1
            if queue1:
                for _ in range(len(queue1)):
                    r, c, dist = queue1.popleft()
                    
                    if (r, c) in visited2:
                        return dist - 1
                    
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        
                        if (0 <= nr < n and 0 <= nc < n and 
                            (nr, nc) not in visited1):
                            visited1.add((nr, nc))
                            queue1.append((nr, nc, dist + 1))
            
            # Expand from island 2
            if queue2:
                for _ in range(len(queue2)):
                    r, c, dist = queue2.popleft()
                    
                    if (r, c) in visited1:
                        return dist - 1
                    
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        
                        if (0 <= nr < n and 0 <= nc < n and 
                            (nr, nc) not in visited2):
                            visited2.add((nr, nc))
                            queue2.append((nr, nc, dist + 1))
        
        return -1
    
    def shortestBridge_approach4_boundary_expansion(self, grid: List[List[int]]) -> int:
        """
        Approach 4: Island Boundary Expansion
        
        Expand island boundaries level by level until they meet.
        
        Time: O(N^2)
        Space: O(N^2)
        """
        n = len(grid)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        def mark_island_and_get_boundary(start_r, start_c, island_id):
            """Mark island with ID and return boundary cells"""
            boundary = []
            stack = [(start_r, start_c)]
            
            while stack:
                r, c = stack.pop()
                
                if (r < 0 or r >= n or c < 0 or c >= n or 
                    grid[r][c] != 1):
                    continue
                
                grid[r][c] = island_id
                
                # Check if this is a boundary cell
                is_boundary = False
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < n and 0 <= nc < n and grid[nr][nc] == 0):
                        is_boundary = True
                        break
                
                if is_boundary:
                    boundary.append((r, c))
                
                # Add neighbors to stack
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < n and 0 <= nc < n and grid[nr][nc] == 1):
                        stack.append((nr, nc))
            
            return boundary
        
        # Find first island and mark it
        first_island_boundary = None
        
        for i in range(n):
            for j in range(n):
                if grid[i][j] == 1:
                    first_island_boundary = mark_island_and_get_boundary(i, j, 2)
                    break
            if first_island_boundary:
                break
        
        # BFS expansion from boundary
        queue = deque([(r, c, 0) for r, c in first_island_boundary])
        
        while queue:
            r, c, dist = queue.popleft()
            
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                
                if 0 <= nr < n and 0 <= nc < n:
                    if grid[nr][nc] == 1:  # Reached second island
                        return dist
                    elif grid[nr][nc] == 0:  # Expand into water
                        grid[nr][nc] = 2  # Mark as visited
                        queue.append((nr, nc, dist + 1))
        
        return -1

def test_shortest_bridge():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (grid, expected)
        ([[0,1],[1,0]], 1),
        ([[0,1,0],[0,0,0],[0,0,1]], 2),
        ([[1,1,1,1,1],[1,0,0,0,1],[1,0,1,0,1],[1,0,0,0,1],[1,1,1,1,1]], 1),
        ([[1,1],[1,1]], 0),  # Already connected (edge case)
        ([[1,0,0,0,1],[0,0,0,0,0],[0,0,1,0,0]], 3),
        ([[1,0,0,1,0],[0,0,0,0,0],[0,0,0,0,0]], 3),
    ]
    
    approaches = [
        ("DFS + BFS", solution.shortestBridge_approach1_dfs_bfs_combination),
        ("Multi-Source BFS", solution.shortestBridge_approach2_multi_source_bfs),
        ("Bidirectional BFS", solution.shortestBridge_approach3_bidirectional_bfs),
        ("Boundary Expansion", solution.shortestBridge_approach4_boundary_expansion),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (grid, expected) in enumerate(test_cases):
            # Create deep copy for testing
            test_grid = [row[:] for row in grid]
            result = func(test_grid)
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} Expected: {expected}, Got: {result}")

def demonstrate_bridge_building():
    """Demonstrate bridge building process"""
    print("\n=== Bridge Building Demo ===")
    
    grid = [[0,1,0],
            [0,0,0],
            [0,0,1]]
    
    print("Original grid (1=land, 0=water):")
    print_bridge_grid(grid)
    
    # Find islands
    n = len(grid)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    def find_island(start_r, start_c, visited):
        """Find all cells of an island"""
        cells = []
        stack = [(start_r, start_c)]
        
        while stack:
            r, c = stack.pop()
            
            if ((r, c) in visited or r < 0 or r >= n or 
                c < 0 or c >= n or grid[r][c] != 1):
                continue
            
            visited.add((r, c))
            cells.append((r, c))
            
            for dr, dc in directions:
                stack.append((r + dr, c + dc))
        
        return cells
    
    # Find both islands
    visited = set()
    islands = []
    
    for i in range(n):
        for j in range(n):
            if grid[i][j] == 1 and (i, j) not in visited:
                island = find_island(i, j, visited)
                islands.append(island)
    
    print(f"\nIsland 1: {islands[0]}")
    print(f"Island 2: {islands[1]}")
    
    # BFS from first island
    queue = deque()
    distances = {}
    
    for r, c in islands[0]:
        queue.append((r, c, 0))
        distances[(r, c)] = 0
    
    print(f"\nBFS expansion from Island 1:")
    
    step = 0
    while queue:
        step += 1
        print(f"\nStep {step}:")
        size = len(queue)
        step_updates = []
        
        for _ in range(size):
            r, c, dist = queue.popleft()
            
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                
                if (0 <= nr < n and 0 <= nc < n and 
                    (nr, nc) not in distances):
                    
                    if (nr, nc) in islands[1]:
                        print(f"  🎯 Reached Island 2 at ({nr},{nc})! Distance: {dist}")
                        print(f"  Bridge length: {dist} water cells to flip")
                        return
                    
                    distances[(nr, nc)] = dist + 1
                    queue.append((nr, nc, dist + 1))
                    step_updates.append(f"({nr},{nc})={dist+1}")
        
        if step_updates:
            print(f"  Updates: {step_updates}")

def print_bridge_grid(grid):
    """Print grid with visual representation"""
    symbols = {0: "🌊", 1: "🏝️"}
    
    for row in grid:
        line = ""
        for cell in row:
            line += symbols[cell]
        print(f"  {line}")
    
    print("  Legend: 🏝️ = Land, 🌊 = Water")

def analyze_bridge_building_strategies():
    """Analyze different bridge building strategies"""
    print("\n=== Bridge Building Strategy Analysis ===")
    
    print("1. DFS + BFS Combination:")
    print("   ✅ Intuitive two-phase approach")
    print("   ✅ DFS finds first island efficiently")
    print("   ✅ BFS guarantees shortest bridge")
    print("   ❌ Modifies grid during DFS")
    print("   • Time: O(N^2), Space: O(N^2)")
    
    print("\n2. Multi-Source BFS:")
    print("   ✅ Preserves original grid structure")
    print("   ✅ Clear separation of island finding and BFS")
    print("   ✅ Flexible for multiple islands")
    print("   ❌ Requires extra space for island storage")
    print("   • Time: O(N^2), Space: O(N^2)")
    
    print("\n3. Bidirectional BFS:")
    print("   ✅ Potentially faster for large distances")
    print("   ✅ Explores from both islands simultaneously")
    print("   ✅ Optimal for symmetric scenarios")
    print("   ❌ More complex implementation")
    print("   • Time: O(N^2), Space: O(N^2)")
    
    print("\n4. Boundary Expansion:")
    print("   ✅ Natural island growth simulation")
    print("   ✅ Efficient boundary tracking")
    print("   ✅ Good for visualization")
    print("   ❌ Requires boundary identification")
    print("   • Time: O(N^2), Space: O(N^2)")
    
    print("\nKey Problem Insights:")
    print("• Two islands guaranteed by problem constraints")
    print("• Bridge length = minimum water cells to flip")
    print("• BFS naturally finds shortest path")
    print("• Island identification crucial first step")
    
    print("\nReal-world Applications:")
    print("• Civil engineering: Actual bridge construction")
    print("• Network design: Connecting isolated components")
    print("• Game development: Island connection mechanics")
    print("• Urban planning: Connecting districts")
    print("• Graph theory: Minimum edge addition problems")

if __name__ == "__main__":
    test_shortest_bridge()
    demonstrate_bridge_building()
    analyze_bridge_building_strategies()

"""
Graph Theory Concepts:
1. Connected Component Bridge Building
2. Multi-Phase Algorithm (DFS + BFS)
3. Island Identification and Connection
4. Shortest Path Between Components

Key Bridge Building Insights:
- Two-phase approach: Find islands, then connect
- DFS efficiently identifies connected components
- BFS guarantees shortest bridge between islands
- Bridge length = minimum cells to flip from 0 to 1

Algorithm Strategy:
1. Identify both islands (connected components)
2. Use multi-source BFS from one island
3. Find shortest path to reach other island
4. Return distance when islands meet

Real-world Applications:
- Civil engineering bridge design
- Network topology optimization
- Game level design
- Urban infrastructure planning
- Graph connectivity problems
- Minimum spanning tree variations

This problem demonstrates practical application of
connected component analysis with shortest path finding.
"""
