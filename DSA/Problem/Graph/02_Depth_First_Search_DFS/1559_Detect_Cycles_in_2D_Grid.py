"""
1559. Detect Cycles in 2D Grid
Difficulty: Medium

Problem:
Given a 2D array of characters grid of size m x n, you need to find if there exists any cycle 
consisting of the same value in grid.

A cycle is a path of length 4 or more in the grid that starts and ends at the same cell. 
From a given cell, you can move to one of the cells adjacent to it - in one of the four 
directions (up, down, left, or right), if it has the same value of the current cell.

Also, you cannot move to the cell that you visited in the previous move. For example, 
the cycle (1, 1) -> (1, 2) -> (1, 1) is not valid because from (1, 2) we visited (1, 1) 
which was the last visited cell.

Return true if any cycle is found, otherwise return false.

Examples:
Input: grid = [["a","a","a","a"],
               ["a","b","b","a"],
               ["a","b","b","a"],
               ["a","a","a","a"]]
Output: true

Input: grid = [["c","c","c","a"],
               ["c","d","c","c"],
               ["c","c","e","c"],
               ["f","c","c","c"]]
Output: true

Input: grid = [["a","b","b"],
               ["b","z","b"],
               ["b","b","a"]]
Output: false

Constraints:
- m == grid.length
- n == grid[i].length
- 1 <= m, n <= 500
- grid consists only of lowercase English letters
"""

from typing import List

class Solution:
    def containsCycle_approach1_dfs_parent_tracking(self, grid: List[List[str]]) -> bool:
        """
        Approach 1: DFS with Parent Tracking
        
        Use DFS and track parent to avoid immediate backtrack.
        If we reach a visited cell that's not our parent, we found a cycle.
        
        Time: O(M*N) - visit each cell once
        Space: O(M*N) - recursion stack + visited set
        """
        if not grid or not grid[0]:
            return False
        
        m, n = len(grid), len(grid[0])
        visited = set()
        
        def dfs(i, j, parent_i, parent_j, char):
            # Mark current cell as visited
            visited.add((i, j))
            
            # Check all 4 directions
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                ni, nj = i + di, j + dj
                
                # Check bounds and character match
                if (0 <= ni < m and 0 <= nj < n and grid[ni][nj] == char):
                    # Skip parent cell (avoid immediate backtrack)
                    if ni == parent_i and nj == parent_j:
                        continue
                    
                    # If visited and not parent, cycle found
                    if (ni, nj) in visited:
                        return True
                    
                    # Recursively explore
                    if dfs(ni, nj, i, j, char):
                        return True
            
            return False
        
        # Try starting DFS from each unvisited cell
        for i in range(m):
            for j in range(n):
                if (i, j) not in visited:
                    if dfs(i, j, -1, -1, grid[i][j]):
                        return True
        
        return False
    
    def containsCycle_approach2_dfs_color_states(self, grid: List[List[str]]) -> bool:
        """
        Approach 2: DFS with Color States (White-Gray-Black)
        
        Use three states: white (unvisited), gray (visiting), black (visited).
        Finding a gray node during DFS indicates a back edge (cycle).
        
        Time: O(M*N)
        Space: O(M*N)
        """
        if not grid or not grid[0]:
            return False
        
        m, n = len(grid), len(grid[0])
        # 0: white (unvisited), 1: gray (visiting), 2: black (visited)
        color = [[0] * n for _ in range(m)]
        
        def dfs(i, j, parent_i, parent_j):
            # Mark as gray (visiting)
            color[i][j] = 1
            
            # Check all 4 directions
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                ni, nj = i + di, j + dj
                
                # Check bounds, character match, and not parent
                if (0 <= ni < m and 0 <= nj < n and 
                    grid[ni][nj] == grid[i][j] and 
                    not (ni == parent_i and nj == parent_j)):
                    
                    # If gray (back edge), cycle found
                    if color[ni][nj] == 1:
                        return True
                    
                    # If white, explore recursively
                    if color[ni][nj] == 0 and dfs(ni, nj, i, j):
                        return True
            
            # Mark as black (visited)
            color[i][j] = 2
            return False
        
        # Try starting from each unvisited cell
        for i in range(m):
            for j in range(n):
                if color[i][j] == 0:
                    if dfs(i, j, -1, -1):
                        return True
        
        return False
    
    def containsCycle_approach3_union_find(self, grid: List[List[str]]) -> bool:
        """
        Approach 3: Union-Find approach
        
        For each cell, try to union with right and down neighbors.
        If union fails (already connected), we found a cycle.
        
        Time: O(M*N*α(M*N))
        Space: O(M*N)
        """
        if not grid or not grid[0]:
            return False
        
        m, n = len(grid), len(grid[0])
        
        class UnionFind:
            def __init__(self, size):
                self.parent = list(range(size))
                self.rank = [0] * size
            
            def find(self, x):
                if self.parent[x] != x:
                    self.parent[x] = self.find(self.parent[x])
                return self.parent[x]
            
            def union(self, x, y):
                px, py = self.find(x), self.find(y)
                if px == py:
                    return False  # Already connected - cycle detected
                
                if self.rank[px] < self.rank[py]:
                    px, py = py, px
                
                self.parent[py] = px
                if self.rank[px] == self.rank[py]:
                    self.rank[px] += 1
                
                return True
        
        uf = UnionFind(m * n)
        
        # Process each cell
        for i in range(m):
            for j in range(n):
                current_id = i * n + j
                
                # Check right neighbor
                if j + 1 < n and grid[i][j] == grid[i][j + 1]:
                    right_id = i * n + (j + 1)
                    if not uf.union(current_id, right_id):
                        return True
                
                # Check down neighbor
                if i + 1 < m and grid[i][j] == grid[i + 1][j]:
                    down_id = (i + 1) * n + j
                    if not uf.union(current_id, down_id):
                        return True
        
        return False
    
    def containsCycle_approach4_iterative_dfs(self, grid: List[List[str]]) -> bool:
        """
        Approach 4: Iterative DFS to avoid recursion limits
        
        Use explicit stack with parent tracking.
        
        Time: O(M*N)
        Space: O(M*N)
        """
        if not grid or not grid[0]:
            return False
        
        m, n = len(grid), len(grid[0])
        visited = set()
        
        def iterative_dfs(start_i, start_j):
            # Stack stores (i, j, parent_i, parent_j)
            stack = [(start_i, start_j, -1, -1)]
            local_visited = set()
            
            while stack:
                i, j, pi, pj = stack.pop()
                
                if (i, j) in local_visited:
                    continue
                
                local_visited.add((i, j))
                visited.add((i, j))
                
                # Check all 4 directions
                for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    ni, nj = i + di, j + dj
                    
                    if (0 <= ni < m and 0 <= nj < n and 
                        grid[ni][nj] == grid[i][j] and 
                        not (ni == pi and nj == pj)):
                        
                        # If already visited in this component, cycle found
                        if (ni, nj) in local_visited:
                            return True
                        
                        # Add to stack for exploration
                        stack.append((ni, nj, i, j))
            
            return False
        
        # Try from each unvisited cell
        for i in range(m):
            for j in range(n):
                if (i, j) not in visited:
                    if iterative_dfs(i, j):
                        return True
        
        return False

def test_detect_cycles():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (grid, expected)
        ([["a","a","a","a"],
          ["a","b","b","a"],
          ["a","b","b","a"],
          ["a","a","a","a"]], True),
        ([["c","c","c","a"],
          ["c","d","c","c"],
          ["c","c","e","c"],
          ["f","c","c","c"]], True),
        ([["a","b","b"],
          ["b","z","b"],
          ["b","b","a"]], False),
        ([["a"]], False),  # Single cell
        ([["a","b"],
          ["b","a"]], False),  # No cycle
        ([["a","a"],
          ["a","a"]], True),  # Small cycle
        ([["a","a","a"],
          ["a","b","a"],
          ["a","a","a"]], True),  # Ring cycle
    ]
    
    approaches = [
        ("DFS Parent Tracking", solution.containsCycle_approach1_dfs_parent_tracking),
        ("DFS Color States", solution.containsCycle_approach2_dfs_color_states),
        ("Union-Find", solution.containsCycle_approach3_union_find),
        ("Iterative DFS", solution.containsCycle_approach4_iterative_dfs),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (grid, expected) in enumerate(test_cases):
            # Create deep copy for testing
            test_grid = [row[:] for row in grid]
            result = func(test_grid)
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} Expected: {expected}, Got: {result}")

def demonstrate_cycle_detection():
    """Demonstrate cycle detection process"""
    print("\n=== Cycle Detection Demo ===")
    
    grid = [
        ["a","a","a"],
        ["a","b","a"],
        ["a","a","a"]
    ]
    
    print("Grid:")
    for i, row in enumerate(grid):
        print(f"  Row {i}: {row}")
    
    # Manual cycle detection trace
    m, n = len(grid), len(grid[0])
    visited = set()
    
    def dfs_trace(i, j, pi, pj, char, path):
        print(f"  Visiting ({i},{j}) with char '{char}', parent ({pi},{pj})")
        print(f"    Current path: {path}")
        
        visited.add((i, j))
        path.append((i, j))
        
        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            ni, nj = i + di, j + dj
            
            if (0 <= ni < m and 0 <= nj < n and grid[ni][nj] == char):
                if ni == pi and nj == pj:
                    print(f"    Skipping parent ({ni},{nj})")
                    continue
                
                if (ni, nj) in visited:
                    print(f"    CYCLE FOUND! Revisiting ({ni},{nj})")
                    print(f"    Cycle path: {path} -> ({ni},{nj})")
                    return True
                
                if dfs_trace(ni, nj, i, j, char, path[:]):
                    return True
        
        return False
    
    print(f"\nTracing DFS for character 'a':")
    found_cycle = False
    for i in range(m):
        for j in range(n):
            if (i, j) not in visited and grid[i][j] == 'a':
                print(f"\nStarting DFS from ({i},{j})")
                if dfs_trace(i, j, -1, -1, 'a', []):
                    found_cycle = True
                    break
        if found_cycle:
            break
    
    print(f"\nCycle found: {found_cycle}")

def visualize_cycles():
    """Create visual representation of cycles in grid"""
    print("\n=== Cycle Visualization ===")
    
    examples = [
        ("Ring Cycle", [["a","a","a"],
                       ["a","b","a"],
                       ["a","a","a"]]),
        ("No Cycle", [["a","b","a"],
                     ["b","c","b"],
                     ["a","b","a"]]),
        ("Large Cycle", [["x","x","x","x"],
                        ["x","y","y","x"],
                        ["x","y","y","x"],
                        ["x","x","x","x"]]),
    ]
    
    for name, grid in examples:
        print(f"\n{name}:")
        
        # Display grid
        for i, row in enumerate(grid):
            print(f"  Row {i}: {row}")
        
        # Check for cycles
        solution = Solution()
        has_cycle = solution.containsCycle_approach1_dfs_parent_tracking(
            [row[:] for row in grid]
        )
        
        print(f"  Has cycle: {has_cycle}")
        
        # Find connected components for each character
        m, n = len(grid), len(grid[0])
        visited = set()
        components = {}
        
        def dfs_component(i, j, char, component):
            if ((i, j) in visited or i < 0 or i >= m or 
                j < 0 or j >= n or grid[i][j] != char):
                return
            
            visited.add((i, j))
            component.append((i, j))
            
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                dfs_component(i + di, j + dj, char, component)
        
        for i in range(m):
            for j in range(n):
                if (i, j) not in visited:
                    char = grid[i][j]
                    component = []
                    dfs_component(i, j, char, component)
                    
                    if len(component) > 1:  # Only show multi-cell components
                        if char not in components:
                            components[char] = []
                        components[char].append(component)
        
        print(f"  Components by character:")
        for char, comps in components.items():
            for idx, comp in enumerate(comps):
                print(f"    '{char}' component {idx+1}: {comp}")

def analyze_cycle_properties():
    """Analyze properties of cycles in 2D grids"""
    print("\n=== Cycle Properties Analysis ===")
    
    print("Key properties of cycles in 2D grids:")
    print("1. Minimum cycle length: 4 (cannot be 3 due to grid constraints)")
    print("2. All cells in cycle must have same character")
    print("3. Cannot revisit immediate parent (avoid trivial back-and-forth)")
    print("4. Must form closed loop in 4-connected grid")
    
    print(f"\nCycle detection strategies:")
    strategies = [
        ("Parent Tracking", "Track immediate parent to avoid backtrack", "Simple, intuitive"),
        ("Color States", "White-Gray-Black coloring for back edge detection", "Classic graph algorithm"),
        ("Union-Find", "Detect cycle when connecting already connected nodes", "Good for offline processing"),
        ("Iterative DFS", "Stack-based to avoid recursion limits", "Handles large grids"),
    ]
    
    print(f"{'Strategy':<15} {'Description':<45} {'Advantage'}")
    print("-" * 75)
    
    for strategy, description, advantage in strategies:
        print(f"{strategy:<15} {description:<45} {advantage}")

def edge_cases_analysis():
    """Analyze edge cases for cycle detection"""
    print("\n=== Edge Cases Analysis ===")
    
    edge_cases = [
        ("Single Cell", [["a"]], "No cycle possible"),
        ("Two Cells", [["a","a"]], "No cycle (need minimum 4)"),
        ("Line", [["a","a","a","a"]], "No cycle (linear path)"),
        ("Corner L", [["a","a"],["a","b"]], "No cycle (L-shape too small)"),
        ("Minimal Square", [["a","a"],["a","a"]], "Has cycle (4-cell square)"),
        ("Diagonal", [["a","b"],["b","a"]], "No cycle (no same-char connections)"),
        ("Mixed Chars", [["a","b","a"],["b","a","b"],["a","b","a"]], "Multiple separate components"),
    ]
    
    solution = Solution()
    
    print(f"{'Case':<15} {'Grid':<25} {'Has Cycle':<10} {'Explanation'}")
    print("-" * 70)
    
    for case_name, grid, explanation in edge_cases:
        has_cycle = solution.containsCycle_approach1_dfs_parent_tracking(
            [row[:] for row in grid]
        )
        grid_str = str(grid)[:22] + "..." if len(str(grid)) > 25 else str(grid)
        
        print(f"{case_name:<15} {grid_str:<25} {str(has_cycle):<10} {explanation}")

if __name__ == "__main__":
    test_detect_cycles()
    demonstrate_cycle_detection()
    visualize_cycles()
    analyze_cycle_properties()
    edge_cases_analysis()

"""
Graph Theory Concepts:
1. Cycle Detection in Undirected Graphs
2. Back Edge Detection
3. Parent Tracking in DFS
4. Graph Coloring (White-Gray-Black)

Key Cycle Detection Concepts:
- Back edge: Edge to an already visited node (not parent)
- Parent tracking: Avoid immediate backtrack in undirected graphs
- Color states: Track visiting state to detect back edges
- Union-Find: Detect cycle when connecting already connected components

Algorithm Variants:
- DFS with parent tracking: Most intuitive approach
- Color-based DFS: Classic graph algorithm technique
- Union-Find: Good for offline cycle detection
- Iterative DFS: Avoids recursion depth limits

Grid-Specific Considerations:
- 4-directional movement only
- Character matching constraint
- Minimum cycle length is 4
- Parent avoidance to prevent trivial cycles

Real-world Applications:
- Circuit design (electrical loops)
- Game level validation (path loops)
- Maze design (ensuring solvability)
- Network topology analysis
- Pattern recognition in images

This problem combines classic cycle detection with grid-specific constraints,
making it an excellent example of adapting graph algorithms to specific domains.
"""
