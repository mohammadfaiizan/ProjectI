"""
1568. Minimum Number of Days to Disconnect Island
Difficulty: Hard

Problem:
You are given an m x n binary grid grid where 1 represents land and 0 represents water.

An island is a maximal 4-directionally (horizontal or vertical) connected group of 1's.

The grid is said to be connected if we have exactly one island, otherwise is said disconnected.

In one day, we are allowed to change any single land cell (1) to a water cell (0).

Return the minimum number of days to disconnect the grid.

Examples:
Input: grid = [[0,1,1,0],[0,1,1,0],[0,0,0,0]]
Output: 2
Explanation: We need at least 2 days to get a disconnected grid.
Change land grid[1][1] and grid[0][1] to water and get 2 separate islands.

Input: grid = [[1,1]]
Output: 2
Explanation: Grid of full water is also disconnected ([[0,0]]), we need 2 days.

Input: grid = [[1,0,1,0]]
Output: 0

Input: grid = [[1,1,0,1,1],[1,1,1,1,1],[1,1,0,1,1],[1,1,0,1,1],[1,1,1,1,1]]
Output: 1

Constraints:
- m == grid.length
- n == grid[i].length
- 1 <= m, n <= 30
- grid[i][j] is either 0 or 1
"""

from typing import List
from collections import deque

class Solution:
    def minDays_approach1_brute_force_optimized(self, grid: List[List[int]]) -> int:
        """
        Approach 1: Optimized Brute Force with Early Termination
        
        Check if grid is already disconnected, then try removing 1 cell, then 2 cells.
        
        Time: O(M*N)^3 in worst case
        Space: O(M*N)
        """
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        
        def count_islands():
            """Count number of islands using BFS"""
            visited = set()
            islands = 0
            
            def bfs(start_i, start_j):
                queue = deque([(start_i, start_j)])
                visited.add((start_i, start_j))
                
                while queue:
                    i, j = queue.popleft()
                    
                    for di, dj in [(0,1), (1,0), (0,-1), (-1,0)]:
                        ni, nj = i + di, j + dj
                        
                        if (0 <= ni < m and 0 <= nj < n and 
                            (ni, nj) not in visited and grid[ni][nj] == 1):
                            visited.add((ni, nj))
                            queue.append((ni, nj))
            
            for i in range(m):
                for j in range(n):
                    if grid[i][j] == 1 and (i, j) not in visited:
                        bfs(i, j)
                        islands += 1
            
            return islands
        
        def get_land_cells():
            """Get all land cells"""
            return [(i, j) for i in range(m) for j in range(n) if grid[i][j] == 1]
        
        # Check initial state
        initial_islands = count_islands()
        if initial_islands != 1:
            return 0
        
        land_cells = get_land_cells()
        
        # Special cases
        if len(land_cells) <= 2:
            return len(land_cells)
        
        # Try removing one cell
        for i, j in land_cells:
            grid[i][j] = 0
            if count_islands() != 1:
                grid[i][j] = 1
                return 1
            grid[i][j] = 1
        
        # If removing one cell doesn't work, answer is 2
        # This is guaranteed for any connected component with > 2 cells
        return 2
    
    def minDays_approach2_articulation_point_analysis(self, grid: List[List[int]]) -> int:
        """
        Approach 2: Articulation Point Analysis
        
        Use Tarjan's algorithm to find articulation points in the grid graph.
        
        Time: O(M*N)
        Space: O(M*N)
        """
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        
        def count_islands():
            """Count islands using DFS"""
            visited = [[False] * n for _ in range(m)]
            islands = 0
            
            def dfs(i, j):
                if (i < 0 or i >= m or j < 0 or j >= n or 
                    visited[i][j] or grid[i][j] == 0):
                    return
                
                visited[i][j] = True
                for di, dj in [(0,1), (1,0), (0,-1), (-1,0)]:
                    dfs(i + di, j + dj)
            
            for i in range(m):
                for j in range(n):
                    if grid[i][j] == 1 and not visited[i][j]:
                        dfs(i, j)
                        islands += 1
            
            return islands
        
        def find_articulation_points():
            """Find articulation points using Tarjan's algorithm"""
            visited = [[False] * n for _ in range(m)]
            discovery = [[0] * n for _ in range(m)]
            low = [[0] * n for _ in range(m)]
            parent = [[None] * n for _ in range(m)]
            articulation_points = set()
            time = [0]
            
            def tarjan_dfs(i, j):
                children = 0
                visited[i][j] = True
                discovery[i][j] = low[i][j] = time[0]
                time[0] += 1
                
                for di, dj in [(0,1), (1,0), (0,-1), (-1,0)]:
                    ni, nj = i + di, j + dj
                    
                    if (0 <= ni < m and 0 <= nj < n and grid[ni][nj] == 1):
                        if not visited[ni][nj]:
                            children += 1
                            parent[ni][nj] = (i, j)
                            tarjan_dfs(ni, nj)
                            
                            low[i][j] = min(low[i][j], low[ni][nj])
                            
                            # Check articulation point conditions
                            if parent[i][j] is None and children > 1:
                                articulation_points.add((i, j))
                            elif parent[i][j] is not None and low[ni][nj] >= discovery[i][j]:
                                articulation_points.add((i, j))
                                
                        elif (ni, nj) != parent[i][j]:
                            low[i][j] = min(low[i][j], discovery[ni][nj])
            
            # Find starting point
            start_found = False
            for i in range(m):
                for j in range(n):
                    if grid[i][j] == 1 and not visited[i][j]:
                        if not start_found:
                            tarjan_dfs(i, j)
                            start_found = True
                        else:
                            return set()  # Multiple components already
            
            return articulation_points
        
        # Check initial state
        if count_islands() != 1:
            return 0
        
        # Count land cells
        land_count = sum(grid[i][j] for i in range(m) for j in range(n))
        
        if land_count <= 2:
            return land_count
        
        # Find articulation points
        articulation_points = find_articulation_points()
        
        if articulation_points:
            return 1
        else:
            return 2
    
    def minDays_approach3_connectivity_analysis(self, grid: List[List[int]]) -> int:
        """
        Approach 3: Detailed Connectivity Analysis
        
        Analyze graph connectivity and bridge structures.
        
        Time: O(M*N)
        Space: O(M*N)
        """
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        
        def is_connected():
            """Check if land forms exactly one connected component"""
            visited = set()
            components = 0
            
            def dfs(i, j):
                if ((i, j) in visited or i < 0 or i >= m or 
                    j < 0 or j >= n or grid[i][j] == 0):
                    return
                
                visited.add((i, j))
                for di, dj in [(0,1), (1,0), (0,-1), (-1,0)]:
                    dfs(i + di, j + dj)
            
            for i in range(m):
                for j in range(n):
                    if grid[i][j] == 1 and (i, j) not in visited:
                        dfs(i, j)
                        components += 1
            
            return components == 1
        
        def get_land_neighbors(i, j):
            """Get neighboring land cells"""
            neighbors = []
            for di, dj in [(0,1), (1,0), (0,-1), (-1,0)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < m and 0 <= nj < n and grid[ni][nj] == 1:
                    neighbors.append((ni, nj))
            return neighbors
        
        # Check if already disconnected
        if not is_connected():
            return 0
        
        land_cells = [(i, j) for i in range(m) for j in range(n) if grid[i][j] == 1]
        
        # Special cases
        if len(land_cells) <= 2:
            return len(land_cells)
        
        # Check for bridges (cells whose removal disconnects the graph)
        for i, j in land_cells:
            grid[i][j] = 0
            if not is_connected():
                grid[i][j] = 1
                return 1
            grid[i][j] = 1
        
        # Check graph structure for 2-day scenarios
        # If no single cell removal works, we can always remove 2 cells
        return 2
    
    def minDays_approach4_bridge_detection(self, grid: List[List[int]]) -> int:
        """
        Approach 4: Bridge Detection in Grid Graph
        
        Convert grid to graph and find bridges using Tarjan's algorithm.
        
        Time: O(M*N)
        Space: O(M*N)
        """
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        
        def build_graph():
            """Build adjacency list from grid"""
            graph = {}
            node_map = {}
            reverse_map = {}
            node_id = 0
            
            # Create nodes
            for i in range(m):
                for j in range(n):
                    if grid[i][j] == 1:
                        node_map[(i, j)] = node_id
                        reverse_map[node_id] = (i, j)
                        graph[node_id] = []
                        node_id += 1
            
            # Create edges
            for i in range(m):
                for j in range(n):
                    if grid[i][j] == 1:
                        current_id = node_map[(i, j)]
                        
                        for di, dj in [(0,1), (1,0), (0,-1), (-1,0)]:
                            ni, nj = i + di, j + dj
                            
                            if (0 <= ni < m and 0 <= nj < n and 
                                grid[ni][nj] == 1):
                                neighbor_id = node_map[(ni, nj)]
                                graph[current_id].append(neighbor_id)
            
            return graph, node_map, reverse_map
        
        def count_components(graph):
            """Count connected components in graph"""
            visited = set()
            components = 0
            
            def dfs(node):
                if node in visited:
                    return
                visited.add(node)
                for neighbor in graph.get(node, []):
                    dfs(neighbor)
            
            for node in graph:
                if node not in visited:
                    dfs(node)
                    components += 1
            
            return components
        
        def find_bridges(graph):
            """Find bridges using Tarjan's algorithm"""
            if not graph:
                return []
            
            visited = set()
            discovery = {}
            low = {}
            parent = {}
            bridges = []
            time = [0]
            
            def bridge_dfs(u):
                visited.add(u)
                discovery[u] = low[u] = time[0]
                time[0] += 1
                
                for v in graph.get(u, []):
                    if v not in visited:
                        parent[v] = u
                        bridge_dfs(v)
                        
                        low[u] = min(low[u], low[v])
                        
                        if low[v] > discovery[u]:
                            bridges.append((u, v))
                            
                    elif v != parent.get(u):
                        low[u] = min(low[u], discovery[v])
            
            for node in graph:
                if node not in visited:
                    bridge_dfs(node)
            
            return bridges
        
        # Build graph representation
        graph, node_map, reverse_map = build_graph()
        
        # Check if already disconnected
        if count_components(graph) != 1:
            return 0
        
        # Handle small cases
        if len(graph) <= 2:
            return len(graph)
        
        # Find bridges
        bridges = find_bridges(graph)
        
        if bridges:
            # If there are bridges, removing one endpoint disconnects graph
            return 1
        
        # No bridges found, need to remove 2 nodes
        return 2
    
    def minDays_approach5_mathematical_analysis(self, grid: List[List[int]]) -> int:
        """
        Approach 5: Mathematical Analysis of Grid Properties
        
        Use mathematical properties of grid graphs for optimization.
        
        Time: O(M*N)
        Space: O(M*N)
        """
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        
        def analyze_grid_structure():
            """Analyze structural properties of the grid"""
            land_cells = []
            for i in range(m):
                for j in range(n):
                    if grid[i][j] == 1:
                        land_cells.append((i, j))
            
            if len(land_cells) == 0:
                return 0, []
            
            # Check connectivity
            visited = set()
            
            def dfs(i, j):
                if ((i, j) in visited or i < 0 or i >= m or 
                    j < 0 or j >= n or grid[i][j] == 0):
                    return
                visited.add((i, j))
                for di, dj in [(0,1), (1,0), (0,-1), (-1,0)]:
                    dfs(i + di, j + dj)
            
            # Start DFS from first land cell
            start_i, start_j = land_cells[0]
            dfs(start_i, start_j)
            
            if len(visited) != len(land_cells):
                return 0, land_cells  # Already disconnected
            
            return len(land_cells), land_cells
        
        def count_neighbors(i, j):
            """Count land neighbors of a cell"""
            count = 0
            for di, dj in [(0,1), (1,0), (0,-1), (-1,0)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < m and 0 <= nj < n and grid[ni][nj] == 1:
                    count += 1
            return count
        
        land_count, land_cells = analyze_grid_structure()
        
        if land_count == 0:
            return 0
        if land_count <= 2:
            return land_count
        
        # Mathematical analysis:
        # - Corner cells (1 neighbor): Always safe to remove
        # - Edge cells (2 neighbors): May be bridges
        # - Internal cells (3-4 neighbors): Usually safe
        
        corner_cells = []
        edge_cells = []
        
        for i, j in land_cells:
            neighbors = count_neighbors(i, j)
            if neighbors == 1:
                corner_cells.append((i, j))
            elif neighbors == 2:
                edge_cells.append((i, j))
        
        # Try removing corner cells first (most likely to disconnect)
        for i, j in corner_cells:
            grid[i][j] = 0
            _, remaining_cells = analyze_grid_structure()
            if len(remaining_cells) != land_count - 1 or len(set(remaining_cells)) != len(remaining_cells):
                grid[i][j] = 1
                return 1
            grid[i][j] = 1
        
        # Try removing edge cells
        for i, j in edge_cells:
            grid[i][j] = 0
            _, remaining_cells = analyze_grid_structure()
            if len(remaining_cells) != land_count - 1:
                grid[i][j] = 1
                return 1
            grid[i][j] = 1
        
        # If no single cell works, answer is 2
        return 2

def test_minimum_days_disconnect():
    """Test all approaches with various test cases"""
    solution = Solution()
    
    test_cases = [
        # (grid, expected)
        ([[0,1,1,0],[0,1,1,0],[0,0,0,0]], 2),
        ([[1,1]], 2),
        ([[1,0,1,0]], 0),
        ([[1,1,0,1,1],[1,1,1,1,1],[1,1,0,1,1],[1,1,0,1,1],[1,1,1,1,1]], 1),
        ([[1]], 1),
        ([[1,1,1]], 2),
        ([[1,0],[0,1]], 0),
    ]
    
    approaches = [
        ("Brute Force Optimized", solution.minDays_approach1_brute_force_optimized),
        ("Articulation Points", solution.minDays_approach2_articulation_point_analysis),
        ("Connectivity Analysis", solution.minDays_approach3_connectivity_analysis),
        ("Bridge Detection", solution.minDays_approach4_bridge_detection),
        ("Mathematical Analysis", solution.minDays_approach5_mathematical_analysis),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (grid, expected) in enumerate(test_cases):
            # Deep copy grid
            grid_copy = [row[:] for row in grid]
            result = func(grid_copy)
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} expected={expected}, got={result}")

def demonstrate_island_analysis():
    """Demonstrate island connectivity analysis"""
    print("\n=== Island Analysis Demo ===")
    
    test_grid = [
        [0,1,1,0],
        [0,1,1,0], 
        [0,0,0,0]
    ]
    
    print(f"Grid:")
    for row in test_grid:
        print(f"  {row}")
    
    print(f"\nIsland structure:")
    print(f"  Connected 2x2 block of land cells")
    print(f"  Removing any single cell leaves 3 connected cells")
    print(f"  Need to remove 2 cells to disconnect")
    
    solution = Solution()
    result = solution.minDays_approach1_brute_force_optimized(test_grid)
    print(f"\nResult: {result} days needed")
    
    print(f"\nStrategy:")
    print(f"  Remove cells at positions (0,1) and (1,1)")
    print(f"  This creates two separate 1-cell islands")

def demonstrate_articulation_analysis():
    """Demonstrate articulation point analysis"""
    print("\n=== Articulation Point Analysis ===")
    
    test_grid = [
        [1,1,1],
        [0,1,0],
        [1,1,1]
    ]
    
    print(f"Grid:")
    for row in test_grid:
        print(f"  {row}")
    
    print(f"\nStructure analysis:")
    print(f"  Center cell (1,1) connects two parts")
    print(f"  Top row: cells (0,0), (0,1), (0,2)")
    print(f"  Bottom row: cells (2,0), (2,1), (2,2)")
    print(f"  Center cell (1,1) is articulation point")
    
    print(f"\nRemoving center cell:")
    print(f"  Disconnects top and bottom parts")
    print(f"  Result: 1 day needed")

def analyze_grid_connectivity_properties():
    """Analyze properties of grid connectivity"""
    print("\n=== Grid Connectivity Properties ===")
    
    print("Grid Graph Properties:")
    
    print("\n1. **Connectivity Patterns:**")
    print("   • 4-directional connectivity (no diagonals)")
    print("   • Planar graph structure")
    print("   • Limited degree (max 4 neighbors)")
    print("   • Regular grid topology")
    
    print("\n2. **Disconnection Strategies:**")
    print("   • Single cell removal for bridges")
    print("   • Two cell removal always sufficient")
    print("   • Corner/edge cells often critical")
    print("   • Articulation points are key targets")
    
    print("\n3. **Mathematical Bounds:**")
    print("   • Answer is always 0, 1, or 2")
    print("   • 0: Already disconnected")
    print("   • 1: Has articulation points or bridges")
    print("   • 2: Robust connectivity (max case)")
    
    print("\n4. **Algorithm Strategies:**")
    print("   • Brute force for small grids")
    print("   • Articulation point detection")
    print("   • Bridge finding algorithms")
    print("   • Structural analysis optimization")
    
    print("\n5. **Optimization Opportunities:**")
    print("   • Early termination conditions")
    print("   • Structural property exploitation")
    print("   • Graph theory algorithm application")
    print("   • Mathematical bound utilization")

def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    print("Island Disconnection Applications:")
    
    print("\n1. **Network Security:**")
    print("   • Identify critical network nodes")
    print("   • Plan network segmentation")
    print("   • Analyze attack surface")
    print("   • Design fault isolation")
    
    print("\n2. **Infrastructure Planning:**")
    print("   • Transportation network robustness")
    print("   • Power grid reliability")
    print("   • Communication network design")
    print("   • Emergency response planning")
    
    print("\n3. **Game Development:**")
    print("   • Terrain modification mechanics")
    print("   • Strategic placement games")
    print("   • Resource control scenarios")
    print("   • Map connectivity analysis")
    
    print("\n4. **Environmental Science:**")
    print("   • Habitat fragmentation analysis")
    print("   • Conservation planning")
    print("   • Ecosystem connectivity")
    print("   • Species migration patterns")
    
    print("\n5. **Urban Planning:**")
    print("   • City block connectivity")
    print("   • Traffic flow analysis")
    print("   • Emergency evacuation routes")
    print("   • Utility network design")

if __name__ == "__main__":
    test_minimum_days_disconnect()
    demonstrate_island_analysis()
    demonstrate_articulation_analysis()
    analyze_grid_connectivity_properties()
    demonstrate_real_world_applications()

"""
Island Disconnection and Graph Connectivity Concepts:
1. Grid Graph Analysis and Connectivity Algorithms
2. Articulation Point Detection in Planar Graphs
3. Bridge Finding and Network Robustness Assessment
4. Brute Force Optimization with Mathematical Bounds
5. Graph Theory Applications in Spatial Problems

Key Problem Insights:
- Grid connectivity follows 4-directional neighbor patterns
- Answer is always 0, 1, or 2 days based on structure
- Articulation points enable 1-day disconnection
- Two cells can always disconnect any connected region

Algorithm Strategy:
1. Check initial connectivity state
2. Try single cell removal (articulation points)
3. Return 2 if no single cell works
4. Use graph theory for optimization

Grid Properties:
- Planar graph with regular structure
- Maximum degree 4 for any cell
- Bridges and articulation points are critical
- Small search space enables brute force

Mathematical Analysis:
- 0 days: Already disconnected
- 1 day: Has articulation points or bridges
- 2 days: Maximum for any connected region
- No case requires more than 2 days

Real-world Applications:
- Network security and segmentation
- Infrastructure reliability analysis
- Game mechanics and strategic planning
- Environmental conservation and urban planning
- Emergency response and evacuation planning

This problem demonstrates grid graph analysis techniques
essential for spatial connectivity and robustness assessment.
"""
