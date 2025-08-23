"""
1559. Detect Cycles in 2D Grid - Multiple Approaches
Difficulty: Medium

Given a 2D array of characters grid of size m x n, you need to find if there 
exists any cycle consisting of the same value in grid.

A cycle is a path of length 4 or more in the grid that starts and ends at the 
same cell. From a given cell, you can move to one of the cells adjacent to it 
- in one of the four directions (up, down, left, or right), if it has the same value.

Also, you cannot move to the cell that you visited in the previous move. 
For example, the cycle (1, 1) -> (1, 2) -> (1, 1) is invalid because from 
(1, 1) we visited (1, 2) and then we cannot go back to (1, 1) right away.

Return true if any cycle is found, otherwise return false.
"""

from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, deque

class DetectCyclesIn2DGrid:
    """Multiple approaches to detect cycles in 2D grid"""
    
    def containsCycle_dfs_recursive(self, grid: List[List[str]]) -> bool:
        """
        Approach 1: DFS Recursive with Parent Tracking
        
        Use DFS to detect cycles by tracking parent to avoid immediate backtrack.
        
        Time: O(m * n)
        Space: O(m * n) for recursion stack and visited set
        """
        if not grid or not grid[0]:
            return False
        
        m, n = len(grid), len(grid[0])
        visited = set()
        
        def dfs(row: int, col: int, parent_row: int, parent_col: int, char: str) -> bool:
            """DFS to detect cycle"""
            visited.add((row, col))
            
            # Check all 4 directions
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                
                # Check bounds and character match
                if (0 <= new_row < m and 0 <= new_col < n and 
                    grid[new_row][new_col] == char):
                    
                    # Skip parent cell (avoid immediate backtrack)
                    if new_row == parent_row and new_col == parent_col:
                        continue
                    
                    # If already visited, we found a cycle
                    if (new_row, new_col) in visited:
                        return True
                    
                    # Continue DFS
                    if dfs(new_row, new_col, row, col, char):
                        return True
            
            return False
        
        # Try starting DFS from each unvisited cell
        for i in range(m):
            for j in range(n):
                if (i, j) not in visited:
                    if dfs(i, j, -1, -1, grid[i][j]):
                        return True
        
        return False
    
    def containsCycle_dfs_iterative(self, grid: List[List[str]]) -> bool:
        """
        Approach 2: DFS Iterative with Stack
        
        Use iterative DFS with explicit stack to avoid recursion limits.
        
        Time: O(m * n)
        Space: O(m * n)
        """
        if not grid or not grid[0]:
            return False
        
        m, n = len(grid), len(grid[0])
        visited = set()
        
        def has_cycle_from(start_row: int, start_col: int) -> bool:
            """Check for cycle starting from given cell"""
            char = grid[start_row][start_col]
            stack = [(start_row, start_col, -1, -1)]  # (row, col, parent_row, parent_col)
            local_visited = set()
            
            while stack:
                row, col, parent_row, parent_col = stack.pop()
                
                if (row, col) in local_visited:
                    return True
                
                local_visited.add((row, col))
                visited.add((row, col))
                
                # Check all 4 directions
                directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
                
                for dr, dc in directions:
                    new_row, new_col = row + dr, col + dc
                    
                    # Check bounds and character match
                    if (0 <= new_row < m and 0 <= new_col < n and 
                        grid[new_row][new_col] == char):
                        
                        # Skip parent cell
                        if new_row == parent_row and new_col == parent_col:
                            continue
                        
                        # If in local visited, cycle found
                        if (new_row, new_col) in local_visited:
                            return True
                        
                        # Add to stack if not globally visited
                        if (new_row, new_col) not in visited:
                            stack.append((new_row, new_col, row, col))
            
            return False
        
        # Try each unvisited cell
        for i in range(m):
            for j in range(n):
                if (i, j) not in visited:
                    if has_cycle_from(i, j):
                        return True
        
        return False
    
    def containsCycle_union_find(self, grid: List[List[str]]) -> bool:
        """
        Approach 3: Union-Find (Disjoint Set Union)
        
        Use Union-Find to detect cycles by checking if two connected cells 
        are already in the same component.
        
        Time: O(m * n * α(m * n)) where α is inverse Ackermann function
        Space: O(m * n)
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
        
        def get_id(row: int, col: int) -> int:
            """Convert 2D coordinates to 1D id"""
            return row * n + col
        
        # Process edges in a specific order to avoid parent issues
        for i in range(m):
            for j in range(n):
                current_id = get_id(i, j)
                
                # Check right neighbor
                if j + 1 < n and grid[i][j] == grid[i][j + 1]:
                    right_id = get_id(i, j + 1)
                    if not uf.union(current_id, right_id):
                        return True
                
                # Check bottom neighbor
                if i + 1 < m and grid[i][j] == grid[i + 1][j]:
                    bottom_id = get_id(i + 1, j)
                    if not uf.union(current_id, bottom_id):
                        return True
        
        return False
    
    def containsCycle_bfs(self, grid: List[List[str]]) -> bool:
        """
        Approach 4: BFS with Parent Tracking
        
        Use BFS to detect cycles by tracking parent relationships.
        
        Time: O(m * n)
        Space: O(m * n)
        """
        if not grid or not grid[0]:
            return False
        
        m, n = len(grid), len(grid[0])
        visited = set()
        
        def has_cycle_bfs(start_row: int, start_col: int) -> bool:
            """BFS to detect cycle from starting cell"""
            char = grid[start_row][start_col]
            queue = deque([(start_row, start_col, -1, -1)])  # (row, col, parent_row, parent_col)
            local_visited = set()
            
            while queue:
                row, col, parent_row, parent_col = queue.popleft()
                
                if (row, col) in local_visited:
                    continue
                
                local_visited.add((row, col))
                visited.add((row, col))
                
                # Check all 4 directions
                directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
                
                for dr, dc in directions:
                    new_row, new_col = row + dr, col + dc
                    
                    # Check bounds and character match
                    if (0 <= new_row < m and 0 <= new_col < n and 
                        grid[new_row][new_col] == char):
                        
                        # Skip parent cell
                        if new_row == parent_row and new_col == parent_col:
                            continue
                        
                        # If already visited in this component, cycle found
                        if (new_row, new_col) in local_visited:
                            return True
                        
                        # Add to queue if not globally visited
                        if (new_row, new_col) not in visited:
                            queue.append((new_row, new_col, row, col))
            
            return False
        
        # Try each unvisited cell
        for i in range(m):
            for j in range(n):
                if (i, j) not in visited:
                    if has_cycle_bfs(i, j):
                        return True
        
        return False
    
    def containsCycle_color_based(self, grid: List[List[str]]) -> bool:
        """
        Approach 5: Color-Based DFS (White-Gray-Black)
        
        Use three colors to track DFS states: unvisited, visiting, visited.
        
        Time: O(m * n)
        Space: O(m * n)
        """
        if not grid or not grid[0]:
            return False
        
        m, n = len(grid), len(grid[0])
        
        # Colors: 0 = white (unvisited), 1 = gray (visiting), 2 = black (visited)
        color = [[0] * n for _ in range(m)]
        
        def dfs(row: int, col: int, parent_row: int, parent_col: int, char: str) -> bool:
            """DFS with color tracking"""
            color[row][col] = 1  # Mark as visiting (gray)
            
            # Check all 4 directions
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                
                # Check bounds and character match
                if (0 <= new_row < m and 0 <= new_col < n and 
                    grid[new_row][new_col] == char):
                    
                    # Skip parent cell
                    if new_row == parent_row and new_col == parent_col:
                        continue
                    
                    # If gray (currently visiting), cycle found
                    if color[new_row][new_col] == 1:
                        return True
                    
                    # If white (unvisited), continue DFS
                    if color[new_row][new_col] == 0:
                        if dfs(new_row, new_col, row, col, char):
                            return True
            
            color[row][col] = 2  # Mark as visited (black)
            return False
        
        # Try each unvisited cell
        for i in range(m):
            for j in range(n):
                if color[i][j] == 0:
                    if dfs(i, j, -1, -1, grid[i][j]):
                        return True
        
        return False
    
    def containsCycle_path_compression(self, grid: List[List[str]]) -> bool:
        """
        Approach 6: Union-Find with Path Compression and Union by Rank
        
        Optimized Union-Find with both path compression and union by rank.
        
        Time: O(m * n * α(m * n))
        Space: O(m * n)
        """
        if not grid or not grid[0]:
            return False
        
        m, n = len(grid), len(grid[0])
        
        class OptimizedUnionFind:
            def __init__(self, size):
                self.parent = list(range(size))
                self.rank = [0] * size
                self.components = size
            
            def find(self, x):
                """Find with path compression"""
                if self.parent[x] != x:
                    self.parent[x] = self.find(self.parent[x])
                return self.parent[x]
            
            def union(self, x, y):
                """Union by rank with cycle detection"""
                px, py = self.find(x), self.find(y)
                
                if px == py:
                    return False  # Cycle detected
                
                # Union by rank
                if self.rank[px] < self.rank[py]:
                    px, py = py, px
                
                self.parent[py] = px
                if self.rank[px] == self.rank[py]:
                    self.rank[px] += 1
                
                self.components -= 1
                return True
            
            def connected(self, x, y):
                return self.find(x) == self.find(y)
        
        uf = OptimizedUnionFind(m * n)
        
        def get_id(row: int, col: int) -> int:
            return row * n + col
        
        # Process all edges
        for i in range(m):
            for j in range(n):
                current_id = get_id(i, j)
                
                # Check right and bottom neighbors only (to avoid duplicates)
                neighbors = []
                if j + 1 < n and grid[i][j] == grid[i][j + 1]:
                    neighbors.append((i, j + 1))
                if i + 1 < m and grid[i][j] == grid[i + 1][j]:
                    neighbors.append((i + 1, j))
                
                for ni, nj in neighbors:
                    neighbor_id = get_id(ni, nj)
                    if not uf.union(current_id, neighbor_id):
                        return True
        
        return False

def test_detect_cycles_2d_grid():
    """Test all approaches with various test cases"""
    solver = DetectCyclesIn2DGrid()
    
    test_cases = [
        # (grid, expected, description)
        ([["a","a","a","a"],["a","b","b","a"],["a","b","b","a"],["a","a","a","a"]], True, "Square cycle"),
        ([["c","c","c","a"],["c","d","c","c"],["c","c","e","c"],["f","c","c","c"]], True, "Complex cycle"),
        ([["a","b","b"],["b","z","b"],["b","b","a"]], False, "No cycle"),
        ([["a"]], False, "Single cell"),
        ([["a","a"],["a","a"]], True, "2x2 cycle"),
        ([["a","b"],["b","a"]], False, "No same character cycle"),
    ]
    
    approaches = [
        ("DFS Recursive", solver.containsCycle_dfs_recursive),
        ("DFS Iterative", solver.containsCycle_dfs_iterative),
        ("Union Find", solver.containsCycle_union_find),
        ("BFS", solver.containsCycle_bfs),
        ("Color Based", solver.containsCycle_color_based),
        ("Path Compression", solver.containsCycle_path_compression),
    ]
    
    print("=== Testing Detect Cycles in 2D Grid ===")
    
    for grid, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Grid: {grid}")
        print(f"Expected: {expected}")
        
        for approach_name, approach_func in approaches:
            try:
                result = approach_func(grid)
                status = "✓" if result == expected else "✗"
                print(f"{approach_name:18} | {status} | Result: {result}")
            except Exception as e:
                print(f"{approach_name:18} | ERROR: {str(e)[:30]}")

def demonstrate_cycle_detection_analysis():
    """Demonstrate cycle detection analysis"""
    print("\n=== Cycle Detection Analysis ===")
    
    grid = [["a","a","a","a"],["a","b","b","a"],["a","b","b","a"],["a","a","a","a"]]
    
    print("Example grid:")
    for row in grid:
        print("  " + " ".join(row))
    
    print(f"\nCycle analysis:")
    print(f"• Character 'a' forms outer cycle: (0,0)→(0,1)→(0,2)→(0,3)→(1,3)→(2,3)→(3,3)→(3,2)→(3,1)→(3,0)→(2,0)→(1,0)→(0,0)")
    print(f"• Character 'b' forms inner cycle: (1,1)→(1,2)→(2,2)→(2,1)→(1,1)")
    print(f"• Both cycles have length ≥ 4, so cycles exist")
    
    solver = DetectCyclesIn2DGrid()
    result = solver.containsCycle_dfs_recursive(grid)
    print(f"\nCycle detected: {result}")

def analyze_algorithm_complexity():
    """Analyze complexity of different approaches"""
    print("\n=== Algorithm Complexity Analysis ===")
    
    print("Approach Comparison:")
    
    print("\n1. **DFS Recursive:**")
    print("   • Time: O(m * n) - visit each cell once")
    print("   • Space: O(m * n) - recursion stack + visited set")
    print("   • Pros: Simple and intuitive")
    print("   • Cons: Stack overflow for large grids")
    
    print("\n2. **DFS Iterative:**")
    print("   • Time: O(m * n)")
    print("   • Space: O(m * n) - explicit stack")
    print("   • Pros: No recursion limit issues")
    print("   • Cons: More complex implementation")
    
    print("\n3. **Union-Find:**")
    print("   • Time: O(m * n * α(m * n)) - nearly linear")
    print("   • Space: O(m * n) - parent and rank arrays")
    print("   • Pros: Efficient for multiple queries")
    print("   • Cons: More complex for single query")
    
    print("\n4. **BFS:**")
    print("   • Time: O(m * n)")
    print("   • Space: O(m * n) - queue and visited set")
    print("   • Pros: Level-by-level exploration")
    print("   • Cons: May use more memory than DFS")
    
    print("\n5. **Color-Based DFS:**")
    print("   • Time: O(m * n)")
    print("   • Space: O(m * n) - color array")
    print("   • Pros: Clear state tracking")
    print("   • Cons: Additional memory for colors")
    
    print("\n6. **Optimized Union-Find:**")
    print("   • Time: O(m * n * α(m * n))")
    print("   • Space: O(m * n)")
    print("   • Pros: Best theoretical complexity")
    print("   • Cons: Constant factors may be higher")

def demonstrate_cycle_properties():
    """Demonstrate properties of cycles in 2D grids"""
    print("\n=== Cycle Properties in 2D Grids ===")
    
    print("Cycle Characteristics:")
    
    print("\n1. **Minimum Cycle Length:**")
    print("   • Must be at least 4 cells")
    print("   • Cannot have 3-cell cycles in grid")
    print("   • Smallest cycle is 2x2 square")
    
    print("\n2. **Same Character Requirement:**")
    print("   • All cells in cycle must have same character")
    print("   • Different characters break the cycle")
    print("   • Multiple cycles can exist for different characters")
    
    print("\n3. **Parent Constraint:**")
    print("   • Cannot immediately return to previous cell")
    print("   • Prevents trivial 2-cell back-and-forth")
    print("   • Ensures meaningful cycle detection")
    
    print("\n4. **Grid Connectivity:**")
    print("   • Only 4-directional movement allowed")
    print("   • No diagonal connections")
    print("   • Each cell has at most 4 neighbors")
    
    print("\n5. **Detection Strategies:**")
    print("   • DFS with parent tracking")
    print("   • Union-Find for component analysis")
    print("   • Color-based state tracking")
    print("   • BFS for systematic exploration")
    
    print("\nCommon Cycle Patterns:")
    print("• Rectangular cycles (most common)")
    print("• L-shaped and irregular cycles")
    print("• Nested cycles within larger regions")
    print("• Multiple disconnected cycles")

def demonstrate_optimization_techniques():
    """Demonstrate optimization techniques for cycle detection"""
    print("\n=== Optimization Techniques ===")
    
    print("Performance Optimizations:")
    
    print("\n1. **Early Termination:**")
    print("   • Return immediately when first cycle found")
    print("   • No need to explore entire grid")
    print("   • Significant speedup for grids with cycles")
    
    print("\n2. **Parent Tracking:**")
    print("   • Avoid immediate backtracking")
    print("   • Reduces false positive detections")
    print("   • Essential for correct cycle identification")
    
    print("\n3. **Visited Set Optimization:**")
    print("   • Use global visited set across components")
    print("   • Avoid re-exploring same regions")
    print("   • Reduces redundant work")
    
    print("\n4. **Union-Find Optimizations:**")
    print("   • Path compression for faster find operations")
    print("   • Union by rank for balanced trees")
    print("   • Nearly constant time operations")
    
    print("\n5. **Memory Optimizations:**")
    print("   • Reuse data structures when possible")
    print("   • Efficient coordinate to ID mapping")
    print("   • Minimize auxiliary space usage")
    
    print("\nAlgorithm Selection Guidelines:")
    print("• Small grids: DFS recursive (simplest)")
    print("• Large grids: DFS iterative (no stack overflow)")
    print("• Multiple queries: Union-Find (amortized efficiency)")
    print("• Memory constrained: Color-based (in-place marking)")

if __name__ == "__main__":
    test_detect_cycles_2d_grid()
    demonstrate_cycle_detection_analysis()
    analyze_algorithm_complexity()
    demonstrate_cycle_properties()
    demonstrate_optimization_techniques()

"""
Detect Cycles in 2D Grid - Key Insights:

1. **Problem Structure:**
   - 2D grid with character values
   - Cycle must have same character throughout
   - Minimum cycle length is 4 cells
   - No immediate backtracking allowed

2. **Algorithm Categories:**
   - Graph Traversal: DFS and BFS with parent tracking
   - Union-Find: Component-based cycle detection
   - State Tracking: Color-based DFS states
   - Optimization: Path compression and early termination

3. **Key Challenges:**
   - Avoiding trivial back-and-forth movements
   - Ensuring minimum cycle length requirement
   - Handling multiple disconnected components
   - Efficient traversal of 2D grid structure

4. **Detection Strategies:**
   - Parent tracking to prevent immediate backtrack
   - Visited set to identify when cycle closes
   - Union-Find to detect when edge creates cycle
   - Color states to track DFS progress

5. **Optimization Techniques:**
   - Early termination on first cycle found
   - Path compression in Union-Find
   - Global visited set to avoid redundant work
   - Efficient coordinate mapping

The problem demonstrates classic cycle detection adapted
for 2D grid constraints with character matching requirements.
"""
