"""
Easy Shortest Path Problems Collection
Difficulty: Easy

This file contains a collection of fundamental shortest path problems that demonstrate
core concepts and serve as building blocks for more advanced algorithms.

Problems included:
1. Simple BFS Shortest Path
2. Shortest Path in Binary Matrix
3. Knight's Shortest Path
4. Minimum Steps to Reach Target
5. Island Perimeter (Path-related)
6. Shortest Bridge
7. Word Ladder Length
8. Jump Game Analysis
"""

from typing import List, Tuple, Set
from collections import deque
import heapq

class Solution:
    def shortestPathBFS_problem1(self, graph: List[List[int]], start: int, end: int) -> int:
        """
        Problem 1: Simple BFS Shortest Path in Unweighted Graph
        
        Find shortest path length between start and end in unweighted graph.
        
        Time: O(V + E)
        Space: O(V)
        """
        if start == end:
            return 0
        
        n = len(graph)
        visited = [False] * n
        queue = deque([(start, 0)])
        visited[start] = True
        
        while queue:
            node, dist = queue.popleft()
            
            for neighbor in graph[node]:
                if neighbor == end:
                    return dist + 1
                
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append((neighbor, dist + 1))
        
        return -1  # No path found
    
    def shortestPathBinaryMatrix_problem2(self, grid: List[List[int]]) -> int:
        """
        Problem 2: Shortest Path in Binary Matrix (8-directional)
        
        Find shortest path from top-left to bottom-right in binary matrix.
        
        Time: O(N^2)
        Space: O(N^2)
        """
        n = len(grid)
        
        if grid[0][0] == 1 or grid[n-1][n-1] == 1:
            return -1
        
        if n == 1:
            return 1
        
        # 8 directions: up, down, left, right, and 4 diagonals
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        queue = deque([(0, 0, 1)])  # (row, col, path_length)
        visited = set([(0, 0)])
        
        while queue:
            row, col, path_length = queue.popleft()
            
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                
                if (0 <= new_row < n and 0 <= new_col < n and 
                    grid[new_row][new_col] == 0 and 
                    (new_row, new_col) not in visited):
                    
                    if new_row == n - 1 and new_col == n - 1:
                        return path_length + 1
                    
                    visited.add((new_row, new_col))
                    queue.append((new_row, new_col, path_length + 1))
        
        return -1
    
    def shortestPathKnight_problem3(self, n: int, start: List[int], end: List[int]) -> int:
        """
        Problem 3: Knight's Shortest Path on Chessboard
        
        Find minimum moves for knight to reach target position.
        
        Time: O(N^2)
        Space: O(N^2)
        """
        if start == end:
            return 0
        
        # Knight moves: 8 possible L-shaped moves
        knight_moves = [(-2,-1), (-2,1), (-1,-2), (-1,2), (1,-2), (1,2), (2,-1), (2,1)]
        
        queue = deque([(start[0], start[1], 0)])
        visited = set([(start[0], start[1])])
        
        while queue:
            row, col, moves = queue.popleft()
            
            for dr, dc in knight_moves:
                new_row, new_col = row + dr, col + dc
                
                if (0 <= new_row < n and 0 <= new_col < n and 
                    (new_row, new_col) not in visited):
                    
                    if new_row == end[0] and new_col == end[1]:
                        return moves + 1
                    
                    visited.add((new_row, new_col))
                    queue.append((new_row, new_col, moves + 1))
        
        return -1  # Should not happen on finite chessboard
    
    def minStepsToTarget_problem4(self, target: int) -> int:
        """
        Problem 4: Minimum Steps to Reach Target
        
        Starting from 0, each step can +1 or -1. Find minimum steps to reach target.
        
        Time: O(target)
        Space: O(target)
        """
        if target == 0:
            return 0
        
        target = abs(target)  # Symmetric problem
        
        queue = deque([(0, 0)])  # (position, steps)
        visited = set([0])
        
        while queue:
            pos, steps = queue.popleft()
            
            # Two possible moves: +1 or -1
            for next_pos in [pos + 1, pos - 1]:
                if next_pos == target:
                    return steps + 1
                
                # Pruning: don't go too far beyond target
                if next_pos not in visited and abs(next_pos) <= target + steps:
                    visited.add(next_pos)
                    queue.append((next_pos, steps + 1))
        
        return -1
    
    def islandPerimeter_problem5(self, grid: List[List[int]]) -> int:
        """
        Problem 5: Island Perimeter (Path-related analysis)
        
        Calculate perimeter of island in grid.
        
        Time: O(M * N)
        Space: O(1)
        """
        rows, cols = len(grid), len(grid[0])
        perimeter = 0
        
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    # Check all 4 directions
                    for di, dj in [(0,1), (1,0), (0,-1), (-1,0)]:
                        ni, nj = i + di, j + dj
                        
                        # Add to perimeter if edge or water
                        if (ni < 0 or ni >= rows or nj < 0 or nj >= cols or 
                            grid[ni][nj] == 0):
                            perimeter += 1
        
        return perimeter
    
    def shortestBridge_problem6(self, grid: List[List[int]]) -> int:
        """
        Problem 6: Shortest Bridge Between Two Islands
        
        Find shortest bridge (path) to connect two islands.
        
        Time: O(N^2)
        Space: O(N^2)
        """
        n = len(grid)
        
        def dfs_find_island(start_i, start_j, island_cells):
            """DFS to find all cells of first island"""
            stack = [(start_i, start_j)]
            
            while stack:
                i, j = stack.pop()
                
                if (i < 0 or i >= n or j < 0 or j >= n or 
                    grid[i][j] != 1 or (i, j) in island_cells):
                    continue
                
                island_cells.add((i, j))
                grid[i][j] = 2  # Mark as visited
                
                # Explore 4 directions
                for di, dj in [(0,1), (1,0), (0,-1), (-1,0)]:
                    stack.append((i + di, j + dj))
        
        # Find first island
        island1 = set()
        found_first = False
        
        for i in range(n):
            for j in range(n):
                if grid[i][j] == 1 and not found_first:
                    dfs_find_island(i, j, island1)
                    found_first = True
                    break
            if found_first:
                break
        
        # BFS from first island to find shortest path to second island
        queue = deque()
        for i, j in island1:
            queue.append((i, j, 0))
        
        while queue:
            i, j, dist = queue.popleft()
            
            for di, dj in [(0,1), (1,0), (0,-1), (-1,0)]:
                ni, nj = i + di, j + dj
                
                if 0 <= ni < n and 0 <= nj < n:
                    if grid[ni][nj] == 1:  # Found second island
                        return dist
                    elif grid[ni][nj] == 0:  # Water, can expand
                        grid[ni][nj] = 2  # Mark as visited
                        queue.append((ni, nj, dist + 1))
        
        return -1
    
    def ladderLength_problem7(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        """
        Problem 7: Word Ladder Length
        
        Find shortest transformation sequence from beginWord to endWord.
        
        Time: O(M^2 * N) where M = word length, N = word count
        Space: O(M * N)
        """
        if endWord not in wordList:
            return 0
        
        word_set = set(wordList)
        queue = deque([(beginWord, 1)])
        
        while queue:
            word, length = queue.popleft()
            
            if word == endWord:
                return length
            
            # Try all possible single character changes
            for i in range(len(word)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    if c != word[i]:
                        new_word = word[:i] + c + word[i+1:]
                        
                        if new_word in word_set:
                            word_set.remove(new_word)  # Avoid revisiting
                            queue.append((new_word, length + 1))
        
        return 0
    
    def canJumpAnalysis_problem8(self, nums: List[int]) -> Tuple[bool, int]:
        """
        Problem 8: Jump Game Analysis (Path-based)
        
        Determine if can reach end and minimum jumps needed.
        
        Time: O(N)
        Space: O(1)
        """
        n = len(nums)
        
        # Check if reachable
        max_reach = 0
        for i in range(n):
            if i > max_reach:
                return False, -1  # Cannot reach this position
            max_reach = max(max_reach, i + nums[i])
            if max_reach >= n - 1:
                break
        
        can_reach = max_reach >= n - 1
        
        if not can_reach:
            return False, -1
        
        # Find minimum jumps
        jumps = 0
        current_reach = 0
        next_reach = 0
        
        for i in range(n - 1):
            next_reach = max(next_reach, i + nums[i])
            
            if i == current_reach:
                jumps += 1
                current_reach = next_reach
        
        return True, jumps

def test_easy_shortest_path_problems():
    """Test all easy shortest path problems"""
    solution = Solution()
    
    print("=== Testing Easy Shortest Path Problems ===")
    
    # Test Problem 1: Simple BFS
    graph = [[1, 2], [0, 3], [0, 3], [1, 2]]
    result = solution.shortestPathBFS_problem1(graph, 0, 3)
    print(f"Problem 1 - Simple BFS: {result} (expected: 2)")
    
    # Test Problem 2: Binary Matrix
    grid = [[0,0,0],[1,1,0],[1,1,0]]
    result = solution.shortestPathBinaryMatrix_problem2(grid)
    print(f"Problem 2 - Binary Matrix: {result} (expected: 4)")
    
    # Test Problem 3: Knight's Path
    result = solution.shortestPathKnight_problem3(8, [0,0], [7,7])
    print(f"Problem 3 - Knight's Path: {result} (expected: 6)")
    
    # Test Problem 4: Min Steps to Target
    result = solution.minStepsToTarget_problem4(3)
    print(f"Problem 4 - Min Steps to Target: {result} (expected: 3)")
    
    # Test Problem 5: Island Perimeter
    grid = [[0,1,0,0],[1,1,1,0],[0,1,0,0],[1,1,0,0]]
    result = solution.islandPerimeter_problem5(grid)
    print(f"Problem 5 - Island Perimeter: {result} (expected: 16)")
    
    # Test Problem 6: Shortest Bridge
    grid = [[0,1],[1,0]]
    result = solution.shortestBridge_problem6([row[:] for row in grid])
    print(f"Problem 6 - Shortest Bridge: {result} (expected: 1)")
    
    # Test Problem 7: Word Ladder
    result = solution.ladderLength_problem7("hit", "cog", ["hot","dot","dog","lot","log","cog"])
    print(f"Problem 7 - Word Ladder: {result} (expected: 5)")
    
    # Test Problem 8: Jump Game
    can_reach, min_jumps = solution.canJumpAnalysis_problem8([2,3,1,1,4])
    print(f"Problem 8 - Jump Game: reachable={can_reach}, jumps={min_jumps} (expected: True, 2)")

def demonstrate_bfs_patterns():
    """Demonstrate common BFS patterns in shortest path problems"""
    print("\n=== BFS Patterns Demo ===")
    
    print("Common BFS Patterns:")
    
    print("\n1. **Level-by-Level BFS:**")
    print("   • Process all nodes at distance d before distance d+1")
    print("   • Use queue size to track levels")
    print("   • Good for problems requiring level information")
    
    print("\n2. **Distance Tracking BFS:**")
    print("   • Store distance in queue with node")
    print("   • Return distance when target found")
    print("   • Most common pattern for shortest path")
    
    print("\n3. **Multi-Source BFS:**")
    print("   • Start BFS from multiple sources simultaneously")
    print("   • Good for problems like 'distance to nearest X'")
    print("   • All sources start at distance 0")
    
    print("\n4. **Bidirectional BFS:**")
    print("   • BFS from both start and end")
    print("   • Meet in the middle")
    print("   • Reduces search space for long paths")
    
    print("\n5. **State Space BFS:**")
    print("   • Nodes represent states, not just positions")
    print("   • Edges represent valid state transitions")
    print("   • Good for puzzle-like problems")

def analyze_shortest_path_fundamentals():
    """Analyze fundamental concepts in shortest path algorithms"""
    print("\n=== Shortest Path Fundamentals ===")
    
    print("Core Concepts:")
    
    print("\n1. **Unweighted vs Weighted Graphs:**")
    print("   • Unweighted: BFS finds shortest path")
    print("   • Weighted: Need Dijkstra or Bellman-Ford")
    print("   • Unit weights: BFS optimal")
    
    print("\n2. **Graph Representations:**")
    print("   • Adjacency List: Space efficient for sparse graphs")
    print("   • Adjacency Matrix: Fast edge lookup, dense graphs")
    print("   • Implicit graphs: Generate neighbors on demand")
    
    print("\n3. **Distance Metrics:**")
    print("   • Manhattan distance: |x1-x2| + |y1-y2|")
    print("   • Euclidean distance: sqrt((x1-x2)² + (y1-y2)²)")
    print("   • Chebyshev distance: max(|x1-x2|, |y1-y2|)")
    
    print("\n4. **Search Space Pruning:**")
    print("   • Visited set: Avoid cycles")
    print("   • Early termination: Stop when target found")
    print("   • Bounds checking: Stay within valid regions")
    
    print("\n5. **Problem Variations:**")
    print("   • Single source, single target")
    print("   • Single source, all targets")
    print("   • All pairs shortest paths")
    print("   • K shortest paths")

def compare_grid_navigation_approaches():
    """Compare different approaches for grid navigation problems"""
    print("\n=== Grid Navigation Approaches ===")
    
    print("Grid Movement Patterns:")
    
    print("\n1. **4-Directional Movement:**")
    print("   • Up, Down, Left, Right")
    print("   • Manhattan distance metric")
    print("   • Most common in maze problems")
    
    print("\n2. **8-Directional Movement:**")
    print("   • Includes diagonal moves")
    print("   • Chebyshev distance metric")
    print("   • Used in games, robotics")
    
    print("\n3. **Knight's Movement:**")
    print("   • L-shaped moves (2+1 pattern)")
    print("   • 8 possible moves from any position")
    print("   • Classic chess problem")
    
    print("\n4. **Custom Movement Patterns:**")
    print("   • Problem-specific rules")
    print("   • Variable step sizes")
    print("   • Conditional movements")
    
    print("\nOptimization Techniques:")
    print("• **Early termination:** Stop when target reached")
    print("• **Bidirectional search:** Meet in middle")
    print("• **A* heuristic:** Guide search toward target")
    print("• **Jump point search:** Skip intermediate points")
    print("• **Hierarchical pathfinding:** Multi-level approach")

def analyze_real_world_applications():
    """Analyze real-world applications of basic shortest path algorithms"""
    print("\n=== Real-World Applications ===")
    
    print("1. **Navigation Systems:**")
    print("   • GPS routing algorithms")
    print("   • Indoor navigation")
    print("   • Robot path planning")
    
    print("\n2. **Game Development:**")
    print("   • NPC pathfinding")
    print("   • Level design analysis")
    print("   • AI movement patterns")
    
    print("\n3. **Network Analysis:**")
    print("   • Social network distances")
    print("   • Internet routing protocols")
    print("   • Communication networks")
    
    print("\n4. **Image Processing:**")
    print("   • Pixel connectivity analysis")
    print("   • Region segmentation")
    print("   • Pattern recognition")
    
    print("\n5. **Logistics:**")
    print("   • Warehouse navigation")
    print("   • Delivery route planning")
    print("   • Supply chain optimization")
    
    print("\n6. **Biology and Medicine:**")
    print("   • Protein folding paths")
    print("   • Neural network analysis")
    print("   • Epidemic spread modeling")
    
    print("\nKey Insights:")
    print("• **Scalability:** Consider graph size and density")
    print("• **Real-time constraints:** Algorithm speed matters")
    print("• **Memory limitations:** Space-efficient representations")
    print("• **Dynamic updates:** Handle changing graphs")
    print("• **Approximation trade-offs:** Speed vs optimality")

if __name__ == "__main__":
    test_easy_shortest_path_problems()
    demonstrate_bfs_patterns()
    analyze_shortest_path_fundamentals()
    compare_grid_navigation_approaches()
    analyze_real_world_applications()

"""
Shortest Path Concepts:
1. Fundamental BFS for Unweighted Graphs
2. Grid-Based Pathfinding Algorithms
3. State Space Search and Representation
4. Distance Metrics and Movement Patterns
5. Search Space Optimization Techniques

Key Problem Patterns:
- Unweighted shortest path: BFS optimal
- Grid navigation: 4/8-directional movement
- State transformation: Word ladder, jump games
- Multi-source problems: Island analysis
- Custom constraints: Knight moves, specific rules

Algorithm Foundations:
- BFS for unweighted graphs: O(V + E)
- Level-by-level exploration with queues
- Visited set for cycle prevention
- Early termination for efficiency
- State space modeling for complex problems

Grid Navigation Techniques:
- Direction vectors for movement patterns
- Boundary checking for valid positions
- Obstacle avoidance and path constraints
- Distance tracking and optimization
- Multi-directional search strategies

Common Optimizations:
- Early termination when target reached
- Bidirectional search for long paths
- State space pruning and bounds checking
- Efficient data structures (sets, deques)
- Memory-conscious implementations

Real-world Applications:
- GPS and navigation systems
- Game AI and pathfinding
- Network routing and analysis
- Image processing and computer vision
- Logistics and supply chain optimization

These fundamental problems provide the building blocks
for understanding advanced shortest path algorithms.
"""
