"""
Path Finding and Search Problems with Backtracking
==================================================

Topics: Maze solving, word search, path enumeration, grid navigation
Companies: Google, Amazon, Microsoft, Facebook, Uber, Tesla, Gaming companies
Difficulty: Medium to Hard
Time Complexity: O(4^(m*n)) for grid problems, varies by constraints
Space Complexity: O(m*n) for visited tracking and recursion stack
"""

from typing import List, Set, Tuple, Dict, Optional, Deque
from collections import deque
import copy

class PathFindingProblems:
    
    def __init__(self):
        """Initialize with solution tracking and path analysis"""
        self.solutions = []
        self.call_count = 0
        self.paths_explored = 0
        self.max_path_length = 0
    
    # ==========================================
    # 1. MAZE SOLVING PROBLEMS
    # ==========================================
    
    def solve_maze_single_path(self, maze: List[List[int]], start: Tuple[int, int], 
                              end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Find a single path through maze from start to end
        
        0 = open path, 1 = wall
        Returns path as list of coordinates, empty if no path exists
        
        Company: Uber, Tesla (for navigation systems)
        Difficulty: Medium
        Time: O(4^(m*n)), Space: O(m*n)
        """
        rows, cols = len(maze), len(maze[0])
        visited = set()
        path = []
        
        def is_valid_move(row: int, col: int) -> bool:
            """Check if move to (row, col) is valid"""
            return (0 <= row < rows and 
                   0 <= col < cols and 
                   maze[row][col] == 0 and 
                   (row, col) not in visited)
        
        def backtrack(row: int, col: int) -> bool:
            self.call_count += 1
            self.paths_explored += 1
            
            print(f"{'  ' * len(path)}Exploring ({row}, {col}), path length: {len(path)}")
            
            # BASE CASE: Reached destination
            if (row, col) == end:
                path.append((row, col))
                print(f"{'  ' * len(path)}✓ Reached destination!")
                return True
            
            # CONSTRAINT CHECK: Invalid move
            if not is_valid_move(row, col):
                return False
            
            # MAKE CHOICE: Add current position to path and mark visited
            path.append((row, col))
            visited.add((row, col))
            self.max_path_length = max(self.max_path_length, len(path))
            
            # TRY: All four directions (up, right, down, left)
            directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
            direction_names = ['up', 'right', 'down', 'left']
            
            for (dr, dc), direction in zip(directions, direction_names):
                new_row, new_col = row + dr, col + dc
                print(f"{'  ' * len(path)}Trying to move {direction} to ({new_row}, {new_col})")
                
                if backtrack(new_row, new_col):
                    return True
            
            # BACKTRACK: Remove current position from path and unmark visited
            path.pop()
            visited.remove((row, col))
            print(f"{'  ' * len(path)}← Backtracking from ({row}, {col})")
            
            return False
        
        print(f"Solving maze from {start} to {end}:")
        self.call_count = 0
        self.paths_explored = 0
        self.max_path_length = 0
        
        start_row, start_col = start
        if backtrack(start_row, start_col):
            print(f"✓ Path found with length {len(path)}")
            print(f"Statistics: {self.call_count} calls, {self.paths_explored} paths explored")
            return path
        else:
            print("✗ No path found!")
            return []
    
    def solve_maze_all_paths(self, maze: List[List[int]], start: Tuple[int, int], 
                            end: Tuple[int, int]) -> List[List[Tuple[int, int]]]:
        """
        Find all possible paths through maze from start to end
        
        Time: O(4^(m*n)), Space: O(4^(m*n))
        """
        rows, cols = len(maze), len(maze[0])
        all_paths = []
        
        def is_valid_move(row: int, col: int, visited: Set[Tuple[int, int]]) -> bool:
            """Check if move is valid"""
            return (0 <= row < rows and 
                   0 <= col < cols and 
                   maze[row][col] == 0 and 
                   (row, col) not in visited)
        
        def backtrack(row: int, col: int, current_path: List[Tuple[int, int]], 
                     visited: Set[Tuple[int, int]]) -> None:
            print(f"{'  ' * len(current_path)}Exploring ({row}, {col}), path: {current_path}")
            
            # BASE CASE: Reached destination
            if (row, col) == end:
                current_path.append((row, col))
                all_paths.append(current_path[:])  # Make a copy
                print(f"{'  ' * len(current_path)}✓ Found path {len(all_paths)}: {current_path}")
                current_path.pop()  # Backtrack for finding more paths
                return
            
            # CONSTRAINT CHECK: Invalid move
            if not is_valid_move(row, col, visited):
                return
            
            # MAKE CHOICE: Add to path and mark visited
            current_path.append((row, col))
            visited.add((row, col))
            
            # TRY: All four directions
            directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
            
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                backtrack(new_row, new_col, current_path, visited)
            
            # BACKTRACK: Remove from path and unmark visited
            current_path.pop()
            visited.remove((row, col))
        
        print(f"Finding all paths from {start} to {end}:")
        start_row, start_col = start
        backtrack(start_row, start_col, [], set())
        
        print(f"Found {len(all_paths)} total paths")
        return all_paths
    
    def print_maze_with_path(self, maze: List[List[int]], path: List[Tuple[int, int]]) -> None:
        """Print maze with path marked"""
        if not path:
            print("No path to display")
            return
        
        path_set = set(path)
        
        print("Maze with path (. = path, # = wall, space = open, S = start, E = end):")
        for i, row in enumerate(maze):
            for j, cell in enumerate(row):
                if (i, j) == path[0]:
                    print('S', end=' ')
                elif (i, j) == path[-1]:
                    print('E', end=' ')
                elif (i, j) in path_set:
                    print('.', end=' ')
                elif cell == 1:
                    print('#', end=' ')
                else:
                    print(' ', end=' ')
            print()
        print()
    
    # ==========================================
    # 2. WORD SEARCH PROBLEMS
    # ==========================================
    
    def word_search(self, board: List[List[str]], word: str) -> bool:
        """
        Search for word in 2D character board
        
        Word can be constructed from adjacent cells (horizontal/vertical)
        Same cell cannot be used more than once per word
        
        Company: Amazon, Microsoft, Google
        Difficulty: Medium
        Time: O(m*n*4^L), Space: O(L) where L is word length
        """
        rows, cols = len(board), len(board[0])
        
        def backtrack(row: int, col: int, index: int, path: List[Tuple[int, int]]) -> bool:
            print(f"{'  ' * index}Checking ({row}, {col}) for '{word[index]}', path: {path}")
            
            # BASE CASE: Found complete word
            if index == len(word):
                print(f"{'  ' * index}✓ Found word: {word}")
                return True
            
            # CONSTRAINT CHECKS
            if (row < 0 or row >= rows or 
                col < 0 or col >= cols or 
                board[row][col] != word[index] or 
                (row, col) in path):
                return False
            
            # MAKE CHOICE: Add current cell to path
            path.append((row, col))
            
            # TRY: All four directions
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                if backtrack(new_row, new_col, index + 1, path):
                    return True
            
            # BACKTRACK: Remove current cell from path
            path.pop()
            print(f"{'  ' * index}← Backtracking from ({row}, {col})")
            
            return False
        
        # Try starting from each cell
        for i in range(rows):
            for j in range(cols):
                if board[i][j] == word[0]:  # Potential starting point
                    print(f"\nTrying to start word search from ({i}, {j})")
                    if backtrack(i, j, 0, []):
                        return True
        
        return False
    
    def word_search_all_paths(self, board: List[List[str]], word: str) -> List[List[Tuple[int, int]]]:
        """
        Find all possible paths that form the given word
        
        Returns list of paths, where each path is list of coordinates
        """
        rows, cols = len(board), len(board[0])
        all_word_paths = []
        
        def backtrack(row: int, col: int, index: int, current_path: List[Tuple[int, int]]) -> None:
            # BASE CASE: Found complete word
            if index == len(word):
                all_word_paths.append(current_path[:])  # Make a copy
                print(f"Found word path {len(all_word_paths)}: {current_path}")
                return
            
            # CONSTRAINT CHECKS
            if (row < 0 or row >= rows or 
                col < 0 or col >= cols or 
                board[row][col] != word[index] or 
                (row, col) in current_path):
                return
            
            # MAKE CHOICE: Add current cell to path
            current_path.append((row, col))
            
            # TRY: All four directions
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                backtrack(new_row, new_col, index + 1, current_path)
            
            # BACKTRACK: Remove current cell from path
            current_path.pop()
        
        print(f"Finding all paths for word '{word}':")
        
        # Try starting from each cell
        for i in range(rows):
            for j in range(cols):
                if board[i][j] == word[0]:
                    backtrack(i, j, 0, [])
        
        print(f"Found {len(all_word_paths)} total paths")
        return all_word_paths
    
    def word_search_multiple_words(self, board: List[List[str]], words: List[str]) -> List[str]:
        """
        Find which words from the list exist in the board
        
        Company: Google, Facebook
        Difficulty: Hard
        Time: O(m*n*4^L*W), Space: O(L*W)
        """
        found_words = []
        
        for word in words:
            print(f"Searching for word: '{word}'")
            if self.word_search(board, word):
                found_words.append(word)
                print(f"✓ Found: '{word}'")
            else:
                print(f"✗ Not found: '{word}'")
            print()
        
        return found_words
    
    # ==========================================
    # 3. GRID PATH PROBLEMS
    # ==========================================
    
    def unique_paths_with_obstacles(self, grid: List[List[int]]) -> int:
        """
        Count unique paths from top-left to bottom-right with obstacles
        
        0 = open, 1 = obstacle
        Can only move right or down
        
        Company: Amazon, Google
        Difficulty: Medium
        Time: O(2^(m+n)), Space: O(m+n)
        """
        rows, cols = len(grid), len(grid[0])
        
        def backtrack(row: int, col: int) -> int:
            # BASE CASE: Reached destination
            if row == rows - 1 and col == cols - 1:
                return 1 if grid[row][col] == 0 else 0
            
            # CONSTRAINT CHECKS: Out of bounds or obstacle
            if (row >= rows or col >= cols or grid[row][col] == 1):
                return 0
            
            # COUNT: Paths going right and down
            paths = 0
            paths += backtrack(row, col + 1)  # Move right
            paths += backtrack(row + 1, col)  # Move down
            
            return paths
        
        print("Counting unique paths with obstacles:")
        if grid[0][0] == 1:  # Starting position is blocked
            return 0
        
        return backtrack(0, 0)
    
    def path_sum_in_grid(self, grid: List[List[int]], target_sum: int) -> List[List[Tuple[int, int]]]:
        """
        Find all paths from top-left to bottom-right with given sum
        
        Can move in all four directions but no cycles
        
        Company: Microsoft, Amazon
        Difficulty: Hard
        Time: O(4^(m*n)), Space: O(m*n)
        """
        rows, cols = len(grid), len(grid[0])
        valid_paths = []
        
        def backtrack(row: int, col: int, current_sum: int, 
                     current_path: List[Tuple[int, int]], visited: Set[Tuple[int, int]]) -> None:
            print(f"{'  ' * len(current_path)}At ({row}, {col}), sum={current_sum}, target={target_sum}")
            
            # BASE CASE: Reached destination
            if row == rows - 1 and col == cols - 1:
                final_sum = current_sum + grid[row][col]
                if final_sum == target_sum:
                    final_path = current_path + [(row, col)]
                    valid_paths.append(final_path)
                    print(f"✓ Found valid path with sum {final_sum}: {final_path}")
                return
            
            # CONSTRAINT CHECKS: Out of bounds or already visited
            if (row < 0 or row >= rows or col < 0 or col >= cols or 
                (row, col) in visited):
                return
            
            # MAKE CHOICE: Add current position
            new_sum = current_sum + grid[row][col]
            current_path.append((row, col))
            visited.add((row, col))
            
            # PRUNING: If current sum already exceeds target, stop exploring
            if new_sum <= target_sum:  # Continue only if we haven't exceeded target
                # TRY: All four directions
                directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
                
                for dr, dc in directions:
                    new_row, new_col = row + dr, col + dc
                    backtrack(new_row, new_col, new_sum, current_path, visited)
            
            # BACKTRACK: Remove current position
            current_path.pop()
            visited.remove((row, col))
        
        print(f"Finding paths with sum {target_sum}:")
        backtrack(0, 0, 0, [], set())
        
        print(f"Found {len(valid_paths)} valid paths")
        return valid_paths
    
    # ==========================================
    # 4. ADVANCED PATH PROBLEMS
    # ==========================================
    
    def shortest_path_all_keys(self, grid: List[str]) -> int:
        """
        Find shortest path collecting all keys
        
        '@' = start, '.' = empty, '#' = wall
        'a'-'f' = keys, 'A'-'F' = locks (need corresponding key)
        
        Company: Google, Amazon (complex pathfinding)
        Difficulty: Hard
        Time: O(m*n*2^k) where k is number of keys
        Space: O(m*n*2^k)
        """
        rows, cols = len(grid), len(grid[0])
        
        # Find start position and count keys
        start = None
        total_keys = 0
        
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == '@':
                    start = (i, j)
                elif 'a' <= grid[i][j] <= 'f':
                    total_keys += 1
        
        if not start:
            return -1
        
        # BFS with state = (row, col, keys_bitmask)
        from collections import deque
        
        queue = deque([(start[0], start[1], 0, 0)])  # row, col, keys, steps
        visited = set([(start[0], start[1], 0)])
        
        while queue:
            row, col, keys, steps = queue.popleft()
            
            # Check if we have all keys
            if keys == (1 << total_keys) - 1:
                return steps
            
            # Try all four directions
            directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
            
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                
                # Check bounds
                if new_row < 0 or new_row >= rows or new_col < 0 or new_col >= cols:
                    continue
                
                cell = grid[new_row][new_col]
                
                # Check if cell is passable
                if cell == '#':
                    continue
                
                # Check if we can pass through locks
                if 'A' <= cell <= 'F':
                    key_needed = ord(cell.lower()) - ord('a')
                    if not (keys & (1 << key_needed)):
                        continue
                
                new_keys = keys
                
                # Pick up key if present
                if 'a' <= cell <= 'f':
                    key_index = ord(cell) - ord('a')
                    new_keys |= (1 << key_index)
                
                state = (new_row, new_col, new_keys)
                if state not in visited:
                    visited.add(state)
                    queue.append((new_row, new_col, new_keys, steps + 1))
        
        return -1  # No path found
    
    def escape_maze_with_time_limit(self, maze: List[List[int]], start: Tuple[int, int], 
                                   end: Tuple[int, int], time_limit: int) -> List[Tuple[int, int]]:
        """
        Find path through maze within time limit
        
        Each move takes 1 time unit
        
        Company: Gaming companies, puzzle games
        Difficulty: Medium
        Time: O(4^min(time_limit, m*n)), Space: O(min(time_limit, m*n))
        """
        rows, cols = len(maze), len(maze[0])
        
        def backtrack(row: int, col: int, time_left: int, 
                     current_path: List[Tuple[int, int]], visited: Set[Tuple[int, int]]) -> bool:
            print(f"{'  ' * len(current_path)}At ({row}, {col}), time left: {time_left}")
            
            # BASE CASE: Reached destination
            if (row, col) == end:
                current_path.append((row, col))
                print(f"✓ Reached destination with {time_left} time remaining!")
                return True
            
            # CONSTRAINT CHECKS: Out of bounds, wall, visited, or no time
            if (row < 0 or row >= rows or col < 0 or col >= cols or 
                maze[row][col] == 1 or (row, col) in visited or time_left <= 0):
                return False
            
            # MAKE CHOICE: Add to path and mark visited
            current_path.append((row, col))
            visited.add((row, col))
            
            # TRY: All four directions
            directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
            
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                if backtrack(new_row, new_col, time_left - 1, current_path, visited):
                    return True
            
            # BACKTRACK: Remove from path and unmark visited
            current_path.pop()
            visited.remove((row, col))
            
            return False
        
        print(f"Finding path from {start} to {end} within {time_limit} moves:")
        path = []
        start_row, start_col = start
        
        if backtrack(start_row, start_col, time_limit, path, set()):
            print(f"✓ Path found: {path}")
            return path
        else:
            print(f"✗ No path found within {time_limit} moves")
            return []

# ==========================================
# DEMONSTRATION AND TESTING
# ==========================================

def demonstrate_path_finding_problems():
    """Demonstrate all path finding and search problems"""
    print("=== PATH FINDING PROBLEMS DEMONSTRATION ===\n")
    
    pf = PathFindingProblems()
    
    # 1. Maze Solving
    print("=== MAZE SOLVING ===")
    
    # Simple maze: 0 = open, 1 = wall
    maze = [
        [0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 0, 1, 0]
    ]
    
    start = (0, 0)
    end = (4, 4)
    
    print("1. Single Path Solution:")
    path = pf.solve_maze_single_path(maze, start, end)
    if path:
        pf.print_maze_with_path(maze, path)
    print()
    
    # Smaller maze for all paths demo
    small_maze = [
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ]
    
    print("2. All Paths Solution (smaller maze):")
    all_paths = pf.solve_maze_all_paths(small_maze, (0, 0), (2, 2))
    for i, path in enumerate(all_paths):
        print(f"Path {i+1}: {path}")
    print()
    
    # 2. Word Search
    print("=== WORD SEARCH ===")
    
    board = [
        ['A', 'B', 'C', 'E'],
        ['S', 'F', 'C', 'S'],
        ['A', 'D', 'E', 'E']
    ]
    
    print("Word search board:")
    for row in board:
        print(f"  {row}")
    print()
    
    words_to_find = ["ABCCED", "SEE", "ABCB"]
    
    print("1. Single Word Search:")
    for word in words_to_find:
        found = pf.word_search(board, word)
        print(f"Word '{word}': {'Found' if found else 'Not found'}")
    print()
    
    print("2. Multiple Words Search:")
    found_words = pf.word_search_multiple_words(board, words_to_find)
    print(f"Found words: {found_words}")
    print()
    
    # 3. Grid Path Problems
    print("=== GRID PATH PROBLEMS ===")
    
    # Grid with obstacles
    obstacle_grid = [
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ]
    
    print("1. Unique Paths with Obstacles:")
    print("Grid (0=open, 1=obstacle):")
    for row in obstacle_grid:
        print(f"  {row}")
    
    unique_paths = pf.unique_paths_with_obstacles(obstacle_grid)
    print(f"Number of unique paths: {unique_paths}")
    print()
    
    # Path sum problem
    sum_grid = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    
    print("2. Path Sum Problem:")
    print("Grid:")
    for row in sum_grid:
        print(f"  {row}")
    
    target_sum = 15
    sum_paths = pf.path_sum_in_grid(sum_grid, target_sum)
    print(f"Paths with sum {target_sum}: {len(sum_paths)}")
    for i, path in enumerate(sum_paths):
        path_sum = sum(sum_grid[r][c] for r, c in path)
        print(f"  Path {i+1}: {path} (sum: {path_sum})")
    print()
    
    # 4. Advanced Problems
    print("=== ADVANCED PATH PROBLEMS ===")
    
    print("1. Escape Maze with Time Limit:")
    time_limit_maze = [
        [0, 0, 0, 0],
        [1, 1, 0, 1],
        [0, 0, 0, 0]
    ]
    
    escape_path = pf.escape_maze_with_time_limit(time_limit_maze, (0, 0), (2, 3), 8)
    if escape_path:
        print(f"Escape path: {escape_path}")
        print(f"Path length: {len(escape_path)} moves")
    print()
    
    print("2. Keys and Locks Problem:")
    keys_grid = [
        "@.a.#",
        "###.#",
        "b.A.B"
    ]
    
    print("Grid (@ = start, a,b = keys, A,B = locks):")
    for row in keys_grid:
        print(f"  {row}")
    
    shortest_steps = pf.shortest_path_all_keys(keys_grid)
    print(f"Shortest path to collect all keys: {shortest_steps} steps")

if __name__ == "__main__":
    demonstrate_path_finding_problems()
    
    print("\n=== PATH FINDING MASTERY GUIDE ===")
    
    print("\n🎯 PROBLEM TYPES:")
    print("• Single Path: Find any valid path")
    print("• All Paths: Enumerate all possible paths")
    print("• Shortest Path: Find minimum cost path")
    print("• Constrained Path: Path with additional constraints")
    
    print("\n📋 SOLUTION APPROACH:")
    print("1. Define valid moves and constraints")
    print("2. Choose appropriate search strategy")
    print("3. Implement backtracking with proper state management")
    print("4. Add pruning for optimization")
    print("5. Handle edge cases and boundary conditions")
    
    print("\n⚡ OPTIMIZATION TECHNIQUES:")
    print("• Visited tracking: Avoid cycles and redundant exploration")
    print("• Early termination: Stop when goal conditions met")
    print("• Pruning: Eliminate impossible paths early")
    print("• Memoization: Cache results for overlapping subproblems")
    print("• Heuristics: Guide search toward promising directions")
    
    print("\n🔍 COMPLEXITY ANALYSIS:")
    print("• Time: Often O(4^(m*n)) for grid problems")
    print("• Space: O(m*n) for visited tracking + recursion stack")
    print("• Actual performance depends heavily on pruning effectiveness")
    print("• Consider BFS for shortest path, DFS for path enumeration")
    
    print("\n📚 REAL-WORLD APPLICATIONS:")
    print("• Navigation: GPS routing, robot path planning")
    print("• Games: Maze solving, puzzle games, AI pathfinding")
    print("• Networks: Network routing, circuit design")
    print("• Logistics: Warehouse optimization, delivery routing")
    print("• Graphics: Ray tracing, collision detection")
