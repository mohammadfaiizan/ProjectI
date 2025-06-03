"""
Backtracking on Graphs
This module implements various backtracking algorithms on graphs including
Hamiltonian path/cycle, Knight's tour, maze solving, and graph coloring.
"""

from collections import defaultdict
from typing import List, Tuple, Set, Dict, Optional

class BacktrackingOnGraphs:
    
    def __init__(self, directed=False):
        self.graph = defaultdict(list)
        self.vertices = set()
        self.directed = directed
    
    def add_edge(self, u, v):
        """Add an edge to the graph"""
        self.vertices.add(u)
        self.vertices.add(v)
        self.graph[u].append(v)
        if not self.directed:
            self.graph[v].append(u)
    
    def get_neighbors(self, vertex):
        """Get neighbors of a vertex"""
        return self.graph[vertex]
    
    # ==================== HAMILTONIAN PATH AND CYCLE ====================
    
    def hamiltonian_path(self, start_vertex=None):
        """
        Find Hamiltonian path (visits each vertex exactly once)
        
        Time Complexity: O(N!)
        Space Complexity: O(N)
        
        Args:
            start_vertex: Starting vertex (optional)
        
        Returns:
            list: Hamiltonian path if exists, None otherwise
        """
        if not self.vertices:
            return []
        
        vertices_list = list(self.vertices)
        n = len(vertices_list)
        
        if start_vertex is None:
            start_vertex = vertices_list[0]
        elif start_vertex not in self.vertices:
            raise ValueError(f"Start vertex {start_vertex} not in graph")
        
        path = [start_vertex]
        visited = {start_vertex}
        
        def backtrack(current_vertex):
            if len(path) == n:
                return True  # Found Hamiltonian path
            
            for neighbor in self.get_neighbors(current_vertex):
                if neighbor not in visited:
                    # Choose
                    path.append(neighbor)
                    visited.add(neighbor)
                    
                    # Explore
                    if backtrack(neighbor):
                        return True
                    
                    # Unchoose (backtrack)
                    path.pop()
                    visited.remove(neighbor)
            
            return False
        
        if backtrack(start_vertex):
            return path.copy()
        else:
            return None
    
    def hamiltonian_cycle(self, start_vertex=None):
        """
        Find Hamiltonian cycle (visits each vertex exactly once and returns to start)
        
        Args:
            start_vertex: Starting vertex (optional)
        
        Returns:
            list: Hamiltonian cycle if exists, None otherwise
        """
        if not self.vertices:
            return []
        
        vertices_list = list(self.vertices)
        n = len(vertices_list)
        
        if start_vertex is None:
            start_vertex = vertices_list[0]
        elif start_vertex not in self.vertices:
            raise ValueError(f"Start vertex {start_vertex} not in graph")
        
        path = [start_vertex]
        visited = {start_vertex}
        
        def backtrack(current_vertex):
            if len(path) == n:
                # Check if we can return to start vertex
                if start_vertex in self.get_neighbors(current_vertex):
                    path.append(start_vertex)
                    return True
                return False
            
            for neighbor in self.get_neighbors(current_vertex):
                if neighbor not in visited:
                    # Choose
                    path.append(neighbor)
                    visited.add(neighbor)
                    
                    # Explore
                    if backtrack(neighbor):
                        return True
                    
                    # Unchoose (backtrack)
                    path.pop()
                    visited.remove(neighbor)
            
            return False
        
        if backtrack(start_vertex):
            return path.copy()
        else:
            return None
    
    def all_hamiltonian_paths(self, start_vertex=None):
        """
        Find all Hamiltonian paths from a given start vertex
        
        Args:
            start_vertex: Starting vertex (optional)
        
        Returns:
            list: List of all Hamiltonian paths
        """
        if not self.vertices:
            return []
        
        vertices_list = list(self.vertices)
        n = len(vertices_list)
        
        if start_vertex is None:
            start_vertex = vertices_list[0]
        
        all_paths = []
        path = [start_vertex]
        visited = {start_vertex}
        
        def backtrack(current_vertex):
            if len(path) == n:
                all_paths.append(path.copy())
                return
            
            for neighbor in self.get_neighbors(current_vertex):
                if neighbor not in visited:
                    # Choose
                    path.append(neighbor)
                    visited.add(neighbor)
                    
                    # Explore
                    backtrack(neighbor)
                    
                    # Unchoose (backtrack)
                    path.pop()
                    visited.remove(neighbor)
        
        backtrack(start_vertex)
        return all_paths
    
    # ==================== KNIGHT'S TOUR ====================
    
    def knights_tour(self, board_size: int, start_row: int = 0, start_col: int = 0):
        """
        Find Knight's tour on a chessboard (visit each square exactly once)
        
        Time Complexity: O(8^(N^2))
        Space Complexity: O(N^2)
        
        Args:
            board_size: Size of the chessboard (N x N)
            start_row: Starting row
            start_col: Starting column
        
        Returns:
            list: Sequence of moves if tour exists, None otherwise
        """
        # Knight moves (8 possible moves)
        knight_moves = [
            (2, 1), (1, 2), (-1, 2), (-2, 1),
            (-2, -1), (-1, -2), (1, -2), (2, -1)
        ]
        
        # Initialize board
        board = [[-1 for _ in range(board_size)] for _ in range(board_size)]
        
        # Track the sequence of moves
        move_sequence = []
        
        def is_safe(row, col):
            return (0 <= row < board_size and 
                   0 <= col < board_size and 
                   board[row][col] == -1)
        
        def backtrack(row, col, move_count):
            # Mark current square with move number
            board[row][col] = move_count
            move_sequence.append((row, col))
            
            # If all squares are visited
            if move_count == board_size * board_size - 1:
                return True
            
            # Try all 8 knight moves
            for dr, dc in knight_moves:
                next_row, next_col = row + dr, col + dc
                
                if is_safe(next_row, next_col):
                    if backtrack(next_row, next_col, move_count + 1):
                        return True
            
            # Backtrack
            board[row][col] = -1
            move_sequence.pop()
            return False
        
        if backtrack(start_row, start_col, 0):
            return move_sequence.copy(), board
        else:
            return None, None
    
    def knights_tour_warnsdorff(self, board_size: int, start_row: int = 0, start_col: int = 0):
        """
        Knight's tour using Warnsdorff's heuristic (choose move with fewest onward moves)
        Much faster than pure backtracking
        
        Args:
            board_size: Size of the chessboard
            start_row: Starting row
            start_col: Starting column
        
        Returns:
            tuple: (move_sequence, board) if successful, (None, None) otherwise
        """
        knight_moves = [
            (2, 1), (1, 2), (-1, 2), (-2, 1),
            (-2, -1), (-1, -2), (1, -2), (2, -1)
        ]
        
        board = [[-1 for _ in range(board_size)] for _ in range(board_size)]
        move_sequence = []
        
        def is_safe(row, col):
            return (0 <= row < board_size and 
                   0 <= col < board_size and 
                   board[row][col] == -1)
        
        def get_degree(row, col):
            """Count number of unvisited squares reachable from (row, col)"""
            count = 0
            for dr, dc in knight_moves:
                if is_safe(row + dr, col + dc):
                    count += 1
            return count
        
        # Start the tour
        current_row, current_col = start_row, start_col
        
        for move_count in range(board_size * board_size):
            board[current_row][current_col] = move_count
            move_sequence.append((current_row, current_col))
            
            if move_count == board_size * board_size - 1:
                break
            
            # Find next move using Warnsdorff's rule
            min_degree = 9
            next_row, next_col = -1, -1
            
            for dr, dc in knight_moves:
                new_row, new_col = current_row + dr, current_col + dc
                
                if is_safe(new_row, new_col):
                    degree = get_degree(new_row, new_col)
                    if degree < min_degree:
                        min_degree = degree
                        next_row, next_col = new_row, new_col
            
            if next_row == -1:  # No valid move found
                return None, None
            
            current_row, current_col = next_row, next_col
        
        return move_sequence, board
    
    # ==================== MAZE SOLVING ====================
    
    def solve_maze(self, maze: List[List[int]], start: Tuple[int, int], 
                   end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Solve maze using backtracking
        
        Args:
            maze: 2D array where 0 is path, 1 is wall
            start: Starting coordinates (row, col)
            end: Ending coordinates (row, col)
        
        Returns:
            list: Path from start to end if exists, None otherwise
        """
        if not maze or not maze[0]:
            return None
        
        rows, cols = len(maze), len(maze[0])
        start_row, start_col = start
        end_row, end_col = end
        
        # Directions: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # Track visited cells
        visited = set()
        path = []
        
        def is_valid(row, col):
            return (0 <= row < rows and 
                   0 <= col < cols and 
                   maze[row][col] == 0 and 
                   (row, col) not in visited)
        
        def backtrack(row, col):
            # Add current cell to path and mark as visited
            path.append((row, col))
            visited.add((row, col))
            
            # Check if we reached the destination
            if (row, col) == (end_row, end_col):
                return True
            
            # Try all four directions
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                
                if is_valid(new_row, new_col):
                    if backtrack(new_row, new_col):
                        return True
            
            # Backtrack
            path.pop()
            visited.remove((row, col))
            return False
        
        if is_valid(start_row, start_col) and backtrack(start_row, start_col):
            return path.copy()
        else:
            return None
    
    def solve_maze_all_paths(self, maze: List[List[int]], start: Tuple[int, int], 
                           end: Tuple[int, int]) -> List[List[Tuple[int, int]]]:
        """
        Find all possible paths through maze
        
        Args:
            maze: 2D array where 0 is path, 1 is wall
            start: Starting coordinates
            end: Ending coordinates
        
        Returns:
            list: All possible paths from start to end
        """
        if not maze or not maze[0]:
            return []
        
        rows, cols = len(maze), len(maze[0])
        start_row, start_col = start
        end_row, end_col = end
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        all_paths = []
        visited = set()
        path = []
        
        def is_valid(row, col):
            return (0 <= row < rows and 
                   0 <= col < cols and 
                   maze[row][col] == 0 and 
                   (row, col) not in visited)
        
        def backtrack(row, col):
            path.append((row, col))
            visited.add((row, col))
            
            if (row, col) == (end_row, end_col):
                all_paths.append(path.copy())
            else:
                for dr, dc in directions:
                    new_row, new_col = row + dr, col + dc
                    if is_valid(new_row, new_col):
                        backtrack(new_row, new_col)
            
            # Backtrack
            path.pop()
            visited.remove((row, col))
        
        if is_valid(start_row, start_col):
            backtrack(start_row, start_col)
        
        return all_paths
    
    # ==================== GRAPH COLORING (M-COLORING PROBLEM) ====================
    
    def graph_coloring(self, m: int) -> Optional[Dict]:
        """
        M-Coloring problem: Color graph with m colors such that no two adjacent vertices have same color
        
        Time Complexity: O(m^V)
        Space Complexity: O(V)
        
        Args:
            m: Number of colors available
        
        Returns:
            dict: Mapping from vertex to color if possible, None otherwise
        """
        if not self.vertices:
            return {}
        
        vertices_list = list(self.vertices)
        n = len(vertices_list)
        colors = {}
        
        def is_safe(vertex, color):
            """Check if it's safe to color vertex with given color"""
            for neighbor in self.get_neighbors(vertex):
                if neighbor in colors and colors[neighbor] == color:
                    return False
            return True
        
        def backtrack(vertex_index):
            if vertex_index == n:
                return True  # All vertices colored successfully
            
            vertex = vertices_list[vertex_index]
            
            # Try all colors
            for color in range(m):
                if is_safe(vertex, color):
                    # Choose color
                    colors[vertex] = color
                    
                    # Recursively color remaining vertices
                    if backtrack(vertex_index + 1):
                        return True
                    
                    # Backtrack
                    del colors[vertex]
            
            return False
        
        if backtrack(0):
            return colors.copy()
        else:
            return None
    
    def chromatic_number(self) -> int:
        """
        Find chromatic number (minimum number of colors needed)
        
        Returns:
            int: Chromatic number
        """
        if not self.vertices:
            return 0
        
        # Try coloring with 1, 2, 3, ... colors until successful
        for m in range(1, len(self.vertices) + 1):
            if self.graph_coloring(m) is not None:
                return m
        
        return len(self.vertices)  # Worst case: each vertex different color
    
    def all_graph_colorings(self, m: int) -> List[Dict]:
        """
        Find all possible colorings of graph with m colors
        
        Args:
            m: Number of colors available
        
        Returns:
            list: All possible colorings
        """
        if not self.vertices:
            return [{}]
        
        vertices_list = list(self.vertices)
        n = len(vertices_list)
        all_colorings = []
        colors = {}
        
        def is_safe(vertex, color):
            for neighbor in self.get_neighbors(vertex):
                if neighbor in colors and colors[neighbor] == color:
                    return False
            return True
        
        def backtrack(vertex_index):
            if vertex_index == n:
                all_colorings.append(colors.copy())
                return
            
            vertex = vertices_list[vertex_index]
            
            for color in range(m):
                if is_safe(vertex, color):
                    colors[vertex] = color
                    backtrack(vertex_index + 1)
                    del colors[vertex]
        
        backtrack(0)
        return all_colorings
    
    def bipartite_coloring(self) -> Optional[Dict]:
        """
        Check if graph is bipartite and return 2-coloring if possible
        
        Returns:
            dict: 2-coloring if graph is bipartite, None otherwise
        """
        return self.graph_coloring(2)
    
    # ==================== UTILITY METHODS ====================
    
    def display(self):
        """Display the graph"""
        for vertex in sorted(self.graph.keys()):
            neighbors = list(self.graph[vertex])
            print(f"{vertex}: {neighbors}")
    
    def display_maze(self, maze: List[List[int]], path: List[Tuple[int, int]] = None):
        """Display maze with optional path"""
        if not maze:
            return
        
        path_set = set(path) if path else set()
        
        for i, row in enumerate(maze):
            line = ""
            for j, cell in enumerate(row):
                if (i, j) in path_set:
                    line += "* "
                elif cell == 0:
                    line += ". "
                else:
                    line += "# "
            print(line)
        print()
    
    def display_knights_board(self, board: List[List[int]]):
        """Display Knight's tour board"""
        if not board:
            return
        
        n = len(board)
        for row in board:
            line = ""
            for cell in row:
                line += f"{cell:2d} "
            print(line)
        print()


# ==================== EXAMPLE USAGE AND TESTING ====================

if __name__ == "__main__":
    print("=== Backtracking on Graphs Demo ===\n")
    
    # Example 1: Hamiltonian Path and Cycle
    print("1. Hamiltonian Path and Cycle:")
    hamiltonian_graph = BacktrackingOnGraphs(directed=False)
    
    # Create a graph with Hamiltonian path
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (1, 3)]
    for u, v in edges:
        hamiltonian_graph.add_edge(u, v)
    
    print("Graph structure:")
    hamiltonian_graph.display()
    
    ham_path = hamiltonian_graph.hamiltonian_path(0)
    print(f"Hamiltonian path from 0: {ham_path}")
    
    ham_cycle = hamiltonian_graph.hamiltonian_cycle(0)
    print(f"Hamiltonian cycle from 0: {ham_cycle}")
    
    all_ham_paths = hamiltonian_graph.all_hamiltonian_paths(0)
    print(f"All Hamiltonian paths from 0: {all_ham_paths}")
    print()
    
    # Example 2: Knight's Tour
    print("2. Knight's Tour:")
    board_size = 5
    
    # Try regular backtracking (might be slow for larger boards)
    print(f"Solving {board_size}x{board_size} Knight's Tour with backtracking:")
    move_sequence, board = BacktrackingOnGraphs().knights_tour(board_size, 0, 0)
    
    if move_sequence:
        print("Knight's tour found!")
        print("Move sequence:", move_sequence[:10], "..." if len(move_sequence) > 10 else "")
        print("Board:")
        BacktrackingOnGraphs().display_knights_board(board)
    else:
        print("No Knight's tour found with backtracking")
    
    # Try Warnsdorff's heuristic
    print(f"Solving {board_size}x{board_size} Knight's Tour with Warnsdorff's heuristic:")
    move_sequence_w, board_w = BacktrackingOnGraphs().knights_tour_warnsdorff(board_size, 0, 0)
    
    if move_sequence_w:
        print("Knight's tour found with Warnsdorff's heuristic!")
        print("Board:")
        BacktrackingOnGraphs().display_knights_board(board_w)
    else:
        print("No Knight's tour found with Warnsdorff's heuristic")
    print()
    
    # Example 3: Maze Solving
    print("3. Maze Solving:")
    maze = [
        [0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 0, 1, 0]
    ]
    
    print("Maze (0=path, 1=wall, *=solution path):")
    start = (0, 0)
    end = (4, 4)
    
    maze_solver = BacktrackingOnGraphs()
    path = maze_solver.solve_maze(maze, start, end)
    
    if path:
        print(f"Path found from {start} to {end}:")
        maze_solver.display_maze(maze, path)
        print(f"Path: {path}")
    else:
        print("No path found")
    
    # Find all paths
    all_paths = maze_solver.solve_maze_all_paths(maze, start, end)
    print(f"Total number of paths: {len(all_paths)}")
    if all_paths:
        print(f"First path: {all_paths[0]}")
    print()
    
    # Example 4: Graph Coloring
    print("4. Graph Coloring (M-Coloring Problem):")
    coloring_graph = BacktrackingOnGraphs(directed=False)
    
    # Create a graph that requires 3 colors (triangle + additional vertex)
    coloring_edges = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3)]
    for u, v in coloring_edges:
        coloring_graph.add_edge(u, v)
    
    print("Graph for coloring:")
    coloring_graph.display()
    
    # Try coloring with different number of colors
    for m in range(2, 5):
        coloring = coloring_graph.graph_coloring(m)
        print(f"Coloring with {m} colors: {coloring}")
    
    chromatic_num = coloring_graph.chromatic_number()
    print(f"Chromatic number: {chromatic_num}")
    
    # Check if bipartite
    bipartite_coloring = coloring_graph.bipartite_coloring()
    print(f"Bipartite (2-coloring): {bipartite_coloring is not None}")
    
    # All colorings with 3 colors
    all_colorings = coloring_graph.all_graph_colorings(3)
    print(f"Number of different 3-colorings: {len(all_colorings)}")
    print(f"First few colorings: {all_colorings[:3]}")
    print()
    
    # Example 5: Bipartite Graph Test
    print("5. Bipartite Graph Test:")
    bipartite_graph = BacktrackingOnGraphs(directed=False)
    
    # Create a bipartite graph
    bipartite_edges = [(0, 3), (0, 4), (1, 3), (1, 4), (2, 3)]
    for u, v in bipartite_edges:
        bipartite_graph.add_edge(u, v)
    
    print("Bipartite graph:")
    bipartite_graph.display()
    
    bipartite_result = bipartite_graph.bipartite_coloring()
    print(f"Is bipartite: {bipartite_result is not None}")
    if bipartite_result:
        print(f"2-coloring: {bipartite_result}")
    
    print("\n=== Demo Complete ===") 