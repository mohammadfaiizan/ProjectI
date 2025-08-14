"""
Specialized Tree Structures - Domain-Specific Trees
==================================================

Topics: Spatial trees, game trees, parsing trees, specialized applications
Companies: Game companies, GIS companies, compiler companies, AI/ML companies
Difficulty: Expert level
Time Complexity: Varies by structure (O(log n) to O(n²))
Space Complexity: O(n) to O(n²) depending on dimensionality and structure
"""

from typing import List, Optional, Dict, Any, Tuple, Union, Callable
from collections import defaultdict, deque
import math
import random

class SpecializedTrees:
    
    def __init__(self):
        """Initialize with specialized tree tracking"""
        self.tree_types_implemented = 0
        self.applications_demonstrated = 0
    
    # ==========================================
    # 1. SPATIAL DATA STRUCTURES
    # ==========================================
    
    def explain_spatial_trees(self) -> None:
        """
        Explain spatial tree structures for geometric data
        """
        print("=== SPATIAL TREE STRUCTURES ===")
        print("Trees for multi-dimensional and geometric data")
        print()
        print("QUADTREES (2D SPACE PARTITIONING):")
        print("• Recursively divide 2D space into quadrants")
        print("• Each node represents a rectangular region")
        print("• Used for: Image processing, collision detection, spatial indexing")
        print("• Operations: O(log n) average, O(n) worst case")
        print()
        print("OCTREES (3D SPACE PARTITIONING):")
        print("• Extension of quadtrees to 3D space")
        print("• Divide space into 8 octants per level")
        print("• Used for: 3D graphics, voxel rendering, 3D collision detection")
        print("• Applications: Game engines, CAD systems, medical imaging")
        print()
        print("K-D TREES (K-DIMENSIONAL BINARY SPACE PARTITIONING):")
        print("• Binary tree for k-dimensional points")
        print("• Alternate splitting dimension at each level")
        print("• Used for: Nearest neighbor search, range queries")
        print("• Excellent for moderate dimensions (k ≤ 20)")
        print()
        print("R-TREES (RECTANGLE TREES):")
        print("• Tree of minimum bounding rectangles (MBRs)")
        print("• Handle overlapping regions efficiently")
        print("• Used for: Geographic Information Systems (GIS)")
        print("• Spatial databases, map applications")
        print()
        print("RANGE TREES:")
        print("• Multi-dimensional range queries")
        print("• Nested tree structures")
        print("• Used for: Computational geometry problems")
        print("• O(log^d n) query time for d dimensions")
    
    def demonstrate_spatial_trees(self) -> None:
        """
        Demonstrate spatial tree implementations
        """
        print("=== SPATIAL TREE DEMONSTRATIONS ===")
        
        # 1. Quadtree demonstration
        print("1. QUADTREE FOR 2D POINT STORAGE")
        print("   Spatial partitioning of 2D points")
        
        quadtree = Quadtree(0, 0, 100, 100)  # 100x100 region
        
        points = [(25, 25), (75, 25), (25, 75), (75, 75), (50, 50), 
                 (10, 10), (90, 90), (20, 80), (80, 20)]
        
        print(f"   Inserting points: {points}")
        
        for x, y in points:
            quadtree.insert(Point2D(x, y))
        
        print(f"   Quadtree depth: {quadtree.get_depth()}")
        print(f"   Total nodes: {quadtree.count_nodes()}")
        
        # Range query
        query_region = Rectangle(20, 20, 60, 60)
        found_points = quadtree.range_query(query_region)
        print(f"   Points in region (20,20)-(60,60): {[(p.x, p.y) for p in found_points]}")
        print()
        
        # 2. K-D Tree demonstration
        print("2. K-D TREE FOR NEAREST NEIGHBOR SEARCH")
        print("   2D points with nearest neighbor queries")
        
        kd_tree = KDTree(2)  # 2-dimensional
        
        for x, y in points:
            kd_tree.insert([x, y])
        
        print(f"   Built K-D tree with {len(points)} points")
        
        query_point = [45, 55]
        nearest = kd_tree.nearest_neighbor(query_point)
        distance = kd_tree.euclidean_distance(query_point, nearest.point)
        
        print(f"   Query point: {query_point}")
        print(f"   Nearest neighbor: {nearest.point}")
        print(f"   Distance: {distance:.2f}")
        print()
        
        # 3. R-Tree demonstration
        print("3. R-TREE FOR RECTANGULAR REGIONS")
        print("   Storing and querying rectangular objects")
        
        rtree = RTree()
        
        rectangles = [
            Rectangle(10, 10, 30, 30, "A"),
            Rectangle(40, 40, 60, 60, "B"),
            Rectangle(20, 50, 50, 80, "C"),
            Rectangle(70, 20, 90, 50, "D"),
            Rectangle(15, 60, 35, 85, "E")
        ]
        
        print("   Inserting rectangles:")
        for rect in rectangles:
            rtree.insert(rect)
            print(f"     {rect.data}: ({rect.x1},{rect.y1})-({rect.x2},{rect.y2})")
        
        # Range query
        query_rect = Rectangle(25, 25, 55, 55)
        intersecting = rtree.range_query(query_rect)
        print(f"   Rectangles intersecting (25,25)-(55,55):")
        for rect in intersecting:
            print(f"     {rect.data}")


class Point2D:
    """2D point for spatial structures"""
    
    def __init__(self, x: float, y: float, data: Any = None):
        self.x = x
        self.y = y
        self.data = data
    
    def __str__(self):
        return f"({self.x}, {self.y})"


class Rectangle:
    """Rectangle for spatial queries"""
    
    def __init__(self, x1: float, y1: float, x2: float, y2: float, data: Any = None):
        self.x1 = min(x1, x2)
        self.y1 = min(y1, y2)
        self.x2 = max(x1, x2)
        self.y2 = max(y1, y2)
        self.data = data
    
    def contains_point(self, point: Point2D) -> bool:
        """Check if rectangle contains point"""
        return (self.x1 <= point.x <= self.x2 and 
                self.y1 <= point.y <= self.y2)
    
    def intersects(self, other: 'Rectangle') -> bool:
        """Check if this rectangle intersects with another"""
        return not (self.x2 < other.x1 or other.x2 < self.x1 or
                   self.y2 < other.y1 or other.y2 < self.y1)
    
    def area(self) -> float:
        """Calculate rectangle area"""
        return (self.x2 - self.x1) * (self.y2 - self.y1)


class QuadtreeNode:
    """Node in quadtree"""
    
    def __init__(self, x: float, y: float, width: float, height: float):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.points: List[Point2D] = []
        self.children: List[Optional['QuadtreeNode']] = [None, None, None, None]
        self.divided = False
        self.capacity = 4  # Maximum points before subdivision


class Quadtree:
    """
    Quadtree implementation for 2D spatial partitioning
    
    Efficiently handles 2D point queries and range searches
    """
    
    def __init__(self, x: float, y: float, width: float, height: float):
        self.root = QuadtreeNode(x, y, width, height)
    
    def insert(self, point: Point2D) -> bool:
        """Insert point into quadtree"""
        return self._insert_recursive(self.root, point)
    
    def _insert_recursive(self, node: QuadtreeNode, point: Point2D) -> bool:
        """Recursive insertion"""
        # Check if point is in this node's boundary
        if not (node.x <= point.x <= node.x + node.width and
                node.y <= point.y <= node.y + node.height):
            return False
        
        # If node has capacity and isn't divided, add point
        if len(node.points) < node.capacity and not node.divided:
            node.points.append(point)
            return True
        
        # Subdivide if necessary
        if not node.divided:
            self._subdivide(node)
        
        # Insert into appropriate child
        return (self._insert_recursive(node.children[0], point) or
                self._insert_recursive(node.children[1], point) or
                self._insert_recursive(node.children[2], point) or
                self._insert_recursive(node.children[3], point))
    
    def _subdivide(self, node: QuadtreeNode) -> None:
        """Subdivide node into four children"""
        half_width = node.width / 2
        half_height = node.height / 2
        
        # Create four children: NW, NE, SW, SE
        node.children[0] = QuadtreeNode(node.x, node.y, half_width, half_height)  # NW
        node.children[1] = QuadtreeNode(node.x + half_width, node.y, half_width, half_height)  # NE
        node.children[2] = QuadtreeNode(node.x, node.y + half_height, half_width, half_height)  # SW
        node.children[3] = QuadtreeNode(node.x + half_width, node.y + half_height, half_width, half_height)  # SE
        
        node.divided = True
        
        # Redistribute existing points
        for point in node.points:
            for child in node.children:
                if self._insert_recursive(child, point):
                    break
        
        node.points.clear()
    
    def range_query(self, range_rect: Rectangle) -> List[Point2D]:
        """Find all points within given rectangle"""
        result = []
        self._range_query_recursive(self.root, range_rect, result)
        return result
    
    def _range_query_recursive(self, node: QuadtreeNode, range_rect: Rectangle, result: List[Point2D]) -> None:
        """Recursive range query"""
        # Check if node boundary intersects with query range
        node_rect = Rectangle(node.x, node.y, node.x + node.width, node.y + node.height)
        
        if not range_rect.intersects(node_rect):
            return
        
        # Check points in this node
        for point in node.points:
            if range_rect.contains_point(point):
                result.append(point)
        
        # Recurse to children if divided
        if node.divided:
            for child in node.children:
                if child:
                    self._range_query_recursive(child, range_rect, result)
    
    def get_depth(self) -> int:
        """Get maximum depth of quadtree"""
        return self._get_depth_recursive(self.root)
    
    def _get_depth_recursive(self, node: QuadtreeNode) -> int:
        """Recursive depth calculation"""
        if not node.divided:
            return 1
        
        max_child_depth = 0
        for child in node.children:
            if child:
                max_child_depth = max(max_child_depth, self._get_depth_recursive(child))
        
        return 1 + max_child_depth
    
    def count_nodes(self) -> int:
        """Count total nodes in quadtree"""
        return self._count_nodes_recursive(self.root)
    
    def _count_nodes_recursive(self, node: QuadtreeNode) -> int:
        """Recursive node counting"""
        count = 1
        
        if node.divided:
            for child in node.children:
                if child:
                    count += self._count_nodes_recursive(child)
        
        return count


class KDTreeNode:
    """Node in K-D tree"""
    
    def __init__(self, point: List[float], dimension: int):
        self.point = point
        self.dimension = dimension
        self.left: Optional['KDTreeNode'] = None
        self.right: Optional['KDTreeNode'] = None


class KDTree:
    """
    K-Dimensional tree for efficient nearest neighbor search
    
    Excellent for moderate-dimensional nearest neighbor queries
    """
    
    def __init__(self, k: int):
        self.k = k  # Number of dimensions
        self.root: Optional[KDTreeNode] = None
    
    def insert(self, point: List[float]) -> None:
        """Insert point into K-D tree"""
        if len(point) != self.k:
            raise ValueError(f"Point must have {self.k} dimensions")
        
        self.root = self._insert_recursive(self.root, point, 0)
    
    def _insert_recursive(self, node: Optional[KDTreeNode], point: List[float], depth: int) -> KDTreeNode:
        """Recursive insertion"""
        if not node:
            return KDTreeNode(point, depth % self.k)
        
        dimension = depth % self.k
        
        if point[dimension] < node.point[dimension]:
            node.left = self._insert_recursive(node.left, point, depth + 1)
        else:
            node.right = self._insert_recursive(node.right, point, depth + 1)
        
        return node
    
    def nearest_neighbor(self, query_point: List[float]) -> Optional[KDTreeNode]:
        """Find nearest neighbor to query point"""
        if not self.root:
            return None
        
        best = [None, float('inf')]  # [node, distance]
        self._nearest_neighbor_recursive(self.root, query_point, 0, best)
        return best[0]
    
    def _nearest_neighbor_recursive(self, node: Optional[KDTreeNode], query: List[float], 
                                  depth: int, best: List[Any]) -> None:
        """Recursive nearest neighbor search"""
        if not node:
            return
        
        # Calculate distance to current node
        distance = self.euclidean_distance(query, node.point)
        
        if distance < best[1]:
            best[0] = node
            best[1] = distance
        
        dimension = depth % self.k
        
        # Determine which side to search first
        if query[dimension] < node.point[dimension]:
            self._nearest_neighbor_recursive(node.left, query, depth + 1, best)
            
            # Check if we need to search the other side
            if abs(query[dimension] - node.point[dimension]) < best[1]:
                self._nearest_neighbor_recursive(node.right, query, depth + 1, best)
        else:
            self._nearest_neighbor_recursive(node.right, query, depth + 1, best)
            
            # Check if we need to search the other side
            if abs(query[dimension] - node.point[dimension]) < best[1]:
                self._nearest_neighbor_recursive(node.left, query, depth + 1, best)
    
    def euclidean_distance(self, p1: List[float], p2: List[float]) -> float:
        """Calculate Euclidean distance between two points"""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


class RTreeNode:
    """Node in R-tree"""
    
    def __init__(self, is_leaf: bool = False):
        self.is_leaf = is_leaf
        self.entries: List[Union[Rectangle, 'RTreeNode']] = []
        self.mbr: Optional[Rectangle] = None  # Minimum bounding rectangle
        self.max_entries = 4  # Maximum entries per node


class RTree:
    """
    R-tree implementation for rectangular spatial data
    
    Efficient for geographic and spatial database applications
    """
    
    def __init__(self):
        self.root = RTreeNode(is_leaf=True)
    
    def insert(self, rect: Rectangle) -> None:
        """Insert rectangle into R-tree"""
        self._insert_recursive(self.root, rect)
    
    def _insert_recursive(self, node: RTreeNode, rect: Rectangle) -> None:
        """Recursive insertion with node splitting"""
        if node.is_leaf:
            # Insert into leaf node
            node.entries.append(rect)
            self._update_mbr(node)
            
            # Split if necessary
            if len(node.entries) > node.max_entries:
                self._split_node(node)
        else:
            # Find best child to insert into
            best_child = self._choose_subtree(node, rect)
            self._insert_recursive(best_child, rect)
            self._update_mbr(node)
    
    def _choose_subtree(self, node: RTreeNode, rect: Rectangle) -> RTreeNode:
        """Choose best child node for insertion"""
        best_child = None
        min_enlargement = float('inf')
        
        for entry in node.entries:
            if isinstance(entry, RTreeNode):
                enlargement = self._calculate_enlargement(entry.mbr, rect)
                if enlargement < min_enlargement:
                    min_enlargement = enlargement
                    best_child = entry
        
        return best_child or node.entries[0]
    
    def _calculate_enlargement(self, mbr: Rectangle, rect: Rectangle) -> float:
        """Calculate area enlargement if rect is added to mbr"""
        if not mbr:
            return rect.area()
        
        new_mbr = self._combine_rectangles(mbr, rect)
        return new_mbr.area() - mbr.area()
    
    def _combine_rectangles(self, rect1: Rectangle, rect2: Rectangle) -> Rectangle:
        """Combine two rectangles into minimum bounding rectangle"""
        return Rectangle(
            min(rect1.x1, rect2.x1),
            min(rect1.y1, rect2.y1),
            max(rect1.x2, rect2.x2),
            max(rect1.y2, rect2.y2)
        )
    
    def _update_mbr(self, node: RTreeNode) -> None:
        """Update minimum bounding rectangle for node"""
        if not node.entries:
            node.mbr = None
            return
        
        if node.is_leaf:
            # MBR of all rectangles
            first_rect = node.entries[0]
            node.mbr = Rectangle(first_rect.x1, first_rect.y1, first_rect.x2, first_rect.y2)
            
            for rect in node.entries[1:]:
                node.mbr = self._combine_rectangles(node.mbr, rect)
        else:
            # MBR of all child nodes
            first_child = node.entries[0]
            node.mbr = Rectangle(first_child.mbr.x1, first_child.mbr.y1, 
                               first_child.mbr.x2, first_child.mbr.y2)
            
            for child in node.entries[1:]:
                if isinstance(child, RTreeNode) and child.mbr:
                    node.mbr = self._combine_rectangles(node.mbr, child.mbr)
    
    def _split_node(self, node: RTreeNode) -> None:
        """Split overflowing node"""
        # Simple split: divide entries roughly in half
        # In practice, more sophisticated algorithms are used
        mid = len(node.entries) // 2
        
        new_node = RTreeNode(node.is_leaf)
        new_node.entries = node.entries[mid:]
        node.entries = node.entries[:mid]
        
        self._update_mbr(node)
        self._update_mbr(new_node)
        
        # If this was root, create new root
        if node == self.root:
            new_root = RTreeNode(is_leaf=False)
            new_root.entries = [node, new_node]
            self._update_mbr(new_root)
            self.root = new_root
    
    def range_query(self, query_rect: Rectangle) -> List[Rectangle]:
        """Find all rectangles that intersect with query rectangle"""
        result = []
        self._range_query_recursive(self.root, query_rect, result)
        return result
    
    def _range_query_recursive(self, node: RTreeNode, query_rect: Rectangle, result: List[Rectangle]) -> None:
        """Recursive range query"""
        if not node.mbr or not query_rect.intersects(node.mbr):
            return
        
        if node.is_leaf:
            for rect in node.entries:
                if query_rect.intersects(rect):
                    result.append(rect)
        else:
            for child in node.entries:
                if isinstance(child, RTreeNode):
                    self._range_query_recursive(child, query_rect, result)


# ==========================================
# 2. GAME TREES AND DECISION TREES
# ==========================================

class GameTrees:
    """
    Game trees for adversarial search and game AI
    """
    
    def explain_game_trees(self) -> None:
        """Explain game tree concepts"""
        print("=== GAME TREES FOR AI ===")
        print("Trees for game playing and adversarial search")
        print()
        print("MINIMAX ALGORITHM:")
        print("• Two-player zero-sum games")
        print("• Maximize player's score, minimize opponent's")
        print("• Perfect play assumption")
        print("• Exponential search space")
        print()
        print("ALPHA-BETA PRUNING:")
        print("• Optimization of minimax")
        print("• Prune branches that cannot affect final decision")
        print("• Can reduce search space exponentially")
        print("• Best case: O(b^(d/2)) instead of O(b^d)")
        print()
        print("MONTE CARLO TREE SEARCH (MCTS):")
        print("• Use random simulations to evaluate positions")
        print("• Four phases: Selection, Expansion, Simulation, Backpropagation")
        print("• Works well for games with large branching factors")
        print("• Used in AlphaGo and other AI breakthroughs")
        print()
        print("APPLICATIONS:")
        print("• Chess, Go, checkers engines")
        print("• Real-time strategy games")
        print("• Puzzle solving")
        print("• Decision making under uncertainty")
    
    def demonstrate_game_trees(self) -> None:
        """Demonstrate game tree implementations"""
        print("=== GAME TREE DEMONSTRATIONS ===")
        
        print("1. TIC-TAC-TOE WITH MINIMAX")
        print("   Perfect play using minimax algorithm")
        
        game = TicTacToe()
        ai_player = MinimaxAI()
        
        print("   Initial board:")
        game.display_board()
        
        # AI makes first move
        move = ai_player.get_best_move(game, 'X')
        game.make_move(move[0], move[1], 'X')
        
        print(f"   AI plays X at position {move}")
        game.display_board()
        
        # Simulate human move
        game.make_move(1, 1, 'O')  # Center
        print("   Human plays O at center:")
        game.display_board()
        
        # AI responds
        move = ai_player.get_best_move(game, 'X')
        game.make_move(move[0], move[1], 'X')
        
        print(f"   AI responds with X at position {move}")
        game.display_board()
        print()
        
        print("2. MONTE CARLO TREE SEARCH DEMO")
        print("   Using MCTS for game tree exploration")
        
        mcts = MonteCarloTreeSearch()
        
        # Create simple game state
        game_state = SimpleGameState([1, 2, 3, 4, 5])  # Example state
        
        print("   Running MCTS simulations...")
        best_move = mcts.search(game_state, simulations=100)
        
        print(f"   Best move found: {best_move}")
        print(f"   Simulations run: 100")
        print(f"   Tree nodes created: {mcts.nodes_created}")


class TicTacToe:
    """Simple Tic-Tac-Toe game implementation"""
    
    def __init__(self):
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.current_player = 'X'
    
    def make_move(self, row: int, col: int, player: str) -> bool:
        """Make a move on the board"""
        if self.is_valid_move(row, col):
            self.board[row][col] = player
            return True
        return False
    
    def is_valid_move(self, row: int, col: int) -> bool:
        """Check if move is valid"""
        return (0 <= row < 3 and 0 <= col < 3 and 
                self.board[row][col] == ' ')
    
    def get_winner(self) -> Optional[str]:
        """Check for winner"""
        # Check rows
        for row in self.board:
            if row[0] == row[1] == row[2] != ' ':
                return row[0]
        
        # Check columns
        for col in range(3):
            if self.board[0][col] == self.board[1][col] == self.board[2][col] != ' ':
                return self.board[0][col]
        
        # Check diagonals
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != ' ':
            return self.board[0][0]
        
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != ' ':
            return self.board[0][2]
        
        return None
    
    def is_game_over(self) -> bool:
        """Check if game is over"""
        return self.get_winner() is not None or self.is_board_full()
    
    def is_board_full(self) -> bool:
        """Check if board is full"""
        return all(cell != ' ' for row in self.board for cell in row)
    
    def get_available_moves(self) -> List[Tuple[int, int]]:
        """Get list of available moves"""
        moves = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == ' ':
                    moves.append((i, j))
        return moves
    
    def copy(self) -> 'TicTacToe':
        """Create a copy of the game state"""
        new_game = TicTacToe()
        new_game.board = [row[:] for row in self.board]
        new_game.current_player = self.current_player
        return new_game
    
    def display_board(self) -> None:
        """Display the current board"""
        print("     0   1   2")
        for i, row in enumerate(self.board):
            print(f"   {i} {' | '.join(row)}")
            if i < 2:
                print("     ---------")


class MinimaxAI:
    """Minimax AI for Tic-Tac-Toe"""
    
    def get_best_move(self, game: TicTacToe, player: str) -> Tuple[int, int]:
        """Get best move using minimax algorithm"""
        _, move = self.minimax(game, player, player, -float('inf'), float('inf'))
        return move
    
    def minimax(self, game: TicTacToe, player: str, original_player: str, 
                alpha: float, beta: float) -> Tuple[int, Optional[Tuple[int, int]]]:
        """
        Minimax with alpha-beta pruning
        
        Returns: (score, best_move)
        """
        winner = game.get_winner()
        
        # Terminal states
        if winner == original_player:
            return 1, None
        elif winner is not None:
            return -1, None
        elif game.is_board_full():
            return 0, None
        
        moves = game.get_available_moves()
        best_move = moves[0] if moves else None
        
        if player == original_player:
            # Maximizing player
            max_score = -float('inf')
            
            for move in moves:
                game_copy = game.copy()
                game_copy.make_move(move[0], move[1], player)
                
                opponent = 'O' if player == 'X' else 'X'
                score, _ = self.minimax(game_copy, opponent, original_player, alpha, beta)
                
                if score > max_score:
                    max_score = score
                    best_move = move
                
                alpha = max(alpha, score)
                if beta <= alpha:
                    break  # Alpha-beta pruning
            
            return max_score, best_move
        else:
            # Minimizing player
            min_score = float('inf')
            
            for move in moves:
                game_copy = game.copy()
                game_copy.make_move(move[0], move[1], player)
                
                opponent = 'O' if player == 'X' else 'X'
                score, _ = self.minimax(game_copy, opponent, original_player, alpha, beta)
                
                if score < min_score:
                    min_score = score
                    best_move = move
                
                beta = min(beta, score)
                if beta <= alpha:
                    break  # Alpha-beta pruning
            
            return min_score, best_move


class MCTSNode:
    """Node in Monte Carlo Tree Search"""
    
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = state.get_legal_actions()[:]
    
    def is_fully_expanded(self) -> bool:
        """Check if all actions have been tried"""
        return len(self.untried_actions) == 0
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal state"""
        return self.state.is_terminal()
    
    def uct_value(self, exploration_constant: float = 1.41) -> float:
        """Calculate UCT (Upper Confidence Bound for Trees) value"""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.value / self.visits
        exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
        
        return exploitation + exploration


class SimpleGameState:
    """Simple game state for MCTS demonstration"""
    
    def __init__(self, state: List[int]):
        self.state = state[:]
        self.current_player = 1
    
    def get_legal_actions(self) -> List[int]:
        """Get list of legal actions"""
        return [i for i, val in enumerate(self.state) if val > 0]
    
    def apply_action(self, action: int) -> 'SimpleGameState':
        """Apply action and return new state"""
        new_state = self.state[:]
        if new_state[action] > 0:
            new_state[action] -= 1
        
        new_game_state = SimpleGameState(new_state)
        new_game_state.current_player = -self.current_player
        return new_game_state
    
    def is_terminal(self) -> bool:
        """Check if game is over"""
        return sum(self.state) == 0
    
    def get_reward(self, player: int) -> float:
        """Get reward for player"""
        if self.is_terminal():
            return 1.0 if player == self.current_player else 0.0
        return 0.0


class MonteCarloTreeSearch:
    """Monte Carlo Tree Search implementation"""
    
    def __init__(self):
        self.nodes_created = 0
    
    def search(self, root_state, simulations: int = 1000) -> Any:
        """Run MCTS and return best action"""
        root = MCTSNode(root_state)
        self.nodes_created = 1
        
        for _ in range(simulations):
            # Selection
            node = self._select(root)
            
            # Expansion
            if not node.is_terminal() and not node.is_fully_expanded():
                node = self._expand(node)
            
            # Simulation
            reward = self._simulate(node.state)
            
            # Backpropagation
            self._backpropagate(node, reward)
        
        # Return best action
        if root.children:
            best_child = max(root.children, key=lambda c: c.visits)
            return best_child.action
        
        return None
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select node using UCT"""
        while not node.is_terminal() and node.is_fully_expanded():
            node = max(node.children, key=lambda c: c.uct_value())
        return node
    
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expand node by adding a child"""
        action = node.untried_actions.pop()
        new_state = node.state.apply_action(action)
        child = MCTSNode(new_state, parent=node, action=action)
        node.children.append(child)
        self.nodes_created += 1
        return child
    
    def _simulate(self, state) -> float:
        """Simulate random game from state"""
        current_state = state
        player = current_state.current_player
        
        while not current_state.is_terminal():
            actions = current_state.get_legal_actions()
            if not actions:
                break
            
            action = random.choice(actions)
            current_state = current_state.apply_action(action)
        
        return current_state.get_reward(player)
    
    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        """Backpropagate reward through tree"""
        while node is not None:
            node.visits += 1
            node.value += reward
            reward = 1 - reward  # Flip reward for opponent
            node = node.parent


# ==========================================
# 3. PARSING AND COMPILER TREES
# ==========================================

class ParsingTrees:
    """
    Trees for parsing and compiler applications
    """
    
    def explain_parsing_trees(self) -> None:
        """Explain parsing tree concepts"""
        print("=== PARSING TREES ===")
        print("Trees for language processing and compilation")
        print()
        print("ABSTRACT SYNTAX TREES (AST):")
        print("• Represent program structure")
        print("• Nodes: operators, keywords, constructs")
        print("• Leaves: identifiers, literals, constants")
        print("• Used for: compilation, interpretation, analysis")
        print()
        print("PARSE TREES:")
        print("• Concrete syntax representation")
        print("• Include all grammar productions")
        print("• More detailed than AST")
        print("• Used for: syntax analysis, error recovery")
        print()
        print("EXPRESSION TREES:")
        print("• Mathematical and logical expressions")
        print("• Binary operators: internal nodes")
        print("• Operands: leaf nodes")
        print("• Support: evaluation, optimization, transformation")
        print()
        print("APPLICATIONS:")
        print("• Programming language compilers")
        print("• Mathematical expression evaluators")
        print("• Query processors in databases")
        print("• Configuration file parsers")
    
    def demonstrate_parsing_trees(self) -> None:
        """Demonstrate parsing tree implementations"""
        print("=== PARSING TREE DEMONSTRATIONS ===")
        
        print("1. EXPRESSION TREE PARSER")
        print("   Parsing mathematical expressions into trees")
        
        parser = ExpressionParser()
        
        expressions = [
            "3 + 4 * 2",
            "(1 + 2) * (3 + 4)",
            "2 ^ 3 ^ 2",  # Right associative
            "10 - 5 - 2"  # Left associative
        ]
        
        for expr in expressions:
            print(f"   Expression: {expr}")
            tree = parser.parse(expr)
            
            print(f"     Infix: {parser.to_infix(tree)}")
            print(f"     Prefix: {parser.to_prefix(tree)}")
            print(f"     Postfix: {parser.to_postfix(tree)}")
            print(f"     Result: {parser.evaluate(tree)}")
            print()
        
        print("2. SIMPLE LANGUAGE AST")
        print("   Building AST for simple programming language")
        
        ast_builder = SimpleASTBuilder()
        
        # Simple program: x = 5; y = x + 3; print(y);
        program = """
        x = 5;
        y = x + 3;
        print(y);
        """
        
        print(f"   Program:")
        print(program)
        
        ast = ast_builder.parse_program(program)
        
        print("   AST structure:")
        ast_builder.display_ast(ast)
        
        print("   Interpretation result:")
        interpreter = SimpleInterpreter()
        interpreter.execute(ast)


class ExpressionTreeNode:
    """Node in expression tree"""
    
    def __init__(self, value: str, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right
        self.is_operator = value in "+-*/^"
    
    def __str__(self):
        return self.value


class ExpressionParser:
    """
    Recursive descent parser for mathematical expressions
    
    Grammar:
    Expression -> Term (('+' | '-') Term)*
    Term -> Factor (('*' | '/') Factor)*
    Factor -> Power
    Power -> Primary ('^' Primary)*
    Primary -> Number | '(' Expression ')'
    """
    
    def __init__(self):
        self.tokens = []
        self.pos = 0
        self.precedence = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3}
        self.right_associative = {'^'}
    
    def parse(self, expression: str) -> ExpressionTreeNode:
        """Parse expression into tree"""
        self.tokens = self._tokenize(expression)
        self.pos = 0
        return self._parse_expression()
    
    def _tokenize(self, expression: str) -> List[str]:
        """Tokenize expression"""
        tokens = []
        i = 0
        
        while i < len(expression):
            if expression[i].isspace():
                i += 1
            elif expression[i].isdigit():
                num = ""
                while i < len(expression) and (expression[i].isdigit() or expression[i] == '.'):
                    num += expression[i]
                    i += 1
                tokens.append(num)
            else:
                tokens.append(expression[i])
                i += 1
        
        return tokens
    
    def _parse_expression(self) -> ExpressionTreeNode:
        """Parse addition and subtraction"""
        left = self._parse_term()
        
        while self.pos < len(self.tokens) and self.tokens[self.pos] in "+-":
            operator = self.tokens[self.pos]
            self.pos += 1
            right = self._parse_term()
            left = ExpressionTreeNode(operator, left, right)
        
        return left
    
    def _parse_term(self) -> ExpressionTreeNode:
        """Parse multiplication and division"""
        left = self._parse_power()
        
        while self.pos < len(self.tokens) and self.tokens[self.pos] in "*/":
            operator = self.tokens[self.pos]
            self.pos += 1
            right = self._parse_power()
            left = ExpressionTreeNode(operator, left, right)
        
        return left
    
    def _parse_power(self) -> ExpressionTreeNode:
        """Parse exponentiation (right associative)"""
        left = self._parse_primary()
        
        if self.pos < len(self.tokens) and self.tokens[self.pos] == '^':
            operator = self.tokens[self.pos]
            self.pos += 1
            right = self._parse_power()  # Right associative
            return ExpressionTreeNode(operator, left, right)
        
        return left
    
    def _parse_primary(self) -> ExpressionTreeNode:
        """Parse numbers and parenthesized expressions"""
        if self.pos >= len(self.tokens):
            raise ValueError("Unexpected end of expression")
        
        token = self.tokens[self.pos]
        
        if token == '(':
            self.pos += 1
            node = self._parse_expression()
            if self.pos >= len(self.tokens) or self.tokens[self.pos] != ')':
                raise ValueError("Missing closing parenthesis")
            self.pos += 1
            return node
        elif token.replace('.', '').isdigit():
            self.pos += 1
            return ExpressionTreeNode(token)
        else:
            raise ValueError(f"Unexpected token: {token}")
    
    def evaluate(self, node: ExpressionTreeNode) -> float:
        """Evaluate expression tree"""
        if not node.is_operator:
            return float(node.value)
        
        left_val = self.evaluate(node.left)
        right_val = self.evaluate(node.right)
        
        if node.value == '+':
            return left_val + right_val
        elif node.value == '-':
            return left_val - right_val
        elif node.value == '*':
            return left_val * right_val
        elif node.value == '/':
            return left_val / right_val if right_val != 0 else float('inf')
        elif node.value == '^':
            return left_val ** right_val
        
        return 0
    
    def to_infix(self, node: ExpressionTreeNode) -> str:
        """Convert tree to infix notation"""
        if not node.is_operator:
            return node.value
        
        left = self.to_infix(node.left)
        right = self.to_infix(node.right)
        
        return f"({left} {node.value} {right})"
    
    def to_prefix(self, node: ExpressionTreeNode) -> str:
        """Convert tree to prefix notation"""
        if not node.is_operator:
            return node.value
        
        left = self.to_prefix(node.left)
        right = self.to_prefix(node.right)
        
        return f"{node.value} {left} {right}"
    
    def to_postfix(self, node: ExpressionTreeNode) -> str:
        """Convert tree to postfix notation"""
        if not node.is_operator:
            return node.value
        
        left = self.to_postfix(node.left)
        right = self.to_postfix(node.right)
        
        return f"{left} {right} {node.value}"


class ASTNode:
    """Base class for AST nodes"""
    
    def __init__(self, node_type: str):
        self.node_type = node_type


class AssignmentNode(ASTNode):
    """Assignment statement node"""
    
    def __init__(self, variable: str, expression):
        super().__init__("assignment")
        self.variable = variable
        self.expression = expression


class BinaryOpNode(ASTNode):
    """Binary operation node"""
    
    def __init__(self, operator: str, left, right):
        super().__init__("binary_op")
        self.operator = operator
        self.left = left
        self.right = right


class VariableNode(ASTNode):
    """Variable reference node"""
    
    def __init__(self, name: str):
        super().__init__("variable")
        self.name = name


class LiteralNode(ASTNode):
    """Literal value node"""
    
    def __init__(self, value):
        super().__init__("literal")
        self.value = value


class PrintNode(ASTNode):
    """Print statement node"""
    
    def __init__(self, expression):
        super().__init__("print")
        self.expression = expression


class ProgramNode(ASTNode):
    """Program root node"""
    
    def __init__(self, statements: List[ASTNode]):
        super().__init__("program")
        self.statements = statements


class SimpleASTBuilder:
    """Simple AST builder for demonstration"""
    
    def parse_program(self, program: str) -> ProgramNode:
        """Parse simple program into AST"""
        lines = [line.strip() for line in program.strip().split('\n') if line.strip()]
        statements = []
        
        for line in lines:
            if line.endswith(';'):
                line = line[:-1]  # Remove semicolon
            
            if '=' in line and not line.startswith('print'):
                # Assignment statement
                var, expr = line.split('=', 1)
                var = var.strip()
                expr = expr.strip()
                
                statements.append(AssignmentNode(var, self._parse_expression(expr)))
            elif line.startswith('print(') and line.endswith(')'):
                # Print statement
                expr = line[6:-1]  # Remove 'print(' and ')'
                statements.append(PrintNode(self._parse_expression(expr)))
        
        return ProgramNode(statements)
    
    def _parse_expression(self, expr: str) -> ASTNode:
        """Parse simple expression"""
        expr = expr.strip()
        
        # Simple binary operations
        for op in ['+', '-', '*', '/']:
            if op in expr:
                parts = expr.split(op, 1)
                if len(parts) == 2:
                    left = self._parse_expression(parts[0].strip())
                    right = self._parse_expression(parts[1].strip())
                    return BinaryOpNode(op, left, right)
        
        # Variable or literal
        if expr.isdigit():
            return LiteralNode(int(expr))
        else:
            return VariableNode(expr)
    
    def display_ast(self, node: ASTNode, depth: int = 0) -> None:
        """Display AST structure"""
        indent = "  " * depth
        
        if isinstance(node, ProgramNode):
            print(f"{indent}Program")
            for stmt in node.statements:
                self.display_ast(stmt, depth + 1)
        elif isinstance(node, AssignmentNode):
            print(f"{indent}Assignment: {node.variable} =")
            self.display_ast(node.expression, depth + 1)
        elif isinstance(node, BinaryOpNode):
            print(f"{indent}BinaryOp: {node.operator}")
            self.display_ast(node.left, depth + 1)
            self.display_ast(node.right, depth + 1)
        elif isinstance(node, VariableNode):
            print(f"{indent}Variable: {node.name}")
        elif isinstance(node, LiteralNode):
            print(f"{indent}Literal: {node.value}")
        elif isinstance(node, PrintNode):
            print(f"{indent}Print:")
            self.display_ast(node.expression, depth + 1)


class SimpleInterpreter:
    """Simple interpreter for AST"""
    
    def __init__(self):
        self.variables = {}
    
    def execute(self, node: ASTNode) -> Any:
        """Execute AST node"""
        if isinstance(node, ProgramNode):
            for stmt in node.statements:
                self.execute(stmt)
        elif isinstance(node, AssignmentNode):
            value = self.execute(node.expression)
            self.variables[node.variable] = value
            print(f"   {node.variable} = {value}")
        elif isinstance(node, PrintNode):
            value = self.execute(node.expression)
            print(f"   Output: {value}")
        elif isinstance(node, BinaryOpNode):
            left = self.execute(node.left)
            right = self.execute(node.right)
            
            if node.operator == '+':
                return left + right
            elif node.operator == '-':
                return left - right
            elif node.operator == '*':
                return left * right
            elif node.operator == '/':
                return left / right if right != 0 else 0
        elif isinstance(node, VariableNode):
            return self.variables.get(node.name, 0)
        elif isinstance(node, LiteralNode):
            return node.value
        
        return None


# ==========================================
# DEMONSTRATION AND TESTING
# ==========================================

def demonstrate_specialized_trees():
    """Demonstrate all specialized tree structures"""
    print("=== SPECIALIZED TREE STRUCTURES COMPREHENSIVE GUIDE ===\n")
    
    specialized = SpecializedTrees()
    
    # 1. Spatial trees
    specialized.explain_spatial_trees()
    print("\n" + "="*60 + "\n")
    
    specialized.demonstrate_spatial_trees()
    print("\n" + "="*60 + "\n")
    
    # 2. Game trees
    game_trees = GameTrees()
    game_trees.explain_game_trees()
    print("\n" + "="*60 + "\n")
    
    game_trees.demonstrate_game_trees()
    print("\n" + "="*60 + "\n")
    
    # 3. Parsing trees
    parsing_trees = ParsingTrees()
    parsing_trees.explain_parsing_trees()
    print("\n" + "="*60 + "\n")
    
    parsing_trees.demonstrate_parsing_trees()


if __name__ == "__main__":
    demonstrate_specialized_trees()
    
    print("\n=== SPECIALIZED TREES MASTERY GUIDE ===")
    
    print("\n🎯 SPECIALIZED TREE CATEGORIES:")
    print("• Spatial Trees: Multi-dimensional data organization")
    print("• Game Trees: Adversarial search and game AI")
    print("• Parsing Trees: Language processing and compilation")
    print("• Domain-Specific Trees: Application-tailored structures")
    
    print("\n📊 COMPLEXITY CHARACTERISTICS:")
    print("• Spatial trees: O(log n) to O(n) depending on dimensionality")
    print("• Game trees: O(b^d) where b=branching factor, d=depth")
    print("• Parsing trees: O(n) for parsing, O(depth) for evaluation")
    print("• Optimizations can reduce complexity significantly")
    
    print("\n⚡ APPLICATION-SPECIFIC OPTIMIZATIONS:")
    print("• Spatial: Choose structure based on data distribution")
    print("• Game: Alpha-beta pruning, transposition tables, move ordering")
    print("• Parsing: Left-factoring, operator precedence, error recovery")
    print("• General: Cache-friendly layouts, parallel processing")
    
    print("\n🔧 IMPLEMENTATION STRATEGIES:")
    print("• Understand domain-specific requirements deeply")
    print("• Choose appropriate splitting/organization criteria")
    print("• Implement efficient pruning and optimization techniques")
    print("• Consider real-time performance constraints")
    print("• Plan for scalability and large datasets")
    
    print("\n🏆 REAL-WORLD APPLICATIONS:")
    print("• GIS and mapping: R-trees, Quadtrees for spatial indexing")
    print("• Game AI: Minimax, MCTS for strategy games")
    print("• Compilers: ASTs for program representation and optimization")
    print("• Graphics: Octrees for 3D rendering and collision detection")
    print("• Databases: Spatial indexes for location-based queries")
    
    print("\n🎓 MASTERY ROADMAP:")
    print("1. Understand the domain problem thoroughly")
    print("2. Study existing specialized solutions")
    print("3. Learn domain-specific optimization techniques")
    print("4. Practice with realistic datasets and constraints")
    print("5. Benchmark against established implementations")
    print("6. Consider hybrid approaches for complex scenarios")
    
    print("\n💡 SUCCESS PRINCIPLES:")
    print("• Domain expertise is as important as algorithmic knowledge")
    print("• Profile and optimize for actual usage patterns")
    print("• Consider both theoretical and practical performance")
    print("• Learn from existing high-quality implementations")
    print("• Stay updated with domain-specific research and advances")
