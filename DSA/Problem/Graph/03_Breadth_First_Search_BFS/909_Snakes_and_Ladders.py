"""
909. Snakes and Ladders
Difficulty: Medium

Problem:
You are given an n x n integer matrix board where the cells are labeled from 1 to n^2 
in a Boustrophedon style starting from the bottom left of the board (i.e. board[n - 1][0]) 
and alternating direction each row.

You start on square 1. In each move, you roll a six-sided die and move forward by that number 
of steps. If you land on the bottom of a ladder, you immediately move to the top of the ladder. 
If you land on a mouth of a snake, you immediately slide down to the tail of the snake.

Return the least number of moves required to reach square n^2. If it is not possible to 
reach the square, return -1.

Examples:
Input: board = [[-1,-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1,-1]]
Output: 2

Input: board = [[-1,-1],[-1,3]]
Output: 1

Constraints:
- n == board.length == board[i].length
- 2 <= n <= 20
- board[i][j] is either -1 or in the range [1, n^2]
- The squares labeled 1 and n^2 do not have any ladders or snakes
"""

from typing import List
from collections import deque

class Solution:
    def snakesAndLadders_approach1_bfs_with_coordinate_mapping(self, board: List[List[int]]) -> int:
        """
        Approach 1: BFS with Coordinate Mapping (Optimal)
        
        Use BFS to find shortest path from 1 to n^2.
        Handle Boustrophedon numbering and snakes/ladders.
        
        Time: O(N^2) - visit each square at most once
        Space: O(N^2) - queue and visited set
        """
        n = len(board)
        target = n * n
        
        def get_coordinates(num):
            """Convert square number to board coordinates"""
            num -= 1  # Convert to 0-based
            row = n - 1 - num // n  # Bottom to top
            col = num % n
            
            # Handle Boustrophedon (snake-like) pattern
            if (n - 1 - row) % 2 == 1:  # Odd rows go right to left
                col = n - 1 - col
            
            return row, col
        
        def get_destination(square):
            """Get final destination considering snakes/ladders"""
            row, col = get_coordinates(square)
            if board[row][col] != -1:
                return board[row][col]
            return square
        
        # BFS
        queue = deque([(1, 0)])  # (square, moves)
        visited = {1}
        
        while queue:
            square, moves = queue.popleft()
            
            if square == target:
                return moves
            
            # Try all dice rolls (1-6)
            for dice in range(1, 7):
                next_square = square + dice
                
                if next_square > target:
                    break
                
                # Get final destination after snakes/ladders
                final_square = get_destination(next_square)
                
                if final_square not in visited:
                    visited.add(final_square)
                    queue.append((final_square, moves + 1))
        
        return -1
    
    def snakesAndLadders_approach2_optimized_bfs(self, board: List[List[int]]) -> int:
        """
        Approach 2: Optimized BFS with Preprocessing
        
        Preprocess board to create direct mapping for efficiency.
        
        Time: O(N^2)
        Space: O(N^2)
        """
        n = len(board)
        target = n * n
        
        # Preprocess: Create mapping from square to destination
        destinations = {}
        
        for square in range(1, target + 1):
            square_idx = square - 1
            row = n - 1 - square_idx // n
            col = square_idx % n
            
            # Handle Boustrophedon pattern
            if (n - 1 - row) % 2 == 1:
                col = n - 1 - col
            
            if board[row][col] != -1:
                destinations[square] = board[row][col]
            else:
                destinations[square] = square
        
        # BFS with preprocessed destinations
        queue = deque([(1, 0)])
        visited = {1}
        
        while queue:
            square, moves = queue.popleft()
            
            for dice in range(1, 7):
                next_square = square + dice
                
                if next_square > target:
                    break
                
                final_square = destinations[next_square]
                
                if final_square == target:
                    return moves + 1
                
                if final_square not in visited:
                    visited.add(final_square)
                    queue.append((final_square, moves + 1))
        
        return -1
    
    def snakesAndLadders_approach3_level_by_level_bfs(self, board: List[List[int]]) -> int:
        """
        Approach 3: Level-by-Level BFS
        
        Process all positions at current move count before next level.
        
        Time: O(N^2)
        Space: O(N^2)
        """
        n = len(board)
        target = n * n
        
        def get_board_value(square):
            """Get board value at given square number"""
            square -= 1
            row = n - 1 - square // n
            col = square % n
            
            if (n - 1 - row) % 2 == 1:
                col = n - 1 - col
            
            return board[row][col]
        
        # Level-by-level BFS
        current_level = {1}
        visited = {1}
        moves = 0
        
        while current_level:
            next_level = set()
            
            for square in current_level:
                if square == target:
                    return moves
                
                # Try all dice outcomes
                for dice in range(1, 7):
                    next_square = square + dice
                    
                    if next_square > target:
                        break
                    
                    # Check for snake or ladder
                    board_value = get_board_value(next_square)
                    final_square = board_value if board_value != -1 else next_square
                    
                    if final_square not in visited:
                        visited.add(final_square)
                        next_level.add(final_square)
            
            current_level = next_level
            moves += 1
        
        return -1
    
    def snakesAndLadders_approach4_dijkstra_alternative(self, board: List[List[int]]) -> int:
        """
        Approach 4: Dijkstra's Algorithm (Alternative Approach)
        
        Use Dijkstra for comparison (overkill for unweighted graph).
        
        Time: O(N^2 log N)
        Space: O(N^2)
        """
        import heapq
        
        n = len(board)
        target = n * n
        
        def get_destination(square):
            """Get final destination considering snakes/ladders"""
            square -= 1
            row = n - 1 - square // n
            col = square % n
            
            if (n - 1 - row) % 2 == 1:
                col = n - 1 - col
            
            if board[row][col] != -1:
                return board[row][col]
            return square + 1
        
        # Dijkstra's algorithm
        pq = [(0, 1)]  # (moves, square)
        distances = {1: 0}
        
        while pq:
            moves, square = heapq.heappop(pq)
            
            if square == target:
                return moves
            
            if moves > distances.get(square, float('inf')):
                continue
            
            for dice in range(1, 7):
                next_square = square + dice
                
                if next_square > target:
                    break
                
                final_square = get_destination(next_square)
                new_moves = moves + 1
                
                if new_moves < distances.get(final_square, float('inf')):
                    distances[final_square] = new_moves
                    heapq.heappush(pq, (new_moves, final_square))
        
        return -1

def test_snakes_and_ladders():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (board, expected)
        ([[-1,-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1,-1]], 2),
        ([[-1,-1],[-1,3]], 1),
        ([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]], 1),
        ([[-1,1,2,-1],[2,13,15,-1],[-1,10,-1,-1],[-1,6,2,8]], 2),
    ]
    
    approaches = [
        ("BFS with Coordinate Mapping", solution.snakesAndLadders_approach1_bfs_with_coordinate_mapping),
        ("Optimized BFS", solution.snakesAndLadders_approach2_optimized_bfs),
        ("Level-by-Level BFS", solution.snakesAndLadders_approach3_level_by_level_bfs),
        ("Dijkstra Alternative", solution.snakesAndLadders_approach4_dijkstra_alternative),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (board, expected) in enumerate(test_cases):
            result = func(board)
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} Expected: {expected}, Got: {result}")

def demonstrate_board_mapping():
    """Demonstrate Boustrophedon board mapping"""
    print("\n=== Boustrophedon Board Mapping Demo ===")
    
    n = 4
    print(f"4x4 Board numbering (Boustrophedon style):")
    
    # Create visual representation
    board_visual = [[0] * n for _ in range(n)]
    
    for square in range(1, n * n + 1):
        square_idx = square - 1
        row = n - 1 - square_idx // n
        col = square_idx % n
        
        # Handle Boustrophedon pattern
        if (n - 1 - row) % 2 == 1:
            col = n - 1 - col
        
        board_visual[row][col] = square
    
    for row in board_visual:
        print(f"  {row}")
    
    print(f"\nCoordinate mapping examples:")
    examples = [1, 4, 5, 8, 9, 12, 13, 16]
    
    for square in examples:
        square_idx = square - 1
        row = n - 1 - square_idx // n
        col = square_idx % n
        
        if (n - 1 - row) % 2 == 1:
            col = n - 1 - col
        
        print(f"  Square {square:2d} -> ({row}, {col})")

def demonstrate_game_simulation():
    """Demonstrate snakes and ladders game simulation"""
    print("\n=== Game Simulation Demo ===")
    
    board = [[-1,-1,-1,-1],
             [-1,-1,-1,-1],
             [-1,13,-1,-1],
             [-1,-1,-1,-1]]
    
    n = len(board)
    target = n * n
    
    print(f"Board size: {n}x{n}, Target: {target}")
    print("Board configuration:")
    for i, row in enumerate(board):
        print(f"  Row {i}: {row}")
    
    # Show snakes and ladders
    snakes_ladders = []
    for square in range(1, target + 1):
        square_idx = square - 1
        row = n - 1 - square_idx // n
        col = square_idx % n
        
        if (n - 1 - row) % 2 == 1:
            col = n - 1 - col
        
        if board[row][col] != -1:
            if board[row][col] > square:
                snakes_ladders.append(f"Ladder: {square} -> {board[row][col]}")
            else:
                snakes_ladders.append(f"Snake: {square} -> {board[row][col]}")
    
    if snakes_ladders:
        print(f"\nSnakes and Ladders:")
        for item in snakes_ladders:
            print(f"  {item}")
    else:
        print(f"\nNo snakes or ladders on this board")
    
    # BFS simulation
    queue = deque([(1, 0, [1])])
    visited = {1}
    
    print(f"\nBFS simulation:")
    
    while queue:
        square, moves, path = queue.popleft()
        
        print(f"  Move {moves}: At square {square}, Path: {path}")
        
        if square == target:
            print(f"  🎯 Reached target in {moves} moves!")
            break
        
        valid_moves = []
        for dice in range(1, 7):
            next_square = square + dice
            
            if next_square > target:
                break
            
            # Check for snakes/ladders
            square_idx = next_square - 1
            row = n - 1 - square_idx // n
            col = square_idx % n
            
            if (n - 1 - row) % 2 == 1:
                col = n - 1 - col
            
            final_square = next_square
            if board[row][col] != -1:
                final_square = board[row][col]
            
            if final_square not in visited:
                visited.add(final_square)
                queue.append((final_square, moves + 1, path + [final_square]))
                
                if final_square != next_square:
                    valid_moves.append(f"dice={dice}: {next_square}->{final_square}")
                else:
                    valid_moves.append(f"dice={dice}: {final_square}")
        
        if valid_moves:
            print(f"    Next moves: {valid_moves}")

def analyze_game_complexity():
    """Analyze complexity of snakes and ladders problem"""
    print("\n=== Game Complexity Analysis ===")
    
    print("Problem Characteristics:")
    print("• State space: Squares 1 to N^2")
    print("• Transitions: 1-6 steps forward (dice roll)")
    print("• Special moves: Snakes (down) and ladders (up)")
    print("• Goal: Minimum moves to reach N^2")
    
    print("\nKey Challenges:")
    print("1. **Boustrophedon Numbering:**")
    print("   • Complex coordinate mapping")
    print("   • Alternating row directions")
    print("   • Requires careful indexing")
    
    print("\n2. **Snakes and Ladders:**")
    print("   • Immediate forced moves")
    print("   • Can help (ladders) or hurt (snakes)")
    print("   • Affect reachability and optimal paths")
    
    print("\n3. **BFS Optimality:**")
    print("   • Guaranteed shortest path")
    print("   • Level-order exploration")
    print("   • Early termination possible")
    
    print("\nAlgorithm Insights:")
    print("• Model as unweighted directed graph")
    print("• Each square is a node")
    print("• Dice rolls create edges (with constraints)")
    print("• Snakes/ladders modify edge destinations")
    print("• BFS finds minimum moves")
    
    print("\nReal-world Applications:")
    print("• Board game AI")
    print("• Game theory analysis")
    print("• Probability calculations")
    print("• Monte Carlo simulations")
    print("• Decision making under uncertainty")

if __name__ == "__main__":
    test_snakes_and_ladders()
    demonstrate_board_mapping()
    demonstrate_game_simulation()
    analyze_game_complexity()

"""
Graph Theory Concepts:
1. Board Game Modeling as Graphs
2. Boustrophedon Pattern Navigation
3. Forced Moves and State Transitions
4. Shortest Path in Game States

Key Game Modeling Insights:
- Each board square is a graph node
- Dice rolls create directed edges (1-6 steps)
- Snakes/ladders modify destination states
- BFS finds minimum moves to goal

Boustrophedon Complexity:
- Bottom-left start with snake-like numbering
- Alternating row directions
- Complex coordinate transformation
- Critical for correct state mapping

Algorithm Strategy:
- Model board as directed graph
- Handle coordinate mapping carefully
- Use BFS for shortest path guarantee
- Process snakes/ladders as immediate transitions

Real-world Applications:
- Board game development
- Game AI and strategy
- Probability analysis
- Educational game design
- Decision making systems

This problem demonstrates graph modeling
of complex board games with special rules.
"""
