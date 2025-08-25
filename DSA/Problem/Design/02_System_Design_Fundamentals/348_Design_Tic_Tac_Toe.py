"""
348. Design Tic-Tac-Toe - Multiple Approaches
Difficulty: Medium

Assume the following rules are for the tic-tac-toe game on an n x n board between two players:

1. A player wins if any row, column, or diagonal is completely filled with their symbols.
2. A game is over when there is a winner or all spaces are filled.

Implement the TicTacToe class:
- TicTacToe(int n) Initializes the object with the size of the board n.
- int move(int row, int col, int player) Indicates that the player (1 or 2) makes a move at the position (row, col). Returns:
  - 0 if the game is ongoing after the move
  - 1 if player 1 wins after the move
  - 2 if player 2 wins after the move
"""

from typing import List, Dict, Set, Tuple, Optional

class TicTacToeBasic:
    """
    Approach 1: Basic Board with Full Check
    
    Store the board and check all win conditions after each move.
    
    Time Complexity: 
    - move: O(n) for checking win conditions
    
    Space Complexity: O(n²)
    """
    
    def __init__(self, n: int):
        self.n = n
        self.board = [[0] * n for _ in range(n)]
    
    def move(self, row: int, col: int, player: int) -> int:
        self.board[row][col] = player
        
        if self._check_win(row, col, player):
            return player
        
        return 0
    
    def _check_win(self, row: int, col: int, player: int) -> bool:
        # Check row
        if all(self.board[row][c] == player for c in range(self.n)):
            return True
        
        # Check column
        if all(self.board[r][col] == player for r in range(self.n)):
            return True
        
        # Check main diagonal
        if row == col and all(self.board[i][i] == player for i in range(self.n)):
            return True
        
        # Check anti-diagonal
        if row + col == self.n - 1 and all(self.board[i][self.n - 1 - i] == player for i in range(self.n)):
            return True
        
        return False

class TicTacToeOptimized:
    """
    Approach 2: Optimized with Counters
    
    Use counters to track progress towards winning conditions.
    
    Time Complexity: 
    - move: O(1)
    
    Space Complexity: O(n)
    """
    
    def __init__(self, n: int):
        self.n = n
        # Track count for each player (1 and 2)
        self.rows = [[0, 0] for _ in range(n)]  # rows[i][player-1]
        self.cols = [[0, 0] for _ in range(n)]  # cols[i][player-1]
        self.diagonal = [0, 0]  # Main diagonal
        self.anti_diagonal = [0, 0]  # Anti-diagonal
    
    def move(self, row: int, col: int, player: int) -> int:
        player_idx = player - 1  # Convert to 0-indexed
        
        # Update counters
        self.rows[row][player_idx] += 1
        self.cols[col][player_idx] += 1
        
        if row == col:
            self.diagonal[player_idx] += 1
        
        if row + col == self.n - 1:
            self.anti_diagonal[player_idx] += 1
        
        # Check win conditions
        if (self.rows[row][player_idx] == self.n or
            self.cols[col][player_idx] == self.n or
            self.diagonal[player_idx] == self.n or
            self.anti_diagonal[player_idx] == self.n):
            return player
        
        return 0

class TicTacToeCompact:
    """
    Approach 3: Compact with Single Array
    
    Use a single array with positive/negative counting.
    
    Time Complexity: 
    - move: O(1)
    
    Space Complexity: O(n)
    """
    
    def __init__(self, n: int):
        self.n = n
        # Use positive for player 1, negative for player 2
        self.rows = [0] * n
        self.cols = [0] * n
        self.diagonal = 0
        self.anti_diagonal = 0
    
    def move(self, row: int, col: int, player: int) -> int:
        # Player 1: +1, Player 2: -1
        to_add = 1 if player == 1 else -1
        
        # Update counters
        self.rows[row] += to_add
        self.cols[col] += to_add
        
        if row == col:
            self.diagonal += to_add
        
        if row + col == self.n - 1:
            self.anti_diagonal += to_add
        
        # Check win conditions
        target = self.n if player == 1 else -self.n
        
        if (self.rows[row] == target or
            self.cols[col] == target or
            self.diagonal == target or
            self.anti_diagonal == target):
            return player
        
        return 0

class TicTacToeWithHistory:
    """
    Approach 4: Enhanced with Move History and Undo
    
    Track game history and support undo operations.
    
    Time Complexity: 
    - move: O(1)
    - undo: O(1)
    
    Space Complexity: O(n + m) where m is number of moves
    """
    
    def __init__(self, n: int):
        self.n = n
        self.rows = [0] * n
        self.cols = [0] * n
        self.diagonal = 0
        self.anti_diagonal = 0
        self.move_history = []  # (row, col, player, game_state)
        self.game_over = False
        self.winner = 0
    
    def move(self, row: int, col: int, player: int) -> int:
        if self.game_over:
            return self.winner
        
        # Save state before move
        prev_state = {
            'rows': self.rows[:],
            'cols': self.cols[:],
            'diagonal': self.diagonal,
            'anti_diagonal': self.anti_diagonal
        }
        
        # Player 1: +1, Player 2: -1
        to_add = 1 if player == 1 else -1
        
        # Update counters
        self.rows[row] += to_add
        self.cols[col] += to_add
        
        if row == col:
            self.diagonal += to_add
        
        if row + col == self.n - 1:
            self.anti_diagonal += to_add
        
        # Check win conditions
        target = self.n if player == 1 else -self.n
        
        result = 0
        if (self.rows[row] == target or
            self.cols[col] == target or
            self.diagonal == target or
            self.anti_diagonal == target):
            result = player
            self.game_over = True
            self.winner = player
        
        # Save move to history
        self.move_history.append((row, col, player, prev_state))
        
        return result
    
    def undo(self) -> bool:
        """Undo the last move"""
        if not self.move_history:
            return False
        
        row, col, player, prev_state = self.move_history.pop()
        
        # Restore previous state
        self.rows = prev_state['rows']
        self.cols = prev_state['cols']
        self.diagonal = prev_state['diagonal']
        self.anti_diagonal = prev_state['anti_diagonal']
        
        self.game_over = False
        self.winner = 0
        
        return True
    
    def get_move_count(self) -> int:
        return len(self.move_history)
    
    def get_last_move(self) -> Optional[Tuple[int, int, int]]:
        if not self.move_history:
            return None
        return self.move_history[-1][:3]

class TicTacToeMultiGame:
    """
    Approach 5: Multi-Game Tournament Support
    
    Support multiple simultaneous games and statistics.
    
    Time Complexity: 
    - move: O(1)
    
    Space Complexity: O(n)
    """
    
    def __init__(self, n: int):
        self.n = n
        self.reset_game()
        
        # Tournament statistics
        self.games_played = 0
        self.player1_wins = 0
        self.player2_wins = 0
        self.draws = 0
        self.total_moves = 0
    
    def reset_game(self) -> None:
        """Reset for a new game"""
        self.rows = [0] * self.n
        self.cols = [0] * self.n
        self.diagonal = 0
        self.anti_diagonal = 0
        self.current_moves = 0
        self.game_active = True
    
    def move(self, row: int, col: int, player: int) -> int:
        if not self.game_active:
            return 0
        
        to_add = 1 if player == 1 else -1
        
        # Update counters
        self.rows[row] += to_add
        self.cols[col] += to_add
        
        if row == col:
            self.diagonal += to_add
        
        if row + col == self.n - 1:
            self.anti_diagonal += to_add
        
        self.current_moves += 1
        self.total_moves += 1
        
        # Check win conditions
        target = self.n if player == 1 else -self.n
        
        if (self.rows[row] == target or
            self.cols[col] == target or
            self.diagonal == target or
            self.anti_diagonal == target):
            
            self.game_active = False
            self.games_played += 1
            
            if player == 1:
                self.player1_wins += 1
            else:
                self.player2_wins += 1
            
            return player
        
        # Check for draw
        if self.current_moves == self.n * self.n:
            self.game_active = False
            self.games_played += 1
            self.draws += 1
        
        return 0
    
    def get_statistics(self) -> Dict[str, int]:
        """Get tournament statistics"""
        return {
            'games_played': self.games_played,
            'player1_wins': self.player1_wins,
            'player2_wins': self.player2_wins,
            'draws': self.draws,
            'total_moves': self.total_moves,
            'avg_moves_per_game': self.total_moves / max(1, self.games_played)
        }


def test_tic_tac_toe_basic():
    """Test basic tic-tac-toe functionality"""
    print("=== Testing Basic Tic-Tac-Toe Functionality ===")
    
    implementations = [
        ("Basic Board", TicTacToeBasic),
        ("Optimized Counters", TicTacToeOptimized),
        ("Compact Array", TicTacToeCompact),
        ("With History", TicTacToeWithHistory),
        ("Multi-Game", TicTacToeMultiGame)
    ]
    
    for name, TicTacToeClass in implementations:
        print(f"\n{name}:")
        
        game = TicTacToeClass(3)
        
        # Simulate a game where player 1 wins
        moves = [
            (0, 0, 1), (0, 1, 2), (1, 0, 1),
            (1, 1, 2), (2, 0, 1)  # Player 1 wins with column 0
        ]
        
        for row, col, player in moves:
            result = game.move(row, col, player)
            print(f"  move({row}, {col}, {player}): {result}")
            
            if result != 0:
                print(f"  Player {result} wins!")
                break

def test_tic_tac_toe_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Tic-Tac-Toe Edge Cases ===")
    
    # Test different board sizes
    sizes = [1, 2, 4, 5]
    
    for size in sizes:
        print(f"\nBoard size {size}x{size}:")
        game = TicTacToeOptimized(size)
        
        # Fill diagonal for player 1
        for i in range(size):
            result = game.move(i, i, 1)
            if result != 0:
                print(f"  Player 1 wins with diagonal after {i+1} moves")
                break
    
    # Test draw scenario (3x3)
    print(f"\nTesting draw scenario:")
    game = TicTacToeCompact(3)
    
    # Moves that lead to a draw
    draw_moves = [
        (0, 0, 1), (0, 1, 2), (0, 2, 1),
        (1, 0, 2), (1, 1, 1), (1, 2, 2),
        (2, 0, 2), (2, 1, 1), (2, 2, 2)
    ]
    
    moves_made = 0
    for row, col, player in draw_moves:
        result = game.move(row, col, player)
        moves_made += 1
        
        if result != 0:
            print(f"  Unexpected win after {moves_made} moves")
            break
    else:
        print(f"  Game ended in draw after {moves_made} moves")

def test_win_conditions():
    """Test all possible win conditions"""
    print("\n=== Testing All Win Conditions ===")
    
    game = TicTacToeOptimized(3)
    
    # Test row win
    print("Testing row win:")
    game = TicTacToeOptimized(3)
    for col in range(3):
        result = game.move(0, col, 1)
    print(f"  Row 0 win: {result == 1}")
    
    # Test column win
    print("Testing column win:")
    game = TicTacToeOptimized(3)
    for row in range(3):
        result = game.move(row, 0, 2)
    print(f"  Column 0 win: {result == 2}")
    
    # Test main diagonal win
    print("Testing main diagonal win:")
    game = TicTacToeOptimized(3)
    for i in range(3):
        result = game.move(i, i, 1)
    print(f"  Main diagonal win: {result == 1}")
    
    # Test anti-diagonal win
    print("Testing anti-diagonal win:")
    game = TicTacToeOptimized(3)
    for i in range(3):
        result = game.move(i, 2 - i, 2)
    print(f"  Anti-diagonal win: {result == 2}")

def test_performance_comparison():
    """Test performance of different implementations"""
    print("\n=== Testing Performance Comparison ===")
    
    import time
    
    implementations = [
        ("Basic Board", TicTacToeBasic),
        ("Optimized Counters", TicTacToeOptimized),
        ("Compact Array", TicTacToeCompact)
    ]
    
    board_sizes = [3, 10, 50]
    
    for size in board_sizes:
        print(f"\nBoard size {size}x{size}:")
        
        for name, TicTacToeClass in implementations:
            game = TicTacToeClass(size)
            
            # Time multiple moves
            start_time = time.time()
            
            # Make moves in a pattern that won't win immediately
            moves_made = 0
            for i in range(min(100, size * size // 2)):
                row = i % size
                col = (i * 2) % size
                player = (i % 2) + 1
                
                result = game.move(row, col, player)
                moves_made += 1
                
                if result != 0:
                    break
            
            elapsed = (time.time() - start_time) * 1000
            
            print(f"  {name}: {elapsed:.3f}ms for {moves_made} moves")

def test_history_functionality():
    """Test history and undo functionality"""
    print("\n=== Testing History Functionality ===")
    
    game = TicTacToeWithHistory(3)
    
    # Make some moves
    moves = [(0, 0, 1), (0, 1, 2), (1, 1, 1), (0, 2, 2)]
    
    print("Making moves:")
    for row, col, player in moves:
        result = game.move(row, col, player)
        count = game.get_move_count()
        last_move = game.get_last_move()
        print(f"  move({row}, {col}, {player}): result={result}, total_moves={count}")
        print(f"    Last move: {last_move}")
    
    # Test undo
    print(f"\nTesting undo:")
    while game.get_move_count() > 0:
        count_before = game.get_move_count()
        success = game.undo()
        count_after = game.get_move_count()
        
        print(f"  Undo: success={success}, moves: {count_before} -> {count_after}")

def test_multi_game_statistics():
    """Test multi-game statistics"""
    print("\n=== Testing Multi-Game Statistics ===")
    
    tournament = TicTacToeMultiGame(3)
    
    # Simulate multiple games
    games = [
        # Game 1: Player 1 wins
        [(0, 0, 1), (0, 1, 2), (1, 0, 1), (1, 1, 2), (2, 0, 1)],
        
        # Game 2: Player 2 wins  
        [(0, 0, 1), (0, 1, 2), (1, 0, 1), (1, 1, 2), (2, 0, 1), (2, 1, 2)],
        
        # Game 3: Draw
        [(0, 0, 1), (0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 1, 1), 
         (1, 2, 2), (2, 0, 2), (2, 1, 1), (2, 2, 2)]
    ]
    
    for game_num, game_moves in enumerate(games, 1):
        print(f"\nGame {game_num}:")
        tournament.reset_game()
        
        for row, col, player in game_moves:
            result = tournament.move(row, col, player)
            if result != 0:
                print(f"  Player {result} wins!")
                break
        else:
            print(f"  Game ended (possibly draw)")
    
    # Display statistics
    stats = tournament.get_statistics()
    print(f"\nTournament Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

def demonstrate_applications():
    """Demonstrate real-world applications"""
    print("\n=== Demonstrating Applications ===")
    
    # Application 1: AI game analysis
    print("Application 1: Game Analysis System")
    
    analyzer = TicTacToeWithHistory(3)
    
    # Simulate a game with analysis
    game_moves = [(1, 1, 1), (0, 0, 2), (0, 1, 1), (2, 2, 2), (2, 1, 1)]
    
    for i, (row, col, player) in enumerate(game_moves):
        result = analyzer.move(row, col, player)
        
        print(f"  Move {i+1}: Player {player} -> ({row}, {col})")
        
        if result != 0:
            print(f"    Game Over: Player {result} wins!")
            print(f"    Total moves: {analyzer.get_move_count()}")
            break
    
    # Application 2: Tournament management
    print(f"\nApplication 2: Tournament Management")
    
    tournament_mgr = TicTacToeMultiGame(3)
    
    # Simulate tournament rounds
    print("  Simulating tournament rounds...")
    
    for round_num in range(3):
        tournament_mgr.reset_game()
        
        # Quick game simulation
        winner = (round_num % 2) + 1  # Alternate winners
        
        # Make winning moves for demonstration
        if winner == 1:
            for i in range(3):
                tournament_mgr.move(i, i, 1)  # Diagonal win
        else:
            for i in range(3):
                tournament_mgr.move(0, i, 2)  # Row win
    
    final_stats = tournament_mgr.get_statistics()
    print(f"  Tournament completed:")
    print(f"    Games played: {final_stats['games_played']}")
    print(f"    Player 1 wins: {final_stats['player1_wins']}")
    print(f"    Player 2 wins: {final_stats['player2_wins']}")

def test_large_boards():
    """Test performance on large boards"""
    print("\n=== Testing Large Boards ===")
    
    import time
    
    large_sizes = [10, 20, 50]
    
    for size in large_sizes:
        print(f"\nTesting {size}x{size} board:")
        
        game = TicTacToeCompact(size)
        
        start_time = time.time()
        
        # Fill diagonal for quick win
        for i in range(size):
            result = game.move(i, i, 1)
            if result != 0:
                elapsed = (time.time() - start_time) * 1000
                print(f"  Player 1 wins after {i+1} moves in {elapsed:.2f}ms")
                break

def benchmark_move_operations():
    """Benchmark move operations"""
    print("\n=== Benchmarking Move Operations ===")
    
    import time
    
    implementations = [
        ("Basic Board", TicTacToeBasic),
        ("Optimized", TicTacToeOptimized),
        ("Compact", TicTacToeCompact)
    ]
    
    board_size = 100
    num_moves = 1000
    
    for name, TicTacToeClass in implementations:
        game = TicTacToeClass(board_size)
        
        start_time = time.time()
        
        # Make random moves
        import random
        for i in range(num_moves):
            row = random.randint(0, board_size - 1)
            col = random.randint(0, board_size - 1)
            player = (i % 2) + 1
            
            game.move(row, col, player)
        
        elapsed = (time.time() - start_time) * 1000
        avg_time = elapsed / num_moves
        
        print(f"  {name}: {elapsed:.2f}ms total, {avg_time:.4f}ms per move")

def test_memory_usage():
    """Test memory usage of different implementations"""
    print("\n=== Testing Memory Usage ===")
    
    implementations = [
        ("Basic Board", TicTacToeBasic),
        ("Optimized", TicTacToeOptimized),
        ("Compact", TicTacToeCompact)
    ]
    
    board_sizes = [3, 10, 100]
    
    for size in board_sizes:
        print(f"\nBoard size {size}x{size}:")
        
        for name, TicTacToeClass in implementations:
            game = TicTacToeClass(size)
            
            # Estimate memory usage (simplified)
            if hasattr(game, 'board'):
                memory_estimate = size * size  # 2D board
            elif hasattr(game, 'rows') and hasattr(game, 'cols'):
                if isinstance(game.rows[0], list):
                    memory_estimate = size * 4  # rows, cols for 2 players
                else:
                    memory_estimate = size * 2  # compact rows, cols
            else:
                memory_estimate = size
            
            print(f"  {name}: ~{memory_estimate} memory units")

if __name__ == "__main__":
    test_tic_tac_toe_basic()
    test_tic_tac_toe_edge_cases()
    test_win_conditions()
    test_performance_comparison()
    test_history_functionality()
    test_multi_game_statistics()
    demonstrate_applications()
    test_large_boards()
    benchmark_move_operations()
    test_memory_usage()

"""
Tic-Tac-Toe Design demonstrates key concepts:

Core Approaches:
1. Basic Board - Store full board, check all conditions O(n)
2. Optimized Counters - Track progress with counters O(1)
3. Compact Array - Single array with +/- counting O(1)
4. With History - Support undo and move tracking
5. Multi-Game - Tournament support with statistics

Key Design Principles:
- Space-time tradeoffs (board vs counters)
- Win condition detection optimization
- State management and history tracking
- Extensibility for tournament features

Performance Characteristics:
- Basic: O(n²) space, O(n) time per move
- Optimized: O(n) space, O(1) time per move
- Compact: O(n) space, O(1) time per move
- History: O(n + m) space where m is moves

Real-world Applications:
- Game development and AI
- Tournament management systems
- Educational programming examples
- Pattern recognition systems
- State machine implementations

The optimized counter approach is most commonly used
due to its O(1) move complexity and moderate space usage.
"""
