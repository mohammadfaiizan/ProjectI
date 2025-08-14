"""
Game Theory and Decision Making with Backtracking
=================================================

Topics: Minimax algorithm, game trees, optimal play, strategic decisions
Companies: Google, Amazon, Microsoft, Gaming companies, AI research labs
Difficulty: Hard
Time Complexity: O(b^d) where b=branching factor, d=depth
Space Complexity: O(d) for recursion stack
"""

from typing import List, Tuple, Optional, Dict, Any
import copy
import math

class GameTheoryProblems:
    
    def __init__(self):
        """Initialize with game state tracking and analysis"""
        self.nodes_evaluated = 0
        self.alpha_beta_pruning_count = 0
        self.game_tree_depth = 0
    
    # ==========================================
    # 1. TIC-TAC-TOE WITH MINIMAX
    # ==========================================
    
    def tic_tac_toe_minimax(self, board: List[List[str]], is_maximizing: bool = True) -> Tuple[int, Optional[Tuple[int, int]]]:
        """
        Solve Tic-Tac-Toe using Minimax algorithm
        
        X = maximizing player, O = minimizing player
        Returns (score, best_move)
        
        Company: Google, Microsoft (game AI)
        Difficulty: Medium
        Time: O(9!), Space: O(9)
        """
        self.nodes_evaluated += 1
        
        def check_winner(board: List[List[str]]) -> Optional[str]:
            """Check if there's a winner"""
            # Check rows
            for row in board:
                if row[0] == row[1] == row[2] != ' ':
                    return row[0]
            
            # Check columns
            for col in range(3):
                if board[0][col] == board[1][col] == board[2][col] != ' ':
                    return board[0][col]
            
            # Check diagonals
            if board[0][0] == board[1][1] == board[2][2] != ' ':
                return board[0][0]
            if board[0][2] == board[1][1] == board[2][0] != ' ':
                return board[0][2]
            
            return None
        
        def is_board_full(board: List[List[str]]) -> bool:
            """Check if board is full"""
            return all(cell != ' ' for row in board for cell in row)
        
        # BASE CASES: Terminal states
        winner = check_winner(board)
        if winner == 'X':
            return (1, None)  # X wins
        elif winner == 'O':
            return (-1, None)  # O wins
        elif is_board_full(board):
            return (0, None)  # Draw
        
        # Get available moves
        available_moves = [(i, j) for i in range(3) for j in range(3) if board[i][j] == ' ']
        
        if is_maximizing:
            # Maximizing player (X)
            max_score = -math.inf
            best_move = None
            
            for move in available_moves:
                row, col = move
                
                # Make move
                board[row][col] = 'X'
                
                # Recurse
                score, _ = self.tic_tac_toe_minimax(board, False)
                
                # Undo move
                board[row][col] = ' '
                
                if score > max_score:
                    max_score = score
                    best_move = move
            
            return (max_score, best_move)
        
        else:
            # Minimizing player (O)
            min_score = math.inf
            best_move = None
            
            for move in available_moves:
                row, col = move
                
                # Make move
                board[row][col] = 'O'
                
                # Recurse
                score, _ = self.tic_tac_toe_minimax(board, True)
                
                # Undo move
                board[row][col] = ' '
                
                if score < min_score:
                    min_score = score
                    best_move = move
            
            return (min_score, best_move)
    
    def play_tic_tac_toe(self) -> None:
        """Demonstrate Tic-Tac-Toe game with AI"""
        board = [[' ' for _ in range(3)] for _ in range(3)]
        
        def print_board(board: List[List[str]]) -> None:
            print("Current board:")
            for i, row in enumerate(board):
                print(f"  {i}: {' | '.join(row)}")
                if i < 2:
                    print("     ---------")
            print()
        
        print("=== TIC-TAC-TOE WITH MINIMAX AI ===")
        print("X = AI (maximizing), O = Human (minimizing)")
        print()
        
        # Let's simulate a game where AI (X) goes first
        current_player = 'X'  # AI starts
        
        for turn in range(9):
            print_board(board)
            
            if current_player == 'X':
                # AI's turn
                print("AI (X) is thinking...")
                self.nodes_evaluated = 0
                score, best_move = self.tic_tac_toe_minimax(board, True)
                
                if best_move:
                    row, col = best_move
                    board[row][col] = 'X'
                    print(f"AI chooses position ({row}, {col})")
                    print(f"Nodes evaluated: {self.nodes_evaluated}")
                    print(f"Expected outcome: {score}")
                
            else:
                # For demo, we'll have AI play both sides
                print("AI playing as O (minimizing)...")
                self.nodes_evaluated = 0
                score, best_move = self.tic_tac_toe_minimax(board, False)
                
                if best_move:
                    row, col = best_move
                    board[row][col] = 'O'
                    print(f"O chooses position ({row}, {col})")
                    print(f"Nodes evaluated: {self.nodes_evaluated}")
                    print(f"Expected outcome: {score}")
            
            # Check for game end
            winner = None
            for row in board:
                if row[0] == row[1] == row[2] != ' ':
                    winner = row[0]
            for col in range(3):
                if board[0][col] == board[1][col] == board[2][col] != ' ':
                    winner = board[0][col]
            if board[0][0] == board[1][1] == board[2][2] != ' ':
                winner = board[0][0]
            if board[0][2] == board[1][1] == board[2][0] != ' ':
                winner = board[0][2]
            
            if winner:
                print_board(board)
                print(f"Game Over! Winner: {winner}")
                return
            
            if all(cell != ' ' for row in board for cell in row):
                print_board(board)
                print("Game Over! It's a draw!")
                return
            
            # Switch players
            current_player = 'O' if current_player == 'X' else 'X'
            print()
    
    # ==========================================
    # 2. MINIMAX WITH ALPHA-BETA PRUNING
    # ==========================================
    
    def minimax_alpha_beta(self, board: List[List[str]], depth: int, alpha: float, beta: float, 
                          is_maximizing: bool) -> Tuple[int, Optional[Tuple[int, int]]]:
        """
        Minimax with Alpha-Beta pruning for better performance
        
        Alpha: Best value maximizer can guarantee
        Beta: Best value minimizer can guarantee
        
        Company: All AI companies
        Difficulty: Hard
        Time: O(b^(d/2)) best case with good ordering
        Space: O(d)
        """
        self.nodes_evaluated += 1
        
        def evaluate_board(board: List[List[str]]) -> int:
            """Evaluate board position"""
            # Check for winner
            for row in board:
                if row[0] == row[1] == row[2] != ' ':
                    return 10 if row[0] == 'X' else -10
            
            for col in range(3):
                if board[0][col] == board[1][col] == board[2][col] != ' ':
                    return 10 if board[0][col] == 'X' else -10
            
            if board[0][0] == board[1][1] == board[2][2] != ' ':
                return 10 if board[0][0] == 'X' else -10
            if board[0][2] == board[1][1] == board[2][0] != ' ':
                return 10 if board[0][2] == 'X' else -10
            
            return 0  # No winner yet
        
        # Terminal test
        score = evaluate_board(board)
        if score != 0 or depth == 0:
            return (score, None)
        
        # Check if board is full
        if all(cell != ' ' for row in board for cell in row):
            return (0, None)
        
        available_moves = [(i, j) for i in range(3) for j in range(3) if board[i][j] == ' ']
        best_move = None
        
        if is_maximizing:
            max_eval = -math.inf
            
            for move in available_moves:
                row, col = move
                board[row][col] = 'X'
                
                eval_score, _ = self.minimax_alpha_beta(board, depth - 1, alpha, beta, False)
                
                board[row][col] = ' '  # Undo move
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                
                alpha = max(alpha, eval_score)
                
                # Alpha-Beta pruning
                if beta <= alpha:
                    self.alpha_beta_pruning_count += 1
                    print(f"    α-β pruning at depth {depth}: α={alpha}, β={beta}")
                    break
            
            return (max_eval, best_move)
        
        else:
            min_eval = math.inf
            
            for move in available_moves:
                row, col = move
                board[row][col] = 'O'
                
                eval_score, _ = self.minimax_alpha_beta(board, depth - 1, alpha, beta, True)
                
                board[row][col] = ' '  # Undo move
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                
                beta = min(beta, eval_score)
                
                # Alpha-Beta pruning
                if beta <= alpha:
                    self.alpha_beta_pruning_count += 1
                    print(f"    α-β pruning at depth {depth}: α={alpha}, β={beta}")
                    break
            
            return (min_eval, best_move)
    
    def compare_minimax_algorithms(self) -> None:
        """Compare regular minimax vs alpha-beta pruning"""
        print("=== MINIMAX vs ALPHA-BETA PRUNING COMPARISON ===")
        
        # Test board with some moves already made
        test_board = [
            ['X', ' ', 'O'],
            [' ', 'X', ' '],
            ['O', ' ', ' ']
        ]
        
        print("Test board:")
        for row in test_board:
            print(f"  {' | '.join(row)}")
        print()
        
        # Test regular minimax
        print("1. Regular Minimax:")
        board_copy1 = [row[:] for row in test_board]
        self.nodes_evaluated = 0
        score1, move1 = self.tic_tac_toe_minimax(board_copy1, True)
        minimax_nodes = self.nodes_evaluated
        
        print(f"   Best move: {move1}")
        print(f"   Score: {score1}")
        print(f"   Nodes evaluated: {minimax_nodes}")
        
        # Test alpha-beta pruning
        print("\n2. Minimax with Alpha-Beta Pruning:")
        board_copy2 = [row[:] for row in test_board]
        self.nodes_evaluated = 0
        self.alpha_beta_pruning_count = 0
        score2, move2 = self.minimax_alpha_beta(board_copy2, 9, -math.inf, math.inf, True)
        ab_nodes = self.nodes_evaluated
        
        print(f"   Best move: {move2}")
        print(f"   Score: {score2}")
        print(f"   Nodes evaluated: {ab_nodes}")
        print(f"   Pruning operations: {self.alpha_beta_pruning_count}")
        
        if minimax_nodes > 0:
            efficiency = (minimax_nodes - ab_nodes) / minimax_nodes * 100
            print(f"   Efficiency improvement: {efficiency:.1f}%")
    
    # ==========================================
    # 3. CONNECT FOUR GAME
    # ==========================================
    
    def connect_four_minimax(self, board: List[List[str]], depth: int, 
                           is_maximizing: bool) -> Tuple[int, Optional[int]]:
        """
        Connect Four using Minimax algorithm
        
        7x6 board, need 4 in a row to win
        
        Company: Gaming companies, AI research
        Difficulty: Hard
        Time: O(7^depth), Space: O(depth)
        """
        def check_winner(board: List[List[str]]) -> Optional[str]:
            """Check for Connect Four winner"""
            rows, cols = len(board), len(board[0])
            
            # Check horizontal
            for r in range(rows):
                for c in range(cols - 3):
                    if (board[r][c] == board[r][c+1] == board[r][c+2] == board[r][c+3] != ' '):
                        return board[r][c]
            
            # Check vertical
            for r in range(rows - 3):
                for c in range(cols):
                    if (board[r][c] == board[r+1][c] == board[r+2][c] == board[r+3][c] != ' '):
                        return board[r][c]
            
            # Check diagonal (top-left to bottom-right)
            for r in range(rows - 3):
                for c in range(cols - 3):
                    if (board[r][c] == board[r+1][c+1] == board[r+2][c+2] == board[r+3][c+3] != ' '):
                        return board[r][c]
            
            # Check diagonal (top-right to bottom-left)
            for r in range(rows - 3):
                for c in range(3, cols):
                    if (board[r][c] == board[r+1][c-1] == board[r+2][c-2] == board[r+3][c-3] != ' '):
                        return board[r][c]
            
            return None
        
        def get_valid_columns(board: List[List[str]]) -> List[int]:
            """Get columns where pieces can be dropped"""
            return [col for col in range(len(board[0])) if board[0][col] == ' ']
        
        def drop_piece(board: List[List[str]], col: int, piece: str) -> Optional[int]:
            """Drop piece in column, return row or None if full"""
            for row in range(len(board) - 1, -1, -1):
                if board[row][col] == ' ':
                    board[row][col] = piece
                    return row
            return None
        
        def remove_piece(board: List[List[str]], row: int, col: int) -> None:
            """Remove piece from board"""
            board[row][col] = ' '
        
        # Terminal test
        winner = check_winner(board)
        if winner == 'R':  # Red (maximizing)
            return (100, None)
        elif winner == 'Y':  # Yellow (minimizing)
            return (-100, None)
        
        valid_cols = get_valid_columns(board)
        if not valid_cols or depth == 0:
            return (0, None)  # Draw or depth limit
        
        if is_maximizing:
            max_eval = -math.inf
            best_col = valid_cols[0]
            
            for col in valid_cols:
                row = drop_piece(board, col, 'R')
                if row is not None:
                    eval_score, _ = self.connect_four_minimax(board, depth - 1, False)
                    remove_piece(board, row, col)
                    
                    if eval_score > max_eval:
                        max_eval = eval_score
                        best_col = col
            
            return (max_eval, best_col)
        
        else:
            min_eval = math.inf
            best_col = valid_cols[0]
            
            for col in valid_cols:
                row = drop_piece(board, col, 'Y')
                if row is not None:
                    eval_score, _ = self.connect_four_minimax(board, depth - 1, True)
                    remove_piece(board, row, col)
                    
                    if eval_score < min_eval:
                        min_eval = eval_score
                        best_col = col
            
            return (min_eval, best_col)
    
    def demonstrate_connect_four(self) -> None:
        """Demonstrate Connect Four AI"""
        print("=== CONNECT FOUR WITH MINIMAX AI ===")
        
        # Initialize 6x7 board
        board = [[' ' for _ in range(7)] for _ in range(6)]
        
        def print_board(board: List[List[str]]) -> None:
            print("Connect Four Board:")
            print("  0 1 2 3 4 5 6")
            print("  ┌─┬─┬─┬─┬─┬─┬─┐")
            for i, row in enumerate(board):
                print(f"{i} │{│'.join(row)}│")
            print("  └─┴─┴─┴─┴─┴─┴─┘")
        
        # Simulate a few moves
        print("Simulating Connect Four game:")
        moves = [(3, 'R'), (3, 'Y'), (4, 'R'), (2, 'Y')]
        
        for col, piece in moves:
            # Drop piece
            for row in range(5, -1, -1):
                if board[row][col] == ' ':
                    board[row][col] = piece
                    break
            
            print_board(board)
            print()
        
        # AI makes next move
        print("AI (Red) calculating best move...")
        score, best_col = self.connect_four_minimax(board, 4, True)
        print(f"AI chooses column {best_col} (score: {score})")
    
    # ==========================================
    # 4. GAME TREE ANALYSIS
    # ==========================================
    
    def analyze_game_tree(self, game_type: str) -> None:
        """
        Analyze game tree characteristics for different games
        """
        print(f"=== GAME TREE ANALYSIS: {game_type.upper()} ===")
        
        if game_type == "tic_tac_toe":
            print("Tic-Tac-Toe Game Tree:")
            print("• Branching factor: Starts at 9, decreases each move")
            print("• Maximum depth: 9 (maximum moves)")
            print("• Total game states: ~3^9 = 19,683")
            print("• Perfect play result: Draw")
            print("• Search complexity: O(9!) without pruning")
            print("• Alpha-beta can reduce to O(√(9!))")
            
        elif game_type == "connect_four":
            print("Connect Four Game Tree:")
            print("• Branching factor: Up to 7 (columns)")
            print("• Maximum depth: 42 (full board)")
            print("• Total game states: ~7^42 (astronomical)")
            print("• First player advantage with perfect play")
            print("• Practical search depth: 8-12 moves")
            print("• Alpha-beta essential for reasonable performance")
            
        elif game_type == "chess":
            print("Chess Game Tree (for reference):")
            print("• Average branching factor: ~35")
            print("• Average game length: ~40 moves per player")
            print("• Total game states: ~10^120")
            print("• Practical search depth: 15-20 moves")
            print("• Requires advanced evaluation functions")
            print("• Uses iterative deepening and transposition tables")
    
    # ==========================================
    # 5. STRATEGIC DECISION PROBLEMS
    # ==========================================
    
    def nim_game_optimal(self, piles: List[int], is_maximizing: bool = True) -> Tuple[bool, Optional[Tuple[int, int]]]:
        """
        Solve Nim game using game theory
        
        Players alternate removing objects from piles
        Last player to move wins
        
        Company: Google (algorithmic thinking)
        Difficulty: Medium
        Time: O(n * max(piles)), Space: O(depth)
        """
        def calculate_nim_sum(piles: List[int]) -> int:
            """Calculate XOR of all pile sizes"""
            nim_sum = 0
            for pile in piles:
                nim_sum ^= pile
            return nim_sum
        
        def get_valid_moves(piles: List[int]) -> List[Tuple[int, int]]:
            """Get all valid moves (pile_index, stones_to_remove)"""
            moves = []
            for i, pile_size in enumerate(piles):
                for stones in range(1, pile_size + 1):
                    moves.append((i, stones))
            return moves
        
        def is_game_over(piles: List[int]) -> bool:
            """Check if game is over (all piles empty)"""
            return all(pile == 0 for pile in piles)
        
        # BASE CASE: Game over
        if is_game_over(piles):
            return (not is_maximizing, None)  # Previous player won
        
        # For Nim, optimal strategy uses XOR (nim-sum)
        nim_sum = calculate_nim_sum(piles)
        
        if is_maximizing:
            # If nim-sum is 0, all moves are losing moves
            if nim_sum == 0:
                # Pick any valid move (all lead to loss with optimal play)
                moves = get_valid_moves(piles)
                return (False, moves[0] if moves else None)
            
            # Find winning move (make nim-sum = 0)
            for pile_idx, pile_size in enumerate(piles):
                target = pile_size ^ nim_sum
                if target < pile_size:
                    stones_to_remove = pile_size - target
                    return (True, (pile_idx, stones_to_remove))
            
            # Fallback (shouldn't reach here with correct nim-sum calculation)
            moves = get_valid_moves(piles)
            return (False, moves[0] if moves else None)
        
        else:
            # Minimizing player - same logic but return opposite
            if nim_sum == 0:
                moves = get_valid_moves(piles)
                return (True, moves[0] if moves else None)  # Opponent will lose
            
            for pile_idx, pile_size in enumerate(piles):
                target = pile_size ^ nim_sum
                if target < pile_size:
                    stones_to_remove = pile_size - target
                    return (False, (pile_idx, stones_to_remove))
            
            moves = get_valid_moves(piles)
            return (True, moves[0] if moves else None)
    
    def demonstrate_nim_game(self) -> None:
        """Demonstrate optimal Nim game play"""
        print("=== NIM GAME WITH OPTIMAL STRATEGY ===")
        
        piles = [3, 5, 7]  # Starting configuration
        current_player = "Player 1"
        
        print(f"Starting piles: {piles}")
        print("Players alternate removing stones from any pile")
        print("Last player to move wins")
        print()
        
        move_count = 0
        while not all(pile == 0 for pile in piles):
            print(f"Move {move_count + 1}: {current_player}'s turn")
            print(f"Current piles: {piles}")
            
            # Calculate optimal move
            is_max = (current_player == "Player 1")
            can_win, best_move = self.nim_game_optimal(piles, is_max)
            
            if best_move:
                pile_idx, stones = best_move
                piles[pile_idx] -= stones
                
                print(f"Optimal move: Remove {stones} stones from pile {pile_idx}")
                print(f"Can win with optimal play: {can_win}")
                print(f"Resulting piles: {piles}")
            
            # Switch players
            current_player = "Player 2" if current_player == "Player 1" else "Player 1"
            move_count += 1
            print()
            
            if move_count > 20:  # Safety break
                break
        
        winner = "Player 2" if current_player == "Player 1" else "Player 1"
        print(f"Game Over! Winner: {winner}")
    
    def predict_winner_game(self, nums: List[int]) -> bool:
        """
        Predict the Winner: Two players pick from array ends
        
        Players alternately pick numbers from either end
        Player 1 wins if their total >= Player 2's total
        
        Company: Google, Amazon
        Difficulty: Medium
        Time: O(n^2), Space: O(n^2)
        """
        memo = {}
        
        def minimax(left: int, right: int, is_player1: bool) -> int:
            """Return score difference (Player1 - Player2)"""
            if left > right:
                return 0
            
            if (left, right, is_player1) in memo:
                return memo[(left, right, is_player1)]
            
            if is_player1:
                # Player 1 wants to maximize score difference
                pick_left = nums[left] + minimax(left + 1, right, False)
                pick_right = nums[right] + minimax(left, right - 1, False)
                result = max(pick_left, pick_right)
            else:
                # Player 2 wants to minimize score difference
                pick_left = -nums[left] + minimax(left + 1, right, True)
                pick_right = -nums[right] + minimax(left, right - 1, True)
                result = min(pick_left, pick_right)
            
            memo[(left, right, is_player1)] = result
            return result
        
        score_diff = minimax(0, len(nums) - 1, True)
        print(f"Array: {nums}")
        print(f"Score difference (Player1 - Player2): {score_diff}")
        print(f"Player 1 can win: {score_diff >= 0}")
        
        return score_diff >= 0

# ==========================================
# DEMONSTRATION AND TESTING
# ==========================================

def demonstrate_game_theory_problems():
    """Demonstrate all game theory and decision making problems"""
    print("=== GAME THEORY PROBLEMS DEMONSTRATION ===\n")
    
    gt = GameTheoryProblems()
    
    # 1. Tic-Tac-Toe with Minimax
    print("=== TIC-TAC-TOE MINIMAX ===")
    gt.play_tic_tac_toe()
    print("\n" + "="*60 + "\n")
    
    # 2. Minimax vs Alpha-Beta comparison
    gt.compare_minimax_algorithms()
    print("\n" + "="*60 + "\n")
    
    # 3. Connect Four demonstration
    gt.demonstrate_connect_four()
    print("\n" + "="*60 + "\n")
    
    # 4. Game tree analysis
    for game in ["tic_tac_toe", "connect_four", "chess"]:
        gt.analyze_game_tree(game)
        print()
    print("="*60 + "\n")
    
    # 5. Nim game optimal strategy
    gt.demonstrate_nim_game()
    print("\n" + "="*60 + "\n")
    
    # 6. Predict winner problem
    print("=== PREDICT WINNER PROBLEM ===")
    test_arrays = [
        [1, 5, 2],
        [1, 5, 233, 7],
        [1, 3, 1]
    ]
    
    for arr in test_arrays:
        can_win = gt.predict_winner_game(arr)
        print()

if __name__ == "__main__":
    demonstrate_game_theory_problems()
    
    print("=== GAME THEORY MASTERY GUIDE ===")
    
    print("\n🎯 CORE CONCEPTS:")
    print("• Minimax: Find optimal play assuming opponent plays optimally")
    print("• Alpha-Beta Pruning: Optimize minimax by eliminating branches")
    print("• Game Tree: Represent all possible game states and moves")
    print("• Evaluation Function: Assess non-terminal positions")
    print("• Zero-Sum Games: One player's gain equals other's loss")
    
    print("\n📋 ALGORITHM COMPONENTS:")
    print("1. State Representation: How to encode game state")
    print("2. Move Generation: Find all legal moves from current state")
    print("3. Terminal Test: Check if game is over")
    print("4. Evaluation Function: Score non-terminal positions")
    print("5. Search Strategy: Minimax, alpha-beta, iterative deepening")
    
    print("\n⚡ OPTIMIZATION TECHNIQUES:")
    print("• Alpha-Beta Pruning: Eliminate branches that won't affect result")
    print("• Move Ordering: Try best moves first for better pruning")
    print("• Transposition Tables: Cache previously computed positions")
    print("• Iterative Deepening: Gradually increase search depth")
    print("• Quiescence Search: Extend search in tactical positions")
    
    print("\n🎮 GAME TYPES:")
    print("• Perfect Information: All game state visible (Chess, Checkers)")
    print("• Imperfect Information: Hidden information (Poker, Battleship)")
    print("• Deterministic: No randomness (Tic-Tac-Toe, Connect Four)")
    print("• Stochastic: Random elements (Backgammon, card games)")
    
    print("\n📊 COMPLEXITY ANALYSIS:")
    print("• Branching Factor (b): Average number of legal moves")
    print("• Search Depth (d): How far ahead to look")
    print("• Time Complexity: O(b^d) without pruning")
    print("• Alpha-Beta: O(b^(d/2)) best case with perfect ordering")
    print("• Space Complexity: O(bd) for search stack")
    
    print("\n🏆 REAL-WORLD APPLICATIONS:")
    print("• Game AI: Chess engines, game bots, puzzle solvers")
    print("• Decision Making: Business strategy, resource allocation")
    print("• Economics: Auction theory, market analysis")
    print("• Security: Adversarial scenarios, threat modeling")
    print("• Robotics: Multi-agent coordination, competitive scenarios")
