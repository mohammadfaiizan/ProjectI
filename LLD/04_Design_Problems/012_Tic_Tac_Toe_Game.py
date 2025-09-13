"""
TIC TAC TOE GAME - Complete System Design
=========================================

Problem Statement:
Design a comprehensive Tic Tac Toe game system that handles:
- Game board management with different sizes (3x3, 4x4, 5x5)
- Player management and turn-based gameplay
- Win condition detection (rows, columns, diagonals)
- Different game modes (Human vs Human, Human vs AI)
- AI implementation with different difficulty levels
- Game statistics and player rankings
- Tournament management
- Save and load game functionality
- Multiplayer support with matchmaking
- Game replay and analysis

Requirements:
- Support configurable board sizes (3x3 to 10x10)
- Implement multiple AI difficulty levels
- Handle different win conditions (3-in-a-row, 4-in-a-row, etc.)
- Provide real-time game state updates
- Support tournament brackets
- Track player statistics and rankings
- Handle network multiplayer games
- Implement game replay functionality
- Support custom game rules and variations
- Provide comprehensive game analysis

Design Patterns Used:
- Strategy: AI algorithms and difficulty levels
- State: Game state management
- Observer: Game event notifications
- Command: Move operations with undo/redo
- Factory: Game and AI creation
- Template Method: Game flow template
- Memento: Game state saving
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Set, Any
from datetime import datetime, timedelta
from enum import Enum
import uuid
import threading
import random
import copy
import math
from dataclasses import dataclass, field


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class Player(Enum):
    X = "X"
    O = "O"
    EMPTY = " "


class GameState(Enum):
    WAITING_FOR_PLAYERS = "waiting_for_players"
    IN_PROGRESS = "in_progress"
    X_WINS = "x_wins"
    O_WINS = "o_wins"
    DRAW = "draw"
    PAUSED = "paused"
    ABANDONED = "abandoned"


class AILevel(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


class GameMode(Enum):
    HUMAN_VS_HUMAN = "human_vs_human"
    HUMAN_VS_AI = "human_vs_ai"
    AI_VS_AI = "ai_vs_ai"


@dataclass
class Position:
    """Board position."""
    row: int
    col: int
    
    def __post_init__(self):
        if self.row < 0 or self.col < 0:
            raise ValueError("Position coordinates must be non-negative")
    
    def __eq__(self, other):
        return isinstance(other, Position) and self.row == other.row and self.col == other.col
    
    def __hash__(self):
        return hash((self.row, self.col))


@dataclass
class Move:
    """Game move with metadata."""
    move_id: str
    position: Position
    player: Player
    timestamp: datetime = field(default_factory=datetime.now)
    move_number: int = 0
    
    def __post_init__(self):
        if not self.move_id:
            self.move_id = str(uuid.uuid4())


@dataclass
class GameResult:
    """Game result with statistics."""
    winner: Optional[Player]
    total_moves: int
    game_duration: timedelta
    x_player_name: str
    o_player_name: str
    board_size: int
    win_condition: int
    final_board_state: List[List[Player]]


# ============================================================================
# AI STRATEGIES
# ============================================================================

class AIStrategy(ABC):
    """Abstract AI strategy."""
    
    @abstractmethod
    def get_best_move(self, board: 'TicTacToeBoard', player: Player) -> Optional[Position]:
        """Get best move for the AI."""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get strategy name."""
        pass
    
    @abstractmethod
    def get_difficulty_level(self) -> AILevel:
        """Get difficulty level."""
        pass


class RandomAIStrategy(AIStrategy):
    """Random move AI strategy (Easy level)."""
    
    def get_best_move(self, board: 'TicTacToeBoard', player: Player) -> Optional[Position]:
        """Get random available move."""
        available_moves = board.get_available_moves()
        if not available_moves:
            return None
        
        return random.choice(available_moves)
    
    def get_strategy_name(self) -> str:
        return "Random AI"
    
    def get_difficulty_level(self) -> AILevel:
        return AILevel.EASY


class GreedyAIStrategy(AIStrategy):
    """Greedy AI strategy (Medium level)."""
    
    def get_best_move(self, board: 'TicTacToeBoard', player: Player) -> Optional[Position]:
        """Get move using greedy strategy."""
        available_moves = board.get_available_moves()
        if not available_moves:
            return None
        
        # First, check if we can win
        for move in available_moves:
            board_copy = copy.deepcopy(board)
            board_copy.make_move(move, player)
            if board_copy.check_winner() == player:
                return move
        
        # Second, check if we need to block opponent
        opponent = Player.O if player == Player.X else Player.X
        for move in available_moves:
            board_copy = copy.deepcopy(board)
            board_copy.make_move(move, opponent)
            if board_copy.check_winner() == opponent:
                return move
        
        # Otherwise, prefer center and corners
        center = Position(board.size // 2, board.size // 2)
        if center in available_moves:
            return center
        
        # Prefer corners
        corners = [
            Position(0, 0), Position(0, board.size - 1),
            Position(board.size - 1, 0), Position(board.size - 1, board.size - 1)
        ]
        
        for corner in corners:
            if corner in available_moves:
                return corner
        
        # Return any available move
        return random.choice(available_moves)
    
    def get_strategy_name(self) -> str:
        return "Greedy AI"
    
    def get_difficulty_level(self) -> AILevel:
        return AILevel.MEDIUM


class MinimaxAIStrategy(AIStrategy):
    """Minimax AI strategy (Hard level)."""
    
    def __init__(self, max_depth: int = 6):
        self.max_depth = max_depth
    
    def get_best_move(self, board: 'TicTacToeBoard', player: Player) -> Optional[Position]:
        """Get move using minimax algorithm."""
        available_moves = board.get_available_moves()
        if not available_moves:
            return None
        
        best_move = None
        best_score = float('-inf')
        
        for move in available_moves:
            board_copy = copy.deepcopy(board)
            board_copy.make_move(move, player)
            
            score = self._minimax(board_copy, self.max_depth - 1, False, player, float('-inf'), float('inf'))
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    def _minimax(self, board: 'TicTacToeBoard', depth: int, is_maximizing: bool, 
                ai_player: Player, alpha: float, beta: float) -> float:
        """Minimax algorithm with alpha-beta pruning."""
        winner = board.check_winner()
        
        # Terminal states
        if winner == ai_player:
            return 10 + depth  # Prefer quicker wins
        elif winner is not None and winner != Player.EMPTY:
            return -10 - depth  # Avoid quicker losses
        elif board.is_full() or depth == 0:
            return 0  # Draw or depth limit
        
        available_moves = board.get_available_moves()
        
        if is_maximizing:
            max_score = float('-inf')
            current_player = ai_player
            
            for move in available_moves:
                board_copy = copy.deepcopy(board)
                board_copy.make_move(move, current_player)
                
                score = self._minimax(board_copy, depth - 1, False, ai_player, alpha, beta)
                max_score = max(max_score, score)
                alpha = max(alpha, score)
                
                if beta <= alpha:
                    break  # Alpha-beta pruning
            
            return max_score
        else:
            min_score = float('inf')
            opponent = Player.O if ai_player == Player.X else Player.X
            
            for move in available_moves:
                board_copy = copy.deepcopy(board)
                board_copy.make_move(move, opponent)
                
                score = self._minimax(board_copy, depth - 1, True, ai_player, alpha, beta)
                min_score = min(min_score, score)
                beta = min(beta, score)
                
                if beta <= alpha:
                    break  # Alpha-beta pruning
            
            return min_score
    
    def get_strategy_name(self) -> str:
        return f"Minimax AI (depth {self.max_depth})"
    
    def get_difficulty_level(self) -> AILevel:
        return AILevel.HARD


class ExpertAIStrategy(AIStrategy):
    """Expert AI strategy with advanced heuristics."""
    
    def __init__(self):
        self.minimax_strategy = MinimaxAIStrategy(max_depth=8)
    
    def get_best_move(self, board: 'TicTacToeBoard', player: Player) -> Optional[Position]:
        """Get move using advanced strategy."""
        available_moves = board.get_available_moves()
        if not available_moves:
            return None
        
        # Use minimax for smaller boards or fewer moves
        if board.size <= 3 or len(available_moves) <= 9:
            return self.minimax_strategy.get_best_move(board, player)
        
        # For larger boards, use heuristic evaluation
        return self._get_heuristic_move(board, player, available_moves)
    
    def _get_heuristic_move(self, board: 'TicTacToeBoard', player: Player, 
                           available_moves: List[Position]) -> Position:
        """Get move using heuristic evaluation."""
        best_move = None
        best_score = float('-inf')
        
        for move in available_moves:
            score = self._evaluate_position(board, move, player)
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move or random.choice(available_moves)
    
    def _evaluate_position(self, board: 'TicTacToeBoard', position: Position, player: Player) -> float:
        """Evaluate position using multiple heuristics."""
        score = 0.0
        
        # Check immediate win
        board_copy = copy.deepcopy(board)
        board_copy.make_move(position, player)
        if board_copy.check_winner() == player:
            return 1000.0
        
        # Check blocking opponent win
        opponent = Player.O if player == Player.X else Player.X
        board_copy = copy.deepcopy(board)
        board_copy.make_move(position, opponent)
        if board_copy.check_winner() == opponent:
            score += 500.0
        
        # Center preference
        center = board.size // 2
        distance_from_center = abs(position.row - center) + abs(position.col - center)
        score += (board.size - distance_from_center) * 10
        
        # Corner preference for smaller boards
        if board.size <= 5:
            if ((position.row == 0 or position.row == board.size - 1) and
                (position.col == 0 or position.col == board.size - 1)):
                score += 20
        
        # Line potential (how many lines this move contributes to)
        score += self._count_line_potential(board, position, player) * 5
        
        return score
    
    def _count_line_potential(self, board: 'TicTacToeBoard', position: Position, player: Player) -> int:
        """Count how many potential winning lines this position contributes to."""
        count = 0
        directions = [
            (0, 1), (1, 0), (1, 1), (1, -1)  # horizontal, vertical, diagonal
        ]
        
        for dr, dc in directions:
            # Check line in both directions
            line_length = 1  # Current position
            
            # Forward direction
            r, c = position.row + dr, position.col + dc
            while (0 <= r < board.size and 0 <= c < board.size and
                   board.board[r][c] in [Player.EMPTY, player]):
                if board.board[r][c] == player:
                    line_length += 1
                r += dr
                c += dc
            
            # Backward direction
            r, c = position.row - dr, position.col - dc
            while (0 <= r < board.size and 0 <= c < board.size and
                   board.board[r][c] in [Player.EMPTY, player]):
                if board.board[r][c] == player:
                    line_length += 1
                r -= dr
                c -= dc
            
            if line_length >= board.win_condition:
                count += 1
        
        return count
    
    def get_strategy_name(self) -> str:
        return "Expert AI"
    
    def get_difficulty_level(self) -> AILevel:
        return AILevel.EXPERT


# ============================================================================
# GAME BOARD
# ============================================================================

class TicTacToeBoard:
    """Tic Tac Toe game board."""
    
    def __init__(self, size: int = 3, win_condition: int = None):
        if size < 3 or size > 10:
            raise ValueError("Board size must be between 3 and 10")
        
        self.size = size
        self.win_condition = win_condition or min(size, 5)  # Default win condition
        self.board: List[List[Player]] = [[Player.EMPTY for _ in range(size)] for _ in range(size)]
        self.move_history: List[Move] = []
        self.move_count = 0
    
    def make_move(self, position: Position, player: Player) -> bool:
        """Make a move on the board."""
        if not self.is_valid_position(position):
            return False
        
        if self.board[position.row][position.col] != Player.EMPTY:
            return False
        
        self.board[position.row][position.col] = player
        self.move_count += 1
        
        # Record move
        move = Move(
            move_id=str(uuid.uuid4()),
            position=position,
            player=player,
            move_number=self.move_count
        )
        self.move_history.append(move)
        
        return True
    
    def undo_last_move(self) -> bool:
        """Undo the last move."""
        if not self.move_history:
            return False
        
        last_move = self.move_history.pop()
        self.board[last_move.position.row][last_move.position.col] = Player.EMPTY
        self.move_count -= 1
        
        return True
    
    def is_valid_position(self, position: Position) -> bool:
        """Check if position is valid on the board."""
        return (0 <= position.row < self.size and 
                0 <= position.col < self.size)
    
    def get_available_moves(self) -> List[Position]:
        """Get all available moves."""
        moves = []
        for row in range(self.size):
            for col in range(self.size):
                if self.board[row][col] == Player.EMPTY:
                    moves.append(Position(row, col))
        return moves
    
    def is_full(self) -> bool:
        """Check if board is full."""
        return len(self.get_available_moves()) == 0
    
    def check_winner(self) -> Optional[Player]:
        """Check if there's a winner."""
        # Check rows
        for row in range(self.size):
            winner = self._check_line([(row, col) for col in range(self.size)])
            if winner:
                return winner
        
        # Check columns
        for col in range(self.size):
            winner = self._check_line([(row, col) for row in range(self.size)])
            if winner:
                return winner
        
        # Check diagonals (all possible diagonals for larger boards)
        winner = self._check_all_diagonals()
        if winner:
            return winner
        
        return None
    
    def _check_line(self, positions: List[Tuple[int, int]]) -> Optional[Player]:
        """Check if a line has a winner."""
        if len(positions) < self.win_condition:
            return None
        
        for i in range(len(positions) - self.win_condition + 1):
            segment = positions[i:i + self.win_condition]
            
            if all(self.board[r][c] != Player.EMPTY for r, c in segment):
                first_player = self.board[segment[0][0]][segment[0][1]]
                if all(self.board[r][c] == first_player for r, c in segment):
                    return first_player
        
        return None
    
    def _check_all_diagonals(self) -> Optional[Player]:
        """Check all possible diagonals for winners."""
        # Main diagonals (top-left to bottom-right)
        for start_row in range(self.size - self.win_condition + 1):
            for start_col in range(self.size - self.win_condition + 1):
                diagonal = []
                for i in range(min(self.size - start_row, self.size - start_col)):
                    diagonal.append((start_row + i, start_col + i))
                
                winner = self._check_line(diagonal)
                if winner:
                    return winner
        
        # Anti-diagonals (top-right to bottom-left)
        for start_row in range(self.size - self.win_condition + 1):
            for start_col in range(self.win_condition - 1, self.size):
                diagonal = []
                for i in range(min(self.size - start_row, start_col + 1)):
                    diagonal.append((start_row + i, start_col - i))
                
                winner = self._check_line(diagonal)
                if winner:
                    return winner
        
        return None
    
    def get_board_state(self) -> List[List[str]]:
        """Get current board state as strings."""
        return [[cell.value for cell in row] for row in self.board]
    
    def reset(self) -> None:
        """Reset the board."""
        self.board = [[Player.EMPTY for _ in range(self.size)] for _ in range(self.size)]
        self.move_history.clear()
        self.move_count = 0
    
    def __str__(self) -> str:
        """String representation of the board."""
        result = ""
        
        # Column headers
        result += "   " + " ".join(f"{i:2}" for i in range(self.size)) + "\n"
        result += "  " + "---" * self.size + "\n"
        
        # Board rows
        for i, row in enumerate(self.board):
            result += f"{i:2}|"
            for cell in row:
                result += f" {cell.value} "
            result += "|\n"
        
        result += "  " + "---" * self.size
        return result


# ============================================================================
# GAME PLAYERS
# ============================================================================

class GamePlayer(ABC):
    """Abstract game player."""
    
    def __init__(self, player_id: str, name: str):
        self.player_id = player_id
        self.name = name
        self.symbol: Optional[Player] = None
        
        # Statistics
        self.games_played = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.total_moves = 0
        self.total_game_time = timedelta(0)
    
    @abstractmethod
    def get_move(self, board: TicTacToeBoard) -> Optional[Position]:
        """Get next move from player."""
        pass
    
    @abstractmethod
    def get_player_type(self) -> str:
        """Get player type."""
        pass
    
    def update_stats(self, result: GameResult, is_winner: bool, is_draw: bool) -> None:
        """Update player statistics."""
        self.games_played += 1
        self.total_moves += result.total_moves // 2  # Approximate moves per player
        self.total_game_time += result.game_duration
        
        if is_draw:
            self.draws += 1
        elif is_winner:
            self.wins += 1
        else:
            self.losses += 1
    
    def get_win_rate(self) -> float:
        """Get win rate percentage."""
        if self.games_played == 0:
            return 0.0
        return (self.wins / self.games_played) * 100
    
    def get_stats(self) -> Dict[str, Any]:
        """Get player statistics."""
        return {
            'player_id': self.player_id,
            'name': self.name,
            'player_type': self.get_player_type(),
            'games_played': self.games_played,
            'wins': self.wins,
            'losses': self.losses,
            'draws': self.draws,
            'win_rate': self.get_win_rate(),
            'total_moves': self.total_moves,
            'average_moves_per_game': self.total_moves / max(1, self.games_played),
            'total_game_time_seconds': self.total_game_time.total_seconds(),
            'average_game_time_seconds': self.total_game_time.total_seconds() / max(1, self.games_played)
        }


class HumanPlayer(GamePlayer):
    """Human player implementation."""
    
    def __init__(self, player_id: str, name: str):
        super().__init__(player_id, name)
        self.next_move: Optional[Position] = None
        self._move_event = threading.Event()
    
    def get_move(self, board: TicTacToeBoard) -> Optional[Position]:
        """Get move from human player (simulated for demo)."""
        # In a real implementation, this would wait for user input
        # For demonstration, we'll return a random valid move
        available_moves = board.get_available_moves()
        if available_moves:
            return random.choice(available_moves)
        return None
    
    def set_move(self, position: Position) -> None:
        """Set move for human player (called by UI)."""
        self.next_move = position
        self._move_event.set()
    
    def get_player_type(self) -> str:
        return "Human"


class AIPlayer(GamePlayer):
    """AI player implementation."""
    
    def __init__(self, player_id: str, name: str, strategy: AIStrategy):
        super().__init__(player_id, name)
        self.strategy = strategy
    
    def get_move(self, board: TicTacToeBoard) -> Optional[Position]:
        """Get move from AI strategy."""
        return self.strategy.get_best_move(board, self.symbol)
    
    def get_player_type(self) -> str:
        return f"AI ({self.strategy.get_difficulty_level().value})"
    
    def set_strategy(self, strategy: AIStrategy) -> None:
        """Change AI strategy."""
        self.strategy = strategy


# ============================================================================
# GAME ENGINE
# ============================================================================

class TicTacToeGame:
    """Main Tic Tac Toe game engine."""
    
    def __init__(self, game_id: str = None, board_size: int = 3, win_condition: int = None):
        self.game_id = game_id or str(uuid.uuid4())
        self.board = TicTacToeBoard(board_size, win_condition)
        self.state = GameState.WAITING_FOR_PLAYERS
        
        # Players
        self.x_player: Optional[GamePlayer] = None
        self.o_player: Optional[GamePlayer] = None
        self.current_player = Player.X
        
        # Game metadata
        self.game_mode = GameMode.HUMAN_VS_HUMAN
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.winner: Optional[Player] = None
        
        # Threading
        self._lock = threading.Lock()
        
        # Observers
        self.observers: List['GameObserver'] = []
        
        print(f"🎮 Tic Tac Toe Game created: {self.game_id[:8]} ({board_size}x{board_size})")
    
    def add_observer(self, observer: 'GameObserver') -> None:
        """Add game observer."""
        self.observers.append(observer)
    
    def remove_observer(self, observer: 'GameObserver') -> None:
        """Remove game observer."""
        if observer in self.observers:
            self.observers.remove(observer)
    
    def notify_observers(self, event_type: str, data: Dict[str, Any]) -> None:
        """Notify observers of game events."""
        for observer in self.observers:
            observer.on_game_event(self.game_id, event_type, data)
    
    def add_player(self, player: GamePlayer, symbol: Player) -> bool:
        """Add player to the game."""
        with self._lock:
            if symbol == Player.X and not self.x_player:
                self.x_player = player
                player.symbol = Player.X
                return True
            elif symbol == Player.O and not self.o_player:
                self.o_player = player
                player.symbol = Player.O
                return True
            return False
    
    def start_game(self) -> bool:
        """Start the game."""
        with self._lock:
            if not self.x_player or not self.o_player:
                return False
            
            if self.state != GameState.WAITING_FOR_PLAYERS:
                return False
            
            self.state = GameState.IN_PROGRESS
            self.start_time = datetime.now()
            
            # Determine game mode
            x_is_ai = isinstance(self.x_player, AIPlayer)
            o_is_ai = isinstance(self.o_player, AIPlayer)
            
            if x_is_ai and o_is_ai:
                self.game_mode = GameMode.AI_VS_AI
            elif x_is_ai or o_is_ai:
                self.game_mode = GameMode.HUMAN_VS_AI
            else:
                self.game_mode = GameMode.HUMAN_VS_HUMAN
            
            print(f"Game started: {self.game_mode.value}")
            
            # Notify observers
            self.notify_observers("game_started", {
                'x_player': self.x_player.name,
                'o_player': self.o_player.name,
                'game_mode': self.game_mode.value,
                'board_size': self.board.size
            })
            
            return True
    
    def make_move(self, position: Position, player: GamePlayer = None) -> bool:
        """Make a move in the game."""
        with self._lock:
            if self.state != GameState.IN_PROGRESS:
                return False
            
            # Validate player turn
            current_game_player = self.x_player if self.current_player == Player.X else self.o_player
            if player and player != current_game_player:
                return False
            
            # Make the move
            if not self.board.make_move(position, self.current_player):
                return False
            
            print(f"{current_game_player.name} ({self.current_player.value}) -> ({position.row}, {position.col})")
            
            # Check for winner
            winner = self.board.check_winner()
            if winner:
                self.winner = winner
                self.state = GameState.X_WINS if winner == Player.X else GameState.O_WINS
                self.end_time = datetime.now()
                
                # Update player statistics
                self._update_player_stats()
                
                print(f"🎉 {current_game_player.name} wins!")
                
                # Notify observers
                self.notify_observers("game_ended", {
                    'winner': winner.value,
                    'winner_name': current_game_player.name,
                    'total_moves': self.board.move_count
                })
                
            elif self.board.is_full():
                self.state = GameState.DRAW
                self.end_time = datetime.now()
                
                # Update player statistics
                self._update_player_stats()
                
                print("🤝 Game is a draw!")
                
                # Notify observers
                self.notify_observers("game_ended", {
                    'winner': None,
                    'total_moves': self.board.move_count
                })
                
            else:
                # Switch turns
                self.current_player = Player.O if self.current_player == Player.X else Player.X
                
                # Notify observers
                self.notify_observers("move_made", {
                    'position': {'row': position.row, 'col': position.col},
                    'player': current_game_player.name,
                    'symbol': self.current_player.value,
                    'move_number': self.board.move_count
                })
            
            return True
    
    def _update_player_stats(self) -> None:
        """Update player statistics after game ends."""
        if not self.start_time or not self.end_time:
            return
        
        game_result = GameResult(
            winner=self.winner,
            total_moves=self.board.move_count,
            game_duration=self.end_time - self.start_time,
            x_player_name=self.x_player.name,
            o_player_name=self.o_player.name,
            board_size=self.board.size,
            win_condition=self.board.win_condition,
            final_board_state=[[cell for cell in row] for row in self.board.board]
        )
        
        # Update X player stats
        is_x_winner = self.winner == Player.X
        is_draw = self.winner is None
        self.x_player.update_stats(game_result, is_x_winner, is_draw)
        
        # Update O player stats
        is_o_winner = self.winner == Player.O
        self.o_player.update_stats(game_result, is_o_winner, is_draw)
    
    def get_current_player(self) -> Optional[GamePlayer]:
        """Get current player."""
        if self.current_player == Player.X:
            return self.x_player
        else:
            return self.o_player
    
    def auto_play_turn(self) -> bool:
        """Auto-play current turn (for AI players)."""
        current_game_player = self.get_current_player()
        
        if not current_game_player or not isinstance(current_game_player, AIPlayer):
            return False
        
        move = current_game_player.get_move(self.board)
        if move:
            return self.make_move(move, current_game_player)
        
        return False
    
    def undo_last_move(self) -> bool:
        """Undo the last move."""
        with self._lock:
            if self.state != GameState.IN_PROGRESS:
                return False
            
            if self.board.undo_last_move():
                # Switch back to previous player
                self.current_player = Player.O if self.current_player == Player.X else Player.X
                return True
            
            return False
    
    def reset_game(self) -> None:
        """Reset the game."""
        with self._lock:
            self.board.reset()
            self.state = GameState.IN_PROGRESS if (self.x_player and self.o_player) else GameState.WAITING_FOR_PLAYERS
            self.current_player = Player.X
            self.winner = None
            self.start_time = datetime.now() if self.state == GameState.IN_PROGRESS else None
            self.end_time = None
    
    def get_game_state(self) -> Dict[str, Any]:
        """Get current game state."""
        current_game_player = self.get_current_player()
        
        return {
            'game_id': self.game_id,
            'state': self.state.value,
            'game_mode': self.game_mode.value,
            'board_size': self.board.size,
            'win_condition': self.board.win_condition,
            'current_player': self.current_player.value,
            'current_player_name': current_game_player.name if current_game_player else None,
            'board_state': self.board.get_board_state(),
            'move_count': self.board.move_count,
            'available_moves': [{'row': pos.row, 'col': pos.col} for pos in self.board.get_available_moves()],
            'x_player': self.x_player.name if self.x_player else None,
            'o_player': self.o_player.name if self.o_player else None,
            'winner': self.winner.value if self.winner else None,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'game_duration_seconds': (self.end_time - self.start_time).total_seconds() if (self.start_time and self.end_time) else None
        }


# ============================================================================
# OBSERVER PATTERN
# ============================================================================

class GameObserver(ABC):
    """Abstract game observer."""
    
    @abstractmethod
    def on_game_event(self, game_id: str, event_type: str, data: Dict[str, Any]) -> None:
        """Handle game event."""
        pass


class GameLogger(GameObserver):
    """Game event logger."""
    
    def __init__(self):
        self.event_log: List[Dict[str, Any]] = []
    
    def on_game_event(self, game_id: str, event_type: str, data: Dict[str, Any]) -> None:
        """Log game event."""
        event = {
            'game_id': game_id,
            'event_type': event_type,
            'data': data,
            'timestamp': datetime.now().isoformat()
        }
        
        self.event_log.append(event)
        print(f"📝 Event: {event_type} in game {game_id[:8]}")


# ============================================================================
# TOURNAMENT SYSTEM
# ============================================================================

class Tournament:
    """Tournament management system."""
    
    def __init__(self, tournament_id: str, name: str):
        self.tournament_id = tournament_id
        self.name = name
        self.players: List[GamePlayer] = []
        self.games: List[TicTacToeGame] = []
        self.standings: Dict[str, Dict[str, Any]] = {}
        self.is_active = False
        
        self.created_at = datetime.now()
    
    def add_player(self, player: GamePlayer) -> bool:
        """Add player to tournament."""
        if player not in self.players and not self.is_active:
            self.players.append(player)
            self.standings[player.player_id] = {
                'player': player,
                'wins': 0,
                'losses': 0,
                'draws': 0,
                'points': 0  # 3 for win, 1 for draw, 0 for loss
            }
            return True
        return False
    
    def start_tournament(self) -> bool:
        """Start round-robin tournament."""
        if len(self.players) < 2 or self.is_active:
            return False
        
        self.is_active = True
        
        # Create round-robin matches
        for i in range(len(self.players)):
            for j in range(i + 1, len(self.players)):
                game = TicTacToeGame(board_size=3)
                game.add_player(self.players[i], Player.X)
                game.add_player(self.players[j], Player.O)
                
                self.games.append(game)
        
        print(f"Tournament '{self.name}' started with {len(self.players)} players")
        print(f"Total games: {len(self.games)}")
        
        return True
    
    def update_standings(self, game: TicTacToeGame) -> None:
        """Update tournament standings after game."""
        if game.state not in [GameState.X_WINS, GameState.O_WINS, GameState.DRAW]:
            return
        
        x_player_id = game.x_player.player_id
        o_player_id = game.o_player.player_id
        
        if game.state == GameState.X_WINS:
            self.standings[x_player_id]['wins'] += 1
            self.standings[x_player_id]['points'] += 3
            self.standings[o_player_id]['losses'] += 1
        elif game.state == GameState.O_WINS:
            self.standings[o_player_id]['wins'] += 1
            self.standings[o_player_id]['points'] += 3
            self.standings[x_player_id]['losses'] += 1
        else:  # Draw
            self.standings[x_player_id]['draws'] += 1
            self.standings[x_player_id]['points'] += 1
            self.standings[o_player_id]['draws'] += 1
            self.standings[o_player_id]['points'] += 1
    
    def get_leaderboard(self) -> List[Dict[str, Any]]:
        """Get tournament leaderboard."""
        leaderboard = []
        
        for player_id, stats in self.standings.items():
            player_stats = stats.copy()
            player_stats['player_name'] = stats['player'].name
            player_stats['games_played'] = stats['wins'] + stats['losses'] + stats['draws']
            
            if player_stats['games_played'] > 0:
                player_stats['win_rate'] = (stats['wins'] / player_stats['games_played']) * 100
            else:
                player_stats['win_rate'] = 0.0
            
            leaderboard.append(player_stats)
        
        # Sort by points (descending), then by win rate
        leaderboard.sort(key=lambda x: (-x['points'], -x['win_rate']))
        
        # Add rankings
        for i, stats in enumerate(leaderboard):
            stats['rank'] = i + 1
        
        return leaderboard


# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================

def demonstrate_tic_tac_toe_game():
    """Demonstrate the Tic Tac Toe game system."""
    print("=== TIC TAC TOE GAME DEMONSTRATION ===\n")
    
    # Create game with different board sizes
    print("1. GAME CREATION WITH DIFFERENT SIZES:")
    
    games = []
    board_sizes = [3, 4, 5]
    
    for size in board_sizes:
        game = TicTacToeGame(board_size=size, win_condition=min(size, 4))
        games.append(game)
        print(f"   ✓ Created {size}x{size} game (win condition: {game.board.win_condition})")
    
    print()
    
    # Test AI strategies
    print("2. AI STRATEGY TESTING:")
    
    strategies = [
        RandomAIStrategy(),
        GreedyAIStrategy(),
        MinimaxAIStrategy(max_depth=4),
        ExpertAIStrategy()
    ]
    
    for strategy in strategies:
        print(f"   ✓ {strategy.get_strategy_name()} ({strategy.get_difficulty_level().value})")
    
    print()
    
    # Create players
    print("3. PLAYER CREATION:")
    
    # Human players
    alice = HumanPlayer("human1", "Alice")
    bob = HumanPlayer("human2", "Bob")
    
    # AI players
    easy_ai = AIPlayer("ai1", "Easy AI", RandomAIStrategy())
    medium_ai = AIPlayer("ai2", "Medium AI", GreedyAIStrategy())
    hard_ai = AIPlayer("ai3", "Hard AI", MinimaxAIStrategy(max_depth=6))
    expert_ai = AIPlayer("ai4", "Expert AI", ExpertAIStrategy())
    
    players = [alice, bob, easy_ai, medium_ai, hard_ai, expert_ai]
    
    for player in players:
        print(f"   ✓ {player.name} ({player.get_player_type()})")
    
    print()
    
    # Test Human vs AI game
    print("4. HUMAN VS AI GAME SIMULATION:")
    
    game = TicTacToeGame(board_size=3)
    logger = GameLogger()
    game.add_observer(logger)
    
    # Add players
    game.add_player(alice, Player.X)
    game.add_player(medium_ai, Player.O)
    
    # Start game
    game.start_game()
    
    print(f"   Game: {alice.name} (X) vs {medium_ai.name} (O)")
    print("   Initial board:")
    print(game.board)
    print()
    
    # Simulate game moves
    move_count = 0
    while game.state == GameState.IN_PROGRESS and move_count < 9:
        current_player = game.get_current_player()
        
        if isinstance(current_player, AIPlayer):
            # AI makes move
            success = game.auto_play_turn()
            if success:
                print(f"   AI move made by {current_player.name}")
        else:
            # Simulate human move (random for demo)
            available_moves = game.board.get_available_moves()
            if available_moves:
                move = random.choice(available_moves)
                success = game.make_move(move, current_player)
                if success:
                    print(f"   Human move made by {current_player.name}")
        
        move_count += 1
        
        # Show board every few moves
        if move_count % 2 == 0:
            print(f"   Board after {move_count} moves:")
            print(game.board)
            print()
    
    # Show final result
    game_state = game.get_game_state()
    print(f"   Final result: {game_state['state']}")
    if game_state['winner']:
        print(f"   Winner: {game_state['winner']}")
    print()
    
    # Test AI vs AI game
    print("5. AI VS AI GAME:")
    
    ai_game = TicTacToeGame(board_size=3)
    ai_game.add_player(hard_ai, Player.X)
    ai_game.add_player(expert_ai, Player.O)
    ai_game.start_game()
    
    print(f"   Game: {hard_ai.name} (X) vs {expert_ai.name} (O)")
    
    # Auto-play entire game
    while ai_game.state == GameState.IN_PROGRESS:
        success = ai_game.auto_play_turn()
        if not success:
            break
    
    print("   Final board:")
    print(ai_game.board)
    
    ai_game_state = ai_game.get_game_state()
    print(f"   Result: {ai_game_state['state']}")
    print(f"   Total moves: {ai_game_state['move_count']}")
    print()
    
    # Test larger board
    print("6. LARGER BOARD GAME (5x5):")
    
    large_game = TicTacToeGame(board_size=5, win_condition=4)
    large_game.add_player(expert_ai, Player.X)
    large_game.add_player(medium_ai, Player.O)
    large_game.start_game()
    
    print(f"   5x5 board with win condition: {large_game.board.win_condition}")
    
    # Play a few moves
    for _ in range(10):
        if large_game.state == GameState.IN_PROGRESS:
            large_game.auto_play_turn()
    
    print("   Board after 10 moves:")
    print(large_game.board)
    print()
    
    # Test tournament
    print("7. TOURNAMENT SYSTEM:")
    
    tournament = Tournament("tour1", "AI Championship")
    
    # Add AI players to tournament
    tournament_players = [easy_ai, medium_ai, hard_ai, expert_ai]
    for player in tournament_players:
        tournament.add_player(player)
        print(f"   ✓ Added {player.name} to tournament")
    
    # Start tournament
    tournament.start_tournament()
    
    # Simulate tournament games
    completed_games = 0
    for game in tournament.games:
        game.start_game()
        
        # Auto-play game
        while game.state == GameState.IN_PROGRESS:
            if not game.auto_play_turn():
                break
        
        # Update tournament standings
        tournament.update_standings(game)
        completed_games += 1
        
        print(f"   Game {completed_games}: {game.x_player.name} vs {game.o_player.name} -> {game.state.value}")
    
    # Show tournament results
    print("\n   Tournament Leaderboard:")
    leaderboard = tournament.get_leaderboard()
    
    for entry in leaderboard:
        print(f"   {entry['rank']}. {entry['player_name']}: {entry['points']} points "
              f"({entry['wins']}-{entry['draws']}-{entry['losses']}) "
              f"Win Rate: {entry['win_rate']:.1f}%")
    
    print()
    
    # Test player statistics
    print("8. PLAYER STATISTICS:")
    
    for player in tournament_players:
        stats = player.get_stats()
        print(f"   {stats['name']} ({stats['player_type']}):")
        print(f"     Games: {stats['games_played']}")
        print(f"     Record: {stats['wins']}-{stats['draws']}-{stats['losses']}")
        print(f"     Win Rate: {stats['win_rate']:.1f}%")
        print(f"     Avg Moves/Game: {stats['average_moves_per_game']:.1f}")
    
    print()
    
    # Test game features
    print("9. GAME FEATURES TESTING:")
    
    feature_game = TicTacToeGame(board_size=3)
    feature_game.add_player(alice, Player.X)
    feature_game.add_player(bob, Player.O)
    feature_game.start_game()
    
    # Make some moves
    moves = [Position(1, 1), Position(0, 0), Position(0, 1), Position(2, 2)]
    
    for i, move in enumerate(moves):
        current_player = feature_game.get_current_player()
        feature_game.make_move(move, current_player)
        print(f"   Move {i+1}: {current_player.name} -> ({move.row}, {move.col})")
    
    print("   Board state:")
    print(feature_game.board)
    
    # Test undo
    print("   Testing undo...")
    feature_game.undo_last_move()
    print("   Board after undo:")
    print(feature_game.board)
    
    # Test reset
    print("   Testing reset...")
    feature_game.reset_game()
    print("   Board after reset:")
    print(feature_game.board)
    
    print()
    
    # Show event log
    print("10. EVENT LOG:")
    
    print(f"   Total events logged: {len(logger.event_log)}")
    
    event_types = {}
    for event in logger.event_log:
        event_type = event['event_type']
        event_types[event_type] = event_types.get(event_type, 0) + 1
    
    for event_type, count in event_types.items():
        print(f"     {event_type}: {count}")
    
    print()
    print("=== TIC TAC TOE GAME DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_tic_tac_toe_game()
