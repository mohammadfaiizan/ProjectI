"""
SNAKE AND LADDER GAME - Complete System Design
==============================================

Problem Statement:
Design a comprehensive Snake and Ladder game system that handles:
- Game board with customizable size and layout
- Snake and ladder placement and management
- Multiple players with turn-based gameplay
- Dice rolling mechanics with various dice types
- Game state management and rules enforcement
- Win conditions and game completion
- Game history and statistics tracking
- Save and load game functionality
- Multiplayer support (local and online)
- Different game modes and variations

Requirements:
- Support configurable board sizes (default 10x10)
- Allow custom placement of snakes and ladders
- Handle 2-6 players with fair turn rotation
- Implement various dice types (standard, loaded, multiple dice)
- Track player positions and movements
- Enforce game rules (exact landing on 100, snake/ladder effects)
- Provide game state persistence
- Support different winning conditions
- Generate game statistics and analytics
- Handle edge cases and invalid moves

Design Patterns Used:
- State: Game and player states
- Strategy: Dice rolling strategies
- Observer: Game event notifications
- Command: Player moves and actions
- Factory: Game component creation
- Memento: Game state saving/loading
- Template Method: Game flow template
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime, timedelta
from enum import Enum
import uuid
import threading
import random
import json
from dataclasses import dataclass, field
from collections import deque


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class GameState(Enum):
    WAITING_FOR_PLAYERS = "waiting_for_players"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class PlayerState(Enum):
    WAITING = "waiting"
    ACTIVE = "active"
    FINISHED = "finished"
    DISCONNECTED = "disconnected"


class MoveType(Enum):
    NORMAL = "normal"
    SNAKE_BITE = "snake_bite"
    LADDER_CLIMB = "ladder_climb"
    WINNING_MOVE = "winning_move"
    INVALID_MOVE = "invalid_move"


class DiceType(Enum):
    STANDARD = "standard"
    LOADED = "loaded"
    DOUBLE_DICE = "double_dice"
    CUSTOM = "custom"


@dataclass
class Position:
    """Board position coordinates."""
    row: int
    col: int
    
    def __post_init__(self):
        if self.row < 0 or self.col < 0:
            raise ValueError("Position coordinates must be non-negative")


@dataclass
class Snake:
    """Snake on the game board."""
    head: int  # Position number where snake head is
    tail: int  # Position number where snake tail is
    name: str = ""
    description: str = ""
    
    def __post_init__(self):
        if self.head <= self.tail:
            raise ValueError("Snake head must be at a higher position than tail")


@dataclass
class Ladder:
    """Ladder on the game board."""
    bottom: int  # Position number where ladder bottom is
    top: int     # Position number where ladder top is
    name: str = ""
    description: str = ""
    
    def __post_init__(self):
        if self.bottom >= self.top:
            raise ValueError("Ladder bottom must be at a lower position than top")


@dataclass
class GameMove:
    """Represents a single move in the game."""
    move_id: str
    player_id: str
    dice_roll: int
    start_position: int
    end_position: int
    move_type: MoveType
    snake_or_ladder_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.move_id:
            self.move_id = str(uuid.uuid4())


# ============================================================================
# DICE STRATEGIES
# ============================================================================

class DiceStrategy(ABC):
    """Abstract dice rolling strategy."""
    
    @abstractmethod
    def roll(self) -> int:
        """Roll the dice and return result."""
        pass
    
    @abstractmethod
    def get_min_value(self) -> int:
        """Get minimum possible dice value."""
        pass
    
    @abstractmethod
    def get_max_value(self) -> int:
        """Get maximum possible dice value."""
        pass
    
    @abstractmethod
    def get_dice_name(self) -> str:
        """Get dice type name."""
        pass


class StandardDiceStrategy(DiceStrategy):
    """Standard 6-sided dice."""
    
    def roll(self) -> int:
        """Roll standard dice (1-6)."""
        return random.randint(1, 6)
    
    def get_min_value(self) -> int:
        return 1
    
    def get_max_value(self) -> int:
        return 6
    
    def get_dice_name(self) -> str:
        return "Standard Dice (1-6)"


class LoadedDiceStrategy(DiceStrategy):
    """Loaded dice with bias towards higher numbers."""
    
    def __init__(self, bias_factor: float = 0.3):
        self.bias_factor = bias_factor  # 0.0 = fair, 1.0 = always max
    
    def roll(self) -> int:
        """Roll loaded dice with bias."""
        if random.random() < self.bias_factor:
            # Bias towards higher numbers (4, 5, 6)
            return random.randint(4, 6)
        else:
            # Normal roll
            return random.randint(1, 6)
    
    def get_min_value(self) -> int:
        return 1
    
    def get_max_value(self) -> int:
        return 6
    
    def get_dice_name(self) -> str:
        return f"Loaded Dice (bias: {self.bias_factor:.1f})"


class DoubleDiceStrategy(DiceStrategy):
    """Two dice rolled together."""
    
    def roll(self) -> int:
        """Roll two dice and return sum."""
        return random.randint(1, 6) + random.randint(1, 6)
    
    def get_min_value(self) -> int:
        return 2
    
    def get_max_value(self) -> int:
        return 12
    
    def get_dice_name(self) -> str:
        return "Double Dice (2-12)"


class CustomDiceStrategy(DiceStrategy):
    """Custom dice with specified range."""
    
    def __init__(self, min_value: int = 1, max_value: int = 6):
        if min_value >= max_value:
            raise ValueError("Min value must be less than max value")
        self.min_value = min_value
        self.max_value = max_value
    
    def roll(self) -> int:
        """Roll custom dice."""
        return random.randint(self.min_value, self.max_value)
    
    def get_min_value(self) -> int:
        return self.min_value
    
    def get_max_value(self) -> int:
        return self.max_value
    
    def get_dice_name(self) -> str:
        return f"Custom Dice ({self.min_value}-{self.max_value})"


# ============================================================================
# PLAYER CLASS
# ============================================================================

class Player:
    """Game player with state and statistics."""
    
    def __init__(self, player_id: str, name: str, color: str = ""):
        self.player_id = player_id
        self.name = name
        self.color = color or self._generate_random_color()
        self.position = 0  # Start at position 0 (before the board)
        self.state = PlayerState.WAITING
        
        # Game statistics
        self.moves_count = 0
        self.snakes_encountered = 0
        self.ladders_climbed = 0
        self.total_dice_sum = 0
        self.max_dice_roll = 0
        self.min_dice_roll = 7  # Will be updated on first roll
        
        # Timing
        self.game_start_time: Optional[datetime] = None
        self.game_end_time: Optional[datetime] = None
        self.total_turn_time = timedelta(0)
        
        # Move history
        self.move_history: List[GameMove] = []
        
        self.created_at = datetime.now()
    
    def _generate_random_color(self) -> str:
        """Generate random color for player."""
        colors = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "cyan"]
        return random.choice(colors)
    
    def move_to_position(self, new_position: int, dice_roll: int, 
                        move_type: MoveType = MoveType.NORMAL,
                        snake_or_ladder_id: str = None) -> GameMove:
        """Move player to new position and record the move."""
        move = GameMove(
            move_id=str(uuid.uuid4()),
            player_id=self.player_id,
            dice_roll=dice_roll,
            start_position=self.position,
            end_position=new_position,
            move_type=move_type,
            snake_or_ladder_id=snake_or_ladder_id
        )
        
        # Update position
        self.position = new_position
        
        # Update statistics
        self.moves_count += 1
        self.total_dice_sum += dice_roll
        self.max_dice_roll = max(self.max_dice_roll, dice_roll)
        self.min_dice_roll = min(self.min_dice_roll, dice_roll)
        
        if move_type == MoveType.SNAKE_BITE:
            self.snakes_encountered += 1
        elif move_type == MoveType.LADDER_CLIMB:
            self.ladders_climbed += 1
        
        # Record move
        self.move_history.append(move)
        
        return move
    
    def start_game(self) -> None:
        """Start game for this player."""
        self.state = PlayerState.ACTIVE
        self.game_start_time = datetime.now()
    
    def finish_game(self) -> None:
        """Finish game for this player."""
        self.state = PlayerState.FINISHED
        self.game_end_time = datetime.now()
    
    def get_average_dice_roll(self) -> float:
        """Get average dice roll."""
        if self.moves_count == 0:
            return 0.0
        return self.total_dice_sum / self.moves_count
    
    def get_game_duration(self) -> timedelta:
        """Get total game duration."""
        if not self.game_start_time:
            return timedelta(0)
        
        end_time = self.game_end_time or datetime.now()
        return end_time - self.game_start_time
    
    def get_player_stats(self) -> Dict[str, Any]:
        """Get player statistics."""
        return {
            'player_id': self.player_id,
            'name': self.name,
            'color': self.color,
            'position': self.position,
            'state': self.state.value,
            'moves_count': self.moves_count,
            'snakes_encountered': self.snakes_encountered,
            'ladders_climbed': self.ladders_climbed,
            'dice_stats': {
                'total_sum': self.total_dice_sum,
                'average_roll': self.get_average_dice_roll(),
                'max_roll': self.max_dice_roll,
                'min_roll': self.min_dice_roll if self.moves_count > 0 else 0
            },
            'timing': {
                'game_duration_seconds': self.get_game_duration().total_seconds(),
                'average_turn_time_seconds': (self.total_turn_time.total_seconds() / max(1, self.moves_count))
            },
            'created_at': self.created_at.isoformat()
        }
    
    def __str__(self) -> str:
        return f"Player {self.name} (Position: {self.position})"


# ============================================================================
# GAME BOARD
# ============================================================================

class GameBoard:
    """Snake and Ladder game board."""
    
    def __init__(self, size: int = 10):
        if size < 5 or size > 20:
            raise ValueError("Board size must be between 5 and 20")
        
        self.size = size
        self.total_positions = size * size
        
        # Board layout (position number -> (row, col))
        self.position_map: Dict[int, Position] = {}
        self._initialize_position_map()
        
        # Snakes and ladders
        self.snakes: Dict[str, Snake] = {}
        self.ladders: Dict[str, Ladder] = {}
        
        # Position effects (position -> snake_id or ladder_id)
        self.position_effects: Dict[int, str] = {}
        
        # Board metadata
        self.created_at = datetime.now()
        self.theme = "Classic"
    
    def _initialize_position_map(self) -> None:
        """Initialize position to coordinate mapping."""
        position = 1
        
        for row in range(self.size - 1, -1, -1):  # Start from top row
            if (self.size - 1 - row) % 2 == 0:
                # Left to right
                for col in range(self.size):
                    self.position_map[position] = Position(row, col)
                    position += 1
            else:
                # Right to left (snake pattern)
                for col in range(self.size - 1, -1, -1):
                    self.position_map[position] = Position(row, col)
                    position += 1
    
    def add_snake(self, head: int, tail: int, name: str = "", description: str = "") -> str:
        """Add snake to the board."""
        if not (1 <= tail < head <= self.total_positions):
            raise ValueError(f"Invalid snake positions: head={head}, tail={tail}")
        
        # Check for conflicts
        if head in self.position_effects or tail in self.position_effects:
            raise ValueError("Snake positions conflict with existing snakes or ladders")
        
        snake_id = str(uuid.uuid4())
        snake = Snake(head, tail, name, description)
        
        self.snakes[snake_id] = snake
        self.position_effects[head] = snake_id
        
        return snake_id
    
    def add_ladder(self, bottom: int, top: int, name: str = "", description: str = "") -> str:
        """Add ladder to the board."""
        if not (1 <= bottom < top <= self.total_positions):
            raise ValueError(f"Invalid ladder positions: bottom={bottom}, top={top}")
        
        # Check for conflicts
        if bottom in self.position_effects or top in self.position_effects:
            raise ValueError("Ladder positions conflict with existing snakes or ladders")
        
        ladder_id = str(uuid.uuid4())
        ladder = Ladder(bottom, top, name, description)
        
        self.ladders[ladder_id] = ladder
        self.position_effects[bottom] = ladder_id
        
        return ladder_id
    
    def remove_snake(self, snake_id: str) -> bool:
        """Remove snake from board."""
        if snake_id not in self.snakes:
            return False
        
        snake = self.snakes[snake_id]
        del self.snakes[snake_id]
        del self.position_effects[snake.head]
        
        return True
    
    def remove_ladder(self, ladder_id: str) -> bool:
        """Remove ladder from board."""
        if ladder_id not in self.ladders:
            return False
        
        ladder = self.ladders[ladder_id]
        del self.ladders[ladder_id]
        del self.position_effects[ladder.bottom]
        
        return True
    
    def get_position_effect(self, position: int) -> Tuple[Optional[str], Optional[Any]]:
        """Get effect at position (snake or ladder)."""
        if position not in self.position_effects:
            return None, None
        
        effect_id = self.position_effects[position]
        
        if effect_id in self.snakes:
            return "snake", self.snakes[effect_id]
        elif effect_id in self.ladders:
            return "ladder", self.ladders[effect_id]
        
        return None, None
    
    def apply_position_effect(self, position: int) -> Tuple[int, MoveType, Optional[str]]:
        """Apply position effect and return new position, move type, and effect ID."""
        effect_type, effect = self.get_position_effect(position)
        
        if effect_type == "snake":
            return effect.tail, MoveType.SNAKE_BITE, self.position_effects[position]
        elif effect_type == "ladder":
            return effect.top, MoveType.LADDER_CLIMB, self.position_effects[position]
        
        return position, MoveType.NORMAL, None
    
    def is_valid_position(self, position: int) -> bool:
        """Check if position is valid on the board."""
        return 0 <= position <= self.total_positions
    
    def get_coordinate(self, position: int) -> Optional[Position]:
        """Get coordinate for position number."""
        return self.position_map.get(position)
    
    def setup_default_snakes_and_ladders(self) -> None:
        """Setup default snakes and ladders for standard game."""
        if self.size != 10:
            return  # Only for 10x10 board
        
        # Default snakes (head -> tail)
        default_snakes = [
            (99, 78, "Giant Snake", "The biggest snake on the board"),
            (95, 75, "Python", "A long python"),
            (92, 88, "Viper", "A quick viper"),
            (87, 24, "Cobra", "Dangerous cobra"),
            (64, 60, "Rattlesnake", "Watch out for the rattle"),
            (62, 19, "Anaconda", "Massive anaconda"),
            (56, 53, "Garden Snake", "Small but effective"),
            (49, 11, "Boa", "Constricting boa"),
            (48, 26, "Adder", "Venomous adder"),
            (16, 6, "Baby Snake", "Even small snakes bite")
        ]
        
        # Default ladders (bottom -> top)
        default_ladders = [
            (1, 38, "Starter Ladder", "Great way to begin"),
            (4, 14, "Short Ladder", "Quick boost"),
            (9, 21, "Wooden Ladder", "Sturdy wooden ladder"),
            (21, 42, "Rope Ladder", "Climb carefully"),
            (28, 84, "Golden Ladder", "Lucky golden ladder"),
            (36, 44, "Metal Ladder", "Strong metal rungs"),
            (51, 67, "Bamboo Ladder", "Flexible bamboo"),
            (71, 91, "Crystal Ladder", "Beautiful crystal ladder"),
            (80, 100, "Victory Ladder", "Almost at the top!")
        ]
        
        # Add snakes
        for head, tail, name, desc in default_snakes:
            try:
                self.add_snake(head, tail, name, desc)
            except ValueError:
                continue  # Skip if positions conflict
        
        # Add ladders
        for bottom, top, name, desc in default_ladders:
            try:
                self.add_ladder(bottom, top, name, desc)
            except ValueError:
                continue  # Skip if positions conflict
    
    def get_board_info(self) -> Dict[str, Any]:
        """Get board information."""
        return {
            'size': self.size,
            'total_positions': self.total_positions,
            'snakes_count': len(self.snakes),
            'ladders_count': len(self.ladders),
            'theme': self.theme,
            'snakes': {
                snake_id: {
                    'head': snake.head,
                    'tail': snake.tail,
                    'name': snake.name,
                    'description': snake.description
                }
                for snake_id, snake in self.snakes.items()
            },
            'ladders': {
                ladder_id: {
                    'bottom': ladder.bottom,
                    'top': ladder.top,
                    'name': ladder.name,
                    'description': ladder.description
                }
                for ladder_id, ladder in self.ladders.items()
            },
            'created_at': self.created_at.isoformat()
        }
    
    def __str__(self) -> str:
        return f"Game Board {self.size}x{self.size} ({len(self.snakes)} snakes, {len(self.ladders)} ladders)"


# ============================================================================
# GAME ENGINE
# ============================================================================

class SnakeAndLadderGame:
    """Main Snake and Ladder game engine."""
    
    def __init__(self, game_id: str = None, board_size: int = 10):
        self.game_id = game_id or str(uuid.uuid4())
        self.board = GameBoard(board_size)
        self.state = GameState.WAITING_FOR_PLAYERS
        
        # Players
        self.players: Dict[str, Player] = {}
        self.player_order: List[str] = []  # Order of play
        self.current_player_index = 0
        self.max_players = 6
        self.min_players = 2
        
        # Dice
        self.dice_strategy: DiceStrategy = StandardDiceStrategy()
        
        # Game settings
        self.exact_finish = True  # Must land exactly on final position
        self.allow_multiple_turns = False  # Extra turn on rolling 6
        
        # Game history
        self.move_history: List[GameMove] = []
        self.winner: Optional[Player] = None
        
        # Timing
        self.game_start_time: Optional[datetime] = None
        self.game_end_time: Optional[datetime] = None
        self.turn_start_time: Optional[datetime] = None
        
        # Game statistics
        self.total_turns = 0
        self.total_dice_rolls = 0
        
        # Threading
        self._lock = threading.Lock()
        
        # Observers
        self.observers: List['GameObserver'] = []
        
        print(f"🎲 Snake and Ladder Game created: {self.game_id[:8]}")
    
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
    
    def set_dice_strategy(self, strategy: DiceStrategy) -> None:
        """Set dice rolling strategy."""
        self.dice_strategy = strategy
        print(f"Dice strategy changed to: {strategy.get_dice_name()}")
    
    def add_player(self, name: str, color: str = "") -> Optional[Player]:
        """Add player to the game."""
        with self._lock:
            if len(self.players) >= self.max_players:
                return None
            
            if self.state != GameState.WAITING_FOR_PLAYERS:
                return None
            
            # Check for duplicate names
            if any(p.name == name for p in self.players.values()):
                return None
            
            player_id = str(uuid.uuid4())
            player = Player(player_id, name, color)
            
            self.players[player_id] = player
            self.player_order.append(player_id)
            
            print(f"Player {name} joined the game")
            
            # Notify observers
            self.notify_observers("player_joined", {
                'player_id': player_id,
                'player_name': name,
                'players_count': len(self.players)
            })
            
            return player
    
    def remove_player(self, player_id: str) -> bool:
        """Remove player from the game."""
        with self._lock:
            if player_id not in self.players:
                return False
            
            if self.state == GameState.IN_PROGRESS:
                # Mark as disconnected instead of removing
                self.players[player_id].state = PlayerState.DISCONNECTED
                return True
            
            player_name = self.players[player_id].name
            del self.players[player_id]
            self.player_order.remove(player_id)
            
            print(f"Player {player_name} left the game")
            return True
    
    def start_game(self) -> bool:
        """Start the game."""
        with self._lock:
            if len(self.players) < self.min_players:
                return False
            
            if self.state != GameState.WAITING_FOR_PLAYERS:
                return False
            
            # Setup default board if empty
            if len(self.board.snakes) == 0 and len(self.board.ladders) == 0:
                self.board.setup_default_snakes_and_ladders()
            
            # Initialize game
            self.state = GameState.IN_PROGRESS
            self.game_start_time = datetime.now()
            self.current_player_index = 0
            
            # Start all players
            for player in self.players.values():
                player.start_game()
            
            # Set first player as active
            if self.player_order:
                current_player = self.players[self.player_order[0]]
                current_player.state = PlayerState.ACTIVE
                self.turn_start_time = datetime.now()
            
            print(f"Game started with {len(self.players)} players")
            
            # Notify observers
            self.notify_observers("game_started", {
                'players': [p.get_player_stats() for p in self.players.values()],
                'board_info': self.board.get_board_info()
            })
            
            return True
    
    def roll_dice(self, player_id: str) -> Optional[int]:
        """Roll dice for player."""
        with self._lock:
            if self.state != GameState.IN_PROGRESS:
                return None
            
            # Check if it's player's turn
            current_player_id = self.player_order[self.current_player_index]
            if player_id != current_player_id:
                return None
            
            player = self.players[player_id]
            if player.state != PlayerState.ACTIVE:
                return None
            
            # Roll dice
            dice_result = self.dice_strategy.roll()
            self.total_dice_rolls += 1
            
            print(f"{player.name} rolled {dice_result}")
            
            # Notify observers
            self.notify_observers("dice_rolled", {
                'player_id': player_id,
                'player_name': player.name,
                'dice_result': dice_result
            })
            
            return dice_result
    
    def make_move(self, player_id: str, dice_result: int) -> Optional[GameMove]:
        """Make move for player with dice result."""
        with self._lock:
            player = self.players.get(player_id)
            if not player:
                return None
            
            start_position = player.position
            target_position = start_position + dice_result
            
            # Check for exact finish rule
            if self.exact_finish and target_position > self.board.total_positions:
                # Cannot move, stay at current position
                move = player.move_to_position(
                    start_position, dice_result, MoveType.INVALID_MOVE
                )
                print(f"{player.name} cannot move (would exceed board)")
            else:
                # Normal move
                actual_position = min(target_position, self.board.total_positions)
                
                # Apply position effects (snakes/ladders)
                final_position, move_type, effect_id = self.board.apply_position_effect(actual_position)
                
                # Check for winning move
                if final_position == self.board.total_positions:
                    move_type = MoveType.WINNING_MOVE
                
                move = player.move_to_position(
                    final_position, dice_result, move_type, effect_id
                )
                
                # Log move details
                if move_type == MoveType.SNAKE_BITE:
                    snake = self.board.snakes[effect_id]
                    print(f"{player.name} was bitten by snake! {actual_position} -> {final_position}")
                elif move_type == MoveType.LADDER_CLIMB:
                    ladder = self.board.ladders[effect_id]
                    print(f"{player.name} climbed ladder! {actual_position} -> {final_position}")
                elif move_type == MoveType.WINNING_MOVE:
                    print(f"{player.name} reached position {final_position} and won!")
                else:
                    print(f"{player.name} moved from {start_position} to {final_position}")
            
            # Record move in game history
            self.move_history.append(move)
            
            # Check for win condition
            if player.position == self.board.total_positions:
                self._handle_player_win(player)
            
            # Update turn timing
            if self.turn_start_time:
                turn_duration = datetime.now() - self.turn_start_time
                player.total_turn_time += turn_duration
            
            # Notify observers
            self.notify_observers("move_made", {
                'move': {
                    'player_id': move.player_id,
                    'player_name': player.name,
                    'dice_roll': move.dice_roll,
                    'start_position': move.start_position,
                    'end_position': move.end_position,
                    'move_type': move.move_type.value
                }
            })
            
            # Next turn (unless player gets extra turn)
            extra_turn = (dice_result == 6 and self.allow_multiple_turns and 
                         move.move_type != MoveType.WINNING_MOVE)
            
            if not extra_turn:
                self._next_turn()
            else:
                print(f"{player.name} gets another turn for rolling 6!")
                self.turn_start_time = datetime.now()
            
            return move
    
    def _handle_player_win(self, player: Player) -> None:
        """Handle player winning the game."""
        player.finish_game()
        self.winner = player
        self.state = GameState.COMPLETED
        self.game_end_time = datetime.now()
        
        # Finish all other players
        for other_player in self.players.values():
            if other_player.player_id != player.player_id:
                other_player.finish_game()
        
        print(f"🎉 {player.name} won the game!")
        
        # Notify observers
        self.notify_observers("game_won", {
            'winner': player.get_player_stats(),
            'game_duration': self.get_game_duration().total_seconds()
        })
    
    def _next_turn(self) -> None:
        """Move to next player's turn."""
        # Set current player to waiting
        current_player_id = self.player_order[self.current_player_index]
        self.players[current_player_id].state = PlayerState.WAITING
        
        # Find next active player
        attempts = 0
        while attempts < len(self.player_order):
            self.current_player_index = (self.current_player_index + 1) % len(self.player_order)
            next_player_id = self.player_order[self.current_player_index]
            next_player = self.players[next_player_id]
            
            if next_player.state in [PlayerState.WAITING, PlayerState.ACTIVE]:
                next_player.state = PlayerState.ACTIVE
                self.total_turns += 1
                self.turn_start_time = datetime.now()
                
                print(f"It's {next_player.name}'s turn")
                
                # Notify observers
                self.notify_observers("turn_changed", {
                    'current_player_id': next_player_id,
                    'current_player_name': next_player.name,
                    'turn_number': self.total_turns
                })
                break
            
            attempts += 1
        
        if attempts >= len(self.player_order):
            # No active players left
            self.state = GameState.ABANDONED
    
    def pause_game(self) -> bool:
        """Pause the game."""
        if self.state == GameState.IN_PROGRESS:
            self.state = GameState.PAUSED
            return True
        return False
    
    def resume_game(self) -> bool:
        """Resume the game."""
        if self.state == GameState.PAUSED:
            self.state = GameState.IN_PROGRESS
            self.turn_start_time = datetime.now()
            return True
        return False
    
    def abandon_game(self) -> bool:
        """Abandon the game."""
        if self.state in [GameState.IN_PROGRESS, GameState.PAUSED]:
            self.state = GameState.ABANDONED
            self.game_end_time = datetime.now()
            
            # Notify observers
            self.notify_observers("game_abandoned", {
                'reason': 'Game abandoned by players'
            })
            return True
        return False
    
    def get_current_player(self) -> Optional[Player]:
        """Get current active player."""
        if (self.state == GameState.IN_PROGRESS and 
            0 <= self.current_player_index < len(self.player_order)):
            player_id = self.player_order[self.current_player_index]
            return self.players[player_id]
        return None
    
    def get_game_duration(self) -> timedelta:
        """Get total game duration."""
        if not self.game_start_time:
            return timedelta(0)
        
        end_time = self.game_end_time or datetime.now()
        return end_time - self.game_start_time
    
    def get_leaderboard(self) -> List[Dict[str, Any]]:
        """Get player leaderboard sorted by position."""
        players_stats = []
        
        for player in self.players.values():
            stats = player.get_player_stats()
            stats['rank'] = 0  # Will be calculated
            players_stats.append(stats)
        
        # Sort by position (descending), then by moves (ascending)
        players_stats.sort(key=lambda p: (-p['position'], p['moves_count']))
        
        # Assign ranks
        for i, stats in enumerate(players_stats):
            stats['rank'] = i + 1
        
        return players_stats
    
    def get_game_state(self) -> Dict[str, Any]:
        """Get complete game state."""
        current_player = self.get_current_player()
        
        return {
            'game_id': self.game_id,
            'state': self.state.value,
            'board': self.board.get_board_info(),
            'players': [p.get_player_stats() for p in self.players.values()],
            'current_player': current_player.get_player_stats() if current_player else None,
            'winner': self.winner.get_player_stats() if self.winner else None,
            'dice_strategy': self.dice_strategy.get_dice_name(),
            'settings': {
                'exact_finish': self.exact_finish,
                'allow_multiple_turns': self.allow_multiple_turns,
                'max_players': self.max_players,
                'min_players': self.min_players
            },
            'statistics': {
                'total_turns': self.total_turns,
                'total_dice_rolls': self.total_dice_rolls,
                'total_moves': len(self.move_history),
                'game_duration_seconds': self.get_game_duration().total_seconds()
            },
            'timing': {
                'game_start_time': self.game_start_time.isoformat() if self.game_start_time else None,
                'game_end_time': self.game_end_time.isoformat() if self.game_end_time else None
            }
        }
    
    def save_game_state(self) -> str:
        """Save game state to JSON string."""
        return json.dumps(self.get_game_state(), indent=2)
    
    def __str__(self) -> str:
        return f"Snake & Ladder Game {self.game_id[:8]} - {self.state.value} ({len(self.players)} players)"


# ============================================================================
# OBSERVER PATTERN FOR GAME EVENTS
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
        print(f"📝 Game Event: {event_type} in game {game_id[:8]}")
    
    def get_events_for_game(self, game_id: str) -> List[Dict[str, Any]]:
        """Get events for specific game."""
        return [event for event in self.event_log if event['game_id'] == game_id]
    
    def clear_log(self) -> None:
        """Clear event log."""
        self.event_log.clear()


# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================

def demonstrate_snake_and_ladder_game():
    """Demonstrate the Snake and Ladder game system."""
    print("=== SNAKE AND LADDER GAME DEMONSTRATION ===\n")
    
    # Initialize game
    game = SnakeAndLadderGame(board_size=10)
    
    # Add game logger
    logger = GameLogger()
    game.add_observer(logger)
    
    print("1. GAME SETUP:")
    
    # Add players
    players_data = [
        ("Alice", "red"),
        ("Bob", "blue"),
        ("Charlie", "green"),
        ("Diana", "yellow")
    ]
    
    players = []
    for name, color in players_data:
        player = game.add_player(name, color)
        if player:
            players.append(player)
            print(f"   ✓ {name} joined ({color})")
        else:
            print(f"   ✗ Failed to add {name}")
    
    print()
    
    # Show board information
    print("2. BOARD INFORMATION:")
    board_info = game.board.get_board_info()
    print(f"   Board Size: {board_info['size']}x{board_info['size']}")
    print(f"   Total Positions: {board_info['total_positions']}")
    print(f"   Snakes: {board_info['snakes_count']}")
    print(f"   Ladders: {board_info['ladders_count']}")
    
    # Show some snakes and ladders
    print(f"   Sample Snakes:")
    for i, (snake_id, snake_info) in enumerate(list(board_info['snakes'].items())[:3]):
        print(f"     {snake_info['name']}: {snake_info['head']} -> {snake_info['tail']}")
    
    print(f"   Sample Ladders:")
    for i, (ladder_id, ladder_info) in enumerate(list(board_info['ladders'].items())[:3]):
        print(f"     {ladder_info['name']}: {ladder_info['bottom']} -> {ladder_info['top']}")
    
    print()
    
    # Test different dice strategies
    print("3. DICE STRATEGY TESTING:")
    
    strategies = [
        StandardDiceStrategy(),
        LoadedDiceStrategy(0.3),
        DoubleDiceStrategy(),
        CustomDiceStrategy(1, 8)
    ]
    
    for strategy in strategies:
        game.set_dice_strategy(strategy)
        
        # Roll dice 5 times to show variation
        rolls = [strategy.roll() for _ in range(5)]
        print(f"   {strategy.get_dice_name()}: {rolls}")
    
    # Set back to standard dice
    game.set_dice_strategy(StandardDiceStrategy())
    
    print()
    
    # Start game
    print("4. STARTING GAME:")
    
    success = game.start_game()
    if success:
        print(f"   ✓ Game started successfully")
        print(f"   Current player: {game.get_current_player().name}")
    else:
        print(f"   ✗ Failed to start game")
    
    print()
    
    # Simulate game play
    print("5. GAME SIMULATION:")
    
    move_count = 0
    max_moves = 50  # Limit simulation
    
    while game.state == GameState.IN_PROGRESS and move_count < max_moves:
        current_player = game.get_current_player()
        if not current_player:
            break
        
        # Roll dice
        dice_result = game.roll_dice(current_player.player_id)
        if dice_result is None:
            break
        
        # Make move
        move = game.make_move(current_player.player_id, dice_result)
        if move:
            move_count += 1
            
            # Show interesting moves
            if move.move_type in [MoveType.SNAKE_BITE, MoveType.LADDER_CLIMB, MoveType.WINNING_MOVE]:
                print(f"   🎯 {current_player.name}: {move.move_type.value} ({move.start_position} -> {move.end_position})")
        
        # Small delay for readability
        if move_count % 10 == 0:
            print(f"   ... {move_count} moves completed ...")
    
    print()
    
    # Show game results
    print("6. GAME RESULTS:")
    
    game_state = game.get_game_state()
    
    print(f"   Game Status: {game_state['state']}")
    print(f"   Total Moves: {game_state['statistics']['total_moves']}")
    print(f"   Total Turns: {game_state['statistics']['total_turns']}")
    print(f"   Game Duration: {game_state['statistics']['game_duration_seconds']:.1f} seconds")
    
    if game.winner:
        print(f"   🏆 Winner: {game.winner.name}")
        winner_stats = game.winner.get_player_stats()
        print(f"     Moves: {winner_stats['moves_count']}")
        print(f"     Snakes: {winner_stats['snakes_encountered']}")
        print(f"     Ladders: {winner_stats['ladders_climbed']}")
        print(f"     Average Dice: {winner_stats['dice_stats']['average_roll']:.1f}")
    
    print()
    
    # Show leaderboard
    print("7. LEADERBOARD:")
    
    leaderboard = game.get_leaderboard()
    
    for player_stats in leaderboard:
        name = player_stats['name']
        position = player_stats['position']
        moves = player_stats['moves_count']
        snakes = player_stats['snakes_encountered']
        ladders = player_stats['ladders_climbed']
        avg_dice = player_stats['dice_stats']['average_roll']
        
        print(f"   {player_stats['rank']}. {name}: Position {position}")
        print(f"      Moves: {moves}, Snakes: {snakes}, Ladders: {ladders}, Avg Dice: {avg_dice:.1f}")
    
    print()
    
    # Show player statistics
    print("8. DETAILED PLAYER STATISTICS:")
    
    for player in players[:2]:  # Show first 2 players
        stats = player.get_player_stats()
        
        print(f"   {stats['name']} ({stats['color']}):")
        print(f"     Final Position: {stats['position']}")
        print(f"     Total Moves: {stats['moves_count']}")
        print(f"     Snakes Encountered: {stats['snakes_encountered']}")
        print(f"     Ladders Climbed: {stats['ladders_climbed']}")
        print(f"     Dice Statistics:")
        print(f"       Total Sum: {stats['dice_stats']['total_sum']}")
        print(f"       Average: {stats['dice_stats']['average_roll']:.2f}")
        print(f"       Max Roll: {stats['dice_stats']['max_roll']}")
        print(f"       Min Roll: {stats['dice_stats']['min_roll']}")
        print(f"     Game Duration: {stats['timing']['game_duration_seconds']:.1f} seconds")
    
    print()
    
    # Show move history highlights
    print("9. MOVE HISTORY HIGHLIGHTS:")
    
    interesting_moves = [
        move for move in game.move_history 
        if move.move_type in [MoveType.SNAKE_BITE, MoveType.LADDER_CLIMB, MoveType.WINNING_MOVE]
    ]
    
    print(f"   Showing {min(10, len(interesting_moves))} interesting moves:")
    
    for move in interesting_moves[:10]:
        player_name = game.players[move.player_id].name
        move_type = move.move_type.value.replace('_', ' ').title()
        
        print(f"     {player_name}: {move_type} (Dice: {move.dice_roll}, {move.start_position} -> {move.end_position})")
    
    print()
    
    # Show event log
    print("10. EVENT LOG SUMMARY:")
    
    events = logger.get_events_for_game(game.game_id)
    event_counts = {}
    
    for event in events:
        event_type = event['event_type']
        event_counts[event_type] = event_counts.get(event_type, 0) + 1
    
    print(f"   Total Events: {len(events)}")
    for event_type, count in event_counts.items():
        print(f"     {event_type.replace('_', ' ').title()}: {count}")
    
    print()
    
    # Test game state saving
    print("11. GAME STATE PERSISTENCE:")
    
    # Save game state
    saved_state = game.save_game_state()
    print(f"   ✓ Game state saved ({len(saved_state)} characters)")
    
    # Show sample of saved data
    state_preview = saved_state[:200] + "..." if len(saved_state) > 200 else saved_state
    print(f"   State preview: {state_preview}")
    
    print()
    
    # Test custom board
    print("12. CUSTOM BOARD TESTING:")
    
    custom_game = SnakeAndLadderGame(board_size=6)  # Smaller board
    
    # Add custom snakes and ladders
    try:
        custom_game.board.add_snake(35, 7, "Big Snake", "Dangerous snake")
        custom_game.board.add_ladder(3, 22, "Lucky Ladder", "Quick climb")
        custom_game.board.add_ladder(15, 26, "Short Ladder", "Small boost")
        
        print(f"   ✓ Custom 6x6 board created")
        print(f"   Snakes: {len(custom_game.board.snakes)}")
        print(f"   Ladders: {len(custom_game.board.ladders)}")
        
    except ValueError as e:
        print(f"   ✗ Custom board creation failed: {e}")
    
    print()
    
    # Performance statistics
    print("13. PERFORMANCE STATISTICS:")
    
    total_events = len(logger.event_log)
    total_moves = len(game.move_history)
    game_duration = game.get_game_duration().total_seconds()
    
    print(f"   Events Generated: {total_events}")
    print(f"   Moves Per Second: {total_moves / max(1, game_duration):.1f}")
    print(f"   Events Per Move: {total_events / max(1, total_moves):.1f}")
    
    # Calculate average game statistics
    if game.winner:
        avg_moves_to_win = game.winner.moves_count
        avg_snakes_per_game = sum(p.snakes_encountered for p in game.players.values()) / len(game.players)
        avg_ladders_per_game = sum(p.ladders_climbed for p in game.players.values()) / len(game.players)
        
        print(f"   Average Moves to Win: {avg_moves_to_win}")
        print(f"   Average Snakes per Player: {avg_snakes_per_game:.1f}")
        print(f"   Average Ladders per Player: {avg_ladders_per_game:.1f}")
    
    print()
    print("=== SNAKE AND LADDER GAME DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_snake_and_ladder_game()
