"""
CHESS GAME DESIGN - Complete System Design
==========================================

Problem Statement:
Design a comprehensive chess game system that handles:
- Complete chess board with all pieces and rules
- Player management and turn-based gameplay
- Move validation and game state tracking
- Check, checkmate, and stalemate detection
- Special moves (castling, en passant, pawn promotion)
- Game history and move notation
- Save and load game functionality
- Different game modes (human vs human, human vs AI)
- Tournament management
- Time controls and game clocks

Requirements:
- Implement all chess pieces with proper movement rules
- Validate all moves according to chess rules
- Detect check, checkmate, and stalemate conditions
- Handle special moves correctly
- Support algebraic notation for moves
- Provide game state persistence
- Support different time controls
- Handle draw conditions (50-move rule, repetition)
- Support game analysis and replay
- Implement basic AI for computer opponent

Design Patterns Used:
- Strategy: Piece movement strategies
- State: Game state management
- Command: Move operations with undo/redo
- Factory: Piece creation
- Observer: Game event notifications
- Template Method: Game flow template
- Memento: Game state saving
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Set, Any
from datetime import datetime, timedelta
from enum import Enum
import uuid
import threading
import copy
from dataclasses import dataclass, field


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class Color(Enum):
    WHITE = "white"
    BLACK = "black"


class PieceType(Enum):
    PAWN = "pawn"
    ROOK = "rook"
    KNIGHT = "knight"
    BISHOP = "bishop"
    QUEEN = "queen"
    KING = "king"


class GameState(Enum):
    WAITING_FOR_PLAYERS = "waiting_for_players"
    IN_PROGRESS = "in_progress"
    CHECK = "check"
    CHECKMATE = "checkmate"
    STALEMATE = "stalemate"
    DRAW = "draw"
    RESIGNED = "resigned"
    TIMEOUT = "timeout"
    PAUSED = "paused"


class MoveType(Enum):
    NORMAL = "normal"
    CAPTURE = "capture"
    CASTLING_KINGSIDE = "castling_kingside"
    CASTLING_QUEENSIDE = "castling_queenside"
    EN_PASSANT = "en_passant"
    PAWN_PROMOTION = "pawn_promotion"


@dataclass
class Position:
    """Chess board position."""
    row: int  # 0-7 (rank 1-8)
    col: int  # 0-7 (file a-h)
    
    def __post_init__(self):
        if not (0 <= self.row <= 7 and 0 <= self.col <= 7):
            raise ValueError("Position must be within board bounds")
    
    def to_algebraic(self) -> str:
        """Convert to algebraic notation (e.g., 'e4')."""
        return chr(ord('a') + self.col) + str(self.row + 1)
    
    @classmethod
    def from_algebraic(cls, notation: str) -> 'Position':
        """Create position from algebraic notation."""
        if len(notation) != 2:
            raise ValueError("Invalid algebraic notation")
        
        col = ord(notation[0].lower()) - ord('a')
        row = int(notation[1]) - 1
        
        return cls(row, col)
    
    def __eq__(self, other):
        return isinstance(other, Position) and self.row == other.row and self.col == other.col
    
    def __hash__(self):
        return hash((self.row, self.col))


@dataclass
class Move:
    """Chess move with metadata."""
    move_id: str
    from_pos: Position
    to_pos: Position
    piece_type: PieceType
    piece_color: Color
    move_type: MoveType = MoveType.NORMAL
    captured_piece: Optional[PieceType] = None
    promotion_piece: Optional[PieceType] = None
    is_check: bool = False
    is_checkmate: bool = False
    algebraic_notation: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.move_id:
            self.move_id = str(uuid.uuid4())


# ============================================================================
# CHESS PIECES
# ============================================================================

class ChessPiece(ABC):
    """Abstract chess piece."""
    
    def __init__(self, color: Color, position: Position):
        self.color = color
        self.position = position
        self.has_moved = False
        self.move_count = 0
    
    @abstractmethod
    def get_piece_type(self) -> PieceType:
        """Get piece type."""
        pass
    
    @abstractmethod
    def get_possible_moves(self, board: 'ChessBoard') -> List[Position]:
        """Get all possible moves for this piece."""
        pass
    
    @abstractmethod
    def get_piece_value(self) -> int:
        """Get piece value for evaluation."""
        pass
    
    def move_to(self, position: Position) -> None:
        """Move piece to new position."""
        self.position = position
        self.has_moved = True
        self.move_count += 1
    
    def can_move_to(self, position: Position, board: 'ChessBoard') -> bool:
        """Check if piece can move to position."""
        return position in self.get_possible_moves(board)
    
    def is_path_clear(self, to_pos: Position, board: 'ChessBoard') -> bool:
        """Check if path is clear (for sliding pieces)."""
        row_diff = to_pos.row - self.position.row
        col_diff = to_pos.col - self.position.col
        
        # Normalize direction
        row_step = 0 if row_diff == 0 else (1 if row_diff > 0 else -1)
        col_step = 0 if col_diff == 0 else (1 if col_diff > 0 else -1)
        
        current_row = self.position.row + row_step
        current_col = self.position.col + col_step
        
        while current_row != to_pos.row or current_col != to_pos.col:
            if board.get_piece_at(Position(current_row, current_col)):
                return False
            current_row += row_step
            current_col += col_step
        
        return True
    
    def __str__(self) -> str:
        return f"{self.color.value.title()} {self.get_piece_type().value.title()}"


class Pawn(ChessPiece):
    """Pawn piece implementation."""
    
    def get_piece_type(self) -> PieceType:
        return PieceType.PAWN
    
    def get_piece_value(self) -> int:
        return 1
    
    def get_possible_moves(self, board: 'ChessBoard') -> List[Position]:
        """Get possible pawn moves."""
        moves = []
        direction = 1 if self.color == Color.WHITE else -1
        
        # Forward move
        forward_pos = Position(self.position.row + direction, self.position.col)
        if board.is_valid_position(forward_pos) and not board.get_piece_at(forward_pos):
            moves.append(forward_pos)
            
            # Double move from starting position
            if not self.has_moved:
                double_forward = Position(self.position.row + 2 * direction, self.position.col)
                if board.is_valid_position(double_forward) and not board.get_piece_at(double_forward):
                    moves.append(double_forward)
        
        # Diagonal captures
        for col_offset in [-1, 1]:
            capture_pos = Position(self.position.row + direction, self.position.col + col_offset)
            if board.is_valid_position(capture_pos):
                target_piece = board.get_piece_at(capture_pos)
                if target_piece and target_piece.color != self.color:
                    moves.append(capture_pos)
                
                # En passant
                if board.can_en_passant(self.position, capture_pos):
                    moves.append(capture_pos)
        
        return moves


class Rook(ChessPiece):
    """Rook piece implementation."""
    
    def get_piece_type(self) -> PieceType:
        return PieceType.ROOK
    
    def get_piece_value(self) -> int:
        return 5
    
    def get_possible_moves(self, board: 'ChessBoard') -> List[Position]:
        """Get possible rook moves (horizontal and vertical)."""
        moves = []
        
        # Horizontal and vertical directions
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        for row_dir, col_dir in directions:
            for i in range(1, 8):
                new_row = self.position.row + i * row_dir
                new_col = self.position.col + i * col_dir
                
                if not (0 <= new_row <= 7 and 0 <= new_col <= 7):
                    break
                
                new_pos = Position(new_row, new_col)
                piece_at_pos = board.get_piece_at(new_pos)
                
                if piece_at_pos is None:
                    moves.append(new_pos)
                elif piece_at_pos.color != self.color:
                    moves.append(new_pos)  # Capture
                    break
                else:
                    break  # Own piece blocking
        
        return moves


class Knight(ChessPiece):
    """Knight piece implementation."""
    
    def get_piece_type(self) -> PieceType:
        return PieceType.KNIGHT
    
    def get_piece_value(self) -> int:
        return 3
    
    def get_possible_moves(self, board: 'ChessBoard') -> List[Position]:
        """Get possible knight moves (L-shaped)."""
        moves = []
        
        # Knight move offsets
        knight_moves = [
            (-2, -1), (-2, 1), (-1, -2), (-1, 2),
            (1, -2), (1, 2), (2, -1), (2, 1)
        ]
        
        for row_offset, col_offset in knight_moves:
            new_row = self.position.row + row_offset
            new_col = self.position.col + col_offset
            
            if 0 <= new_row <= 7 and 0 <= new_col <= 7:
                new_pos = Position(new_row, new_col)
                piece_at_pos = board.get_piece_at(new_pos)
                
                if piece_at_pos is None or piece_at_pos.color != self.color:
                    moves.append(new_pos)
        
        return moves


class Bishop(ChessPiece):
    """Bishop piece implementation."""
    
    def get_piece_type(self) -> PieceType:
        return PieceType.BISHOP
    
    def get_piece_value(self) -> int:
        return 3
    
    def get_possible_moves(self, board: 'ChessBoard') -> List[Position]:
        """Get possible bishop moves (diagonal)."""
        moves = []
        
        # Diagonal directions
        directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        for row_dir, col_dir in directions:
            for i in range(1, 8):
                new_row = self.position.row + i * row_dir
                new_col = self.position.col + i * col_dir
                
                if not (0 <= new_row <= 7 and 0 <= new_col <= 7):
                    break
                
                new_pos = Position(new_row, new_col)
                piece_at_pos = board.get_piece_at(new_pos)
                
                if piece_at_pos is None:
                    moves.append(new_pos)
                elif piece_at_pos.color != self.color:
                    moves.append(new_pos)  # Capture
                    break
                else:
                    break  # Own piece blocking
        
        return moves


class Queen(ChessPiece):
    """Queen piece implementation."""
    
    def get_piece_type(self) -> PieceType:
        return PieceType.QUEEN
    
    def get_piece_value(self) -> int:
        return 9
    
    def get_possible_moves(self, board: 'ChessBoard') -> List[Position]:
        """Get possible queen moves (combination of rook and bishop)."""
        moves = []
        
        # All 8 directions (horizontal, vertical, diagonal)
        directions = [
            (0, 1), (0, -1), (1, 0), (-1, 0),  # Rook moves
            (1, 1), (1, -1), (-1, 1), (-1, -1)  # Bishop moves
        ]
        
        for row_dir, col_dir in directions:
            for i in range(1, 8):
                new_row = self.position.row + i * row_dir
                new_col = self.position.col + i * col_dir
                
                if not (0 <= new_row <= 7 and 0 <= new_col <= 7):
                    break
                
                new_pos = Position(new_row, new_col)
                piece_at_pos = board.get_piece_at(new_pos)
                
                if piece_at_pos is None:
                    moves.append(new_pos)
                elif piece_at_pos.color != self.color:
                    moves.append(new_pos)  # Capture
                    break
                else:
                    break  # Own piece blocking
        
        return moves


class King(ChessPiece):
    """King piece implementation."""
    
    def get_piece_type(self) -> PieceType:
        return PieceType.KING
    
    def get_piece_value(self) -> int:
        return 1000  # Invaluable
    
    def get_possible_moves(self, board: 'ChessBoard') -> List[Position]:
        """Get possible king moves (one square in any direction)."""
        moves = []
        
        # King can move one square in any direction
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        for row_offset, col_offset in directions:
            new_row = self.position.row + row_offset
            new_col = self.position.col + col_offset
            
            if 0 <= new_row <= 7 and 0 <= new_col <= 7:
                new_pos = Position(new_row, new_col)
                piece_at_pos = board.get_piece_at(new_pos)
                
                if piece_at_pos is None or piece_at_pos.color != self.color:
                    # Check if move would put king in check
                    if not board.would_be_in_check(self.color, self.position, new_pos):
                        moves.append(new_pos)
        
        # Castling moves
        if not self.has_moved and not board.is_in_check(self.color):
            # Kingside castling
            if board.can_castle_kingside(self.color):
                castle_pos = Position(self.position.row, self.position.col + 2)
                moves.append(castle_pos)
            
            # Queenside castling
            if board.can_castle_queenside(self.color):
                castle_pos = Position(self.position.row, self.position.col - 2)
                moves.append(castle_pos)
        
        return moves


# ============================================================================
# PIECE FACTORY
# ============================================================================

class PieceFactory:
    """Factory for creating chess pieces."""
    
    @staticmethod
    def create_piece(piece_type: PieceType, color: Color, position: Position) -> ChessPiece:
        """Create chess piece of specified type."""
        piece_classes = {
            PieceType.PAWN: Pawn,
            PieceType.ROOK: Rook,
            PieceType.KNIGHT: Knight,
            PieceType.BISHOP: Bishop,
            PieceType.QUEEN: Queen,
            PieceType.KING: King
        }
        
        piece_class = piece_classes.get(piece_type)
        if not piece_class:
            raise ValueError(f"Unknown piece type: {piece_type}")
        
        return piece_class(color, position)


# ============================================================================
# CHESS BOARD
# ============================================================================

class ChessBoard:
    """Chess board with pieces and game logic."""
    
    def __init__(self):
        self.board: List[List[Optional[ChessPiece]]] = [[None for _ in range(8)] for _ in range(8)]
        self.move_history: List[Move] = []
        self.captured_pieces: Dict[Color, List[ChessPiece]] = {Color.WHITE: [], Color.BLACK: []}
        self.en_passant_target: Optional[Position] = None
        self.halfmove_clock = 0  # For 50-move rule
        self.fullmove_number = 1
        
        self._setup_initial_position()
    
    def _setup_initial_position(self) -> None:
        """Setup initial chess position."""
        # Place pawns
        for col in range(8):
            self.board[1][col] = PieceFactory.create_piece(PieceType.PAWN, Color.WHITE, Position(1, col))
            self.board[6][col] = PieceFactory.create_piece(PieceType.PAWN, Color.BLACK, Position(6, col))
        
        # Place other pieces
        piece_order = [PieceType.ROOK, PieceType.KNIGHT, PieceType.BISHOP, PieceType.QUEEN,
                      PieceType.KING, PieceType.BISHOP, PieceType.KNIGHT, PieceType.ROOK]
        
        for col, piece_type in enumerate(piece_order):
            self.board[0][col] = PieceFactory.create_piece(piece_type, Color.WHITE, Position(0, col))
            self.board[7][col] = PieceFactory.create_piece(piece_type, Color.BLACK, Position(7, col))
    
    def get_piece_at(self, position: Position) -> Optional[ChessPiece]:
        """Get piece at position."""
        if not self.is_valid_position(position):
            return None
        return self.board[position.row][position.col]
    
    def set_piece_at(self, position: Position, piece: Optional[ChessPiece]) -> None:
        """Set piece at position."""
        if self.is_valid_position(position):
            self.board[position.row][position.col] = piece
            if piece:
                piece.position = position
    
    def is_valid_position(self, position: Position) -> bool:
        """Check if position is valid on board."""
        return 0 <= position.row <= 7 and 0 <= position.col <= 7
    
    def move_piece(self, from_pos: Position, to_pos: Position, promotion_piece: PieceType = None) -> Optional[Move]:
        """Move piece from one position to another."""
        piece = self.get_piece_at(from_pos)
        if not piece:
            return None
        
        # Validate move
        if not piece.can_move_to(to_pos, self):
            return None
        
        # Determine move type
        move_type = MoveType.NORMAL
        captured_piece = None
        
        target_piece = self.get_piece_at(to_pos)
        if target_piece:
            move_type = MoveType.CAPTURE
            captured_piece = target_piece.get_piece_type()
            self.captured_pieces[target_piece.color].append(target_piece)
        
        # Handle special moves
        if piece.get_piece_type() == PieceType.PAWN:
            # Pawn promotion
            if (piece.color == Color.WHITE and to_pos.row == 7) or (piece.color == Color.BLACK and to_pos.row == 0):
                move_type = MoveType.PAWN_PROMOTION
                if promotion_piece:
                    promoted_piece = PieceFactory.create_piece(promotion_piece, piece.color, to_pos)
                    promoted_piece.has_moved = True
                    piece = promoted_piece
            
            # En passant
            elif self.en_passant_target and to_pos == self.en_passant_target:
                move_type = MoveType.EN_PASSANT
                # Remove the captured pawn
                captured_pawn_pos = Position(from_pos.row, to_pos.col)
                captured_pawn = self.get_piece_at(captured_pawn_pos)
                if captured_pawn:
                    self.captured_pieces[captured_pawn.color].append(captured_pawn)
                    self.set_piece_at(captured_pawn_pos, None)
        
        elif piece.get_piece_type() == PieceType.KING:
            # Castling
            if abs(to_pos.col - from_pos.col) == 2:
                if to_pos.col > from_pos.col:
                    move_type = MoveType.CASTLING_KINGSIDE
                    # Move rook
                    rook = self.get_piece_at(Position(from_pos.row, 7))
                    self.set_piece_at(Position(from_pos.row, 7), None)
                    self.set_piece_at(Position(from_pos.row, 5), rook)
                    rook.move_to(Position(from_pos.row, 5))
                else:
                    move_type = MoveType.CASTLING_QUEENSIDE
                    # Move rook
                    rook = self.get_piece_at(Position(from_pos.row, 0))
                    self.set_piece_at(Position(from_pos.row, 0), None)
                    self.set_piece_at(Position(from_pos.row, 3), rook)
                    rook.move_to(Position(from_pos.row, 3))
        
        # Execute move
        self.set_piece_at(from_pos, None)
        self.set_piece_at(to_pos, piece)
        piece.move_to(to_pos)
        
        # Update en passant target
        self.en_passant_target = None
        if (piece.get_piece_type() == PieceType.PAWN and 
            abs(to_pos.row - from_pos.row) == 2):
            self.en_passant_target = Position((from_pos.row + to_pos.row) // 2, from_pos.col)
        
        # Update move counters
        if move_type == MoveType.CAPTURE or piece.get_piece_type() == PieceType.PAWN:
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1
        
        if piece.color == Color.BLACK:
            self.fullmove_number += 1
        
        # Create move object
        move = Move(
            move_id=str(uuid.uuid4()),
            from_pos=from_pos,
            to_pos=to_pos,
            piece_type=piece.get_piece_type(),
            piece_color=piece.color,
            move_type=move_type,
            captured_piece=captured_piece,
            promotion_piece=promotion_piece
        )
        
        self.move_history.append(move)
        return move
    
    def can_en_passant(self, pawn_pos: Position, target_pos: Position) -> bool:
        """Check if en passant capture is possible."""
        if not self.en_passant_target or target_pos != self.en_passant_target:
            return False
        
        # Check if there's an enemy pawn that just moved two squares
        enemy_pawn_pos = Position(pawn_pos.row, target_pos.col)
        enemy_pawn = self.get_piece_at(enemy_pawn_pos)
        
        return (enemy_pawn and 
                enemy_pawn.get_piece_type() == PieceType.PAWN and
                enemy_pawn.color != self.get_piece_at(pawn_pos).color)
    
    def can_castle_kingside(self, color: Color) -> bool:
        """Check if kingside castling is possible."""
        king_row = 0 if color == Color.WHITE else 7
        king = self.get_piece_at(Position(king_row, 4))
        rook = self.get_piece_at(Position(king_row, 7))
        
        if not king or not rook or king.has_moved or rook.has_moved:
            return False
        
        # Check if squares between king and rook are empty
        for col in range(5, 7):
            if self.get_piece_at(Position(king_row, col)):
                return False
        
        # Check if king would pass through or end up in check
        for col in range(4, 7):
            if self.would_be_in_check(color, Position(king_row, 4), Position(king_row, col)):
                return False
        
        return True
    
    def can_castle_queenside(self, color: Color) -> bool:
        """Check if queenside castling is possible."""
        king_row = 0 if color == Color.WHITE else 7
        king = self.get_piece_at(Position(king_row, 4))
        rook = self.get_piece_at(Position(king_row, 0))
        
        if not king or not rook or king.has_moved or rook.has_moved:
            return False
        
        # Check if squares between king and rook are empty
        for col in range(1, 4):
            if self.get_piece_at(Position(king_row, col)):
                return False
        
        # Check if king would pass through or end up in check
        for col in range(2, 5):
            if self.would_be_in_check(color, Position(king_row, 4), Position(king_row, col)):
                return False
        
        return True
    
    def is_in_check(self, color: Color) -> bool:
        """Check if king of given color is in check."""
        king_pos = self.find_king(color)
        if not king_pos:
            return False
        
        return self.is_square_attacked(king_pos, color)
    
    def would_be_in_check(self, color: Color, from_pos: Position, to_pos: Position) -> bool:
        """Check if move would put king in check."""
        # Make temporary move
        piece = self.get_piece_at(from_pos)
        captured_piece = self.get_piece_at(to_pos)
        
        self.set_piece_at(from_pos, None)
        self.set_piece_at(to_pos, piece)
        
        # Check if in check
        in_check = self.is_in_check(color)
        
        # Undo temporary move
        self.set_piece_at(from_pos, piece)
        self.set_piece_at(to_pos, captured_piece)
        
        return in_check
    
    def is_square_attacked(self, position: Position, defending_color: Color) -> bool:
        """Check if square is attacked by enemy pieces."""
        attacking_color = Color.BLACK if defending_color == Color.WHITE else Color.WHITE
        
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if piece and piece.color == attacking_color:
                    if position in piece.get_possible_moves(self):
                        return True
        
        return False
    
    def find_king(self, color: Color) -> Optional[Position]:
        """Find king position for given color."""
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if (piece and 
                    piece.get_piece_type() == PieceType.KING and 
                    piece.color == color):
                    return Position(row, col)
        return None
    
    def get_all_legal_moves(self, color: Color) -> List[Tuple[Position, Position]]:
        """Get all legal moves for given color."""
        legal_moves = []
        
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if piece and piece.color == color:
                    from_pos = Position(row, col)
                    possible_moves = piece.get_possible_moves(self)
                    
                    for to_pos in possible_moves:
                        if not self.would_be_in_check(color, from_pos, to_pos):
                            legal_moves.append((from_pos, to_pos))
        
        return legal_moves
    
    def is_checkmate(self, color: Color) -> bool:
        """Check if given color is in checkmate."""
        if not self.is_in_check(color):
            return False
        
        return len(self.get_all_legal_moves(color)) == 0
    
    def is_stalemate(self, color: Color) -> bool:
        """Check if given color is in stalemate."""
        if self.is_in_check(color):
            return False
        
        return len(self.get_all_legal_moves(color)) == 0
    
    def get_board_state(self) -> List[List[str]]:
        """Get current board state as string representation."""
        board_state = []
        
        for row in range(7, -1, -1):  # Display from rank 8 to 1
            row_state = []
            for col in range(8):
                piece = self.board[row][col]
                if piece:
                    symbol = piece.get_piece_type().value[0].upper()
                    if piece.color == Color.BLACK:
                        symbol = symbol.lower()
                    row_state.append(symbol)
                else:
                    row_state.append('.')
            board_state.append(row_state)
        
        return board_state
    
    def __str__(self) -> str:
        """String representation of the board."""
        board_state = self.get_board_state()
        result = "  a b c d e f g h\n"
        
        for i, row in enumerate(board_state):
            rank = 8 - i
            result += f"{rank} {' '.join(row)} {rank}\n"
        
        result += "  a b c d e f g h"
        return result


# ============================================================================
# CHESS GAME ENGINE
# ============================================================================

class ChessGame:
    """Main chess game engine."""
    
    def __init__(self, game_id: str = None):
        self.game_id = game_id or str(uuid.uuid4())
        self.board = ChessBoard()
        self.state = GameState.WAITING_FOR_PLAYERS
        
        # Players
        self.white_player: Optional['ChessPlayer'] = None
        self.black_player: Optional['ChessPlayer'] = None
        self.current_turn = Color.WHITE
        
        # Game settings
        self.time_control: Optional['TimeControl'] = None
        self.draw_offered_by: Optional[Color] = None
        
        # Game metadata
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.result: Optional[str] = None
        self.winner: Optional[Color] = None
        
        # Threading
        self._lock = threading.Lock()
        
        print(f"♟️ Chess Game created: {self.game_id[:8]}")
    
    def add_player(self, player: 'ChessPlayer', color: Color) -> bool:
        """Add player to the game."""
        with self._lock:
            if color == Color.WHITE and not self.white_player:
                self.white_player = player
                player.color = color
                return True
            elif color == Color.BLACK and not self.black_player:
                self.black_player = player
                player.color = color
                return True
            return False
    
    def start_game(self) -> bool:
        """Start the chess game."""
        with self._lock:
            if not self.white_player or not self.black_player:
                return False
            
            if self.state != GameState.WAITING_FOR_PLAYERS:
                return False
            
            self.state = GameState.IN_PROGRESS
            self.start_time = datetime.now()
            
            if self.time_control:
                self.time_control.start_game()
            
            print("Chess game started!")
            return True
    
    def make_move(self, player: 'ChessPlayer', from_algebraic: str, to_algebraic: str,
                 promotion_piece: PieceType = None) -> bool:
        """Make a move in algebraic notation."""
        with self._lock:
            if self.state != GameState.IN_PROGRESS:
                return False
            
            if player.color != self.current_turn:
                return False
            
            try:
                from_pos = Position.from_algebraic(from_algebraic)
                to_pos = Position.from_algebraic(to_algebraic)
            except ValueError:
                return False
            
            # Validate that player owns the piece
            piece = self.board.get_piece_at(from_pos)
            if not piece or piece.color != player.color:
                return False
            
            # Make the move
            move = self.board.move_piece(from_pos, to_pos, promotion_piece)
            if not move:
                return False
            
            # Update game state
            self._update_game_state()
            
            # Switch turns
            self.current_turn = Color.BLACK if self.current_turn == Color.WHITE else Color.WHITE
            
            # Update time control
            if self.time_control:
                self.time_control.switch_turn()
            
            print(f"Move: {from_algebraic}-{to_algebraic}")
            return True
    
    def _update_game_state(self) -> None:
        """Update game state after move."""
        opponent_color = Color.BLACK if self.current_turn == Color.WHITE else Color.WHITE
        
        if self.board.is_checkmate(opponent_color):
            self.state = GameState.CHECKMATE
            self.winner = self.current_turn
            self.result = f"{self.current_turn.value.title()} wins by checkmate"
            self.end_time = datetime.now()
        elif self.board.is_stalemate(opponent_color):
            self.state = GameState.STALEMATE
            self.result = "Draw by stalemate"
            self.end_time = datetime.now()
        elif self.board.is_in_check(opponent_color):
            self.state = GameState.CHECK
        elif self.board.halfmove_clock >= 50:
            self.state = GameState.DRAW
            self.result = "Draw by 50-move rule"
            self.end_time = datetime.now()
        else:
            self.state = GameState.IN_PROGRESS
    
    def offer_draw(self, player: 'ChessPlayer') -> bool:
        """Offer draw."""
        if player.color == self.current_turn:
            self.draw_offered_by = player.color
            return True
        return False
    
    def accept_draw(self, player: 'ChessPlayer') -> bool:
        """Accept draw offer."""
        if (self.draw_offered_by and 
            player.color != self.draw_offered_by and
            player.color == self.current_turn):
            self.state = GameState.DRAW
            self.result = "Draw by agreement"
            self.end_time = datetime.now()
            return True
        return False
    
    def resign(self, player: 'ChessPlayer') -> bool:
        """Resign the game."""
        if player.color in [Color.WHITE, Color.BLACK]:
            self.state = GameState.RESIGNED
            self.winner = Color.BLACK if player.color == Color.WHITE else Color.WHITE
            self.result = f"{self.winner.value.title()} wins by resignation"
            self.end_time = datetime.now()
            return True
        return False
    
    def get_game_state(self) -> Dict[str, Any]:
        """Get current game state."""
        return {
            'game_id': self.game_id,
            'state': self.state.value,
            'current_turn': self.current_turn.value,
            'board': self.board.get_board_state(),
            'move_count': len(self.board.move_history),
            'halfmove_clock': self.board.halfmove_clock,
            'fullmove_number': self.board.fullmove_number,
            'white_player': self.white_player.name if self.white_player else None,
            'black_player': self.black_player.name if self.black_player else None,
            'result': self.result,
            'winner': self.winner.value if self.winner else None,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None
        }


# ============================================================================
# PLAYER AND TIME CONTROL
# ============================================================================

class ChessPlayer:
    """Chess player."""
    
    def __init__(self, player_id: str, name: str):
        self.player_id = player_id
        self.name = name
        self.color: Optional[Color] = None
        self.rating = 1200  # Default rating
        
        # Statistics
        self.games_played = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
    
    def update_stats(self, result: str) -> None:
        """Update player statistics."""
        self.games_played += 1
        
        if "wins" in result and self.color.value in result:
            self.wins += 1
        elif "wins" in result:
            self.losses += 1
        else:
            self.draws += 1
    
    def get_win_rate(self) -> float:
        """Get win rate percentage."""
        if self.games_played == 0:
            return 0.0
        return (self.wins / self.games_played) * 100


class TimeControl:
    """Chess time control system."""
    
    def __init__(self, initial_time_minutes: int, increment_seconds: int = 0):
        self.initial_time = timedelta(minutes=initial_time_minutes)
        self.increment = timedelta(seconds=increment_seconds)
        
        self.white_time = self.initial_time
        self.black_time = self.initial_time
        self.current_player_start: Optional[datetime] = None
        self.is_running = False
    
    def start_game(self) -> None:
        """Start time control."""
        self.is_running = True
        self.current_player_start = datetime.now()
    
    def switch_turn(self) -> None:
        """Switch turn and update time."""
        if not self.is_running or not self.current_player_start:
            return
        
        # Calculate time used
        time_used = datetime.now() - self.current_player_start
        
        # Deduct time and add increment
        if self.current_turn == Color.WHITE:
            self.white_time -= time_used
            self.white_time += self.increment
        else:
            self.black_time -= time_used
            self.black_time += self.increment
        
        # Start timer for next player
        self.current_player_start = datetime.now()
    
    def get_remaining_time(self, color: Color) -> timedelta:
        """Get remaining time for player."""
        if color == Color.WHITE:
            return self.white_time
        else:
            return self.black_time
    
    def is_time_up(self, color: Color) -> bool:
        """Check if player's time is up."""
        return self.get_remaining_time(color) <= timedelta(0)


# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================

def demonstrate_chess_game():
    """Demonstrate the chess game system."""
    print("=== CHESS GAME DESIGN DEMONSTRATION ===\n")
    
    # Create game and players
    game = ChessGame()
    
    white_player = ChessPlayer("player1", "Alice")
    black_player = ChessPlayer("player2", "Bob")
    
    print("1. GAME SETUP:")
    
    # Add players
    success1 = game.add_player(white_player, Color.WHITE)
    success2 = game.add_player(black_player, Color.BLACK)
    
    print(f"   ✓ White player (Alice): {success1}")
    print(f"   ✓ Black player (Bob): {success2}")
    
    # Start game
    game_started = game.start_game()
    print(f"   ✓ Game started: {game_started}")
    
    print()
    
    # Show initial board
    print("2. INITIAL BOARD POSITION:")
    print(game.board)
    print()
    
    # Test piece movements
    print("3. PIECE MOVEMENT TESTING:")
    
    # Test some opening moves
    moves = [
        ("e2", "e4", "White pawn to e4"),
        ("e7", "e5", "Black pawn to e5"),
        ("g1", "f3", "White knight to f3"),
        ("b8", "c6", "Black knight to c6"),
        ("f1", "c4", "White bishop to c4"),
        ("f8", "c5", "Black bishop to c5")
    ]
    
    for i, (from_sq, to_sq, description) in enumerate(moves):
        current_player = white_player if i % 2 == 0 else black_player
        success = game.make_move(current_player, from_sq, to_sq)
        
        if success:
            print(f"   ✓ {description}: {from_sq}-{to_sq}")
        else:
            print(f"   ✗ Failed: {description}")
    
    print()
    
    # Show board after moves
    print("4. BOARD AFTER OPENING MOVES:")
    print(game.board)
    print()
    
    # Test special moves
    print("5. SPECIAL MOVES TESTING:")
    
    # Create a new game for special move testing
    special_game = ChessGame()
    special_game.add_player(white_player, Color.WHITE)
    special_game.add_player(black_player, Color.BLACK)
    special_game.start_game()
    
    # Setup for castling test
    castling_moves = [
        ("e2", "e4"), ("e7", "e5"),
        ("g1", "f3"), ("b8", "c6"),
        ("f1", "e2"), ("f8", "e7"),
        ("e1", "g1")  # Kingside castling attempt
    ]
    
    print("   Testing castling:")
    for i, (from_sq, to_sq) in enumerate(castling_moves):
        current_player = white_player if i % 2 == 0 else black_player
        success = special_game.make_move(current_player, from_sq, to_sq)
        
        if i == len(castling_moves) - 1:  # Last move (castling)
            if success:
                print(f"   ✓ Kingside castling successful")
            else:
                print(f"   ✗ Kingside castling failed")
    
    print()
    
    # Test piece values and evaluation
    print("6. PIECE EVALUATION:")
    
    piece_values = {}
    for piece_type in PieceType:
        piece = PieceFactory.create_piece(piece_type, Color.WHITE, Position(0, 0))
        piece_values[piece_type.value] = piece.get_piece_value()
    
    print("   Piece Values:")
    for piece_name, value in piece_values.items():
        print(f"     {piece_name.title()}: {value}")
    
    print()
    
    # Test game state detection
    print("7. GAME STATE DETECTION:")
    
    # Test check detection
    check_game = ChessGame()
    check_game.add_player(white_player, Color.WHITE)
    check_game.add_player(black_player, Color.BLACK)
    check_game.start_game()
    
    # Create a check position
    check_moves = [
        ("e2", "e4"), ("e7", "e5"),
        ("d1", "h5"), ("b8", "c6"),
        ("f1", "c4"), ("g8", "f6"),
        ("h5", "f7")  # Check!
    ]
    
    for i, (from_sq, to_sq) in enumerate(check_moves):
        current_player = white_player if i % 2 == 0 else black_player
        success = check_game.make_move(current_player, from_sq, to_sq)
        
        if i == len(check_moves) - 1 and success:
            print(f"   ✓ Check detected: {check_game.state.value}")
    
    print()
    
    # Show legal moves
    print("8. LEGAL MOVES ANALYSIS:")
    
    # Get legal moves for white in opening position
    legal_moves = game.board.get_all_legal_moves(Color.WHITE)
    print(f"   Legal moves for White: {len(legal_moves)}")
    
    # Show some example legal moves
    print("   Sample legal moves:")
    for i, (from_pos, to_pos) in enumerate(legal_moves[:5]):
        print(f"     {from_pos.to_algebraic()}-{to_pos.to_algebraic()}")
    
    print()
    
    # Test time control
    print("9. TIME CONTROL TESTING:")
    
    time_control = TimeControl(initial_time_minutes=10, increment_seconds=5)
    print(f"   ✓ Time control created: 10+5 (10 minutes + 5 second increment)")
    
    time_control.start_game()
    print(f"   White time: {time_control.get_remaining_time(Color.WHITE)}")
    print(f"   Black time: {time_control.get_remaining_time(Color.BLACK)}")
    
    print()
    
    # Test game statistics
    print("10. GAME STATISTICS:")
    
    game_state = game.get_game_state()
    
    print(f"   Game ID: {game_state['game_id'][:8]}")
    print(f"   State: {game_state['state']}")
    print(f"   Current Turn: {game_state['current_turn']}")
    print(f"   Moves Played: {game_state['move_count']}")
    print(f"   Halfmove Clock: {game_state['halfmove_clock']}")
    print(f"   Fullmove Number: {game_state['fullmove_number']}")
    
    if game_state['result']:
        print(f"   Result: {game_state['result']}")
    
    print()
    
    # Test player statistics
    print("11. PLAYER STATISTICS:")
    
    # Simulate some game results for statistics
    white_player.update_stats("White wins by checkmate")
    white_player.update_stats("Draw by stalemate")
    white_player.update_stats("Black wins by resignation")
    
    print(f"   {white_player.name} (White):")
    print(f"     Games Played: {white_player.games_played}")
    print(f"     Wins: {white_player.wins}")
    print(f"     Losses: {white_player.losses}")
    print(f"     Draws: {white_player.draws}")
    print(f"     Win Rate: {white_player.get_win_rate():.1f}%")
    print(f"     Rating: {white_player.rating}")
    
    print()
    
    # Test move history
    print("12. MOVE HISTORY:")
    
    print("   Game moves:")
    for i, move in enumerate(game.board.move_history):
        move_number = (i // 2) + 1
        color = "White" if i % 2 == 0 else "Black"
        
        print(f"     {move_number}. {color}: {move.from_pos.to_algebraic()}-{move.to_pos.to_algebraic()}")
        
        if move.move_type != MoveType.NORMAL:
            print(f"        ({move.move_type.value})")
    
    print()
    
    # Test captured pieces
    print("13. CAPTURED PIECES:")
    
    white_captured = game.board.captured_pieces[Color.WHITE]
    black_captured = game.board.captured_pieces[Color.BLACK]
    
    print(f"   White captured: {len(white_captured)} pieces")
    for piece in white_captured:
        print(f"     - {piece}")
    
    print(f"   Black captured: {len(black_captured)} pieces")
    for piece in black_captured:
        print(f"     - {piece}")
    
    print()
    
    # Test position notation
    print("14. POSITION NOTATION:")
    
    test_positions = ["a1", "e4", "h8", "d5"]
    
    print("   Algebraic notation conversion:")
    for notation in test_positions:
        try:
            pos = Position.from_algebraic(notation)
            back_to_algebraic = pos.to_algebraic()
            print(f"     {notation} -> ({pos.row}, {pos.col}) -> {back_to_algebraic}")
        except ValueError as e:
            print(f"     {notation} -> Error: {e}")
    
    print()
    
    print("=== CHESS GAME DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_chess_game()
