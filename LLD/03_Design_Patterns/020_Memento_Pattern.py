"""
MEMENTO PATTERN - Behavioral Design Pattern
============================================

Problem Statement:
Implement the Memento pattern to capture and externalize an object's internal
state without violating encapsulation, so that the object can be restored to
this state later:
- State capture and restoration without exposing internals
- Undo/redo functionality implementation
- Checkpoint and rollback mechanisms
- Version control and history management
- Game state saving and loading

Learning Objectives:
- Understand Memento vs Command pattern for undo operations
- Implement state capture without breaking encapsulation
- Design caretaker and originator relationships
- Handle large state objects and memory optimization
- Create robust state management systems
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Generic, TypeVar
import time
import json
import copy
import pickle
import hashlib
from datetime import datetime
from enum import Enum
import threading
from dataclasses import dataclass, field


# ============================================================================
# MEMENTO INTERFACE
# ============================================================================

T = TypeVar('T')

class Memento(ABC):
    """Abstract memento interface."""
    
    @abstractmethod
    def get_state_id(self) -> str:
        """Get unique identifier for this state."""
        pass
    
    @abstractmethod
    def get_timestamp(self) -> datetime:
        """Get timestamp when memento was created."""
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about this memento."""
        pass


class Originator(ABC):
    """Abstract originator that can create and restore mementos."""
    
    @abstractmethod
    def create_memento(self, description: str = "") -> Memento:
        """Create memento of current state."""
        pass
    
    @abstractmethod
    def restore_memento(self, memento: Memento) -> None:
        """Restore state from memento."""
        pass
    
    @abstractmethod
    def get_current_state_info(self) -> Dict[str, Any]:
        """Get information about current state."""
        pass


class Caretaker:
    """Caretaker manages mementos without accessing their content."""
    
    def __init__(self, max_mementos: int = 100):
        self.mementos: List[Memento] = []
        self.max_mementos = max_mementos
        self.current_index = -1
        
    def save_memento(self, memento: Memento) -> None:
        """Save memento and manage history."""
        # Remove any mementos after current index (for redo functionality)
        if self.current_index < len(self.mementos) - 1:
            self.mementos = self.mementos[:self.current_index + 1]
        
        # Add new memento
        self.mementos.append(memento)
        self.current_index += 1
        
        # Limit number of mementos
        if len(self.mementos) > self.max_mementos:
            self.mementos.pop(0)
            self.current_index -= 1
        
        print(f"Saved memento: {memento.get_state_id()}")
    
    def get_memento(self, index: int = None) -> Optional[Memento]:
        """Get memento by index (current if None)."""
        if index is None:
            index = self.current_index
        
        if 0 <= index < len(self.mementos):
            return self.mementos[index]
        return None
    
    def undo(self) -> Optional[Memento]:
        """Get previous memento for undo operation."""
        if self.current_index > 0:
            self.current_index -= 1
            return self.mementos[self.current_index]
        return None
    
    def redo(self) -> Optional[Memento]:
        """Get next memento for redo operation."""
        if self.current_index < len(self.mementos) - 1:
            self.current_index += 1
            return self.mementos[self.current_index]
        return None
    
    def can_undo(self) -> bool:
        """Check if undo is possible."""
        return self.current_index > 0
    
    def can_redo(self) -> bool:
        """Check if redo is possible."""
        return self.current_index < len(self.mementos) - 1
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get history of all mementos."""
        return [
            {
                'index': i,
                'state_id': memento.get_state_id(),
                'timestamp': memento.get_timestamp().isoformat(),
                'metadata': memento.get_metadata(),
                'is_current': i == self.current_index
            }
            for i, memento in enumerate(self.mementos)
        ]
    
    def clear_history(self) -> None:
        """Clear all mementos."""
        self.mementos.clear()
        self.current_index = -1
        print("Cleared memento history")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get caretaker statistics."""
        return {
            'total_mementos': len(self.mementos),
            'current_index': self.current_index,
            'can_undo': self.can_undo(),
            'can_redo': self.can_redo(),
            'max_mementos': self.max_mementos,
            'memory_usage_estimate': len(self.mementos) * 1024  # Rough estimate
        }


# ============================================================================
# TEXT EDITOR WITH MEMENTO
# ============================================================================

@dataclass
class TextEditorState:
    """Text editor state data."""
    content: str = ""
    cursor_position: int = 0
    selection_start: int = 0
    selection_end: int = 0
    font_size: int = 12
    font_family: str = "Arial"
    is_bold: bool = False
    is_italic: bool = False
    zoom_level: float = 1.0
    
    def __post_init__(self):
        """Validate state after initialization."""
        self.cursor_position = max(0, min(self.cursor_position, len(self.content)))
        self.selection_start = max(0, min(self.selection_start, len(self.content)))
        self.selection_end = max(0, min(self.selection_end, len(self.content)))


class TextEditorMemento(Memento):
    """Memento for text editor state."""
    
    def __init__(self, state: TextEditorState, description: str = ""):
        self._state = copy.deepcopy(state)
        self._state_id = self._generate_state_id()
        self._timestamp = datetime.now()
        self._description = description
        self._metadata = {
            'description': description,
            'content_length': len(state.content),
            'cursor_position': state.cursor_position,
            'has_selection': state.selection_start != state.selection_end,
            'font_size': state.font_size,
            'zoom_level': state.zoom_level
        }
    
    def _generate_state_id(self) -> str:
        """Generate unique state ID based on content."""
        state_str = f"{self._state.content}_{self._state.cursor_position}_{self._state.font_size}"
        return hashlib.md5(state_str.encode()).hexdigest()[:8]
    
    def get_state_id(self) -> str:
        """Get state identifier."""
        return self._state_id
    
    def get_timestamp(self) -> datetime:
        """Get creation timestamp."""
        return self._timestamp
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get memento metadata."""
        return self._metadata.copy()
    
    def get_state(self) -> TextEditorState:
        """Get the stored state (only accessible by originator)."""
        return copy.deepcopy(self._state)


class TextEditor(Originator):
    """Text editor with memento support."""
    
    def __init__(self):
        self._state = TextEditorState()
        self.caretaker = Caretaker(max_mementos=50)
        self.auto_save_enabled = True
        self.auto_save_interval = 10  # characters
        self.characters_since_save = 0
        
        # Save initial state
        self.save_checkpoint("Initial state")
    
    def create_memento(self, description: str = "") -> Memento:
        """Create memento of current state."""
        return TextEditorMemento(self._state, description)
    
    def restore_memento(self, memento: Memento) -> None:
        """Restore state from memento."""
        if isinstance(memento, TextEditorMemento):
            self._state = memento.get_state()
            print(f"Restored state: {memento.get_metadata()['description']}")
    
    def get_current_state_info(self) -> Dict[str, Any]:
        """Get current state information."""
        return {
            'content_length': len(self._state.content),
            'cursor_position': self._state.cursor_position,
            'has_selection': self._state.selection_start != self._state.selection_end,
            'font_size': self._state.font_size,
            'font_family': self._state.font_family,
            'formatting': {
                'bold': self._state.is_bold,
                'italic': self._state.is_italic
            },
            'zoom_level': self._state.zoom_level
        }
    
    def insert_text(self, text: str, position: int = None) -> None:
        """Insert text at specified position."""
        if position is None:
            position = self._state.cursor_position
        
        # Insert text
        content = self._state.content
        self._state.content = content[:position] + text + content[position:]
        self._state.cursor_position = position + len(text)
        
        # Auto-save if enabled
        self.characters_since_save += len(text)
        if self.auto_save_enabled and self.characters_since_save >= self.auto_save_interval:
            self.save_checkpoint(f"Auto-save after inserting '{text[:20]}...'")
            self.characters_since_save = 0
        
        print(f"Inserted text: '{text}' at position {position}")
    
    def delete_text(self, start: int, end: int) -> str:
        """Delete text between start and end positions."""
        deleted_text = self._state.content[start:end]
        self._state.content = self._state.content[:start] + self._state.content[end:]
        
        # Adjust cursor position
        if self._state.cursor_position > end:
            self._state.cursor_position -= (end - start)
        elif self._state.cursor_position > start:
            self._state.cursor_position = start
        
        print(f"Deleted text: '{deleted_text}' from position {start}-{end}")
        return deleted_text
    
    def set_selection(self, start: int, end: int) -> None:
        """Set text selection."""
        self._state.selection_start = max(0, min(start, len(self._state.content)))
        self._state.selection_end = max(0, min(end, len(self._state.content)))
        self._state.cursor_position = self._state.selection_end
        
        print(f"Selection set: {self._state.selection_start}-{self._state.selection_end}")
    
    def set_font_size(self, size: int) -> None:
        """Set font size."""
        self._state.font_size = max(8, min(size, 72))
        print(f"Font size set to: {self._state.font_size}")
    
    def set_font_family(self, family: str) -> None:
        """Set font family."""
        self._state.font_family = family
        print(f"Font family set to: {family}")
    
    def toggle_bold(self) -> None:
        """Toggle bold formatting."""
        self._state.is_bold = not self._state.is_bold
        print(f"Bold formatting: {'ON' if self._state.is_bold else 'OFF'}")
    
    def toggle_italic(self) -> None:
        """Toggle italic formatting."""
        self._state.is_italic = not self._state.is_italic
        print(f"Italic formatting: {'ON' if self._state.is_italic else 'OFF'}")
    
    def set_zoom_level(self, zoom: float) -> None:
        """Set zoom level."""
        self._state.zoom_level = max(0.5, min(zoom, 3.0))
        print(f"Zoom level set to: {self._state.zoom_level:.1f}x")
    
    def save_checkpoint(self, description: str = "") -> None:
        """Save current state as checkpoint."""
        memento = self.create_memento(description)
        self.caretaker.save_memento(memento)
    
    def undo(self) -> bool:
        """Undo last operation."""
        memento = self.caretaker.undo()
        if memento:
            self.restore_memento(memento)
            return True
        return False
    
    def redo(self) -> bool:
        """Redo last undone operation."""
        memento = self.caretaker.redo()
        if memento:
            self.restore_memento(memento)
            return True
        return False
    
    def get_content(self) -> str:
        """Get current content."""
        return self._state.content
    
    def get_cursor_position(self) -> int:
        """Get current cursor position."""
        return self._state.cursor_position
    
    def get_selection(self) -> tuple:
        """Get current selection."""
        return (self._state.selection_start, self._state.selection_end)
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get edit history."""
        return self.caretaker.get_history()


# ============================================================================
# GAME STATE MEMENTO
# ============================================================================

@dataclass
class GameState:
    """Game state data."""
    level: int = 1
    score: int = 0
    lives: int = 3
    player_position: tuple = field(default_factory=lambda: (0, 0))
    inventory: List[str] = field(default_factory=list)
    health: int = 100
    mana: int = 50
    experience: int = 0
    achievements: List[str] = field(default_factory=list)
    game_time: float = 0.0
    difficulty: str = "normal"
    
    def __post_init__(self):
        """Validate state after initialization."""
        self.level = max(1, self.level)
        self.score = max(0, self.score)
        self.lives = max(0, self.lives)
        self.health = max(0, min(self.health, 100))
        self.mana = max(0, min(self.mana, 100))
        self.experience = max(0, self.experience)


class GameStateMemento(Memento):
    """Memento for game state."""
    
    def __init__(self, state: GameState, description: str = "", save_type: str = "manual"):
        self._state = copy.deepcopy(state)
        self._state_id = self._generate_state_id()
        self._timestamp = datetime.now()
        self._description = description
        self._save_type = save_type
        self._metadata = {
            'description': description,
            'save_type': save_type,
            'level': state.level,
            'score': state.score,
            'lives': state.lives,
            'health': state.health,
            'game_time': state.game_time,
            'inventory_size': len(state.inventory),
            'achievements_count': len(state.achievements)
        }
    
    def _generate_state_id(self) -> str:
        """Generate unique state ID."""
        state_str = f"{self._state.level}_{self._state.score}_{self._state.lives}_{self._state.game_time}"
        return hashlib.md5(state_str.encode()).hexdigest()[:8]
    
    def get_state_id(self) -> str:
        return self._state_id
    
    def get_timestamp(self) -> datetime:
        return self._timestamp
    
    def get_metadata(self) -> Dict[str, Any]:
        return self._metadata.copy()
    
    def get_state(self) -> GameState:
        """Get the stored state."""
        return copy.deepcopy(self._state)
    
    def get_save_type(self) -> str:
        """Get save type (manual, auto, checkpoint)."""
        return self._save_type


class Game(Originator):
    """Game with save/load functionality using memento pattern."""
    
    def __init__(self, player_name: str):
        self.player_name = player_name
        self._state = GameState()
        self.caretaker = Caretaker(max_mementos=20)
        self.auto_save_enabled = True
        self.auto_save_interval = 60.0  # seconds
        self.last_auto_save = time.time()
        
        # Save initial state
        self.save_game("New game started", "manual")
    
    def create_memento(self, description: str = "") -> Memento:
        """Create memento of current game state."""
        return GameStateMemento(self._state, description)
    
    def restore_memento(self, memento: Memento) -> None:
        """Restore game state from memento."""
        if isinstance(memento, GameStateMemento):
            self._state = memento.get_state()
            print(f"Game loaded: {memento.get_metadata()['description']}")
    
    def get_current_state_info(self) -> Dict[str, Any]:
        """Get current game state information."""
        return {
            'player_name': self.player_name,
            'level': self._state.level,
            'score': self._state.score,
            'lives': self._state.lives,
            'health': self._state.health,
            'mana': self._state.mana,
            'experience': self._state.experience,
            'position': self._state.player_position,
            'inventory_items': len(self._state.inventory),
            'achievements': len(self._state.achievements),
            'game_time': self._state.game_time,
            'difficulty': self._state.difficulty
        }
    
    def update_game_time(self, delta_time: float) -> None:
        """Update game time."""
        self._state.game_time += delta_time
        
        # Auto-save if enabled
        current_time = time.time()
        if (self.auto_save_enabled and 
            current_time - self.last_auto_save >= self.auto_save_interval):
            self.save_game("Auto-save", "auto")
            self.last_auto_save = current_time
    
    def move_player(self, x: int, y: int) -> None:
        """Move player to new position."""
        self._state.player_position = (x, y)
        print(f"Player moved to position: ({x}, {y})")
    
    def gain_score(self, points: int) -> None:
        """Add points to score."""
        self._state.score += points
        print(f"Gained {points} points. Total score: {self._state.score}")
    
    def lose_life(self) -> bool:
        """Lose a life. Returns True if game over."""
        self._state.lives -= 1
        print(f"Lost a life. Lives remaining: {self._state.lives}")
        
        if self._state.lives <= 0:
            print("Game Over!")
            return True
        return False
    
    def gain_life(self) -> None:
        """Gain an extra life."""
        self._state.lives += 1
        print(f"Gained a life. Total lives: {self._state.lives}")
    
    def take_damage(self, damage: int) -> bool:
        """Take damage. Returns True if player died."""
        self._state.health -= damage
        print(f"Took {damage} damage. Health: {self._state.health}")
        
        if self._state.health <= 0:
            self._state.health = 0
            return self.lose_life()
        return False
    
    def heal(self, amount: int) -> None:
        """Heal player."""
        self._state.health = min(100, self._state.health + amount)
        print(f"Healed {amount} HP. Health: {self._state.health}")
    
    def use_mana(self, amount: int) -> bool:
        """Use mana. Returns True if successful."""
        if self._state.mana >= amount:
            self._state.mana -= amount
            print(f"Used {amount} mana. Mana: {self._state.mana}")
            return True
        else:
            print("Not enough mana!")
            return False
    
    def restore_mana(self, amount: int) -> None:
        """Restore mana."""
        self._state.mana = min(100, self._state.mana + amount)
        print(f"Restored {amount} mana. Mana: {self._state.mana}")
    
    def gain_experience(self, exp: int) -> bool:
        """Gain experience. Returns True if leveled up."""
        self._state.experience += exp
        exp_needed = self._state.level * 100  # Simple leveling formula
        
        if self._state.experience >= exp_needed:
            self.level_up()
            return True
        
        print(f"Gained {exp} XP. Total: {self._state.experience}/{exp_needed}")
        return False
    
    def level_up(self) -> None:
        """Level up player."""
        self._state.level += 1
        self._state.experience = 0
        self._state.health = 100  # Full heal on level up
        self._state.mana = 100    # Full mana on level up
        
        print(f"LEVEL UP! Now level {self._state.level}")
        self.save_game(f"Level {self._state.level} reached", "checkpoint")
    
    def add_to_inventory(self, item: str) -> None:
        """Add item to inventory."""
        self._state.inventory.append(item)
        print(f"Added '{item}' to inventory")
    
    def remove_from_inventory(self, item: str) -> bool:
        """Remove item from inventory."""
        if item in self._state.inventory:
            self._state.inventory.remove(item)
            print(f"Removed '{item}' from inventory")
            return True
        else:
            print(f"'{item}' not found in inventory")
            return False
    
    def unlock_achievement(self, achievement: str) -> None:
        """Unlock achievement."""
        if achievement not in self._state.achievements:
            self._state.achievements.append(achievement)
            print(f"🏆 Achievement unlocked: {achievement}")
    
    def set_difficulty(self, difficulty: str) -> None:
        """Set game difficulty."""
        valid_difficulties = ["easy", "normal", "hard", "nightmare"]
        if difficulty in valid_difficulties:
            self._state.difficulty = difficulty
            print(f"Difficulty set to: {difficulty}")
        else:
            print(f"Invalid difficulty. Valid options: {valid_difficulties}")
    
    def save_game(self, description: str = "", save_type: str = "manual") -> None:
        """Save current game state."""
        memento = GameStateMemento(self._state, description, save_type)
        self.caretaker.save_memento(memento)
    
    def load_game(self, save_index: int = None) -> bool:
        """Load game from save."""
        if save_index is None:
            # Load most recent save
            memento = self.caretaker.get_memento()
        else:
            memento = self.caretaker.get_memento(save_index)
        
        if memento:
            self.restore_memento(memento)
            return True
        else:
            print("No save found to load")
            return False
    
    def get_save_list(self) -> List[Dict[str, Any]]:
        """Get list of all saves."""
        return self.caretaker.get_history()


# ============================================================================
# CONFIGURATION MANAGER WITH MEMENTO
# ============================================================================

class ConfigurationMemento(Memento):
    """Memento for configuration state."""
    
    def __init__(self, config_data: Dict[str, Any], description: str = ""):
        self._config_data = copy.deepcopy(config_data)
        self._state_id = self._generate_state_id()
        self._timestamp = datetime.now()
        self._description = description
        self._metadata = {
            'description': description,
            'config_keys': list(config_data.keys()),
            'config_count': len(config_data),
            'data_size': len(json.dumps(config_data))
        }
    
    def _generate_state_id(self) -> str:
        """Generate state ID based on configuration."""
        config_str = json.dumps(self._config_data, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def get_state_id(self) -> str:
        return self._state_id
    
    def get_timestamp(self) -> datetime:
        return self._timestamp
    
    def get_metadata(self) -> Dict[str, Any]:
        return self._metadata.copy()
    
    def get_config_data(self) -> Dict[str, Any]:
        """Get configuration data."""
        return copy.deepcopy(self._config_data)


class ConfigurationManager(Originator):
    """Configuration manager with version control using memento pattern."""
    
    def __init__(self, config_name: str):
        self.config_name = config_name
        self._config: Dict[str, Any] = {}
        self.caretaker = Caretaker(max_mementos=30)
        self._change_listeners: List[Callable] = []
        
        # Default configuration
        self._config = {
            'database': {
                'host': 'localhost',
                'port': 5432,
                'name': 'myapp',
                'pool_size': 10
            },
            'cache': {
                'enabled': True,
                'ttl': 3600,
                'max_size': 1000
            },
            'logging': {
                'level': 'INFO',
                'file': 'app.log',
                'max_size': '10MB'
            },
            'features': {
                'feature_a': True,
                'feature_b': False,
                'feature_c': True
            }
        }
        
        # Save initial configuration
        self.save_configuration("Initial configuration")
    
    def create_memento(self, description: str = "") -> Memento:
        """Create memento of current configuration."""
        return ConfigurationMemento(self._config, description)
    
    def restore_memento(self, memento: Memento) -> None:
        """Restore configuration from memento."""
        if isinstance(memento, ConfigurationMemento):
            old_config = copy.deepcopy(self._config)
            self._config = memento.get_config_data()
            
            # Notify listeners of configuration change
            self._notify_change_listeners(old_config, self._config)
            
            print(f"Configuration restored: {memento.get_metadata()['description']}")
    
    def get_current_state_info(self) -> Dict[str, Any]:
        """Get current configuration information."""
        return {
            'config_name': self.config_name,
            'total_keys': len(self._config),
            'sections': list(self._config.keys()),
            'data_size': len(json.dumps(self._config)),
            'listeners': len(self._change_listeners)
        }
    
    def set_value(self, key_path: str, value: Any, description: str = "") -> None:
        """Set configuration value using dot notation."""
        keys = key_path.split('.')
        config = self._config
        
        # Navigate to parent
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set value
        old_value = config.get(keys[-1])
        config[keys[-1]] = value
        
        print(f"Configuration updated: {key_path} = {value} (was: {old_value})")
        
        # Save configuration change
        if not description:
            description = f"Updated {key_path} to {value}"
        self.save_configuration(description)
    
    def get_value(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key_path.split('.')
        config = self._config
        
        try:
            for key in keys:
                config = config[key]
            return config
        except (KeyError, TypeError):
            return default
    
    def delete_value(self, key_path: str, description: str = "") -> bool:
        """Delete configuration value."""
        keys = key_path.split('.')
        config = self._config
        
        try:
            # Navigate to parent
            for key in keys[:-1]:
                config = config[key]
            
            # Delete key
            if keys[-1] in config:
                del config[keys[-1]]
                
                print(f"Configuration deleted: {key_path}")
                
                if not description:
                    description = f"Deleted {key_path}"
                self.save_configuration(description)
                
                return True
        except (KeyError, TypeError):
            pass
        
        return False
    
    def merge_configuration(self, new_config: Dict[str, Any], description: str = "") -> None:
        """Merge new configuration with existing."""
        def deep_merge(target: Dict, source: Dict) -> None:
            for key, value in source.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    deep_merge(target[key], value)
                else:
                    target[key] = value
        
        deep_merge(self._config, new_config)
        
        if not description:
            description = f"Merged configuration with {len(new_config)} keys"
        self.save_configuration(description)
        
        print(f"Configuration merged: {description}")
    
    def reset_section(self, section: str, description: str = "") -> bool:
        """Reset configuration section to default."""
        defaults = {
            'database': {
                'host': 'localhost',
                'port': 5432,
                'name': 'myapp',
                'pool_size': 10
            },
            'cache': {
                'enabled': True,
                'ttl': 3600,
                'max_size': 1000
            },
            'logging': {
                'level': 'INFO',
                'file': 'app.log',
                'max_size': '10MB'
            },
            'features': {
                'feature_a': True,
                'feature_b': False,
                'feature_c': True
            }
        }
        
        if section in defaults:
            self._config[section] = copy.deepcopy(defaults[section])
            
            if not description:
                description = f"Reset {section} section to defaults"
            self.save_configuration(description)
            
            print(f"Section '{section}' reset to defaults")
            return True
        
        return False
    
    def save_configuration(self, description: str = "") -> None:
        """Save current configuration state."""
        memento = self.create_memento(description)
        self.caretaker.save_memento(memento)
    
    def rollback_configuration(self) -> bool:
        """Rollback to previous configuration."""
        memento = self.caretaker.undo()
        if memento:
            self.restore_memento(memento)
            return True
        return False
    
    def rollforward_configuration(self) -> bool:
        """Roll forward to next configuration."""
        memento = self.caretaker.redo()
        if memento:
            self.restore_memento(memento)
            return True
        return False
    
    def add_change_listener(self, listener: Callable[[Dict, Dict], None]) -> None:
        """Add configuration change listener."""
        self._change_listeners.append(listener)
    
    def remove_change_listener(self, listener: Callable) -> None:
        """Remove configuration change listener."""
        if listener in self._change_listeners:
            self._change_listeners.remove(listener)
    
    def _notify_change_listeners(self, old_config: Dict[str, Any], new_config: Dict[str, Any]) -> None:
        """Notify all change listeners."""
        for listener in self._change_listeners:
            try:
                listener(old_config, new_config)
            except Exception as e:
                print(f"Error in change listener: {e}")
    
    def get_configuration_history(self) -> List[Dict[str, Any]]:
        """Get configuration change history."""
        return self.caretaker.get_history()
    
    def export_configuration(self) -> str:
        """Export current configuration as JSON."""
        return json.dumps(self._config, indent=2)
    
    def import_configuration(self, config_json: str, description: str = "") -> bool:
        """Import configuration from JSON."""
        try:
            new_config = json.loads(config_json)
            self._config = new_config
            
            if not description:
                description = "Imported configuration from JSON"
            self.save_configuration(description)
            
            print("Configuration imported successfully")
            return True
        except json.JSONDecodeError as e:
            print(f"Failed to import configuration: {e}")
            return False


def demonstrate_memento_pattern():
    """
    Demonstrate Memento pattern implementations.
    """
    print("=== MEMENTO PATTERN DEMONSTRATION ===\n")
    
    # 1. Text Editor with Undo/Redo
    print("1. TEXT EDITOR WITH UNDO/REDO:")
    
    editor = TextEditor()
    
    print("\n   Text editing operations:")
    print("   " + "=" * 40)
    
    # Perform editing operations
    editor.insert_text("Hello World!")
    editor.save_checkpoint("Added greeting")
    
    editor.insert_text(" How are you?", len(editor.get_content()))
    editor.save_checkpoint("Added question")
    
    editor.set_selection(0, 5)  # Select "Hello"
    editor.delete_text(0, 5)
    editor.insert_text("Hi", 0)
    editor.save_checkpoint("Changed greeting")
    
    editor.set_font_size(16)
    editor.toggle_bold()
    editor.save_checkpoint("Changed formatting")
    
    print(f"\n   Current content: '{editor.get_content()}'")
    print(f"   Current state: {editor.get_current_state_info()}")
    
    # Test undo/redo
    print(f"\n   Testing undo/redo:")
    print(f"   Undo available: {editor.caretaker.can_undo()}")
    
    editor.undo()
    print(f"   After undo: '{editor.get_content()}'")
    
    editor.undo()
    print(f"   After undo: '{editor.get_content()}'")
    
    editor.redo()
    print(f"   After redo: '{editor.get_content()}'")
    
    # Show edit history
    history = editor.get_history()
    print(f"\n   Edit History ({len(history)} states):")
    for entry in history:
        marker = " -> " if entry['is_current'] else "    "
        print(f"   {marker}{entry['metadata']['description']} "
              f"(Length: {entry['metadata']['content_length']})")
    
    print()
    
    # 2. Game State Save/Load System
    print("2. GAME STATE SAVE/LOAD SYSTEM:")
    
    game = Game("Player1")
    
    print("\n   Game progression:")
    print("   " + "=" * 40)
    
    # Simulate game progression
    game.move_player(10, 5)
    game.gain_score(100)
    game.add_to_inventory("Health Potion")
    game.add_to_inventory("Magic Sword")
    game.save_game("Found magic sword", "manual")
    
    # Level progression
    game.gain_experience(150)  # Should level up
    game.move_player(25, 15)
    game.gain_score(250)
    game.unlock_achievement("First Level Up")
    
    # Combat simulation
    game.take_damage(30)
    game.use_mana(20)
    game.heal(15)
    game.save_game("After combat", "manual")
    
    # More progression
    game.gain_experience(200)  # Another level up
    game.set_difficulty("hard")
    game.unlock_achievement("Difficulty Increased")
    game.save_game("Level 3 on hard mode", "checkpoint")
    
    print(f"\n   Current game state: {game.get_current_state_info()}")
    
    # Show save list
    saves = game.get_save_list()
    print(f"\n   Save Games ({len(saves)} saves):")
    for save in saves:
        marker = " -> " if save['is_current'] else "    "
        metadata = save['metadata']
        print(f"   {marker}[{save['index']}] {metadata['description']} "
              f"(Level {metadata['level']}, Score: {metadata['score']})")
    
    # Test loading previous save
    print(f"\n   Loading previous save...")
    game.load_game(1)  # Load second save
    print(f"   After loading: {game.get_current_state_info()}")
    
    print()
    
    # 3. Configuration Manager with Version Control
    print("3. CONFIGURATION MANAGER WITH VERSION CONTROL:")
    
    config_manager = ConfigurationManager("AppConfig")
    
    # Add change listener
    def config_change_listener(old_config: Dict, new_config: Dict):
        print(f"   [LISTENER] Configuration changed")
    
    config_manager.add_change_listener(config_change_listener)
    
    print("\n   Configuration management:")
    print("   " + "=" * 40)
    
    # Show initial configuration
    print(f"   Initial config: {config_manager.get_current_state_info()}")
    
    # Make configuration changes
    config_manager.set_value("database.host", "production-db.example.com", "Updated database host")
    config_manager.set_value("database.port", 5433, "Changed database port")
    config_manager.set_value("cache.ttl", 7200, "Increased cache TTL")
    
    # Add new feature flag
    config_manager.set_value("features.feature_d", True, "Added new feature flag")
    
    # Merge new configuration
    new_config = {
        "api": {
            "version": "v2",
            "timeout": 30,
            "retries": 3
        },
        "features": {
            "feature_e": False
        }
    }
    config_manager.merge_configuration(new_config, "Added API configuration")
    
    # Show current values
    print(f"\n   Current database host: {config_manager.get_value('database.host')}")
    print(f"   Current cache TTL: {config_manager.get_value('cache.ttl')}")
    print(f"   API timeout: {config_manager.get_value('api.timeout')}")
    
    # Test rollback
    print(f"\n   Rolling back configuration...")
    config_manager.rollback_configuration()
    print(f"   Database host after rollback: {config_manager.get_value('database.host')}")
    
    # Roll forward
    print(f"   Rolling forward configuration...")
    config_manager.rollforward_configuration()
    print(f"   Database host after roll forward: {config_manager.get_value('database.host')}")
    
    # Show configuration history
    config_history = config_manager.get_configuration_history()
    print(f"\n   Configuration History ({len(config_history)} versions):")
    for entry in config_history:
        marker = " -> " if entry['is_current'] else "    "
        metadata = entry['metadata']
        print(f"   {marker}[{entry['index']}] {metadata['description']} "
              f"({metadata['config_count']} keys)")
    
    # Export configuration
    exported_config = config_manager.export_configuration()
    print(f"\n   Exported configuration size: {len(exported_config)} characters")
    
    print()
    
    # 4. Advanced Memento Features
    print("4. ADVANCED MEMENTO FEATURES:")
    
    # Create a caretaker with limited history
    limited_caretaker = Caretaker(max_mementos=3)
    
    # Create multiple mementos
    test_editor = TextEditor()
    test_editor.caretaker = limited_caretaker
    
    print("\n   Testing limited history caretaker:")
    print("   " + "=" * 40)
    
    # Add more mementos than the limit
    for i in range(5):
        test_editor.insert_text(f"Text {i+1} ")
        test_editor.save_checkpoint(f"Added text {i+1}")
    
    # Show history (should only have last 3)
    limited_history = test_editor.get_history()
    print(f"   Limited history ({len(limited_history)} entries, max 3):")
    for entry in limited_history:
        print(f"     {entry['metadata']['description']}")
    
    # Test caretaker statistics
    stats = limited_caretaker.get_statistics()
    print(f"\n   Caretaker statistics:")
    print(f"     Total mementos: {stats['total_mementos']}")
    print(f"     Current index: {stats['current_index']}")
    print(f"     Can undo: {stats['can_undo']}")
    print(f"     Can redo: {stats['can_redo']}")
    print(f"     Memory usage estimate: {stats['memory_usage_estimate']} bytes")
    
    print()
    
    # 5. Thread-Safe Memento Operations
    print("5. THREAD-SAFE MEMENTO OPERATIONS:")
    
    class ThreadSafeCaretaker(Caretaker):
        """Thread-safe version of caretaker."""
        
        def __init__(self, max_mementos: int = 100):
            super().__init__(max_mementos)
            self._lock = threading.Lock()
        
        def save_memento(self, memento: Memento) -> None:
            with self._lock:
                super().save_memento(memento)
        
        def undo(self) -> Optional[Memento]:
            with self._lock:
                return super().undo()
        
        def redo(self) -> Optional[Memento]:
            with self._lock:
                return super().redo()
    
    # Create thread-safe editor
    thread_safe_editor = TextEditor()
    thread_safe_editor.caretaker = ThreadSafeCaretaker()
    
    print("\n   Testing thread-safe operations:")
    print("   " + "=" * 40)
    
    # Simulate concurrent operations
    def concurrent_editing(editor, thread_id, operations):
        for i, text in enumerate(operations):
            editor.insert_text(f"[T{thread_id}] {text}")
            editor.save_checkpoint(f"Thread {thread_id} - Operation {i+1}")
            time.sleep(0.01)  # Small delay
    
    import threading
    
    # Create threads for concurrent editing
    thread1 = threading.Thread(
        target=concurrent_editing,
        args=(thread_safe_editor, 1, ["Hello", "World"])
    )
    
    thread2 = threading.Thread(
        target=concurrent_editing,
        args=(thread_safe_editor, 2, ["Foo", "Bar"])
    )
    
    # Start threads
    thread1.start()
    thread2.start()
    
    # Wait for completion
    thread1.join()
    thread2.join()
    
    print(f"   Final content: '{thread_safe_editor.get_content()}'")
    
    concurrent_history = thread_safe_editor.get_history()
    print(f"   Concurrent operations history ({len(concurrent_history)} entries):")
    for entry in concurrent_history[-4:]:  # Show last 4
        print(f"     {entry['metadata']['description']}")
    
    print()
    
    # 6. Memento Pattern Benefits
    print("6. MEMENTO PATTERN BENEFITS:")
    print("   ✓ Encapsulation: Internal state is captured without exposing implementation")
    print("   ✓ Undo/Redo Support: Easy implementation of undo and redo functionality")
    print("   ✓ State History: Complete history of state changes is maintained")
    print("   ✓ Rollback Capability: Can restore to any previous state")
    print("   ✓ Snapshot Creation: Point-in-time snapshots for backup and recovery")
    print("   ✓ Version Control: Track changes and manage different versions")
    print("   ✓ Debugging Aid: State history helps in debugging and analysis")
    print("   ✓ User Experience: Users can safely experiment knowing they can undo")
    print("   ✓ Data Integrity: Prevents accidental loss of important state")
    print("   ✓ Flexibility: Can be combined with other patterns for complex scenarios")
    print()
    
    print("=== MEMENTO PATTERN DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_memento_pattern()
