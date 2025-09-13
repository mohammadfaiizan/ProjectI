"""
FLYWEIGHT PATTERN - Structural Design Pattern
=============================================

Problem Statement:
Implement the Flyweight pattern to minimize memory usage when working with
large numbers of similar objects:
- Share common state among multiple objects (intrinsic state)
- Store unique state externally (extrinsic state)
- Reduce memory footprint for object-heavy applications
- Implement flyweight factories for object management
- Handle large collections efficiently

Learning Objectives:
- Understand intrinsic vs extrinsic state separation
- Implement memory-efficient object sharing
- Design flyweight factories and pools
- Handle large-scale object collections
- Optimize memory usage in graphics and gaming applications
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Set
import weakref
import sys
from enum import Enum
import random
import time


# ============================================================================
# FLYWEIGHT INTERFACE AND CONTEXT
# ============================================================================

class Flyweight(ABC):
    """Abstract flyweight interface."""
    
    @abstractmethod
    def operation(self, extrinsic_state: Dict[str, Any]) -> str:
        """Perform operation using intrinsic and extrinsic state."""
        pass
    
    @abstractmethod
    def get_intrinsic_state(self) -> Dict[str, Any]:
        """Get the intrinsic (shared) state."""
        pass


class Context:
    """Context that holds extrinsic state and references to flyweights."""
    
    def __init__(self, flyweight: Flyweight, extrinsic_state: Dict[str, Any]):
        self.flyweight = flyweight
        self.extrinsic_state = extrinsic_state
    
    def operation(self) -> str:
        """Delegate operation to flyweight with extrinsic state."""
        return self.flyweight.operation(self.extrinsic_state)
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get memory usage information."""
        return {
            'flyweight_id': id(self.flyweight),
            'context_id': id(self),
            'extrinsic_state_size': sys.getsizeof(self.extrinsic_state),
            'flyweight_shared': True
        }


# ============================================================================
# CHARACTER SYSTEM FLYWEIGHTS
# ============================================================================

class CharacterType(Enum):
    WARRIOR = "warrior"
    MAGE = "mage"
    ARCHER = "archer"
    ROGUE = "rogue"


class CharacterFlyweight(Flyweight):
    """Flyweight for game characters sharing common properties."""
    
    def __init__(self, character_type: CharacterType, sprite_data: str, 
                 base_stats: Dict[str, int]):
        # Intrinsic state (shared among all characters of this type)
        self.character_type = character_type
        self.sprite_data = sprite_data  # Shared sprite/texture data
        self.base_stats = base_stats.copy()  # Base health, mana, etc.
        self.abilities = self._get_abilities_for_type(character_type)
        
        print(f"CharacterFlyweight created for {character_type.value}")
    
    def _get_abilities_for_type(self, char_type: CharacterType) -> List[str]:
        """Get abilities based on character type."""
        abilities_map = {
            CharacterType.WARRIOR: ["Sword Strike", "Shield Block", "Charge"],
            CharacterType.MAGE: ["Fireball", "Heal", "Teleport"],
            CharacterType.ARCHER: ["Arrow Shot", "Multi-Shot", "Eagle Eye"],
            CharacterType.ROGUE: ["Backstab", "Stealth", "Poison Blade"]
        }
        return abilities_map.get(char_type, [])
    
    def operation(self, extrinsic_state: Dict[str, Any]) -> str:
        """Perform character operation using both intrinsic and extrinsic state."""
        # Extrinsic state: position, current health, level, equipment, etc.
        position = extrinsic_state.get('position', (0, 0))
        current_health = extrinsic_state.get('current_health', self.base_stats['health'])
        level = extrinsic_state.get('level', 1)
        equipment = extrinsic_state.get('equipment', [])
        
        # Calculate effective stats (intrinsic + extrinsic)
        effective_health = self.base_stats['health'] + (level - 1) * 10
        effective_damage = self.base_stats['damage'] + len(equipment) * 5
        
        return (f"{self.character_type.value.title()} at {position} - "
                f"Level {level}, Health: {current_health}/{effective_health}, "
                f"Damage: {effective_damage}, Equipment: {len(equipment)} items")
    
    def get_intrinsic_state(self) -> Dict[str, Any]:
        """Get intrinsic (shared) state."""
        return {
            'character_type': self.character_type.value,
            'sprite_data_size': len(self.sprite_data),
            'base_stats': self.base_stats,
            'abilities': self.abilities
        }
    
    def render(self, position: Tuple[int, int], scale: float = 1.0) -> str:
        """Render character at specific position (using extrinsic state)."""
        return f"Rendering {self.character_type.value} sprite at {position} with scale {scale}"
    
    def get_ability_damage(self, ability_name: str, character_level: int) -> int:
        """Calculate ability damage based on intrinsic ability and extrinsic level."""
        if ability_name not in self.abilities:
            return 0
        
        base_damage = self.base_stats['damage']
        ability_multiplier = 1.0 + (self.abilities.index(ability_name) * 0.2)
        level_bonus = character_level * 2
        
        return int(base_damage * ability_multiplier + level_bonus)


class GameCharacter:
    """Context class that uses character flyweight."""
    
    def __init__(self, character_id: str, flyweight: CharacterFlyweight, 
                 position: Tuple[int, int], level: int = 1):
        self.character_id = character_id
        self.flyweight = flyweight
        
        # Extrinsic state (unique to each character instance)
        self.position = position
        self.level = level
        self.current_health = flyweight.base_stats['health']
        self.current_mana = flyweight.base_stats.get('mana', 0)
        self.equipment = []
        self.experience = 0
        self.status_effects = []
    
    def move_to(self, new_position: Tuple[int, int]) -> None:
        """Move character to new position."""
        self.position = new_position
    
    def add_equipment(self, item: str) -> None:
        """Add equipment to character."""
        self.equipment.append(item)
    
    def take_damage(self, damage: int) -> None:
        """Apply damage to character."""
        self.current_health = max(0, self.current_health - damage)
    
    def heal(self, amount: int) -> None:
        """Heal character."""
        max_health = self.flyweight.base_stats['health'] + (self.level - 1) * 10
        self.current_health = min(max_health, self.current_health + amount)
    
    def get_status(self) -> str:
        """Get character status using flyweight."""
        extrinsic_state = {
            'position': self.position,
            'current_health': self.current_health,
            'level': self.level,
            'equipment': self.equipment
        }
        return self.flyweight.operation(extrinsic_state)
    
    def render(self) -> str:
        """Render character using flyweight."""
        scale = 1.0 + (self.level - 1) * 0.1  # Scale based on level
        return self.flyweight.render(self.position, scale)
    
    def use_ability(self, ability_name: str) -> str:
        """Use character ability."""
        damage = self.flyweight.get_ability_damage(ability_name, self.level)
        return f"{self.character_id} uses {ability_name} for {damage} damage!"


# ============================================================================
# TEXT FORMATTING FLYWEIGHTS
# ============================================================================

class FontStyle(Enum):
    NORMAL = "normal"
    BOLD = "bold"
    ITALIC = "italic"
    UNDERLINE = "underline"


class TextFormatFlyweight(Flyweight):
    """Flyweight for text formatting properties."""
    
    def __init__(self, font_family: str, font_size: int, color: str, style: FontStyle):
        # Intrinsic state (shared formatting properties)
        self.font_family = font_family
        self.font_size = font_size
        self.color = color
        self.style = style
        
        print(f"TextFormatFlyweight created: {font_family} {font_size}pt {color} {style.value}")
    
    def operation(self, extrinsic_state: Dict[str, Any]) -> str:
        """Format text using intrinsic formatting and extrinsic content."""
        text_content = extrinsic_state.get('text', '')
        position = extrinsic_state.get('position', (0, 0))
        
        # Apply formatting
        formatted_text = text_content
        if self.style == FontStyle.BOLD:
            formatted_text = f"**{formatted_text}**"
        elif self.style == FontStyle.ITALIC:
            formatted_text = f"*{formatted_text}*"
        elif self.style == FontStyle.UNDERLINE:
            formatted_text = f"_{formatted_text}_"
        
        return (f"<text font='{self.font_family}' size='{self.font_size}' "
                f"color='{self.color}' x='{position[0]}' y='{position[1]}'>"
                f"{formatted_text}</text>")
    
    def get_intrinsic_state(self) -> Dict[str, Any]:
        """Get intrinsic formatting state."""
        return {
            'font_family': self.font_family,
            'font_size': self.font_size,
            'color': self.color,
            'style': self.style.value
        }
    
    def get_css_style(self) -> str:
        """Get CSS representation of the format."""
        style_map = {
            FontStyle.NORMAL: 'normal',
            FontStyle.BOLD: 'bold',
            FontStyle.ITALIC: 'italic',
            FontStyle.UNDERLINE: 'underline'
        }
        
        css = f"font-family: {self.font_family}; "
        css += f"font-size: {self.font_size}pt; "
        css += f"color: {self.color}; "
        
        if self.style == FontStyle.BOLD:
            css += "font-weight: bold; "
        elif self.style == FontStyle.ITALIC:
            css += "font-style: italic; "
        elif self.style == FontStyle.UNDERLINE:
            css += "text-decoration: underline; "
        
        return css


class FormattedText:
    """Context for formatted text using flyweight."""
    
    def __init__(self, text: str, format_flyweight: TextFormatFlyweight, 
                 position: Tuple[int, int]):
        self.text = text
        self.format_flyweight = format_flyweight
        self.position = position
    
    def render(self) -> str:
        """Render formatted text."""
        extrinsic_state = {
            'text': self.text,
            'position': self.position
        }
        return self.format_flyweight.operation(extrinsic_state)
    
    def get_css(self) -> str:
        """Get CSS for this formatted text."""
        return self.format_flyweight.get_css_style()
    
    def change_text(self, new_text: str) -> None:
        """Change text content (extrinsic state)."""
        self.text = new_text
    
    def move_to(self, new_position: Tuple[int, int]) -> None:
        """Move text to new position (extrinsic state)."""
        self.position = new_position


# ============================================================================
# TREE RENDERING FLYWEIGHTS
# ============================================================================

class TreeType(Enum):
    OAK = "oak"
    PINE = "pine"
    BIRCH = "birch"
    MAPLE = "maple"


class TreeFlyweight(Flyweight):
    """Flyweight for tree rendering in a forest."""
    
    def __init__(self, tree_type: TreeType, sprite_data: str, color_palette: List[str]):
        # Intrinsic state (shared among all trees of this type)
        self.tree_type = tree_type
        self.sprite_data = sprite_data  # Large sprite/texture data
        self.color_palette = color_palette
        self.base_size = self._get_base_size(tree_type)
        
        print(f"TreeFlyweight created for {tree_type.value} tree")
    
    def _get_base_size(self, tree_type: TreeType) -> Tuple[int, int]:
        """Get base size for tree type."""
        size_map = {
            TreeType.OAK: (80, 120),
            TreeType.PINE: (60, 150),
            TreeType.BIRCH: (50, 100),
            TreeType.MAPLE: (70, 110)
        }
        return size_map.get(tree_type, (60, 100))
    
    def operation(self, extrinsic_state: Dict[str, Any]) -> str:
        """Render tree using intrinsic type data and extrinsic position/size."""
        position = extrinsic_state.get('position', (0, 0))
        scale = extrinsic_state.get('scale', 1.0)
        season = extrinsic_state.get('season', 'summer')
        
        # Calculate actual size based on scale
        actual_width = int(self.base_size[0] * scale)
        actual_height = int(self.base_size[1] * scale)
        
        # Choose color based on season
        color = self._get_seasonal_color(season)
        
        return (f"Rendering {self.tree_type.value} tree at {position} - "
                f"Size: {actual_width}x{actual_height}, Color: {color}, Season: {season}")
    
    def _get_seasonal_color(self, season: str) -> str:
        """Get tree color based on season."""
        seasonal_colors = {
            'spring': self.color_palette[0] if self.color_palette else 'light_green',
            'summer': self.color_palette[1] if len(self.color_palette) > 1 else 'green',
            'autumn': self.color_palette[2] if len(self.color_palette) > 2 else 'orange',
            'winter': self.color_palette[3] if len(self.color_palette) > 3 else 'brown'
        }
        return seasonal_colors.get(season, 'green')
    
    def get_intrinsic_state(self) -> Dict[str, Any]:
        """Get intrinsic tree state."""
        return {
            'tree_type': self.tree_type.value,
            'sprite_data_size': len(self.sprite_data),
            'color_palette': self.color_palette,
            'base_size': self.base_size
        }
    
    def get_memory_footprint(self) -> int:
        """Get approximate memory footprint of the flyweight."""
        return (sys.getsizeof(self.sprite_data) + 
                sys.getsizeof(self.color_palette) + 
                sys.getsizeof(self.base_size) + 
                64)  # Approximate overhead


class Tree:
    """Context for individual tree instances."""
    
    def __init__(self, tree_id: str, flyweight: TreeFlyweight, 
                 position: Tuple[int, int], scale: float = 1.0):
        self.tree_id = tree_id
        self.flyweight = flyweight
        
        # Extrinsic state (unique to each tree instance)
        self.position = position
        self.scale = scale
        self.age = random.randint(1, 100)  # Tree age affects appearance
        self.health = random.uniform(0.5, 1.0)  # Tree health
    
    def render(self, season: str = 'summer') -> str:
        """Render tree for specific season."""
        # Adjust scale based on age and health
        effective_scale = self.scale * (0.5 + self.age / 200) * self.health
        
        extrinsic_state = {
            'position': self.position,
            'scale': effective_scale,
            'season': season
        }
        return self.flyweight.operation(extrinsic_state)
    
    def get_bounds(self) -> Tuple[int, int, int, int]:
        """Get tree bounding box."""
        base_size = self.flyweight.base_size
        width = int(base_size[0] * self.scale)
        height = int(base_size[1] * self.scale)
        
        return (self.position[0], self.position[1], 
                self.position[0] + width, self.position[1] + height)
    
    def grow(self, growth_factor: float = 0.1) -> None:
        """Make tree grow (change extrinsic state)."""
        self.scale += growth_factor
        self.age += 1


# ============================================================================
# FLYWEIGHT FACTORIES
# ============================================================================

class CharacterFlyweightFactory:
    """Factory for managing character flyweights."""
    
    def __init__(self):
        self._flyweights: Dict[CharacterType, CharacterFlyweight] = {}
        self._creation_count = 0
    
    def get_character_flyweight(self, character_type: CharacterType) -> CharacterFlyweight:
        """Get or create character flyweight."""
        if character_type not in self._flyweights:
            # Create sprite data (simulated large data)
            sprite_data = f"SPRITE_DATA_FOR_{character_type.value.upper()}_" + "X" * 1000
            
            # Define base stats for each character type
            base_stats_map = {
                CharacterType.WARRIOR: {'health': 100, 'mana': 20, 'damage': 15},
                CharacterType.MAGE: {'health': 60, 'mana': 100, 'damage': 25},
                CharacterType.ARCHER: {'health': 80, 'mana': 40, 'damage': 20},
                CharacterType.ROGUE: {'health': 70, 'mana': 30, 'damage': 22}
            }
            
            base_stats = base_stats_map.get(character_type, {'health': 50, 'mana': 50, 'damage': 10})
            
            self._flyweights[character_type] = CharacterFlyweight(
                character_type, sprite_data, base_stats
            )
            self._creation_count += 1
        
        return self._flyweights[character_type]
    
    def get_flyweight_count(self) -> int:
        """Get number of flyweight instances created."""
        return len(self._flyweights)
    
    def get_total_memory_usage(self) -> int:
        """Get approximate total memory usage of all flyweights."""
        total = 0
        for flyweight in self._flyweights.values():
            total += sys.getsizeof(flyweight.sprite_data)
            total += sys.getsizeof(flyweight.base_stats)
            total += sys.getsizeof(flyweight.abilities)
            total += 64  # Approximate object overhead
        return total
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get factory statistics."""
        return {
            'flyweight_types': len(self._flyweights),
            'total_creations': self._creation_count,
            'memory_usage_bytes': self.get_total_memory_usage(),
            'available_types': [t.value for t in self._flyweights.keys()]
        }


class TextFormatFlyweightFactory:
    """Factory for managing text format flyweights."""
    
    def __init__(self):
        self._flyweights: Dict[str, TextFormatFlyweight] = {}
    
    def get_format_flyweight(self, font_family: str, font_size: int, 
                           color: str, style: FontStyle) -> TextFormatFlyweight:
        """Get or create text format flyweight."""
        # Create unique key for this combination
        key = f"{font_family}_{font_size}_{color}_{style.value}"
        
        if key not in self._flyweights:
            self._flyweights[key] = TextFormatFlyweight(font_family, font_size, color, style)
        
        return self._flyweights[key]
    
    def get_flyweight_count(self) -> int:
        """Get number of format flyweight instances."""
        return len(self._flyweights)
    
    def get_common_formats(self) -> List[str]:
        """Get list of commonly used formats."""
        return list(self._flyweights.keys())


class TreeFlyweightFactory:
    """Factory for managing tree flyweights."""
    
    def __init__(self):
        self._flyweights: Dict[TreeType, TreeFlyweight] = {}
    
    def get_tree_flyweight(self, tree_type: TreeType) -> TreeFlyweight:
        """Get or create tree flyweight."""
        if tree_type not in self._flyweights:
            # Create large sprite data (simulated)
            sprite_data = f"TREE_SPRITE_{tree_type.value.upper()}_" + "X" * 5000
            
            # Define color palettes for each tree type
            color_palettes = {
                TreeType.OAK: ['light_green', 'dark_green', 'orange', 'brown'],
                TreeType.PINE: ['dark_green', 'forest_green', 'dark_green', 'dark_brown'],
                TreeType.BIRCH: ['light_green', 'green', 'yellow', 'white'],
                TreeType.MAPLE: ['light_green', 'green', 'red', 'brown']
            }
            
            color_palette = color_palettes.get(tree_type, ['green', 'green', 'brown', 'brown'])
            
            self._flyweights[tree_type] = TreeFlyweight(tree_type, sprite_data, color_palette)
        
        return self._flyweights[tree_type]
    
    def get_flyweight_count(self) -> int:
        """Get number of tree flyweight instances."""
        return len(self._flyweights)
    
    def get_total_sprite_memory(self) -> int:
        """Get total memory used by sprite data."""
        total = 0
        for flyweight in self._flyweights.values():
            total += flyweight.get_memory_footprint()
        return total


# ============================================================================
# FLYWEIGHT MANAGER AND DEMO APPLICATIONS
# ============================================================================

class GameWorld:
    """Game world that manages many characters using flyweights."""
    
    def __init__(self):
        self.character_factory = CharacterFlyweightFactory()
        self.characters: List[GameCharacter] = []
        self.world_size = (1000, 1000)
    
    def spawn_character(self, character_type: CharacterType, position: Tuple[int, int], 
                       level: int = 1) -> str:
        """Spawn a new character in the world."""
        flyweight = self.character_factory.get_character_flyweight(character_type)
        character_id = f"{character_type.value}_{len(self.characters) + 1}"
        
        character = GameCharacter(character_id, flyweight, position, level)
        self.characters.append(character)
        
        return character_id
    
    def spawn_army(self, character_type: CharacterType, count: int) -> List[str]:
        """Spawn an army of characters."""
        army = []
        for i in range(count):
            x = random.randint(0, self.world_size[0])
            y = random.randint(0, self.world_size[1])
            level = random.randint(1, 10)
            
            character_id = self.spawn_character(character_type, (x, y), level)
            army.append(character_id)
        
        return army
    
    def get_character_by_id(self, character_id: str) -> Optional[GameCharacter]:
        """Get character by ID."""
        for character in self.characters:
            if character.character_id == character_id:
                return character
        return None
    
    def render_all_characters(self) -> List[str]:
        """Render all characters in the world."""
        rendered = []
        for character in self.characters:
            rendered.append(character.render())
        return rendered
    
    def get_world_statistics(self) -> Dict[str, Any]:
        """Get world statistics including memory usage."""
        factory_stats = self.character_factory.get_statistics()
        
        # Calculate context memory usage
        context_memory = 0
        for character in self.characters:
            context_memory += sys.getsizeof(character)
            context_memory += sys.getsizeof(character.equipment)
            context_memory += sys.getsizeof(character.status_effects)
        
        return {
            'total_characters': len(self.characters),
            'flyweight_types': factory_stats['flyweight_types'],
            'flyweight_memory_bytes': factory_stats['memory_usage_bytes'],
            'context_memory_bytes': context_memory,
            'total_memory_bytes': factory_stats['memory_usage_bytes'] + context_memory,
            'memory_saved_ratio': self._calculate_memory_savings()
        }
    
    def _calculate_memory_savings(self) -> float:
        """Calculate memory savings from using flyweight pattern."""
        if not self.characters:
            return 0.0
        
        # Memory with flyweight pattern
        flyweight_memory = self.character_factory.get_total_memory_usage()
        context_memory = len(self.characters) * 200  # Approximate context size
        total_with_flyweight = flyweight_memory + context_memory
        
        # Memory without flyweight pattern (each character has its own sprite data)
        memory_without_flyweight = len(self.characters) * 1500  # Approximate full object size
        
        if memory_without_flyweight == 0:
            return 0.0
        
        savings_ratio = (memory_without_flyweight - total_with_flyweight) / memory_without_flyweight
        return max(0.0, savings_ratio)


class DocumentEditor:
    """Document editor using text format flyweights."""
    
    def __init__(self):
        self.format_factory = TextFormatFlyweightFactory()
        self.formatted_texts: List[FormattedText] = []
    
    def add_text(self, text: str, font_family: str, font_size: int, 
                color: str, style: FontStyle, position: Tuple[int, int]) -> None:
        """Add formatted text to document."""
        format_flyweight = self.format_factory.get_format_flyweight(
            font_family, font_size, color, style
        )
        
        formatted_text = FormattedText(text, format_flyweight, position)
        self.formatted_texts.append(formatted_text)
    
    def render_document(self) -> List[str]:
        """Render entire document."""
        rendered = []
        for formatted_text in self.formatted_texts:
            rendered.append(formatted_text.render())
        return rendered
    
    def get_document_statistics(self) -> Dict[str, Any]:
        """Get document statistics."""
        return {
            'total_text_elements': len(self.formatted_texts),
            'unique_formats': self.format_factory.get_flyweight_count(),
            'common_formats': self.format_factory.get_common_formats()
        }


class Forest:
    """Forest simulation using tree flyweights."""
    
    def __init__(self, width: int, height: int):
        self.tree_factory = TreeFlyweightFactory()
        self.trees: List[Tree] = []
        self.forest_size = (width, height)
        self.current_season = 'summer'
    
    def plant_tree(self, tree_type: TreeType, position: Tuple[int, int], 
                  scale: float = 1.0) -> str:
        """Plant a single tree."""
        flyweight = self.tree_factory.get_tree_flyweight(tree_type)
        tree_id = f"{tree_type.value}_{len(self.trees) + 1}"
        
        tree = Tree(tree_id, flyweight, position, scale)
        self.trees.append(tree)
        
        return tree_id
    
    def plant_forest(self, tree_counts: Dict[TreeType, int]) -> Dict[TreeType, List[str]]:
        """Plant multiple trees to create a forest."""
        planted_trees = {}
        
        for tree_type, count in tree_counts.items():
            planted_trees[tree_type] = []
            
            for _ in range(count):
                x = random.randint(0, self.forest_size[0])
                y = random.randint(0, self.forest_size[1])
                scale = random.uniform(0.5, 1.5)
                
                tree_id = self.plant_tree(tree_type, (x, y), scale)
                planted_trees[tree_type].append(tree_id)
        
        return planted_trees
    
    def change_season(self, new_season: str) -> None:
        """Change forest season."""
        self.current_season = new_season
        print(f"Forest season changed to {new_season}")
    
    def render_forest(self) -> List[str]:
        """Render entire forest."""
        rendered = []
        for tree in self.trees:
            rendered.append(tree.render(self.current_season))
        return rendered
    
    def simulate_growth(self) -> None:
        """Simulate tree growth over time."""
        for tree in self.trees:
            if random.random() < 0.1:  # 10% chance of growth
                tree.grow(random.uniform(0.05, 0.15))
    
    def get_forest_statistics(self) -> Dict[str, Any]:
        """Get forest statistics."""
        tree_type_counts = {}
        for tree in self.trees:
            tree_type = tree.flyweight.tree_type
            tree_type_counts[tree_type.value] = tree_type_counts.get(tree_type.value, 0) + 1
        
        return {
            'total_trees': len(self.trees),
            'tree_types': tree_type_counts,
            'flyweight_instances': self.tree_factory.get_flyweight_count(),
            'sprite_memory_bytes': self.tree_factory.get_total_sprite_memory(),
            'current_season': self.current_season,
            'forest_size': self.forest_size
        }


def demonstrate_flyweight_pattern():
    """
    Demonstrate Flyweight pattern implementations.
    """
    print("=== FLYWEIGHT PATTERN DEMONSTRATION ===\n")
    
    # 1. Character System with Flyweights
    print("1. CHARACTER SYSTEM WITH FLYWEIGHTS:")
    
    game_world = GameWorld()
    
    # Spawn individual characters
    warrior_id = game_world.spawn_character(CharacterType.WARRIOR, (100, 200), 5)
    mage_id = game_world.spawn_character(CharacterType.MAGE, (150, 250), 3)
    
    print(f"   Spawned individual characters: {warrior_id}, {mage_id}")
    
    # Spawn armies (many characters of same type)
    print("\n   Spawning armies...")
    warrior_army = game_world.spawn_army(CharacterType.WARRIOR, 50)
    mage_army = game_world.spawn_army(CharacterType.MAGE, 30)
    archer_army = game_world.spawn_army(CharacterType.ARCHER, 40)
    rogue_army = game_world.spawn_army(CharacterType.ROGUE, 20)
    
    print(f"   Warrior army: {len(warrior_army)} characters")
    print(f"   Mage army: {len(mage_army)} characters")
    print(f"   Archer army: {len(archer_army)} characters")
    print(f"   Rogue army: {len(rogue_army)} characters")
    
    # Show character operations
    warrior = game_world.get_character_by_id(warrior_id)
    if warrior:
        print(f"\n   Warrior status: {warrior.get_status()}")
        print(f"   Warrior ability: {warrior.use_ability('Sword Strike')}")
        
        # Modify extrinsic state
        warrior.add_equipment("Magic Sword")
        warrior.add_equipment("Steel Armor")
        print(f"   After equipment: {warrior.get_status()}")
    
    # Show memory statistics
    world_stats = game_world.get_world_statistics()
    print(f"\n   World Statistics:")
    print(f"     Total characters: {world_stats['total_characters']}")
    print(f"     Flyweight types: {world_stats['flyweight_types']}")
    print(f"     Flyweight memory: {world_stats['flyweight_memory_bytes']} bytes")
    print(f"     Context memory: {world_stats['context_memory_bytes']} bytes")
    print(f"     Total memory: {world_stats['total_memory_bytes']} bytes")
    print(f"     Memory saved: {world_stats['memory_saved_ratio']:.1%}")
    
    print()
    
    # 2. Text Formatting System
    print("2. TEXT FORMATTING SYSTEM:")
    
    editor = DocumentEditor()
    
    # Add various formatted texts
    editor.add_text("Title", "Arial", 24, "black", FontStyle.BOLD, (10, 10))
    editor.add_text("Subtitle", "Arial", 18, "gray", FontStyle.ITALIC, (10, 40))
    editor.add_text("Body text paragraph 1", "Times", 12, "black", FontStyle.NORMAL, (10, 70))
    editor.add_text("Body text paragraph 2", "Times", 12, "black", FontStyle.NORMAL, (10, 90))
    editor.add_text("Important note", "Arial", 14, "red", FontStyle.BOLD, (10, 120))
    editor.add_text("Footer", "Arial", 10, "gray", FontStyle.ITALIC, (10, 150))
    
    # Add more text with same formatting (reuses flyweights)
    for i in range(10):
        editor.add_text(f"Paragraph {i+3}", "Times", 12, "black", FontStyle.NORMAL, (10, 170 + i*20))
    
    print("   Document created with multiple formatted texts")
    
    # Show document statistics
    doc_stats = editor.get_document_statistics()
    print(f"   Total text elements: {doc_stats['total_text_elements']}")
    print(f"   Unique formats (flyweights): {doc_stats['unique_formats']}")
    print(f"   Format reuse ratio: {1 - doc_stats['unique_formats']/doc_stats['total_text_elements']:.1%}")
    
    # Show some rendered text
    rendered_texts = editor.render_document()
    print(f"\n   Sample rendered texts:")
    for i, text in enumerate(rendered_texts[:3]):
        print(f"     {i+1}. {text}")
    print(f"     ... and {len(rendered_texts)-3} more")
    
    print()
    
    # 3. Forest Simulation
    print("3. FOREST SIMULATION:")
    
    forest = Forest(1000, 800)
    
    # Plant a diverse forest
    tree_counts = {
        TreeType.OAK: 100,
        TreeType.PINE: 150,
        TreeType.BIRCH: 80,
        TreeType.MAPLE: 70
    }
    
    print("   Planting forest...")
    planted = forest.plant_forest(tree_counts)
    
    for tree_type, tree_ids in planted.items():
        print(f"   {tree_type.value.title()} trees: {len(tree_ids)}")
    
    # Show forest statistics
    forest_stats = forest.get_forest_statistics()
    print(f"\n   Forest Statistics:")
    print(f"     Total trees: {forest_stats['total_trees']}")
    print(f"     Tree types: {forest_stats['tree_types']}")
    print(f"     Flyweight instances: {forest_stats['flyweight_instances']}")
    print(f"     Sprite memory: {forest_stats['sprite_memory_bytes']} bytes")
    print(f"     Current season: {forest_stats['current_season']}")
    
    # Demonstrate seasonal changes
    print(f"\n   Demonstrating seasonal changes:")
    seasons = ['spring', 'summer', 'autumn', 'winter']
    
    for season in seasons:
        forest.change_season(season)
        # Render a few trees to show seasonal differences
        rendered_trees = forest.render_forest()
        print(f"     {season.title()}: {rendered_trees[0]}")
    
    # Simulate growth
    print(f"\n   Simulating forest growth...")
    forest.simulate_growth()
    print(f"   Growth simulation completed")
    
    print()
    
    # 4. Memory Usage Comparison
    print("4. MEMORY USAGE COMPARISON:")
    
    # Calculate memory usage with and without flyweight pattern
    def calculate_memory_without_flyweight(num_objects: int, object_size: int) -> int:
        """Calculate memory usage without flyweight pattern."""
        return num_objects * object_size
    
    def calculate_memory_with_flyweight(num_objects: int, num_flyweights: int, 
                                      flyweight_size: int, context_size: int) -> int:
        """Calculate memory usage with flyweight pattern."""
        return (num_flyweights * flyweight_size) + (num_objects * context_size)
    
    # Character system comparison
    char_count = world_stats['total_characters']
    char_flyweights = world_stats['flyweight_types']
    
    without_flyweight = calculate_memory_without_flyweight(char_count, 1500)  # Full character object
    with_flyweight = world_stats['total_memory_bytes']
    
    print(f"   Character System Memory Comparison:")
    print(f"     Characters: {char_count}")
    print(f"     Without flyweight: {without_flyweight:,} bytes")
    print(f"     With flyweight: {with_flyweight:,} bytes")
    print(f"     Memory saved: {without_flyweight - with_flyweight:,} bytes ({(1-with_flyweight/without_flyweight):.1%})")
    
    # Forest system comparison
    tree_count = forest_stats['total_trees']
    tree_flyweights = forest_stats['flyweight_instances']
    
    without_flyweight_trees = calculate_memory_without_flyweight(tree_count, 5500)  # Full tree object
    with_flyweight_trees = forest_stats['sprite_memory_bytes'] + (tree_count * 100)  # Context size
    
    print(f"\n   Forest System Memory Comparison:")
    print(f"     Trees: {tree_count}")
    print(f"     Without flyweight: {without_flyweight_trees:,} bytes")
    print(f"     With flyweight: {with_flyweight_trees:,} bytes")
    print(f"     Memory saved: {without_flyweight_trees - with_flyweight_trees:,} bytes ({(1-with_flyweight_trees/without_flyweight_trees):.1%})")
    
    print()
    
    # 5. Flyweight Pattern Benefits and Trade-offs
    print("5. FLYWEIGHT PATTERN BENEFITS AND TRADE-OFFS:")
    
    print("   Benefits:")
    print("   ✓ Memory Efficiency: Significant memory savings with many similar objects")
    print("   ✓ Performance: Reduced object creation overhead")
    print("   ✓ Scalability: Can handle large numbers of objects efficiently")
    print("   ✓ Centralized Management: Flyweight factories provide centralized control")
    print("   ✓ Consistency: Shared intrinsic state ensures consistency")
    
    print("\n   Trade-offs:")
    print("   ⚠ Complexity: Requires careful separation of intrinsic/extrinsic state")
    print("   ⚠ Context Management: Extrinsic state must be managed by clients")
    print("   ⚠ Method Parameters: Operations may require many parameters")
    print("   ⚠ Immutability: Flyweights should be immutable for safe sharing")
    print("   ⚠ Factory Overhead: Additional complexity in factory management")
    
    print("\n   Best Use Cases:")
    print("   • Graphics/Gaming: Sprites, textures, particle systems")
    print("   • Text Processing: Character formatting, fonts")
    print("   • UI Components: Icons, buttons with shared styling")
    print("   • Simulation: Large numbers of similar entities")
    print("   • Data Visualization: Chart elements, graph nodes")
    
    print()
    
    print("=== FLYWEIGHT PATTERN DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_flyweight_pattern()
