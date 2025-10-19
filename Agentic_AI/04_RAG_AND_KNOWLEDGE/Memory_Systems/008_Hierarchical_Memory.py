#!/usr/bin/env python3
"""
Hierarchical Memory: Multi-Level Memory Architecture for Intelligent Storage
==========================================================================

WHAT IS THE PROBLEM?
==================
AI systems need sophisticated memory architectures that can:
- Store information at different levels of detail and importance
- Enable efficient retrieval from immediate to long-term memory
- Provide automatic promotion and demotion of memory items
- Support different memory types with appropriate storage policies
- Scale from working memory to permanent knowledge storage
- Maintain performance across different temporal and spatial scales

Example: Flat Memory Architecture Problems
WITHOUT HIERARCHICAL MEMORY (Traditional):
- All memories stored at same level regardless of importance
- No distinction between temporary and permanent information
- Slow retrieval due to searching through all memory
- No automatic organization by recency or importance
- Memory conflicts between short-term and long-term needs
- Result: Poor performance, inefficient storage, confusing organization

REAL WORLD EXAMPLE:
=================
How does human hierarchical memory work?

HUMAN MEMORY HIERARCHY:
1. SENSORY MEMORY: Immediate processing (milliseconds)
   - Raw sensory input, automatic filtering
2. SHORT-TERM/WORKING MEMORY: Active processing (seconds to minutes)
   - Limited capacity (7±2 items), conscious manipulation
3. LONG-TERM MEMORY: Permanent storage (lifetime)
   - Unlimited capacity, organized by patterns and associations
4. EPISODIC MEMORY: Personal experiences with context
   - Specific events, spatial and temporal details
5. SEMANTIC MEMORY: General knowledge and facts
   - Abstracted from episodes, context-independent
6. PROCEDURAL MEMORY: Skills and habits
   - Automatic, unconscious execution

BENEFITS OF HIERARCHICAL MEMORY:
- Optimizes performance for different memory access patterns
- Enables natural promotion/demotion based on importance and usage
- Supports efficient retrieval through appropriate memory level
- Provides automatic organization and maintenance
- Scales from immediate processing to lifetime knowledge storage
- Matches human cognitive architecture for intuitive interaction

THE HIERARCHICAL ADVANTAGE:
==========================
FLAT: All memories equal → Poor performance, confused organization
HIERARCHICAL: Multi-level organization → Optimal access, natural structure

HIERARCHICAL MEMORY COMPONENTS:
==============================
1. SENSORY BUFFER: Immediate input processing and filtering
2. WORKING MEMORY: Active information manipulation with limited capacity
3. SHORT-TERM STORE: Recent information bridge to long-term memory
4. LONG-TERM STORE: Permanent organized knowledge storage
5. EPISODIC BUFFER: Specific experiences with rich contextual details
6. SEMANTIC NETWORK: Abstract knowledge and fact relationships
7. PROCEDURAL STORE: Skills, habits, and automatic behaviors

WHY THIS IS REVOLUTIONARY:
========================
- Enables AI systems to process information like human cognition
- Critical for agents that need both immediate and long-term memory
- Supports natural learning progression from experience to knowledge
- Optimizes storage and retrieval for different information types
- Foundation for human-like reasoning and decision making
- Enables efficient scaling from working memory to vast knowledge bases
"""

import asyncio
import time
import json
import uuid
import math
import random
from typing import Dict, List, Any, Optional, Tuple, Union, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta
import numpy as np
from contextlib import contextmanager
import hashlib
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class MemoryLevel(Enum):
    """Levels in the hierarchical memory system"""
    SENSORY = "sensory"          # Immediate input processing
    WORKING = "working"          # Active manipulation
    SHORT_TERM = "short_term"    # Recent information bridge
    LONG_TERM = "long_term"      # Permanent storage
    EPISODIC = "episodic"        # Experience memories
    SEMANTIC = "semantic"        # Knowledge and facts
    PROCEDURAL = "procedural"    # Skills and procedures

class MemoryType(Enum):
    """Types of memory content"""
    EXPERIENCE = "experience"    # Specific events and interactions
    KNOWLEDGE = "knowledge"      # Facts and information
    SKILL = "skill"             # Procedures and abilities
    PREFERENCE = "preference"    # User preferences and patterns
    CONTEXT = "context"         # Environmental and situational
    GOAL = "goal"               # Objectives and intentions
    PATTERN = "pattern"         # Recognized patterns and rules

class PromotionTrigger(Enum):
    """Triggers for memory promotion to higher levels"""
    FREQUENT_ACCESS = "frequent_access"      # Accessed many times
    HIGH_IMPORTANCE = "high_importance"      # Marked as important
    STRONG_ASSOCIATIONS = "strong_associations"  # Connected to other memories
    PATTERN_RECOGNITION = "pattern_recognition"  # Part of recognized pattern
    EXPLICIT_LEARNING = "explicit_learning"  # Deliberately learned
    EMOTIONAL_SIGNIFICANCE = "emotional_significance"  # Emotionally important
    TEMPORAL_PERSISTENCE = "temporal_persistence"  # Persists over time

@dataclass
class MemoryItem:
    """Represents an item in hierarchical memory"""
    
    id: str
    content: Dict[str, Any]
    memory_type: MemoryType
    current_level: MemoryLevel
    
    # Temporal information
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    
    # Hierarchy management
    importance_score: float = 1.0
    activation_level: float = 1.0  # Current activation strength
    base_activation: float = 1.0   # Base activation level
    
    # Associations and connections
    associations: Dict[str, float] = field(default_factory=dict)  # memory_id -> strength
    context_associations: Dict[str, float] = field(default_factory=dict)  # context -> strength
    
    # Level-specific metadata
    level_metadata: Dict[MemoryLevel, Dict[str, Any]] = field(default_factory=dict)
    
    # Promotion/demotion tracking
    promotion_history: List[Tuple[MemoryLevel, MemoryLevel, datetime, str]] = field(default_factory=list)
    promotion_resistance: float = 1.0  # Resistance to level changes
    
    # Quality and confidence
    confidence_score: float = 1.0
    coherence_score: float = 1.0    # How well it fits with other memories
    
    # Context and tags
    context_tags: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
    
    def update_access(self) -> None:
        """Update access tracking and activation"""
        self.access_count += 1
        self.last_accessed = datetime.now()
        
        # Boost activation on access
        self.activation_level = min(2.0, self.activation_level * 1.2)
    
    def calculate_promotion_score(self, target_level: MemoryLevel) -> float:
        """Calculate score for promotion to target level"""
        
        # Base factors
        frequency_factor = min(1.0, self.access_count / 10.0)
        recency_factor = self._calculate_recency_factor()
        importance_factor = self.importance_score
        activation_factor = self.activation_level / 2.0
        
        # Association strength
        association_factor = self._calculate_association_strength()
        
        # Level-specific factors
        level_factor = self._calculate_level_compatibility(target_level)
        
        # Confidence and coherence
        quality_factor = (self.confidence_score + self.coherence_score) / 2.0
        
        # Weighted combination based on target level
        if target_level == MemoryLevel.WORKING:
            score = (
                recency_factor * 0.4 +
                activation_factor * 0.3 +
                importance_factor * 0.2 +
                quality_factor * 0.1
            )
        elif target_level == MemoryLevel.SHORT_TERM:
            score = (
                frequency_factor * 0.3 +
                recency_factor * 0.3 +
                importance_factor * 0.2 +
                association_factor * 0.2
            )
        elif target_level in [MemoryLevel.LONG_TERM, MemoryLevel.SEMANTIC]:
            score = (
                frequency_factor * 0.2 +
                importance_factor * 0.3 +
                association_factor * 0.2 +
                quality_factor * 0.3
            )
        elif target_level == MemoryLevel.EPISODIC:
            score = (
                importance_factor * 0.4 +
                quality_factor * 0.3 +
                association_factor * 0.3
            )
        else:
            score = (
                frequency_factor * 0.25 +
                importance_factor * 0.25 +
                activation_factor * 0.25 +
                quality_factor * 0.25
            )
        
        # Apply resistance
        score = score / self.promotion_resistance
        
        return min(1.0, score)
    
    def _calculate_recency_factor(self) -> float:
        """Calculate recency factor for memory"""
        
        hours_since_access = (datetime.now() - self.last_accessed).total_seconds() / 3600
        
        # Exponential decay with different rates for different levels
        if self.current_level == MemoryLevel.WORKING:
            decay_rate = 1.0  # Fast decay for working memory
        elif self.current_level == MemoryLevel.SHORT_TERM:
            decay_rate = 0.1  # Medium decay for short-term
        else:
            decay_rate = 0.01  # Slow decay for long-term
        
        return math.exp(-decay_rate * hours_since_access / 24)
    
    def _calculate_association_strength(self) -> float:
        """Calculate overall association strength"""
        
        if not self.associations:
            return 0.0
        
        # Average of association strengths
        total_strength = sum(self.associations.values())
        return min(1.0, total_strength / len(self.associations))
    
    def _calculate_level_compatibility(self, target_level: MemoryLevel) -> float:
        """Calculate compatibility with target memory level"""
        
        # Content type compatibility
        type_compatibility = {
            MemoryLevel.SENSORY: {
                MemoryType.EXPERIENCE: 0.9,
                MemoryType.CONTEXT: 0.8,
                MemoryType.PATTERN: 0.3
            },
            MemoryLevel.WORKING: {
                MemoryType.EXPERIENCE: 0.8,
                MemoryType.GOAL: 0.9,
                MemoryType.CONTEXT: 0.7,
                MemoryType.KNOWLEDGE: 0.6
            },
            MemoryLevel.SHORT_TERM: {
                MemoryType.EXPERIENCE: 0.9,
                MemoryType.PREFERENCE: 0.8,
                MemoryType.KNOWLEDGE: 0.7
            },
            MemoryLevel.LONG_TERM: {
                MemoryType.KNOWLEDGE: 0.9,
                MemoryType.PATTERN: 0.8,
                MemoryType.PREFERENCE: 0.7
            },
            MemoryLevel.EPISODIC: {
                MemoryType.EXPERIENCE: 1.0,
                MemoryType.CONTEXT: 0.8
            },
            MemoryLevel.SEMANTIC: {
                MemoryType.KNOWLEDGE: 1.0,
                MemoryType.PATTERN: 0.9,
                MemoryType.PREFERENCE: 0.6
            },
            MemoryLevel.PROCEDURAL: {
                MemoryType.SKILL: 1.0,
                MemoryType.PATTERN: 0.8
            }
        }
        
        return type_compatibility.get(target_level, {}).get(self.memory_type, 0.5)
    
    def add_association(self, other_memory_id: str, strength: float) -> None:
        """Add association with another memory"""
        self.associations[other_memory_id] = strength
    
    def update_activation(self, base_decay: float = 0.95) -> None:
        """Update activation level with decay"""
        
        # Natural decay
        self.activation_level *= base_decay
        
        # Boost from associations
        association_boost = self._calculate_association_strength() * 0.1
        self.activation_level = min(2.0, self.activation_level + association_boost)
    
    def promote_to_level(self, new_level: MemoryLevel, reason: str) -> None:
        """Promote memory to new level"""
        
        old_level = self.current_level
        
        # Record promotion
        self.promotion_history.append((old_level, new_level, datetime.now(), reason))
        
        # Update level
        self.current_level = new_level
        
        # Adjust properties for new level
        self._adjust_for_level(new_level)
        
        # Increase promotion resistance (harder to change again soon)
        self.promotion_resistance = min(2.0, self.promotion_resistance * 1.2)
    
    def _adjust_for_level(self, level: MemoryLevel) -> None:
        """Adjust memory properties for specific level"""
        
        if level == MemoryLevel.WORKING:
            # Working memory: high activation, faster decay
            self.activation_level = min(2.0, self.activation_level * 1.5)
            
        elif level == MemoryLevel.LONG_TERM:
            # Long-term: stable activation, slower decay
            self.base_activation = max(0.5, self.base_activation)
            
        elif level == MemoryLevel.SEMANTIC:
            # Semantic: emphasize associations
            for assoc_id in list(self.associations.keys()):
                self.associations[assoc_id] = min(1.0, self.associations[assoc_id] * 1.2)
    
    def get_content_hash(self) -> str:
        """Get hash of content for similarity comparison"""
        content_str = json.dumps(self.content, sort_keys=True)
        return hashlib.md5(content_str.encode()).hexdigest()

class MemoryLevelManager(ABC):
    """Abstract base class for managing specific memory levels"""
    
    def __init__(self, level: MemoryLevel, capacity: int = 1000):
        self.level = level
        self.capacity = capacity
        self.memories: Dict[str, MemoryItem] = {}
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    @abstractmethod
    def can_accept(self, memory: MemoryItem) -> bool:
        """Check if this level can accept the memory"""
        pass
    
    @abstractmethod
    def get_eviction_candidates(self, count: int) -> List[str]:
        """Get candidates for eviction from this level"""
        pass
    
    @abstractmethod
    def update_level(self) -> None:
        """Update all memories in this level"""
        pass
    
    def add_memory(self, memory: MemoryItem) -> bool:
        """Add memory to this level"""
        
        if not self.can_accept(memory):
            return False
        
        # Check capacity
        if len(self.memories) >= self.capacity:
            # Evict least suitable memory
            candidates = self.get_eviction_candidates(1)
            if candidates:
                self.remove_memory(candidates[0])
        
        self.memories[memory.id] = memory
        memory.current_level = self.level
        
        return True
    
    def remove_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """Remove memory from this level"""
        
        return self.memories.pop(memory_id, None)
    
    def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """Get memory from this level"""
        
        if memory_id in self.memories:
            memory = self.memories[memory_id]
            memory.update_access()
            return memory
        
        return None

class SensoryMemoryManager(MemoryLevelManager):
    """Manages sensory memory - immediate input processing"""
    
    def __init__(self):
        super().__init__(MemoryLevel.SENSORY, capacity=50)
        self.retention_time = timedelta(seconds=30)  # Very short retention
    
    def can_accept(self, memory: MemoryItem) -> bool:
        """Sensory memory accepts immediate experiences and context"""
        
        return memory.memory_type in [MemoryType.EXPERIENCE, MemoryType.CONTEXT]
    
    def get_eviction_candidates(self, count: int) -> List[str]:
        """Evict oldest memories first"""
        
        sorted_memories = sorted(
            self.memories.values(),
            key=lambda m: m.created_at
        )
        
        return [m.id for m in sorted_memories[:count]]
    
    def update_level(self) -> None:
        """Remove expired sensory memories"""
        
        cutoff_time = datetime.now() - self.retention_time
        expired_ids = []
        
        for memory_id, memory in self.memories.items():
            if memory.created_at < cutoff_time:
                expired_ids.append(memory_id)
        
        for memory_id in expired_ids:
            self.remove_memory(memory_id)

class WorkingMemoryManager(MemoryLevelManager):
    """Manages working memory - active manipulation"""
    
    def __init__(self):
        super().__init__(MemoryLevel.WORKING, capacity=7)  # Miller's magic number
        self.activation_threshold = 0.3
    
    def can_accept(self, memory: MemoryItem) -> bool:
        """Working memory accepts currently relevant memories"""
        
        return (memory.activation_level > self.activation_threshold and
                memory.memory_type in [MemoryType.EXPERIENCE, MemoryType.GOAL, 
                                     MemoryType.CONTEXT, MemoryType.KNOWLEDGE])
    
    def get_eviction_candidates(self, count: int) -> List[str]:
        """Evict memories with lowest activation"""
        
        sorted_memories = sorted(
            self.memories.values(),
            key=lambda m: m.activation_level
        )
        
        return [m.id for m in sorted_memories[:count]]
    
    def update_level(self) -> None:
        """Update activation levels and evict low-activation memories"""
        
        for memory in self.memories.values():
            memory.update_activation(base_decay=0.8)  # Faster decay in working memory
        
        # Remove memories below threshold
        to_remove = []
        
        for memory_id, memory in self.memories.items():
            if memory.activation_level < self.activation_threshold:
                to_remove.append(memory_id)
        
        for memory_id in to_remove:
            self.remove_memory(memory_id)

class ShortTermMemoryManager(MemoryLevelManager):
    """Manages short-term memory - recent information bridge"""
    
    def __init__(self):
        super().__init__(MemoryLevel.SHORT_TERM, capacity=200)
        self.retention_time = timedelta(hours=24)  # 24-hour retention
    
    def can_accept(self, memory: MemoryItem) -> bool:
        """Short-term memory accepts recent and relevant memories"""
        
        age_hours = (datetime.now() - memory.created_at).total_seconds() / 3600
        
        return (age_hours < 48 and  # Recent enough
                memory.memory_type in [MemoryType.EXPERIENCE, MemoryType.PREFERENCE,
                                     MemoryType.KNOWLEDGE, MemoryType.CONTEXT])
    
    def get_eviction_candidates(self, count: int) -> List[str]:
        """Evict oldest and least accessed memories"""
        
        sorted_memories = sorted(
            self.memories.values(),
            key=lambda m: (m.access_count, m.last_accessed)
        )
        
        return [m.id for m in sorted_memories[:count]]
    
    def update_level(self) -> None:
        """Remove expired memories and update activations"""
        
        cutoff_time = datetime.now() - self.retention_time
        expired_ids = []
        
        for memory_id, memory in self.memories.items():
            memory.update_activation(base_decay=0.9)
            
            if memory.last_accessed < cutoff_time:
                expired_ids.append(memory_id)
        
        for memory_id in expired_ids:
            self.remove_memory(memory_id)

class LongTermMemoryManager(MemoryLevelManager):
    """Manages long-term memory - permanent storage"""
    
    def __init__(self):
        super().__init__(MemoryLevel.LONG_TERM, capacity=10000)
        self.consolidation_threshold = 5  # Minimum access count for consolidation
    
    def can_accept(self, memory: MemoryItem) -> bool:
        """Long-term memory accepts important and frequently accessed memories"""
        
        return (memory.access_count >= self.consolidation_threshold or
                memory.importance_score >= 0.8 or
                len(memory.associations) >= 3)  # Well-connected memories
    
    def get_eviction_candidates(self, count: int) -> List[str]:
        """Evict memories with lowest importance and associations"""
        
        sorted_memories = sorted(
            self.memories.values(),
            key=lambda m: (m.importance_score, len(m.associations), m.access_count)
        )
        
        return [m.id for m in sorted_memories[:count]]
    
    def update_level(self) -> None:
        """Slow decay for long-term memories"""
        
        for memory in self.memories.values():
            memory.update_activation(base_decay=0.95)  # Very slow decay

class EpisodicMemoryManager(MemoryLevelManager):
    """Manages episodic memory - specific experiences"""
    
    def __init__(self):
        super().__init__(MemoryLevel.EPISODIC, capacity=5000)
    
    def can_accept(self, memory: MemoryItem) -> bool:
        """Episodic memory accepts rich experiences with context"""
        
        return (memory.memory_type == MemoryType.EXPERIENCE and
                len(memory.context_tags) >= 2 and  # Rich context
                memory.confidence_score >= 0.6)
    
    def get_eviction_candidates(self, count: int) -> List[str]:
        """Evict memories with poor context and low importance"""
        
        sorted_memories = sorted(
            self.memories.values(),
            key=lambda m: (len(m.context_tags), m.importance_score, m.coherence_score)
        )
        
        return [m.id for m in sorted_memories[:count]]
    
    def update_level(self) -> None:
        """Update coherence scores based on associations"""
        
        for memory in self.memories.values():
            memory.update_activation(base_decay=0.93)
            
            # Update coherence based on associations
            if memory.associations:
                avg_association = sum(memory.associations.values()) / len(memory.associations)
                memory.coherence_score = (memory.coherence_score + avg_association) / 2

class SemanticMemoryManager(MemoryLevelManager):
    """Manages semantic memory - knowledge and facts"""
    
    def __init__(self):
        super().__init__(MemoryLevel.SEMANTIC, capacity=20000)
    
    def can_accept(self, memory: MemoryItem) -> bool:
        """Semantic memory accepts knowledge and patterns"""
        
        return (memory.memory_type in [MemoryType.KNOWLEDGE, MemoryType.PATTERN, MemoryType.PREFERENCE] and
                memory.confidence_score >= 0.7)
    
    def get_eviction_candidates(self, count: int) -> List[str]:
        """Evict least connected and lowest confidence memories"""
        
        sorted_memories = sorted(
            self.memories.values(),
            key=lambda m: (len(m.associations), m.confidence_score, m.access_count)
        )
        
        return [m.id for m in sorted_memories[:count]]
    
    def update_level(self) -> None:
        """Strengthen associations and update confidence"""
        
        for memory in self.memories.values():
            memory.update_activation(base_decay=0.97)  # Very slow decay
            
            # Strengthen well-connected memories
            if len(memory.associations) >= 5:
                memory.confidence_score = min(1.0, memory.confidence_score * 1.01)

class ProceduralMemoryManager(MemoryLevelManager):
    """Manages procedural memory - skills and procedures"""
    
    def __init__(self):
        super().__init__(MemoryLevel.PROCEDURAL, capacity=1000)
    
    def can_accept(self, memory: MemoryItem) -> bool:
        """Procedural memory accepts skills and procedures"""
        
        return (memory.memory_type in [MemoryType.SKILL, MemoryType.PATTERN] and
                memory.access_count >= 3)  # Practiced skills
    
    def get_eviction_candidates(self, count: int) -> List[str]:
        """Evict least practiced skills"""
        
        sorted_memories = sorted(
            self.memories.values(),
            key=lambda m: (m.access_count, m.importance_score)
        )
        
        return [m.id for m in sorted_memories[:count]]
    
    def update_level(self) -> None:
        """Skills improve with practice"""
        
        for memory in self.memories.values():
            memory.update_activation(base_decay=0.96)
            
            # Skills improve with practice
            if memory.access_count >= 10:
                memory.importance_score = min(1.0, memory.importance_score * 1.02)

class HierarchicalMemorySystem:
    """Complete hierarchical memory system"""
    
    def __init__(self):
        # Initialize memory level managers
        self.level_managers = {
            MemoryLevel.SENSORY: SensoryMemoryManager(),
            MemoryLevel.WORKING: WorkingMemoryManager(),
            MemoryLevel.SHORT_TERM: ShortTermMemoryManager(),
            MemoryLevel.LONG_TERM: LongTermMemoryManager(),
            MemoryLevel.EPISODIC: EpisodicMemoryManager(),
            MemoryLevel.SEMANTIC: SemanticMemoryManager(),
            MemoryLevel.PROCEDURAL: ProceduralMemoryManager()
        }
        
        # Memory promotion/demotion rules
        self.promotion_rules = self._initialize_promotion_rules()
        
        # Global memory index
        self.memory_index: Dict[str, MemoryLevel] = {}
        
        # Association network
        self.association_network: Dict[str, Set[str]] = defaultdict(set)
        
        # Statistics
        self.stats = {
            'memories_added': 0,
            'promotions': 0,
            'demotions': 0,
            'retrievals': 0,
            'associations_created': 0
        }
        
        # Background processing
        self.background_executor = ThreadPoolExecutor(max_workers=2)
        
        self.logger = logging.getLogger("HierarchicalMemorySystem")
    
    async def initialize(self) -> None:
        """Initialize the hierarchical memory system"""
        self.logger.info("Hierarchical memory system initialized")
    
    def _initialize_promotion_rules(self) -> Dict[MemoryLevel, List[Tuple[MemoryLevel, float]]]:
        """Initialize promotion rules between memory levels"""
        
        return {
            MemoryLevel.SENSORY: [
                (MemoryLevel.WORKING, 0.8),      # High activation -> working
                (MemoryLevel.SHORT_TERM, 0.6)    # Moderate activation -> short-term
            ],
            MemoryLevel.WORKING: [
                (MemoryLevel.SHORT_TERM, 0.7),   # Sustained activation -> short-term
                (MemoryLevel.LONG_TERM, 0.9)     # Very high importance -> long-term
            ],
            MemoryLevel.SHORT_TERM: [
                (MemoryLevel.LONG_TERM, 0.8),    # High consolidation score -> long-term
                (MemoryLevel.EPISODIC, 0.7),     # Rich experience -> episodic
                (MemoryLevel.SEMANTIC, 0.6)      # Knowledge content -> semantic
            ],
            MemoryLevel.LONG_TERM: [
                (MemoryLevel.EPISODIC, 0.8),     # Experience content -> episodic
                (MemoryLevel.SEMANTIC, 0.7),     # Abstract knowledge -> semantic
                (MemoryLevel.PROCEDURAL, 0.9)    # Skill content -> procedural
            ]
        }
    
    def add_memory(self, content: Dict[str, Any], memory_type: MemoryType,
                  importance: float = 1.0, context_tags: Set[str] = None,
                  initial_level: MemoryLevel = MemoryLevel.SENSORY) -> str:
        """Add a new memory to the system"""
        
        memory = MemoryItem(
            id="",
            content=content,
            memory_type=memory_type,
            current_level=initial_level,
            importance_score=importance,
            context_tags=context_tags or set()
        )
        
        # Add to appropriate level
        level_manager = self.level_managers[initial_level]
        
        if level_manager.add_memory(memory):
            self.memory_index[memory.id] = initial_level
            self.stats['memories_added'] += 1
            
            self.logger.debug(f"Added memory {memory.id[:8]}... to {initial_level.value}")
            
            return memory.id
        else:
            self.logger.warning(f"Failed to add memory to {initial_level.value}")
            return ""
    
    def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """Retrieve a memory from any level"""
        
        if memory_id not in self.memory_index:
            return None
        
        level = self.memory_index[memory_id]
        level_manager = self.level_managers[level]
        
        memory = level_manager.get_memory(memory_id)
        
        if memory:
            self.stats['retrievals'] += 1
            
            # Trigger promotion evaluation after access
            self._evaluate_promotion(memory)
        
        return memory
    
    def search_memories(self, query: str, memory_types: Set[MemoryType] = None,
                       levels: Set[MemoryLevel] = None, max_results: int = 10) -> List[MemoryItem]:
        """Search memories across levels and types"""
        
        results = []
        search_levels = levels or set(self.level_managers.keys())
        search_types = memory_types or set(MemoryType)
        
        for level in search_levels:
            level_manager = self.level_managers[level]
            
            for memory in level_manager.memories.values():
                if memory.memory_type not in search_types:
                    continue
                
                # Simple text matching (can be enhanced with semantic search)
                content_str = str(memory.content).lower()
                
                if query.lower() in content_str:
                    memory.update_access()
                    results.append(memory)
        
        # Sort by relevance (activation + importance + recency)
        results.sort(key=lambda m: (
            m.activation_level + m.importance_score + m._calculate_recency_factor()
        ), reverse=True)
        
        return results[:max_results]
    
    def create_association(self, memory_id1: str, memory_id2: str, strength: float = 1.0) -> bool:
        """Create association between two memories"""
        
        memory1 = self.get_memory(memory_id1)
        memory2 = self.get_memory(memory_id2)
        
        if not memory1 or not memory2:
            return False
        
        # Bidirectional association
        memory1.add_association(memory_id2, strength)
        memory2.add_association(memory_id1, strength)
        
        # Update global association network
        self.association_network[memory_id1].add(memory_id2)
        self.association_network[memory_id2].add(memory_id1)
        
        self.stats['associations_created'] += 1
        
        return True
    
    def _evaluate_promotion(self, memory: MemoryItem) -> None:
        """Evaluate if memory should be promoted to higher level"""
        
        current_level = memory.current_level
        
        if current_level not in self.promotion_rules:
            return
        
        promotion_candidates = self.promotion_rules[current_level]
        
        for target_level, threshold in promotion_candidates:
            promotion_score = memory.calculate_promotion_score(target_level)
            
            if promotion_score >= threshold:
                if self._attempt_promotion(memory, target_level):
                    break  # Stop after first successful promotion
    
    def _attempt_promotion(self, memory: MemoryItem, target_level: MemoryLevel) -> bool:
        """Attempt to promote memory to target level"""
        
        target_manager = self.level_managers[target_level]
        
        if not target_manager.can_accept(memory):
            return False
        
        # Remove from current level
        current_level = memory.current_level
        current_manager = self.level_managers[current_level]
        current_manager.remove_memory(memory.id)
        
        # Add to target level
        if target_manager.add_memory(memory):
            # Update index
            self.memory_index[memory.id] = target_level
            
            # Record promotion
            memory.promote_to_level(target_level, f"Promoted from {current_level.value}")
            
            self.stats['promotions'] += 1
            
            self.logger.debug(f"Promoted memory {memory.id[:8]}... from {current_level.value} to {target_level.value}")
            
            return True
        else:
            # Failed to add to target, restore to current level
            current_manager.add_memory(memory)
            return False
    
    async def update_system(self) -> Dict[str, Any]:
        """Update all memory levels and process promotions"""
        
        update_results = {}
        
        # Update each level
        for level, manager in self.level_managers.items():
            initial_count = len(manager.memories)
            manager.update_level()
            final_count = len(manager.memories)
            
            update_results[level.value] = {
                'initial_memories': initial_count,
                'final_memories': final_count,
                'memories_removed': initial_count - final_count
            }
        
        # Process promotions
        promotion_count = await self._process_background_promotions()
        update_results['promotions_processed'] = promotion_count
        
        # Update associations
        self._strengthen_associations()
        
        return update_results
    
    async def _process_background_promotions(self) -> int:
        """Process memory promotions in background"""
        
        promotion_count = 0
        
        # Check memories in lower levels for promotion opportunities
        for level in [MemoryLevel.SENSORY, MemoryLevel.WORKING, MemoryLevel.SHORT_TERM]:
            manager = self.level_managers[level]
            
            for memory in list(manager.memories.values()):
                # Skip recently promoted memories
                if (memory.promotion_history and 
                    (datetime.now() - memory.promotion_history[-1][2]).total_seconds() < 3600):
                    continue
                
                self._evaluate_promotion(memory)
                promotion_count += 1
        
        return promotion_count
    
    def _strengthen_associations(self) -> None:
        """Strengthen associations between co-accessed memories"""
        
        # Find memories accessed in similar time windows
        time_window = timedelta(hours=1)
        recent_accesses = defaultdict(list)
        
        for level_manager in self.level_managers.values():
            for memory in level_manager.memories.values():
                if memory.last_accessed:
                    time_key = memory.last_accessed.replace(minute=0, second=0, microsecond=0)
                    recent_accesses[time_key].append(memory.id)
        
        # Create associations between co-accessed memories
        for access_time, memory_ids in recent_accesses.items():
            if len(memory_ids) >= 2:
                for i, memory_id1 in enumerate(memory_ids):
                    for memory_id2 in memory_ids[i + 1:]:
                        # Strengthen existing association or create new one
                        memory1 = self.get_memory(memory_id1)
                        memory2 = self.get_memory(memory_id2)
                        
                        if memory1 and memory2:
                            current_strength = memory1.associations.get(memory_id2, 0.0)
                            new_strength = min(1.0, current_strength + 0.1)
                            
                            memory1.add_association(memory_id2, new_strength)
                            memory2.add_association(memory_id1, new_strength)
    
    def get_memory_by_level(self, level: MemoryLevel) -> List[MemoryItem]:
        """Get all memories from a specific level"""
        
        manager = self.level_managers[level]
        return list(manager.memories.values())
    
    def get_associated_memories(self, memory_id: str, strength_threshold: float = 0.5) -> List[MemoryItem]:
        """Get memories associated with given memory"""
        
        memory = self.get_memory(memory_id)
        
        if not memory:
            return []
        
        associated = []
        
        for assoc_id, strength in memory.associations.items():
            if strength >= strength_threshold:
                assoc_memory = self.get_memory(assoc_id)
                if assoc_memory:
                    associated.append(assoc_memory)
        
        # Sort by association strength
        associated.sort(key=lambda m: memory.associations.get(m.id, 0.0), reverse=True)
        
        return associated
    
    def analyze_memory_distribution(self) -> Dict[str, Any]:
        """Analyze memory distribution across levels and types"""
        
        distribution = {}
        
        for level, manager in self.level_managers.items():
            level_stats = {
                'total_memories': len(manager.memories),
                'capacity': manager.capacity,
                'utilization': len(manager.memories) / manager.capacity,
                'memory_types': defaultdict(int),
                'avg_activation': 0.0,
                'avg_importance': 0.0,
                'avg_associations': 0.0
            }
            
            if manager.memories:
                activations = []
                importances = []
                association_counts = []
                
                for memory in manager.memories.values():
                    level_stats['memory_types'][memory.memory_type.value] += 1
                    activations.append(memory.activation_level)
                    importances.append(memory.importance_score)
                    association_counts.append(len(memory.associations))
                
                level_stats['avg_activation'] = sum(activations) / len(activations)
                level_stats['avg_importance'] = sum(importances) / len(importances)
                level_stats['avg_associations'] = sum(association_counts) / len(association_counts)
            
            distribution[level.value] = level_stats
        
        return distribution
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        
        total_memories = sum(len(manager.memories) for manager in self.level_managers.values())
        total_associations = sum(len(network) for network in self.association_network.values()) // 2
        
        # Level utilization
        level_utilization = {}
        for level, manager in self.level_managers.items():
            level_utilization[level.value] = len(manager.memories) / manager.capacity
        
        # Memory flow statistics
        promotion_flow = defaultdict(int)
        for level_manager in self.level_managers.values():
            for memory in level_manager.memories.values():
                for old_level, new_level, timestamp, reason in memory.promotion_history:
                    promotion_flow[f"{old_level.value}_to_{new_level.value}"] += 1
        
        return {
            'overview': {
                'total_memories': total_memories,
                'total_associations': total_associations,
                'memories_added': self.stats['memories_added'],
                'promotions': self.stats['promotions'],
                'retrievals': self.stats['retrievals']
            },
            'level_utilization': level_utilization,
            'promotion_flow': dict(promotion_flow),
            'performance_metrics': self.stats
        }
    
    def consolidate_memories(self) -> Dict[str, int]:
        """Consolidate similar memories to optimize storage"""
        
        consolidation_results = {}
        
        for level, manager in self.level_managers.items():
            if level in [MemoryLevel.SENSORY, MemoryLevel.WORKING]:
                continue  # Don't consolidate highly dynamic levels
            
            consolidated_count = self._consolidate_level(manager)
            consolidation_results[level.value] = consolidated_count
        
        return consolidation_results
    
    def _consolidate_level(self, manager: MemoryLevelManager) -> int:
        """Consolidate similar memories within a level"""
        
        memories = list(manager.memories.values())
        consolidated_count = 0
        
        # Find similar memory groups
        similarity_threshold = 0.8
        processed_ids = set()
        
        for i, memory1 in enumerate(memories):
            if memory1.id in processed_ids:
                continue
            
            similar_group = [memory1]
            
            for memory2 in memories[i + 1:]:
                if memory2.id in processed_ids:
                    continue
                
                # Calculate similarity based on content hash and context
                if self._calculate_memory_similarity(memory1, memory2) >= similarity_threshold:
                    similar_group.append(memory2)
                    processed_ids.add(memory2.id)
            
            # Consolidate group if it has multiple members
            if len(similar_group) > 1:
                self._consolidate_memory_group(similar_group, manager)
                consolidated_count += len(similar_group) - 1  # Count merged memories
            
            processed_ids.add(memory1.id)
        
        return consolidated_count
    
    def _calculate_memory_similarity(self, memory1: MemoryItem, memory2: MemoryItem) -> float:
        """Calculate similarity between two memories"""
        
        # Content similarity
        if memory1.get_content_hash() == memory2.get_content_hash():
            content_sim = 1.0
        else:
            # Simple content comparison (can be enhanced)
            content1_str = str(memory1.content).lower()
            content2_str = str(memory2.content).lower()
            
            common_words = set(content1_str.split()) & set(content2_str.split())
            total_words = set(content1_str.split()) | set(content2_str.split())
            
            content_sim = len(common_words) / len(total_words) if total_words else 0.0
        
        # Context similarity
        if memory1.context_tags and memory2.context_tags:
            common_tags = memory1.context_tags & memory2.context_tags
            total_tags = memory1.context_tags | memory2.context_tags
            context_sim = len(common_tags) / len(total_tags)
        else:
            context_sim = 0.5  # Neutral if one has no context
        
        # Type similarity
        type_sim = 1.0 if memory1.memory_type == memory2.memory_type else 0.0
        
        # Combined similarity
        similarity = (content_sim * 0.5 + context_sim * 0.3 + type_sim * 0.2)
        
        return similarity
    
    def _consolidate_memory_group(self, memory_group: List[MemoryItem], 
                                manager: MemoryLevelManager) -> None:
        """Consolidate a group of similar memories"""
        
        if len(memory_group) <= 1:
            return
        
        # Sort by importance and access count to find the best representative
        memory_group.sort(key=lambda m: (m.importance_score, m.access_count), reverse=True)
        
        primary_memory = memory_group[0]
        memories_to_merge = memory_group[1:]
        
        # Merge properties into primary memory
        for memory in memories_to_merge:
            # Combine access counts
            primary_memory.access_count += memory.access_count
            
            # Take maximum activation and importance
            primary_memory.activation_level = max(primary_memory.activation_level, memory.activation_level)
            primary_memory.importance_score = max(primary_memory.importance_score, memory.importance_score)
            
            # Merge associations
            for assoc_id, strength in memory.associations.items():
                current_strength = primary_memory.associations.get(assoc_id, 0.0)
                primary_memory.associations[assoc_id] = max(current_strength, strength)
            
            # Merge context tags
            primary_memory.context_tags.update(memory.context_tags)
            
            # Remove merged memory
            manager.remove_memory(memory.id)
            if memory.id in self.memory_index:
                del self.memory_index[memory.id]

# EXAMPLE DEMONSTRATIONS
# =====================

async def demo_basic_hierarchical_memory():
    """Demo: Basic hierarchical memory operations"""
    print("\nDEMO 1: BASIC HIERARCHICAL MEMORY")
    print("=" * 50)
    
    system = HierarchicalMemorySystem()
    await system.initialize()
    
    print("Adding memories to different levels:")
    
    # Add immediate sensory input
    sensory_id = system.add_memory(
        {"input": "user_says_hello", "timestamp": "now", "modality": "text"},
        MemoryType.EXPERIENCE,
        importance=0.6,
        context_tags={"conversation", "greeting"},
        initial_level=MemoryLevel.SENSORY
    )
    print(f"  Added sensory memory: {sensory_id[:8]}...")
    
    # Add working memory item (current task)
    working_id = system.add_memory(
        {"task": "help_user_with_coding", "status": "active", "priority": "high"},
        MemoryType.GOAL,
        importance=0.9,
        context_tags={"task", "coding", "active"},
        initial_level=MemoryLevel.WORKING
    )
    print(f"  Added working memory: {working_id[:8]}...")
    
    # Add knowledge to semantic memory
    semantic_id = system.add_memory(
        {"concept": "python_list", "definition": "ordered collection of items", "methods": ["append", "pop", "index"]},
        MemoryType.KNOWLEDGE,
        importance=0.8,
        context_tags={"programming", "python", "data_structures"},
        initial_level=MemoryLevel.SEMANTIC
    )
    print(f"  Added semantic memory: {semantic_id[:8]}...")
    
    # Add personal experience
    episodic_id = system.add_memory(
        {"event": "user_learned_loops", "context": "yesterday", "outcome": "successful", "difficulty": "medium"},
        MemoryType.EXPERIENCE,
        importance=0.7,
        context_tags={"learning", "loops", "programming", "yesterday"},
        initial_level=MemoryLevel.EPISODIC
    )
    print(f"  Added episodic memory: {episodic_id[:8]}...")
    
    # Create associations between related memories
    print(f"\nCreating associations:")
    
    system.create_association(working_id, semantic_id, 0.8)  # Current task relates to knowledge
    print(f"  Associated working and semantic memories")
    
    system.create_association(semantic_id, episodic_id, 0.6)  # Knowledge relates to past experience
    print(f"  Associated semantic and episodic memories")
    
    # Retrieve and access memories
    print(f"\nAccessing memories:")
    
    for memory_id in [sensory_id, working_id, semantic_id, episodic_id]:
        memory = system.get_memory(memory_id)
        if memory:
            print(f"  {memory_id[:8]}... - Level: {memory.current_level.value}, Activation: {memory.activation_level:.2f}")
    
    # Show memory distribution
    print(f"\nMemory distribution:")
    distribution = system.analyze_memory_distribution()
    
    for level, stats in distribution.items():
        if stats['total_memories'] > 0:
            print(f"  {level}: {stats['total_memories']} memories (utilization: {stats['utilization']:.1%})")

async def demo_memory_promotion():
    """Demo: Memory promotion between levels"""
    print("\nDEMO 2: MEMORY PROMOTION")
    print("=" * 50)
    
    system = HierarchicalMemorySystem()
    await system.initialize()
    
    print("Simulating memory promotion through repeated access:")
    
    # Add a memory to sensory level
    memory_id = system.add_memory(
        {"user_preference": "coffee_order", "preference": "latte_with_oat_milk", "time": "morning"},
        MemoryType.PREFERENCE,
        importance=0.5,
        context_tags={"coffee", "morning", "preference"},
        initial_level=MemoryLevel.SENSORY
    )
    
    print(f"  Initial memory level: {system.get_memory(memory_id).current_level.value}")
    
    # Simulate repeated access to increase importance
    print(f"\nSimulating repeated access over time:")
    
    for day in range(7):
        for access in range(3):  # Access 3 times per day
            memory = system.get_memory(memory_id)
            
            if memory:
                # Increase importance with repetition
                memory.importance_score = min(1.0, memory.importance_score + 0.05)
                
                print(f"    Day {day + 1}, Access {access + 1}: Level={memory.current_level.value}, "
                      f"Activation={memory.activation_level:.2f}, Importance={memory.importance_score:.2f}")
                
                # Check for promotion
                await system.update_system()
                
                # Memory might have been promoted
                updated_memory = system.get_memory(memory_id)
                if updated_memory and updated_memory.current_level != memory.current_level:
                    print(f"      -> PROMOTED to {updated_memory.current_level.value}!")
        
        print()
    
    # Show final state and promotion history
    final_memory = system.get_memory(memory_id)
    if final_memory:
        print(f"Final memory state:")
        print(f"  Level: {final_memory.current_level.value}")
        print(f"  Access count: {final_memory.access_count}")
        print(f"  Importance: {final_memory.importance_score:.2f}")
        print(f"  Activation: {final_memory.activation_level:.2f}")
        
        print(f"\nPromotion history:")
        for old_level, new_level, timestamp, reason in final_memory.promotion_history:
            print(f"  {old_level.value} -> {new_level.value}: {reason}")

async def demo_association_network():
    """Demo: Building association networks between memories"""
    print("\nDEMO 3: ASSOCIATION NETWORK")
    print("=" * 50)
    
    system = HierarchicalMemorySystem()
    await system.initialize()
    
    print("Building a network of related programming memories:")
    
    # Add related programming concepts
    concepts = [
        ("python_variables", "Variables store data values", {"programming", "python", "basics"}),
        ("python_functions", "Functions are reusable code blocks", {"programming", "python", "functions"}),
        ("python_loops", "Loops repeat code execution", {"programming", "python", "control_flow"}),
        ("python_lists", "Lists store ordered collections", {"programming", "python", "data_structures"}),
        ("debugging_tips", "Use print statements and debugger", {"programming", "debugging", "best_practices"}),
        ("code_organization", "Structure code in modules and classes", {"programming", "organization", "best_practices"}),
    ]
    
    memory_ids = {}
    
    for concept, description, tags in concepts:
        memory_id = system.add_memory(
            {"concept": concept, "description": description, "domain": "programming"},
            MemoryType.KNOWLEDGE,
            importance=0.7,
            context_tags=tags,
            initial_level=MemoryLevel.SEMANTIC
        )
        memory_ids[concept] = memory_id
        print(f"  Added: {concept}")
    
    # Create logical associations
    print(f"\nCreating associations based on relationships:")
    
    associations = [
        ("python_variables", "python_functions", 0.8, "Functions use variables"),
        ("python_functions", "python_loops", 0.7, "Functions often contain loops"),
        ("python_loops", "python_lists", 0.9, "Loops commonly iterate over lists"),
        ("python_functions", "code_organization", 0.8, "Functions are part of code organization"),
        ("debugging_tips", "python_functions", 0.6, "Functions need debugging"),
        ("debugging_tips", "python_loops", 0.7, "Loops often need debugging"),
        ("code_organization", "debugging_tips", 0.5, "Good organization aids debugging"),
    ]
    
    for concept1, concept2, strength, reason in associations:
        id1 = memory_ids[concept1]
        id2 = memory_ids[concept2]
        
        system.create_association(id1, id2, strength)
        print(f"  {concept1} <-> {concept2} (strength: {strength:.1f}) - {reason}")
    
    # Demonstrate associative retrieval
    print(f"\nDemonstrating associative retrieval:")
    
    # Start with one concept and find related ones
    start_concept = "python_functions"
    start_id = memory_ids[start_concept]
    
    associated_memories = system.get_associated_memories(start_id, strength_threshold=0.6)
    
    print(f"  Memories associated with '{start_concept}':")
    for memory in associated_memories:
        concept_name = memory.content.get("concept", "unknown")
        strength = system.get_memory(start_id).associations.get(memory.id, 0.0)
        print(f"    {concept_name} (strength: {strength:.1f})")
    
    # Show network effect through multiple accesses
    print(f"\nSimulating learning session - accessing related concepts:")
    
    learning_sequence = ["python_variables", "python_functions", "python_loops", "python_lists"]
    
    for concept in learning_sequence:
        memory_id = memory_ids[concept]
        memory = system.get_memory(memory_id)
        
        # Access the memory multiple times
        for _ in range(2):
            system.get_memory(memory_id)
        
        print(f"  Studied: {concept} (activation: {memory.activation_level:.2f})")
    
    # Update system to strengthen co-accessed associations
    await system.update_system()
    
    print(f"\nAssociation strengths after learning session:")
    
    for concept1, concept2, _, _ in associations:
        id1 = memory_ids[concept1]
        memory1 = system.get_memory(id1)
        
        if id2 := memory_ids[concept2]:
            strength = memory1.associations.get(id2, 0.0)
            print(f"  {concept1} <-> {concept2}: {strength:.2f}")

async def demo_memory_consolidation():
    """Demo: Memory consolidation and optimization"""
    print("\nDEMO 4: MEMORY CONSOLIDATION")
    print("=" * 50)
    
    system = HierarchicalMemorySystem()
    await system.initialize()
    
    print("Adding similar memories for consolidation:")
    
    # Add multiple similar memories (simulating repeated similar experiences)
    coffee_orders = [
        {"order": "latte", "size": "medium", "milk": "oat", "time": "morning"},
        {"order": "latte", "size": "medium", "milk": "oat_milk", "time": "morning"},  # Slightly different
        {"order": "medium_latte", "milk_type": "oat", "time": "morning"},  # Different structure
        {"order": "latte", "size": "medium", "milk": "oat", "time": "9am"},  # Different time format
        {"order": "cappuccino", "size": "small", "milk": "regular", "time": "afternoon"},  # Different order
    ]
    
    memory_ids = []
    
    for i, order in enumerate(coffee_orders):
        memory_id = system.add_memory(
            order,
            MemoryType.PREFERENCE,
            importance=0.6,
            context_tags={"coffee", "order", "preference"},
            initial_level=MemoryLevel.SHORT_TERM
        )
        memory_ids.append(memory_id)
        print(f"  Added coffee order {i + 1}: {order}")
    
    print(f"\nBefore consolidation:")
    distribution = system.analyze_memory_distribution()
    short_term_count = distribution['short_term']['total_memories']
    print(f"  Short-term memories: {short_term_count}")
    
    # Perform consolidation
    print(f"\nPerforming memory consolidation:")
    
    consolidation_results = system.consolidate_memories()
    
    for level, consolidated_count in consolidation_results.items():
        if consolidated_count > 0:
            print(f"  {level}: consolidated {consolidated_count} memories")
    
    print(f"\nAfter consolidation:")
    distribution = system.analyze_memory_distribution()
    new_short_term_count = distribution['short_term']['total_memories']
    print(f"  Short-term memories: {new_short_term_count}")
    print(f"  Memories consolidated: {short_term_count - new_short_term_count}")
    
    # Show remaining memories
    print(f"\nRemaining memories in short-term:")
    short_term_memories = system.get_memory_by_level(MemoryLevel.SHORT_TERM)
    
    for memory in short_term_memories:
        if memory.memory_type == MemoryType.PREFERENCE:
            print(f"  {memory.content}")
            print(f"    Access count: {memory.access_count}")
            print(f"    Importance: {memory.importance_score:.2f}")
            print(f"    Associations: {len(memory.associations)}")
            print()

async def demo_complete_memory_lifecycle():
    """Demo: Complete memory lifecycle from sensory to long-term"""
    print("\nDEMO 5: COMPLETE MEMORY LIFECYCLE")
    print("=" * 50)
    
    system = HierarchicalMemorySystem()
    await system.initialize()
    
    print("Simulating complete memory lifecycle:")
    
    # Phase 1: Sensory input
    print(f"\nPhase 1: Sensory input")
    
    memory_id = system.add_memory(
        {"event": "user_asks_about_recursion", "content": "How does recursion work in programming?", 
         "timestamp": datetime.now().isoformat()},
        MemoryType.EXPERIENCE,
        importance=0.4,  # Start with low importance
        context_tags={"programming", "question", "recursion"},
        initial_level=MemoryLevel.SENSORY
    )
    
    memory = system.get_memory(memory_id)
    print(f"  Created sensory memory: Level={memory.current_level.value}, Activation={memory.activation_level:.2f}")
    
    # Phase 2: Move to working memory through attention
    print(f"\nPhase 2: Attention moves to working memory")
    
    # Simulate attention by increasing activation
    for _ in range(3):
        memory = system.get_memory(memory_id)
        memory.importance_score = min(1.0, memory.importance_score + 0.2)
    
    await system.update_system()
    
    memory = system.get_memory(memory_id)
    print(f"  After attention: Level={memory.current_level.value}, Importance={memory.importance_score:.2f}")
    
    # Phase 3: Sustained attention promotes to short-term
    print(f"\nPhase 3: Sustained attention and processing")
    
    # Simulate processing by repeated access and adding context
    for i in range(5):
        memory = system.get_memory(memory_id)
        
        # Add more context as we process the question
        if i == 2:
            memory.context_tags.add("problem_solving")
        if i == 4:
            memory.context_tags.add("teaching_opportunity")
        
        await system.update_system()
        
        print(f"    Processing step {i + 1}: Level={memory.current_level.value}, "
              f"Activation={memory.activation_level:.2f}")
    
    # Phase 4: Connect to existing knowledge
    print(f"\nPhase 4: Connecting to existing knowledge")
    
    # Add related knowledge
    related_knowledge_id = system.add_memory(
        {"concept": "recursion", "definition": "function that calls itself", 
         "examples": ["factorial", "fibonacci"], "applications": ["tree_traversal"]},
        MemoryType.KNOWLEDGE,
        importance=0.8,
        context_tags={"programming", "recursion", "concepts"},
        initial_level=MemoryLevel.SEMANTIC
    )
    
    # Create association
    system.create_association(memory_id, related_knowledge_id, 0.9)
    print(f"  Connected to existing knowledge with strength 0.9")
    
    # Phase 5: Consolidation to long-term memory
    print(f"\nPhase 5: Consolidation to long-term")
    
    # Simulate importance through successful teaching
    memory = system.get_memory(memory_id)
    memory.importance_score = 0.9
    memory.context_tags.add("successful_teaching")
    
    # Multiple updates to trigger promotion
    for _ in range(3):
        await system.update_system()
    
    memory = system.get_memory(memory_id)
    print(f"  Final state: Level={memory.current_level.value}, Importance={memory.importance_score:.2f}")
    
    # Phase 6: Show complete promotion history
    print(f"\nPhase 6: Complete promotion history")
    
    for i, (old_level, new_level, timestamp, reason) in enumerate(memory.promotion_history, 1):
        print(f"  {i}. {old_level.value} -> {new_level.value} ({reason})")
    
    # Show final system state
    print(f"\nFinal system state:")
    stats = system.get_system_statistics()
    
    print(f"  Total memories: {stats['overview']['total_memories']}")
    print(f"  Total promotions: {stats['overview']['promotions']}")
    print(f"  Total associations: {stats['overview']['total_associations']}")
    
    print(f"\nLevel utilization:")
    for level, utilization in stats['level_utilization'].items():
        if utilization > 0:
            print(f"  {level}: {utilization:.1%}")

async def main():
    """
    Demonstrate Hierarchical Memory for multi-level intelligent storage
    
    WHAT YOU'LL LEARN:
    ================
    1. How to implement multi-level memory hierarchies like human cognition
    2. How to design automatic promotion and demotion between memory levels
    3. How to build association networks that strengthen with co-access
    4. How to create memory consolidation that merges similar experiences
    5. How to manage complete memory lifecycles from sensory to long-term
    6. How to optimize memory systems for different temporal and access patterns
    
    REAL WORLD APPLICATIONS:
    =======================
    - AI assistants with human-like memory organization and recall
    - Educational systems that track learning progression naturally
    - Personal AI that remembers user interactions at appropriate detail levels
    - Gaming AI with realistic memory and learning behaviors
    - Research assistants that organize information hierarchically
    - Conversational AI with natural memory formation and retrieval
    """
    
    print("HIERARCHICAL MEMORY DEMONSTRATION")
    print("Multi-level memory architecture for intelligent storage!")
    
    await demo_basic_hierarchical_memory()
    await demo_memory_promotion()
    await demo_association_network()
    await demo_memory_consolidation()
    await demo_complete_memory_lifecycle()
    
    print("\nWHAT WE LEARNED:")
    print("=" * 40)
    print("✓ Hierarchical memory optimizes storage and retrieval for different information types")
    print("✓ Automatic promotion creates natural learning progression from experience to knowledge")
    print("✓ Association networks strengthen through co-access and related experiences")
    print("✓ Memory consolidation optimizes storage by merging similar experiences")
    print("✓ Complete lifecycle management mimics human cognitive memory processes")
    print("✓ Multi-level architecture scales from immediate processing to lifetime knowledge")
    print("\nTHE POWER OF HIERARCHICAL MEMORY:")
    print("- Enables AI systems to process information like human cognition")
    print("- Supports natural learning progression from experience to knowledge")
    print("- Optimizes storage and retrieval for different information types")
    print("- Creates intuitive and human-like memory organization")
    print("- Essential for AI systems that need both immediate and long-term memory")

if __name__ == "__main__":
    asyncio.run(main())
