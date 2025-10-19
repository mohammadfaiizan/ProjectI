#!/usr/bin/env python3
"""
Memory Pruning System: Intelligent Forgetting for Optimal Performance
====================================================================

WHAT IS THE PROBLEM?
==================
AI systems accumulate vast amounts of information over time but lack mechanisms to:
- Remove irrelevant, outdated, or contradictory information
- Forget details while preserving important patterns and knowledge
- Optimize memory usage by removing redundant information
- Maintain system performance as memory grows
- Prevent information overload that degrades decision quality
- Balance between retention and forgetting for optimal learning

Example: Information Overload
WITHOUT PRUNING (Traditional):
- AI assistant learns user prefers coffee every morning for 6 months
- Stores 180+ individual instances of same preference
- Memory bloated with redundant information
- Slower retrieval due to information overload
- No distinction between important patterns and noise
- Result: Poor performance, storage waste, information confusion

REAL WORLD EXAMPLE:
=================
How does human memory pruning work?

HUMAN FORGETTING MECHANISMS:
1. INTERFERENCE: New information replaces old conflicting information
2. DECAY: Unused memories fade over time naturally
3. RETRIEVAL-INDUCED FORGETTING: Recalling some memories weakens related ones
4. MOTIVATED FORGETTING: Intentionally suppress traumatic or irrelevant memories
5. CONSOLIDATION: Important memories strengthened while details fade
6. SCHEMA-BASED FORGETTING: Remember patterns, forget specific instances
7. ADAPTIVE FORGETTING: Forget outdated information to learn new patterns

BENEFITS OF MEMORY PRUNING:
- Maintains optimal system performance with bounded memory usage
- Improves retrieval speed by reducing information noise
- Enhances learning by removing conflicting or outdated information
- Preserves important patterns while forgetting irrelevant details
- Prevents information overload that degrades decision quality
- Enables continuous adaptation to changing environments

THE PRUNING ADVANTAGE:
=====================
UNPRUNED: Accumulated noise → Poor performance, information overload
PRUNED: Optimized knowledge → Efficient retrieval, clear patterns

PRUNING COMPONENTS:
==================
1. RELEVANCE ASSESSMENT: Determining which memories are still useful
2. REDUNDANCY DETECTION: Identifying duplicate or similar information
3. DECAY MECHANISMS: Time-based forgetting with different rates
4. CONFLICT RESOLUTION: Handling contradictory information
5. PATTERN PRESERVATION: Keeping important patterns while pruning details
6. ADAPTIVE THRESHOLDS: Dynamic pruning based on system performance
7. SELECTIVE RETENTION: Preserving critical information during pruning

WHY THIS IS REVOLUTIONARY:
========================
- Enables AI systems to operate efficiently with bounded memory
- Critical for long-running systems that accumulate information continuously
- Improves learning by removing noise and conflicting information
- Maintains system responsiveness as information scales
- Foundation for adaptive AI that evolves with changing conditions
- Mimics human cognitive efficiency through intelligent forgetting
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class MemoryImportance(Enum):
    """Importance levels for memory retention"""
    CRITICAL = 5      # Never prune (system memories, core knowledge)
    HIGH = 4          # Rarely prune (user preferences, important patterns)
    MEDIUM = 3        # Prune when redundant (general conversations)
    LOW = 2           # Prune regularly (temporary information)
    MINIMAL = 1       # Prune aggressively (noise, errors)

class PruningStrategy(Enum):
    """Strategies for memory pruning"""
    TIME_DECAY = "time_decay"                    # Remove based on age
    FREQUENCY_BASED = "frequency_based"          # Remove rarely accessed
    REDUNDANCY_REMOVAL = "redundancy_removal"    # Remove duplicate information
    CONFIDENCE_BASED = "confidence_based"        # Remove low-confidence memories
    RELEVANCE_SCORING = "relevance_scoring"      # Remove irrelevant memories
    ADAPTIVE_PRUNING = "adaptive_pruning"        # Dynamic pruning based on performance
    INTERFERENCE_BASED = "interference_based"    # Remove conflicting information

class PruningTrigger(Enum):
    """Triggers for memory pruning"""
    MEMORY_LIMIT = "memory_limit"        # Memory usage exceeds threshold
    PERFORMANCE_DEGRADATION = "performance_degradation"  # System performance drops
    TIME_BASED = "time_based"           # Scheduled pruning
    REDUNDANCY_THRESHOLD = "redundancy_threshold"  # Too much redundant information
    CONFLICT_DETECTION = "conflict_detection"  # Conflicting information found
    MANUAL = "manual"                   # User-initiated pruning

@dataclass
class MemoryItem:
    """Represents a memory item for pruning analysis"""
    
    id: str
    content: Dict[str, Any]
    importance: MemoryImportance
    
    # Temporal information
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    
    # Quality metrics
    confidence_score: float = 1.0
    relevance_score: float = 1.0
    
    # Relationships
    related_memories: Set[str] = field(default_factory=set)
    superseded_by: Optional[str] = None
    supersedes: Set[str] = field(default_factory=set)
    
    # Pruning tracking
    pruning_resistance: float = 1.0  # Higher = harder to prune
    last_pruning_evaluation: Optional[datetime] = None
    pruning_attempts: int = 0
    
    # Context and metadata
    context_tags: Set[str] = field(default_factory=set)
    source: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
    
    def update_access(self) -> None:
        """Update access tracking"""
        self.access_count += 1
        self.last_accessed = datetime.now()
        
        # Accessing memory increases its resistance to pruning
        self.pruning_resistance = min(2.0, self.pruning_resistance * 1.1)
    
    def calculate_retention_score(self) -> float:
        """Calculate overall score for memory retention"""
        
        # Time factors
        age_hours = (datetime.now() - self.created_at).total_seconds() / 3600
        recency_hours = (datetime.now() - self.last_accessed).total_seconds() / 3600
        
        # Age factor (newer is better, but with diminishing returns)
        age_factor = 1.0 / (1.0 + age_hours / 168)  # 168 hours = 1 week
        
        # Recency factor (recently accessed is better)
        recency_factor = 1.0 / (1.0 + recency_hours / 24)  # 24 hours
        
        # Frequency factor (more accessed is better)
        frequency_factor = min(1.0, self.access_count / 10.0)
        
        # Quality factors
        importance_factor = self.importance.value / 5.0
        confidence_factor = self.confidence_score
        relevance_factor = self.relevance_score
        resistance_factor = min(1.0, self.pruning_resistance / 2.0)
        
        # Weighted combination
        retention_score = (
            age_factor * 0.15 +
            recency_factor * 0.2 +
            frequency_factor * 0.15 +
            importance_factor * 0.25 +
            confidence_factor * 0.1 +
            relevance_factor * 0.1 +
            resistance_factor * 0.05
        )
        
        return min(1.0, retention_score)
    
    def calculate_redundancy_score(self, other_memories: List['MemoryItem']) -> float:
        """Calculate how redundant this memory is compared to others"""
        
        if not other_memories:
            return 0.0
        
        max_similarity = 0.0
        
        for other_memory in other_memories:
            if other_memory.id == self.id:
                continue
            
            similarity = self._calculate_similarity(other_memory)
            max_similarity = max(max_similarity, similarity)
        
        return max_similarity
    
    def _calculate_similarity(self, other_memory: 'MemoryItem') -> float:
        """Calculate similarity between this and another memory"""
        
        # Content similarity
        content_sim = self._calculate_content_similarity(other_memory.content)
        
        # Context similarity
        context_sim = self._calculate_context_similarity(other_memory.context_tags)
        
        # Temporal similarity (memories close in time are more similar)
        time_diff_hours = abs((self.created_at - other_memory.created_at).total_seconds() / 3600)
        temporal_sim = 1.0 / (1.0 + time_diff_hours / 24.0)  # Decay over days
        
        # Combined similarity
        similarity = (content_sim * 0.6 + context_sim * 0.3 + temporal_sim * 0.1)
        
        return similarity
    
    def _calculate_content_similarity(self, other_content: Dict[str, Any]) -> float:
        """Calculate content similarity between memory contents"""
        
        all_keys = set(self.content.keys()) | set(other_content.keys())
        
        if not all_keys:
            return 0.0
        
        matching_score = 0.0
        
        for key in all_keys:
            value1 = self.content.get(key)
            value2 = other_content.get(key)
            
            if value1 == value2:
                matching_score += 1.0
            elif value1 is not None and value2 is not None:
                # Partial similarity for different values of same key
                matching_score += 0.3
        
        return matching_score / len(all_keys)
    
    def _calculate_context_similarity(self, other_context: Set[str]) -> float:
        """Calculate context tag similarity"""
        
        if not self.context_tags and not other_context:
            return 1.0
        
        if not self.context_tags or not other_context:
            return 0.0
        
        intersection = self.context_tags & other_context
        union = self.context_tags | other_context
        
        return len(intersection) / len(union)  # Jaccard similarity
    
    def get_content_hash(self) -> str:
        """Get hash of content for exact duplicate detection"""
        
        content_str = json.dumps(self.content, sort_keys=True)
        return hashlib.md5(content_str.encode()).hexdigest()

class MemoryPruner:
    """Handles different memory pruning strategies"""
    
    def __init__(self):
        self.pruning_strategies = {
            PruningStrategy.TIME_DECAY: self._prune_by_time_decay,
            PruningStrategy.FREQUENCY_BASED: self._prune_by_frequency,
            PruningStrategy.REDUNDANCY_REMOVAL: self._prune_by_redundancy,
            PruningStrategy.CONFIDENCE_BASED: self._prune_by_confidence,
            PruningStrategy.RELEVANCE_SCORING: self._prune_by_relevance,
            PruningStrategy.ADAPTIVE_PRUNING: self._prune_adaptively,
            PruningStrategy.INTERFERENCE_BASED: self._prune_by_interference
        }
        
        self.logger = logging.getLogger("MemoryPruner")
    
    def prune_memories(self, memories: List[MemoryItem], strategy: PruningStrategy,
                      target_reduction: float = 0.2) -> Tuple[List[str], Dict[str, Any]]:
        """Prune memories using specified strategy"""
        
        if strategy not in self.pruning_strategies:
            self.logger.warning(f"Unknown pruning strategy: {strategy}")
            return [], {}
        
        pruning_func = self.pruning_strategies[strategy]
        
        try:
            return pruning_func(memories, target_reduction)
        except Exception as e:
            self.logger.error(f"Pruning failed with strategy {strategy}: {e}")
            return [], {}
    
    def _prune_by_time_decay(self, memories: List[MemoryItem], 
                           target_reduction: float) -> Tuple[List[str], Dict[str, Any]]:
        """Prune memories based on time decay"""
        
        # Calculate age-based pruning scores
        scored_memories = []
        
        for memory in memories:
            if memory.importance == MemoryImportance.CRITICAL:
                continue  # Never prune critical memories
            
            age_hours = (datetime.now() - memory.created_at).total_seconds() / 3600
            recency_hours = (datetime.now() - memory.last_accessed).total_seconds() / 3600
            
            # Exponential decay based on age and recency
            age_decay = math.exp(-age_hours / 168)  # Weekly decay
            recency_decay = math.exp(-recency_hours / 24)  # Daily decay
            
            # Combined decay score (lower = more likely to prune)
            decay_score = (age_decay + recency_decay) / 2
            
            # Adjust by importance
            importance_multiplier = memory.importance.value / 5.0
            final_score = decay_score * importance_multiplier
            
            scored_memories.append((memory.id, final_score))
        
        # Sort by score (lowest first = most likely to prune)
        scored_memories.sort(key=lambda x: x[1])
        
        # Prune target percentage
        target_count = int(len(scored_memories) * target_reduction)
        to_prune = scored_memories[:target_count]
        
        pruned_ids = [memory_id for memory_id, _ in to_prune]
        
        pruning_info = {
            'strategy': 'time_decay',
            'total_evaluated': len(memories),
            'target_reduction': target_reduction,
            'actual_pruned': len(pruned_ids),
            'avg_age_pruned': self._calculate_avg_age([m for m in memories if m.id in pruned_ids]),
            'criteria': 'Age and recency-based exponential decay'
        }
        
        return pruned_ids, pruning_info
    
    def _prune_by_frequency(self, memories: List[MemoryItem], 
                          target_reduction: float) -> Tuple[List[str], Dict[str, Any]]:
        """Prune memories based on access frequency"""
        
        # Calculate frequency-based scores
        scored_memories = []
        
        for memory in memories:
            if memory.importance == MemoryImportance.CRITICAL:
                continue
            
            # Frequency score based on access count and recency
            frequency_score = memory.access_count
            
            # Boost score for recently accessed memories
            recency_hours = (datetime.now() - memory.last_accessed).total_seconds() / 3600
            recency_boost = 1.0 / (1.0 + recency_hours / 24)
            
            final_score = frequency_score * (1.0 + recency_boost)
            
            # Adjust by importance
            importance_multiplier = memory.importance.value / 5.0
            final_score *= importance_multiplier
            
            scored_memories.append((memory.id, final_score))
        
        # Sort by score (lowest first = least accessed)
        scored_memories.sort(key=lambda x: x[1])
        
        # Prune target percentage
        target_count = int(len(scored_memories) * target_reduction)
        to_prune = scored_memories[:target_count]
        
        pruned_ids = [memory_id for memory_id, _ in to_prune]
        
        pruning_info = {
            'strategy': 'frequency_based',
            'total_evaluated': len(memories),
            'target_reduction': target_reduction,
            'actual_pruned': len(pruned_ids),
            'avg_access_count_pruned': self._calculate_avg_access_count([m for m in memories if m.id in pruned_ids]),
            'criteria': 'Low access frequency and poor recency'
        }
        
        return pruned_ids, pruning_info
    
    def _prune_by_redundancy(self, memories: List[MemoryItem], 
                           target_reduction: float) -> Tuple[List[str], Dict[str, Any]]:
        """Prune memories based on redundancy"""
        
        # Find redundant memories
        redundant_groups = self._find_redundant_groups(memories)
        
        # For each redundant group, keep the best one and prune others
        to_prune = []
        
        for group in redundant_groups:
            if len(group) <= 1:
                continue
            
            # Score memories in group by retention score
            group_scores = [(memory, memory.calculate_retention_score()) for memory in group]
            group_scores.sort(key=lambda x: x[1], reverse=True)  # Best first
            
            # Keep the best one, prune the rest
            to_keep = group_scores[0][0]
            to_prune_in_group = [memory for memory, _ in group_scores[1:]]
            
            # Don't prune critical memories
            to_prune_in_group = [m for m in to_prune_in_group if m.importance != MemoryImportance.CRITICAL]
            
            to_prune.extend(to_prune_in_group)
        
        # Limit to target reduction
        if len(to_prune) > int(len(memories) * target_reduction):
            # Sort by retention score and take worst ones
            to_prune.sort(key=lambda x: x.calculate_retention_score())
            to_prune = to_prune[:int(len(memories) * target_reduction)]
        
        pruned_ids = [memory.id for memory in to_prune]
        
        pruning_info = {
            'strategy': 'redundancy_removal',
            'total_evaluated': len(memories),
            'target_reduction': target_reduction,
            'actual_pruned': len(pruned_ids),
            'redundant_groups_found': len(redundant_groups),
            'avg_similarity_pruned': self._calculate_avg_redundancy_score(to_prune, memories),
            'criteria': 'High similarity to retained memories'
        }
        
        return pruned_ids, pruning_info
    
    def _prune_by_confidence(self, memories: List[MemoryItem], 
                           target_reduction: float) -> Tuple[List[str], Dict[str, Any]]:
        """Prune memories based on confidence scores"""
        
        # Score memories by confidence
        scored_memories = []
        
        for memory in memories:
            if memory.importance == MemoryImportance.CRITICAL:
                continue
            
            # Base confidence score
            confidence_score = memory.confidence_score
            
            # Adjust by relevance
            adjusted_score = confidence_score * memory.relevance_score
            
            # Factor in importance
            importance_multiplier = memory.importance.value / 5.0
            final_score = adjusted_score * importance_multiplier
            
            scored_memories.append((memory.id, final_score))
        
        # Sort by score (lowest first = least confident)
        scored_memories.sort(key=lambda x: x[1])
        
        # Prune target percentage
        target_count = int(len(scored_memories) * target_reduction)
        to_prune = scored_memories[:target_count]
        
        pruned_ids = [memory_id for memory_id, _ in to_prune]
        
        pruning_info = {
            'strategy': 'confidence_based',
            'total_evaluated': len(memories),
            'target_reduction': target_reduction,
            'actual_pruned': len(pruned_ids),
            'avg_confidence_pruned': self._calculate_avg_confidence([m for m in memories if m.id in pruned_ids]),
            'confidence_threshold': to_prune[-1][1] if to_prune else 0.0,
            'criteria': 'Low confidence and relevance scores'
        }
        
        return pruned_ids, pruning_info
    
    def _prune_by_relevance(self, memories: List[MemoryItem], 
                          target_reduction: float) -> Tuple[List[str], Dict[str, Any]]:
        """Prune memories based on relevance scoring"""
        
        # Calculate context relevance
        context_popularity = self._calculate_context_popularity(memories)
        
        # Score memories by relevance
        scored_memories = []
        
        for memory in memories:
            if memory.importance == MemoryImportance.CRITICAL:
                continue
            
            # Base relevance score
            relevance_score = memory.relevance_score
            
            # Context relevance (how popular are this memory's contexts)
            context_relevance = 0.0
            if memory.context_tags:
                context_scores = [context_popularity.get(tag, 0.0) for tag in memory.context_tags]
                context_relevance = sum(context_scores) / len(context_scores)
            
            # Temporal relevance (more recent is more relevant)
            age_hours = (datetime.now() - memory.created_at).total_seconds() / 3600
            temporal_relevance = 1.0 / (1.0 + age_hours / 168)  # Weekly decay
            
            # Combined relevance score
            combined_relevance = (
                relevance_score * 0.5 +
                context_relevance * 0.3 +
                temporal_relevance * 0.2
            )
            
            # Factor in importance
            importance_multiplier = memory.importance.value / 5.0
            final_score = combined_relevance * importance_multiplier
            
            scored_memories.append((memory.id, final_score))
        
        # Sort by score (lowest first = least relevant)
        scored_memories.sort(key=lambda x: x[1])
        
        # Prune target percentage
        target_count = int(len(scored_memories) * target_reduction)
        to_prune = scored_memories[:target_count]
        
        pruned_ids = [memory_id for memory_id, _ in to_prune]
        
        pruning_info = {
            'strategy': 'relevance_scoring',
            'total_evaluated': len(memories),
            'target_reduction': target_reduction,
            'actual_pruned': len(pruned_ids),
            'avg_relevance_pruned': self._calculate_avg_relevance([m for m in memories if m.id in pruned_ids]),
            'context_popularity_considered': len(context_popularity),
            'criteria': 'Low relevance, context popularity, and temporal relevance'
        }
        
        return pruned_ids, pruning_info
    
    def _prune_adaptively(self, memories: List[MemoryItem], 
                         target_reduction: float) -> Tuple[List[str], Dict[str, Any]]:
        """Adaptive pruning based on memory characteristics"""
        
        # Analyze memory distribution
        analysis = self._analyze_memory_distribution(memories)
        
        # Choose strategy based on analysis
        if analysis['high_redundancy']:
            return self._prune_by_redundancy(memories, target_reduction)
        elif analysis['many_old_memories']:
            return self._prune_by_time_decay(memories, target_reduction)
        elif analysis['low_confidence_memories']:
            return self._prune_by_confidence(memories, target_reduction)
        else:
            # Default to frequency-based pruning
            return self._prune_by_frequency(memories, target_reduction)
    
    def _prune_by_interference(self, memories: List[MemoryItem], 
                             target_reduction: float) -> Tuple[List[str], Dict[str, Any]]:
        """Prune memories that interfere with each other"""
        
        # Find conflicting memories
        conflicting_pairs = self._find_conflicting_memories(memories)
        
        # For each conflict, prune the weaker memory
        to_prune = []
        
        for memory1, memory2 in conflicting_pairs:
            if memory1.importance == MemoryImportance.CRITICAL and memory2.importance == MemoryImportance.CRITICAL:
                continue  # Don't prune critical memories
            
            # Compare memories and prune the weaker one
            score1 = memory1.calculate_retention_score()
            score2 = memory2.calculate_retention_score()
            
            if score1 < score2 and memory1.importance != MemoryImportance.CRITICAL:
                to_prune.append(memory1)
            elif score2 < score1 and memory2.importance != MemoryImportance.CRITICAL:
                to_prune.append(memory2)
        
        # Remove duplicates
        to_prune = list(set(to_prune))
        
        # Limit to target reduction
        if len(to_prune) > int(len(memories) * target_reduction):
            to_prune.sort(key=lambda x: x.calculate_retention_score())
            to_prune = to_prune[:int(len(memories) * target_reduction)]
        
        pruned_ids = [memory.id for memory in to_prune]
        
        pruning_info = {
            'strategy': 'interference_based',
            'total_evaluated': len(memories),
            'target_reduction': target_reduction,
            'actual_pruned': len(pruned_ids),
            'conflicts_found': len(conflicting_pairs),
            'conflicts_resolved': len(to_prune),
            'criteria': 'Conflicting information with lower retention scores'
        }
        
        return pruned_ids, pruning_info
    
    def _find_redundant_groups(self, memories: List[MemoryItem]) -> List[List[MemoryItem]]:
        """Find groups of redundant memories"""
        
        similarity_threshold = 0.8
        groups = []
        used_memories = set()
        
        for i, memory1 in enumerate(memories):
            if memory1.id in used_memories:
                continue
            
            group = [memory1]
            used_memories.add(memory1.id)
            
            for j, memory2 in enumerate(memories[i + 1:], i + 1):
                if memory2.id in used_memories:
                    continue
                
                similarity = memory1._calculate_similarity(memory2)
                
                if similarity >= similarity_threshold:
                    group.append(memory2)
                    used_memories.add(memory2.id)
            
            if len(group) > 1:
                groups.append(group)
        
        return groups
    
    def _find_conflicting_memories(self, memories: List[MemoryItem]) -> List[Tuple[MemoryItem, MemoryItem]]:
        """Find pairs of conflicting memories"""
        
        conflicts = []
        
        for i, memory1 in enumerate(memories):
            for memory2 in memories[i + 1:]:
                if self._are_conflicting(memory1, memory2):
                    conflicts.append((memory1, memory2))
        
        return conflicts
    
    def _are_conflicting(self, memory1: MemoryItem, memory2: MemoryItem) -> bool:
        """Check if two memories are conflicting"""
        
        # Check for same keys with different values
        for key in memory1.content.keys():
            if key in memory2.content:
                value1 = memory1.content[key]
                value2 = memory2.content[key]
                
                # If same key has different values, it's a conflict
                if value1 != value2 and isinstance(value1, (str, int, float, bool)) and isinstance(value2, (str, int, float, bool)):
                    # Check if they're in similar contexts (more likely to be actual conflicts)
                    context_similarity = memory1._calculate_context_similarity(memory2.context_tags)
                    
                    if context_similarity > 0.5:  # Similar contexts make conflict more likely
                        return True
        
        return False
    
    def _analyze_memory_distribution(self, memories: List[MemoryItem]) -> Dict[str, bool]:
        """Analyze memory characteristics to guide adaptive pruning"""
        
        if not memories:
            return {'high_redundancy': False, 'many_old_memories': False, 'low_confidence_memories': False}
        
        # Calculate redundancy
        total_similarity = 0
        pair_count = 0
        
        for i, memory1 in enumerate(memories):
            for memory2 in memories[i + 1:]:
                total_similarity += memory1._calculate_similarity(memory2)
                pair_count += 1
        
        avg_similarity = total_similarity / pair_count if pair_count > 0 else 0
        high_redundancy = avg_similarity > 0.6
        
        # Calculate age distribution
        now = datetime.now()
        old_memory_count = 0
        
        for memory in memories:
            age_hours = (now - memory.created_at).total_seconds() / 3600
            if age_hours > 168:  # Older than 1 week
                old_memory_count += 1
        
        many_old_memories = (old_memory_count / len(memories)) > 0.5
        
        # Calculate confidence distribution
        low_confidence_count = 0
        
        for memory in memories:
            if memory.confidence_score < 0.6:
                low_confidence_count += 1
        
        low_confidence_memories = (low_confidence_count / len(memories)) > 0.3
        
        return {
            'high_redundancy': high_redundancy,
            'many_old_memories': many_old_memories,
            'low_confidence_memories': low_confidence_memories
        }
    
    def _calculate_context_popularity(self, memories: List[MemoryItem]) -> Dict[str, float]:
        """Calculate popularity scores for context tags"""
        
        context_counts = defaultdict(int)
        
        for memory in memories:
            for tag in memory.context_tags:
                context_counts[tag] += 1
        
        total_memories = len(memories)
        context_popularity = {}
        
        for tag, count in context_counts.items():
            popularity = count / total_memories
            context_popularity[tag] = popularity
        
        return context_popularity
    
    def _calculate_avg_age(self, memories: List[MemoryItem]) -> float:
        """Calculate average age of memories in hours"""
        
        if not memories:
            return 0.0
        
        now = datetime.now()
        total_age = sum((now - memory.created_at).total_seconds() / 3600 for memory in memories)
        
        return total_age / len(memories)
    
    def _calculate_avg_access_count(self, memories: List[MemoryItem]) -> float:
        """Calculate average access count of memories"""
        
        if not memories:
            return 0.0
        
        total_access = sum(memory.access_count for memory in memories)
        
        return total_access / len(memories)
    
    def _calculate_avg_confidence(self, memories: List[MemoryItem]) -> float:
        """Calculate average confidence score of memories"""
        
        if not memories:
            return 0.0
        
        total_confidence = sum(memory.confidence_score for memory in memories)
        
        return total_confidence / len(memories)
    
    def _calculate_avg_relevance(self, memories: List[MemoryItem]) -> float:
        """Calculate average relevance score of memories"""
        
        if not memories:
            return 0.0
        
        total_relevance = sum(memory.relevance_score for memory in memories)
        
        return total_relevance / len(memories)
    
    def _calculate_avg_redundancy_score(self, pruned_memories: List[MemoryItem], 
                                      all_memories: List[MemoryItem]) -> float:
        """Calculate average redundancy score of pruned memories"""
        
        if not pruned_memories:
            return 0.0
        
        total_redundancy = 0
        
        for memory in pruned_memories:
            others = [m for m in all_memories if m.id != memory.id]
            redundancy = memory.calculate_redundancy_score(others)
            total_redundancy += redundancy
        
        return total_redundancy / len(pruned_memories)

class MemoryPruningSystem:
    """Complete memory pruning system with multiple strategies and triggers"""
    
    def __init__(self, max_memories: int = 10000):
        # Core components
        self.memory_pruner = MemoryPruner()
        
        # Memory storage
        self.memories: Dict[str, MemoryItem] = {}
        self.pruned_memories: Dict[str, MemoryItem] = {}  # Keep record of pruned memories
        
        # Configuration
        self.max_memories = max_memories
        self.pruning_threshold = 0.9  # Trigger pruning at 90% capacity
        self.default_pruning_reduction = 0.2  # Remove 20% of memories by default
        
        # Pruning schedule and triggers
        self.auto_pruning_enabled = True
        self.pruning_schedule: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.performance_metrics = {
            'retrieval_times': deque(maxlen=100),
            'memory_usage': deque(maxlen=100),
            'access_patterns': defaultdict(int)
        }
        
        # Statistics
        self.stats = {
            'total_memories_added': 0,
            'total_memories_pruned': 0,
            'pruning_operations': 0,
            'auto_prunings_triggered': 0,
            'manual_prunings': 0,
            'memory_capacity_savings': 0
        }
        
        self.logger = logging.getLogger("MemoryPruningSystem")
    
    async def initialize(self) -> None:
        """Initialize the memory pruning system"""
        self.logger.info("Memory pruning system initialized")
    
    def add_memory(self, content: Dict[str, Any], importance: MemoryImportance = MemoryImportance.MEDIUM,
                  confidence: float = 1.0, relevance: float = 1.0,
                  context_tags: Set[str] = None) -> str:
        """Add a new memory to the system"""
        
        memory = MemoryItem(
            id="",
            content=content,
            importance=importance,
            confidence_score=confidence,
            relevance_score=relevance,
            context_tags=context_tags or set()
        )
        
        self.memories[memory.id] = memory
        self.stats['total_memories_added'] += 1
        
        # Check if pruning should be triggered
        if self.auto_pruning_enabled:
            self._check_pruning_triggers()
        
        self.logger.debug(f"Added memory {memory.id[:8]}... ({len(self.memories)} total)")
        
        return memory.id
    
    def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """Get a memory and update its access tracking"""
        
        if memory_id in self.memories:
            memory = self.memories[memory_id]
            memory.update_access()
            
            # Track performance
            self.performance_metrics['access_patterns'][memory_id] += 1
            
            return memory
        
        return None
    
    def _check_pruning_triggers(self) -> None:
        """Check if any pruning triggers are activated"""
        
        current_memory_count = len(self.memories)
        
        # Memory limit trigger
        if current_memory_count >= self.max_memories * self.pruning_threshold:
            self._schedule_pruning(
                PruningTrigger.MEMORY_LIMIT,
                PruningStrategy.ADAPTIVE_PRUNING,
                self.default_pruning_reduction
            )
        
        # Performance degradation trigger
        if self._is_performance_degraded():
            self._schedule_pruning(
                PruningTrigger.PERFORMANCE_DEGRADATION,
                PruningStrategy.FREQUENCY_BASED,
                0.15  # Smaller reduction for performance issues
            )
        
        # Redundancy threshold trigger
        if self._check_redundancy_threshold():
            self._schedule_pruning(
                PruningTrigger.REDUNDANCY_THRESHOLD,
                PruningStrategy.REDUNDANCY_REMOVAL,
                0.25  # Higher reduction for redundancy
            )
    
    def _is_performance_degraded(self) -> bool:
        """Check if system performance has degraded"""
        
        if len(self.performance_metrics['retrieval_times']) < 10:
            return False
        
        recent_times = list(self.performance_metrics['retrieval_times'])[-10:]
        avg_recent = sum(recent_times) / len(recent_times)
        
        # Compare with overall average
        all_times = list(self.performance_metrics['retrieval_times'])
        avg_overall = sum(all_times) / len(all_times)
        
        # Performance degraded if recent times are 50% slower
        return avg_recent > avg_overall * 1.5
    
    def _check_redundancy_threshold(self) -> bool:
        """Check if redundancy exceeds threshold"""
        
        if len(self.memories) < 50:  # Need minimum memories to assess redundancy
            return False
        
        # Sample memories for efficiency
        sample_size = min(100, len(self.memories))
        memory_sample = random.sample(list(self.memories.values()), sample_size)
        
        # Calculate average redundancy
        total_redundancy = 0
        count = 0
        
        for memory in memory_sample:
            others = [m for m in memory_sample if m.id != memory.id]
            redundancy = memory.calculate_redundancy_score(others)
            total_redundancy += redundancy
            count += 1
        
        avg_redundancy = total_redundancy / count if count > 0 else 0
        
        return avg_redundancy > 0.7  # 70% redundancy threshold
    
    def _schedule_pruning(self, trigger: PruningTrigger, strategy: PruningStrategy,
                         target_reduction: float) -> None:
        """Schedule a pruning operation"""
        
        pruning_task = {
            'id': str(uuid.uuid4()),
            'trigger': trigger,
            'strategy': strategy,
            'target_reduction': target_reduction,
            'scheduled_time': datetime.now(),
            'priority': self._calculate_pruning_priority(trigger),
            'status': 'scheduled'
        }
        
        self.pruning_schedule.append(pruning_task)
        
        # Sort by priority
        self.pruning_schedule.sort(key=lambda x: x['priority'], reverse=True)
        
        self.logger.debug(f"Scheduled pruning: {trigger.value} -> {strategy.value} "
                         f"(reduction: {target_reduction:.1%})")
    
    def _calculate_pruning_priority(self, trigger: PruningTrigger) -> float:
        """Calculate priority for pruning operations"""
        
        priority_map = {
            PruningTrigger.MEMORY_LIMIT: 1.0,           # Highest priority
            PruningTrigger.PERFORMANCE_DEGRADATION: 0.9,
            PruningTrigger.CONFLICT_DETECTION: 0.8,
            PruningTrigger.REDUNDANCY_THRESHOLD: 0.6,
            PruningTrigger.TIME_BASED: 0.4,
            PruningTrigger.MANUAL: 0.5
        }
        
        return priority_map.get(trigger, 0.3)
    
    async def execute_scheduled_pruning(self, max_operations: int = 3) -> List[Dict[str, Any]]:
        """Execute scheduled pruning operations"""
        
        executed_operations = []
        operations_count = 0
        
        while self.pruning_schedule and operations_count < max_operations:
            task = self.pruning_schedule.pop(0)
            
            try:
                result = await self._execute_pruning_task(task)
                
                if result:
                    executed_operations.append(result)
                    operations_count += 1
                    
                    self.logger.info(f"Executed pruning: {task['trigger'].value} -> "
                                   f"{result['memories_pruned']} memories removed")
                
            except Exception as e:
                self.logger.error(f"Pruning execution failed: {e}")
                task['status'] = 'failed'
                task['error'] = str(e)
        
        return executed_operations
    
    async def _execute_pruning_task(self, task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute a single pruning task"""
        
        memories_list = list(self.memories.values())
        
        if not memories_list:
            return None
        
        # Execute pruning
        pruned_ids, pruning_info = self.memory_pruner.prune_memories(
            memories_list,
            task['strategy'],
            task['target_reduction']
        )
        
        if not pruned_ids:
            return None
        
        # Move pruned memories to pruned storage
        for memory_id in pruned_ids:
            if memory_id in self.memories:
                self.pruned_memories[memory_id] = self.memories[memory_id]
                del self.memories[memory_id]
        
        # Update statistics
        self.stats['total_memories_pruned'] += len(pruned_ids)
        self.stats['pruning_operations'] += 1
        self.stats['memory_capacity_savings'] += len(pruned_ids)
        
        if task['trigger'] != PruningTrigger.MANUAL:
            self.stats['auto_prunings_triggered'] += 1
        else:
            self.stats['manual_prunings'] += 1
        
        # Create execution result
        execution_result = {
            'task_id': task['id'],
            'trigger': task['trigger'].value,
            'strategy': task['strategy'].value,
            'target_reduction': task['target_reduction'],
            'memories_pruned': len(pruned_ids),
            'memories_remaining': len(self.memories),
            'execution_time': datetime.now(),
            'pruning_details': pruning_info
        }
        
        return execution_result
    
    async def manual_prune(self, strategy: PruningStrategy, 
                          target_reduction: float = 0.2) -> Dict[str, Any]:
        """Manually trigger memory pruning"""
        
        task = {
            'id': str(uuid.uuid4()),
            'trigger': PruningTrigger.MANUAL,
            'strategy': strategy,
            'target_reduction': target_reduction,
            'scheduled_time': datetime.now(),
            'priority': 0.5,
            'status': 'scheduled'
        }
        
        result = await self._execute_pruning_task(task)
        
        if result:
            self.logger.info(f"Manual pruning completed: {result['memories_pruned']} memories removed")
        
        return result or {}
    
    def analyze_memory_patterns(self) -> Dict[str, Any]:
        """Analyze current memory patterns and health"""
        
        if not self.memories:
            return {'status': 'empty', 'recommendations': ['Add memories to analyze patterns']}
        
        memories_list = list(self.memories.values())
        
        # Age analysis
        now = datetime.now()
        ages = [(now - memory.created_at).total_seconds() / 3600 for memory in memories_list]
        avg_age = sum(ages) / len(ages)
        
        # Access pattern analysis
        access_counts = [memory.access_count for memory in memories_list]
        avg_access = sum(access_counts) / len(access_counts)
        
        # Importance distribution
        importance_dist = defaultdict(int)
        for memory in memories_list:
            importance_dist[memory.importance.value] += 1
        
        # Confidence and relevance analysis
        confidences = [memory.confidence_score for memory in memories_list]
        relevances = [memory.relevance_score for memory in memories_list]
        avg_confidence = sum(confidences) / len(confidences)
        avg_relevance = sum(relevances) / len(relevances)
        
        # Context tag analysis
        all_tags = set()
        for memory in memories_list:
            all_tags.update(memory.context_tags)
        
        # Generate recommendations
        recommendations = []
        
        if avg_age > 168:  # More than 1 week
            recommendations.append("Consider time-based pruning for old memories")
        
        if avg_access < 2:
            recommendations.append("Many memories have low access; consider frequency-based pruning")
        
        if avg_confidence < 0.7:
            recommendations.append("Low average confidence; consider confidence-based pruning")
        
        if len(all_tags) / len(memories_list) < 0.5:
            recommendations.append("Improve context tagging for better organization")
        
        return {
            'status': 'analyzed',
            'memory_count': len(memories_list),
            'capacity_utilization': len(memories_list) / self.max_memories,
            'age_statistics': {
                'average_age_hours': avg_age,
                'oldest_memory_hours': max(ages),
                'newest_memory_hours': min(ages)
            },
            'access_statistics': {
                'average_access_count': avg_access,
                'max_access_count': max(access_counts),
                'unaccessed_memories': access_counts.count(0)
            },
            'importance_distribution': dict(importance_dist),
            'quality_metrics': {
                'average_confidence': avg_confidence,
                'average_relevance': avg_relevance,
                'low_confidence_count': sum(1 for c in confidences if c < 0.5),
                'low_relevance_count': sum(1 for r in relevances if r < 0.5)
            },
            'context_analysis': {
                'total_unique_tags': len(all_tags),
                'average_tags_per_memory': sum(len(m.context_tags) for m in memories_list) / len(memories_list),
                'untagged_memories': sum(1 for m in memories_list if not m.context_tags)
            },
            'recommendations': recommendations
        }
    
    def get_pruning_candidates(self, strategy: PruningStrategy, 
                             count: int = 10) -> List[Tuple[str, float, str]]:
        """Get top candidates for pruning without actually pruning them"""
        
        memories_list = list(self.memories.values())
        
        if not memories_list:
            return []
        
        pruned_ids, pruning_info = self.memory_pruner.prune_memories(
            memories_list, strategy, target_reduction=1.0  # Get all candidates
        )
        
        # Score candidates
        candidates = []
        
        for memory_id in pruned_ids[:count]:
            memory = self.memories.get(memory_id)
            
            if memory:
                retention_score = memory.calculate_retention_score()
                reason = self._generate_pruning_reason(memory, strategy)
                
                candidates.append((memory_id, retention_score, reason))
        
        return candidates
    
    def _generate_pruning_reason(self, memory: MemoryItem, strategy: PruningStrategy) -> str:
        """Generate human-readable reason for pruning a memory"""
        
        reasons = []
        
        # Age-based reasons
        age_hours = (datetime.now() - memory.created_at).total_seconds() / 3600
        if age_hours > 168:
            reasons.append(f"Old memory ({age_hours/24:.1f} days)")
        
        # Access-based reasons
        if memory.access_count == 0:
            reasons.append("Never accessed")
        elif memory.access_count < 2:
            reasons.append("Rarely accessed")
        
        # Quality-based reasons
        if memory.confidence_score < 0.5:
            reasons.append("Low confidence")
        
        if memory.relevance_score < 0.5:
            reasons.append("Low relevance")
        
        # Strategy-specific reasons
        if strategy == PruningStrategy.REDUNDANCY_REMOVAL:
            reasons.append("High redundancy with other memories")
        elif strategy == PruningStrategy.INTERFERENCE_BASED:
            reasons.append("Conflicts with other memories")
        
        return "; ".join(reasons) if reasons else "Low overall retention score"
    
    def restore_pruned_memory(self, memory_id: str) -> bool:
        """Restore a previously pruned memory"""
        
        if memory_id in self.pruned_memories:
            memory = self.pruned_memories[memory_id]
            
            # Check if we have space
            if len(self.memories) >= self.max_memories:
                self.logger.warning("Cannot restore memory: at capacity limit")
                return False
            
            # Restore memory
            self.memories[memory_id] = memory
            del self.pruned_memories[memory_id]
            
            # Update access tracking
            memory.update_access()
            
            self.logger.info(f"Restored pruned memory {memory_id[:8]}...")
            
            return True
        
        return False
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        
        # Memory statistics
        total_memories = len(self.memories)
        total_pruned = len(self.pruned_memories)
        
        # Performance statistics
        avg_retrieval_time = 0.0
        if self.performance_metrics['retrieval_times']:
            avg_retrieval_time = sum(self.performance_metrics['retrieval_times']) / len(self.performance_metrics['retrieval_times'])
        
        # Pruning effectiveness
        pruning_effectiveness = 0.0
        if self.stats['total_memories_added'] > 0:
            pruning_effectiveness = self.stats['total_memories_pruned'] / self.stats['total_memories_added']
        
        return {
            'memory_statistics': {
                'active_memories': total_memories,
                'pruned_memories': total_pruned,
                'total_memories_processed': self.stats['total_memories_added'],
                'capacity_utilization': total_memories / self.max_memories,
                'memory_efficiency': 1.0 - pruning_effectiveness
            },
            'pruning_statistics': {
                'total_pruning_operations': self.stats['pruning_operations'],
                'auto_prunings': self.stats['auto_prunings_triggered'],
                'manual_prunings': self.stats['manual_prunings'],
                'pruning_effectiveness': pruning_effectiveness,
                'capacity_savings': self.stats['memory_capacity_savings']
            },
            'performance_metrics': {
                'average_retrieval_time': avg_retrieval_time,
                'pending_pruning_tasks': len(self.pruning_schedule),
                'auto_pruning_enabled': self.auto_pruning_enabled
            },
            'configuration': {
                'max_memories': self.max_memories,
                'pruning_threshold': self.pruning_threshold,
                'default_reduction': self.default_pruning_reduction
            }
        }

# EXAMPLE DEMONSTRATIONS
# =====================

async def demo_basic_memory_pruning():
    """Demo: Basic memory pruning functionality"""
    print("\nDEMO 1: BASIC MEMORY PRUNING")
    print("=" * 50)
    
    system = MemoryPruningSystem(max_memories=20)  # Small limit for demo
    await system.initialize()
    
    print("Adding various types of memories:")
    
    # Add different types of memories
    memory_data = [
        # Recent important memories
        ({"type": "user_preference", "preference": "coffee", "context": "morning"}, MemoryImportance.HIGH, 0.9, 0.9),
        ({"type": "task", "task": "daily_standup", "time": "09:00"}, MemoryImportance.MEDIUM, 0.8, 0.8),
        
        # Older memories
        ({"type": "note", "content": "remember to buy milk"}, MemoryImportance.LOW, 0.6, 0.4),
        ({"type": "temp_info", "info": "weather was sunny yesterday"}, MemoryImportance.MINIMAL, 0.3, 0.2),
        
        # Redundant memories
        ({"type": "user_preference", "preference": "coffee", "context": "morning"}, MemoryImportance.MEDIUM, 0.8, 0.8),
        ({"type": "user_preference", "preference": "coffee", "context": "morning"}, MemoryImportance.MEDIUM, 0.7, 0.7),
        
        # Various other memories to fill up capacity
        *[({"type": "filler", "data": f"item_{i}"}, MemoryImportance.LOW, 0.5, 0.5) for i in range(15)]
    ]
    
    memory_ids = []
    
    for i, (content, importance, confidence, relevance) in enumerate(memory_data, 1):
        memory_id = system.add_memory(
            content, importance, confidence, relevance,
            context_tags={content.get("type", "unknown")}
        )
        memory_ids.append(memory_id)
        print(f"  {i:2d}. Added {content.get('type', 'unknown')} memory: {memory_id[:8]}...")
        
        # Simulate some access patterns
        if i <= 5:  # Access first 5 memories more
            for _ in range(3):
                system.get_memory(memory_id)
    
    print(f"\nTotal memories: {len(system.memories)}")
    print(f"Capacity utilization: {len(system.memories) / system.max_memories:.1%}")
    
    # Trigger automatic pruning
    print(f"\nTriggering automatic pruning:")
    
    pruning_results = await system.execute_scheduled_pruning()
    
    for result in pruning_results:
        print(f"  Pruning executed:")
        print(f"    Trigger: {result['trigger']}")
        print(f"    Strategy: {result['strategy']}")
        print(f"    Memories pruned: {result['memories_pruned']}")
        print(f"    Memories remaining: {result['memories_remaining']}")
        print(f"    Criteria: {result['pruning_details']['criteria']}")
    
    print(f"\nAfter pruning:")
    print(f"  Active memories: {len(system.memories)}")
    print(f"  Pruned memories: {len(system.pruned_memories)}")
    print(f"  New utilization: {len(system.memories) / system.max_memories:.1%}")

async def demo_pruning_strategies():
    """Demo: Different pruning strategies"""
    print("\nDEMO 2: PRUNING STRATEGIES")
    print("=" * 50)
    
    system = MemoryPruningSystem(max_memories=50)
    await system.initialize()
    
    # Disable auto-pruning for manual demonstration
    system.auto_pruning_enabled = False
    
    print("Adding memories with different characteristics:")
    
    # Create memories with different patterns for strategy testing
    base_time = datetime.now()
    
    memories_to_add = [
        # Old, rarely accessed memories
        *[{"type": "old_note", "content": f"old note {i}", "created": "2 weeks ago"} for i in range(5)],
        
        # Recent, frequently accessed memories  
        *[{"type": "recent_task", "content": f"recent task {i}", "created": "today"} for i in range(5)],
        
        # Low confidence memories
        *[{"type": "uncertain", "content": f"uncertain info {i}", "confidence": "low"} for i in range(5)],
        
        # Redundant memories
        *[{"type": "duplicate", "content": "same information", "instance": i} for i in range(5)],
        
        # High importance memories
        *[{"type": "critical", "content": f"critical info {i}", "importance": "high"} for i in range(3)],
    ]
    
    for i, content in enumerate(memories_to_add):
        # Vary importance, confidence, and age
        if "old_note" in content["type"]:
            importance = MemoryImportance.LOW
            confidence = 0.6
            relevance = 0.4
            # Simulate old memory by manually setting timestamp
            memory_id = system.add_memory(content, importance, confidence, relevance)
            system.memories[memory_id].created_at = base_time - timedelta(days=14)
            
        elif "recent_task" in content["type"]:
            importance = MemoryImportance.MEDIUM
            confidence = 0.8
            relevance = 0.9
            memory_id = system.add_memory(content, importance, confidence, relevance)
            # Simulate frequent access
            for _ in range(5):
                system.get_memory(memory_id)
                
        elif "uncertain" in content["type"]:
            importance = MemoryImportance.MEDIUM
            confidence = 0.3  # Low confidence
            relevance = 0.5
            memory_id = system.add_memory(content, importance, confidence, relevance)
            
        elif "duplicate" in content["type"]:
            importance = MemoryImportance.LOW
            confidence = 0.7
            relevance = 0.6
            memory_id = system.add_memory(content, importance, confidence, relevance)
            
        elif "critical" in content["type"]:
            importance = MemoryImportance.CRITICAL
            confidence = 0.9
            relevance = 0.9
            memory_id = system.add_memory(content, importance, confidence, relevance)
        
        print(f"  {i+1:2d}. Added {content['type']} memory")
    
    print(f"\nTesting different pruning strategies:")
    
    strategies_to_test = [
        (PruningStrategy.TIME_DECAY, "Remove old memories"),
        (PruningStrategy.FREQUENCY_BASED, "Remove rarely accessed memories"),
        (PruningStrategy.CONFIDENCE_BASED, "Remove low confidence memories"),
        (PruningStrategy.REDUNDANCY_REMOVAL, "Remove redundant memories"),
    ]
    
    for strategy, description in strategies_to_test:
        print(f"\n{strategy.value.upper()}:")
        print(f"  Description: {description}")
        
        # Get pruning candidates without actually pruning
        candidates = system.get_pruning_candidates(strategy, count=5)
        
        print(f"  Top candidates for pruning:")
        for memory_id, retention_score, reason in candidates:
            memory = system.memories[memory_id]
            print(f"    {memory_id[:8]}... - {memory.content.get('type', 'unknown')} "
                  f"(retention: {retention_score:.2f}, reason: {reason})")

async def demo_adaptive_pruning():
    """Demo: Adaptive pruning based on system state"""
    print("\nDEMO 3: ADAPTIVE PRUNING")
    print("=" * 50)
    
    system = MemoryPruningSystem(max_memories=30)
    await system.initialize()
    
    print("Simulating different memory states for adaptive pruning:")
    
    # Scenario 1: High redundancy
    print(f"\nScenario 1: High redundancy state")
    
    # Add many similar memories
    for i in range(10):
        system.add_memory(
            {"action": "coffee_order", "time": "morning", "preference": "latte", "instance": i},
            MemoryImportance.MEDIUM, 0.8, 0.7,
            context_tags={"coffee", "morning"}
        )
    
    # Add some unique memories
    for i in range(5):
        system.add_memory(
            {"action": "unique_task", "task": f"task_{i}"},
            MemoryImportance.MEDIUM, 0.7, 0.8
        )
    
    print(f"  Added {len(system.memories)} memories with high redundancy")
    
    # Analyze and prune
    analysis = system.analyze_memory_patterns()
    print(f"  System recommendations: {analysis['recommendations']}")
    
    result = await system.manual_prune(PruningStrategy.ADAPTIVE_PRUNING, 0.3)
    print(f"  Adaptive pruning result: {result['strategy']} removed {result['memories_pruned']} memories")
    
    # Scenario 2: Many old memories
    print(f"\nScenario 2: Many old memories state")
    
    base_time = datetime.now()
    
    # Add old memories
    for i in range(10):
        memory_id = system.add_memory(
            {"type": "old_data", "data": f"old_item_{i}"},
            MemoryImportance.LOW, 0.6, 0.5
        )
        # Make them old
        system.memories[memory_id].created_at = base_time - timedelta(days=30)
    
    print(f"  Added {10} old memories")
    
    # Analyze and prune  
    analysis = system.analyze_memory_patterns()
    print(f"  System recommendations: {analysis['recommendations']}")
    
    result = await system.manual_prune(PruningStrategy.ADAPTIVE_PRUNING, 0.4)
    print(f"  Adaptive pruning result: {result['strategy']} removed {result['memories_pruned']} memories")
    
    # Scenario 3: Low confidence memories
    print(f"\nScenario 3: Low confidence memories state")
    
    # Add low confidence memories
    for i in range(8):
        system.add_memory(
            {"type": "uncertain_info", "info": f"uncertain_{i}"},
            MemoryImportance.MEDIUM, 0.3, 0.4  # Low confidence and relevance
        )
    
    print(f"  Added {8} low confidence memories")
    
    # Analyze and prune
    analysis = system.analyze_memory_patterns()
    print(f"  System recommendations: {analysis['recommendations']}")
    
    result = await system.manual_prune(PruningStrategy.ADAPTIVE_PRUNING, 0.3)
    print(f"  Adaptive pruning result: {result['strategy']} removed {result['memories_pruned']} memories")
    
    # Final analysis
    print(f"\nFinal system state:")
    final_stats = system.get_system_statistics()
    print(f"  Active memories: {final_stats['memory_statistics']['active_memories']}")
    print(f"  Pruned memories: {final_stats['memory_statistics']['pruned_memories']}")
    print(f"  Pruning effectiveness: {final_stats['pruning_statistics']['pruning_effectiveness']:.2f}")

async def demo_memory_restoration():
    """Demo: Memory restoration and pruning history"""
    print("\nDEMO 4: MEMORY RESTORATION")
    print("=" * 50)
    
    system = MemoryPruningSystem(max_memories=15)
    await system.initialize()
    
    print("Adding memories and tracking pruning history:")
    
    # Add memories with different importance levels
    important_memories = []
    regular_memories = []
    
    for i in range(8):
        # Add important memory
        important_id = system.add_memory(
            {"type": "important", "task": f"critical_task_{i}", "priority": "high"},
            MemoryImportance.HIGH, 0.9, 0.9,
            context_tags={"important", "task"}
        )
        important_memories.append(important_id)
        
        # Add regular memory
        regular_id = system.add_memory(
            {"type": "regular", "note": f"regular_note_{i}", "priority": "medium"},
            MemoryImportance.MEDIUM, 0.6, 0.6,
            context_tags={"regular", "note"}
        )
        regular_memories.append(regular_id)
    
    print(f"  Added {len(important_memories)} important and {len(regular_memories)} regular memories")
    
    # Access important memories more frequently
    for memory_id in important_memories:
        for _ in range(3):
            system.get_memory(memory_id)
    
    # Trigger pruning
    print(f"\nTriggering memory pruning:")
    
    result = await system.manual_prune(PruningStrategy.FREQUENCY_BASED, 0.4)
    
    print(f"  Pruned {result['memories_pruned']} memories")
    print(f"  Remaining memories: {result['memories_remaining']}")
    
    # Show pruned memory details
    print(f"\nPruned memories details:")
    
    for i, (memory_id, memory) in enumerate(list(system.pruned_memories.items())[:5], 1):
        print(f"  {i}. {memory_id[:8]}... - {memory.content.get('type', 'unknown')}")
        print(f"     Access count: {memory.access_count}")
        print(f"     Confidence: {memory.confidence_score:.2f}")
        print(f"     Age: {(datetime.now() - memory.created_at).total_seconds() / 3600:.1f} hours")
    
    # Restore some important memories
    print(f"\nRestoring important memories:")
    
    restored_count = 0
    
    for memory_id, memory in list(system.pruned_memories.items()):
        if memory.importance == MemoryImportance.HIGH and restored_count < 3:
            if system.restore_pruned_memory(memory_id):
                print(f"  Restored: {memory_id[:8]}... - {memory.content.get('task', 'unknown')}")
                restored_count += 1
    
    # Show final state
    print(f"\nFinal memory state:")
    
    final_stats = system.get_system_statistics()
    
    print(f"  Active memories: {final_stats['memory_statistics']['active_memories']}")
    print(f"  Pruned memories: {final_stats['memory_statistics']['pruned_memories']}")
    print(f"  Total processed: {final_stats['memory_statistics']['total_memories_processed']}")
    print(f"  Restoration capability: {len(system.pruned_memories)} memories available for restoration")

async def demo_performance_based_pruning():
    """Demo: Performance-based pruning triggers"""
    print("\nDEMO 5: PERFORMANCE-BASED PRUNING")
    print("=" * 50)
    
    system = MemoryPruningSystem(max_memories=100)
    await system.initialize()
    
    print("Simulating performance degradation scenario:")
    
    # Add many memories to simulate load
    print(f"\nPhase 1: Adding memories and simulating normal performance")
    
    for i in range(40):
        system.add_memory(
            {"id": i, "data": f"data_item_{i}", "category": f"cat_{i % 5}"},
            MemoryImportance.MEDIUM, 0.7, 0.7
        )
        
        # Simulate normal retrieval time
        system.performance_metrics['retrieval_times'].append(0.1 + random.uniform(-0.02, 0.02))
    
    print(f"  Added {len(system.memories)} memories")
    print(f"  Average retrieval time: {sum(system.performance_metrics['retrieval_times']) / len(system.performance_metrics['retrieval_times']):.3f}s")
    
    # Simulate performance degradation
    print(f"\nPhase 2: Simulating performance degradation")
    
    for i in range(20):
        system.add_memory(
            {"id": i + 40, "data": f"heavy_item_{i}", "size": "large"},
            MemoryImportance.LOW, 0.5, 0.5
        )
        
        # Simulate slower retrieval times
        degraded_time = 0.15 + random.uniform(0.05, 0.1)  # Slower
        system.performance_metrics['retrieval_times'].append(degraded_time)
    
    print(f"  Added {20} more memories")
    print(f"  New average retrieval time: {sum(list(system.performance_metrics['retrieval_times'])[-10:]) / 10:.3f}s")
    
    # Check if performance degradation triggers pruning
    print(f"\nPhase 3: Checking performance-based triggers")
    
    is_degraded = system._is_performance_degraded()
    print(f"  Performance degradation detected: {is_degraded}")
    
    if is_degraded:
        # Force trigger performance-based pruning
        system._schedule_pruning(
            PruningTrigger.PERFORMANCE_DEGRADATION,
            PruningStrategy.FREQUENCY_BASED,
            0.2
        )
        
        print(f"  Scheduled performance-based pruning")
        
        # Execute pruning
        results = await system.execute_scheduled_pruning()
        
        for result in results:
            print(f"  Executed: {result['strategy']} pruning")
            print(f"    Trigger: {result['trigger']}")
            print(f"    Memories pruned: {result['memories_pruned']}")
            print(f"    Remaining: {result['memories_remaining']}")
    
    # Simulate performance improvement after pruning
    print(f"\nPhase 4: Performance after pruning")
    
    for i in range(10):
        # Simulate improved retrieval times
        improved_time = 0.08 + random.uniform(-0.01, 0.02)
        system.performance_metrics['retrieval_times'].append(improved_time)
    
    final_avg = sum(list(system.performance_metrics['retrieval_times'])[-10:]) / 10
    print(f"  Post-pruning retrieval time: {final_avg:.3f}s")
    
    # Show comprehensive statistics
    print(f"\nPerformance statistics:")
    
    stats = system.get_system_statistics()
    
    print(f"  Memory efficiency: {stats['memory_statistics']['memory_efficiency']:.2f}")
    print(f"  Pruning operations: {stats['pruning_statistics']['total_pruning_operations']}")
    print(f"  Auto-triggered prunings: {stats['pruning_statistics']['auto_prunings']}")
    print(f"  Capacity utilization: {stats['memory_statistics']['capacity_utilization']:.1%}")

async def main():
    """
    Demonstrate Memory Pruning System for intelligent forgetting and optimization
    
    WHAT YOU'LL LEARN:
    ================
    1. How to implement intelligent memory pruning with multiple strategies
    2. How to design time-decay, frequency-based, and redundancy removal algorithms
    3. How to build adaptive pruning that responds to system state
    4. How to create performance-based pruning triggers
    5. How to implement memory restoration and pruning history tracking
    6. How to balance retention and forgetting for optimal system performance
    
    REAL WORLD APPLICATIONS:
    =======================
    - Long-running AI assistants that accumulate conversation history
    - Customer service systems managing large interaction histories
    - Recommendation systems optimizing user preference storage
    - IoT systems managing sensor data and event histories
    - Educational platforms tracking learning progress and forgetting curves
    - Content management systems with automatic archival and cleanup
    """
    
    print("MEMORY PRUNING SYSTEM DEMONSTRATION")
    print("Intelligent forgetting for optimal performance!")
    
    await demo_basic_memory_pruning()
    await demo_pruning_strategies()
    await demo_adaptive_pruning()
    await demo_memory_restoration()
    await demo_performance_based_pruning()
    
    print("\nWHAT WE LEARNED:")
    print("=" * 40)
    print("✓ Memory pruning maintains optimal system performance with bounded storage")
    print("✓ Different pruning strategies handle various memory patterns effectively")
    print("✓ Adaptive pruning responds intelligently to changing system conditions")
    print("✓ Performance-based triggers prevent degradation from memory overload")
    print("✓ Memory restoration provides safety nets for important information")
    print("✓ Intelligent forgetting enables continuous learning and adaptation")
    print("\nTHE POWER OF MEMORY PRUNING:")
    print("- Enables AI systems to operate efficiently with bounded memory")
    print("- Improves learning by removing noise and conflicting information")
    print("- Maintains system responsiveness as information scales")
    print("- Mimics human cognitive efficiency through intelligent forgetting")
    print("- Essential for production AI systems with continuous operation")

if __name__ == "__main__":
    asyncio.run(main())
