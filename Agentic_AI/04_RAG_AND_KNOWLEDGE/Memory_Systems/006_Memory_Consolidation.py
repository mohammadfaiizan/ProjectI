#!/usr/bin/env python3
"""
Memory Consolidation: Converting Temporary Information into Permanent Knowledge
============================================================================

WHAT IS THE PROBLEM?
==================
AI agents accumulate vast amounts of temporary information but lack mechanisms to:
- Convert important temporary memories into permanent knowledge
- Identify which experiences deserve long-term storage
- Compress and organize memories for efficient retrieval
- Remove redundant or irrelevant information over time
- Create abstract patterns from specific experiences
- Maintain memory consistency and avoid contradictions

Example: Learning Without Consolidation
WITHOUT CONSOLIDATION (Traditional):
- Agent learns user prefers coffee over tea during conversation
- Information stored as isolated interaction record
- Same preference learned again in future conversations  
- No pattern recognition across similar preferences
- Memory grows linearly without organization
- Result: Inefficient storage, poor pattern recognition, repetitive learning

REAL WORLD EXAMPLE:
=================
How does human memory consolidation work?

HUMAN MEMORY CONSOLIDATION:
1. ENCODING: Initial temporary storage of experiences in hippocampus
2. REHEARSAL: Repeated activation strengthens important memories
3. INTEGRATION: New information integrated with existing knowledge schemas
4. ABSTRACTION: Specific details generalized into patterns and rules
5. TRANSFER: Important memories moved to long-term cortical storage
6. FORGETTING: Irrelevant details fade while core patterns remain
7. RECONSOLIDATION: Memories updated when recalled and re-stored

BENEFITS OF MEMORY CONSOLIDATION:
- Transforms raw experiences into structured knowledge
- Enables pattern recognition and learning from repetition
- Reduces memory storage requirements through compression
- Creates hierarchical knowledge organization
- Supports transfer learning and generalization
- Maintains memory coherence and consistency

THE CONSOLIDATION ADVANTAGE:
===========================
UNCONSOLIDATED: Raw experiences → Inefficient isolated memories
CONSOLIDATED: Processed patterns → Structured retrievable knowledge

CONSOLIDATION COMPONENTS:
========================
1. IMPORTANCE ASSESSMENT: Identifying which memories deserve consolidation
2. PATTERN EXTRACTION: Finding common themes across multiple experiences
3. ABSTRACTION CREATION: Generalizing specific instances into rules
4. KNOWLEDGE INTEGRATION: Merging new patterns with existing knowledge
5. CONFLICT RESOLUTION: Handling contradictory information
6. COMPRESSION ALGORITHMS: Efficient encoding of consolidated knowledge
7. MAINTENANCE CYCLES: Periodic reorganization and cleanup

WHY THIS IS REVOLUTIONARY:
========================
- Enables AI to learn efficiently from accumulated experiences
- Critical for agents that need to improve over time
- Supports human-like learning and knowledge development
- Reduces memory footprint while increasing knowledge quality
- Foundation for continuous learning and adaptation
- Enables transfer of learning across domains and tasks
"""

import asyncio
import time
import json
import uuid
import re
import math
from typing import Dict, List, Any, Optional, Tuple, Union, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import numpy as np
from contextlib import contextmanager
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class MemoryType(Enum):
    """Types of memories for consolidation"""
    EPISODIC = "episodic"           # Specific experiences and events
    SEMANTIC = "semantic"           # Facts and general knowledge
    PROCEDURAL = "procedural"       # Skills and procedures
    PREFERENCE = "preference"       # User preferences and patterns
    PATTERN = "pattern"             # Recognized behavioral patterns
    RULE = "rule"                   # Derived rules and principles

class ConsolidationTrigger(Enum):
    """Triggers for memory consolidation"""
    FREQUENCY = "frequency"         # Repeated similar experiences
    IMPORTANCE = "importance"       # High-importance memories
    TIME_BASED = "time_based"       # Scheduled consolidation
    SIMILARITY = "similarity"       # Similar memory clusters
    CONFLICT = "conflict"           # Conflicting information
    MANUAL = "manual"               # User-initiated consolidation

class ConsolidationStrategy(Enum):
    """Strategies for memory consolidation"""
    PATTERN_EXTRACTION = "pattern_extraction"
    FREQUENCY_BASED = "frequency_based"
    SIMILARITY_CLUSTERING = "similarity_clustering"
    RULE_INDUCTION = "rule_induction"
    ABSTRACTION_HIERARCHY = "abstraction_hierarchy"
    COMPRESSION_BASED = "compression_based"

@dataclass
class MemoryItem:
    """Represents a memory item for consolidation"""
    
    id: str
    content: Dict[str, Any]
    memory_type: MemoryType
    
    # Temporal information
    timestamp: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    
    # Importance and relevance
    importance_score: float = 1.0
    confidence_score: float = 1.0
    
    # Consolidation tracking
    is_consolidated: bool = False
    consolidation_count: int = 0
    
    # Relationships
    related_memories: Set[str] = field(default_factory=set)
    derived_from: Set[str] = field(default_factory=set)
    contributes_to: Set[str] = field(default_factory=set)
    
    # Context
    context_tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
    
    def update_access(self) -> None:
        """Update access tracking"""
        self.access_count += 1
        self.last_accessed = datetime.now()
    
    def calculate_retention_score(self) -> float:
        """Calculate how likely this memory should be retained"""
        
        # Combine multiple factors
        age_factor = self._calculate_age_factor()
        frequency_factor = min(1.0, self.access_count / 10.0)
        importance_factor = self.importance_score
        confidence_factor = self.confidence_score
        
        # Weighted combination
        retention_score = (
            age_factor * 0.2 +
            frequency_factor * 0.3 + 
            importance_factor * 0.3 +
            confidence_factor * 0.2
        )
        
        return min(1.0, retention_score)
    
    def _calculate_age_factor(self) -> float:
        """Calculate age-based retention factor"""
        
        age_hours = (datetime.now() - self.timestamp).total_seconds() / 3600
        
        # Exponential decay with different rates for different memory types
        if self.memory_type == MemoryType.EPISODIC:
            decay_rate = 0.1  # Faster decay for specific episodes
        elif self.memory_type == MemoryType.SEMANTIC:
            decay_rate = 0.05  # Slower decay for general knowledge
        else:
            decay_rate = 0.075  # Medium decay for other types
        
        return math.exp(-decay_rate * age_hours / 24)  # Daily decay
    
    def get_content_hash(self) -> str:
        """Get hash of content for similarity comparison"""
        
        content_str = json.dumps(self.content, sort_keys=True)
        return hashlib.md5(content_str.encode()).hexdigest()

@dataclass
class ConsolidatedMemory:
    """Represents consolidated memory from multiple sources"""
    
    id: str
    consolidated_content: Dict[str, Any]
    memory_type: MemoryType
    
    # Source tracking
    source_memories: Set[str] = field(default_factory=set)
    consolidation_strategy: ConsolidationStrategy = ConsolidationStrategy.PATTERN_EXTRACTION
    
    # Quality metrics
    confidence_score: float = 1.0
    compression_ratio: float = 1.0
    pattern_strength: float = 1.0
    
    # Temporal information
    consolidation_timestamp: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    # Usage tracking
    retrieval_count: int = 0
    application_count: int = 0
    
    # Metadata
    context_tags: Set[str] = field(default_factory=set)
    derived_patterns: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
    
    def update_retrieval(self) -> None:
        """Update retrieval tracking"""
        self.retrieval_count += 1
    
    def update_application(self) -> None:
        """Update application tracking"""
        self.application_count += 1

class PatternExtractor:
    """Extracts patterns from memory collections"""
    
    def __init__(self):
        self.logger = logging.getLogger("PatternExtractor")
    
    def extract_patterns(self, memories: List[MemoryItem]) -> List[Dict[str, Any]]:
        """Extract patterns from a collection of memories"""
        
        if len(memories) < 2:
            return []
        
        patterns = []
        
        # Extract different types of patterns
        patterns.extend(self._extract_frequency_patterns(memories))
        patterns.extend(self._extract_sequence_patterns(memories))
        patterns.extend(self._extract_attribute_patterns(memories))
        patterns.extend(self._extract_conditional_patterns(memories))
        
        return patterns
    
    def _extract_frequency_patterns(self, memories: List[MemoryItem]) -> List[Dict[str, Any]]:
        """Extract patterns based on frequency of occurrence"""
        
        patterns = []
        
        # Group memories by content attributes
        attribute_counts = defaultdict(Counter)
        
        for memory in memories:
            for key, value in memory.content.items():
                if isinstance(value, (str, int, float, bool)):
                    attribute_counts[key][str(value)] += 1
        
        # Find frequent patterns
        for attribute, value_counts in attribute_counts.items():
            total_memories = len(memories)
            
            for value, count in value_counts.items():
                frequency = count / total_memories
                
                if frequency >= 0.3:  # 30% threshold for pattern
                    patterns.append({
                        'type': 'frequency_pattern',
                        'attribute': attribute,
                        'value': value,
                        'frequency': frequency,
                        'support_count': count,
                        'confidence': frequency,
                        'description': f"{attribute} is frequently '{value}' ({frequency:.1%} of cases)"
                    })
        
        return patterns
    
    def _extract_sequence_patterns(self, memories: List[MemoryItem]) -> List[Dict[str, Any]]:
        """Extract temporal sequence patterns"""
        
        patterns = []
        
        # Sort memories by timestamp
        sorted_memories = sorted(memories, key=lambda m: m.timestamp)
        
        # Look for action sequences
        if len(sorted_memories) >= 3:
            sequences = []
            
            for i in range(len(sorted_memories) - 2):
                sequence = [
                    sorted_memories[i].content.get('action', 'unknown'),
                    sorted_memories[i + 1].content.get('action', 'unknown'),
                    sorted_memories[i + 2].content.get('action', 'unknown')
                ]
                sequences.append(tuple(sequence))
            
            # Count sequence frequencies
            sequence_counts = Counter(sequences)
            
            for sequence, count in sequence_counts.items():
                if count >= 2:  # Appeared at least twice
                    frequency = count / (len(sorted_memories) - 2)
                    
                    patterns.append({
                        'type': 'sequence_pattern',
                        'sequence': list(sequence),
                        'frequency': frequency,
                        'support_count': count,
                        'confidence': frequency,
                        'description': f"Action sequence: {' → '.join(sequence)} (occurs {frequency:.1%} of time)"
                    })
        
        return patterns
    
    def _extract_attribute_patterns(self, memories: List[MemoryItem]) -> List[Dict[str, Any]]:
        """Extract patterns based on attribute relationships"""
        
        patterns = []
        
        # Find correlations between attributes
        attribute_pairs = defaultdict(lambda: defaultdict(Counter))
        
        for memory in memories:
            content_items = list(memory.content.items())
            
            for i, (key1, value1) in enumerate(content_items):
                for key2, value2 in content_items[i + 1:]:
                    if isinstance(value1, (str, int, float, bool)) and isinstance(value2, (str, int, float, bool)):
                        attribute_pairs[key1][str(value1)][f"{key2}={value2}"] += 1
        
        # Find strong correlations
        for attr1, value_map in attribute_pairs.items():
            for value1, correlations in value_map.items():
                total_occurrences = sum(correlations.values())
                
                for correlation, count in correlations.items():
                    confidence = count / total_occurrences
                    
                    if confidence >= 0.7:  # 70% confidence threshold
                        patterns.append({
                            'type': 'attribute_pattern',
                            'condition': f"{attr1}={value1}",
                            'consequence': correlation,
                            'confidence': confidence,
                            'support_count': count,
                            'description': f"When {attr1}='{value1}', then {correlation} ({confidence:.1%} confidence)"
                        })
        
        return patterns
    
    def _extract_conditional_patterns(self, memories: List[MemoryItem]) -> List[Dict[str, Any]]:
        """Extract conditional patterns and rules"""
        
        patterns = []
        
        # Group memories by context tags
        context_groups = defaultdict(list)
        
        for memory in memories:
            for tag in memory.context_tags:
                context_groups[tag].append(memory)
        
        # Find patterns within contexts
        for context, context_memories in context_groups.items():
            if len(context_memories) >= 3:
                
                # Extract common attributes within this context
                common_attributes = defaultdict(Counter)
                
                for memory in context_memories:
                    for key, value in memory.content.items():
                        if isinstance(value, (str, int, float, bool)):
                            common_attributes[key][str(value)] += 1
                
                # Find dominant patterns in this context
                for attribute, value_counts in common_attributes.items():
                    total_in_context = len(context_memories)
                    
                    for value, count in value_counts.items():
                        frequency = count / total_in_context
                        
                        if frequency >= 0.6:  # 60% frequency in context
                            patterns.append({
                                'type': 'conditional_pattern',
                                'context': context,
                                'attribute': attribute,
                                'value': value,
                                'frequency': frequency,
                                'support_count': count,
                                'confidence': frequency,
                                'description': f"In context '{context}', {attribute} is usually '{value}' ({frequency:.1%})"
                            })
        
        return patterns

class MemoryCompressor:
    """Compresses related memories into consolidated forms"""
    
    def __init__(self):
        self.logger = logging.getLogger("MemoryCompressor")
    
    def compress_memories(self, memories: List[MemoryItem], 
                         strategy: ConsolidationStrategy) -> Dict[str, Any]:
        """Compress multiple memories using specified strategy"""
        
        if not memories:
            return {}
        
        if strategy == ConsolidationStrategy.PATTERN_EXTRACTION:
            return self._compress_by_patterns(memories)
        elif strategy == ConsolidationStrategy.FREQUENCY_BASED:
            return self._compress_by_frequency(memories)
        elif strategy == ConsolidationStrategy.SIMILARITY_CLUSTERING:
            return self._compress_by_similarity(memories)
        elif strategy == ConsolidationStrategy.RULE_INDUCTION:
            return self._compress_by_rules(memories)
        elif strategy == ConsolidationStrategy.ABSTRACTION_HIERARCHY:
            return self._compress_by_abstraction(memories)
        else:
            return self._compress_by_patterns(memories)  # Default
    
    def _compress_by_patterns(self, memories: List[MemoryItem]) -> Dict[str, Any]:
        """Compress memories by extracting patterns"""
        
        pattern_extractor = PatternExtractor()
        patterns = pattern_extractor.extract_patterns(memories)
        
        # Organize patterns by type
        pattern_summary = defaultdict(list)
        
        for pattern in patterns:
            pattern_summary[pattern['type']].append(pattern)
        
        # Create compressed representation
        compressed = {
            'compression_type': 'pattern_based',
            'source_count': len(memories),
            'patterns': dict(pattern_summary),
            'pattern_count': len(patterns),
            'confidence_score': self._calculate_pattern_confidence(patterns),
            'summary': self._create_pattern_summary(patterns)
        }
        
        return compressed
    
    def _compress_by_frequency(self, memories: List[MemoryItem]) -> Dict[str, Any]:
        """Compress memories by frequency analysis"""
        
        # Count frequency of all attributes and values
        attribute_frequencies = defaultdict(Counter)
        
        for memory in memories:
            for key, value in memory.content.items():
                if isinstance(value, (str, int, float, bool)):
                    attribute_frequencies[key][str(value)] += 1
        
        # Extract most frequent patterns
        frequent_patterns = {}
        
        for attribute, value_counts in attribute_frequencies.items():
            total_count = sum(value_counts.values())
            
            # Get most frequent value
            most_frequent_value, frequency = value_counts.most_common(1)[0]
            frequency_ratio = frequency / total_count
            
            if frequency_ratio >= 0.3:  # 30% threshold
                frequent_patterns[attribute] = {
                    'value': most_frequent_value,
                    'frequency': frequency,
                    'frequency_ratio': frequency_ratio,
                    'alternatives': dict(value_counts.most_common(3))
                }
        
        compressed = {
            'compression_type': 'frequency_based',
            'source_count': len(memories),
            'frequent_patterns': frequent_patterns,
            'confidence_score': self._calculate_frequency_confidence(frequent_patterns),
            'summary': f"Identified {len(frequent_patterns)} frequent patterns from {len(memories)} memories"
        }
        
        return compressed
    
    def _compress_by_similarity(self, memories: List[MemoryItem]) -> Dict[str, Any]:
        """Compress memories by clustering similar ones"""
        
        # Calculate similarity matrix
        similarity_clusters = self._cluster_similar_memories(memories)
        
        # Compress each cluster
        compressed_clusters = []
        
        for cluster in similarity_clusters:
            if len(cluster) >= 2:
                # Find common attributes in cluster
                common_attributes = self._find_common_attributes(cluster)
                
                cluster_summary = {
                    'cluster_size': len(cluster),
                    'common_attributes': common_attributes,
                    'representative_memory': cluster[0].id,
                    'similarity_score': self._calculate_cluster_similarity(cluster)
                }
                
                compressed_clusters.append(cluster_summary)
        
        compressed = {
            'compression_type': 'similarity_based',
            'source_count': len(memories),
            'clusters': compressed_clusters,
            'cluster_count': len(compressed_clusters),
            'confidence_score': self._calculate_similarity_confidence(compressed_clusters),
            'summary': f"Grouped {len(memories)} memories into {len(compressed_clusters)} similarity clusters"
        }
        
        return compressed
    
    def _compress_by_rules(self, memories: List[MemoryItem]) -> Dict[str, Any]:
        """Compress memories by inducing rules"""
        
        # Extract conditions and outcomes
        rules = []
        
        # Group memories by outcomes
        outcome_groups = defaultdict(list)
        
        for memory in memories:
            outcome = memory.content.get('outcome', memory.content.get('result', 'unknown'))
            outcome_groups[str(outcome)].append(memory)
        
        # For each outcome, find common conditions
        for outcome, outcome_memories in outcome_groups.items():
            if len(outcome_memories) >= 2:
                
                # Find common conditions
                condition_counts = defaultdict(int)
                
                for memory in outcome_memories:
                    for key, value in memory.content.items():
                        if key not in ['outcome', 'result'] and isinstance(value, (str, int, float, bool)):
                            condition_counts[f"{key}={value}"] += 1
                
                # Create rules for frequent conditions
                total_outcomes = len(outcome_memories)
                
                for condition, count in condition_counts.items():
                    frequency = count / total_outcomes
                    
                    if frequency >= 0.6:  # 60% threshold for rule
                        rules.append({
                            'condition': condition,
                            'outcome': outcome,
                            'frequency': frequency,
                            'support': count,
                            'confidence': frequency,
                            'rule': f"IF {condition} THEN {outcome} ({frequency:.1%} confidence)"
                        })
        
        compressed = {
            'compression_type': 'rule_based',
            'source_count': len(memories),
            'rules': rules,
            'rule_count': len(rules),
            'confidence_score': self._calculate_rule_confidence(rules),
            'summary': f"Induced {len(rules)} rules from {len(memories)} memories"
        }
        
        return compressed
    
    def _compress_by_abstraction(self, memories: List[MemoryItem]) -> Dict[str, Any]:
        """Compress memories by creating abstraction hierarchy"""
        
        # Create hierarchical categories
        categories = defaultdict(lambda: defaultdict(list))
        
        for memory in memories:
            # Categorize by memory type and context
            memory_type = memory.memory_type.value
            
            # Use context tags as subcategories
            if memory.context_tags:
                for tag in memory.context_tags:
                    categories[memory_type][tag].append(memory)
            else:
                categories[memory_type]['general'].append(memory)
        
        # Create abstraction hierarchy
        hierarchy = {}
        
        for category, subcategories in categories.items():
            category_abstraction = {}
            
            for subcategory, subcategory_memories in subcategories.items():
                # Abstract common features
                common_features = self._abstract_common_features(subcategory_memories)
                
                category_abstraction[subcategory] = {
                    'memory_count': len(subcategory_memories),
                    'common_features': common_features,
                    'abstraction_level': self._calculate_abstraction_level(common_features),
                    'representative_memories': [m.id for m in subcategory_memories[:3]]
                }
            
            hierarchy[category] = category_abstraction
        
        compressed = {
            'compression_type': 'abstraction_hierarchy',
            'source_count': len(memories),
            'hierarchy': hierarchy,
            'category_count': len(hierarchy),
            'confidence_score': self._calculate_abstraction_confidence(hierarchy),
            'summary': f"Created {len(hierarchy)} category hierarchy from {len(memories)} memories"
        }
        
        return compressed
    
    def _cluster_similar_memories(self, memories: List[MemoryItem]) -> List[List[MemoryItem]]:
        """Cluster memories by similarity"""
        
        # Simple clustering based on content similarity
        clusters = []
        used_memories = set()
        
        for i, memory1 in enumerate(memories):
            if memory1.id in used_memories:
                continue
            
            cluster = [memory1]
            used_memories.add(memory1.id)
            
            for j, memory2 in enumerate(memories[i + 1:], i + 1):
                if memory2.id in used_memories:
                    continue
                
                similarity = self._calculate_content_similarity(memory1, memory2)
                
                if similarity >= 0.7:  # 70% similarity threshold
                    cluster.append(memory2)
                    used_memories.add(memory2.id)
            
            clusters.append(cluster)
        
        return clusters
    
    def _calculate_content_similarity(self, memory1: MemoryItem, memory2: MemoryItem) -> float:
        """Calculate similarity between two memories"""
        
        content1 = memory1.content
        content2 = memory2.content
        
        # Get all keys from both memories
        all_keys = set(content1.keys()) | set(content2.keys())
        
        if not all_keys:
            return 0.0
        
        matching_attributes = 0
        
        for key in all_keys:
            value1 = content1.get(key)
            value2 = content2.get(key)
            
            if value1 == value2:
                matching_attributes += 1
            elif value1 is not None and value2 is not None:
                # Partial similarity for different values
                matching_attributes += 0.5
        
        similarity = matching_attributes / len(all_keys)
        
        return similarity
    
    def _find_common_attributes(self, memories: List[MemoryItem]) -> Dict[str, Any]:
        """Find attributes common to all memories in cluster"""
        
        if not memories:
            return {}
        
        # Start with first memory's attributes
        common_attributes = dict(memories[0].content)
        
        # Remove attributes that don't match in other memories
        for memory in memories[1:]:
            keys_to_remove = []
            
            for key, value in common_attributes.items():
                if key not in memory.content or memory.content[key] != value:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del common_attributes[key]
        
        return common_attributes
    
    def _abstract_common_features(self, memories: List[MemoryItem]) -> Dict[str, Any]:
        """Abstract common features from memories"""
        
        # Count frequency of each attribute-value pair
        feature_counts = defaultdict(Counter)
        
        for memory in memories:
            for key, value in memory.content.items():
                if isinstance(value, (str, int, float, bool)):
                    feature_counts[key][str(value)] += 1
        
        # Abstract features that appear in majority of memories
        common_features = {}
        threshold = len(memories) * 0.6  # 60% threshold
        
        for attribute, value_counts in feature_counts.items():
            for value, count in value_counts.items():
                if count >= threshold:
                    common_features[attribute] = {
                        'value': value,
                        'frequency': count / len(memories),
                        'abstraction_type': 'frequent_value'
                    }
                    break  # Take most frequent value
        
        return common_features
    
    def _calculate_pattern_confidence(self, patterns: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for extracted patterns"""
        
        if not patterns:
            return 0.0
        
        total_confidence = sum(pattern.get('confidence', 0.0) for pattern in patterns)
        
        return total_confidence / len(patterns)
    
    def _calculate_frequency_confidence(self, frequent_patterns: Dict[str, Any]) -> float:
        """Calculate confidence score for frequency-based compression"""
        
        if not frequent_patterns:
            return 0.0
        
        total_frequency = sum(pattern['frequency_ratio'] for pattern in frequent_patterns.values())
        
        return total_frequency / len(frequent_patterns)
    
    def _calculate_similarity_confidence(self, clusters: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for similarity-based compression"""
        
        if not clusters:
            return 0.0
        
        total_similarity = sum(cluster.get('similarity_score', 0.0) for cluster in clusters)
        
        return total_similarity / len(clusters)
    
    def _calculate_rule_confidence(self, rules: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for rule-based compression"""
        
        if not rules:
            return 0.0
        
        total_confidence = sum(rule.get('confidence', 0.0) for rule in rules)
        
        return total_confidence / len(rules)
    
    def _calculate_abstraction_confidence(self, hierarchy: Dict[str, Any]) -> float:
        """Calculate confidence score for abstraction-based compression"""
        
        if not hierarchy:
            return 0.0
        
        total_abstraction = 0
        category_count = 0
        
        for category, subcategories in hierarchy.items():
            for subcategory, data in subcategories.items():
                total_abstraction += data.get('abstraction_level', 0.0)
                category_count += 1
        
        return total_abstraction / category_count if category_count > 0 else 0.0
    
    def _calculate_cluster_similarity(self, cluster: List[MemoryItem]) -> float:
        """Calculate average similarity within a cluster"""
        
        if len(cluster) < 2:
            return 1.0
        
        total_similarity = 0
        pair_count = 0
        
        for i, memory1 in enumerate(cluster):
            for memory2 in cluster[i + 1:]:
                total_similarity += self._calculate_content_similarity(memory1, memory2)
                pair_count += 1
        
        return total_similarity / pair_count if pair_count > 0 else 0.0
    
    def _calculate_abstraction_level(self, common_features: Dict[str, Any]) -> float:
        """Calculate abstraction level of common features"""
        
        if not common_features:
            return 0.0
        
        # Higher abstraction = more general features with high frequency
        total_abstraction = 0
        
        for feature_data in common_features.values():
            frequency = feature_data.get('frequency', 0.0)
            # Higher frequency = higher abstraction
            total_abstraction += frequency
        
        return total_abstraction / len(common_features)
    
    def _create_pattern_summary(self, patterns: List[Dict[str, Any]]) -> str:
        """Create text summary of extracted patterns"""
        
        if not patterns:
            return "No patterns found"
        
        pattern_types = defaultdict(int)
        
        for pattern in patterns:
            pattern_types[pattern['type']] += 1
        
        summary_parts = []
        
        for pattern_type, count in pattern_types.items():
            summary_parts.append(f"{count} {pattern_type.replace('_', ' ')} patterns")
        
        return f"Found {len(patterns)} total patterns: " + ", ".join(summary_parts)

class MemoryConsolidationEngine:
    """Main engine for memory consolidation processes"""
    
    def __init__(self):
        # Core components
        self.pattern_extractor = PatternExtractor()
        self.memory_compressor = MemoryCompressor()
        
        # Memory storage
        self.raw_memories: Dict[str, MemoryItem] = {}
        self.consolidated_memories: Dict[str, ConsolidatedMemory] = {}
        
        # Consolidation tracking
        self.consolidation_schedule: List[Dict[str, Any]] = []
        self.consolidation_history: List[Dict[str, Any]] = []
        
        # Configuration
        self.consolidation_threshold = 5  # Minimum memories for consolidation
        self.similarity_threshold = 0.7
        self.pattern_confidence_threshold = 0.6
        
        # Statistics
        self.stats = {
            'memories_processed': 0,
            'consolidations_performed': 0,
            'patterns_extracted': 0,
            'memories_compressed': 0,
            'storage_saved': 0
        }
        
        self.logger = logging.getLogger("MemoryConsolidationEngine")
    
    async def initialize(self) -> None:
        """Initialize the consolidation engine"""
        self.logger.info("Memory consolidation engine initialized")
    
    def add_memory(self, content: Dict[str, Any], memory_type: MemoryType,
                  importance_score: float = 1.0, context_tags: Set[str] = None) -> str:
        """Add a new memory for potential consolidation"""
        
        memory = MemoryItem(
            id="",
            content=content,
            memory_type=memory_type,
            importance_score=importance_score,
            context_tags=context_tags or set()
        )
        
        self.raw_memories[memory.id] = memory
        self.stats['memories_processed'] += 1
        
        # Check if consolidation should be triggered
        self._check_consolidation_triggers(memory)
        
        self.logger.debug(f"Added memory {memory.id[:8]}... for consolidation")
        
        return memory.id
    
    def _check_consolidation_triggers(self, new_memory: MemoryItem) -> None:
        """Check if new memory triggers consolidation"""
        
        # Frequency trigger: similar memories
        similar_memories = self._find_similar_memories(new_memory)
        
        if len(similar_memories) >= self.consolidation_threshold:
            self._schedule_consolidation(
                similar_memories,
                ConsolidationTrigger.FREQUENCY,
                ConsolidationStrategy.PATTERN_EXTRACTION
            )
        
        # Context trigger: memories with same context tags
        if new_memory.context_tags:
            context_memories = self._find_memories_by_context(new_memory.context_tags)
            
            if len(context_memories) >= self.consolidation_threshold:
                self._schedule_consolidation(
                    context_memories,
                    ConsolidationTrigger.SIMILARITY,
                    ConsolidationStrategy.SIMILARITY_CLUSTERING
                )
        
        # Time-based trigger: check for pending consolidations
        self._check_time_based_consolidation()
    
    def _find_similar_memories(self, target_memory: MemoryItem) -> List[MemoryItem]:
        """Find memories similar to target memory"""
        
        similar_memories = [target_memory]
        
        for memory in self.raw_memories.values():
            if memory.id != target_memory.id and not memory.is_consolidated:
                similarity = self.memory_compressor._calculate_content_similarity(
                    target_memory, memory
                )
                
                if similarity >= self.similarity_threshold:
                    similar_memories.append(memory)
        
        return similar_memories
    
    def _find_memories_by_context(self, context_tags: Set[str]) -> List[MemoryItem]:
        """Find memories with overlapping context tags"""
        
        context_memories = []
        
        for memory in self.raw_memories.values():
            if not memory.is_consolidated and memory.context_tags & context_tags:
                context_memories.append(memory)
        
        return context_memories
    
    def _schedule_consolidation(self, memories: List[MemoryItem],
                              trigger: ConsolidationTrigger,
                              strategy: ConsolidationStrategy) -> None:
        """Schedule a consolidation task"""
        
        consolidation_task = {
            'id': str(uuid.uuid4()),
            'memories': [m.id for m in memories],
            'trigger': trigger,
            'strategy': strategy,
            'scheduled_time': datetime.now(),
            'priority': self._calculate_consolidation_priority(memories, trigger),
            'status': 'scheduled'
        }
        
        self.consolidation_schedule.append(consolidation_task)
        
        # Sort schedule by priority
        self.consolidation_schedule.sort(key=lambda x: x['priority'], reverse=True)
        
        self.logger.debug(f"Scheduled consolidation for {len(memories)} memories "
                         f"(trigger: {trigger.value}, strategy: {strategy.value})")
    
    def _calculate_consolidation_priority(self, memories: List[MemoryItem],
                                        trigger: ConsolidationTrigger) -> float:
        """Calculate priority for consolidation task"""
        
        # Base priority from trigger type
        trigger_priority = {
            ConsolidationTrigger.IMPORTANCE: 1.0,
            ConsolidationTrigger.CONFLICT: 0.9,
            ConsolidationTrigger.FREQUENCY: 0.8,
            ConsolidationTrigger.SIMILARITY: 0.7,
            ConsolidationTrigger.TIME_BASED: 0.6,
            ConsolidationTrigger.MANUAL: 0.5
        }
        
        base_priority = trigger_priority.get(trigger, 0.5)
        
        # Adjust based on memory characteristics
        avg_importance = sum(m.importance_score for m in memories) / len(memories)
        avg_age_hours = sum((datetime.now() - m.timestamp).total_seconds() / 3600 for m in memories) / len(memories)
        
        # Higher importance = higher priority
        importance_factor = avg_importance
        
        # Older memories get higher priority (need consolidation)
        age_factor = min(1.0, avg_age_hours / 24.0)  # Normalize to days
        
        # More memories = higher priority
        quantity_factor = min(1.0, len(memories) / 10.0)
        
        priority = (base_priority * 0.4 + 
                   importance_factor * 0.3 + 
                   age_factor * 0.2 + 
                   quantity_factor * 0.1)
        
        return priority
    
    def _check_time_based_consolidation(self) -> None:
        """Check for time-based consolidation opportunities"""
        
        # Group memories by age
        age_groups = defaultdict(list)
        
        for memory in self.raw_memories.values():
            if not memory.is_consolidated:
                age_hours = (datetime.now() - memory.timestamp).total_seconds() / 3600
                
                if age_hours >= 24:  # Older than 1 day
                    age_groups['old'].append(memory)
                elif age_hours >= 6:  # 6-24 hours
                    age_groups['medium'].append(memory)
                else:  # Less than 6 hours
                    age_groups['recent'].append(memory)
        
        # Schedule consolidation for old memories
        if len(age_groups['old']) >= self.consolidation_threshold:
            self._schedule_consolidation(
                age_groups['old'],
                ConsolidationTrigger.TIME_BASED,
                ConsolidationStrategy.ABSTRACTION_HIERARCHY
            )
    
    async def process_consolidation_queue(self, max_tasks: int = 5) -> List[Dict[str, Any]]:
        """Process pending consolidation tasks"""
        
        processed_tasks = []
        tasks_processed = 0
        
        while self.consolidation_schedule and tasks_processed < max_tasks:
            task = self.consolidation_schedule.pop(0)
            
            try:
                result = await self._execute_consolidation_task(task)
                
                if result:
                    processed_tasks.append(result)
                    tasks_processed += 1
                    
                    self.logger.debug(f"Completed consolidation task {task['id'][:8]}...")
                
            except Exception as e:
                self.logger.error(f"Consolidation task failed: {e}")
                task['status'] = 'failed'
                task['error'] = str(e)
        
        return processed_tasks
    
    async def _execute_consolidation_task(self, task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute a single consolidation task"""
        
        # Get memories for consolidation
        memory_ids = task['memories']
        memories = [self.raw_memories[mid] for mid in memory_ids if mid in self.raw_memories]
        
        if len(memories) < 2:
            return None
        
        # Perform consolidation
        strategy = task['strategy']
        compressed_content = self.memory_compressor.compress_memories(memories, strategy)
        
        if not compressed_content:
            return None
        
        # Create consolidated memory
        consolidated_memory = ConsolidatedMemory(
            id="",
            consolidated_content=compressed_content,
            memory_type=memories[0].memory_type,  # Use first memory's type
            source_memories={m.id for m in memories},
            consolidation_strategy=strategy,
            confidence_score=compressed_content.get('confidence_score', 0.0),
            compression_ratio=len(memories)  # Simple ratio
        )
        
        # Extract patterns if available
        if 'patterns' in compressed_content:
            patterns = compressed_content['patterns']
            consolidated_memory.derived_patterns = [
                pattern.get('description', str(pattern)) 
                for pattern_list in patterns.values() 
                for pattern in pattern_list
            ]
        
        # Store consolidated memory
        self.consolidated_memories[consolidated_memory.id] = consolidated_memory
        
        # Mark source memories as consolidated
        for memory in memories:
            memory.is_consolidated = True
            memory.consolidation_count += 1
            memory.contributes_to.add(consolidated_memory.id)
        
        # Update statistics
        self.stats['consolidations_performed'] += 1
        self.stats['memories_compressed'] += len(memories)
        self.stats['patterns_extracted'] += len(consolidated_memory.derived_patterns)
        
        # Record consolidation
        consolidation_record = {
            'task_id': task['id'],
            'consolidated_memory_id': consolidated_memory.id,
            'source_memory_count': len(memories),
            'strategy': strategy.value,
            'trigger': task['trigger'].value,
            'timestamp': datetime.now(),
            'quality_metrics': {
                'confidence_score': consolidated_memory.confidence_score,
                'compression_ratio': consolidated_memory.compression_ratio,
                'pattern_count': len(consolidated_memory.derived_patterns)
            }
        }
        
        self.consolidation_history.append(consolidation_record)
        
        return consolidation_record
    
    async def consolidate_by_pattern(self, pattern_type: str) -> List[str]:
        """Manually consolidate memories that match a specific pattern"""
        
        # Find memories matching pattern
        matching_memories = []
        
        for memory in self.raw_memories.values():
            if not memory.is_consolidated:
                # Simple pattern matching (can be made more sophisticated)
                content_str = str(memory.content).lower()
                
                if pattern_type.lower() in content_str:
                    matching_memories.append(memory)
        
        if len(matching_memories) < 2:
            return []
        
        # Perform consolidation
        compressed_content = self.memory_compressor.compress_memories(
            matching_memories, ConsolidationStrategy.PATTERN_EXTRACTION
        )
        
        if not compressed_content:
            return []
        
        # Create consolidated memory
        consolidated_memory = ConsolidatedMemory(
            id="",
            consolidated_content=compressed_content,
            memory_type=MemoryType.PATTERN,
            source_memories={m.id for m in matching_memories},
            consolidation_strategy=ConsolidationStrategy.PATTERN_EXTRACTION
        )
        
        self.consolidated_memories[consolidated_memory.id] = consolidated_memory
        
        # Mark source memories as consolidated
        for memory in matching_memories:
            memory.is_consolidated = True
        
        self.logger.info(f"Manually consolidated {len(matching_memories)} memories by pattern '{pattern_type}'")
        
        return [consolidated_memory.id]
    
    def query_consolidated_memories(self, query: str, memory_type: Optional[MemoryType] = None,
                                  max_results: int = 10) -> List[ConsolidatedMemory]:
        """Query consolidated memories"""
        
        results = []
        
        for consolidated in self.consolidated_memories.values():
            # Filter by memory type if specified
            if memory_type and consolidated.memory_type != memory_type:
                continue
            
            # Simple text matching (can be improved with semantic search)
            content_str = str(consolidated.consolidated_content).lower()
            
            if query.lower() in content_str:
                consolidated.update_retrieval()
                results.append(consolidated)
        
        # Sort by relevance (retrieval count and confidence)
        results.sort(key=lambda x: (x.retrieval_count, x.confidence_score), reverse=True)
        
        return results[:max_results]
    
    def get_consolidation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive consolidation statistics"""
        
        # Analyze current state
        total_raw_memories = len(self.raw_memories)
        unconsolidated_memories = len([m for m in self.raw_memories.values() if not m.is_consolidated])
        total_consolidated = len(self.consolidated_memories)
        
        # Memory type distribution
        raw_type_distribution = defaultdict(int)
        consolidated_type_distribution = defaultdict(int)
        
        for memory in self.raw_memories.values():
            raw_type_distribution[memory.memory_type.value] += 1
        
        for consolidated in self.consolidated_memories.values():
            consolidated_type_distribution[consolidated.memory_type.value] += 1
        
        # Consolidation efficiency
        if total_raw_memories > 0:
            consolidation_ratio = total_consolidated / total_raw_memories
            compression_efficiency = (total_raw_memories - unconsolidated_memories) / total_raw_memories
        else:
            consolidation_ratio = 0.0
            compression_efficiency = 0.0
        
        # Pattern statistics
        all_patterns = []
        for consolidated in self.consolidated_memories.values():
            all_patterns.extend(consolidated.derived_patterns)
        
        return {
            'memory_counts': {
                'total_raw_memories': total_raw_memories,
                'unconsolidated_memories': unconsolidated_memories,
                'consolidated_memories': total_consolidated,
                'pending_consolidations': len(self.consolidation_schedule)
            },
            'type_distributions': {
                'raw_memories': dict(raw_type_distribution),
                'consolidated_memories': dict(consolidated_type_distribution)
            },
            'efficiency_metrics': {
                'consolidation_ratio': consolidation_ratio,
                'compression_efficiency': compression_efficiency,
                'average_compression_ratio': sum(c.compression_ratio for c in self.consolidated_memories.values()) / len(self.consolidated_memories) if self.consolidated_memories else 0.0
            },
            'pattern_metrics': {
                'total_patterns_extracted': len(all_patterns),
                'average_patterns_per_consolidation': len(all_patterns) / len(self.consolidated_memories) if self.consolidated_memories else 0.0
            },
            'performance_stats': self.stats,
            'consolidation_history_count': len(self.consolidation_history)
        }
    
    def cleanup_old_memories(self, retention_days: int = 30) -> int:
        """Clean up old unconsolidated memories"""
        
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        memories_removed = 0
        
        memories_to_remove = []
        
        for memory_id, memory in self.raw_memories.items():
            if (not memory.is_consolidated and 
                memory.timestamp < cutoff_date and
                memory.calculate_retention_score() < 0.3):  # Low retention score
                
                memories_to_remove.append(memory_id)
        
        # Remove memories
        for memory_id in memories_to_remove:
            del self.raw_memories[memory_id]
            memories_removed += 1
        
        self.logger.info(f"Cleaned up {memories_removed} old unconsolidated memories")
        
        return memories_removed

# EXAMPLE DEMONSTRATIONS
# =====================

async def demo_basic_consolidation():
    """Demo: Basic memory consolidation"""
    print("\nDEMO 1: BASIC MEMORY CONSOLIDATION")
    print("=" * 50)
    
    engine = MemoryConsolidationEngine()
    await engine.initialize()
    
    print("Adding user preference memories:")
    
    # Add user preference memories
    preferences = [
        {"user": "john", "preference": "coffee", "context": "morning", "strength": "strong"},
        {"user": "john", "preference": "coffee", "context": "afternoon", "strength": "medium"},
        {"user": "john", "preference": "tea", "context": "evening", "strength": "light"},
        {"user": "john", "preference": "coffee", "context": "morning", "strength": "strong"},
        {"user": "john", "preference": "coffee", "context": "work", "strength": "strong"},
        {"user": "sarah", "preference": "tea", "context": "morning", "strength": "strong"},
        {"user": "sarah", "preference": "tea", "context": "afternoon", "strength": "medium"},
    ]
    
    memory_ids = []
    
    for i, pref in enumerate(preferences, 1):
        memory_id = engine.add_memory(
            pref, MemoryType.PREFERENCE, 
            importance_score=0.8,
            context_tags={"user_preferences", pref["context"]}
        )
        memory_ids.append(memory_id)
        print(f"  {i}. Added preference: {pref['user']} likes {pref['preference']} in {pref['context']}")
    
    print(f"\nTotal memories added: {len(memory_ids)}")
    print(f"Pending consolidations: {len(engine.consolidation_schedule)}")
    
    # Process consolidation queue
    print(f"\nProcessing consolidation queue:")
    
    consolidation_results = await engine.process_consolidation_queue()
    
    for result in consolidation_results:
        print(f"  Consolidated {result['source_memory_count']} memories")
        print(f"    Strategy: {result['strategy']}")
        print(f"    Trigger: {result['trigger']}")
        print(f"    Confidence: {result['quality_metrics']['confidence_score']:.2f}")
        print(f"    Patterns found: {result['quality_metrics']['pattern_count']}")
    
    # Show consolidated memories
    print(f"\nConsolidated memories:")
    
    for consolidated_id, consolidated in engine.consolidated_memories.items():
        print(f"  {consolidated_id[:8]}... - {consolidated.memory_type.value}")
        print(f"    Source memories: {len(consolidated.source_memories)}")
        print(f"    Patterns: {len(consolidated.derived_patterns)}")
        
        if consolidated.derived_patterns:
            for pattern in consolidated.derived_patterns[:3]:  # Show first 3 patterns
                print(f"      • {pattern}")
        
        print()

async def demo_pattern_extraction():
    """Demo: Pattern extraction from experiences"""
    print("\nDEMO 2: PATTERN EXTRACTION")
    print("=" * 50)
    
    engine = MemoryConsolidationEngine()
    await engine.initialize()
    
    print("Adding task completion experiences:")
    
    # Add task completion memories with patterns
    task_experiences = [
        {"task": "email", "time": "morning", "duration": 15, "interruptions": 0, "completion": "success"},
        {"task": "coding", "time": "morning", "duration": 120, "interruptions": 1, "completion": "success"},
        {"task": "meeting", "time": "afternoon", "duration": 60, "interruptions": 3, "completion": "partial"},
        {"task": "email", "time": "morning", "duration": 10, "interruptions": 0, "completion": "success"},
        {"task": "coding", "time": "afternoon", "duration": 90, "interruptions": 4, "completion": "partial"},
        {"task": "documentation", "time": "evening", "duration": 45, "interruptions": 0, "completion": "success"},
        {"task": "email", "time": "afternoon", "duration": 20, "interruptions": 2, "completion": "partial"},
        {"task": "coding", "time": "morning", "duration": 150, "interruptions": 0, "completion": "success"},
    ]
    
    for i, exp in enumerate(task_experiences, 1):
        memory_id = engine.add_memory(
            exp, MemoryType.EPISODIC,
            importance_score=0.7,
            context_tags={"work_patterns", exp["task"], exp["time"]}
        )
        print(f"  {i}. {exp['task']} in {exp['time']}: {exp['completion']} ({exp['duration']}min, {exp['interruptions']} interruptions)")
    
    print(f"\nProcessing pattern extraction:")
    
    # Force consolidation by processing queue
    results = await engine.process_consolidation_queue(max_tasks=3)
    
    for result in results:
        print(f"\nConsolidation Result:")
        print(f"  Memories processed: {result['source_memory_count']}")
        print(f"  Strategy: {result['strategy']}")
        print(f"  Confidence: {result['quality_metrics']['confidence_score']:.2f}")
        
        # Get the consolidated memory to show patterns
        consolidated_id = result['consolidated_memory_id']
        consolidated = engine.consolidated_memories[consolidated_id]
        
        print(f"  Patterns extracted:")
        for pattern in consolidated.derived_patterns:
            print(f"    • {pattern}")
    
    # Query for specific patterns
    print(f"\nQuerying for 'morning' patterns:")
    
    morning_results = engine.query_consolidated_memories("morning", max_results=3)
    
    for result in morning_results:
        print(f"  Found: {result.memory_type.value} memory")
        print(f"    Confidence: {result.confidence_score:.2f}")
        print(f"    Retrieval count: {result.retrieval_count}")
        print(f"    Pattern summary: {result.consolidated_content.get('summary', 'No summary')}")

async def demo_rule_induction():
    """Demo: Rule induction from repeated patterns"""
    print("\nDEMO 3: RULE INDUCTION")
    print("=" * 50)
    
    engine = MemoryConsolidationEngine()
    await engine.initialize()
    
    print("Adding cause-effect experiences:")
    
    # Add cause-effect memories for rule induction
    experiences = [
        {"condition": "rainy_weather", "action": "work_from_home", "outcome": "productive_day", "satisfaction": "high"},
        {"condition": "sunny_weather", "action": "work_from_office", "outcome": "collaborative_day", "satisfaction": "high"},
        {"condition": "rainy_weather", "action": "work_from_home", "outcome": "productive_day", "satisfaction": "high"},
        {"condition": "meeting_heavy_day", "action": "block_focus_time", "outcome": "completed_tasks", "satisfaction": "medium"},
        {"condition": "deadline_pressure", "action": "skip_meetings", "outcome": "on_time_delivery", "satisfaction": "high"},
        {"condition": "rainy_weather", "action": "work_from_office", "outcome": "commute_stress", "satisfaction": "low"},
        {"condition": "meeting_heavy_day", "action": "attend_all_meetings", "outcome": "incomplete_tasks", "satisfaction": "low"},
        {"condition": "deadline_pressure", "action": "skip_meetings", "outcome": "on_time_delivery", "satisfaction": "high"},
        {"condition": "sunny_weather", "action": "work_from_office", "outcome": "collaborative_day", "satisfaction": "high"},
    ]
    
    for i, exp in enumerate(experiences, 1):
        memory_id = engine.add_memory(
            exp, MemoryType.EPISODIC,
            importance_score=0.8,
            context_tags={"decision_patterns", exp["condition"]}
        )
        print(f"  {i}. {exp['condition']} → {exp['action']} → {exp['outcome']} (satisfaction: {exp['satisfaction']})")
    
    # Manually trigger rule-based consolidation
    print(f"\nInducing rules from experiences:")
    
    # Get all episodic memories for rule induction
    episodic_memories = [m for m in engine.raw_memories.values() if m.memory_type == MemoryType.EPISODIC]
    
    # Use rule induction strategy
    compressed_content = engine.memory_compressor.compress_memories(
        episodic_memories, ConsolidationStrategy.RULE_INDUCTION
    )
    
    print(f"Rules discovered:")
    
    if 'rules' in compressed_content:
        for rule in compressed_content['rules']:
            print(f"  • {rule['rule']}")
            print(f"    Support: {rule['support']} cases")
            print(f"    Confidence: {rule['confidence']:.1%}")
            print()
    
    # Create consolidated memory with rules
    consolidated_memory = ConsolidatedMemory(
        id="",
        consolidated_content=compressed_content,
        memory_type=MemoryType.RULE,
        consolidation_strategy=ConsolidationStrategy.RULE_INDUCTION,
        confidence_score=compressed_content.get('confidence_score', 0.0)
    )
    
    engine.consolidated_memories[consolidated_memory.id] = consolidated_memory
    
    print(f"Consolidated {len(episodic_memories)} experiences into {len(compressed_content.get('rules', []))} rules")
    print(f"Overall confidence: {compressed_content.get('confidence_score', 0.0):.2f}")

async def demo_hierarchical_abstraction():
    """Demo: Hierarchical abstraction of knowledge"""
    print("\nDEMO 4: HIERARCHICAL ABSTRACTION")
    print("=" * 50)
    
    engine = MemoryConsolidationEngine()
    await engine.initialize()
    
    print("Adding domain knowledge memories:")
    
    # Add memories from different domains and levels
    knowledge_items = [
        # Programming knowledge
        {"domain": "programming", "level": "syntax", "concept": "for_loop", "language": "python", "difficulty": "basic"},
        {"domain": "programming", "level": "syntax", "concept": "list_comprehension", "language": "python", "difficulty": "intermediate"},
        {"domain": "programming", "level": "paradigm", "concept": "object_oriented", "language": "python", "difficulty": "intermediate"},
        {"domain": "programming", "level": "paradigm", "concept": "functional", "language": "python", "difficulty": "advanced"},
        {"domain": "programming", "level": "algorithm", "concept": "sorting", "language": "python", "difficulty": "intermediate"},
        
        # Data science knowledge
        {"domain": "data_science", "level": "tool", "concept": "pandas", "language": "python", "difficulty": "intermediate"},
        {"domain": "data_science", "level": "tool", "concept": "numpy", "language": "python", "difficulty": "basic"},
        {"domain": "data_science", "level": "concept", "concept": "regression", "language": "python", "difficulty": "intermediate"},
        {"domain": "data_science", "level": "concept", "concept": "clustering", "language": "python", "difficulty": "advanced"},
        
        # Machine learning knowledge
        {"domain": "machine_learning", "level": "algorithm", "concept": "linear_regression", "language": "python", "difficulty": "basic"},
        {"domain": "machine_learning", "level": "algorithm", "concept": "neural_networks", "language": "python", "difficulty": "advanced"},
        {"domain": "machine_learning", "level": "concept", "concept": "supervised_learning", "language": "python", "difficulty": "intermediate"},
    ]
    
    for i, item in enumerate(knowledge_items, 1):
        memory_id = engine.add_memory(
            item, MemoryType.SEMANTIC,
            importance_score=0.6,
            context_tags={item["domain"], item["level"], item["difficulty"]}
        )
        print(f"  {i}. {item['domain']}.{item['level']}: {item['concept']} ({item['difficulty']})")
    
    print(f"\nCreating hierarchical abstraction:")
    
    # Get all semantic memories
    semantic_memories = [m for m in engine.raw_memories.values() if m.memory_type == MemoryType.SEMANTIC]
    
    # Use abstraction hierarchy strategy
    compressed_content = engine.memory_compressor.compress_memories(
        semantic_memories, ConsolidationStrategy.ABSTRACTION_HIERARCHY
    )
    
    print(f"Knowledge hierarchy created:")
    
    if 'hierarchy' in compressed_content:
        hierarchy = compressed_content['hierarchy']
        
        for domain, subcategories in hierarchy.items():
            print(f"\n{domain.upper()}:")
            
            for subcategory, data in subcategories.items():
                print(f"  {subcategory}:")
                print(f"    Memories: {data['memory_count']}")
                print(f"    Abstraction level: {data['abstraction_level']:.2f}")
                
                if data['common_features']:
                    print(f"    Common features:")
                    for feature, feature_data in data['common_features'].items():
                        print(f"      {feature}: {feature_data['value']} ({feature_data['frequency']:.1%})")
    
    print(f"\nHierarchy statistics:")
    print(f"  Categories: {compressed_content.get('category_count', 0)}")
    print(f"  Source memories: {compressed_content.get('source_count', 0)}")
    print(f"  Confidence: {compressed_content.get('confidence_score', 0.0):.2f}")

async def demo_memory_lifecycle():
    """Demo: Complete memory lifecycle with consolidation"""
    print("\nDEMO 5: COMPLETE MEMORY LIFECYCLE")
    print("=" * 50)
    
    engine = MemoryConsolidationEngine()
    await engine.initialize()
    
    print("Simulating complete memory lifecycle:")
    
    # Phase 1: Initial memory accumulation
    print(f"\nPhase 1: Memory accumulation")
    
    daily_activities = [
        {"activity": "email_check", "time": "09:00", "duration": 15, "mood": "neutral", "productivity": "medium"},
        {"activity": "standup_meeting", "time": "09:30", "duration": 15, "mood": "positive", "productivity": "high"},
        {"activity": "deep_work", "time": "10:00", "duration": 120, "mood": "focused", "productivity": "high"},
        {"activity": "lunch_break", "time": "12:00", "duration": 60, "mood": "relaxed", "productivity": "none"},
        {"activity": "code_review", "time": "13:00", "duration": 45, "mood": "analytical", "productivity": "high"},
        {"activity": "documentation", "time": "14:00", "duration": 90, "mood": "neutral", "productivity": "medium"},
        {"activity": "team_meeting", "time": "15:30", "duration": 60, "mood": "collaborative", "productivity": "medium"},
        {"activity": "email_check", "time": "16:30", "duration": 20, "mood": "tired", "productivity": "low"},
    ]
    
    for activity in daily_activities:
        memory_id = engine.add_memory(
            activity, MemoryType.EPISODIC,
            importance_score=0.6,
            context_tags={"daily_routine", activity["time"][:2] + "h"}  # Hour context
        )
        print(f"  {activity['time']}: {activity['activity']} ({activity['duration']}min, mood: {activity['mood']})")
    
    stats_phase1 = engine.get_consolidation_statistics()
    print(f"  Total memories: {stats_phase1['memory_counts']['total_raw_memories']}")
    print(f"  Pending consolidations: {stats_phase1['memory_counts']['pending_consolidations']}")
    
    # Phase 2: Automatic consolidation
    print(f"\nPhase 2: Automatic consolidation")
    
    consolidation_results = await engine.process_consolidation_queue()
    
    for result in consolidation_results:
        print(f"  Consolidated {result['source_memory_count']} memories using {result['strategy']}")
        print(f"    Confidence: {result['quality_metrics']['confidence_score']:.2f}")
        print(f"    Patterns: {result['quality_metrics']['pattern_count']}")
    
    # Phase 3: Pattern-based queries
    print(f"\nPhase 3: Pattern-based queries")
    
    queries = ["morning", "productivity", "meeting"]
    
    for query in queries:
        results = engine.query_consolidated_memories(query, max_results=2)
        print(f"  Query '{query}': {len(results)} results")
        
        for result in results:
            summary = result.consolidated_content.get('summary', 'No summary')
            print(f"    • {summary}")
    
    # Phase 4: Manual pattern consolidation
    print(f"\nPhase 4: Manual pattern consolidation")
    
    pattern_consolidations = await engine.consolidate_by_pattern("productivity")
    print(f"  Consolidated {len(pattern_consolidations)} memory groups by 'productivity' pattern")
    
    # Phase 5: Memory cleanup
    print(f"\nPhase 5: Memory maintenance")
    
    cleaned_count = engine.cleanup_old_memories(retention_days=0)  # Clean immediately for demo
    print(f"  Cleaned up {cleaned_count} old memories")
    
    # Final statistics
    print(f"\nFinal statistics:")
    
    final_stats = engine.get_consolidation_statistics()
    
    print(f"  Raw memories: {final_stats['memory_counts']['total_raw_memories']}")
    print(f"  Consolidated memories: {final_stats['memory_counts']['consolidated_memories']}")
    print(f"  Consolidation ratio: {final_stats['efficiency_metrics']['consolidation_ratio']:.2f}")
    print(f"  Compression efficiency: {final_stats['efficiency_metrics']['compression_efficiency']:.2f}")
    print(f"  Total patterns extracted: {final_stats['pattern_metrics']['total_patterns_extracted']}")
    print(f"  Consolidations performed: {final_stats['performance_stats']['consolidations_performed']}")

async def main():
    """
    Demonstrate Memory Consolidation for converting temporary information into permanent knowledge
    
    WHAT YOU'LL LEARN:
    ================
    1. How to implement pattern extraction from accumulated experiences
    2. How to build rule induction systems that learn from repetition
    3. How to create hierarchical knowledge organization through abstraction
    4. How to design compression algorithms for efficient knowledge storage
    5. How to manage the complete memory consolidation lifecycle
    6. How to build systems that learn and improve over time
    
    REAL WORLD APPLICATIONS:
    =======================
    - Personal AI assistants that learn user preferences over time
    - Customer service systems that improve from interaction patterns
    - Educational AI that adapts based on learning patterns
    - Recommendation systems that consolidate user behavior into preferences
    - Research assistants that build knowledge from multiple sources
    - Workflow optimization systems that learn from usage patterns
    """
    
    print("MEMORY CONSOLIDATION DEMONSTRATION")
    print("Converting temporary information into permanent knowledge!")
    
    await demo_basic_consolidation()
    await demo_pattern_extraction()
    await demo_rule_induction()
    await demo_hierarchical_abstraction()
    await demo_memory_lifecycle()
    
    print("\nWHAT WE LEARNED:")
    print("=" * 40)
    print("✓ Memory consolidation transforms raw experiences into structured knowledge")
    print("✓ Pattern extraction reveals hidden relationships in accumulated data")
    print("✓ Rule induction creates actionable insights from repeated patterns")
    print("✓ Hierarchical abstraction organizes knowledge into meaningful categories")
    print("✓ Compression algorithms reduce storage while preserving important information")
    print("✓ Complete lifecycle management enables continuous learning and improvement")
    print("\nTHE POWER OF MEMORY CONSOLIDATION:")
    print("- Enables AI to learn efficiently from accumulated experiences")
    print("- Creates human-like learning and knowledge development")
    print("- Reduces memory requirements while improving knowledge quality")
    print("- Supports transfer learning and generalization across domains")
    print("- Essential for AI systems that need to improve over time")

if __name__ == "__main__":
    asyncio.run(main())
