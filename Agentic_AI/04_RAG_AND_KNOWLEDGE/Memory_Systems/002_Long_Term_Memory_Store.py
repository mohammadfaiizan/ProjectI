#!/usr/bin/env python3
"""
Long-Term Memory Store: Persistent Knowledge Storage and Retrieval
================================================================

WHAT IS THE PROBLEM?
==================
AI agents lack persistent memory across sessions and interactions:
- Knowledge acquired during conversations is lost when the session ends
- No way to accumulate learning and experiences over time
- Cannot build upon previous interactions or maintain continuity
- Unable to develop preferences, patterns, or long-term relationships
- Missing institutional memory that enables continuous improvement
- No mechanism to store and retrieve lessons learned from past experiences

Example: Personal AI Assistant Without Long-Term Memory
WITHOUT PERSISTENT MEMORY (Traditional):
- User teaches assistant about their preferences and work patterns
- Session ends and all learned information is permanently lost
- Next session starts fresh with no knowledge of previous interactions
- User must repeatedly provide the same information and preferences
- Assistant cannot improve or adapt based on past experiences
- No continuity in relationship or understanding of user needs
- Result: Frustrating user experience, inefficient interactions, no learning

REAL WORLD EXAMPLE:
=================
How does human long-term memory enable learning and adaptation?

HUMAN LONG-TERM MEMORY SYSTEM:
1. ENCODING: Experiences are processed and stored in permanent memory
2. CONSOLIDATION: Important memories are strengthened through repetition
3. ORGANIZATION: Information is categorized and linked for efficient retrieval
4. RETRIEVAL: Past experiences inform current decisions and actions
5. ADAPTATION: Learning from mistakes and successes shapes future behavior
6. PATTERN RECOGNITION: Identifying recurring themes and relationships
7. WISDOM ACCUMULATION: Building understanding through accumulated experience

BENEFITS OF LONG-TERM MEMORY:
- Enables continuous learning and improvement across sessions
- Maintains context and relationships over extended periods
- Supports personalization and adaptation to user preferences
- Allows building upon previous knowledge and experiences
- Enables pattern recognition across multiple interactions
- Creates foundation for wisdom and expertise development

THE MEMORY ADVANTAGE:
===================
NO PERSISTENCE: Each session independent → No learning or growth
WITH PERSISTENCE: Accumulated knowledge → Continuous improvement

LONG-TERM MEMORY COMPONENTS:
===========================
1. KNOWLEDGE STORAGE: Persistent storage of facts, experiences, and patterns
2. MEMORY INDEXING: Efficient organization and retrieval mechanisms
3. MEMORY CONSOLIDATION: Strengthening important memories over time
4. FORGETTING MECHANISMS: Selective removal of outdated or irrelevant information
5. ASSOCIATIVE NETWORKS: Connecting related memories for better retrieval
6. TEMPORAL ORGANIZATION: Time-based organization of memories and experiences
7. PRIORITY MANAGEMENT: Importance-based retention and access patterns

WHY THIS IS REVOLUTIONARY:
========================
- Enables AI agents to learn and grow like humans do
- Critical for personalized AI assistants and long-term user relationships
- Foundation for adaptive systems that improve through experience
- Supports complex reasoning that builds on accumulated knowledge
- Enables development of expertise and specialized knowledge domains
- Creates truly intelligent systems that learn from every interaction
"""

import asyncio
import time
import json
import uuid
import hashlib
import pickle
import sqlite3
import os
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
import logging
from collections import defaultdict
from datetime import datetime, timedelta
import threading
import gzip
import shutil
from pathlib import Path
import numpy as np
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class MemoryType(Enum):
    """Types of long-term memories"""
    FACTUAL = "factual"           # Facts and explicit knowledge
    EXPERIENTIAL = "experiential" # Past experiences and events
    PROCEDURAL = "procedural"     # Skills and procedures
    EPISODIC = "episodic"        # Personal experiences with context
    SEMANTIC = "semantic"        # General knowledge and concepts
    PREFERENCE = "preference"     # User preferences and patterns
    PATTERN = "pattern"          # Learned patterns and relationships

class MemoryStrength(Enum):
    """Memory strength levels"""
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4
    PERMANENT = 5

class RetrievalCue(Enum):
    """Types of retrieval cues"""
    TEMPORAL = "temporal"         # Time-based retrieval
    SEMANTIC = "semantic"         # Meaning-based retrieval
    ASSOCIATIVE = "associative"   # Association-based retrieval
    CONTEXTUAL = "contextual"     # Context-based retrieval
    SIMILARITY = "similarity"     # Similarity-based retrieval

@dataclass
class LongTermMemory:
    """Represents a long-term memory"""
    
    id: str
    content: Any
    memory_type: MemoryType
    
    # Memory metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    
    # Memory strength and consolidation
    strength: MemoryStrength = MemoryStrength.WEAK
    consolidation_level: float = 0.0  # 0.0 to 1.0
    
    # Context and associations
    context: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    associations: Set[str] = field(default_factory=set)  # IDs of related memories
    
    # Retrieval and forgetting
    retrieval_success_rate: float = 1.0
    forgetting_rate: float = 0.1
    importance_score: float = 0.5
    
    # Versioning and updates
    version: int = 1
    last_updated: datetime = field(default_factory=datetime.now)
    update_history: List[str] = field(default_factory=list)
    
    # Embedding for similarity search
    embedding: Optional[List[float]] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
    
    def access(self) -> None:
        """Record memory access"""
        self.last_accessed = datetime.now()
        self.access_count += 1
        
        # Strengthen memory on access
        self._strengthen_memory()
    
    def _strengthen_memory(self) -> None:
        """Strengthen memory based on access"""
        
        # Increase consolidation level
        self.consolidation_level = min(1.0, self.consolidation_level + 0.1)
        
        # Potentially increase strength
        if self.consolidation_level > 0.8 and self.strength.value < MemoryStrength.VERY_STRONG.value:
            new_strength_value = min(MemoryStrength.PERMANENT.value, self.strength.value + 1)
            self.strength = MemoryStrength(new_strength_value)
        
        # Improve retrieval success rate
        self.retrieval_success_rate = min(1.0, self.retrieval_success_rate + 0.05)
    
    def apply_forgetting(self, time_delta: timedelta) -> None:
        """Apply forgetting curve"""
        
        # Don't forget permanent memories
        if self.strength == MemoryStrength.PERMANENT:
            return
        
        # Calculate forgetting based on time and strength
        days_passed = time_delta.total_seconds() / (24 * 3600)
        forgetting_factor = self.forgetting_rate * days_passed / self.strength.value
        
        # Reduce consolidation level
        self.consolidation_level = max(0.0, self.consolidation_level - forgetting_factor)
        
        # Reduce retrieval success rate
        self.retrieval_success_rate = max(0.1, self.retrieval_success_rate - forgetting_factor * 0.5)
    
    def update_content(self, new_content: Any, change_description: str = "") -> None:
        """Update memory content"""
        
        self.content = new_content
        self.version += 1
        self.last_updated = datetime.now()
        
        if change_description:
            self.update_history.append(f"v{self.version}: {change_description}")
        
        # Reset some metrics on update
        self.consolidation_level = max(0.3, self.consolidation_level)  # Partial reset
    
    def calculate_relevance_score(self, query_embedding: List[float] = None,
                                 query_tags: Set[str] = None,
                                 query_context: Dict[str, Any] = None) -> float:
        """Calculate relevance to a query"""
        
        scores = []
        
        # Embedding similarity
        if query_embedding and self.embedding:
            similarity = self._cosine_similarity(query_embedding, self.embedding)
            scores.append(similarity * 0.4)
        
        # Tag overlap
        if query_tags and self.tags:
            tag_overlap = len(query_tags.intersection(self.tags)) / len(query_tags.union(self.tags))
            scores.append(tag_overlap * 0.3)
        
        # Context similarity
        if query_context and self.context:
            context_similarity = self._context_similarity(query_context, self.context)
            scores.append(context_similarity * 0.2)
        
        # Memory strength and recency
        strength_score = self.strength.value / MemoryStrength.PERMANENT.value
        recency_score = max(0, 1.0 - (datetime.now() - self.last_accessed).days / 365)
        scores.append((strength_score + recency_score) / 2 * 0.1)
        
        return sum(scores) if scores else 0.0
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between vectors"""
        
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm_a = sum(a * a for a in vec1) ** 0.5
        norm_b = sum(b * b for b in vec2) ** 0.5
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def _context_similarity(self, ctx1: Dict[str, Any], ctx2: Dict[str, Any]) -> float:
        """Calculate context similarity"""
        
        common_keys = set(ctx1.keys()).intersection(set(ctx2.keys()))
        if not common_keys:
            return 0.0
        
        matches = sum(1 for key in common_keys if ctx1[key] == ctx2[key])
        return matches / len(common_keys)

class MemoryStorage(ABC):
    """Abstract base class for memory storage backends"""
    
    @abstractmethod
    async def store_memory(self, memory: LongTermMemory) -> bool:
        """Store a memory"""
        pass
    
    @abstractmethod
    async def retrieve_memory(self, memory_id: str) -> Optional[LongTermMemory]:
        """Retrieve a memory by ID"""
        pass
    
    @abstractmethod
    async def search_memories(self, query: str, memory_types: List[MemoryType] = None,
                            tags: Set[str] = None, limit: int = 10) -> List[LongTermMemory]:
        """Search for memories"""
        pass
    
    @abstractmethod
    async def update_memory(self, memory: LongTermMemory) -> bool:
        """Update an existing memory"""
        pass
    
    @abstractmethod
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory"""
        pass

class SQLiteMemoryStorage(MemoryStorage):
    """SQLite-based memory storage"""
    
    def __init__(self, db_path: str = "long_term_memory.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        
        self.logger = logging.getLogger("SQLiteMemoryStorage")
        
        # Initialize database
        self._initialize_database()
    
    def _initialize_database(self) -> None:
        """Initialize the SQLite database"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create memories table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content_json TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    last_accessed TIMESTAMP NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    strength INTEGER NOT NULL,
                    consolidation_level REAL NOT NULL,
                    context_json TEXT,
                    tags_json TEXT,
                    associations_json TEXT,
                    retrieval_success_rate REAL DEFAULT 1.0,
                    forgetting_rate REAL DEFAULT 0.1,
                    importance_score REAL DEFAULT 0.5,
                    version INTEGER DEFAULT 1,
                    last_updated TIMESTAMP NOT NULL,
                    update_history_json TEXT,
                    embedding_json TEXT
                )
            ''')
            
            # Create indexes for efficient searching
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(memory_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON memories(created_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_last_accessed ON memories(last_accessed)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_strength ON memories(strength)')
            
            conn.commit()
    
    async def store_memory(self, memory: LongTermMemory) -> bool:
        """Store a memory in the database"""
        
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute('''
                        INSERT OR REPLACE INTO memories (
                            id, content_json, memory_type, created_at, last_accessed,
                            access_count, strength, consolidation_level, context_json,
                            tags_json, associations_json, retrieval_success_rate,
                            forgetting_rate, importance_score, version, last_updated,
                            update_history_json, embedding_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        memory.id,
                        json.dumps(memory.content, default=str),
                        memory.memory_type.value,
                        memory.created_at.isoformat(),
                        memory.last_accessed.isoformat(),
                        memory.access_count,
                        memory.strength.value,
                        memory.consolidation_level,
                        json.dumps(memory.context),
                        json.dumps(list(memory.tags)),
                        json.dumps(list(memory.associations)),
                        memory.retrieval_success_rate,
                        memory.forgetting_rate,
                        memory.importance_score,
                        memory.version,
                        memory.last_updated.isoformat(),
                        json.dumps(memory.update_history),
                        json.dumps(memory.embedding) if memory.embedding else None
                    ))
                    
                    conn.commit()
                    return True
                    
        except Exception as e:
            self.logger.error(f"Failed to store memory {memory.id}: {e}")
            return False
    
    async def retrieve_memory(self, memory_id: str) -> Optional[LongTermMemory]:
        """Retrieve a memory by ID"""
        
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute('SELECT * FROM memories WHERE id = ?', (memory_id,))
                    row = cursor.fetchone()
                    
                    if row:
                        return self._row_to_memory(row)
                    
                    return None
                    
        except Exception as e:
            self.logger.error(f"Failed to retrieve memory {memory_id}: {e}")
            return None
    
    async def search_memories(self, query: str, memory_types: List[MemoryType] = None,
                            tags: Set[str] = None, limit: int = 10) -> List[LongTermMemory]:
        """Search for memories"""
        
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Build search query
                    conditions = []
                    params = []
                    
                    if memory_types:
                        type_placeholders = ','.join('?' * len(memory_types))
                        conditions.append(f'memory_type IN ({type_placeholders})')
                        params.extend([mt.value for mt in memory_types])
                    
                    if query:
                        conditions.append('content_json LIKE ?')
                        params.append(f'%{query}%')
                    
                    where_clause = 'WHERE ' + ' AND '.join(conditions) if conditions else ''
                    
                    sql = f'''
                        SELECT * FROM memories
                        {where_clause}
                        ORDER BY last_accessed DESC, strength DESC
                        LIMIT ?
                    '''
                    params.append(limit)
                    
                    cursor.execute(sql, params)
                    rows = cursor.fetchall()
                    
                    memories = [self._row_to_memory(row) for row in rows]
                    
                    # Filter by tags if specified
                    if tags:
                        memories = [m for m in memories if tags.intersection(m.tags)]
                    
                    return memories
                    
        except Exception as e:
            self.logger.error(f"Failed to search memories: {e}")
            return []
    
    async def update_memory(self, memory: LongTermMemory) -> bool:
        """Update an existing memory"""
        return await self.store_memory(memory)  # SQLite handles upsert
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory"""
        
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute('DELETE FROM memories WHERE id = ?', (memory_id,))
                    conn.commit()
                    
                    return cursor.rowcount > 0
                    
        except Exception as e:
            self.logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False
    
    def _row_to_memory(self, row: Tuple) -> LongTermMemory:
        """Convert database row to LongTermMemory object"""
        
        (id, content_json, memory_type, created_at, last_accessed, access_count,
         strength, consolidation_level, context_json, tags_json, associations_json,
         retrieval_success_rate, forgetting_rate, importance_score, version,
         last_updated, update_history_json, embedding_json) = row
        
        memory = LongTermMemory(
            id=id,
            content=json.loads(content_json),
            memory_type=MemoryType(memory_type),
            created_at=datetime.fromisoformat(created_at),
            last_accessed=datetime.fromisoformat(last_accessed),
            access_count=access_count,
            strength=MemoryStrength(strength),
            consolidation_level=consolidation_level,
            context=json.loads(context_json) if context_json else {},
            tags=set(json.loads(tags_json)) if tags_json else set(),
            associations=set(json.loads(associations_json)) if associations_json else set(),
            retrieval_success_rate=retrieval_success_rate,
            forgetting_rate=forgetting_rate,
            importance_score=importance_score,
            version=version,
            last_updated=datetime.fromisoformat(last_updated),
            update_history=json.loads(update_history_json) if update_history_json else [],
            embedding=json.loads(embedding_json) if embedding_json else None
        )
        
        return memory
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics"""
        
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Total memories
                    cursor.execute('SELECT COUNT(*) FROM memories')
                    total_memories = cursor.fetchone()[0]
                    
                    # Memories by type
                    cursor.execute('''
                        SELECT memory_type, COUNT(*) 
                        FROM memories 
                        GROUP BY memory_type
                    ''')
                    type_distribution = dict(cursor.fetchall())
                    
                    # Memories by strength
                    cursor.execute('''
                        SELECT strength, COUNT(*) 
                        FROM memories 
                        GROUP BY strength
                    ''')
                    strength_distribution = dict(cursor.fetchall())
                    
                    # Database size
                    db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
                    
                    return {
                        'total_memories': total_memories,
                        'type_distribution': type_distribution,
                        'strength_distribution': strength_distribution,
                        'database_size_bytes': db_size
                    }
                    
        except Exception as e:
            self.logger.error(f"Failed to get statistics: {e}")
            return {}

class MemoryConsolidator:
    """Handles memory consolidation and strengthening"""
    
    def __init__(self, storage: MemoryStorage):
        self.storage = storage
        self.consolidation_rules = self._define_consolidation_rules()
        
        self.logger = logging.getLogger("MemoryConsolidator")
    
    def _define_consolidation_rules(self) -> Dict[str, Any]:
        """Define rules for memory consolidation"""
        
        return {
            'repetition_threshold': 3,      # Access count to trigger consolidation
            'time_threshold': timedelta(days=7),  # Time window for consolidation
            'importance_boost': 0.2,        # Importance increase on consolidation
            'strength_requirements': {
                MemoryStrength.WEAK: {'accesses': 2, 'days': 1},
                MemoryStrength.MODERATE: {'accesses': 5, 'days': 3},
                MemoryStrength.STRONG: {'accesses': 10, 'days': 7},
                MemoryStrength.VERY_STRONG: {'accesses': 20, 'days': 14}
            }
        }
    
    async def consolidate_memory(self, memory: LongTermMemory) -> bool:
        """Attempt to consolidate a memory"""
        
        # Check if memory qualifies for consolidation
        if not self._qualifies_for_consolidation(memory):
            return False
        
        # Apply consolidation
        original_strength = memory.strength
        
        # Increase strength
        if memory.strength.value < MemoryStrength.PERMANENT.value:
            new_strength_value = min(MemoryStrength.PERMANENT.value, memory.strength.value + 1)
            memory.strength = MemoryStrength(new_strength_value)
        
        # Boost consolidation level
        memory.consolidation_level = min(1.0, memory.consolidation_level + 0.3)
        
        # Boost importance
        memory.importance_score = min(1.0, memory.importance_score + self.consolidation_rules['importance_boost'])
        
        # Reduce forgetting rate
        memory.forgetting_rate = max(0.01, memory.forgetting_rate * 0.8)
        
        # Update memory
        success = await self.storage.update_memory(memory)
        
        if success:
            self.logger.debug(f"Consolidated memory {memory.id}: {original_strength.name} -> {memory.strength.name}")
        
        return success
    
    def _qualifies_for_consolidation(self, memory: LongTermMemory) -> bool:
        """Check if memory qualifies for consolidation"""
        
        # Already at maximum strength
        if memory.strength == MemoryStrength.PERMANENT:
            return False
        
        rules = self.consolidation_rules['strength_requirements']
        
        # Get requirements for current strength level
        if memory.strength in rules:
            requirements = rules[memory.strength]
            
            # Check access count
            if memory.access_count < requirements['accesses']:
                return False
            
            # Check age
            age = datetime.now() - memory.created_at
            if age < timedelta(days=requirements['days']):
                return False
            
            return True
        
        return False
    
    async def batch_consolidation(self, memory_ids: List[str] = None) -> Dict[str, int]:
        """Perform batch consolidation"""
        
        results = {
            'processed': 0,
            'consolidated': 0,
            'failed': 0
        }
        
        # If no specific IDs provided, get candidates
        if memory_ids is None:
            memory_ids = await self._get_consolidation_candidates()
        
        for memory_id in memory_ids:
            results['processed'] += 1
            
            try:
                memory = await self.storage.retrieve_memory(memory_id)
                
                if memory:
                    success = await self.consolidate_memory(memory)
                    
                    if success:
                        results['consolidated'] += 1
                    else:
                        results['failed'] += 1
                else:
                    results['failed'] += 1
                    
            except Exception as e:
                self.logger.error(f"Failed to consolidate memory {memory_id}: {e}")
                results['failed'] += 1
        
        return results
    
    async def _get_consolidation_candidates(self) -> List[str]:
        """Get memories that are candidates for consolidation"""
        
        # This would need to be implemented based on storage backend
        # For SQLite, we could query for memories with sufficient access counts
        
        # Placeholder implementation
        return []

class MemoryRetriever:
    """Handles memory retrieval and search"""
    
    def __init__(self, storage: MemoryStorage):
        self.storage = storage
        
        self.logger = logging.getLogger("MemoryRetriever")
    
    async def retrieve_by_cue(self, cue_type: RetrievalCue, cue_value: Any,
                            memory_types: List[MemoryType] = None,
                            limit: int = 10) -> List[LongTermMemory]:
        """Retrieve memories using different cues"""
        
        if cue_type == RetrievalCue.TEMPORAL:
            return await self._retrieve_by_time(cue_value, memory_types, limit)
        elif cue_type == RetrievalCue.SEMANTIC:
            return await self._retrieve_by_meaning(cue_value, memory_types, limit)
        elif cue_type == RetrievalCue.ASSOCIATIVE:
            return await self._retrieve_by_association(cue_value, memory_types, limit)
        elif cue_type == RetrievalCue.CONTEXTUAL:
            return await self._retrieve_by_context(cue_value, memory_types, limit)
        elif cue_type == RetrievalCue.SIMILARITY:
            return await self._retrieve_by_similarity(cue_value, memory_types, limit)
        else:
            return []
    
    async def _retrieve_by_time(self, time_range: Tuple[datetime, datetime],
                               memory_types: List[MemoryType] = None,
                               limit: int = 10) -> List[LongTermMemory]:
        """Retrieve memories from a time range"""
        
        # This would need storage backend support
        # For now, get all memories and filter
        
        all_memories = await self.storage.search_memories("", memory_types, None, 1000)
        
        start_time, end_time = time_range
        
        filtered_memories = [
            memory for memory in all_memories
            if start_time <= memory.created_at <= end_time
        ]
        
        # Sort by recency and return top results
        filtered_memories.sort(key=lambda m: m.created_at, reverse=True)
        
        return filtered_memories[:limit]
    
    async def _retrieve_by_meaning(self, query: str,
                                  memory_types: List[MemoryType] = None,
                                  limit: int = 10) -> List[LongTermMemory]:
        """Retrieve memories by semantic meaning"""
        
        # Use storage search (could be enhanced with embeddings)
        memories = await self.storage.search_memories(query, memory_types, None, limit)
        
        # Apply access to retrieved memories
        for memory in memories:
            memory.access()
            await self.storage.update_memory(memory)
        
        return memories
    
    async def _retrieve_by_association(self, memory_id: str,
                                      memory_types: List[MemoryType] = None,
                                      limit: int = 10) -> List[LongTermMemory]:
        """Retrieve memories associated with a given memory"""
        
        # Get the source memory
        source_memory = await self.storage.retrieve_memory(memory_id)
        
        if not source_memory:
            return []
        
        # Get associated memories
        associated_memories = []
        
        for associated_id in source_memory.associations:
            memory = await self.storage.retrieve_memory(associated_id)
            
            if memory and (not memory_types or memory.memory_type in memory_types):
                memory.access()
                await self.storage.update_memory(memory)
                associated_memories.append(memory)
        
        # Sort by strength and recency
        associated_memories.sort(
            key=lambda m: (m.strength.value, m.last_accessed),
            reverse=True
        )
        
        return associated_memories[:limit]
    
    async def _retrieve_by_context(self, context: Dict[str, Any],
                                  memory_types: List[MemoryType] = None,
                                  limit: int = 10) -> List[LongTermMemory]:
        """Retrieve memories by context similarity"""
        
        # Get all memories and calculate context similarity
        all_memories = await self.storage.search_memories("", memory_types, None, 1000)
        
        # Calculate relevance scores
        scored_memories = []
        
        for memory in all_memories:
            relevance = memory.calculate_relevance_score(query_context=context)
            
            if relevance > 0.1:  # Minimum relevance threshold
                scored_memories.append((memory, relevance))
        
        # Sort by relevance
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        
        # Apply access and return memories
        result_memories = []
        
        for memory, score in scored_memories[:limit]:
            memory.access()
            await self.storage.update_memory(memory)
            result_memories.append(memory)
        
        return result_memories
    
    async def _retrieve_by_similarity(self, target_memory: LongTermMemory,
                                     memory_types: List[MemoryType] = None,
                                     limit: int = 10) -> List[LongTermMemory]:
        """Retrieve memories similar to target memory"""
        
        if not target_memory.embedding:
            return []
        
        # Get all memories with embeddings
        all_memories = await self.storage.search_memories("", memory_types, None, 1000)
        
        # Calculate similarities
        similar_memories = []
        
        for memory in all_memories:
            if memory.id != target_memory.id and memory.embedding:
                similarity = target_memory._cosine_similarity(
                    target_memory.embedding, memory.embedding
                )
                
                if similarity > 0.3:  # Minimum similarity threshold
                    similar_memories.append((memory, similarity))
        
        # Sort by similarity
        similar_memories.sort(key=lambda x: x[1], reverse=True)
        
        # Apply access and return memories
        result_memories = []
        
        for memory, similarity in similar_memories[:limit]:
            memory.access()
            await self.storage.update_memory(memory)
            result_memories.append(memory)
        
        return result_memories

class LongTermMemorySystem:
    """Complete long-term memory system"""
    
    def __init__(self, storage_path: str = "long_term_memory.db"):
        # Core components
        self.storage = SQLiteMemoryStorage(storage_path)
        self.consolidator = MemoryConsolidator(self.storage)
        self.retriever = MemoryRetriever(self.storage)
        
        # Memory management
        self.association_graph: Dict[str, Set[str]] = defaultdict(set)
        
        # Statistics
        self.stats = {
            'memories_created': 0,
            'memories_retrieved': 0,
            'consolidations_performed': 0,
            'associations_created': 0
        }
        
        self.logger = logging.getLogger("LongTermMemorySystem")
    
    async def initialize(self) -> None:
        """Initialize the long-term memory system"""
        self.logger.info("Long-term memory system initialized")
    
    async def store_memory(self, content: Any, memory_type: MemoryType,
                          context: Dict[str, Any] = None,
                          tags: Set[str] = None,
                          importance: float = 0.5,
                          embedding: List[float] = None) -> str:
        """Store a new memory"""
        
        memory = LongTermMemory(
            id="",
            content=content,
            memory_type=memory_type,
            context=context or {},
            tags=tags or set(),
            importance_score=importance,
            embedding=embedding
        )
        
        success = await self.storage.store_memory(memory)
        
        if success:
            self.stats['memories_created'] += 1
            self.logger.debug(f"Stored memory: {memory.id}")
            return memory.id
        else:
            raise Exception("Failed to store memory")
    
    async def retrieve_memory(self, memory_id: str) -> Optional[LongTermMemory]:
        """Retrieve a specific memory"""
        
        memory = await self.storage.retrieve_memory(memory_id)
        
        if memory:
            memory.access()
            await self.storage.update_memory(memory)
            self.stats['memories_retrieved'] += 1
        
        return memory
    
    async def search_memories(self, query: str = "",
                            memory_types: List[MemoryType] = None,
                            tags: Set[str] = None,
                            limit: int = 10) -> List[LongTermMemory]:
        """Search for memories"""
        
        memories = await self.storage.search_memories(query, memory_types, tags, limit)
        
        # Update access for retrieved memories
        for memory in memories:
            memory.access()
            await self.storage.update_memory(memory)
        
        self.stats['memories_retrieved'] += len(memories)
        
        return memories
    
    async def create_association(self, memory_id1: str, memory_id2: str,
                               bidirectional: bool = True) -> bool:
        """Create association between memories"""
        
        # Get both memories
        memory1 = await self.storage.retrieve_memory(memory_id1)
        memory2 = await self.storage.retrieve_memory(memory_id2)
        
        if not memory1 or not memory2:
            return False
        
        # Add associations
        memory1.associations.add(memory_id2)
        
        if bidirectional:
            memory2.associations.add(memory_id1)
        
        # Update storage
        success1 = await self.storage.update_memory(memory1)
        success2 = await self.storage.update_memory(memory2) if bidirectional else True
        
        if success1 and success2:
            self.stats['associations_created'] += 1
            self.association_graph[memory_id1].add(memory_id2)
            
            if bidirectional:
                self.association_graph[memory_id2].add(memory_id1)
            
            return True
        
        return False
    
    async def consolidate_memories(self, memory_ids: List[str] = None) -> Dict[str, int]:
        """Perform memory consolidation"""
        
        results = await self.consolidator.batch_consolidation(memory_ids)
        self.stats['consolidations_performed'] += results['consolidated']
        
        return results
    
    async def retrieve_by_cue(self, cue_type: RetrievalCue, cue_value: Any,
                            memory_types: List[MemoryType] = None,
                            limit: int = 10) -> List[LongTermMemory]:
        """Retrieve memories using different retrieval cues"""
        
        memories = await self.retriever.retrieve_by_cue(
            cue_type, cue_value, memory_types, limit
        )
        
        self.stats['memories_retrieved'] += len(memories)
        
        return memories
    
    async def update_memory_content(self, memory_id: str, new_content: Any,
                                  change_description: str = "") -> bool:
        """Update memory content"""
        
        memory = await self.storage.retrieve_memory(memory_id)
        
        if not memory:
            return False
        
        memory.update_content(new_content, change_description)
        
        return await self.storage.update_memory(memory)
    
    async def forget_memories(self, criteria: Dict[str, Any]) -> int:
        """Forget memories based on criteria"""
        
        # This would implement forgetting logic
        # For now, just a placeholder
        
        return 0
    
    async def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        
        storage_stats = self.storage.get_statistics()
        
        return {
            'system_statistics': self.stats,
            'storage_statistics': storage_stats,
            'association_graph_size': len(self.association_graph),
            'total_associations': sum(len(assocs) for assocs in self.association_graph.values())
        }
    
    async def export_memories(self, export_path: str,
                            memory_types: List[MemoryType] = None) -> bool:
        """Export memories to file"""
        
        try:
            memories = await self.storage.search_memories("", memory_types, None, 10000)
            
            export_data = {
                'exported_at': datetime.now().isoformat(),
                'total_memories': len(memories),
                'memories': [asdict(memory) for memory in memories]
            }
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export memories: {e}")
            return False
    
    @contextmanager
    def memory_session(self, session_context: Dict[str, Any]):
        """Context manager for memory operations"""
        
        # This could track operations within a session
        session_id = str(uuid.uuid4())
        
        self.logger.debug(f"Started memory session: {session_id}")
        
        try:
            yield self
        finally:
            self.logger.debug(f"Ended memory session: {session_id}")

# EXAMPLE DEMONSTRATIONS
# =====================

async def demo_basic_memory_operations():
    """Demo: Basic long-term memory operations"""
    print("\nDEMO 1: BASIC MEMORY OPERATIONS")
    print("=" * 50)
    
    memory_system = LongTermMemorySystem("demo_memory.db")
    await memory_system.initialize()
    
    # Store different types of memories
    print("Storing various types of memories:")
    
    # Factual memory
    fact_id = await memory_system.store_memory(
        {"fact": "Paris is the capital of France", "confidence": 1.0},
        MemoryType.FACTUAL,
        context={"domain": "geography", "source": "textbook"},
        tags={"geography", "europe", "capital"},
        importance=0.8
    )
    print(f"  Stored factual memory: {fact_id[:8]}...")
    
    # Experiential memory
    experience_id = await memory_system.store_memory(
        {
            "event": "User asked about weather in London",
            "response": "Provided current weather data",
            "outcome": "User satisfied",
            "timestamp": datetime.now().isoformat()
        },
        MemoryType.EXPERIENTIAL,
        context={"interaction_type": "weather_query", "user_satisfaction": "high"},
        tags={"weather", "london", "query"},
        importance=0.6
    )
    print(f"  Stored experiential memory: {experience_id[:8]}...")
    
    # Preference memory
    preference_id = await memory_system.store_memory(
        {"user_preference": "prefers detailed explanations", "strength": 0.9},
        MemoryType.PREFERENCE,
        context={"user_id": "user123", "domain": "learning"},
        tags={"preference", "explanation_style"},
        importance=0.7
    )
    print(f"  Stored preference memory: {preference_id[:8]}...")
    
    # Retrieve memories
    print(f"\nRetrieving stored memories:")
    
    fact_memory = await memory_system.retrieve_memory(fact_id)
    print(f"  Retrieved fact: {fact_memory.content}")
    print(f"    Access count: {fact_memory.access_count}")
    
    experience_memory = await memory_system.retrieve_memory(experience_id)
    print(f"  Retrieved experience: {experience_memory.content['event']}")
    
    # Search memories
    print(f"\nSearching memories:")
    
    geography_memories = await memory_system.search_memories(
        query="France",
        tags={"geography"}
    )
    print(f"  Found {len(geography_memories)} geography memories about France")
    
    weather_memories = await memory_system.search_memories(
        memory_types=[MemoryType.EXPERIENTIAL],
        tags={"weather"}
    )
    print(f"  Found {len(weather_memories)} weather-related experiences")
    
    # Show memory statistics
    stats = await memory_system.get_memory_statistics()
    print(f"\nMemory system statistics:")
    print(f"  Memories created: {stats['system_statistics']['memories_created']}")
    print(f"  Memories retrieved: {stats['system_statistics']['memories_retrieved']}")
    print(f"  Total in storage: {stats['storage_statistics']['total_memories']}")

async def demo_memory_associations():
    """Demo: Creating and using memory associations"""
    print("\nDEMO 2: MEMORY ASSOCIATIONS")
    print("=" * 50)
    
    memory_system = LongTermMemorySystem("demo_associations.db")
    await memory_system.initialize()
    
    print("Building a network of associated memories:")
    
    # Create related memories about Python programming
    python_id = await memory_system.store_memory(
        {"concept": "Python", "description": "High-level programming language"},
        MemoryType.FACTUAL,
        tags={"programming", "python", "language"}
    )
    
    oop_id = await memory_system.store_memory(
        {"concept": "Object-Oriented Programming", "description": "Programming paradigm based on objects"},
        MemoryType.FACTUAL,
        tags={"programming", "oop", "paradigm"}
    )
    
    django_id = await memory_system.store_memory(
        {"framework": "Django", "description": "Python web framework"},
        MemoryType.FACTUAL,
        tags={"python", "web", "framework"}
    )
    
    project_id = await memory_system.store_memory(
        {
            "project": "Built web application using Django",
            "duration": "2 weeks",
            "outcome": "successful deployment"
        },
        MemoryType.EXPERIENTIAL,
        tags={"project", "django", "web"}
    )
    
    print(f"  Created 4 related memories")
    
    # Create associations
    print(f"\nCreating associations:")
    
    # Python is related to OOP
    await memory_system.create_association(python_id, oop_id)
    print(f"  Associated Python with OOP")
    
    # Python is related to Django
    await memory_system.create_association(python_id, django_id)
    print(f"  Associated Python with Django")
    
    # Django project uses Django framework
    await memory_system.create_association(project_id, django_id)
    print(f"  Associated project with Django")
    
    # Test associative retrieval
    print(f"\nTesting associative retrieval:")
    
    # Find memories associated with Python
    python_associated = await memory_system.retrieve_by_cue(
        RetrievalCue.ASSOCIATIVE, python_id
    )
    
    print(f"  Memories associated with Python: {len(python_associated)}")
    for memory in python_associated:
        if 'concept' in memory.content:
            print(f"    {memory.content['concept']}")
        elif 'framework' in memory.content:
            print(f"    {memory.content['framework']}")
    
    # Find memories associated with Django
    django_associated = await memory_system.retrieve_by_cue(
        RetrievalCue.ASSOCIATIVE, django_id
    )
    
    print(f"  Memories associated with Django: {len(django_associated)}")
    for memory in django_associated:
        if 'concept' in memory.content:
            print(f"    {memory.content['concept']}")
        elif 'project' in memory.content:
            print(f"    {memory.content['project']}")
    
    # Show association statistics
    stats = await memory_system.get_memory_statistics()
    print(f"\nAssociation statistics:")
    print(f"  Associations created: {stats['system_statistics']['associations_created']}")
    print(f"  Association graph size: {stats['association_graph_size']}")
    print(f"  Total associations: {stats['total_associations']}")

async def demo_memory_consolidation():
    """Demo: Memory consolidation and strengthening"""
    print("\nDEMO 3: MEMORY CONSOLIDATION")
    print("=" * 50)
    
    memory_system = LongTermMemorySystem("demo_consolidation.db")
    await memory_system.initialize()
    
    print("Demonstrating memory consolidation through repeated access:")
    
    # Create a memory that will be frequently accessed
    important_fact_id = await memory_system.store_memory(
        {"fact": "Machine learning is a subset of artificial intelligence"},
        MemoryType.FACTUAL,
        context={"domain": "AI", "importance": "high"},
        tags={"AI", "machine_learning", "fact"},
        importance=0.9
    )
    
    # Create a memory that will be rarely accessed
    trivial_fact_id = await memory_system.store_memory(
        {"fact": "There are 24 hours in a day"},
        MemoryType.FACTUAL,
        context={"domain": "general", "importance": "low"},
        tags={"time", "fact"},
        importance=0.3
    )
    
    print(f"  Created two memories with different importance levels")
    
    # Show initial state
    important_memory = await memory_system.retrieve_memory(important_fact_id)
    trivial_memory = await memory_system.retrieve_memory(trivial_fact_id)
    
    print(f"\nInitial memory states:")
    print(f"  Important memory:")
    print(f"    Strength: {important_memory.strength.name}")
    print(f"    Consolidation level: {important_memory.consolidation_level:.2f}")
    print(f"    Access count: {important_memory.access_count}")
    
    print(f"  Trivial memory:")
    print(f"    Strength: {trivial_memory.strength.name}")
    print(f"    Consolidation level: {trivial_memory.consolidation_level:.2f}")
    print(f"    Access count: {trivial_memory.access_count}")
    
    # Simulate repeated access to important memory
    print(f"\nSimulating repeated access to important memory:")
    
    for i in range(8):
        memory = await memory_system.retrieve_memory(important_fact_id)
        print(f"  Access {i+1}: Strength={memory.strength.name}, "
              f"Consolidation={memory.consolidation_level:.2f}, "
              f"Count={memory.access_count}")
    
    # Perform consolidation
    print(f"\nPerforming memory consolidation:")
    
    consolidation_results = await memory_system.consolidate_memories([important_fact_id, trivial_fact_id])
    
    print(f"  Consolidation results: {consolidation_results}")
    
    # Show final states
    important_memory = await memory_system.retrieve_memory(important_fact_id)
    trivial_memory = await memory_system.retrieve_memory(trivial_fact_id)
    
    print(f"\nFinal memory states:")
    print(f"  Important memory:")
    print(f"    Strength: {important_memory.strength.name}")
    print(f"    Consolidation level: {important_memory.consolidation_level:.2f}")
    print(f"    Forgetting rate: {important_memory.forgetting_rate:.3f}")
    
    print(f"  Trivial memory:")
    print(f"    Strength: {trivial_memory.strength.name}")
    print(f"    Consolidation level: {trivial_memory.consolidation_level:.2f}")
    print(f"    Forgetting rate: {trivial_memory.forgetting_rate:.3f}")

async def demo_contextual_retrieval():
    """Demo: Context-based memory retrieval"""
    print("\nDEMO 4: CONTEXTUAL RETRIEVAL")
    print("=" * 50)
    
    memory_system = LongTermMemorySystem("demo_context.db")
    await memory_system.initialize()
    
    print("Storing memories with different contexts:")
    
    # Work-related memories
    work_memory1 = await memory_system.store_memory(
        {"task": "Code review", "outcome": "Found 3 bugs", "time_spent": "2 hours"},
        MemoryType.EXPERIENTIAL,
        context={"environment": "work", "activity": "programming", "team": "backend"},
        tags={"work", "programming", "review"}
    )
    
    work_memory2 = await memory_system.store_memory(
        {"meeting": "Sprint planning", "decisions": ["Feature A priority", "Bug fix deadline"]},
        MemoryType.EXPERIENTIAL,
        context={"environment": "work", "activity": "planning", "team": "backend"},
        tags={"work", "meeting", "planning"}
    )
    
    # Personal memories
    personal_memory1 = await memory_system.store_memory(
        {"activity": "Read book about AI", "enjoyment": "high", "pages": 50},
        MemoryType.EXPERIENTIAL,
        context={"environment": "home", "activity": "learning", "mood": "curious"},
        tags={"personal", "reading", "AI"}
    )
    
    personal_memory2 = await memory_system.store_memory(
        {"activity": "Cooked dinner", "recipe": "pasta carbonara", "success": True},
        MemoryType.EXPERIENTIAL,
        context={"environment": "home", "activity": "cooking", "mood": "relaxed"},
        tags={"personal", "cooking", "food"}
    )
    
    print(f"  Created 4 memories with work and personal contexts")
    
    # Test contextual retrieval
    print(f"\nTesting contextual retrieval:")
    
    # Retrieve work-related memories
    work_context = {"environment": "work"}
    work_memories = await memory_system.retrieve_by_cue(
        RetrievalCue.CONTEXTUAL, work_context
    )
    
    print(f"  Work-related memories: {len(work_memories)}")
    for memory in work_memories:
        if 'task' in memory.content:
            print(f"    Task: {memory.content['task']}")
        elif 'meeting' in memory.content:
            print(f"    Meeting: {memory.content['meeting']}")
    
    # Retrieve programming-related memories
    programming_context = {"activity": "programming"}
    programming_memories = await memory_system.retrieve_by_cue(
        RetrievalCue.CONTEXTUAL, programming_context
    )
    
    print(f"  Programming-related memories: {len(programming_memories)}")
    for memory in programming_memories:
        print(f"    {memory.content}")
    
    # Retrieve home/personal memories
    home_context = {"environment": "home"}
    home_memories = await memory_system.retrieve_by_cue(
        RetrievalCue.CONTEXTUAL, home_context
    )
    
    print(f"  Home-related memories: {len(home_memories)}")
    for memory in home_memories:
        print(f"    Activity: {memory.content['activity']}")
    
    # Retrieve learning-related memories across contexts
    learning_context = {"activity": "learning"}
    learning_memories = await memory_system.retrieve_by_cue(
        RetrievalCue.CONTEXTUAL, learning_context
    )
    
    print(f"  Learning-related memories: {len(learning_memories)}")
    for memory in learning_memories:
        print(f"    {memory.content}")

async def demo_temporal_retrieval():
    """Demo: Time-based memory retrieval"""
    print("\nDEMO 5: TEMPORAL RETRIEVAL")
    print("=" * 50)
    
    memory_system = LongTermMemorySystem("demo_temporal.db")
    await memory_system.initialize()
    
    print("Creating memories with different timestamps:")
    
    # Create memories with different time periods
    now = datetime.now()
    
    # Recent memory (today)
    recent_id = await memory_system.store_memory(
        {"event": "Learned about neural networks", "satisfaction": "high"},
        MemoryType.EXPERIENTIAL,
        context={"period": "recent"},
        tags={"learning", "AI"}
    )
    
    # Manually set recent timestamp
    recent_memory = await memory_system.retrieve_memory(recent_id)
    recent_memory.created_at = now
    await memory_system.storage.update_memory(recent_memory)
    
    # Medium-term memory (1 week ago)
    medium_id = await memory_system.store_memory(
        {"event": "Completed Python course", "grade": "A+"},
        MemoryType.EXPERIENTIAL,
        context={"period": "medium"},
        tags={"learning", "python", "course"}
    )
    
    medium_memory = await memory_system.retrieve_memory(medium_id)
    medium_memory.created_at = now - timedelta(days=7)
    await memory_system.storage.update_memory(medium_memory)
    
    # Old memory (1 month ago)
    old_id = await memory_system.store_memory(
        {"event": "Started programming journey", "excitement": "very high"},
        MemoryType.EXPERIENTIAL,
        context={"period": "old"},
        tags={"learning", "programming", "beginning"}
    )
    
    old_memory = await memory_system.retrieve_memory(old_id)
    old_memory.created_at = now - timedelta(days=30)
    await memory_system.storage.update_memory(old_memory)
    
    print(f"  Created memories from different time periods")
    
    # Test temporal retrieval
    print(f"\nTesting temporal retrieval:")
    
    # Retrieve recent memories (last 3 days)
    recent_range = (now - timedelta(days=3), now)
    recent_memories = await memory_system.retrieve_by_cue(
        RetrievalCue.TEMPORAL, recent_range
    )
    
    print(f"  Recent memories (last 3 days): {len(recent_memories)}")
    for memory in recent_memories:
        print(f"    {memory.content['event']} (created: {memory.created_at.strftime('%Y-%m-%d')})")
    
    # Retrieve memories from last 2 weeks
    two_weeks_range = (now - timedelta(days=14), now)
    two_week_memories = await memory_system.retrieve_by_cue(
        RetrievalCue.TEMPORAL, two_weeks_range
    )
    
    print(f"  Memories from last 2 weeks: {len(two_week_memories)}")
    for memory in two_week_memories:
        print(f"    {memory.content['event']} (created: {memory.created_at.strftime('%Y-%m-%d')})")
    
    # Retrieve all learning memories chronologically
    all_learning = await memory_system.search_memories(
        tags={"learning"}
    )
    
    # Sort by creation time
    all_learning.sort(key=lambda m: m.created_at)
    
    print(f"  All learning memories (chronological): {len(all_learning)}")
    for memory in all_learning:
        print(f"    {memory.created_at.strftime('%Y-%m-%d')}: {memory.content['event']}")

async def main():
    """
    Demonstrate Long-Term Memory Store for persistent knowledge storage
    
    WHAT YOU'LL LEARN:
    ================
    1. How to implement persistent memory storage with SQLite backend
    2. How to create and manage different types of long-term memories
    3. How to build associative memory networks for related information
    4. How to implement memory consolidation and strengthening mechanisms
    5. How to perform contextual and temporal memory retrieval
    6. How to create complete long-term memory systems for AI agents
    
    REAL WORLD APPLICATIONS:
    =======================
    - Personal AI assistants that remember user preferences and history
    - Educational systems that track student learning progress
    - Customer service bots that maintain conversation history
    - Research assistants that accumulate domain knowledge
    - Therapeutic AI that builds understanding of patient needs
    - Collaborative AI that learns from team interactions
    """
    
    print("LONG-TERM MEMORY STORE DEMONSTRATION")
    print("Persistent knowledge storage and retrieval!")
    
    await demo_basic_memory_operations()
    await demo_memory_associations()
    await demo_memory_consolidation()
    await demo_contextual_retrieval()
    await demo_temporal_retrieval()
    
    print("\nWHAT WE LEARNED:")
    print("=" * 40)
    print("✓ Persistent memory enables continuous learning across sessions")
    print("✓ Memory associations create networks of related knowledge")
    print("✓ Consolidation strengthens important memories over time")
    print("✓ Contextual retrieval finds relevant memories for current situations")
    print("✓ Temporal retrieval enables chronological memory access")
    print("✓ Complete systems support various memory types and retrieval patterns")
    print("\nTHE POWER OF LONG-TERM MEMORY:")
    print("- Enables AI agents to learn and grow like humans do")
    print("- Supports personalization and adaptation to user needs")
    print("- Creates continuity across interactions and sessions")
    print("- Allows building upon previous knowledge and experiences")
    print("- Enables development of expertise through accumulated learning")

if __name__ == "__main__":
    asyncio.run(main())
