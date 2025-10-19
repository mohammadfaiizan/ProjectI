#!/usr/bin/env python3
"""
Short-Term Memory: Temporary Information Storage for Active Processing
====================================================================

WHAT IS THE PROBLEM?
==================
AI agents lack the ability to temporarily hold and manipulate information:
- Models can only process information within their context window
- No mechanism to temporarily store working information during complex reasoning
- Cannot maintain intermediate results across multiple processing steps
- Inability to hold multiple pieces of information while performing operations
- Limited working memory capacity restricts complex problem-solving abilities
- Context switches cause loss of working information and disrupted processing

Example: Multi-Step Mathematical Problem
WITHOUT SHORT-TERM MEMORY (Traditional):
- Agent receives: "Calculate the total cost of 5 laptops at $899 each, 3 monitors at $299 each, with 8.5% sales tax"
- Must process everything in a single pass without intermediate storage
- Cannot break down the problem into manageable sub-calculations
- Risk of computational errors due to cognitive overload
- No way to verify intermediate results before proceeding
- Result: Errors in complex calculations, inability to handle multi-step reasoning

REAL WORLD EXAMPLE:
=================
How does human working memory function during problem-solving?

HUMAN SHORT-TERM MEMORY:
1. INFORMATION INTAKE: Receives problem statement and extracts key elements
2. TEMPORARY STORAGE: Holds intermediate calculations and results in working memory
3. MANIPULATION: Performs operations on stored information
4. PROGRESSIVE BUILDING: Uses intermediate results to build toward final solution
5. VERIFICATION: Checks intermediate steps against stored context
6. ADAPTIVE CAPACITY: Manages information load based on complexity
7. CONTEXTUAL INTEGRATION: Combines working memory with long-term knowledge

BENEFITS OF SHORT-TERM MEMORY:
- Enables complex multi-step reasoning and problem decomposition
- Maintains working context during extended processing sessions
- Allows verification and error correction during computation
- Supports intermediate result storage for iterative processes
- Enables flexible information manipulation and recombination
- Improves accuracy and reliability of complex cognitive tasks

THE MEMORY ADVANTAGE:
===================
NO MEMORY: Single-pass processing → Limited complexity handling
WITH MEMORY: Multi-step processing → Complex reasoning capabilities

SHORT-TERM MEMORY COMPONENTS:
============================
1. BUFFER MANAGEMENT: Temporary storage with automatic capacity management
2. ITEM PRIORITY: Importance-based retention and eviction policies
3. TEMPORAL DECAY: Time-based forgetting of less relevant information
4. CAPACITY LIMITS: Realistic constraints matching cognitive limitations
5. RAPID ACCESS: Fast retrieval and modification of stored items
6. CONTEXT INTEGRATION: Seamless integration with ongoing processing
7. WORKING OPERATIONS: Support for manipulation of stored information

WHY THIS IS REVOLUTIONARY:
========================
- Enables AI agents to handle complex multi-step reasoning like humans
- Critical for mathematical computation, logical reasoning, and planning
- Foundation for advanced cognitive architectures and agent systems
- Supports natural problem decomposition and solution building
- Enables verification and error correction during processing
- Creates more reliable and capable autonomous AI systems
"""

import asyncio
import time
import json
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
from collections import deque, OrderedDict
from datetime import datetime, timedelta
import threading
import weakref
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class ItemPriority(Enum):
    """Priority levels for memory items"""
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    MINIMAL = 1

class MemoryOperation(Enum):
    """Types of memory operations"""
    STORE = "store"
    RETRIEVE = "retrieve"
    UPDATE = "update"
    DELETE = "delete"
    CLEAR = "clear"
    PRUNE = "prune"

@dataclass
class MemoryItem:
    """Represents an item in short-term memory"""
    
    id: str
    content: Any
    
    # Metadata
    priority: ItemPriority = ItemPriority.MEDIUM
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    
    # Decay and retention
    decay_rate: float = 0.1  # How quickly importance decreases
    importance: float = 1.0  # Current importance score
    
    # Context
    tags: List[str] = field(default_factory=list)
    context: str = ""
    
    # Relationships
    related_items: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
    
    def update_access(self) -> None:
        """Update access statistics"""
        self.last_accessed = datetime.now()
        self.access_count += 1
        
        # Boost importance on access
        self.importance = min(2.0, self.importance + 0.1)
    
    def apply_decay(self, time_delta: timedelta) -> None:
        """Apply temporal decay to importance"""
        decay_factor = self.decay_rate * (time_delta.total_seconds() / 3600)  # Decay per hour
        self.importance = max(0.0, self.importance - decay_factor)
    
    def calculate_retention_score(self) -> float:
        """Calculate score for retention decisions"""
        # Combine multiple factors
        priority_weight = self.priority.value / 5.0
        importance_weight = self.importance
        recency_weight = max(0, 1.0 - (datetime.now() - self.last_accessed).total_seconds() / 3600)
        frequency_weight = min(1.0, self.access_count / 10.0)
        
        # Weighted combination
        return (priority_weight * 0.3 + 
                importance_weight * 0.3 + 
                recency_weight * 0.2 + 
                frequency_weight * 0.2)

class EvictionPolicy(ABC):
    """Abstract base class for memory eviction policies"""
    
    @abstractmethod
    def select_for_eviction(self, items: List[MemoryItem], 
                          capacity: int) -> List[str]:
        """Select items to evict when memory is full"""
        pass

class LRUEvictionPolicy(EvictionPolicy):
    """Least Recently Used eviction policy"""
    
    def select_for_eviction(self, items: List[MemoryItem], 
                          capacity: int) -> List[str]:
        """Select least recently used items for eviction"""
        
        if len(items) <= capacity:
            return []
        
        # Sort by last accessed (oldest first)
        sorted_items = sorted(items, key=lambda x: x.last_accessed)
        
        # Select items to evict
        items_to_evict = sorted_items[:len(items) - capacity]
        
        return [item.id for item in items_to_evict]

class ImportanceBasedEvictionPolicy(EvictionPolicy):
    """Importance-based eviction policy"""
    
    def select_for_eviction(self, items: List[MemoryItem], 
                          capacity: int) -> List[str]:
        """Select least important items for eviction"""
        
        if len(items) <= capacity:
            return []
        
        # Sort by retention score (lowest first)
        sorted_items = sorted(items, key=lambda x: x.calculate_retention_score())
        
        # Select items to evict
        items_to_evict = sorted_items[:len(items) - capacity]
        
        return [item.id for item in items_to_evict]

class HybridEvictionPolicy(EvictionPolicy):
    """Hybrid eviction policy combining multiple factors"""
    
    def select_for_eviction(self, items: List[MemoryItem], 
                          capacity: int) -> List[str]:
        """Select items for eviction using hybrid scoring"""
        
        if len(items) <= capacity:
            return []
        
        # Calculate hybrid scores for all items
        scored_items = []
        
        for item in items:
            # Combine retention score with priority protection
            retention_score = item.calculate_retention_score()
            
            # Protect critical items
            if item.priority == ItemPriority.CRITICAL:
                retention_score += 2.0
            elif item.priority == ItemPriority.HIGH:
                retention_score += 1.0
            
            scored_items.append((item, retention_score))
        
        # Sort by score (lowest first for eviction)
        scored_items.sort(key=lambda x: x[1])
        
        # Select items to evict
        items_to_evict = scored_items[:len(items) - capacity]
        
        return [item.id for item, score in items_to_evict]

class MemoryBuffer:
    """Core buffer for short-term memory storage"""
    
    def __init__(self, capacity: int = 100, 
                 eviction_policy: EvictionPolicy = None):
        self.capacity = capacity
        self.eviction_policy = eviction_policy or HybridEvictionPolicy()
        
        # Storage
        self.items: Dict[str, MemoryItem] = {}
        self.access_order: deque = deque()  # For LRU tracking
        
        # Statistics
        self.stats = {
            'total_stores': 0,
            'total_retrievals': 0,
            'total_evictions': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Threading support
        self._lock = threading.RLock()
        
        self.logger = logging.getLogger("MemoryBuffer")
    
    def store(self, item: MemoryItem) -> bool:
        """Store an item in the buffer"""
        
        with self._lock:
            try:
                # Update existing item
                if item.id in self.items:
                    self.items[item.id] = item
                    item.update_access()
                    self._update_access_order(item.id)
                    return True
                
                # Check capacity and evict if necessary
                if len(self.items) >= self.capacity:
                    self._evict_items()
                
                # Store new item
                self.items[item.id] = item
                self.access_order.append(item.id)
                
                self.stats['total_stores'] += 1
                
                self.logger.debug(f"Stored item {item.id} (buffer size: {len(self.items)})")
                
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to store item: {e}")
                return False
    
    def retrieve(self, item_id: str) -> Optional[MemoryItem]:
        """Retrieve an item from the buffer"""
        
        with self._lock:
            self.stats['total_retrievals'] += 1
            
            if item_id in self.items:
                item = self.items[item_id]
                item.update_access()
                self._update_access_order(item_id)
                
                self.stats['cache_hits'] += 1
                
                return item
            else:
                self.stats['cache_misses'] += 1
                return None
    
    def update_item(self, item_id: str, new_content: Any = None,
                   new_priority: ItemPriority = None, 
                   new_tags: List[str] = None) -> bool:
        """Update an existing item"""
        
        with self._lock:
            if item_id not in self.items:
                return False
            
            item = self.items[item_id]
            
            if new_content is not None:
                item.content = new_content
            
            if new_priority is not None:
                item.priority = new_priority
            
            if new_tags is not None:
                item.tags = new_tags
            
            item.update_access()
            self._update_access_order(item_id)
            
            return True
    
    def delete(self, item_id: str) -> bool:
        """Delete an item from the buffer"""
        
        with self._lock:
            if item_id in self.items:
                del self.items[item_id]
                
                # Remove from access order
                try:
                    self.access_order.remove(item_id)
                except ValueError:
                    pass  # Item might not be in deque
                
                return True
            
            return False
    
    def clear(self) -> None:
        """Clear all items from the buffer"""
        
        with self._lock:
            self.items.clear()
            self.access_order.clear()
            
            self.logger.debug("Buffer cleared")
    
    def get_items_by_priority(self, priority: ItemPriority) -> List[MemoryItem]:
        """Get all items with specified priority"""
        
        with self._lock:
            return [item for item in self.items.values() 
                   if item.priority == priority]
    
    def get_items_by_tags(self, tags: List[str]) -> List[MemoryItem]:
        """Get items containing any of the specified tags"""
        
        with self._lock:
            matching_items = []
            
            for item in self.items.values():
                if any(tag in item.tags for tag in tags):
                    item.update_access()
                    matching_items.append(item)
            
            return matching_items
    
    def search_content(self, query: str) -> List[MemoryItem]:
        """Search items by content"""
        
        with self._lock:
            matching_items = []
            query_lower = query.lower()
            
            for item in self.items.values():
                # Simple content search
                content_str = str(item.content).lower()
                context_str = item.context.lower()
                
                if query_lower in content_str or query_lower in context_str:
                    item.update_access()
                    matching_items.append(item)
            
            return matching_items
    
    def apply_decay(self) -> None:
        """Apply temporal decay to all items"""
        
        with self._lock:
            current_time = datetime.now()
            
            for item in self.items.values():
                time_delta = current_time - item.last_accessed
                item.apply_decay(time_delta)
    
    def _evict_items(self) -> None:
        """Evict items according to eviction policy"""
        
        items_list = list(self.items.values())
        items_to_evict = self.eviction_policy.select_for_eviction(
            items_list, self.capacity - 1  # Make room for new item
        )
        
        for item_id in items_to_evict:
            if item_id in self.items:
                del self.items[item_id]
                
                try:
                    self.access_order.remove(item_id)
                except ValueError:
                    pass
                
                self.stats['total_evictions'] += 1
        
        if items_to_evict:
            self.logger.debug(f"Evicted {len(items_to_evict)} items")
    
    def _update_access_order(self, item_id: str) -> None:
        """Update access order for LRU tracking"""
        
        try:
            # Remove from current position
            self.access_order.remove(item_id)
        except ValueError:
            pass  # Item might not be in deque
        
        # Add to end (most recent)
        self.access_order.append(item_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        
        with self._lock:
            cache_hit_rate = (self.stats['cache_hits'] / 
                            max(1, self.stats['total_retrievals']))
            
            priority_distribution = {}
            for priority in ItemPriority:
                priority_distribution[priority.name] = len(
                    self.get_items_by_priority(priority)
                )
            
            return {
                'capacity': self.capacity,
                'current_size': len(self.items),
                'utilization': len(self.items) / self.capacity,
                'cache_hit_rate': cache_hit_rate,
                'priority_distribution': priority_distribution,
                'operations': self.stats.copy()
            }

class WorkingMemoryManager:
    """Manages working memory operations and context"""
    
    def __init__(self, buffer_capacity: int = 100):
        self.buffer = MemoryBuffer(capacity=buffer_capacity)
        
        # Context tracking
        self.current_context = ""
        self.context_stack: List[str] = []
        
        # Operation tracking
        self.operation_history: List[Dict[str, Any]] = []
        
        self.logger = logging.getLogger("WorkingMemoryManager")
    
    def store_item(self, content: Any, priority: ItemPriority = ItemPriority.MEDIUM,
                  tags: List[str] = None, context: str = "",
                  item_id: str = None) -> str:
        """Store an item in working memory"""
        
        if tags is None:
            tags = []
        
        if item_id is None:
            item_id = str(uuid.uuid4())
        
        # Create memory item
        item = MemoryItem(
            id=item_id,
            content=content,
            priority=priority,
            tags=tags,
            context=context or self.current_context
        )
        
        # Store in buffer
        success = self.buffer.store(item)
        
        if success:
            self._log_operation(MemoryOperation.STORE, item_id, 
                              f"Stored item with priority {priority.name}")
            return item_id
        else:
            raise Exception("Failed to store item in working memory")
    
    def retrieve_item(self, item_id: str) -> Optional[Any]:
        """Retrieve an item from working memory"""
        
        item = self.buffer.retrieve(item_id)
        
        if item:
            self._log_operation(MemoryOperation.RETRIEVE, item_id, "Retrieved item")
            return item.content
        else:
            self._log_operation(MemoryOperation.RETRIEVE, item_id, "Item not found")
            return None
    
    def update_item(self, item_id: str, new_content: Any = None,
                   new_priority: ItemPriority = None,
                   new_tags: List[str] = None) -> bool:
        """Update an existing item"""
        
        success = self.buffer.update_item(item_id, new_content, new_priority, new_tags)
        
        if success:
            self._log_operation(MemoryOperation.UPDATE, item_id, "Updated item")
        
        return success
    
    def find_by_tags(self, tags: List[str]) -> List[Tuple[str, Any]]:
        """Find items by tags"""
        
        items = self.buffer.get_items_by_tags(tags)
        return [(item.id, item.content) for item in items]
    
    def search(self, query: str) -> List[Tuple[str, Any]]:
        """Search items by content"""
        
        items = self.buffer.search_content(query)
        return [(item.id, item.content) for item in items]
    
    @contextmanager
    def context(self, context_name: str):
        """Context manager for working memory context"""
        
        # Push new context
        self.context_stack.append(self.current_context)
        self.current_context = context_name
        
        self.logger.debug(f"Entered context: {context_name}")
        
        try:
            yield self
        finally:
            # Pop context
            self.current_context = self.context_stack.pop() if self.context_stack else ""
            self.logger.debug(f"Exited context: {context_name}")
    
    def clear_context(self, context_name: str) -> int:
        """Clear all items from a specific context"""
        
        items_to_remove = []
        
        for item_id, item in self.buffer.items.items():
            if item.context == context_name:
                items_to_remove.append(item_id)
        
        # Remove items
        for item_id in items_to_remove:
            self.buffer.delete(item_id)
        
        self._log_operation(MemoryOperation.CLEAR, "", 
                          f"Cleared context {context_name} ({len(items_to_remove)} items)")
        
        return len(items_to_remove)
    
    def prune_low_importance(self, threshold: float = 0.3) -> int:
        """Remove items with low importance scores"""
        
        items_to_remove = []
        
        for item_id, item in self.buffer.items.items():
            if item.importance < threshold and item.priority != ItemPriority.CRITICAL:
                items_to_remove.append(item_id)
        
        # Remove items
        for item_id in items_to_remove:
            self.buffer.delete(item_id)
        
        self._log_operation(MemoryOperation.PRUNE, "", 
                          f"Pruned {len(items_to_remove)} low-importance items")
        
        return len(items_to_remove)
    
    def maintain_memory(self) -> Dict[str, int]:
        """Perform memory maintenance operations"""
        
        # Apply decay
        self.buffer.apply_decay()
        
        # Prune low-importance items
        pruned_count = self.prune_low_importance()
        
        return {
            'items_pruned': pruned_count,
            'current_size': len(self.buffer.items),
            'utilization': len(self.buffer.items) / self.buffer.capacity
        }
    
    def _log_operation(self, operation: MemoryOperation, item_id: str, 
                      description: str) -> None:
        """Log memory operation"""
        
        log_entry = {
            'timestamp': datetime.now(),
            'operation': operation.value,
            'item_id': item_id,
            'description': description,
            'context': self.current_context
        }
        
        self.operation_history.append(log_entry)
        
        # Keep only recent history
        if len(self.operation_history) > 1000:
            self.operation_history = self.operation_history[-500:]
    
    def get_status(self) -> Dict[str, Any]:
        """Get current memory status"""
        
        buffer_stats = self.buffer.get_statistics()
        
        return {
            'buffer_statistics': buffer_stats,
            'current_context': self.current_context,
            'context_stack_depth': len(self.context_stack),
            'operation_count': len(self.operation_history),
            'recent_operations': self.operation_history[-5:] if self.operation_history else []
        }

class ShortTermMemorySystem:
    """Complete short-term memory system with advanced features"""
    
    def __init__(self, capacity: int = 200, enable_auto_maintenance: bool = True):
        # Core components
        self.working_memory = WorkingMemoryManager(buffer_capacity=capacity)
        
        # Auto-maintenance
        self.auto_maintenance = enable_auto_maintenance
        self.maintenance_interval = 300  # 5 minutes
        self.last_maintenance = datetime.now()
        
        # Advanced features
        self.relationship_graph: Dict[str, List[str]] = {}
        self.temporal_chunks: Dict[str, List[str]] = {}  # Time-based grouping
        
        # Statistics
        self.stats = {
            'total_sessions': 0,
            'items_processed': 0,
            'contexts_used': 0,
            'maintenance_runs': 0
        }
        
        self.logger = logging.getLogger("ShortTermMemorySystem")
    
    async def initialize(self) -> None:
        """Initialize the short-term memory system"""
        self.logger.info("Short-term memory system initialized")
    
    def create_session(self, session_name: str) -> 'MemorySession':
        """Create a new memory session"""
        
        self.stats['total_sessions'] += 1
        return MemorySession(self, session_name)
    
    def add_relationship(self, item1_id: str, item2_id: str, 
                        bidirectional: bool = True) -> None:
        """Add relationship between memory items"""
        
        if item1_id not in self.relationship_graph:
            self.relationship_graph[item1_id] = []
        
        if item2_id not in self.relationship_graph[item1_id]:
            self.relationship_graph[item1_id].append(item2_id)
        
        if bidirectional:
            if item2_id not in self.relationship_graph:
                self.relationship_graph[item2_id] = []
            
            if item1_id not in self.relationship_graph[item2_id]:
                self.relationship_graph[item2_id].append(item1_id)
    
    def get_related_items(self, item_id: str, depth: int = 1) -> List[str]:
        """Get related items up to specified depth"""
        
        if depth <= 0 or item_id not in self.relationship_graph:
            return []
        
        related = []
        to_explore = [(item_id, 0)]
        visited = set()
        
        while to_explore:
            current_id, current_depth = to_explore.pop(0)
            
            if current_id in visited or current_depth >= depth:
                continue
            
            visited.add(current_id)
            
            if current_id in self.relationship_graph:
                for related_id in self.relationship_graph[current_id]:
                    if related_id not in visited:
                        related.append(related_id)
                        to_explore.append((related_id, current_depth + 1))
        
        return related
    
    def create_temporal_chunk(self, chunk_id: str, item_ids: List[str]) -> None:
        """Group items into temporal chunks"""
        
        self.temporal_chunks[chunk_id] = item_ids
    
    def get_temporal_chunk(self, chunk_id: str) -> List[Tuple[str, Any]]:
        """Get all items in a temporal chunk"""
        
        if chunk_id not in self.temporal_chunks:
            return []
        
        items = []
        for item_id in self.temporal_chunks[chunk_id]:
            content = self.working_memory.retrieve_item(item_id)
            if content is not None:
                items.append((item_id, content))
        
        return items
    
    def perform_maintenance(self) -> Dict[str, Any]:
        """Perform system maintenance"""
        
        if not self.auto_maintenance:
            return {}
        
        current_time = datetime.now()
        time_since_maintenance = current_time - self.last_maintenance
        
        if time_since_maintenance.total_seconds() < self.maintenance_interval:
            return {}
        
        # Perform maintenance
        maintenance_results = self.working_memory.maintain_memory()
        
        # Clean up broken relationships
        cleanup_results = self._cleanup_relationships()
        maintenance_results.update(cleanup_results)
        
        # Update maintenance timestamp
        self.last_maintenance = current_time
        self.stats['maintenance_runs'] += 1
        
        self.logger.debug(f"Maintenance completed: {maintenance_results}")
        
        return maintenance_results
    
    def _cleanup_relationships(self) -> Dict[str, int]:
        """Clean up relationships to non-existent items"""
        
        existing_items = set(self.working_memory.buffer.items.keys())
        relationships_removed = 0
        
        for item_id in list(self.relationship_graph.keys()):
            if item_id not in existing_items:
                del self.relationship_graph[item_id]
                relationships_removed += 1
            else:
                # Clean up references to non-existent items
                valid_relationships = [
                    rel_id for rel_id in self.relationship_graph[item_id]
                    if rel_id in existing_items
                ]
                
                if len(valid_relationships) != len(self.relationship_graph[item_id]):
                    relationships_removed += len(self.relationship_graph[item_id]) - len(valid_relationships)
                    self.relationship_graph[item_id] = valid_relationships
        
        return {'relationships_cleaned': relationships_removed}
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        memory_status = self.working_memory.get_status()
        
        return {
            'memory_status': memory_status,
            'relationships': {
                'total_items_with_relationships': len(self.relationship_graph),
                'total_relationships': sum(len(rels) for rels in self.relationship_graph.values())
            },
            'temporal_chunks': {
                'total_chunks': len(self.temporal_chunks),
                'total_chunked_items': sum(len(items) for items in self.temporal_chunks.values())
            },
            'system_statistics': self.stats,
            'maintenance': {
                'auto_maintenance_enabled': self.auto_maintenance,
                'last_maintenance': self.last_maintenance,
                'maintenance_runs': self.stats['maintenance_runs']
            }
        }

class MemorySession:
    """A working session with short-term memory"""
    
    def __init__(self, memory_system: ShortTermMemorySystem, session_name: str):
        self.memory_system = memory_system
        self.session_name = session_name
        self.session_id = str(uuid.uuid4())
        
        # Session-specific tracking
        self.session_items: List[str] = []
        self.created_at = datetime.now()
        
        self.logger = logging.getLogger(f"MemorySession-{session_name}")
    
    def __enter__(self):
        """Enter the memory session context"""
        self.context_manager = self.memory_system.working_memory.context(self.session_name)
        self.context_manager.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the memory session context"""
        self.context_manager.__exit__(exc_type, exc_val, exc_tb)
    
    def store(self, content: Any, priority: ItemPriority = ItemPriority.MEDIUM,
             tags: List[str] = None) -> str:
        """Store an item in this session"""
        
        item_id = self.memory_system.working_memory.store_item(
            content, priority, tags, self.session_name
        )
        
        self.session_items.append(item_id)
        return item_id
    
    def retrieve(self, item_id: str) -> Optional[Any]:
        """Retrieve an item"""
        return self.memory_system.working_memory.retrieve_item(item_id)
    
    def search(self, query: str) -> List[Tuple[str, Any]]:
        """Search items in this session"""
        return self.memory_system.working_memory.search(query)
    
    def find_by_tags(self, tags: List[str]) -> List[Tuple[str, Any]]:
        """Find items by tags in this session"""
        return self.memory_system.working_memory.find_by_tags(tags)
    
    def create_relationship(self, item1_id: str, item2_id: str) -> None:
        """Create relationship between items"""
        self.memory_system.add_relationship(item1_id, item2_id)
    
    def get_related(self, item_id: str) -> List[str]:
        """Get items related to the given item"""
        return self.memory_system.get_related_items(item_id)
    
    def clear_session(self) -> int:
        """Clear all items from this session"""
        return self.memory_system.working_memory.clear_context(self.session_name)
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get information about this session"""
        
        active_items = sum(1 for item_id in self.session_items 
                          if item_id in self.memory_system.working_memory.buffer.items)
        
        return {
            'session_name': self.session_name,
            'session_id': self.session_id,
            'created_at': self.created_at,
            'total_items_created': len(self.session_items),
            'active_items': active_items,
            'duration': datetime.now() - self.created_at
        }

# EXAMPLE DEMONSTRATIONS
# =====================

async def demo_basic_memory_operations():
    """Demo: Basic short-term memory operations"""
    print("\nDEMO 1: BASIC MEMORY OPERATIONS")
    print("=" * 50)
    
    memory_system = ShortTermMemorySystem(capacity=10)
    await memory_system.initialize()
    
    # Store different types of information
    print("Storing various items in short-term memory:")
    
    # Mathematical calculation components
    item1_id = memory_system.working_memory.store_item(
        {"operation": "multiplication", "operand1": 25, "operand2": 4, "result": 100},
        ItemPriority.HIGH,
        ["math", "calculation"]
    )
    print(f"  Stored calculation: 25 × 4 = 100 (ID: {item1_id[:8]}...)")
    
    # Temporary text processing
    item2_id = memory_system.working_memory.store_item(
        "The quick brown fox jumps over the lazy dog",
        ItemPriority.MEDIUM,
        ["text", "processing"]
    )
    print(f"  Stored text for processing (ID: {item2_id[:8]}...)")
    
    # Problem-solving context
    item3_id = memory_system.working_memory.store_item(
        {"problem": "route_planning", "origin": "A", "destination": "C", "intermediate": "B"},
        ItemPriority.CRITICAL,
        ["planning", "route"]
    )
    print(f"  Stored route planning context (ID: {item3_id[:8]}...)")
    
    # Retrieve items
    print(f"\nRetrieving stored items:")
    
    calc_result = memory_system.working_memory.retrieve_item(item1_id)
    print(f"  Retrieved calculation: {calc_result}")
    
    route_context = memory_system.working_memory.retrieve_item(item3_id)
    print(f"  Retrieved route context: {route_context}")
    
    # Search by tags
    print(f"\nSearching by tags:")
    
    math_items = memory_system.working_memory.find_by_tags(["math"])
    print(f"  Math-tagged items: {len(math_items)}")
    for item_id, content in math_items:
        print(f"    {item_id[:8]}...: {content}")
    
    # Search by content
    print(f"\nSearching by content:")
    
    text_items = memory_system.working_memory.search("fox")
    print(f"  Items containing 'fox': {len(text_items)}")
    for item_id, content in text_items:
        print(f"    {item_id[:8]}...: {content[:50]}...")
    
    # Show memory status
    print(f"\nMemory status:")
    status = memory_system.get_comprehensive_status()
    
    buffer_stats = status['memory_status']['buffer_statistics']
    print(f"  Capacity: {buffer_stats['capacity']}")
    print(f"  Current size: {buffer_stats['current_size']}")
    print(f"  Utilization: {buffer_stats['utilization']:.1%}")

async def demo_memory_sessions():
    """Demo: Using memory sessions for context management"""
    print("\nDEMO 2: MEMORY SESSIONS")
    print("=" * 50)
    
    memory_system = ShortTermMemorySystem(capacity=20)
    await memory_system.initialize()
    
    # Simulate multiple problem-solving sessions
    print("Running multiple problem-solving sessions:")
    
    # Session 1: Mathematical problem
    print(f"\nSession 1: Solving compound interest problem")
    with memory_system.create_session("compound_interest") as session:
        
        # Store problem parameters
        principal_id = session.store(
            {"name": "principal", "value": 1000, "description": "Initial investment"},
            ItemPriority.CRITICAL,
            ["principal", "money"]
        )
        
        rate_id = session.store(
            {"name": "interest_rate", "value": 0.05, "description": "Annual interest rate"},
            ItemPriority.CRITICAL,
            ["rate", "percentage"]
        )
        
        time_id = session.store(
            {"name": "time_period", "value": 3, "description": "Years"},
            ItemPriority.CRITICAL,
            ["time", "years"]
        )
        
        # Store intermediate calculation
        compound_factor_id = session.store(
            {"calculation": "(1 + rate)^time", "value": (1 + 0.05)**3, "result": 1.157625},
            ItemPriority.HIGH,
            ["calculation", "intermediate"]
        )
        
        # Final result
        final_result_id = session.store(
            {"calculation": "principal × compound_factor", "value": 1000 * 1.157625, "result": 1157.63},
            ItemPriority.HIGH,
            ["result", "final"]
        )
        
        # Create relationships
        session.create_relationship(principal_id, final_result_id)
        session.create_relationship(rate_id, compound_factor_id)
        session.create_relationship(compound_factor_id, final_result_id)
        
        print(f"    Stored 5 items for compound interest calculation")
        
        # Verify calculation by retrieving components
        principal = session.retrieve(principal_id)
        final_result = session.retrieve(final_result_id)
        
        print(f"    Principal: ${principal['value']}")
        print(f"    Final amount: ${final_result['result']:.2f}")
        
        session_info = session.get_session_info()
        print(f"    Session items: {session_info['active_items']}")
    
    # Session 2: Text analysis problem
    print(f"\nSession 2: Text analysis task")
    with memory_system.create_session("text_analysis") as session:
        
        # Store text to analyze
        text_id = session.store(
            "Artificial intelligence will revolutionize how we work and live",
            ItemPriority.CRITICAL,
            ["text", "input"]
        )
        
        # Store analysis results
        word_count_id = session.store(
            {"metric": "word_count", "value": 10},
            ItemPriority.MEDIUM,
            ["analysis", "count"]
        )
        
        keywords_id = session.store(
            {"metric": "keywords", "value": ["artificial", "intelligence", "revolutionize", "work", "live"]},
            ItemPriority.MEDIUM,
            ["analysis", "keywords"]
        )
        
        sentiment_id = session.store(
            {"metric": "sentiment", "value": "positive", "confidence": 0.85},
            ItemPriority.MEDIUM,
            ["analysis", "sentiment"]
        )
        
        # Create relationships
        session.create_relationship(text_id, word_count_id)
        session.create_relationship(text_id, keywords_id)
        session.create_relationship(text_id, sentiment_id)
        
        print(f"    Stored text and 3 analysis results")
        
        # Show related items
        related_to_text = session.get_related(text_id)
        print(f"    Items related to text: {len(related_to_text)}")
        
        session_info = session.get_session_info()
        print(f"    Session items: {session_info['active_items']}")
    
    # Show overall system status
    print(f"\nOverall system status after sessions:")
    status = memory_system.get_comprehensive_status()
    
    print(f"  Total sessions: {status['system_statistics']['total_sessions']}")
    print(f"  Current memory size: {status['memory_status']['buffer_statistics']['current_size']}")
    print(f"  Items with relationships: {status['relationships']['total_items_with_relationships']}")
    print(f"  Total relationships: {status['relationships']['total_relationships']}")

async def demo_memory_eviction():
    """Demo: Memory eviction and capacity management"""
    print("\nDEMO 3: MEMORY EVICTION AND CAPACITY")
    print("=" * 50)
    
    # Create small memory system to demonstrate eviction
    memory_system = ShortTermMemorySystem(capacity=5)
    await memory_system.initialize()
    
    print("Creating memory system with small capacity (5 items)")
    print("Demonstrating eviction policies when memory fills up")
    
    # Store items with different priorities
    print(f"\nStoring items with different priorities:")
    
    items_data = [
        ("Critical system info", ItemPriority.CRITICAL, ["system", "critical"]),
        ("High priority task", ItemPriority.HIGH, ["task", "important"]),
        ("Medium priority data", ItemPriority.MEDIUM, ["data", "normal"]),
        ("Low priority note", ItemPriority.LOW, ["note", "minor"]),
        ("Minimal importance item", ItemPriority.MINIMAL, ["temp", "minimal"])
    ]
    
    stored_ids = []
    
    for content, priority, tags in items_data:
        item_id = memory_system.working_memory.store_item(content, priority, tags)
        stored_ids.append(item_id)
        
        status = memory_system.working_memory.get_status()
        current_size = status['buffer_statistics']['current_size']
        print(f"  Stored '{content}' ({priority.name}): Memory size: {current_size}")
    
    print(f"\nMemory is now at capacity: {len(stored_ids)} items")
    
    # Add more items to trigger eviction
    print(f"\nAdding more items to trigger eviction:")
    
    additional_items = [
        ("New critical item", ItemPriority.CRITICAL, ["new", "critical"]),
        ("New high priority", ItemPriority.HIGH, ["new", "high"]),
        ("New medium item", ItemPriority.MEDIUM, ["new", "medium"])
    ]
    
    for content, priority, tags in additional_items:
        print(f"\n  Adding '{content}' ({priority.name})")
        
        # Check current items before adding
        before_items = list(memory_system.working_memory.buffer.items.keys())
        
        item_id = memory_system.working_memory.store_item(content, priority, tags)
        
        # Check which items remain after adding
        after_items = list(memory_system.working_memory.buffer.items.keys())
        
        evicted_items = set(before_items) - set(after_items)
        
        if evicted_items:
            print(f"    Evicted {len(evicted_items)} items")
            for evicted_id in evicted_items:
                # Try to identify what was evicted
                original_item = next((data for data in items_data + additional_items 
                                    if data[0] in str(evicted_id)), None)
                print(f"      Evicted item (ID: {evicted_id[:8]}...)")
        else:
            print(f"    No items evicted")
        
        status = memory_system.working_memory.get_status()
        print(f"    Current memory size: {status['buffer_statistics']['current_size']}")
    
    # Show final memory contents
    print(f"\nFinal memory contents:")
    for item_id, item in memory_system.working_memory.buffer.items.items():
        print(f"  {item.content} ({item.priority.name})")
    
    # Show eviction statistics
    buffer_stats = memory_system.working_memory.get_status()['buffer_statistics']
    print(f"\nEviction statistics:")
    print(f"  Total evictions: {buffer_stats['operations']['total_evictions']}")
    print(f"  Cache hit rate: {buffer_stats['cache_hit_rate']:.2%}")

async def demo_memory_relationships():
    """Demo: Memory relationships and associative retrieval"""
    print("\nDEMO 4: MEMORY RELATIONSHIPS")
    print("=" * 50)
    
    memory_system = ShortTermMemorySystem(capacity=15)
    await memory_system.initialize()
    
    print("Building a knowledge network with relationships")
    
    # Create a network of related concepts
    with memory_system.create_session("knowledge_network") as session:
        
        # Core concepts
        ai_id = session.store(
            {"concept": "artificial_intelligence", "definition": "Intelligence exhibited by machines"},
            ItemPriority.CRITICAL,
            ["ai", "core"]
        )
        
        ml_id = session.store(
            {"concept": "machine_learning", "definition": "Learning without explicit programming"},
            ItemPriority.HIGH,
            ["ml", "learning"]
        )
        
        dl_id = session.store(
            {"concept": "deep_learning", "definition": "Learning with deep neural networks"},
            ItemPriority.HIGH,
            ["dl", "neural"]
        )
        
        # Applications
        nlp_id = session.store(
            {"concept": "natural_language_processing", "application": "Understanding human language"},
            ItemPriority.MEDIUM,
            ["nlp", "language"]
        )
        
        cv_id = session.store(
            {"concept": "computer_vision", "application": "Understanding visual information"},
            ItemPriority.MEDIUM,
            ["cv", "vision"]
        )
        
        # Techniques
        nn_id = session.store(
            {"concept": "neural_networks", "technique": "Interconnected nodes mimicking brain"},
            ItemPriority.MEDIUM,
            ["nn", "network"]
        )
        
        # Create hierarchical relationships
        print(f"\nCreating relationships:")
        
        # AI is parent of ML
        session.create_relationship(ai_id, ml_id)
        print(f"  AI ↔ Machine Learning")
        
        # ML is parent of DL
        session.create_relationship(ml_id, dl_id)
        print(f"  Machine Learning ↔ Deep Learning")
        
        # DL uses Neural Networks
        session.create_relationship(dl_id, nn_id)
        print(f"  Deep Learning ↔ Neural Networks")
        
        # Applications use ML techniques
        session.create_relationship(ml_id, nlp_id)
        session.create_relationship(ml_id, cv_id)
        print(f"  Machine Learning ↔ NLP")
        print(f"  Machine Learning ↔ Computer Vision")
        
        # Cross-connections
        session.create_relationship(dl_id, nlp_id)
        session.create_relationship(dl_id, cv_id)
        print(f"  Deep Learning ↔ NLP")
        print(f"  Deep Learning ↔ Computer Vision")
        
        # Demonstrate associative retrieval
        print(f"\nDemonstrating associative retrieval:")
        
        # Find items related to AI
        ai_related = session.get_related(ai_id)
        print(f"\nItems directly related to AI: {len(ai_related)}")
        for related_id in ai_related:
            item = session.retrieve(related_id)
            print(f"  {item['concept']}")
        
        # Find items related to Deep Learning
        dl_related = session.get_related(dl_id)
        print(f"\nItems directly related to Deep Learning: {len(dl_related)}")
        for related_id in dl_related:
            item = session.retrieve(related_id)
            print(f"  {item['concept']}")
        
        # Show multi-hop relationships
        print(f"\nExploring relationship network:")
        all_related = memory_system.get_related_items(ai_id, depth=3)
        print(f"Items related to AI (up to 3 hops): {len(all_related)}")
        
        for related_id in all_related:
            item = session.retrieve(related_id)
            if item:
                print(f"  {item['concept']}")
    
    # Show relationship statistics
    print(f"\nRelationship network statistics:")
    status = memory_system.get_comprehensive_status()
    
    print(f"  Items with relationships: {status['relationships']['total_items_with_relationships']}")
    print(f"  Total relationships: {status['relationships']['total_relationships']}")

async def demo_memory_maintenance():
    """Demo: Automatic memory maintenance and optimization"""
    print("\nDEMO 5: MEMORY MAINTENANCE")
    print("=" * 50)
    
    memory_system = ShortTermMemorySystem(capacity=20, enable_auto_maintenance=True)
    await memory_system.initialize()
    
    print("Demonstrating automatic memory maintenance")
    
    # Simulate different types of memory usage
    print(f"\nSimulating various memory usage patterns:")
    
    with memory_system.create_session("maintenance_demo") as session:
        
        # Create items with different access patterns
        frequently_used_id = session.store(
            "Frequently accessed data",
            ItemPriority.HIGH,
            ["frequent", "important"]
        )
        
        occasionally_used_id = session.store(
            "Occasionally accessed data",
            ItemPriority.MEDIUM,
            ["occasional", "normal"]
        )
        
        rarely_used_id = session.store(
            "Rarely accessed data",
            ItemPriority.LOW,
            ["rare", "archival"]
        )
        
        print(f"    Created items with different usage patterns")
        
        # Simulate access patterns
        print(f"\nSimulating access patterns over time:")
        
        # Frequent access to first item
        for i in range(10):
            session.retrieve(frequently_used_id)
        print(f"    Accessed frequent item 10 times")
        
        # Occasional access to second item
        for i in range(3):
            session.retrieve(occasionally_used_id)
        print(f"    Accessed occasional item 3 times")
        
        # Rare access to third item (only once)
        session.retrieve(rarely_used_id)
        print(f"    Accessed rare item 1 time")
        
        # Show items before maintenance
        print(f"\nMemory state before maintenance:")
        for item_id, item in memory_system.working_memory.buffer.items.items():
            print(f"    {item.content[:30]}... (importance: {item.importance:.2f}, "
                  f"access count: {item.access_count}, priority: {item.priority.name})")
        
        # Manually apply decay to simulate time passage
        print(f"\nApplying temporal decay to simulate time passage...")
        memory_system.working_memory.buffer.apply_decay()
        
        # Update importance scores based on access patterns
        for item in memory_system.working_memory.buffer.items.values():
            # Simulate time-based decay
            if item.access_count < 2:
                item.importance *= 0.5  # Heavy decay for rarely accessed items
            elif item.access_count < 5:
                item.importance *= 0.8  # Moderate decay
            # Frequently accessed items maintain importance
        
        print(f"    Applied decay based on access patterns")
        
        # Show items after decay
        print(f"\nMemory state after decay:")
        for item_id, item in memory_system.working_memory.buffer.items.items():
            print(f"    {item.content[:30]}... (importance: {item.importance:.2f}, "
                  f"access count: {item.access_count})")
        
        # Perform maintenance
        print(f"\nPerforming memory maintenance:")
        maintenance_results = memory_system.perform_maintenance()
        
        print(f"    Maintenance results: {maintenance_results}")
        
        # Show items after maintenance
        print(f"\nMemory state after maintenance:")
        remaining_items = len(memory_system.working_memory.buffer.items)
        print(f"    Remaining items: {remaining_items}")
        
        for item_id, item in memory_system.working_memory.buffer.items.items():
            print(f"    {item.content[:30]}... (importance: {item.importance:.2f}, "
                  f"retained due to: {item.priority.name} priority)")
        
        # Show maintenance statistics
        print(f"\nMaintenance statistics:")
        status = memory_system.get_comprehensive_status()
        
        maintenance_info = status['maintenance']
        print(f"    Auto-maintenance enabled: {maintenance_info['auto_maintenance_enabled']}")
        print(f"    Maintenance runs: {maintenance_info['maintenance_runs']}")
        print(f"    Last maintenance: {maintenance_info['last_maintenance']}")
        
        # Show buffer statistics
        buffer_stats = status['memory_status']['buffer_statistics']
        print(f"    Current utilization: {buffer_stats['utilization']:.1%}")
        print(f"    Cache hit rate: {buffer_stats['cache_hit_rate']:.1%}")

async def main():
    """
    Demonstrate Short-Term Memory for temporary information storage and processing
    
    WHAT YOU'LL LEARN:
    ================
    1. How to implement working memory buffers with capacity management
    2. How to use different eviction policies (LRU, importance-based, hybrid)
    3. How to manage memory sessions and contexts for organized processing
    4. How to create and use memory relationships for associative retrieval
    5. How to implement automatic memory maintenance and optimization
    6. How to balance memory capacity with processing requirements
    
    REAL WORLD APPLICATIONS:
    =======================
    - Mathematical computation engines maintaining intermediate results
    - Natural language processing systems holding working context
    - Problem-solving agents managing multi-step reasoning
    - Code generation systems maintaining variable and function contexts
    - Planning agents storing intermediate goals and constraints
    - Game AI maintaining current game state and tactical information
    """
    
    print("SHORT-TERM MEMORY DEMONSTRATION")
    print("Temporary information storage for active processing!")
    
    await demo_basic_memory_operations()
    await demo_memory_sessions()
    await demo_memory_eviction()
    await demo_memory_relationships()
    await demo_memory_maintenance()
    
    print("\nWHAT WE LEARNED:")
    print("=" * 40)
    print("✓ Working memory enables complex multi-step reasoning")
    print("✓ Eviction policies manage memory capacity intelligently")
    print("✓ Memory sessions provide organized context management")
    print("✓ Relationships enable associative memory retrieval")
    print("✓ Automatic maintenance optimizes memory performance")
    print("✓ Priority systems protect critical working information")
    print("\nTHE POWER OF SHORT-TERM MEMORY:")
    print("- Enables AI agents to handle complex reasoning like humans")
    print("- Supports natural problem decomposition and solution building")
    print("- Provides temporary workspace for cognitive operations")
    print("- Creates foundation for advanced cognitive architectures")
    print("- Enables verification and error correction during processing")

if __name__ == "__main__":
    asyncio.run(main())
