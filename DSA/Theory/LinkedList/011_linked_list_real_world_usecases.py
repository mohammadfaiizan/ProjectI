"""
Linked List Real-World Use Cases - Practical Applications in Software Systems
This module demonstrates how linked lists are used in real-world applications and systems.
"""

from typing import Optional, Any, List, Dict, Tuple, Callable
import hashlib
import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

class ListNode:
    """Basic node class for linked list"""
    def __init__(self, key: Any = None, val: Any = None, next: Optional['ListNode'] = None):
        self.key = key
        self.val = val
        self.next = next
    
    def __repr__(self):
        return f"ListNode({self.key}: {self.val})"

class DoublyListNode:
    """Node for doubly linked list"""
    def __init__(self, key: Any = None, val: Any = None, 
                 prev: Optional['DoublyListNode'] = None, 
                 next: Optional['DoublyListNode'] = None):
        self.key = key
        self.val = val
        self.prev = prev
        self.next = next
    
    def __repr__(self):
        return f"DoublyListNode({self.key}: {self.val})"

# ==================== HASHMAP WITH LINKED LIST BUCKETS ====================

class HashMapWithLL:
    """
    HashMap implementation using linked lists for collision resolution
    
    This demonstrates how linked lists are used in real hash table implementations
    to handle collisions via chaining method.
    """
    
    def __init__(self, initial_capacity: int = 16, load_factor: float = 0.75):
        """
        Initialize HashMap with linked list buckets
        
        Args:
            initial_capacity: Initial number of buckets
            load_factor: Threshold for resizing (size/capacity)
        """
        self.capacity = initial_capacity
        self.size = 0
        self.load_factor = load_factor
        self.buckets: List[Optional[ListNode]] = [None] * self.capacity
        self.collision_count = 0
        self.resize_count = 0
    
    def _hash(self, key: Any) -> int:
        """
        Hash function to map keys to bucket indices
        
        Args:
            key: Key to hash
        
        Returns:
            Bucket index
        """
        # Simple hash using Python's hash function
        return hash(key) % self.capacity
    
    def put(self, key: Any, value: Any) -> None:
        """
        Insert or update key-value pair
        
        Time Complexity: O(1) average, O(n) worst case
        Space Complexity: O(1)
        
        Args:
            key: Key to insert/update
            value: Value to store
        """
        bucket_index = self._hash(key)
        head = self.buckets[bucket_index]
        
        # Search for existing key
        current = head
        while current:
            if current.key == key:
                # Update existing key
                current.val = value
                return
            current = current.next
        
        # Key not found, insert new node at beginning of chain
        new_node = ListNode(key, value)
        new_node.next = head
        self.buckets[bucket_index] = new_node
        self.size += 1
        
        # Count collision if bucket wasn't empty
        if head is not None:
            self.collision_count += 1
        
        # Resize if load factor exceeded
        if self.size > self.capacity * self.load_factor:
            self._resize()
    
    def get(self, key: Any) -> Any:
        """
        Retrieve value by key
        
        Time Complexity: O(1) average, O(n) worst case
        
        Args:
            key: Key to retrieve
        
        Returns:
            Value or None if not found
        """
        bucket_index = self._hash(key)
        current = self.buckets[bucket_index]
        
        while current:
            if current.key == key:
                return current.val
            current = current.next
        
        return None
    
    def delete(self, key: Any) -> bool:
        """
        Delete key-value pair
        
        Args:
            key: Key to delete
        
        Returns:
            True if deleted, False if not found
        """
        bucket_index = self._hash(key)
        current = self.buckets[bucket_index]
        prev = None
        
        while current:
            if current.key == key:
                if prev:
                    prev.next = current.next
                else:
                    self.buckets[bucket_index] = current.next
                
                self.size -= 1
                return True
            
            prev = current
            current = current.next
        
        return False
    
    def _resize(self) -> None:
        """Resize the hash table when load factor is exceeded"""
        old_buckets = self.buckets
        old_capacity = self.capacity
        
        # Double the capacity
        self.capacity *= 2
        self.size = 0
        self.collision_count = 0
        self.resize_count += 1
        self.buckets = [None] * self.capacity
        
        # Rehash all existing elements
        for head in old_buckets:
            current = head
            while current:
                self.put(current.key, current.val)
                current = current.next
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics about the hash table"""
        # Calculate chain lengths
        chain_lengths = []
        max_chain_length = 0
        empty_buckets = 0
        
        for head in self.buckets:
            length = 0
            current = head
            while current:
                length += 1
                current = current.next
            
            chain_lengths.append(length)
            max_chain_length = max(max_chain_length, length)
            if length == 0:
                empty_buckets += 1
        
        avg_chain_length = sum(chain_lengths) / len(chain_lengths)
        
        return {
            "size": self.size,
            "capacity": self.capacity,
            "load_factor": self.size / self.capacity,
            "collision_count": self.collision_count,
            "resize_count": self.resize_count,
            "max_chain_length": max_chain_length,
            "avg_chain_length": avg_chain_length,
            "empty_buckets": empty_buckets,
            "bucket_utilization": (self.capacity - empty_buckets) / self.capacity
        }
    
    def display_buckets(self, max_buckets: int = 10) -> List[str]:
        """Display contents of first few buckets"""
        result = []
        
        for i in range(min(max_buckets, self.capacity)):
            bucket_content = []
            current = self.buckets[i]
            
            while current:
                bucket_content.append(f"{current.key}:{current.val}")
                current = current.next
            
            if bucket_content:
                result.append(f"Bucket {i}: {' -> '.join(bucket_content)}")
            else:
                result.append(f"Bucket {i}: Empty")
        
        return result

class ThreadSafeHashMap:
    """
    Thread-safe HashMap using linked lists with locking
    """
    
    def __init__(self, initial_capacity: int = 16):
        """Initialize thread-safe HashMap"""
        self.hashmap = HashMapWithLL(initial_capacity)
        self.lock = threading.RLock()
    
    def put(self, key: Any, value: Any) -> None:
        """Thread-safe put operation"""
        with self.lock:
            self.hashmap.put(key, value)
    
    def get(self, key: Any) -> Any:
        """Thread-safe get operation"""
        with self.lock:
            return self.hashmap.get(key)
    
    def delete(self, key: Any) -> bool:
        """Thread-safe delete operation"""
        with self.lock:
            return self.hashmap.delete(key)
    
    def size(self) -> int:
        """Get current size thread-safely"""
        with self.lock:
            return self.hashmap.size

# ==================== UNDO/REDO OPERATIONS ====================

class OperationType(Enum):
    """Types of operations for undo/redo system"""
    INSERT = "insert"
    DELETE = "delete"
    MODIFY = "modify"
    MOVE = "move"
    REPLACE = "replace"

@dataclass
class Operation:
    """Represents an operation that can be undone/redone"""
    op_type: OperationType
    data: Dict[str, Any]
    timestamp: float
    description: str
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class UndoRedoSystem:
    """
    Undo/Redo system using doubly linked list for operations history
    
    This demonstrates how text editors, IDEs, and other applications
    implement undo/redo functionality using linked lists.
    """
    
    def __init__(self, max_operations: int = 100):
        """
        Initialize undo/redo system
        
        Args:
            max_operations: Maximum number of operations to keep in history
        """
        self.max_operations = max_operations
        self.current = None  # Current position in history
        self.size = 0
        self.operation_count = 0
    
    def execute_operation(self, operation: Operation) -> None:
        """
        Execute new operation and add to history
        
        Args:
            operation: Operation to execute
        """
        # Create new node for this operation
        new_node = DoublyListNode(key=self.operation_count, val=operation)
        self.operation_count += 1
        
        if not self.current:
            # First operation
            self.current = new_node
        else:
            # Clear forward history (operations after current)
            self.current.next = None
            
            # Add new operation
            new_node.prev = self.current
            self.current.next = new_node
            self.current = new_node
        
        self.size += 1
        
        # Maintain max operations limit
        if self.size > self.max_operations:
            self._remove_oldest()
    
    def undo(self) -> Optional[Operation]:
        """
        Undo last operation
        
        Returns:
            Operation that was undone, or None if nothing to undo
        """
        if not self.current:
            return None
        
        undone_operation = self.current.val
        
        # Move current pointer back
        if self.current.prev:
            self.current = self.current.prev
        else:
            # At the beginning, but keep the node for potential redo
            pass
        
        return undone_operation
    
    def redo(self) -> Optional[Operation]:
        """
        Redo next operation
        
        Returns:
            Operation that was redone, or None if nothing to redo
        """
        if not self.current:
            return None
        
        # Check if there's a next operation to redo
        if self.current.next:
            self.current = self.current.next
            return self.current.val
        
        return None
    
    def can_undo(self) -> bool:
        """Check if undo is possible"""
        return self.current is not None
    
    def can_redo(self) -> bool:
        """Check if redo is possible"""
        return self.current is not None and self.current.next is not None
    
    def get_history(self) -> List[Operation]:
        """Get complete operation history"""
        history = []
        
        # Find the beginning
        start = self.current
        while start and start.prev:
            start = start.prev
        
        # Collect all operations
        current = start
        while current:
            history.append(current.val)
            current = current.next
        
        return history
    
    def get_current_position(self) -> int:
        """Get current position in history (0-indexed)"""
        if not self.current:
            return -1
        
        position = 0
        temp = self.current
        
        while temp.prev:
            position += 1
            temp = temp.prev
        
        return position
    
    def _remove_oldest(self) -> None:
        """Remove oldest operation to maintain size limit"""
        # Find the oldest (first) operation
        oldest = self.current
        while oldest.prev:
            oldest = oldest.prev
        
        # Remove oldest
        if oldest.next:
            oldest.next.prev = None
        
        self.size -= 1
    
    def clear_history(self) -> None:
        """Clear all operation history"""
        self.current = None
        self.size = 0
        self.operation_count = 0

class TextEditorWithUndo:
    """
    Simple text editor demonstrating undo/redo functionality
    """
    
    def __init__(self):
        """Initialize text editor"""
        self.content = ""
        self.undo_redo = UndoRedoSystem()
    
    def insert_text(self, position: int, text: str) -> None:
        """Insert text at position"""
        # Create operation
        operation = Operation(
            op_type=OperationType.INSERT,
            data={"position": position, "text": text},
            timestamp=time.time(),
            description=f"Insert '{text}' at position {position}"
        )
        
        # Execute operation
        self.content = self.content[:position] + text + self.content[position:]
        
        # Add to history
        self.undo_redo.execute_operation(operation)
    
    def delete_text(self, start: int, end: int) -> None:
        """Delete text from start to end position"""
        deleted_text = self.content[start:end]
        
        # Create operation
        operation = Operation(
            op_type=OperationType.DELETE,
            data={"start": start, "end": end, "deleted_text": deleted_text},
            timestamp=time.time(),
            description=f"Delete '{deleted_text}' from {start} to {end}"
        )
        
        # Execute operation
        self.content = self.content[:start] + self.content[end:]
        
        # Add to history
        self.undo_redo.execute_operation(operation)
    
    def undo(self) -> bool:
        """Undo last operation"""
        operation = self.undo_redo.undo()
        
        if not operation:
            return False
        
        # Reverse the operation
        if operation.op_type == OperationType.INSERT:
            # Remove inserted text
            pos = operation.data["position"]
            text_len = len(operation.data["text"])
            self.content = self.content[:pos] + self.content[pos + text_len:]
        
        elif operation.op_type == OperationType.DELETE:
            # Restore deleted text
            start = operation.data["start"]
            deleted_text = operation.data["deleted_text"]
            self.content = self.content[:start] + deleted_text + self.content[start:]
        
        return True
    
    def redo(self) -> bool:
        """Redo next operation"""
        operation = self.undo_redo.redo()
        
        if not operation:
            return False
        
        # Re-execute the operation
        if operation.op_type == OperationType.INSERT:
            pos = operation.data["position"]
            text = operation.data["text"]
            self.content = self.content[:pos] + text + self.content[pos:]
        
        elif operation.op_type == OperationType.DELETE:
            start = operation.data["start"]
            end = operation.data["end"]
            self.content = self.content[:start] + self.content[end:]
        
        return True
    
    def get_content(self) -> str:
        """Get current content"""
        return self.content

# ==================== LRU CACHE IMPLEMENTATION ====================

class LRUCache:
    """
    LRU (Least Recently Used) Cache implementation using doubly linked list
    
    This is commonly used in:
    - Operating system page replacement
    - CPU cache management
    - Database buffer management
    - Web browser cache
    """
    
    def __init__(self, capacity: int):
        """
        Initialize LRU cache
        
        Args:
            capacity: Maximum number of items to store
        """
        self.capacity = capacity
        self.cache = {}  # key -> node mapping
        
        # Create dummy head and tail for easier manipulation
        self.head = DoublyListNode()
        self.tail = DoublyListNode()
        self.head.next = self.tail
        self.tail.prev = self.head
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def _add_node(self, node: DoublyListNode) -> None:
        """Add node right after head (most recently used position)"""
        node.prev = self.head
        node.next = self.head.next
        
        self.head.next.prev = node
        self.head.next = node
    
    def _remove_node(self, node: DoublyListNode) -> None:
        """Remove an existing node from the list"""
        prev_node = node.prev
        next_node = node.next
        
        prev_node.next = next_node
        next_node.prev = prev_node
    
    def _move_to_head(self, node: DoublyListNode) -> None:
        """Move node to head (mark as most recently used)"""
        self._remove_node(node)
        self._add_node(node)
    
    def _pop_tail(self) -> DoublyListNode:
        """Pop the last node (least recently used)"""
        last_node = self.tail.prev
        self._remove_node(last_node)
        return last_node
    
    def get(self, key: Any) -> Any:
        """
        Get value by key and mark as recently used
        
        Time Complexity: O(1)
        
        Args:
            key: Key to retrieve
        
        Returns:
            Value or None if not found
        """
        node = self.cache.get(key)
        
        if not node:
            self.misses += 1
            return None
        
        # Move to head (most recently used)
        self._move_to_head(node)
        self.hits += 1
        
        return node.val
    
    def put(self, key: Any, value: Any) -> None:
        """
        Put key-value pair in cache
        
        Time Complexity: O(1)
        
        Args:
            key: Key to store
            value: Value to store
        """
        node = self.cache.get(key)
        
        if not node:
            # Create new node
            new_node = DoublyListNode(key, value)
            
            self.cache[key] = new_node
            self._add_node(new_node)
            
            # Check capacity
            if len(self.cache) > self.capacity:
                # Remove least recently used
                tail = self._pop_tail()
                del self.cache[tail.key]
                self.evictions += 1
        else:
            # Update existing node
            node.val = value
            self._move_to_head(node)
    
    def delete(self, key: Any) -> bool:
        """Delete key from cache"""
        node = self.cache.get(key)
        
        if not node:
            return False
        
        self._remove_node(node)
        del self.cache[key]
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            "capacity": self.capacity,
            "current_size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": hit_rate,
            "miss_rate": 1 - hit_rate
        }
    
    def get_cache_state(self) -> List[Tuple[Any, Any]]:
        """Get current cache state from most to least recently used"""
        result = []
        current = self.head.next
        
        while current != self.tail:
            result.append((current.key, current.val))
            current = current.next
        
        return result
    
    def clear(self) -> None:
        """Clear all items from cache"""
        self.cache.clear()
        self.head.next = self.tail
        self.tail.prev = self.head
        
        # Reset statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0

# ==================== MEMORY MANAGEMENT SIMULATION ====================

class MemoryBlock:
    """Represents a block of memory"""
    def __init__(self, start_address: int, size: int, is_free: bool = True,
                 process_id: Optional[int] = None):
        self.start_address = start_address
        self.size = size
        self.is_free = is_free
        self.process_id = process_id
        self.next = None
    
    def __repr__(self):
        status = "FREE" if self.is_free else f"ALLOCATED(P{self.process_id})"
        return f"Block[{self.start_address}-{self.start_address + self.size - 1}]: {status}"

class MemoryManager:
    """
    Memory management system using linked list for free/allocated blocks
    
    This simulates how operating systems manage memory allocation
    using linked lists to track free and allocated memory blocks.
    """
    
    def __init__(self, total_memory: int):
        """
        Initialize memory manager
        
        Args:
            total_memory: Total memory size to manage
        """
        self.total_memory = total_memory
        
        # Initialize with one large free block
        self.head = MemoryBlock(0, total_memory, is_free=True)
        
        # Statistics
        self.allocated_blocks = 0
        self.free_blocks = 1
        self.total_allocated = 0
        self.fragmentation_count = 0
    
    def allocate(self, size: int, process_id: int) -> Optional[int]:
        """
        Allocate memory block using first-fit algorithm
        
        Args:
            size: Size of memory to allocate
            process_id: ID of process requesting memory
        
        Returns:
            Start address of allocated block or None if allocation failed
        """
        current = self.head
        prev = None
        
        # Find first free block that can accommodate the request
        while current:
            if current.is_free and current.size >= size:
                # Found suitable block
                start_address = current.start_address
                
                if current.size == size:
                    # Exact fit - mark block as allocated
                    current.is_free = False
                    current.process_id = process_id
                    self.free_blocks -= 1
                else:
                    # Split the block
                    new_free_block = MemoryBlock(
                        current.start_address + size,
                        current.size - size,
                        is_free=True
                    )
                    
                    # Update current block
                    current.size = size
                    current.is_free = False
                    current.process_id = process_id
                    
                    # Insert new free block
                    new_free_block.next = current.next
                    current.next = new_free_block
                    
                    # No change in free_blocks count (one split into allocated + free)
                
                self.allocated_blocks += 1
                self.total_allocated += size
                
                return start_address
            
            prev = current
            current = current.next
        
        # No suitable block found
        return None
    
    def deallocate(self, process_id: int) -> bool:
        """
        Deallocate all memory blocks for a process
        
        Args:
            process_id: ID of process to deallocate memory for
        
        Returns:
            True if any blocks were deallocated
        """
        current = self.head
        deallocated = False
        
        while current:
            if not current.is_free and current.process_id == process_id:
                # Mark block as free
                current.is_free = True
                current.process_id = None
                
                self.allocated_blocks -= 1
                self.free_blocks += 1
                self.total_allocated -= current.size
                deallocated = True
            
            current = current.next
        
        # Coalesce adjacent free blocks
        if deallocated:
            self._coalesce_free_blocks()
        
        return deallocated
    
    def _coalesce_free_blocks(self) -> None:
        """Merge adjacent free blocks to reduce fragmentation"""
        current = self.head
        coalesced = 0
        
        while current and current.next:
            if (current.is_free and current.next.is_free and
                current.start_address + current.size == current.next.start_address):
                
                # Merge current with next
                next_block = current.next
                current.size += next_block.size
                current.next = next_block.next
                
                self.free_blocks -= 1
                coalesced += 1
            else:
                current = current.next
        
        if coalesced > 0:
            self.fragmentation_count += coalesced
    
    def get_memory_layout(self) -> List[str]:
        """Get visual representation of memory layout"""
        layout = []
        current = self.head
        
        while current:
            layout.append(str(current))
            current = current.next
        
        return layout
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory management statistics"""
        free_memory = self.total_memory - self.total_allocated
        fragmentation = self._calculate_fragmentation()
        
        return {
            "total_memory": self.total_memory,
            "allocated_memory": self.total_allocated,
            "free_memory": free_memory,
            "allocated_blocks": self.allocated_blocks,
            "free_blocks": self.free_blocks,
            "memory_utilization": self.total_allocated / self.total_memory,
            "fragmentation_ratio": fragmentation,
            "largest_free_block": self._get_largest_free_block()
        }
    
    def _calculate_fragmentation(self) -> float:
        """Calculate memory fragmentation ratio"""
        if self.free_blocks <= 1:
            return 0.0
        
        # Count free block sizes
        free_sizes = []
        current = self.head
        
        while current:
            if current.is_free:
                free_sizes.append(current.size)
            current = current.next
        
        if not free_sizes:
            return 0.0
        
        # Fragmentation = 1 - (largest_free_block / total_free_memory)
        total_free = sum(free_sizes)
        largest_free = max(free_sizes)
        
        return 1 - (largest_free / total_free) if total_free > 0 else 0.0
    
    def _get_largest_free_block(self) -> int:
        """Get size of largest free block"""
        max_size = 0
        current = self.head
        
        while current:
            if current.is_free and current.size > max_size:
                max_size = current.size
            current = current.next
        
        return max_size

# ==================== ADVANCED USE CASES ====================

class ProcessScheduler:
    """
    Process scheduler using linked list for ready queue
    Demonstrates how OS schedulers use linked lists
    """
    
    class Process:
        def __init__(self, pid: int, priority: int, burst_time: int):
            self.pid = pid
            self.priority = priority
            self.burst_time = burst_time
            self.remaining_time = burst_time
            self.arrival_time = time.time()
            self.next = None
        
        def __repr__(self):
            return f"P{self.pid}(pri={self.priority}, burst={self.burst_time})"
    
    def __init__(self, scheduling_algorithm: str = "FCFS"):
        """
        Initialize process scheduler
        
        Args:
            scheduling_algorithm: FCFS, SJF, or Priority
        """
        self.algorithm = scheduling_algorithm
        self.ready_queue = None
        self.running_process = None
        self.completed_processes = []
        self.context_switches = 0
    
    def add_process(self, pid: int, priority: int, burst_time: int) -> None:
        """Add process to ready queue"""
        new_process = self.Process(pid, priority, burst_time)
        
        if self.algorithm == "FCFS":
            self._add_to_end(new_process)
        elif self.algorithm == "SJF":
            self._add_by_burst_time(new_process)
        elif self.algorithm == "Priority":
            self._add_by_priority(new_process)
    
    def _add_to_end(self, process: 'Process') -> None:
        """Add process to end of queue (FCFS)"""
        if not self.ready_queue:
            self.ready_queue = process
        else:
            current = self.ready_queue
            while current.next:
                current = current.next
            current.next = process
    
    def _add_by_burst_time(self, process: 'Process') -> None:
        """Add process sorted by burst time (SJF)"""
        if not self.ready_queue or process.burst_time < self.ready_queue.burst_time:
            process.next = self.ready_queue
            self.ready_queue = process
        else:
            current = self.ready_queue
            while (current.next and 
                   current.next.burst_time <= process.burst_time):
                current = current.next
            
            process.next = current.next
            current.next = process
    
    def _add_by_priority(self, process: 'Process') -> None:
        """Add process sorted by priority (higher number = higher priority)"""
        if not self.ready_queue or process.priority > self.ready_queue.priority:
            process.next = self.ready_queue
            self.ready_queue = process
        else:
            current = self.ready_queue
            while (current.next and 
                   current.next.priority >= process.priority):
                current = current.next
            
            process.next = current.next
            current.next = process
    
    def schedule_next(self) -> Optional['Process']:
        """Get next process to run"""
        if not self.ready_queue:
            return None
        
        # Remove from ready queue
        next_process = self.ready_queue
        self.ready_queue = self.ready_queue.next
        next_process.next = None
        
        if self.running_process:
            self.context_switches += 1
        
        self.running_process = next_process
        return next_process
    
    def get_queue_state(self) -> List[str]:
        """Get current state of ready queue"""
        result = []
        current = self.ready_queue
        
        while current:
            result.append(str(current))
            current = current.next
        
        return result

# ==================== EXAMPLE USAGE AND TESTING ====================

if __name__ == "__main__":
    print("=== Linked List Real-World Use Cases Demo ===\n")
    
    # Example 1: HashMap with Linked List Buckets
    print("1. HashMap with Linked List Collision Resolution:")
    
    hashmap = HashMapWithLL(initial_capacity=8)
    
    # Insert some key-value pairs
    test_data = [
        ("apple", 5), ("banana", 7), ("cherry", 3), ("date", 9),
        ("elderberry", 12), ("fig", 6), ("grape", 8), ("honeydew", 4)
    ]
    
    print("Inserting data into HashMap:")
    for key, value in test_data:
        hashmap.put(key, value)
        print(f"Put {key}: {value}")
    
    # Display bucket contents
    print(f"\nBucket contents:")
    for bucket_info in hashmap.display_buckets():
        print(f"  {bucket_info}")
    
    # Show statistics
    stats = hashmap.get_statistics()
    print(f"\nHashMap Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    # Test retrieval and deletion
    print(f"\nTesting operations:")
    print(f"Get 'apple': {hashmap.get('apple')}")
    print(f"Get 'nonexistent': {hashmap.get('nonexistent')}")
    print(f"Delete 'banana': {hashmap.delete('banana')}")
    print(f"Get 'banana' after deletion: {hashmap.get('banana')}")
    print()
    
    # Example 2: Undo/Redo System
    print("2. Text Editor with Undo/Redo:")
    
    editor = TextEditorWithUndo()
    
    print("Text editing operations:")
    editor.insert_text(0, "Hello")
    print(f"After insert 'Hello': '{editor.get_content()}'")
    
    editor.insert_text(5, " World")
    print(f"After insert ' World': '{editor.get_content()}'")
    
    editor.insert_text(11, "!")
    print(f"After insert '!': '{editor.get_content()}'")
    
    editor.delete_text(5, 11)  # Delete " World"
    print(f"After delete ' World': '{editor.get_content()}'")
    
    # Test undo operations
    print(f"\nUndo operations:")
    while editor.undo_redo.can_undo():
        operation = editor.undo()
        print(f"Undo: '{editor.get_content()}'")
    
    # Test redo operations
    print(f"\nRedo operations:")
    while editor.undo_redo.can_redo():
        operation = editor.redo()
        print(f"Redo: '{editor.get_content()}'")
    
    # Show operation history
    print(f"\nOperation history:")
    for i, op in enumerate(editor.undo_redo.get_history()):
        current_marker = " <-- CURRENT" if i == editor.undo_redo.get_current_position() else ""
        print(f"  {i}: {op.description}{current_marker}")
    print()
    
    # Example 3: LRU Cache
    print("3. LRU Cache Implementation:")
    
    lru_cache = LRUCache(capacity=3)
    
    # Test cache operations
    cache_ops = [
        ("put", 1, "One"),
        ("put", 2, "Two"), 
        ("put", 3, "Three"),
        ("get", 1, None),
        ("put", 4, "Four"),  # Should evict key 2
        ("get", 2, None),    # Should miss
        ("get", 3, None),
        ("get", 4, None),
        ("put", 5, "Five"),  # Should evict key 1
    ]
    
    print("Cache operations:")
    for op, key, value in cache_ops:
        if op == "put":
            lru_cache.put(key, value)
            print(f"Put ({key}, {value}) -> Cache state: {lru_cache.get_cache_state()}")
        else:
            result = lru_cache.get(key)
            print(f"Get {key} -> {result}, Cache state: {lru_cache.get_cache_state()}")
    
    # Show cache statistics
    cache_stats = lru_cache.get_statistics()
    print(f"\nCache Statistics:")
    for key, value in cache_stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    print()
    
    # Example 4: Memory Management
    print("4. Operating System Memory Management:")
    
    memory_manager = MemoryManager(total_memory=1000)
    
    print("Memory allocation simulation:")
    
    # Allocate memory for different processes
    allocations = [
        (100, 1, "Process 1"),
        (200, 2, "Process 2"),
        (150, 3, "Process 3"),
        (300, 4, "Process 4"),
        (100, 5, "Process 5"),  # Should fail - not enough memory
    ]
    
    for size, pid, name in allocations:
        start_addr = memory_manager.allocate(size, pid)
        if start_addr is not None:
            print(f"Allocated {size} bytes for {name} at address {start_addr}")
        else:
            print(f"Failed to allocate {size} bytes for {name}")
    
    print(f"\nMemory layout after allocations:")
    for block in memory_manager.get_memory_layout():
        print(f"  {block}")
    
    # Deallocate some processes
    print(f"\nDeallocating Process 2 and Process 4:")
    memory_manager.deallocate(2)
    memory_manager.deallocate(4)
    
    print(f"Memory layout after deallocation:")
    for block in memory_manager.get_memory_layout():
        print(f"  {block}")
    
    # Show memory statistics
    mem_stats = memory_manager.get_statistics()
    print(f"\nMemory Statistics:")
    for key, value in mem_stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    print()
    
    # Example 5: Process Scheduler
    print("5. Process Scheduler with Linked List:")
    
    # Test different scheduling algorithms
    schedulers = ["FCFS", "SJF", "Priority"]
    
    for algorithm in schedulers:
        print(f"\n{algorithm} Scheduling:")
        scheduler = ProcessScheduler(algorithm)
        
        # Add processes
        processes = [
            (1, 3, 10),  # (pid, priority, burst_time)
            (2, 1, 5),
            (3, 4, 8),
            (4, 2, 3),
            (5, 5, 12)
        ]
        
        for pid, priority, burst_time in processes:
            scheduler.add_process(pid, priority, burst_time)
        
        print(f"Ready queue: {scheduler.get_queue_state()}")
        
        # Schedule processes
        execution_order = []
        while True:
            next_process = scheduler.schedule_next()
            if not next_process:
                break
            execution_order.append(str(next_process))
        
        print(f"Execution order: {' -> '.join(execution_order)}")
        print(f"Context switches: {scheduler.context_switches}")
    
    # Example 6: Performance Analysis
    print(f"\n6. Performance Analysis:")
    
    # Compare HashMap performance with different load factors
    print("HashMap performance with different load factors:")
    
    load_factors = [0.5, 0.75, 1.0, 1.5]
    for lf in load_factors:
        test_map = HashMapWithLL(initial_capacity=16, load_factor=lf)
        
        # Insert many items
        for i in range(100):
            test_map.put(f"key{i}", f"value{i}")
        
        stats = test_map.get_statistics()
        print(f"Load factor {lf}: avg_chain_length={stats['avg_chain_length']:.2f}, "
              f"max_chain_length={stats['max_chain_length']}, "
              f"collisions={stats['collision_count']}")
    
    # LRU Cache hit rate analysis
    print(f"\nLRU Cache hit rate analysis:")
    
    cache_sizes = [5, 10, 20, 50]
    for size in cache_sizes:
        test_cache = LRUCache(size)
        
        # Simulate access pattern
        import random
        for _ in range(200):
            key = random.randint(1, 30)  # Keys 1-30, cache sizes vary
            test_cache.get(key)
            if random.random() < 0.1:  # 10% chance to put new data
                test_cache.put(key, f"value{key}")
        
        stats = test_cache.get_statistics()
        print(f"Cache size {size}: hit_rate={stats['hit_rate']:.3f}, "
              f"evictions={stats['evictions']}")
    
    # Real-world application insights
    print(f"\nReal-World Application Insights:")
    print(f"1. HashMap Buckets: Used in Python dict, Java HashMap, database indexing")
    print(f"2. Undo/Redo: Text editors (VSCode, Word), IDEs, graphics software")
    print(f"3. LRU Cache: CPU caches, OS page replacement, web browser cache")
    print(f"4. Memory Management: OS malloc/free, garbage collectors, embedded systems")
    print(f"5. Process Scheduling: Linux CFS, Windows scheduler, real-time systems")
    
    print("\n=== Demo Complete ===") 