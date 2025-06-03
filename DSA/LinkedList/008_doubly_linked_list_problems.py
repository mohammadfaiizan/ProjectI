"""
Doubly Linked List Problems - Advanced DLL Applications and Designs
This module implements complex applications and design problems using doubly linked lists.
"""

from typing import Optional, Dict, List, Any
from collections import OrderedDict
import random

class DoublyListNode:
    """Node class for doubly linked list"""
    def __init__(self, val: int = 0, prev: Optional['DoublyListNode'] = None, 
                 next: Optional['DoublyListNode'] = None):
        self.val = val
        self.prev = prev
        self.next = next
    
    def __repr__(self):
        return f"DoublyListNode({self.val})"

class KeyValueNode:
    """Node class for key-value pairs"""
    def __init__(self, key: int = 0, val: int = 0, prev: Optional['KeyValueNode'] = None, 
                 next: Optional['KeyValueNode'] = None):
        self.key = key
        self.val = val
        self.prev = prev
        self.next = next
    
    def __repr__(self):
        return f"KeyValueNode({self.key}: {self.val})"

class LRUCache:
    """
    LRU (Least Recently Used) Cache implementation using doubly linked list + hashmap
    
    Time Complexity: O(1) for all operations
    Space Complexity: O(capacity)
    """
    
    def __init__(self, capacity: int):
        """
        Initialize LRU cache with given capacity
        
        Args:
            capacity: Maximum number of key-value pairs to store
        """
        self.capacity = capacity
        self.cache = {}  # key -> node mapping
        
        # Create dummy head and tail for easier manipulation
        self.head = KeyValueNode()
        self.tail = KeyValueNode()
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def _add_node(self, node: KeyValueNode) -> None:
        """Add node right after head (most recently used position)"""
        node.prev = self.head
        node.next = self.head.next
        
        self.head.next.prev = node
        self.head.next = node
    
    def _remove_node(self, node: KeyValueNode) -> None:
        """Remove an existing node from the list"""
        prev_node = node.prev
        next_node = node.next
        
        prev_node.next = next_node
        next_node.prev = prev_node
    
    def _move_to_head(self, node: KeyValueNode) -> None:
        """Move node to head (mark as most recently used)"""
        self._remove_node(node)
        self._add_node(node)
    
    def _pop_tail(self) -> KeyValueNode:
        """Pop the last node (least recently used)"""
        last_node = self.tail.prev
        self._remove_node(last_node)
        return last_node
    
    def get(self, key: int) -> int:
        """
        Get value by key and mark as recently used
        
        Args:
            key: Key to get value for
        
        Returns:
            Value or -1 if not found
        """
        node = self.cache.get(key)
        
        if not node:
            return -1
        
        # Move to head (most recently used)
        self._move_to_head(node)
        
        return node.val
    
    def put(self, key: int, value: int) -> None:
        """
        Put key-value pair in cache
        
        Args:
            key: Key to store
            value: Value to store
        """
        node = self.cache.get(key)
        
        if not node:
            # Create new node
            new_node = KeyValueNode(key, value)
            
            self.cache[key] = new_node
            self._add_node(new_node)
            
            # Check capacity
            if len(self.cache) > self.capacity:
                # Remove least recently used
                tail = self._pop_tail()
                del self.cache[tail.key]
        else:
            # Update existing node
            node.val = value
            self._move_to_head(node)
    
    def display_cache_state(self) -> List[tuple]:
        """Display current cache state from most to least recently used"""
        result = []
        current = self.head.next
        
        while current != self.tail:
            result.append((current.key, current.val))
            current = current.next
        
        return result
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get comprehensive cache information"""
        return {
            "capacity": self.capacity,
            "current_size": len(self.cache),
            "cache_state": self.display_cache_state(),
            "available_space": self.capacity - len(self.cache)
        }

class LFUCache:
    """
    LFU (Least Frequently Used) Cache implementation
    """
    
    def __init__(self, capacity: int):
        """Initialize LFU cache"""
        self.capacity = capacity
        self.cache = {}  # key -> node
        self.frequencies = {}  # frequency -> doubly linked list
        self.key_frequencies = {}  # key -> frequency
        self.min_frequency = 0
    
    def _add_to_frequency_list(self, key: int, frequency: int) -> None:
        """Add key to frequency list"""
        if frequency not in self.frequencies:
            self.frequencies[frequency] = {
                'head': KeyValueNode(),
                'tail': KeyValueNode()
            }
            self.frequencies[frequency]['head'].next = self.frequencies[frequency]['tail']
            self.frequencies[frequency]['tail'].prev = self.frequencies[frequency]['head']
        
        node = self.cache[key]
        # Add to head of frequency list
        head = self.frequencies[frequency]['head']
        node.next = head.next
        node.prev = head
        head.next.prev = node
        head.next = node
    
    def _remove_from_frequency_list(self, key: int, frequency: int) -> None:
        """Remove key from frequency list"""
        if frequency in self.frequencies:
            node = self.cache[key]
            node.prev.next = node.next
            node.next.prev = node.prev
    
    def get(self, key: int) -> int:
        """Get value and update frequency"""
        if key not in self.cache:
            return -1
        
        # Update frequency
        old_freq = self.key_frequencies[key]
        new_freq = old_freq + 1
        
        self._remove_from_frequency_list(key, old_freq)
        self.key_frequencies[key] = new_freq
        self._add_to_frequency_list(key, new_freq)
        
        # Update min frequency if needed
        if old_freq == self.min_frequency and not self._has_keys_with_frequency(old_freq):
            self.min_frequency += 1
        
        return self.cache[key].val
    
    def put(self, key: int, value: int) -> None:
        """Put key-value pair"""
        if self.capacity <= 0:
            return
        
        if key in self.cache:
            # Update existing
            self.cache[key].val = value
            self.get(key)  # This will update frequency
        else:
            # Add new
            if len(self.cache) >= self.capacity:
                # Remove LFU
                self._evict_lfu()
            
            # Add new key
            new_node = KeyValueNode(key, value)
            self.cache[key] = new_node
            self.key_frequencies[key] = 1
            self._add_to_frequency_list(key, 1)
            self.min_frequency = 1
    
    def _has_keys_with_frequency(self, frequency: int) -> bool:
        """Check if there are keys with given frequency"""
        if frequency not in self.frequencies:
            return False
        
        freq_list = self.frequencies[frequency]
        return freq_list['head'].next != freq_list['tail']
    
    def _evict_lfu(self) -> None:
        """Evict least frequently used key"""
        if self.min_frequency in self.frequencies:
            freq_list = self.frequencies[self.min_frequency]
            # Remove from tail (least recently used among same frequency)
            tail_node = freq_list['tail'].prev
            
            if tail_node != freq_list['head']:
                key_to_remove = tail_node.key
                
                # Remove from all structures
                self._remove_from_frequency_list(key_to_remove, self.min_frequency)
                del self.cache[key_to_remove]
                del self.key_frequencies[key_to_remove]

class BrowserHistory:
    """
    Browser History implementation using doubly linked list
    Supports forward/backward navigation
    """
    
    def __init__(self, homepage: str):
        """
        Initialize browser history with homepage
        
        Args:
            homepage: Initial homepage URL
        """
        self.current = DoublyListNode(homepage)
        self.current.val = homepage  # Store URL as value
    
    def visit(self, url: str) -> None:
        """
        Visit new URL (clears forward history)
        
        Args:
            url: URL to visit
        """
        # Create new node
        new_node = DoublyListNode(url)
        new_node.val = url
        
        # Link with current
        new_node.prev = self.current
        self.current.next = new_node
        
        # Move to new page
        self.current = new_node
    
    def back(self, steps: int) -> str:
        """
        Go back in history by given steps
        
        Args:
            steps: Number of steps to go back
        
        Returns:
            Current URL after going back
        """
        for _ in range(steps):
            if self.current.prev:
                self.current = self.current.prev
            else:
                break
        
        return self.current.val
    
    def forward(self, steps: int) -> str:
        """
        Go forward in history by given steps
        
        Args:
            steps: Number of steps to go forward
        
        Returns:
            Current URL after going forward
        """
        for _ in range(steps):
            if self.current.next:
                self.current = self.current.next
            else:
                break
        
        return self.current.val
    
    def get_current_url(self) -> str:
        """Get current URL"""
        return self.current.val
    
    def get_history_path(self) -> List[str]:
        """Get complete history path"""
        # Go to beginning
        start = self.current
        while start.prev:
            start = start.prev
        
        # Collect all URLs
        history = []
        current = start
        while current:
            history.append(current.val)
            current = current.next
        
        return history
    
    def can_go_back(self) -> bool:
        """Check if can go back"""
        return self.current.prev is not None
    
    def can_go_forward(self) -> bool:
        """Check if can go forward"""
        return self.current.next is not None

class MiddleOperationList:
    """
    Doubly linked list with O(1) operations on middle element
    """
    
    def __init__(self):
        """Initialize list with O(1) middle operations"""
        self.head = DoublyListNode()
        self.tail = DoublyListNode()
        self.head.next = self.tail
        self.tail.prev = self.head
        
        self.middle = None
        self.size = 0
        self.is_even_size = True
    
    def insert_middle(self, val: int) -> None:
        """
        Insert value at middle position
        
        Time Complexity: O(1)
        
        Args:
            val: Value to insert
        """
        new_node = DoublyListNode(val)
        
        if self.size == 0:
            # First element
            new_node.next = self.tail
            new_node.prev = self.head
            self.head.next = new_node
            self.tail.prev = new_node
            self.middle = new_node
        else:
            if self.is_even_size:
                # Insert before middle
                new_node.next = self.middle
                new_node.prev = self.middle.prev
                self.middle.prev.next = new_node
                self.middle.prev = new_node
                # Middle doesn't change for even size
            else:
                # Insert after middle
                new_node.next = self.middle.next
                new_node.prev = self.middle
                self.middle.next.prev = new_node
                self.middle.next = new_node
                self.middle = new_node
        
        self.size += 1
        self.is_even_size = not self.is_even_size
    
    def delete_middle(self) -> Optional[int]:
        """
        Delete middle element
        
        Time Complexity: O(1)
        
        Returns:
            Value of deleted middle element or None if empty
        """
        if self.size == 0:
            return None
        
        deleted_val = self.middle.val
        
        if self.size == 1:
            # Last element
            self.head.next = self.tail
            self.tail.prev = self.head
            self.middle = None
        else:
            if self.is_even_size:
                # Move middle to next
                new_middle = self.middle.next
                self.middle.prev.next = self.middle.next
                self.middle.next.prev = self.middle.prev
                self.middle = new_middle
            else:
                # Move middle to previous
                new_middle = self.middle.prev
                self.middle.prev.next = self.middle.next
                self.middle.next.prev = self.middle.prev
                self.middle = new_middle
        
        self.size -= 1
        self.is_even_size = not self.is_even_size
        
        return deleted_val
    
    def get_middle(self) -> Optional[int]:
        """Get middle element value"""
        return self.middle.val if self.middle else None
    
    def insert_at_beginning(self, val: int) -> None:
        """Insert at beginning and update middle if needed"""
        new_node = DoublyListNode(val)
        
        new_node.next = self.head.next
        new_node.prev = self.head
        self.head.next.prev = new_node
        self.head.next = new_node
        
        if self.size == 0:
            self.middle = new_node
        elif not self.is_even_size:
            # Move middle towards tail
            self.middle = self.middle.next
        
        self.size += 1
        self.is_even_size = not self.is_even_size
    
    def insert_at_end(self, val: int) -> None:
        """Insert at end and update middle if needed"""
        new_node = DoublyListNode(val)
        
        new_node.next = self.tail
        new_node.prev = self.tail.prev
        self.tail.prev.next = new_node
        self.tail.prev = new_node
        
        if self.size == 0:
            self.middle = new_node
        elif self.is_even_size:
            # Move middle towards tail
            self.middle = self.middle.next
        
        self.size += 1
        self.is_even_size = not self.is_even_size
    
    def display(self) -> List[int]:
        """Display all elements"""
        result = []
        current = self.head.next
        
        while current != self.tail:
            result.append(current.val)
            current = current.next
        
        return result
    
    def get_size(self) -> int:
        """Get current size"""
        return self.size

class MultiLevelUndoRedo:
    """
    Multi-level undo/redo system using doubly linked list
    """
    
    def __init__(self, max_history: int = 100):
        """
        Initialize undo/redo system
        
        Args:
            max_history: Maximum number of operations to keep in history
        """
        self.max_history = max_history
        self.current = None
        self.size = 0
    
    def execute_operation(self, operation: str, data: Any) -> None:
        """
        Execute new operation and add to history
        
        Args:
            operation: Operation name
            data: Operation data
        """
        new_node = DoublyListNode()
        new_node.val = {"operation": operation, "data": data, "timestamp": random.randint(1000, 9999)}
        
        if not self.current:
            # First operation
            self.current = new_node
        else:
            # Clear forward history
            self.current.next = None
            
            # Add new operation
            new_node.prev = self.current
            self.current.next = new_node
            self.current = new_node
        
        self.size += 1
        
        # Maintain max history
        if self.size > self.max_history:
            self._remove_oldest()
    
    def undo(self) -> Optional[Dict[str, Any]]:
        """
        Undo last operation
        
        Returns:
            Operation data that was undone or None
        """
        if not self.current:
            return None
        
        undone_operation = self.current.val
        
        if self.current.prev:
            self.current = self.current.prev
        else:
            # No more operations to undo
            pass
        
        return undone_operation
    
    def redo(self) -> Optional[Dict[str, Any]]:
        """
        Redo next operation
        
        Returns:
            Operation data that was redone or None
        """
        if not self.current or not self.current.next:
            return None
        
        self.current = self.current.next
        return self.current.val
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get complete operation history"""
        # Find beginning
        start = self.current
        while start and start.prev:
            start = start.prev
        
        # Collect all operations
        history = []
        current = start
        while current:
            history.append(current.val)
            current = current.next
        
        return history
    
    def can_undo(self) -> bool:
        """Check if undo is possible"""
        return self.current is not None
    
    def can_redo(self) -> bool:
        """Check if redo is possible"""
        return self.current is not None and self.current.next is not None
    
    def _remove_oldest(self) -> None:
        """Remove oldest operation to maintain max history"""
        # Find the head
        start = self.current
        while start.prev:
            start = start.prev
        
        # Remove head
        if start.next:
            start.next.prev = None
        
        self.size -= 1

class SnapshotDataStructure:
    """
    Data structure that supports snapshots using doubly linked list
    """
    
    def __init__(self):
        """Initialize snapshot data structure"""
        self.snapshots = DoublyListNode()  # Dummy head
        self.current_snapshot = None
        self.snapshot_count = 0
        self.data = {}
    
    def set(self, key: str, value: Any) -> None:
        """Set key-value pair in current snapshot"""
        self.data[key] = value
    
    def get(self, key: str) -> Any:
        """Get value for key from current snapshot"""
        return self.data.get(key)
    
    def take_snapshot(self) -> int:
        """
        Take snapshot of current state
        
        Returns:
            Snapshot ID
        """
        # Create new snapshot node
        snapshot_node = DoublyListNode()
        snapshot_node.val = {
            "id": self.snapshot_count,
            "data": self.data.copy(),
            "timestamp": random.randint(10000, 99999)
        }
        
        # Link to list
        if not self.current_snapshot:
            # First snapshot
            self.snapshots.next = snapshot_node
            snapshot_node.prev = self.snapshots
        else:
            # Add after current
            snapshot_node.prev = self.current_snapshot
            snapshot_node.next = self.current_snapshot.next
            
            if self.current_snapshot.next:
                self.current_snapshot.next.prev = snapshot_node
            
            self.current_snapshot.next = snapshot_node
        
        self.current_snapshot = snapshot_node
        snapshot_id = self.snapshot_count
        self.snapshot_count += 1
        
        return snapshot_id
    
    def restore_snapshot(self, snapshot_id: int) -> bool:
        """
        Restore to specific snapshot
        
        Args:
            snapshot_id: ID of snapshot to restore
        
        Returns:
            True if successful, False if snapshot not found
        """
        # Find snapshot
        current = self.snapshots.next
        
        while current:
            if current.val["id"] == snapshot_id:
                # Restore data
                self.data = current.val["data"].copy()
                self.current_snapshot = current
                return True
            current = current.next
        
        return False
    
    def list_snapshots(self) -> List[Dict[str, Any]]:
        """List all available snapshots"""
        snapshots = []
        current = self.snapshots.next
        
        while current:
            snapshots.append({
                "id": current.val["id"],
                "timestamp": current.val["timestamp"],
                "data_size": len(current.val["data"])
            })
            current = current.next
        
        return snapshots
    
    def delete_snapshot(self, snapshot_id: int) -> bool:
        """Delete specific snapshot"""
        current = self.snapshots.next
        
        while current:
            if current.val["id"] == snapshot_id:
                # Remove from list
                if current.prev:
                    current.prev.next = current.next
                if current.next:
                    current.next.prev = current.prev
                
                # Update current if deleted
                if self.current_snapshot == current:
                    self.current_snapshot = current.prev if current.prev != self.snapshots else current.next
                
                return True
            current = current.next
        
        return False

# ==================== EXAMPLE USAGE AND TESTING ====================

if __name__ == "__main__":
    print("=== Doubly Linked List Problems Demo ===\n")
    
    # Example 1: LRU Cache
    print("1. LRU Cache Implementation:")
    
    lru = LRUCache(3)
    
    print("Cache operations:")
    operations = [
        ("put", 1, 10),
        ("put", 2, 20),
        ("put", 3, 30),
        ("get", 1, None),
        ("put", 4, 40),  # Should evict key 2
        ("get", 2, None),  # Should return -1
        ("get", 3, None),
        ("get", 4, None),
        ("put", 5, 50),  # Should evict key 1
    ]
    
    for i, op in enumerate(operations):
        if op[0] == "put":
            lru.put(op[1], op[2])
            print(f"  Put ({op[1]}, {op[2]}) -> Cache: {lru.display_cache_state()}")
        else:
            result = lru.get(op[1])
            print(f"  Get {op[1]} -> {result}, Cache: {lru.display_cache_state()}")
    
    print(f"Final cache info: {lru.get_cache_info()}")
    print()
    
    # Example 2: Browser History
    print("2. Browser History:")
    
    browser = BrowserHistory("google.com")
    print(f"Initial: {browser.get_current_url()}")
    
    history_operations = [
        ("visit", "youtube.com"),
        ("visit", "facebook.com"),
        ("visit", "twitter.com"),
        ("back", 1),
        ("back", 1),
        ("forward", 1),
        ("visit", "instagram.com"),  # Should clear forward history
        ("forward", 1),  # Should stay at instagram
        ("back", 2),
    ]
    
    for op, param in history_operations:
        if op == "visit":
            browser.visit(param)
            print(f"Visit {param} -> Current: {browser.get_current_url()}")
        elif op == "back":
            result = browser.back(param)
            print(f"Back {param} -> Current: {result}")
        elif op == "forward":
            result = browser.forward(param)
            print(f"Forward {param} -> Current: {result}")
    
    print(f"Complete history: {browser.get_history_path()}")
    print(f"Can go back: {browser.can_go_back()}, Can go forward: {browser.can_go_forward()}")
    print()
    
    # Example 3: Middle Operations List
    print("3. O(1) Middle Operations:")
    
    middle_list = MiddleOperationList()
    
    print("Insert at middle operations:")
    for val in [1, 2, 3, 4, 5]:
        middle_list.insert_middle(val)
        print(f"Insert {val} -> List: {middle_list.display()}, Middle: {middle_list.get_middle()}")
    
    print("\nInsert at beginning:")
    middle_list.insert_at_beginning(0)
    print(f"Insert 0 at beginning -> List: {middle_list.display()}, Middle: {middle_list.get_middle()}")
    
    print("\nInsert at end:")
    middle_list.insert_at_end(6)
    print(f"Insert 6 at end -> List: {middle_list.display()}, Middle: {middle_list.get_middle()}")
    
    print("\nDelete middle operations:")
    for _ in range(3):
        deleted = middle_list.delete_middle()
        print(f"Delete middle -> Deleted: {deleted}, List: {middle_list.display()}, Middle: {middle_list.get_middle()}")
    print()
    
    # Example 4: Multi-level Undo/Redo
    print("4. Multi-level Undo/Redo:")
    
    undo_redo = MultiLevelUndoRedo(5)
    
    print("Execute operations:")
    operations = [
        ("create_file", "document.txt"),
        ("write_text", "Hello World"),
        ("format_bold", "Hello"),
        ("add_image", "image.png"),
        ("save_file", "document.txt"),
    ]
    
    for op, data in operations:
        undo_redo.execute_operation(op, data)
        print(f"Execute {op}({data})")
    
    print(f"\nComplete history: {len(undo_redo.get_history())} operations")
    
    print("\nUndo operations:")
    for _ in range(3):
        undone = undo_redo.undo()
        if undone:
            print(f"Undo: {undone['operation']}({undone['data']})")
        print(f"  Can undo: {undo_redo.can_undo()}, Can redo: {undo_redo.can_redo()}")
    
    print("\nRedo operations:")
    for _ in range(2):
        redone = undo_redo.redo()
        if redone:
            print(f"Redo: {redone['operation']}({redone['data']})")
        print(f"  Can undo: {undo_redo.can_undo()}, Can redo: {undo_redo.can_redo()}")
    print()
    
    # Example 5: Snapshot Data Structure
    print("5. Snapshot Data Structure:")
    
    snapshot_ds = SnapshotDataStructure()
    
    print("Initial data operations:")
    snapshot_ds.set("name", "John")
    snapshot_ds.set("age", 25)
    snapshot_ds.set("city", "New York")
    
    snapshot1 = snapshot_ds.take_snapshot()
    print(f"Snapshot {snapshot1} taken")
    
    # Modify data
    snapshot_ds.set("age", 26)
    snapshot_ds.set("job", "Engineer")
    
    snapshot2 = snapshot_ds.take_snapshot()
    print(f"Snapshot {snapshot2} taken")
    
    # More modifications
    snapshot_ds.set("city", "San Francisco")
    snapshot_ds.set("salary", 100000)
    
    snapshot3 = snapshot_ds.take_snapshot()
    print(f"Snapshot {snapshot3} taken")
    
    print(f"\nCurrent data: name={snapshot_ds.get('name')}, age={snapshot_ds.get('age')}, city={snapshot_ds.get('city')}")
    
    print("\nAvailable snapshots:")
    for snapshot in snapshot_ds.list_snapshots():
        print(f"  ID: {snapshot['id']}, Timestamp: {snapshot['timestamp']}, Data size: {snapshot['data_size']}")
    
    # Restore to snapshot 1
    print(f"\nRestore to snapshot {snapshot1}:")
    snapshot_ds.restore_snapshot(snapshot1)
    print(f"Data after restore: name={snapshot_ds.get('name')}, age={snapshot_ds.get('age')}, city={snapshot_ds.get('city')}")
    print(f"Job: {snapshot_ds.get('job')}, Salary: {snapshot_ds.get('salary')}")
    print()
    
    # Example 6: Performance Analysis
    print("6. Performance Analysis:")
    
    # Large LRU cache
    large_lru = LRUCache(1000)
    
    print("Testing large LRU cache (1000 capacity):")
    
    # Fill cache
    for i in range(1000):
        large_lru.put(i, i * 10)
    
    print(f"Cache filled with 1000 items")
    
    # Access pattern (some gets, some puts)
    access_count = 0
    eviction_count = 0
    
    for i in range(500):
        # Get existing keys
        large_lru.get(i)
        access_count += 1
        
        # Add new keys (should cause evictions)
        large_lru.put(1000 + i, (1000 + i) * 10)
        eviction_count += 1
    
    cache_info = large_lru.get_cache_info()
    print(f"After {access_count} gets and {eviction_count} puts:")
    print(f"  Cache size: {cache_info['current_size']}/{cache_info['capacity']}")
    print(f"  Available space: {cache_info['available_space']}")
    
    # Test middle operations on large list
    large_middle_list = MiddleOperationList()
    
    print(f"\nTesting O(1) middle operations on large list:")
    
    # Insert 1000 elements at middle
    for i in range(100):  # Reduced for demo
        large_middle_list.insert_middle(i)
    
    print(f"Inserted 100 elements")
    print(f"List size: {large_middle_list.get_size()}")
    print(f"Middle element: {large_middle_list.get_middle()}")
    print(f"List sample: {large_middle_list.display()[:10]}...{large_middle_list.display()[-10:]}")
    
    # Delete from middle
    deleted_count = 0
    while large_middle_list.get_size() > 50:
        large_middle_list.delete_middle()
        deleted_count += 1
    
    print(f"Deleted {deleted_count} elements from middle")
    print(f"Final size: {large_middle_list.get_size()}")
    print(f"Final middle: {large_middle_list.get_middle()}")
    
    print("\n=== Demo Complete ===") 