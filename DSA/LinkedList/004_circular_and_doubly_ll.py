"""
Circular & Doubly Linked Lists - Advanced Linked List Variants
This module implements comprehensive circular and doubly linked list operations with advanced algorithms.
"""

from typing import Optional, List, Tuple, Any
from collections import deque

class DoublyListNode:
    """Node class for doubly linked list"""
    def __init__(self, val: int = 0, prev: Optional['DoublyListNode'] = None, 
                 next: Optional['DoublyListNode'] = None):
        self.val = val
        self.prev = prev
        self.next = next
    
    def __repr__(self):
        return f"DoublyListNode({self.val})"

class ListNode:
    """Basic node class for singly linked list"""
    def __init__(self, val: int = 0, next: Optional['ListNode'] = None):
        self.val = val
        self.next = next
    
    def __repr__(self):
        return f"ListNode({self.val})"

class CircularLinkedList:
    """
    Comprehensive Circular Singly Linked List Implementation
    """
    
    def __init__(self):
        """Initialize empty circular linked list"""
        self.head: Optional[ListNode] = None
        self.tail: Optional[ListNode] = None
        self.size = 0
    
    def insert_at_beginning(self, val: int) -> None:
        """
        Insert node at the beginning
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        
        Args:
            val: Value to insert
        """
        new_node = ListNode(val)
        
        if not self.head:
            self.head = self.tail = new_node
            new_node.next = new_node
        else:
            new_node.next = self.head
            self.tail.next = new_node
            self.head = new_node
        
        self.size += 1
    
    def insert_at_end(self, val: int) -> None:
        """
        Insert node at the end
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        
        Args:
            val: Value to insert
        """
        new_node = ListNode(val)
        
        if not self.head:
            self.head = self.tail = new_node
            new_node.next = new_node
        else:
            new_node.next = self.head
            self.tail.next = new_node
            self.tail = new_node
        
        self.size += 1
    
    def insert_at_position(self, pos: int, val: int) -> bool:
        """
        Insert node at specific position (0-indexed)
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Args:
            pos: Position to insert at
            val: Value to insert
        
        Returns:
            bool: True if insertion successful
        """
        if pos < 0 or pos > self.size:
            return False
        
        if pos == 0:
            self.insert_at_beginning(val)
            return True
        
        if pos == self.size:
            self.insert_at_end(val)
            return True
        
        new_node = ListNode(val)
        current = self.head
        
        for _ in range(pos - 1):
            current = current.next
        
        new_node.next = current.next
        current.next = new_node
        self.size += 1
        return True
    
    def delete_by_value(self, val: int) -> bool:
        """
        Delete first occurrence of value
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Args:
            val: Value to delete
        
        Returns:
            bool: True if deletion successful
        """
        if not self.head:
            return False
        
        # Single node case
        if self.size == 1 and self.head.val == val:
            self.head = self.tail = None
            self.size -= 1
            return True
        
        # Delete head
        if self.head.val == val:
            self.tail.next = self.head.next
            self.head = self.head.next
            self.size -= 1
            return True
        
        # Delete non-head node
        current = self.head
        while current.next != self.head:
            if current.next.val == val:
                if current.next == self.tail:
                    self.tail = current
                current.next = current.next.next
                self.size -= 1
                return True
            current = current.next
        
        return False
    
    def delete_by_position(self, pos: int) -> bool:
        """
        Delete node at specific position
        
        Args:
            pos: Position to delete (0-indexed)
        
        Returns:
            bool: True if deletion successful
        """
        if pos < 0 or pos >= self.size or not self.head:
            return False
        
        if pos == 0:
            return self.delete_by_value(self.head.val)
        
        current = self.head
        for _ in range(pos - 1):
            current = current.next
        
        if current.next == self.tail:
            self.tail = current
        
        current.next = current.next.next
        self.size -= 1
        return True
    
    def search(self, val: int) -> int:
        """
        Search for value and return position
        
        Args:
            val: Value to search for
        
        Returns:
            int: Position of value (-1 if not found)
        """
        if not self.head:
            return -1
        
        current = self.head
        position = 0
        
        while True:
            if current.val == val:
                return position
            current = current.next
            position += 1
            if current == self.head:
                break
        
        return -1
    
    def display(self) -> List[int]:
        """Display all values in the list"""
        if not self.head:
            return []
        
        values = []
        current = self.head
        
        while True:
            values.append(current.val)
            current = current.next
            if current == self.head:
                break
        
        return values
    
    def length(self) -> int:
        """Get length of list"""
        return self.size
    
    def is_empty(self) -> bool:
        """Check if list is empty"""
        return self.head is None
    
    def clear(self) -> None:
        """Clear all nodes"""
        self.head = self.tail = None
        self.size = 0
    
    def split_list(self) -> Tuple[Optional['CircularLinkedList'], Optional['CircularLinkedList']]:
        """
        Split circular list into two halves
        
        Returns:
            Tuple of two circular lists
        """
        if self.size < 2:
            return self, None
        
        # Find middle
        slow = fast = self.head
        slow_prev = None
        
        while fast.next != self.head and fast.next.next != self.head:
            slow_prev = slow
            slow = slow.next
            fast = fast.next.next
        
        # If even number of nodes
        if fast.next.next == self.head:
            slow_prev = slow
            slow = slow.next
        
        # Create first half
        first_half = CircularLinkedList()
        first_half.head = self.head
        first_half.tail = slow_prev
        slow_prev.next = self.head
        first_half.size = (self.size + 1) // 2
        
        # Create second half
        second_half = CircularLinkedList()
        second_half.head = slow
        second_half.tail = self.tail
        self.tail.next = slow
        second_half.size = self.size // 2
        
        return first_half, second_half
    
    def josephus_problem(self, k: int) -> int:
        """
        Solve Josephus problem using circular list
        
        Args:
            k: Every k-th person is eliminated
        
        Returns:
            Position of survivor (0-indexed)
        """
        if not self.head or k <= 0:
            return -1
        
        current = self.head
        
        while self.size > 1:
            # Move k-1 steps
            for _ in range(k - 1):
                current = current.next
            
            # Store next node before deletion
            next_node = current.next
            
            # Delete current node
            self.delete_by_value(current.val)
            
            # Move to next node
            current = next_node
        
        return self.head.val if self.head else -1
    
    def print_structure(self) -> str:
        """Print circular structure"""
        if not self.head:
            return "Empty Circular List"
        
        values = self.display()
        return " -> ".join(map(str, values)) + f" -> {values[0]} (circular)"

class DoublyLinkedList:
    """
    Comprehensive Doubly Linked List Implementation
    """
    
    def __init__(self):
        """Initialize empty doubly linked list"""
        self.head: Optional[DoublyListNode] = None
        self.tail: Optional[DoublyListNode] = None
        self.size = 0
    
    def insert_at_beginning(self, val: int) -> None:
        """
        Insert node at beginning
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        
        Args:
            val: Value to insert
        """
        new_node = DoublyListNode(val)
        
        if not self.head:
            self.head = self.tail = new_node
        else:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node
        
        self.size += 1
    
    def insert_at_end(self, val: int) -> None:
        """
        Insert node at end
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        
        Args:
            val: Value to insert
        """
        new_node = DoublyListNode(val)
        
        if not self.tail:
            self.head = self.tail = new_node
        else:
            new_node.prev = self.tail
            self.tail.next = new_node
            self.tail = new_node
        
        self.size += 1
    
    def insert_before_node(self, target_node: DoublyListNode, val: int) -> bool:
        """
        Insert node before target node
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        
        Args:
            target_node: Node to insert before
            val: Value to insert
        
        Returns:
            bool: True if insertion successful
        """
        if not target_node:
            return False
        
        new_node = DoublyListNode(val)
        
        new_node.next = target_node
        new_node.prev = target_node.prev
        
        if target_node.prev:
            target_node.prev.next = new_node
        else:
            self.head = new_node
        
        target_node.prev = new_node
        self.size += 1
        return True
    
    def insert_after_node(self, target_node: DoublyListNode, val: int) -> bool:
        """
        Insert node after target node
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        
        Args:
            target_node: Node to insert after
            val: Value to insert
        
        Returns:
            bool: True if insertion successful
        """
        if not target_node:
            return False
        
        new_node = DoublyListNode(val)
        
        new_node.prev = target_node
        new_node.next = target_node.next
        
        if target_node.next:
            target_node.next.prev = new_node
        else:
            self.tail = new_node
        
        target_node.next = new_node
        self.size += 1
        return True
    
    def delete_node(self, node: DoublyListNode) -> bool:
        """
        Delete specific node
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        
        Args:
            node: Node to delete
        
        Returns:
            bool: True if deletion successful
        """
        if not node:
            return False
        
        # Update previous node
        if node.prev:
            node.prev.next = node.next
        else:
            self.head = node.next
        
        # Update next node
        if node.next:
            node.next.prev = node.prev
        else:
            self.tail = node.prev
        
        self.size -= 1
        return True
    
    def delete_by_value(self, val: int) -> bool:
        """
        Delete first occurrence of value
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Args:
            val: Value to delete
        
        Returns:
            bool: True if deletion successful
        """
        current = self.head
        
        while current:
            if current.val == val:
                return self.delete_node(current)
            current = current.next
        
        return False
    
    def search_node(self, val: int) -> Optional[DoublyListNode]:
        """
        Search for node with given value
        
        Args:
            val: Value to search for
        
        Returns:
            Node with value or None
        """
        current = self.head
        
        while current:
            if current.val == val:
                return current
            current = current.next
        
        return None
    
    def search_optimized(self, val: int) -> Optional[DoublyListNode]:
        """
        Optimized search using both directions
        
        Args:
            val: Value to search for
        
        Returns:
            Node with value or None
        """
        if not self.head:
            return None
        
        left = self.head
        right = self.tail
        
        while left and right and left != right and left.prev != right:
            if left.val == val:
                return left
            if right.val == val:
                return right
            
            left = left.next
            right = right.prev
        
        # Check if they meet at the middle
        if left and left.val == val:
            return left
        
        return None
    
    def display_forward(self) -> List[int]:
        """Display values forward (head to tail)"""
        values = []
        current = self.head
        
        while current:
            values.append(current.val)
            current = current.next
        
        return values
    
    def display_backward(self) -> List[int]:
        """Display values backward (tail to head)"""
        values = []
        current = self.tail
        
        while current:
            values.append(current.val)
            current = current.prev
        
        return values
    
    def reverse(self) -> None:
        """
        Reverse the doubly linked list
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        current = self.head
        
        while current:
            # Swap next and prev
            current.next, current.prev = current.prev, current.next
            current = current.prev  # Move to what was next
        
        # Swap head and tail
        self.head, self.tail = self.tail, self.head
    
    def sort(self) -> None:
        """
        Sort doubly linked list using merge sort
        
        Time Complexity: O(n log n)
        Space Complexity: O(log n)
        """
        if self.size <= 1:
            return
        
        self.head = self._merge_sort(self.head)
        
        # Update tail
        current = self.head
        while current.next:
            current = current.next
        self.tail = current
    
    def _merge_sort(self, head: DoublyListNode) -> DoublyListNode:
        """Helper method for merge sort"""
        if not head or not head.next:
            return head
        
        # Split the list
        mid = self._get_middle(head)
        mid_next = mid.next
        
        mid.next = None
        if mid_next:
            mid_next.prev = None
        
        # Recursively sort both halves
        left = self._merge_sort(head)
        right = self._merge_sort(mid_next)
        
        # Merge sorted halves
        return self._merge(left, right)
    
    def _get_middle(self, head: DoublyListNode) -> DoublyListNode:
        """Get middle node using slow-fast pointer"""
        slow = fast = head
        
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
        
        return slow
    
    def _merge(self, left: DoublyListNode, right: DoublyListNode) -> DoublyListNode:
        """Merge two sorted doubly linked lists"""
        dummy = DoublyListNode(0)
        current = dummy
        
        while left and right:
            if left.val <= right.val:
                current.next = left
                left.prev = current
                left = left.next
            else:
                current.next = right
                right.prev = current
                right = right.next
            current = current.next
        
        # Append remaining nodes
        if left:
            current.next = left
            left.prev = current
        
        if right:
            current.next = right
            right.prev = current
        
        # Set up return node
        result = dummy.next
        if result:
            result.prev = None
        
        return result
    
    def length(self) -> int:
        """Get length of list"""
        return self.size
    
    def is_empty(self) -> bool:
        """Check if list is empty"""
        return self.head is None
    
    def print_structure_forward(self) -> str:
        """Print forward structure"""
        if not self.head:
            return "Empty List"
        
        values = self.display_forward()
        return "NULL <-> " + " <-> ".join(map(str, values)) + " <-> NULL"

class CircularDoublyLinkedList:
    """
    Circular Doubly Linked List Implementation
    """
    
    def __init__(self):
        """Initialize empty circular doubly linked list"""
        self.head: Optional[DoublyListNode] = None
        self.size = 0
    
    def insert_at_beginning(self, val: int) -> None:
        """
        Insert node at beginning
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        
        Args:
            val: Value to insert
        """
        new_node = DoublyListNode(val)
        
        if not self.head:
            self.head = new_node
            new_node.next = new_node.prev = new_node
        else:
            tail = self.head.prev
            
            new_node.next = self.head
            new_node.prev = tail
            self.head.prev = new_node
            tail.next = new_node
            self.head = new_node
        
        self.size += 1
    
    def insert_at_end(self, val: int) -> None:
        """
        Insert node at end
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        
        Args:
            val: Value to insert
        """
        if not self.head:
            self.insert_at_beginning(val)
            return
        
        new_node = DoublyListNode(val)
        tail = self.head.prev
        
        new_node.next = self.head
        new_node.prev = tail
        tail.next = new_node
        self.head.prev = new_node
        
        self.size += 1
    
    def delete_by_value(self, val: int) -> bool:
        """
        Delete first occurrence of value
        
        Args:
            val: Value to delete
        
        Returns:
            bool: True if deletion successful
        """
        if not self.head:
            return False
        
        current = self.head
        
        # Single node case
        if self.size == 1 and current.val == val:
            self.head = None
            self.size -= 1
            return True
        
        # Search for node to delete
        while True:
            if current.val == val:
                # Update connections
                current.prev.next = current.next
                current.next.prev = current.prev
                
                # Update head if necessary
                if current == self.head:
                    self.head = current.next
                
                self.size -= 1
                return True
            
            current = current.next
            if current == self.head:
                break
        
        return False
    
    def display_forward(self) -> List[int]:
        """Display values forward"""
        if not self.head:
            return []
        
        values = []
        current = self.head
        
        while True:
            values.append(current.val)
            current = current.next
            if current == self.head:
                break
        
        return values
    
    def display_backward(self) -> List[int]:
        """Display values backward"""
        if not self.head:
            return []
        
        values = []
        current = self.head.prev  # Start from tail
        
        while True:
            values.append(current.val)
            current = current.prev
            if current == self.head.prev:
                break
        
        return values
    
    def length(self) -> int:
        """Get length of list"""
        return self.size
    
    def rotate(self, k: int) -> None:
        """
        Rotate list by k positions
        
        Args:
            k: Number of positions to rotate (positive = right, negative = left)
        """
        if not self.head or self.size <= 1:
            return
        
        # Normalize k
        k = k % self.size
        if k == 0:
            return
        
        # Find new head
        current = self.head
        for _ in range(k):
            current = current.next
        
        self.head = current

class LRUCache:
    """
    LRU Cache implementation using doubly linked list
    """
    
    def __init__(self, capacity: int):
        """
        Initialize LRU cache
        
        Args:
            capacity: Maximum capacity of cache
        """
        self.capacity = capacity
        self.cache = {}
        
        # Create dummy head and tail
        self.head = DoublyListNode(0)
        self.tail = DoublyListNode(0)
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def _add_node(self, node: DoublyListNode) -> None:
        """Add node right after head"""
        node.prev = self.head
        node.next = self.head.next
        
        self.head.next.prev = node
        self.head.next = node
    
    def _remove_node(self, node: DoublyListNode) -> None:
        """Remove an existing node"""
        prev_node = node.prev
        next_node = node.next
        
        prev_node.next = next_node
        next_node.prev = prev_node
    
    def _move_to_head(self, node: DoublyListNode) -> None:
        """Move node to head"""
        self._remove_node(node)
        self._add_node(node)
    
    def _pop_tail(self) -> DoublyListNode:
        """Pop the last node"""
        last_node = self.tail.prev
        self._remove_node(last_node)
        return last_node
    
    def get(self, key: int) -> int:
        """
        Get value from cache
        
        Args:
            key: Key to get
        
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
            key: Key to put
            value: Value to put
        """
        node = self.cache.get(key)
        
        if not node:
            new_node = DoublyListNode(value)
            new_node.val = value  # Store value
            
            self.cache[key] = new_node
            self._add_node(new_node)
            
            if len(self.cache) > self.capacity:
                # Remove least recently used
                tail = self._pop_tail()
                # Find and remove from cache
                key_to_remove = None
                for k, v in self.cache.items():
                    if v == tail:
                        key_to_remove = k
                        break
                if key_to_remove is not None:
                    del self.cache[key_to_remove]
        else:
            # Update value and move to head
            node.val = value
            self._move_to_head(node)

class AdvancedLinkedListOperations:
    """Advanced operations on linked lists"""
    
    @staticmethod
    def merge_k_sorted_lists(lists: List[DoublyLinkedList]) -> DoublyLinkedList:
        """
        Merge k sorted doubly linked lists
        
        Time Complexity: O(n log k) where n is total nodes, k is number of lists
        
        Args:
            lists: List of sorted doubly linked lists
        
        Returns:
            Merged sorted doubly linked list
        """
        if not lists:
            return DoublyLinkedList()
        
        while len(lists) > 1:
            merged_lists = []
            
            for i in range(0, len(lists), 2):
                list1 = lists[i]
                list2 = lists[i + 1] if i + 1 < len(lists) else DoublyLinkedList()
                
                merged = AdvancedLinkedListOperations._merge_two_sorted_lists(list1, list2)
                merged_lists.append(merged)
            
            lists = merged_lists
        
        return lists[0]
    
    @staticmethod
    def _merge_two_sorted_lists(list1: DoublyLinkedList, list2: DoublyLinkedList) -> DoublyLinkedList:
        """Merge two sorted doubly linked lists"""
        result = DoublyLinkedList()
        
        p1, p2 = list1.head, list2.head
        
        while p1 and p2:
            if p1.val <= p2.val:
                result.insert_at_end(p1.val)
                p1 = p1.next
            else:
                result.insert_at_end(p2.val)
                p2 = p2.next
        
        # Add remaining nodes
        while p1:
            result.insert_at_end(p1.val)
            p1 = p1.next
        
        while p2:
            result.insert_at_end(p2.val)
            p2 = p2.next
        
        return result
    
    @staticmethod
    def flatten_multilevel_list(head: DoublyListNode) -> DoublyListNode:
        """
        Flatten a multilevel doubly linked list
        (Assuming nodes have a child pointer)
        
        Args:
            head: Head of multilevel list
        
        Returns:
            Head of flattened list
        """
        if not head:
            return None
        
        stack = []
        current = head
        
        while current:
            if hasattr(current, 'child') and current.child:
                if current.next:
                    stack.append(current.next)
                
                current.next = current.child
                current.child.prev = current
                current.child = None
            
            if not current.next and stack:
                next_node = stack.pop()
                current.next = next_node
                next_node.prev = current
            
            current = current.next
        
        return head

# ==================== EXAMPLE USAGE AND TESTING ====================

if __name__ == "__main__":
    print("=== Circular & Doubly Linked Lists Demo ===\n")
    
    # Example 1: Circular Singly Linked List
    print("1. Circular Singly Linked List:")
    
    cll = CircularLinkedList()
    
    # Insert operations
    print("Inserting elements:")
    cll.insert_at_beginning(3)
    cll.insert_at_beginning(2)
    cll.insert_at_beginning(1)
    cll.insert_at_end(4)
    cll.insert_at_end(5)
    cll.insert_at_position(2, 10)
    
    print(f"List: {cll.display()}")
    print(f"Structure: {cll.print_structure()}")
    print(f"Length: {cll.length()}")
    
    # Search and delete
    print(f"Search 10: position {cll.search(10)}")
    print(f"Delete 10: {cll.delete_by_value(10)}")
    print(f"After deletion: {cll.display()}")
    
    # Split list
    first_half, second_half = cll.split_list()
    print(f"First half: {first_half.display() if first_half else 'None'}")
    print(f"Second half: {second_half.display() if second_half else 'None'}")
    
    # Josephus problem
    josephus_list = CircularLinkedList()
    for i in range(1, 8):
        josephus_list.insert_at_end(i)
    
    print(f"Josephus problem with k=3: {josephus_list.josephus_problem(3)}")
    print()
    
    # Example 2: Doubly Linked List
    print("2. Doubly Linked List:")
    
    dll = DoublyLinkedList()
    
    # Insert operations
    print("Inserting elements:")
    dll.insert_at_beginning(2)
    dll.insert_at_beginning(1)
    dll.insert_at_end(3)
    dll.insert_at_end(4)
    
    print(f"Forward: {dll.display_forward()}")
    print(f"Backward: {dll.display_backward()}")
    print(f"Structure: {dll.print_structure_forward()}")
    
    # Insert before/after specific nodes
    node_2 = dll.search_node(2)
    if node_2:
        dll.insert_after_node(node_2, 25)
        dll.insert_before_node(node_2, 15)
    
    print(f"After insert before/after node 2: {dll.display_forward()}")
    
    # Delete operations
    node_25 = dll.search_node(25)
    if node_25:
        dll.delete_node(node_25)
    
    print(f"After deleting node 25: {dll.display_forward()}")
    
    # Reverse list
    dll.reverse()
    print(f"After reverse: {dll.display_forward()}")
    
    # Sort list
    dll.insert_at_end(1)
    dll.insert_at_end(6)
    dll.insert_at_end(2)
    print(f"Before sort: {dll.display_forward()}")
    dll.sort()
    print(f"After sort: {dll.display_forward()}")
    print()
    
    # Example 3: Circular Doubly Linked List
    print("3. Circular Doubly Linked List:")
    
    cdll = CircularDoublyLinkedList()
    
    # Insert operations
    cdll.insert_at_beginning(3)
    cdll.insert_at_beginning(2)
    cdll.insert_at_beginning(1)
    cdll.insert_at_end(4)
    cdll.insert_at_end(5)
    
    print(f"Forward: {cdll.display_forward()}")
    print(f"Backward: {cdll.display_backward()}")
    
    # Rotate operations
    print(f"Original: {cdll.display_forward()}")
    cdll.rotate(2)
    print(f"After rotate right by 2: {cdll.display_forward()}")
    cdll.rotate(-3)
    print(f"After rotate left by 3: {cdll.display_forward()}")
    
    # Delete operation
    print(f"Delete 3: {cdll.delete_by_value(3)}")
    print(f"After deletion: {cdll.display_forward()}")
    print()
    
    # Example 4: LRU Cache
    print("4. LRU Cache:")
    
    lru = LRUCache(3)
    
    # Cache operations
    lru.put(1, 10)
    lru.put(2, 20)
    lru.put(3, 30)
    
    print(f"Get 1: {lru.get(1)}")  # Should return 10
    print(f"Get 2: {lru.get(2)}")  # Should return 20
    
    lru.put(4, 40)  # This should evict key 3
    
    print(f"Get 3: {lru.get(3)}")  # Should return -1 (not found)
    print(f"Get 4: {lru.get(4)}")  # Should return 40
    print(f"Get 1: {lru.get(1)}")  # Should return 10
    print(f"Get 2: {lru.get(2)}")  # Should return 20
    print()
    
    # Example 5: Advanced Operations
    print("5. Advanced Operations:")
    
    # Create multiple sorted lists
    list1 = DoublyLinkedList()
    list2 = DoublyLinkedList()
    list3 = DoublyLinkedList()
    
    for val in [1, 4, 7]:
        list1.insert_at_end(val)
    
    for val in [2, 5, 8]:
        list2.insert_at_end(val)
    
    for val in [3, 6, 9]:
        list3.insert_at_end(val)
    
    print(f"List 1: {list1.display_forward()}")
    print(f"List 2: {list2.display_forward()}")
    print(f"List 3: {list3.display_forward()}")
    
    # Merge k sorted lists
    merged = AdvancedLinkedListOperations.merge_k_sorted_lists([list1, list2, list3])
    print(f"Merged: {merged.display_forward()}")
    print()
    
    # Example 6: Performance Analysis
    print("6. Performance Analysis:")
    
    # Large circular list operations
    large_cll = CircularLinkedList()
    
    print("Inserting 1000 elements in circular list...")
    for i in range(1000):
        large_cll.insert_at_end(i)
    
    print(f"Large circular list length: {large_cll.length()}")
    
    # Search operations
    search_results = []
    for val in [100, 500, 999, 1000]:
        pos = large_cll.search(val)
        search_results.append((val, pos))
    
    print("Search results (value, position):")
    for val, pos in search_results:
        print(f"  {val}: {pos}")
    
    # Large doubly linked list
    large_dll = DoublyLinkedList()
    
    print("\nInserting 1000 elements in doubly linked list...")
    for i in range(1000):
        large_dll.insert_at_end(i)
    
    print(f"Large doubly linked list length: {large_dll.length()}")
    
    # Optimized search
    node_500 = large_dll.search_optimized(500)
    print(f"Optimized search for 500: {node_500.val if node_500 else 'Not found'}")
    
    # Performance comparison
    normal_search_node = large_dll.search_node(750)
    optimized_search_node = large_dll.search_optimized(750)
    
    print(f"Normal search for 750: {normal_search_node.val if normal_search_node else 'Not found'}")
    print(f"Optimized search for 750: {optimized_search_node.val if optimized_search_node else 'Not found'}")
    
    print("\n=== Demo Complete ===") 