"""
Linked List Data Structures - Implementation of Various Data Structures Using Linked Lists
This module implements fundamental data structures using linked lists as the underlying storage mechanism.
"""

from typing import Optional, Any, List, Tuple
import heapq
from abc import ABC, abstractmethod

class ListNode:
    """Basic node class for linked list"""
    def __init__(self, val: Any = None, next: Optional['ListNode'] = None):
        self.val = val
        self.next = next
    
    def __repr__(self):
        return f"ListNode({self.val})"

class PriorityNode:
    """Node class for priority queue with priority and data"""
    def __init__(self, priority: int, data: Any, next: Optional['PriorityNode'] = None):
        self.priority = priority
        self.data = data
        self.next = next
    
    def __repr__(self):
        return f"PriorityNode(priority={self.priority}, data={self.data})"
    
    def __lt__(self, other):
        """For comparison in heap operations"""
        return self.priority < other.priority
    
    def __eq__(self, other):
        return self.priority == other.priority

# ==================== STACK USING LINKED LIST ====================

class LinkedListStack:
    """
    Stack implementation using linked list
    
    All operations are O(1) time complexity
    Space complexity: O(n) where n is number of elements
    """
    
    def __init__(self):
        """Initialize empty stack"""
        self.top = None
        self.size = 0
    
    def push(self, val: Any) -> None:
        """
        Push element onto stack
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        
        Args:
            val: Value to push onto stack
        """
        new_node = ListNode(val)
        new_node.next = self.top
        self.top = new_node
        self.size += 1
    
    def pop(self) -> Any:
        """
        Pop element from stack
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        
        Returns:
            Top element from stack
        
        Raises:
            IndexError: If stack is empty
        """
        if self.is_empty():
            raise IndexError("pop from empty stack")
        
        popped_val = self.top.val
        self.top = self.top.next
        self.size -= 1
        return popped_val
    
    def peek(self) -> Any:
        """
        Get top element without removing it
        
        Returns:
            Top element from stack
        
        Raises:
            IndexError: If stack is empty
        """
        if self.is_empty():
            raise IndexError("peek from empty stack")
        
        return self.top.val
    
    def is_empty(self) -> bool:
        """Check if stack is empty"""
        return self.top is None
    
    def get_size(self) -> int:
        """Get current size of stack"""
        return self.size
    
    def clear(self) -> None:
        """Clear all elements from stack"""
        self.top = None
        self.size = 0
    
    def to_list(self) -> List[Any]:
        """Convert stack to list (top to bottom)"""
        result = []
        current = self.top
        while current:
            result.append(current.val)
            current = current.next
        return result
    
    def search(self, val: Any) -> int:
        """
        Search for element in stack
        
        Args:
            val: Value to search for
        
        Returns:
            Distance from top (0-indexed), -1 if not found
        """
        current = self.top
        position = 0
        
        while current:
            if current.val == val:
                return position
            current = current.next
            position += 1
        
        return -1
    
    def reverse(self) -> None:
        """Reverse the stack in-place"""
        if self.size <= 1:
            return
        
        # Use recursion to reverse
        def reverse_recursive(node):
            if not node or not node.next:
                return node
            
            reversed_head = reverse_recursive(node.next)
            node.next.next = node
            node.next = None
            return reversed_head
        
        self.top = reverse_recursive(self.top)

class StackWithMin:
    """
    Stack that supports getting minimum element in O(1) time
    """
    
    def __init__(self):
        """Initialize stack with min tracking"""
        self.main_stack = LinkedListStack()
        self.min_stack = LinkedListStack()
    
    def push(self, val: Any) -> None:
        """Push element and update minimum"""
        self.main_stack.push(val)
        
        if self.min_stack.is_empty() or val <= self.min_stack.peek():
            self.min_stack.push(val)
    
    def pop(self) -> Any:
        """Pop element and update minimum"""
        if self.main_stack.is_empty():
            raise IndexError("pop from empty stack")
        
        popped = self.main_stack.pop()
        
        if popped == self.min_stack.peek():
            self.min_stack.pop()
        
        return popped
    
    def peek(self) -> Any:
        """Get top element"""
        return self.main_stack.peek()
    
    def get_min(self) -> Any:
        """Get minimum element in O(1) time"""
        if self.min_stack.is_empty():
            raise IndexError("get_min from empty stack")
        return self.min_stack.peek()
    
    def is_empty(self) -> bool:
        """Check if stack is empty"""
        return self.main_stack.is_empty()

# ==================== QUEUE USING LINKED LIST ====================

class LinkedListQueue:
    """
    Queue implementation using linked list
    
    All operations are O(1) time complexity
    Space complexity: O(n) where n is number of elements
    """
    
    def __init__(self):
        """Initialize empty queue"""
        self.front = None
        self.rear = None
        self.size = 0
    
    def enqueue(self, val: Any) -> None:
        """
        Add element to rear of queue
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        
        Args:
            val: Value to add to queue
        """
        new_node = ListNode(val)
        
        if self.rear is None:
            # First element
            self.front = self.rear = new_node
        else:
            self.rear.next = new_node
            self.rear = new_node
        
        self.size += 1
    
    def dequeue(self) -> Any:
        """
        Remove element from front of queue
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        
        Returns:
            Front element from queue
        
        Raises:
            IndexError: If queue is empty
        """
        if self.is_empty():
            raise IndexError("dequeue from empty queue")
        
        dequeued_val = self.front.val
        self.front = self.front.next
        
        if self.front is None:
            # Queue became empty
            self.rear = None
        
        self.size -= 1
        return dequeued_val
    
    def peek_front(self) -> Any:
        """Get front element without removing it"""
        if self.is_empty():
            raise IndexError("peek from empty queue")
        return self.front.val
    
    def peek_rear(self) -> Any:
        """Get rear element without removing it"""
        if self.is_empty():
            raise IndexError("peek from empty queue")
        return self.rear.val
    
    def is_empty(self) -> bool:
        """Check if queue is empty"""
        return self.front is None
    
    def get_size(self) -> int:
        """Get current size of queue"""
        return self.size
    
    def clear(self) -> None:
        """Clear all elements from queue"""
        self.front = None
        self.rear = None
        self.size = 0
    
    def to_list(self) -> List[Any]:
        """Convert queue to list (front to rear)"""
        result = []
        current = self.front
        while current:
            result.append(current.val)
            current = current.next
        return result
    
    def reverse(self) -> None:
        """Reverse the queue"""
        if self.size <= 1:
            return
        
        # Use stack to reverse
        stack = LinkedListStack()
        
        # Transfer all elements to stack
        while not self.is_empty():
            stack.push(self.dequeue())
        
        # Transfer back to queue
        while not stack.is_empty():
            self.enqueue(stack.pop())

class CircularQueue:
    """
    Circular queue implementation using linked list
    """
    
    def __init__(self, capacity: int):
        """
        Initialize circular queue with fixed capacity
        
        Args:
            capacity: Maximum number of elements
        """
        self.capacity = capacity
        self.size = 0
        self.front = None
        self.rear = None
    
    def enqueue(self, val: Any) -> bool:
        """Add element if space available"""
        if self.is_full():
            return False
        
        new_node = ListNode(val)
        
        if self.is_empty():
            self.front = self.rear = new_node
            new_node.next = new_node  # Point to itself
        else:
            new_node.next = self.rear.next
            self.rear.next = new_node
            self.rear = new_node
        
        self.size += 1
        return True
    
    def dequeue(self) -> Any:
        """Remove front element"""
        if self.is_empty():
            raise IndexError("dequeue from empty circular queue")
        
        dequeued_val = self.front.val
        
        if self.size == 1:
            # Last element
            self.front = self.rear = None
        else:
            self.rear.next = self.front.next
            self.front = self.front.next
        
        self.size -= 1
        return dequeued_val
    
    def is_empty(self) -> bool:
        """Check if queue is empty"""
        return self.size == 0
    
    def is_full(self) -> bool:
        """Check if queue is full"""
        return self.size == self.capacity
    
    def peek(self) -> Any:
        """Get front element"""
        if self.is_empty():
            raise IndexError("peek from empty circular queue")
        return self.front.val

class Deque:
    """
    Double-ended queue (deque) implementation using doubly linked list
    """
    
    class DoublyListNode:
        """Node for doubly linked list"""
        def __init__(self, val: Any = None, prev: Optional['DoublyListNode'] = None, 
                     next: Optional['DoublyListNode'] = None):
            self.val = val
            self.prev = prev
            self.next = next
    
    def __init__(self):
        """Initialize empty deque"""
        # Use dummy nodes for easier manipulation
        self.head = self.DoublyListNode()
        self.tail = self.DoublyListNode()
        self.head.next = self.tail
        self.tail.prev = self.head
        self.size = 0
    
    def add_front(self, val: Any) -> None:
        """Add element to front"""
        new_node = self.DoublyListNode(val)
        
        new_node.next = self.head.next
        new_node.prev = self.head
        self.head.next.prev = new_node
        self.head.next = new_node
        
        self.size += 1
    
    def add_rear(self, val: Any) -> None:
        """Add element to rear"""
        new_node = self.DoublyListNode(val)
        
        new_node.next = self.tail
        new_node.prev = self.tail.prev
        self.tail.prev.next = new_node
        self.tail.prev = new_node
        
        self.size += 1
    
    def remove_front(self) -> Any:
        """Remove element from front"""
        if self.is_empty():
            raise IndexError("remove from empty deque")
        
        node_to_remove = self.head.next
        removed_val = node_to_remove.val
        
        self.head.next = node_to_remove.next
        node_to_remove.next.prev = self.head
        
        self.size -= 1
        return removed_val
    
    def remove_rear(self) -> Any:
        """Remove element from rear"""
        if self.is_empty():
            raise IndexError("remove from empty deque")
        
        node_to_remove = self.tail.prev
        removed_val = node_to_remove.val
        
        self.tail.prev = node_to_remove.prev
        node_to_remove.prev.next = self.tail
        
        self.size -= 1
        return removed_val
    
    def peek_front(self) -> Any:
        """Get front element"""
        if self.is_empty():
            raise IndexError("peek from empty deque")
        return self.head.next.val
    
    def peek_rear(self) -> Any:
        """Get rear element"""
        if self.is_empty():
            raise IndexError("peek from empty deque")
        return self.tail.prev.val
    
    def is_empty(self) -> bool:
        """Check if deque is empty"""
        return self.size == 0
    
    def get_size(self) -> int:
        """Get current size"""
        return self.size

# ==================== PRIORITY QUEUE USING LINKED LIST ====================

class LinkedListPriorityQueue:
    """
    Priority Queue implementation using sorted linked list
    
    Lower priority values have higher priority (min-heap behavior)
    """
    
    def __init__(self, max_heap: bool = False):
        """
        Initialize priority queue
        
        Args:
            max_heap: If True, higher values have higher priority
        """
        self.head = None
        self.size = 0
        self.max_heap = max_heap
    
    def enqueue(self, priority: int, data: Any) -> None:
        """
        Add element with given priority
        
        Time Complexity: O(n) - need to find correct position
        Space Complexity: O(1)
        
        Args:
            priority: Priority value (lower = higher priority for min-heap)
            data: Data to store
        """
        new_node = PriorityNode(priority, data)
        
        # Empty queue or new node has highest priority
        if (not self.head or 
            (not self.max_heap and priority < self.head.priority) or
            (self.max_heap and priority > self.head.priority)):
            new_node.next = self.head
            self.head = new_node
        else:
            # Find correct position
            current = self.head
            while (current.next and 
                   ((not self.max_heap and current.next.priority <= priority) or
                    (self.max_heap and current.next.priority >= priority))):
                current = current.next
            
            new_node.next = current.next
            current.next = new_node
        
        self.size += 1
    
    def dequeue(self) -> Tuple[int, Any]:
        """
        Remove and return highest priority element
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        
        Returns:
            Tuple of (priority, data)
        
        Raises:
            IndexError: If queue is empty
        """
        if self.is_empty():
            raise IndexError("dequeue from empty priority queue")
        
        priority = self.head.priority
        data = self.head.data
        self.head = self.head.next
        self.size -= 1
        
        return priority, data
    
    def peek(self) -> Tuple[int, Any]:
        """Get highest priority element without removing"""
        if self.is_empty():
            raise IndexError("peek from empty priority queue")
        
        return self.head.priority, self.head.data
    
    def is_empty(self) -> bool:
        """Check if queue is empty"""
        return self.head is None
    
    def get_size(self) -> int:
        """Get current size"""
        return self.size
    
    def to_list(self) -> List[Tuple[int, Any]]:
        """Convert to list of (priority, data) tuples"""
        result = []
        current = self.head
        while current:
            result.append((current.priority, current.data))
            current = current.next
        return result

class HeapBasedPriorityQueue:
    """
    Priority Queue using Python's heapq module for comparison
    
    This provides O(log n) enqueue/dequeue operations
    """
    
    def __init__(self, max_heap: bool = False):
        """Initialize heap-based priority queue"""
        self.heap = []
        self.max_heap = max_heap
        self.size = 0
    
    def enqueue(self, priority: int, data: Any) -> None:
        """Add element - O(log n)"""
        if self.max_heap:
            # Negate priority for max heap behavior
            heapq.heappush(self.heap, (-priority, data))
        else:
            heapq.heappush(self.heap, (priority, data))
        self.size += 1
    
    def dequeue(self) -> Tuple[int, Any]:
        """Remove highest priority element - O(log n)"""
        if self.is_empty():
            raise IndexError("dequeue from empty priority queue")
        
        priority, data = heapq.heappop(self.heap)
        self.size -= 1
        
        if self.max_heap:
            priority = -priority
        
        return priority, data
    
    def peek(self) -> Tuple[int, Any]:
        """Get highest priority element - O(1)"""
        if self.is_empty():
            raise IndexError("peek from empty priority queue")
        
        priority, data = self.heap[0]
        if self.max_heap:
            priority = -priority
        
        return priority, data
    
    def is_empty(self) -> bool:
        """Check if empty"""
        return len(self.heap) == 0
    
    def get_size(self) -> int:
        """Get size"""
        return self.size

class MultiLevelPriorityQueue:
    """
    Multi-level priority queue with separate queues for different priority levels
    """
    
    def __init__(self, num_levels: int = 3):
        """
        Initialize multi-level priority queue
        
        Args:
            num_levels: Number of priority levels (0 = highest)
        """
        self.num_levels = num_levels
        self.queues = [LinkedListQueue() for _ in range(num_levels)]
        self.size = 0
    
    def enqueue(self, priority_level: int, data: Any) -> None:
        """Add element to specific priority level"""
        if 0 <= priority_level < self.num_levels:
            self.queues[priority_level].enqueue(data)
            self.size += 1
        else:
            raise ValueError(f"Priority level must be between 0 and {self.num_levels - 1}")
    
    def dequeue(self) -> Any:
        """Remove from highest priority non-empty queue"""
        if self.is_empty():
            raise IndexError("dequeue from empty multi-level priority queue")
        
        for level in range(self.num_levels):
            if not self.queues[level].is_empty():
                self.size -= 1
                return self.queues[level].dequeue()
        
        raise IndexError("All queues are empty")
    
    def peek(self) -> Any:
        """Get highest priority element"""
        if self.is_empty():
            raise IndexError("peek from empty multi-level priority queue")
        
        for level in range(self.num_levels):
            if not self.queues[level].is_empty():
                return self.queues[level].peek_front()
        
        raise IndexError("All queues are empty")
    
    def is_empty(self) -> bool:
        """Check if all queues are empty"""
        return self.size == 0
    
    def get_size(self) -> int:
        """Get total size across all levels"""
        return self.size
    
    def get_level_sizes(self) -> List[int]:
        """Get sizes of each priority level"""
        return [queue.get_size() for queue in self.queues]

# ==================== SPECIALIZED DATA STRUCTURES ====================

class StackWithTwoQueues:
    """
    Stack implementation using two queues
    Demonstrates how different data structures can be implemented using others
    """
    
    def __init__(self):
        """Initialize stack using two queues"""
        self.q1 = LinkedListQueue()
        self.q2 = LinkedListQueue()
    
    def push(self, val: Any) -> None:
        """
        Push operation - O(n)
        Strategy: Add to q1, then move all previous elements from q2 to q1
        """
        self.q1.enqueue(val)
        
        # Move all elements from q2 to q1
        while not self.q2.is_empty():
            self.q1.enqueue(self.q2.dequeue())
        
        # Swap q1 and q2
        self.q1, self.q2 = self.q2, self.q1
    
    def pop(self) -> Any:
        """Pop operation - O(1)"""
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self.q2.dequeue()
    
    def peek(self) -> Any:
        """Peek operation - O(1)"""
        if self.is_empty():
            raise IndexError("peek from empty stack")
        return self.q2.peek_front()
    
    def is_empty(self) -> bool:
        """Check if empty"""
        return self.q2.is_empty()

class QueueWithTwoStacks:
    """
    Queue implementation using two stacks
    """
    
    def __init__(self):
        """Initialize queue using two stacks"""
        self.inbox = LinkedListStack()   # For enqueue operations
        self.outbox = LinkedListStack()  # For dequeue operations
    
    def enqueue(self, val: Any) -> None:
        """Enqueue operation - O(1)"""
        self.inbox.push(val)
    
    def dequeue(self) -> Any:
        """
        Dequeue operation - Amortized O(1)
        Move elements from inbox to outbox when outbox is empty
        """
        if self.outbox.is_empty():
            if self.inbox.is_empty():
                raise IndexError("dequeue from empty queue")
            
            # Transfer all elements from inbox to outbox
            while not self.inbox.is_empty():
                self.outbox.push(self.inbox.pop())
        
        return self.outbox.pop()
    
    def peek(self) -> Any:
        """Peek operation - Amortized O(1)"""
        if self.outbox.is_empty():
            if self.inbox.is_empty():
                raise IndexError("peek from empty queue")
            
            # Transfer elements
            while not self.inbox.is_empty():
                self.outbox.push(self.inbox.pop())
        
        return self.outbox.peek()
    
    def is_empty(self) -> bool:
        """Check if empty"""
        return self.inbox.is_empty() and self.outbox.is_empty()

class LFUCache:
    """
    LFU (Least Frequently Used) Cache using linked lists
    """
    
    class Node:
        def __init__(self, key: int, val: Any, freq: int = 1):
            self.key = key
            self.val = val
            self.freq = freq
            self.prev = None
            self.next = None
    
    def __init__(self, capacity: int):
        """Initialize LFU cache"""
        self.capacity = capacity
        self.min_freq = 0
        self.key_to_node = {}  # key -> node
        self.freq_to_list = {}  # frequency -> doubly linked list
    
    def _add_node(self, node: 'Node', freq: int) -> None:
        """Add node to frequency list"""
        if freq not in self.freq_to_list:
            self.freq_to_list[freq] = {'head': self.Node(0, 0), 'tail': self.Node(0, 0)}
            self.freq_to_list[freq]['head'].next = self.freq_to_list[freq]['tail']
            self.freq_to_list[freq]['tail'].prev = self.freq_to_list[freq]['head']
        
        head = self.freq_to_list[freq]['head']
        node.next = head.next
        node.prev = head
        head.next.prev = node
        head.next = node
    
    def _remove_node(self, node: 'Node') -> None:
        """Remove node from its current list"""
        node.prev.next = node.next
        node.next.prev = node.prev
    
    def get(self, key: int) -> Any:
        """Get value and update frequency"""
        if key not in self.key_to_node:
            return None
        
        node = self.key_to_node[key]
        self._remove_node(node)
        
        # Update frequency
        old_freq = node.freq
        node.freq += 1
        self._add_node(node, node.freq)
        
        # Update min_freq if needed
        if old_freq == self.min_freq and self._is_list_empty(old_freq):
            self.min_freq += 1
        
        return node.val
    
    def put(self, key: int, value: Any) -> None:
        """Put key-value pair"""
        if self.capacity == 0:
            return
        
        if key in self.key_to_node:
            # Update existing
            node = self.key_to_node[key]
            node.val = value
            self.get(key)  # This will update frequency
        else:
            # Add new
            if len(self.key_to_node) >= self.capacity:
                # Remove LFU
                tail = self.freq_to_list[self.min_freq]['tail']
                lfu_node = tail.prev
                self._remove_node(lfu_node)
                del self.key_to_node[lfu_node.key]
            
            # Add new node
            new_node = self.Node(key, value)
            self.key_to_node[key] = new_node
            self._add_node(new_node, 1)
            self.min_freq = 1
    
    def _is_list_empty(self, freq: int) -> bool:
        """Check if frequency list is empty"""
        if freq not in self.freq_to_list:
            return True
        head = self.freq_to_list[freq]['head']
        tail = self.freq_to_list[freq]['tail']
        return head.next == tail

# ==================== EXAMPLE USAGE AND TESTING ====================

if __name__ == "__main__":
    print("=== Linked List Data Structures Demo ===\n")
    
    # Example 1: Stack Operations
    print("1. Stack Using Linked List:")
    
    stack = LinkedListStack()
    
    # Basic operations
    operations = [("push", 1), ("push", 2), ("push", 3), ("pop", None), ("peek", None), ("push", 4)]
    
    for op, val in operations:
        if op == "push":
            stack.push(val)
            print(f"Push {val} -> Stack: {stack.to_list()}, Size: {stack.get_size()}")
        elif op == "pop":
            popped = stack.pop()
            print(f"Pop -> {popped}, Stack: {stack.to_list()}, Size: {stack.get_size()}")
        elif op == "peek":
            peeked = stack.peek()
            print(f"Peek -> {peeked}, Stack: {stack.to_list()}")
    
    # Stack with min
    print(f"\nStack with minimum tracking:")
    min_stack = StackWithMin()
    
    min_operations = [3, 1, 4, 1, 5, 0, 2]
    for val in min_operations:
        min_stack.push(val)
        print(f"Push {val} -> Min: {min_stack.get_min()}")
    
    while not min_stack.is_empty():
        popped = min_stack.pop()
        min_val = min_stack.get_min() if not min_stack.is_empty() else "N/A"
        print(f"Pop {popped} -> Min: {min_val}")
    print()
    
    # Example 2: Queue Operations
    print("2. Queue Using Linked List:")
    
    queue = LinkedListQueue()
    
    # Basic operations
    queue_ops = [("enqueue", 'A'), ("enqueue", 'B'), ("enqueue", 'C'), 
                 ("dequeue", None), ("enqueue", 'D'), ("dequeue", None)]
    
    for op, val in queue_ops:
        if op == "enqueue":
            queue.enqueue(val)
            print(f"Enqueue {val} -> Queue: {queue.to_list()}, Size: {queue.get_size()}")
        elif op == "dequeue":
            dequeued = queue.dequeue()
            print(f"Dequeue -> {dequeued}, Queue: {queue.to_list()}, Size: {queue.get_size()}")
    
    # Circular Queue
    print(f"\nCircular Queue (capacity=3):")
    circular_q = CircularQueue(3)
    
    circular_ops = [("enqueue", 1), ("enqueue", 2), ("enqueue", 3), 
                   ("enqueue", 4), ("dequeue", None), ("enqueue", 4)]
    
    for op, val in circular_ops:
        if op == "enqueue":
            success = circular_q.enqueue(val)
            print(f"Enqueue {val} -> Success: {success}, Full: {circular_q.is_full()}")
        elif op == "dequeue":
            dequeued = circular_q.dequeue()
            print(f"Dequeue -> {dequeued}, Empty: {circular_q.is_empty()}")
    
    # Deque
    print(f"\nDouble-ended Queue (Deque):")
    deque = Deque()
    
    deque_ops = [("add_front", 1), ("add_rear", 2), ("add_front", 0), 
                ("remove_rear", None), ("remove_front", None)]
    
    for op, val in deque_ops:
        if "add" in op:
            if "front" in op:
                deque.add_front(val)
            else:
                deque.add_rear(val)
            print(f"{op.replace('_', ' ').title()} {val} -> Front: {deque.peek_front()}, Rear: {deque.peek_rear()}")
        else:
            if "front" in op:
                removed = deque.remove_front()
            else:
                removed = deque.remove_rear()
            if not deque.is_empty():
                print(f"{op.replace('_', ' ').title()} -> {removed}, Front: {deque.peek_front()}, Rear: {deque.peek_rear()}")
            else:
                print(f"{op.replace('_', ' ').title()} -> {removed}, Deque is empty")
    print()
    
    # Example 3: Priority Queue
    print("3. Priority Queue Using Linked List:")
    
    # Min-heap priority queue
    pq = LinkedListPriorityQueue()
    
    priority_ops = [(3, "Low"), (1, "High"), (2, "Medium"), (1, "Critical"), (4, "Lowest")]
    
    print("Enqueue operations (lower number = higher priority):")
    for priority, data in priority_ops:
        pq.enqueue(priority, data)
        print(f"Enqueue ({priority}, {data}) -> Queue: {pq.to_list()}")
    
    print(f"\nDequeue operations:")
    while not pq.is_empty():
        priority, data = pq.dequeue()
        print(f"Dequeue -> ({priority}, {data}), Remaining: {pq.to_list()}")
    
    # Max-heap priority queue
    print(f"\nMax-heap Priority Queue:")
    max_pq = LinkedListPriorityQueue(max_heap=True)
    
    for priority, data in [(1, "Low"), (5, "High"), (3, "Medium")]:
        max_pq.enqueue(priority, data)
        print(f"Enqueue ({priority}, {data}) -> Queue: {max_pq.to_list()}")
    
    # Multi-level priority queue
    print(f"\nMulti-level Priority Queue:")
    mlpq = MultiLevelPriorityQueue(num_levels=3)
    
    ml_ops = [(0, "Emergency"), (2, "Normal"), (1, "Important"), (0, "Critical"), (2, "Routine")]
    
    for level, data in ml_ops:
        mlpq.enqueue(level, data)
        print(f"Enqueue Level {level}: {data} -> Sizes: {mlpq.get_level_sizes()}")
    
    print(f"Dequeue from multi-level queue:")
    while not mlpq.is_empty():
        dequeued = mlpq.dequeue()
        print(f"Dequeue -> {dequeued}, Remaining sizes: {mlpq.get_level_sizes()}")
    print()
    
    # Example 4: Hybrid Data Structures
    print("4. Hybrid Data Structures:")
    
    # Stack using two queues
    print("Stack implemented with two queues:")
    stack_q = StackWithTwoQueues()
    
    for val in [1, 2, 3]:
        stack_q.push(val)
        print(f"Push {val} -> Top: {stack_q.peek()}")
    
    while not stack_q.is_empty():
        popped = stack_q.pop()
        top = stack_q.peek() if not stack_q.is_empty() else "Empty"
        print(f"Pop -> {popped}, Top: {top}")
    
    # Queue using two stacks
    print(f"\nQueue implemented with two stacks:")
    queue_s = QueueWithTwoStacks()
    
    for val in ['X', 'Y', 'Z']:
        queue_s.enqueue(val)
        print(f"Enqueue {val} -> Front: {queue_s.peek()}")
    
    while not queue_s.is_empty():
        dequeued = queue_s.dequeue()
        front = queue_s.peek() if not queue_s.is_empty() else "Empty"
        print(f"Dequeue -> {dequeued}, Front: {front}")
    print()
    
    # Example 5: Performance Comparison
    print("5. Performance Comparison:")
    
    # Compare linked list vs heap-based priority queue
    print("Priority Queue Performance Comparison:")
    
    # Linked list implementation
    ll_pq = LinkedListPriorityQueue()
    
    # Heap-based implementation
    heap_pq = HeapBasedPriorityQueue()
    
    import time
    import random
    
    # Test data
    test_data = [(random.randint(1, 100), f"Task{i}") for i in range(100)]
    
    # Linked List Priority Queue
    start_time = time.time()
    for priority, data in test_data:
        ll_pq.enqueue(priority, data)
    ll_enqueue_time = time.time() - start_time
    
    start_time = time.time()
    while not ll_pq.is_empty():
        ll_pq.dequeue()
    ll_dequeue_time = time.time() - start_time
    
    # Heap-based Priority Queue
    start_time = time.time()
    for priority, data in test_data:
        heap_pq.enqueue(priority, data)
    heap_enqueue_time = time.time() - start_time
    
    start_time = time.time()
    while not heap_pq.is_empty():
        heap_pq.dequeue()
    heap_dequeue_time = time.time() - start_time
    
    print(f"Linked List PQ - Enqueue: {ll_enqueue_time:.6f}s, Dequeue: {ll_dequeue_time:.6f}s")
    print(f"Heap-based PQ - Enqueue: {heap_enqueue_time:.6f}s, Dequeue: {heap_dequeue_time:.6f}s")
    print(f"Heap is ~{ll_enqueue_time/heap_enqueue_time:.1f}x faster for enqueue")
    print(f"Heap is ~{ll_dequeue_time/heap_dequeue_time:.1f}x faster for dequeue")
    
    # Memory usage comparison
    print(f"\nMemory Usage Analysis:")
    print(f"Stack: O(n) - one pointer per element")
    print(f"Queue: O(n) - one pointer per element")
    print(f"Priority Queue (LL): O(n) - one pointer + priority per element")
    print(f"Priority Queue (Heap): O(n) - array-based, better cache locality")
    print(f"Deque: O(n) - two pointers per element")
    
    # Use case recommendations
    print(f"\nUse Case Recommendations:")
    print(f"Stack LL: Simple LIFO operations, recursive algorithms")
    print(f"Queue LL: FIFO operations, BFS, producer-consumer")
    print(f"Priority Queue LL: Small datasets, simple implementation")
    print(f"Priority Queue Heap: Large datasets, performance critical")
    print(f"Circular Queue: Fixed-size buffer, ring buffer applications")
    print(f"Deque: Both ends operations, sliding window algorithms")
    
    print("\n=== Demo Complete ===") 