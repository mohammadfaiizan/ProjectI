"""
Deque Basic Implementation - Multiple Approaches
Difficulty: Easy

Implement a deque (double-ended queue) data structure with various approaches.
Support insertion and deletion at both ends efficiently.
"""

from typing import Any, Optional, List, Iterator

class ArrayDeque:
    """
    Approach 1: Array-based Deque with Dynamic Resizing
    
    Use dynamic array with front and rear pointers.
    
    Time: O(1) amortized for all operations, Space: O(n)
    """
    
    def __init__(self, capacity: int = 16):
        self._items = [None] * capacity
        self._capacity = capacity
        self._front = 0
        self._rear = 0
        self._size = 0
    
    def push_front(self, item: Any) -> None:
        """Add item to front of deque"""
        if self._size >= self._capacity:
            self._resize()
        
        self._front = (self._front - 1) % self._capacity
        self._items[self._front] = item
        self._size += 1
    
    def push_back(self, item: Any) -> None:
        """Add item to back of deque"""
        if self._size >= self._capacity:
            self._resize()
        
        self._items[self._rear] = item
        self._rear = (self._rear + 1) % self._capacity
        self._size += 1
    
    def pop_front(self) -> Any:
        """Remove and return item from front"""
        if self._size == 0:
            raise IndexError("Deque is empty")
        
        item = self._items[self._front]
        self._items[self._front] = None  # Help GC
        self._front = (self._front + 1) % self._capacity
        self._size -= 1
        return item
    
    def pop_back(self) -> Any:
        """Remove and return item from back"""
        if self._size == 0:
            raise IndexError("Deque is empty")
        
        self._rear = (self._rear - 1) % self._capacity
        item = self._items[self._rear]
        self._items[self._rear] = None  # Help GC
        self._size -= 1
        return item
    
    def front(self) -> Any:
        """Peek at front item"""
        if self._size == 0:
            raise IndexError("Deque is empty")
        return self._items[self._front]
    
    def back(self) -> Any:
        """Peek at back item"""
        if self._size == 0:
            raise IndexError("Deque is empty")
        rear_index = (self._rear - 1) % self._capacity
        return self._items[rear_index]
    
    def is_empty(self) -> bool:
        """Check if deque is empty"""
        return self._size == 0
    
    def size(self) -> int:
        """Get deque size"""
        return self._size
    
    def _resize(self) -> None:
        """Resize internal array"""
        old_capacity = self._capacity
        self._capacity *= 2
        new_items = [None] * self._capacity
        
        # Copy items in order
        for i in range(self._size):
            old_index = (self._front + i) % old_capacity
            new_items[i] = self._items[old_index]
        
        self._items = new_items
        self._front = 0
        self._rear = self._size


class LinkedListDeque:
    """
    Approach 2: Doubly Linked List Deque
    
    Use doubly linked list for true O(1) operations.
    
    Time: O(1) for all operations, Space: O(n)
    """
    
    class Node:
        def __init__(self, data: Any, prev_node: Optional['Node'] = None, 
                     next_node: Optional['Node'] = None):
            self.data = data
            self.prev = prev_node
            self.next = next_node
    
    def __init__(self):
        # Sentinel nodes for easier implementation
        self._head = self.Node(None)
        self._tail = self.Node(None)
        self._head.next = self._tail
        self._tail.prev = self._head
        self._size = 0
    
    def push_front(self, item: Any) -> None:
        """Add item to front"""
        new_node = self.Node(item, self._head, self._head.next)
        self._head.next.prev = new_node
        self._head.next = new_node
        self._size += 1
    
    def push_back(self, item: Any) -> None:
        """Add item to back"""
        new_node = self.Node(item, self._tail.prev, self._tail)
        self._tail.prev.next = new_node
        self._tail.prev = new_node
        self._size += 1
    
    def pop_front(self) -> Any:
        """Remove and return item from front"""
        if self._size == 0:
            raise IndexError("Deque is empty")
        
        node = self._head.next
        data = node.data
        
        self._head.next = node.next
        node.next.prev = self._head
        
        self._size -= 1
        return data
    
    def pop_back(self) -> Any:
        """Remove and return item from back"""
        if self._size == 0:
            raise IndexError("Deque is empty")
        
        node = self._tail.prev
        data = node.data
        
        self._tail.prev = node.prev
        node.prev.next = self._tail
        
        self._size -= 1
        return data
    
    def front(self) -> Any:
        """Peek at front item"""
        if self._size == 0:
            raise IndexError("Deque is empty")
        return self._head.next.data
    
    def back(self) -> Any:
        """Peek at back item"""
        if self._size == 0:
            raise IndexError("Deque is empty")
        return self._tail.prev.data
    
    def is_empty(self) -> bool:
        """Check if deque is empty"""
        return self._size == 0
    
    def size(self) -> int:
        """Get deque size"""
        return self._size


class TwoStackDeque:
    """
    Approach 3: Two-Stack Implementation
    
    Use two stacks to implement deque operations.
    
    Time: O(1) amortized, Space: O(n)
    """
    
    def __init__(self):
        self._front_stack = []  # For front operations
        self._back_stack = []   # For back operations
    
    def push_front(self, item: Any) -> None:
        """Add item to front"""
        self._front_stack.append(item)
    
    def push_back(self, item: Any) -> None:
        """Add item to back"""
        self._back_stack.append(item)
    
    def pop_front(self) -> Any:
        """Remove and return item from front"""
        if self._front_stack:
            return self._front_stack.pop()
        elif self._back_stack:
            # Transfer half of back stack to front stack
            self._rebalance()
            if self._front_stack:
                return self._front_stack.pop()
            else:
                return self._back_stack.pop(0)
        else:
            raise IndexError("Deque is empty")
    
    def pop_back(self) -> Any:
        """Remove and return item from back"""
        if self._back_stack:
            return self._back_stack.pop()
        elif self._front_stack:
            # Transfer half of front stack to back stack
            self._rebalance()
            if self._back_stack:
                return self._back_stack.pop()
            else:
                return self._front_stack.pop(0)
        else:
            raise IndexError("Deque is empty")
    
    def front(self) -> Any:
        """Peek at front item"""
        if self._front_stack:
            return self._front_stack[-1]
        elif self._back_stack:
            return self._back_stack[0]
        else:
            raise IndexError("Deque is empty")
    
    def back(self) -> Any:
        """Peek at back item"""
        if self._back_stack:
            return self._back_stack[-1]
        elif self._front_stack:
            return self._front_stack[0]
        else:
            raise IndexError("Deque is empty")
    
    def is_empty(self) -> bool:
        """Check if deque is empty"""
        return len(self._front_stack) == 0 and len(self._back_stack) == 0
    
    def size(self) -> int:
        """Get deque size"""
        return len(self._front_stack) + len(self._back_stack)
    
    def _rebalance(self) -> None:
        """Rebalance stacks for better performance"""
        if not self._front_stack and self._back_stack:
            # Move half of back stack to front stack (reversed)
            mid = len(self._back_stack) // 2
            self._front_stack = self._back_stack[:mid][::-1]
            self._back_stack = self._back_stack[mid:]
        elif not self._back_stack and self._front_stack:
            # Move half of front stack to back stack (reversed)
            mid = len(self._front_stack) // 2
            self._back_stack = self._front_stack[:mid][::-1]
            self._front_stack = self._front_stack[mid:]


class CircularBufferDeque:
    """
    Approach 4: Fixed-Size Circular Buffer Deque
    
    Use fixed-size circular buffer for memory-efficient deque.
    
    Time: O(1) for all operations, Space: O(capacity)
    """
    
    def __init__(self, capacity: int):
        self._items = [None] * capacity
        self._capacity = capacity
        self._front = 0
        self._rear = 0
        self._size = 0
    
    def push_front(self, item: Any) -> None:
        """Add item to front"""
        if self._size >= self._capacity:
            raise OverflowError("Deque is full")
        
        self._front = (self._front - 1) % self._capacity
        self._items[self._front] = item
        self._size += 1
    
    def push_back(self, item: Any) -> None:
        """Add item to back"""
        if self._size >= self._capacity:
            raise OverflowError("Deque is full")
        
        self._items[self._rear] = item
        self._rear = (self._rear + 1) % self._capacity
        self._size += 1
    
    def pop_front(self) -> Any:
        """Remove and return item from front"""
        if self._size == 0:
            raise IndexError("Deque is empty")
        
        item = self._items[self._front]
        self._items[self._front] = None
        self._front = (self._front + 1) % self._capacity
        self._size -= 1
        return item
    
    def pop_back(self) -> Any:
        """Remove and return item from back"""
        if self._size == 0:
            raise IndexError("Deque is empty")
        
        self._rear = (self._rear - 1) % self._capacity
        item = self._items[self._rear]
        self._items[self._rear] = None
        self._size -= 1
        return item
    
    def front(self) -> Any:
        """Peek at front item"""
        if self._size == 0:
            raise IndexError("Deque is empty")
        return self._items[self._front]
    
    def back(self) -> Any:
        """Peek at back item"""
        if self._size == 0:
            raise IndexError("Deque is empty")
        back_index = (self._rear - 1) % self._capacity
        return self._items[back_index]
    
    def is_empty(self) -> bool:
        """Check if deque is empty"""
        return self._size == 0
    
    def is_full(self) -> bool:
        """Check if deque is full"""
        return self._size >= self._capacity
    
    def size(self) -> int:
        """Get deque size"""
        return self._size


class IterableDeque:
    """
    Approach 5: Iterable Deque with Python Collections Integration
    
    Deque that integrates well with Python's iteration protocols.
    
    Time: O(1) for operations, O(n) for iteration, Space: O(n)
    """
    
    def __init__(self):
        from collections import deque
        self._items = deque()
    
    def push_front(self, item: Any) -> None:
        """Add item to front"""
        self._items.appendleft(item)
    
    def push_back(self, item: Any) -> None:
        """Add item to back"""
        self._items.append(item)
    
    def pop_front(self) -> Any:
        """Remove and return item from front"""
        if not self._items:
            raise IndexError("Deque is empty")
        return self._items.popleft()
    
    def pop_back(self) -> Any:
        """Remove and return item from back"""
        if not self._items:
            raise IndexError("Deque is empty")
        return self._items.pop()
    
    def front(self) -> Any:
        """Peek at front item"""
        if not self._items:
            raise IndexError("Deque is empty")
        return self._items[0]
    
    def back(self) -> Any:
        """Peek at back item"""
        if not self._items:
            raise IndexError("Deque is empty")
        return self._items[-1]
    
    def is_empty(self) -> bool:
        """Check if deque is empty"""
        return len(self._items) == 0
    
    def size(self) -> int:
        """Get deque size"""
        return len(self._items)
    
    def __iter__(self) -> Iterator[Any]:
        """Make deque iterable"""
        return iter(self._items)
    
    def __len__(self) -> int:
        """Support len() function"""
        return len(self._items)
    
    def __str__(self) -> str:
        """String representation"""
        return f"Deque({list(self._items)})"


def test_deque_implementations():
    """Test all deque implementations"""
    
    implementations = [
        ("Array Deque", ArrayDeque),
        ("LinkedList Deque", LinkedListDeque),
        ("Two Stack Deque", TwoStackDeque),
        ("Circular Buffer", lambda: CircularBufferDeque(10)),
        ("Iterable Deque", IterableDeque),
    ]
    
    print("=== Testing Deque Implementations ===")
    
    for name, deque_factory in implementations:
        print(f"\n--- {name} ---")
        
        try:
            deque = deque_factory()
            
            # Test basic operations
            print("Testing basic operations:")
            
            # Push operations
            deque.push_front("front1")
            deque.push_back("back1")
            deque.push_front("front2")
            deque.push_back("back2")
            
            print(f"  After pushes: size = {deque.size()}")
            print(f"  Front: {deque.front()}, Back: {deque.back()}")
            
            # Pop operations
            front_item = deque.pop_front()
            back_item = deque.pop_back()
            
            print(f"  Popped front: {front_item}, back: {back_item}")
            print(f"  After pops: size = {deque.size()}")
            print(f"  Front: {deque.front()}, Back: {deque.back()}")
            
            print(f"  {name} tests passed ✓")
            
        except Exception as e:
            print(f"  {name} error: {str(e)[:50]}")


def demonstrate_deque_operations():
    """Demonstrate deque operations step by step"""
    print("\n=== Deque Operations Demonstration ===")
    
    deque = ArrayDeque()
    
    operations = [
        ("push_front", "A"),
        ("push_back", "B"),
        ("push_front", "C"),
        ("push_back", "D"),
        ("pop_front", None),
        ("push_front", "E"),
        ("pop_back", None),
        ("push_back", "F"),
    ]
    
    print("Step-by-step deque operations:")
    
    for i, (operation, value) in enumerate(operations):
        print(f"\nStep {i+1}: {operation}" + (f"({value})" if value else "()"))
        
        if operation == "push_front":
            deque.push_front(value)
        elif operation == "push_back":
            deque.push_back(value)
        elif operation == "pop_front":
            if not deque.is_empty():
                value = deque.pop_front()
                print(f"  Popped: {value}")
        elif operation == "pop_back":
            if not deque.is_empty():
                value = deque.pop_back()
                print(f"  Popped: {value}")
        
        if not deque.is_empty():
            print(f"  Front: {deque.front()}, Back: {deque.back()}, Size: {deque.size()}")
        else:
            print(f"  Deque is empty")


def analyze_time_complexity():
    """Analyze time complexity of different implementations"""
    print("\n=== Time Complexity Analysis ===")
    
    implementations = [
        ("Array Deque", "O(1)*", "O(1)*", "O(1)", "O(1)", "Amortized due to resizing"),
        ("LinkedList Deque", "O(1)", "O(1)", "O(1)", "O(1)", "True O(1) operations"),
        ("Two Stack Deque", "O(1)*", "O(1)*", "O(1)*", "O(1)*", "Amortized due to rebalancing"),
        ("Circular Buffer", "O(1)", "O(1)", "O(1)", "O(1)", "Fixed capacity"),
        ("Iterable Deque", "O(1)", "O(1)", "O(1)", "O(1)", "Built on collections.deque"),
    ]
    
    print(f"{'Implementation':<20} | {'Push Front':<12} | {'Push Back':<12} | {'Pop Front':<12} | {'Pop Back':<12} | {'Notes'}")
    print("-" * 100)
    
    for impl, push_front, push_back, pop_front, pop_back, notes in implementations:
        print(f"{impl:<20} | {push_front:<12} | {push_back:<12} | {pop_front:<12} | {pop_back:<12} | {notes}")


def test_edge_cases():
    """Test edge cases for deque implementations"""
    print("\n=== Testing Edge Cases ===")
    
    deque = LinkedListDeque()
    
    edge_cases = [
        ("Empty deque pop_front", lambda: deque.pop_front(), IndexError),
        ("Empty deque pop_back", lambda: deque.pop_back(), IndexError),
        ("Empty deque front", lambda: deque.front(), IndexError),
        ("Empty deque back", lambda: deque.back(), IndexError),
        ("Single item operations", lambda: test_single_item(deque), None),
        ("Alternating operations", lambda: test_alternating(deque), None),
    ]
    
    def test_single_item(d):
        d.push_front("item")
        front = d.front()
        back = d.back()
        popped = d.pop_back()
        return front == back == popped == "item"
    
    def test_alternating(d):
        for i in range(5):
            d.push_front(f"f{i}")
            d.push_back(f"b{i}")
        
        results = []
        while not d.is_empty():
            if d.size() % 2 == 0:
                results.append(d.pop_front())
            else:
                results.append(d.pop_back())
        
        return len(results) == 10
    
    for description, operation, expected_exception in edge_cases:
        try:
            result = operation()
            if expected_exception:
                print(f"{description:25} | ✗ Expected {expected_exception.__name__}")
            else:
                print(f"{description:25} | ✓ Result: {result}")
        except Exception as e:
            if expected_exception and isinstance(e, expected_exception):
                print(f"{description:25} | ✓ Correctly raised {type(e).__name__}")
            else:
                print(f"{description:25} | ✗ Unexpected error: {type(e).__name__}")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications of deques"""
    print("\n=== Real-World Applications ===")
    
    # Application 1: Browser history with back/forward
    print("1. Browser History Management:")
    
    class BrowserHistory:
        def __init__(self):
            self.history = LinkedListDeque()
            self.current_page = None
        
        def visit_page(self, url: str):
            if self.current_page:
                self.history.push_back(self.current_page)
            self.current_page = url
            print(f"   Visited: {url}")
        
        def go_back(self):
            if not self.history.is_empty():
                self.history.push_front(self.current_page)
                self.current_page = self.history.pop_back()
                print(f"   Went back to: {self.current_page}")
            else:
                print("   No history to go back to")
        
        def go_forward(self):
            if not self.history.is_empty():
                self.history.push_back(self.current_page)
                self.current_page = self.history.pop_front()
                print(f"   Went forward to: {self.current_page}")
            else:
                print("   No forward history")
    
    browser = BrowserHistory()
    browser.visit_page("google.com")
    browser.visit_page("github.com")
    browser.visit_page("stackoverflow.com")
    browser.go_back()
    browser.go_back()
    browser.go_forward()
    
    # Application 2: Sliding window maximum
    print(f"\n2. Sliding Window Maximum:")
    
    def sliding_window_maximum(arr: List[int], k: int) -> List[int]:
        """Find maximum in each sliding window using deque"""
        deque = ArrayDeque()
        result = []
        
        for i in range(len(arr)):
            # Remove elements outside current window
            while not deque.is_empty() and deque.front() <= i - k:
                deque.pop_front()
            
            # Remove smaller elements from back
            while not deque.is_empty() and arr[deque.back()] <= arr[i]:
                deque.pop_back()
            
            deque.push_back(i)
            
            # Add maximum to result if window is complete
            if i >= k - 1:
                result.append(arr[deque.front()])
        
        return result
    
    arr = [1, 3, -1, -3, 5, 3, 6, 7]
    k = 3
    maximums = sliding_window_maximum(arr, k)
    print(f"   Array: {arr}")
    print(f"   Window size: {k}")
    print(f"   Sliding window maximums: {maximums}")
    
    # Application 3: Palindrome checker
    print(f"\n3. Palindrome Checker:")
    
    def is_palindrome(s: str) -> bool:
        """Check if string is palindrome using deque"""
        deque = IterableDeque()
        
        # Add only alphanumeric characters
        for char in s.lower():
            if char.isalnum():
                deque.push_back(char)
        
        # Compare from both ends
        while deque.size() > 1:
            if deque.pop_front() != deque.pop_back():
                return False
        
        return True
    
    test_strings = ["racecar", "A man a plan a canal Panama", "race a car", "hello"]
    
    for test_str in test_strings:
        result = is_palindrome(test_str)
        print(f"   '{test_str}' -> {'Palindrome' if result else 'Not palindrome'}")


def benchmark_deque_implementations():
    """Benchmark different deque implementations"""
    print("\n=== Performance Benchmark ===")
    
    import time
    
    implementations = [
        ("Array Deque", ArrayDeque),
        ("LinkedList Deque", LinkedListDeque),
        ("Iterable Deque", IterableDeque),
    ]
    
    n_operations = 10000
    
    for name, deque_class in implementations:
        try:
            deque = deque_class()
            
            # Benchmark mixed operations
            start_time = time.time()
            
            for i in range(n_operations // 4):
                deque.push_front(i)
                deque.push_back(i)
                
                if i % 10 == 0 and not deque.is_empty():
                    deque.pop_front()
                    deque.pop_back()
            
            end_time = time.time()
            
            print(f"{name:20} | Time: {end_time - start_time:.4f}s | Final size: {deque.size()}")
            
        except Exception as e:
            print(f"{name:20} | Error: {str(e)[:30]}")


if __name__ == "__main__":
    test_deque_implementations()
    demonstrate_deque_operations()
    analyze_time_complexity()
    test_edge_cases()
    demonstrate_real_world_applications()
    benchmark_deque_implementations()

"""
Deque Basic Implementation demonstrates various approaches to implementing
double-ended queues including array-based, linked list, two-stack, circular
buffer, and iterable implementations with comprehensive testing and analysis.
"""