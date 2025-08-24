"""
Custom Stack Implementation - Multiple Approaches
Difficulty: Easy

Implement a custom stack data structure with various optimization techniques and features.
Focus on memory efficiency, performance optimization, and advanced functionality.
"""

from typing import Any, Optional, List, Iterator, Generic, TypeVar
import threading
from collections import deque
import sys

T = TypeVar('T')

class BasicStack:
    """
    Approach 1: Basic Array-based Stack
    
    Simple implementation using Python list.
    
    Time: O(1) amortized for push/pop, Space: O(n)
    """
    
    def __init__(self, capacity: Optional[int] = None):
        self._items = []
        self._capacity = capacity
    
    def push(self, item: Any) -> None:
        """Push item onto stack"""
        if self._capacity and len(self._items) >= self._capacity:
            raise OverflowError("Stack overflow")
        self._items.append(item)
    
    def pop(self) -> Any:
        """Pop item from stack"""
        if self.is_empty():
            raise IndexError("Stack underflow")
        return self._items.pop()
    
    def peek(self) -> Any:
        """Peek at top item without removing"""
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self._items[-1]
    
    def is_empty(self) -> bool:
        """Check if stack is empty"""
        return len(self._items) == 0
    
    def size(self) -> int:
        """Get stack size"""
        return len(self._items)
    
    def clear(self) -> None:
        """Clear all items"""
        self._items.clear()


class LinkedListStack:
    """
    Approach 2: Linked List-based Stack
    
    Implementation using linked list for memory efficiency.
    
    Time: O(1) for all operations, Space: O(n)
    """
    
    class Node:
        def __init__(self, data: Any, next_node: Optional['Node'] = None):
            self.data = data
            self.next = next_node
    
    def __init__(self):
        self._top = None
        self._size = 0
    
    def push(self, item: Any) -> None:
        """Push item onto stack"""
        new_node = self.Node(item, self._top)
        self._top = new_node
        self._size += 1
    
    def pop(self) -> Any:
        """Pop item from stack"""
        if self.is_empty():
            raise IndexError("Stack underflow")
        
        data = self._top.data
        self._top = self._top.next
        self._size -= 1
        return data
    
    def peek(self) -> Any:
        """Peek at top item"""
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self._top.data
    
    def is_empty(self) -> bool:
        """Check if stack is empty"""
        return self._top is None
    
    def size(self) -> int:
        """Get stack size"""
        return self._size
    
    def clear(self) -> None:
        """Clear all items"""
        self._top = None
        self._size = 0


class MemoryOptimizedStack:
    """
    Approach 3: Memory-Optimized Stack
    
    Use __slots__ and memory pooling for optimization.
    
    Time: O(1) for operations, Space: O(n) with reduced overhead
    """
    
    __slots__ = ['_items', '_capacity', '_size']
    
    def __init__(self, capacity: int = 1000):
        self._items = [None] * capacity
        self._capacity = capacity
        self._size = 0
    
    def push(self, item: Any) -> None:
        """Push item onto stack"""
        if self._size >= self._capacity:
            self._resize()
        
        self._items[self._size] = item
        self._size += 1
    
    def pop(self) -> Any:
        """Pop item from stack"""
        if self.is_empty():
            raise IndexError("Stack underflow")
        
        self._size -= 1
        item = self._items[self._size]
        self._items[self._size] = None  # Help GC
        return item
    
    def peek(self) -> Any:
        """Peek at top item"""
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self._items[self._size - 1]
    
    def is_empty(self) -> bool:
        """Check if stack is empty"""
        return self._size == 0
    
    def size(self) -> int:
        """Get stack size"""
        return self._size
    
    def _resize(self) -> None:
        """Resize internal array"""
        old_capacity = self._capacity
        self._capacity *= 2
        new_items = [None] * self._capacity
        
        for i in range(old_capacity):
            new_items[i] = self._items[i]
        
        self._items = new_items


class MinMaxStack:
    """
    Approach 4: Min/Max Stack
    
    Stack that tracks minimum and maximum elements efficiently.
    
    Time: O(1) for all operations, Space: O(n)
    """
    
    def __init__(self):
        self._items = []
        self._min_stack = []
        self._max_stack = []
    
    def push(self, item: Any) -> None:
        """Push item and update min/max"""
        self._items.append(item)
        
        # Update min stack
        if not self._min_stack or item <= self._min_stack[-1]:
            self._min_stack.append(item)
        
        # Update max stack
        if not self._max_stack or item >= self._max_stack[-1]:
            self._max_stack.append(item)
    
    def pop(self) -> Any:
        """Pop item and update min/max"""
        if self.is_empty():
            raise IndexError("Stack underflow")
        
        item = self._items.pop()
        
        # Update min stack
        if item == self._min_stack[-1]:
            self._min_stack.pop()
        
        # Update max stack
        if item == self._max_stack[-1]:
            self._max_stack.pop()
        
        return item
    
    def peek(self) -> Any:
        """Peek at top item"""
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self._items[-1]
    
    def get_min(self) -> Any:
        """Get minimum element in O(1)"""
        if not self._min_stack:
            raise IndexError("Stack is empty")
        return self._min_stack[-1]
    
    def get_max(self) -> Any:
        """Get maximum element in O(1)"""
        if not self._max_stack:
            raise IndexError("Stack is empty")
        return self._max_stack[-1]
    
    def is_empty(self) -> bool:
        """Check if stack is empty"""
        return len(self._items) == 0
    
    def size(self) -> int:
        """Get stack size"""
        return len(self._items)


class ThreadSafeStack:
    """
    Approach 5: Thread-Safe Stack
    
    Stack implementation with thread safety using locks.
    
    Time: O(1) for operations, Space: O(n)
    """
    
    def __init__(self, capacity: Optional[int] = None):
        self._items = []
        self._capacity = capacity
        self._lock = threading.RLock()
    
    def push(self, item: Any) -> None:
        """Thread-safe push"""
        with self._lock:
            if self._capacity and len(self._items) >= self._capacity:
                raise OverflowError("Stack overflow")
            self._items.append(item)
    
    def pop(self) -> Any:
        """Thread-safe pop"""
        with self._lock:
            if self.is_empty():
                raise IndexError("Stack underflow")
            return self._items.pop()
    
    def peek(self) -> Any:
        """Thread-safe peek"""
        with self._lock:
            if self.is_empty():
                raise IndexError("Stack is empty")
            return self._items[-1]
    
    def is_empty(self) -> bool:
        """Thread-safe empty check"""
        with self._lock:
            return len(self._items) == 0
    
    def size(self) -> int:
        """Thread-safe size check"""
        with self._lock:
            return len(self._items)


class GenericStack(Generic[T]):
    """
    Approach 6: Generic Type-Safe Stack
    
    Type-safe stack implementation with generics.
    
    Time: O(1) for operations, Space: O(n)
    """
    
    def __init__(self, capacity: Optional[int] = None):
        self._items: List[T] = []
        self._capacity = capacity
    
    def push(self, item: T) -> None:
        """Push typed item"""
        if self._capacity and len(self._items) >= self._capacity:
            raise OverflowError("Stack overflow")
        self._items.append(item)
    
    def pop(self) -> T:
        """Pop typed item"""
        if self.is_empty():
            raise IndexError("Stack underflow")
        return self._items.pop()
    
    def peek(self) -> T:
        """Peek at typed item"""
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self._items[-1]
    
    def is_empty(self) -> bool:
        """Check if empty"""
        return len(self._items) == 0
    
    def size(self) -> int:
        """Get size"""
        return len(self._items)
    
    def __iter__(self) -> Iterator[T]:
        """Make stack iterable"""
        return reversed(self._items)


def test_stack_implementations():
    """Test all stack implementations"""
    
    implementations = [
        ("Basic Stack", BasicStack),
        ("LinkedList Stack", LinkedListStack),
        ("Memory Optimized", MemoryOptimizedStack),
        ("MinMax Stack", MinMaxStack),
        ("Thread Safe", ThreadSafeStack),
        ("Generic Stack", GenericStack),
    ]
    
    print("=== Testing Stack Implementations ===")
    
    for name, stack_class in implementations:
        print(f"\n--- {name} ---")
        
        try:
            # Create stack instance
            if name == "Memory Optimized":
                stack = stack_class(10)
            else:
                stack = stack_class()
            
            # Test basic operations
            print("Testing basic operations:")
            
            # Push operations
            for i in range(5):
                stack.push(i)
                print(f"  Pushed {i}, size: {stack.size()}")
            
            # Peek operation
            top = stack.peek()
            print(f"  Top element: {top}")
            
            # Pop operations
            while not stack.is_empty():
                item = stack.pop()
                print(f"  Popped {item}, size: {stack.size()}")
            
            # Test MinMax stack specific features
            if name == "MinMax Stack":
                print("Testing min/max features:")
                for val in [3, 1, 4, 1, 5, 9, 2]:
                    stack.push(val)
                    print(f"  Pushed {val}, min: {stack.get_min()}, max: {stack.get_max()}")
            
            print(f"  {name} tests passed ✓")
            
        except Exception as e:
            print(f"  {name} error: {str(e)[:50]}")


def demonstrate_memory_optimization():
    """Demonstrate memory optimization techniques"""
    print("\n=== Memory Optimization Demonstration ===")
    
    # Compare memory usage
    import sys
    
    # Basic stack
    basic_stack = BasicStack()
    for i in range(1000):
        basic_stack.push(i)
    
    basic_size = sys.getsizeof(basic_stack._items)
    print(f"Basic stack memory usage: {basic_size} bytes")
    
    # Memory optimized stack
    opt_stack = MemoryOptimizedStack(1000)
    for i in range(1000):
        opt_stack.push(i)
    
    opt_size = sys.getsizeof(opt_stack._items)
    print(f"Optimized stack memory usage: {opt_size} bytes")
    
    print(f"Memory savings: {basic_size - opt_size} bytes ({((basic_size - opt_size) / basic_size * 100):.1f}%)")


def demonstrate_thread_safety():
    """Demonstrate thread safety"""
    print("\n=== Thread Safety Demonstration ===")
    
    import threading
    import time
    import random
    
    # Shared thread-safe stack
    safe_stack = ThreadSafeStack()
    results = []
    
    def producer(stack, thread_id):
        """Producer thread"""
        for i in range(10):
            value = thread_id * 100 + i
            stack.push(value)
            results.append(f"Thread {thread_id} pushed {value}")
            time.sleep(random.uniform(0.001, 0.01))
    
    def consumer(stack, thread_id):
        """Consumer thread"""
        for _ in range(5):
            try:
                if not stack.is_empty():
                    value = stack.pop()
                    results.append(f"Thread {thread_id} popped {value}")
                time.sleep(random.uniform(0.001, 0.01))
            except IndexError:
                pass
    
    # Create and start threads
    threads = []
    
    # Producer threads
    for i in range(2):
        t = threading.Thread(target=producer, args=(safe_stack, i))
        threads.append(t)
        t.start()
    
    # Consumer threads
    for i in range(2, 4):
        t = threading.Thread(target=consumer, args=(safe_stack, i))
        threads.append(t)
        t.start()
    
    # Wait for all threads
    for t in threads:
        t.join()
    
    print("Thread operations completed:")
    for result in results[:10]:  # Show first 10 operations
        print(f"  {result}")
    
    print(f"Final stack size: {safe_stack.size()}")


def benchmark_stack_implementations():
    """Benchmark different stack implementations"""
    print("\n=== Performance Benchmark ===")
    
    import time
    
    implementations = [
        ("Basic Stack", BasicStack),
        ("LinkedList Stack", LinkedListStack),
        ("Memory Optimized", MemoryOptimizedStack),
    ]
    
    n_operations = 100000
    
    for name, stack_class in implementations:
        try:
            if name == "Memory Optimized":
                stack = stack_class(n_operations)
            else:
                stack = stack_class()
            
            # Benchmark push operations
            start_time = time.time()
            for i in range(n_operations):
                stack.push(i)
            push_time = time.time() - start_time
            
            # Benchmark pop operations
            start_time = time.time()
            for _ in range(n_operations):
                stack.pop()
            pop_time = time.time() - start_time
            
            print(f"{name:20} | Push: {push_time:.4f}s | Pop: {pop_time:.4f}s")
            
        except Exception as e:
            print(f"{name:20} | Error: {str(e)[:30]}")


def demonstrate_advanced_features():
    """Demonstrate advanced stack features"""
    print("\n=== Advanced Features Demonstration ===")
    
    # Generic stack with type safety
    print("1. Generic Type-Safe Stack:")
    int_stack: GenericStack[int] = GenericStack()
    
    for i in range(5):
        int_stack.push(i)
    
    print(f"   Integer stack: {list(int_stack)}")
    
    # String stack
    str_stack: GenericStack[str] = GenericStack()
    for word in ["hello", "world", "stack"]:
        str_stack.push(word)
    
    print(f"   String stack: {list(str_stack)}")
    
    # MinMax stack demonstration
    print("\n2. MinMax Stack Features:")
    minmax_stack = MinMaxStack()
    
    values = [3, 1, 4, 1, 5, 9, 2, 6]
    for val in values:
        minmax_stack.push(val)
        print(f"   After pushing {val}: min={minmax_stack.get_min()}, max={minmax_stack.get_max()}")


def test_edge_cases():
    """Test edge cases for stack implementations"""
    print("\n=== Testing Edge Cases ===")
    
    stack = BasicStack()
    
    edge_cases = [
        ("Empty stack pop", lambda: stack.pop(), IndexError),
        ("Empty stack peek", lambda: stack.peek(), IndexError),
        ("Push then pop", lambda: (stack.push(1), stack.pop())[1], None),
        ("Multiple operations", lambda: test_multiple_ops(stack), None),
    ]
    
    def test_multiple_ops(s):
        s.clear()
        for i in range(3):
            s.push(i)
        return [s.pop() for _ in range(3)]
    
    for description, operation, expected_exception in edge_cases:
        try:
            result = operation()
            if expected_exception:
                print(f"{description:20} | ✗ Expected {expected_exception.__name__}")
            else:
                print(f"{description:20} | ✓ Result: {result}")
        except Exception as e:
            if expected_exception and isinstance(e, expected_exception):
                print(f"{description:20} | ✓ Correctly raised {type(e).__name__}")
            else:
                print(f"{description:20} | ✗ Unexpected error: {type(e).__name__}")


def analyze_time_complexity():
    """Analyze time complexity of different implementations"""
    print("\n=== Time Complexity Analysis ===")
    
    implementations = [
        ("Basic Stack", "O(1)*", "O(1)", "O(n)", "Amortized due to dynamic array"),
        ("LinkedList Stack", "O(1)", "O(1)", "O(n)", "True O(1) operations"),
        ("Memory Optimized", "O(1)*", "O(1)", "O(n)", "Amortized with pre-allocation"),
        ("MinMax Stack", "O(1)", "O(1)", "O(n)", "Additional min/max tracking"),
        ("Thread Safe", "O(1)", "O(1)", "O(n)", "Lock overhead negligible"),
        ("Generic Stack", "O(1)*", "O(1)", "O(n)", "Type safety with same performance"),
    ]
    
    print(f"{'Implementation':<20} | {'Push':<8} | {'Pop':<8} | {'Space':<8} | {'Notes'}")
    print("-" * 80)
    
    for impl, push, pop, space, notes in implementations:
        print(f"{impl:<20} | {push:<8} | {pop:<8} | {space:<8} | {notes}")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    # Application 1: Undo/Redo system
    print("1. Undo/Redo System:")
    
    class UndoRedoSystem:
        def __init__(self):
            self.undo_stack = BasicStack()
            self.redo_stack = BasicStack()
        
        def execute_command(self, command):
            self.undo_stack.push(command)
            self.redo_stack.clear()  # Clear redo stack on new command
            print(f"   Executed: {command}")
        
        def undo(self):
            if not self.undo_stack.is_empty():
                command = self.undo_stack.pop()
                self.redo_stack.push(command)
                print(f"   Undid: {command}")
            else:
                print("   Nothing to undo")
        
        def redo(self):
            if not self.redo_stack.is_empty():
                command = self.redo_stack.pop()
                self.undo_stack.push(command)
                print(f"   Redid: {command}")
            else:
                print("   Nothing to redo")
    
    system = UndoRedoSystem()
    system.execute_command("Type 'Hello'")
    system.execute_command("Type ' World'")
    system.undo()
    system.undo()
    system.redo()
    
    # Application 2: Expression evaluator
    print(f"\n2. Expression Evaluator:")
    
    def evaluate_postfix(expression):
        stack = BasicStack()
        
        for token in expression.split():
            if token in ['+', '-', '*', '/']:
                b = stack.pop()
                a = stack.pop()
                
                if token == '+':
                    result = a + b
                elif token == '-':
                    result = a - b
                elif token == '*':
                    result = a * b
                elif token == '/':
                    result = a / b
                
                stack.push(result)
                print(f"   {a} {token} {b} = {result}")
            else:
                stack.push(float(token))
                print(f"   Pushed {token}")
        
        return stack.pop()
    
    expr = "3 4 + 2 * 7 /"
    result = evaluate_postfix(expr)
    print(f"   Expression '{expr}' = {result}")


if __name__ == "__main__":
    test_stack_implementations()
    demonstrate_memory_optimization()
    demonstrate_thread_safety()
    benchmark_stack_implementations()
    demonstrate_advanced_features()
    test_edge_cases()
    analyze_time_complexity()
    demonstrate_real_world_applications()

"""
Custom Stack Implementation demonstrates various optimization techniques
including memory efficiency, thread safety, type safety, and advanced
features for high-performance stack data structures.
"""
