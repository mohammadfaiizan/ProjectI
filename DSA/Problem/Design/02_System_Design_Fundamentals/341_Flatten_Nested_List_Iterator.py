"""
341. Flatten Nested List Iterator - Multiple Approaches
Difficulty: Medium

You are given a nested list of integers nestedList. Each element is either an integer or a list whose elements may also be integers or other lists. Implement an iterator to flatten it.

Implement the NestedIterator class:
- NestedIterator(List<NestedInteger> nestedList) Initializes the iterator with the nested list nestedList.
- int next() Returns the next integer in the nested list.
- boolean hasNext() Returns true if there are still some integers in the nested list.

Note: Your code will be tested with the inputs and expected outputs shown below. You must not deserialize the nested list given to you. Your algorithm should operate on the NestedInteger interface:

interface NestedInteger {
    public boolean isInteger()
    public Integer getInteger()
    public List<NestedInteger> getList()
}
"""

from typing import List, Optional, Iterator as IteratorType
from collections import deque

class NestedInteger:
    """Mock NestedInteger class for demonstration"""
    def __init__(self, value=None):
        if isinstance(value, int):
            self._integer = value
            self._list = None
        else:
            self._integer = None
            self._list = value or []
    
    def isInteger(self) -> bool:
        return self._integer is not None
    
    def getInteger(self) -> int:
        return self._integer
    
    def getList(self) -> List['NestedInteger']:
        return self._list

class NestedIteratorPrecompute:
    """
    Approach 1: Precompute Flattened List
    
    Flatten the entire nested structure during initialization.
    
    Time Complexity: 
    - Constructor: O(n) where n is total elements
    - hasNext(), next(): O(1)
    
    Space Complexity: O(n)
    """
    
    def __init__(self, nestedList: List[NestedInteger]):
        self.flattened = []
        self.index = 0
        self._flatten(nestedList)
    
    def _flatten(self, nested_list: List[NestedInteger]) -> None:
        for nested in nested_list:
            if nested.isInteger():
                self.flattened.append(nested.getInteger())
            else:
                self._flatten(nested.getList())
    
    def next(self) -> int:
        if self.hasNext():
            val = self.flattened[self.index]
            self.index += 1
            return val
        return -1
    
    def hasNext(self) -> bool:
        return self.index < len(self.flattened)

class NestedIteratorStack:
    """
    Approach 2: Stack-based Lazy Evaluation
    
    Use stack to maintain current position in nested structure.
    
    Time Complexity: 
    - Constructor: O(1)
    - hasNext(): O(d) where d is max depth
    - next(): O(1) amortized
    
    Space Complexity: O(d) where d is max depth
    """
    
    def __init__(self, nestedList: List[NestedInteger]):
        # Stack stores (list, index) pairs
        self.stack = [(nestedList, 0)]
    
    def next(self) -> int:
        if not self.hasNext():
            return -1
        
        # Get current position
        nested_list, index = self.stack[-1]
        
        # Move to next position
        self.stack[-1] = (nested_list, index + 1)
        
        return nested_list[index].getInteger()
    
    def hasNext(self) -> bool:
        while self.stack:
            nested_list, index = self.stack[-1]
            
            if index >= len(nested_list):
                # Current list is exhausted
                self.stack.pop()
                continue
            
            nested = nested_list[index]
            
            if nested.isInteger():
                return True
            else:
                # Replace current position with the nested list
                self.stack[-1] = (nested_list, index + 1)
                self.stack.append((nested.getList(), 0))
        
        return False

class NestedIteratorQueue:
    """
    Approach 3: Queue-based with Eager Expansion
    
    Use queue and expand nested lists as encountered.
    
    Time Complexity: 
    - Constructor: O(1)
    - hasNext(): O(1)
    - next(): O(d) where d is depth of next element
    
    Space Complexity: O(w) where w is maximum width at any level
    """
    
    def __init__(self, nestedList: List[NestedInteger]):
        self.queue = deque(nestedList)
    
    def next(self) -> int:
        if not self.hasNext():
            return -1
        
        return self.queue.popleft().getInteger()
    
    def hasNext(self) -> bool:
        while self.queue:
            if self.queue[0].isInteger():
                return True
            
            # Expand the nested list
            nested_list = self.queue.popleft().getList()
            
            # Add all elements to front of queue
            for i in range(len(nested_list) - 1, -1, -1):
                self.queue.appendleft(nested_list[i])
        
        return False

class NestedIteratorGenerator:
    """
    Approach 4: Python Generator
    
    Use generator for elegant recursive flattening.
    
    Time Complexity: 
    - Constructor: O(1)
    - hasNext(): O(1)
    - next(): O(1) amortized
    
    Space Complexity: O(d) where d is max depth
    """
    
    def __init__(self, nestedList: List[NestedInteger]):
        self.generator = self._flatten_generator(nestedList)
        self.current_value = None
        self.has_current = True
        self._advance()
    
    def _flatten_generator(self, nested_list: List[NestedInteger]):
        for nested in nested_list:
            if nested.isInteger():
                yield nested.getInteger()
            else:
                yield from self._flatten_generator(nested.getList())
    
    def _advance(self) -> None:
        try:
            self.current_value = next(self.generator)
        except StopIteration:
            self.has_current = False
    
    def next(self) -> int:
        if not self.hasNext():
            return -1
        
        val = self.current_value
        self._advance()
        return val
    
    def hasNext(self) -> bool:
        return self.has_current

class NestedIteratorStateful:
    """
    Approach 5: Stateful with Position Tracking
    
    Enhanced iterator with position tracking and debugging.
    
    Time Complexity: 
    - Constructor: O(1)
    - hasNext(): O(d)
    - next(): O(1) amortized
    
    Space Complexity: O(d)
    """
    
    def __init__(self, nestedList: List[NestedInteger]):
        # Stack stores (list, index, depth) tuples
        self.stack = [(nestedList, 0, 0)]
        self.total_elements = 0
        self.current_position = 0
        self.max_depth_seen = 0
    
    def next(self) -> int:
        if not self.hasNext():
            return -1
        
        # Get current position
        nested_list, index, depth = self.stack[-1]
        
        # Move to next position
        self.stack[-1] = (nested_list, index + 1, depth)
        
        self.current_position += 1
        return nested_list[index].getInteger()
    
    def hasNext(self) -> bool:
        while self.stack:
            nested_list, index, depth = self.stack[-1]
            
            # Update max depth seen
            self.max_depth_seen = max(self.max_depth_seen, depth)
            
            if index >= len(nested_list):
                # Current list is exhausted
                self.stack.pop()
                continue
            
            nested = nested_list[index]
            
            if nested.isInteger():
                return True
            else:
                # Replace current position with the nested list
                self.stack[-1] = (nested_list, index + 1, depth)
                self.stack.append((nested.getList(), 0, depth + 1))
        
        return False
    
    def get_current_depth(self) -> int:
        """Get current nesting depth"""
        return len(self.stack)
    
    def get_max_depth_seen(self) -> int:
        """Get maximum depth encountered"""
        return self.max_depth_seen
    
    def get_position(self) -> int:
        """Get current position in flattened sequence"""
        return self.current_position


def create_nested_integer(value) -> NestedInteger:
    """Helper to create NestedInteger from various inputs"""
    if isinstance(value, int):
        return NestedInteger(value)
    elif isinstance(value, list):
        nested_list = [create_nested_integer(item) for item in value]
        return NestedInteger(nested_list)
    else:
        return NestedInteger([])

def test_nested_iterator_basic():
    """Test basic nested iterator functionality"""
    print("=== Testing Basic Nested Iterator Functionality ===")
    
    implementations = [
        ("Precompute", NestedIteratorPrecompute),
        ("Stack-based", NestedIteratorStack),
        ("Queue-based", NestedIteratorQueue),
        ("Generator", NestedIteratorGenerator),
        ("Stateful", NestedIteratorStateful)
    ]
    
    # Test case: [[1,1],2,[1,1]]
    test_case = [
        create_nested_integer([1, 1]),
        create_nested_integer(2),
        create_nested_integer([1, 1])
    ]
    
    for name, IteratorClass in implementations:
        print(f"\n{name}:")
        
        iterator = IteratorClass(test_case)
        result = []
        
        while iterator.hasNext():
            result.append(iterator.next())
        
        print(f"  Flattened: {result}")

def test_nested_iterator_complex():
    """Test with complex nested structures"""
    print("\n=== Testing Complex Nested Structures ===")
    
    # Test case: [1,[4,[6]]]
    complex_case = [
        create_nested_integer(1),
        create_nested_integer([
            4,
            create_nested_integer([6])
        ])
    ]
    
    iterator = NestedIteratorStack(complex_case)
    result = []
    
    print("Complex nesting [1,[4,[6]]]:")
    while iterator.hasNext():
        result.append(iterator.next())
    
    print(f"  Flattened: {result}")
    
    # Test case: [[[[1]]]]
    deep_case = [
        create_nested_integer([
            create_nested_integer([
                create_nested_integer([1])
            ])
        ])
    ]
    
    iterator2 = NestedIteratorStack(deep_case)
    result2 = []
    
    print(f"\nDeep nesting [[[[1]]]]:")
    while iterator2.hasNext():
        result2.append(iterator2.next())
    
    print(f"  Flattened: {result2}")

def test_nested_iterator_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    # Empty list
    print("Empty nested list:")
    empty_iterator = NestedIteratorStack([])
    print(f"  hasNext(): {empty_iterator.hasNext()}")
    
    # List with empty sublists
    empty_sublists = [
        create_nested_integer([]),
        create_nested_integer(1),
        create_nested_integer([])
    ]
    
    iterator = NestedIteratorStack(empty_sublists)
    result = []
    
    print(f"\nList with empty sublists [[],1,[]]:")
    while iterator.hasNext():
        result.append(iterator.next())
    
    print(f"  Flattened: {result}")
    
    # Only nested empty lists
    only_empty = [
        create_nested_integer([]),
        create_nested_integer([create_nested_integer([])])
    ]
    
    iterator2 = NestedIteratorStack(only_empty)
    result2 = []
    
    print(f"\nOnly empty nested lists [[],[[]]]:")
    while iterator2.hasNext():
        result2.append(iterator2.next())
    
    print(f"  Flattened: {result2}")

def test_iterator_performance():
    """Test performance of different implementations"""
    print("\n=== Testing Iterator Performance ===")
    
    import time
    
    # Create large nested structure
    def create_large_nested(depth: int, width: int) -> List[NestedInteger]:
        if depth <= 0:
            return [create_nested_integer(i) for i in range(width)]
        
        nested_list = []
        for i in range(width):
            if i % 2 == 0:
                nested_list.append(create_nested_integer(i))
            else:
                sub_nested = create_large_nested(depth - 1, width // 2)
                nested_list.append(create_nested_integer(sub_nested))
        
        return nested_list
    
    large_nested = create_large_nested(3, 10)
    
    implementations = [
        ("Precompute", NestedIteratorPrecompute),
        ("Stack-based", NestedIteratorStack),
        ("Queue-based", NestedIteratorQueue),
        ("Generator", NestedIteratorGenerator)
    ]
    
    for name, IteratorClass in implementations:
        # Test initialization time
        start_time = time.time()
        iterator = IteratorClass(large_nested)
        init_time = (time.time() - start_time) * 1000
        
        # Test iteration time
        start_time = time.time()
        count = 0
        while iterator.hasNext():
            iterator.next()
            count += 1
        iteration_time = (time.time() - start_time) * 1000
        
        print(f"  {name}:")
        print(f"    Init: {init_time:.2f}ms, Iteration: {iteration_time:.2f}ms ({count} elements)")

def test_stateful_features():
    """Test stateful iterator features"""
    print("\n=== Testing Stateful Features ===")
    
    # Test case: [1,[2,3],4,[5,[6,7]]]
    test_case = [
        create_nested_integer(1),
        create_nested_integer([2, 3]),
        create_nested_integer(4),
        create_nested_integer([5, create_nested_integer([6, 7])])
    ]
    
    iterator = NestedIteratorStateful(test_case)
    
    print("Tracking depth and position:")
    while iterator.hasNext():
        depth = iterator.get_current_depth()
        position = iterator.get_position()
        value = iterator.next()
        
        print(f"  Position {position}: value={value}, depth={depth}")
    
    max_depth = iterator.get_max_depth_seen()
    print(f"  Maximum depth encountered: {max_depth}")

def demonstrate_applications():
    """Demonstrate real-world applications"""
    print("\n=== Demonstrating Applications ===")
    
    # Application 1: JSON-like structure processing
    print("Application 1: Nested Configuration Processing")
    
    # Simulate nested configuration: {"servers": [1, 2], "databases": {"primary": 10, "backup": [11, 12]}}
    config_structure = [
        create_nested_integer([1, 2]),  # servers
        create_nested_integer([10, create_nested_integer([11, 12])])  # databases
    ]
    
    iterator = NestedIteratorStack(config_structure)
    server_ids = []
    
    while iterator.hasNext():
        server_ids.append(iterator.next())
    
    print(f"  Extracted server/database IDs: {server_ids}")
    
    # Application 2: Nested comment thread processing
    print(f"\nApplication 2: Comment Thread Flattening")
    
    # Comments structure: comment1, [reply1, reply2], comment2, [reply3, [sub-reply1]]
    comments = [
        create_nested_integer(101),  # comment1
        create_nested_integer([201, 202]),  # replies to comment1
        create_nested_integer(102),  # comment2
        create_nested_integer([203, create_nested_integer([301])])  # replies with sub-reply
    ]
    
    iterator = NestedIteratorQueue(comments)
    comment_order = []
    
    while iterator.hasNext():
        comment_order.append(iterator.next())
    
    print(f"  Flattened comment processing order: {comment_order}")
    
    # Application 3: Nested menu navigation
    print(f"\nApplication 3: Menu Navigation System")
    
    # Menu: File, [New, Open, [Recent1, Recent2]], Edit, [Cut, Copy]
    menu_structure = [
        create_nested_integer(1),  # File
        create_nested_integer([11, 12, create_nested_integer([121, 122])]),  # File submenu
        create_nested_integer(2),  # Edit
        create_nested_integer([21, 22])  # Edit submenu
    ]
    
    iterator = NestedIteratorGenerator(menu_structure)
    menu_items = []
    
    while iterator.hasNext():
        menu_items.append(iterator.next())
    
    print(f"  Menu traversal order: {menu_items}")

def test_memory_efficiency():
    """Test memory efficiency of different approaches"""
    print("\n=== Testing Memory Efficiency ===")
    
    # Create nested structure with known characteristics
    def create_test_structure(levels: int) -> List[NestedInteger]:
        if levels <= 1:
            return [create_nested_integer(i) for i in range(5)]
        
        return [
            create_nested_integer(0),
            create_nested_integer(create_test_structure(levels - 1)),
            create_nested_integer(1)
        ]
    
    test_levels = [2, 3, 4]
    
    for levels in test_levels:
        print(f"\nNesting depth {levels}:")
        
        test_structure = create_test_structure(levels)
        
        # Test different implementations
        implementations = [
            ("Precompute", NestedIteratorPrecompute),
            ("Stack-based", NestedIteratorStack),
            ("Generator", NestedIteratorGenerator)
        ]
        
        for name, IteratorClass in implementations:
            iterator = IteratorClass(test_structure)
            
            # Count elements
            count = 0
            while iterator.hasNext():
                iterator.next()
                count += 1
            
            # Estimate memory usage (simplified)
            if hasattr(iterator, 'flattened'):
                memory_estimate = len(iterator.flattened)
            elif hasattr(iterator, 'stack'):
                memory_estimate = len(iterator.stack) * 2  # Rough estimate
            else:
                memory_estimate = levels  # Generator uses O(depth) space
            
            print(f"  {name}: {count} elements, ~{memory_estimate} memory units")

def test_hasNext_behavior():
    """Test hasNext() behavior and side effects"""
    print("\n=== Testing hasNext() Behavior ===")
    
    test_case = [
        create_nested_integer([]),
        create_nested_integer(1),
        create_nested_integer([2, 3]),
        create_nested_integer([])
    ]
    
    iterator = NestedIteratorStack(test_case)
    
    print("Testing hasNext() multiple calls:")
    
    call_count = 0
    while call_count < 10 and iterator.hasNext():
        call_count += 1
        has_next = iterator.hasNext()
        print(f"  hasNext() call {call_count}: {has_next}")
        
        if call_count % 3 == 0:  # Consume element every 3 calls
            value = iterator.next()
            print(f"    Consumed: {value}")

def benchmark_different_structures():
    """Benchmark performance on different structure types"""
    print("\n=== Benchmarking Different Structure Types ===")
    
    import time
    
    structures = [
        ("Flat list", [create_nested_integer(i) for i in range(1000)]),
        ("Deep nesting", [create_nested_integer([create_nested_integer([i]) for i in range(100)])]),
        ("Wide nesting", [create_nested_integer(list(range(100))) for _ in range(10)])
    ]
    
    for structure_name, structure in structures:
        print(f"\n{structure_name}:")
        
        implementations = [
            ("Precompute", NestedIteratorPrecompute),
            ("Stack-based", NestedIteratorStack)
        ]
        
        for impl_name, IteratorClass in implementations:
            start_time = time.time()
            
            iterator = IteratorClass(structure)
            count = 0
            
            while iterator.hasNext():
                iterator.next()
                count += 1
            
            elapsed = (time.time() - start_time) * 1000
            
            print(f"  {impl_name}: {elapsed:.2f}ms for {count} elements")

if __name__ == "__main__":
    test_nested_iterator_basic()
    test_nested_iterator_complex()
    test_nested_iterator_edge_cases()
    test_iterator_performance()
    test_stateful_features()
    demonstrate_applications()
    test_memory_efficiency()
    test_hasNext_behavior()
    benchmark_different_structures()

"""
Nested List Iterator Design demonstrates key concepts:

Core Approaches:
1. Precompute - Flatten entire structure upfront
2. Stack-based - Lazy evaluation using stack for position tracking
3. Queue-based - Eager expansion of nested lists as encountered
4. Generator - Python-specific recursive generator approach
5. Stateful - Enhanced with debugging and position tracking

Key Design Principles:
- Lazy vs eager evaluation trade-offs
- Memory efficiency in deeply nested structures
- State management for complex traversal
- Iterator pattern for nested data structures

Performance Characteristics:
- Precompute: O(n) init, O(1) operations, O(n) space
- Stack-based: O(1) init, O(d) hasNext, O(d) space
- Queue-based: O(1) init, O(1) hasNext, O(w) space
- Generator: O(1) init, O(1) operations, O(d) space

Real-world Applications:
- JSON/XML document processing
- Configuration file parsing
- Nested comment thread systems
- Menu navigation and tree traversal
- Compiler AST traversal

The stack-based approach is most commonly used
due to its optimal balance of memory usage and
performance for deeply nested structures.
"""
