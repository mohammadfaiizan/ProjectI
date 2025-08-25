"""
284. Peeking Iterator - Multiple Approaches
Difficulty: Medium

Design an iterator that supports the peek operation on an existing iterator in addition to the hasNext and the next operations.

Implement the PeekingIterator class:
- PeekingIterator(Iterator<int> iterator) Initializes the object with the given integer iterator iterator.
- int next() Returns the next element in the array and moves the pointer to the next element.
- boolean hasNext() Returns true if there are still elements in the array.
- int peek() Returns the next element in the array without moving the pointer.

Note: Each language may have a different implementation of the constructor and Iterator, but they all support the int next() and boolean hasNext() functions.
"""

from typing import Optional, List, Iterator as IteratorType

class Iterator:
    """Mock iterator class for demonstration"""
    def __init__(self, nums: List[int]):
        self.nums = nums
        self.index = 0
    
    def next(self) -> int:
        if self.hasNext():
            val = self.nums[self.index]
            self.index += 1
            return val
        return -1
    
    def hasNext(self) -> bool:
        return self.index < len(self.nums)

class PeekingIteratorSimple:
    """
    Approach 1: Simple Caching with Flag
    
    Cache the next value and use a flag to track if it's consumed.
    
    Time Complexity: O(1) for all operations
    Space Complexity: O(1)
    """
    
    def __init__(self, iterator: Iterator):
        self.iterator = iterator
        self.has_peeked = False
        self.peeked_value = None
    
    def peek(self) -> int:
        if not self.has_peeked:
            if self.iterator.hasNext():
                self.peeked_value = self.iterator.next()
                self.has_peeked = True
        
        return self.peeked_value if self.has_peeked else -1
    
    def next(self) -> int:
        if self.has_peeked:
            self.has_peeked = False
            return self.peeked_value
        
        return self.iterator.next() if self.iterator.hasNext() else -1
    
    def hasNext(self) -> bool:
        return self.has_peeked or self.iterator.hasNext()

class PeekingIteratorBuffer:
    """
    Approach 2: Single Element Buffer
    
    Always maintain one element in buffer for peek operations.
    
    Time Complexity: O(1) for all operations
    Space Complexity: O(1)
    """
    
    def __init__(self, iterator: Iterator):
        self.iterator = iterator
        self.buffer = None
        self._advance()
    
    def _advance(self) -> None:
        """Advance the buffer to next element"""
        if self.iterator.hasNext():
            self.buffer = self.iterator.next()
        else:
            self.buffer = None
    
    def peek(self) -> int:
        return self.buffer if self.buffer is not None else -1
    
    def next(self) -> int:
        current = self.buffer
        self._advance()
        return current if current is not None else -1
    
    def hasNext(self) -> bool:
        return self.buffer is not None

class PeekingIteratorLookahead:
    """
    Approach 3: Multi-element Lookahead
    
    Support looking ahead multiple elements.
    
    Time Complexity: O(1) for standard operations, O(k) for lookahead
    Space Complexity: O(k) where k is lookahead distance
    """
    
    def __init__(self, iterator: Iterator):
        self.iterator = iterator
        self.buffer = []
        self.buffer_size = 0
    
    def _ensure_buffer(self, size: int) -> None:
        """Ensure buffer has at least 'size' elements"""
        while len(self.buffer) < size and self.iterator.hasNext():
            self.buffer.append(self.iterator.next())
    
    def peek(self, distance: int = 0) -> int:
        """Peek at element 'distance' positions ahead"""
        self._ensure_buffer(distance + 1)
        
        if distance < len(self.buffer):
            return self.buffer[distance]
        return -1
    
    def next(self) -> int:
        self._ensure_buffer(1)
        
        if self.buffer:
            return self.buffer.pop(0)
        return -1
    
    def hasNext(self) -> bool:
        self._ensure_buffer(1)
        return len(self.buffer) > 0

class PeekingIteratorStateful:
    """
    Approach 4: Stateful with Multiple Operations
    
    Enhanced iterator with additional state tracking.
    
    Time Complexity: O(1) for all operations
    Space Complexity: O(1)
    """
    
    def __init__(self, iterator: Iterator):
        self.iterator = iterator
        self.current_value = None
        self.has_current = False
        self.position = 0
        self._fetch_next()
    
    def _fetch_next(self) -> None:
        """Fetch next value from underlying iterator"""
        if self.iterator.hasNext():
            self.current_value = self.iterator.next()
            self.has_current = True
        else:
            self.current_value = None
            self.has_current = False
    
    def peek(self) -> int:
        return self.current_value if self.has_current else -1
    
    def next(self) -> int:
        if not self.has_current:
            return -1
        
        value = self.current_value
        self.position += 1
        self._fetch_next()
        return value
    
    def hasNext(self) -> bool:
        return self.has_current
    
    def position(self) -> int:
        """Get current position in iteration"""
        return self.position
    
    def reset_if_possible(self) -> bool:
        """Reset iterator if possible (for demonstration)"""
        # This would require the underlying iterator to support reset
        return False

class PeekingIteratorRobust:
    """
    Approach 5: Robust with Error Handling
    
    Enhanced error handling and edge case management.
    
    Time Complexity: O(1) for all operations
    Space Complexity: O(1)
    """
    
    def __init__(self, iterator: Iterator):
        if iterator is None:
            raise ValueError("Iterator cannot be None")
        
        self.iterator = iterator
        self.cached_value = None
        self.has_cached = False
        self.is_exhausted = False
        self.operation_count = 0
    
    def peek(self) -> int:
        if self.is_exhausted:
            return -1
        
        if not self.has_cached:
            try:
                if self.iterator.hasNext():
                    self.cached_value = self.iterator.next()
                    self.has_cached = True
                else:
                    self.is_exhausted = True
                    return -1
            except Exception:
                self.is_exhausted = True
                return -1
        
        self.operation_count += 1
        return self.cached_value
    
    def next(self) -> int:
        if self.is_exhausted:
            return -1
        
        if self.has_cached:
            value = self.cached_value
            self.has_cached = False
            self.cached_value = None
        else:
            try:
                if self.iterator.hasNext():
                    value = self.iterator.next()
                else:
                    self.is_exhausted = True
                    return -1
            except Exception:
                self.is_exhausted = True
                return -1
        
        self.operation_count += 1
        return value
    
    def hasNext(self) -> bool:
        if self.is_exhausted:
            return False
        
        if self.has_cached:
            return True
        
        try:
            return self.iterator.hasNext()
        except Exception:
            self.is_exhausted = True
            return False
    
    def get_operation_count(self) -> int:
        """Get total number of operations performed"""
        return self.operation_count


def test_peeking_iterator_basic():
    """Test basic peeking iterator functionality"""
    print("=== Testing Basic Peeking Iterator Functionality ===")
    
    implementations = [
        ("Simple Caching", PeekingIteratorSimple),
        ("Buffer-based", PeekingIteratorBuffer),
        ("Lookahead", PeekingIteratorLookahead),
        ("Stateful", PeekingIteratorStateful),
        ("Robust", PeekingIteratorRobust)
    ]
    
    test_data = [1, 2, 3, 4, 5]
    
    for name, PeekingClass in implementations:
        print(f"\n{name}:")
        
        base_iterator = Iterator(test_data)
        peeking_iter = PeekingClass(base_iterator)
        
        # Test sequence: peek, next, peek, next, etc.
        operations = [
            ("peek", None), ("peek", None), ("next", None),
            ("peek", None), ("next", None), ("hasNext", None),
            ("next", None), ("peek", None), ("next", None)
        ]
        
        for op, _ in operations:
            if op == "peek":
                if hasattr(peeking_iter, 'peek'):
                    result = peeking_iter.peek()
                    print(f"  peek(): {result}")
            elif op == "next":
                result = peeking_iter.next()
                print(f"  next(): {result}")
            elif op == "hasNext":
                result = peeking_iter.hasNext()
                print(f"  hasNext(): {result}")

def test_peeking_iterator_edge_cases():
    """Test peeking iterator edge cases"""
    print("\n=== Testing Peeking Iterator Edge Cases ===")
    
    # Test with empty iterator
    print("Empty iterator:")
    empty_iter = Iterator([])
    peeking_empty = PeekingIteratorSimple(empty_iter)
    
    print(f"  hasNext(): {peeking_empty.hasNext()}")
    print(f"  peek(): {peeking_empty.peek()}")
    print(f"  next(): {peeking_empty.next()}")
    
    # Test with single element
    print(f"\nSingle element iterator:")
    single_iter = Iterator([42])
    peeking_single = PeekingIteratorBuffer(single_iter)
    
    print(f"  peek(): {peeking_single.peek()}")
    print(f"  peek(): {peeking_single.peek()}")  # Multiple peeks
    print(f"  hasNext(): {peeking_single.hasNext()}")
    print(f"  next(): {peeking_single.next()}")
    print(f"  hasNext(): {peeking_single.hasNext()}")
    print(f"  peek(): {peeking_single.peek()}")
    
    # Test peek without next
    print(f"\nPeek without consuming:")
    peek_test_iter = Iterator([10, 20, 30])
    peeking_peek = PeekingIteratorSimple(peek_test_iter)
    
    for i in range(5):
        peek_val = peeking_peek.peek()
        print(f"  peek() call {i+1}: {peek_val}")

def test_lookahead_functionality():
    """Test lookahead functionality"""
    print("\n=== Testing Lookahead Functionality ===")
    
    test_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    base_iterator = Iterator(test_data)
    lookahead_iter = PeekingIteratorLookahead(base_iterator)
    
    print("Testing multi-element lookahead:")
    
    # Look ahead at different distances
    for distance in range(5):
        value = lookahead_iter.peek(distance)
        print(f"  peek({distance}): {value}")
    
    # Consume some elements and test again
    print(f"\nAfter consuming 3 elements:")
    for _ in range(3):
        consumed = lookahead_iter.next()
        print(f"  next(): {consumed}")
    
    print(f"New lookahead values:")
    for distance in range(3):
        value = lookahead_iter.peek(distance)
        print(f"  peek({distance}): {value}")

def test_multiple_peek_patterns():
    """Test various peek usage patterns"""
    print("\n=== Testing Multiple Peek Patterns ===")
    
    test_data = [1, 2, 3, 4, 5]
    
    patterns = [
        ("Peek before every next", "peek,next,peek,next,peek,next"),
        ("Multiple peeks", "peek,peek,peek,next,peek,peek,next"),
        ("Peek at end", "next,next,next,next,peek,next,peek"),
        ("Only peeks", "peek,peek,peek,peek,peek")
    ]
    
    for pattern_name, operations in patterns:
        print(f"\n{pattern_name}:")
        
        base_iterator = Iterator(test_data)
        peeking_iter = PeekingIteratorSimple(base_iterator)
        
        ops = operations.split(",")
        for op in ops:
            if op == "peek":
                result = peeking_iter.peek()
                print(f"  peek(): {result}")
            elif op == "next":
                result = peeking_iter.next()
                print(f"  next(): {result}")

def test_performance_comparison():
    """Test performance of different implementations"""
    print("\n=== Testing Performance Comparison ===")
    
    import time
    
    implementations = [
        ("Simple Caching", PeekingIteratorSimple),
        ("Buffer-based", PeekingIteratorBuffer),
        ("Stateful", PeekingIteratorStateful),
        ("Robust", PeekingIteratorRobust)
    ]
    
    # Large test data
    large_data = list(range(10000))
    operations_count = 20000
    
    for name, PeekingClass in implementations:
        base_iterator = Iterator(large_data)
        peeking_iter = PeekingClass(base_iterator)
        
        start_time = time.time()
        
        # Mixed operations
        import random
        for _ in range(operations_count):
            op = random.choice(['peek', 'next', 'hasNext'])
            
            if op == 'peek':
                peeking_iter.peek()
            elif op == 'next':
                if peeking_iter.hasNext():
                    peeking_iter.next()
            else:  # hasNext
                peeking_iter.hasNext()
        
        elapsed = (time.time() - start_time) * 1000
        
        print(f"  {name}: {elapsed:.2f}ms for {operations_count} operations")

def demonstrate_applications():
    """Demonstrate real-world applications"""
    print("\n=== Demonstrating Applications ===")
    
    # Application 1: Parser lookahead
    print("Application 1: Simple Expression Parser")
    
    tokens = ["(", "1", "+", "2", ")", "*", "3"]
    token_iterator = Iterator(tokens)
    parser_iter = PeekingIteratorLookahead(token_iterator)
    
    def parse_expression():
        result = []
        
        while parser_iter.hasNext():
            current_token = parser_iter.peek()
            
            if current_token == "(":
                result.append("START_GROUP")
                parser_iter.next()
            elif current_token == ")":
                result.append("END_GROUP")
                parser_iter.next()
            elif current_token in ["+", "*", "-", "/"]:
                result.append(f"OPERATOR_{current_token}")
                parser_iter.next()
            else:
                result.append(f"NUMBER_{current_token}")
                parser_iter.next()
        
        return result
    
    parsed = parse_expression()
    print(f"  Tokens: {tokens}")
    print(f"  Parsed: {parsed[:5]}...")  # Show first 5 for brevity
    
    # Application 2: Data validation with lookahead
    print(f"\nApplication 2: Data Validation")
    
    data_stream = [1, 2, 4, 8, 16, 32, 64]  # Powers of 2
    data_iterator = Iterator(data_stream)
    validator_iter = PeekingIteratorSimple(data_iterator)
    
    def validate_doubling_sequence():
        valid = True
        
        if not validator_iter.hasNext():
            return True
        
        current = validator_iter.next()
        
        while validator_iter.hasNext():
            next_val = validator_iter.peek()
            
            if next_val != current * 2:
                print(f"    Validation failed: {current} -> {next_val} (expected {current * 2})")
                valid = False
                break
            
            current = validator_iter.next()
        
        return valid
    
    is_valid = validate_doubling_sequence()
    print(f"  Data stream: {data_stream}")
    print(f"  Is valid doubling sequence: {is_valid}")
    
    # Application 3: Stream processing with buffering
    print(f"\nApplication 3: Stream Processing with Buffering")
    
    sensor_data = [23.5, 24.1, 23.8, 25.2, 24.9, 23.7, 24.3]
    sensor_iterator = Iterator([int(x * 10) for x in sensor_data])  # Convert to integers
    buffer_iter = PeekingIteratorBuffer(sensor_iterator)
    
    def detect_anomalies(threshold=20):  # 2.0 degree threshold
        anomalies = []
        
        if not buffer_iter.hasNext():
            return anomalies
        
        current = buffer_iter.next() / 10.0
        
        while buffer_iter.hasNext():
            next_val = buffer_iter.peek() / 10.0
            
            if abs(next_val - current) > threshold / 10.0:
                anomalies.append((current, next_val))
            
            current = buffer_iter.next() / 10.0
        
        return anomalies
    
    anomalies = detect_anomalies()
    print(f"  Sensor readings: {sensor_data}")
    print(f"  Anomalies detected: {len(anomalies)}")

def test_stateful_features():
    """Test stateful iterator features"""
    print("\n=== Testing Stateful Features ===")
    
    test_data = [10, 20, 30, 40, 50]
    base_iterator = Iterator(test_data)
    stateful_iter = PeekingIteratorStateful(base_iterator)
    
    print("Testing position tracking:")
    
    operations = ["peek", "next", "peek", "next", "next"]
    
    for op in operations:
        if op == "peek":
            value = stateful_iter.peek()
            print(f"  peek(): {value}, position: {stateful_iter.position()}")
        else:  # next
            value = stateful_iter.next()
            print(f"  next(): {value}, position: {stateful_iter.position()}")

def test_robust_error_handling():
    """Test robust error handling"""
    print("\n=== Testing Robust Error Handling ===")
    
    test_data = [1, 2, 3]
    base_iterator = Iterator(test_data)
    robust_iter = PeekingIteratorRobust(base_iterator)
    
    print("Normal operations:")
    while robust_iter.hasNext():
        peeked = robust_iter.peek()
        next_val = robust_iter.next()
        print(f"  peek: {peeked}, next: {next_val}")
    
    print(f"\nOperations after exhaustion:")
    print(f"  hasNext(): {robust_iter.hasNext()}")
    print(f"  peek(): {robust_iter.peek()}")
    print(f"  next(): {robust_iter.next()}")
    
    print(f"  Total operations: {robust_iter.get_operation_count()}")

def benchmark_peek_vs_next_ratio():
    """Benchmark different peek to next ratios"""
    print("\n=== Benchmarking Peek vs Next Ratios ===")
    
    import time
    
    test_data = list(range(1000))
    ratios = [
        ("Peek heavy (10:1)", 10, 1),
        ("Balanced (1:1)", 1, 1),
        ("Next heavy (1:10)", 1, 10)
    ]
    
    for ratio_name, peek_ratio, next_ratio in ratios:
        print(f"\n{ratio_name}:")
        
        base_iterator = Iterator(test_data)
        peeking_iter = PeekingIteratorSimple(base_iterator)
        
        start_time = time.time()
        
        operation_cycle = 0
        total_ops = 0
        
        while peeking_iter.hasNext() and total_ops < 5000:
            cycle_pos = operation_cycle % (peek_ratio + next_ratio)
            
            if cycle_pos < peek_ratio:
                peeking_iter.peek()
            else:
                peeking_iter.next()
            
            operation_cycle += 1
            total_ops += 1
        
        elapsed = (time.time() - start_time) * 1000
        
        print(f"  {total_ops} operations in {elapsed:.2f}ms")

if __name__ == "__main__":
    test_peeking_iterator_basic()
    test_peeking_iterator_edge_cases()
    test_lookahead_functionality()
    test_multiple_peek_patterns()
    test_performance_comparison()
    demonstrate_applications()
    test_stateful_features()
    test_robust_error_handling()
    benchmark_peek_vs_next_ratio()

"""
Peeking Iterator Design demonstrates key concepts:

Core Approaches:
1. Simple Caching - Basic flag-based approach for single peek
2. Buffer-based - Always maintain next element in buffer
3. Lookahead - Support multiple element lookahead
4. Stateful - Enhanced with position tracking and state
5. Robust - Comprehensive error handling and edge cases

Key Design Principles:
- Lazy evaluation and caching strategies
- State management for iterator patterns
- Lookahead capabilities for parsing applications
- Error handling and robustness considerations

Performance Characteristics:
- All approaches provide O(1) operations
- Space complexity varies from O(1) to O(k) for lookahead
- Trade-offs between simplicity and functionality

Real-world Applications:
- Parser implementations requiring lookahead
- Data validation and pattern detection
- Stream processing with buffering
- Compiler design and tokenization
- Protocol parsing and state machines

The simple caching approach is most commonly used
due to its simplicity and optimal performance for
standard peek operations.
"""
