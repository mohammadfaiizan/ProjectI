"""
251. Flatten 2D Vector - Multiple Approaches
Difficulty: Medium

Design an iterator to flatten a 2d vector. It should support the next and hasNext operations.

Implement the Vector2D class:
- Vector2D(int[][] vec) initializes the object with the 2d vector vec.
- next() returns the next element from the 2d vector and moves the pointer in the right direction.
- hasNext() returns true if there are still some elements left.
"""

from typing import List, Iterator

class Vector2DSimple:
    """
    Approach 1: Flatten on Initialization
    
    Flatten the 2D vector into 1D during initialization.
    
    Time Complexity:
    - __init__: O(n) where n is total elements
    - next: O(1)
    - hasNext: O(1)
    
    Space Complexity: O(n)
    """
    
    def __init__(self, vec: List[List[int]]):
        self.flattened = []
        for row in vec:
            self.flattened.extend(row)
        self.index = 0
    
    def next(self) -> int:
        if self.hasNext():
            result = self.flattened[self.index]
            self.index += 1
            return result
        raise StopIteration("No more elements")
    
    def hasNext(self) -> bool:
        return self.index < len(self.flattened)

class Vector2DLazy:
    """
    Approach 2: Lazy Evaluation with Two Pointers
    
    Use two pointers to track current position without pre-flattening.
    
    Time Complexity:
    - __init__: O(1)
    - next: O(1) amortized
    - hasNext: O(1) amortized
    
    Space Complexity: O(1)
    """
    
    def __init__(self, vec: List[List[int]]):
        self.vec = vec
        self.row = 0
        self.col = 0
        self._advance_to_next()
    
    def _advance_to_next(self) -> None:
        """Advance pointers to next valid element"""
        while self.row < len(self.vec):
            if self.col < len(self.vec[self.row]):
                return  # Found valid position
            
            # Move to next row
            self.row += 1
            self.col = 0
    
    def next(self) -> int:
        if not self.hasNext():
            raise StopIteration("No more elements")
        
        result = self.vec[self.row][self.col]
        self.col += 1
        self._advance_to_next()
        return result
    
    def hasNext(self) -> bool:
        return self.row < len(self.vec)

class Vector2DGenerator:
    """
    Approach 3: Generator-based Implementation
    
    Use Python generator for lazy evaluation.
    
    Time Complexity:
    - __init__: O(1)
    - next: O(1) amortized
    - hasNext: O(1)
    
    Space Complexity: O(1)
    """
    
    def __init__(self, vec: List[List[int]]):
        self.generator = self._create_generator(vec)
        self.next_value = None
        self.has_next_value = False
        self._advance()
    
    def _create_generator(self, vec: List[List[int]]) -> Iterator[int]:
        """Create generator for flattened vector"""
        for row in vec:
            for element in row:
                yield element
    
    def _advance(self) -> None:
        """Get next value from generator"""
        try:
            self.next_value = next(self.generator)
            self.has_next_value = True
        except StopIteration:
            self.has_next_value = False
    
    def next(self) -> int:
        if not self.hasNext():
            raise StopIteration("No more elements")
        
        result = self.next_value
        self._advance()
        return result
    
    def hasNext(self) -> bool:
        return self.has_next_value

class Vector2DAdvanced:
    """
    Approach 4: Advanced with Features and Analytics
    
    Enhanced version with statistics and additional functionality.
    
    Time Complexity:
    - __init__: O(1)
    - next: O(1) amortized
    - hasNext: O(1)
    
    Space Complexity: O(1) + analytics
    """
    
    def __init__(self, vec: List[List[int]]):
        self.vec = vec
        self.row = 0
        self.col = 0
        
        # Analytics
        self.total_elements = sum(len(row) for row in vec)
        self.elements_consumed = 0
        self.next_calls = 0
        self.has_next_calls = 0
        
        # Features
        self.history = []  # Track accessed elements
        
        self._advance_to_next()
    
    def _advance_to_next(self) -> None:
        """Advance to next valid element"""
        while self.row < len(self.vec):
            # Skip empty rows
            if self.col < len(self.vec[self.row]):
                return
            
            self.row += 1
            self.col = 0
    
    def next(self) -> int:
        self.next_calls += 1
        
        if not self.hasNext():
            raise StopIteration("No more elements")
        
        result = self.vec[self.row][self.col]
        
        # Update analytics
        self.elements_consumed += 1
        self.history.append((self.row, self.col, result))
        
        # Advance position
        self.col += 1
        self._advance_to_next()
        
        return result
    
    def hasNext(self) -> bool:
        self.has_next_calls += 1
        return self.row < len(self.vec)
    
    def get_progress(self) -> dict:
        """Get iteration progress"""
        progress_percent = (self.elements_consumed / max(1, self.total_elements)) * 100
        
        return {
            'total_elements': self.total_elements,
            'elements_consumed': self.elements_consumed,
            'progress_percent': progress_percent,
            'current_position': (self.row, self.col) if self.hasNext() else None,
            'next_calls': self.next_calls,
            'has_next_calls': self.has_next_calls
        }
    
    def peek(self) -> int:
        """Peek at next element without consuming it"""
        if not self.hasNext():
            raise StopIteration("No more elements")
        
        return self.vec[self.row][self.col]
    
    def get_history(self) -> List[tuple]:
        """Get history of consumed elements"""
        return self.history.copy()
    
    def reset(self) -> None:
        """Reset iterator to beginning"""
        self.row = 0
        self.col = 0
        self.elements_consumed = 0
        self.next_calls = 0
        self.has_next_calls = 0
        self.history.clear()
        self._advance_to_next()

class Vector2DMemoryOptimized:
    """
    Approach 5: Memory-Optimized for Large Vectors
    
    Optimized for memory usage with minimal overhead.
    
    Time Complexity:
    - __init__: O(1)
    - next: O(1) amortized
    - hasNext: O(1)
    
    Space Complexity: O(1)
    """
    
    def __init__(self, vec: List[List[int]]):
        self.vec = vec
        self.current_pos = self._find_first_element()
    
    def _find_first_element(self) -> tuple:
        """Find first valid element position"""
        for row_idx, row in enumerate(self.vec):
            if row:  # Non-empty row
                return (row_idx, 0)
        return (len(self.vec), 0)  # No elements found
    
    def _find_next_element(self, row: int, col: int) -> tuple:
        """Find next valid element position"""
        # Try next column in current row
        if row < len(self.vec) and col + 1 < len(self.vec[row]):
            return (row, col + 1)
        
        # Find next non-empty row
        for next_row in range(row + 1, len(self.vec)):
            if self.vec[next_row]:
                return (next_row, 0)
        
        return (len(self.vec), 0)  # End of vector
    
    def next(self) -> int:
        if not self.hasNext():
            raise StopIteration("No more elements")
        
        row, col = self.current_pos
        result = self.vec[row][col]
        
        self.current_pos = self._find_next_element(row, col)
        
        return result
    
    def hasNext(self) -> bool:
        return self.current_pos[0] < len(self.vec)


def test_vector2d_basic():
    """Test basic Vector2D functionality"""
    print("=== Testing Basic Vector2D Functionality ===")
    
    implementations = [
        ("Simple Flatten", Vector2DSimple),
        ("Lazy Two Pointers", Vector2DLazy),
        ("Generator", Vector2DGenerator),
        ("Advanced", Vector2DAdvanced),
        ("Memory Optimized", Vector2DMemoryOptimized)
    ]
    
    test_cases = [
        [[1, 2], [3], [4, 5, 6]],
        [[1], [2], [3]],
        [[1, 2, 3, 4, 5, 6]],
        []
    ]
    
    for test_case in test_cases:
        print(f"\nTest case: {test_case}")
        
        # Generate expected output
        expected = []
        for row in test_case:
            expected.extend(row)
        
        for name, Vector2DClass in implementations:
            try:
                iterator = Vector2DClass(test_case)
                result = []
                
                while iterator.hasNext():
                    result.append(iterator.next())
                
                correct = result == expected
                print(f"  {name}: {result} - {'✓' if correct else '✗'}")
                
            except Exception as e:
                print(f"  {name}: Error - {e}")

def test_vector2d_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Vector2D Edge Cases ===")
    
    # Test empty vector
    print("Empty vector:")
    iterator = Vector2DLazy([])
    
    print(f"  hasNext(): {iterator.hasNext()}")
    
    try:
        iterator.next()
        print("  next(): Should have raised exception")
    except StopIteration:
        print("  next(): Correctly raised StopIteration")
    
    # Test vector with empty rows
    print(f"\nVector with empty rows:")
    iterator = Vector2DAdvanced([[], [1, 2], [], [3], []])
    
    result = []
    while iterator.hasNext():
        result.append(iterator.next())
    
    print(f"  Result: {result}")
    print(f"  Expected: [1, 2, 3]")
    
    # Test single element
    print(f"\nSingle element:")
    iterator = Vector2DMemoryOptimized([[5]])
    
    print(f"  hasNext(): {iterator.hasNext()}")
    print(f"  next(): {iterator.next()}")
    print(f"  hasNext(): {iterator.hasNext()}")
    
    # Test calling next() when empty
    print(f"\nCalling next() when exhausted:")
    iterator = Vector2DSimple([[1]])
    
    iterator.next()  # Consume the element
    
    try:
        iterator.next()
        print("  Should have raised exception")
    except StopIteration:
        print("  Correctly raised StopIteration")

def test_performance_comparison():
    """Test performance of different implementations"""
    print("\n=== Testing Performance Comparison ===")
    
    import time
    
    # Create large test vector
    large_vector = [[i * 100 + j for j in range(100)] for i in range(100)]  # 10,000 elements
    
    implementations = [
        ("Simple Flatten", Vector2DSimple),
        ("Lazy Two Pointers", Vector2DLazy),
        ("Generator", Vector2DGenerator),
        ("Memory Optimized", Vector2DMemoryOptimized)
    ]
    
    for name, Vector2DClass in implementations:
        # Time initialization
        start_time = time.time()
        iterator = Vector2DClass(large_vector)
        init_time = (time.time() - start_time) * 1000
        
        # Time iteration
        start_time = time.time()
        count = 0
        while iterator.hasNext():
            iterator.next()
            count += 1
        iteration_time = (time.time() - start_time) * 1000
        
        print(f"  {name}:")
        print(f"    Init: {init_time:.2f}ms")
        print(f"    Iteration: {iteration_time:.2f}ms")
        print(f"    Elements: {count}")

def test_advanced_features():
    """Test advanced features"""
    print("\n=== Testing Advanced Features ===")
    
    test_vector = [[1, 2], [3, 4, 5], [6]]
    iterator = Vector2DAdvanced(test_vector)
    
    # Test progress tracking
    print("Progress tracking:")
    
    for i in range(3):
        if iterator.hasNext():
            element = iterator.next()
            progress = iterator.get_progress()
            
            print(f"  Element {i+1}: {element}")
            print(f"    Progress: {progress['progress_percent']:.1f}%")
            print(f"    Position: {progress['current_position']}")
    
    # Test peek functionality
    print(f"\nPeek functionality:")
    
    if iterator.hasNext():
        peeked = iterator.peek()
        next_element = iterator.next()
        
        print(f"  Peeked: {peeked}")
        print(f"  Next: {next_element}")
        print(f"  Match: {peeked == next_element}")
    
    # Test history
    print(f"\nHistory:")
    history = iterator.get_history()
    for i, (row, col, value) in enumerate(history):
        print(f"  Step {i+1}: vec[{row}][{col}] = {value}")
    
    # Test reset
    print(f"\nReset functionality:")
    
    before_reset = iterator.get_progress()
    iterator.reset()
    after_reset = iterator.get_progress()
    
    print(f"  Before reset: {before_reset['elements_consumed']} consumed")
    print(f"  After reset: {after_reset['elements_consumed']} consumed")

def demonstrate_applications():
    """Demonstrate real-world applications"""
    print("\n=== Demonstrating Applications ===")
    
    # Application 1: Matrix processing
    print("Application 1: Matrix Data Processing")
    
    matrix_data = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    
    processor = Vector2DAdvanced(matrix_data)
    
    print("  Processing matrix row by row:")
    
    # Process elements in flattened order
    sum_total = 0
    count = 0
    
    while processor.hasNext():
        value = processor.next()
        sum_total += value
        count += 1
        
        if count % 3 == 0:  # Every 3 elements (end of row)
            progress = processor.get_progress()
            print(f"    Processed row, running sum: {sum_total}, progress: {progress['progress_percent']:.1f}%")
    
    print(f"  Final sum: {sum_total}")
    
    # Application 2: Sparse data structure
    print(f"\nApplication 2: Sparse Data Structure Processing")
    
    sparse_data = [
        [],           # Empty row
        [10, 20],     # Some data
        [],           # Empty row
        [30],         # Single element
        [],           # Empty row
        [40, 50, 60]  # More data
    ]
    
    sparse_processor = Vector2DLazy(sparse_data)
    
    print("  Processing sparse data structure:")
    
    non_zero_elements = []
    while sparse_processor.hasNext():
        element = sparse_processor.next()
        non_zero_elements.append(element)
    
    print(f"    Non-zero elements: {non_zero_elements}")
    print(f"    Count: {len(non_zero_elements)}")
    
    # Application 3: Batch processing system
    print(f"\nApplication 3: Batch Processing System")
    
    batches = [
        [1001, 1002, 1003],     # Batch 1: Order IDs
        [1004, 1005],           # Batch 2: Order IDs  
        [1006, 1007, 1008, 1009] # Batch 3: Order IDs
    ]
    
    batch_processor = Vector2DMemoryOptimized(batches)
    
    print("  Processing order batches:")
    
    processed_orders = 0
    current_batch = 1
    orders_in_current_batch = 0
    
    while batch_processor.hasNext():
        order_id = batch_processor.next()
        processed_orders += 1
        orders_in_current_batch += 1
        
        # Simulate processing
        print(f"    Processing order {order_id}")
        
        # Check if we've processed all orders in current batch
        if orders_in_current_batch >= len(batches[current_batch - 1]):
            print(f"    Completed batch {current_batch} ({orders_in_current_batch} orders)")
            current_batch += 1
            orders_in_current_batch = 0
    
    print(f"  Total orders processed: {processed_orders}")
    
    # Application 4: Image pixel processing
    print(f"\nApplication 4: Image Pixel Processing")
    
    # Simulate RGB image as 2D array (simplified to single channel)
    image_pixels = [
        [255, 128, 64],   # Row 1: Pixel values
        [192, 96, 32],    # Row 2: Pixel values
        [224, 160, 80]    # Row 3: Pixel values
    ]
    
    pixel_processor = Vector2DGenerator(image_pixels)
    
    print("  Processing image pixels:")
    
    # Apply simple filter (example: brightness adjustment)
    brightness_factor = 0.8
    processed_pixels = []
    
    while pixel_processor.hasNext():
        pixel_value = pixel_processor.next()
        adjusted_value = int(pixel_value * brightness_factor)
        processed_pixels.append(adjusted_value)
    
    print(f"    Original pixels: {[pixel for row in image_pixels for pixel in row]}")
    print(f"    Processed pixels: {processed_pixels}")
    print(f"    Brightness factor: {brightness_factor}")

def test_memory_efficiency():
    """Test memory efficiency"""
    print("\n=== Testing Memory Efficiency ===")
    
    # Create vectors with different characteristics
    test_cases = [
        ("Dense small", [[i for i in range(10)] for _ in range(10)]),
        ("Sparse", [[] if i % 3 == 0 else [i] for i in range(100)]),
        ("Single large row", [[i for i in range(1000)]]),
        ("Many small rows", [[i] for i in range(1000)])
    ]
    
    implementations = [
        ("Simple Flatten", Vector2DSimple),
        ("Lazy", Vector2DLazy),
        ("Memory Optimized", Vector2DMemoryOptimized)
    ]
    
    for test_name, test_vector in test_cases:
        print(f"\n{test_name}:")
        
        for impl_name, Vector2DClass in implementations:
            iterator = Vector2DClass(test_vector)
            
            # Estimate memory usage
            if hasattr(iterator, 'flattened'):
                memory_estimate = len(iterator.flattened)
                approach = "Pre-flattened"
            else:
                memory_estimate = 1  # Just pointer tracking
                approach = "Lazy"
            
            # Count total elements
            total_elements = sum(len(row) for row in test_vector)
            
            print(f"    {impl_name} ({approach}): ~{memory_estimate} vs {total_elements} elements")

def stress_test_vector2d():
    """Stress test Vector2D implementations"""
    print("\n=== Stress Testing Vector2D ===")
    
    import time
    
    # Create very large vector
    large_rows = 1000
    elements_per_row = 100
    
    print(f"Creating {large_rows}x{elements_per_row} vector...")
    
    large_vector = []
    for i in range(large_rows):
        if i % 10 == 0:  # Every 10th row is empty (sparse pattern)
            large_vector.append([])
        else:
            large_vector.append([i * elements_per_row + j for j in range(elements_per_row)])
    
    total_elements = sum(len(row) for row in large_vector)
    print(f"Total elements: {total_elements}")
    
    # Test different implementations
    implementations = [
        ("Lazy", Vector2DLazy),
        ("Memory Optimized", Vector2DMemoryOptimized)
    ]
    
    for name, Vector2DClass in implementations:
        print(f"\n{name}:")
        
        start_time = time.time()
        iterator = Vector2DClass(large_vector)
        init_time = time.time() - start_time
        
        start_time = time.time()
        count = 0
        sum_total = 0
        
        while iterator.hasNext():
            value = iterator.next()
            sum_total += value
            count += 1
            
            # Progress update every 10000 elements
            if count % 10000 == 0:
                elapsed = time.time() - start_time
                rate = count / elapsed if elapsed > 0 else 0
                print(f"    Processed {count} elements, rate: {rate:.0f} elements/sec")
        
        total_time = time.time() - start_time + init_time
        
        print(f"    Total time: {total_time:.3f}s")
        print(f"    Elements processed: {count}")
        print(f"    Sum: {sum_total}")
        print(f"    Rate: {count / total_time:.0f} elements/sec")

def test_concurrent_access():
    """Test behavior with concurrent-like access patterns"""
    print("\n=== Testing Concurrent-like Access Patterns ===")
    
    # Test multiple iterators on same data
    shared_vector = [[i, i+1, i+2] for i in range(0, 30, 3)]
    
    print("Multiple iterators on same vector:")
    
    iterator1 = Vector2DLazy(shared_vector)
    iterator2 = Vector2DLazy(shared_vector)
    
    # Interleave access
    results1 = []
    results2 = []
    
    while iterator1.hasNext() or iterator2.hasNext():
        if iterator1.hasNext():
            results1.append(iterator1.next())
        
        if iterator2.hasNext():
            results2.append(iterator2.next())
    
    print(f"  Iterator 1 result: {results1[:10]}...")  # Show first 10
    print(f"  Iterator 2 result: {results2[:10]}...")  # Show first 10
    print(f"  Results match: {results1 == results2}")

def benchmark_different_patterns():
    """Benchmark different vector patterns"""
    print("\n=== Benchmarking Different Patterns ===")
    
    import time
    
    patterns = [
        ("Dense uniform", [[i for i in range(100)] for _ in range(100)]),
        ("Sparse random", [[] if i % 5 == 0 else [i] for i in range(500)]),
        ("Single long row", [[i for i in range(10000)]]),
        ("Many short rows", [[i] for i in range(10000)]),
        ("Pyramid", [[j for j in range(i)] for i in range(1, 101)])
    ]
    
    for pattern_name, pattern_vector in patterns:
        total_elements = sum(len(row) for row in pattern_vector)
        
        iterator = Vector2DLazy(pattern_vector)
        
        start_time = time.time()
        
        count = 0
        while iterator.hasNext():
            iterator.next()
            count += 1
        
        elapsed = (time.time() - start_time) * 1000
        
        print(f"  {pattern_name}:")
        print(f"    Elements: {total_elements}")
        print(f"    Time: {elapsed:.2f}ms")
        print(f"    Rate: {count / (elapsed / 1000):.0f} elements/sec")

if __name__ == "__main__":
    test_vector2d_basic()
    test_vector2d_edge_cases()
    test_performance_comparison()
    test_advanced_features()
    demonstrate_applications()
    test_memory_efficiency()
    stress_test_vector2d()
    test_concurrent_access()
    benchmark_different_patterns()

"""
Flatten 2D Vector Design demonstrates key concepts:

Core Approaches:
1. Simple Flatten - Pre-flatten during initialization for O(1) access
2. Lazy Two Pointers - Use row/col pointers for space-efficient iteration
3. Generator - Python generator for clean lazy evaluation
4. Advanced - Enhanced with analytics, history, and additional features
5. Memory Optimized - Minimal overhead for large-scale data processing

Key Design Principles:
- Iterator pattern implementation with hasNext/next interface
- Lazy vs eager evaluation trade-offs
- Memory efficiency for large datasets
- Support for sparse data structures

Performance Characteristics:
- Simple: O(n) space, O(1) operations after initialization
- Lazy: O(1) space, O(1) amortized operations
- Generator: O(1) space, minimal overhead
- Advanced: O(1) space + analytics overhead

Real-world Applications:
- Matrix and multi-dimensional data processing
- Sparse data structure traversal
- Batch processing systems for ordered data
- Image and signal processing pipelines
- Database result set flattening
- API response data iteration

The lazy two-pointer approach provides optimal balance
of space efficiency and performance for most use cases,
especially when dealing with large or sparse datasets.
"""
